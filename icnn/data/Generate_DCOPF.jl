using Pkg

using PowerModels, JuMP, Ipopt, Gurobi
using PowerModels
using Distributions
using BSON
using Plots

function generate_truncated_scales(data::Dict, nsamples::Int)
    # Get the total number of buses in the system
    nbus = length(keys(data["bus"]))

    # === Define scaling factors for active (P) and reactive (Q) power ===
    # Active power (P) scaling factors are generated from a truncated normal distribution
    # with mean 0 and limits [-0.3, 0.3], then shifted by +1 to keep values around 1.
    scale_P = 1 .+ rand(truncated(Normal(0), -0.3, 0.3), nbus, nsamples)

    # Reactive power (Q) scaling factors are generated similarly,
    # but limited to the range [-0.15, 0], meaning Q is reduced slightly.
    scale_Q = 1 .+ rand(truncated(Normal(0), -0.15, 0.0), nbus, nsamples)

    # === Extract base demand values from PowerModels data ===
    Pd_base = zeros(nbus)
    Qd_base = zeros(nbus)

    # Create a mapping from bus IDs to their index positions
    bus_ids = sort(parse.(Int, keys(data["bus"])))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # Fill the base demand vectors for each bus
    for (load_id, load_data) in data["load"]
        bus_id = load_data["load_bus"]
        idx = bus_idx_map[bus_id]
        Pd_base[idx] = load_data["pd"]
        Qd_base[idx] = load_data["qd"]
    end

    # === Generate demand samples by applying scaling factors ===
    # Each sample represents a different scenario of P and Q demand variations.
    P_samples = scale_P' .* Pd_base'
    Q_samples = scale_Q' .* Qd_base'

    return P_samples, Q_samples
end


function run_ac_dc_batch_from_data(
    data_orig,
    P_samples,
    Q_samples,
    nsamples;
    solver=Ipopt.Optimizer
)
    # === Extract problem dimensions ===
    total_available, nbus = size(P_samples)

    # Build reference AC model to determine number of generators (ngen)
    data0 = deepcopy(data_orig)
    spec0 = PowerModels.instantiate_model(data0, PowerModels.ACRPowerModel, PowerModels.build_opf)
    ngen = length(spec0.ref[:it][:pm][:nw][0][:gen])

    # === Preallocate results for nsamples requested ===
    Demand = zeros(nsamples, 2 * nbus)
    DispatchAC = zeros(nsamples, ngen)
    DispatchDC = zeros(nsamples, ngen)
    TimeAC = zeros(nsamples)
    TimeDC = zeros(nsamples)
    ObjAC = zeros(nsamples)
    ObjDC = zeros(nsamples)

    valid_k = 0  # counter of valid samples
    j = 1        # index of current sample

    @info "Searching for $nsamples valid samples (total available = $total_available)..."

    # === Main Sampling Loop ===
    while valid_k < nsamples && j <= total_available
        @info "Evaluating sample $j of $total_available"

        Pd_i = P_samples[j, :]
        Qd_i = Q_samples[j, :]

        # ================================
        # === 1) Solve DC-OPF for sample
        # ================================
        data_dc = deepcopy(data_orig)
        t_dc = 0.0
        res_dc = nothing
        dc_success = false

        try
            t1 = time()
            res_dc = DCPM(data_dc, Pd_i, Qd_i; solver=Gurobi.Optimizer)
            t_dc = time() - t1
            dc_success = res_dc["flag"]
        catch e
            @warn "Sample $j failed in DC with error: $e"
            j += 1
            continue
        end

        # If DC is infeasible, skip the sample
        if !dc_success
            @warn "Sample $j discarded (DC infeasible or no solution)"
            j += 1
            continue
        end

        # ================================
        # === 2) Solve AC-OPF for sample
        # ================================
        data_ac = deepcopy(data_orig)
        t_ac = 0.0
        res_ac = nothing

        try
            t0 = time()
            res_ac = ACPM(data_ac, Pd_i, Qd_i; solver=solver)
            t_ac = time() - t0
        catch e
            @warn "Sample $j failed in AC with error: $e"
            j += 1
            continue
        end

        # AC must be feasible
        if !res_ac["flag"]
            @warn "Sample $j discarded (AC infeasible)"
            j += 1
            continue
        end

        # ========================================
        # === 3) Store results of a valid sample
        # ========================================
        valid_k += 1
        Demand[valid_k, :] = vcat(Pd_i, Qd_i)
        DispatchAC[valid_k, :] = res_ac["Pg"]
        DispatchDC[valid_k, :] = res_dc["Pg"]
        TimeAC[valid_k] = t_ac
        TimeDC[valid_k] = t_dc
        ObjAC[valid_k] = res_ac["objective"]
        ObjDC[valid_k] = res_dc["objective"]

        j += 1
    end

    # === Trim output if fewer than nsamples were obtained ===
    if valid_k < nsamples
        @warn "Only $valid_k valid samples obtained out of $nsamples requested."

        Demand = Demand[1:valid_k, :]
        DispatchAC = DispatchAC[1:valid_k, :]
        DispatchDC = DispatchDC[1:valid_k, :]
        TimeAC = TimeAC[1:valid_k]
        TimeDC = TimeDC[1:valid_k]
        ObjAC = ObjAC[1:valid_k]
        ObjDC = ObjDC[1:valid_k]
    end

    # === Return all results ===
    return Dict(
        "Demand" => Demand,
        "DispatchAC" => DispatchAC,
        "DispatchDC" => DispatchDC,
        "TimeAC" => TimeAC,
        "TimeDC" => TimeDC,
        "ObjAC" => ObjAC,
        "ObjDC" => ObjDC,
        "n_valid" => valid_k
    )
end



function DCPM(data, Pd=nothing, Qd=nothing; solver=Gurobi.Optimizer)

    # === Step 1: Sort and map bus IDs ===
    # Sort all bus IDs numerically to maintain a consistent ordering.
    # Create a mapping from bus_id → index (for accessing Pd/Qd vectors easily).
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # === Step 2: Update load demands if custom values are provided ===
    # If Pd and Qd vectors are passed as arguments, update the corresponding
    # load entries in the data dictionary before running the DC-OPF.
    if Pd !== nothing && Qd !== nothing
        for (load_id, load_data) in data["load"]
            bus_id = load_data["load_bus"]
            idx = bus_idx_map[bus_id]
            data["load"][load_id]["pd"] = Pd[idx]
            data["load"][load_id]["qd"] = Qd[idx]
        end
    end

    # === Step 3: Build and solve the DC Optimal Power Flow (DC-OPF) model ===
    # Instantiate the DC model using PowerModels, based on the provided data.
    spec = PowerModels.instantiate_model(
        data,
        PowerModels.DCPPowerModel,   # DC Power Flow formulation
        PowerModels.build_opf        # Build a standard OPF problem
    )

    # Run the optimization silently (suppress solver output)
    set_silent(spec.model)
    results = PowerModels.optimize_model!(spec; optimizer=solver)

    # === Step 4: Extract generator data and results ===
    # Reference to generator metadata
    ref_gen = spec.ref[:it][:pm][:nw][0][:gen]

    # Variable references for generator active power outputs
    pg_da = spec.var[:it][:pm][:nw][0][:pg]

    # Retrieve generator IDs in sorted order to align Pg results consistently
    gen_ids_sorted = sort(collect(keys(ref_gen)))

    # Extract the optimized generator dispatch values (Pg)
    Pg_ordered = [
        JuMP.value(pg_da[g_id])
        for g_id in gen_ids_sorted
    ]

    # Display the solver termination status for debugging/logging
    println("status DCOPF ", JuMP.termination_status(spec.model))

    # === Step 5: Return all results in a dictionary ===
    return Dict(
        "flag" => (JuMP.termination_status(spec.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(spec.model),
        "Pg" => Pg_ordered,
        "results" => results,
    )
end

function ACPM(data, Pd=nothing, Qd=nothing; solver=Ipopt.Optimizer)
    # Paso 1: ordenar los IDs de buses y crear el mapa
    bus_ids = sort(parse.(Int, collect(keys(data["bus"]))))
    bus_idx_map = Dict(bus_id => i for (i, bus_id) in enumerate(bus_ids))

    # step 2  Update the loads 
    if Pd !== nothing && Qd !== nothing
        for (load_id, load_data) in data["load"]
            bus_id = load_data["load_bus"]
            idx = bus_idx_map[bus_id]
            data["load"][load_id]["pd"] = Pd[idx]
            data["load"][load_id]["qd"] = Qd[idx]
        end
    end

    # === Build the model ===
    spec = PowerModels.instantiate_model(
        data,
        PowerModels.ACRPowerModel,
        PowerModels.build_opf
    )
    set_time_limit_sec(spec.model, 1100.0)
    set_silent(spec.model)
    results = PowerModels.optimize_model!(spec; optimizer=solver)

    # Sort the generators index
    ref_gen = spec.ref[:it][:pm][:nw][0][:gen]
    pg_da = spec.var[:it][:pm][:nw][0][:pg]
    gen_ids_sorted = sort(collect(keys(ref_gen)))
    Pg_ordered = [
        JuMP.value(pg_da[g_id])
        for g_id in gen_ids_sorted
    ]

    println("satus ACOPF ", JuMP.termination_status(spec.model))
    return Dict(
        "flag" => (JuMP.termination_status(spec.model) in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)),
        "objective" => JuMP.objective_value(spec.model),
        "Pg" => Pg_ordered,
        "results" => results
    )
end






limit_sample = 100 # Number random samples 
nsamples = 100 # Number until reach lnsamples

path_data = "test_systems/data-opf"

system_name = "pglib_opf_case118_ieee"

file_name = joinpath(path_data, "$(system_name).m")

data = PowerModels.parse_file(file_name)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

P_samples, Q_samples = generate_truncated_scales(data, limit_sample)

results = run_ac_dc_batch_from_data(data, P_samples, Q_samples, nsamples)


# Save and load model (save .json)
# BSON.@save "test_systems/data_$system_name.bson" results

# k = BSON.load("test_systems/data_$system_name.bson")

# a = k["ObjAC"]
# b = k["ObjDC"]
# scatter(a, a)
# scatter!(a, b)