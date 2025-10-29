using Pkg

using PowerModels, JuMP, Ipopt, Gurobi
using PowerModels
using Gurobi
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


function run_dc_batch_from_data(data_orig, P_samples, Q_samples, nsamples, solver=Ipopt.Optimizer)
    # === Extract dimensions ===
    # total_available: total number of available samples (rows in P_samples)
    # nbus: number of buses (columns in P_samples)
    total_available, nbus = size(P_samples)

    # === Prepare reference model ===
    # Create a deep copy of the original data to avoid modifying it
    data0 = deepcopy(data_orig)

    # Instantiate a base AC model to determine the number of generators (ngen)
    spec0 = PowerModels.instantiate_model(data0, PowerModels.ACRPowerModel, PowerModels.build_opf)
    ngen = length(spec0.ref[:it][:pm][:nw][0][:gen])

    # === Preallocate result arrays ===
    # Demand: concatenated active (P) and reactive (Q) demand per bus
    # DispatchDC: generator dispatch solutions for each valid DC sample
    # TimeDC: elapsed time for each DC optimization run
    # ObjDC: objective value of each DC run
    Demand = zeros(nsamples, 2 * nbus)
    DispatchDC = zeros(nsamples, ngen)
    TimeDC = zeros(nsamples)
    ObjDC = zeros(nsamples)

    # valid_k: counter for successfully solved samples
    # j: index of the current sample being processed
    valid_k = 0
    j = 1

    @info "Searching for $nsamples valid samples (total available = $total_available)..."

    # === Main sampling loop ===
    while valid_k < nsamples && j <= total_available
        @info "Evaluating sample $j out of $total_available"
        Pd_i, Qd_i = P_samples[j, :], Q_samples[j, :]

        data_dc = deepcopy(data_orig)
        t1 = time()
        res_dc = nothing
        t_dc = 0.0
        dc_success = false

        try
            # Run the DC optimization model for this sample
            res_dc = DCPM(data_dc, Pd_i, Qd_i; solver=Gurobi.Optimizer)
            t_dc = time() - t1
            dc_success = res_dc["flag"]
        catch e
            # If any error occurs during the DC run, log it and continue
            @warn "Sample $j failed in DC with error: $e"
            j += 1
            continue
        end

        # Skip the sample if the DC model did not converge successfully
        if !dc_success
            @warn "Sample $j discarded (DC infeasible or no solution)"
            j += 1
            continue
        end

        # === Store results of valid samples ===
        if res_dc["flag"]
            valid_k += 1
            Demand[valid_k, :] = vcat(Pd_i, Qd_i)
            DispatchDC[valid_k, :] = res_dc["Pg"]
            TimeDC[valid_k] = t_dc
            ObjDC[valid_k] = res_dc["objective"]
        else
            @warn "Sample $j discarded (DC infeasible)"
        end
    end

    # === Final check ===
    # If fewer valid samples were found than requested, trim the arrays
    if valid_k < nsamples
        @warn "Only $valid_k valid samples obtained out of $nsamples requested."
        Demand = Demand[1:valid_k, :]
        DispatchDC = DispatchDC[1:valid_k, :]
        TimeDC = TimeDC[1:valid_k]
        ObjDC = ObjDC[1:valid_k]
    end

    # === Return all results in a dictionary ===
    return Dict(
        "Demand" => Demand,
        "DispatchDC" => DispatchDC,
        "TimeDC" => TimeDC,
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

limit_sample = 200 # Number random samples 
nsamples = 10 # Number until reach lnsamples

path_data = "test_systems/data-opf"

system_name = "pglib_opf_case118_ieee"

file_name = joinpath(path_data, "$(system_name).m")

data = PowerModels.parse_file(file_name)
PowerModels.standardize_cost_terms!(data, order=2)
PowerModels.calc_thermal_limits!(data)

P_samples, Q_samples = generate_truncated_scales(data, limit_sample)

results = run_dc_batch_from_data(data, P_samples, Q_samples, nsamples)

println("\n" * "="^70)
println("DCOPF Data Generation Summary")
println("="^70)
println("System: $system_name")
println("Samples generated: $(results["n_valid"])")
println("Demand dimension: $(size(results["Demand"], 2))")
println("Objective range: [$(minimum(results["ObjDC"])), $(maximum(results["ObjDC"]))]")
println("="^70)

# Save results to BSON file
output_file = "test_systems/data_$system_name.bson"
BSON.@save output_file results
println("\n✅ Data saved to: $output_file")

# k = BSON.load("test_systems/data_$system_name.bson")

# a = k["ObjAC"]
# b = k["ObjDC"]
# scatter(a, a)
# scatter!(a, b)