"""
Fair Benchmark Comparison with Multiple Reduction Targets
Runs benchmarks for 20%, 30%, 40%, 50%, and 60% cost reduction
"""

using Pkg
Pkg.activate(".")

using BSON
using Printf
using Statistics
using Random
using DataFrames
using CSV
using JuMP
const MOI = JuMP.MOI

# Load ICNN module
include(joinpath(@__DIR__, "..", "icnn", "ICNN.jl"))
using .ICNN

# Load data loader
include(joinpath(@__DIR__, "..", "icnn", "data", "dcopf_loader.jl"))

# Load counterfactual algorithms
include(joinpath(@__DIR__, "..", "counterfactuals", "model_loader.jl"))
include(joinpath(@__DIR__, "..", "counterfactuals", "algorithms", "outer_approximation.jl"))
include(joinpath(@__DIR__, "..", "counterfactuals", "algorithms", "mip_counterfactual.jl"))

println("="^80)
println("Multiple Reduction Targets Benchmark: ECP vs ESH vs MILP")
println("="^80)
println()

# Configuration
n_trials = 3
n_test_cases = 5
random_seed = 42
time_limit_milp = 600.0
reduction_factors = [0.20, 0.30, 0.40, 0.50, 0.60]  # 20%, 30%, 40%, 50%, 60%

println("Benchmark Configuration:")
println("  Trials per method per case: $n_trials")
println("  Test cases: $n_test_cases")
println("  Reduction targets: ", join([string(Int(r*100), "%") for r in reduction_factors], ", "))
println("  MILP time limit: $(time_limit_milp)s")
println()

# Load model and data ONCE
model_path = joinpath(@__DIR__, "..", "tmp", "dcopf_experiment", "best_model.bson")
data_path = joinpath(@__DIR__, "..", "icnn", "data", "data_pglib_opf_case118_ieee.bson")

println("Loading resources...")
model_data = BSON.load(model_path)
icnn_model_ref = model_data[:model]

dataset = prepare_dcopf_dataset(
    data_path;
    train_ratio=0.8,
    normalize_method=:none,
    shuffle=true,
    seed=42
)

X_test = dataset.X_test
Y_test = dataset.Y_test
Y_pred_test = predict(icnn_model_ref, X_test)

# Select diverse test cases
quantiles = [0.6, 0.7, 0.75, 0.8, 0.85]
test_indices = Int[]
for q in quantiles
    threshold = quantile(Y_pred_test[:, 1], q)
    candidates = findall(y -> y >= threshold, Y_pred_test[:, 1])
    if !isempty(candidates)
        push!(test_indices, candidates[min(3, length(candidates))])
    end
end
test_indices = unique(test_indices)[1:min(n_test_cases, end)]

println("✓ Resources loaded")
println("✓ Selected $(length(test_indices)) test cases")
println()

# Common parameters
epsilon_factor = 0.002
sparsity_weight = 0.05
max_iterations_oa = 50

x_min = minimum(X_test)
x_max = maximum(X_test)
x_bounds = (Float64(x_min), Float64(x_max + 0.5))

# Warmup
println("Warmup (compiling Julia code)...")
x_warmup = Float32.(X_test[test_indices[1], :])
y_warmup_factual = Float64(icnn_model_ref(reshape(x_warmup, 1, :))[1, 1])
y_warmup_target = y_warmup_factual * 0.6
epsilon_warmup = epsilon_factor * abs(y_warmup_target)

generate_counterfactual_oa(
    icnn_model_ref, x_warmup, y_warmup_target;
    epsilon=epsilon_warmup, sparsity_weight=sparsity_weight,
    x_bounds=x_bounds, max_iterations=max_iterations_oa,
    cut_strategy=:ecp, verbose=false
)
println("✓ Warmup complete")
println()

# Storage for all results
all_results = []

# Run benchmarks for each reduction factor
for reduction_factor in reduction_factors
    println("="^80)
    println("TESTING $(Int(reduction_factor*100))% REDUCTION")
    println("="^80)
    println()
    
    # Main benchmark loop for this reduction
    for (case_idx, test_idx) in enumerate(test_indices)
        println("-"^80)
        println("Reduction $(Int(reduction_factor*100))% - Case $case_idx/$n_test_cases (Sample #$test_idx)")
        println("-"^80)
        
        x_factual = Float32.(X_test[test_idx, :])
        y_factual = Float64(icnn_model_ref(reshape(x_factual, 1, :))[1, 1])
        y_target = y_factual * (1.0 - reduction_factor)
        epsilon = epsilon_factor * abs(y_target)
        
        println("  y_factual = $(round(y_factual, digits=2))")
        println("  y_target = $(round(y_target, digits=2)) ($(Int(reduction_factor*100))% reduction)")
        println()
        
        # Create trial schedule
        trial_schedule = []
        for trial in 1:n_trials
            push!(trial_schedule, (:ecp, trial))
            push!(trial_schedule, (:esh, trial))
            push!(trial_schedule, (:milp, trial))
        end
        
        Random.seed!(random_seed + case_idx + Int(reduction_factor*1000))
        shuffle!(trial_schedule)
        
        # Storage for this case
        case_results = Dict(:ecp => [], :esh => [], :milp => [])
        
        # Run trials
        print("  Running trials: ")
        for (method, trial) in trial_schedule
            GC.gc()
            fresh_model_data = BSON.load(model_path)
            fresh_icnn = fresh_model_data[:model]
            
            if method == :ecp
                result = generate_counterfactual_oa(
                    fresh_icnn, x_factual, y_target;
                    epsilon=epsilon, sparsity_weight=sparsity_weight,
                    x_bounds=x_bounds, max_iterations=max_iterations_oa,
                    cut_strategy=:ecp, verbose=false
                )
                push!(case_results[:ecp], result)
                status = result[:status] == :optimal ? "✓" : "✗"
                print("E$trial$(status) ")
                
            elseif method == :esh
                result = generate_counterfactual_oa(
                    fresh_icnn, x_factual, y_target;
                    epsilon=epsilon, sparsity_weight=sparsity_weight,
                    x_bounds=x_bounds, max_iterations=max_iterations_oa,
                    cut_strategy=:esh, verbose=false
                )
                push!(case_results[:esh], result)
                status = result[:status] == :optimal ? "✓" : "✗"
                print("S$trial$(status) ")
                
            elseif method == :milp
                result = generate_counterfactual(
                    fresh_icnn, x_factual, y_target;
                    epsilon=epsilon, sparsity_weight=sparsity_weight,
                    x_bounds=x_bounds, time_limit=time_limit_milp
                )
                push!(case_results[:milp], result)
                status = (result[:status] == MOI.OPTIMAL || result[:status] == MOI.FEASIBLE_POINT) ? "✓" : "✗"
                print("M$trial$(status) ")
            end
        end
        println()
        
        # Compute statistics
        function compute_stats(results, metric_key)
            function is_success(status)
                return status in [:optimal, :OPTIMAL, :FEASIBLE_POINT] || 
                       status == MOI.OPTIMAL || status == MOI.FEASIBLE_POINT
            end
            
            values = [r[metric_key] for r in results if is_success(r[:status])]
            if isempty(values)
                return (mean=NaN, median=NaN, std=NaN, min=NaN, max=NaN, n_success=0)
            end
            return (
                mean=mean(values),
                median=median(values),
                std=std(values),
                min=minimum(values),
                max=maximum(values),
                n_success=length(values)
            )
        end
        
        ecp_time_stats = compute_stats(case_results[:ecp], :solve_time)
        esh_time_stats = compute_stats(case_results[:esh], :solve_time)
        milp_time_stats = compute_stats(case_results[:milp], :solve_time)
        
        ecp_iter_stats = compute_stats(case_results[:ecp], :iterations)
        esh_iter_stats = compute_stats(case_results[:esh], :iterations)
        
        println("  Times: ECP=$(round(ecp_time_stats.mean,digits=3))s, ESH=$(round(esh_time_stats.mean,digits=3))s, MILP=$(isnan(milp_time_stats.mean) ? "timeout" : string(round(milp_time_stats.mean,digits=3),"s"))")
        println()
        
        # Store results
        push!(all_results, Dict(
            :reduction_pct => reduction_factor * 100,
            :case_idx => case_idx,
            :test_idx => test_idx,
            :y_factual => y_factual,
            :y_target => y_target,
            :ecp_time_mean => ecp_time_stats.mean,
            :ecp_time_std => ecp_time_stats.std,
            :ecp_iter_mean => ecp_iter_stats.mean,
            :ecp_success => ecp_time_stats.n_success,
            :esh_time_mean => esh_time_stats.mean,
            :esh_time_std => esh_time_stats.std,
            :esh_iter_mean => esh_iter_stats.mean,
            :esh_success => esh_time_stats.n_success,
            :milp_time_mean => milp_time_stats.mean,
            :milp_time_std => milp_time_stats.std,
            :milp_success => milp_time_stats.n_success,
            :n_trials => n_trials
        ))
    end
end

# Save all results
println("="^80)
println("SAVING RESULTS")
println("="^80)

df = DataFrame(all_results)
output_dir = joinpath(@__DIR__, "..", "tmp", "benchmark_results")
mkpath(output_dir)
csv_path = joinpath(output_dir, "multiple_reductions_results.csv")
CSV.write(csv_path, df)

println("✓ Results saved to: $csv_path")
println()

# Print summary
println("="^80)
println("SUMMARY BY REDUCTION TARGET")
println("="^80)
println()

for reduction_pct in sort(unique(df.reduction_pct))
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    
    ecp_avg = mean(df_subset.ecp_time_mean)
    esh_avg = mean(df_subset.esh_time_mean)
    milp_times = filter(!isnan, df_subset.milp_time_mean)
    
    println("$(Int(reduction_pct))% Reduction:")
    println("  ECP:  $(round(ecp_avg, digits=3))s")
    println("  ESH:  $(round(esh_avg, digits=3))s")
    if !isempty(milp_times)
        println("  MILP: $(round(mean(milp_times), digits=3))s")
    else
        println("  MILP: timeout")
    end
    println()
end

println("="^80)
println("Benchmark complete!")
println("="^80)

