"""
Detailed Benchmark with Full Counterfactual Information
Captures: factual cost, counterfactual cost, changed features with values
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
using JSON
const MOI = JuMP.MOI

# Load modules
include(joinpath(@__DIR__, "..", "icnn", "ICNN.jl"))
using .ICNN
include(joinpath(@__DIR__, "..", "icnn", "data", "dcopf_loader.jl"))
include(joinpath(@__DIR__, "..", "counterfactuals", "model_loader.jl"))
include(joinpath(@__DIR__, "..", "counterfactuals", "algorithms", "outer_approximation.jl"))
include(joinpath(@__DIR__, "..", "counterfactuals", "algorithms", "mip_counterfactual.jl"))

println("="^80)
println("Detailed Counterfactual Benchmark")
println("="^80)
println()

# Configuration
n_trials = 1  # 1 trial to get clean examples
n_test_cases = 3  # Fewer cases for detailed analysis
random_seed = 42
time_limit_milp = 600.0
reduction_factors = [0.20, 0.30, 0.40, 0.50, 0.60]

println("Configuration:")
println("  Reduction targets: ", join([string(Int(r*100), "%") for r in reduction_factors], ", "))
println("  Test cases: $n_test_cases")
println()

# Load resources
model_path = joinpath(@__DIR__, "..", "tmp", "dcopf_experiment", "best_model.bson")
data_path = joinpath(@__DIR__, "..", "icnn", "data", "data_pglib_opf_case118_ieee.bson")

model_data = BSON.load(model_path)
icnn_model_ref = model_data[:model]

dataset = prepare_dcopf_dataset(data_path; train_ratio=0.8, normalize_method=:none, shuffle=true, seed=42)
X_test = dataset.X_test
Y_test = dataset.Y_test
Y_pred_test = predict(icnn_model_ref, X_test)

# Select test cases
quantiles = [0.7, 0.8, 0.85]
test_indices = Int[]
for q in quantiles
    threshold = quantile(Y_pred_test[:, 1], q)
    candidates = findall(y -> y >= threshold, Y_pred_test[:, 1])
    if !isempty(candidates)
        push!(test_indices, candidates[3])
    end
end
test_indices = unique(test_indices)[1:min(n_test_cases, end)]

println("✓ Resources loaded, $(length(test_indices)) test cases selected")
println()

# Common parameters
epsilon_factor = 0.002
sparsity_weight = 0.05
max_iterations_oa = 50
x_min = minimum(X_test)
x_max = maximum(X_test)
x_bounds = (Float64(x_min), Float64(x_max + 0.5))

# Storage for detailed results
detailed_results = []

# Function to extract changed features
function get_changed_features(x_factual, x_cf; threshold=1e-6)
    changes = Dict{Int, Dict{String, Float64}}()
    for i in 1:length(x_factual)
        if abs(x_factual[i] - x_cf[i]) > threshold
            changes[i] = Dict(
                "original" => Float64(x_factual[i]),
                "counterfactual" => Float64(x_cf[i]),
                "delta" => Float64(x_cf[i] - x_factual[i])
            )
        end
    end
    return changes
end

# Run benchmark
for reduction_factor in reduction_factors
    println("="^80)
    println("Testing $(Int(reduction_factor*100))% Reduction")
    println("="^80)
    println()
    
    for (case_idx, test_idx) in enumerate(test_indices)
        println("Case $case_idx (reduction $(Int(reduction_factor*100))%):")
        
        x_factual = Float32.(X_test[test_idx, :])
        y_factual = Float64(icnn_model_ref(reshape(x_factual, 1, :))[1, 1])
        y_target = y_factual * (1.0 - reduction_factor)
        epsilon = epsilon_factor * abs(y_target)
        
        println("  Factual cost: $(round(y_factual, digits=2))")
        println("  Target cost: $(round(y_target, digits=2))")
        
        # Test each method
        for method in [:ecp, :esh, :milp]
            print("  Testing $method... ")
            
            try
                fresh_model_data = BSON.load(model_path)
                fresh_icnn = fresh_model_data[:model]
                
                result = if method == :milp
                    generate_counterfactual(
                        fresh_icnn, x_factual, y_target;
                        epsilon=epsilon, sparsity_weight=sparsity_weight,
                        x_bounds=x_bounds, time_limit=time_limit_milp
                    )
                else
                    generate_counterfactual_oa(
                        fresh_icnn, x_factual, y_target;
                        epsilon=epsilon, sparsity_weight=sparsity_weight,
                        x_bounds=x_bounds, max_iterations=max_iterations_oa,
                        cut_strategy=method, verbose=false
                    )
                end
                
                # Check success
                is_success = if method == :milp
                    result[:status] == MOI.OPTIMAL || result[:status] == MOI.FEASIBLE_POINT
                else
                    result[:status] == :optimal
                end
                
                if is_success && result[:counterfactual] !== nothing
                    x_cf = result[:counterfactual]
                    y_cf = Float64(fresh_icnn(reshape(x_cf, 1, :))[1, 1])
                    
                    # Get changed features
                    changed_features = get_changed_features(x_factual, x_cf)
                    n_changed = length(changed_features)
                    
                    # Extract timing breakdown
                    timing = get(result, :timing_breakdown, Dict())
                    
                    println("✓ Success")
                    println("    CF cost: $(round(y_cf, digits=2)), Changed: $n_changed features")
                    println("    Iterations: $(get(result, :iterations, 1)), Time: $(round(result[:solve_time], digits=3))s")
                    
                    # Store detailed result
                    push!(detailed_results, Dict(
                        :reduction_pct => reduction_factor * 100,
                        :case_idx => case_idx,
                        :test_idx => test_idx,
                        :method => string(method),
                        :factual_cost => y_factual,
                        :target_cost => y_target,
                        :cf_cost => y_cf,
                        :epsilon => epsilon,
                        :n_features_changed => n_changed,
                        :solve_time => result[:solve_time],
                        :iterations => get(result, :iterations, 1),
                        :changed_features => JSON.json(changed_features),
                        :success => true,
                        :timing_breakdown => JSON.json(timing)
                    ))
                else
                    println("✗ Failed")
                    timing = get(result, :timing_breakdown, Dict())
                    push!(detailed_results, Dict(
                        :reduction_pct => reduction_factor * 100,
                        :case_idx => case_idx,
                        :test_idx => test_idx,
                        :method => string(method),
                        :factual_cost => y_factual,
                        :target_cost => y_target,
                        :cf_cost => missing,
                        :epsilon => epsilon,
                        :n_features_changed => missing,
                        :solve_time => result[:solve_time],
                        :iterations => get(result, :iterations, missing),
                        :changed_features => "{}",
                        :success => false,
                        :timing_breakdown => JSON.json(timing)
                    ))
                end
                
            catch e
                println("✗ Error: $e")
                push!(detailed_results, Dict(
                    :reduction_pct => reduction_factor * 100,
                    :case_idx => case_idx,
                    :test_idx => test_idx,
                    :method => string(method),
                    :factual_cost => y_factual,
                    :target_cost => y_target,
                    :cf_cost => missing,
                    :epsilon => epsilon,
                    :n_features_changed => missing,
                    :solve_time => missing,
                    :iterations => missing,
                    :changed_features => "{}",
                    :success => false,
                    :timing_breakdown => "{}"
                ))
            end
        end
        println()
    end
end

# Save results
println("="^80)
println("Saving detailed results...")
println("="^80)

df = DataFrame(detailed_results)
output_dir = joinpath(@__DIR__, "..", "tmp", "benchmark_results")
mkpath(output_dir)

csv_path = joinpath(output_dir, "detailed_counterfactuals.csv")
CSV.write(csv_path, df)

println("✓ Results saved to: $csv_path")
println("  Total entries: $(nrow(df))")
println("  Successful: $(sum(df.success))/$(nrow(df))")
println()
println("="^80)
println("Benchmark complete!")
println("="^80)

