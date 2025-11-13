"""
Benchmark Comparison: ECP vs ESH vs MILP

Tests all three counterfactual generation methods on multiple problem instances:
- ECP (Extended Cutting Plane): Standard OA method
- ESH (Extended Supporting Hyperplane): Advanced OA with tighter cuts
- MILP: Full mixed-integer programming with complete NN encoding

Collects statistics: solve time, iterations, distance, quality, success rate
"""

using Pkg
Pkg.activate(".")

using BSON
using Printf
using Statistics
using DataFrames
using CSV

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
println("Comprehensive Benchmark: ECP vs ESH vs MILP")
println("="^80)
println()

# Load trained ICNN model
model_path = joinpath(@__DIR__, "..", "tmp", "dcopf_experiment", "best_model.bson")
println("Loading model: $model_path")
model_data = BSON.load(model_path)
icnn_model = model_data[:model]
println("✓ Model loaded")
println()

# Load test data
data_path = joinpath(@__DIR__, "..", "icnn", "data", "data_pglib_opf_case118_ieee.bson")
println("Loading data: $data_path")
dataset = prepare_dcopf_dataset(
    data_path;
    train_ratio=0.8,
    normalize_method=:none,
    shuffle=true,
    seed=42
)

X_test = dataset.X_test
Y_test = dataset.Y_test
n_test = size(X_test, 1)
println("✓ Loaded $n_test test samples")
println()

# Select diverse test cases
Y_pred_test = predict(icnn_model, X_test)

# Get test indices from different cost quantiles
quantiles = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
test_indices = Int[]

for q in quantiles
    threshold = quantile(Y_pred_test[:, 1], q)
    candidates = findall(y -> y >= threshold, Y_pred_test[:, 1])
    if !isempty(candidates)
        push!(test_indices, candidates[min(3, length(candidates))])
    end
end

# Remove duplicates
test_indices = unique(test_indices)
n_cases = length(test_indices)

println("Selected $n_cases test cases from cost quantiles: $(quantiles)")
println()

# Common parameters
epsilon_factor = 0.002  # 0.2% of target
sparsity_weight = 0.05
max_iterations_oa = 50
time_limit_milp = 300.0  # 5 minutes for MILP

# Data bounds
x_min = minimum(X_test)
x_max = maximum(X_test)
x_bounds = (Float64(x_min), Float64(x_max + 0.5))

# Target reduction
reduction_factor = 0.40  # 40% cost reduction

println("Benchmark Parameters:")
println("  Test cases: $n_cases")
println("  Reduction target: $(reduction_factor * 100)%")
println("  Epsilon: $(epsilon_factor * 100)% of target")
println("  Sparsity weight: $sparsity_weight")
println("  OA max iterations: $max_iterations_oa")
println("  MILP time limit: $(time_limit_milp)s")
println("  X bounds: [$(round(x_bounds[1], digits=2)), $(round(x_bounds[2], digits=2))]")
println()

# Storage for results
results = []

# Run benchmark on each test case
for (case_num, test_idx) in enumerate(test_indices)
    println("="^80)
    println("Test Case $case_num/$n_cases (Sample #$test_idx)")
    println("="^80)
    
    # Get factual point
    x_factual = Float32.(X_test[test_idx, :])
    y_factual = Float64(icnn_model(reshape(x_factual, 1, :))[1, 1])
    y_target = y_factual * (1.0 - reduction_factor)
    epsilon = epsilon_factor * abs(y_target)
    
    println("  y_factual = $(round(y_factual, digits=2))")
    println("  y_target = $(round(y_target, digits=2))")
    println("  Reduction: $(reduction_factor * 100)%")
    println("  Epsilon: $(round(epsilon, digits=4))")
    println()
    
    # Initialize result dict for this case
    case_result = Dict{Symbol, Any}(
        :case_num => case_num,
        :test_idx => test_idx,
        :y_factual => y_factual,
        :y_target => y_target,
        :epsilon => epsilon
    )
    
    # Test 1: ECP
    println("-"^80)
    println("Method 1/3: ECP (Extended Cutting Plane)")
    println("-"^80)
    
    try
        result_ecp = generate_counterfactual_oa(
            icnn_model,
            x_factual,
            y_target;
            epsilon=epsilon,
            sparsity_weight=sparsity_weight,
            x_bounds=x_bounds,
            max_iterations=max_iterations_oa,
            cut_strategy=:ecp,
            verbose=false
        )
        
        case_result[:ecp_status] = result_ecp[:status]
        case_result[:ecp_time] = result_ecp[:solve_time]
        case_result[:ecp_iterations] = result_ecp[:iterations]
        case_result[:ecp_distance] = result_ecp[:distance]
        case_result[:ecp_num_changed] = result_ecp[:num_changed]
        case_result[:ecp_prediction] = result_ecp[:prediction]
        case_result[:ecp_objective] = result_ecp[:upper_bound]
        case_result[:ecp_cuts] = result_ecp[:ecp_cuts]
        
        println("  Status: $(result_ecp[:status])")
        println("  Time: $(round(result_ecp[:solve_time], digits=2))s")
        println("  Iterations: $(result_ecp[:iterations])")
        if result_ecp[:status] == :optimal
            println("  Distance: $(round(result_ecp[:distance], digits=4))")
            println("  Changed: $(result_ecp[:num_changed]) features")
            println("  Prediction: $(round(result_ecp[:prediction], digits=2))")
        end
    catch e
        println("  ✗ ERROR: $e")
        case_result[:ecp_status] = :error
        case_result[:ecp_time] = NaN
        case_result[:ecp_iterations] = NaN
        case_result[:ecp_distance] = Inf
        case_result[:ecp_num_changed] = NaN
        case_result[:ecp_prediction] = NaN
        case_result[:ecp_objective] = Inf
        case_result[:ecp_cuts] = NaN
    end
    println()
    
    # Test 2: ESH
    println("-"^80)
    println("Method 2/3: ESH (Extended Supporting Hyperplane)")
    println("-"^80)
    
    try
        result_esh = generate_counterfactual_oa(
            icnn_model,
            x_factual,
            y_target;
            epsilon=epsilon,
            sparsity_weight=sparsity_weight,
            x_bounds=x_bounds,
            max_iterations=max_iterations_oa,
            cut_strategy=:esh,
            verbose=false
        )
        
        case_result[:esh_status] = result_esh[:status]
        case_result[:esh_time] = result_esh[:solve_time]
        case_result[:esh_iterations] = result_esh[:iterations]
        case_result[:esh_distance] = result_esh[:distance]
        case_result[:esh_num_changed] = result_esh[:num_changed]
        case_result[:esh_prediction] = result_esh[:prediction]
        case_result[:esh_objective] = result_esh[:upper_bound]
        case_result[:esh_cuts] = result_esh[:esh_cuts]
        
        println("  Status: $(result_esh[:status])")
        println("  Time: $(round(result_esh[:solve_time], digits=2))s")
        println("  Iterations: $(result_esh[:iterations])")
        if result_esh[:status] == :optimal
            println("  Distance: $(round(result_esh[:distance], digits=4))")
            println("  Changed: $(result_esh[:num_changed]) features")
            println("  Prediction: $(round(result_esh[:prediction], digits=2))")
        end
    catch e
        println("  ✗ ERROR: $e")
        case_result[:esh_status] = :error
        case_result[:esh_time] = NaN
        case_result[:esh_iterations] = NaN
        case_result[:esh_distance] = Inf
        case_result[:esh_num_changed] = NaN
        case_result[:esh_prediction] = NaN
        case_result[:esh_objective] = Inf
        case_result[:esh_cuts] = NaN
    end
    println()
    
    # Test 3: MILP
    println("-"^80)
    println("Method 3/3: MILP (Full Mixed-Integer Programming)")
    println("-"^80)
    
    try
        result_milp = generate_counterfactual(
            icnn_model,
            x_factual,
            y_target;
            epsilon=epsilon,
            sparsity_weight=sparsity_weight,
            x_bounds=x_bounds,
            time_limit=time_limit_milp
        )
        
        case_result[:milp_status] = result_milp[:status]
        case_result[:milp_time] = result_milp[:solve_time]
        case_result[:milp_distance] = result_milp[:distance]
        case_result[:milp_num_changed] = result_milp[:num_changed]
        case_result[:milp_prediction] = result_milp[:prediction]
        
        println("  Status: $(result_milp[:status])")
        println("  Time: $(round(result_milp[:solve_time], digits=2))s")
        if result_milp[:status] in [:OPTIMAL, :FEASIBLE_POINT, :optimal]
            println("  Distance: $(round(result_milp[:distance], digits=4))")
            println("  Changed: $(result_milp[:num_changed]) features")
            println("  Prediction: $(round(result_milp[:prediction], digits=2))")
        end
    catch e
        println("  ✗ ERROR: $e")
        case_result[:milp_status] = :error
        case_result[:milp_time] = NaN
        case_result[:milp_distance] = Inf
        case_result[:milp_num_changed] = NaN
        case_result[:milp_prediction] = NaN
    end
    println()
    
    push!(results, case_result)
end

# Create summary statistics
println("="^80)
println("BENCHMARK SUMMARY")
println("="^80)
println()

# Convert to DataFrame for analysis
df = DataFrame(results)

# Count successes
ecp_success = count(x -> x in [:optimal, :OPTIMAL], df.ecp_status)
esh_success = count(x -> x in [:optimal, :OPTIMAL], df.esh_status)
# MILP uses JuMP termination status enum
milp_success = count(x -> string(x) in ["OPTIMAL", "FEASIBLE_POINT"], string.(df.milp_status))

println("Success Rate:")
println(@sprintf("  ECP:  %2d/%2d (%.1f%%)", ecp_success, n_cases, 100*ecp_success/n_cases))
println(@sprintf("  ESH:  %2d/%2d (%.1f%%)", esh_success, n_cases, 100*esh_success/n_cases))
println(@sprintf("  MILP: %2d/%2d (%.1f%%)", milp_success, n_cases, 100*milp_success/n_cases))
println()

# Filter to successful cases for comparison
# MILP uses JuMP TerminationStatusCode enum, so we need to check string representation
milp_success_mask = [string(x) in ["OPTIMAL", "FEASIBLE_POINT"] for x in df.milp_status]

df_success = df[
    (df.ecp_status .== :optimal) .& 
    (df.esh_status .== :optimal) .& 
    milp_success_mask,
    :
]
n_success = nrow(df_success)

if n_success > 0
    println("Statistics on $n_success successful cases:")
    println()
    
    # Solve time comparison
    println("Solve Time (seconds):")
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7.2f  Max: %7.2f",
        "ECP", 
        mean(df_success.ecp_time),
        median(df_success.ecp_time),
        minimum(df_success.ecp_time),
        maximum(df_success.ecp_time)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7.2f  Max: %7.2f",
        "ESH",
        mean(df_success.esh_time),
        median(df_success.esh_time),
        minimum(df_success.esh_time),
        maximum(df_success.esh_time)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7.2f  Max: %7.2f",
        "MILP",
        mean(df_success.milp_time),
        median(df_success.milp_time),
        minimum(df_success.milp_time),
        maximum(df_success.milp_time)))
    println()
    
    # Iterations comparison (OA methods only)
    println("Iterations (OA methods only):")
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7d  Max: %7d",
        "ECP",
        mean(df_success.ecp_iterations),
        median(df_success.ecp_iterations),
        minimum(df_success.ecp_iterations),
        maximum(df_success.ecp_iterations)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7d  Max: %7d",
        "ESH",
        mean(df_success.esh_iterations),
        median(df_success.esh_iterations),
        minimum(df_success.esh_iterations),
        maximum(df_success.esh_iterations)))
    println()
    
    # Distance comparison
    println("L1 Distance:")
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7.2f  Max: %7.2f",
        "ECP",
        mean(df_success.ecp_distance),
        median(df_success.ecp_distance),
        minimum(df_success.ecp_distance),
        maximum(df_success.ecp_distance)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7.2f  Max: %7.2f",
        "ESH",
        mean(df_success.esh_distance),
        median(df_success.esh_distance),
        minimum(df_success.esh_distance),
        maximum(df_success.esh_distance)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7.2f  Max: %7.2f",
        "MILP",
        mean(df_success.milp_distance),
        median(df_success.milp_distance),
        minimum(df_success.milp_distance),
        maximum(df_success.milp_distance)))
    println()
    
    # Features changed
    println("Features Changed:")
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7d  Max: %7d",
        "ECP",
        mean(df_success.ecp_num_changed),
        median(df_success.ecp_num_changed),
        minimum(df_success.ecp_num_changed),
        maximum(df_success.ecp_num_changed)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7d  Max: %7d",
        "ESH",
        mean(df_success.esh_num_changed),
        median(df_success.esh_num_changed),
        minimum(df_success.esh_num_changed),
        maximum(df_success.esh_num_changed)))
    
    println(@sprintf("  %-10s  Mean: %7.2f  Median: %7.2f  Min: %7d  Max: %7d",
        "MILP",
        mean(df_success.milp_num_changed),
        median(df_success.milp_num_changed),
        minimum(df_success.milp_num_changed),
        maximum(df_success.milp_num_changed)))
    println()
    
    # Speedup analysis
    println("Speedup vs MILP:")
    ecp_speedup = df_success.milp_time ./ df_success.ecp_time
    esh_speedup = df_success.milp_time ./ df_success.esh_time
    
    println(@sprintf("  ECP:  Mean: %.2fx  Median: %.2fx  Min: %.2fx  Max: %.2fx",
        mean(ecp_speedup), median(ecp_speedup), minimum(ecp_speedup), maximum(ecp_speedup)))
    
    println(@sprintf("  ESH:  Mean: %.2fx  Median: %.2fx  Min: %.2fx  Max: %.2fx",
        mean(esh_speedup), median(esh_speedup), minimum(esh_speedup), maximum(esh_speedup)))
    println()
    
    # ESH vs ECP comparison
    println("ESH vs ECP:")
    time_improvement = df_success.ecp_time .- df_success.esh_time
    iter_improvement = df_success.ecp_iterations .- df_success.esh_iterations
    
    esh_faster = count(x -> x > 0, time_improvement)
    esh_fewer_iters = count(x -> x > 0, iter_improvement)
    
    println(@sprintf("  ESH faster: %d/%d cases (%.1f%%)", 
        esh_faster, n_success, 100*esh_faster/n_success))
    println(@sprintf("  ESH fewer iterations: %d/%d cases (%.1f%%)",
        esh_fewer_iters, n_success, 100*esh_fewer_iters/n_success))
    println(@sprintf("  Mean time improvement: %+.2fs", mean(time_improvement)))
    println(@sprintf("  Mean iteration reduction: %+.2f", mean(iter_improvement)))
    println()
end

# Save results
output_dir = joinpath(@__DIR__, "..", "tmp", "benchmark_results")
mkpath(output_dir)

# Replace nothing with missing for CSV compatibility
df_csv = copy(df)
for col in names(df_csv)
    df_csv[!, col] = replace(df_csv[!, col], nothing => missing)
end

csv_path = joinpath(output_dir, "benchmark_all_methods.csv")
CSV.write(csv_path, df_csv)
println("Results saved to: $csv_path")

# Create detailed comparison table
println()
println("="^80)
println("DETAILED CASE-BY-CASE COMPARISON")
println("="^80)
println()

println(@sprintf("%-6s │ %8s %8s %8s │ %7s %7s %7s │ %6s %6s",
    "Case", "ECP(s)", "ESH(s)", "MILP(s)", "ECP(it)", "ESH(it)", "ECP-ESH", "ECP/ML", "ESH/ML"))
println("─"^80)

for row in eachrow(df)
    milp_ok = string(row.milp_status) in ["OPTIMAL", "FEASIBLE_POINT"]
    
    if row.ecp_status == :optimal && row.esh_status == :optimal && milp_ok
        
        iter_diff = row.ecp_iterations - row.esh_iterations
        ecp_speedup = row.milp_time / row.ecp_time
        esh_speedup = row.milp_time / row.esh_time
        
        println(@sprintf("%-6d │ %8.2f %8.2f %8.2f │ %7d %7d %+7d │ %6.1fx %6.1fx",
            row.case_num,
            row.ecp_time, row.esh_time, row.milp_time,
            row.ecp_iterations, row.esh_iterations, iter_diff,
            ecp_speedup, esh_speedup))
    else
        # Show failed cases
        ecp_str = row.ecp_status == :optimal ? @sprintf("%8.2f", row.ecp_time) : "  FAILED"
        esh_str = row.esh_status == :optimal ? @sprintf("%8.2f", row.esh_time) : "  FAILED"
        milp_str = milp_ok ? @sprintf("%8.2f", row.milp_time) : "  FAILED"
        
        println(@sprintf("%-6d │ %8s %8s %8s │ %7s %7s %7s │ %6s %6s",
            row.case_num, ecp_str, esh_str, milp_str, "-", "-", "-", "-", "-"))
    end
end

println()
println("="^80)
println("Benchmark Complete!")
println("="^80)

