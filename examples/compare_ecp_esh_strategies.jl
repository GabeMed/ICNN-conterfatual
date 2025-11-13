"""
Compare ECP vs ESH Cut Strategies for OA Counterfactual Generation

This script demonstrates the difference between the two cut generation strategies:
- Extended Cutting Plane (ECP): Standard approach, adds cuts at infeasible points
- Extended Supporting Hyperplane (ESH): Advanced approach, finds boundary points for tighter cuts

The ESH method may converge faster for problems with tight constraints, but requires
additional computational effort for bisection search.
"""

using Pkg
Pkg.activate(".")

using BSON
using Printf
using Statistics

# Load ICNN module
include(joinpath(@__DIR__, "..", "icnn", "ICNN.jl"))
using .ICNN

# Load data loader
include(joinpath(@__DIR__, "..", "icnn", "data", "dcopf_loader.jl"))

# Load counterfactual generation
include(joinpath(@__DIR__, "..", "counterfactuals", "model_loader.jl"))
include(joinpath(@__DIR__, "..", "counterfactuals", "algorithms", "outer_approximation.jl"))

println("=" ^ 80)
println("Comparing ECP vs ESH Cut Strategies for OA Counterfactuals")
println("=" ^ 80)
println()

# Load trained ICNN model
model_path = joinpath(@__DIR__, "..", "tmp", "dcopf_experiment", "best_model.bson")
println("Loading trained ICNN model from: $model_path")
model_data = BSON.load(model_path)
icnn_model = model_data[:model]
println("✓ Model loaded successfully")
println()

# Load data for test cases
data_path = joinpath(@__DIR__, "..", "icnn", "data", "data_pglib_opf_case118_ieee.bson")
println("Loading test data from: $data_path")

# Prepare dataset using same preprocessing as training
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

# Select a test case from high-cost instances (like the working example)
Y_pred_test = predict(icnn_model, X_test)
high_cost_threshold = quantile(Y_pred_test[:, 1], 0.75)
high_cost_indices = findall(y -> y >= high_cost_threshold, Y_pred_test[:, 1])
test_idx = high_cost_indices[min(5, length(high_cost_indices))]

x_factual = Float32.(X_test[test_idx, :])
y_factual_batch = icnn_model(reshape(x_factual, 1, :))
y_factual = Float64(y_factual_batch[1, 1])

println("Test Case #$test_idx (high-cost instance)")
println("  Factual prediction: y = $(round(y_factual, digits=4))")
println()

# Set target (reduce by 10% - REALISTIC TEST)
reduction_factor = 0.30  # Very aggressive to test ESH advantage
y_target = y_factual * (1.0 - reduction_factor)
println("Target: Reduce cost by $(reduction_factor * 100)%")
println("  Target value: y = $(round(y_target, digits=4))")
println()

# Common parameters (matching the working example)
epsilon = 0.002 * abs(y_target)  # Percentage-based epsilon
sparsity_weight = 0.05
max_iterations = 100

# Compute actual data bounds (CRITICAL - data is not normalized to [0,1])
x_min = minimum(X_test)
x_max = maximum(X_test)
x_bounds = (Float64(x_min), Float64(x_max + 0.5))

println("Parameters:")
println("  Epsilon: $(round(epsilon, digits=4)) (0.2% of target)")
println("  Sparsity weight: $sparsity_weight")
println("  X bounds: [$(round(x_bounds[1], digits=2)), $(round(x_bounds[2], digits=2))]")
println()

println()
println("=" ^ 80)
println("Strategy 2: Extended Supporting Hyperplane (ESH) - Advanced Method")
println("=" ^ 80)
println()

result_esh = generate_counterfactual_oa(
    icnn_model,
    x_factual,
    y_target;
    epsilon=epsilon,
    sparsity_weight=sparsity_weight,
    x_bounds=x_bounds,
    max_iterations=max_iterations,
    cut_strategy=:esh,
    verbose=true
)
println("=" ^ 80)
println("Strategy 1: Extended Cutting Plane (ECP) - Standard Method")
println("=" ^ 80)
println()

result_ecp = generate_counterfactual_oa(
    icnn_model,
    x_factual,
    y_target;
    epsilon=epsilon,
    sparsity_weight=sparsity_weight,
    x_bounds=x_bounds,
    max_iterations=max_iterations,
    cut_strategy=:ecp,
    verbose=true
)


println()
println("=" ^ 80)
println("Comparison Summary")
println("=" ^ 80)
println()

# Compare results
println(@sprintf("%-30s %-20s %-20s", "Metric", "ECP", "ESH"))
println("-" ^ 80)

# Add detailed diagnostics function
function analyze_convergence(result, strategy_name)
    println("\n" * "=" ^ 80)
    println("Detailed Analysis: $strategy_name")
    println("=" ^ 80)
    
    history = result[:iteration_history]
    
    # Count cut types
    cut_types = [log.cut_type for log in history]
    ecp_cuts = count(==(  "ECP"), cut_types)
    esh_cuts = count(==("ESH"), cut_types)
    
    println("\nCut Statistics:")
    println("  ECP cuts: $ecp_cuts")
    println("  ESH cuts: $esh_cuts")
    println("  Total iterations: $(length(history))")
    
    # Convergence trajectory
    println("\nObjective Convergence:")
    objs = [log.obj_k for log in history]
    println("  Initial: $(round(objs[1], digits=4))")
    println("  Final: $(round(objs[end], digits=4))")
    println("  Total change: $(round(objs[end] - objs[1], digits=4))")
    
    # Feasibility achievement
    feasible_iters = [i for (i, log) in enumerate(history) if log.feasible]
    if !isempty(feasible_iters)
        println("\nFeasibility:")
        println("  First feasible at iteration: $(feasible_iters[1])")
        println("  Feasible in $(length(feasible_iters))/$(length(history)) iterations")
    end
    
    # MILP solve times
    milp_times = [log.master_solve_time for log in history]
    println("\nMILP Solve Times:")
    println("  Average: $(round(mean(milp_times), digits=4))s")
    println("  Min: $(round(minimum(milp_times), digits=4))s")
    println("  Max: $(round(maximum(milp_times), digits=4))s")
    println("  Total MILP time: $(round(sum(milp_times), digits=3))s")
    
    # ESH-specific diagnostics
    if esh_cuts > 0
        bisection_times = [log.bisection_info.bisection_time for log in history if log.bisection_info !== nothing]
        if !isempty(bisection_times)
            println("\nESH Bisection Times:")
            println("  Average: $(round(mean(bisection_times), digits=4))s")
            println("  Total bisection time: $(round(sum(bisection_times), digits=3))s")
            
            # Distances
            d_ints = [log.bisection_info.dist_to_interior for log in history if log.bisection_info !== nothing]
            d_infs = [log.bisection_info.dist_to_infeasible for log in history if log.bisection_info !== nothing]
            
            println("\nESH Boundary Point Quality:")
            println("  Avg distance to interior: $(round(mean(d_ints), digits=3))")
            println("  Avg distance to infeasible: $(round(mean(d_infs), digits=3))")
        end
    end
    
    # Show convergence pattern (first 10 iterations)
    println("\nConvergence Pattern (first 10 iters):")
    println(@sprintf("%-6s %-10s %-10s %-10s %-6s %-8s", "Iter", "Objective", "f(x)", "Error", "Feas", "Δobj"))
    println("-" ^ 65)
    for i in 1:min(10, length(history))
        log = history[i]
        delta = i > 1 ? log.obj_k - history[i-1].obj_k : 0.0
        feas_str = log.feasible ? "✓" : "✗"
        @printf("%-6d %-10.4f %-10.1f %-10.4f %-6s %+.4f\n",
            i, log.obj_k, log.f_k, log.target_error, feas_str, delta)
    end
end

if result_ecp[:status] == :optimal && result_esh[:status] == :optimal
    println(@sprintf("%-30s %-20s %-20s", 
            "Status", 
            String(result_ecp[:status]), 
            String(result_esh[:status])))
    
    println(@sprintf("%-30s %-20d %-20d", 
            "Iterations", 
            result_ecp[:iterations], 
            result_esh[:iterations]))
    
    println(@sprintf("%-30s %-20.4f %-20.4f", 
            "Solve time (s)", 
            result_ecp[:solve_time], 
            result_esh[:solve_time]))
    
    println(@sprintf("%-30s %-20.6f %-20.6f", 
            "Final objective", 
            result_ecp[:upper_bound], 
            result_esh[:upper_bound]))
    
    println(@sprintf("%-30s %-20.4f %-20.4f", 
            "L1 distance", 
            result_ecp[:distance], 
            result_esh[:distance]))
    
    println(@sprintf("%-30s %-20d %-20d", 
            "Features changed", 
            result_ecp[:num_changed], 
            result_esh[:num_changed]))
    
    println(@sprintf("%-30s %-20.6f %-20.6f", 
            "Final prediction", 
            result_ecp[:prediction], 
            result_esh[:prediction]))
    
    println()
    
    # Determine which is better
    iter_improvement = result_ecp[:iterations] - result_esh[:iterations]
    time_improvement = result_ecp[:solve_time] - result_esh[:solve_time]
    
    println("Analysis:")
    if iter_improvement > 0
        println("  ✓ ESH converged $(iter_improvement) iteration(s) faster")
    elseif iter_improvement < 0
        println("  ✓ ECP converged $(-iter_improvement) iteration(s) faster")
    else
        println("  → Both methods converged in same number of iterations")
    end
    
    if time_improvement > 0
        println("  ✓ ESH was $(round(time_improvement, digits=2))s faster overall")
    elseif time_improvement < 0
        println("  ⚠ ECP was $(round(-time_improvement, digits=2))s faster overall")
        println("    (ESH bisection search adds computational overhead)")
    else
        println("  → Both methods took similar time")
    end
    
    obj_improvement = result_ecp[:upper_bound] - result_esh[:upper_bound]
    if abs(obj_improvement) < 1e-4
        println("  → Both methods achieved similar objective values")
    elseif obj_improvement > 0
        println("  ✓ ESH found better solution (lower objective by $(round(obj_improvement, digits=6)))")
    else
        println("  ✓ ECP found better solution (lower objective by $(round(-obj_improvement, digits=6)))")
    end
else
    println("One or both methods failed to find optimal solution")
    println(@sprintf("  ECP status: %s", result_ecp[:status]))
    println(@sprintf("  ESH status: %s", result_esh[:status]))
end

# Add detailed diagnostics
if result_ecp[:status] == :optimal
    analyze_convergence(result_ecp, "ECP (Extended Cutting Plane)")
end

if result_esh[:status] == :optimal
    analyze_convergence(result_esh, "ESH (Extended Supporting Hyperplane)")
end



