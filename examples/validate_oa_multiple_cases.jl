"""
Multiple Test Cases Validation: OA Counterfactual Recovery

This script runs the perturbation recovery experiment on multiple test cases
to get comprehensive validation statistics.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Load modules
include("../icnn/ICNN.jl")
using .ICNN

include("../counterfactuals/utils/gradient_utils.jl")
include("../counterfactuals/algorithms/outer_approximation.jl")

using Printf
using Random
using Statistics

println("=" ^ 80)
println("OA VALIDATION: MULTIPLE TEST CASES")
println("=" ^ 80)
println()

# ============================================================================
# Load Model and Data
# ============================================================================
println("Loading trained model and dataset...")

model_path = "/home/gabemed/purdue/ICNN-conterfatual/tmp/dcopf_experiment/best_model.bson"
model = load_model(model_path)

data_path = "/home/gabemed/purdue/ICNN-conterfatual/icnn/data/data_pglib_opf_case118_ieee.bson"
dataset = prepare_dcopf_dataset(data_path; train_ratio=0.8, normalize_method=:standardize, seed=42)

X_train = dataset.X_train
Y_train = dataset.Y_train
n_features = dataset.n_features
n_train = size(X_train, 1)

println("✓ Model and data loaded")
println()

# ============================================================================
# Test Configuration
# ============================================================================
n_test_cases = 5
Random.seed!(123)  # Different seed for test case selection

# Different perturbation magnitudes to test
perturbation_configs = [
    (p1=0.5f0, p2=-0.3f0, name="Small"),
    (p1=0.8f0, p2=-0.6f0, name="Medium"),
    (p1=1.2f0, p2=-0.9f0, name="Large"),
]

println("Test Configuration:")
println("  Number of test cases per perturbation: $n_test_cases")
println("  Perturbation magnitudes: $(length(perturbation_configs))")
println("  Total tests: $(n_test_cases * length(perturbation_configs))")
println()

# ============================================================================
# Run Validation Tests
# ============================================================================
results_summary = []

for (config_idx, config) in enumerate(perturbation_configs)
    println("=" ^ 80)
    println("PERTURBATION CONFIG $(config_idx): $(config.name)")
    println("  Perturbations: [$(config.p1), $(config.p2)]")
    println("=" ^ 80)
    println()

    for test_idx in 1:n_test_cases
        println("-" ^ 80)
        println("Test Case $(test_idx)/$(n_test_cases)")
        println("-" ^ 80)

        # Select random training point
        point1_idx = rand(1:n_train)
        point1 = Float32.(X_train[point1_idx, :])
        cost1 = model(reshape(point1, 1, :))[1, 1]

        # Create perturbed point
        point2 = copy(point1)
        point2[1] += config.p1
        point2[2] += config.p2
        cost2 = model(reshape(point2, 1, :))[1, 1]

        println("Point1: f = $(round(cost1, digits=4))")
        println("Point2: f = $(round(cost2, digits=4)) (Δ = $(round(cost2 - cost1, digits=4)))")

        # Setup counterfactual problem
        target_margin = min(0.05f0, abs(cost2 - cost1) * 0.1f0)
        target_cost = Float64(cost1 - target_margin)
        epsilon = 0.05
        immutable_indices = collect(3:n_features)

        # Run OA (silently)
        result = generate_counterfactual_oa(
            model,
            point2,
            target_cost;
            epsilon=epsilon,
            sparsity_weight=0.05,
            x_bounds=(-10.0, 10.0),
            max_iterations=50,
            time_limit_per_iter=30.0,
            tolerance=1e-5,
            immutable_indices=immutable_indices,
            verbose=false  # Suppress output
        )

        # Analyze results
        if result[:status] == :optimal || result[:status] == :already_at_target
            cf = result[:counterfactual]
            cf_cost = result[:prediction]

            # Compute metrics
            recovery_error_1 = abs(cf[1] - point1[1])
            recovery_error_2 = abs(cf[2] - point1[2])
            recovery_l2 = sqrt((cf[1] - point1[1])^2 + (cf[2] - point1[2])^2)
            perturbation_l2 = sqrt(config.p1^2 + config.p2^2)
            recovery_ratio = recovery_l2 / perturbation_l2

            cost_achieved = abs(cf_cost - target_cost) <= epsilon || cf_cost < target_cost
            immutable_check = all(isapprox.(cf[3:end], point2[3:end], atol=1e-5))

            # Store results
            test_result = (
                config_name = config.name,
                test_idx = test_idx,
                point1_idx = point1_idx,
                cost1 = cost1,
                cost2 = cost2,
                cf_cost = cf_cost,
                target_cost = target_cost,
                recovery_error_1 = recovery_error_1,
                recovery_error_2 = recovery_error_2,
                recovery_l2 = recovery_l2,
                perturbation_l2 = perturbation_l2,
                recovery_ratio = recovery_ratio,
                cost_achieved = cost_achieved,
                immutable_check = immutable_check,
                iterations = result[:iterations],
                solve_time = result[:solve_time],
                num_changed = result[:num_changed],
                status = result[:status]
            )
            push!(results_summary, test_result)

            # Print summary
            status_str = cost_achieved && immutable_check ? "✓ PASS" : "⚠ PARTIAL"
            println("Result: $status_str")
            println("  Recovery ratio: $(round(recovery_ratio * 100, digits=1))%")
            println("  Cost achieved: $(cost_achieved ? "✓" : "✗")")
            println("  Immutability: $(immutable_check ? "✓" : "✗")")
            println("  Features changed: $(result[:num_changed])")
            println("  Iterations: $(result[:iterations]), Time: $(round(result[:solve_time], digits=2))s")
        else
            println("Result: ✗ FAIL - Status: $(result[:status])")

            test_result = (
                config_name = config.name,
                test_idx = test_idx,
                point1_idx = point1_idx,
                cost1 = cost1,
                cost2 = cost2,
                cf_cost = nothing,
                target_cost = target_cost,
                recovery_error_1 = nothing,
                recovery_error_2 = nothing,
                recovery_l2 = nothing,
                perturbation_l2 = sqrt(config.p1^2 + config.p2^2),
                recovery_ratio = nothing,
                cost_achieved = false,
                immutable_check = false,
                iterations = result[:iterations],
                solve_time = result[:solve_time],
                num_changed = nothing,
                status = result[:status]
            )
            push!(results_summary, test_result)
        end

        println()
    end
end

# ============================================================================
# Aggregate Statistics
# ============================================================================
println("=" ^ 80)
println("AGGREGATE VALIDATION RESULTS")
println("=" ^ 80)
println()

# Filter successful cases
successful = filter(r -> r.status == :optimal || r.status == :already_at_target, results_summary)
n_successful = length(successful)
n_total = length(results_summary)

println("Overall Success Rate:")
println("  Successful: $n_successful / $n_total ($(round(n_successful/n_total*100, digits=1))%)")
println()

if n_successful > 0
    # Statistics per perturbation magnitude
    println("Statistics by Perturbation Magnitude:")
    println("-" ^ 80)
    for config in perturbation_configs
        config_results = filter(r -> r.config_name == config.name, successful)
        n_config = length(config_results)

        if n_config > 0
            avg_recovery = mean([r.recovery_ratio for r in config_results])
            avg_iterations = mean([r.iterations for r in config_results])
            avg_time = mean([r.solve_time for r in config_results])
            avg_changed = mean([r.num_changed for r in config_results])
            n_cost_achieved = count(r -> r.cost_achieved, config_results)
            n_immutable = count(r -> r.immutable_check, config_results)

            println("\n$(config.name) Perturbation ($(config.p1), $(config.p2)):")
            println("  Success rate: $n_config / $n_test_cases")
            println("  Avg recovery ratio: $(round(avg_recovery * 100, digits=1))%")
            println("  Avg iterations: $(round(avg_iterations, digits=1))")
            println("  Avg solve time: $(round(avg_time, digits=2))s")
            println("  Avg features changed: $(round(avg_changed, digits=1))")
            println("  Cost achieved: $n_cost_achieved / $n_config")
            println("  Immutability satisfied: $n_immutable / $n_config")
        else
            println("\n$(config.name) Perturbation: No successful cases")
        end
    end
    println()

    # Overall statistics
    println("Overall Statistics (successful cases):")
    println("-" ^ 80)
    avg_recovery = mean([r.recovery_ratio for r in successful])
    median_recovery = median([r.recovery_ratio for r in successful])
    min_recovery = minimum([r.recovery_ratio for r in successful])
    max_recovery = maximum([r.recovery_ratio for r in successful])

    println("Recovery Ratio (distance from Point1 / perturbation magnitude):")
    println("  Mean:   $(round(avg_recovery * 100, digits=1))%")
    println("  Median: $(round(median_recovery * 100, digits=1))%")
    println("  Min:    $(round(min_recovery * 100, digits=1))%")
    println("  Max:    $(round(max_recovery * 100, digits=1))%")
    println()

    avg_time = mean([r.solve_time for r in successful])
    avg_iters = mean([r.iterations for r in successful])
    println("Performance:")
    println("  Avg solve time: $(round(avg_time, digits=2))s")
    println("  Avg iterations: $(round(avg_iters, digits=1))")
    println()

    n_cost_achieved = count(r -> r.cost_achieved, successful)
    n_immutable = count(r -> r.immutable_check, successful)
    println("Constraint Satisfaction:")
    println("  Cost target achieved: $n_cost_achieved / $n_successful ($(round(n_cost_achieved/n_successful*100, digits=1))%)")
    println("  Immutability satisfied: $n_immutable / $n_successful ($(round(n_immutable/n_successful*100, digits=1))%)")
end

println()
println("=" ^ 80)
println("VALIDATION ASSESSMENT")
println("=" ^ 80)

if n_successful == n_total
    println("✓✓✓ EXCELLENT: All test cases succeeded!")
elseif n_successful >= 0.8 * n_total
    println("✓✓ GOOD: Most test cases succeeded (≥80%)")
elseif n_successful >= 0.5 * n_total
    println("✓ ACCEPTABLE: Majority of test cases succeeded (≥50%)")
else
    println("⚠ NEEDS IMPROVEMENT: Less than half of test cases succeeded")
end

if n_successful > 0
    avg_recovery = mean([r.recovery_ratio for r in successful if r.recovery_ratio !== nothing])
    if avg_recovery < 0.3
        println("✓✓✓ EXCELLENT RECOVERY: Algorithm closely recovers original points (<30% error)")
    elseif avg_recovery < 0.5
        println("✓✓ GOOD RECOVERY: Algorithm reasonably recovers original points (<50% error)")
    elseif avg_recovery < 0.8
        println("✓ MODERATE RECOVERY: Algorithm finds solutions but not full recovery (<80% error)")
    else
        println("⚠ WEAK RECOVERY: Algorithm finds alternative solutions (≥80% error)")
    end

    n_cost_achieved = count(r -> r.cost_achieved, successful)
    if n_cost_achieved == n_successful
        println("✓✓✓ PERFECT COST ACHIEVEMENT: All solutions meet target cost")
    elseif n_cost_achieved >= 0.9 * n_successful
        println("✓✓ EXCELLENT COST ACHIEVEMENT: Almost all solutions meet target (≥90%)")
    else
        println("⚠ INCONSISTENT COST ACHIEVEMENT: Some solutions miss target")
    end
end

println()
println("Conclusion:")
if n_successful >= 0.8 * n_total && n_successful > 0
    avg_recovery = mean([r.recovery_ratio for r in successful if r.recovery_ratio !== nothing])
    if avg_recovery < 0.5
        println("  The OA algorithm is working correctly and produces meaningful")
        println("  counterfactuals that recover the original points or find")
        println("  equally good alternatives.")
    else
        println("  The OA algorithm successfully finds solutions that meet the")
        println("  target cost, though it may find alternative solutions rather")
        println("  than fully recovering the original points. This is valid behavior")
        println("  for optimization - multiple solutions can achieve the same cost.")
    end
else
    println("  The OA algorithm may need tuning or the problem instances may be")
    println("  challenging. Consider adjusting penalty weights, tolerances, or")
    println("  testing with different perturbation magnitudes.")
end

println()
println("=" ^ 80)
println("VALIDATION COMPLETE")
println("=" ^ 80)
