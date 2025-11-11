"""
Validation Experiment: OA Counterfactual Recovery from Known Perturbation

This experiment validates that the OA algorithm produces meaningful counterfactuals by:
1. Taking a known point Point1 from training data with f(Point1) = cost1
2. Perturbing only the first 2 features to create Point2 with f(Point2) = cost2
3. Asking: "What changes in Point2 make f(x) ≤ cost1?"
4. Expected: Algorithm should recover Point1 (or close to it) by reversing the perturbation

This provides ground truth validation: we know Point1 achieves the target cost,
so a correct algorithm should find it (or a similarly good solution).
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Load modules
include("../icnn/ICNN.jl")
using .ICNN

include("../counterfactuals/utils/gradient_utils.jl")
include("../counterfactuals/algorithms/outer_approximation.jl")

using Printf
using Plots
using Random
using Statistics
using CSV
using DataFrames

# ============================================================================
# Visualization Function (defined early to be available)
# ============================================================================
"""
    plot_convergence_2d(x1_traj, x2_traj, point1, point2, cf, c1, c2, c_cf)

Visualize OA convergence in 2D feature space.
"""
function plot_convergence_2d(x1_trajectory, x2_trajectory,
                            point1, point2, counterfactual,
                            cost1, cost2, cost_cf)

    # Create main trajectory plot
    p1 = plot(x1_trajectory, x2_trajectory,
             linewidth=2,
             alpha=0.7,
             arrow=:arrow,
             label="OA trajectory",
             xlabel="Feature 1",
             ylabel="Feature 2",
             title="Convergence in 2D Feature Space",
             legend=:best,
             grid=true,
             size=(800, 600))

    # Add markers for iterations
    scatter!(p1, x1_trajectory, x2_trajectory,
            marker=:circle,
            markersize=3,
            alpha=0.5,
            label="",
            color=:gray)

    # Mark special points
    scatter!(p1, [point2[1]], [point2[2]],
            marker=:x,
            markersize=12,
            markerstrokewidth=3,
            label="Start (Point2)",
            color=:red)

    scatter!(p1, [point1[1]], [point1[2]],
            marker=:star5,
            markersize=15,
            label="Target (Point1)",
            color=:green)

    scatter!(p1, [counterfactual[1]], [counterfactual[2]],
            marker=:diamond,
            markersize=12,
            label="Counterfactual",
            color=:blue)

    # Annotate costs
    annotate!(p1, point1[1], point1[2] + 0.1,
             text("f=$(round(cost1, digits=3))", 8, :green))
    annotate!(p1, point2[1], point2[2] - 0.1,
             text("f=$(round(cost2, digits=3))", 8, :red))
    annotate!(p1, counterfactual[1], counterfactual[2] + 0.1,
             text("f=$(round(cost_cf, digits=3))", 8, :blue))

    # Create convergence metrics plot
    iterations = 1:length(x1_trajectory)
    distances = [sqrt((x1_trajectory[i] - point1[1])^2 +
                     (x2_trajectory[i] - point1[2])^2)
                for i in iterations]

    p2 = plot(iterations, distances,
             linewidth=2,
             marker=:circle,
             label="L2 distance to Point1",
             xlabel="Iteration",
             ylabel="Distance",
             title="Convergence to Target Point",
             grid=true,
             legend=:best)

    # Combine plots
    p = plot(p1, p2, layout=(2, 1), size=(800, 1000))

    # Save plot
    output_path = "/home/gabemed/purdue/ICNN-conterfatual/tmp/validation_convergence.png"
    savefig(p, output_path)
    println("✓ Convergence plot saved to: $output_path")
    println()

    # Print trajectory statistics
    println("Trajectory Statistics:")
    println("  Total iterations: $(length(x1_trajectory))")
    println("  Initial distance: $(round(distances[1], digits=4))")
    println("  Final distance: $(round(distances[end], digits=4))")
    println("  Distance reduction: $(round((1 - distances[end]/distances[1]) * 100, digits=1))%")
    println()

    return p
end

println("=" ^ 80)
println("OA VALIDATION EXPERIMENT: PERTURBATION RECOVERY")
println("=" ^ 80)
println()
println("Objective: Verify OA can recover from a known perturbation")
println()

# ============================================================================
# STEP 1: Load Trained Model and Data
# ============================================================================
println("STEP 1: Loading trained model and dataset...")
println("-" ^ 80)

# Load trained FICNN model
model_path = "/home/gabemed/purdue/ICNN-conterfatual/tmp/dcopf_experiment/best_model.bson"
if !isfile(model_path)
    error("Model not found at $model_path. Please train a model first.")
end

model, scaler_X, scaler_Y = load_model(model_path)
println("✓ Model loaded from: $model_path")

# Load DCOPF dataset
data_path = "/home/gabemed/purdue/ICNN-conterfatual/icnn/data/data_pglib_opf_case118_ieee.bson"
dataset = prepare_dcopf_dataset(data_path; train_ratio=0.8, normalize_method=:none, seed=42)

X_train = dataset.X_train
Y_train = dataset.Y_train
X_test = dataset.X_test
Y_test = dataset.Y_test

n_features = dataset.n_features
n_train = size(X_train, 1)

println("✓ Dataset loaded:")
println("  Training samples: $n_train")
println("  Test samples: $(size(X_test, 1))")
println("  Features: $n_features")
println()

# ============================================================================
# STEP 2: Select Original Point (Point1)
# ============================================================================
println("STEP 2: Selecting original point from training data...")
println("-" ^ 80)

# Select a random point from training data (or specific index for reproducibility)
Random.seed!(42)
point1_idx = rand(1:n_train)  # Random selection
# point1_idx = 100  # Or use specific index

point1 = Float32.(X_train[point1_idx, :])
cost1 = model(reshape(point1, 1, :))[1, 1]

println("Selected Point1 (training sample #$point1_idx):")
println("  First 2 features: [$(round(point1[1], digits=4)), $(round(point1[2], digits=4))]")
println("  Model prediction: f(Point1) = $(round(cost1, digits=4))")
println()

# ============================================================================
# STEP 3: Create Perturbed Point (Point2)
# ============================================================================
println("STEP 3: Creating perturbed point...")
println("-" ^ 80)

# Copy Point1 and perturb only first 2 features
point2 = copy(point1)

# Define perturbations (adaptive based on data range)
# For standardized data, typical range is ~[-3, 3]
# We'll perturb by a moderate amount (e.g., 0.5-1.0 standard deviations)
perturbation1 = 0.8f0
perturbation2 = 0.0f0

point2[1] += perturbation1
point2[2] += perturbation2

# Evaluate cost at perturbed point
cost2 = model(reshape(point2, 1, :))[1, 1]

println("Perturbed Point2:")
println("  First 2 features: [$(round(point2[1], digits=4)), $(round(point2[2], digits=4))]")
println("  Perturbations: [$(round(perturbation1, digits=4)), $(round(perturbation2, digits=4))]")
println("  Model prediction: f(Point2) = $(round(cost2, digits=4))")
println()

# ============================================================================
# STEP 4: Setup Counterfactual Problem
# ============================================================================
println("STEP 4: Setting up counterfactual problem...")
println("-" ^ 80)

# Compute cost change from perturbation
cost_change = cost2 - cost1

# Target: Recover cost1 (or better)
# We'll set target slightly below cost1 to encourage finding Point1 or better
target_margin = min(0.05f0, abs(cost_change) * 0.1f0)  # Small margin
target_cost = Float64(cost1 - target_margin)
epsilon = 0.05  # Tolerance for target achievement

# Immutability: Lock all features EXCEPT first 2
immutable_indices = collect(3:n_features)

println("Counterfactual Question:")
println("  Starting from Point2 (perturbed), what changes to features 1-2")
println("  will achieve cost ≤ $(round(target_cost, digits=4)) ± $epsilon ?")
println()
println("  Features 1-2: Mutable (can be changed)")
println("  Features 3-$n_features: Immutable (locked to Point2 values)")
println()
println("Expected Result:")
println("  Algorithm should recover Point1 by reversing the perturbation")
println("  (or find an equally good solution)")
println()

# ============================================================================
# STEP 5: Run OA Algorithm
# ============================================================================
println("=" ^ 80)
println("STEP 5: Running OA Counterfactual Generation")
println("=" ^ 80)
println()

# Configure OA parameters
# For standardized data, typical range is ~[-3, 3], but we'll use wider bounds
# to avoid constraining the search space
oa_params = (
    epsilon = epsilon,
    sparsity_weight = 0.05,  # Low weight since we expect exactly 2 features to change  
    x_bounds = (-10.0, 10.0),  # Wide bounds for standardized data (99.9% of values)
    max_iterations = 50,
    time_limit_per_iter = 3000.0,
    tolerance = 1e-5,
    immutable_indices = immutable_indices,
    verbose = true
)

# Run OA
result = generate_counterfactual_oa(
    model,
    point2,  # Start from perturbed point (factual)
    target_cost;
    oa_params...
)

println()


# ============================================================================
# STEP 6: Analyze Results and Validate Recovery
# ============================================================================
println("=" ^ 80)
println("STEP 6: VALIDATION RESULTS")
println("=" ^ 80)
println()

if result[:status] == :optimal || result[:status] == :already_at_target
    counterfactual = result[:counterfactual]
    cf_cost = result[:prediction]

    # ========================================================================
    # Feature Recovery Analysis
    # ========================================================================
    println("Feature Recovery Analysis (first 2 features):")
    println("-" ^ 80)
    println(@sprintf("  %-20s %12s %12s %12s", "Point", "Feature 1", "Feature 2", "Cost"))
    println("-" ^ 80)
    println(@sprintf("  %-20s %12.4f %12.4f %12.4f", "Point1 (original)", point1[1], point1[2], cost1))
    println(@sprintf("  %-20s %12.4f %12.4f %12.4f", "Point2 (perturbed)", point2[1], point2[2], cost2))
    println(@sprintf("  %-20s %12.4f %12.4f %12.4f", "Counterfactual", counterfactual[1], counterfactual[2], cf_cost))
    println("-" ^ 80)
    println()

    # Compute recovery metrics
    recovery_error_1 = abs(counterfactual[1] - point1[1])
    recovery_error_2 = abs(counterfactual[2] - point1[2])
    recovery_l2 = sqrt((counterfactual[1] - point1[1])^2 + (counterfactual[2] - point1[2])^2)
    perturbation_l2 = sqrt(perturbation1^2 + perturbation2^2)

    println("Recovery Errors:")
    println("  Feature 1 error: $(round(recovery_error_1, digits=4))")
    println("  Feature 2 error: $(round(recovery_error_2, digits=4))")
    println("  L2 distance from Point1: $(round(recovery_l2, digits=4))")
    println("  Original perturbation L2: $(round(perturbation_l2, digits=4))")
    println("  Recovery ratio: $(round(recovery_l2 / perturbation_l2 * 100, digits=2))%")
    println()

    # ========================================================================
    # Immutability Constraint Verification
    # ========================================================================
    immutable_check = all(isapprox.(counterfactual[3:end], point2[3:end], atol=1e-5))
    if immutable_check
        println("✓ Immutability constraint satisfied: Features 3-$n_features unchanged")
    else
        max_violation = maximum(abs.(counterfactual[3:end] .- point2[3:end]))
        println("✗ WARNING: Immutability constraint violated!")
        println("  Maximum violation: $(round(max_violation, digits=6))")
    end
    println()

    # ========================================================================
    # Cost Achievement Verification
    # ========================================================================
    println("Cost Achievement:")
    println("  Point1 (original):    $(round(cost1, digits=4))")
    println("  Point2 (perturbed):   $(round(cost2, digits=4))")
    println("  Counterfactual:       $(round(cf_cost, digits=4))")
    println("  Target:               $(round(target_cost, digits=4)) ± $epsilon")

    cost_achieved = abs(cf_cost - target_cost) <= epsilon || cf_cost < target_cost
    if cost_achieved
        println("  ✓ Target achieved! (error: $(round(abs(cf_cost - target_cost), digits=6)))")
    else
        println("  ✗ Target not achieved (error: $(round(abs(cf_cost - target_cost), digits=6)))")
    end
    println()

    # ========================================================================
    # Overall Validation Assessment
    # ========================================================================
    println("=" ^ 80)
    println("OVERALL VALIDATION ASSESSMENT")
    println("=" ^ 80)

    # Define success criteria
    recovery_threshold = 0.3  # Allow 30% of perturbation magnitude as error
    strict_recovery = recovery_l2 / perturbation_l2 < recovery_threshold
    loose_recovery = recovery_l2 / perturbation_l2 < 0.5

    validation_passed = false

    if strict_recovery && immutable_check && cost_achieved
        println("✓✓✓ VALIDATION PASSED - EXCELLENT!")
        println("  → Algorithm successfully recovered original point Point1")
        println("  → Recovery error: $(round(recovery_l2 / perturbation_l2 * 100, digits=1))% of perturbation")
        println("  → Immutability constraints satisfied")
        println("  → Target cost achieved")
        validation_passed = true
    elseif loose_recovery && immutable_check && cost_achieved
        println("✓✓ VALIDATION PASSED - GOOD")
        println("  → Algorithm converged to near-original point")
        println("  → Recovery error: $(round(recovery_l2 / perturbation_l2 * 100, digits=1))% of perturbation")
        println("  → Immutability constraints satisfied")
        println("  → Target cost achieved")
        validation_passed = true
    elseif cost_achieved && immutable_check
        println("✓ PARTIAL SUCCESS")
        println("  → Target cost achieved")
        println("  → Immutability constraints satisfied")
        println("  → But did not fully recover Point1 (found alternative solution)")
        println("  → Recovery error: $(round(recovery_l2 / perturbation_l2 * 100, digits=1))% of perturbation")
    else
        println("⚠ VALIDATION INCONCLUSIVE")
        if !cost_achieved
            println("  ✗ Target cost not achieved")
        end
        if !immutable_check
            println("  ✗ Immutability constraints violated")
        end
        println("  → Algorithm may have limitations with this problem instance")
    end

    println()
    println("Optimization Statistics:")
    println("  Iterations: $(result[:iterations])")
    println("  Solve time: $(round(result[:solve_time], digits=2))s")
    println("  Best objective: $(round(result[:upper_bound], digits=6))")
    println("  Features changed: $(result[:num_changed]) (expected: 2)")

    # ========================================================================
    # Convergence Visualization
    # ========================================================================
    if haskey(result, :iteration_history) && length(result[:iteration_history]) > 0
        println()
        println("=" ^ 80)
        println("CONVERGENCE VISUALIZATION")
        println("=" ^ 80)
        println()

        history = result[:iteration_history]

        # Extract trajectories for first 2 features
        x1_trajectory = [h.x_k[1] for h in history]
        x2_trajectory = [h.x_k[2] for h in history]

        # Create convergence plot
        plot_convergence_2d(
            x1_trajectory, x2_trajectory,
            point1, point2, counterfactual,
            cost1, cost2, cf_cost
        )
    end

else
    # ========================================================================
    # Failure Case
    # ========================================================================
    println("✗ VALIDATION FAILED: Algorithm did not find solution")
    println("  Status: $(result[:status])")
    println("  Iterations: $(result[:iterations])")
    println("  Solve time: $(round(result[:solve_time], digits=2))s")
    println()

    if haskey(result, :iteration_history) && length(result[:iteration_history]) > 0
        println("Iteration History (last 10):")
        history = result[:iteration_history]
        n_show = min(10, length(history))
        for i in (length(history)-n_show+1):length(history)
            h = history[i]
            @printf("  Iter %2d: obj=%8.4f  UB=%8.4f  f(x)=%7.4f  err=%6.4f  %s\n",
                   h.iteration, h.obj_k, h.UB, h.f_k, h.target_error,
                   h.feasible ? "✓" : "✗")
        end
    end

    println()
    println("Possible reasons for failure:")
    println("  - Target cost too aggressive (try larger epsilon)")
    println("  - Perturbation too large (try smaller perturbations)")
    println("  - Model limitations (non-convex regions, poor approximation)")
    println("  - Algorithm parameters need tuning (penalty weights, iterations)")
end

println()
println("=" ^ 80)
println("VALIDATION EXPERIMENT COMPLETE")
println("=" ^ 80)
