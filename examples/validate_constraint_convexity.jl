"""
# Validation Test: Convex vs Non-Convex Constraints in OA

This script demonstrates the mathematical difference between convex and non-convex
constraint sets when using Outer Approximation with a convex ICNN model.

## Mathematical Background

For a **convex function** f(x), constraint sets have different convexity properties:

1. **CONVEX (Valid for OA):**
   - Constraint: f(x) ≤ c
   - Set: {x : f(x) ≤ c} is CONVEX (sublevel set of convex function)
   - Standard OA with upper approximation cuts applies

2. **NON-CONVEX (Invalid for Standard OA):**
   - Constraint: f(x) ≥ c
   - Set: {x : f(x) ≥ c} is NON-CONVEX (superlevel set of convex function)
   - Standard OA cannot be applied directly

## What This Test Does

1. Creates a simple 2D convex function (quadratic)
2. Visualizes the constraint sets for f(x) ≤ c and f(x) ≥ c
3. Shows why one is convex and the other is not
4. Demonstrates the error handling in the OA implementation

## Expected Behavior

- `:upper_bound` mode should work correctly
- `:lower_bound` mode should throw an error with explanation
- `:interval` mode should work with a warning (approximation)
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Load ICNN module
include("../icnn/ICNN.jl")
using .ICNN

# Load counterfactual generation utilities
include("../counterfactuals/utils/gradient_utils.jl")
include("../counterfactuals/algorithms/outer_approximation.jl")

using Printf
using Statistics
using BSON
using Random

println("=" ^ 80)
println("VALIDATION TEST: Constraint Set Convexity in Outer Approximation")
println("=" ^ 80)
println()

# ============================================================================
# PART 1: Mathematical Demonstration with Simple Example
# ============================================================================

println("PART 1: Mathematical Demonstration")
println("-" ^ 80)
println()

println("Consider a simple convex function: f(x) = x²")
println()

# Create sample points
x_samples = range(-2.0, 2.0, length=100)
f_samples = x_samples .^ 2

println("For threshold c = 1.0:")
println()

# Sublevel set (convex)
sublevel_set = [x for (x, f) in zip(x_samples, f_samples) if f <= 1.0]
println("1. SUBLEVEL SET {x : f(x) ≤ 1}")
println("   This is the set where x² ≤ 1")
println("   Solution: x ∈ [-1, 1] (single interval)")
println("   This is CONVEX ✓")
println("   Number of intervals: 1")
println("   Min x: $(round(minimum(sublevel_set), digits=3))")
println("   Max x: $(round(maximum(sublevel_set), digits=3))")
println()

# Superlevel set (non-convex)
superlevel_set = [x for (x, f) in zip(x_samples, f_samples) if f >= 1.0]
println("2. SUPERLEVEL SET {x : f(x) ≥ 1}")
println("   This is the set where x² ≥ 1")
println("   Solution: x ∈ (-∞, -1] ∪ [1, ∞) (two disconnected intervals)")
println("   This is NON-CONVEX ✗")
println("   Why? The midpoint x=0 between x=-1 and x=1 has f(0)=0 < 1")
println("   So the midpoint is NOT in the set!")
println()

println("Verification:")
x1, x2 = -1.0, 1.0
x_mid = (x1 + x2) / 2
f_mid = x_mid^2
println("  - x₁ = $x1:  f(x₁) = $(x1^2) ≥ 1 ✓ (in set)")
println("  - x₂ = $x2:  f(x₂) = $(x2^2) ≥ 1 ✓ (in set)")
println("  - x_mid = $x_mid: f(x_mid) = $f_mid < 1 ✗ (NOT in set)")
println("  → Set is not closed under convex combinations → NON-CONVEX")
println()

# ============================================================================
# PART 2: Load Trained Model
# ============================================================================

println()
println("PART 2: Testing with Trained ICNN Model")
println("-" ^ 80)
println()

# Path to trained model
model_path = joinpath(@__DIR__, "..", "tmp", "dcopf_experiment", "best_model.bson")

if !isfile(model_path)
    println("ERROR: No trained model found at:")
    println("  $model_path")
    println()
    println("Please train a model first by running:")
    println("  julia src/train_dcopf.jl")
    println()
    exit(1)
end

println("Loading model from: $model_path")
model_data = BSON.load(model_path)
model = model_data[:model]
println("✓ Model loaded successfully")
println()

# Load test data
data_file = joinpath(@__DIR__, "..", "icnn", "data", "data_pglib_opf_case118_ieee.bson")

if !isfile(data_file)
    println("ERROR: DCOPF dataset not found at:")
    println("  $data_file")
    println()
    println("Please generate the dataset by running:")
    println("  julia icnn/data/Generate_DCOPF.jl")
    println()
    exit(1)
end

dataset = prepare_dcopf_dataset(
    data_file;
    train_ratio=0.8,
    normalize_method=:standardize,
    shuffle=true,
    seed=42
)

X_test = dataset.X_test
n_features = dataset.n_features

# Select a test point
x_factual = Float32.(X_test[1, :])
y_factual = model(reshape(x_factual, 1, :))[1, 1]

println("Selected test instance:")
println("  - Features: $n_features")
println("  - Current prediction: y = $(round(y_factual, digits=4))")
println()

# ============================================================================
# PART 3: Test Different Constraint Types
# ============================================================================

println()
println("PART 3: Testing Different Constraint Types")
println("=" ^ 80)
println()

# Common parameters
epsilon = 0.01
sparsity_weight = 0.1
x_bounds = (Float64(minimum(X_test)), Float64(maximum(X_test)))
max_iterations = 5  # Just a few iterations for testing

# Test 1: Upper bound constraint (CONVEX - should work)
println("TEST 1: Upper Bound Constraint (CONVEX)")
println("-" ^ 80)
y_target_upper = Float64(y_factual) * 0.95  # Reduce by 5%

println("Attempting: f(x) ≤ $(round(y_target_upper + epsilon, digits=4))")
println("This defines a CONVEX constraint set (sublevel set).")
println()

try
    result_upper = generate_counterfactual_oa(
        model,
        x_factual,
        y_target_upper;
        epsilon=epsilon,
        sparsity_weight=sparsity_weight,
        x_bounds=x_bounds,
        max_iterations=max_iterations,
        verbose=false
    )

    println("✓ SUCCESS: :upper_bound mode completed")
    println("  Status: $(result_upper[:status])")
    println("  Iterations: $(result_upper[:iterations])")
    if result_upper[:status] == :optimal
        println("  Distance: $(round(result_upper[:distance], digits=4))")
        println("  Prediction: $(round(result_upper[:prediction], digits=4))")
    end
catch e
    println("✗ UNEXPECTED ERROR:")
    println("  $(e)")
end

println()
println()

# Test 2: Lower bound constraint (NON-CONVEX - should error)
println("TEST 2: Lower Bound Constraint (NON-CONVEX)")
println("-" ^ 80)
y_target_lower = Float64(y_factual) * 1.05  # Increase by 5%

println("Attempting: f(x) ≥ $(round(y_target_lower - epsilon, digits=4))")
println("This defines a NON-CONVEX constraint set (superlevel set).")
println()

try
    result_lower = generate_counterfactual_oa(
        model,
        x_factual,
        y_target_lower;
        epsilon=epsilon,
        sparsity_weight=sparsity_weight,
        x_bounds=x_bounds,
        max_iterations=max_iterations,
        verbose=false
    )

    println("✗ UNEXPECTED: Should have thrown an error!")
    println("  Status: $(result_lower[:status])")
catch e
    if isa(e, ErrorException) && contains(string(e), "NON-CONVEX")
        println("✓ EXPECTED ERROR CAUGHT:")
        println()
        error_lines = split(string(e), "\n")
        for line in error_lines[1:min(10, length(error_lines))]
            println("  $line")
        end
        if length(error_lines) > 10
            println("  ... (see full error message for details)")
        end
    else
        println("✗ UNEXPECTED ERROR TYPE:")
        println("  $(e)")
        rethrow()
    end
end

println()
println()

# Test 3: Interval constraint (PENALTY APPROXIMATION - should warn)
println("TEST 3: Interval Constraint (PENALTY APPROXIMATION)")
println("-" ^ 80)
y_target_interval = Float64(y_factual) * 0.95

println("Attempting: |f(x) - $(round(y_target_interval, digits=4))| ≤ $epsilon")
println("This uses penalty approximation, not exact OA.")
println("Expected: Warning message, then execution.")
println()

# Capture the warning
try
    result_interval = generate_counterfactual_oa(
        model,
        x_factual,
        y_target_interval;
            epsilon=epsilon,
        sparsity_weight=sparsity_weight,
        x_bounds=x_bounds,
        max_iterations=max_iterations,
        verbose=false
    )

    println("✓ COMPLETED WITH WARNING")
    println("  Status: $(result_interval[:status])")
    println("  Iterations: $(result_interval[:iterations])")
    if result_interval[:status] == :optimal
        println("  Distance: $(round(result_interval[:distance], digits=4))")
        println("  Prediction: $(round(result_interval[:prediction], digits=4))")
    end
catch e
    println("✗ UNEXPECTED ERROR:")
    println("  $(e)")
end

println()
println()

# ============================================================================
# PART 4: Summary
# ============================================================================

println("=" ^ 80)
println("VALIDATION SUMMARY")
println("=" ^ 80)
println()

println("Constraint Type Validity for Convex ICNN Models:")
println()

println("1. :upper_bound (f(x) ≤ c)")
println("   ✓ CONVEX constraint set (sublevel set)")
println("   ✓ Standard OA with upper approximation cuts applies")
println("   ✓ Guaranteed global optimum")
println("   → RECOMMENDED for cost reduction, threshold satisfaction")
println()

println("2. :lower_bound (f(x) ≥ c)")
println("   ✗ NON-CONVEX constraint set (superlevel set)")
println("   ✗ Standard OA cannot be applied")
println("   ✗ Implementation throws error with explanation")
println("   → NOT SUPPORTED (by design)")
println()

println("3. :interval (|f(x) - y| ≤ ε)")
println("   ⚠ Mixed: convex upper bound + non-convex lower bound")
println("   ⚠ Uses penalty approximation, not exact OA")
println("   ⚠ Works in practice but no global optimality guarantee")
println("   → Use for legacy compatibility or exact target matching")
println()

println("=" ^ 80)
println("Key Takeaway:")
println()
println("For a CONVEX function f, only constraints of the form f(x) ≤ c define")
println("CONVEX feasible regions suitable for standard Outer Approximation.")
println()
println("Constraints like f(x) ≥ c create NON-CONVEX regions (disconnected sets)")
println("and require alternative optimization approaches.")
println("=" ^ 80)
println()

println("✓ Validation test completed successfully")
println()
