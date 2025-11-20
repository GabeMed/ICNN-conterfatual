"""
Test script to verify StandardNN fixes.

This tests:
1. StandardNN has correct fields (input_layer, hidden_layers)
2. Forward pass works correctly
3. Model is compatible with trainer's collect_metrics=true
"""

using Pkg
Pkg.activate(".")

include("ICNN.jl")
using .ICNN

using Random
using Printf
using Statistics

println("=" ^ 70)
println("Testing StandardNN Fixes")
println("=" ^ 70)

# ============================================================================
# Test 1: Model Structure
# ============================================================================
println("\n[Test 1] Verifying Model Structure")
println("-" ^ 70)

n_features = 10
n_output = 1
hidden_sizes = [20, 20]

model = StandardNN(n_features, n_output; hidden_sizes=hidden_sizes)

# Check fields exist
@assert hasfield(StandardNN, :input_layer) "Missing input_layer field"
@assert hasfield(StandardNN, :hidden_layers) "Missing hidden_layers field"
@assert !hasfield(StandardNN, :network) "Should not have network field"

println("✅ Model has correct fields:")
println("   - input_layer: $(typeof(model.input_layer))")
println("   - hidden_layers: Vector{Dense} with $(length(model.hidden_layers)) layers")

# Verify dimensions
@assert model.n_features == n_features
@assert model.n_output == n_output
@assert length(model.layers) == length(hidden_sizes) + 1
@assert length(model.hidden_layers) == length(model.layers) - 1

println("✅ Dimensions are correct:")
println("   - n_features: $(model.n_features)")
println("   - n_output: $(model.n_output)")
println("   - layers: $(model.layers)")
println("   - hidden_layers count: $(length(model.hidden_layers))")

# ============================================================================
# Test 2: Forward Pass
# ============================================================================
println("\n[Test 2] Testing Forward Pass")
println("-" ^ 70)

Random.seed!(42)
batch_size = 5
X = rand(Float32, batch_size, n_features)

try
    y_pred = model(X)

    @assert size(y_pred) == (batch_size, n_output) "Wrong output shape: $(size(y_pred))"
    @assert eltype(y_pred) == Float32 "Wrong output type: $(eltype(y_pred))"

    println("✅ Forward pass successful:")
    println("   Input shape: $(size(X))")
    println("   Output shape: $(size(y_pred))")
    println("   Sample predictions: $(y_pred[1:min(3, batch_size), 1])")
catch e
    println("❌ Forward pass FAILED:")
    println("   Error: $e")
    rethrow(e)
end

# ============================================================================
# Test 3: Gradient Computation
# ============================================================================
println("\n[Test 3] Testing Gradient Computation")
println("-" ^ 70)

using Flux

# Generate synthetic data
X_train = rand(Float32, 10, n_features)
Y_train = rand(Float32, 10, 1)

try
    # Compute loss and gradients
    loss_value, grads = Flux.withgradient(model) do m
        y_pred = m(X_train)
        mean((y_pred .- Y_train) .^ 2)
    end

    @assert !isnothing(grads[1]) "Gradients are nothing"
    @assert loss_value > 0 "Loss should be positive"

    println("✅ Gradient computation successful:")
    @printf("   Loss: %.6f\n", loss_value)
    println("   Gradients computed for all layers")
catch e
    println("❌ Gradient computation FAILED:")
    println("   Error: $e")
    rethrow(e)
end

# ============================================================================
# Test 4: Metrics Collection Compatibility
# ============================================================================
println("\n[Test 4] Testing Metrics Collection Compatibility")
println("-" ^ 70)

try
    # Simulate what trainer.jl does with collect_metrics=true

    # Extract input layer weights
    weights = vec(model.input_layer.weight)
    sample_weights = Float64.(weights[1:min(20, length(weights))])
    println("✅ Input layer weights extracted:")
    println("   Sample (first 5): $(sample_weights[1:min(5, length(sample_weights))])")

    # Extract hidden layer weights
    for (i, layer) in enumerate(model.hidden_layers)
        weights = vec(layer.weight)
        sample_weights = Float64.(weights[1:min(20, length(weights))])

        n_negative = sum(weights .< -1e-6)
        @printf("✅ Hidden layer %d weights extracted:\n", i)
        @printf("   Min: %.6f, Max: %.6f\n", minimum(weights), maximum(weights))
        println("   Negative weights: $n_negative (expected for StandardNN)")
    end

    println("\n✅ Metrics collection compatible - no crashes!")

catch e
    println("❌ Metrics collection FAILED:")
    println("   Error: $e")
    rethrow(e)
end

# ============================================================================
# Test 5: Interface Compatibility Functions
# ============================================================================
println("\n[Test 5] Testing Interface Functions")
println("-" ^ 70)

try
    # These should not crash (they're no-ops for StandardNN)
    enforcing_convexity!(model)
    println("✅ enforcing_convexity! - OK (no-op for StandardNN)")

    initialize_convex!(model)
    println("✅ initialize_convex! - OK (no-op for StandardNN)")

catch e
    println("❌ Interface functions FAILED:")
    println("   Error: $e")
    rethrow(e)
end

# ============================================================================
# Test 6: All Layers Have Bias
# ============================================================================
println("\n[Test 6] Verifying All Layers Have Bias")
println("-" ^ 70)

try
    # Check input layer has bias
    @assert !isnothing(model.input_layer.bias) "Input layer missing bias"
    @assert length(model.input_layer.bias) > 0 "Input layer bias is empty"
    println("✅ Input layer has bias: $(length(model.input_layer.bias)) units")

    # Check all hidden layers have bias
    for (i, layer) in enumerate(model.hidden_layers)
        @assert !isnothing(layer.bias) "Hidden layer $i missing bias"
        @assert length(layer.bias) > 0 "Hidden layer $i bias is empty"
        println("✅ Hidden layer $i has bias: $(length(layer.bias)) units")
    end

    println("\n✅ All layers have bias (unlike FICNN which has bias=false for hidden layers)")

catch e
    println("❌ Bias check FAILED:")
    println("   Error: $e")
    rethrow(e)
end

# ============================================================================
# Summary
# ============================================================================
println("\n" * "=" ^ 70)
println("✅ All Tests PASSED!")
println("=" ^ 70)
println("\nStandardNN is now:")
println("  - Compatible with FICNN interface (input_layer, hidden_layers)")
println("  - Forward pass works correctly")
println("  - Gradient computation works")
println("  - Metrics collection will NOT crash")
println("  - All layers have bias (correct for standard NN)")
println("\nYou can now safely use StandardNN with collect_metrics=true!")
println("=" ^ 70)
