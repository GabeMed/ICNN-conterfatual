"""
Quick test to validate the new FICNN architecture.

This script:
1. Creates a simple FICNN model
2. Tests forward pass with random data
3. Validates convexity constraints
4. Tests training for a few iterations
"""

using Pkg
Pkg.activate(".")

include("../src/ICNN.jl")
using .ICNN

using Random
using Statistics

println("="^70)
println("Testing New FICNN Architecture")
println("="^70)

# ============================================================================
# 1. Create Model
# ============================================================================

println("\n🔧 Creating FICNN model...")
n_features = 20  # Example: 20 input features
n_output = 1     # Scalar output
hidden_sizes = [50, 50]

model = FICNN(n_features, n_output; hidden_sizes=hidden_sizes)

println("✅ Model created successfully!")
println("   Input: $n_features features")
println("   Hidden: $hidden_sizes")
println("   Output: $n_output")

# ============================================================================
# 2. Test Forward Pass
# ============================================================================

println("\n🔧 Testing forward pass...")
Random.seed!(42)

# Create dummy data
batch_size = 16
X_dummy = randn(Float32, batch_size, n_features)

# Forward pass
y_pred = model(X_dummy)

println("✅ Forward pass successful!")
println("   Input shape: $(size(X_dummy))")
println("   Output shape: $(size(y_pred))")
println("   Output range: [$(minimum(y_pred)), $(maximum(y_pred))]")

# ============================================================================
# 3. Test Convexity Initialization
# ============================================================================

println("\n🔧 Testing convexity initialization...")
initialize_convex!(model)

println("✅ Convexity initialized!")
for (i, layer) in enumerate(model.hidden_layers)
    w = layer.weight
    n_negative = sum(w .< 0)
    println("   Hidden layer $i: min=$(round(minimum(w), digits=6)), " *
            "max=$(round(maximum(w), digits=6)), negatives=$n_negative")
end

# ============================================================================
# 4. Test Loss Computation
# ============================================================================

println("\n🔧 Testing loss computation...")
Y_dummy = randn(Float32, batch_size, n_output)
loss = mse_loss(model, X_dummy, Y_dummy)

println("✅ Loss computation successful!")
println("   MSE loss: $loss")

# ============================================================================
# 5. Test Gradient Computation
# ============================================================================

println("\n🔧 Testing gradient computation...")
using Flux

val, grads = Flux.withgradient(model) do m
    mse_loss(m, X_dummy, Y_dummy)
end

println("✅ Gradient computation successful!")
println("   Loss value: $val")
println("   Gradient for input_layer: $(grads[1].input_layer !== nothing ? "✓" : "✗")")
for i in 1:length(model.hidden_layers)
    has_grad = grads[1].hidden_layers[i] !== nothing
    println("   Gradient for hidden_layer[$i]: $(has_grad ? "✓" : "✗")")
end

# ============================================================================
# 6. Test Convexity Enforcement
# ============================================================================

println("\n🔧 Testing convexity enforcement...")
# Manually set some weights negative
model.hidden_layers[1].weight[1, 1] = -0.5f0
println("   Set weight to -0.5 before enforcement")

enforcing_convexity!(model)
new_val = model.hidden_layers[1].weight[1, 1]
println("   Weight value after enforcement: $new_val")

if new_val >= 0
    println("✅ Convexity enforcement working correctly!")
else
    println("❌ Convexity enforcement failed!")
end

# ============================================================================
# 7. Mini Training Test
# ============================================================================

println("\n🔧 Testing mini training loop (3 epochs)...")

# Generate synthetic data
n_samples = 100
X_train = randn(Float32, n_samples, n_features)
Y_train = randn(Float32, n_samples, 1)

# Reinitialize model
model = FICNN(n_features, n_output; hidden_sizes=hidden_sizes)

# Train for 3 epochs
model = train!(
    model, X_train, Y_train, 3;
    learning_rate=0.001f0,
    batch_size=32,
    save_dir="./tmp/test_architecture",
    is_convex=true,
    collect_metrics=false
)

println("✅ Mini training completed!")

# Test prediction
y_pred = predict(model, X_train[1:5, :])
println("\n   Sample predictions:")
println("   Shape: $(size(y_pred))")
println("   Values: $(y_pred[1:min(5, end)])")

# Final convexity check
println("\n🔧 Final convexity check:")
all_positive = true
for (i, layer) in enumerate(model.hidden_layers)
    w = layer.weight
    n_negative = sum(w .< -1e-6)
    if n_negative > 0
        all_positive = false
        println("   ⚠️  Hidden layer $i has $n_negative negative weights")
    else
        println("   ✅ Hidden layer $i: all weights ≥ 0")
    end
end

# ============================================================================
# Summary
# ============================================================================

println("\n" * "="^70)
if all_positive
    println("✅ ALL TESTS PASSED! Architecture is working correctly.")
else
    println("⚠️  Some convexity constraints violated. Check implementation.")
end
println("="^70)
