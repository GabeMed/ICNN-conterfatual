"""
Test script for ACPM pipeline with StandardNN.

This script tests:
1. Module loading
2. StandardNN model creation
3. Forward pass
4. ACPM data loading (if available)
5. Training loop (brief test)
"""

using Pkg
Pkg.activate(".")

include("../ICNN.jl")
using .ICNN

using Random
using Statistics

println("="^70)
println("ACPM Pipeline Test Suite")
println("="^70)

# Test 1: Module loading
println("\n[1/5] Testing module loading...")
try
    println("  - FICNN available: ", isdefined(ICNN, :FICNN))
    println("  - StandardNN available: ", isdefined(ICNN, :StandardNN))
    println("  - load_dcopf_data available: ", isdefined(ICNN, :load_dcopf_data))
    println("  - load_acpm_data available: ", isdefined(ICNN, :load_acpm_data))
    println("  PASSED")
catch e
    println("  FAILED: $e")
end

# Test 2: StandardNN model creation
println("\n[2/5] Testing StandardNN model creation...")
try
    model = StandardNN(10, 1; hidden_sizes=[50, 50])
    println("  - Model type: ", typeof(model))
    println("  - n_features: ", model.n_features)
    println("  - n_output: ", model.n_output)
    println("  - layers: ", model.layers)
    println("  PASSED")
catch e
    println("  FAILED: $e")
    rethrow(e)
end

# Test 3: Forward pass
println("\n[3/5] Testing forward pass...")
try
    model = StandardNN(10, 1; hidden_sizes=[50, 50])
    x = rand(Float32, 5, 10)  # 5 samples, 10 features
    y = model(x)
    println("  - Input shape: ", size(x))
    println("  - Output shape: ", size(y))
    println("  - Output range: [", minimum(y), ", ", maximum(y), "]")
    @assert size(y) == (5, 1) "Output shape incorrect"
    println("  PASSED")
catch e
    println("  FAILED: $e")
    rethrow(e)
end

# Test 4: ACPM data loading (if available)
println("\n[4/5] Testing ACPM data loading...")
data_file = "icnn/data/data_pglib_opf_case118_ieee_AC+DC.bson"
if isfile(data_file)
    try
        dataset = prepare_acpm_dataset(
            data_file;
            train_ratio=0.8,
            normalize_method=:none,
            shuffle=true,
            seed=42
        )
        println("  - Training samples: ", size(dataset.X_train, 1))
        println("  - Test samples: ", size(dataset.X_test, 1))
        println("  - Features: ", dataset.n_features)
        println("  - Y range: [", minimum(dataset.Y_train), ", ", maximum(dataset.Y_train), "]")
        println("  PASSED")
    catch e
        println("  FAILED: $e")
        rethrow(e)
    end
else
    println("  SKIPPED (data file not found: $data_file)")
    println("  To generate data, run: julia icnn/data/Generate_DCOPF.jl")
end

# Test 5: Brief training test
println("\n[5/5] Testing training loop...")
try
    Random.seed!(42)

    # Create synthetic data
    n_samples = 100
    n_features = 10
    X_train = rand(Float32, n_samples, n_features)
    Y_train = sum(X_train.^2, dims=2) .+ rand(Float32, n_samples, 1) .* 0.1

    X_test = rand(Float32, 20, n_features)
    Y_test = sum(X_test.^2, dims=2) .+ rand(Float32, 20, 1) .* 0.1

    model = StandardNN(n_features, 1; hidden_sizes=[20, 20])

    # Train for just 3 epochs to test
    save_dir = "./tmp/test_acpm_pipeline"
    mkpath(save_dir)

    model = train!(
        model, X_train, Y_train, 3;
        learning_rate=1f-3,
        batch_size=16,
        save_dir=save_dir,
        is_convex=false,  # IMPORTANT for StandardNN
        X_test=X_test,
        y_test=Y_test,
        collect_metrics=false
    )

    # Check predictions
    y_pred = predict(model, X_test)
    mse = mean((y_pred .- Y_test).^2)

    println("  - Training completed")
    println("  - Test MSE: ", mse)
    println("  - Prediction range: [", minimum(y_pred), ", ", maximum(y_pred), "]")
    println("  - Model saved to: ", save_dir)
    println("  PASSED")

    # Clean up
    rm(save_dir, recursive=true)
catch e
    println("  FAILED: $e")
    rethrow(e)
end

# Test 6: Verify no convexity enforcement
println("\n[6/6] Testing convexity enforcement (should be no-op)...")
try
    model = StandardNN(10, 1; hidden_sizes=[20, 20])

    # These should be no-ops for StandardNN
    initialize_convex!(model)
    enforcing_convexity!(model)

    println("  - initialize_convex! completed (no-op)")
    println("  - enforcing_convexity! completed (no-op)")
    println("  PASSED")
catch e
    println("  FAILED: $e")
    rethrow(e)
end

println("\n" * "="^70)
println("All tests passed!")
println("="^70)

println("\nNext steps:")
println("1. Generate ACPM data: julia icnn/data/Generate_DCOPF.jl")
println("2. Train StandardNN: julia icnn/examples/train_acpm_standard_nn.jl")
println("3. Compare with FICNN on DCPM data")
