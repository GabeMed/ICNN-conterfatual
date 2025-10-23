# Run FICNN with detailed metrics for comparison with Python
using Pkg
using Random
using Statistics
using Flux

# Include the main module
include("../src/ICNN.jl")
using .ICNN

function main()
    # Set random seed for reproducibility (same as Python)
    Random.seed!(42)

    # Load and preprocess the Adult Income dataset
    println("Loading dataset...")
    df = load_adult_income()
    X, y, feature_names = preprocess_adult_income(df)
    X = Float32.(X)
    y = Float32.(y)

    # Split the data into training and test sets (same as Python: 80/20)
    println("\nSplitting data...")
    X_train, y_train, X_test, y_test = split_data(X, y, train_ratio=0.8, seed=42)

    println("\nTraining set size: $(size(X_train, 1)) samples")
    println("Test set size: $(size(X_test, 1)) samples")
    println("Number of features: $(size(X_train, 2))")

    # Create FICNN model with same architecture as Python
    println("\nCreating FICNN model...")
    model = FICNN(
        size(X_train, 2),  # number of features
        1,                 # number of outputs (binary classification)
        hidden_sizes=[200, 200],  # Match Python exactly
        n_gradient_iterations=10   # REDUCED from 30 to 10 for speed
    )

    # Validate model architecture
    println("\n🔍 Model Architecture Validation:")
    println("   n_features: $(model.n_features)")
    println("   n_labels: $(model.n_labels)")
    println("   layers: $(model.layers)")
    println("   n_gradient_iterations: $(model.n_gradient_iterations)")
    println("\n   Layer Dimensions:")
    for (i, layer) in enumerate(model.input_x_layers)
        w_shape = size(layer.weight)
        b_shape = layer.bias === false ? "none" : size(layer.bias)
        println("      input_x_layer[$i]: W=$(w_shape), b=$(b_shape)")
    end
    for (i, layer) in enumerate(model.input_y_layers)
        w_shape = size(layer.weight)
        b_shape = layer.bias === false ? "none" : size(layer.bias)
        println("      input_y_layer[$i]: W=$(w_shape), b=$(b_shape)")
    end
    for (i, layer) in enumerate(model.hidden_layers)
        w_shape = size(layer.weight)
        has_bias = layer.bias !== false
        println("      hidden_layer[$i]: W=$(w_shape), has_bias=$(has_bias)")
    end

    # Test forward pass before training
    println("\n🧪 Pre-training Forward Pass Test:")
    x_sample = X_train[1:5, :]
    y_sample = y_train[1:5, :]
    y_init_sample = fill(0.5f0, size(y_sample))

    println("   Testing model(x, y)...")
    energy = model(x_sample, y_init_sample)
    println("      ✓ Energy output: $(size(energy)) range [$(round(minimum(energy), digits=4)), $(round(maximum(energy), digits=4))]")

    println("   Testing predict(model, x, y_init)...")
    y_pred_sample = predict(model, x_sample, y_init_sample)
    println("      ✓ Prediction output: $(size(y_pred_sample)) range [$(round(minimum(y_pred_sample), digits=4)), $(round(maximum(y_pred_sample), digits=4))]")

    println("   Testing mse_loss(model, x, y_init, y_true)...")
    loss_sample = mse_loss(model, x_sample, y_init_sample, y_sample)
    println("      ✓ Loss: $(round(loss_sample, digits=6))")

    println("   Testing gradient computation...")
    loss_val, grads = Flux.withgradient(model) do m
        mse_loss(m, x_sample, y_init_sample, y_sample)
    end
    if grads[1] !== nothing
        println("      ✓ Gradients computed successfully (loss=$(round(loss_val, digits=6)))")
    else
        println("      ✗ ERROR: Gradients are nothing!")
    end

    # Train the model with metrics collection
    println("\nTraining model with metrics collection...")
    println("="^70)
    println("JULIA/FLUX IMPLEMENTATION")
    println("="^70)
    
    # DIFFERENTIATION METHOD (both are paper-compliant):
    # - "unrolled": Unroll PGD solver, AD tracks through iterations (WORKS) ✓
    # - "implicit": Implicit differentiation of argmin (currently uses unrolled)
    # - "none": Standard nested AD (broken, don't use)
    #
    # Paper (Amos et al. ICML'17, Sec 5.1) mentions BOTH approaches:
    # 1. "differentiate through the argmin" → implicit diff
    # 2. "unroll the optimization procedure" → unrolled
    #
    # Both are valid! Use "unrolled" for now (simpler, works reliably).
    diff_method = "unrolled"
    
    println("\n🎯 Differentiation method: $diff_method")
    if diff_method == "implicit"
        println("   Using IMPLICIT DIFFERENTIATION (Paper Sec 5.1)")
        println("   (Currently falls back to unrolled - full impl needs param packing)")
    elseif diff_method == "unrolled"
        println("   Using UNROLLED SOLVER (Paper Sec 5.1 - alternative approach)")
        println("   AD tracks through PGD iterations - paper-compliant!")
    else
        println("   ⚠️  Using nested AD (broken, don't use)")
    end
    
    save_dir = joinpath(@__DIR__, "results_julia")
    model = train!(
        model,
        X_train,
        y_train,
        5;  # 5 epochs like Python
        learning_rate=0.001,
        batch_size=128,  # INCREASED from 32 to 128 for speed
        save_dir=save_dir,
        X_test=X_test,
        y_test=y_test,
        collect_metrics=true,
        diff_method=diff_method  # Use selected differentiation method
    )

    # Make final predictions
    println("\nFinal Evaluation...")
    y_init = fill(0.5f0, size(y_test))
    y_pred = predict(model, X_test, y_init)

    # Calculate accuracy
    accuracy = mean((y_pred .> 0.5) .== y_test)
    println("\nFinal Test Accuracy: $(round(accuracy * 100, digits=2))%")

    # ===== PREDICTION INSPECTION =====
    println("\n" * "="^70)
    println("PREDICTION INSPECTION")
    println("="^70)

    # 1. Prediction statistics
    println("\n1. Prediction Statistics:")
    println("   Mean prediction: $(round(mean(y_pred), digits=4))")
    println("   Std prediction:  $(round(std(y_pred), digits=4))")
    println("   Min prediction:  $(round(minimum(y_pred), digits=4))")
    println("   Max prediction:  $(round(maximum(y_pred), digits=4))")
    println("   Median prediction: $(round(median(y_pred), digits=4))")

    # 2. Distribution of predictions
    println("\n2. Prediction Distribution:")
    n_low = sum(y_pred .< 0.3)
    n_mid = sum((y_pred .>= 0.3) .& (y_pred .<= 0.7))
    n_high = sum(y_pred .> 0.7)
    println("   < 0.3 (confident negative): $n_low ($(round(100*n_low/length(y_pred), digits=1))%)")
    println("   0.3-0.7 (uncertain):        $n_mid ($(round(100*n_mid/length(y_pred), digits=1))%)")
    println("   > 0.7 (confident positive): $n_high ($(round(100*n_high/length(y_pred), digits=1))%)")

    # 3. Confusion matrix
    println("\n3. Confusion Matrix:")
    y_pred_binary = (y_pred .> 0.5)
    y_test_binary = (y_test .> 0.5)

    true_pos = sum(y_pred_binary .& y_test_binary)
    true_neg = sum(.!y_pred_binary .& .!y_test_binary)
    false_pos = sum(y_pred_binary .& .!y_test_binary)
    false_neg = sum(.!y_pred_binary .& y_test_binary)

    println("   True Positives:  $true_pos")
    println("   True Negatives:  $true_neg")
    println("   False Positives: $false_pos")
    println("   False Negatives: $false_neg")

    # Calculate metrics
    precision = true_pos / (true_pos + false_pos + 1e-10)
    recall = true_pos / (true_pos + false_neg + 1e-10)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

    println("\n   Precision: $(round(precision * 100, digits=2))%")
    println("   Recall:    $(round(recall * 100, digits=2))%")
    println("   F1 Score:  $(round(f1 * 100, digits=2))%")

    # 4. Show sample predictions
    println("\n4. Sample Predictions (first 20):")
    println("   idx | y_true | y_pred | prediction | correct?")
    println("   " * "-"^54)
    for i in 1:min(20, length(y_test))
        true_val = y_test[i, 1]
        pred_val = y_pred[i, 1]
        pred_label = pred_val > 0.5 ? "POS" : "NEG"
        is_correct = (pred_val > 0.5) == (true_val > 0.5) ? "✓" : "✗"
        println("   $(lpad(i, 3)) | $(lpad(Int(true_val), 6)) | $(lpad(round(pred_val, digits=4), 6)) | $(lpad(pred_label, 10)) | $is_correct")
    end

    # 5. Check for mode collapse
    println("\n5. Mode Collapse Detection:")
    unique_preds = length(unique(round.(y_pred, digits=2)))
    println("   Unique predictions (rounded to 2 decimals): $unique_preds")
    if unique_preds < 10
        println("   ⚠️  WARNING: Very few unique predictions - possible mode collapse!")
        println("   Top 5 most common predictions:")
        pred_counts = Dict{Float32, Int}()
        for p in round.(y_pred, digits=2)
            pred_counts[p] = get(pred_counts, p, 0) + 1
        end
        sorted_preds = sort(collect(pred_counts), by=x->x[2], rev=true)
        for (i, (pred, count)) in enumerate(sorted_preds[1:min(5, length(sorted_preds))])
            println("      $i. y=$(pred): $count samples ($(round(100*count/length(y_pred), digits=1))%)")
        end
    else
        println("   ✓ Good diversity in predictions")
    end

    # 6. Loss analysis
    println("\n6. Loss Analysis:")
    mse_test = mean((y_pred .- y_test) .^ 2)
    mae_test = mean(abs.(y_pred .- y_test))
    println("   MSE (test): $(round(mse_test, digits=6))")
    println("   MAE (test): $(round(mae_test, digits=6))")

    # Find worst predictions
    errors = abs.(y_pred .- y_test)
    worst_indices = sortperm(vec(errors), rev=true)[1:min(5, length(errors))]
    println("\n   Worst 5 predictions:")
    println("   idx | y_true | y_pred | error")
    println("   " * "-"^38)
    for idx in worst_indices
        println("   $(lpad(idx, 3)) | $(lpad(round(y_test[idx,1], digits=4), 6)) | $(lpad(round(y_pred[idx,1], digits=4), 6)) | $(lpad(round(errors[idx,1], digits=4), 6))")
    end

    # 7. Check if model is learning or just predicting majority class
    println("\n7. Baseline Comparison:")
    majority_class = mean(y_test)
    always_majority_acc = max(majority_class, 1 - majority_class)
    println("   Always predict majority class accuracy: $(round(always_majority_acc * 100, digits=2))%")
    println("   Model accuracy: $(round(accuracy * 100, digits=2))%")
    improvement = accuracy - always_majority_acc
    println("   Improvement over baseline: $(round(improvement * 100, digits=2))%")

    if improvement < 0.01
        println("   ⚠️  WARNING: Model is barely better than baseline!")
        println("   The model may not be learning meaningful patterns.")
    else
        println("   ✓ Model shows improvement over baseline")
    end

    println("\n" * "="^70)
    println("JULIA TRAINING COMPLETE")
    println("Metrics saved to: $save_dir/metrics_julia.json")
    println("="^70)
end

# Run the main function
main()

