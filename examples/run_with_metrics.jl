# Run FICNN with detailed metrics for comparison with Python
using Pkg
using Random
using Statistics

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
        hidden_sizes=[100, 100],  # Testing with smaller network first
        n_gradient_iterations=20   # Reduced iterations
    )

    # Train the model with metrics collection
    println("\nTraining model with metrics collection...")
    println("="^70)
    println("JULIA/FLUX IMPLEMENTATION")
    println("="^70)
    
    save_dir = joinpath(@__DIR__, "results_julia")
    model = train!(
        model,
        X_train,
        y_train,
        5;  # 5 epochs like Python
        learning_rate=0.001,
        batch_size=32,
        save_dir=save_dir,
        X_test=X_test,
        y_test=y_test,
        collect_metrics=true
    )

    # Make final predictions
    println("\nFinal Evaluation...")
    y_init = fill(0.5f0, size(y_test))
    y_pred = predict(model, X_test, y_init)

    # Calculate accuracy
    accuracy = mean((y_pred .> 0.5) .== y_test)
    println("\nFinal Test Accuracy: $(round(accuracy * 100, digits=2))%")
    
    println("\n" * "="^70)
    println("JULIA TRAINING COMPLETE")
    println("Metrics saved to: $save_dir/metrics_julia.json")
    println("="^70)
end

# Run the main function
main()

