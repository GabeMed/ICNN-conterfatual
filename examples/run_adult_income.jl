# Example script to run the ICNN model on the Adult Income dataset
using Pkg

# First, activate the project environment
# Pkg.activate(".")  # Uncomment this if you have a Project.toml

# Add required packages if not already installed
# Pkg.add(["Flux", "Random", "Statistics", "Plots", "Printf", "CSV", "DataFrames", "Downloads"])

# Include the main module
include("../src/ICNN.jl")
using .ICNN

function main()
    # Set random seed for reproducibility
    Random.seed!(42)

    # Load and preprocess the Adult Income dataset
    println("Loading dataset...")
    df = load_adult_income()
    X, y, feature_names = preprocess_adult_income(df)

    # Split the data into training and test sets
    println("\nSplitting data...")
    X_train, y_train, X_test, y_test = split_data(X, y, train_ratio=0.8)

    println("\nTraining set size: $(size(X_train, 1)) samples")
    println("Test set size: $(size(X_test, 1)) samples")
    println("Number of features: $(size(X_train, 2))")

    # Create and initialize the FICNN model
    println("\nCreating FICNN model...")
    model = FICNN(
        size(X_train, 2),  # number of features
        1,                 # number of outputs (binary classification)
        hidden_sizes=[100, 100],  # smaller network for faster training
        n_gradient_iterations=20
    )

    # Train the model
    println("\nTraining model...")
    save_dir = joinpath(@__DIR__, "results")
    model = train!(
        model,
        X_train,
        y_train,
        epochs=10,        # reduced for example
        learning_rate=0.001,
        batch_size=32,
        save_dir=save_dir
    )

    # Make predictions on test set
    println("\nEvaluating model...")
    y_init = fill(0.5f0, size(y_test))
    y_pred = predict(model, X_test, y_init)

    # Calculate accuracy
    accuracy = mean((y_pred .> 0.5) .== y_test)
    println("\nTest accuracy: $(round(accuracy * 100, digits=2))%")
end

# Run the main function
main()