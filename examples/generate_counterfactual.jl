# Example: Generate counterfactual explanations using trained ICNN model

using Pkg
Pkg.activate(".")

# Load ICNN module
include("../src/ICNN.jl")
using .ICNN

# Load counterfactuals module
include("../counterfactuals/model_loader.jl")
include("../counterfactuals/algorithms/optimization.jl")

"""
Simple example of counterfactual generation workflow.

Steps:
1. Load trained ICNN model
2. Load/create a factual input
3. Generate counterfactual for opposite class
4. Analyze results
"""
function main()
    println("="^70)
    println("COUNTERFACTUAL GENERATION WITH ICNN")
    println("="^70)
    
    # ========================================
    # PART 1: Load trained model and scaler
    # ========================================
    model_path = joinpath(@__DIR__, "results_julia/best_model.bson")
    scaler_path = joinpath(@__DIR__, "results_julia/scaler.bson")

    println("\n📦 Loading model and scaler...")
    local icnn_model, scaler, feature_info
    try
        icnn_model = load_icnn_model(model_path)
        scaler_data = BSON.load(scaler_path)
        scaler = scaler_data[:scaler]
        feature_info = scaler_data[:feature_info]
        println("  Loaded: model + scaler + feature_info")
    catch e
        println("❌ Error loading: $e")
        println("\nPlease train a model first:")
        println("  julia --project=. examples/train_icnn.jl")
        return
    end
    
    # ========================================
    # PART 2: Load real test data (already normalized!)
    # ========================================
    println("\n📊 Loading test data...")

    # Load dataset and apply same preprocessing
    df = load_adult_income()
    X_raw, y_raw, _, _ = preprocess_adult_income(df)
    X_raw = Float32.(X_raw)
    y_raw = Float32.(y_raw)

    # Split with same seed as training
    X_train_raw, y_train_raw, X_test_raw, y_test_raw = split_data(X_raw, y_raw, train_ratio=0.8, seed=42)

    # Apply SAME scaler fitted during training
    X_test = copy(X_test_raw)
    transform!(X_test, scaler)

    # Pick first test sample as factual
    sample_idx = 1
    x_factual = X_test[sample_idx, :]
    y_true = y_test_raw[sample_idx, 1]

    println("  Using test sample #$sample_idx")
    println("  True label: $(Int(y_true))")
    println("  Features: $(length(x_factual)) (all in [0,1], categoricals are binary)")
    
    # ========================================
    # PART 3: Get current prediction
    # ========================================
    println("\n🔮 Current prediction...")
    predicted_class, probability = test_model_prediction(icnn_model, x_factual)
    
    println("  Predicted class: $predicted_class")
    println("  Probability: $(round(probability, digits=4))")
    
    # ========================================
    # PART 4: Generate counterfactual
    # ========================================
    println("\n🎯 Generating counterfactual...")
    
    # Target is the opposite class
    y_target = predicted_class == 0 ? 1.0f0 : 0.0f0
    
    println("  Target class: $y_target")
    
    # Generate counterfactuals with different configurations
    println("\n" * "="^70)
    println("Testing different sparsity weights")
    println("="^70)

    results = Dict()

    # Configuration 1: Light sparsity
    println("\n### Config 1: Light sparsity (5.0) ###")
    try
        result = generate_counterfactual(
            icnn_model, x_factual, y_target;
            alpha=2.0, sparsity_weight=5.0, time_limit=60.0, feature_info=feature_info
        )
        results[:light_sparsity] = result
    catch e
        println("❌ Error: $e")
        showerror(stdout, e, catch_backtrace())
    end

    # Configuration 2: Medium sparsity
    println("\n### Config 2: Medium sparsity (20.0) ###")
    try
        result = generate_counterfactual(
            icnn_model, x_factual, y_target;
            alpha=2.0, sparsity_weight=20.0, time_limit=60.0, feature_info=feature_info
        )
        results[:medium_sparsity] = result
    catch e
        println("❌ Error: $e")
        showerror(stdout, e, catch_backtrace())
    end

    # Configuration 3: Strong sparsity
    println("\n### Config 3: Strong sparsity (80.0) ###")
    try
        result = generate_counterfactual(
            icnn_model, x_factual, y_target;
            alpha=2.0, sparsity_weight=80.0, time_limit=60.0, feature_info=feature_info
        )
        results[:strong_sparsity] = result
    catch e
        println("❌ Error: $e")
        showerror(stdout, e, catch_backtrace())
    end
    
    # ========================================
    # PART 5: Analyze results
    # ========================================
    println("\n" * "="^70)
    println("RESULTS")
    println("="^70)

    if isempty(results)
        println("No counterfactuals found")
        return
    end

    config_order = [:light_sparsity, :medium_sparsity, :strong_sparsity]
    config_labels = ["Light (5.0)", "Medium (20.0)", "Strong (80.0)"]

    for (i, config) in enumerate(config_order)
        if haskey(results, config) && results[config][:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
            r = results[config]
            println("$(config_labels[i]): dist=$(round(r[:distance], digits=4)), changed=$(r[:num_changed]), pred=$(round(r[:prediction], digits=3)), time=$(round(r[:solve_time], digits=2))s")
        end
    end

    if haskey(results, :light_sparsity) && results[:light_sparsity][:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        baseline = results[:light_sparsity]
        println("\nSparsity tradeoff:")
        for config in [:medium_sparsity, :strong_sparsity]
            if haskey(results, config) && results[config][:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
                r = results[config]
                Δdist = r[:distance] - baseline[:distance]
                Δfeats = baseline[:num_changed] - r[:num_changed]
                if Δfeats > 0
                    println("  Saved $Δfeats features at cost +$(round(Δdist, digits=4)) distance")
                end
            end
        end
    end

    println("="^70)
end

# Run example
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

