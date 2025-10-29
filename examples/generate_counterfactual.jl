"""
Generate counterfactual explanations for DCOPF using trained ICNN model.

Example use case:
Given a demand vector x with predicted cost y, find a nearby demand x'
such that the cost is reduced to a target value y_target.

This is useful for:
1. Understanding how demand changes affect costs
2. Finding cost-efficient demand patterns
3. Sensitivity analysis of the power system
"""

using Pkg
Pkg.activate(".")

# Load ICNN module
include("../src/ICNN.jl")
using .ICNN

# Load counterfactuals module
include("../counterfactuals/model_loader.jl")
include("../counterfactuals/algorithms/mip_counterfactual.jl")

using Random
using Statistics
using Printf

"""
Main function to generate counterfactuals for DCOPF.
"""
function main()
    println("="^70)
    println("COUNTERFACTUAL GENERATION FOR DC-OPF")
    println("="^70)

    # ========================================================================
    # PART 1: Configuration
    # ========================================================================

    # Paths (adjust these to your trained model)
    model_path = "tmp/dcopf_experiment/best_model.bson"
    data_path = "test_systems/data_pglib_opf_case118_ieee.bson"

    # Counterfactual settings
    sparsity_weight = 0.1     # Weight for sparsity penalty
    epsilon = 0.01            # Tolerance for target matching (normalized scale)
    time_limit = 120.0        # Solver time limit (seconds)

    # Target scenarios
    cost_reduction = 0.05  # Reduce cost by 5% (normalized)

    println("\n📋 Configuration:")
    println("   Model: $model_path")
    println("   Data: $data_path")
    println("   Sparsity weight: $sparsity_weight")
    println("   Epsilon (tolerance): $epsilon")
    println("   Time limit: $(time_limit)s")
    println("   Cost reduction target: $(cost_reduction*100)%")

    # ========================================================================
    # PART 2: Load trained model
    # ========================================================================

    println("\n" * "="^70)
    println("Loading Model and Data")
    println("="^70)

    if !isfile(model_path)
        error("""
        Model file not found: $model_path

        Please train a model first:
            julia examples/train_dcopf.jl

        This will create the model file.
        """)
    end

    println("\n📦 Loading model...")
    icnn_model = load_icnn_model(model_path)

    # ========================================================================
    # PART 3: Load test data
    # ========================================================================

    println("\n📊 Loading test data...")
    dataset = prepare_dcopf_dataset(
        data_path;
        train_ratio=0.8,
        normalize_method=:standardize,
        shuffle=true,
        seed=42
    )

    X_test = dataset.X_test
    Y_test = dataset.Y_test
    scaler_X = dataset.scaler_X
    scaler_Y = dataset.scaler_Y

    println("✓ Test set loaded: $(size(X_test, 1)) samples")

    # ========================================================================
    # PART 4: Select a factual sample
    # ========================================================================

    println("\n" * "="^70)
    println("Selecting Factual Sample")
    println("="^70)

    # Pick a test sample (you can change this index)
    sample_idx = 1
    x_factual = X_test[sample_idx, :]
    y_true = Y_test[sample_idx, 1]

    # Get current prediction
    y_pred = test_model_prediction(icnn_model, x_factual)

    println("\nFactual sample #$sample_idx:")
    println("  Features: $(length(x_factual)) (normalized)")
    println("  True cost (normalized): $(round(y_true, digits=4))")
    println("  Predicted cost (normalized): $(round(y_pred, digits=4))")
    println("  Prediction error: $(round(abs(y_pred - y_true), digits=6))")

    # Denormalize for interpretation
    y_true_denorm = denormalize_output([y_true], scaler_Y)[1]
    y_pred_denorm = denormalize_output([y_pred], scaler_Y)[1]

    println("\n  Original scale:")
    println("    True cost: \$$(round(y_true_denorm, digits=2))")
    println("    Predicted cost: \$$(round(y_pred_denorm, digits=2))")

    # ========================================================================
    # PART 5: Define counterfactual targets
    # ========================================================================

    println("\n" * "="^70)
    println("Defining Counterfactual Targets")
    println("="^70)

    # Target 1: Reduce cost by specified percentage
    y_target_1 = y_pred - cost_reduction

    # Target 2: Match true cost (if prediction is off)
    y_target_2 = y_true

    # Target 3: Increase cost (for exploration)
    y_target_3 = y_pred + cost_reduction

    targets = [
        ("Cost reduction by $(cost_reduction*100)%", y_target_1),
        ("Match true cost", y_target_2),
        ("Cost increase by $(cost_reduction*100)%", y_target_3)
    ]

    println("\nCounterfactual targets:")
    for (i, (desc, target)) in enumerate(targets)
        target_denorm = denormalize_output([target], scaler_Y)[1]
        println("  $i. $desc")
        println("     Normalized: $(round(target, digits=4))")
        println("     Original: \$$(round(target_denorm, digits=2))")
    end

    # ========================================================================
    # PART 6: Generate counterfactuals
    # ========================================================================

    println("\n" * "="^70)
    println("Generating Counterfactuals")
    println("="^70)

    results = Dict()

    for (i, (desc, y_target)) in enumerate(targets)
        println("\n### Scenario $i: $desc ###")

        try
            result = generate_counterfactual(
                icnn_model,
                x_factual,
                Float64(y_target);
                epsilon=epsilon,
                sparsity_weight=sparsity_weight,
                x_bounds=(minimum(X_test), maximum(X_test)),
                time_limit=time_limit,
                immutable_indices=Int[]  # Can specify features that can't change
            )
            results[Symbol("scenario_$i")] = result
        catch e
            println("❌ Error: $e")
            showerror(stdout, e, catch_backtrace())
        end
    end

    # ========================================================================
    # PART 7: Analyze and compare results
    # ========================================================================

    println("\n" * "="^70)
    println("RESULTS SUMMARY")
    println("="^70)

    if isempty(results)
        println("No counterfactuals found.")
        return
    end

    println("\n" * "-"^70)
    @printf("%-30s | %8s | %10s | %10s | %8s\n",
            "Scenario", "Status", "Distance", "Changed", "Time (s)")
    println("-"^70)

    for (i, (desc, y_target)) in enumerate(targets)
        key = Symbol("scenario_$i")
        if haskey(results, key)
            r = results[key]
            status_str = r[:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT] ? "✓ Found" : "✗ $(r[:status])"
            dist_str = r[:distance] == Inf ? "∞" : "$(round(r[:distance], digits=3))"
            changed_str = r[:num_changed] === nothing ? "-" : "$(r[:num_changed])"
            time_str = "$(round(r[:solve_time], digits=2))"

            @printf("%-30s | %8s | %10s | %10s | %8s\n",
                   desc[1:min(30, length(desc))], status_str, dist_str, changed_str, time_str)
        end
    end
    println("-"^70)

    # ========================================================================
    # PART 8: Detailed analysis of best counterfactual
    # ========================================================================

    println("\n" * "="^70)
    println("DETAILED ANALYSIS - Best Counterfactual")
    println("="^70)

    # Find the best counterfactual (smallest distance with valid solution)
    best_key = nothing
    best_dist = Inf

    for (key, result) in results
        if result[:status] in [MOI.OPTIMAL, MOI.FEASIBLE_POINT] &&
           result[:distance] < best_dist
            best_key = key
            best_dist = result[:distance]
        end
    end

    if best_key === nothing
        println("No valid counterfactual found.")
        return
    end

    best_result = results[best_key]
    x_cf = best_result[:counterfactual]
    changed_indices = best_result[:changed_indices]

    println("\nBest counterfactual: $(best_key)")
    println("  Distance: $(round(best_result[:distance], digits=4))")
    println("  Features changed: $(best_result[:num_changed])/$(length(x_factual))")
    println("  Predicted cost (normalized): $(round(best_result[:prediction], digits=4))")

    # Denormalize
    y_cf_denorm = denormalize_output([best_result[:prediction]], scaler_Y)[1]
    println("  Predicted cost (original): \$$(round(y_cf_denorm, digits=2))")

    cost_change = y_cf_denorm - y_pred_denorm
    cost_change_pct = (cost_change / y_pred_denorm) * 100

    println("\n  Cost change: \$$(round(cost_change, digits=2)) ($(round(cost_change_pct, digits=2))%)")

    # Show top changed features
    if !isempty(changed_indices)
        n_show = min(10, length(changed_indices))
        println("\n  Top $n_show changed features:")

        # Sort by absolute change
        changes = abs.(x_cf[changed_indices] .- x_factual[changed_indices])
        sorted_idx = sortperm(changes, rev=true)

        for i in 1:n_show
            idx = changed_indices[sorted_idx[i]]
            @printf("    Feature %3d: %8.4f → %8.4f (Δ = %+8.4f)\n",
                   idx, x_factual[idx], x_cf[idx], x_cf[idx] - x_factual[idx])
        end
    end

    # ========================================================================
    # PART 9: Validation
    # ========================================================================

    println("\n" * "="^70)
    println("VALIDATION")
    println("="^70)

    # Verify the counterfactual prediction
    y_cf_verify = test_model_prediction(icnn_model, x_cf)
    println("\nCounterfactual verification:")
    println("  MIP prediction: $(round(best_result[:prediction], digits=6))")
    println("  Direct forward pass: $(round(y_cf_verify, digits=6))")
    println("  Difference: $(round(abs(y_cf_verify - best_result[:prediction]), digits=8))")

    if abs(y_cf_verify - best_result[:prediction]) < 1e-4
        println("  ✓ Predictions match!")
    else
        println("  ⚠️  Predictions differ (may be due to numerical precision)")
    end

    println("\n" * "="^70)
    println("✅ Counterfactual generation complete!")
    println("="^70)
end

# Run the main function
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
