"""Projects W^(z) ≥ 0 to maintain convexity. Call after each update."""
function enforcing_convexity!(model::AbstractICNN)
    @inbounds for layer in model.hidden_layers
        layer.weight .= max.(layer.weight, 0.0f0)
    end
end

"""Initializes W^(z) to small positive values."""
function initialize_convex!(model::AbstractICNN)
    @inbounds for layer in model.hidden_layers
        layer.weight .= abs.(layer.weight) ./ 10.0f0
    end
end

"""
Direct forward pass for regression.
Simply calls the model to get predictions.
"""
function predict(model::AbstractICNN, x)
    return model(Float32.(x))
end

"""
MSE loss for regression: direct comparison between prediction and ground truth.
No nested optimization - simple supervised learning.
"""
function mse_loss(model::AbstractICNN, x, y_true)
    y_pred = model(Float32.(x))
    return mean((y_pred .- Float32.(y_true)) .^ 2)
end

"""
Train ICNN with mini-batch gradient descent for regression.

# Arguments
- `model`: FICNN model to train
- `data_x`: Training input features (n_samples, n_features)
- `data_y`: Training target values (n_samples, n_output)
- `epochs`: Number of training epochs
- `learning_rate`: Learning rate for Adam optimizer
- `batch_size`: Mini-batch size
- `save_dir`: Directory to save models and logs
- `is_convex`: Whether to enforce convexity constraints
- `X_test`: Optional test input features
- `y_test`: Optional test target values
- `collect_metrics`: Whether to collect detailed training metrics
"""
function train!(model::AbstractICNN, data_x, data_y, epochs::Int = 100;
                learning_rate=0.001f0, batch_size=32, save_dir = "./tmp",
                is_convex=true, X_test=nothing, y_test=nothing, collect_metrics=false)
    # Create save directory and log file
    mkpath(save_dir)
    log_file = joinpath(save_dir, "training_log.csv")
    
    # Prepare metrics collection
    metrics = Dict(
        "train_losses" => Float64[],
        "test_losses" => Float64[],
        "test_rmse" => Float64[],
        "test_mae" => Float64[],
        "predictions" => Float64[],
        "final_predictions_sample" => Float64[],
        "weights_sample" => Dict()
    )

    # CSV header
    if X_test !== nothing && y_test !== nothing
        open(log_file, "w") do io
            println(io, "epoch,train_loss,test_loss,test_rmse,test_mae,time")
        end
    else
        open(log_file, "w") do io
            println(io, "epoch,train_loss,time")
        end
    end
    
    if is_convex
        initialize_convex!(model)
        println("\n🔧 Convexity initialized:")
        for (i, layer) in enumerate(model.hidden_layers)
            w = layer.weight
            println("   hidden_layer[$i]: min=$(round(minimum(w), digits=4)), max=$(round(maximum(w), digits=4))")
        end
    end

    opt = Flux.setup(Adam(learning_rate), model)

    best_mse = Inf
    n_samples = size(data_x, 1)
    n_batches = Int(ceil(n_samples / batch_size))

    println("\n📊 Training Configuration:")
    println("   Total samples: $n_samples")
    println("   Batch size: $batch_size")
    println("   Batches per epoch: $n_batches")
    println("   Learning rate: $learning_rate")
    println("   Epochs: $epochs")
    println("   Task: Regression (MSE loss)")
    println("   Convexity constraint: $is_convex")

    for epoch in 1:epochs
        println("=== Epoch $epoch ===")
        start_time = time()

        indexes = randperm(n_samples)
        epoch_loss = 0.0

        for batch_idx in 1:n_batches
            start_idx = (batch_idx - 1) * batch_size + 1
            end_idx = min(batch_idx * batch_size, n_samples)
            batch_indexes = indexes[start_idx:end_idx]

            x_batch = data_x[batch_indexes, :]
            y_batch = data_y[batch_indexes, :]

            val, grads = Flux.withgradient(model) do m
                mse_loss(m, x_batch, y_batch)
            end

            Flux.update!(opt, model, grads[1])

            if is_convex
                enforcing_convexity!(model)
            end

            epoch_loss += val

            if batch_idx % 50 == 0
                print("\r  Batch $batch_idx/$n_batches (loss: $(round(val, digits=6)))  ")
                flush(stdout)
            end
        end
        println()
        epoch_loss /= n_batches

        if is_convex && (epoch % 10 == 0 || epoch == 1)
            println("\n   🔧 Convexity check:")
            for (i, layer) in enumerate(model.hidden_layers)
                w = layer.weight
                n_neg = sum(w .< -1e-6)
                if n_neg > 0
                    println("      ⚠️  hidden_layer[$i]: $n_neg negative weights!")
                end
                println("      hidden_layer[$i]: min=$(round(minimum(w), digits=6)), max=$(round(maximum(w), digits=6))")
            end
        end

        test_loss = 0.0f0
        test_rmse = 0.0f0
        test_mae = 0.0f0
        if X_test !== nothing && y_test !== nothing
            y_pred_test = predict(model, Float32.(X_test))
            test_loss = mean((y_pred_test .- Float32.(y_test)) .^ 2)
            test_rmse = sqrt(test_loss)
            test_mae = mean(abs.(y_pred_test .- Float32.(y_test)))

            println("\n   📈 Test Metrics:")
            println("      MSE: $(round(test_loss, digits=6))")
            println("      RMSE: $(round(test_rmse, digits=6))")
            println("      MAE: $(round(test_mae, digits=6))")

            println("\n   📊 Prediction Stats:")
            println("      Mean: $(round(mean(y_pred_test), digits=4)), Std: $(round(std(y_pred_test), digits=4))")
            println("      Range: [$(round(minimum(y_pred_test), digits=4)), $(round(maximum(y_pred_test), digits=4))]")
            println("      Target mean: $(round(mean(y_test), digits=4)), Target std: $(round(std(y_test), digits=4))")

            if collect_metrics
                push!(metrics["test_losses"], test_loss)
                push!(metrics["test_rmse"], test_rmse)
                push!(metrics["test_mae"], test_mae)
                if epoch == epochs
                    metrics["final_predictions_sample"] = vec(y_pred_test)[1:min(100, length(y_pred_test))]
                end
            end
        end

        elapsed_time = time() - start_time
        
        if collect_metrics
            push!(metrics["train_losses"], epoch_loss)
        end
        
        @printf(" + train_loss: %.5e\n", epoch_loss)
        if X_test !== nothing && y_test !== nothing
            @printf(" + test_loss: %.5e\n", test_loss)
            @printf(" + test_rmse: %.5e\n", test_rmse)
            @printf(" + test_mae: %.5e\n", test_mae)
        end
        @printf(" + time: %.2f s\n", elapsed_time)

        open(log_file, "a") do io
            if X_test !== nothing && y_test !== nothing
                println(io, "$epoch,$epoch_loss,$test_loss,$test_rmse,$test_mae,$elapsed_time")
            else
                println(io, "$epoch,$epoch_loss,$elapsed_time")
            end
        end

        if epoch_loss < best_mse
            best_mse = epoch_loss
            best_model_path = joinpath(save_dir, "best_model.bson")
            save_model(model, best_model_path)
        end

        if epoch % 10 == 0
            checkpoint_path = joinpath(save_dir, "checkpoint_epoch_$epoch.bson")
            save_model(model, checkpoint_path)
        end
    end

    final_model_path = joinpath(save_dir, "final_model.bson")
    save_model(model, final_model_path)
    
    if collect_metrics
        # Collect input layer weights
        weights = vec(model.input_layer.weight)
        metrics["weights_sample"]["input_layer"] = Float64.(weights[1:min(20, length(weights))])

        # Collect hidden layer weights
        for (i, layer) in enumerate(model.hidden_layers)
            weights = vec(layer.weight)
            metrics["weights_sample"]["hidden_layer_$i"] = Float64.(weights[1:min(20, length(weights))])

            n_negative = sum(weights .< -1e-6)
            if n_negative > 0
                @warn "Hidden layer $i has $n_negative negative weights!"
            end
            @printf("  Hidden layer %d: min=%.6f, max=%.6f\n", i, minimum(weights), maximum(weights))
        end

        metrics_file = joinpath(save_dir, "metrics_julia.json")
        save_training_metrics(metrics, metrics_file)
    end
    
    return model
end