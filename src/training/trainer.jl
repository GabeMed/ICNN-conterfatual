"""
    enforcing_convexity!(model::AbstractICNN)

Projects W^(z) weights to be non-negative to maintain convexity in y.
Must be called after each training step.
"""
function enforcing_convexity!(model::AbstractICNN)
    @inbounds for layer in model.hidden_layers
        layer.weight .= max.(layer.weight, 0.0f0)
    end
end

"""
    initialize_convex!(model::AbstractICNN)

Initializes W^(z) weights to small positive values.
Should be called before training starts
"""
function initialize_convex!(model::AbstractICNN)
    @inbounds for layer in model.hidden_layers
        # small positive init
        layer.weight .= abs.(layer.weight) ./ 10.0f0
    end
end

"""
    predict(model::AbstractICNN, x, y_init; learning_rate=0.01f0, momentum=0.9f0)

Performs inference by minimizing the energy function f(x, y) w.r.t. y via gradient descent.

Given observed input x, finds optimal y:
    y* = argmin_y f(x, y)

Using gradient descent with momentum starting from y_init.
"""
function predict(model::AbstractICNN, x, y_init; 
                learning_rate=0.01f0, 
                momentum=0.9f0)
    
    x = Float32.(x)
    y = Float32.(copy(y_init))
    velocity = zero(y)
    
    for iter in 1:model.n_gradient_iterations
        # Compute ∇_y f(x, y) via pullback (differentiable for the outer grad)
        _, back = Zygote.pullback(y -> sum(model(x, y)), y)
        grad_y = back(1f0)[1]
        
        # Momentum update 
        prev_velocity = velocity
        velocity = momentum .* prev_velocity .- learning_rate .* grad_y
        y = y .- momentum .* prev_velocity .+ (1.0f0 + momentum) .* velocity
    end
    
    return y
end

"""
    mse_loss(model::AbstractICNN, x, y_init, y_true)

Computes MSE loss between predicted y and true y.

The prediction is obtained by minimizing the energy function:
    y_pred = argmin_y f(x, y)

Then we compute MSE(y_pred, y_true).

"""
function mse_loss(model::AbstractICNN, x, y_init, y_true)
    x = Float32.(x)
    y_init = Float32.(y_init)
    y_true = Float32.(y_true)
    
    # Minimize energy to get prediction 
    y_pred = predict(model, x, y_init)
    
    # MSE loss
    return mean((y_pred .- y_true) .^ 2)
end

"""
    train!(model::AbstractICNN, data_x, data_y, epochs::Int = 100;
           learning_rate=0.001, batch_size=32, save_dir="./tmp", is_convex=true)

Train the ICNN model using mini-batch gradient descent.
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
        "test_accuracies" => Float64[],
        "predictions" => Float64[],
        "final_predictions_sample" => Float64[],
        "weights_sample" => Dict()
    )
    
    # CSV header
    if X_test !== nothing && y_test !== nothing
        open(log_file, "w") do io
            println(io, "epoch,train_loss,test_loss,test_accuracy,time")
        end
    else
        open(log_file, "w") do io
            println(io, "epoch,train_loss,time")
        end
    end

    if is_convex
        initialize_convex!(model)
    end

    opt = Flux.setup(Adam(learning_rate), model)
    
    best_mse = Inf
    n_samples = size(data_x, 1)
    n_batches = Int(ceil(n_samples / batch_size))

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
            y_init = fill(0.5f0, size(y_batch))

            # Flux.withgradient returns val and grads
            val, grads = Flux.withgradient(model) do m
                mse_loss(m, x_batch, y_init, y_batch)
            end
            
            Flux.update!(opt, model, grads[1])
            epoch_loss += val
        end

        epoch_loss /= n_batches

        if is_convex
            enforcing_convexity!(model)
        end

        # Compute test metrics if test set provided
        test_loss = 0.0f0
        test_accuracy = 0.0f0
        if X_test !== nothing && y_test !== nothing
            y_init_test = fill(0.5f0, size(y_test))
            # Use predict for evaluation
            y_pred_test = predict(model, Float32.(X_test), y_init_test)
            test_loss = mean((y_pred_test .- Float32.(y_test)) .^ 2)
            test_accuracy = mean((y_pred_test .> 0.5f0) .== Float32.(y_test))
            
            if collect_metrics
                push!(metrics["test_losses"], test_loss)
                push!(metrics["test_accuracies"], test_accuracy)
                
                # Store predictions on last epoch
                if epoch == epochs
                    metrics["final_predictions_sample"] = vec(y_pred_test)[1:min(100, length(y_pred_test))]
                end
            end
        end

        elapsed_time = time() - start_time
        
        # Collect training loss
        if collect_metrics
            push!(metrics["train_losses"], epoch_loss)
        end
        
        # Print metrics
        @printf(" + train_loss: %.5e\n", epoch_loss)
        if X_test !== nothing && y_test !== nothing
            @printf(" + test_loss: %.5e\n", test_loss)
            @printf(" + test_accuracy: %.2f%%\n", test_accuracy * 100)
        end
        @printf(" + time: %.2f s\n", elapsed_time)

        # Log training progress
        open(log_file, "a") do io
            if X_test !== nothing && y_test !== nothing
                println(io, "$epoch,$epoch_loss,$test_loss,$test_accuracy,$elapsed_time")
            else
                println(io, "$epoch,$epoch_loss,$elapsed_time")
            end
        end

        # Save best model
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
    
    # Collect weight samples for comparison
    if collect_metrics
        # Sample from x input layers
        for (i, layer) in enumerate(model.input_x_layers)
            weights = vec(layer.weight)
            sample_size = min(20, length(weights))
            metrics["weights_sample"]["input_x_$i"] =
                Float64.(weights[1:sample_size])
        end
        
        # Sample from y input layers
        for (i, layer) in enumerate(model.input_y_layers)
            weights = vec(layer.weight)
            sample_size = min(20, length(weights))
            metrics["weights_sample"]["input_y_$i"] =
                Float64.(weights[1:sample_size])
        end

        # Sample from hidden layers (W^(z) - should be non-negative)
        for (i, layer) in enumerate(model.hidden_layers)
            weights = vec(layer.weight)
            sample_size = min(20, length(weights))
            metrics["weights_sample"]["hidden_layer_$i"] =
                Float64.(weights[1:sample_size])

            # Verify non-negativity (important!)
            n_negative = sum(weights .< -1e-6)  # tolerance
            if n_negative > 0
                @warn "Hidden layer $i has $n_negative significantly negative weights!"
            end
            @printf("  Hidden layer %d: min=%.6f, max=%.6f\n",
                    i, minimum(weights), maximum(weights))
        end
        
        # Save metrics to JSON
        metrics_file = joinpath(save_dir, "metrics_julia.json")
        save_training_metrics(metrics, metrics_file)
    end
    
    return model
end