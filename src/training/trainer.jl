"""
    enforcing_convexity!(model::AbstractICNN)

Enforces convexity by ensuring all weights in hidden layers are non-negative.
"""
function enforcing_convexity!(model::AbstractICNN)
    for layer in model.hidden_layers
        if layer !== nothing
            layer.weight .= max.(layer.weight, 0.0)
        end
    end
end

"""
    initialize_convex!(model::AbstractICNN)

Initializes the model weights to ensure convexity.
"""
function initialize_convex!(model::AbstractICNN)
    for layer in model.hidden_layers
        if layer !== nothing
            layer.weight .= abs.(layer.weight) ./ 10.0
        end
    end
end

"""
    predict(model::AbstractICNN, x, y_init; learning_rate=0.01, momentum=0.9)

Predict output y for input x using gradient descent.
"""
function predict(model::AbstractICNN, x, y_init; learning_rate=0.01, momentum=0.9)
    x = Float32.(x)
    y = Float32.(copy(y_init))
    velocity = zero(y)
    
    for _ in 1:model.n_gradient_iterations
        # Compute gradient of energy with respect to y
        grad = Zygote.ignore_derivatives() do
            Zygote.gradient(y_i -> sum(model(x, y_i)), y)[1]
        end
        
        previous_vel = velocity
        velocity = momentum .* previous_vel .- learning_rate .* grad
        y = y .- momentum .* previous_vel .+ (1.0 + momentum) .* velocity
    end
    
    return y
end

"""
    mse_loss(model::AbstractICNN, x, y_init, y_true)

Compute Mean Squared Error loss between predicted and true values.
"""
function mse_loss(model::AbstractICNN, x, y_init, y_true)
    x = Float32.(x)
    y_true = Float32.(y_true)
    # Evaluate energy at true y
    energy_true = model(x, y_true)
    # We want low energy for correct (x, y) pairs
    loss = mean(energy_true .^ 2)
    return loss, y_true
end

"""
    train!(model::AbstractICNN, data_x, data_y, epochs::Int = 100;
           learning_rate=0.001, batch_size=32, save_dir="./tmp", is_convex=true)

Train the ICNN model using mini-batch gradient descent.
"""
function train!(model::AbstractICNN, data_x, data_y, epochs::Int = 100;
                learning_rate=0.001, batch_size=32, save_dir = "./tmp",
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

    opt = Flux.setup(ADAM(learning_rate), model)
    
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

            batch_loss = 0.0            
            gs = gradient(model) do model
                loss, _ = mse_loss(model, x_batch, y_init, y_batch)
                batch_loss = loss
                loss
            end

            Flux.update!(opt, model, gs[1])
            epoch_loss += batch_loss
        end

        epoch_loss /= n_batches

        if is_convex
            enforcing_convexity!(model)
        end

        # Compute test metrics if test set provided
        test_loss = 0.0
        test_accuracy = 0.0
        if X_test !== nothing && y_test !== nothing
            y_init_test = fill(0.5f0, size(y_test))
            # Use predict for evaluation (30 iterations)
            y_pred_test = predict(model, Float32.(X_test), y_init_test)
            test_loss = mean((y_pred_test .- Float32.(y_test)) .^ 2)
            test_accuracy = mean((y_pred_test .> 0.5) .== Float32.(y_test))
            
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
        for (i, layer) in enumerate(model.input_insertion_layers)
            weights = vec(layer.weight)
            metrics["weights_sample"]["input_layer_$i"] = weights[1:min(20, length(weights))]
        end
        for (i, layer) in enumerate(model.hidden_layers)
            if layer !== nothing
                weights = vec(layer.weight)
                metrics["weights_sample"]["hidden_layer_$i"] = weights[1:min(20, length(weights))]
            end
        end
        
        # Save metrics to JSON
        metrics_file = joinpath(save_dir, "metrics_julia.json")
        save_training_metrics(metrics, metrics_file)
    end
    
    return model
end