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
    y = copy(y_init)
    velocity = zeros(size(y))

    for _ in 1:model.n_gradient_iterations
        energy = model(x, y)
        grad = gradient(y_i -> sum(model(x, y_i)), y)[1]

        previous_vel = copy(velocity)
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
    y_pred = predict(model, x, y_init)
    return mean((y_pred .- y_true).^2), y_pred
end

"""
    train!(model::AbstractICNN, data_x, data_y, epochs::Int = 100;
           learning_rate=0.001, batch_size=32, save_dir="./tmp", is_convex=true)

Train the ICNN model using mini-batch gradient descent.
"""
function train!(model::AbstractICNN, data_x, data_y, epochs::Int = 100;
                learning_rate=0.001, batch_size=32, save_dir = "./tmp",
                is_convex=true)
    # Create save directory and log file
    mkpath(save_dir)
    log_file = joinpath(save_dir, "training_log.csv")
    open(log_file, "w") do io
        println(io, "epoch,train_loss,val_loss")
    end

    params = Flux.params(model.input_insertion_layers, model.hidden_layers)
    opt = ADAM(learning_rate)
    
    if is_convex
        initialize_convex!(model)
    end

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
            gs = gradient(params) do
                loss, _ = mse_loss(model, x_batch, y_init, y_batch)
                batch_loss = loss
                loss
            end

            Flux.update!(opt, params, gs)
            epoch_loss += batch_loss
        end

        epoch_loss /= n_batches

        if is_convex
            enforcing_convexity!(model)
        end

        elapsed_time = time() - start_time
        @printf(" + loss: %.5e\n", epoch_loss)
        @printf(" + time: %.2f s\n", elapsed_time)

        # Log training progress
        open(log_file, "a") do io
            println(io, "$epoch,$epoch_loss,$elapsed_time")
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
    
    return model
end