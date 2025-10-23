"""
Structure for Fully Input Convex Neural Network (FICNN).

This implementation uses SEPARATE layers for x and y instead of concatenation.
This approach:
1. Simplifies gradient computation (avoids nested differentiation issues)
2. Maintains convexity in y (critical for MIP/Outer Approximation)
3. Is easier to integrate with JuMP/MathOptAI for counterfactual generation

Architecture ensures f(x,y) is convex in y through non-negative weights in hidden_layers.
"""
mutable struct FICNN <: AbstractICNN
    n_features::Int
    n_labels::Int
    layers::Vector{Int}
    input_x_layers::Vector{Dense}      # W^(x) - processes x input
    input_y_layers::Vector{Dense}      # W^(y) - processes y input
    hidden_layers::Vector{Dense}       # W^(z) - MUST be non-negative for convexity
    n_gradient_iterations::Int
end

Flux.@layer FICNN trainable=(input_x_layers, input_y_layers, hidden_layers)

"""
    FICNN(n_features::Int, n_labels::Int; hidden_sizes=[200, 200], n_gradient_iterations=30)

Constructor for FICNN model with separate x and y processing.

# Arguments
- `n_features::Int`: Number of input features (x)
- `n_labels::Int`: Number of output labels (y)
- `hidden_sizes::Vector{Int}`: Sizes of hidden layers
- `n_gradient_iterations::Int`: Number of gradient descent iterations for prediction

# Architecture (with separate x and y layers)
    z_0 = W_0^(x) * x + W_0^(y) * y + b_0
    z_i = ReLU(W_i^(x) * x + W_i^(y) * y + W_i^(z) * z_{i-1}) for i > 0

where W_i^(z) must be non-negative for convexity in y.
Convexity: Since ReLU preserves convexity and W^(z) ≥ 0, f(x,y) is convex in y.
"""
function FICNN(n_features::Int, n_labels::Int;
               hidden_sizes=[200, 200], n_gradient_iterations=30)

    layers = vcat(hidden_sizes, 1)  # e.g. [200, 200, 1]
    nL = length(layers)

    # Create separate layers for x and y inputs
    input_x_layers = Vector{Dense}(undef, nL)
    input_y_layers = Vector{Dense}(undef, nL)
    hidden_layers = Vector{Dense}(undef, nL-1)

    for i in 1:nL
        sz = layers[i]
        # Layers for x and y (with bias only on x layer to avoid redundancy)
        input_x_layers[i] = Dense(n_features => sz, bias=true)    # W^(x)_i
        input_y_layers[i] = Dense(n_labels => sz, bias=false)     # W^(y)_i (no bias)

        # Hidden layers (must be non-negative for convexity)
        if i > 1
            hidden_layers[i-1] = Dense(layers[i-1] => sz, bias=false)  # W^(z)_i
        end
    end

    model = FICNN(n_features, n_labels, layers,
                 input_x_layers, input_y_layers, hidden_layers,
                 n_gradient_iterations)

    # Validate architecture
    @assert length(model.input_x_layers) == length(model.layers)
    @assert length(model.input_y_layers) == length(model.layers)
    @assert length(model.hidden_layers) == length(model.layers) - 1

    return model
end

"""
    (model::FICNN)(x, y; reuse=false)

Forward pass through the FICNN model with separate x and y processing.

# Architecture
For each layer i:
    z_i = W^(x)_i * x + W^(y)_i * y + W^(z)_i * z_{i-1} (if i > 1)
    z_i = ReLU(z_i) if i < nL (hidden layers)
    z_i = z_i (linear) if i == nL (output layer)

Convexity: f(x,y) is convex in y because:
1. W^(y) and W^(z) weights contribute linearly to y
2. ReLU is convex
3. W^(z) ≥ 0 ensures convexity is preserved through layers
"""
function (model::FICNN)(x, y; reuse=false)
    # Ensure concrete Float32 arrays
    x = Float32.(x)
    y = Float32.(y)

    # Transpose to (features, batch) format for Flux Dense layers
    x_t = permutedims(x, (2, 1))  # (n_features, batch)
    y_t = permutedims(y, (2, 1))  # (n_labels, batch)

    prevZ = nothing
    z = nothing  # Initialize z outside loop to avoid scope issues
    nL = length(model.layers)

    @inbounds for i in 1:nL
        # Compute z_x = W^(x)_i * x + b_i
        z_x = model.input_x_layers[i](x_t)

        # Compute z_y = W^(y)_i * y (no bias)
        z_y = model.input_y_layers[i](y_t)

        # Combine x and y contributions
        z = z_x .+ z_y

        # Add hidden layer contribution if not first layer
        if prevZ !== nothing
            z_z = model.hidden_layers[i-1](prevZ)
            z = z .+ z_z
        end

        # Apply ReLU for hidden layers, linear for output layer
        if i < nL
            z = relu.(z)
        end

        prevZ = z
    end

    return permutedims(z, (2, 1))  # (batch, 1)
end

include("../training/trainer.jl")  # Include training-specific functions