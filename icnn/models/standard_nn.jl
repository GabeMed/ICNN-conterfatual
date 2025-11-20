"""
Standard feedforward neural network for non-convex regression tasks.

This is a regular multilayer perceptron (MLP) with ReLU activations,
suitable for learning non-convex mappings such as the AC Optimal Power Flow problem.

Unlike FICNN, this model:
- Does NOT maintain convexity in the output
- Does NOT require non-negative weight constraints
- Does NOT need convexity enforcement after gradient updates
- Uses standard backpropagation without special constraints

Architecture:
    z_0 = ReLU(W_0 * x + b_0)
    z_i = ReLU(W_i * z_{i-1} + b_i)  for i > 0
    y = W_final * z_final + b_final (linear output)
"""
mutable struct StandardNN <: AbstractICNN
    n_features::Int
    n_output::Int
    layers::Vector{Int}
    input_layer::Dense              # First layer (with bias)
    hidden_layers::Vector{Dense}    # All subsequent layers (all with bias, unlike FICNN)
end

Flux.@layer StandardNN trainable=(input_layer, hidden_layers)

"""
    StandardNN(n_features::Int, n_output::Int=1; hidden_sizes=[200, 200])

Constructor for standard feedforward neural network for regression.

# Arguments
- `n_features::Int`: Number of input features (x)
- `n_output::Int`: Number of output values (y) - typically 1 for scalar regression
- `hidden_sizes::Vector{Int}`: Sizes of hidden layers

# Architecture
A standard MLP:
    z_0 = ReLU(W_0 * x + b_0)
    z_i = ReLU(W_i * z_{i-1} + b_i)  for i > 0
    y = W_final * z_final + b_final (linear output)

All weights can be positive or negative - no convexity constraints.

# Example
```julia
model = StandardNN(236, 1; hidden_sizes=[500, 500])
y_pred = model(x)  # Forward pass
```
"""
function StandardNN(n_features::Int, n_output::Int=1; hidden_sizes=[500, 500])

    layers = vcat(hidden_sizes, n_output)  # e.g. [200, 200, 1]
    nL = length(layers)

    # First layer: processes input x (with bias)
    input_layer = Dense(n_features => layers[1], bias=true)

    # Hidden layers: all with bias (unlike FICNN which has bias=false)
    hidden_layers = Vector{Dense}(undef, nL-1)
    for i in 1:(nL-1)
        hidden_layers[i] = Dense(layers[i] => layers[i+1], bias=true)
    end

    model = StandardNN(n_features, n_output, layers, input_layer, hidden_layers)

    # Validate architecture
    @assert length(model.hidden_layers) == length(model.layers) - 1

    return model
end

"""
    (model::StandardNN)(x)

Forward pass through the standard neural network.

# Architecture
    z_0 = ReLU(W_0 * x + b_0)
    z_i = ReLU(W_i * z_{i-1} + b_i)  for i > 0
    y = W_final * z_final + b_final (linear output)

Standard feedforward pass - no convexity constraints.
"""
function (model::StandardNN)(x)
    # Ensure concrete Float32 arrays
    x = Float32.(x)

    # Transpose to (features, batch) format for Flux Dense layers
    x_t = permutedims(x, (2, 1))  # (n_features, batch)

    # Input layer with ReLU
    z = model.input_layer(x_t)
    z = relu.(z)

    # Hidden layers
    nL = length(model.hidden_layers)
    @inbounds for i in 1:nL
        z = model.hidden_layers[i](z)
        # Apply ReLU for all layers except the last (output is linear)
        if i < nL
            z = relu.(z)
        end
    end

    return permutedims(z, (2, 1))  # (batch, n_output)
end

"""
No convexity enforcement needed for standard NN.
This function is a no-op for StandardNN but maintains interface compatibility.
"""
function enforcing_convexity!(model::StandardNN)
    # Standard NN - no convexity constraints to enforce
    return nothing
end

"""
Standard initialization for neural networks.
This function is a no-op for StandardNN but maintains interface compatibility.
"""
function initialize_convex!(model::StandardNN)
    # Standard NN uses default Flux initialization
    # No special initialization needed
    return nothing
end

# Note: trainer.jl is included in the main ICNN.jl module
# No need to include it here to avoid duplicate docstring warnings
