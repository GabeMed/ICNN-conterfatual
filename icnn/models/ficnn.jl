"""
Structure for Fully Input Convex Neural Network (FICNN).

Simplified feed-forward architecture for regression tasks:
- Direct mapping: x → y
- No energy-based learning
- Maintains convexity in output through non-negative weights in hidden layers

Architecture:
    z_0 = W_0 * x + b_0
    z_i = ReLU(W_i * z_{i-1})  for i > 0, where W_i ≥ 0
    y = z_final (linear output)

Convexity: Since W_i ≥ 0 and ReLU is convex, the network output is convex.
"""
mutable struct FICNN <: AbstractICNN
    n_features::Int
    n_output::Int
    layers::Vector{Int}
    input_layer::Dense              # W_0 - first layer (can be any value)
    hidden_layers::Vector{Dense}    # W_i - MUST be non-negative for convexity
end

Flux.@layer FICNN trainable=(input_layer, hidden_layers)

"""
    FICNN(n_features::Int, n_output::Int; hidden_sizes=[200, 200])

Constructor for simplified FICNN model for regression.

# Arguments
- `n_features::Int`: Number of input features (x)
- `n_output::Int`: Number of output values (y) - typically 1 for scalar regression
- `hidden_sizes::Vector{Int}`: Sizes of hidden layers

# Architecture
    z_0 = W_0 * x + b_0
    z_i = ReLU(W_i * z_{i-1})  for i > 0, where W_i ≥ 0
    y = z_final (linear output)

where W_i (hidden_layers) must be non-negative for convexity.
"""
function FICNN(n_features::Int, n_output::Int=1; hidden_sizes=[200, 200])

    layers = vcat(hidden_sizes, n_output)  # e.g. [200, 200, 1]
    nL = length(layers)

    # First layer: processes input x (no convexity constraint)
    input_layer = Dense(n_features => layers[1], bias=true)

    # Hidden layers: must be non-negative for convexity
    hidden_layers = Vector{Dense}(undef, nL-1)
    for i in 1:(nL-1)
        hidden_layers[i] = Dense(layers[i] => layers[i+1], bias=false)
    end

    model = FICNN(n_features, n_output, layers, input_layer, hidden_layers)

    # Validate architecture
    @assert length(model.hidden_layers) == length(model.layers) - 1

    return model
end

"""
    (model::FICNN)(x)

Forward pass through the simplified FICNN model.

# Architecture
    z_0 = W_0 * x + b_0
    z_i = ReLU(W_i * z_{i-1})  for i > 0, where W_i ≥ 0
    y = z_final (linear output)

Convexity: The output is convex w.r.t. input because:
1. First layer is affine
2. ReLU is convex and preserves convexity
3. W_i ≥ 0 ensures positive linear combinations preserve convexity
"""
function (model::FICNN)(x)
    # Ensure concrete Float32 arrays
    x = Float32.(x)

    # Transpose to (features, batch) format for Flux Dense layers
    x_t = permutedims(x, (2, 1))  # (n_features, batch)

    # First layer: z_0 = W_0 * x + b_0
    z = model.input_layer(x_t)
    z = relu.(z)

    # Hidden layers: z_i = ReLU(W_i * z_{i-1})
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

include("../training/trainer.jl")  # Include training-specific functions