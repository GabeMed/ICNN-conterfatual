"""
Structure for Fully Input Convex Neural Network (FICNN).
Split input layers to avoid concatenation during nested AD.
"""
mutable struct FICNN <: AbstractICNN
    n_features::Int
    n_labels::Int
    layers::Vector{Int}
    input_x_layers::Vector{Dense}                      # W^x for x inputs
    input_y_layers::Vector{Dense}                      # W^y for y inputs  
    hidden_layers::Vector{Dense}                       # W(z), length = nL-1
    n_gradient_iterations::Int
end

Flux.@layer FICNN trainable=(input_x_layers, input_y_layers, hidden_layers)

"""
    FICNN(n_features::Int, n_labels::Int; hidden_sizes=[200, 200], n_gradient_iterations=30)

Constructor for FICNN model.

# Arguments
- `n_features::Int`: Number of input features
- `n_labels::Int`: Number of output labels
- `hidden_sizes::Vector{Int}`: Sizes of hidden layers
- `n_gradient_iterations::Int`: Number of gradient descent iterations for prediction

# Architecture
The FICNN implements a fully input convex neural network where each layer's output
is computed as: z_i = ReLU(W_i^x * x + W_i^y * y + W_i^z * z_{i-1})
No concatenation needed - avoids nested AD issues with ChainRules.
"""
function FICNN(n_features::Int, n_labels::Int;
               hidden_sizes=[200, 200], n_gradient_iterations=30)

    layers = vcat(hidden_sizes, 1)                     # e.g. [200,200,1]
    nL = length(layers)

    # Separate x and y input layers - NO CONCATENATION in forward pass
    input_x_layers = Vector{Dense}(undef, nL)
    input_y_layers = Vector{Dense}(undef, nL)
    hidden_layers  = Vector{Dense}(undef, nL-1)

    for i in 1:nL
        sz = layers[i]
        input_x_layers[i] = Dense(n_features => sz)                    # W^x_i
        input_y_layers[i] = Dense(n_labels => sz)                      # W^y_i
        if i > 1
            hidden_layers[i-1] = Dense(layers[i-1] => sz, bias=false)  # W^z_i
        end
    end

    model = FICNN(n_features, n_labels, layers,
                 input_x_layers, input_y_layers, hidden_layers,
                 n_gradient_iterations)
    
    # Assert the vector lengths
    @assert length(model.input_x_layers) == length(model.layers)
    @assert length(model.input_y_layers) == length(model.layers)
    @assert length(model.hidden_layers) == length(model.layers) - 1
    
    return model
end

"""
    (model::FICNN)(x, y; reuse=false)

Forward pass through the FICNN model.
NO concatenation - processes x and y separately then adds.
This is critical for nested AD compatibility.
"""
function (model::FICNN)(x, y; reuse=false)
    # Ensure concrete Float32 arrays
    x = Float32.(x)
    y = Float32.(y)
    
    # Transpose to (features, batch) format for Flux Dense layers
    x_t = permutedims(x, (2, 1))  # (n_features, batch)
    y_t = permutedims(y, (2, 1))  # (n_labels, batch)

    z = nothing
    nL = length(model.layers)

    @inbounds for i in 1:nL
        # Process x and y separately, then ADD (no concatenation!)
        a_x = model.input_x_layers[i](x_t)   # W^x_i * x
        a_y = model.input_y_layers[i](y_t)   # W^y_i * y
        a_xy = a_x .+ a_y                     # Sum instead of concat
        
        # pre-activation = W^x_i*x + W^y_i*y  (+ W^z_i * z_{i-1} when i>1)
        pre = (i == 1) ? a_xy : (a_xy .+ model.hidden_layers[i-1](z))

        # hidden layers use ReLU; last layer is linear
        z = (i < nL) ? relu.(pre) : pre
    end

    return permutedims(z, (2, 1))  # (batch, 1)
end

include("../training/trainer.jl")  # Include training-specific functions