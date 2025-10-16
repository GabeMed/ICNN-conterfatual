"""
Structure for Fully Input Convex Neural Network (FICNN).
"""
mutable struct FICNN <: AbstractICNN
    n_features::Int
    n_labels::Int
    layers::Vector{Any}
    input_insertion_layers::Vector{Any}  # W(y) in the article
    hidden_layers::Vector{Any}           # W(z) in the article
    n_gradient_iterations::Int
end

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
is computed as: z_i = ReLU(W_i^(y) * [x, y] + W_i^(z) * z_{i-1})
"""
function FICNN(n_features::Int, n_labels::Int;
              hidden_sizes=[200, 200], n_gradient_iterations=30)
    layers = vcat(hidden_sizes, [1])
    input_insertion_layers = []
    hidden_layers = []
    input_size = n_features + n_labels

    for (i, sz) in enumerate(layers)
        push!(input_insertion_layers, Dense(input_size, sz))
        if i > 1
            push!(hidden_layers, Dense(layers[i-1], sz, bias=false))
        end
    end

    return FICNN(n_features, n_labels, layers,
                input_insertion_layers, hidden_layers, n_gradient_iterations)
end

"""
    (model::FICNN)(x, y; reuse=false)

Forward pass through the FICNN model.
"""
function (model::FICNN)(x, y; reuse=false)
    xy = hcat(x, y)
    prev_z = nothing

    for (i, sz) in enumerate(model.layers)
        z_components = []
        push!(z_components, model.input_insertion_layers[i](xy))

        if i > 1 && prev_z !== nothing
            push!(z_components, model.hidden_layers[i-1](prev_z))
        end

        z_total = sum(z_components)
        prev_z = sz != 1 ? relu.(z_total) : z_total
    end

    return prev_z
end

include("../training/trainer.jl")  # Include training-specific functions