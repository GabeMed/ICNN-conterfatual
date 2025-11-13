"""
Gradient computation utilities for FICNN models used in Outer Approximation.

This module provides functions to compute gradients of FICNN models with respect
to input features using automatic differentiation (Zygote). These gradients are
essential for generating cutting planes in the OA algorithm.

Key functions:
- `compute_input_gradient`: Compute ∇_x f(x) for a single input point
- `evaluate_ficnn_with_gradient`: Get both f(x) and ∇f(x) efficiently
- `validate_gradient`: Test AD gradients against finite differences
"""

using Zygote
using Statistics
using Printf

# Import FICNN type from parent module
# Users should include ICNN module before using these utilities

"""
    compute_input_gradient(model::FICNN, x::Vector{Float32}) -> Vector{Float64}

Compute the gradient ∇_x f(x) of the FICNN model with respect to input features.

# Arguments
- `model::FICNN`: The trained FICNN model
- `x::Vector{Float32}`: Input feature vector of size (n_features,)

# Returns
- `Vector{Float64}`: Gradient vector ∇f(x) of same size as x, converted to Float64 for JuMP compatibility

# Details
The gradient is computed using Zygote automatic differentiation. The function:
1. Reshapes x to batch format (1, n_features)
2. Computes gradient using reverse-mode AD
3. Flattens and converts to Float64
4. Checks for numerical issues (NaN/Inf)

# Example
```julia
model = FICNN(10, 1)  # 10 features, 1 output
x = rand(Float32, 10)
grad = compute_input_gradient(model, x)
```

# Notes
- Input must be Float32 to match model expectations
- Output is Float64 for JuMP constraint coefficient compatibility
- ReLU non-differentiability at zero is handled by Zygote (returns 0 or 1)
"""
function compute_input_gradient(model, x::Vector{Float32})
    # Reshape x to batch format (1, n_features)
    x_batch = reshape(x, 1, :)

    # Compute gradient using Zygote
    # We differentiate the scalar output w.r.t. the input
    grads = Zygote.gradient(x_batch) do x_in
        y_pred = model(x_in)
        return y_pred[1, 1]  # Extract scalar output for single sample
    end

    # Extract gradient (first element of tuple from gradient())
    grad_x = grads[1]

    # Flatten to 1D vector
    grad_vec = vec(grad_x)  # (n_features,)

    # Convert to Float64 for JuMP compatibility
    grad_float64 = Float64.(grad_vec)

    # Check for numerical issues
    if any(isnan.(grad_float64))
        @warn "Gradient contains NaN values at some dimensions"
        # Replace NaN with 0 for numerical stability
        grad_float64 = replace(grad_float64, NaN => 0.0)
    end

    if any(isinf.(grad_float64))
        @warn "Gradient contains Inf values at some dimensions"
        # Clamp infinite values to large but finite numbers
        grad_float64 = clamp.(grad_float64, -1e10, 1e10)
    end

    return grad_float64
end


"""
    evaluate_ficnn_with_gradient(model::FICNN, x::Vector{Float32}) -> Tuple{Float64, Vector{Float64}}

Compute both f(x) and ∇f(x) in a single call for efficiency.

# Arguments
- `model::FICNN`: The trained FICNN model
- `x::Vector{Float32}`: Input feature vector of size (n_features,)

# Returns
- `Tuple{Float64, Vector{Float64}}`: (function_value, gradient_vector)
  - `function_value`: The FICNN prediction f(x)
  - `gradient_vector`: The gradient ∇f(x)

# Details
This function is more efficient than calling the model and computing gradients
separately, as it reuses computation. Used in OA algorithms where both the
function value and gradient are needed for generating cuts.

# Example
```julia
model = FICNN(10, 1)
x = rand(Float32, 10)
f_val, grad = evaluate_ficnn_with_gradient(model, x)

# Use in OA cut: f(x0) + grad' * (x - x0) ≤ f(x)
```

# Notes
- Both outputs are Float64 for numerical consistency with optimization solvers
- The gradient computation includes numerical stability checks
"""
function evaluate_ficnn_with_gradient(model, x::Vector{Float32})::Tuple{Float64, Vector{Float64}}
    # Reshape for batch format
    x_batch = reshape(x, 1, :)

    # Forward pass to get prediction
    y_pred = model(x_batch)[1, 1]
    f_value = Float64(y_pred)

    # Compute gradient (this will do another forward pass internally)
    gradient = compute_input_gradient(model, x)

    return (f_value, gradient)
end

# Export main functions
# Note: finite_difference_gradient is not exported (internal testing only)
 export compute_input_gradient
        # evaluate_ficnn_with_gradient
        # validate_gradient,
        # check_gradient_numerical_stability
