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


# """
#     finite_difference_gradient(model::FICNN, x::Vector{Float32}; h::Float64=1e-5) -> Vector{Float64}

# Compute gradient using central finite differences for validation/testing.

# # Arguments
# - `model::FICNN`: The trained FICNN model
# - `x::Vector{Float32}`: Input feature vector of size (n_features,)
# - `h::Float64`: Step size for finite differences (default: 1e-5)

# # Returns
# - `Vector{Float64}`: Approximate gradient vector computed via finite differences

# # Details
# Uses central differences formula for each dimension i:
#     ∂f/∂x_i ≈ [f(x + h*e_i) - f(x - h*e_i)] / (2h)

# where e_i is the i-th unit vector.

# # Example
# ```julia
# model = FICNN(10, 1)
# x = rand(Float32, 10)
# grad_fd = finite_difference_gradient(model, x, h=1e-6)
# grad_ad = compute_input_gradient(model, x)
# @assert isapprox(grad_fd, grad_ad, rtol=1e-4)
# ```

# # Notes
# - This is for testing only - much slower than automatic differentiation
# - Accuracy depends on step size h (too small → roundoff, too large → truncation)
# - Not recommended for production use in OA algorithm
# """
# function finite_difference_gradient(model, x::Vector{Float32}; h::Float64=1e-5)::Vector{Float64}
#     n = length(x)
#     grad = zeros(Float64, n)

#     # Compute partial derivatives using central differences
#     for i in 1:n
#         # Forward step: x + h*e_i
#         x_plus = copy(x)
#         x_plus[i] += Float32(h)
#         x_plus_batch = reshape(x_plus, 1, :)
#         f_plus = Float64(model(x_plus_batch)[1, 1])

#         # Backward step: x - h*e_i
#         x_minus = copy(x)
#         x_minus[i] -= Float32(h)
#         x_minus_batch = reshape(x_minus, 1, :)
#         f_minus = Float64(model(x_minus_batch)[1, 1])

#         # Central difference: [f(x+h) - f(x-h)] / (2h)
#         grad[i] = (f_plus - f_minus) / (2 * h)
#     end

#     return grad
# end


# """
#     validate_gradient(model::FICNN, x::Vector{Float32}; rtol::Float64=1e-4, verbose::Bool=true) -> Bool

# Validate automatic differentiation gradient against finite differences.

# # Arguments
# - `model::FICNN`: The trained FICNN model
# - `x::Vector{Float32}`: Input feature vector to test at
# - `rtol::Float64`: Relative tolerance for comparison (default: 1e-4)
# - `verbose::Bool`: Print diagnostic information (default: true)

# # Returns
# - `Bool`: true if gradients match within tolerance, false otherwise

# # Details
# Computes gradients using both automatic differentiation and finite differences,
# then checks if they agree within relative tolerance. Useful for:
# - Debugging custom layers or operations
# - Verifying Zygote compatibility
# - Ensuring numerical stability

# # Example
# ```julia
# model = FICNN(10, 1)
# initialize_convex!(model)
# x = rand(Float32, 10)

# # Validate gradient computation
# is_valid = validate_gradient(model, x, rtol=1e-3, verbose=true)
# ```

# # Notes
# - Finite differences are approximate, so perfect agreement is not expected
# - ReLU creates discontinuities where finite differences may differ from AD
# - If validation fails, check for:
#   * In-place operations in forward pass
#   * Type instabilities
#   * Custom operations without gradient rules
# """
# function validate_gradient(model, x::Vector{Float32}; rtol::Float64=1e-4, verbose::Bool=true)::Bool
#     # Compute gradients both ways
#     grad_auto = compute_input_gradient(model, x)
#     grad_fd = finite_difference_gradient(model, x)

#     # Compute differences
#     abs_diff = abs.(grad_auto .- grad_fd)
#     max_abs_diff = maximum(abs_diff)

#     # Compute relative error (avoid division by zero)
#     rel_diff = abs_diff ./ (abs.(grad_fd) .+ 1e-10)
#     max_rel_diff = maximum(rel_diff)

#     # Check if they're close
#     matches = isapprox(grad_auto, grad_fd, rtol=rtol)

#     if verbose
#         println("=" ^ 60)
#         println("Gradient Validation Report")
#         println("=" ^ 60)
#         println("Input dimension: ", length(x))
#         println("Max absolute difference: ", round(max_abs_diff, digits=8))
#         println("Max relative difference: ", round(max_rel_diff, digits=8))
#         println("Relative tolerance: ", rtol)
#         println("Match: ", matches ? "✓" : "✗")

#         if !matches
#             println("\n" * "─" ^ 60)
#             println("First 10 dimensions comparison:")
#             println("─" ^ 60)
#             println("Dim |   Auto Diff   |  Finite Diff  | Abs Diff  | Rel Diff")
#             println("─" ^ 60)
#             for i in 1:min(10, length(grad_auto))
#                 @printf("%3d | %13.6e | %13.6e | %.3e | %.3e\n",
#                         i, grad_auto[i], grad_fd[i], abs_diff[i], rel_diff[i])
#             end
#             println("─" ^ 60)

#             # Show statistics
#             println("\nGradient Statistics:")
#             println("  Auto Diff - Mean: ", round(mean(grad_auto), digits=6))
#             println("  Auto Diff - Std:  ", round(std(grad_auto), digits=6))
#             println("  Auto Diff - Min:  ", round(minimum(grad_auto), digits=6))
#             println("  Auto Diff - Max:  ", round(maximum(grad_auto), digits=6))
#             println("\n  Finite Diff - Mean: ", round(mean(grad_fd), digits=6))
#             println("  Finite Diff - Std:  ", round(std(grad_fd), digits=6))
#             println("  Finite Diff - Min:  ", round(minimum(grad_fd), digits=6))
#             println("  Finite Diff - Max:  ", round(maximum(grad_fd), digits=6))
#         else
#             println("\nGradient validation passed! ✓")
#         end
#         println("=" ^ 60)
#     end

#     return matches
# end


# """
#     check_gradient_numerical_stability(model::FICNN, x_samples::Vector{Vector{Float32}}; verbose::Bool=true) -> Dict

# Check gradient numerical stability across multiple sample points.

# # Arguments
# - `model::FICNN`: The trained FICNN model
# - `x_samples::Vector{Vector{Float32}}`: Collection of input points to test
# - `verbose::Bool`: Print diagnostic information (default: true)

# # Returns
# - `Dict`: Statistics about gradient stability including:
#   * `nan_count`: Number of samples with NaN gradients
#   * `inf_count`: Number of samples with Inf gradients
#   * `mean_gradient_norm`: Average L2 norm of gradients
#   * `max_gradient_norm`: Maximum L2 norm of gradients
#   * `gradient_norms`: Vector of all gradient norms

# # Example
# ```julia
# model = FICNN(10, 1)
# samples = [rand(Float32, 10) for _ in 1:100]
# stats = check_gradient_numerical_stability(model, samples)
# ```
# """
# function check_gradient_numerical_stability(
#     model,
#     x_samples::Vector{Vector{Float32}};
#     verbose::Bool=true
# )
#     n_samples = length(x_samples)
#     nan_count = 0
#     inf_count = 0
#     gradient_norms = Float64[]

#     for x in x_samples
#         grad = compute_input_gradient(model, x)

#         if any(isnan.(grad))
#             nan_count += 1
#         end

#         if any(isinf.(grad))
#             inf_count += 1
#         end

#         # Compute L2 norm
#         grad_norm = sqrt(sum(grad .^ 2))
#         push!(gradient_norms, grad_norm)
#     end

#     stats = Dict(
#         "nan_count" => nan_count,
#         "inf_count" => inf_count,
#         "mean_gradient_norm" => mean(gradient_norms),
#         "max_gradient_norm" => maximum(gradient_norms),
#         "min_gradient_norm" => minimum(gradient_norms),
#         "std_gradient_norm" => std(gradient_norms),
#         "gradient_norms" => gradient_norms
#     )

#     if verbose
#         println("\n" * "=" ^ 60)
#         println("Gradient Numerical Stability Report")
#         println("=" ^ 60)
#         println("Total samples tested: ", n_samples)
#         println("NaN gradients: ", nan_count, " (", round(100*nan_count/n_samples, digits=2), "%)")
#         println("Inf gradients: ", inf_count, " (", round(100*inf_count/n_samples, digits=2), "%)")
#         println("\nGradient Norm Statistics:")
#         println("  Mean: ", round(stats["mean_gradient_norm"], digits=6))
#         println("  Std:  ", round(stats["std_gradient_norm"], digits=6))
#         println("  Min:  ", round(stats["min_gradient_norm"], digits=6))
#         println("  Max:  ", round(stats["max_gradient_norm"], digits=6))
#         println("=" ^ 60)
#     end

#     return stats
# end


# Export main functions
# Note: finite_difference_gradient is not exported (internal testing only)
 export compute_input_gradient
        # evaluate_ficnn_with_gradient
        # validate_gradient,
        # check_gradient_numerical_stability
