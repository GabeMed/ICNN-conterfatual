"""
Outer Approximation (OA) algorithm for ICNN counterfactual generation.

Given a factual input x_factual and a target output y_target, find a nearby
counterfactual x' such that NN(x') satisfies the target constraint by iteratively
solving a MILP master problem with linearized neural network constraints (OA cuts).

The OA method exploits the convexity of the FICNN to replace the full MIP
encoding with a series of linear cuts, resulting in significantly faster solve times.

# Algorithm Overview

1. Build master MILP with decision variables x, sparsity indicators
2. Add direct convex constraint: f(x) ≤ y_target + ε via outer approximation cuts
3. Initialize with OA cut at x_factual
4. Iterate:
   a. Solve master MILP → get candidate x_k
   b. Evaluate NN at x_k → get f(x_k) and ∇f(x_k)
   c. Check feasibility: f(x_k) ≤ y_target + ε
   d. Update upper bound if feasible
   e. Add OA cut to tighten approximation
   f. Check convergence
5. Return best feasible solution

# Key Differences from Full MIP
- No hidden layer variables z_i
- No ReLU big-M constraints
- Neural network represented implicitly via OA cuts
- Master problem grows by one constraint per iteration
- Exploits convexity for provably correct solutions
"""

using JuMP
using Gurobi
const MOI = JuMP.MOI  # Use MOI from JuMP
using LinearAlgebra
using Printf

# Import FICNN model type
# Users should ensure ICNN module is loaded before using this
# include("../../icnn/ICNN.jl")
# using .ICNN: FICNN

# Import gradient utilities
include("../utils/gradient_utils.jl")
using .Main: evaluate_ficnn_with_gradient, compute_input_gradient

# Track cut count globally for naming
mutable struct CutCounter
    count::Int
end

const cut_counter = CutCounter(0)

function reset_cut_counter!()
    cut_counter.count = 0
end


"""
    build_master_problem_oa(
        n_features::Int,
        x_factual::Vector{Float32},
        y_target::Float64;
        sparsity_weight::Float64=0.1,
        x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
        epsilon::Float64=0.01,
        immutable_indices::Vector{Int}=Int[]
    ) -> Tuple{Model, Vector{VariableRef}, Vector{VariableRef}}

Build the master MILP problem for outer approximation.

# Formulation

Minimize: ||x - x_factual||_1 + sparsity_weight * num_changed
Subject to: f(x) ≤ y_target + ε (enforced via OA cuts)

The convex constraint f(x) ≤ y_target + ε is approximated by outer approximation cuts:
  f(x_k) + ∇f(x_k)·(x - x_k) ≤ y_target + ε

Since f is convex, this linearization over-estimates f(x), creating a progressively
tighter outer approximation of the feasible set.

# Arguments
- `n_features::Int`: Number of input features
- `x_factual::Vector{Float32}`: Factual input point
- `y_target::Float64`: Target output value
- `sparsity_weight::Float64`: Weight for sparsity penalty (default: 0.1)
- `x_bounds::Tuple{Float64, Float64}`: Bounds for input variables (default: (0.0, 1.0))
- `epsilon::Float64`: Tolerance for target matching (default: 0.01)
- `immutable_indices::Vector{Int}`: Indices that cannot be changed (default: [])

# Returns
- `Tuple{Model, Vector{VariableRef}, Vector{VariableRef}}`:
  - `model`: JuMP model
  - `x_var`: Counterfactual input variables
  - `changed_var`: Binary sparsity indicators

# Details
- No hidden layer variables (z_i)
- No ReLU big-M constraints
- Neural network constraint represented implicitly by OA cuts
- OA cuts added iteratively during solve

# Objective
Minimize: ||x - x_factual||_1 + sparsity_weight * num_changed
         = sum(delta_pos + delta_neg) + sparsity_weight * sum(changed)

# Example
```julia
model, x, changed = build_master_problem_oa(
    10, x_factual, 0.5;
    sparsity_weight=0.1,
    epsilon=0.01
)
```
"""
function build_master_problem_oa(
    n_features::Int,
    x_factual::Vector{Float32},
    y_target::Float64;
    sparsity_weight::Float64=0.1,
    x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
    epsilon::Float64=0.01,
    immutable_indices::Vector{Int}=Int[]
)

    # Create JuMP model with Gurobi
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # Decision variables: counterfactual input
    @variable(model, x_bounds[1] <= x[i=1:n_features] <= x_bounds[2])

    # Distance decomposition variables (L1 norm = delta_pos + delta_neg)
    @variable(model, delta_pos[i=1:n_features] >= 0)
    @variable(model, delta_neg[i=1:n_features] >= 0)

    # Sparsity: binary indicators for feature changes
    @variable(model, changed[i=1:n_features], Bin)

    # Distance decomposition: x[i] - x_factual[i] = delta_pos[i] - delta_neg[i]
    @constraint(model, distance_decomp[i=1:n_features],
                x[i] - x_factual[i] == delta_pos[i] - delta_neg[i])

    # Big-M constraints for sparsity linking
    M = x_bounds[2] - x_bounds[1]  # Maximum possible change
    @constraint(model, sparsity_pos[i=1:n_features],
                delta_pos[i] <= M * changed[i])
    @constraint(model, sparsity_neg[i=1:n_features],
                delta_neg[i] <= M * changed[i])

    # Immutability constraints: fix certain features
    for i in immutable_indices
        fix(x[i], x_factual[i], force=true)
    end

    # Objective: minimize distance + sparsity
    # The NN constraint f(x) ≤ y_target + epsilon is enforced via OA cuts
    @objective(model, Min,
                sum(delta_pos[i] + delta_neg[i] for i in 1:n_features) +
                sparsity_weight * sum(changed[i] for i in 1:n_features))

    return (model, x, changed)
end


"""
    add_oa_cut!(
        master_model::Model,
        x_var::Vector{VariableRef},
        x_point::Vector{Float64},
        f_value::Float64,
        gradient::Vector{Float64};
        y_target::Float64=0.0,
        epsilon::Float64=0.01
    )

Add an outer approximation cut to the master problem for convex constraint f(x) ≤ y_target + ε.

# Mathematical Formulation

At point x_k, the first-order Taylor approximation of convex f is:
    f(x) ≥ f(x_k) + ∇f(x_k)·(x - x_k)

Rearranging: f(x_k) + ∇f(x_k)·(x - x_k) ≤ f(x)

For constraint f(x) ≤ y_target + ε, we add the outer approximation cut:
    f(x_k) + ∇f(x_k)·(x - x_k) ≤ y_target + ε

Equivalently: [f(x_k) - ∇f(x_k)·x_k] + ∇f(x_k)·x ≤ y_target + ε

Since f is convex, this linearization over-estimates f(x), creating a progressively
tighter outer approximation of the feasible set {x : f(x) ≤ y_target + ε}.

# Arguments
- `master_model::Model`: JuMP master problem model
- `x_var::Vector{VariableRef}`: Input decision variables
- `x_point::Vector{Float64}`: Point at which to linearize
- `f_value::Float64`: Neural network value f(x_point)
- `gradient::Vector{Float64}`: Gradient ∇f(x_point) 
- `y_target::Float64`: Target value
- `epsilon::Float64`: Tolerance

# Example
```julia
x_k = Float64.(value.(x_var))
f_k, grad_k = evaluate_ficnn_with_gradient(model, Float32.(x_k))
add_oa_cut!(master_model, x_var, x_k, f_k, grad_k;
            y_target=target, epsilon=eps)
```
"""
function add_oa_cut!(
    master_model::Model,
    x_var::Vector{VariableRef},
    x_point::Vector{Float64},
    f_value::Float64,
    gradient::Vector{Float64};
    y_target::Float64=0.0,
    epsilon::Float64=0.1
)
    n = length(x_var)

    # Validate inputs
    @assert length(gradient) == n "Gradient dimension mismatch"
    @assert !isnan(f_value) && !isinf(f_value) "Invalid f_value: $f_value"
    @assert !any(isnan.(gradient)) "Gradient contains NaN"
    @assert !any(isinf.(gradient)) "Gradient contains Inf"

    # Compute constant term: f(x_k) - ∇f(x_k)·x_k
    constant = f_value - dot(gradient, x_point)

    # Build linear expression: constant + sum(gradient[i] * x[i])
    linear_expr = @expression(master_model,
                              constant + sum(gradient[i] * x_var[i] for i in 1:n))

    cut_counter.count += 1
    cut_name = "oa_cut_$(cut_counter.count)"

    con = @constraint(master_model, linear_expr <= y_target + epsilon)
    set_name(con, cut_name)

    return cut_name
end


"""
    add_esh_cut!(
        master_model::Model,
        x_var::Vector{VariableRef},
        x_boundary::Vector{Float64},
        gradient::Vector{Float64}
    )

Add an Extended Supporting Hyperplane (ESH) cut to the master problem.

# Mathematical Formulation

ESH cuts are supporting hyperplanes that pass through a boundary point x'
where f(x') = y_target + ε. Unlike ECP cuts, ESH cuts do NOT include the
function value term, creating a pure supporting hyperplane.

From "The supporting hyperplane optimization toolkit for convex MINLP" (page 6, Eq. 3):
    ∇g_m(x')·(x - x') ≤ 0

For our convex constraint f(x) ≤ y_target + ε, the ESH cut is:
    ∇f(x')·(x - x') ≤ 0

Rearranging:
    ∇f(x')·x ≤ ∇f(x')·x'

Since f is convex and x' is on the boundary (f(x') = y_target + ε), this
hyperplane supports the feasible set {x : f(x) ≤ y_target + ε} at x'.

# Key Difference from ECP (add_oa_cut!)

**ECP cut (Outer Approximation)**:
    f(x_k) + ∇f(x_k)·(x - x_k) ≤ y_target + ε

Includes the function value f(x_k), used at infeasible points where
f(x_k) > y_target + ε.

**ESH cut (Supporting Hyperplane)**:
    ∇f(x')·(x - x') ≤ 0

Does NOT include function value, used at boundary points where
f(x') = y_target + ε. This is a tighter cut because it exactly
represents the tangent hyperplane at the boundary.

# Theoretical Property

For convex f and boundary point x' where f(x') = y_target + ε:
- The hyperplane ∇f(x')·(x - x') = 0 is tangent to the level set
- All feasible points satisfy ∇f(x')·(x - x') ≤ 0
- All infeasible points violate ∇f(x')·(x - x') > 0

This makes ESH cuts tighter than ECP cuts, potentially improving convergence.

# Arguments
- `master_model::Model`: JuMP master problem model
- `x_var::Vector{VariableRef}`: Input decision variables
- `x_boundary::Vector{Float64}`: Boundary point where f(x') ≈ y_target + ε
- `gradient::Vector{Float64}`: Gradient ∇f(x_boundary)

# Returns
- Constraint name of the added cut

# Example
```julia
# Find boundary point via bisection
x_boundary = find_boundary_point_bisection(
    x_interior, x_infeasible, icnn_model, y_target + epsilon
)

# Compute gradient at boundary
f_boundary, grad_boundary = evaluate_ficnn_with_gradient(icnn_model, x_boundary)

# Add ESH cut
add_esh_cut!(master_model, x_var, Float64.(x_boundary), grad_boundary)
```

# Notes
- Used exclusively in ESH strategy when boundary point is available
- Does not require y_target or epsilon (unlike add_oa_cut!)
- Generates tighter cuts than ECP for convex problems
- Requires finding boundary point first (via bisection or other method)
"""
function add_esh_cut!(
    master_model::Model,
    x_var::Vector{VariableRef},
    x_boundary::Vector{Float64},
    gradient::Vector{Float64}
)
    n = length(x_var)

    # Validate inputs
    @assert length(x_boundary) == n "Boundary point dimension mismatch"
    @assert length(gradient) == n "Gradient dimension mismatch"
    @assert !any(isnan.(x_boundary)) "Boundary point contains NaN"
    @assert !any(isinf.(x_boundary)) "Boundary point contains Inf"
    @assert !any(isnan.(gradient)) "Gradient contains NaN"
    @assert !any(isinf.(gradient)) "Gradient contains Inf"

    # ESH cut: ∇f(x')·(x - x') ≤ 0
    # Rearranged: ∇f(x')·x ≤ ∇f(x')·x'
    # This creates a supporting hyperplane at the boundary point
    rhs = dot(gradient, x_boundary)

    # Build linear expression: sum(gradient[i] * x[i])
    linear_expr = @expression(master_model,
                              sum(gradient[i] * x_var[i] for i in 1:n))

    cut_counter.count += 1
    cut_name = "esh_cut_$(cut_counter.count)"

    con = @constraint(master_model, linear_expr <= rhs)
    set_name(con, cut_name)

    return cut_name
end


"""
    add_initial_cut!(
        master_model::Model,
        x_var::Vector{VariableRef},
        icnn_model,
        x_factual::Vector{Float32}
    )

Add initial OA cut at the factual point to provide starting linearization.

This ensures the master problem has at least one cut and prevents unbounded
solutions in the first iteration.

# Arguments
- `master_model::Model`: JuMP master problem model
- `x_var::Vector{VariableRef}`: Input decision variables
- `icnn_model`: Trained FICNN model
- `x_factual::Vector{Float32}`: Factual input point

# Returns
- Constraint name of the added cut

# Example
```julia
model, x, changed = build_master_problem_oa(n, x_factual, y_target)
add_initial_cut!(model, x, icnn_model, x_factual)
```
"""
function add_initial_cut!(
    master_model::Model,
    x_var::Vector{VariableRef},
    icnn_model,
    x_factual::Vector{Float32};
    y_target::Float64=0.0,
    epsilon::Float64=0.01
)
    # Evaluate NN at factual point
    f_factual, grad_factual = evaluate_ficnn_with_gradient(icnn_model, x_factual)

    # Add OA cut at factual point
    cut_name = add_oa_cut!(
        master_model,
        x_var,
        Float64.(x_factual),
        f_factual,
        grad_factual;
        y_target=y_target,
        epsilon=epsilon
    )

    return cut_name
end


"""
    find_interior_point_oa(
        icnn_model,
        x_factual::Vector{Float32},
        y_target::Float64,
        epsilon::Float64,
        x_bounds::Tuple{Float64, Float64},
        immutable_indices::Vector{Int};
        max_iter::Int=20,
        tolerance::Float64=1e-4
    ) -> Union{Vector{Float32}, Nothing}

Find an interior point for ESH cuts by solving the Minimax formulation (Prob. MM) from the paper.

# Mathematical Background

The paper describes finding an interior point by solving a "Minimax" problem (Appendix A):
    max ν
    s.t. f(x) + ν ≤ y_target + ε
         x ∈ bounds
         
Rearranged: f(x) ≤ (y_target + ε) - ν

We seek the largest ν > 0 such that there exists an x satisfying the constraint.
A point x with ν > 0 is strictly feasible (interior to the constraint set).

# Algorithm

Uses a simple cutting-plane algorithm (Appendix A):
1. Build LP with variables x[1:n] and ν ≥ 0
2. Objective: maximize ν
3. Add bounds and immutability constraints
4. Iterate:
   a. Add cut: f(x_k) + ∇f(x_k)·(x - x_k) + ν ≤ y_target + ε
   b. Solve LP → get x_k, ν_k
   c. If ν_k > tolerance: SUCCESS, return x_k as interior point
   d. Else: evaluate f(x_k), ∇f(x_k) and add tighter cut
5. If max_iter reached without success: return nothing

# Arguments
- `icnn_model`: Trained FICNN model
- `x_factual::Vector{Float32}`: Factual input (used for first cut)
- `y_target::Float64`: Target output value
- `epsilon::Float64`: Tolerance for target matching
- `x_bounds::Tuple{Float64, Float64}`: Bounds for input variables
- `immutable_indices::Vector{Int}`: Indices that cannot be changed
- `max_iter::Int`: Maximum iterations for minimax OA loop (default: 20)
- `tolerance::Float64`: Minimum ν required for success (default: 1e-4)

# Returns
- `Vector{Float32}`: Interior point x_int where f(x_int) < y_target + ε
- `Nothing`: If no interior point found within max_iter

# Example
```julia
x_int = find_interior_point_oa(
    icnn_model, x_factual, y_target, epsilon, 
    (0.0, 1.0), Int[];
    max_iter=20, tolerance=1e-4
)

if x_int !== nothing
    println("Found interior point for ESH cuts")
end
```

# Notes
- Called before main OA loop when cut_strategy == :esh
- Enables ESH cuts from iteration 1 if successful
- Falls back to ECP if this returns nothing
- Typically converges in 5-15 iterations if interior point exists
"""
function find_interior_point_oa(
    icnn_model,
    x_start::Vector{Float32},  # Starting point for search (typically x_factual or first feasible solution)
    y_target::Float64,
    epsilon::Float64,
    x_bounds::Tuple{Float64, Float64},
    immutable_indices::Vector{Int};
    max_iter::Int=20,  # Reduced from 50 per paper's recommendation
    tolerance::Float64=1e-4,
    verbose::Bool=false
)
    n_features = length(x_start)

    if verbose
        println("\n" * "="^70)
        println("DEBUG: find_interior_point_oa() - Prob. MM Minimax Solver")
        println("="^70)
        println("Goal: Find x with f(x) - ν ≤ y_target + ε, where ν > $(tolerance)")
        println("      (Maximizing ν finds deepest interior point)")
        println("Target: y_target + ε = $(round(y_target + epsilon, digits=6))")
        println("Max iterations: $max_iter")
        println()
    end

    # Build simple LP for minimax problem
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # Variables: x[1:n] and nu >= 0
    @variable(model, x_bounds[1] <= x[i=1:n_features] <= x_bounds[2])
    @variable(model, nu >= 0)

    # Objective: maximize nu
    @objective(model, Max, nu)

    # Immutability constraints
    for i in immutable_indices
        fix(x[i], x_start[i], force=true)
    end

    # Initial bound to prevent unboundedness (without cuts, max nu would be ∞)
    # This bound will be tightened by OA cuts as they're added
    @constraint(model, nu <= y_target + epsilon)

    # Initialize x_k from x_start (provided by caller)
    # This could be x_factual (infeasible) or a feasible solution from ECP
    x_k = copy(x_start)

    if verbose
        f_init = Float64(icnn_model(reshape(x_k, 1, :))[1, 1])
        println("Initialized from x_start:")
        println("  f(x_start) = $(round(f_init, digits=4))")
        println("  Target = $(round(y_target + epsilon, digits=4))")
        violation = max(0.0, f_init - (y_target + epsilon))
        if violation > 0
            println("  Initial violation: $(round(violation, digits=4))")
        else
            println("  Already feasible (starting from ECP solution)")
        end
        println("  Initial bound: ν ≤ $(round(y_target + epsilon, digits=2))")
        println()
        println("Now starting cutting plane iterations...")
        println("Iter │    ν         │   f(x_k)    │ LP Status │ Verification")
        println("─────┼──────────────┼─────────────┼───────────┼──────────────")
    end

    # Minimax OA loop - NOW starting from feasible x₁
    failure_reason = "unknown"
    for iter in 1:max_iter
        # Evaluate NN at current point
        f_k, grad_k = evaluate_ficnn_with_gradient(icnn_model, x_k)

        # Skip cuts where gradient is near-zero (flat regions)
        # Such cuts don't provide useful bounds and can make LP unbounded
        grad_norm = norm(grad_k)
        if grad_norm < 1e-6
            if verbose && iter <= 3
                println("  Iter $iter: Skipping cut (gradient norm = $(grad_norm) ≈ 0)")
            end
            # Move to a random point within bounds to escape flat region
            x_k = Float32.(rand(n_features) .* (x_bounds[2] - x_bounds[1]) .+ x_bounds[1])
            for i in immutable_indices
                x_k[i] = x_start[i]
            end
            continue
        end

        # Correct Prob. MM formulation (Appendix A, Minimax Problem)
        # Goal: Find deepest interior point by maximizing ν such that:
        #       f(x) ≤ (y_target + ε) - ν  (equivalently: f(x) + ν ≤ y_target + ε)
        # If ν > 0, then f(x) < y_target + ε (strict interior)
        # OA linearization: f(x_k) + ∇f(x_k)·(x - x_k) + ν ≤ y_target + ε
        # As ν increases, constraint on f(x) TIGHTENS, pushing solution deeper into feasible region
        constant = f_k - dot(grad_k, Float64.(x_k))
        linear_expr = @expression(model,
                                  constant + sum(grad_k[i] * x[i] for i in 1:n_features) + nu)
        @constraint(model, linear_expr <= y_target + epsilon)
        
        # Solve LP
        optimize!(model)
        
        status = termination_status(model)
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            # No interior point exists
            failure_reason = "LP became infeasible (no interior point exists)"
            if verbose
                println(@sprintf("%4d │ %-12s │ %11.6f │ INFEAS    │ N/A", 
                                iter, "N/A", f_k))
                println("\n✗ FAILED: $failure_reason")
                println("="^70)
            end
            return nothing
        end
        
        if status ∉ [MOI.OPTIMAL, MOI.TIME_LIMIT]
            # Solve failed
            failure_reason = "LP solver failed with status: $status"
            if verbose
                println(@sprintf("%4d │ %-12s │ %11.6f │ %9s │ N/A", 
                                iter, "N/A", f_k, string(status)[1:min(9,end)]))
                println("\n✗ FAILED: $failure_reason")
                println("="^70)
            end
            return nothing
        end
        
        # Extract solution
        x_k = Float32.(value.(x))
        nu_k = value(nu)

        # Diagnostic output for first few iterations
        if verbose && iter <= 3
            x_batch = reshape(x_k, 1, :)
            f_k_actual = Float64(icnn_model(x_batch)[1, 1])
            target_val = y_target + epsilon
            is_interior = (nu_k > tolerance && f_k_actual <= target_val)

            println("  Prob. MM Iter $iter:")
            println("    ν = $(round(nu_k, digits=6))")
            println("    f(x) = $(round(f_k_actual, digits=4))")
            println("    Target = $(round(target_val, digits=4))")
            println("    Interior? $is_interior")

            # Show cut quality metrics
            x_k_float = Float64.(x_k)
            cut_value_at_solution = constant + dot(grad_k, x_k_float) + nu_k
            cut_rhs = y_target + epsilon
            cut_slack = cut_rhs - cut_value_at_solution
            effective_bound = cut_rhs - nu_k

            println("    Cut diagnostics:")
            println(@sprintf("      Cut LHS at x*:     %.2f", cut_value_at_solution))
            println(@sprintf("      Cut RHS:           %.2f", cut_rhs))
            println(@sprintf("      Cut slack:         %.2f", cut_slack))
            println(@sprintf("      Effective f bound: %.2f (should be < %.2f)",
                           effective_bound, y_target + epsilon))

            if cut_slack > 1000.0
                @warn "Cut has large slack ($cut_slack) - may indicate formulation issue!"
            end
        end

        # Check for success: nu > tolerance means we have strict feasibility
        if nu_k > tolerance
            # Verify that x_k is actually feasible
            x_batch = reshape(x_k, 1, :)
            f_verify = Float64(icnn_model(x_batch)[1, 1])

            # Check: f(x*) should be strictly less than target + epsilon
            interior_slack = (y_target + epsilon) - f_verify

            if verbose
                verify_status = interior_slack > tolerance ? "✓ SUCCESS" : @sprintf("✗ Slack=%.6f", interior_slack)
                println(@sprintf("%4d │ %12.8f │ %11.6f │ OPTIMAL   │ %s",
                                iter, nu_k, f_k, verify_status))
            end

            if interior_slack > tolerance && nu_k > tolerance
                # SUCCESS: strictly interior point
                if verbose
                    println("\n✓ SUCCESS: Found interior point!")
                    println("  ν = $(round(nu_k, digits=8)) > tolerance")
                    println("  f(x) = $(round(f_verify, digits=6)) < $(round(y_target + epsilon, digits=6))")
                    println("  Interior slack: $(round(interior_slack, digits=6)) > tolerance")
                    println("="^70)
                end
                return x_k
            else
                # ν > tolerance but verification failed (nonconvexity issue?)
                # Continue searching
                failure_reason = "Verification failed: interior_slack = $(round(interior_slack, digits=8)) ≤ tolerance"
            end
        else
            if verbose
                println(@sprintf("%4d │ %12.8f │ %11.6f │ OPTIMAL   │ ν ≤ tol", 
                                iter, nu_k, f_k))
            end
            failure_reason = "Max iterations reached with ν ≤ tolerance"
        end
    end
    
    # Max iterations reached without finding interior point
    if verbose
        println("\n✗ FAILED: $failure_reason")
        println("  Final ν = $(round(value(nu), digits=8)) ≤ $(tolerance)")
        println("  Need ν > $(tolerance) for strict interior point")
        println("="^70)
    end
    return nothing
end


"""
    find_boundary_point_bisection(
        x_feasible::Vector{Float32},
        x_infeasible::Vector{Float32},
        icnn_model,
        target_value::Float64;
        max_iterations::Int=30,
        tolerance::Float64=1e-5
    ) -> Vector{Float32}

Find boundary point on line segment between feasible and infeasible points using bisection.

This function implements the root-finding step required for Extended Supporting Hyperplane (ESH)
cuts. Given a feasible point (interior to the constraint set) and an infeasible point (exterior),
it finds the boundary point x' where f(x') ≈ target_value.

# Mathematical Background

For ESH cuts, we need to find x' on the line segment:
    x(λ) = λ * x_feasible + (1 - λ) * x_infeasible,  λ ∈ [0, 1]

such that f(x') = target_value (typically y_target + ε).

Since f is continuous and convex:
- f(x_feasible) ≤ target_value (interior point)
- f(x_infeasible) > target_value (exterior point)

By intermediate value theorem, there exists λ* where f(x(λ*)) = target_value.

# Algorithm

Bisection search on λ ∈ [0, 1]:
1. Start with λ_low = 0 (feasible), λ_high = 1 (infeasible)
2. Test midpoint λ_mid = (λ_low + λ_high) / 2
3. Evaluate f(x(λ_mid))
4. If f(x(λ_mid)) < target_value: λ_low = λ_mid (move toward infeasible)
5. If f(x(λ_mid)) > target_value: λ_high = λ_mid (move toward feasible)
6. Repeat until |f(x(λ_mid)) - target_value| < tolerance

# Arguments
- `x_feasible::Vector{Float32}`: Interior point (f(x) ≤ target_value)
- `x_infeasible::Vector{Float32}`: Exterior point (f(x) > target_value)
- `icnn_model`: Trained FICNN model
- `target_value::Float64`: Target boundary value (y_target + ε)
- `max_iterations::Int`: Maximum bisection iterations (default: 30)
- `tolerance::Float64`: Convergence tolerance for |f(x') - target_value| (default: 1e-5)

# Returns
- `Vector{Float32}`: Boundary point x' where f(x') ≈ target_value

# Example
```julia
# Find boundary point for ESH cut
x_boundary = find_boundary_point_bisection(
    x_interior,
    x_k,
    icnn_model,
    y_target + epsilon;
    max_iterations=30,
    tolerance=1e-5
)
f_boundary, grad_boundary = evaluate_ficnn_with_gradient(icnn_model, x_boundary)
add_oa_cut!(master_model, x_var, x_boundary, f_boundary, grad_boundary, ...)
```

# Notes
- Requires exactly one feasible and one infeasible point
- Bisection is robust and guaranteed to converge for convex f
- Typically converges in 15-25 iterations with tolerance 1e-5
- More sophisticated methods (secant, regula falsi) possible but bisection is simple and reliable
"""
function find_boundary_point_bisection(
    x_feasible::Vector{Float32},
    x_infeasible::Vector{Float32},
    icnn_model,
    target_value::Float64;
    max_iterations::Int=30,
    tolerance::Float64=1e-5
)
    λ_low = 0.0   # Corresponds to x_feasible (interior)
    λ_high = 1.0  # Corresponds to x_infeasible (exterior)
    
    # Verify initial conditions
    x_batch_feasible = reshape(x_feasible, 1, :)
    f_feasible = Float64(icnn_model(x_batch_feasible)[1, 1])
    
    x_batch_infeasible = reshape(x_infeasible, 1, :)
    f_infeasible = Float64(icnn_model(x_batch_infeasible)[1, 1])
    
    # Sanity check: ensure feasible point is actually feasible
    if f_feasible > target_value
        @warn "find_boundary_point_bisection: x_feasible has f(x) > target_value. Using x_feasible as-is."
        return x_feasible
    end
    
    # Sanity check: ensure infeasible point is actually infeasible
    if f_infeasible <= target_value
        @warn "find_boundary_point_bisection: x_infeasible has f(x) ≤ target_value. Using x_infeasible as-is."
        return x_infeasible
    end
    
    # Bisection loop
    λ_mid = 0.5
    x_mid = similar(x_feasible)
    f_mid = 0.0
    
    for iter in 1:max_iterations
        # Compute midpoint on line segment
        λ_mid = (λ_low + λ_high) / 2.0
        x_mid = Float32.((1.0 - λ_mid) .* x_feasible .+ λ_mid .* x_infeasible)
        
        # Evaluate neural network at midpoint
        x_batch_mid = reshape(x_mid, 1, :)
        f_mid = Float64(icnn_model(x_batch_mid)[1, 1])
        
        # Check convergence
        error = abs(f_mid - target_value)
        if error < tolerance
            break
        end
        
        # Update bisection interval
        if f_mid < target_value
            # x_mid is still interior (feasible), move toward infeasible
            λ_low = λ_mid
        else
            # x_mid is exterior (infeasible), move toward feasible
            λ_high = λ_mid
        end
    end
    
    return x_mid
end


"""
    generate_counterfactual_oa(
        icnn_model,
        x_factual::Vector{Float32},
        y_target::Float64;
        epsilon::Float64=0.01,
        sparsity_weight::Float64=0.1,
        x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
        max_iterations::Int=50,
        time_limit_per_iter::Float64=60.0,
        tolerance::Float64=1e-6,
        immutable_indices::Vector{Int}=Int[],
        cut_strategy::Symbol=:ecp,
        verbose::Bool=true
    ) -> Dict

Generate counterfactual using Outer Approximation algorithm for convex constraint f(x) ≤ y_target + ε.

# Problem Formulation

Minimize: ||x - x_factual||_1 + sparsity_weight * num_changed
Subject to: f(x) ≤ y_target + ε

where f is a convex FICNN model.

# Algorithm
1. Build master MILP 
2. Add initial OA cut at x_factual
3. Main loop (until convergence or max_iterations):
   a. Solve master MILP → candidate solution x_k
   b. Evaluate NN(x_k) → get f_k and ∇f_k
   c. Check feasibility: f(x_k) ≤ y_target + ε
   d. Update best solution if feasible
   e. Add OA cut to tighten approximation (ECP or ESH)
   f. Check convergence (solution stabilization)
4. Return best feasible solution or infeasible status

# Cut Strategies

## Extended Cutting Plane (ECP) - Default
- Adds cut at infeasible point x_k directly
- Cut: f(x_k) + ∇f(x_k)·(x - x_k) ≤ y_target + ε
- Simple and efficient for most problems

## Extended Supporting Hyperplane (ESH) - Advanced
- Requires an interior point (feasible solution)
- Finds boundary point x' on line between interior and infeasible points
- Cut: f(x') + ∇f(x')·(x - x') ≤ y_target + ε
- Generates tighter cuts, may improve convergence for difficult problems
- Automatically falls back to ECP until first feasible solution found

# Arguments
- `icnn_model`: Trained FICNN model
- `x_factual::Vector{Float32}`: Factual input point
- `y_target::Float64`: Target output value
- `epsilon::Float64`: Tolerance for target matching (default: 0.01)
- `sparsity_weight::Float64`: Weight for sparsity penalty (default: 0.1)
- `x_bounds::Tuple{Float64, Float64}`: Bounds for input features (default: (0.0, 1.0))
- `max_iterations::Int`: Maximum OA iterations (default: 50)
- `time_limit_per_iter::Float64`: Time limit per MILP solve (default: 60s)
- `tolerance::Float64`: Convergence tolerance for solution change (default: 1e-6)
- `immutable_indices::Vector{Int}`: Indices that cannot be changed (default: [])
- `cut_strategy::Symbol`: Cut generation strategy - :ecp or :esh (default: :ecp)
- `verbose::Bool`: Print iteration progress (default: true)

# Returns
- `Dict` with keys:
  - `:status`: Optimization status (:optimal, :infeasible, :time_limit, etc.)
  - `:counterfactual`: Best counterfactual solution (or nothing if infeasible)
  - `:prediction`: Neural network prediction at counterfactual
  - `:distance`: L1 distance from factual
  - `:num_changed`: Number of changed features
  - `:changed_indices`: Indices of changed features
  - `:iterations`: Number of OA iterations performed
  - `:solve_time`: Total solution time
  - `:upper_bound`: Best objective value found (for feasible solutions)
  - `:iteration_history`: Array of iteration logs

# Example
```julia
# Using default ECP strategy
result = generate_counterfactual_oa(
    model, x_factual, 0.5;
    epsilon=0.01,
    sparsity_weight=0.1,
    max_iterations=30,
    verbose=true
)

# Using ESH strategy for tighter cuts
result = generate_counterfactual_oa(
    model, x_factual, 0.5;
    epsilon=0.01,
    cut_strategy=:esh,
    verbose=true
)

if result[:status] == :optimal
    println("Found counterfactual: ", result[:counterfactual])
    println("Distance: ", result[:distance])
    println("Changed features: ", result[:num_changed])
end
```

# Convergence
For convex ICNN models with good initialization:
- First OA cut from x_factual often provides strong approximation
- Typically converges in 1-5 iterations when feasible solution exists
- Convergence criteria: solution stabilizes (no change between iterations)
- Fast convergence is EXPECTED for well-posed problems
- ESH may converge faster for problems with tight constraints

# Notes
- Much faster than full MIP for large networks (no hidden layer variables)
- Exploits convexity for provably correct solutions
- Number of OA cuts grows linearly with iterations
- Master problem size: O(n_features) variables, O(iterations) constraints
- ESH requires bisection search (adds ~30 NN evaluations per cut)
"""
function generate_counterfactual_oa(
    icnn_model,
    x_factual::Vector{Float32},
    y_target::Float64;
    epsilon::Float64=0.01,
    sparsity_weight::Float64=0.1,
    x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
    max_iterations::Int=50,
    time_limit_per_iter::Float64=60.0,
    tolerance::Float64=1e-6,
    immutable_indices::Vector{Int}=Int[],
    cut_strategy::Symbol=:ecp,
    verbose::Bool=true
)
    n_features = length(x_factual)

    # Validate cut strategy
    @assert cut_strategy in [:ecp, :esh] "cut_strategy must be :ecp or :esh, got: $cut_strategy"

    # Reset cut counter
    reset_cut_counter!()

    if verbose
        println("=" ^ 70)
        println("Outer Approximation Counterfactual Generation")
        println("=" ^ 70)
    end

    # Evaluate current prediction
    x_batch = reshape(x_factual, 1, :)
    y_current = icnn_model(x_batch)[1, 1]

    if verbose
        println("Factual prediction: y = $(round(y_current, digits=4))")

        println("Target constraint: f(x) ≤ $(round(y_target + epsilon, digits=4))")
        println("Parameters:")
        println("  - Features: $n_features")
        println("  - Sparsity weight: $sparsity_weight")
        println("  - Cut strategy: $cut_strategy")
        println("  - Max iterations: $max_iterations")
        println("  - Time limit per iter: $(time_limit_per_iter)s")
        println("  - Convergence tolerance: $tolerance")
        println("  - Immutable features: $(length(immutable_indices))")
        println()
    end

    # FIX #1: Start timing BEFORE any preprocessing
    # This ensures solve_time includes all algorithm overhead
    start_time = time()

    # Check if already at target (depends on constraint type)
    already_satisfied = y_current <= y_target + epsilon

    if already_satisfied
        if verbose
            println("✓ Already satisfies target constraint!")
        end
        early_exit_time = time() - start_time
        return Dict(
            :status => :already_at_target,
            :counterfactual => x_factual,
            :prediction => Float64(y_current),
            :distance => 0.0,
            :num_changed => 0,
            :changed_indices => Int[],
            :iterations => 0,
            :solve_time => early_exit_time,
            :upper_bound => 0.0,
            :iteration_history => [],
            :timing_breakdown => Dict(
                :total => early_exit_time,
                :model_build => 0.0,
                :interior_point_search => 0.0,
                :total_milp_solve => 0.0,
                :total_nn_eval => 0.0,
                :total_cut_generation => 0.0,
                :total_bisection => 0.0
            )
        )
    end

    # Initialize timing breakdown FIRST
    timing_breakdown = Dict(
        :total => 0.0,
        :model_build => 0.0,
        :interior_point_search => 0.0,
        :total_milp_solve => 0.0,
        :total_nn_eval => 0.0,
        :total_cut_generation => 0.0,
        :total_bisection => 0.0
    )
    
    # Build master problem
    model_build_start = time()
    if verbose
        println("Building master MILP...")
    end
    master_model, x_var, changed_var = build_master_problem_oa(
        n_features,
        x_factual,
        y_target;
        sparsity_weight=sparsity_weight,
        x_bounds=x_bounds,
        epsilon=epsilon,
        immutable_indices=immutable_indices
    )

    # Add initial cut at factual point
    add_initial_cut!(master_model, x_var, icnn_model, x_factual;
                     y_target=y_target, epsilon=epsilon)

    if verbose
        println("✓ Master problem built with initial cut")
    end
    
    timing_breakdown[:model_build] = time() - model_build_start

    # FIX: For ESH, find interior point BEFORE main loop
    # This enables ESH cuts from iteration 1 instead of waiting until first feasible
    x_interior_point = nothing
    interior_search_start = time()
    
    if cut_strategy == :esh
        if verbose
            println()
            println("ESH strategy selected - searching for interior point...")
        end
        
        # CRITICAL FIX: For ESH to be effective, interior point must be CLOSE to feasible boundary
        # Using x=0 (lower bounds) gives f=0, which is too deep - bisection finds boundaries near infeasible points
        # Solution: Search for interior point using Prob. MM from a reasonable starting point
        
        target_value = y_target + epsilon
        
        # First check if lower bounds work AND are close enough to target
        x_candidate = fill(Float32(x_bounds[1]), n_features)
        for i in immutable_indices
            x_candidate[i] = x_factual[i]
        end
        
        x_batch = reshape(x_candidate, 1, :)
        f_candidate = Float64(icnn_model(x_batch)[1, 1])
        
        # Only use lower bounds if they're reasonably close to target (not too deep)
        # If f(lower) < 0.5 * target, it's too deep to be useful for ESH
        if f_candidate < target_value - 1e-6 && f_candidate > 0.5 * target_value
            x_interior_point = x_candidate
            if verbose
                println("✓ Lower bounds are good interior point!")
                println("  f(lower_bounds) = $(round(f_candidate, digits=2))")
                println("  Target = $(round(target_value, digits=2))")
            end
        else
            # Lower bounds too deep or infeasible, use Prob. MM to find better interior
            if verbose
                if f_candidate >= target_value
                    println("  Lower bounds infeasible (f = $(round(f_candidate, digits=2)) >= $(round(target_value, digits=2)))")
                else
                    println("  Lower bounds too deep (f = $(round(f_candidate, digits=2)) << $(round(target_value, digits=2)))")
                end
                println("  Searching for better interior point via Prob. MM...")
            end
            
            # Start Prob. MM from midpoint between lower bounds and factual
            # This gives a starting point closer to the feasible region
            x_start = Float32.(0.5 * (x_candidate + x_factual))
            
            x_interior_point = find_interior_point_oa(
                icnn_model, x_start, y_target, epsilon, x_bounds, immutable_indices;
                max_iter=20, tolerance=1e-4, verbose=verbose
            )
        end
        
        if x_interior_point !== nothing
            if verbose
                println("✓ Interior point found! ESH cuts will be used for infeasible iterations.")
            end
        else
            if verbose
                println("⚠ No interior point found. Falling back to ECP strategy.")
            end
        end
    end
    
    timing_breakdown[:interior_point_search] = time() - interior_search_start

    if verbose
        println()
        println("Starting OA iterations...")
        println("-" ^ 70)
    end

    # Initialize tracking
    # Note: For constraint satisfaction problems with OA, we only track upper bound
    # (objective value of feasible solutions). There is no true lower bound.
    UB = Inf
    best_x = nothing
    best_f = nothing
    best_obj = Inf
    iteration_history = []
    prev_x = nothing  # Track previous solution for convergence
    
    # Track cut types for diagnostics
    ecp_cut_count = 0
    esh_cut_count = 0

    # Main OA loop
    for iter in 1:max_iterations
        # Step 1: Solve master MILP
        iter_milp_start = time()
        set_time_limit_sec(master_model, time_limit_per_iter)
        set_silent(master_model)  # Suppress Gurobi output unless explicitly enabled
        optimize!(master_model)

        status = termination_status(master_model)
        master_solve_time = JuMP.solve_time(master_model)
        timing_breakdown[:total_milp_solve] += time() - iter_milp_start

        # Check termination status
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            if verbose
                println("\n✗ Master problem is infeasible - no counterfactual exists")
            end

            solve_time = time() - start_time
            timing_breakdown[:total] = solve_time
            return Dict(
                :status => :infeasible,
                :counterfactual => nothing,
                :prediction => nothing,
                :distance => Inf,
                :num_changed => nothing,
                :changed_indices => nothing,
                :iterations => iter,
                :solve_time => solve_time,
                :upper_bound => UB,
                :iteration_history => iteration_history,
                :ecp_cuts => ecp_cut_count,
                :esh_cuts => esh_cut_count,
                :timing_breakdown => timing_breakdown
            )
        end

        if status ∉ [MOI.OPTIMAL, MOI.TIME_LIMIT, MOI.INTERRUPTED]
            if verbose
                println("\n⚠ Master problem solve failed with status: $status")
            end
            break
        end

        # Step 2: Extract solution
        x_k = Float32.(value.(x_var))
        obj_k = objective_value(master_model)

        # Step 3: Evaluate neural network at x_k
        iter_nn_start = time()
        f_k, grad_k = evaluate_ficnn_with_gradient(icnn_model, x_k)
        timing_breakdown[:total_nn_eval] += time() - iter_nn_start

        # Step 4: Check feasibility w.r.t. target
        # Add small numerical tolerance to avoid floating-point issues
        numerical_tol = 1e-6
        target_error = max(0.0, f_k - (y_target + epsilon))  # Violation amount
        feasible = f_k <= y_target + epsilon + numerical_tol

        # Step 5: Update best solution
        # For constraint satisfaction problems, only feasible solutions provide bounds.
        # Feasible solution → objective value is an upper bound on optimal objective.
        if feasible && obj_k < UB
            UB = obj_k
            best_x = copy(x_k)
            best_f = f_k
            best_obj = obj_k
            
            # CRITICAL FIX for ESH: Replace deep interior point (x=0) with first feasible
            # This gives ESH cuts that are much tighter because interior is near optimal region
            if cut_strategy == :esh && x_interior_point !== nothing
                x_int_batch = reshape(x_interior_point, 1, :)
                f_int = Float64(icnn_model(x_int_batch)[1, 1])
                
                # If interior is at lower bounds (f ≈ 0), replace with feasible solution
                if f_int < 1.0
                    x_interior_point = copy(x_k)
                    if verbose
                        println("  → Switched interior point: lower bounds (f=0) → feasible (f=$(round(f_k, digits=2)))")
                    end
                end
            end
        end

        # Step 6: Add OA cut (ECP or ESH strategy)
        # This cut will tighten the approximation in the next iteration
        # Even if the current solution is feasible, the cut may reveal better solutions
        
        # Choose cut strategy:
        # - Use ECP if: (a) cut_strategy == :ecp, OR (b) no interior point available yet
        # - Use ESH if: (a) cut_strategy == :esh, AND (b) interior point available
        use_esh = (cut_strategy == :esh) && (x_interior_point !== nothing)
        
        # Track cut diagnostics
        cut_point = x_k
        cut_f = f_k
        cut_type = "ECP"
        bisection_info = nothing
        

        iter_cut_start = time()
        if use_esh && !feasible
            # ESH Strategy: Find boundary point between interior and infeasible points
            # This generates tighter supporting hyperplane cuts
            target_boundary = y_target + epsilon

            bisection_start = time()
            x_boundary = find_boundary_point_bisection(
                x_interior_point,
                x_k,
                icnn_model,
                target_boundary;
                max_iterations=30,
                tolerance=1e-5
            )
            bisection_time = time() - bisection_start
            timing_breakdown[:total_bisection] += bisection_time

            # Evaluate at boundary point
            f_boundary, grad_boundary = evaluate_ficnn_with_gradient(icnn_model, x_boundary)

            # Store diagnostics
            cut_type = "ESH"
            cut_point = x_boundary
            cut_f = f_boundary
            bisection_info = (
                boundary_f=f_boundary,
                infeasible_f=f_k,
                dist_to_interior=norm(x_boundary - x_interior_point),
                dist_to_infeasible=norm(x_boundary - x_k),
                bisection_time=bisection_time
            )

            # Add ESH cut at boundary point (supporting hyperplane)
            # ESH cut: ∇f(x')·(x - x') ≤ 0
            # This is the correct formulation per the paper (Section 2.2, Equation 3)
            # Unlike ECP, ESH cuts do NOT include the function value term
            add_esh_cut!(
                master_model,
                x_var,
                Float64.(x_boundary),
                grad_boundary
            )

            # Increment ESH cut counter
            esh_cut_count += 1
        else
            # ECP Strategy (Default): Add cut at infeasible point directly
            # This is the original/standard outer approximation approach
            # ECP cut: f(x_k) + ∇f(x_k)·(x - x_k) ≤ y_target + ε
            add_oa_cut!(
                master_model,
                x_var,
                Float64.(x_k),
                f_k,
                grad_k;
                y_target=y_target,
                epsilon=epsilon
            )

            # Increment ECP cut counter
            ecp_cut_count += 1
        end
        timing_breakdown[:total_cut_generation] += time() - iter_cut_start
        
        # Calculate cut quality metrics
        num_cons = JuMP.num_constraints(master_model; count_variable_in_set_constraints=false)
        gap = f_k - (y_target + epsilon)  # How far from feasibility
        step_change = iter > 1 ? obj_k - iteration_history[end].obj_k : NaN

        # Step 7: Log iteration BEFORE convergence check
        iter_log = (
            iteration=iter,
            obj_k=obj_k,
            UB=UB,
            f_k=f_k,
            target_error=target_error,
            feasible=feasible,
            x_k=copy(x_k),  # Store point for convergence analysis
            cut_type=cut_type,
            num_constraints=num_cons,
            gap=gap,
            step_change=step_change,
            master_solve_time=master_solve_time,
            bisection_info=bisection_info
        )
        push!(iteration_history, iter_log)

        if verbose
            feasible_str = feasible ? "✓" : "✗"
            ub_str = UB < Inf ? @sprintf("UB=%8.4f", UB) : "UB=     Inf"
            @printf("Iter %2d: %s  obj=%8.4f  f(x)=%7.4f  err=%6.4f  %s  [%s] cuts=%d milp=%.3fs",
                    iter, ub_str, obj_k, f_k, target_error, feasible_str, cut_type, num_cons, master_solve_time)
            
            # Show step improvement
            if !isnan(step_change)
                @printf("  Δobj=%+.4f", step_change)
            end
            
            # Show ESH-specific diagnostics
            if bisection_info !== nothing
                @printf("  [bisect=%.3fs d_int=%.2f d_inf=%.2f]",
                    bisection_info.bisection_time,
                    bisection_info.dist_to_interior,
                    bisection_info.dist_to_infeasible)
            end
            
            println()
        end

        # Step 8: Check convergence
        # OA convergence for constraint satisfaction problems:
        #
        # Key challenge: The master objective (distance + sparsity) doesn't include
        # the NN output directly. The NN constraint f(x) ≤ target is approximated
        # by OA cuts. This means:
        # - If solution is feasible → it's a valid upper bound
        # - But we need multiple cuts to ensure we've found the BEST feasible solution
        #
        # Convergence criteria (ALL must be satisfied):
        # 1. Found at least one feasible solution (UB < Inf)
        # 2. Solution hasn't changed in last 2 iterations (stable)
        # 3. Completed at least 3 iterations (ensures cuts have tightened approximation)

        solution_changed = true
        if prev_x !== nothing
            max_change = maximum(abs.(x_k .- prev_x))
            if max_change < tolerance
                solution_changed = false
            end
        end

        prev_x = copy(x_k)

        # Check convergence
        # Require: (1) feasible, (2) stable solution, (3) min 3 iterations
        if iter >= 3 && feasible && !solution_changed
            if verbose
                println("-" ^ 70)
                println("✓ Converged! Solution stabilized after $(iter) iterations")
            end
            break
        end

        # Alternative early stop: excellent solution that's been stable
        if iter >= 3 && feasible && target_error <= epsilon / 10 && !solution_changed
            if verbose
                println("-" ^ 70)
                println("✓ Found excellent solution with error = $(round(target_error, digits=6))")
            end
            break
        end
    end

    solve_time = time() - start_time
    timing_breakdown[:total] = solve_time

    if verbose
        println("-" ^ 70)
        println("\nTiming Breakdown:")
        println("  Model build:     $(round(timing_breakdown[:model_build], digits=3))s")
        println("  Interior search: $(round(timing_breakdown[:interior_point_search], digits=3))s")
        println("  MILP solving:    $(round(timing_breakdown[:total_milp_solve], digits=3))s")
        println("  NN evaluations:  $(round(timing_breakdown[:total_nn_eval], digits=3))s")
        println("  Cut generation:  $(round(timing_breakdown[:total_cut_generation], digits=3))s")
        if timing_breakdown[:total_bisection] > 0
            println("    - Bisection:   $(round(timing_breakdown[:total_bisection], digits=3))s")
        end
        println("  Total:           $(round(timing_breakdown[:total], digits=3))s")
        println()
    end

    # Return results
    if best_x !== nothing
        # Compute final statistics
        distance = sum(abs.(best_x .- x_factual))
        changed_indices = findall(abs.(best_x .- x_factual) .> 1e-5)
        num_changed = length(changed_indices)

        if verbose
            println("✓ Counterfactual found!")
            println("  Status: optimal")
            println("  Iterations: $(length(iteration_history))")
            println("  Solve time: $(round(solve_time, digits=2))s")
            println("  Best objective: $(round(UB, digits=6))")
            println()
            println("  Distance (L1): $(round(distance, digits=4))")
            println("  Features changed: $num_changed / $n_features")
            println("  Prediction: $(round(best_f, digits=4))")
            println("  Target: $(round(y_target, digits=4))")
            println("  Error: $(round(abs(best_f - y_target), digits=6))")

            if num_changed > 0 && num_changed <= 20
                println()
                println("  Changed features (showing up to 20):")
                for idx in changed_indices[1:min(20, num_changed)]
                    delta = best_x[idx] - x_factual[idx]
                    @printf("    Feature %3d: %.4f → %.4f (Δ = %+.4f)\n",
                           idx, x_factual[idx], best_x[idx], delta)
                end
            end
            println("=" ^ 70)
        end

        return Dict(
            :status => :optimal,
            :counterfactual => best_x,
            :prediction => best_f,
            :distance => distance,
            :num_changed => num_changed,
            :changed_indices => changed_indices,
            :iterations => length(iteration_history),
            :solve_time => solve_time,
            :upper_bound => UB,
            :iteration_history => iteration_history,
            :ecp_cuts => ecp_cut_count,
            :esh_cuts => esh_cut_count,
            :timing_breakdown => timing_breakdown
        )
    else
        if verbose
            println("✗ No feasible counterfactual found")
            println("  Iterations: $(length(iteration_history))")
            println("  Solve time: $(round(solve_time, digits=2))s")
            println("=" ^ 70)
        end

        return Dict(
            :status => :no_solution,
            :counterfactual => nothing,
            :prediction => nothing,
            :distance => Inf,
            :num_changed => nothing,
            :changed_indices => nothing,
            :iterations => length(iteration_history),
            :solve_time => solve_time,
            :upper_bound => UB,
            :iteration_history => iteration_history,
            :ecp_cuts => ecp_cut_count,
            :esh_cuts => esh_cut_count,
            :timing_breakdown => timing_breakdown
        )
    end
end


"""
    add_integer_cut!(
        master_model::Model,
        changed_var::Vector{VariableRef},
        changed_pattern::Vector{Float64}
    )

Add an integer cut to prevent revisiting the same sparsity pattern.

This is an optional backup mechanism to prevent cycling. If the algorithm visits
the same sparsity pattern multiple times, this cut excludes that exact pattern
from future iterations.

# Arguments
- `master_model::Model`: JuMP master problem model
- `changed_var::Vector{VariableRef}`: Binary sparsity indicator variables
- `changed_pattern::Vector{Float64}`: Current sparsity pattern (solution values)

# Mechanism
For a sparsity pattern where changed[i] = 1 for indices in set S, add constraint:
    sum(changed[i] for i in S) <= |S| - 1

This ensures at least one feature in S must be un-changed in future solutions,
preventing the exact same pattern from recurring.

# Example
```julia
# After solving master problem
changed_sol = value.(changed_var)
add_integer_cut!(master_model, changed_var, changed_sol)
```

# Notes
- Only useful when cycling is observed (same sparsity patterns repeated)
- Not needed with proper penalty formulation in most cases
- Can make master problem harder to solve if many cuts are added
"""
function add_integer_cut!(
    master_model::Model,
    changed_var::Vector{VariableRef},
    changed_pattern::Vector{Float64}
)
    # Find indices where feature was changed (changed[i] ≈ 1)
    S = findall(x -> x > 0.5, changed_pattern)

    if !isempty(S)
        # Add constraint: sum(changed[i] for i in S) <= |S| - 1
        # This prevents exact repetition of this sparsity pattern
        cut_counter.count += 1
        cut_name = "integer_cut_$(cut_counter.count)"
        con = @constraint(master_model, sum(changed_var[i] for i in S) <= length(S) - 1)
        set_name(con, cut_name)

        return cut_name
    end

    return nothing
end


# Export main functions
export generate_counterfactual_oa,
       build_master_problem_oa,
       add_oa_cut!,
       add_esh_cut!,
       add_initial_cut!,
       add_integer_cut!,
       find_boundary_point_bisection,
       find_interior_point_oa
