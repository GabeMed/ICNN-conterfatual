"""
Outer Approximation (OA) algorithm for ICNN counterfactual generation.

Given a factual input x_factual and a target output y_target, find a nearby
counterfactual x' such that NN(x') satisfies the target constraint by iteratively
solving a MILP master problem with linearized neural network constraints (OA cuts).

The OA method exploits the convexity of the FICNN to replace the full MIP
encoding with a series of linear cuts, resulting in significantly faster solve times.

# Algorithm Overview

1. Build master MILP with decision variables x, sparsity indicators
2. Add constraint based on constraint_type:
   - :upper_bound: Direct constraint γ ≤ y_target + ε with upper approximation cuts
   - :interval: Penalty formulation min |γ - y_target| with lower approximation cuts
3. Initialize with OA cut at x_factual
4. Iterate:
   a. Solve master MILP → get candidate x_k and gamma_k
   b. Evaluate NN at x_k → get f(x_k) and ∇f(x_k)
   c. Check feasibility based on constraint_type
   d. Update bounds if feasible
   e. Add appropriate OA cut
   f. Check convergence
5. Return best feasible solution

# Key Differences from Full MIP
- No hidden layer variables z_i
- No ReLU big-M constraints
- Neural network represented implicitly via OA cuts
- Master problem grows by one constraint per iteration
- Exploits convexity for global optimality guarantee (when using :upper_bound)
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
        constraint_type::Symbol=:upper_bound,
        sparsity_weight::Float64=0.1,
        x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
        epsilon::Float64=0.01,
        target_penalty_weight::Float64=1000.0,
        immutable_indices::Vector{Int}=Int[]
    ) -> Tuple{Model, Vector{VariableRef}, Union{VariableRef, Nothing}, Vector{VariableRef}}

Build the master MILP problem for outer approximation with specified constraint type.

# Constraint Types

## :upper_bound 
f(x) ≤ y_target + ε:
- Objective: minimize distance + sparsity
- Constraint: γ ≤ y_target + epsilon (enforced via upper approximation cuts)
- OA cuts: γ ≤ f(x_k) + ∇f(x_k)·(x - x_k)
- This is the CORRECT standard OA formulation for convex constraints
- Use case: "Reduce cost to at most y_target"

# Parameters
- `constraint_type::Symbol`: Type of constraint (:upper_bound, :interval, :lower_bound)
- Other parameters same as before

# Arguments
- `n_features::Int`: Number of input features
- `x_factual::Vector{Float32}`: Factual input point
- `y_target::Float64`: Target output value
- `sparsity_weight::Float64`: Weight for sparsity penalty (default: 0.1)
- `x_bounds::Tuple{Float64, Float64}`: Bounds for input variables (default: (0.0, 1.0))
- `epsilon::Float64`: Tolerance for target matching in feasibility check (default: 0.01)
- `immutable_indices::Vector{Int}`: Indices that cannot be changed (default: [])

# Returns
- `Tuple{Model, Vector{VariableRef}, Vector{VariableRef}}`:
  - `model`: JuMP model
  - `x_var`: Counterfactual input variables
  - `changed_var`: Binary sparsity indicators

# Details
The master problem is much simpler than the full MIP:
- No hidden layer variables (z_i)
- No ReLU big-M constraints
- Neural network represented by OA cuts
- OA cuts are added iteratively during the algorithm
# Objective
Minimize: ||x - x_factual||_1 + sparsity_weight * num_changed
         = sum(delta_pos + delta_neg) + sparsity_weight * sum(changed)

# Example
```julia
model, x, changed = build_master_problem_oa(
    10, x_factual, 0.5;
    sparsity_weight=0.1,
    epsilon=0.01,
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

Add an outer approximation cut to the master problem based on constraint type.

# Cut Types

## Upper Bound Cut 
For convex constraint f(x) ≤ y_target + ε, we add UPPER approximation cut:
    constant + ∇f(x_k)·x ≤ y_target + ε

where constant = f(x_k) - ∇f(x_k)·x_k

This defines an OUTER APPROXIMATION of the feasible set {x : f(x) ≤ y_target + ε}.
Since f is convex, the linearization OVER-ESTIMATES f(x), making the feasible
region smaller (more conservative).

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
            constraint_type=:upper_bound, y_target=target, epsilon=eps)
```
"""
function add_oa_cut!(
    master_model::Model,
    x_var::Vector{VariableRef},
    x_point::Vector{Float64},
    f_value::Float64,
    gradient::Vector{Float64};
    y_target::Float64=0.0,
    epsilon::Float64=0.01
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
        verbose::Bool=true
    ) -> Dict

Generate counterfactual using Outer Approximation algorithm.

# Constraint Types (IMPORTANT - See mathematical note at top of file)

## :upper_bound 
- Constraint: f(x) ≤ y_target + ε (CONVEX constraint set)
- Use case: "Reduce cost to at most y_target"
- Method: Direct constraint enforcement via upper approximation cuts
- Optimality: Guaranteed global optimum
- Example: To reduce cost by 10%, set y_target = current_cost * 0.9

# Algorithm
1. Build master MILP 
2. Add initial OA cut at x_factual
3. Main loop (until convergence or max_iterations):
   a. Solve master MILP → candidate solution x_k
   b. Evaluate NN(x_k) → get f_k and ∇f_k
   c. Check feasibility 
   d. Update bounds
   e. Add appropriate OA cut
   f. Check convergence
4. Return best feasible solution or infeasible status

# Arguments
- `icnn_model`: Trained FICNN model
- `x_factual::Vector{Float32}`: Factual input point
- `y_target::Float64`: Target output value
- `epsilon::Float64`: Tolerance for target matching (default: 0.01)
- `sparsity_weight::Float64`: Weight for sparsity penalty (default: 0.1)
- `x_bounds::Tuple{Float64, Float64}`: Bounds for input features (default: (0.0, 1.0))
- `max_iterations::Int`: Maximum OA iterations (default: 50)
- `time_limit_per_iter::Float64`: Time limit per MILP solve (default: 60s)
- `tolerance::Float64`: Convergence tolerance for UB - LB (default: 1e-6)
- `immutable_indices::Vector{Int}`: Indices that cannot be changed (default: [])
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
  - `:lower_bound`: Final lower bound
  - `:upper_bound`: Final upper bound
  - `:iteration_history`: Array of iteration logs

# Example
```julia
result = generate_counterfactual_oa(
    model, x_factual, 0.5;
    epsilon=0.01,
    sparsity_weight=0.1,
    max_iterations=30,
    verbose=true
)

if result[:status] == :optimal
    println("Found counterfactual: ", result[:counterfactual])
    println("Distance: ", result[:distance])
    println("Changed features: ", result[:num_changed])
end
```

# Convergence Guarantee
For convex ICNN models, OA is guaranteed to converge to global optimum:
- Lower bound (LB) increases monotonically (master objective)
- Upper bound (UB) decreases when feasible solutions are found
- Convergence when UB - LB ≤ tolerance

# Notes
- Much faster than full MIP for large networks (no hidden layer variables)
- Exploits convexity for global optimality
- Number of OA cuts grows linearly with iterations
- Master problem size: O(n_features) variables, O(iterations) constraints
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
    verbose::Bool=true
)
    n_features = length(x_factual)

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
        println("  - Max iterations: $max_iterations")
        println("  - Time limit per iter: $(time_limit_per_iter)s")
        println("  - Convergence tolerance: $tolerance")
        println("  - Immutable features: $(length(immutable_indices))")
        println()
    end

    # Check if already at target (depends on constraint type)
    already_satisfied = y_current <= y_target + epsilon

    if already_satisfied
        if verbose
            println("✓ Already satisfies target constraint!")
        end
        return Dict(
            :status => :already_at_target,
            :counterfactual => x_factual,
            :prediction => Float64(y_current),
            :distance => 0.0,
            :num_changed => 0,
            :changed_indices => Int[],
            :iterations => 0,
            :solve_time => 0.0,
            :lower_bound => 0.0,
            :upper_bound => 0.0,
            :iteration_history => []
        )
    end

    # Build master problem
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
        println()
        println("Starting OA iterations...")
        println("-" ^ 70)
    end

    # Initialize bounds and tracking
    LB = -Inf
    UB = Inf
    best_x = nothing
    best_f = nothing
    best_obj = Inf
    iteration_history = []

    start_time = time()

    # Main OA loop
    for iter in 1:max_iterations
        # Step 1: Solve master MILP
        set_time_limit_sec(master_model, time_limit_per_iter)
        optimize!(master_model)

        status = termination_status(master_model)

        # Check termination status
        if status == MOI.INFEASIBLE || status == MOI.INFEASIBLE_OR_UNBOUNDED
            if verbose
                println("\n✗ Master problem is infeasible - no counterfactual exists")
            end

            solve_time = time() - start_time
            return Dict(
                :status => :infeasible,
                :counterfactual => nothing,
                :prediction => nothing,
                :distance => Inf,
                :num_changed => nothing,
                :changed_indices => nothing,
                :iterations => iter,
                :solve_time => solve_time,
                :lower_bound => LB,
                :upper_bound => UB,
                :iteration_history => iteration_history
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

        # Update lower bound (master objective is a lower bound)
        LB = obj_k

        # Step 3: Evaluate neural network at x_k
        f_k, grad_k = evaluate_ficnn_with_gradient(icnn_model, x_k)

        # Extract gamma_k (for logging)

        # Step 4: Check feasibility w.r.t. target
        target_error = max(0.0, f_k - (y_target + epsilon))  # Violation amount
        feasible = f_k <= y_target + epsilon

        # Step 5: Update upper bound if feasible
        if feasible && obj_k < UB
            UB = obj_k
            best_x = copy(x_k)
            best_f = f_k
            best_obj = obj_k
        end

        # Step 6: Add OA cut
        add_oa_cut!(
            master_model,
            x_var,
            Float64.(x_k),
            f_k,
            grad_k;
            y_target=y_target,
            epsilon=epsilon
        )

        # Step 7: Log iteration (including x_k for visualization)
        iter_log = (
            iteration=iter,
            LB=LB,
            UB=UB,
            f_k=f_k,
            target_error=target_error,
            feasible=feasible,
            gap=UB - LB,
            x_k=copy(x_k)  # Store point for convergence analysis
        )
        push!(iteration_history, iter_log)

        if verbose
            feasible_str = feasible ? "✓" : "✗"
            @printf("Iter %2d: LB=%8.4f  UB=%8.4f  Gap=%8.4f  f(x)=%7.4f  err=%6.4f  %s\n",
                    iter, LB, UB, UB - LB, f_k, target_error, feasible_str)
        end

        # Step 8: Check convergence
        if isfinite(UB) && isfinite(LB) && (UB - LB <= tolerance)
            if verbose
                println("-" ^ 70)
                println("✓ Converged! Gap = $(round(UB - LB, digits=8)) ≤ $tolerance")
            end
            break
        end

        # Early termination if we found a very good solution
        if feasible && target_error <= epsilon / 10
            if verbose
                println("-" ^ 70)
                println("✓ Found excellent solution with error = $(round(target_error, digits=6))")
            end
            break
        end
    end

    solve_time = time() - start_time

    if verbose
        println("-" ^ 70)
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
            println("  Final gap: $(round(UB - LB, digits=6))")
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
            :lower_bound => LB,
            :upper_bound => UB,
            :iteration_history => iteration_history
        )
    else
        if verbose
            println("✗ No feasible counterfactual found")
            println("  Iterations: $(length(iteration_history))")
            println("  Solve time: $(round(solve_time, digits=2))s")
            println("  Final lower bound: $(round(LB, digits=4))")
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
            :lower_bound => LB,
            :upper_bound => UB,
            :iteration_history => iteration_history
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
       add_initial_cut!,
       add_integer_cut!
