"""
MIP-based counterfactual generation for ICNN models.

Given a demand vector x and a desired cost y_target,
find a nearby demand x' such that NN(x') ≈ y_target.

Uses Mixed-Integer Programming to encode the neural network
and minimize the distance while satisfying the target constraint.
"""

using JuMP
using Gurobi
using Statistics
using LinearAlgebra
using Printf

"""
    neural_network_ficnn!(jump_model, icnn_model, x)

Builds FICNN constraints in JuMP.
Architecture: y = NN(x) with feed-forward layers.

Returns: output variable (scalar)
"""
function neural_network_ficnn!(jump_model::Model, icnn_model::FICNN,
                                          x::Vector{VariableRef})
    n_layers = length(icnn_model.hidden_layers)

    # First layer: z_0 = ReLU(W_0 * x + b_0)
    W_0 = Float64.(icnn_model.input_layer.weight)
    b_0 = Float64.(icnn_model.input_layer.bias)
    layer_size = size(W_0, 1)

    z_0 = @variable(jump_model, [1:layer_size])

    for j in 1:layer_size
        a_expr = @expression(jump_model,
            sum(W_0[j, k] * x[k] for k in 1:icnn_model.n_features) + b_0[j])
        # ReLU: z >= a and z >= 0
        @constraint(jump_model, z_0[j] >= a_expr)
        @constraint(jump_model, z_0[j] >= 0)
    end

    z_prev = z_0
    z_layers = [z_0]

    # Hidden layers: z_i = ReLU(W_i * z_{i-1}) or linear for last layer
    for i in 1:n_layers
        W_i = Float64.(icnn_model.hidden_layers[i].weight)
        layer_size = size(W_i, 1)

        z_i = @variable(jump_model, [1:layer_size])

        for j in 1:layer_size
            a_expr = @expression(jump_model,
                sum(W_i[j, k] * z_prev[k] for k in 1:length(z_prev)))

            if i < n_layers  # ReLU for hidden layers
                @constraint(jump_model, z_i[j] >= a_expr)
                @constraint(jump_model, z_i[j] >= 0)
            else  # Linear for output layer
                @constraint(jump_model, z_i[j] == a_expr)
            end
        end

        push!(z_layers, z_i)
        z_prev = z_i
    end

    # Output is scalar (first element of last layer)
    output = z_layers[end][1]

    return output
end

"""
    build_counterfactual_model(icnn_model, n_features;
                              sparsity_weight, x_bounds, epsilon)

Builds counterfactual JuMP model.

Objective: min ||x - x_factual||_1 + sparsity * num_changed
Subject to: |NN(x) - y_target| <= epsilon

Returns: (model, x, y_pred, x_factual, delta_pos, delta_neg)
"""
function build_counterfactual_model(icnn_model::FICNN,
                                               n_features::Int;
                                               sparsity_weight::Float64=0.1,
                                               x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
                                               epsilon::Float64=0.01)
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    # Decision variables: counterfactual input
    @variable(model, x_bounds[1] <= x[i=1:n_features] <= x_bounds[2])

    # Factual input (will be fixed later)
    @variable(model, x_factual[i=1:n_features])

    # Build neural network
    y_pred = neural_network_ficnn!(model, icnn_model, x)

    # Distance variables (L1 norm)
    @variable(model, delta_pos[i=1:n_features] >= 0)
    @variable(model, delta_neg[i=1:n_features] >= 0)

    distance_expr = sum(delta_pos[i] + delta_neg[i] for i in 1:n_features)

    # Sparsity: binary variables indicating feature changes
    @variable(model, changed[i=1:n_features], Bin)
    M = x_bounds[2] - x_bounds[1]  # Big-M value

    @constraint(model, big_m_pos[i=1:n_features], delta_pos[i] <= M * changed[i])
    @constraint(model, big_m_neg[i=1:n_features], delta_neg[i] <= M * changed[i])

    # Objective: minimize distance + sparsity penalty
    @objective(model, Min,
               distance_expr + sparsity_weight * sum(changed[i] for i in 1:n_features))

    return model, x, y_pred, x_factual, delta_pos, delta_neg
end

"""
    set_factual_constraints!(model, x_factual_data, x, x_factual,
                            delta_pos, delta_neg; immutable_indices)

Fixes factual values and adds delta constraints.
Optionally fixes immutable features (e.g., bus locations in power systems).
"""
function set_factual_constraints!(model, x_factual_data::Vector{Float32},
                                   x::Vector{VariableRef},
                                   x_factual::Vector{VariableRef},
                                   delta_pos::Vector{VariableRef},
                                   delta_neg::Vector{VariableRef};
                                   immutable_indices::Vector{Int}=Int[])
    n_features = length(x_factual_data)

    for i in 1:n_features
        # Fix factual value
        fix(x_factual[i], x_factual_data[i])

        # Delta constraint: x[i] - x_factual[i] = delta_pos[i] - delta_neg[i]
        @constraint(model, x[i] - x_factual[i] == delta_pos[i] - delta_neg[i])

        # Fix immutable features
        if i in immutable_indices
            fix(x[i], x_factual_data[i], force=true)
        end
    end
end

"""
    add_target_constraint!(model, y_pred, y_target, epsilon)

Adds regression target constraint: |NN(x) - y_target| <= epsilon.
This is linearized as:
    y_target - epsilon <= y_pred <= y_target + epsilon
"""
function add_target_constraint!(model, y_pred::VariableRef,
                               y_target::Float64, epsilon::Float64)
    @constraint(model, target_lower, y_pred >= y_target - epsilon)
    @constraint(model, target_upper, y_pred <= y_target + epsilon)
end

"""
    generate_counterfactual(icnn_model, x_factual, y_target;
                                      epsilon, sparsity_weight, x_bounds,
                                      time_limit, immutable_indices)

Generates counterfactual for regression tasks.

# Arguments
- `icnn_model`: Trained FICNN model
- `x_factual`: Factual input (e.g., current demand vector)
- `y_target`: Target output value (e.g., desired cost)
- `epsilon`: Tolerance for target matching (default: 0.01)
- `sparsity_weight`: Weight for sparsity penalty (default: 0.1)
- `x_bounds`: Bounds for input features (default: (0.0, 1.0) for normalized)
- `time_limit`: Solver time limit in seconds (default: 300)
- `immutable_indices`: Indices of features that cannot be changed (default: [])

# Returns
Dictionary with:
- `:counterfactual`: Counterfactual input x'
- `:distance`: L1 distance ||x' - x||_1
- `:num_changed`: Number of changed features
- `:changed_indices`: Indices of changed features
- `:prediction`: Predicted value NN(x')
- `:solve_time`: Solver time
- `:status`: Optimization status
"""
function generate_counterfactual(icnn_model::FICNN,
                                           x_factual::Vector{Float32},
                                           y_target::Float64;
                                           epsilon::Float64=0.01,
                                           sparsity_weight::Float64=0.1,
                                           x_bounds::Tuple{Float64, Float64}=(0.0, 1.0),
                                           time_limit::Float64=300.0,
                                           immutable_indices::Vector{Int}=Int[])

    # Current prediction
    y_current = predict(icnn_model, reshape(x_factual, 1, :))[1, 1]

    println("Factual: y = $(round(y_current, digits=4))")
    println("Target: y = $(round(y_target, digits=4)) ± $(epsilon)")
    println("Parameters: sparsity=$sparsity_weight, time_limit=$(time_limit)s")

    # Check if already at target
    if abs(y_current - y_target) <= epsilon
        println("✓ Already at target!")
        return Dict(
            :counterfactual => x_factual,
            :distance => 0.0,
            :num_changed => 0,
            :changed_indices => Int[],
            :prediction => y_current,
            :solve_time => 0.0,
            :status => :already_at_target
        )
    end

    # Build MIP model
    n_features = length(x_factual)
    model, x, y_pred, x_factual_var, delta_pos, delta_neg =
        build_counterfactual_model(
            icnn_model, n_features;
            sparsity_weight=sparsity_weight,
            x_bounds=x_bounds,
            epsilon=epsilon
        )

    # Set factual constraints
    set_factual_constraints!(model, x_factual, x, x_factual_var,
                            delta_pos, delta_neg;
                            immutable_indices=immutable_indices)

    # Add target constraint
    add_target_constraint!(model, y_pred, Float64(y_target), epsilon)

    # Solver settings
    set_time_limit_sec(model, time_limit)

    # Solve
    println("Solving MIP...")
    start_time = time()
    optimize!(model)
    solve_time = time() - start_time

    status = termination_status(model)

    # Extract solution
    if status in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        x_cf = value.(x)
        distance = sum(abs.(x_cf .- x_factual))
        y_pred_val = value(y_pred)
        changed_indices = findall(abs.(x_cf .- x_factual) .> 1e-5)
        num_changed = length(changed_indices)

        println("✓ Found counterfactual!")
        println("  Distance: $(round(distance, digits=4))")
        println("  Changed features: $num_changed/$(n_features)")
        println("  Predicted value: $(round(y_pred_val, digits=4))")
        println("  Target error: $(round(abs(y_pred_val - y_target), digits=6))")
        println("  Solve time: $(round(solve_time, digits=2))s")

        if num_changed > 0 && num_changed <= 20
            println("\n  Changed features (showing up to 20):")
            for idx in changed_indices[1:min(20, num_changed)]
                @printf("    Feature %3d: %.4f → %.4f (Δ = %+.4f)\n",
                       idx, x_factual[idx], x_cf[idx], x_cf[idx] - x_factual[idx])
            end
        end

        return Dict(
            :counterfactual => Float32.(x_cf),
            :distance => distance,
            :num_changed => num_changed,
            :changed_indices => changed_indices,
            :prediction => y_pred_val,
            :solve_time => solve_time,
            :status => status
        )
    else
        println("✗ No solution found")
        println("  Status: $status")
        println("  Solve time: $(round(solve_time, digits=2))s")

        return Dict(
            :counterfactual => nothing,
            :distance => Inf,
            :num_changed => nothing,
            :changed_indices => nothing,
            :prediction => nothing,
            :solve_time => solve_time,
            :status => status
        )
    end
end
