using JuMP
using Gurobi
using MathOptAI
using Statistics
using LinearAlgebra
using Printf

"""
Builds FICNN constraints in JuMP using epigraph ReLU (z >= a, z >= 0).
Returns scalar output variable.
"""
function build_ficnn_constraints!(jump_model::Model, icnn_model::FICNN,
                                   x::Vector{VariableRef}, y_target::Float32)
    n_layers = length(icnn_model.layers)

    y_vec = fill(Float64(y_target), icnn_model.n_labels)
    z_layers = Vector{Vector{VariableRef}}(undef, n_layers)

    for i in 1:n_layers
        layer_size = icnn_model.layers[i]
        W_x = Float64.(icnn_model.input_x_layers[i].weight)
        b_x = Float64.(icnn_model.input_x_layers[i].bias)
        W_y = Float64.(icnn_model.input_y_layers[i].weight)
        y_contrib = W_y * y_vec

        z = @variable(jump_model, [1:layer_size])
        z_layers[i] = z

        if i == 1
            for j in 1:layer_size
                a_expr = @expression(jump_model,
                    sum(W_x[j, k] * x[k] for k in 1:icnn_model.n_features) +
                    y_contrib[j] + b_x[j])
                @constraint(jump_model, z[j] >= a_expr)  # ReLU 
                @constraint(jump_model, z[j] >= 0)       # ReLU 
            end
        else
            W_z = Float64.(icnn_model.hidden_layers[i-1].weight)
            z_prev = z_layers[i-1]

            for j in 1:layer_size
                a_expr = @expression(jump_model,
                    sum(W_x[j, k] * x[k] for k in 1:icnn_model.n_features) +
                    sum(W_z[j, k] * z_prev[k] for k in 1:length(z_prev)) +
                    y_contrib[j] + b_x[j])

                if i < n_layers
                    @constraint(jump_model, z[j] >= a_expr) # ReLU 
                    @constraint(jump_model, z[j] >= 0) # ReLU 
                else
                    @constraint(jump_model, z[j] == a_expr)
                end
            end
        end
    end

    output = z_layers[end][1]
    @constraint(jump_model, output >= 0) # Problem, we are clamping the energy to [0,1] 
    @constraint(jump_model, output <= 1) # but its just so we can use the margin for binary classification

    return output
end

"""
Builds counterfactual JuMP model with proper categorical/numeric handling.
Returns: (model, x, y_pred, x_factual, delta_pos, delta_neg)
"""
function build_counterfactual_model_icnn(icnn_model::FICNN, y_target::Float32,
                                          n_features::Int; sparsity_weight::Float64=0.1,
                                          feature_info::Union{Dict, Nothing}=nothing)
    model = Model(Gurobi.Optimizer)
    set_silent(model)

    n_numeric = isnothing(feature_info) ? 6 : feature_info[:n_numeric]
    feature_types = isnothing(feature_info) ? fill(:numeric, n_features) : feature_info[:feature_types]
    feature_groups = isnothing(feature_info) ? Dict{String, Vector{Int}}() : feature_info[:feature_groups]

    x = Vector{VariableRef}(undef, n_features)
    for i in 1:n_features
        if feature_types[i] == :numeric
            x[i] = @variable(model, lower_bound=0.0, upper_bound=1.0)
        else
            x[i] = @variable(model, binary=true)
        end
    end

    @variable(model, x_factual[i=1:n_features])

    y_pred = build_ficnn_constraints!(model, icnn_model, x, y_target)

    @variable(model, delta_pos[i=1:n_features] >= 0)
    @variable(model, delta_neg[i=1:n_features] >= 0)

    distance_expr = sum(delta_pos[i] + delta_neg[i] for i in 1:n_features)

    @variable(model, changed[i=1:n_features], Bin)
    M = 1.0
    @constraint(model, big_m_pos[i=1:n_features], delta_pos[i] <= M * changed[i])
    @constraint(model, big_m_neg[i=1:n_features], delta_neg[i] <= M * changed[i])

    for (group_name, group_indices) in feature_groups
        @constraint(model, sum(x[j] for j in group_indices) <= 1)
    end

    @objective(model, Min, distance_expr + sparsity_weight * sum(changed[i] for i in 1:n_features))

    model.ext[:y_target] = y_target

    return model, x, y_pred, x_factual, delta_pos, delta_neg
end

"""
Fixes factual values, adds delta constraints, and fixes immutable features.
"""
function set_factual_constraints!(model, x_factual_data::Vector{Float32},
                                   x::Vector{VariableRef}, x_factual::Vector{VariableRef},
                                   delta_pos::Vector{VariableRef}, delta_neg::Vector{VariableRef};
                                   feature_info::Union{Dict, Nothing}=nothing)
    n_features = length(x_factual_data)

    immutable_indices = Int[]
    if !isnothing(feature_info)
        immutable_names = feature_info[:immutable_features]
        feature_names = feature_info[:feature_names]
        for (i, fname) in enumerate(feature_names)
            for immut in immutable_names
                if startswith(fname, immut)
                    push!(immutable_indices, i)
                    break
                end
            end
        end
    end

    for i in 1:n_features
        fix(x_factual[i], x_factual_data[i])
        @constraint(model, x[i] - x_factual[i] == delta_pos[i] - delta_neg[i])

        if i in immutable_indices
            fix(x[i], x_factual_data[i], force=true)
        end
    end
end

"""
Adds classification constraint with margin: y_pred on correct side of 0.5 ± margin.
"""
function add_classification_constraints!(model, y_pred::VariableRef, y_target::Float32,
                                          alpha::Float64=1.5)
    margin = (alpha - 1.0) / (2.0 * alpha)

    if y_target == 0.0f0
        @constraint(model, cf_classification, y_pred <= 0.5 - margin)
    elseif y_target == 1.0f0
        @constraint(model, cf_classification, y_pred >= 0.5 + margin)
    else
        error("y_target must be 0.0 or 1.0, got $y_target")
    end
end

"""
Generates counterfactual: min ||x - x_factual||_1 + sparsity_weight*(# changed).
"""
function generate_counterfactual(icnn_model::FICNN, x_factual::Vector{Float32},
                                  y_target::Float32;
                                  alpha::Float64=1.5,
                                  sparsity_weight::Float64=0.1,
                                  time_limit::Float64=300.0,
                                  feature_info::Union{Dict, Nothing}=nothing)

    y_init = fill(0.5f0, 1, 1)
    y_current = predict(icnn_model, reshape(x_factual, 1, :), y_init)[1, 1]
    current_class = y_current > 0.5f0 ? 1.0f0 : 0.0f0

    println("Factual: class=$(Int(current_class)), pred=$(round(y_current, digits=3))")
    println("Target: class=$(Int(y_target)), alpha=$alpha, sparsity=$sparsity_weight")

    if current_class == y_target
        return Dict(:counterfactual => x_factual, :distance => 0.0, :num_changed => 0,
                    :changed_indices => Int[], :prediction => y_current,
                    :solve_time => 0.0, :status => :already_target_class)
    end

    n_features = length(x_factual)
    model, x, y_pred, x_factual_var, delta_pos, delta_neg = build_counterfactual_model_icnn(
        icnn_model, y_target, n_features;
        sparsity_weight=sparsity_weight,
        feature_info=feature_info
    )

    set_factual_constraints!(model, x_factual, x, x_factual_var, delta_pos, delta_neg;
                              feature_info=feature_info)
    add_classification_constraints!(model, y_pred, y_target, alpha)

    set_time_limit_sec(model, time_limit)

    println("Solving...")
    start_time = time()
    optimize!(model)
    solve_time = time() - start_time

    status = termination_status(model)

    if status in [MOI.OPTIMAL, MOI.FEASIBLE_POINT]
        x_cf = value.(x)
        distance = sum(abs.(x_cf .- x_factual))
        y_pred_val = value(y_pred)
        changed_indices = findall(abs.(x_cf .- x_factual) .> 1e-5)
        num_changed = length(changed_indices)

        println("✓ Found: dist=$(round(distance, digits=4)), changed=$num_changed/$(n_features), pred=$(round(y_pred_val, digits=3)), time=$(round(solve_time, digits=2))s")

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
        println("✗ Status: $status, time=$(round(solve_time, digits=2))s")
        return Dict(:counterfactual => nothing, :distance => Inf, :num_changed => nothing,
                    :changed_indices => nothing, :prediction => nothing,
                    :solve_time => solve_time, :status => status)
    end
end

