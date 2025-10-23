using CSV
using DataFrames
using Downloads
using Random   

"""
    load_adult_income()

Load the Adult Income dataset from UCI Machine Learning Repository.
Returns a DataFrame containing the raw data.
"""
function load_adult_income()
    train_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
    
    col_names = ["age", "workclass", "fnlwgt", "education", "education_num",
                 "marital_status", "occupation", "relationship", "race", "sex",
                 "capital_gain", "capital_loss", "hours_per_week", "native_country",
                 "income"]
    
    println("Downloading Adult Income dataset...")
    train_file = Downloads.download(train_url)
    test_file = Downloads.download(test_url)
    
    train_df = CSV.read(train_file, DataFrame, header=false, skipto=1)
    test_df = CSV.read(test_file, DataFrame, header=false, skipto=2)
    
    rename!(train_df, col_names)
    rename!(test_df, col_names)
    
    df = vcat(train_df, test_df)
    
    println("Dataset loaded: $(nrow(df)) samples")
    return df
end

struct MinMaxScaler
    xmin::Vector{Float64}
    xmax::Vector{Float64}
end

function fit_minmax(X::AbstractMatrix, num_features::Int=6)
    xmin = [minimum(X[:, j]) for j in 1:num_features]
    xmax = [maximum(X[:, j]) for j in 1:num_features]
    return MinMaxScaler(xmin, xmax)
end

function transform!(X::AbstractMatrix, scaler::MinMaxScaler)
    for j in 1:length(scaler.xmin)
        r = scaler.xmax[j] - scaler.xmin[j]
        if r > 0
            X[:, j] = (X[:, j] .- scaler.xmin[j]) ./ r
        else
            X[:, j] .= 0.0
        end
    end
    return X
end

"""
Preprocesses Adult Income: min-max [0,1] for numerics, one-hot for categoricals.
Returns: (X, y, feature_names, feature_info)
"""
function preprocess_adult_income(df::DataFrame)
    for col in names(df)
        if eltype(df[!, col]) <: AbstractString
            df[!, col] = strip.(df[!, col])
        end
    end

    for col in names(df)
        if eltype(df[!, col]) <: AbstractString
            df = df[df[!, col] .!= "?", :]
        end
    end

    df[!, :income_binary] = [occursin(">50K", x) ? 1.0f0 : 0.0f0 for x in df.income]

    numeric_features = ["age", "fnlwgt", "education_num", "capital_gain",
                       "capital_loss", "hours_per_week"]
    categorical_features = ["workclass", "marital_status", "occupation",
                          "relationship", "race", "sex", "native_country"]

    immutable_features = ["race", "sex", "native_country"]

    feature_matrix = []
    feature_names = []
    feature_types = []
    feature_groups = Dict{String, Vector{Int}}()

    for feat in numeric_features
        push!(feature_matrix, Float32.(df[!, feat]))
        push!(feature_names, feat)
        push!(feature_types, :numeric)
    end

    for feat in categorical_features
        unique_vals = unique(df[!, feat])
        group_indices = []
        for val in unique_vals[2:end]
            binary_feat = Float32.(df[!, feat] .== val)
            push!(feature_matrix, binary_feat)
            fname = "$(feat)_$(val)"
            push!(feature_names, fname)
            push!(feature_types, :categorical)
            push!(group_indices, length(feature_names))
        end
        feature_groups[feat] = group_indices
    end

    X = hcat(feature_matrix...)'
    X = X'
    y = reshape(df.income_binary, :, 1)

    feature_info = Dict(
        :n_numeric => length(numeric_features),
        :feature_names => feature_names,
        :feature_types => feature_types,
        :feature_groups => feature_groups,
        :immutable_features => immutable_features
    )

    println("Preprocessing complete:")
    println("  Features: $(size(X, 2)) ($(length(numeric_features)) numeric, $(length(feature_names) - length(numeric_features)) categorical)")
    println("  Samples: $(size(X, 1))")

    return X, y, feature_names, feature_info
end

"""
    split_data(X, y; train_ratio=0.8, seed=42)

Split data into training and test sets.
"""
function split_data(X, y; train_ratio=0.8, seed=42)
    Random.seed!(seed)
    n = size(X, 1)
    indices = randperm(n)
    train_size = Int(floor(n * train_ratio))
    
    train_idx = indices[1:train_size]
    test_idx = indices[train_size+1:end]
    
    return (X[train_idx, :], y[train_idx, :], 
            X[test_idx, :], y[test_idx, :])
end