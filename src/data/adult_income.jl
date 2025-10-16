using CSV
using DataFrames
using Downloads   

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

"""
    preprocess_adult_income(df::DataFrame)

Preprocess the Adult Income dataset:
1. Clean whitespace
2. Remove missing values
3. Convert categorical variables to one-hot encoding
4. Normalize numerical features
"""
function preprocess_adult_income(df::DataFrame)
    # Clean whitespace
    for col in names(df)
        if eltype(df[!, col]) <: AbstractString
            df[!, col] = strip.(df[!, col])
        end
    end
    
    # Remove missing values
    for col in names(df)
        if eltype(df[!, col]) <: AbstractString
            df = df[df[!, col] .!= "?", :]
        end
    end
    
    # Convert target to binary
    df[!, :income_binary] = [occursin(">50K", x) ? 1.0f0 : 0.0f0 for x in df.income]
    
    # Select features
    numeric_features = ["age", "fnlwgt", "education_num", "capital_gain", 
                       "capital_loss", "hours_per_week"]
    categorical_features = ["workclass", "marital_status", "occupation", 
                          "relationship", "race", "sex", "native_country"]
    
    feature_matrix = []
    feature_names = []
    
    # Add numeric features
    for feat in numeric_features
        push!(feature_matrix, Float32.(df[!, feat]))
        push!(feature_names, feat)
    end
    
    # Add categorical features (one-hot)
    for feat in categorical_features
        unique_vals = unique(df[!, feat])
        for val in unique_vals[2:end]
            binary_feat = Float32.(df[!, feat] .== val)
            push!(feature_matrix, binary_feat)
            push!(feature_names, "$(feat)_$(val)")
        end
    end
    
    X = hcat(feature_matrix...)'
    
    # Normalize numeric features
    for i in 1:6
        m = mean(X[i, :])
        sigma = std(X[i, :])
        if sigma > 0
            X[i, :] = (X[i, :] .- m) ./ sigma
        end
    end
    
    X = X'
    y = reshape(df.income_binary, :, 1)
    
    println("Preprocessing complete:")
    println("  Features: $(size(X, 2))")
    println("  Samples: $(size(X, 1))")
    println("  Positive class ratio: $(mean(y))")
    
    return X, y, feature_names
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