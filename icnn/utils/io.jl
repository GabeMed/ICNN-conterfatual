"""
    save_model(model::AbstractICNN, filepath::String; scaler_X=nothing, scaler_Y=nothing)

Save the model and normalization parameters to a BSON file.

# Arguments
- `model::AbstractICNN`: The model to save
- `filepath::String`: Path to save the model
- `scaler_X`: Normalization parameters for X (optional but HIGHLY recommended)
- `scaler_Y`: Normalization parameters for Y (optional but HIGHLY recommended)

# Example
```julia
dataset = prepare_dcopf_dataset("data.bson"; normalize_method=:minmax)
model = FICNN(...)
train!(model, dataset.X_train, dataset.Y_train, ...)
save_model(model, "model.bson"; scaler_X=dataset.scaler_X, scaler_Y=dataset.scaler_Y)
```
"""
function save_model(model::AbstractICNN, filepath::String; scaler_X=nothing, scaler_Y=nothing)
    mkpath(dirname(filepath))
    
    if scaler_X !== nothing && scaler_Y !== nothing
        @save filepath model scaler_X scaler_Y
        println("Model saved in: $filepath")
        println("  ✓ Normalization parameters saved: $(scaler_Y[:method])")
    else
        @save filepath model
        println("Model saved in: $filepath")
        println("  ⚠ WARNING: Normalization parameters NOT saved - inference may be incorrect!")
    end
end

"""
    load_model(filepath::String)

Load a model and normalization parameters from a BSON file.

# Returns
If scalers are present: (model, scaler_X, scaler_Y)
If scalers are absent: model only (with warning)

# Example
```julia
# With scalers (recommended)
model, scaler_X, scaler_Y = load_model("model.bson")

# Without scalers (old models)
model = load_model("model.bson")
```
"""
function load_model(filepath::String)
    data = BSON.load(filepath)
    
    println("Model loaded from: $filepath")
    
    if haskey(data, :scaler_X) && haskey(data, :scaler_Y)
        println("  ✓ Normalization parameters found: $(data[:scaler_Y][:method])")
        return data[:model], data[:scaler_X], data[:scaler_Y]
    else
        println("  ⚠ WARNING: Normalization parameters NOT found!")
        println("    Model may have been trained with different normalization.")
        println("    Predictions may be incorrect if normalization mismatch exists.")
        return data[:model]
    end
end

"""
    save_training_metrics(metrics::Dict, filepath::String)

Save training metrics to JSON file for comparison with Python implementation.
"""
function save_training_metrics(metrics::Dict, filepath::String)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        JSON.print(io, metrics, 2)
    end
    println("Metrics saved to: $filepath")
end