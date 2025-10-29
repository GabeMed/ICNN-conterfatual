using BSON
using Flux

"""
    load_icnn_model(model_path::String)

Loads a trained FICNN model from a BSON file.

Args:
- model_path: path to .bson file 

Returns: FICNN model ready for inference or counterfactual generation
"""
function load_icnn_model(model_path::String)
    if !isfile(model_path)
        error("Model file not found: $model_path")
    end
    
    println("Loading ICNN model from: $model_path")
    
    # Load the BSON file
    model_data = BSON.load(model_path)
    
    # Extract the model (assuming it was saved with key :model)
    if haskey(model_data, :model)
        model = model_data[:model]
    else
        # Try to extract directly if saved differently
        model = model_data
    end
    
    println("Model loaded successfully!")
    println("  Type: $(typeof(model))")
    
    if isa(model, FICNN)
        println("  Input features: $(model.n_features)")
        println("  Output dimension: $(model.n_output)")
        println("  Hidden layers: $(model.layers)")
    end
    
    return model
end

"""
    test_model_prediction(model::FICNN, x_sample::Vector{Float32})

Tests the model on a sample input to verify it works.
For regression tasks, returns the predicted value.

Returns: predicted_value (scalar Float32)
"""
function test_model_prediction(model::FICNN, x_sample::Vector{Float32})
    # Reshape to (1, n_features)
    x_matrix = reshape(x_sample, 1, :)

    # Run inference (direct forward pass)
    y_pred = predict(model, x_matrix)

    # Extract scalar prediction
    return y_pred[1, 1]
end