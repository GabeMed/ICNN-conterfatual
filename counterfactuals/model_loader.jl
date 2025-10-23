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
        println("  Features: $(model.n_features)")
        println("  Labels: $(model.n_labels)")
        println("  Architecture: $(model.layers)")
    end
    
    return model
end

"""
    test_model_prediction(model::FICNN, x_sample::Vector{Float32})

Tests the model on a sample input to verify it works.

Returns: (predicted_class, probability)
"""
function test_model_prediction(model::FICNN, x_sample::Vector{Float32})
    # Reshape to (1, n_features)
    x_matrix = reshape(x_sample, 1, :)
    
    # Initialize y for optimization
    y_init = fill(0.5f0, 1, 1)
    
    # Run inference
    y_pred = predict(model, x_matrix, y_init)
    
    # Extract scalar prediction
    prob = y_pred[1, 1]
    predicted_class = prob > 0.5f0 ? 1 : 0
    
    return predicted_class, prob
end