"""
    save_model(model::AbstractICNN, filepath::String)

Save the model to a BSON file.
"""
function save_model(model::AbstractICNN, filepath::String)
    mkpath(dirname(filepath))
    @save filepath model
    println("Model saved in: $filepath")
end

"""
    load_model(filepath::String)

Load a model from a BSON file.
"""
function load_model(filepath::String)
    @load filepath model
    println("Model loaded from: $filepath")
    return model
end