module ICNN

using Flux
using Random
using Statistics
using Plots
using Printf
using CSV
using DataFrames
using Downloads
using BSON: @save, @load
using Zygote
using Zygote: gradient
using JSON
using ImplicitDifferentiation

# Export models
export FICNN, PICNN

# Export training functions
export train!, predict, mse_loss
export enforcing_convexity!, initialize_convex!

# Export data loading functions
export load_adult_income, preprocess_adult_income, split_data

# Export utility functions
export save_model, load_model

# Export metrics functions
export save_training_metrics


# Include model definitions
include("models/base.jl")
include("models/ficnn.jl")

# Include data handling
include("data/adult_income.jl")

# Include training utilities
include("training/trainer.jl")

# Include utilities
include("utils/io.jl")
include("utils/visualization.jl")

end # module