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
export FICNN, PICNN, StandardNN

# Export training functions
export train!, predict, mse_loss
export enforcing_convexity!, initialize_convex!

# Export data loading functions
# DCPM (convex)
export load_dcopf_data, prepare_dcopf_dataset
# ACPM (non-convex)
export load_acpm_data, prepare_acpm_dataset
# Common utilities
export normalize_data, denormalize_output, split_data

# Export utility functions
export save_model, load_model

# Export metrics functions
export save_training_metrics


# Include model definitions
include("models/base.jl")
include("models/ficnn.jl")
include("models/standard_nn.jl")

# Include data handling
include("data/dcopf_loader.jl")
include("data/acpm_loader.jl")

# Include training utilities
include("training/trainer.jl")

# Include utilities
include("utils/io.jl")
include("utils/visualization.jl")

end # module