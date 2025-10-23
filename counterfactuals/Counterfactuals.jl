"""
Counterfactuals Module

Algorithms for generating counterfactual explanations using ICNN as blackbox.

This module provides various algorithms for counterfactual generation:
- Optimization-based methods
- Gradient-based methods  
- Search-based methods

The ICNN model serves as the predictive blackbox model.
"""
module Counterfactuals

using ..ICNN

# Exports will be added as algorithms are implemented
# export generate_counterfactual, DiCE, Wachter, etc.

"""
    AbstractCounterfactualMethod

Base type for counterfactual generation algorithms.

All counterfactual methods should subtype this and implement:
- `generate(method, model, x_original, target; kwargs...)`
"""
abstract type AbstractCounterfactualMethod end

# Algorithm implementations will be included here
# include("algorithms/wachter.jl")
# include("algorithms/dice.jl")
# include("algorithms/growing_spheres.jl")
# etc.

end # module

