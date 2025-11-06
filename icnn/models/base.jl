# Base functionality for Input Convex Neural Networks

"""
    AbstractICNN

Abstract type for Input Convex Neural Networks.
All ICNN variants should inherit from this type.
"""
abstract type AbstractICNN end

"""
    predict(model::AbstractICNN, x, y_init; learning_rate=0.01, momentum=0.9)

Generic prediction function for ICNN models.
Must be implemented by concrete subtypes.
"""
function predict end

"""
    mse_loss(model::AbstractICNN, x, y_init, y_true)

Generic loss function for ICNN models.
Must be implemented by concrete subtypes.
"""
function mse_loss end