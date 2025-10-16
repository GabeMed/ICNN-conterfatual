"""
    plot_decision_boundary(model::AbstractICNN, data_x, data_y, filename::String)

Plot the decision boundary of the model along with the data points.
"""
function plot_decision_boundary(model::AbstractICNN, data_x, data_y, filename::String)
    x_min, x_max = minimum(data_x[:,1]) - 0.5, maximum(data_x[:,1]) + 0.5
    y_min, y_max = minimum(data_x[:,2]) - 0.5, maximum(data_x[:,2]) + 0.5

    xx = range(x_min, x_max, length=20)
    yy = range(y_min, y_max, length=20)

    grid = hcat([[x,y] for x in xx, y in yy]...)' |> collect
    y_init = fill(0.5f0, size(grid,1))

    yn = predict(model, grid, y_init)
    yn = clamp.(yn, 0.0, 1.0)
    zz = reshape(1.0 .- yn[:, 1], length(yy), length(xx))
    
    p = contourf(xx, yy, zz, levels=10, c=:RdBu, alpha=0.5)
    
    # Plot points
    mask0 = data_y[:, 1] .== 0
    scatter!(p, data_x[mask0, 1], data_x[mask0, 2], 
            color=:red, label="Class 0", markersize=3)
    scatter!(p, data_x[.!mask0, 1], data_x[.!mask0, 2], 
            color=:blue, label="Class 1", markersize=3)
    
    savefig(p, "$filename.png")
end