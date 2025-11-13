"""
Visualize fair benchmark results comparing ECP and ESH methods
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Plots
using Statistics
using Printf
using Dates

# Read benchmark results
results_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "fair_benchmark_results.csv")

if !isfile(results_path)
    println("Error: Benchmark results not found at: $results_path")
    println("Run examples/benchmark_fair_comparison.jl first")
    exit(1)
end

df = CSV.read(results_path, DataFrame)
println("Loaded $(nrow(df)) test cases")
println()

# Fair benchmark only contains successful cases
n_cases = nrow(df)
println("All $n_cases cases are successful")
println()

# Create output directory
output_dir = joinpath(@__DIR__, "..", "tmp", "benchmark_results")
mkpath(output_dir)

# Set plot defaults
default(fontfamily="Computer Modern", framestyle=:box, grid=:y, gridstyle=:dash, 
        gridalpha=0.3, legendfontsize=10, guidefontsize=12, tickfontsize=10)

println("Generating visualizations...")
println()

# 1. Solve time comparison
println("Creating solve time comparison...")

p1 = plot(
    title="Solve Time Comparison (Mean ± Std)",
    xlabel="Test Case",
    ylabel="Time (seconds)",
    legend=:topleft,
    size=(1000, 600)
)

case_ids = 1:n_cases
plot!(p1, case_ids, df.ecp_time_mean, 
      ribbon=df.ecp_time_std,
      label="ECP", marker=:circle, linewidth=2, color=:blue,
      fillalpha=0.2)
plot!(p1, case_ids, df.esh_time_mean, 
      ribbon=df.esh_time_std,
      label="ESH", marker=:square, linewidth=2, color=:green,
      fillalpha=0.2)

savefig(p1, joinpath(output_dir, "1_solve_time_comparison.png"))
println("  Saved: 1_solve_time_comparison.png")

# 2. Iterations comparison
println("Creating iterations comparison...")

p2 = plot(
    title="Iterations Comparison (OA Methods)",
    xlabel="Test Case",
    ylabel="Number of Iterations",
    legend=:topleft,
    size=(1000, 600)
)

plot!(p2, case_ids, df.ecp_iter_mean, 
      label="ECP", marker=:circle, linewidth=2, color=:blue)
plot!(p2, case_ids, df.esh_iter_mean, 
      label="ESH", marker=:square, linewidth=2, color=:green)

savefig(p2, joinpath(output_dir, "2_iterations_comparison.png"))
println("  Saved: 2_iterations_comparison.png")

# 3. Time per iteration
println("Creating time per iteration comparison...")

time_per_iter_ecp = df.ecp_time_mean ./ df.ecp_iter_mean
time_per_iter_esh = df.esh_time_mean ./ df.esh_iter_mean

p3 = plot(
    title="Time per Iteration",
    xlabel="Test Case",
    ylabel="Time/Iteration (seconds)",
    legend=:topleft,
    size=(1000, 600)
)

plot!(p3, case_ids, time_per_iter_ecp, 
      label="ECP", marker=:circle, linewidth=2, color=:blue)
plot!(p3, case_ids, time_per_iter_esh, 
      label="ESH", marker=:square, linewidth=2, color=:green)

savefig(p3, joinpath(output_dir, "3_time_per_iteration.png"))
println("  Saved: 3_time_per_iteration.png")

# 4. ESH vs ECP scatter plot (time)
println("Creating ESH vs ECP time scatter...")

p4 = scatter(
    df.ecp_time_mean,
    df.esh_time_mean,
    xlabel="ECP Time (s)",
    ylabel="ESH Time (s)",
    title="ESH vs ECP: Solve Time",
    label="Test cases",
    color=:blue,
    marker=:circle,
    markersize=8,
    size=(800, 800),
    aspect_ratio=:equal
)

# Add diagonal line (y=x)
max_time = max(maximum(df.ecp_time_mean), maximum(df.esh_time_mean))
min_time = min(minimum(df.ecp_time_mean), minimum(df.esh_time_mean))
plot!(p4, [min_time, max_time], [min_time, max_time], 
      label="Equal time", linestyle=:dash, color=:red, linewidth=2)

# Add text annotations for which is better
annotate!(p4, max_time*0.8, max_time*0.2, 
         text("ECP faster", :blue, 12))
annotate!(p4, max_time*0.2, max_time*0.8, 
         text("ESH faster", :green, 12))

savefig(p4, joinpath(output_dir, "4_esh_vs_ecp_time_scatter.png"))
println("  Saved: 4_esh_vs_ecp_time_scatter.png")

# 5. ESH vs ECP scatter plot (iterations)
println("Creating ESH vs ECP iterations scatter...")

p5 = scatter(
    df.ecp_iter_mean,
    df.esh_iter_mean,
    xlabel="ECP Iterations",
    ylabel="ESH Iterations",
    title="ESH vs ECP: Iterations",
    label="Test cases",
    color=:green,
    marker=:square,
    markersize=8,
    size=(800, 800),
    aspect_ratio=:equal
)

# Add diagonal line
max_iter = max(maximum(df.ecp_iter_mean), maximum(df.esh_iter_mean))
min_iter = min(minimum(df.ecp_iter_mean), minimum(df.esh_iter_mean))
plot!(p5, [min_iter, max_iter], [min_iter, max_iter], 
      label="Equal iterations", linestyle=:dash, color=:red, linewidth=2)

annotate!(p5, max_iter*0.8, max_iter*0.2, 
         text("ECP fewer", :blue, 12))
annotate!(p5, max_iter*0.2, max_iter*0.8, 
         text("ESH fewer", :green, 12))

savefig(p5, joinpath(output_dir, "5_esh_vs_ecp_iterations_scatter.png"))
println("  Saved: 5_esh_vs_ecp_iterations_scatter.png")

# 6. Summary statistics plot
println("Creating summary statistics...")

p6 = plot(layout=(2,2), size=(1200, 1000), 
          title=["Mean Solve Time" "Median Solve Time" "Mean Iterations" "Total Speedup"])

# Mean time
bar!(p6[1], ["ECP", "ESH"], 
     [mean(df.ecp_time_mean), mean(df.esh_time_mean)],
     color=[:blue :green],
     ylabel="Time (s)",
     legend=false)

# Median time
bar!(p6[2], ["ECP", "ESH"], 
     [median(df.ecp_time_mean), median(df.esh_time_mean)],
     color=[:blue :green],
     ylabel="Time (s)",
     legend=false)

# Mean iterations
bar!(p6[3], ["ECP", "ESH"], 
     [mean(df.ecp_iter_mean), mean(df.esh_iter_mean)],
     color=[:blue :green],
     ylabel="Iterations",
     legend=false)

# Time ratio (ESH/ECP)
time_ratios = df.esh_time_mean ./ df.ecp_time_mean
bar!(p6[4], case_ids, time_ratios,
     color=ifelse.(time_ratios .< 1.0, :green, :blue),
     ylabel="ESH/ECP Ratio",
     xlabel="Test Case",
     legend=false)
hline!(p6[4], [1.0], linestyle=:dash, color=:red, linewidth=2)

savefig(p6, joinpath(output_dir, "6_summary_statistics.png"))
println("  Saved: 6_summary_statistics.png")

# Print summary statistics
println()
println("="^80)
println("SUMMARY STATISTICS")
println("="^80)
println()

println("Solve Time (seconds):")
println(@sprintf("  ECP:  Mean=%.3f  Median=%.3f  Std=%.3f", 
                 mean(df.ecp_time_mean), median(df.ecp_time_mean), std(df.ecp_time_mean)))
println(@sprintf("  ESH:  Mean=%.3f  Median=%.3f  Std=%.3f", 
                 mean(df.esh_time_mean), median(df.esh_time_mean), std(df.esh_time_mean)))
println()

println("Iterations:")
println(@sprintf("  ECP:  Mean=%.1f  Median=%.1f  Std=%.1f", 
                 mean(df.ecp_iter_mean), median(df.ecp_iter_mean), std(df.ecp_iter_mean)))
println(@sprintf("  ESH:  Mean=%.1f  Median=%.1f  Std=%.1f", 
                 mean(df.esh_iter_mean), median(df.esh_iter_mean), std(df.esh_iter_mean)))
println()

println("ESH vs ECP Comparison:")
faster_count = count(df.esh_time_mean .< df.ecp_time_mean)
println(@sprintf("  ESH faster in %d/%d cases (%.1f%%)", 
                 faster_count, n_cases, 100*faster_count/n_cases))

avg_speedup = mean(df.ecp_time_mean) / mean(df.esh_time_mean)
if avg_speedup > 1.0
    println(@sprintf("  ESH is %.1f%% faster on average (%.2fx speedup)", 
                     100*(avg_speedup-1), avg_speedup))
else
    println(@sprintf("  ESH is %.1f%% slower on average (%.2fx)", 
                     100*(1-avg_speedup), avg_speedup))
end

iter_diff = mean(df.ecp_iter_mean) - mean(df.esh_iter_mean)
if iter_diff > 0
    println(@sprintf("  ESH uses %.1f fewer iterations on average", iter_diff))
else
    println(@sprintf("  ESH uses %.1f more iterations on average", -iter_diff))
end

println()
println("="^80)
println("All visualizations saved to: $output_dir")
println("="^80)

# Generate HTML report
println()
println("Generating HTML report...")

html_path = joinpath(output_dir, "fair_benchmark_report.html")

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fair Benchmark Report: ECP vs ESH</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 10px;
            margin-top: 30px;
        }
        .summary-box {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-card h3 {
            margin-top: 0;
            color: #2c3e50;
            font-size: 14px;
            text-transform: uppercase;
        }
        .stat-value {
            font-size: 32px;
            font-weight: bold;
            color: #3498db;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .comparison-table th {
            background-color: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
        }
        .comparison-table td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        .comparison-table tr:hover {
            background-color: #f8f9fa;
        }
        .better {
            color: #27ae60;
            font-weight: bold;
        }
        .worse {
            color: #e74c3c;
            font-weight: bold;
        }
        .plot-container {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .methodology {
            background: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            margin: 20px 0;
        }
        .methodology ul {
            margin: 10px 0;
        }
        .methodology li {
            margin: 5px 0;
        }
        .footer {
            text-align: center;
            color: #7f8c8d;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #bdc3c7;
        }
    </style>
</head>
<body>
    <h1>Fair Benchmark Report: ECP vs ESH</h1>
    
    <div class="summary-box">
        <p><strong>Generated:</strong> $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))</p>
        <p><strong>Test Cases:</strong> $n_cases</p>
        <p><strong>Trials per Method:</strong> $(df.n_trials[1])</p>
    </div>

    <h2>Executive Summary</h2>
    <div class="stats-grid">
        <div class="stat-card">
            <h3>ECP Mean Time</h3>
            <div class="stat-value">$(round(mean(df.ecp_time_mean), digits=3))s</div>
            <div class="stat-label">± $(round(std(df.ecp_time_mean), digits=3))s</div>
        </div>
        <div class="stat-card">
            <h3>ESH Mean Time</h3>
            <div class="stat-value">$(round(mean(df.esh_time_mean), digits=3))s</div>
            <div class="stat-label">± $(round(std(df.esh_time_mean), digits=3))s</div>
        </div>
        <div class="stat-card">
            <h3>ESH Faster</h3>
            <div class="stat-value">$faster_count/$n_cases</div>
            <div class="stat-label">$(round(100*faster_count/n_cases, digits=1))% of cases</div>
        </div>
        <div class="stat-card">
            <h3>Average Speedup</h3>
            <div class="stat-value">$(round(avg_speedup, digits=2))x</div>
            <div class="stat-label">$(avg_speedup > 1.0 ? "ESH faster" : "ECP faster")</div>
        </div>
    </div>

    <h2>Detailed Comparison</h2>
    <table class="comparison-table">
        <thead>
            <tr>
                <th>Case</th>
                <th>ECP Time (s)</th>
                <th>ESH Time (s)</th>
                <th>Speedup</th>
                <th>ECP Iters</th>
                <th>ESH Iters</th>
                <th>Iter Diff</th>
            </tr>
        </thead>
        <tbody>
"""

for i in 1:n_cases
    ecp_time = round(df.ecp_time_mean[i], digits=3)
    esh_time = round(df.esh_time_mean[i], digits=3)
    speedup = round(ecp_time / esh_time, digits=2)
    speedup_class = speedup > 1.0 ? "better" : "worse"
    
    ecp_iter = Int(df.ecp_iter_mean[i])
    esh_iter = Int(df.esh_iter_mean[i])
    local iter_diff = ecp_iter - esh_iter
    iter_class = iter_diff > 0 ? "better" : "worse"
    
    global html_content *= """
            <tr>
                <td>Case $i</td>
                <td>$ecp_time ± $(round(df.ecp_time_std[i], digits=3))</td>
                <td>$esh_time ± $(round(df.esh_time_std[i], digits=3))</td>
                <td class="$speedup_class">$(speedup)x</td>
                <td>$ecp_iter</td>
                <td>$esh_iter</td>
                <td class="$iter_class">$(iter_diff > 0 ? "+" : "")$iter_diff</td>
            </tr>
"""
end

html_content *= """
        </tbody>
    </table>

    <h2>Visualizations</h2>
    
    <div class="plot-container">
        <h3>Solve Time Comparison</h3>
        <img src="1_solve_time_comparison.png" alt="Solve Time Comparison">
    </div>

    <div class="plot-container">
        <h3>Iterations Comparison</h3>
        <img src="2_iterations_comparison.png" alt="Iterations Comparison">
    </div>

    <div class="plot-container">
        <h3>Time per Iteration</h3>
        <img src="3_time_per_iteration.png" alt="Time per Iteration">
    </div>

    <div class="plot-container">
        <h3>ESH vs ECP: Solve Time</h3>
        <img src="4_esh_vs_ecp_time_scatter.png" alt="ESH vs ECP Time Scatter">
        <p>Points below the diagonal line indicate ESH is faster</p>
    </div>

    <div class="plot-container">
        <h3>ESH vs ECP: Iterations</h3>
        <img src="5_esh_vs_ecp_iterations_scatter.png" alt="ESH vs ECP Iterations Scatter">
        <p>Points below the diagonal line indicate ESH uses fewer iterations</p>
    </div>

    <div class="plot-container">
        <h3>Summary Statistics</h3>
        <img src="6_summary_statistics.png" alt="Summary Statistics">
    </div>

    <h2>Statistical Analysis</h2>
    <div class="summary-box">
        <h3>Solve Time Statistics</h3>
        <ul>
            <li><strong>ECP:</strong> Mean = $(round(mean(df.ecp_time_mean), digits=3))s, Median = $(round(median(df.ecp_time_mean), digits=3))s, Std = $(round(std(df.ecp_time_mean), digits=3))s</li>
            <li><strong>ESH:</strong> Mean = $(round(mean(df.esh_time_mean), digits=3))s, Median = $(round(median(df.esh_time_mean), digits=3))s, Std = $(round(std(df.esh_time_mean), digits=3))s</li>
        </ul>
        
        <h3>Iterations Statistics</h3>
        <ul>
            <li><strong>ECP:</strong> Mean = $(round(mean(df.ecp_iter_mean), digits=1)), Median = $(round(median(df.ecp_iter_mean), digits=1)), Std = $(round(std(df.ecp_iter_mean), digits=1))</li>
            <li><strong>ESH:</strong> Mean = $(round(mean(df.esh_iter_mean), digits=1)), Median = $(round(median(df.esh_iter_mean), digits=1)), Std = $(round(std(df.esh_iter_mean), digits=1))</li>
        </ul>
        
        <h3>Key Findings</h3>
        <ul>
            <li>ESH was faster in <strong>$faster_count out of $n_cases</strong> test cases ($(round(100*faster_count/n_cases, digits=1))%)</li>
            <li>Average speedup: <strong>$(round(avg_speedup, digits=2))x</strong> $(avg_speedup > 1.0 ? "(ESH faster)" : "(ECP faster)")</li>
            <li>Average iteration difference: <strong>$(round(iter_diff, digits=1))</strong> $(iter_diff > 0 ? "(ESH uses fewer)" : "(ESH uses more)")</li>
        </ul>
    </div>

    <h2>Benchmark Methodology</h2>
    <div class="methodology">
        <p><strong>This benchmark follows rigorous best practices to ensure fair comparison:</strong></p>
        <ul>
            <li>✓ <strong>Randomized execution order</strong> - Eliminates position bias</li>
            <li>✓ <strong>Multiple trials per method</strong> - $(df.n_trials[1]) trials each for statistical validity</li>
            <li>✓ <strong>Fresh model reload per trial</strong> - Eliminates warmstart effects</li>
            <li>✓ <strong>Garbage collection between trials</strong> - Fair memory state</li>
            <li>✓ <strong>Warmup runs excluded</strong> - Julia compilation effects eliminated</li>
            <li>✓ <strong>Statistical analysis</strong> - Mean, median, and standard deviation reported</li>
            <li>✓ <strong>Controlled random seed</strong> - Results are reproducible</li>
        </ul>
        <p><strong>Results are FAIR and ORDER-INDEPENDENT</strong></p>
    </div>

    <div class="footer">
        <p>Generated by Fair Benchmark Visualization Script</p>
        <p>ICNN Counterfactual Generation Project</p>
    </div>
</body>
</html>
"""

open(html_path, "w") do f
    write(f, html_content)
end

println("✓ HTML report saved to: $html_path")
println()
