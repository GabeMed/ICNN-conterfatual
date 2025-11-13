"""
Generate simple HTML report for benchmark results
Always specifies test cases and reduction percentage for all timing results
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
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
n_cases = nrow(df)
n_trials = df.n_trials[1]
reduction_pct = df.reduction_pct[1]

println("Generating HTML report...")

# HTML Header and CSS
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results - $(round(reduction_pct, digits=0))% Cost Reduction</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        .experiment-config {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .experiment-config p {
            margin: 5px 0;
            font-size: 16px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .metric-value {
            font-weight: bold;
            color: #2980b9;
        }
        .success-100 {
            color: #27ae60;
            font-weight: bold;
        }
        .success-partial {
            color: #f39c12;
            font-weight: bold;
        }
        .success-fail {
            color: #e74c3c;
            font-weight: bold;
        }
        .highlight-best {
            background-color: #d5f4e6;
            font-weight: bold;
        }
        .note {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
        }
        .summary-box {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .comparison-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border: 2px solid #dee2e6;
        }
        .comparison-card h3 {
            margin-top: 0;
            color: #495057;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Benchmark Results: Counterfactual Generation Methods</h1>
        
        <div class="experiment-config">
            <p><strong>Experiment:</strong> $(round(reduction_pct, digits=0))% cost reduction</p>
            <p><strong>Number of test cases:</strong> $n_cases</p>
            <p><strong>Trials per method:</strong> $n_trials</p>
            <p><strong>Methods compared:</strong> ECP (Extended Cutting Plane), ESH (Extended Supporting Hyperplane), MILP (Mixed-Integer Linear Programming)</p>
        </div>

        <h2>Table 1: Average Performance for $(round(reduction_pct, digits=0))% Reduction</h2>
"""

# Compute averages
total_trials = sum(df.n_trials)
ecp_success_rate = 100 * sum(df.ecp_success) / total_trials
esh_success_rate = 100 * sum(df.esh_success) / total_trials
milp_success_rate = 100 * sum(df.milp_success) / total_trials

ecp_time_avg = mean(df.ecp_time_mean)
esh_time_avg = mean(df.esh_time_mean)
ecp_iter_avg = mean(df.ecp_iter_mean)
esh_iter_avg = mean(df.esh_iter_mean)

# Check MILP
if sum(df.milp_success) > 0
    milp_times = filter(!isnan, df.milp_time_mean)
    milp_time_avg = mean(milp_times)
    has_milp = true
else
    milp_time_avg = NaN
    has_milp = false
end

# Add comparison cards
html *= """
        <div class="comparison">
            <div class="comparison-card">
                <h3>ECP Method</h3>
                <p><strong>Success Rate:</strong> <span class="$(ecp_success_rate == 100 ? "success-100" : "success-partial")">$(round(ecp_success_rate, digits=0))%</span></p>
                <p><strong>Avg Time ($(round(reduction_pct, digits=0))% reduction):</strong> <span class="metric-value">$(round(ecp_time_avg, digits=3))s</span></p>
                <p><strong>Avg Iterations:</strong> <span class="metric-value">$(round(ecp_iter_avg, digits=1))</span></p>
            </div>
            <div class="comparison-card">
                <h3>ESH Method</h3>
                <p><strong>Success Rate:</strong> <span class="$(esh_success_rate == 100 ? "success-100" : "success-partial")">$(round(esh_success_rate, digits=0))%</span></p>
                <p><strong>Avg Time ($(round(reduction_pct, digits=0))% reduction):</strong> <span class="metric-value">$(round(esh_time_avg, digits=3))s</span></p>
                <p><strong>Avg Iterations:</strong> <span class="metric-value">$(round(esh_iter_avg, digits=1))</span></p>
            </div>
            <div class="comparison-card">
                <h3>MILP Method</h3>
                <p><strong>Success Rate:</strong> <span class="$(milp_success_rate == 100 ? "success-100" : milp_success_rate > 0 ? "success-partial" : "success-fail")">$(round(milp_success_rate, digits=0))%</span></p>
"""

if has_milp
    global html *= """
                <p><strong>Avg Time ($(round(reduction_pct, digits=0))% reduction):</strong> <span class="metric-value">$(round(milp_time_avg, digits=3))s</span></p>
                <p><strong>Avg Iterations:</strong> <span class="metric-value">N/A</span></p>
"""
else
    global html *= """
                <p><strong>Avg Time ($(round(reduction_pct, digits=0))% reduction):</strong> <span class="metric-value">timeout</span></p>
                <p><strong>Avg Iterations:</strong> <span class="metric-value">N/A</span></p>
"""
end

html *= """
            </div>
        </div>
"""

# Summary box
html *= """
        <div class="summary-box">
            <h3>Key Results for $(round(reduction_pct, digits=0))% Cost Reduction:</h3>
"""

if has_milp
    ecp_ratio = ecp_time_avg / milp_time_avg
    esh_ratio = esh_time_avg / milp_time_avg
    if ecp_ratio < 1.0
        global html *= "<p><strong>ECP:</strong> $(round(1.0/ecp_ratio, digits=2))x faster than MILP ($(round(ecp_time_avg, digits=3))s vs $(round(milp_time_avg, digits=3))s)</p>"
    else
        global html *= "<p><strong>ECP:</strong> $(round(ecp_ratio, digits=2))x slower than MILP ($(round(ecp_time_avg, digits=3))s vs $(round(milp_time_avg, digits=3))s)</p>"
    end
    if esh_ratio < 1.0
        global html *= "<p><strong>ESH:</strong> $(round(1.0/esh_ratio, digits=2))x faster than MILP ($(round(esh_time_avg, digits=3))s vs $(round(milp_time_avg, digits=3))s)</p>"
    else
        global html *= "<p><strong>ESH:</strong> $(round(esh_ratio, digits=2))x slower than MILP ($(round(esh_time_avg, digits=3))s vs $(round(milp_time_avg, digits=3))s)</p>"
    end
end

esh_vs_ecp = esh_time_avg / ecp_time_avg
if esh_vs_ecp > 1.0
    pct = (esh_vs_ecp - 1.0) * 100
    global html *= "<p><strong>ESH vs ECP:</strong> ESH is $(round(pct, digits=1))% slower ($(round(esh_time_avg, digits=3))s vs $(round(ecp_time_avg, digits=3))s)</p>"
else
    pct = (1.0 - esh_vs_ecp) * 100
    global html *= "<p><strong>ESH vs ECP:</strong> ESH is $(round(pct, digits=1))% faster ($(round(esh_time_avg, digits=3))s vs $(round(ecp_time_avg, digits=3))s)</p>"
end

html *= """
        </div>

        <h2>Table 2: Results by Test Case ($(round(reduction_pct, digits=0))% Reduction)</h2>
        <table>
            <thead>
                <tr>
                    <th>Case</th>
                    <th>Factual Cost</th>
                    <th>Target Cost</th>
                    <th>ECP Time (s)</th>
                    <th>ESH Time (s)</th>
                    <th>MILP Time (s)</th>
                    <th>ECP Iter</th>
                    <th>ESH Iter</th>
                    <th>Success (E/S/M)</th>
                </tr>
            </thead>
            <tbody>
"""

# Add rows for each case
for i in 1:n_cases
    factual = round(df.y_factual[i], digits=0)
    target = round(df.y_target[i], digits=0)
    ecp_time = round(df.ecp_time_mean[i], digits=3)
    esh_time = round(df.esh_time_mean[i], digits=3)
    ecp_iter = round(df.ecp_iter_mean[i], digits=0)
    esh_iter = round(df.esh_iter_mean[i], digits=0)
    
    milp_str = isnan(df.milp_time_mean[i]) ? "timeout" : string(round(df.milp_time_mean[i], digits=3))
    
    # Determine best time (among successful methods)
    times = [df.ecp_time_mean[i], df.esh_time_mean[i]]
    if !isnan(df.milp_time_mean[i])
        push!(times, df.milp_time_mean[i])
    end
    best_time = minimum(times)
    
    ecp_class = df.ecp_time_mean[i] == best_time ? " class=\"highlight-best\"" : ""
    esh_class = df.esh_time_mean[i] == best_time ? " class=\"highlight-best\"" : ""
    milp_class = !isnan(df.milp_time_mean[i]) && df.milp_time_mean[i] == best_time ? " class=\"highlight-best\"" : ""
    
    global html *= """
                <tr>
                    <td>$i</td>
                    <td>$factual</td>
                    <td>$target</td>
                    <td$ecp_class>$ecp_time</td>
                    <td$esh_class>$esh_time</td>
                    <td$milp_class>$milp_str</td>
                    <td>$ecp_iter</td>
                    <td>$esh_iter</td>
                    <td>$(df.ecp_success[i])/$(df.esh_success[i])/$(df.milp_success[i])</td>
                </tr>
"""
end

html *= """
            </tbody>
        </table>
        <p class="note">Note: Best time for each case is highlighted. All times are for $(round(reduction_pct, digits=0))% cost reduction.</p>
        <p class="note">Success column shows ECP/ESH/MILP successful trials out of $n_trials.</p>

        <h2>Statistical Summary ($(round(reduction_pct, digits=0))% Reduction)</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Mean</th>
                    <th>Median</th>
                    <th>Min</th>
                    <th>Max</th>
                    <th>Std Dev</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>ECP Time (s)</strong></td>
                    <td>$(round(mean(df.ecp_time_mean), digits=3))</td>
                    <td>$(round(median(df.ecp_time_mean), digits=3))</td>
                    <td>$(round(minimum(df.ecp_time_mean), digits=3))</td>
                    <td>$(round(maximum(df.ecp_time_mean), digits=3))</td>
                    <td>$(round(std(df.ecp_time_mean), digits=3))</td>
                </tr>
                <tr>
                    <td><strong>ESH Time (s)</strong></td>
                    <td>$(round(mean(df.esh_time_mean), digits=3))</td>
                    <td>$(round(median(df.esh_time_mean), digits=3))</td>
                    <td>$(round(minimum(df.esh_time_mean), digits=3))</td>
                    <td>$(round(maximum(df.esh_time_mean), digits=3))</td>
                    <td>$(round(std(df.esh_time_mean), digits=3))</td>
                </tr>
"""

if has_milp
    global html *= """
                <tr>
                    <td><strong>MILP Time (s)</strong></td>
                    <td>$(round(mean(milp_times), digits=3))</td>
                    <td>$(round(median(milp_times), digits=3))</td>
                    <td>$(round(minimum(milp_times), digits=3))</td>
                    <td>$(round(maximum(milp_times), digits=3))</td>
                    <td>$(round(std(milp_times), digits=3))</td>
                </tr>
"""
end

html *= """
                <tr>
                    <td><strong>ECP Iterations</strong></td>
                    <td>$(round(mean(df.ecp_iter_mean), digits=1))</td>
                    <td>$(round(median(df.ecp_iter_mean), digits=1))</td>
                    <td>$(round(minimum(df.ecp_iter_mean), digits=0))</td>
                    <td>$(round(maximum(df.ecp_iter_mean), digits=0))</td>
                    <td>$(round(std(df.ecp_iter_mean), digits=1))</td>
                </tr>
                <tr>
                    <td><strong>ESH Iterations</strong></td>
                    <td>$(round(mean(df.esh_iter_mean), digits=1))</td>
                    <td>$(round(median(df.esh_iter_mean), digits=1))</td>
                    <td>$(round(minimum(df.esh_iter_mean), digits=0))</td>
                    <td>$(round(maximum(df.esh_iter_mean), digits=0))</td>
                    <td>$(round(std(df.esh_iter_mean), digits=1))</td>
                </tr>
            </tbody>
        </table>

        <h2>Methodology</h2>
        <div class="summary-box">
            <p><strong>Benchmark Design:</strong></p>
            <ul>
                <li>Randomized execution order to eliminate position bias</li>
                <li>Multiple trials per method ($n_trials trials each)</li>
                <li>Fresh model reload per trial to eliminate warmstart effects</li>
                <li>Garbage collection between trials for fair memory state</li>
                <li>Warmup runs excluded from results (Julia compilation)</li>
                <li>Statistical analysis with mean, median, and standard deviation</li>
                <li>Controlled random seed for reproducibility</li>
            </ul>
            <p><strong>Reduction Target:</strong> $(round(reduction_pct, digits=0))% cost reduction across all test cases</p>
        </div>

        <p class="note" style="margin-top: 30px; text-align: center;">
            Generated on $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        </p>
    </div>
</body>
</html>
"""

# Save HTML
output_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "benchmark_report.html")
open(output_path, "w") do io
    write(io, html)
end

println("✓ HTML report generated: $output_path")
println()

