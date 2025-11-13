"""
Generate comprehensive HTML report for multiple reduction targets
Shows results for 20%, 30%, 40%, 50%, and 60% cost reduction
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Printf
using Dates

# Read benchmark results
results_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "multiple_reductions_results.csv")

if !isfile(results_path)
    println("Error: Multiple reductions results not found at: $results_path")
    println("Run examples/benchmark_multiple_reductions.jl first")
    exit(1)
end

df = CSV.read(results_path, DataFrame)
n_trials = df.n_trials[1]
reduction_targets = sort(unique(df.reduction_pct))

println("Generating comprehensive HTML report...")
println("Reduction targets: ", join([string(Int(r), "%") for r in reduction_targets], ", "))

# HTML Header and CSS
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Benchmark Results - Multiple Reduction Targets</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1400px;
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
            margin-top: 40px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        h3 {
            color: #34495e;
            margin-top: 25px;
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
        .highlight-best {
            background-color: #d5f4e6;
            font-weight: bold;
        }
        .success-100 { color: #27ae60; font-weight: bold; }
        .success-partial { color: #f39c12; font-weight: bold; }
        .success-fail { color: #e74c3c; font-weight: bold; }
        .note {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
            font-size: 14px;
        }
        .summary-box {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .reduction-section {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }
        .reduction-header {
            background-color: #3498db;
            color: white;
            padding: 10px 15px;
            margin: -20px -20px 20px -20px;
            border-radius: 6px 6px 0 0;
            font-size: 20px;
            font-weight: bold;
        }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .comparison-card {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            border: 2px solid #dee2e6;
            text-align: center;
        }
        .comparison-card h4 {
            margin-top: 0;
            color: #495057;
        }
        .big-number {
            font-size: 28px;
            font-weight: bold;
            color: #2980b9;
            margin: 10px 0;
        }
        .chart-container {
            margin: 30px 0;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Comprehensive Benchmark: Multiple Cost Reduction Targets</h1>
        
        <div class="experiment-config">
            <p><strong>Reduction Targets:</strong> $(join([string(Int(r), "%") for r in reduction_targets], ", "))</p>
            <p><strong>Test cases per reduction:</strong> $(length(unique(df[df.reduction_pct .== reduction_targets[1], :case_idx])))</p>
            <p><strong>Trials per method:</strong> $n_trials</p>
            <p><strong>Methods compared:</strong> ECP (Extended Cutting Plane), ESH (Extended Supporting Hyperplane), MILP (Mixed-Integer Linear Programming)</p>
        </div>

        <h2>Performance Overview Across All Reduction Targets</h2>
        <table>
            <thead>
                <tr>
                    <th>Reduction</th>
                    <th>ECP Time (s)</th>
                    <th>ESH Time (s)</th>
                    <th>MILP Time (s)</th>
                    <th>ECP Iter</th>
                    <th>ESH Iter</th>
                    <th>Best Method</th>
                </tr>
            </thead>
            <tbody>
"""

# Add overview rows for each reduction
for reduction_pct in reduction_targets
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    
    ecp_avg = mean(df_subset.ecp_time_mean)
    esh_avg = mean(df_subset.esh_time_mean)
    ecp_iter = mean(df_subset.ecp_iter_mean)
    esh_iter = mean(df_subset.esh_iter_mean)
    
    milp_times = filter(!isnan, df_subset.milp_time_mean)
    milp_str = isempty(milp_times) ? "timeout" : @sprintf("%.3f", mean(milp_times))
    
    # Determine best method
    times = [ecp_avg, esh_avg]
    methods = ["ECP", "ESH"]
    if !isempty(milp_times)
        push!(times, mean(milp_times))
        push!(methods, "MILP")
    end
    best_idx = argmin(times)
    best_method = methods[best_idx]
    
    global html *= """
                <tr>
                    <td><strong>$(Int(reduction_pct))%</strong></td>
                    <td$(best_method == "ECP" ? " class=\"highlight-best\"" : "")>$(round(ecp_avg, digits=3))</td>
                    <td$(best_method == "ESH" ? " class=\"highlight-best\"" : "")>$(round(esh_avg, digits=3))</td>
                    <td$(best_method == "MILP" ? " class=\"highlight-best\"" : "")>$milp_str</td>
                    <td>$(round(ecp_iter, digits=1))</td>
                    <td>$(round(esh_iter, digits=1))</td>
                    <td><strong>$best_method</strong></td>
                </tr>
"""
end

html *= """
            </tbody>
        </table>
        <p class="note">Best time for each reduction target is highlighted.</p>
"""

# Detailed sections for each reduction
for reduction_pct in reduction_targets
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    n_cases = nrow(df_subset)
    
    total_trials = sum(df_subset.n_trials)
    ecp_success_rate = 100 * sum(df_subset.ecp_success) / total_trials
    esh_success_rate = 100 * sum(df_subset.esh_success) / total_trials
    milp_success_rate = 100 * sum(df_subset.milp_success) / total_trials
    
    ecp_time_avg = mean(df_subset.ecp_time_mean)
    esh_time_avg = mean(df_subset.esh_time_mean)
    ecp_iter_avg = mean(df_subset.ecp_iter_mean)
    esh_iter_avg = mean(df_subset.esh_iter_mean)
    
    milp_times = filter(!isnan, df_subset.milp_time_mean)
    has_milp = !isempty(milp_times)
    milp_time_avg = has_milp ? mean(milp_times) : NaN
    
    global html *= """
        <div class="reduction-section">
            <div class="reduction-header">
                $(Int(reduction_pct))% Cost Reduction
            </div>
            
            <div class="comparison-grid">
                <div class="comparison-card">
                    <h4>ECP Method</h4>
                    <div class="big-number">$(round(ecp_time_avg, digits=3))s</div>
                    <p>Success: <span class="$(ecp_success_rate == 100 ? "success-100" : "success-partial")">$(round(ecp_success_rate, digits=0))%</span></p>
                    <p>Iterations: $(round(ecp_iter_avg, digits=1))</p>
                </div>
                <div class="comparison-card">
                    <h4>ESH Method</h4>
                    <div class="big-number">$(round(esh_time_avg, digits=3))s</div>
                    <p>Success: <span class="$(esh_success_rate == 100 ? "success-100" : "success-partial")">$(round(esh_success_rate, digits=0))%</span></p>
                    <p>Iterations: $(round(esh_iter_avg, digits=1))</p>
                </div>
"""
    
    if has_milp
        global html *= """
                <div class="comparison-card">
                    <h4>MILP Method</h4>
                    <div class="big-number">$(round(milp_time_avg, digits=3))s</div>
                    <p>Success: <span class="$(milp_success_rate == 100 ? "success-100" : "success-partial")">$(round(milp_success_rate, digits=0))%</span></p>
                    <p>Iterations: N/A</p>
                </div>
"""
    else
        global html *= """
                <div class="comparison-card">
                    <h4>MILP Method</h4>
                    <div class="big-number">timeout</div>
                    <p>Success: <span class="success-fail">$(round(milp_success_rate, digits=0))%</span></p>
                    <p>Iterations: N/A</p>
                </div>
"""
    end
    
    global html *= """
            </div>
            
            <h3>Results by Test Case ($(Int(reduction_pct))% Reduction)</h3>
            <table>
                <thead>
                    <tr>
                        <th>Case</th>
                        <th>Factual Cost</th>
                        <th>Target Cost</th>
                        <th>ECP Time</th>
                        <th>ESH Time</th>
                        <th>MILP Time</th>
                        <th>Success</th>
                    </tr>
                </thead>
                <tbody>
"""
    
    for i in 1:n_cases
        row = df_subset[i, :]
        factual = round(row.y_factual, digits=0)
        target = round(row.y_target, digits=0)
        ecp_time = round(row.ecp_time_mean, digits=3)
        esh_time = round(row.esh_time_mean, digits=3)
        milp_str = isnan(row.milp_time_mean) ? "timeout" : string(round(row.milp_time_mean, digits=3))
        
        # Best time for this case
        times = [row.ecp_time_mean, row.esh_time_mean]
        if !isnan(row.milp_time_mean)
            push!(times, row.milp_time_mean)
        end
        best_time = minimum(times)
        
        ecp_class = row.ecp_time_mean == best_time ? " class=\"highlight-best\"" : ""
        esh_class = row.esh_time_mean == best_time ? " class=\"highlight-best\"" : ""
        milp_class = !isnan(row.milp_time_mean) && row.milp_time_mean == best_time ? " class=\"highlight-best\"" : ""
        
        global html *= """
                    <tr>
                        <td>$(row.case_idx)</td>
                        <td>$factual</td>
                        <td>$target</td>
                        <td$ecp_class>$ecp_time</td>
                        <td$esh_class>$esh_time</td>
                        <td$milp_class>$milp_str</td>
                        <td>$(row.ecp_success)/$(row.esh_success)/$(row.milp_success)</td>
                    </tr>
"""
    end
    
    global html *= """
                </tbody>
            </table>
            <p class="note">Success column shows ECP/ESH/MILP successful trials out of $n_trials each.</p>
"""
    
    # Key insights for this reduction
    global html *= """
            <div class="summary-box">
                <h4>Key Results for $(Int(reduction_pct))% Reduction:</h4>
"""
    
    if has_milp
        ecp_ratio = ecp_time_avg / milp_time_avg
        esh_ratio = esh_time_avg / milp_time_avg
        if ecp_ratio < 1.0
            global html *= "<p><strong>ECP:</strong> $(round(1.0/ecp_ratio, digits=2))x faster than MILP</p>"
        else
            global html *= "<p><strong>ECP:</strong> $(round(ecp_ratio, digits=2))x slower than MILP</p>"
        end
        if esh_ratio < 1.0
            global html *= "<p><strong>ESH:</strong> $(round(1.0/esh_ratio, digits=2))x faster than MILP</p>"
        else
            global html *= "<p><strong>ESH:</strong> $(round(esh_ratio, digits=2))x slower than MILP</p>"
        end
    end
    
    esh_vs_ecp = esh_time_avg / ecp_time_avg
    if esh_vs_ecp > 1.0
        pct = (esh_vs_ecp - 1.0) * 100
        global html *= "<p><strong>ESH vs ECP:</strong> ESH is $(round(pct, digits=1))% slower</p>"
    else
        pct = (1.0 - esh_vs_ecp) * 100
        global html *= "<p><strong>ESH vs ECP:</strong> ESH is $(round(pct, digits=1))% faster</p>"
    end
    
    global html *= """
            </div>
        </div>
"""
end

# Methodology and footer
html *= """
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
            <p><strong>Test Configuration:</strong> Each reduction target tested on $(length(unique(df[df.reduction_pct .== reduction_targets[1], :case_idx]))) diverse test cases</p>
        </div>

        <p class="note" style="margin-top: 30px; text-align: center;">
            Generated on $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        </p>
    </div>
</body>
</html>
"""

# Save HTML
output_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "multi_reduction_report.html")
open(output_path, "w") do io
    write(io, html)
end

println("✓ Comprehensive HTML report generated: $output_path")
println()

