"""
Generate HTML report with detailed counterfactual analysis
Shows: factual cost, counterfactual cost, and which features changed
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Printf
using JSON
using Dates

# Read detailed results
results_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "detailed_counterfactuals.csv")

if !isfile(results_path)
    println("Error: Detailed results not found at: $results_path")
    println("Run examples/benchmark_detailed_counterfactuals.jl first")
    exit(1)
end

df = CSV.read(results_path, DataFrame)
reduction_targets = sort(unique(df.reduction_pct))

println("Generating detailed counterfactual report...")
println("Total entries: $(nrow(df))")
println()

# HTML Header
html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Detailed Counterfactual Analysis</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1600px;
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
            margin-top: 35px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }
        h3 {
            color: #34495e;
            margin-top: 25px;
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 13px;
        }
        th {
            background-color: #3498db;
            color: white;
            padding: 10px 8px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 8px;
            border-bottom: 1px solid #ecf0f1;
        }
        tr:hover {
            background-color: #f8f9fa;
        }
        .success { color: #27ae60; font-weight: bold; }
        .failure { color: #e74c3c; font-weight: bold; }
        .info-box {
            background-color: #e8f4f8;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 20px 0;
        }
        .method-card {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .method-header {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
        }
        .features-table {
            font-size: 12px;
            margin-top: 10px;
        }
        .features-table th {
            background-color: #95a5a6;
        }
        .delta-positive {
            color: #27ae60;
            font-weight: bold;
        }
        .delta-negative {
            color: #e74c3c;
            font-weight: bold;
        }
        .note {
            font-style: italic;
            color: #7f8c8d;
            font-size: 13px;
            margin-top: 10px;
        }
        .cost-comparison {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }
        .cost-card {
            background-color: white;
            border: 2px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
        }
        .cost-card h4 {
            margin: 0 0 10px 0;
            color: #495057;
            font-size: 14px;
        }
        .cost-value {
            font-size: 24px;
            font-weight: bold;
            color: #2980b9;
        }
        .summary-section {
            background-color: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 20px;
            margin: 25px 0;
        }
        .reduction-header {
            background-color: #3498db;
            color: white;
            padding: 12px 20px;
            margin: -20px -20px 20px -20px;
            border-radius: 6px 6px 0 0;
            font-size: 18px;
            font-weight: bold;
        }
        .case-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }
        .case-box {
            background-color: white;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
        }
        .case-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 5px;
        }
        .method-result {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        .method-result:last-child {
            border-bottom: none;
        }
        .method-name {
            font-weight: 600;
            color: #34495e;
            min-width: 60px;
        }
        .result-metrics {
            display: flex;
            gap: 15px;
            font-size: 13px;
        }
        .metric {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .metric-label {
            font-size: 11px;
            color: #7f8c8d;
        }
        .metric-value {
            font-weight: bold;
            color: #2c3e50;
        }
        details {
            margin: 10px 0;
        }
        summary {
            cursor: pointer;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 5px;
            font-weight: 600;
            color: #2c3e50;
        }
        summary:hover {
            background-color: #d5dbdb;
        }
        .compact-table {
            font-size: 12px;
            margin: 10px 0;
        }
        .compact-table td {
            padding: 5px 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Detailed Counterfactual Analysis</h1>
        
        <div class="info-box">
            <p><strong>Analysis includes:</strong></p>
            <ul>
                <li>Factual Cost: Original cost before intervention</li>
                <li>Counterfactual Cost: Cost achieved by each method</li>
                <li>Changed Features: Which features were modified and their values</li>
                <li>Feature Changes: Original value → New value (Δ change)</li>
            </ul>
        </div>

        <h2>Overview: Results by Reduction Target</h2>
        <p class="note">Click on each reduction to expand/collapse details</p>
"""

# Group results by reduction
for reduction_pct in reduction_targets
    df_reduction = df[df.reduction_pct .== reduction_pct, :]
    cases_in_reduction = sort(unique(df_reduction.case_idx))
    
    global html *= """
        <details open>
            <summary>$(Int(reduction_pct))% Cost Reduction - $(length(cases_in_reduction)) Test Cases</summary>
            <div class="summary-section">
                <div class="case-grid">
"""
    
    # For each case in this reduction
    for case_idx in cases_in_reduction
        df_case = df_reduction[df_reduction.case_idx .== case_idx, :]
        
        if nrow(df_case) == 0
            continue
        end
        
        factual = df_case[1, :factual_cost]
        target = df_case[1, :target_cost]
        
        global html *= """
                    <div class="case-box">
                        <div class="case-title">Case $case_idx</div>
                        <p class="note" style="margin: 5px 0;">Factual: $(Int(round(factual))) → Target: $(Int(round(target)))</p>
"""
        
        # Show each method's result
        for method_row in eachrow(df_case)
            method = uppercase(method_row.method)
            
            if method_row.success && !ismissing(method_row.cf_cost)
                cf_cost = Int(round(method_row.cf_cost))
                n_features = Int(method_row.n_features_changed)
                time = @sprintf("%.3f", method_row.solve_time)
                
                global html *= """
                        <div class="method-result">
                            <span class="method-name">$method</span>
                            <div class="result-metrics">
                                <div class="metric">
                                    <span class="metric-label">CF Cost</span>
                                    <span class="metric-value success">$cf_cost</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Features</span>
                                    <span class="metric-value">$n_features</span>
                                </div>
                                <div class="metric">
                                    <span class="metric-label">Time</span>
                                    <span class="metric-value">$(time)s</span>
                                </div>
                            </div>
                        </div>
"""
            else
                global html *= """
                        <div class="method-result">
                            <span class="method-name">$method</span>
                            <div class="result-metrics">
                                <span class="failure">✗ Failed</span>
                            </div>
                        </div>
"""
            end
        end
        
        global html *= """
                    </div>
"""
    end
    
    global html *= """
                </div>
            </div>
        </details>
"""
end

# Detailed analysis by reduction and case
for reduction_pct in reduction_targets
    df_reduction = df[df.reduction_pct .== reduction_pct, :]
    cases = unique(df_reduction.case_idx)
    
    global html *= """
        <h2>$(Int(reduction_pct))% Cost Reduction - Detailed Analysis</h2>
"""
    
    for case_idx in cases
        df_case = df_reduction[df_reduction.case_idx .== case_idx, :]
        
        if nrow(df_case) == 0
            continue
        end
        
        factual_cost = df_case[1, :factual_cost]
        target_cost = df_case[1, :target_cost]
        epsilon = df_case[1, :epsilon]
        
        global html *= """
        <h3>Case $case_idx: $(Int(reduction_pct))% Reduction</h3>
        
        <div class="cost-comparison">
            <div class="cost-card">
                <h4>Factual Cost</h4>
                <div class="cost-value">$(Int(round(factual_cost)))</div>
            </div>
            <div class="cost-card">
                <h4>Target Cost</h4>
                <div class="cost-value">$(Int(round(target_cost)))</div>
                <p class="note">± $(Int(round(epsilon)))</p>
            </div>
            <div class="cost-card">
                <h4>Reduction</h4>
                <div class="cost-value">$(Int(reduction_pct))%</div>
                <p class="note">$(Int(round(factual_cost - target_cost))) units</p>
            </div>
        </div>
"""
        
        # Show each method's results
        for method_row in eachrow(df_case)
            method = uppercase(method_row.method)
            
            global html *= """
            <div class="method-card">
                <div class="method-header">$method Method</div>
"""
            
            if method_row.success && !ismissing(method_row.cf_cost)
                cf_cost = method_row.cf_cost
                n_changed = Int(method_row.n_features_changed)
                solve_time = method_row.solve_time
                
                deviation = cf_cost - target_cost
                
                global html *= """
                <p><strong>Status:</strong> <span class="success">✓ Success</span></p>
                <p><strong>Counterfactual Cost:</strong> $(Int(round(cf_cost))) 
                   (deviation: $(deviation > 0 ? "+" : "")$(round(deviation, digits=1)))</p>
                <p><strong>Features Changed:</strong> $n_changed / 236</p>
                <p><strong>Solve Time:</strong> $(round(solve_time, digits=3))s</p>
"""
                
                # Parse and display changed features
                try
                    changes_dict = JSON.parse(method_row.changed_features)
                    
                    if !isempty(changes_dict)
                        global html *= """
                <table class="features-table">
                    <thead>
                        <tr>
                            <th>Feature</th>
                            <th>Original</th>
                            <th>→</th>
                            <th>Counterfactual</th>
                            <th>Δ Change</th>
                        </tr>
                    </thead>
                    <tbody>
"""
                        
                        # Sort by absolute delta
                        sorted_features = sort(collect(changes_dict), 
                            by=x -> abs(x[2]["delta"]), rev=true)
                        
                        # Show top 15 changes
                        for (feat_idx_str, change) in sorted_features[1:min(15, end)]
                            feat_idx = feat_idx_str
                            orig = change["original"]
                            cf = change["counterfactual"]
                            delta = change["delta"]
                            
                            delta_class = delta > 0 ? "delta-positive" : "delta-negative"
                            delta_sign = delta > 0 ? "+" : ""
                            
                            global html *= """
                        <tr>
                            <td><strong>Feature $feat_idx</strong></td>
                            <td>$(round(orig, digits=4))</td>
                            <td>→</td>
                            <td>$(round(cf, digits=4))</td>
                            <td class="$delta_class">$(delta_sign)$(round(delta, digits=4))</td>
                        </tr>
"""
                        end
                        
                        global html *= """
                    </tbody>
                </table>
"""
                        
                        if length(sorted_features) > 15
                            global html *= """
                <p class="note">Showing top 15 of $n_changed changed features (sorted by magnitude)</p>
"""
                        end
                    end
                catch e
                    global html *= """
                <p class="note">Could not parse feature changes</p>
"""
                end
                
            else
                global html *= """
                <p><strong>Status:</strong> <span class="failure">✗ Failed</span></p>
                <p>Method did not find a valid counterfactual</p>
"""
            end
            
            global html *= """
            </div>
"""
        end
    end
end

# Footer
html *= """
        <div class="info-box" style="margin-top: 40px;">
            <h3 style="margin-top: 0;">Reading the Results</h3>
            <ul>
                <li><strong>Factual Cost:</strong> Original cost before any changes</li>
                <li><strong>Target Cost:</strong> Desired cost (Factual × (1 - Reduction%))</li>
                <li><strong>CF Cost:</strong> Actual cost achieved by the counterfactual</li>
                <li><strong>Feature Changes:</strong> Shows original → new value for each modified feature</li>
                <li><strong>Δ Change:</strong> Difference (positive = increase, negative = decrease)</li>
            </ul>
        </div>

        <p class="note" style="text-align: center; margin-top: 30px;">
            Generated on $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))
        </p>
    </div>
</body>
</html>
"""

# Save
output_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "counterfactual_details.html")
open(output_path, "w") do io
    write(io, html)
end

println("✓ Detailed counterfactual report generated: $output_path")
println()

