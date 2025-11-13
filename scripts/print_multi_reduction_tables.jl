"""
Print simple tables for multiple reduction targets
Quick terminal visualization of all benchmark results
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Printf

# Read results
results_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "multiple_reductions_results.csv")

if !isfile(results_path)
    println("Error: Results not found at: $results_path")
    exit(1)
end

df = CSV.read(results_path, DataFrame)
reduction_targets = sort(unique(df.reduction_pct))

println("="^80)
println("BENCHMARK RESULTS: MULTIPLE REDUCTION TARGETS")
println("="^80)
println()

# Overall summary table
println("="^80)
println("Table 1: Performance Overview Across All Reduction Targets")
println("="^80)
println()
println(@sprintf("%-10s %12s %12s %12s %12s %12s", 
    "Reduction", "ECP Time", "ESH Time", "MILP Time", "ECP Iter", "ESH Iter"))
println("-"^80)

for reduction_pct in reduction_targets
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    
    ecp_avg = mean(skipmissing(df_subset.ecp_time_mean))
    esh_avg = mean(skipmissing(df_subset.esh_time_mean))
    ecp_iter = mean(skipmissing(df_subset.ecp_iter_mean))
    esh_iter = mean(skipmissing(df_subset.esh_iter_mean))
    
    milp_times = filter(!isnan, df_subset.milp_time_mean)
    milp_str = isempty(milp_times) ? "timeout" : @sprintf("%.3f", mean(milp_times))
    
    ecp_str = isnan(ecp_avg) ? "failed" : @sprintf("%.3f", ecp_avg)
    esh_str = isnan(esh_avg) ? "failed" : @sprintf("%.3f", esh_avg)
    ecp_iter_str = isnan(ecp_iter) ? "-" : @sprintf("%.1f", ecp_iter)
    esh_iter_str = isnan(esh_iter) ? "-" : @sprintf("%.1f", esh_iter)
    
    println(@sprintf("%-10s %12s %12s %12s %12s %12s",
        string(Int(reduction_pct), "%"), ecp_str, esh_str, milp_str, 
        ecp_iter_str, esh_iter_str))
end
println()

# Success rates
println("="^80)
println("Table 2: Success Rates by Reduction Target")
println("="^80)
println()
println(@sprintf("%-10s %15s %15s %15s", 
    "Reduction", "ECP Success", "ESH Success", "MILP Success"))
println("-"^80)

for reduction_pct in reduction_targets
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    total_trials = sum(df_subset.n_trials)
    
    ecp_rate = 100 * sum(df_subset.ecp_success) / total_trials
    esh_rate = 100 * sum(df_subset.esh_success) / total_trials
    milp_rate = 100 * sum(df_subset.milp_success) / total_trials
    
    println(@sprintf("%-10s %15.0f%% %15.0f%% %15.0f%%",
        string(Int(reduction_pct), "%"), ecp_rate, esh_rate, milp_rate))
end
println()

# Detailed results for each reduction
for reduction_pct in reduction_targets
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    n_cases = nrow(df_subset)
    
    println("="^80)
    println("$(Int(reduction_pct))% Cost Reduction - Detailed Results")
    println("="^80)
    println()
    
    # Summary stats
    ecp_avg = mean(skipmissing(df_subset.ecp_time_mean))
    esh_avg = mean(skipmissing(df_subset.esh_time_mean))
    milp_times = filter(!isnan, df_subset.milp_time_mean)
    
    println("Average Performance:")
    if !isnan(ecp_avg)
        println(@sprintf("  ECP:  %.3fs", ecp_avg))
    else
        println("  ECP:  Failed")
    end
    if !isnan(esh_avg)
        println(@sprintf("  ESH:  %.3fs", esh_avg))
    else
        println("  ESH:  Failed")
    end
    if !isempty(milp_times)
        println(@sprintf("  MILP: %.3fs", mean(milp_times)))
    else
        println("  MILP: Timeout")
    end
    println()
    
    # Per-case results
    println(@sprintf("%-6s %12s %12s %12s %12s", 
        "Case", "ECP Time", "ESH Time", "MILP Time", "Success"))
    println("-"^80)
    
    for i in 1:n_cases
        row = df_subset[i, :]
        ecp_str = isnan(row.ecp_time_mean) ? "failed" : @sprintf("%.3f", row.ecp_time_mean)
        esh_str = isnan(row.esh_time_mean) ? "failed" : @sprintf("%.3f", row.esh_time_mean)
        milp_str = isnan(row.milp_time_mean) ? "timeout" : @sprintf("%.3f", row.milp_time_mean)
        success_str = @sprintf("%d/%d/%d", row.ecp_success, row.esh_success, row.milp_success)
        
        println(@sprintf("%-6d %12s %12s %12s %12s",
            row.case_idx, ecp_str, esh_str, milp_str, success_str))
    end
    println()
end

# Key insights
println("="^80)
println("KEY INSIGHTS")
println("="^80)
println()

println("Best Method by Reduction Target:")
println()

for reduction_pct in reduction_targets
    df_subset = df[df.reduction_pct .== reduction_pct, :]
    
    ecp_avg = mean(skipmissing(df_subset.ecp_time_mean))
    esh_avg = mean(skipmissing(df_subset.esh_time_mean))
    milp_times = filter(!isnan, df_subset.milp_time_mean)
    
    times = Float64[]
    methods = String[]
    
    if !isnan(ecp_avg)
        push!(times, ecp_avg)
        push!(methods, "ECP")
    end
    if !isnan(esh_avg)
        push!(times, esh_avg)
        push!(methods, "ESH")
    end
    if !isempty(milp_times)
        push!(times, mean(milp_times))
        push!(methods, "MILP")
    end
    
    if !isempty(times)
        best_idx = argmin(times)
        best_method = methods[best_idx]
        best_time = times[best_idx]
        println(@sprintf("  %s reduction: %s (%.3fs)", 
            string(Int(reduction_pct), "%"), best_method, best_time))
    end
end

println()
println("="^80)
println("Report files:")
println("  - HTML: tmp/benchmark_results/multi_reduction_report.html")
println("  - CSV:  tmp/benchmark_results/multiple_reductions_results.csv")
println("  - MD:   tmp/benchmark_results/MULTI_REDUCTION_SUMMARY.md")
println("="^80)

