"""
Simple table-based visualization of benchmark results
Focuses on clear, concise presentation of key metrics
"""

using Pkg
Pkg.activate(".")

using CSV
using DataFrames
using Statistics
using Printf

# Read benchmark results
results_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "fair_benchmark_results.csv")

if !isfile(results_path)
    println("Error: Benchmark results not found at: $results_path")
    println("Run examples/benchmark_fair_comparison.jl first")
    exit(1)
end

df = CSV.read(results_path, DataFrame)
n_cases = nrow(df)

println("="^80)
println("BENCHMARK RESULTS SUMMARY")
println("="^80)
println()

# Experiment parameters
reduction_pct = (1.0 - mean(df.y_target ./ df.y_factual)) * 100
n_trials = df.n_trials[1]

println("Experiment Configuration:")
println("  Target reduction: $(round(reduction_pct, digits=1))%")
println("  Number of test cases: $n_cases")
println("  Trials per method: $n_trials")
println()

# Table 1: Average Performance Across All Cases
println("="^80)
println("Table 1: Average Performance (All Cases)")
println("="^80)
println()
println(@sprintf("%-10s %12s %12s %12s %20s", 
    "Method", "Success", "Time (s)", "Iterations", "vs MILP"))
println("-"^80)

ecp_time_avg = mean(df.ecp_time_mean)
esh_time_avg = mean(df.esh_time_mean)

ecp_iter_avg = mean(df.ecp_iter_mean)
esh_iter_avg = mean(df.esh_iter_mean)

# Success rates
total_trials = sum(df.n_trials)
ecp_success_rate = 100 * sum(df.ecp_success) / total_trials
esh_success_rate = 100 * sum(df.esh_success) / total_trials
milp_success_rate = 100 * sum(df.milp_success) / total_trials

# Check if MILP has any successful runs
if sum(df.milp_success) > 0
    milp_times = filter(!isnan, df.milp_time_mean)
    milp_time_avg = mean(milp_times)
    ecp_ratio = ecp_time_avg / milp_time_avg
    esh_ratio = esh_time_avg / milp_time_avg
    
    ecp_vs_milp = ecp_ratio < 1.0 ? @sprintf("%.2fx faster", 1.0/ecp_ratio) : @sprintf("%.2fx slower", ecp_ratio)
    esh_vs_milp = esh_ratio < 1.0 ? @sprintf("%.2fx faster", 1.0/esh_ratio) : @sprintf("%.2fx slower", esh_ratio)
    
    println(@sprintf("%-10s %11.0f%% %12.3f %12.1f %20s", 
        "ECP", ecp_success_rate, ecp_time_avg, ecp_iter_avg, ecp_vs_milp))
    println(@sprintf("%-10s %11.0f%% %12.3f %12.1f %20s", 
        "ESH", esh_success_rate, esh_time_avg, esh_iter_avg, esh_vs_milp))
    println(@sprintf("%-10s %11.0f%% %12.3f %12s %20s", 
        "MILP", milp_success_rate, milp_time_avg, "-", "baseline"))
else
    println(@sprintf("%-10s %11.0f%% %12.3f %12.1f %20s", 
        "ECP", ecp_success_rate, ecp_time_avg, ecp_iter_avg, "N/A"))
    println(@sprintf("%-10s %11.0f%% %12.3f %12.1f %20s", 
        "ESH", esh_success_rate, esh_time_avg, esh_iter_avg, "N/A"))
    println(@sprintf("%-10s %11.0f%% %12s %12s %20s", 
        "MILP", milp_success_rate, "timeout", "-", "-"))
end
println()

# Table 2: Detailed Per-Case Results
println("="^80)
println("Table 2: Detailed Results by Case")
println("="^80)
println()
println(@sprintf("%-6s %11s %11s %11s %10s %10s %10s", 
    "Case", "ECP Time", "ESH Time", "MILP Time", "ECP Iter", "ESH Iter", "Success"))
println("-"^80)

for i in 1:n_cases
    milp_str = isnan(df.milp_time_mean[i]) ? "timeout" : @sprintf("%.3f", df.milp_time_mean[i])
    success_str = @sprintf("%d/%d/%d", df.ecp_success[i], df.esh_success[i], df.milp_success[i])
    println(@sprintf("%-6d %11.3f %11.3f %11s %10.0f %10.0f %10s",
        i, df.ecp_time_mean[i], df.esh_time_mean[i], milp_str,
        df.ecp_iter_mean[i], df.esh_iter_mean[i], success_str))
end
println("Note: Success shows ECP/ESH/MILP successful trials")
println()

# Table 3: ESH vs ECP Comparison
println("="^80)
println("Table 3: ESH vs ECP Comparison")
println("="^80)
println()

esh_vs_ecp_time = esh_time_avg / ecp_time_avg
esh_vs_ecp_iter = esh_iter_avg / ecp_iter_avg
time_diff = esh_time_avg - ecp_time_avg
time_diff_pct = (esh_vs_ecp_time - 1.0) * 100

println(@sprintf("Average time: ECP=%.3fs, ESH=%.3fs", ecp_time_avg, esh_time_avg))
if time_diff > 0
    println(@sprintf("Result: ESH is %.1f%% slower (+%.3fs)", time_diff_pct, time_diff))
else
    println(@sprintf("Result: ESH is %.1f%% faster (%.3fs)", abs(time_diff_pct), abs(time_diff)))
end
println()
println(@sprintf("Average iterations: ECP=%.1f, ESH=%.1f (ratio: %.2fx)", 
    ecp_iter_avg, esh_iter_avg, esh_vs_ecp_iter))
println()

# Table 4: Statistical Summary
println("="^80)
println("Table 4: Statistical Summary (Across Cases)")
println("="^80)
println()
println(@sprintf("%-10s %10s %10s %10s %10s", "Method", "Mean", "Median", "Min", "Max"))
println("-"^80)

println(@sprintf("%-10s %10.3f %10.3f %10.3f %10.3f", 
    "ECP Time", mean(df.ecp_time_mean), median(df.ecp_time_mean),
    minimum(df.ecp_time_mean), maximum(df.ecp_time_mean)))

println(@sprintf("%-10s %10.3f %10.3f %10.3f %10.3f", 
    "ESH Time", mean(df.esh_time_mean), median(df.esh_time_mean),
    minimum(df.esh_time_mean), maximum(df.esh_time_mean)))

println(@sprintf("%-10s %10.1f %10.1f %10.1f %10.1f", 
    "ECP Iter", mean(df.ecp_iter_mean), median(df.ecp_iter_mean),
    minimum(df.ecp_iter_mean), maximum(df.ecp_iter_mean)))

println(@sprintf("%-10s %10.1f %10.1f %10.1f %10.1f", 
    "ESH Iter", mean(df.esh_iter_mean), median(df.esh_iter_mean),
    minimum(df.esh_iter_mean), maximum(df.esh_iter_mean)))
println()

# Save to text file
output_path = joinpath(@__DIR__, "..", "tmp", "benchmark_results", "results_table.txt")
open(output_path, "w") do io
    write(io, "="^80 * "\n")
    write(io, "BENCHMARK RESULTS SUMMARY\n")
    write(io, "="^80 * "\n\n")
    
    write(io, "Experiment: $(round(reduction_pct, digits=1))% cost reduction\n")
    write(io, "Cases: $n_cases, Trials per method: $n_trials\n\n")
    
    write(io, "="^80 * "\n")
    write(io, "Average Performance\n")
    write(io, "="^80 * "\n\n")
    write(io, @sprintf("%-10s %15s %15s\n", "Method", "Time (s)", "Iterations"))
    write(io, "-"^80 * "\n")
    write(io, @sprintf("%-10s %15.3f %15.1f\n", "ECP", ecp_time_avg, ecp_iter_avg))
    write(io, @sprintf("%-10s %15.3f %15.1f\n", "ESH", esh_time_avg, esh_iter_avg))
    write(io, "\n")
    
    write(io, "ESH vs ECP: ")
    if time_diff > 0
        write(io, @sprintf("ESH is %.1f%% slower\n", time_diff_pct))
    else
        write(io, @sprintf("ESH is %.1f%% faster\n", abs(time_diff_pct)))
    end
end

println("Results saved to: $output_path")
println()
println("="^80)

