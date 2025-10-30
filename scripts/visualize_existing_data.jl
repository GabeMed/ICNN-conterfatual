"""
Visualize existing DC-OPF dataset without regenerating.

Quick visualization script for already generated BSON files.

Usage:
    julia scripts/visualize_existing_data.jl [path_to_bson_file]

If no path provided, uses default: src/data/data_pglib_opf_case118_ieee.bson
"""

using Pkg
Pkg.activate(".")

using BSON
using Statistics
using Plots
using Printf

# Parse command line arguments
data_file = if length(ARGS) > 0
    ARGS[1]
else
    joinpath(@__DIR__, "../src/data/data_pglib_opf_case118_ieee.bson")
end

println("=" ^ 70)
println("DC-OPF Data Visualization")
println("=" ^ 70)
println("\n📂 Loading data from: $data_file")

if !isfile(data_file)
    error("""
    ❌ Data file not found: $data_file

    Please generate the data first by running:
        julia src/data/Generate_DCOPF.jl

    Or specify a different file path as argument:
        julia scripts/visualize_existing_data.jl path/to/your/data.bson
    """)
end

# Load the data
raw_data = BSON.load(data_file)

# Check if data is wrapped in "results" key (from Generate_DCOPF.jl)
# Try both symbol and string keys
if haskey(raw_data, :results)
    data = raw_data[:results]
elseif haskey(raw_data, "results")
    data = raw_data["results"]
else
    data = raw_data
end

# Extract data components
Demand = data["Demand"]
DispatchDC = data["DispatchDC"]
ObjDC = data["ObjDC"]
TimeDC = data["TimeDC"]
n_valid = data["n_valid"]

n_samples, n_features = size(Demand)
n_buses = n_features ÷ 2  # Half for P, half for Q

println("\n✅ Data loaded successfully!")
println("\n📊 Dataset Statistics:")
println("─" ^ 70)
println("Total samples: $n_valid")
println("Number of buses: $n_buses")
println("Number of generators: $(size(DispatchDC, 2))")
println("Feature dimension: $n_features ($(n_buses) buses × 2 [P,Q])")

# Separate P and Q demands
P_demand = Demand[:, 1:n_buses]
Q_demand = Demand[:, (n_buses+1):end]

# ============================================================================
# Quick Statistics
# ============================================================================

println("\n" * "=" ^ 70)
println("Quick Statistics")
println("=" ^ 70)

println("\n1️⃣  Objective Values (Costs):")
@printf("   Mean:   %12.2f\n", mean(ObjDC))
@printf("   Std:    %12.2f\n", std(ObjDC))
@printf("   Min:    %12.2f\n", minimum(ObjDC))
@printf("   Max:    %12.2f\n", maximum(ObjDC))
@printf("   Median: %12.2f\n", median(ObjDC))

println("\n2️⃣  Active Power (P) Demand:")
@printf("   Mean: %.4f ± %.4f\n", mean(P_demand), std(P_demand))
@printf("   Range: [%.4f, %.4f]\n", minimum(P_demand), maximum(P_demand))

println("\n3️⃣  Reactive Power (Q) Demand:")
@printf("   Mean: %.4f ± %.4f\n", mean(Q_demand), std(Q_demand))
@printf("   Range: [%.4f, %.4f]\n", minimum(Q_demand), maximum(Q_demand))

println("\n4️⃣  Solution Times:")
@printf("   Mean:   %.4f seconds\n", mean(TimeDC))
@printf("   Median: %.4f seconds\n", median(TimeDC))
@printf("   Max:    %.4f seconds\n", maximum(TimeDC))

# ============================================================================
# Data Quality Checks
# ============================================================================

println("\n" * "=" ^ 70)
println("Data Quality Checks")
println("=" ^ 70)

checks_passed = 0
total_checks = 4

print("\n1️⃣  NaN values: ")
if any(isnan.(Demand)) || any(isnan.(ObjDC))
    println("⚠️  FOUND - Data may have issues")
else
    println("✅ NONE")
    checks_passed += 1
end

print("2️⃣  Inf values: ")
if any(isinf.(Demand)) || any(isinf.(ObjDC))
    println("⚠️  FOUND - Data may have issues")
else
    println("✅ NONE")
    checks_passed += 1
end

print("3️⃣  Duplicate samples: ")
n_unique = size(unique(Demand, dims=1), 1)
if n_unique < n_samples
    println("⚠️  $(n_samples - n_unique) duplicates found")
else
    println("✅ NONE")
    checks_passed += 1
end

print("4️⃣  Objective values: ")
if all(ObjDC .> 0)
    println("✅ All positive (valid costs)")
    checks_passed += 1
else
    println("⚠️  Some non-positive values detected")
end

println("\n📊 Quality Score: $checks_passed/$total_checks checks passed")

# ============================================================================
# Visualization
# ============================================================================

println("\n" * "=" ^ 70)
println("Creating Visualizations")
println("=" ^ 70)

# Create output directory
viz_dir = joinpath(@__DIR__, "../tmp/dcopf_visualization")
mkpath(viz_dir)

println("\n🎨 Generating plots...")

# Plot 1: Objective Value Distribution
p1 = histogram(ObjDC, bins=50,
               xlabel="Objective Value (Cost)",
               ylabel="Frequency",
               title="Distribution of Optimal Costs",
               legend=false,
               color=:steelblue,
               size=(600, 400))
savefig(p1, joinpath(viz_dir, "objective_distribution.png"))
println("   ✅ objective_distribution.png")

# Plot 2: Cost vs Total Demand (scatter)
total_P = sum(P_demand, dims=2)[:]
p2 = scatter(total_P, ObjDC,
             xlabel="Total Active Power Demand",
             ylabel="Optimal Cost",
             title="Cost vs Total Demand",
             legend=false,
             alpha=0.5,
             markersize=3,
             color=:coral,
             size=(600, 400))
savefig(p2, joinpath(viz_dir, "cost_vs_demand.png"))
println("   ✅ cost_vs_demand.png")

# Plot 3: Demand Profiles Sample
sample_size = min(10, n_samples)
sample_indices = 1:sample_size
p3 = plot(layout=(2,1), size=(800, 600))
for i in sample_indices
    plot!(p3[1], 1:n_buses, P_demand[i, :],
          alpha=0.6, label=nothing, legend=false, color=:blue)
    plot!(p3[2], 1:n_buses, Q_demand[i, :],
          alpha=0.6, label=nothing, legend=false, color=:red)
end
xlabel!(p3[1], "Bus Index")
ylabel!(p3[1], "P Demand")
title!(p3[1], "Active Power Demand Profiles (first $sample_size samples)")
xlabel!(p3[2], "Bus Index")
ylabel!(p3[2], "Q Demand")
title!(p3[2], "Reactive Power Demand Profiles (first $sample_size samples)")
savefig(p3, joinpath(viz_dir, "demand_profiles.png"))
println("   ✅ demand_profiles.png")

# Plot 4: Generator Dispatch Statistics
n_generators = size(DispatchDC, 2)
gen_means = mean(DispatchDC, dims=1)[:]
gen_stds = std(DispatchDC, dims=1)[:]
p4 = bar(1:n_generators, gen_means,
         yerror=gen_stds,
         xlabel="Generator Index",
         ylabel="Average Dispatch",
         title="Average Generator Dispatch (± std)",
         legend=false,
         color=:teal,
         size=(800, 400))
savefig(p4, joinpath(viz_dir, "generator_dispatch.png"))
println("   ✅ generator_dispatch.png")

# Plot 5: Solution Time Distribution
p5 = histogram(TimeDC, bins=30,
               xlabel="Solution Time (seconds)",
               ylabel="Frequency",
               title="DC-OPF Solution Time Distribution",
               legend=false,
               color=:purple,
               size=(600, 400))
savefig(p5, joinpath(viz_dir, "solution_times.png"))
println("   ✅ solution_times.png")

# Plot 6: Heatmap of first 50 samples across all buses
n_show_samples = min(50, n_samples)
n_show_buses = min(50, n_buses)
p6 = heatmap(P_demand[1:n_show_samples, 1:n_show_buses]',
             xlabel="Sample Index",
             ylabel="Bus Index",
             title="Active Power Demand Heatmap (first $n_show_samples samples, $n_show_buses buses)",
             color=:viridis,
             size=(900, 600))
savefig(p6, joinpath(viz_dir, "demand_heatmap.png"))
println("   ✅ demand_heatmap.png")

# Combined summary plot
p_summary = plot(p1, p2, p4, p5,
                 layout=(2,2),
                 size=(1200, 900),
                 plot_title="DC-OPF Dataset Summary")
savefig(p_summary, joinpath(viz_dir, "summary.png"))
println("   ✅ summary.png")

# ============================================================================
# Sample Data Preview
# ============================================================================

println("\n" * "=" ^ 70)
println("Sample Data Preview")
println("=" ^ 70)

println("\n📋 First 5 samples:")
println("\n   Demand (first 5 features) | Objective | Time")
println("   " * "-" ^ 60)
for i in 1:min(5, n_samples)
    features_str = join([@sprintf("%.3f", Demand[i, j]) for j in 1:min(5, n_features)], ", ")
    @printf("   [%s, ...] | %10.2f | %.4fs\n", features_str, ObjDC[i], TimeDC[i])
end

# ============================================================================
# Convexity Assessment
# ============================================================================

println("\n" * "=" ^ 70)
println("Convexity Assessment")
println("=" ^ 70)

println("\n📐 Testing if objective appears convex in demand:")
println("   (Checking if higher total demand generally means higher cost)")

# Compute correlation between total demand and cost
total_demand = sum(Demand, dims=2)[:]
correlation = cor(total_demand, ObjDC)
@printf("\n   Correlation(Total Demand, Cost): %.4f\n", correlation)

if correlation > 0.7
    println("   ✅ Strong positive correlation - likely convex relationship")
elseif correlation > 0.4
    println("   ⚠️  Moderate correlation - relationship may be convex")
else
    println("   ⚠️  Weak correlation - check if data is appropriate for ICNN")
end

# ============================================================================
# Final Summary
# ============================================================================

println("\n" * "="^70)
println("✅ Visualization Complete!")
println("="^70)

println("\n📁 All visualizations saved to:")
println("   $viz_dir")

println("\n📊 Generated files:")
files = [
    "objective_distribution.png - Distribution of optimization costs",
    "cost_vs_demand.png - Cost as function of total demand",
    "demand_profiles.png - Sample demand patterns across buses",
    "generator_dispatch.png - Average generator outputs",
    "solution_times.png - Computational time distribution",
    "demand_heatmap.png - Demand heatmap visualization",
    "summary.png - Combined overview"
]
for (i, file) in enumerate(files)
    println("   $i. $file")
end

println("\n🎯 Recommended Actions:")
if checks_passed == total_checks
    println("   ✅ Data quality looks good!")
    println("   → Ready to train ICNN: julia examples/train_dcopf.jl")
else
    println("   ⚠️  Some quality issues detected")
    println("   → Review the warnings above")
    println("   → Consider regenerating data if needed")
end

if correlation < 0.4
    println("   ⚠️  Low correlation between demand and cost")
    println("   → May not be ideal for ICNN training")
    println("   → Check if DC-OPF formulation is correct")
end

println("\n" * "="^70)
println("📚 For more details, check the CLAUDE.md documentation")
println("="^70)
