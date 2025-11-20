# Timing Analysis Updates - Detailed Breakdown

## Overview
Added comprehensive timing breakdowns to all counterfactual generation algorithms to clearly identify where time is spent during execution.

## Changes Made

### 1. MILP Algorithm (`counterfactuals/algorithms/mip_counterfactual.jl`)

**New Timing Components Tracked:**
- **Initial Evaluation**: Time to evaluate factual point
- **Model Build**: Time to construct JuMP model with neural network constraints
- **MIP Solve**: Time spent in Gurobi solver
- **Result Extraction**: Time to extract and process solution

**Console Output:**
```
✓ Found counterfactual!
  Total time: 5.234s
    - Model build: 0.123s
    - MIP solve: 5.089s
    - Extraction: 0.022s
```

**Return Dictionary:**
- Added `:iterations` field (always 1 for MILP - single-shot)
- Added `:timing_breakdown` dictionary with all components

### 2. Outer Approximation Algorithm (`counterfactuals/algorithms/outer_approximation.jl`)

**New Timing Components Tracked:**
- **Model Build**: Time to construct master MILP
- **Interior Point Search**: Time for Prob. MM (ESH only)
- **Total MILP Solve**: Cumulative time across all iterations
- **Total NN Evaluations**: Time evaluating neural network + gradients
- **Total Cut Generation**: Time generating OA/ESH cuts
- **Total Bisection**: Time finding boundary points (ESH only)

**Console Output:**
```
Timing Breakdown:
  Model build:     0.045s
  Interior search: 0.234s
  MILP solving:    1.456s
  NN evaluations:  0.123s
  Cut generation:  0.089s
    - Bisection:   0.034s
  Total:           1.947s
```

**Per-Iteration Console:**
```
Iter  1: UB=     Inf  obj=  12.3456  f(x)= 8234.56  err=234.56  ✗  [ECP] cuts=2 milp=0.123s
Iter  2: UB=  15.6789  obj=  15.6789  f(x)= 7989.23  err=  9.23  ✓  [ESH] cuts=3 milp=0.145s  [bisect=0.034s d_int=2.34 d_inf=5.67]
```

**Return Dictionary:**
- Enhanced `:iterations` field with actual iteration count
- Added `:timing_breakdown` dictionary with all components

### 3. Benchmark Script (`examples/benchmark_detailed_counterfactuals.jl`)

**Enhancements:**
- Captures timing breakdown from algorithm results
- Stores as JSON in CSV for detailed analysis
- Displays iterations and time in console during benchmark

**Console Output:**
```
✓ Success
    CF cost: 7989.23, Changed: 15 features
    Iterations: 3, Time: 1.947s
```

**CSV Columns Added:**
- `timing_breakdown`: JSON string with full timing data
- Enhanced `iterations` column (properly tracked)

### 4. HTML Report (`scripts/generate_counterfactual_report.jl`)

**New Visualizations:**

#### Overview Section:
- Added **Iters** metric showing iteration count
- Display format: `ECP: ✓ CF=7989, Features=15, Iters=3, Time=1.947s`

#### Detailed Analysis Section:
- **Iterations badge**: Prominent display of iteration count
- **Timing Breakdown Table**: Component-by-component time analysis

**MILP Timing Table:**
```
Component            Time (s)    % of Total
Model Build          0.123       2.4%
MIP Solve            5.089       97.2%
Result Extraction    0.022       0.4%
```

**OA Timing Table:**
```
Component            Time (s)    % of Total
Model Build          0.045       2.3%
Interior Search      0.234       12.0%
MILP Solving         1.456       74.8%
NN Evaluations       0.123       6.3%
Cut Generation       0.089       4.6%
  └─ Bisection       0.034       1.7%
```

## Key Insights Enabled

### 1. Algorithm Bottleneck Identification
- **MILP**: >95% time in solver (expected for large MILPs)
- **OA-ECP**: Majority in MILP solves (fast iterations)
- **OA-ESH**: Additional bisection overhead for tighter cuts

### 2. Iteration Efficiency
- Compare iterations needed: MILP=1 vs OA-ECP=3-5 vs OA-ESH=3-5
- Time per iteration visible in console logs
- Convergence patterns trackable

### 3. Method Comparison
With detailed timing, you can now answer:
- Where does MILP spend time? → Solver (large model)
- Where does OA spend time? → MILP solves (multiple smaller problems)
- Is ESH worth it? → Compare bisection overhead vs convergence speed
- Which is fastest overall? → See total time breakdown

### 4. Scalability Analysis
- Model build time shows encoding complexity
- MILP solve time shows problem difficulty
- Iteration count shows convergence behavior

## Usage

### Run Benchmark:
```bash
julia examples/benchmark_detailed_counterfactuals.jl
```

### Generate Report:
```bash
julia scripts/generate_counterfactual_report.jl
```

### View Results:
```bash
open tmp/benchmark_results/counterfactual_details.html
```

## Interpretation Guide

### MILP Method
- **Fast model build**: Good - encoding efficient
- **Slow MIP solve**: Expected - large problem
- **1 iteration**: Always - single-shot optimization

### OA-ECP Method
- **Fast model build**: Good - master MILP is small
- **Fast per-iteration**: Good - cuts tighten quickly
- **3-5 iterations**: Expected - convex problem converges fast
- **Low NN eval time**: Good - only evaluate at candidates

### OA-ESH Method
- **Similar to ECP**: Expected base behavior
- **Additional bisection time**: Cost of finding boundary points
- **Potentially fewer iterations**: Benefit of tighter cuts
- **Trade-off visible**: Compare bisection overhead vs iteration savings

## Benefits for Research

1. **Transparent Performance**: No more black-box timing
2. **Bottleneck Identification**: Know exactly where to optimize
3. **Method Comparison**: Fair apples-to-apples timing breakdown
4. **Publication-Ready**: Detailed tables and visualizations
5. **Reproducibility**: Complete timing data in CSV

## Notes

- All times in seconds with 3 decimal precision
- Percentages calculated relative to total time
- JSON format allows future analysis in Python/R
- HTML report maintains clean layout with expandable sections

