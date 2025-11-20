# Bug Fixes and Timing Enhancements Summary

**Date:** November 20, 2025

## Critical Bugs Fixed

### Issue: 30 out of 45 tests failing (67% failure rate for ECP/ESH)

**Root Cause:** Variable initialization order error in `outer_approximation.jl`

**Specific Problems:**
1. **Line 1009:** `start_time` used before definition
   - Code tried to use `start_time` in early exit path before it was initialized at line 1035
   - Caused immediate crash for all OA-based methods (ECP and ESH)

2. **Line 1124:** `timing_breakdown` dictionary used before initialization
   - Dictionary was defined at line 1147 but used at line 1124
   - Caused crashes when trying to record timing data

3. **Line 1062:** `interior_search_start` used for timing before `timing_breakdown` existed
   - Timing variables were out of order

**Fix Applied:**
```julia
# BEFORE (BROKEN):
# Check if already at target (line 1002-1031)
if already_satisfied
    early_exit_time = time() - start_time  # ❌ start_time undefined!
    ...
end
start_time = time()  # Defined AFTER use! (line 1035)

# AFTER (FIXED):
start_time = time()  # ✅ Define FIRST
# Initialize timing_breakdown dictionary BEFORE any use
timing_breakdown = Dict(...)
# Then check if already at target
if already_satisfied
    early_exit_time = time() - start_time  # ✅ Now works!
    ...
end
```

## Results After Fix

### Success Rates
| Method | Before | After | Improvement |
|--------|--------|-------|-------------|
| MILP   | 100%   | 100%  | Stable      |
| ECP    | 0%     | 86.7% | **+86.7%**  |
| ESH    | 0%     | 86.7% | **+86.7%**  |
| **Overall** | **33.3%** | **91.1%** | **+57.8%** |

### Performance Comparison (Successful Cases)

| Metric                    | MILP  | ECP   | ESH   | Best      |
|---------------------------|-------|-------|-------|-----------|
| Success Rate              | 100%  | 86.7% | 86.7% | MILP      |
| Avg Time (s)              | 1.306 | 1.965 | 1.743 | MILP      |
| Avg Iterations            | 1.0   | 18.5  | 19.0  | MILP      |
| Avg Features Changed      | 13.5  | 11.8  | 11.8  | ECP/ESH   |

**Key Insights:**
- **MILP:** Most reliable (100% success) and fastest, but changes more features
- **ECP/ESH:** Find sparser solutions (fewer features changed), good for interpretability
- **ESH vs ECP:** ESH slightly faster than ECP when both succeed

## Timing Enhancements

### 1. Enhanced Timing Breakdown Structure

Both MILP and OA now provide detailed timing with these components:

**MILP Timing:**
```julia
timing_breakdown => Dict(
    :total => <total_time>,
    :initial_eval => <evaluation_time>,
    :model_build => <build_time>,
    :mip_solve => <solve_time>,
    :result_extraction => <extraction_time>
)
```

**OA (ECP/ESH) Timing:**
```julia
timing_breakdown => Dict(
    :total => <total_time>,
    :model_build => <build_time>,
    :interior_point_search => <interior_search_time>,
    :total_milp_solve => <cumulative_milp_time>,
    :total_nn_eval => <cumulative_nn_eval_time>,
    :total_cut_generation => <cumulative_cut_time>,
    :total_bisection => <bisection_time>  # ESH only
)
```

### 2. Visual Enhancements in HTML Report

#### New: Performance Comparison Table
Shows side-by-side comparison of all methods with:
- Success rate
- Average time
- Average iterations
- Average features changed
- Success/total cases

Best values highlighted in green.

#### Enhanced: Timing Breakdown Visualization
- **Before:** Simple table with numbers
- **After:** Visual bars with colors showing proportion of time spent in each phase

Colors used:
- 🔵 Model Build: Blue (#3498db)
- 🟣 Interior Search: Purple (#9b59b6)
- 🔴 MILP Solving: Red (#e74c3c)
- 🟠 NN Evaluations: Orange (#f39c12)
- 🟢 Cut Generation: Green (#2ecc71)
- 🟦 Bisection: Teal (#16a085)

#### Enhanced: Iterations Display
- **Before:** Plain text "Iterations: 5"
- **After:** Badge format with color: `[5 iterations]`

#### Enhanced: Solve Time Display
- Monospace font for better readability
- Bold and larger size
- Stands out more in the report

### 3. New CSS Classes Added

```css
.comparison-table      /* Summary table styling */
.best-value           /* Highlights best performance (green) */
.worst-value          /* Highlights worst performance (red) */
.timing-visual        /* Container for visual timing breakdown */
.timing-row           /* Individual timing component row */
.timing-label         /* Component name */
.timing-bar-container /* Background bar */
.timing-bar-fill      /* Colored fill showing proportion */
.timing-value         /* Time value in seconds */
```

## Files Modified

### Core Algorithm Files
1. `counterfactuals/algorithms/outer_approximation.jl`
   - Fixed variable initialization order (3 bugs)
   - Timing structure already good, just fixed access order

2. `counterfactuals/algorithms/mip_counterfactual.jl`
   - Already had good timing structure
   - No changes needed

### Reporting Files
3. `scripts/generate_counterfactual_report.jl`
   - Added performance comparison summary table
   - Added visual timing bars with colors
   - Enhanced iterations and time display
   - Added 70+ lines of CSS for better styling
   - Added statistics calculation for summary table

4. `examples/benchmark_detailed_counterfactuals.jl`
   - No changes needed (already capturing all data correctly)

## Example: Timing Breakdown for a Typical Case

**Case: 20% reduction, ECP method**
```
Total Time: 5.098s
├─ Model Build:      0.063s  (1.2%)  🔵
├─ Interior Search:  0.000s  (0.0%)  🟣
├─ MILP Solving:     4.906s (96.2%)  🔴
├─ NN Evaluations:   0.045s  (0.9%)  🟠
└─ Cut Generation:   0.037s  (0.7%)  🟢

Iterations: 8
Features Changed: 5/236
```

**Case: 20% reduction, MILP method**
```
Total Time: 1.466s
├─ Initial Eval:     0.000s  (0.0%)
├─ Model Build:      0.574s (39.2%)  🔵
├─ MIP Solve:        0.889s (60.6%)  🔴
└─ Result Extract:   0.004s  (0.3%)  🟢

Iterations: 1
Features Changed: 5/236
```

## Testing

**Test Command:**
```bash
cd /home/gabemed/purdue/ICNN-conterfatual
julia --project=. examples/benchmark_detailed_counterfactuals.jl
julia --project=. scripts/generate_counterfactual_report.jl
```

**Test Results:**
- ✅ All 3 methods (MILP, ECP, ESH) now work correctly
- ✅ 41/45 tests successful (91.1%)
- ✅ Only 4 failures on most aggressive 60% reduction case
- ✅ Timing breakdowns display correctly in HTML
- ✅ Visual elements render properly

## Recommendations

1. **For maximum reliability:** Use MILP (100% success rate)
2. **For sparse solutions:** Use ECP or ESH (fewer features changed)
3. **For aggressive reductions (>50%):** Prefer MILP over OA methods
4. **For moderate reductions (<40%):** All methods work well

## Visualization Location

The enhanced HTML report is available at:
```
tmp/benchmark_results/counterfactual_details.html
```

Open in a web browser to see:
- Interactive collapse/expand sections
- Color-coded timing bars
- Comparison table with highlighting
- Detailed feature changes for each case

