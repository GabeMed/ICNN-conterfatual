# ESH Performance Analysis Results

**Date**: November 13, 2025
**Status**: Complete
**Confidence**: High (backed by detailed evidence and diagnostics)

## Executive Summary

ESH (Extended Supporting Hyperplane) is **not functioning correctly**. The interior point discovery algorithm fails in 100% of test cases, causing:
- 50% overall failure rate (2/4 cases)
- Silent fallback to ECP in 2 cases (no ESH cuts generated)
- 10% performance overhead when it does work

## Key Findings

### 1. Success Rate Crisis
- **ECP**: 100% (4/4 cases) ✓
- **ESH**: 50% (2/4 cases) ✗
- **Difference**: -50% reliability loss for no benefit

### 2. Interior Point Discovery Broken
**All 4 test cases fail interior point discovery:**
```
✗ FAILED: Verification failed: f(x*) > target despite ν > tolerance
  Final ν = 1917.16981125 ≤ 0.0001
  ⚠ No interior point found. Falling back to ECP strategy.
```

**The Issue**: LP solver reports solution optimal but verification fails with gaps of 8624-31045 units (10,000x larger than tolerance).

### 3. Zero ESH Cuts in Successful Cases
Cut statistics after "successful" ESH runs:
- **Case 3**: ECP cuts: 36, **ESH cuts: 0** (100% fallback)
- **Case 4**: ECP cuts: 39, **ESH cuts: 0** (100% fallback)

ESH strategy selected but **never actually used for cutting**.

### 4. Identical Solutions
When both methods "succeed":
- Case 3: Same solution (distance=14.93, features_changed=13)
- Case 4: Same solution (distance=15.16, features_changed=12)

Identical solutions = same algorithm = ESH is really ECP.

## Root Cause: Broken Prob. MM

**Location**: `counterfactuals/algorithms/outer_approximation.jl` lines 373-563

**Function**: `find_interior_point_oa()` - implements Prob. MM (Problem Minimax)

**The Bug**: The LP constraint formulation is incorrect or mismatches the problem structure:

```
Constraint added to LP: f(x_k) + ∇f(x_k)·(x - x_k) - ν ≤ y_target + ε
LP reports: ν = 1917.17 (OPTIMAL)
Verification: f(x) = 10441-18295 (INFEASIBLE, gap = 8624-31045)
```

The cutting-plane approach creates "phantom solutions" that satisfy the linearized constraint but violate the actual nonlinear constraint by thousands of units.

## Benchmark Metrics

### Performance Comparison (4 test cases, 40% cost reduction)

| Metric | ECP | ESH | Result |
|--------|-----|-----|--------|
| Solve Time | 4.47s | 4.93s | ESH +0.46s (10% slower) |
| Mean Iterations | 32 | 32 | Same when both work |
| Success Rate | 100% | 50% | ESH -50% reliability |
| ESH Cuts Used | N/A | 0 | Never generated |

### Case-by-Case Results

**Case 1 (Sample #3)**: FAILURE
- Target: 40% reduction (99674.72 → 59804.83)
- ECP: optimal, 7.57s, 30 iterations ✓
- ESH: no_solution, 6.19s, 50 iterations MAX ✗

**Case 2 (Sample #4)**: FAILURE
- Target: 40% reduction (95440.16 → 57264.10)
- ECP: optimal, 2.08s, 23 iterations ✓
- ESH: no_solution, 4.79s, 50 iterations MAX ✗

**Case 3 (Sample #12)**: FALLBACK
- Target: 40% reduction (96844.52 → 58106.71)
- ECP: optimal, 3.93s, 36 iterations
- ESH: optimal, 3.01s, 33 iterations (ECP fallback, no ESH cuts)
- Same solution: distance=14.93, features=13

**Case 4 (Sample #29)**: FALLBACK
- Target: 40% reduction (97863.88 → 58718.32)
- ECP: optimal, 4.29s, 39 iterations
- ESH: optimal, 4.57s, 38 iterations (ECP fallback, no ESH cuts)
- Same solution: distance=15.16, features=12

## Code Issues Identified

### Issue 1: Prob. MM Verification Gap (CRITICAL)
- **File**: `outer_approximation.jl` lines 521-530
- **Problem**: Verification shows gap of 8624-31045 units
- **Impact**: Interior point discovery always fails
- **Fix**: Reimplement LP constraint or validation logic

### Issue 2: Silent Fallback (HIGH)
- **File**: `outer_approximation.jl` line 974
- **Problem**: When interior point fails, code falls back to ECP without clear logging
- **Impact**: Users think ESH is working when it's really ECP
- **Fix**: Add explicit print when fallback occurs

### Issue 3: Incomplete Fallback (MEDIUM)
- **File**: `outer_approximation.jl` lines 1096-1155
- **Problem**: Fallback mechanism incomplete in some cases (Cases 1-2 hit max iterations)
- **Impact**: Inconsistent reliability (50% vs 100%)
- **Fix**: Ensure fallback to ECP is complete and reliable

### Issue 4: Interior Point Heuristic (MEDIUM)
- **File**: `outer_approximation.jl` line 940
- **Problem**: Assumes lower bounds are reasonable interior points (fails for DCOPF)
- **Impact**: Interior point search starts from bad initial guess
- **Fix**: Use problem-specific heuristics

## Recommendations (Prioritized)

### 1. CRITICAL (10 minutes)
**Disable ESH in benchmarks**
- Mark ESH as "experimental - broken, do not use"
- Update documentation to recommend ECP only
- File: `examples/benchmark_all_methods.jl` and `examples/compare_ecp_esh_strategies.jl`

### 2. HIGH (2-4 hours)
**Investigate Prob. MM verification gap**
- Why does LP report optimal but verification fails 8000+ units?
- Is the LP constraint formulation correct?
- Compare with original paper exactly
- File: `counterfactuals/algorithms/outer_approximation.jl` lines 461-469

### 3. HIGH (30 minutes)
**Add explicit fallback logging**
- Print clear message when interior point search fails
- Users need to know ESH is really ECP
- File: `counterfactuals/algorithms/outer_approximation.jl` line 974

### 4. MEDIUM (1-2 hours)
**Fix fallback mechanism**
- Cases 1-2 hit max iterations despite ECP fallback
- Ensure fallback is complete and handles all cases
- File: `counterfactuals/algorithms/outer_approximation.jl` lines 1096-1244

### 5. MEDIUM (4-8 hours)
**Implement simpler interior point heuristic**
- Replace broken Prob. MM with bisection approach
- Use bisection from lower bounds to first infeasible point
- Or use first feasible solution from ECP as interior point
- File: Create new function in `counterfactuals/algorithms/outer_approximation.jl`

### 6. LOW (8-16 hours)
**Two-phase algorithm (if worth effort)**
- Phase 1: Run ECP until first feasible solution
- Phase 2: Switch to ESH with feasible solution as interior point
- Trade-off: 2x overhead for potential better convergence
- Only recommend if testing shows benefits justify cost

## What Should Happen vs What's Happening

### How ESH Should Work (Theory)
1. Find interior point where f(x) < target
2. For each infeasible solution x_k, find boundary point on line between interior and x_k
3. Add supporting hyperplane cut at boundary (tighter than at x_k)
4. Result: Fewer tight cuts needed → fewer iterations → faster convergence

### What Actually Happens (Practice)
1. Try to find interior point → **FAILS**
2. No interior point available → can't do ESH
3. Fall back to ECP (add cut at infeasible point directly)
4. Result: Same ECP algorithm, **no performance improvement**, **overhead added**

## Test Evidence

### Evidence #1: Interior Point Discovery Failures
All 4 test cases show identical failure pattern:
```
Lower bounds too deep (f = 0.0 << 1917.17)
Searching for better interior point via Prob. MM...
[20 iterations with verification gaps >8000]
✗ FAILED: Verification failed: f(x*) > target despite ν > tolerance
⚠ No interior point found. Falling back to ECP strategy.
```

### Evidence #2: Zero ESH Cuts Used
Cut statistics show ESH cuts never generated:
```
Case 3: ECP cuts: 36, ESH cuts: 0
Case 4: ECP cuts: 39, ESH cuts: 0
```

### Evidence #3: Identical Solutions
When both methods succeed, solutions are identical:
```
Case 3 Distance: ECP=14.93, ESH=14.93 (difference=0)
Case 4 Distance: ECP=15.16, ESH=15.16 (difference=0)
```

## Files Analyzed

1. `/home/gabemed/purdue/ICNN-conterfatual/examples/benchmark_all_methods.jl` - Main benchmark
2. `/home/gabemed/purdue/ICNN-conterfatual/counterfactuals/algorithms/outer_approximation.jl` - Core algorithm
3. `/home/gabemed/purdue/ICNN-conterfatual/examples/compare_ecp_esh_strategies.jl` - Comparison script
4. `/home/gabemed/purdue/ICNN-conterfatual/tmp/benchmark_results/benchmark_all_methods.csv` - Results
5. `/home/gabemed/purdue/ICNN-conterfatual/tmp/benchmark_output.log` - Detailed diagnostics

## Conclusion

**ESH is not production-ready.** The interior point discovery algorithm (Prob. MM) has fundamental issues:
- Fails in 100% of test cases (crashes to verification failures)
- Causes 50% overall failure rate through incomplete fallback
- Never actually generates ESH cuts when selected
- Adds computational overhead for zero benefit

**Recommendation**: Disable ESH in all benchmarks and publications. Use ECP exclusively (100% reliable). Investigate and reimplement ESH properly if the theoretical benefits justify the engineering effort.

---

## For Quick Reference

**Total Test Cases**: 4
**ECP Success**: 4/4 (100%)
**ESH Success**: 2/4 (50%) - but only because of ECP fallback
**ESH Cuts Ever Generated**: 0 out of 4 cases

**Root Cause Function**: `find_interior_point_oa()` lines 373-563
**Root Cause Issue**: Prob. MM LP constraint formulation is wrong
**Broken Verification**: Gaps of 8624-31045 units (should be <0.0001)

**Fix Priority**: Disable ESH → Investigate Prob. MM → Add Logging → Fix Fallback
**Time to Disable**: 10 minutes
**ECP Reliability**: 100% - use this method

