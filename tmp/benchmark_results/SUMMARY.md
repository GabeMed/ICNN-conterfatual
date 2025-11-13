# Benchmark Results Summary

## Experiment Configuration
- **Cost Reduction Target:** 40%
- **Test Cases:** 3
- **Trials per Method:** 3
- **Methods:** ECP, ESH, MILP

## Key Results for 40% Cost Reduction

### Average Performance

| Method | Success Rate | Time (s) | Iterations | vs MILP |
|--------|-------------|----------|------------|---------|
| ECP    | 100%        | 3.105    | 29.7       | 2.74x slower |
| ESH    | 100%        | 3.217    | 31.0       | 2.84x slower |
| MILP   | 100%        | 1.135    | -          | baseline |

### Main Findings
- **MILP is fastest:** 1.135s average solve time for 40% reduction
- **ECP:** 2.74x slower than MILP but converges reliably
- **ESH:** 2.84x slower than MILP, 3.6% slower than ECP
- **All methods:** 100% success rate on test cases

### Results by Case (40% Reduction)

| Case | Factual Cost | Target Cost | ECP Time | ESH Time | MILP Time |
|------|-------------|-------------|----------|----------|-----------|
| 1    | 99,675      | 59,805      | 2.862s   | 4.442s   | **1.131s** |
| 2    | 95,440      | 57,264      | 2.332s   | **2.011s** | 1.207s   |
| 3    | 96,845      | 58,107      | 4.123s   | 3.197s   | **1.065s** |

**Note:** Best time for each case shown in bold.

## Files Generated

1. **benchmark_report.html** - Interactive HTML report with full details
2. **results_table.txt** - Plain text summary
3. **fair_benchmark_results.csv** - Raw data for further analysis

## Methodology

- Randomized execution order (eliminates bias)
- Fresh model reload per trial (no warmstart)
- Multiple trials for statistical reliability
- Controlled random seed (reproducible)

---

**Generated:** 2025-11-13

