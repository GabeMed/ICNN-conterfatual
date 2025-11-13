# Comprehensive Benchmark Results
## Multiple Cost Reduction Targets

**Date:** 2025-11-13  
**Methods:** ECP, ESH, MILP  
**Test Cases:** 5 per reduction target  
**Trials:** 3 per method per case

---

## Overall Performance Summary

| Reduction | ECP Time (s) | ESH Time (s) | MILP Time (s) | Best Method | ECP Success | ESH Success | MILP Success |
|-----------|--------------|--------------|---------------|-------------|-------------|-------------|--------------|
| **20%**   | 0.487        | 0.513        | **0.973**     | ECP         | 100%        | 100%        | 100%         |
| **30%**   | 1.700        | 1.965        | **1.253**     | MILP        | 100%        | 100%        | 100%         |
| **40%**   | 3.193        | 3.362        | **1.121**     | MILP        | 100%        | 100%        | 100%         |
| **50%**   | 0.939        | **0.881**    | 1.254         | ESH         | 100%        | 100%        | 100%         |
| **60%**   | N/A          | N/A          | **1.456**     | MILP        | 40%         | 60%         | 100%         |

---

## Detailed Results by Reduction Target

### 20% Cost Reduction
**Target:** Reduce cost from factual to 80% of original

| Method | Avg Time | Avg Iterations | Success Rate |
|--------|----------|----------------|--------------|
| ECP    | 0.487s   | 7.7            | 100%         |
| ESH    | 0.513s   | 7.3            | 100%         |
| MILP   | 0.973s   | -              | 100%         |

**Key Finding:** For small reductions (20%), ECP/ESH are ~2x faster than MILP

---

### 30% Cost Reduction
**Target:** Reduce cost from factual to 70% of original

| Method | Avg Time | Avg Iterations | Success Rate |
|--------|----------|----------------|--------------|
| ECP    | 1.700s   | 16.0           | 100%         |
| ESH    | 1.965s   | 17.3           | 100%         |
| MILP   | 1.253s   | -              | 100%         |

**Key Finding:** MILP becomes competitive, ECP ~1.4x slower than MILP

---

### 40% Cost Reduction
**Target:** Reduce cost from factual to 60% of original

| Method | Avg Time | Avg Iterations | Success Rate |
|--------|----------|----------------|--------------|
| ECP    | 3.193s   | 29.7           | 100%         |
| ESH    | 3.362s   | 31.0           | 100%         |
| MILP   | 1.121s   | -              | 100%         |

**Key Finding:** MILP significantly faster (~3x) as problem difficulty increases

---

### 50% Cost Reduction
**Target:** Reduce cost from factual to 50% of original

| Method | Avg Time | Avg Iterations | Success Rate |
|--------|----------|----------------|--------------|
| ECP    | 0.939s   | 13.7           | 100%         |
| ESH    | 0.881s   | 13.3           | 100%         |
| MILP   | 1.254s   | -              | 100%         |

**Key Finding:** Surprising! ESH fastest at 50% reduction, MILP slowest

---

### 60% Cost Reduction  
**Target:** Reduce cost from factual to 40% of original

| Method | Avg Time | Avg Iterations | Success Rate |
|--------|----------|----------------|--------------|
| ECP    | 3.965s   | 40.0           | 40%          |
| ESH    | 4.452s   | 40.0           | 60%          |
| MILP   | 1.456s   | -              | 100%         |

**Key Finding:** At extreme reductions (60%), OA methods struggle. MILP is most reliable.

---

## Key Insights

### Performance by Reduction Level

1. **Low Reductions (20%):** ECP/ESH are faster due to fewer iterations needed
2. **Moderate Reductions (30-40%):** MILP becomes dominant
3. **High Reductions (50%):** Methods converge in performance
4. **Extreme Reductions (60%):** MILP is only reliable method

### Method Comparison

**ECP (Extended Cutting Plane):**
- Fast for small reductions (<30%)
- Struggles at extreme reductions (60%)
- Average performance: 2.06s (excluding failures)

**ESH (Extended Supporting Hyperplane):**
- Similar to ECP with slight variations
- Best at 50% reduction
- Average performance: 2.24s (excluding failures)

**MILP (Mixed-Integer Linear Programming):**
- Consistent across all reduction targets
- Most reliable (100% success across all levels)
- Average performance: 1.21s
- **Winner for high-difficulty problems**

### Success Rates
- **ECP/ESH:** 100% success up to 50% reduction, degrades at 60%
- **MILP:** 100% success at all reduction levels

---

## Recommendations

1. **For real-time applications (<30% reduction):** Use ECP for speed
2. **For moderate reductions (30-50%):** Use MILP for best performance
3. **For extreme reductions (>50%):** Use MILP exclusively for reliability
4. **For guaranteed solutions:** Always use MILP

---

## Files Generated

1. **multi_reduction_report.html** - Interactive comprehensive report
2. **multiple_reductions_results.csv** - Raw data
3. **MULTI_REDUCTION_SUMMARY.md** - This summary

Open the HTML report for interactive visualization:
```
tmp/benchmark_results/multi_reduction_report.html
```

---

**Generated:** 2025-11-13  
**Total benchmark runtime:** ~15 minutes  
**Total test cases:** 25 (5 reductions × 5 cases each)  
**Total trials:** 225 (25 cases × 3 methods × 3 trials)

