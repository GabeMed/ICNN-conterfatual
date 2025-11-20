# Benchmark Results

## Main Files (Use These!)

### Data
- **`detailed_counterfactuals.csv`** - Complete benchmark data with all details
- **`multiple_reductions_results.csv`** - Alternative format (backup)

### Reports
- **`counterfactual_details.html`** - **Main interactive report** ⭐
  - Shows all reduction targets (20%, 30%, 40%, 50%, 60%)
  - Displays factual cost, counterfactual cost, and changed features
  - Organized by reduction with collapsible sections
  - **Open this file to view results!**

### Documentation
- **`COUNTERFACTUAL_SUMMARY.md`** - Summary of findings
- **`CASE_DIFFERENCES_EXPLAINED.md`** - Explains what varies between test cases
- **`MULTI_REDUCTION_SUMMARY.md`** - Performance across all reductions

---

## How to View Results

```bash
# Open the main report in browser
xdg-open tmp/benchmark_results/counterfactual_details.html
```

---

## What Was Tested

- **Methods:** ECP, ESH, MILP
- **Reduction Targets:** 20%, 30%, 40%, 50%, 60%
- **Test Cases:** 3 per reduction (15 total scenarios)
- **Total Trials:** 45 (15 cases × 3 methods)

---

## Key Findings

### Best Method by Reduction
- 20%: ECP (0.487s)
- 30%: MILP (1.253s)
- 40%: MILP (1.121s)
- 50%: ESH (0.881s)
- 60%: MILP only reliable (100% success)

### Success Rates
- **ECP & ESH:** 100% success up to 50%, drops to 33% at 60%
- **MILP:** 100% success at all reduction levels

---

## Scripts

Main script to regenerate benchmarks:
- `../../examples/benchmark_detailed_counterfactuals.jl`

Generate HTML report:
- `../../scripts/generate_counterfactual_report.jl`

