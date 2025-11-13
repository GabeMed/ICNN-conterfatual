# Detailed Counterfactual Analysis Results

## Summary

**Total Test Cases:** 15 (5 reductions × 3 cases each)  
**Methods Tested:** ECP, ESH, MILP  
**Successful Counterfactuals:** 41/45 (91.1%)

---

## What This Report Shows

### For Each Test Case, You Can See:

1. **Factual Cost** - Original cost before any intervention
2. **Target Cost** - Desired cost based on reduction percentage
3. **Counterfactual Cost** - Actual cost achieved by each method
4. **Features Changed** - Which specific features were modified
5. **Feature Values** - Original → New (with Δ change)

### Example from the Data:

**Case 1, 20% Reduction:**
- **Factual Cost:** 99,675
- **Target Cost:** 79,740 (20% reduction)
- **Acceptable Range:** [79,580, 79,900]

**ECP Method:**
- **CF Cost Achieved:** 79,899 ✓ (within range)
- **Features Changed:** 5 out of 236
  - Feature 59: 3.155 → 0.000 (Δ -3.155)
  - Feature 49: 0.751 → 0.000 (Δ -0.751)
  - Feature 56: 0.851 → 0.000 (Δ -0.851)
  - Feature 62: 0.832 → 0.000 (Δ -0.832)
  - Feature 116: 2.261 → 0.336 (Δ -1.925)

---

## Key Findings

### Features Changed by Reduction Level

| Reduction | ECP (avg) | ESH (avg) | MILP (avg) |
|-----------|-----------|-----------|------------|
| 20%       | 4.3       | 4.3       | 4.3        |
| 30%       | 8.0       | 8.0       | 8.0        |
| 40%       | 15.0      | 14.7      | 14.7       |
| 50%       | 17.7      | 18.3      | 17.7       |
| 60%       | 23.0      | 23.0      | 23.0       |

**Observation:** More aggressive reductions require changing more features

### Cost Achievement

**All successful methods achieved costs within acceptable range:**
- Target ± ε (where ε = 0.2% of target)
- This validates that all methods respect the constraint

### Method Reliability by Reduction

| Reduction | ECP Success | ESH Success | MILP Success |
|-----------|-------------|-------------|--------------|
| 20%       | 3/3 (100%)  | 3/3 (100%)  | 3/3 (100%)   |
| 30%       | 3/3 (100%)  | 3/3 (100%)  | 3/3 (100%)   |
| 40%       | 3/3 (100%)  | 3/3 (100%)  | 3/3 (100%)   |
| 50%       | 3/3 (100%)  | 3/3 (100%)  | 3/3 (100%)   |
| 60%       | 1/3 (33%)   | 1/3 (33%)   | 3/3 (100%)   |

---

## Common Feature Patterns

### Frequently Modified Features Across Methods:

Most methods consistently modify these features:
- **Feature 59** - Often reduced to 0
- **Feature 49** - Often reduced to 0
- **Feature 56** - Often reduced to 0
- **Feature 116** - Partially reduced (not to 0)

**Interpretation:** These features have the highest impact on cost reduction

### Modification Strategies:

1. **Small Reductions (20-30%):** 
   - Modify 4-8 features
   - Focus on high-impact features
   - Minimal disruption

2. **Moderate Reductions (40-50%):**
   - Modify 15-18 features
   - Broader set of changes
   - More complex interventions

3. **Extreme Reductions (60%):**
   - Require 23+ feature changes
   - Only MILP reliably finds solutions
   - High complexity

---

## Files Generated

1. **counterfactual_details.html** - Full interactive report
   - Shows all cases with feature-level details
   - Includes which features changed and by how much
   - Visual cost comparisons

2. **detailed_counterfactuals.csv** - Raw data
   - All counterfactual information
   - JSON encoded feature changes
   - Ready for further analysis

---

## How to Use the HTML Report

Open `tmp/benchmark_results/counterfactual_details.html` to see:

1. **Overview Table:** Quick comparison of all results
2. **Detailed Cards:** Each case shows:
   - Cost comparison (Factual → Target → CF)
   - Method-specific results
   - Top 15 feature changes with values
   - Success/failure status

3. **Feature Changes:** For each successful counterfactual:
   - Original value
   - New value  
   - Delta (change amount)

---

## Practical Insights

### For Practitioners:

1. **Choose MILP for reliability** - 100% success across all reduction levels
2. **Use ECP/ESH for speed** - When you need quick results (<40% reduction)
3. **Monitor feature count** - More changes = more complex intervention
4. **Focus on high-impact features** - Features 59, 49, 56, 116 are key

### Understanding Counterfactuals:

- Negative Δ = Feature decreased
- Positive Δ = Feature increased
- Magnitude of Δ = Size of change needed
- Number of changes = Complexity of intervention

---

**Generated:** 2025-11-13  
**View Full Report:** `tmp/benchmark_results/counterfactual_details.html`

