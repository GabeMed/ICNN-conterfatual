# OA Counterfactual Validation Experiments

This directory contains validation experiments to verify the correctness of the Outer Approximation (OA) counterfactual algorithm.

## Overview

The validation experiments use a "perturbation recovery" methodology to test if the OA algorithm can find counterfactuals that recover from known perturbations. This provides ground-truth validation with concrete, measurable success criteria.

## Experiment Scripts

### 1. Single Case Validation with Visualization

**File:** `validate_oa_perturbation.jl`

**Purpose:** Run a single test case with detailed output and convergence visualization

**Features:**
- Detailed step-by-step output
- Feature recovery analysis
- Constraint satisfaction verification
- 2D convergence plot showing trajectory in feature space

**Usage:**
```bash
julia validate_oa_perturbation.jl
```

**Output:**
- Console: Detailed validation report
- File: `../tmp/validation_convergence.png` - Convergence visualization

### 2. Multiple Cases Batch Validation

**File:** `validate_oa_multiple_cases.jl`

**Purpose:** Run comprehensive validation across multiple test cases with different perturbation magnitudes

**Features:**
- Tests 15 cases (5 per perturbation magnitude)
- Aggregate statistics
- Success rate analysis
- Performance metrics

**Usage:**
```bash
julia validate_oa_multiple_cases.jl
```

**Output:**
- Console: Aggregate validation statistics and assessment

## Validation Methodology

### Experimental Design

1. **Point1 (Original):** Select a point from training data with f(Point1) = cost1
2. **Point2 (Perturbed):** Perturb features 1-2 to create Point2 with f(Point2) = cost2
3. **Counterfactual Question:** "What changes in Point2 achieve f(x) ≤ cost1?"
4. **Expected:** Algorithm should find a solution that achieves the target cost

### Constraints

- **Mutable features:** 1-2 (can be changed by algorithm)
- **Immutable features:** 3-236 (locked to Point2 values)
- **Target:** f(x) ≤ cost1 ± epsilon

### Success Criteria

A test case passes if:
1. ✓ Counterfactual found (status = optimal)
2. ✓ Target cost achieved: |f(counterfactual) - target| ≤ epsilon
3. ✓ Immutability satisfied: features 3-236 unchanged

## Results Summary

**Overall:** 100% success rate (15/15 test cases)

### By Perturbation Magnitude

| Magnitude | Success Rate | Avg Recovery | Avg Time | Cost Achieved | Immutability |
|-----------|-------------|--------------|----------|---------------|--------------|
| Small (0.5, -0.3) | 5/5 (100%) | 90.4% | 0.18s | 5/5 ✓ | 5/5 ✓ |
| Medium (0.8, -0.6) | 5/5 (100%) | 103.1% | 0.0s | 5/5 ✓ | 5/5 ✓ |
| Large (1.2, -0.9) | 5/5 (100%) | 87.1% | 0.0s | 5/5 ✓ | 5/5 ✓ |

**Recovery Ratio:** Distance from Point1 / Perturbation magnitude
- Mean: 93.6%
- Median: 100.0%

### Interpretation

The algorithm successfully finds optimal counterfactuals that:
- Achieve target cost (100% of cases)
- Satisfy constraints (100% of cases)
- Converge quickly (<1 second average)

**Note on Recovery:** The algorithm may not fully recover Point1 exactly, because it finds the **optimal solution** according to the objective function (minimize distance + sparsity). Multiple solutions can achieve the same cost, and the algorithm finds the most efficient one (often changing fewer features).

## Visualization Example

The convergence plot shows:
- **Red X:** Starting point (Point2, perturbed)
- **Green Star:** Target point (Point1, original)
- **Blue Diamond:** Counterfactual solution found by OA
- **Bottom panel:** Distance to Point1 over iterations

Example from validation run:

```
Point1 (original):  [-1.2853,  0.3443]  f = 1.0552
Point2 (perturbed): [-0.4853, -0.2557]  f = 1.1214
Counterfactual:     [-0.4853,  0.1839]  f = 1.0486

Result: Changed only Feature 2 (more efficient than changing both)
```

## Customization

### Modify Test Parameters

Edit the scripts to customize:

```julia
# Number of test cases
n_test_cases = 5

# Perturbation magnitudes
perturbation_configs = [
    (p1=0.5f0, p2=-0.3f0, name="Small"),
    (p1=0.8f0, p2=-0.6f0, name="Medium"),
    (p1=1.2f0, p2=-0.9f0, name="Large"),
]

# OA parameters
oa_params = (
    epsilon = 0.05,
    sparsity_weight = 0.05,
    target_penalty_weight = 1000.0,
    x_bounds = (-10.0, 10.0),
    max_iterations = 50,
    # ...
)
```

### Random Seeds

- Single case: `Random.seed!(42)` (line ~68)
- Multiple cases: `Random.seed!(123)` (line ~32)

Change seeds to test different random samples.

## Requirements

### Dependencies

- Julia packages: ICNN.jl, JuMP.jl, Gurobi.jl, Plots.jl
- Trained FICNN model
- DCOPF dataset

### Required Files

- Model: `../tmp/dcopf_experiment/best_model.bson`
- Data: `../icnn/data/data_pglib_opf_case118_ieee.bson`

If files are missing, train a model first:
```bash
julia ../src/train_dcopf.jl
```

## Troubleshooting

### Error: Model not found

Train a model first:
```bash
cd /home/gabemed/purdue/ICNN-conterfatual
julia src/train_dcopf.jl
```

### Error: Gurobi license

Ensure Gurobi is properly licensed:
```bash
gurobi_cl --license
```

### Error: Invalid coefficient -Inf

Check `x_bounds` parameter - should be finite values, e.g., `(-10.0, 10.0)` not `(-Inf, Inf)` for standardized data.

## Further Reading

- **Full validation report:** `../VALIDATION_RESULTS.md`
- **OA algorithm:** `../counterfactuals/algorithms/outer_approximation.jl`
- **ICNN module:** `../icnn/ICNN.jl`

## Contact

For questions or issues with the validation experiments, please refer to the main project documentation or open an issue.

---

**Last Updated:** 2025-11-07
**Validation Status:** ✓ PASSED - Algorithm validated as correct
