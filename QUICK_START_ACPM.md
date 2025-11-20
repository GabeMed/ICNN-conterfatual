# Quick Start: ACPM with StandardNN

## Overview
Train a standard (non-convex) neural network on AC Optimal Power Flow data.

## Quick Commands

### 1. Generate ACPM Data (AC+DC)
```bash
cd /home/gabemed/purdue/ICNN-conterfatual

# Edit icnn/data/Generate_DCOPF.jl:
# Set mode = "AC+DC"
julia icnn/data/Generate_DCOPF.jl
```

### 2. Test the Pipeline
```bash
julia icnn/examples/test_acpm_pipeline.jl
```

### 3. Train StandardNN on ACPM
```bash
julia icnn/examples/train_acpm_standard_nn.jl
```

## Key Differences: DCPM vs ACPM

| Aspect | DCPM (Convex) | ACPM (Non-convex) |
|--------|---------------|-------------------|
| **Model** | FICNN | StandardNN |
| **Target** | ObjDC | ObjAC |
| **Convexity** | Yes (W ≥ 0) | No constraints |
| **Training flag** | `is_convex=true` | `is_convex=false` |
| **Data file** | `data_*_DC.bson` | `data_*_AC+DC.bson` |
| **Loader** | `load_dcopf_data` | `load_acpm_data` |

## Code Examples

### Load and Train
```julia
using Pkg
Pkg.activate(".")

include("icnn/ICNN.jl")
using .ICNN

# Load ACPM data
dataset = prepare_acpm_dataset(
    "icnn/data/data_pglib_opf_case118_ieee_AC+DC.bson"
)

# Create StandardNN (non-convex)
model = StandardNN(dataset.n_features, 1; hidden_sizes=[200, 200])

# Train (IMPORTANT: is_convex=false)
model = train!(
    model, dataset.X_train, dataset.Y_train, 50;
    is_convex=false,  # CRITICAL!
    X_test=dataset.X_test,
    y_test=dataset.Y_test
)

# Predict
y_pred = predict(model, dataset.X_test)
```

## File Locations

### Created Files
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/models/standard_nn.jl`
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/data/acpm_loader.jl`
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/examples/train_acpm_standard_nn.jl`
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/examples/test_acpm_pipeline.jl`

### Modified Files
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/ICNN.jl` (exports)
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/data/Generate_DCOPF.jl` (ACPM functions)
- `/home/gabemed/purdue/ICNN-conterfatual/icnn/models/ficnn.jl` (removed duplicate include)

## Common Mistakes

1. **Forgetting `is_convex=false`**: Always set this for StandardNN!
2. **Using wrong data loader**: Use `load_acpm_data` for ACPM, not `load_dcopf_data`
3. **Wrong target**: Use `ObjAC` (AC objectives) for ACPM, not `ObjDC`
4. **Data generation mode**: Set `mode="AC+DC"` in Generate_DCOPF.jl

## Troubleshooting

### "Data file not found"
```bash
# Generate data first
julia icnn/data/Generate_DCOPF.jl
```

### "Module not loading"
```julia
# Use this pattern:
include("icnn/ICNN.jl")
using .ICNN
# NOT: using ICNN
```

### "Poor training performance"
- Increase hidden layer sizes: `[200, 200]` → `[500, 500]`
- Try normalization: `normalize_method=:standardize`
- Increase epochs: `50` → `100`

## What's Next?

1. Compare FICNN (DCPM) vs StandardNN (ACPM) performance
2. Implement counterfactual generation for non-convex models
3. Explore hybrid approaches (start with DC, refine with AC)
