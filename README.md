# ICNN-Counterfactual

Counterfactual explanation algorithms using Input Convex Neural Networks (ICNN) as blackbox predictors.

## Structure

```
ICNN-conterfatual/
├── src/                           # ICNN implementation (blackbox model)
│   ├── ICNN.jl                   # Main module
│   ├── models/                   # ICNN architectures
│   │   ├── base.jl              # Abstract types
│   │   └── ficnn.jl             # Fully Input Convex NN
│   ├── training/                 # Training procedures
│   │   ├── trainer.jl           # Train loop with convexity projection
│   │   └── implicit_predict.jl  # Differentiable argmin (paper-compliant)
│   ├── data/                     # Dataset loaders
│   │   └── adult_income.jl
│   └── utils/                    # Utilities
│       ├── io.jl
│       └── visualization.jl
├── counterfactuals/              # Counterfactual generation algorithms
│   ├── Counterfactuals.jl       # Main module
│   └── algorithms/              # Algorithm implementations
│       └── (to be added)
├── examples/                     # Usage examples
│   ├── train_icnn.jl            # Train ICNN on dataset
│   └── results/                 # Training outputs
├── experiments/                  # Benchmarks and experiments
├── test/                        # Unit tests
├── docs/                        # Documentation
├── archive/                     # Old/reference files
└── Project.toml                 # Dependencies
```

## ICNN Implementation

The ICNN implementation follows **Amos, Xu, Kolter (ICML'17)**:

### Key Features

✅ **Convexity in y**: Architecture ensures f(x,y) is convex in y through:
- Non-negative weights W^(z) ≥ 0 in hidden layers
- ReLU activations (convex, non-decreasing)
- Post-update projection to maintain W^(z) ≥ 0

✅ **Inference via PGD**: Minimizes f(x,y) using projected gradient descent on [0,1]^p

✅ **Paper-compliant training**: Uses **unrolled solver differentiation**
- AD tracks through PGD iterations
- Alternative to implicit differentiation mentioned in paper Sec 5.1
- More stable than nested AD

### Architecture

```julia
# Forward pass (separate x and y processing)
z_0 = W_0^(x) * x + W_0^(y) * y + b_0
z_i = ReLU(W_i^(x) * x + W_i^(y) * y + W_i^(z) * z_{i-1})  for i > 0
```

Where W^(z) ≥ 0 ensures convexity in y.

## Quick Start

### Train ICNN

```julia
using Pkg
Pkg.activate(".")

include("examples/train_icnn.jl")
```

This will:
1. Load Adult Income dataset
2. Train FICNN model (5 epochs)
3. Save model and metrics to `examples/results/`

### Use Trained Model

```julia
include("src/ICNN.jl")
using .ICNN

# Load model
model = load_model("examples/results/best_model.bson")

# Make prediction
x = ... # input features
y_init = fill(0.5f0, 1, 1)
y_pred = predict(model, x, y_init)
```

## Differentiation Methods

The implementation supports multiple differentiation approaches for training:

| Method | Status | Paper-Compliant | Notes |
|--------|--------|----------------|-------|
| `"unrolled"` | ✅ Works | ✅ Yes | Unroll PGD, AD tracks through iterations |
| `"implicit"` | 🔄 Partial | ✅ Yes | Falls back to unrolled (needs param packing) |
| `"none"` | ❌ Broken | ❌ No | Nested AD doesn't work with Zygote |

**Recommended**: Use `diff_method="unrolled"` (default in examples).

## Paper Reference

```bibtex
@inproceedings{amos2017input,
  title={Input convex neural networks},
  author={Amos, Brandon and Xu, Lei and Kolter, J Zico},
  booktitle={International Conference on Machine Learning},
  pages={146--155},
  year={2017},
  organization={PMLR}
}
```

## Dependencies

```toml
[deps]
BSON = "fbb218c0-5317-5bc6-957e-2ee96dd4b1f0"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Flux = "587475ba-b1d2-4670-a9a8-e4f34e0d6d79"
ImplicitDifferentiation = "57b37032-215b-411f-ba27-7a5596611d64"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"
```

## Next Steps

- [ ] Implement counterfactual generation algorithms
- [ ] Add benchmarks comparing different methods
- [ ] Create experiments on various datasets
- [ ] Add comprehensive tests

## License

[Add license information]

