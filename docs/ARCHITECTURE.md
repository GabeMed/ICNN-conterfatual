# Architecture

## Overview

This repository implements counterfactual explanation algorithms using ICNN as a blackbox model.

## ICNN as Blackbox

The ICNN serves as the predictive model (blackbox) for counterfactual generation. Key properties:

### 1. Convexity in y
f(x, y) is **convex in y** for any fixed x. This is ensured by:
- W^(z) ≥ 0 (non-negative weights in hidden layers)
- ReLU activations (convex, non-decreasing)
- Post-update projection during training

### 2. Differentiable Predictions
The model supports:
- Forward pass: energy f(x, y)
- Inference: y* = argmin_y f(x, y) via PGD
- Gradients: ∂f/∂x, ∂f/∂y available

### 3. Training
Paper-compliant training via unrolled solver:
```julia
# Forward: minimize energy to get y*
y_pred = solve via PGD on f(x, y)

# Backward: AD tracks through PGD iterations
loss = MSE(y_pred, y_true)
gradients = differentiate through unrolled solver
```

## Module Structure

### src/ - ICNN Implementation

**Purpose**: Blackbox predictive model

**Components**:
- `models/`: ICNN architectures (FICNN, etc.)
- `training/`: Training procedures with convexity constraints
- `data/`: Dataset loaders
- `utils/`: I/O and visualization

**Interface**:
```julia
# Train model
model = FICNN(n_features, n_labels)
trained_model = train!(model, X, y, epochs)

# Make predictions
y_pred = predict(model, x, y_init)
energy = model(x, y)
```

### counterfactuals/ - Counterfactual Algorithms

**Purpose**: Generate counterfactual explanations

**Interface** (to be implemented):
```julia
# Generate counterfactual
method = WachterMethod(lambda=0.1)
x_cf = generate(method, model, x_original, target)

# Evaluate
validity = is_valid(x_cf, model, target)
distance = norm(x_cf - x_original)
```

**Planned Algorithms**:
- Wachter et al. (gradient-based)
- DiCE (diversity)
- Growing Spheres (search-based)
- Counterfactual Latent Uncertainty Explanations (CLUE)

### experiments/ - Benchmarks

**Purpose**: Compare counterfactual algorithms

Structure:
```
experiments/
├── benchmarks/
│   ├── compare_methods.jl
│   └── metrics.jl
├── datasets/
│   └── prepare_datasets.jl
└── results/
    └── (experiment outputs)
```

### test/ - Unit Tests

**Purpose**: Ensure correctness

Structure:
```
test/
├── runtests.jl
├── test_icnn.jl
├── test_training.jl
└── test_counterfactuals.jl
```

## Workflow

### 1. Train ICNN
```bash
julia --project=. examples/train_icnn.jl
```

Outputs:
- `examples/results/best_model.bson`
- `examples/results/metrics.json`
- `examples/results/training_log.csv`

### 2. Generate Counterfactuals (to be implemented)
```julia
include("src/ICNN.jl")
include("counterfactuals/Counterfactuals.jl")

model = load_model("examples/results/best_model.bson")
method = WachterMethod()
x_cf = generate(method, model, x_original, target=1.0)
```

### 3. Run Experiments (to be implemented)
```bash
julia --project=. experiments/benchmarks/compare_methods.jl
```

## Design Principles

1. **Separation of Concerns**
   - ICNN: predictive model only
   - Counterfactuals: explanation generation
   - Experiments: evaluation and comparison

2. **Modularity**
   - Each algorithm is independent
   - Common interface via `AbstractCounterfactualMethod`
   - Easy to add new algorithms

3. **Reproducibility**
   - Fixed random seeds
   - Saved models and configs
   - Detailed logging

4. **Paper Compliance**
   - ICNN follows Amos et al. ICML'17
   - Counterfactual algorithms follow original papers
   - Citations in code

## Next Steps

1. **Implement base counterfactual algorithm** (e.g., Wachter)
2. **Add evaluation metrics** (validity, proximity, sparsity)
3. **Create benchmark suite** comparing methods
4. **Add visualization tools** for explanations

