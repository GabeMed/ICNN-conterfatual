# Counterfactual Generation via FICNN Optimization

## Problem Formulation

Given a trained Fully Input Convex Neural Network (FICNN) $f: \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}$ and a factual input $\mathbf{x}_{\text{fact}} \in [0,1]^n$, we seek a counterfactual $\mathbf{x}_{\text{cf}} \in [0,1]^n$ that:

- Minimizes the distance to the factual: $\|\mathbf{x}_{\text{cf}} - \mathbf{x}_{\text{fact}}\|_1$
- Minimizes the number of changed features (sparsity)
- Satisfies the target classification constraint
- Respects domain constraints (immutability, one-hot groups)

## Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| $\mathbf{x}$ | $[0,1]^n$ | Counterfactual features |
| $\mathbf{x}_{\text{fact}}$ | $\mathbb{R}^n$ | Fixed factual features |
| $\boldsymbol{\delta}^+$ | $\mathbb{R}_+^n$ | Positive deviations |
| $\boldsymbol{\delta}^-$ | $\mathbb{R}_+^n$ | Negative deviations |
| $\mathbf{c}$ | $\{0,1\}^n$ | Binary indicators for changed features |
| $\mathbf{z}^{(l)}$ | $\mathbb{R}^{d_l}$ | FICNN hidden layer activations, $l=1,\ldots,L$ |
| $y_{\text{pred}}$ | $[0,1]$ | FICNN output (prediction) |

**Dimensions:**
- $n$ = number of features
- $L$ = number of FICNN layers
- $d_l$ = dimension of layer $l$

## FICNN Architecture Constraints

The FICNN $f(\mathbf{x}, y_{\text{target}})$ is convex in $y$ with skip connections from $\mathbf{x}$ to all layers. We encode it as a feedforward network with ReLU activations using the **epigraph formulation**.

### Layer 1 (First Hidden Layer)

For each neuron $j = 1, \ldots, d_1$:

$$
\begin{align}
a_j^{(1)} &= \sum_{k=1}^{n} W_{jk}^{(x,1)} x_k + \sum_{k=1}^{m} W_{jk}^{(y,1)} y_{\text{target},k} + b_j^{(x,1)} \\
z_j^{(1)} &\geq a_j^{(1)} \quad \text{(ReLU epigraph)} \\
z_j^{(1)} &\geq 0 \quad \text{(ReLU non-negativity)}
\end{align}
$$

**Weights:**
- $W^{(x,1)} \in \mathbb{R}^{d_1 \times n}$ - from input $\mathbf{x}$ to layer 1
- $W^{(y,1)} \in \mathbb{R}^{d_1 \times m}$ - from output $y_{\text{target}}$ to layer 1
- $b^{(x,1)} \in \mathbb{R}^{d_1}$ - biases
- $y_{\text{target}} \in \{0, 1\}^m$ - target output (fixed)

### Hidden Layers $l = 2, \ldots, L-1$

For each neuron $j = 1, \ldots, d_l$:

$$
\begin{align}
a_j^{(l)} &= \sum_{k=1}^{n} W_{jk}^{(x,l)} x_k + \sum_{k=1}^{d_{l-1}} W_{jk}^{(z,l-1)} z_k^{(l-1)} + \sum_{k=1}^{m} W_{jk}^{(y,l)} y_{\text{target},k} + b_j^{(x,l)} \\
z_j^{(l)} &\geq a_j^{(l)} \quad \text{(ReLU epigraph)} \\
z_j^{(l)} &\geq 0 \quad \text{(ReLU non-negativity)}
\end{align}
$$

where $W^{(z,l-1)} \in \mathbb{R}_+^{d_l \times d_{l-1}}$ are **non-negative** weights (enforced during training to maintain convexity).

### Output Layer $L$

For the scalar output:

$$
\begin{align}
y_{\text{pred}} &= \sum_{k=1}^{n} W_{k}^{(x,L)} x_k + \sum_{k=1}^{d_{L-1}} W_{k}^{(z,L-1)} z_k^{(L-1)} + \sum_{k=1}^{m} W_{k}^{(y,L)} y_{\text{target},k} + b^{(x,L)} \\
y_{\text{pred}} &\geq 0 \\
y_{\text{pred}} &\leq 1
\end{align}
$$

The output layer is **linear** (no ReLU) and clamped to $[0,1]$ for binary classification.

## Distance Decomposition

The L1 distance is decomposed into positive and negative deviations:

$$
\begin{align}
x_i - x_{\text{fact},i} &= \delta_i^+ - \delta_i^-, \quad \forall i = 1, \ldots, n \\
\delta_i^+, \delta_i^- &\geq 0, \quad \forall i = 1, \ldots, n
\end{align}
$$

Total L1 distance:

$$
d(\mathbf{x}, \mathbf{x}_{\text{fact}}) = \sum_{i=1}^{n} (\delta_i^+ + \delta_i^-)
$$

## Sparsity Constraints (Big-M)

Binary indicators $c_i \in \{0,1\}$ track whether feature $i$ changed:

$$
\begin{align}
\delta_i^+ &\leq M \cdot c_i, \quad \forall i = 1, \ldots, n \\
\delta_i^- &\leq M \cdot c_i, \quad \forall i = 1, \ldots, n
\end{align}
$$

where $M = 1.0$ (since all features are normalized to $[0,1]$).

Number of changed features:

$$
\text{sparsity} = \sum_{i=1}^{n} c_i
$$

## Classification Constraint (Margin)

To ensure the counterfactual is classified as the target class with confidence:

$$
\begin{align}
\text{If } y_{\text{target}} = 0: &\quad y_{\text{pred}} \leq 0.5 - \text{margin} \\
\text{If } y_{\text{target}} = 1: &\quad y_{\text{pred}} \geq 0.5 + \text{margin}
\end{align}
$$

where the margin is:

$$
\text{margin} = \frac{\alpha - 1}{2\alpha}
$$

For $\alpha = 2.0$: margin = 0.25 (prediction must be $\leq 0.25$ or $\geq 0.75$).

## Domain Constraints

### Immutable Features

For features that cannot be changed (e.g., race, sex, native_country), indexed by $\mathcal{I} \subseteq \{1, \ldots, n\}$:

$$
x_i = x_{\text{fact},i}, \quad \forall i \in \mathcal{I}
$$

### One-Hot Group Constraints

For each categorical feature group $G_k = \{i_1, i_2, \ldots, i_{|G_k|}\}$ (e.g., marital_status):

$$
\sum_{i \in G_k} x_i \leq 1
$$

This allows at most one category to be active (or none).

**Note on Categorical Features:** While one-hot encoded categorical features are stored as continuous variables in $[0,1]$ for compatibility with the trained FICNN, the optimization naturally pushes them toward binary values due to the sparsity constraint and one-hot group constraints.

## Objective Function

The complete optimization problem minimizes:

$$
\boxed{
\min_{\mathbf{x}, \boldsymbol{\delta}^+, \boldsymbol{\delta}^-, \mathbf{c}, \{\mathbf{z}^{(l)}\}, y_{\text{pred}}} \quad
\underbrace{\sum_{i=1}^{n} (\delta_i^+ + \delta_i^-)}_{\text{L1 distance}} +
\lambda \underbrace{\sum_{i=1}^{n} c_i}_{\text{sparsity}}
}
$$

**Parameters:**
- $\lambda$ = sparsity weight (e.g., 0.01, 0.1, 1.0)

## Complete MILP Formulation

$$
\begin{align}
\min \quad & \sum_{i=1}^{n} (\delta_i^+ + \delta_i^-) + \lambda \sum_{i=1}^{n} c_i \\
\text{s.t.} \quad & x_i - x_{\text{fact},i} = \delta_i^+ - \delta_i^-, && \forall i = 1, \ldots, n \\
& \delta_i^+ \leq M \cdot c_i, && \forall i = 1, \ldots, n \\
& \delta_i^- \leq M \cdot c_i, && \forall i = 1, \ldots, n \\
& x_i = x_{\text{fact},i}, && \forall i \in \mathcal{I} \\
& \sum_{i \in G_k} x_i \leq 1, && \forall k \\
& \text{FICNN constraints (epigraph ReLU)} \\
& \text{Classification constraint (margin)} \\
& 0 \leq x_i \leq 1, && \forall i = 1, \ldots, n \\
& \delta_i^+, \delta_i^- \geq 0, && \forall i = 1, \ldots, n \\
& c_i \in \{0, 1\}, && \forall i = 1, \ldots, n \\
& z_j^{(l)} \geq 0, && \forall j, l \\
& 0 \leq y_{\text{pred}} \leq 1
\end{align}
$$

## Model Statistics

For the Adult Income dataset with **81 features** (6 numeric + 75 categorical):

### Variables
- **Continuous**:
  - x: 81
  - δ⁺, δ⁻: 162
  - FICNN hidden: 200 + 200 + 1 = 401
  - **Total continuous: 644**

- **Binary**:
  - c (changed indicators): 81
  - **Total binary: 81**

- **Total variables: 725**

### Constraints
- Distance decomposition: 81
- Big-M: 162
- Immutables: ~20
- One-hot groups: 7
- FICNN ReLU: ~800
- Classification: 1
- **Total constraints: ~1071**

### Solver
- **Gurobi (MILP)**
- **Typical solve time: 0.5-2 seconds**

## Implementation Notes

1. **FICNN training**: All features (numeric and categorical) are Float32 in [0,1]. Categorical one-hot values are 0.0 or 1.0 but stored as continuous.

2. **Compatibility**: We keep x continuous in the optimization to match the trained FICNN. The combination of sparsity constraints and one-hot group constraints naturally pushes categorical features toward binary values.

3. **Normalization**: Features are normalized via Min-Max scaling fitted on training data only (no data leakage). Scaler is saved and applied to test/counterfactual inputs.

4. **Sparsity vs Distance**: Larger λ reduces number of changed features at the cost of larger L1 distance:
   - λ = 0.01: More features changed, smaller distance per feature
   - λ = 0.1: Medium trade-off (recommended)
   - λ = 1.0: Fewer features changed, larger distance per feature
   
   **Note**: Values of λ > 1 can cause the sparsity term to dominate, making all solutions converge to changing the minimum number of features regardless of distance magnitude.

## Code Mapping

### File: `counterfactuals/algorithms/optimization.jl`

#### Function: `build_ficnn_constraints!()` (lines 12-62)
Implements FICNN architecture constraints (Sections 3.1-3.3)

#### Function: `build_counterfactual_model_icnn()` (lines 68-105)
Builds the complete MILP model including:
- Decision variables
- Distance decomposition
- Sparsity constraints (Big-M)
- One-hot group constraints
- Objective function

#### Function: `set_factual_constraints!()` (lines 110-138)
Sets factual values and immutable features

#### Function: `add_classification_constraints!()` (lines 143-154)
Adds margin-based classification constraint

#### Function: `generate_counterfactual()` (lines 159-234)
Main interface that combines all components and solves with Gurobi
