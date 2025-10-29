# Complete Mathematical Formulation: Counterfactual Generation with ICNN

---

## Full Optimization Problem

### Decision Variables

- **x' ∈ ℝ^n**: counterfactual input
- **δ⁺ ∈ ℝ^n**: positive changes (δ⁺ᵢ ≥ 0)
- **δ⁻ ∈ ℝ^n**: negative changes (δ⁻ᵢ ≥ 0)
- **c ∈ {0,1}^n**: binary change indicators
- **z⁽⁰⁾ ∈ ℝ^{h₁}**: activations of layer 0
- **z⁽¹⁾ ∈ ℝ^{h₂}**: activations of layer 1
- **⋮**
- **z⁽ᴸ⁻¹⁾ ∈ ℝ^{hₗ}**: activations of layer L-1
- **y_pred ∈ ℝ**: predicted output

---

## Objective Function

```
minimize    Σⁿᵢ₌₁ (δ⁺ᵢ + δ⁻ᵢ) + λ Σⁿᵢ₌₁ cᵢ
```

---

## Constraints

### 1. Target Constraint

```
y_pred ≥ y_target - ε
y_pred ≤ y_target + ε
```

### 2. Feature Bounds

```
x'ᵢ ≥ x_min,i        ∀i = 1,...,n
x'ᵢ ≤ x_max,i        ∀i = 1,...,n
```

### 3. L1 Distance Decomposition

```
x'ᵢ - x*ᵢ = δ⁺ᵢ - δ⁻ᵢ        ∀i = 1,...,n
δ⁺ᵢ ≥ 0                      ∀i = 1,...,n
δ⁻ᵢ ≥ 0                      ∀i = 1,...,n
```

### 4. Sparsity Enforcement (Big-M)

```
δ⁺ᵢ ≤ M · cᵢ        ∀i = 1,...,n
δ⁻ᵢ ≤ M · cᵢ        ∀i = 1,...,n
cᵢ ∈ {0,1}          ∀i = 1,...,n
```

where M = max{x_max,i - x_min,i : i = 1,...,n}

### 5. Immutable Features (Optional)

```
x'ᵢ = x*ᵢ        ∀i ∈ I_immutable
```

### 6. Neural Network Constraints

#### Layer 0 (Input Layer)

For j = 1, ..., h₁:

```
z⁽⁰⁾ⱼ ≥ Σⁿᵢ₌₁ W⁽⁰⁾ⱼᵢ · x'ᵢ + b⁽⁰⁾ⱼ

z⁽⁰⁾ⱼ ≥ 0
```

Expanded form for all neurons in layer 0:

```
z⁽⁰⁾₁ ≥ W⁽⁰⁾₁₁ x'₁ + W⁽⁰⁾₁₂ x'₂ + ... + W⁽⁰⁾₁ₙ x'ₙ + b⁽⁰⁾₁
z⁽⁰⁾₁ ≥ 0

z⁽⁰⁾₂ ≥ W⁽⁰⁾₂₁ x'₁ + W⁽⁰⁾₂₂ x'₂ + ... + W⁽⁰⁾₂ₙ x'ₙ + b⁽⁰⁾₂
z⁽⁰⁾₂ ≥ 0

⋮

z⁽⁰⁾ₕ₁ ≥ W⁽⁰⁾ₕ₁₁ x'₁ + W⁽⁰⁾ₕ₁₂ x'₂ + ... + W⁽⁰⁾ₕ₁ₙ x'ₙ + b⁽⁰⁾ₕ₁
z⁽⁰⁾ₕ₁ ≥ 0
```

**Total constraints for layer 0:** 2h₁

#### Layer 1 (First Hidden Layer)

For j = 1, ..., h₂:

```
z⁽¹⁾ⱼ ≥ Σʰ¹ₖ₌₁ W⁽¹⁾ⱼₖ · z⁽⁰⁾ₖ

z⁽¹⁾ⱼ ≥ 0
```

Expanded form for all neurons in layer 1:

```
z⁽¹⁾₁ ≥ W⁽¹⁾₁₁ z⁽⁰⁾₁ + W⁽¹⁾₁₂ z⁽⁰⁾₂ + ... + W⁽¹⁾₁ₕ₁ z⁽⁰⁾ₕ₁
z⁽¹⁾₁ ≥ 0

z⁽¹⁾₂ ≥ W⁽¹⁾₂₁ z⁽⁰⁾₁ + W⁽¹⁾₂₂ z⁽⁰⁾₂ + ... + W⁽¹⁾₂ₕ₁ z⁽⁰⁾ₕ₁
z⁽¹⁾₂ ≥ 0

⋮

z⁽¹⁾ₕ₂ ≥ W⁽¹⁾ₕ₂₁ z⁽⁰⁾₁ + W⁽¹⁾ₕ₂₂ z⁽⁰⁾₂ + ... + W⁽¹⁾ₕ₂ₕ₁ z⁽⁰⁾ₕ₁
z⁽¹⁾ₕ₂ ≥ 0
```

**Note:** W⁽¹⁾ⱼₖ ≥ 0 (fixed, non-negative weights from trained model)

**Total constraints for layer 1:** 2h₂

#### Layer l (General Hidden Layer, for l = 2, ..., L-1)

For j = 1, ..., h_{l+1}:

```
z⁽ˡ⁾ⱼ ≥ Σʰˡₖ₌₁ W⁽ˡ⁾ⱼₖ · z⁽ˡ⁻¹⁾ₖ

z⁽ˡ⁾ⱼ ≥ 0
```

**Note:** W⁽ˡ⁾ⱼₖ ≥ 0 for all l ≥ 1 (convexity constraint)

**Total constraints for layer l:** 2h_{l+1}

#### Output Layer (Layer L)

```
y_pred = Σʰᴸ⁻¹ₖ₌₁ W⁽ᴸ⁾ₖ · z⁽ᴸ⁻¹⁾ₖ
```

Expanded form:

```
y_pred = W⁽ᴸ⁾₁ z⁽ᴸ⁻¹⁾₁ + W⁽ᴸ⁾₂ z⁽ᴸ⁻¹⁾₂ + ... + W⁽ᴸ⁾ₕₗ₋₁ z⁽ᴸ⁻¹⁾ₕₗ₋₁
```

**Note:** W⁽ᴸ⁾ₖ ≥ 0 (convexity constraint)

**Total constraints for output:** 1

---

## Complete Problem Statement

```
minimize    Σⁿᵢ₌₁ (δ⁺ᵢ + δ⁻ᵢ) + λ Σⁿᵢ₌₁ cᵢ

subject to:

    # Target
    y_pred ≥ y_target - ε
    y_pred ≤ y_target + ε

    # Bounds
    x_min,i ≤ x'ᵢ ≤ x_max,i                     ∀i = 1,...,n

    # Distance
    x'ᵢ - x*ᵢ = δ⁺ᵢ - δ⁻ᵢ                        ∀i = 1,...,n
    δ⁺ᵢ ≥ 0                                     ∀i = 1,...,n
    δ⁻ᵢ ≥ 0                                     ∀i = 1,...,n

    # Sparsity
    δ⁺ᵢ ≤ M · cᵢ                                ∀i = 1,...,n
    δ⁻ᵢ ≤ M · cᵢ                                ∀i = 1,...,n
    cᵢ ∈ {0,1}                                  ∀i = 1,...,n

    # Immutable (optional)
    x'ᵢ = x*ᵢ                                   ∀i ∈ I_immutable

    # Neural Network - Layer 0
    z⁽⁰⁾ⱼ ≥ Σⁿᵢ₌₁ W⁽⁰⁾ⱼᵢ x'ᵢ + b⁽⁰⁾ⱼ            ∀j = 1,...,h₁
    z⁽⁰⁾ⱼ ≥ 0                                   ∀j = 1,...,h₁

    # Neural Network - Hidden Layers
    z⁽ˡ⁾ⱼ ≥ Σʰˡₖ₌₁ W⁽ˡ⁾ⱼₖ z⁽ˡ⁻¹⁾ₖ               ∀j = 1,...,h_{l+1}, ∀l = 1,...,L-1
    z⁽ˡ⁾ⱼ ≥ 0                                   ∀j = 1,...,h_{l+1}, ∀l = 1,...,L-1

    # Neural Network - Output
    y_pred = Σʰᴸ⁻¹ₖ₌₁ W⁽ᴸ⁾ₖ z⁽ᴸ⁻¹⁾ₖ

where:
    W⁽ˡ⁾ ≥ 0  for l = 1,...,L    (fixed from trained ICNN)
```

---

## Example: Two Hidden Layers (h₁=200, h₂=200)

### Architecture

```
Input: x' ∈ ℝⁿ (e.g., n=236)
    ↓
Layer 0: z⁽⁰⁾ ∈ ℝ²⁰⁰
    ↓
Layer 1: z⁽¹⁾ ∈ ℝ²⁰⁰
    ↓
Output: y_pred ∈ ℝ
```

### Constraints Count

| Constraint Type | Count |
|----------------|-------|
| Target | 2 |
| Bounds | 2n |
| Distance | n |
| Non-negativity | 2n |
| Sparsity | 2n |
| Binary | n |
| Immutable | |I_immutable| |
| Layer 0 ReLU | 2 × 200 = 400 |
| Layer 1 ReLU | 2 × 200 = 400 |
| Output | 1 |
| **Total** | **803 + 8n + |I_immutable|** |

For n=236: **2,691 constraints**

### Variables Count

| Variable Type | Count |
|--------------|-------|
| x' | n |
| δ⁺ | n |
| δ⁻ | n |
| c | n (binary) |
| z⁽⁰⁾ | 200 |
| z⁽¹⁾ | 200 |
| y_pred | 1 |
| **Total Continuous** | **3n + 401** |
| **Total Binary** | **n** |

For n=236: **1,109 continuous + 236 binary = 1,345 variables**

---

## ReLU Linearization Explanation

The ReLU activation function:

```
ReLU(a) = max(0, a)
```

is nonlinear but can be exactly represented by the epigraph formulation:

```
z ≥ a
z ≥ 0
```

This is valid because:
- If a ≥ 0: minimization will set z = a (to minimize objective)
- If a < 0: z must be ≥ 0, so z = 0

The optimization naturally enforces z = max(0, a) without requiring binary variables for ReLU.

---

## Problem Classification

**Type:** Mixed-Integer Linear Program (MILP)

**Complexity:** NP-hard

**Variables:**
- Continuous: O(n + Σₗ hₗ)
- Binary: O(n)

**Constraints:**
- Linear: O(n + Σₗ hₗ)

---

## Parameters Summary

| Symbol | Description | Example Value |
|--------|-------------|---------------|
| n | Input dimension | 236 |
| x* | Factual input | Given data |
| y_target | Target output | 0.85 |
| ε | Tolerance | 0.01 |
| λ | Sparsity weight | 0.1 |
| M | Big-M | 1.0 (for normalized data) |
| h₁, h₂, ... | Hidden layer sizes | [200, 200] |
| W⁽ˡ⁾, b⁽⁰⁾ | Network weights | From trained ICNN |

---

**Authors:** Gabriel Medeiros & Claude

**Date:** 2025-10-29
