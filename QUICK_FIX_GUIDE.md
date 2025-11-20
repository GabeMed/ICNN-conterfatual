# Quick Fix Guide - ESH Interior Point Bug

## The Bug

Interior point LP (Prob. MM) is missing gradient normalization, causing:
- ν measured in function value units (wrong)
- Should be r measured in geometric distance units (correct)
- Result: 8000+ unit "verification gap" even though points are valid

## The Fix (One Line!)

**File**: `/home/gabemed/purdue/ICNN-conterfatual/counterfactuals/algorithms/outer_approximation.jl`

**Line 589**: Change from:
```julia
linear_expr = @expression(model,
    constant + sum(grad_k[i] * x[i] for i in 1:n_features) + nu)
```

To:
```julia
grad_norm = norm(grad_k)
linear_expr = @expression(model,
    constant + sum(grad_k[i] * x[i] for i in 1:n_features) + nu * grad_norm)
```

## Why This Works

**Without normalization**:
- Constraint: `∇f·x + ν ≤ RHS`
- ν = unnormalized slack (depends on gradient magnitude)
- Large gradients → misleadingly large ν
- Result: ν and f(x) don't match

**With normalization**:
- Constraint: `∇f·x + r * ||∇f|| ≤ RHS`
- r = geometric radius (independent of gradient scale)
- This is the **Chebyshev center** formulation
- Result: r and f(x) are consistent

## Testing

Before fix:
```bash
cd examples
julia run_esh_debug.jl
# Expected: ν ≈ 1e-4, f(x) ≈ -8000, "verification gap" warning
```

After fix:
```bash
julia run_esh_debug.jl
# Expected: r ≈ 80-100, f(x) ≈ -8000, no warnings, consistent!
```

## Additional Fixes (Optional but Recommended)

### Fix 2: Tighten Initial Bound (line 536)
```julia
# From:
@constraint(model, nu <= y_target + epsilon)

# To:
@constraint(model, nu <= sqrt(Float64(length(x_start))))
```

### Fix 3: Update Tolerance (line 713)
```julia
# From:
tolerance::Float64=1e-5

# To:
tolerance::Float64=1e-8
```

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| LP variable | ν (unnormalized) | r (geometric radius) |
| Typical value | 1e-4 (tiny) | 80-100 (large) |
| f(x*) value | -8000 (deeply interior) | -8000 (deeply interior) |
| Consistency | ✗ Mismatch | ✓ Aligned |
| Formulation | Unnormalized slack | Chebyshev center |

## References

- Full analysis: `ESH_IMPLEMENTATION_ANALYSIS.md`
- Detailed fixes: `FIXES_REQUIRED.md`
- Math reference: `ESH_MATH_FORMULATION.md`
- Research summary: `RESEARCH_SUMMARY.md`
