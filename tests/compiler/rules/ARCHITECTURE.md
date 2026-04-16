# Rules Test Architecture

## Purpose

Every graph transformation rule **must** be tested for numerical correctness,
not just structural properties. The pattern:

1. Build a graph containing the target op with concrete shapes.
2. Run the graph through `NumpyBackend` → **before** values.
3. Apply the decomposition rule via `Pass.apply()`.
4. Run the rewritten graph through `NumpyBackend` → **after** values.
5. Assert `before ≈ after` with `np.testing.assert_allclose`.

This catches semantic bugs that structural tests (checking which ops are
present) cannot: wrong axis in a reduction, swapped operands, missing
scale constant, incorrect coordinate mapping, etc.

## File Layout

```
tests/compiler/rules/
└── test_decompose_rules.py   # all decomposition rules (structural + correctness)
```

## Covered Rules

| Rule file | Op | Structural | Correctness |
|---|---|---|---|
| `001_decompose_sdpa.py` | `SdpaOp` | ✓ | ✓ |
| `002_decompose_silu.py` | `ElementwiseOp("silu")` | ✓ | ✓ |
| `003_decompose_pow.py` | `ElementwiseOp("pow")` | ✓ | ✓ |
| `004_decompose_linear.py` | `LinearOp` | ✓ | ✓ (± bias) |
| `005_decompose_matmul.py` | `MatmulOp` | ✓ | ✓ (± bias) |
| `007_decompose_mean.py` | `MeanOp` | ✓ | ✓ |
| `010_unsqueeze_to_indexmap.py` | `UnsqueezeOp` | — | ✓ (dim=0, dim=-1) |
| `011_transpose_to_indexmap.py` | `TransposeOp` | — | ✓ |
| `012_reshape_to_indexmap.py` | `ReshapeOp` | — | ✓ |
| `013_slice_to_indexmap.py` | `SliceOp` | — | ✓ |
| `014_cat_to_indexmap.py` | `CatOp` | — | ✓ |

## Adding a New Rule Test

When adding a new rewrite rule, add both test types in `test_decompose_rules.py`:

```python
def test_<op>_decomposes():
    """Structural: verify the original op is gone and expected ops are present."""
    ...

def test_<op>_correctness():
    """Numerical: verify before == after through the numpy backend."""
    g = _make_<op>_graph()
    inputs = {"x": rng.standard_normal(...).astype(np.float32)}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("<rule_file>.py")), inputs)
    _assert_close(before, after)
```

Use small concrete shapes (avoid symbolic dims) so the numpy backend
can execute the graph. `IndexMapOp.forward` iterates in Python, so keep
tensor sizes under ~1000 elements for fast tests.
