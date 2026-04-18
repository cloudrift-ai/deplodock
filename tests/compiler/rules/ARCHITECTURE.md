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
├── test_decompose_rules.py      # decomposition rules (structural + correctness)
├── test_optimization_rules.py   # optimization rules (structural + correctness)
└── test_fusion_rules.py         # fusion rules (structural only — LoopOp not numpy-executable)
```

## Covered Rules

### Decomposition (`rules/decomposition/`)

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

### Optimization (`rules/optimization/`)

| Rule file | Op | Structural | Correctness |
|---|---|---|---|
| `002_insert_broadcast_indexmap.py` | `ElementwiseOp` (broadcast) | ✓ | ✓ (1D, scalar, 3D, RMSNorm chain) |

### Fusion (`rules/fusion/`)

The fusion pass is driven by lift-then-merge: each tensor op becomes a
trivial `LoopOp` (one lift rule per op), then adjacent LoopOp pairs
merge via a σ-based legality check. `test_fusion_rules.py` runs the
whole directory as a single pass; `test_merge_core.py` unit-tests the
σ solver and merge bookkeeping directly.

| Rule file | Op | Tested via |
|---|---|---|
| `001_lift_elementwise.py` | `ElementwiseOp` → `LoopOp` | `test_fusion_rules.py` (pass fixpoint) |
| `002_lift_reduce.py`      | `ReduceOp` → `LoopOp`      | `test_fusion_rules.py`, `test_merge_core.py::test_merge_reduce_then_elementwise` |
| `003_lift_indexmap.py`    | `IndexMapOp` → `LoopOp`    | `test_optimization_rules.py::test_matmul_with_transpose_fuses_to_one_kernel` (e2e) |
| `004_lift_gather.py`      | `GatherOp` → `LoopOp`      | `test_torch_ops.py::test_gather` |
| `005_merge_loop_ops.py`   | `LoopOp → LoopOp`          | `test_merge_core.py` (17 unit tests) + `test_fusion_rules.py` (fixpoint) |

Numerical correctness for lifted + merged kernels runs through the
numpy backends in three places:

- `test_fusion_rules.py::test_*_correctness` — runs the pre- and
  post-fusion graph through `NumpyBackend` (which uses `LoopOp.forward`
  post-fusion) and asserts outputs match.
- `test_e2e_accuracy.py` — full-pipeline coverage on toy shapes
  (pointwise, reduce, matmul, RMSNorm, softmax) via the `run_graph`
  fixture parameterized over `numpy` / `loop` / `cuda`.
- `test_block_accuracy.py` — real transformer block (TinyLlama layer 0
  with random weights, `seq_len=8` for the CPU lane, `seq_len=32` for
  the CUDA lane) compiled end-to-end and compared against PyTorch
  eager. The `_cpu` variant runs `LoopBackend` + CPU eager (always
  on, ~3s); the `_cuda` variants are gated by `@requires_cuda`.

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
