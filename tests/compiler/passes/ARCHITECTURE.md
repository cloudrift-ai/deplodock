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
tests/compiler/passes/
├── conftest.py                     # RecordingDump fixture for rule-fired assertions
├── test_decompose_rules.py         # decomposition rules (structural + correctness)
├── test_optimization_rules.py      # optimization rules (structural + correctness)
├── test_fusion_rules.py            # fusion rules (structural — LoopOp not numpy-executable)
├── test_matcher.py                 # Pattern matcher unit tests
├── test_matmul_rules.py            # matmul-specific rewrite rules
├── test_reduction_rules.py         # reduction-pattern rewrite rules
├── test_register_tile_rules.py     # register-tile lowering rules
├── test_partition_planner_rules.py # partition-planner pass
├── test_partition_planner_forks.py # partition-planner fork generation
├── test_launch_geometry_rules.py   # launch-geometry pass
├── test_masked_tile.py             # masked-tile pass (dynamic-shape boundary guard)
├── test_stage_inputs_classify.py   # Stage-input classifier
├── test_lowering_accuracy.py       # 040 / 060 / 070 + TMA end-to-end CUDA accuracy
├── test_lowering_blocked_gemm.py   # FN > 1 matmul accuracy (per-cell + replicator)
├── test_knob_pinning.py            # DEPLODOCK_KNOBS-pinned regression configs
├── test_tile_naming.py             # provenance-driven k_<op>_<suffix> kernel naming
└── test_pipeline_semantics.py      # full pass chain (decompose → opt → fuse) vs numpy
```

The `tests/compiler/conftest.py` exposes `matmul_graph(m, k, n)` — the
shared (m,k)@(k,n)→(m,n) graph builder used by the lowering / backend /
e2e tests, plus the `requires_cuda` skip marker and the `run_graph`
parametrized fixture.

## Covered Rules

### Decomposition (`passes/frontend/decomposition/`)

| Rule file          | Op                      | Structural | Correctness       |
|--------------------|-------------------------|------------|-------------------|
| `010_sdpa.py`      | `SdpaOp`                | ✓          | ✓                 |
| `020_silu.py`      | `ElementwiseOp("silu")` | ✓          | ✓                 |
| `030_pow.py`       | `ElementwiseOp("pow")`  | ✓          | ✓                 |
| `040_linear.py`    | `LinearOp`              | ✓          | ✓ (± bias)        |
| `070_matmul.py`    | `MatmulOp`              | ✓          | ✓ (± bias)        |
| `090_mean.py`      | `MeanOp`                | ✓          | ✓                 |
| `110_unsqueeze.py` | `UnsqueezeOp`           | —          | ✓ (dim=0, dim=-1) |
| `120_transpose.py` | `TransposeOp`           | —          | ✓                 |
| `130_reshape.py`   | `ReshapeOp`             | —          | ✓                 |
| `140_slice.py`     | `SliceOp`               | —          | ✓                 |
| `150_cat.py`       | `CatOp`                 | —          | ✓                 |

### Optimization (`passes/frontend/optimization/`)

| Rule file                          | Op                          | Structural | Correctness                       |
|------------------------------------|-----------------------------|------------|-----------------------------------|
| `002_insert_broadcast_indexmap.py` | `ElementwiseOp` (broadcast) | ✓          | ✓ (1D, scalar, 3D, RMSNorm chain) |

### Fusion (`passes/loop/lifting/` + `passes/loop/fusion/`)

Lifting wraps each surviving tensor primitive (elementwise / reduce /
indexmap / gather) in a trivial single-op `LoopOp`. Fusion then splices
adjacent `LoopOp` pairs by inlining the producer body at each consumer
`Load` that reads it. `test_fusion_rules.py` runs lifting followed by
fusion as a single pass; the splicer's behaviour is exercised
end-to-end there (no separate unit-test file — the old
`test_merge_core.py` was retired with `_merge_core.py`).

| Rule file                              | Op                         | Tested via                                                                         |
|----------------------------------------|----------------------------|------------------------------------------------------------------------------------|
| `loop/lifting/010_lift_elementwise.py` | `ElementwiseOp` → `LoopOp` | `test_fusion_rules.py` (pass fixpoint)                                             |
| `loop/lifting/020_lift_reduce.py`      | `ReduceOp` → `LoopOp`      | `test_fusion_rules.py::test_contraction_*`                                         |
| `loop/lifting/030_lift_indexmap.py`    | `IndexMapOp` → `LoopOp`    | `test_optimization_rules.py::test_matmul_with_transpose_fuses_to_one_kernel` (e2e) |
| `loop/lifting/040_lift_gather.py`      | `GatherOp` → `LoopOp`      | `test_torch_ops.py::test_gather`                                                   |
| `loop/fusion/010_merge_loop_ops.py`    | `LoopOp → LoopOp` (splice) | `test_fusion_rules.py` (fixpoint)                                                  |

Numerical correctness for lifted + merged kernels runs through the
numpy backends in three places:

- `test_fusion_rules.py::test_*_correctness` — runs the pre- and
  post-fusion graph through `NumpyBackend` (which uses `LoopOp.forward`
  post-fusion) and asserts outputs match.
- `tests/compiler/e2e/test_accuracy.py` — full-pipeline coverage on
  toy shapes (pointwise, reduce, matmul, RMSNorm, softmax) via the
  `run_graph` fixture parameterized over `numpy` / `loop` / `cuda`.
- `tests/compiler/e2e/test_block.py` — real transformer block (TinyLlama
  layer 0 with random weights, `seq_len=8` for the CPU lane, `seq_len=32`
  for the CUDA lane) compiled end-to-end and compared against PyTorch
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
