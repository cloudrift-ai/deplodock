# Compiler Architecture

## Three-Layer Shape

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 · Frontend (backend-agnostic)                                                                               │
│                                                                                                                      │
│  torch_trace.py ─→ Graph populated with frontend ops                                                                 │
│      ir/graph.py:    Tensor, Node[T_Op], Graph, Hints                                                                │
│      ir/base.py:     Op, InputOp, ConstantOp                                                                         │
│      ir/frontend_ir.py: Torch-captured ops (LinearOp, MatmulOp, SdpaOp, MeanOp, UnsqueezeOp, TransposeOp,               │
│                      ReshapeOp, SliceOp, CatOp)                                                                      │
│      ir/tensor_ir.py:   minimal IR survives decomposition (ElementwiseOp, ReduceOp, IndexMapOp, ...)                    │
│      ir/expr.py:     Expr AST + coord_expr helpers                                                                   │
│                                                                                                                      │
│  RULE: No GPU, no CUDA, no backend imports.                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 · Lowering → LoopProgram                                                                                    │
│                                                                                                                      │
│  pipeline.py: compile_graph(graph) -> LoopProgram                                                                    │
│    After fusion, ir/simplify.simplify_loop_op is applied to every LoopOp (const fold + clamp elimination).           │
│    LoopProgram (program/loop.py):                                                                                    │
│       LoopBuffer (shape, role) × N                                                                                   │
│       LoopLaunch (LoopOp + input/output buffer names) × N                                                            │
│       + graph_inputs/outputs/constants/constant_values                                                               │
│                                                                                                                      │
│  Each LoopOp (ir/loop_ir.py) is one GPU kernel as an SSA program:                                                       │
│      axes   (tuple[Axis, ...])        — named iteration variables                                                    │
│      inputs (tuple[Port, ...])        — per-input access patterns                                                    │
│      accumulators (tuple[Accumulator, ...]) — reduce accumulators                                                   │
│      body   (tuple[Stmt, ...])        — Assign | Update | Write | Select                                             │
│  Reductions = Accumulator(combine) + Update; writes are inline.                                                      │
│                                                                                                                      │
│  RULE: Operates on Graph + Loop IR only. No backend imports.                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 · Backends                                                                                                  │
│                                                                                                                      │
│  backend/base.py:  Backend ABC — compile(graph), run, benchmark                                                      │
│  backend/numpy/:   NumpyBackend — Graph interpreter (pre-fusion)                                                     │
│  backend/loop/:    LoopBackend  — LoopProgram interpreter (numpy)                                                    │
│  backend/cuda/:    CudaBackend  — LoopProgram → GpuProgram → nvcc                                                    │
│                                                                                                                      │
│  All three backends share:                                                                                           │
│    backend.compile(graph) → compiled                                                                                 │
│    backend.run(compiled, input_data=…) → ProgramResult (outputs=dict[name, ndarray], time_ms)                        │
│                                                                                                                      │
│  Codegen internals (CUDA):                                                                                           │
│    backend/kernel_codegen.py: GpuKernel → C source                                                                   │
│    backend/cuda/emit.py:      LoopProgram → GpuProgram                                                               │
│    backend/cuda/program.py:   CudaLaunch(GpuLaunch), nvcc, run                                                       │
│    backend/cuda/runner.py:    single-kernel compile + run harness                                                    │
│                                                                                                                      │
│  Program forms (shared across backends):                                                                             │
│      program/loop.py: LoopProgram + LoopBuffer + LoopLaunch                                                          │
│      program/gpu.py:  GpuProgram + GpuBuffer + GpuLaunch                                                             │
│  The imperative C-like kernel AST lives upstream in ir/kernel_ir.py (GpuKernel, Stmt, ArrayAccess, ...);                   │
│  the backend consumes it.                                                                                            │
│                                                                                                                      │
│  RULE: GPU specifics live here; everything above is portable.                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

See `ir/ARCHITECTURE.md` for the per-dialect breakdown of each IR level,
and `program/ARCHITECTURE.md` for the Program-form pairing.

## Canonical Data Flow

```
PyTorch module
   │  torch_trace.trace_module(...)
   ▼
Graph (Layer 1)
   │  backend.compile(graph)  — unified across numpy / loop / cuda
   │    ├─ NumpyBackend: wrap Graph; numpy walk in run
   │    ├─ LoopBackend:  compile_graph(graph) → LoopProgram; interpret in run
   │    └─ CudaBackend:  compile_graph(graph) → LoopProgram
   │                     compile_kernels(lp)  → GpuProgram
   ▼
LoopProgram (Layer 2: LoopBuffers + LoopLaunches)
   │  [CUDA path]
   │    for launch in loop_program.launches:
   │      emit.emit_kernel(launch, name, loop_program)  → GpuKernel
   │      kernel_codegen.emit_kernel(GpuKernel) → C source
   │      wrap as CudaLaunch, collect in GpuProgram
   ▼
GpuProgram (Layer 3)
   │  backend/cuda/program.generate_source(program)
   ▼
.cu  →  nvcc  →  GPU
```

## Structural LoopOp Cheat Sheet

SSA invariants are enforced by `LoopOp.__post_init__`: unique names,
defined-before-use, no forward references.

| Slot                                         | Type                                      | Used for                                                       |
|----------------------------------------------|-------------------------------------------|----------------------------------------------------------------|
| `Axis.name` / `extent` / `kind`              | `str` / `int` / `"free"                   | One iteration variable; reduce axes pair with accumulators     |
| `Port.index`                                 | `tuple[Expr, ...]`                        | Per-input-dim access pattern over axis Vars                    |
| `Accumulator.name` / `combine` / `init`      | `str` / `ElementwiseOp` / `Expr`          | Reduce accumulator declaration                                 |
| `Assign.op`                                  | `ElementwiseOp`                           | Pure SSA body op; ReduceOp is NOT valid here                   |
| `Update.target` / `value`                    | `str` / `str`                             | Fold value into an Accumulator                                 |
| `Write.output` / `index` / `value`           | `int` / `tuple[Expr, ...]` / `str`        | Inline store at a position                                     |
| `Select.branches[i].value` / `select`        | `str` / `Expr`                            | Coord-predicated SSA binding (replaces old Mux)                |
| `LoopOp.axes` / `inputs` / `accumulators` / `body` | 4 tuples                            | Iteration + access patterns + state + SSA statements           |
| `LoopLaunch.input_names`                     | `list[str]`                               | Per-Port external buffer name (program-level)                  |
| `LoopLaunch.output_name`                     | `str`                                     | External buffer written by this LoopOp                         |

Analogies (use in module/class docstrings):

- **Named iteration axes** like Halide ``Var``/``RVar`` or MLIR
  ``linalg.generic`` iterator types.
- **Accumulator state** like MLIR ``scf.for`` iter_args — the
  ``Update`` statement is the combine-and-write.
- **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store) for `LoopOp`.

## Shape inference

Every ``Op`` subclass implements ``infer_output_shape(input_shapes) → tuple``.
The compiler never stores shapes redundantly — shapes are derived when
needed by calling ``infer_output_shape`` on the op instance.

| Op type                                               | Rule                                                                                                                                                                         |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ``ElementwiseOp``                                     | ``broadcast_shapes(*input_shapes)`` — NumPy right-aligned broadcast. Inputs MUST be broadcast-compatible; matmul decomposition inserts unsqueeze IndexMapOps to ensure this. |
| ``ReduceOp``                                          | Drop the ``axis`` dim from the input shape.                                                                                                                                  |
| ``IndexMapOp``                                        | Returns ``self.out_shape`` — the output shape is part of the op definition because it cannot be derived from the coord_map + input shape (e.g., reshape ``(12,) → (3, 4)``). |
| ``TransposeOp``                                       | Permute the input shape by ``self.axes``.                                                                                                                                    |
| ``ReshapeOp``                                         | Returns ``self.shape``.                                                                                                                                                      |
| ``LinearOp`` / ``MatmulOp`` / ``SdpaOp`` / ``MeanOp`` | High-level ops; decomposed before shape inference runs on the kernel IR.                                                                                                     |
| ``LoopOp``                                            | ``infer_output_shape()`` returns the tuple of free-axis extents.                                                                                                             |

**Invariant**: after decomposition + optimization, every ``ElementwiseOp("mul")``
in the graph has broadcast-compatible inputs. Matmul ``A(..., M, K) × B(..., K, N)``
is decomposed with unsqueeze IndexMapOps (``A → A[..., M, K, 1]``,
``B → B[..., 1, K, N]``) so the mul broadcasts to ``(..., M, K, N)`` and
the reduce over ``K`` (axis -2) yields ``(..., M, N)``. No special-case
shape inference for contractions.

## Fusion policy (rules/fusion/)

Lift-then-merge, driven by the rewriter's fixpoint loop:

1. **Lift each tensor op into a trivial one-op ``LoopOp``** — one rule
   per tensor op (``001_lift_elementwise``, ``002_lift_reduce``,
   ``003_lift_indexmap``, ``004_lift_gather``). Each produces a kernel
   whose iteration space matches its op's natural shape, with Ports
   expressing the input access pattern directly as ``Expr``s over axis
   Vars. Reductions contribute one ``Accumulator`` accumulator and one
   ``Update`` statement; multi-source IndexMapOps (cat / concat)
   contribute a ``Select`` prelude choosing among per-source Ports.
2. **Merge adjacent ``LoopOp`` pairs** (``005_merge_loop_ops``). The
   grammar matches a ``LoopOp`` whose sole consumer is another
   ``LoopOp``. Merging is defined by a single substitution σ:
   - Solve ``writer.index[k] == reader.index[k]`` at each output dim
     for every producer axis. Supported forms: direct ``Var(a)``,
     ``Var(a) ± c``, and ``Literal(0)`` broadcast slots (no binding).
     Unsupported forms (multiplicative, modular, data-dependent) make
     the merge return ``None``.
   - Every unbound producer axis must be kind ``"reduce"`` — a leaking
     free axis would require replicating producer work per consumer
     slot, which we refuse.
   - The merged kernel must have at most one reduce axis (matches the
     current single-reduce CUDA backend — multi-reduce kernels feeding
     reductions across different data axes stay separate).
   - **Rank-growth guard**: refuse merging a reducing producer into a
     consumer whose ``Write`` has higher rank. The extra consumer free
     dims would make the producer's reduce body run once per iteration
     of each new dim — the MLP pathology where ``gate*up`` (reduce over
     hidden) would feed a rank-4 broadcast-then-mul before ``down_proj``
     and blow up per-layer work by ~3500× on Qwen-scale models.
     Rank-preserving reductions (keepdim) share the consumer's Write
     rank, so softmax-style merges are unaffected.
   - Axes and SSA names are renamed to avoid collisions; producer
     Ports are substituted through σ; the producer's connecting
     ``Write`` becomes ``Assign(bridge, ElementwiseOp("copy"), …)``
     that the consumer reads in place of the consumed Port.
3. **Fixpoint** — lift rules fire once per tensor op; merge fires
   repeatedly on adjacent LoopOps. The process terminates because
   every lift consumes one tensor op and every merge consumes two
   LoopOps.

The merge rule subsumes two passes the old pipeline had as separate
concerns:

- **Layout composition** (chained single-source IndexMapOps): merging
  two lifted IndexMap kernels composes their ``coord_map``s via σ, so
  no dedicated pre-pass is needed.
- **Absorption of identity-alias / linear-offset layouts**: transposes,
  keep-dim reductions, ``+``/``-`` offset slices (e.g. rotary
  ``rotate_half``) all fall out of σ-based binding — the consumer's
  reader index and the producer's writer index align affinely, so the
  producer's Port becomes the merged kernel's Port under the solved σ.

``optimization/002_insert_broadcast_indexmap.py`` survives because it
addresses a different concern — promoting implicit elementwise
broadcasts to explicit ``IndexMapOp``s — which merge does not handle.

## Codegen policy (backend/cuda/emit.py)

Walks the SSA body — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Maintains a `values: dict[str, Expr]` mapping Assign names to C expressions.

- `_emit_port_load(port, buf, src_shape, env) -> Expr`: emits `ArrayAccess(buffer, coord)` for a Port.index pattern evaluated in the axis environment. Select statements inside the SSA body are handled separately by `_emit_select`, which lowers `SelectBranch`es into a nested `Ternary` chain.
- `_emit_body`: unified entry for one kernel body. Calls `ir.loop_plan.analyze_kernel(loop, dollar_shapes, out_shape)` to produce a `KernelPlan`, then delegates to `_emit_plan`. The same plan powers `LoopProgram.pretty_print_launch` so the dump view mirrors the loop nest codegen emits.
- `_emit_plan`: walks `plan.steps` — each step is either `Inline` (a straight-line block of `Assign` / `Select` / `Write` / `Update`) or `Loop` (a K-loop over a reduce axis). Per-element port loads referenced inside a loop are deferred into that loop; other port loads happen upfront. Grid is 1D over flat free-axis extents; block is `(256, 1, 1)` for pure-elementwise plans, `(1, 1, 1)` for plans containing a `Loop` step. Empty bodies (copy kernels) are a pointwise subcase.
- Reductions (single `Loop` step) emit an accumulator `Accumulator` + `Update`; max reductions use `fmaxf` instead of `AugAssign`. Softmax-style cross-iteration patterns (reduce_max → sub+exp → reduce_sum → div) and contractions (mul → reduce_sum) fall out naturally from multiple `Loop` steps recomputing per-element values as needed.

The naive schedule is correctness-first — no shared memory, no async copies, no TMA, no vectorization. Performance work lives in follow-up commits.

## Numpy backend (`backend/numpy/backend.py`)

`NumpyBackend` derives from `Backend` (same ABC as `CudaBackend`).
Every `Op` subclass implements `forward(*inputs: np.ndarray) -> np.ndarray` — the
numpy equivalent of the tensor operation. `NumpyBackend.compile(graph)` stores the
graph; `run()` walks it in topological order, calling `forward` at each node,
reshaping outputs to match declared `node.output.shape`. No GPU required.

Covered ops: all elementwise functions, reductions, scans, gather/scatter,
transpose/reshape/unsqueeze/slice/cat, linear, matmul, SDPA, mean, plus
`IndexMapOp` and `LoopOp`. `LoopOp.forward` delegates to
`backend/loop/backend.py::execute_loop_op`, the same numpy interpreter that
`LoopBackend` uses — this lets `NumpyBackend` execute graphs *after* fusion,
so fusion rules can be validated numerically on CPU without CUDA.

## Testing

- `tests/compiler/test_ir.py`, `test_shape_inference.py`, `test_indexmap.py`, `test_backend_ir.py` — Layer 1 unit tests.
- `tests/compiler/test_kernel_op.py` — structural IR construction + invariant violations.
- `tests/compiler/test_lower.py` — `compile_graph` group-discovery + LoopOp assembly.
- `tests/compiler/test_emit.py` — recursive-descent codegen source-level assertions + on-GPU numerical checks.
- `tests/compiler/test_pipeline.py` — end-to-end on small synthetic graphs.
- `tests/compiler/test_torch_trace*.py`, `test_real_trace.py`, `test_hints.py` — tracer / hint coverage.
- `tests/compiler/test_torch_ops.py` — Op.forward() + numpy backend: per-op tests + torch cross-checks.
- `tests/compiler/rules/` — all rewrite rules (decomposition, optimization, fusion).

Full-model E2E (TinyLlama layer) comes back in a follow-up commit once decomposition of higher-level ops is ported into the new lowering.
