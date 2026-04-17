# Compiler Architecture

## Three-Layer Shape

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 · Frontend (backend-agnostic)                                                                               │
│                                                                                                                      │
│  torch_trace.py ─→ Graph populated with frontend ops                                                                 │
│      ir/graph.py:    Tensor, Node[T_Op], Graph, Hints                                                                │
│      ir/base.py:     Op, InputOp, ConstantOp                                                                         │
│      ir/frontend.py: Torch-captured ops (LinearOp, MatmulOp, SdpaOp, MeanOp, UnsqueezeOp, TransposeOp,               │
│                      ReshapeOp, SliceOp, CatOp)                                                                      │
│      ir/tensor.py:   minimal IR survives decomposition (ElementwiseOp, ReduceOp, IndexMapOp, ...)                    │
│      ir/expr.py:     Expr AST + coord_expr helpers                                                                   │
│                                                                                                                      │
│  RULE: No GPU, no CUDA, no backend imports.                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 · Lowering → LoopProgram                                                                                    │
│                                                                                                                      │
│  pipeline.py: compile_graph(graph) -> LoopProgram                                                                    │
│    LoopProgram (program/loop.py):                                                                                    │
│       LoopBuffer (shape, role) × N                                                                                   │
│       LoopLaunch (LoopOp + input/output buffer names) × N                                                            │
│       + graph_inputs/outputs/constants/constant_values                                                               │
│                                                                                                                      │
│  Each LoopOp (ir/loop.py) is one GPU kernel as an SSA program:                                                       │
│      axes   (tuple[Axis, ...])        — named iteration variables                                                    │
│      inputs (tuple[Port, ...])        — per-input access patterns                                                    │
│      locals (tuple[LocalBuffer, ...]) — thread-local accumulators                                                    │
│      body   (tuple[Stmt, ...])        — Assign | Update | Write | Select                                             │
│  Reductions = LocalBuffer(combine) + Update; writes are inline.                                                      │
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
│  The imperative C-like kernel AST lives upstream in ir/gpu.py (GpuKernel, Stmt, ArrayAccess, ...);                   │
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
| `LocalBuffer.name` / `init` / `combine`      | `str` / `Expr` / `ElementwiseOp \| None`  | Accumulator state (combine set) or plain scratch               |
| `Assign.op`                                  | `ElementwiseOp`                           | Pure SSA body op; ReduceOp is NOT valid here                   |
| `Update.target` / `value`                    | `str` / `str`                             | Fold value into a LocalBuffer accumulator                      |
| `Write.output` / `index` / `value`           | `int` / `tuple[Expr, ...]` / `str`        | Inline store at a position                                     |
| `Select.branches[i].value` / `select`        | `str` / `Expr`                            | Coord-predicated SSA binding (replaces old Mux)                |
| `LoopOp.axes` / `inputs` / `locals` / `body` | 4 tuples                                  | Iteration + access patterns + state + SSA statements           |
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

## Lowering policy (rules/fusion/assemble_kernels.py)

Backward-cone region growing via the rewriter:

1. **Forward BFS** from seed — collect all downstream fusable primitives
   (``ElementwiseOp``, ``ReduceOp``), enforcing that reductions form a
   single linear chain (no parallel reductions).
2. **Find output** — last node in topo order with no consumers in the
   forward set.
3. **Backward cone** — from output, absorb nodes whose consumers are
   all in the region. Trims nodes with side-outputs.
4. **Body** — topologically sorted ``Assign`` statements from the region.
5. ``InputOp`` / ``ConstantOp`` / ``IndexMapOp`` nodes emit no kernel —
   they survive as external buffers referenced by ``LoopLaunch.input_names``
   in ``Port`` order. Identity-alias IndexMapOps at boundaries are baked
   into the consuming ``Port.index`` Exprs during fusion (see step 6).
6. **Layout absorption covers permutations + identity aliases**: a
   ``coord_map`` entry must be either ``Literal(0)`` at a size-1 source
   dim or ``Var("out_coord_k")`` for any ``k`` (any order — transposes
   qualify), with every non-size-1 output dim covered by some
   placeholder. Broadcasts into fresh non-unit output dims and arithmetic
   reshapes (``//``, ``%``, ``+`` offsets, GQA ``/N``) are rejected by
   ``_same_rank`` and lowered as standalone kernels via
   ``003_wrap_indexmap``. Absorbed IndexMapOps are baked directly into
   the consuming ``Port.index`` Exprs; there is no separate ``indexmap``
   field on Port. Port indices align source-buffer dims to axis extents
   right-to-left rather than by positional left-pad, so extra iteration
   axes (e.g. reduce axes or absorbed unsqueezes) don't misalign the read.
7. **IndexMap composition** (``optimization/001_compose_indexmap``):
   before fusion, adjacent single-source IndexMapOp chains collapse into
   one op with a substituted ``coord_map``. This reduces double-hop
   chains (e.g. matmul's ``unsqueeze → broadcast``) to a single
   IndexMapOp, which rule 001 can then absorb into the consuming kernel's
   ``Port.index``.

## Codegen policy (backend/cuda/emit.py)

Walks the SSA body — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Maintains a `values: dict[str, Expr]` mapping Assign names to C expressions.

- `_emit_port_load(port, buf, src_shape, env) -> Expr`: emits `ArrayAccess(buffer, coord)` for a Port.index pattern evaluated in the axis environment. Select statements inside the SSA body are handled separately by `_emit_select`, which lowers `SelectBranch`es into a nested `Ternary` chain.
- `_emit_body`: unified entry for one kernel body. Calls `ir.loop_plan.analyze_kernel(loop, dollar_shapes, out_shape)` to produce a `KernelPlan`, then delegates to `_emit_plan`. The same plan powers `LoopProgram.pretty_print_launch` so the dump view mirrors the loop nest codegen emits.
- `_emit_plan`: walks `plan.steps` — each step is either `Inline` (a straight-line block of `Assign` / `Select` / `Write` / `Update`) or `Loop` (a K-loop over a reduce axis). Per-element port loads referenced inside a loop are deferred into that loop; other port loads happen upfront. Grid is 1D over flat free-axis extents; block is `(256, 1, 1)` for pure-elementwise plans, `(1, 1, 1)` for plans containing a `Loop` step. Empty bodies (copy kernels) are a pointwise subcase.
- Reductions (single `Loop` step) emit an accumulator `LocalBuffer` + `Update`; max reductions use `fmaxf` instead of `AugAssign`. Softmax-style cross-iteration patterns (reduce_max → sub+exp → reduce_sum → div) and contractions (mul → reduce_sum) fall out naturally from multiple `Loop` steps recomputing per-element values as needed.

The naive schedule is correctness-first — no shared memory, no async copies, no TMA, no vectorization. Performance work lives in follow-up commits.

## Numpy backend (`backend/numpy/backend.py`)

`NumpyBackend` derives from `Backend` (same ABC as `CudaBackend`).
Every `Op` subclass implements `forward(*inputs: np.ndarray) -> np.ndarray` — the
numpy equivalent of the tensor operation. `NumpyBackend.compile(graph)` stores the
graph; `run()` walks it in topological order, calling `forward` at each node,
reshaping outputs to match declared `node.output.shape`. No GPU required.

Covered ops: all elementwise functions, reductions, scans, gather/scatter,
transpose/reshape/unsqueeze/slice/cat, linear, matmul, SDPA, mean.
`IndexMapOp` and `LoopOp` raise `NotImplementedError` (structural IR).

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
