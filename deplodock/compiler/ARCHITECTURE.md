# Compiler Architecture

## Three-Layer Shape

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1 · Frontend (backend-agnostic)                                                                               │
│                                                                                                                      │
│  trace/torch.py ─→ Graph populated with frontend ops                                                                 │
│      ir/graph.py:    Tensor, Node[T_Op], Graph, Hints                                                                │
│      ir/base.py:     Op, InputOp, ConstantOp                                                                         │
│      ir/frontend/ir.py: Torch-captured ops (LinearOp, MatmulOp, SdpaOp, MeanOp, UnsqueezeOp, TransposeOp,               │
│                      ReshapeOp, SliceOp, CatOp)                                                                      │
│      ir/tensor/ir.py:   minimal IR survives decomposition (ElementwiseOp, ReduceOp, IndexMapOp, ...)                    │
│      ir/expr.py:     Expr AST + coord_expr helpers                                                                   │
│                                                                                                                      │
│  RULE: No GPU, no CUDA, no backend imports.                                                                          │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2 · Fusion (pipeline.compile_graph)                                                                           │
│                                                                                                                      │
│  pipeline.py: compile_graph(graph) -> Graph[LoopOp + InputOp + ConstantOp]                                            │
│    Runs decomposition → optimization → fusion passes from passes/.                                                   │
│    After fusion, ir/simplify.simplify_loop_op is applied to every LoopOp (const fold + clamp elimination).           │
│                                                                                                                      │
│  Each LoopOp (ir/loop/ir.py) is one kernel worth of compute as an SSA program:                                        │
│      body   (tuple[Stmt, ...])        — Assign | Accum | Write | Select | Loop | Load (sole stored field)            │
│      axes   (tuple[Axis, ...])        — iteration space (computed from body's Loop tree)                             │
│      loads  (tuple[Load, ...])        — body-form external reads (computed from body)                                │
│      accums (tuple[Accum, ...])       — reduce accumulators (computed from body)                                     │
│                                                                                                                      │
│  Buffers and launch order live on the graph directly: shape = node.output.shape, role derives from                   │
│  graph.inputs / graph.outputs / ConstantOp membership, launch order = graph.topological_order().                     │
│                                                                                                                      │
│  RULE: Operates on Graph + Loop IR only. No backend imports.                                                         │
├──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3 · Backends                                                                                                  │
│                                                                                                                      │
│  backend/base.py:  Backend ABC — compile(graph), run, benchmark                                                      │
│  backend/numpy/:   NumpyBackend — Graph interpreter (pre-fusion)                                                     │
│  backend/loop/:    LoopBackend  — Graph[LoopOp] interpreter (numpy)                                                  │
│  backend/cuda/:    CudaBackend  — Graph[LoopOp] → Graph[KernelOp] → Graph[CudaOp] → cupy.RawKernel (NVRTC)           │
│                                                                                                                      │
│  All three backends share:                                                                                           │
│    backend.compile(graph) → compiled                                                                                 │
│    backend.run(compiled, input_data=…) → ProgramResult (outputs=dict[name, ndarray], time_ms)                        │
│                                                                                                                      │
│  Lowering passes (backend/cuda path):                                                                                │
│    passes/lowering/kernel/lower_loopop.py   — LoopOp  → KernelOp  (uses backend/cuda/emit)                           │
│    passes/lowering/cuda/lower_kernelop.py   — KernelOp → CudaOp   (uses ir/cuda/emit)                      │
│                                                                                                                      │
│  Codegen internals (CUDA):                                                                                           │
│    ir/kernel/ir.py:           GpuKernel (AST) + KernelOp (graph-op wrapping GpuKernel + launch geometry)             │
│    ir/kernel/emit.py:         per-node codegen: LoopOp → GpuKernel (emit_kernel, launch_config)                      │
│    ir/cuda/ir.py:             CudaOp   (graph-op with rendered CUDA source)                                          │
│    ir/cuda/emit.py: GpuKernel → C source                                                                   │
│    backend/cuda/emit.py:      per-kernel LoopOp → GpuKernel helpers                                                  │
│    backend/cuda/program.py:   Graph[CudaOp] → cupy.RawKernel compile + dispatch + per-kernel event timing            │
│    backend/cuda/runner.py:    single-kernel cupy dispatch (for tuning/diagnostics scripts)                           │
│                                                                                                                      │
│  RULE: GPU specifics live here; everything above is portable.                                                        │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

See `ir/ARCHITECTURE.md` for the per-dialect breakdown of each IR level.

`trace/huggingface.py` holds helpers for tracing whole HF `CausalLM` models:
`build_full_model_wrapper()` returns an `nn.Module` with a precomputed
causal mask (so HF's dynamic mask-construction ops stay out of the
traced graph), and `collect_const_feed()` resolves tracer-emitted
placeholder names (e.g. `p_attn_q_proj_weight`) back to their module
attributes using `ExportedProgram.graph_signature`. Use
`trace_module_with_constants()` to get both the IR graph and the
placeholder→attribute map in one pass.

## Canonical Data Flow

```
PyTorch module
   │  trace.torch.trace_module(...)
   ▼
Graph (frontend ops — Layer 1)
   │  backend.compile(graph)  — unified across numpy / loop / cuda
   │    ├─ NumpyBackend: wrap Graph; numpy walk in run
   │    ├─ LoopBackend:  compile_graph(graph) → Graph[LoopOp]; interpret in run
   │    └─ CudaBackend:  compile_graph(graph) → Graph[LoopOp]
   │                     lower_to_kernel(g)   → Graph[KernelOp]
   │                     lower_to_cuda(g)     → Graph[CudaOp]
   ▼
Graph[LoopOp]  (Layer 2: fused kernels as SSA LoopOps)
   │  [CUDA path]
   │    passes/lowering/kernel/:
   │      for each LoopOp node:
   │        emit_kernel(node, name, graph)  → GpuKernel
   │        node.op = KernelOp(kernel, grid, block, arg_order, …)
   │          # KernelOp.__post_init__ → normalize_kernel (const fold / clamp eliminate)
   │    passes/lowering/cuda/:
   │      for each KernelOp node:
   │        emit_kernel_source(op.kernel) → CUDA source string
   │        node.op = CudaOp(kernel_source, kernel_name, …)
   ▼
Graph[CudaOp]  (Layer 3)
   │  backend/cuda/program._compile(graph)
   ▼
cupy.RawKernel (NVRTC)  →  cupy.ndarray dispatch  →  GPU
```

## Structural LoopOp Cheat Sheet

SSA invariants are enforced by `LoopOp.__post_init__`: unique names,
defined-before-use, no forward references.

| Slot                                         | Type                                      | Used for                                                       |
|----------------------------------------------|-------------------------------------------|----------------------------------------------------------------|
| `Axis.name` / `extent` / `kind`              | `str` / `int` / `"free"                   | One iteration variable; reduce axes pair with accumulators     |
| `Load.name` / `source` / `index`             | `str` / `int` / `tuple[Expr, ...]`        | Body-form external read: ``name = load src[index...]``          |
| `Accum.name` / `value` / `op`                | `str` / `str` / `ElementwiseOp`           | Reduce fold: ``name = op(name, value)`` inside a reduce Loop   |
| `Assign.op`                                  | `ElementwiseOp`                           | Pure SSA body op; ReduceOp is NOT valid here                   |
| `Write.output` / `index` / `value`           | `int` / `tuple[Expr, ...]` / `str`        | Inline store at a position                                     |
| `Select.branches[i].value` / `select`        | `str` / `Expr`                            | Coord-predicated SSA binding (replaces old Mux)                |
| `LoopOp.body`                                | `tuple[Stmt, ...]`                        | Sole stored field; `axes` / `loads` / `accums` are computed    |
| `node.inputs`                                | `list[str]`                               | Per-source external buffer node id (indexed by `Load.source`)  |
| `node.id`                                    | `str`                                     | The output buffer written by this LoopOp                       |

Analogies (use in module/class docstrings):

- **Named iteration axes** like Halide ``Var``/``RVar`` or MLIR
  ``linalg.generic`` iterator types.
- **Accumulator state** like MLIR ``scf.for`` iter_args — the
  ``Accum`` statement is the combine-and-write.
- **Tiled dataflow pipeline** (CUTLASS mainloop → MMA → epilogue → store) for `LoopOp`.

## Shape inference

Every ``Op`` subclass implements ``infer_output_shape(input_shapes) → tuple``.
The compiler never stores shapes redundantly — shapes are derived when
needed by calling ``infer_output_shape`` on the op instance.

| Op type                                               | Rule                                                                                                                                                                         |
|-------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ``ElementwiseOp``                                     | All inputs must already share the output shape — callers wrap mismatched inputs via ``ir/broadcast.broadcast_to`` to insert an explicit IndexMapOp.                          |
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

## Fusion policy (passes/fusion/)

Lift-then-splice, driven by the rewriter's fixpoint loop:

1. **Lift each tensor op into a trivial one-op ``LoopOp``** — one rule
   per tensor op (``001_lift_elementwise``, ``002_lift_reduce``,
   ``003_lift_indexmap``, ``004_lift_gather``). Each produces a kernel
   whose iteration space matches its op's output shape, with axes named
   ``a0, a1, ...`` and a Write at the identity index. Body-form
   ``Load`` statements express each input's access pattern directly as
   ``Expr``s over axis Vars. Reductions contribute one ``Accum``
   statement inside a reduce ``Loop`` over the reduction axis;
   multi-source IndexMapOps (cat / concat) contribute a ``Select``
   prelude choosing among per-source Loads.
2. **Splice adjacent ``LoopOp`` pairs** (``005_merge_loop_ops`` calls
   ``ir/loop/splicer.py::splice_graph``; ``splice_loop_ops`` /
   ``splice_loops`` are thin pairwise / tag-generic entry points over
   the same ``_Splicer``). The splicer is worklist-driven: seed with
   every ``Write`` of the sink, resolve one pending dep at a time,
   queue each def's own deps. A minimal σ handles non-identity Write
   forms:
   - ``Var(p)`` in ``writer.index[k]`` → bind ``p → reader.index[k]``.
   - ``Literal(c)`` in ``writer.index[k]`` → no binding (keep-dim
     reduction slot, broadcast position).
   Anything else in the writer (``Var(p) ± c``, ``Cast``,
   multiplicative) → splice refuses and the kernels stay separate.

   **Unified dedup table.** Each emission is keyed on
   ``(origin, name, emit_scope, σ.restrict(live_axes))`` — the tag/name
   identity, where the stmt lands in the merged body, and the σ
   restricted to axes actually reachable through that stmt's Expr
   subtrees (``LoopMeta.live_axes``). Same key → share one emission;
   different emit scope or σ-binding → emit again.

   **Accum placement.** A producer ``Accum`` reference materializes as
   ``Loop(fresh_reduce_axis, Accum(...))`` at
   ``_scope_for_axes(ref_scope, required_c_axes)`` — the innermost
   consumer scope covering the σ-mapped enclosing axes. SDPA-style
   patterns where QK^T is referenced inside softmax's max-loop and
   sum-loop resolve to two distinct emit scopes and emit twice, each
   under its own j-reduce — different keys, separate emissions.
3. **Input dedup + sibling reduce axis unification.** The graph-based
   ``splice_graph`` walks the subgraph and assigns each distinct
   external input node a slot in first-seen order — shared external
   buffers land in one slot by construction. The merged body then
   re-runs ``ir/loop/normalize.unify_sibling_reduce_axes``, which at
   every scope finds sibling reduce Loops whose reduce axes index the
   same ``(Load.source, dim)`` position and renames them to a single
   canonical axis name. Together these collapse
   ``sum(x, -1) + max(x, -1)`` into one kernel with two accumulators
   sharing one reduce sweep name, and make softmax's max-over-K and
   sum-over-K report as a single reduce axis.
4. **Fixpoint.** Lift rules fire once per tensor op; the splice fires
   repeatedly on adjacent LoopOps. Patterns the splicer can't handle
   (σ unsolvable, or an accumulator's required enclosing axes can't be
   mapped to the current consumer scope) simply stay as separate
   kernels.
   Copy-elimination and SSA canonicalisation are not fusion-pass rules —
   they live in ``ir/loop/normalize.py`` and run inside
   ``LoopOp.__post_init__`` on every constructed body.

The splicer subsumes two passes the old pipeline had as separate concerns:

- **Layout composition** (chained single-source IndexMapOps): the
  producer's ``coord_map`` survives as body Load indices; when the
  consumer's Load is an identity read, the splice trivially inlines
  without σ, so composed layouts come out correctly.
- **Absorption of identity-alias / keep-dim reductions**: keep-dim
  reduction writes use ``Literal(0)`` on the reduced dim — the splicer
  allows this without binding, and the reducer's ``Accum`` lands at the
  merged kernel's reduce scope naturally.

``optimization/001_compose_indexmaps.py`` runs between decomposition and
fusion to collapse chains of single-source / single-consumer
``IndexMapOp`` nodes into one composed coord_map. Matmul's decomposition
emits a ``b → unsqueeze → broadcast → mul`` chain where each layout op
is its own ``IndexMapOp``; without composition the unsqueeze lifts to a
trivial copy kernel that the splicer can't merge (its output has no
counterpart at the consumer's outermost axis). Composing before lift
keeps the matmul at one launch.

## Codegen policy (backend/cuda/emit.py)

Walks the SSA body — no classification pass, no `Schedule` dataclass, no `LoopIR` intermediate. Maintains a `values: dict[str, Expr]` mapping Assign names to C expressions.

- `_emit_load_access(index, buf, src_shape, env) -> Expr`: emits `ArrayAccess(buffer, coord)` for a body `Load.index` pattern evaluated in the axis environment. Select statements inside the SSA body are handled separately by `_emit_select`, which lowers `SelectBranch`es into a nested `Ternary` chain.
- `_emit_body`: unified entry for one kernel body. Calls `ir.loop.plan.analyze_kernel(loop, out_shape)` to produce a `KernelPlan`, then delegates to `_emit_plan`. The same plan powers the per-LoopOp pretty-printer so dump views mirror the loop nest codegen emits.
- `_emit_plan`: walks `plan.steps` — each step is either `Inline` (a straight-line block of `Assign` / `Select` / `Write` / `Accum` / `Load`) or `Loop` (a K-loop over a reduce axis). Loads inside a reduce Loop's body materialize inside that loop; Loads in prelude Inline steps happen once per row. Grid is 1D over flat free-axis extents; block is `(256, 1, 1)` for pure-elementwise plans, `(1, 1, 1)` for plans containing a `Loop` step. Empty bodies (copy kernels) are a pointwise subcase.
- Reductions (single `Loop` step) emit an accumulator init + `Accum` fold; max reductions use `fmaxf` instead of `AugAssign`. Softmax-style cross-iteration patterns (reduce_max → sub+exp → reduce_sum → div) and contractions (mul → reduce_sum) fall out naturally from multiple `Loop` steps recomputing per-element values as needed.

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
- `tests/compiler/passes/` — all rewrite passes (decomposition, optimization, fusion, lowering).

Full-model E2E (TinyLlama layer) comes back in a follow-up commit once decomposition of higher-level ops is ported into the new lowering.
