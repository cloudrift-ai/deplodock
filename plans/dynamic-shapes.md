# Dynamic Shapes in the Deplodock Compiler

Scope: support symbolic `seq_len` end-to-end so a single compiled artifact runs at multiple sequence lengths. Batch,
num_heads, and other dims stay static in v1.

## Motivation

Today the compiler is statically shaped: every `Tensor.shape` element is an `int`, every `Axis.extent` is an `int`, and
every cached kernel is keyed on the exact concrete shape. The "full-model compile + `deplodock run`" work
([[project_full_model_compile]]) and the autoregressive generation loop both want one compiled graph reused across
prefill + decode lengths. The minimum needed to unlock that is one symbolic free dim: `seq_len`.

## Representation: a `Dim` helper

Introduce `compiler/dim.py` *before* touching any pass. The current `tuple[int | str, ...]` union on `Tensor.shape` is
the source of the audit pain — every consumer has to branch on `isinstance(..., int)` and most don't, so a string
silently means "skip this optimization" or crashes deep inside a backend.

```python
@dataclass(frozen=True)
class Dim:
    """One tensor / axis extent. Static int or symbolic name. Future: Expr."""
    value: int | str

    @property
    def is_static(self) -> bool: ...
    def as_static(self) -> int: ...        # raises on symbolic — loud failure at static-only sites
    def __eq__(self, other): ...           # accepts int for migration ergonomics
```

No `__index__` / `__int__` — we *want* `int(dim)` and `range(dim)` to fail loudly at symbolic sites; that is the type's
job. Construction is explicit (`Dim(32)`, `Dim("seq_len")`); reads use `.value` or `.as_static()`.

Later additions (do not add until something forces them):

- `multiple_of: int | None` — for `Axis.split` divisibility checks (M3).
- `min / max: int | None` — for shape inference on `slice` / `select`, and for autotune bucketing.
- `value: int | str | Expr` — when we hit `2*seq_len` style arithmetic.
- `resolve(env: dict[str, int]) -> int` — for runtime launch-dim computation in the CUDA backend (M2).

Rejected alternatives:

- A `Symbol` / `StaticDim` class hierarchy — forks every call site between two branches, same pain as the union.
- Auto-converting `int` to `Dim` at every callsite — keeps construction explicit; reads stay readable via `.value`.
- Hiding `value` behind an opaque API — would hurt debugger / repr legibility for no real benefit.

## Validation slice

Smallest end-to-end thing that exercises the path:

```
deplodock compile --code "torch.nn.RMSNorm(2048)(torch.randn(1, S, 2048))"
deplodock run --code "..."     # twice with different S, assert single cached kernel
```

RMSNorm is the minimal case: one *static* reduce axis (2048), free axes over `(1, S, 2048)`. It exercises
lifting → tile-planner → CUDA emit without touching matmul K-chunking, SDPA mask construction, or split-K — all the
places most likely to need separate fixes.

Follow-on slices, in order of difficulty:

1. **Softmax over symbolic `seq_len`** — reduce axis is dynamic. Forces chunk_reduce / cooperative reduce to handle a
   non-int extent.
2. **Single-layer SDPA with symbolic `seq_len`** — pulls in mask shape, GQA reshape, K reduction.
3. **TinyLlama whole-model compile** — generation loop reuses one graph for prefill + decode.

## Milestones (single branch, milestone commits per [[feedback_single_branch_milestones]])

| M  | Slice | Validation |
|----|-------|------------|
| M0 | Add `compiler/dim.py`; switch `Tensor.shape: tuple[Dim, ...]`, `Axis.extent: Dim`, and `IndexMapOp.out_shape: tuple[Dim, ...]`; migrate the ~15 `int(d)` and ~30 `int(ax.extent)` sites to `d.as_static()`; fold `Dim.value` into `Graph.structural_key()` | `make test` green; no behavior change |
| M1 | Lifting passes (`010_lift_elementwise`, `020_lift_reduce`, `030_lift_indexmap`, `040_lift_gather`) preserve symbolic free-axis extents; `Loop.forward()` resolves them from input-array shapes at `ir/loop/runner.py:115` (already shape-from-array) | `compile --code RMSNorm` with symbolic S prints loop IR with `Axis(extent=Dim("seq_len"))` |
| M2 | Kernel signature carries `seq_len` as an i32 runtime arg; CUDA emitter renders `Axis` extent as that arg in grid bounds and free-axis loop limits | `run --code RMSNorm` with two different S values, one cached kernel |
| M3 | Tile planner handles symbolic extent on free axes (no split — bind whole axis to BLOCK or stride); explicit error on symbolic reduce axes | RMSNorm passes; softmax-over-S fails cleanly with a known message |
| M4 | SDPA decomposition + reshape handle symbolic `seq_len` (causal mask passed as input, not constructed inline); HF wrapper produces a symbolic-aware mask | Single-layer SDPA traces with symbolic seq_len |
| M5 | Reduce-axis dynamism: chunk_reduce / cooperative reduce emit a runtime loop count instead of a baked-in `int` | Softmax-over-S runs; SDPA-over-S runs |
| M6 | Trace path: `compile <hf_model> --dynamic seq_len` end-to-end on TinyLlama whole-model; generation loop reuses one graph | Two-token vs sixteen-token decode share one compiled artifact |

M0–M3 is the load-bearing scope — once a free dim threads cleanly through to launch geometry, the rest is mechanical.
M4–M6 is where the bodies are buried: mask construction, reduce-axis loop count, autotune cache keying.

## Surfaces that need work (audit)

1. **Trace** (`compiler/trace/torch.py`, `compiler/trace/huggingface.py`) — feed `dynamic_shapes={…}` into
   `torch.export.export()`; capture `torch.SymInt` from FX meta as `Dim("seq_len")`; rework
   `build_full_model_wrapper` so causal mask + `position_ids` are either symbolic-length tensors or graph inputs.
2. **Tensor / Axis** (`compiler/tensor.py`, `compiler/ir/axis.py`) — shape elements and axis extents become `Dim`.
   `Axis.split(factor)` needs to refuse symbolic extents (M3).
3. **Frontend decomposition** (`compiler/pipeline/passes/frontend/decomposition/`) — most rules propagate shapes and
   are fine, and several already `isinstance(..., int)`-guard symbolic dims (`010_sdpa.py:35-36,68,80`,
   `090_mean.py:17,21`, `_matmul_helpers.py:61`, `150_cat.py:45`), so the audit work for compound math ops is
   partially anticipated. Remaining risk sites: `010_sdpa.py` (mask shape derived from seq_len);
   `ReshapeOp` (`ir/frontend/ir.py:74,78` do `in_numel *= int(d)` and `known *= int(d)`); `SliceOp`
   (`ir/frontend/ir.py:96`) also carries `tuple[int | str, ...]` but only forwards it — its `forward` reads
   start/end from constant inputs, so symbolic dims pass through without a cast; `_broadcast.py` / `_helpers.py`.
4. **Loop lifting** (`compiler/pipeline/passes/loop/lifting/`) — all four lifting passes do `Axis(extent=int(d))`. Each
   becomes `Axis(extent=d)` with `d: Dim`. `IndexMapOp.out_shape` (`ir/tensor/ir.py:254`) is currently
   `tuple[int, ...]` and must widen too, since `030_lift_indexmap.py:26` reads from it.
5. **Tile / Kernel / CUDA lowering** (`compiler/pipeline/passes/lowering/{tile,kernel,cuda}/`,
   `compiler/backend/cuda/`) — kernel signature gains a `seq_len: int` runtime arg; launch grid
   (`backend/cuda/program.py` — `_Buffer.shape: tuple[int, ...]` at line 68, statically resolved at launch
   time via `_buffers()` at line 93), TMA descriptors (`backend/cuda/_tma.py:101`), shared-mem sizing
   (`tile/070_pad_smem.py:222`, `tile/020_stage_inputs.py:166,274,538`) all resolve the symbolic dim from the actual
   input tensor at launch time. ~20 `int(ax.extent)` call sites across `tile/` and `kernel/` will need to either pull
   from the runtime env or refuse symbolic axes (M3).
6. **Autotune cache** (`Graph.structural_key()` in `compiler/graph.py` at lines 563–621 — folds `tuple(out.shape)` into
   the digest at line 611; tune DB schema in `compiler/pipeline/search/{db.py,keys.py}`) — symbolic dims must hash by
   name, not by current binding, or the cache busts every batch.

## Open decisions

- **Divisibility on symbolic axes.** `Axis.split` requires `extent % factor == 0`. v1 answer: refuse to split symbolic
  axes — bind whole axis to BLOCK or use cooperative stride. Matches the current "no residue tail" stance in `axis.py`.
  Revisit when M5 needs to chunk a symbolic reduce.
- **Where the runtime binding lives.** Two options. (a) `KernelOp` carries a `runtime_args: dict[str, Dim]` and the
  backend resolves at launch from input shapes. (b) Each `CudaOp` re-derives bindings by walking its inputs. (a) is
  cleaner; (b) is closer to today. Defer until M2.
- **`Dim` value promotion to `Expr`.** Keep `value: int | str` in v1. Graduate to `Expr` only if a real pass needs
  `2*seq_len` or `seq_len + 1`. Most mask / position-id cases can be expressed as a separate symbolic name plus a
  graph-level offset rather than as shape arithmetic.

## Progress

- **M0 — done.** `compiler/dim.py` lives. `Tensor.shape: tuple[Dim, ...]`, `Axis.extent: Dim`, and
  `IndexMapOp.out_shape: tuple[Dim, ...]` all converted with `__post_init__` coercion (producer call sites unchanged).
  68 `int(<expr>.extent)` and ~10 shape-side reader sites migrated to `.as_static()`; the few remaining `int(d)` calls
  operate on `_Buffer.shape` (already `tuple[int, ...]`) or numpy `arr.shape`. `Graph.structural_key()` and
  `_serialize_field` both unwrap `Dim` to `Dim.value` so cached digests and on-disk JSON round-trip past the
  static-int era unchanged. `Dim.__str__` returns the bare value so pretty IR (`for i in 0..32`) and
  `Body.structural_key` digests don't shift either. All 1197 tests green.
- **M1 — done.** `010_lift_elementwise` / `040_lift_gather` / `030_lift_indexmap` drop the all-static guard;
  `020_lift_reduce` keeps a static-reduce-axis check (M5 owns symbolic reduce). `LoopOp.forward` binds symbolic
  axis names from input-array Load positions and specializes the body before C++ rendering (one cached kernel per
  runtime-shape today; per-axis runtime args land in M2). Body normalize / simplify guard their
  `as_static()` calls behind `is_static`. New `tests/compiler/test_dynamic_shapes.py` covers elementwise +
  static-reduce lift preservation and forward-time specialization at two different `seq_len` values.
- **M2 + M3 — done together.** ``CudaOp.grid`` / ``CudaOp.block`` widened to ``tuple[GridDimSpec, GridDimSpec,
  GridDimSpec]`` where each spec is a tuple of ``int | str`` factors multiplied at launch time; new
  ``runtime_args: tuple[str, ...]`` lists symbolic axis names. CUDA renderer emits ``int <name>`` kernel params after
  buffers + TMA descriptors, and grid-axis decode + index-flatten both substitute symbolic names directly into the
  rendered C (``int a0 = blockIdx.x / (4);``, ``x[a0 * 2048 + ...]``). ``partition_loops`` accepts symbolic free axes
  for pointwise + cooperative-reduce paths (BM/BN forced to 1, whole axis bound to grid), refuses symbolic reduce
  axes or symbolic matmul M/N/K with a clean ``RuleSkipped``. ``050_use_tma`` bails on any symbolic shape. The CUDA
  backend ``_compile`` walks ``graph.inputs`` to build ``symbolic_bindings: name → (input_buf, dim_index)``;
  ``CompiledProgram.build`` resolves them once from ``input_data``, and ``_launch`` substitutes those values into the
  grid spec and tail-appends them to the kernel arg pack. End-to-end: symbolic elementwise `exp(x)` and `RMSNorm`
  (traced + free-dim rewritten to ``Dim('seq_len')``) compile to a single ``CudaOp`` whose kernel source contains
  ``int seq_len`` and run correctly at multiple seq_len values. 1203 tests green.
- **M4 — done.** ``_matmul_helpers._max_dim`` handles broadcast against symbolic dims (size-1-on-other-side
  collapses cleanly); SDPA decomposition (``010_sdpa.py``) traces fine for symbolic seq_len because the inline
  causal mask is already an ``IndexMapOp`` whose ``select`` clauses reference placeholder coords, not the extent.
  ``130_reshape`` rewrites its coord-map strides as ``Expr`` trees (``Var('seq_len')`` for symbolic factors,
  ``Literal`` for static) and runs them through ``simplify`` so a reshape through a symbolic dim threads cleanly.
  ``build_full_model_wrapper`` gains a ``dynamic=True`` mode: forward becomes
  ``forward(input_ids, attention_mask, position_ids)`` so the caller supplies a per-call mask sized to the
  runtime seq_len; ``build_causal_mask`` is exposed for that purpose. M4 validation slice ("single-layer SDPA
  traces with symbolic seq_len") covered by ``test_symbolic_sdpa_traces_and_decomposes``. The remaining
  ``ReduceOp`` over symbolic seq_len (softmax max/sum + attn@V) survives decomposition but stays un-lifted
  pending M5. 1204 tests green.
- **M5 — done.** ``020_lift_reduce`` drops the static-reduce-axis check; ``unify_sibling_reduce_axes`` and
  the ``reduce`` merge phase compare extents via ``Dim`` equality so symbolic siblings unify cleanly.
  ``SerialTile.render`` uses ``str(axis.extent)`` so the rendered C ``for`` loop bound resolves to the
  symbolic kernel arg name. ``_launch_geometry`` now walks the entire kernel body (``_collect_symbolic_axis_names``)
  to harvest symbolic names from inner loops, not just grid / thread tiles. ``partition_loops`` accepts:
  pointwise / cooperative-reduce with all-symbolic free axes (``allow_empty_threads``: single-thread-per-CTA
  variant, slow but correct); matmul with symbolic M / N / K (``_build_split_body`` uses the symbolic ``Dim``
  directly for ``K_o`` when ``params.bk=1``). Ring-buffer and pipelined-stage passes defer cleanly on
  symbolic K; ``mark_unroll`` returns a placeholder trip count so it declines to unroll runtime-bound loops.
  End-to-end validation: ``softmax_over_S`` and full causal ``SDPA_over_S`` both compile to a single set of
  cached kernels and run correctly at multiple seq_len values within fp32 tolerance. 1206 tests green.
- **M6 — initial path landed.** ``compiler/trace/dynamic.py`` exposes ``make_dynamic(graph, name, value)``
  which post-trace rewrites every ``Dim(value)`` in ``node.output.shape`` plus the ``shape`` field on
  ``ReshapeOp`` / ``SliceOp`` to the symbolic ``Dim(name)``. The trace flow becomes a two-step recipe:
  ``trace_module`` at a canonical seq_len, then ``make_dynamic`` to swap the dim. Validation tests cover
  a real ``torch.nn.RMSNorm`` (1208 green; tested at seq_len 8, 32) and a real ``torch.nn.Linear``
  matmul on symbolic M (tested at seq_len 4, 16, 32). The earlier M4 ``build_full_model_wrapper(dynamic=True)``
  + ``build_causal_mask`` pair sits on top of this for HF whole-model use. Full TinyLlama whole-model with
  RoPE / KV cache / generation-loop is the remaining stretch — the planner / codegen / launch path is
  ready to receive it; the missing piece is tracer coverage for the permute+slice patterns in HF
  attention blocks.
- **Position-based ``--dynamic NAME@INPUT:AXIS`` CLI + torch.export SymInt path.** ``deplodock {compile,
  tune, run}`` accept ``--dynamic NAME@INPUT:AXIS`` (repeatable). ``compiler/trace/dynamic.py`` parses
  the spec strings to ``(name, input, axis)`` triples and converts to ``torch.export.Dim`` instances
  via ``build_torch_dynamic_shapes``. ``trace_module`` / ``trace_module_with_constants`` accept a
  ``dynamic_shapes`` kwarg that flows straight to ``torch.export.export``; torch's SymInt
  propagation determines which downstream FX tensors carry the dynamic dim. The FX walker
  (``_get_shape`` / ``_wrap_shape`` / ``_op_shape``) converts ``SymInt`` to ``Dim``, with a
  ``_sym_rename_map`` that maps torch's internal names (``s0``, ``s27``) back to the user's
  ``Dim('seq_len')`` so the IR reads cleanly. ``_expand_dynamic_shapes`` auto-fills ``None`` for
  any forward-arg the user didn't mark dynamic (torch requires ALL keys present); container args
  (HF's ``position_embeddings`` tuple) get a structurally-matching ``(None, None)`` spec via
  ``_static_spec_for``. The ``aten.sym_size.int`` FX nodes that torch emits to extract symbolic
  dim sizes (e.g. ``b, s, d = x.shape``) are skipped during the IR walk — they're consumed inline
  by reshape's ``_op_shape``, not represented as graph nodes. Value-based ``make_dynamic`` /
  ``parse_specs`` / ``apply_specs`` are gone; the only path is position-based + torch.export.
  Eliminates the value-collision class (``--seq-len 32`` colliding with ``num_heads=32``) by
  construction. Caveat: per-layer HF trace (``compile <model> --layer N``) eagerly computes
  ``cos/sin`` from a concrete seq_len, so position embeddings specialise the dim — only the
  ``--code`` and whole-model paths work end-to-end today.

## Explicitly out of scope (v1)

- Dynamic batch, num_heads, or any dim other than seq_len.
- Symbolic arithmetic in shapes (`2*seq_len`, `seq_len - 1`). Decompose to a separate symbolic name if encountered.
- Whole-model dynamic compile through `bench` / `deploy` paths. Stay on `compile --code` + `run --code` until M6.
- Residue tails for non-divisible static splits. Unchanged from today.
