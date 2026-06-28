# IR Dialects

Per-dialect op definitions. A `Graph` (`compiler/graph.py`) hosts nodes
from every dialect; the population shifts as passes run. For the
top-level layer/pass picture see `compiler/ARCHITECTURE.md`.

## Dialects at a glance

| Dialect           | When populated                  | Ops                                                                                                   |
|-------------------|---------------------------------|-------------------------------------------------------------------------------------------------------|
| `base`            | always                          | `Op` (base), `InputOp`, `ConstantOp`                                                                  |
| `frontend/ir`     | after tracing                   | `LinearOp`, `MatmulOp`, `SdpaOp`, `MeanOp`, `UnsqueezeOp`, `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp` |
| `tensor/ir`       | after decomposition             | `ElementwiseOp`, `ReduceOp`, `ScanOp`, `GatherOp`, `ScatterOp`, `IndexMapOp`                          |
| `loop/ir`         | after fusion                    | `LoopOp` + body types (`Load`, `Assign`, `Accum`, `Write`, `Select`, `Loop`, `Axis`)                  |
| `tile/ir`         | after `lowering/tile`           | `TileOp` + scheduling stmts (`Tile`, sources-only `Stage` carrying `Source`/`CacheDim` per-operand layouts, `StageBundle` carrying policy + optional `compute` phase, `StridedLoop`) |
| `kernel/ir`       | after `lowering/kernel`         | `KernelOp` + hardware stmts (`Tile`, `Smem`, `Sync`, `TreeHalve`)                                     |
| `cuda/ir`         | after `lowering/cuda`           | `CudaOp` (rendered `__global__` source)                                                               |

## Invariants by stage

- **Frontend → tensor** (after `decomposition`): `LinearOp`, `MatmulOp`,
  `SdpaOp`, `MeanOp`, and the layout ops are gone. Only
  `ElementwiseOp`, `ReduceOp`, `IndexMapOp`, scan/gather/scatter, plus
  boundaries survive. (The broadcast-explicit invariant for
  `ElementwiseOp` inputs lives in `compiler/ARCHITECTURE.md`.)
- **Tensor → loop** (after `fusion`): only `LoopOp` + boundaries.
  Tensor-IR ops survive only *inside* `LoopOp.body` as `Assign.op` or
  `Accum.op` (`ElementwiseOp` only — `ReduceOp` is not a valid body
  op; reductions are `Accum` statements inside a reduce `Loop`).
- **Loop → tile** (after `lowering/tile`): `LoopOp` nodes replaced by
  `TileOp` whose body is a `Tile` carrying scheduling decisions
  (`BIND_THREAD`/`BIND_BLOCK` axes, `Stage`). Cooperative-reduce
  emission, atomic-write classification, and broadcast-write guards
  are derived by ``escape_analysis`` at materialize / render time
  rather than carried as explicit Stmts.
- **Tile → kernel** (after `lowering/kernel`): `TileOp` materialized to
  `KernelOp` whose body uses hardware primitives (`Smem`, `Sync`,
  `TreeHalve`, `StridedLoop`).
- **Kernel → CUDA** (after `lowering/cuda`): `KernelOp` replaced by
  `CudaOp` carrying rendered source.

`Op.source` is the rewrite-chain predecessor — the engine's
`_apply_one` stamps it on every 1:1 in-place rebind, so a fully
lowered `CudaOp` carries the full chain back to its originating
`LoopOp` (`cuda.source.source.source`) without any rule needing to
pass it explicitly. The base-class field is keyword-only and
`compare=False`, so subclass positional construction and equality
keep working unchanged. `source` is excluded from
`Graph.structural_key` and from `op_cache_key` — kernels rendered
along different lowering paths still dedup in the tuning cache.

**Stmt subclasses are `@dataclass(frozen=True)`** — every concrete Loop-IR
/ Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves, `GridTile`,
`ThreadTile`, `WarpTile`, `Stage`, `StageBundle`, `Smem`, `Sync`,
`CpAsyncCopy`, `TmaDescriptor`, …) is immutable + hashable. `Body` is a `tuple[Stmt, ...]`
subclass, so a full body tree hashes structurally end-to-end. This makes
`Body.structural_key()` and any other bodies-as-cache-keys path work
without a try/except fallback for unhashable stmts. To "edit" a frozen
Stmt, return a fresh instance via `dataclasses.replace(stmt, field=value)`;
`__post_init__` coercions use `object.__setattr__`. Ops, by contrast,
are NOT frozen — the engine mutates `op.source` / `op.knobs` / `op.inputs` /
`op.outputs` post-construction. Op fields stored inside Stmts (e.g.
`Assign.op`) must be lightweight value objects (e.g. `ElementwiseImpl`,
not `ElementwiseOp`) so the surrounding Stmt's hashability isn't poisoned.

## `base.py`

Cross-cutting root. Imported by every dialect, imports nothing from
them.

| Symbol          | Role                                                                           |
|-----------------|--------------------------------------------------------------------------------|
| `Op`            | Base class. Subclasses implement `infer_output_shape` and `forward` (numpy).   |
| `InputOp`       | Sentinel: graph input tensor. Value supplied by the executor.                  |
| `ConstantOp`    | Sentinel: weights / scalar constants. Scalars carry `value`; tensors carry `source_path` / `source_shape` / `source_dtype` (the safetensors / `nn.Module` address) plus `load_ops` — a chain of frontend ops applied at bind time by the loader. |
| `_keepdim_axis` | Shape helper shared by `ReduceOp` (tensor) and `MeanOp` (frontend).            |

## `expr.py`

Shared expression sublanguage used by every IR layer: `Load.index`,
`Write.index`, `SelectBranch.select`, `IndexMapOp.coord_map`,
`StridedLoop.start`/`step`, `Cond.cond`, etc. Imports nothing from
other IR files.

| Symbol                                                                       | Role                                                                     |
|------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `Var`, `Literal`, `BinaryExpr`, `Builtin`, `FuncCallExpr`, `TernaryExpr`, `CastExpr` | Expression nodes. Each has `eval(env) → value/ndarray`, `pretty()`, `substitute(mapping)`, `free_vars()`. |
| `_ExprOps`                                                                   | Mixin: Python operator overloading for expression building; default `NotImplementedError` for `pretty`/`substitute`/`free_vars`. |
| `Expr`                                                                       | Union type alias.                                                        |
| `PLACEHOLDER_PREFIX`, `placeholder`, `is_placeholder`                        | Convention for output-coord placeholders in coord maps.                  |

## `frontend/ir.py`

Ops captured directly from PyTorch. Every one has a decomposition rule
under `pipeline/passes/frontend/decomposition/`; after that pass none of these
remain.

| Group         | Ops                                                                                       |
|---------------|-------------------------------------------------------------------------------------------|
| Layout-only   | `TransposeOp`, `ReshapeOp`, `SliceOp`, `CatOp`, `UnsqueezeOp` — rewrite to `IndexMapOp`.  |
| Compound math | `LinearOp`, `MatmulOp`, `SdpaOp`, `MeanOp`, `RmsNormOp`, `LayerNormOp`, `SoftmaxOp` — rewrite to elementwise + reduce chains. |

## `tensor/ir.py`

Minimal IR fusion consumes. `IndexMapOp` is the unified layout-only op;
it replaces the frontend layout ops via `coord_map` expressions.

| Symbol                               | Role                                                           |
|--------------------------------------|----------------------------------------------------------------|
| `ElementwiseOp`                      | Per-element scalar function (`add`/`mul`/`exp`/…).             |
| `ReduceOp`                           | Collapse one axis via associative binary op.                   |
| `ScanOp`                             | Cumulative variant of reduce.                                  |
| `GatherOp`, `ScatterOp`              | Data-dependent reads / writes.                                 |
| `IndexMapOp` + `IndexSource`         | Unified layout-only op over `Expr`.                            |

Op metadata (arity / `commutative` / `associative` / `identity` /
`has_identity` / `selecting` / `semiring_product`) lives on `ElementwiseImpl` in
`ir/elementwise.py` — the single source of truth shared across elementwise,
reduce, scan, and accumulator use sites. The algebraic traits are what
reassociation gates (split-K, cooperative tree-combine) query instead of matching
op names. The per-op trait *properties* (`op.semiring_product` — is this op a `⊗`
in some semiring) and the binary method `⊗.distributes_over(⊕)` (does this product
distribute over that reduce — the `_SEMIRING` table, only `(+, ×)` today) live on
`ElementwiseImpl`; the module's op-name-free **role queries** the planner /
atom-cell matchers / flash recognizer ask round them out: `reduce_canon` (alias →
base combine, `sum` → `add` …) and the `_REDUCE_SPELLING` registry
(`reduce_spelling`) — the single op-keyed table
behind the four sites that used to switch on the reduce op name (`Accum.render`'s
`+=` / `*=` / `fmax` / `fmin`, `kernel/ir._binary_combine_expr`, and
`ReduceOp.forward` / `ScanOp.forward`'s numpy reductions). `op.selecting` (the
max/min family) drives the init-placement dtype choice.

## `loop/`

One `LoopOp` = one GPU kernel described as an SSA program over named
iteration axes. Free vs reduce is inferred from body structure — a
`Loop` is a reduce Loop iff its body contains a `ReduceCarrier` (the shared
base of `Accum`, its tensor-core form `Mma`, and the general monoid `Monoid`;
`is_reduce`, axis threading, and the other carrier-agnostic checks key off the
base, not an `isinstance(s, (Accum, Mma))` ladder). Rules that need the combine's
algebra read the carrier's `associative` / `commutative` / `has_identity` traits
directly (`Accum` forwards to its scalar `op`; `Mma` reports the additive-fold
constants; `Monoid` reports `associative` / `has_identity` `True` by construction
with a per-instance `commutative` field).

**The algebra is in the body, not a tag** (`ir/stmt/algebra.py` — the consolidated
algebraic vocabulary: the lift `Map`, the carrier `Monoid` + `Twist`, and the
`Semiring` contraction view). There is no stored / derived `AlgebraKind`: a kernel's
algebra is read directly off its carriers and partial structure where a pass needs it.
The fold ⊕ is the carrier (`Accum` scalar fold, or `Monoid` + `Twist`); the lift is
the partial — a unary value (a reduction: sum / max / online softmax) or a ⊗-product
over several contraction operands (a contraction: matmul). A non-reduce scope is
pointwise.

The one structural shape the schedule must recognize lives here: `Semiring` (the
`reduce(⊕) ∘ map(⊗)` view) and `Semiring.match(loop)`, computed on demand. A
reduce is a contraction not by "two loads" but by the genuine algebra — the lift
⊗ **distributes over** the fold ⊕ (`multiply` over `add`; *not* `add` over `add`,
a sum of two operands) and contracts ≥ 2 distinct operands (`x·x` is a squared
reduce, not a contraction). The view exposes `fold` / `lift` / `operands` /
`reduce_axis` (and `is_additive` — the `(×, +)` semiring the mma atom implements).
`lowering/tile/010_recognize` uses `Semiring.match(...) is None` to keep a
matmul's `Accum` an `Accum` (rather than degenerate-monoidizing it like a plain
reduce); `020_schedule` gates flash structurally (a reduce loop nested inside a
reduce loop); the mma atom tier reads the operands + `is_additive` to pick the
tensor-core cell.

`Monoid` is the general loop-carried **monoid** carrier — *(identity element,
associative operation, internal state)* made explicit: `state` (the carried SSA
names), `partial` (this step's contribution), `identity` (one `Expr` per state
component, seeded by the enclosing `Init`), a `commutative` flag, and a `twist`
(below) that holds the operation. The whole operation lives inside the carrier,
not as loose body statements, so the online-algorithm gates (`accums_independent`,
`classify_fragment_epilogue`) never see the cross-state coupling. `carried_names()`
/ `defines()` return `state`; `deps()` / `partial_deps()` return `partial` (the
carried read is implicit, like `Accum` / `Mma`).

**The `Twist` — the part that varies, extracted from the algebra.** Transport of
structure: a monoid `(·, e)` conjugated by a bijection ψ gives the twisted combine
`x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))`. The monoid algebra above is shared; ψ is the twist.
`Monoid.twist` is a `Twist` holding the operation **as data** — `merge` (a short
`Assign` program that reads old state + partial and reassigns the new state;
state-targeting Assigns are updates, the rest local temps) and `combine_states`
(the **state-merges-state** form the cross-partition combine needs —
cooperative-tree / split-KV / split-K cross-CTA reduce, where each partition holds
a complete state, reading the second operand `state_b`, default `"<s>__o"`). ψ
lives entirely in those programs — a plain reduction's identity twist
(`Twist.degenerate`: componentwise `state_i = op_i(state_i, partial_i)`, used by
`Accum.as_monoid`), online softmax's max-rescale, a future mma-fragment
realization — all the *same* monoid, differing only in the combine. Readers reach
it directly off the carrier (`monoid.twist.merge` / `.combine_states` /
`.state_b`). `Monoid.__post_init__` completes the twist
against the state: defaults `state_b` and, for an **additive** carrier whose
partial lifts to a state (`len(partial) == len(state)`), auto-derives
`combine_states` from `merge` (partial reads swapped for `state_b`); an asymmetric
monoid (flash's LSE) authors both on the twist (`flash_combine`).
`as_state_merge(other)` returns a one-shot `Monoid` whose `merge` IS
`combine_states` with `state_b` renamed to `other`, so a two-partition merge
renders through the same machinery as a streaming step.

`Monoid.render` emits the `merge` program in fp32: each `Assign` targeting a
`state` name is a reassignment of the carried value (declared by an enclosing
`Init`); every other `Assign` declares a local temp. Statement order is
load-bearing (a state update follows every read of that state's old value).
`rewrite` threads the merge through the SSA renamer (state / partial refs map;
the carrier-internal temps pass through). `LoopOp` validation threads `Init` (a
carried-name binding site) and `Monoid` (partials must be in scope; state is
loop-carried and exports), so a hand-written streaming nest type-checks and runs
through `LoopOp.forward`. **Example** — flash attention's online softmax (the
log-sum-exp monoid): state `(m, l, O)`, partial `(score, value)`, identity
`(−inf, 0, 0)`, merge `m_new=max(m,s); alpha=exp(m−m_new); l=l·alpha+exp(s−m_new);
O=O·alpha+exp(s−m_new)·v; m=m_new` (built by `loop/fusion/_flash.flash_combine`).

### `loop/ir.py` — LoopOp types

| Symbol                       | Role                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `Axis`                       | Named iteration variable (`name`, `extent`). Defined in `ir/axis.py`, re-exported here. Carries optional `source_axis` (pre-split origin) and `real_extent` (pre-ceil-div bound for masked tiles — block axis covers `ceil_div(real_extent, BN·FN)`; materializer reads it to gate boundary lanes). Both excluded from equality / hashing. |
| `LoopOp`                     | One kernel. Stored field: `body` (nested `Loop` tree). Computed: `axes`, `loads`, `accums`.                       |
| `Load`                       | Body-form external read: `name = load(input)[index...]`. `input` matches the producing graph node's id.           |
| `Assign`                     | SSA body stmt: `name = op(args)` with `op: ElementwiseImpl`.                                                      |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to its op's identity. ``axes`` lists the reduction axis names — propagated through Sigma renames (including σ-splits via `Expr.free_vars()`); the escape-analysis helper derives cross-thread cooperativity from ``axes ∩ enclosing ThreadTile.axes``. |
| `Init`                       | Explicit accumulator initialization at an outer scope (matmul chunked-K).                                         |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Serial iteration block: `axis` + nested `body`.                                                                   |
| `StridedLoop`                | Strided iteration (`start`, `step`) — cooperative thread-stride loop reused by Tile/Kernel IR.                    |
| `Cond`                       | If/else block over an `Expr` predicate.                                                                           |
| `Stmt`                       | Base class — every body statement subclasses it. Leaves and control-flow nodes live in `ir/stmt.py`.              |

Body walkers: `iter_body(body)` (pre-order; powers `for s in loop_op`),
`map_body(body, fn)` (transformer), `Stmt.rewrite(rename_ssa, sigma)`
(per-stmt copy with SSA rename + Expr substitution),
`Stmt.pretty(indent)` (rendered lines for kernel dumps; block stmts
recurse via `pretty_body`).

Dependence cones (`ir/stmt/body.py`): `Body.backward_cone(roots)` / `Body.forward_cone(seeds)` build a `Cone` —
the subset of the body's immediate stmts closed under SSA dependence (a wrapper joins as a unit; internally-bound
axes excluded), plus `external_reads`, the names read from outside (axis vars and enclosing/sibling scopes alike).
Construction never fails: unresolved names are data, and chaining scope levels means seeding the next level's
`backward_cone` with the previous one's `external_reads`. `Body.defs_die_at(members, roots=…, allowed=…)` is the
matching escape check (may the cone be cut out, with only the designated consumers reading its roots?). This is
the shared substrate behind the rules that slice cones (`010_split_demoted`'s producer cut, `assembly/_slab._hoist_masked`'s masked-load
guard) — eligibility judgments stay in the rules, per `pipeline/passes/ARCHITECTURE.md`. Two dataflow walks
deliberately do NOT use it: `classify_fragment_epilogue` (single pass interleaving reduce-scope flags with its
negative-form blocker reporting) and `030_hoist_invariant_compute` (all-deps saturation under an axis-invariance
predicate — a different operator than the cone's any-dep taint).

`rewrite` has two distinct rename channels that must stay disjoint:
`rename_ssa` carries **SSA-name** renames, `sigma` carries **axis**
substitutions. `Load`/`Write` index exprs apply *both*
(`_rename_ssa_vars_in_expr(sigma.apply(e), rename)`) so an indirect
(gather) index Var gets renamed exactly once. Putting the same name in
both maps renames it twice — and if the two passes form a chain (e.g.
`x → in5` and a pre-existing `in5 → in26`) the double application
collapses it transitively, silently wiring a gather to the wrong row.

### `loop/normalize.py` — structural canonicalization

Pure `body → body` passes run from `LoopOp.__post_init__` so every
constructed `LoopOp` (including intermediate fusion results) is
canonicalized before validation:

- `topo_sort_siblings` — stable Kahn reorder so SSA defs precede their uses
  within each body (fixes splicer-produced use-before-def).
- `drop_size_one_free_axes` — inline extent-1 free Loops.
- `canonicalize_free_axis_order` — sort outer free Loops by axis name.
- `eliminate_copy_aliases` — drop `y = copy(x)` Assigns.
- `unify_sibling_reduce_axes` — rename sibling reduce Loops whose
  reduce-axis Load positions overlap on any `(source, dim)` pair so
  they share one canonical axis name (softmax's max + sum sweeps; the
  two matmul reductions in `silu(x@Wg) * (x@Wu)` that both index `x`
  at the same K slot). Union-find groups all transitively-overlapping
  Loops at one scope.
- `merge_sibling_reduce_loops` — concatenate sibling reduce Loops that
  share `axis.name` / `extent` into one Loop body. Gated on disjoint
  SSA defs across the two halves, the second body not reading any name
  the first body defines (blocks softmax-style sequential reduces
  where sum-exp reads `acc_max`), and no between-stmt def consumed by
  the second loop. Eliminates the duplicate K traversal in patterns
  like `silu(x@Wg) * (x@Wu)`; downstream `dedup_loads` then collapses
  the duplicate `x` loads, and the lowering passes stage both weight
  tensors symmetrically.
- `split_invariant_divides` — rewrite `divide(x, y)` into
  `reciprocal(y) + multiply(x, recip)` when `y` is loop-invariant
  w.r.t. some axis `x` depends on, so the rcp can hoist out of the
  inner loop and the per-iter cost drops from XU divide to FMA
  multiply.
- `hoist_loop_invariants` — pull loop-invariant Assigns out of reduce
  Loops.
- `rename_ssa_sequential` — cosmetic: `Load` names become `in0, in1,
  …`, Assign/Select `v0, v1, …`, Accum `acc0, …`, in definition order.
  Records renames only in the SSA channel (`rename`), never the axis
  channel (`sigma`) — see the `rewrite` two-channel rule above; an SSA
  name leaking into `sigma` double-renames indirect (gather) indices.
- `canonicalize_buffer_names` — rename `Load.input` / `Write.output` to
  `b0, b1, …` in encounter order. Off by default (buffer names bind to
  graph nodes) — opt in via `normalize_body(..., canonical_buffers=True)`.
  Used by `Body.structural_key()` for dedup queries where buffer identity
  doesn't matter.
- `sort_commutative_args` — sort `Assign.args` for commutative ops
  (`add` / `multiply` / `maximum` / `minimum`) so two bodies that
  differ only by argument order land in the same canonical form.
  Runs last so the sort key is the post-rename canonical SSA / buffer
  names.

`Body.structural_key()` re-runs `normalize_body(self, hoist=False,
canonical_buffers=True)` and joins `pretty_body`'s line list — a
`cached_property` returning the canonical text rendering. Two bodies
that differ only by SSA / axis names, commutative-arg order, or
external-buffer names produce the same key. Use it as a dict key /
set member when deduping candidate bodies in a search.

### `loop/simplify.py` — Expr simplification

Called inside `normalize_body`. Generic bottom-up Expr rewriter:
constant folding, algebraic identities, range-based comparison folding
(`(k0 > 2047 ? 2047 : k0) < 0 ? 0 : k0` → `k0`). `SimplifyCtx`/`Interval`
track integer ranges from axis extents (`axis.extend_simplify_ctx` pushes
each loop axis into the ctx). `SimplifyCtx.bounds` additionally tracks a
*symbolic* exclusive upper bound per var (`i < seq_len`) so a modulo by a
non-literal divisor folds — `i % seq_len → i` when `i`'s loop extent is
`seq_len` (`_mod_below_divisor`). This collapses the delinearized seq
coordinate `((i*stride + feat) / stride) % seq_len` that compose-indexmaps
emits back to `i`, the symbolic-shape counterpart of the literal-divisor
`_div_mod_decompose` cleanup (a static `seq_len` already constant-folds it).
Symbolic-extent axes get `[0, sentinel]` ranges (non-negativity for the inner
`(i*c + …)//c → i` div fold) instead of being dropped.

Also used by `ir/kernel/normalize.py` for GpuKernel Expr simplification.

### `loop/splicer.py` — LoopOp merger

The machinery `pipeline/passes/loop/fusion/010_merge_loop_ops.py` calls to
splice adjacent `LoopOp` pairs. `Sigma` (from `ir/sigma.py`) is the
axis-substitution bookkeeping threaded through the merge.

`splice_graph` derives splice edges as `(node_id, node_id)` — it **assumes a
producer LoopOp's sole `Write.output` buf is its node id** (the buf-name ==
node-id invariant the whole graph maintains). A rule that emits a LoopOp whose
`Write.output` doesn't match its node id silently breaks every later fold of
that node: the edge points at a Write that doesn't exist, so the splicer raises
`_NotSupported` and the node survives as its own kernel. Rules that rename a
node must rename its body `Write.output` to match (`fusion/_helpers.py::rename_write_output`).
Every `_NotSupported` carries a reason string, logged at DEBUG by `splice_loops`
— `compile -vv` shows which pattern a rejected edge hit.

### `loop/interpret.py` — numpy interpreter

`execute_loop_op(loop, input_arrays, out_shape) → ndarray` walks the
LoopOp body against pre-provided input arrays. Powers `LoopOp.forward`
— so post-fusion graphs run through the default `Backend.run`
topo-walk like any pre-fusion graph.

### `loop/builder.py` — fluent construction

`LoopBuilder` helper used by decomposition/fusion tests to construct
LoopOp bodies without spelling out every `Loop(Axis(…))` nest.

## `tile/`

Tile IR encodes scheduling decisions structurally — `Tile.axes` carry
`BIND_THREAD` / `BIND_BLOCK` bindings, `Stage` wraps consumer subtrees
that read smem-cached operands. Coordination decisions are derived from
the body at materialize / render time via ``ir/tile/escape_analysis.py``:
cooperative-reduce combine emission from ``Accum.axes ∩ ThreadTile.axes``,
atomic-write classification from enclosing ``GridTile.axes`` vs
``Write.index``, broadcast-write guards from cooperative thread axes vs
``Write.index``. One cooperative analysis covers every reduce carrier — a scalar
``Accum`` is the degenerate 1-component monoid (its ``combine_partials`` is the
one-``Assign`` op-fold), the general ``Monoid`` (flash online-softmax) the
multi-component case — keyed by the carrier's first carried name. The
materializer's single ``emit_combine`` backend (reached via the one
``carrier_algebra.realize(carrier, dist)`` composer) emits the cross-thread fold off the
carrier's ``carried_names`` / ``combine_operands`` / ``combine_partials``:
``WarpShuffle`` (register ``__shfl_xor_sync`` butterfly, ≤ warp), a hierarchical
per-warp ``WarpShuffle`` + ``n_warps``-wide ``TreeHalve`` (power-of-two warp
multiple), or a block-wide per-component ``TreeHalve`` (> warp) — all folding via
the carrier's ``combine_states`` and reassigning the state in place (no ``_b``
rename). No explicit coordination stmt or per-tile tag
carries this information. Compute leaves (`Load` / `Assign` / `Accum` / `Write`)
and control flow (`Loop` / `StridedLoop` / `Cond`) come from `ir/stmt.py`;
``Accum.axes`` carries the names of the loops being reduced over and is
the source of truth for cooperativity.

**Binding-tier tile flavors.** Three `ParallelTile` subclasses bind a
parallel coord: `GridTile` (one coord = one CTA, lifts to `blockIdx`),
`ThreadTile` (one coord = one thread, lifts to `threadIdx`), and
`WarpTile` (one coord = one warp; the body presumes 32 lanes execute it
collectively, with `lane = threadIdx.x & 31` exposed unconditionally).
`ThreadTile` and `WarpTile` are mutually exclusive inside one
`TileOp.body` — both bind `threadIdx`. `RegisterTile` (per-thread
register cell) and `AtomTile` (hardware-atomic MMA cell — one coord =
one fragment) are both consumed before kernel render: `RegisterTile` by
`kernel/010_split_register_axes` (cell-body replication); `AtomTile` by
`kernel/005_lower_atom_tile`. `RegisterTile.reduce` (default `False` = the
FM/FN output-cell tile) flags the **reduce-axis** `K_f` tile the planner
emits for the `FK` multiple-accumulator optimization (non-matmul reduces;
see `plans/fk-register-tile-reductions.md`): it strip-mines the K serial
loop so each cell owns an independent accumulator, and
`010_split_register_axes` appends a cross-accumulator tree-fold (collapsing
`acc_0..acc_{FK-1}` back to one `acc` after the K serial loops) when the tile
wraps `Accum`s — the flag also routes `FK`-vs-`FM`/`FN` knob stamping. `partition_loops` stamps the `Atom` spec (cell
shape + operand dtypes — `Atom` lives in `ir/tile/ir.py`) **onto the `AtomTile`
itself** (`AtomTile.atom`) — the structural "this matmul factorizes through
tensor cores" signal, carried in the IR rather than re-derived from a knob.
Right after, `tile/enumeration/050_warp_build` reads `.atom` off the tile and
collapses the cell's `Assign(multiply) + Accum` into a single `Mma` op
(`c += a @ b`, a reduce-accumulate sibling of `Accum` — both subclass
`ReduceCarrier`, so the `Mma` makes its loop `is_reduce` with no special-casing,
and reports the additive-fold traits `associative` / `commutative` /
`has_identity` directly) that carries that `Atom` and names its A/B
operand `Load`s by SSA
value. The operand loads stay **plain** — the `Mma` is the sole tensor-core
marker downstream. Both are ordinary IR the staging passes carry through (the
loads stage like any `Load`); `kernel/005_lower_atom_tile` recovers each
operand's role from the co-located `Mma` and lowers the loads →
`RegFragment`+`LdmatrixLoad` and the `Mma` → `MmaSyncPtx`, with a final
`RegStore`. The `ATOM_KIND` knob on the enclosing `TileOp` is the *tuning*
shadow of the same choice (DB / config / search identity), not the semantic
source. Downstream consumer plans (MMA fragment
factorization, warp-specialize refactor) emit `WarpTile` to drive
warp-cooperative codegen.

The `Mma` also carries optional **explicit masked-tile guards** (`m_guard` /
`n_guard` / `k_zero`, each `(base, bound)` or `None`) for a HAND-BUILT cell — the
symbolic-`seq_len` warp-chain flash, where `kernel/005` can't derive guards from a
Write boundary `Cond` (a fragment-output / fragment-A cell has no Write) or the
operand tensor shape (the flash uses flat single-index Loads). When set, `005`
routes them straight to the operand `LdmatrixLoad`s (A row clamp / B col clamp / B
reduce-row zero-fill); `None` (the default, the enumeration-σ path) keeps `005`
deriving guards from the Write `Cond` + operand shape as before.

**Stage + StageBundle:** `Stage` is a sources-only group of gmem
transport operands behind one barrier — it carries no body. The
producer (cooperative `Load+Write` per source) is synthesized at
materialize time from `Stage.sources` — each `Source` carries `name`
(smem buffer), `buf` (gmem operand), `cache_dims`, `origin`, optional
`pad`, and a stored `addressing` of type `AffineAddressing |
TemplateAddressing` describing how cache vars decode into source-buffer
indices. Multi-source bundles (e.g. matmul A+B) load all behind one sync
boundary. The `StageBundle` owns the consumer `body`, the transport
`policy` (`StagePolicy.SYNC/BUFFERED/ASYNC/TMA` + policy fields
`buffer_count`/`phase`/`pipeline_depth`), and an optional
hoisted-invariant `compute` phase (a self-describing cooperative body
that reads sibling slabs and writes a fresh fused slab — emitted after
the transport sources, before the consumer body).

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `TileOp`              | Graph-op carrying a `Tile`-rooted body. One per kernel.                                                                              |
| `Tile`                | Axis-bound scope wrapper (`axes: tuple[BoundAxis, ...]` + body).                                                                     |
| `StageBundle`         | Single-policy cooperative-staging unit: ``sources: tuple[Source, ...]`` (gmem transport operands, per-operand smem layouts) wrapping one consumer `body`. Materialize emits leading `Sync` + per-source cooperative `Load+Write` (or `CpAsyncCopy` / per-source TMA box copy per the policy) + trailing `Sync`. Carries the transport `policy` (`StagePolicy.SYNC/BUFFERED/ASYNC/TMA`) + policy fields (`buffer_count` ≥ 2 rotating slabs selected by `phase`; `pipeline_depth` > 1 marks temporal pipelining expanded by the pipeline-stages pass), and an optional `compute` phase. ASYNC `cp.async` loads are **vectorized** — `_stage_expand._cp_async_width` picks the widest legal copy (`CpAsyncCopy.nbytes` ∈ {4,8,16}; 16⇒`cp.async.cg`) gated on inner-axis stride-1 contiguity + alloc/gmem-stride alignment. TMA shared-memory swizzle is per-`Source` (`Source.swizzle`), not a bundle field. |
| `StageBundle.compute` | Optional hoisted-invariant cooperative compute phase (a `Body`, set by `030_hoist_invariant_compute`): `Load`s reading sibling cone slabs + `Assign`s + a single `Write` into a fresh fused slab. Self-describing — the materializer recovers the slab name / loop domain / dtype from the body's `Write` + the cone sources (no output `Source`). Emitted after the transport sources, before the consumer body. |
| `Source`              | One gmem operand staged into one smem slab. Fields: `name`, `buf`, `cache_dims`, `origin`, `pad`, `addressing`, `dtype`, `swizzle` (per-operand TMA smem-swizzle mode; `NONE` except on mma.sync ldmatrix operands). A `StageBundle` carries one or more. |
| `CacheDim`            | One cache (smem) axis paired with the source-buffer dim it maps to. `Source.cache_dims` is a tuple of these.                          |
| `AffineAddressing`    | Stored addressing variant: `source_index[d] = origin[d] + decoded_coord(dims[i] == d)`. Fast path; no symbolic substitution. Optional per-cache-dim `block` multiplier grows the slab and producer iteration range by `block[i]` per cache dim (e.g. MMA atom factor); default `()` keeps coef-1 semantics. |
| `TemplateAddressing`  | Stored addressing variant: source index expressed verbatim with cache-axis Vars; materialize Sigma-substitutes them. Used for collapsed-reshape views and any other case where `origin + decoded` can't reconstruct the load. |
| `AsyncWait`           | Explicit wait carrier for pipelined async / TMA schedules. Emitted by `080_pipeline_stages` between issue / consume halves of each steady-state K_o iter and at the epilogue drain. ``keep`` is the cp.async ``wait_group`` arg; ``phase`` / ``slot`` are TMA mbarrier-test args. Sync-style (``pipeline_depth==1``) stages don't carry one — the materializer emits an implicit wait at the wrap boundary. |

## `kernel/`

### `kernel/ir.py` — fully-scheduled kernel form

Reuses `Tile` + leaf stmts from Tile IR; adds hardware primitives
materialized from scheduling decisions. `KernelOp` carries the body
directly (no separate AST class).

| Symbol             | Role                                                              |
|--------------------|-------------------------------------------------------------------|
| `KernelOp`         | Graph-op wrapper around a `Tile`-rooted body. One per kernel.     |
| `Smem`             | `__shared__` array allocation (name + dtype + extents + optional `align`). Swizzled TMA operand slabs align to their full swizzle atom (`8 × swizzle_width` B: B128→1024, B64→512, B32→256) — the coordinate-only `ldmatrix` XOR only reproduces the hardware's absolute-address swizzle when the base zeroes the swizzle's source-address bits; non-swizzled TMA keeps 128 B, fp16 16 B. `pack_smem` (the shared pool packer used by `smem_bytes` and the renderer) pads each buffer to `max(sizeof(dtype), align)` so the static-vs-dynamic gate and the launch-time dynamic-pool size agree. |
| `Sync`             | `__syncthreads()` barrier.                                        |
| `TreeHalve`        | Cross-thread tree reduction over a smem buffer.                   |
| `RegFragment`      | mma.sync (s16816) per-thread register array decl (one per operand role `"a"`/`"b"`/`"c"`): `unsigned a[4]`/`b[2]` (16-bit operands, 2 elems/reg) or `float c[4]` (f32 acc, zero-init at decl — no separate fill). Carries the cell shape `(M, N, K)` + dtype. Emitted by the MMA cell lowering pass (`kernel/005_lower_atom_tile`). The sole tensor-core fragment family (the opaque `nvcuda::wmma` nodes were removed). |
| `LdmatrixLoad`     | Load one operand into a `RegFragment`. `staged=True` (default): `ldmatrix.sync.aligned.m8n8.x{4,trans}.b16` from smem (`role="a"` → x4; `role="b"` → x2.trans; each lane derives its row address from `threadIdx.x & 31`; `swizzle` applies the per-lane chunk XOR for a TMA-swizzled slab). `staged=False`: operand not staged into smem (ldmatrix is smem-only) → renders a **gmem-direct fragment load** (`dpl_mma_load_{a,b}_gmem`) reading the fragment straight from gmem with the same m16n8k16 lane→element map — slower (no smem reuse) but lets an unstageable MMA tile compile instead of crashing. `b_trans=True` (role "b" only) marks a transposed-B operand stored `[N, K]` (the native `mma.row.col` col-major B — a Q@K^T cell): gmem-direct via `dpl_mma_load_b_gmem_trans` (k contiguous; masked → `_trans_nclamp`). |
| `MmaSyncPtx`       | `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32` — one s16816 MMA via inline PTX (`c += a @ b`). `ab_dtype` (`"f16"`/`"bf16"`) picks the `dpl_mma_…` wrapper. |
| `RegStore`         | Per-lane epilogue store of the f32 `c[4]` accumulator to the output (no `store_matrix_sync` for mma.sync) — direct for f32 dst, `__float2half` downconvert for f16. Optional `epilogue` (a `RegEpilogue`: leaf `EpilogueLoad`s with per-dim `m`/`n`/`fixed` roles + `(name, op, args)` chain in topo order, plus `selects` — coord-predicated causal-mask ternaries) carries a fused pointwise chain — residual adds, bias/scale broadcasts, activations, the causal attention mask — evaluated per element in f32 at the element's own (row, col), leaves loaded at each buffer's own dim stride, ops rendered via `op_to_expr` (folded in by `kernel/005_lower_atom_tile._scan_epilogue` after the shared negative-form gate `lowering/_predicates.classify_fragment_epilogue` admits the slice; leaf buffers declared via `external_reads` so they stay in the kernel signature after their scalar Loads are stripped). Each `selects` entry `(name, ((cond|None, value), …))` renders as a per-element ternary, its `__M__`/`__N__` placeholder coords substituted with the element's absolute (row, col). |
| Shared from `tile` | `Tile` (launch geometry); from `ir/stmt.py`: `Loop`, `StridedLoop`, `Load`, `Assign`, `Accum`, `Write`, `Select`, `Cond`. |

## `cuda/ir.py`

| Symbol    | Role                                                                        |
|-----------|-----------------------------------------------------------------------------|
| `CudaOp`  | Graph-op carrying `kernel_source`, `kernel_name`, `arg_order`, `grid`, `block`, `smem_bytes`, `zero_outputs`, `comment`. Produced by `pipeline/passes/lowering/cuda` (renders the `KernelOp` body to a `__global__` source string). |

## Graph as the single program form

There is no separate program type. A `Graph` is the execution plan:
node ids are buffer names, `node.output.shape` is the buffer shape,
`graph.topological_order()` is the launch order, and
`graph.inputs` / `graph.outputs` / `ConstantOp` membership gives each
buffer its role (input / output / constant / scratch).
