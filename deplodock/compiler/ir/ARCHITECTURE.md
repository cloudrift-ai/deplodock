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
| `tile/ir`         | after `lowering/tile`           | `TileOp` carrying a typed `Kernel` (`tile/schedule`: the op-tree `Map` paired with one kind-free `TileSchedule`); the schedule holds the free→grid `Placement` + the reduce `ReducePlan` |
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
  `TileOp` carrying a typed `Kernel` (`tile/schedule` — the op-tree
  `Map` paired with one kind-free `TileSchedule`; a kernel's structure
  is read off its annotated reduce loop's `AxisRole`, not a Python
  type). `010_recognize` lifts the `Map` (a thin `Body` wrapper over
  the annotated loop nest, each reduce `Loop` carrying its `AxisRole` +
  `Carrier`) with an UNMAPPED `Placement`; `020_schedule` maps the free
  axes onto the grid and decides the reduce `ReducePlan` via the single
  `REDUCE` codec knob (`g<n>` cta / `b<n>` coop / `r<n>` reg; the
  decision hierarchy = env pin > search/prior fork > conservative
  default). The knob is ephemeral — resolved here into the schedule's
  `ReducePlan`; the combine stays on the loop's `Carrier`. Any static
  `PLANAR` / `TWISTED` reduce is cooperation-eligible (degenerate
  `sum`/`max`/`mean` AND twisted online-softmax / flash, scalar AND
  full-row outputs); the default cooperates a wide reduce feeding an
  under-occupied grid.
- **Tile → kernel** (after `lowering/kernel`): `TileOp` materialized to
  `KernelOp` whose body is a `Tile` (the thread-grid decode) over the
  lowered op tree. A cooperative `ReducePlan` lowers the reduce as a
  `StridedLoop` (lane-strided fold) + the derived carrier-generic
  cross-thread combine (`_combine.emit_combine` → `WarpShuffle` /
  `Smem`+`Sync`+`TreeHalve`, multi-component for a twisted carrier) +
  the projection (a full-row output sweep distributed across the coop
  lanes, a scalar output guarded to lane 0); the `Tile` gains the coop
  lane axis and `block_threads = coop`. A **symbolic reduce axis**
  (dynamic `seq_len`) is supported — the `StridedLoop`'s `< seq_len`
  bound is the runtime-extent mask (idle lanes fold the identity; no
  ceil-div / clamp) and the `Dim` name is threaded as a runtime `int`
  arg. The cross-CTA split (`030_split`), `reg` fold, a symbolic FREE
  axis (dynamic grid), strided rows, and the tensor-core `warp_tile`
  (incl. flash's warp tier) are reserved future tiers.
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
/ Tile-IR / Kernel-IR statement (`Loop`, `Cond`, leaves, `Tile`, `Smem`, `Sync`,
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
`Loop` is a reduce Loop iff its body holds a carrier — `Accum` or its
tensor-core form `Mma`. `is_reduce` (and axis threading and the other
carrier-agnostic checks) test exactly that `isinstance(s, (Accum, Mma))`
tuple — there is no shared base class; the carriers are plain `Stmt`s
that happen to share the reduce-surface methods. `Accum` / `Mma` expose
`associative` / `commutative` / `has_identity` traits (`Accum` forwards to its scalar
`op`; `Mma` reports the additive-fold constants). A reduce `Loop` also carries its
scheduling `AxisRole` (`loop.role`) and the decoupled `Carrier` algebra payload
(`loop.carrier`), both stamped by tile-lowering recognition (commutativity is unused —
split/reorder legality is a future cooperative-tier concern, recorded structurally when
it returns).

**The algebra is in the body, not a tag** (`ir/stmt/algebra.py` — the consolidated
algebraic vocabulary). There is no stored / derived `AlgebraKind` and no op-tree node zoo:
a kernel's compute is ONE `Map` — a thin `Body` wrapper over the per-cell loop-IR stmts
(operand `Load`s, the lift `Assign`s, an optional annotated reduce `Loop`, then the
post-reduce projection) — and a pass reads its algebra **structurally** off the annotated
reduce loop where it needs it, never a Python type:

- `Map` — the pointwise lift wrapper: a `body` (a `Body` of stmts) + a derived `out`
  property (the carried state of a trailing reduce `Loop`, else the body's last def). It
  HAS a `Body`, it is not one; there is no `source` / nested-node field — composition is
  just stmt order in the body.
- A reduction is a `Map` whose body holds the **annotated reduce `Loop`** followed by its
  projection. The `Loop` carries its `AxisRole` (`loop.role`) and a `Carrier`
  (`loop.carrier`); `ops.reduce_loop(op)` returns the outermost annotated reduce `Loop` and
  `ops.axis_role(op)` reads its role — `PLANAR` (plain `sum`/`max`/`mean`), `TWISTED`
  (online-softmax / flash), `CONTRACTION` (matmul), or `FREE` (pointwise / flat fallback).
- A contraction is a `Map` whose reduce `Loop` is `CONTRACTION`: the `⊗` lift `Assign` sits
  in the loop body and the additive fold's degenerate `Carrier` rides the loop. The shared
  builder `ops.contraction_loop(lift, fold, operand_bodies, reduce_axis)` builds it in the
  recognizable `Accum`-in-`Loop` form (used by flash's score producer and the scalar matmul).

`ir.tile.ops.lower(op)` is now just the `Map`'s body verbatim — the carriers were already
dissolved into loose fold `Accum`s (and the streaming `merge` for a twisted carrier) at
recognition, and the reduce `Loop`s carry their role/carrier annotations, so one `lower`
call emits the kernel's per-cell body with nothing left to expand.

A reduce is a contraction not by "two loads" but by the genuine algebra — the lift ⊗
**distributes over** the fold ⊕ (`multiply` over `add`; *not* `add` over `add`, a sum of two
operands) and contracts ≥ 2 distinct operand buffers (`x·x` is a squared reduce, not a
contraction). Recognition stamps the `CONTRACTION` role on that form (keeping the matmul's
`Accum` a loose `Accum` rather than degenerate-folding it like a plain reduce);
`020_schedule` gates flash structurally (a reduce loop nested inside a reduce loop); the mma
atom tier reads the operands off the annotated loop to pick the tensor-core cell.

`Carrier` (`ir/stmt/algebra.py`) is the carrier **algebra** of a reduce — its carried
`State` plus the ψ-conjugated `Twist` combine — decoupled from any loop position. It is NOT
a `Stmt` and carries no `partial` / `axis`; a reduce `Loop` carries one (`loop.carrier`) so
the streaming / cooperative / cross-CTA materializers read the combine off the loop. It
derives the streaming fold (`merge`), the cross-partition fold (`combine_states`), the
degenerate `Accum` form (`as_accums`), and the one-shot cross-partition combine
(`as_state_merge` → a renderable `StateMerge` stmt) from `(state, twist)`. A *degenerate*
carrier (the `id` twist) is a plain `sum`/`max`/`mean` reduce; a *twisted* one (`exp`) is
online-softmax / flash; a contraction's carrier is the degenerate carrier of its additive
fold. `Accum.as_carrier()` is that additive `Accum` AS the degenerate 1-component `Carrier`
it already is. `State` bundles the internal-state SSA `names`. The neutral element (seed) is
NOT stored on `State` — a carrier dissolves into its fold `Accum`s (`Carrier.as_accums`) and
each fold's seed is its `op.identity`, so there is one source of truth for the seed.
`State.other` is the second-operand names `"<n>__o"` the cross-partition combine reads.

**The `Twist` — the part that varies, generated not hand-authored.** Transport of structure: a
monoid `(·, e)` conjugated by a bijection ψ gives the twisted combine `x ⊕ y = ψ(ψ⁻¹(x) · ψ⁻¹(y))`.
The carrier algebra above is shared; ψ is the twist, carried on `Carrier.twist` **as data** in one of
two modes. In **SPEC mode** the twist is a name-free `(family, channels)` — a tuple of `Channel`s
(`fold` ⊕, `term`, `lift` ⊗) — and the combine *programs* are GENERATED on demand by the
mode-dispatching `Carrier` accessors `carrier.merge` (streaming fold) / `.combine_states` (cross-partition
state⊕state, reading the `state_b` operand `"<n>__o"`) / `.state_b`. Generation (`ir/stmt/carrier.py`)
builds the naive `ψ∘base∘(ψ⁻¹×ψ⁻¹)` combine — associativity inherited from the base monoid for free —
then a per-family stabilizer rewrites it to the numerically-stable form (distribute the ψ-rescale,
fuse exponentials, fold identities, DCE/CSE) and a structural certificate asserts every surviving
`exp` has a `≤ 0` argument. `family="exp"` is the online-softmax / flash log-sum-exp carrier (built
from `pivot`/`denom`/`expect` channels — softmax is flash minus the `expect` channel); `family="id"`
is the degenerate identity twist (a plain reduce, `Accum.as_carrier`). In **BOUND mode**
(`family is None`) the programs are stored verbatim — used only by `as_state_merge(other)`, the
one-shot finalize `StateMerge` whose `merge` IS its `combine_states` (regenerated with temps keyed on
`other[0]`), so a two-partition merge renders through the same machinery as a streaming step. **Example** — flash
attention's online softmax (the log-sum-exp carrier): state `(m, l, O)`, partial `(score, value)`,
identity `(−inf, 0, 0)`, merge `m_new=max(m,s); alpha=exp(m−m_new); l=l·alpha+exp(s−m_new);
O=O·alpha+exp(s−m_new)·v; m=m_new`.

### `loop/ir.py` — LoopOp types

| Symbol                       | Role                                                                                                              |
|------------------------------|-------------------------------------------------------------------------------------------------------------------|
| `Axis`                       | Named iteration variable (`name`, `extent`). Defined in `ir/axis.py`, re-exported here. Carries optional `source_axis` (pre-split origin) and `real_extent` (pre-ceil-div bound for masked tiles — block axis covers `ceil_div(real_extent, BN·FN)`; materializer reads it to gate boundary lanes). Both excluded from equality / hashing. |
| `LoopOp`                     | One kernel. Stored field: `body` (nested `Loop` tree). Computed: `axes`, `loads`, `accums`.                       |
| `Load`                       | Body-form external read: `name = load(input)[index...]`. `input` matches the producing graph node's id.           |
| `Assign`                     | SSA body stmt: `name = op(args)` with `op: ElementwiseImpl`.                                                      |
| `Accum`                      | Reduce accumulator: `name = op(name, value)` inside a reduce `Loop`. Initialized to its op's identity. ``axes`` lists the reduction axis names — propagated through Sigma renames (including σ-splits via `Expr.free_vars()`); the escape-analysis helper derives cross-thread cooperativity from ``axes ∩ enclosing ThreadTile.axes``. |
| `Init`                       | Explicit `<dtype> name = identity;` seed at this scope (`name` + scalar `identity` + `dtype`). Used for a `Carrier`'s state (one per `State` component, via `State.inits()`), emitted above the streaming `Loop`. Scope-bound (never hoisted); shadows a deeper same-named `Accum` init. |
| `Write`                      | Write an SSA value to output at `index`.                                                                          |
| `Select` + `SelectBranch`    | Coord-predicated binding (replaces the old Mux).                                                                  |
| `Loop`                       | Serial iteration block: `axis` + nested `body`.                                                                   |
| `StridedLoop`                | Strided iteration (`start`, `step`) — cooperative thread-stride loop reused by Tile/Kernel IR.                    |
| `Cond`                       | If/else block over an `Expr` predicate.                                                                           |
| `Stmt`                       | Base class — every body statement subclasses it. Leaves and control-flow nodes live in `ir/stmt/`.               |

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
the shared substrate behind the rules that slice cones (the demoted-operand producer cut in
`lowering/tile/030_split`) — eligibility judgments stay in the rules, per `pipeline/passes/ARCHITECTURE.md`. The
`classify_fragment_epilogue` walk (`ir/stmt/algebra.py`) deliberately does NOT use it: it is a single pass
interleaving reduce-scope flags with its negative-form blocker reporting, a different operator than the cone's
any-dep taint.

`rewrite` has two distinct rename channels that must stay disjoint:
`rename_ssa` carries **SSA-name** renames, `sigma` carries **axis**
substitutions. `Load`/`Write` index exprs apply *both*
(`_rename_ssa_vars_in_expr(sigma.apply(e), rename)`) so an indirect
(gather) index Var gets renamed exactly once. Putting the same name in
both maps renames it twice — and if the two passes form a chain (e.g.
`x → in5` and a pre-existing `in5 → in26`) the double application
collapses it transitively, silently wiring a gather to the wrong row.

### `ir/stmt/normalize.py` — structural canonicalization

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

### `ir/expr.py` — Expr simplification

`simplify` (called inside `normalize_body`). Generic bottom-up Expr rewriter:
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

### `loop/runner.py` — C++ JIT executor

`execute_loop_op_cpp(loop, input_arrays, out_shape) → ndarray` renders the
LoopOp body to a C++ source string and JIT-compiles it in-process via cppyy /
Cling (cached by the rendered source), then calls it with raw pointers to the
input arrays. Powers `LoopOp.forward` — so post-fusion graphs run through the
default `Backend.run` topo-walk like any pre-fusion graph.

### `loop/builder.py` — fluent construction

`LoopBuilder` helper used by decomposition/fusion tests to construct
LoopOp bodies without spelling out every `Loop(Axis(…))` nest.

## `tile/`

Tile IR (`tile/ir.py`, `tile/schedule.py`, `tile/ops.py`) carries the scheduling decisions on a typed `Kernel`
rather than re-deriving them from the body. A `TileOp` holds exactly one `Kernel` — the op-tree `Map`
(from `ir/stmt/algebra.py`, a thin `Body` wrapper over the annotated loop nest) paired with one kind-free
`TileSchedule` (`tile/schedule.py`); a kernel's structure is read off its annotated reduce loop's `AxisRole`
(`ops.axis_role`), not a Python type. The schedule holds the free→grid `Placement` and, for a reduction, the `ReducePlan`.

`ReducePlan` (`tile/schedule.py`) is a list of `ReduceStage`s, one per hardware `Level` the reduce axis is
partitioned across, coarse→fine: `GRID` (split-K across CTAs), `BLOCK` (cooperative threads within a CTA), `REG`
(ILP register-fold), `SERIAL` (the per-thread remainder). The per-level combine `Fold` (`SHFL` lane butterfly /
`SMEM` block tree / `ATOMIC` cross-CTA finalize) is **derived** from the level (`ReduceStage.combine`), not stored
or tuned. The single `REDUCE` codec knob decides the plan in `020_schedule`; the combine itself stays in the op
tree.

All four schedule codecs — `REDUCE`, `TILE` (scalar or warp `TilePlan`), `STAGE`, and `WSPEC` — share one
schema-driven ser/de engine (`tile/codec.py`): a `Schema` of typed `Field`s plus generic `desugar` / `decode` /
`encode`. Each codec class keeps its `parse` / `spell` API and its semantics, delegating only the string ↔ struct
conversion to the engine, so the featurizer and `020_schedule` call sites — and the on-disk golden wire format — are
unchanged. The grammar collapses int and pair widths into one tuple kind and supports per-field params (the recursive
`WSPEC` role case); the one non-uniform value codec is the `REDUCE` `g<n>[a|k]` finalize letter, kept inside the value
so the round-trip stays byte-identical.

`WSPEC` (warp specialization) is the worker-mapping pin — a role→warp-count allocation (`WarpSpec`; role descriptors in
`tile/role.py`, the COMPUTE consumer implicit and sized by `TilePlan.units`) carried on an **orthogonal**
`workers: WarpSpec | None` field of the uniform schedule (`None` = uniform SIMT), **not** a union arm: it adds a warp
split over the fixed pipeline rather than replacing it. Pin-only this cut — `020_schedule` stamps `workers` from a
`DEPLODOCK_WSPEC` pin (gated on a warp `TILE` + a `STAGE`, since the producer needs a load half to drive); the
producer/consumer codegen in `lowering/kernel` is a documented `TODO(warp-spec)`.

`tile/ops.py` `lower(op)` returns the `Map`'s body verbatim — the loop nest with its annotated reduce `Loop`s, the
carriers already dissolved into loose folds + the streaming `merge` at recognition; `pretty(op)` renders it for
dumps. The tensor-core,
cooperative-combine, and staging (cp.async / TMA) tiers are materialized downstream in `lowering/kernel` against the
op tree + schedule; warp specialization has its schedule codec (`WSPEC` → `workers`) but its producer/consumer codegen
is still a `TODO(warp-spec)`. The older tile-level `GridTile` / `ThreadTile` / `Stage` structures were removed in the
tile-IR rebuild and are being rebuilt there as the schedules return (see `pipeline/passes/ARCHITECTURE.md`).

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
| `RegFragment`      | mma.sync (s16816) per-thread register array decl (one per operand role `"a"`/`"b"`/`"c"`): `unsigned a[4]`/`b[2]` (16-bit operands, 2 elems/reg) or `float c[4]` (f32 acc, zero-init at decl — no separate fill). Carries the cell shape `(M, N, K)` + dtype. Emitted by the MMA lowering pass. The sole tensor-core fragment family (the opaque `nvcuda::wmma` nodes were removed). |
| `LdmatrixLoad`     | Load one operand into a `RegFragment`. `staged=True` (default): `ldmatrix.sync.aligned.m8n8.x{4,trans}.b16` from smem (`role="a"` → x4; `role="b"` → x2.trans; each lane derives its row address from `threadIdx.x & 31`; `swizzle` applies the per-lane chunk XOR for a TMA-swizzled slab). `staged=False`: operand not staged into smem (ldmatrix is smem-only) → renders a **gmem-direct fragment load** (`dpl_mma_load_{a,b}_gmem`) reading the fragment straight from gmem with the same m16n8k16 lane→element map — slower (no smem reuse) but lets an unstageable MMA tile compile instead of crashing. `b_trans=True` (role "b" only) marks a transposed-B operand stored `[N, K]` (the native `mma.row.col` col-major B — a Q@K^T cell): gmem-direct via `dpl_mma_load_b_gmem_trans` (k contiguous; masked → `_trans_nclamp`). |
| `MmaSyncPtx`       | `mma.sync.aligned.m16n8k16.row.col.f32.{f16,bf16}.{f16,bf16}.f32` — one s16816 MMA via inline PTX (`c += a @ b`). `ab_dtype` (`"f16"`/`"bf16"`) picks the `dpl_mma_…` wrapper. |
| `RegStore`         | Per-lane epilogue store of the f32 `c[4]` accumulator to the output (no `store_matrix_sync` for mma.sync) — direct for f32 dst, `__float2half` downconvert for f16. Optional `epilogue` (a `RegEpilogue`: leaf `EpilogueLoad`s with per-dim `m`/`n`/`fixed` roles + `(name, op, args)` chain in topo order, plus `selects` — coord-predicated causal-mask ternaries) carries a fused pointwise chain — residual adds, bias/scale broadcasts, activations, the causal attention mask — evaluated per element in f32 at the element's own (row, col), leaves loaded at each buffer's own dim stride, ops rendered via `op_to_expr` (folded in by the MMA lowering pass after the shared negative-form gate `classify_fragment_epilogue` (`ir/stmt/algebra.py`) admits the slice; leaf buffers declared via `external_reads` so they stay in the kernel signature after their scalar Loads are stripped). Each `selects` entry `(name, ((cond|None, value), …))` renders as a per-element ternary, its `__M__`/`__N__` placeholder coords substituted with the element's absolute (row, col). |
| Shared from `tile` | `Tile` (launch geometry); from `ir/stmt/`: `Loop`, `StridedLoop`, `Load`, `Assign`, `Accum`, `Write`, `Select`, `Cond`. |

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
