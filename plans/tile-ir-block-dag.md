# A new Tile IR: a block DAG where scheduling is annotation over an invariant algorithm

Redesign the Tile IR so that **staging, pipelining, warp-specialization, register tiling, split/cooperative-K, and
block placement are all the same kind of operation** — and that operation is a **Schedule annotation over an invariant
algorithm**, applied by a single deterministic `assemble` step. The algorithm is a DAG of compute blocks; everything
that today lives in the tree shape (binding hierarchy, smem staging, pipeline structure, warp roles) becomes either a
**derived view** of the body or an entry in the **Schedule**. Nothing is stored that the body already implies.

This follows the codebase's own discipline. `Loop.algebra_kind` and `iter_dag` are already "computed on demand… a
derived cache, not a second source of truth… never enters equality / `op_cache_key`, always consistent with the body."
The new IR extends that to the whole tile phase: `reads`, `writes`, `carrier`, `atom`, and the edge topology are all
*derived from the body + connectivity*, never stored as attributes that could drift.

The IR below is the result of three co-design rounds that re-expressed every current tile pass against it; each
pass's rewrite is in "Pass rewrites." The earlier strawmen and round-by-round feedback are not reproduced — only the
converged result.

## Why the current Tile IR fights us

Today's Tile IR (`ir/tile/ir.py`) is a **tree**, and nesting is overloaded to mean four different things at once:

- **binding hierarchy** — `GridTile > ThreadTile > RegisterTile > SerialTile` nesting *is* the blockIdx/threadIdx/
  register/loop assignment;
- **dataflow** — a `StageBundle` *wraps* its consumer body, so "A feeds B" is encoded as "A's bundle contains B";
- **staging** — smem lifetime is the structural extent of `StageBundle.body`;
- **pipeline structure** — `080_pipeline_stages` peels prologue/steady/epilogue by *replicating* `StageBundle`s and
  splicing `AsyncWait` Stmts into the tree; `085_warp_specialize` splits the body into a producer/consumer `Cond`.

Because the schedule lives *in the tree shape*, every scheduling pass is bespoke tree surgery that re-derives structure
it shouldn't have to. The pass map bears this out — `015/017/020/021/026/030/040/050/060/080/085` are all
dataflow/scheduling rewrites; only `011/025/070/090` are genuinely local. And the surgery has brittle, undocumented
ordering lore: `080` requires `060`'s `ASYNC` policy and `pipeline_depth=1`; `085` requires `080`'s `depth=2`; `021`
must run after `020` but restructure the mask `Cond` `020` was blind to. Each pass re-pattern-matches the tree, mutates
it, and hopes the next pass's matcher still fires.

The root cause: **the invariant (what depends on what) and the variant (how it's scheduled) are tangled in one
structure, and large parts of that structure duplicate the body.** Rearranging the variant means rewriting the thing
that also holds the invariant.

## The model: algorithm (derived views) + Schedule

Three strata, with a strict rule about where each piece of information lives:

- **Algorithm** — the *invariant*. A DAG of `Block`s; each block is `name + domain + compute`, where `compute` is the
  scalar Loop-IR body (`Load`/`Assign`/`Select`/`Write`/`Accum`/`Mma`/`Monoid`) over logical buffers. This is the single
  source of truth. Only the **algebra-/dependency-changing** moves touch it — `tile_axis` (σ-split an axis),
  `partition_reduce` (insert a combine block), `atomize` (fuse a cell to `Mma`).
- **Derived views** — *projections of the algorithm*, computed on demand, never stored: `Block.reads`/`writes`
  (`AccessMap`s read off the body's `Load`/`Write` index exprs), `Block.carrier` (the `ReduceCarrier` in the body),
  `Block.atom` (the `Mma`'s atom), `Carrier.kind`/traits/`mask`, and the **edge topology** (`A` writes `X`, `B` reads
  `X`). Because they are computed, they cannot drift and they do not enter `op_cache_key`.
- **Schedule** — the *variant*: every scheduling choice, keyed by block / axis / read-site. `binding`, `scope`, `role`,
  `launch`, `staged` (+transport), `distance`, `ring_depth`, `cohort`, `pad`, `reg_budget`, `unroll`, `grid_swizzle`.
  The scheduling moves only edit the Schedule; they never touch the body. `assemble` applies the Schedule to the
  algorithm and emits today's `KernelOp` tower — so the migration oracle is **byte-identical emitted CUDA**.

The sharp consequence: **smem slabs, cooperative producers, pipeline peels, warp-spec `Cond`s, and combine kernels do
not exist in the IR.** They are *synthesized by `assemble`* from the Schedule. `stage(A)` is the annotation
`staged[(A→mm)] = SYNC`; `assemble` materializes the slab + the cooperative load. Two reads of the same buffer at the
same access collapse to one slab at `assemble` — so sibling-stage dedup (`026`) is automatic and has no move at all.

## The IR

Minimal frozen-dataclass classes. The body is stored; the projections are properties; the choices are in `Schedule`.

```python
from __future__ import annotations

import enum
from dataclasses import dataclass

from deplodock.compiler.dtype import DataType
from deplodock.compiler.ir.algebra import classify_algebra
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.stmt import Body, ReduceCarrier
from deplodock.compiler.ir.tile.ir import Atom


class Space(enum.Enum):
    GMEM = "gmem"
    SMEM = "smem"  # only ever an assemble artifact (a staged slab); never a stored Buffer
    REG = "reg"


class Binding(enum.Enum):
    GRID = "grid"  # blockIdx        — scope-creating
    SERIAL = "serial"  # for-loop     — scope-creating
    WARP = "warp"  # warp_id          — replication
    THREAD = "thread"  # threadIdx    — replication
    REGISTER = "register"  # unrolled cell — replication
    ATOM = "atom"  # one tensor-core cell — non-addressable (excluded from AccessMap)


class Transport(enum.Enum):
    SYNC = "sync"
    CPASYNC = "cpasync"  # sm_80+
    TMA = "tma"  # sm_90+


class Role(enum.Enum):
    PRODUCER = "producer"
    CONSUMER = "consumer"


class AddrKind(enum.Enum):
    AFFINE = "affine"  # source_index[d] = offset[d] + Σ_{i: dims[i]==d} (composite-stride)·Var(axes[i])
    TEMPLATE = "template"  # verbatim coords, domain vars symbolic (collapsed reshape `/`,`%`)


@dataclass(frozen=True)
class AccessMap:
    """A DERIVED value: how one `Load`/`Write` in a body indexes one buffer.
    Produced by classifying the leaf's index Expr (today's `_classify`); not
    stored on blocks. AFFINE carries the structure `assemble` needs to size
    slabs, decide TMA box-eligibility, pick a swizzle, and clamp."""

    kind: AddrKind
    axes: tuple[str, ...] = ()  # domain axes indexing this buffer (AFFINE)
    dims: tuple[int, ...] = ()  # axes[i] -> source dim
    block: tuple[int, ...] = ()  # per-axis atom-cell stride multiplier
    offset: tuple[Expr, ...] = ()  # per-source-dim CTA-uniform anchor
    template: tuple[Expr, ...] = ()  # TEMPLATE: verbatim source coords
    clamp: tuple[Expr | None, ...] = ()  # per-source-dim safe-read bound — itself derived from the gmem Buffer.shape

    def free_axes(self) -> frozenset[str]: ...  # the domain axes this access depends on (drives hoist legality)


@dataclass(frozen=True)
class Port:
    """A DERIVED dataflow endpoint: (buffer, AccessMap) read off one body leaf."""

    buffer: str
    access: AccessMap


@dataclass(frozen=True)
class Carrier:
    """A DERIVED view of a folding block's reduce algebra — the legality oracle
    for the reduce-restructuring moves. `kind`/traits come from
    `classify_algebra`; `mask` (the symbolic-K identity-fill bound) is read off
    the block's domain. Nothing here is stored: it is recomputed from the body
    + domain, like `Loop.algebra_kind`."""

    carrier: ReduceCarrier
    mask: tuple[str, Expr] | None = None  # (reduce-axis, runtime bound) — derived from a symbolic reduce axis

    @property
    def kind(self):
        return classify_algebra(self.carrier)


@dataclass(frozen=True)
class Block:
    """A DAG node: the algorithm at one compute site. STORED state is only
    `name`, `domain`, `compute`. Everything else is a projection of `compute`
    (+ domain), computed on demand — so it can never drift and never enters
    `op_cache_key`."""

    name: str
    domain: tuple[Axis, ...]  # iteration axes (extent / real_extent / symbolic) the body references
    compute: Body  # the scalar algorithm over logical buffers — THE source of truth

    @property
    def reads(self) -> tuple[Port, ...]: ...  # Load leaves of compute → (buffer, AccessMap)
    @property
    def writes(self) -> tuple[Port, ...]: ...  # Write leaves
    @property
    def carrier(self) -> Carrier | None: ...  # the ReduceCarrier in compute (+ derived mask/kind), else None
    @property
    def atom(self) -> Atom | None: ...  # the Mma's atom once atomized, else None


@dataclass(frozen=True)
class Buffer:
    """A LOGICAL value-store: a kernel input/output or an inter-block
    intermediate. SMEM slabs are not Buffers — they are assemble artifacts of a
    `staged` annotation. `pad` is a schedule property of the slab, not here."""

    name: str
    shape: tuple[Expr, ...]
    dtype: DataType
    space: Space = Space.GMEM


# Edge is a DERIVED value (not stored): one per (producer-or-input, consumer, buffer) from the body's buffer def-use.
@dataclass(frozen=True)
class Edge:
    src: str  # producer block name, or an input Buffer name
    dst: str  # consumer block name
    buffer: str


@dataclass(frozen=True)
class Schedule:
    """The variant — every scheduling choice. The scheduling moves edit only
    this; `assemble` applies it to the algorithm. Staging keys are read-sites
    (the derived `Edge`); a read absent from `staged` is gmem-direct."""

    binding: dict[str, Binding]  # axis -> hardware role (applies to a block only when axis ∈ block.domain)
    scope: dict[str, tuple[str, ...]]  # block -> enclosing nest override (default = max-hoist, derived from reads.free_axes)
    role: dict[str, Role]  # block -> producer/consumer (warp-spec; absent = unspecialized)
    launch: dict[str, int]  # block -> launch group (one group = one kernel)
    staged: dict[Edge, Transport]  # read-site -> SMEM fill transport (presence = staged)
    distance: dict[Edge, tuple[tuple[str, int], ...]]  # read-site -> per-serial-axis retiming offset
    cohort: dict[Edge, int]  # read-site -> barrier / pipeline / transport cohort id
    ring_depth: dict[Edge, int]  # staged read-site -> ring slots; INVARIANT >= max(distance)+1
    pad: dict[Edge, tuple[int, ...]]  # staged read-site -> bank-conflict pad of its slab
    reg_budget: dict[Role, int]  # warp-spec register redistribution (SetMaxNReg)
    unroll: dict[str, bool]  # SERIAL axis -> #pragma unroll
    grid_swizzle: dict[str, int]  # GRID block -> L2 row-group remap


@dataclass
class TileGraph:
    """The new Tile IR. `assemble(TileGraph) -> KernelOp | Graph[KernelOp]`
    (one kernel per launch group). The edge topology is derived."""

    name: str
    buffers: dict[str, Buffer]  # logical only (inputs / outputs / intermediates)
    blocks: tuple[Block, ...]
    schedule: Schedule

    @property
    def edges(self) -> tuple[Edge, ...]: ...  # buffer def-use across blocks (+ input-source edges)
```

### Derived vs stored, and which moves touch the body

| Information                                                                                                 | Where it lives                                                         |
|-------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `Block.reads`/`writes`, `AccessMap` structure, `clamp`                                                      | **derived** from `compute`'s `Load`/`Write` index exprs (+ gmem shape) |
| `Block.carrier`, `Carrier.kind`/traits/`mask`                                                               | **derived** from `compute` (the `ReduceCarrier`) + `domain`            |
| `Block.atom`                                                                                                | **derived** from `compute` (the `Mma`'s atom)                          |
| edge topology                                                                                               | **derived** from buffer def-use across blocks                          |
| `Block.{name, domain, compute}`, `Buffer.{name, shape, dtype, space}`                                       | **stored** (intrinsic identity / the body itself)                      |
| `binding, scope, role, launch, staged, distance, ring_depth, cohort, pad, reg_budget, unroll, grid_swizzle` | **stored** in `Schedule` (the choices)                                 |

This yields a crisp rule. **Body-mutating moves** (the algebra-/dependency-licensed ones): `tile_axis` (σ-split — a
dependency-preserving index rewrite), `partition_reduce` (insert a combine block — `commutative`), `atomize` (fuse a
cell to `Mma` — `SEMIRING`). **Annotation moves** (touch only `Schedule`): `stage`, `retime`, `promote_transport`,
`specialize_warps`, `hoist` (sets `scope`), `pad`/`swizzle`/`unroll`. Everything a scheduling move once did by tree
surgery is now a dict write the body never sees.

The payoff lands on the old "canonical form for caching" worry: **`op_cache_key` = `canonical(compute bodies + edge
topology) + Schedule`.** The derived projections never enter the key (same reason `algebra_kind` doesn't today), so the
canonical-form surface is just the bodies and the choices.

**Mechanics note.** `functools.cached_property` does not compose with `frozen=True` (it needs to write `__dict__`). Use
a plain `@property` that recomputes — these are compile-time structures and this is the `algebra_kind` precedent — and
add a frozen-safe memo (an `object.__setattr__` descriptor or a side `WeakKeyDictionary`) only where profiling demands.
The one derivation worth caching is `AccessMap` (the `_classify` affine analysis is heuristic and falls back to
`TEMPLATE`).

## An operation in the IR

A 256×256×256 tiled matmul `C = A @ B` (fp16 in, fp32 accumulate). The algorithm is **one block** reading logical gmem
`A`, `B` and writing `C` — there are no slab buffers and no stage blocks. Staging A and B through smem is two `Schedule`
annotations; `assemble` synthesizes the slabs + cooperative loads. Pretty-printed (`reads`/`writes`/`edges` shown as the
derived projections they are):

```
tilegraph k_matmul:                                    # C[256,256] = A[256,256] @ B[256,256]
  buffers:  A gmem f16 [256,256]   B gmem f16 [256,256]   C gmem f32 [256,256]    # logical only

  axes (binding):
    M_b:4  N_b:4   grid          M_t:16 N_t:16  thread          M_r:4 N_r:4  register
    K_o:16 serial                K_i:16 serial

  block mm:
    compute   acc += A[M_b*64+M_t*4+M_r, K_o*16+K_i] * B[K_o*16+K_i, N_b*64+N_t*4+N_r]
              C[M_b*64+M_t*4+M_r, N_b*64+N_t*4+N_r] = acc
    derived:  reads   = { A : affine, B : affine }          # off the two Loads
              writes  = { C : affine }
              carrier = SEMIRING(+) over (K_o, K_i)         # off the Accum
              edges   = { A→mm, B→mm }                      # input-source reads

  schedule:
    staged       (A→mm)=SYNC   (B→mm)=SYNC                  # ← the only staging state
    cohort       (A→mm),(B→mm) → 0                          # one __syncthreads
    ring_depth   (A→mm)=1  (B→mm)=1
    grid_swizzle mm=8
```

Every transformation is now an edit to the `schedule` block, leaving `mm.compute` untouched:

- **double-buffered cp.async**: `distance{(A→mm),(B→mm)} = ((K_o,1))`, `ring_depth = 2`, `staged = CPASYNC`. `assemble`
  peels the prologue/steady/epilogue + waits.
- **tensor cores**: `atomize(mm)` — the one body edit (bind a `16×8×16` ATOM cell, fuse the cell to `Mma`); `Block.atom`
  then *derives* from that `Mma`.
- **split-K across CTAs**: `partition_reduce(K_o, GRID)` inserts a combine block (body edit, `commutative`-licensed);
  `assemble` synthesizes the atomic add or the second-kernel reduce from `binding`/`launch`.

## The moves — one structure, two legality checks

**Dep** = preserves the edge topology; **Alg** = a carrier-trait query (`partition/decompose.py::legal_decomps`,
generalized). "Touches" = whether the move edits the body (algorithm) or only the `Schedule`.

| Move                                    | Effect                                                            | Touches   | Legality                                           | Subsumes today                                    |
|-----------------------------------------|-------------------------------------------------------------------|-----------|----------------------------------------------------|---------------------------------------------------|
| `tile_axis(A → A_b·A_t·A_r·…)`          | σ-split an axis; rewrite the body's index exprs                   | body      | Dep (map) / **Alg `associative`** (reduce)         | `TileMap`/`TileSerial`, `_split_free_axis`        |
| `partition_reduce(K, GRID\|THREAD)`     | insert a combine `Block`; bind the partition factor               | body      | **Alg `commutative`** (+ `has_identity` if masked) | split-K (`015`/`017`), cooperative-K              |
| `atomize`                               | fuse `[Load,Load,mul,Accum]`→`Mma` (atom cell)                    | body      | **Alg `SEMIRING`** + atom eligibility              | `AtomTile` + `011_lower_atom_cell`                |
| `bind(axis, role)`                      | `Schedule.binding[axis] = …`                                      | schedule  | Dep + resource budget                              | `_wrap_tower` Role→flavor                         |
| `register_tile`                         | `bind(axis, REGISTER)` → assemble replicates per cell             | schedule  | Dep + cell budget                                  | `010_split_register_axes` + `RegisterTile`        |
| `stage(read)`                           | `Schedule.staged[edge] = SYNC` → assemble synth slab + producer   | schedule  | Dep + reuse across a parallel axis                 | `020_stage_inputs` (+ `026` dedup is automatic)   |
| `hoist(block)`                          | `Schedule.scope[block] = …` (default = max-hoist, derived)        | schedule  | Dep                                                | `030` + prologue/epilogue/`leading`/`mid`         |
| `retime(read, +k along S)`              | `Schedule.distance[edge]`, raise `ring_depth`                     | schedule  | Dep + `ring_depth ≥ k+1`                           | `040_use_ring_buffers` + `080_pipeline_stages`    |
| `promote_transport(read, →CPASYNC/TMA)` | `Schedule.staged[edge] = …`                                       | schedule  | Dep + cc gate; TMA needs box-affine map            | `050_use_tma`, `060_use_async_copy`               |
| `specialize_warps(2-coloring)`          | `Schedule.role[block] = …`                                        | schedule  | Dep (cut antichain) + crossing edges async/TMA     | `085_warp_specialize`                             |
| `pad/swizzle/unroll`                    | `Schedule.pad`/`grid_swizzle`/`unroll`                            | schedule  | perf-only                                          | `070`, `025`, `090`                               |
| `guard`                                 | derived from `domain.real_extent` vs tile (not a move)            | derived   | —                                                  | masked-tile `Cond` / `real_extent` / `kmask`      |

This table catalogs the scheduling *operations* and their legality; it is orthogonal to *where each runs*. Of the
perf-only row, only `swizzle` is a genuine enumeration fork; `pad` and `unroll` are fixed-logic (a formula / a threshold)
and run as deterministic post-`assemble` passes, not search moves — the fork-vs-fixed-logic split and pass placement live
in "The pass structure."

The two headline consequences from the original design hold and are now literal: **prologue/epilogue are not special**
(a fused-prologue reduce is a block whose reads are {M,K}; its default `scope` ends at `M_scope`, outside the N register
tier — no special-case builder), and **pipelining / warp-spec stop being hand-peeled** (the peel / `Cond` / mbarrier
ring is emitted deterministically from `distance` / `role` by the materialization stage — the `peel` post-pass +
`assemble` — never an enumeration fork interleaved with the search).

## The pass structure: enumeration passes + one `assemble`

The model above resolves into two *sides* split at `assemble`. The whole tile phase is: a deterministic `tile/build`
seeds the root tile node; a pipeline of **enumeration passes** (genuine forks) widens the search tree move-family by
move-family — the **search side**; then the deterministic **materialization stage** lowers the chosen leaf — the
`assemble` core plus the fixed-logic post-passes (`peel` / `mask_order` / `pad` / `unroll`). Only the enumeration side
branches the MCTS; `build`, `assemble`, and the post-passes are all choice-free. This is the concrete realization of "the
scheduling moves only edit the Schedule; the tower is built deterministically from them." (In the pass-rewrite and
lowering sections below, "`assemble`" often names this whole materialization stage; where a specific fixed-logic
post-pass does the work — `peel`, `mask_order`, `pad`, `unroll` — it is the one named in "Directory layout.")

One question decides where a family lives, and it dominates every other: **is it a genuine fork?** — does it offer ≥2
ranked candidates the search must *bench* to choose between (a tile size, a ring depth, transport on/off, specialize
on/off)? Enumeration exists to hand the MCTS something to branch on; a family that has nothing to rank must not be an
enumeration pass, because it would plant a tree node with a single child — pure overhead that contaminates the search.

- **Genuine forks → enumeration passes, all pre-`assemble`.** They *must* be pre-`assemble`: the search explores them
  before any leaf is materialized. Every genuine fork's legality reads only derived projections + the current `Schedule`,
  so they all sit pre-`assemble` with no exception.
- **Fixed-logic (a deterministic rule — a threshold, a bank-conflict formula, a σ-peel driven by an already-chosen
  depth) → NOT an enumeration pass.** It runs as a **deterministic post-`assemble` pass** (the default — a
  `KernelOp`→`KernelOp` total function that keeps `assemble` minimal), or **inside `assemble`** when it is a
  by-construction property of materialization (e.g. slab coalescing).

So there is **no post-`assemble` enumeration**: post-`assemble` is deterministic-only. The materialization side is one
deterministic *stage* — `assemble` (core) followed by the deterministic post-passes — and the search tree sits entirely
in front of it. This is the load-bearing discipline: **enumeration is for genuine forks; fixed logic stays out of it.**

- **Enumeration passes** — *generate subtrees for the search.* Each owns one move family. It matches the tile node,
  computes its legal offers from the **derived projections + the current `Schedule`** (never from tower shape), and
  returns a `Fork` whose children apply that family's moves — each child a new tile node with an edited `Schedule` (or,
  for the three body moves, edited `Block`s). An enumeration pass **never emits a `KernelOp`**; it only widens the tree.
  This is the existing `moves.py`/`tree.py` model, unchanged, now spanning the whole tile phase rather than ending at the
  partition head.
- **One materialization pass — `assemble`** — *materializes the chosen leaf.* It matches a fully-scheduled tile node and
  deterministically lowers `(algorithm + Schedule) → KernelOp | Graph[KernelOp]`. It is **total and choice-free**: every
  decision already lives in the leaf, so `assemble` makes none. It is the **only** pass that emits the tower.

**The inter-pass IR is the tile node, not the tower.** Today `010` resolves its `Fork` to a `TileOp` and the (now
deleted) `020`–`090` rewrote that tower; the brittle ordering lore — "did the previous pass leave the tree in the shape
my matcher wants" — was a direct consequence of scheduling living in tree shape. In the new structure every enumeration
pass matches and returns the **same stable `TileGraph`-bearing node**, so the pass order is just the MCTS decision order:
`retime` runs after `stage` because `retime`'s offer set reads a `staged` edge — an explicit fact on derived state, not a
matcher coincidence. (The plumbing reuses the engine unchanged: a tile node is a graph node whose op payload is a
`TileGraph`; a `Fork` resolves to a new such node, exactly as `010`'s `Fork` resolves to a `TileOp` today.)

Concrete pass list — pipeline order *is* decision order:

| pass                    | kind              | move family                          | edits                             | predecessor (deleted / legacy)         |
|-------------------------|-------------------|--------------------------------------|-----------------------------------|----------------------------------------|
| `tile/build`            | deterministic     | — derive algorithm DAG from `LoopOp` | seeds node + reference `Schedule` | `010` front half (`build_dag.py`)      |
| `tile/tensorize`        | enum (body)       | `atomize`                            | `Block.compute`                   | `011` + warp tower                     |
| `tile/partition_reduce` | enum (body)       | split-K / cooperative-K              | insert combine `Block`            | `015`/`017`                            |
| `tile/register_tile`    | enum (schedule)   | `register_tile` = `bind(REGISTER)`   | `binding`                         | `010` reg split                        |
| `tile/stage`            | enum (schedule)   | `stage`, `hoist`                     | `staged`, `scope`                 | `020`/`021`/`026`/`030`                |
| `tile/retime`           | enum (schedule)   | `retime`                             | `distance`, `ring_depth`          | `040`/`080`                            |
| `tile/transport`        | enum (schedule)   | `promote_transport`                  | `staged` value                    | `050`/`060`                            |
| `tile/warp_spec`        | enum (schedule)   | `specialize_warps`                   | `role`, `reg_budget`              | `085`                                  |
| `tile/swizzle`          | enum (schedule)   | `swizzle`                            | `grid_swizzle`                    | `025`                                  |
| `tile/assemble`         | **materialize**   | apply `Schedule` → basic tower       | TileGraph → KernelOp              | the tower builders + `assemble_block`  |
| `tile/peel`             | **det. (post)**   | pipeline σ-peel from `ring_depth`    | KernelOp serial nest              | `080`                                  |
| `tile/mask_order`       | **det. (post)**   | cooperative load above mask `Cond`   | KernelOp masked tile              | `021`                                  |
| `tile/pad`              | **det. (post)**   | bank-conflict pad (formula)          | KernelOp smem-slab alloc          | `070`                                  |
| `tile/unroll`           | **det. (post)**   | `#pragma unroll` (trip threshold)    | KernelOp SERIAL loop              | `090`                                  |

Every `enum` row is a **genuine fork** (a ranked knob the search benches); `enum (body)` additionally edits `Block`s.
Every `det. (post)` row is **fixed-logic** — a deterministic `KernelOp`→`KernelOp` total function, no knob to branch, run
after `assemble` so it never enters the search tree. (`tile_axis` is not its own pass — the free-axis σ-splits fold into
`tile/build` and the reduce σ-split rides `partition_reduce`/`tensorize`, as `_split_free_axis` / `_replace_k_scalar` do
in `build_dag.py` today.)

One ex-pass dissolves with no pass at all: **`026`** (sibling-stage dedup) becomes automatic the moment `assemble`
coalesces slabs by `(buffer, access, distance)` — the duplicate is never created, so there is nothing to deduplicate. It
is the one fixed-logic case that is `assemble`-internal *by construction* rather than a post-pass. The peel (`080`) and
mask-ordering (`021`) could likewise be folded into `assemble`; they are listed as separate `det. (post)` passes per the
preference for a minimal `assemble` core with composable deterministic post-passes — the assemble/post boundary among the
fixed-logic mechanics is a modularity choice, but **none of them is enumeration**, which is the property that matters.

**`pad` and `unroll` are both `det. (post)` — and for the same reason: neither is a fork.** An earlier cut here split
them on "does the decision read a materialized artifact" (`pad` yes, `unroll` no) and put `unroll` pre-`assemble`. That
test is real but secondary; the governing test is the fork test, and both fail it. `070_pad_smem` computes one pad value
from a bank-conflict *formula* (`_max_conflict` / `lane_bank_distribution`), and `090_mark_unroll` unrolls every `SERIAL`
loop whose logical trip product is ≤ a threshold — **neither offers candidates to rank.** So both leave the search tree
and run as deterministic post-`assemble` passes. `pad` *also* happens to need the materialized slab (its safety gate
reads the realized `ld.shared.v4` width + the MMA atom-strided `AccessMap.block` stamp), which is why post-`assemble` is
its *only* legal home; `unroll` could in principle compute its threshold pre-`assemble` from logical axis extents, but
since it has no fork it belongs out of enumeration anyway, and running it on the materialized loop nest is the natural
home for a deterministic `#pragma`-stamping pass. The `Schedule.pad` / `Schedule.unroll` fields persist the chosen value
for variant identity, written *by* the deterministic post-pass, not consumed by `assemble`.

**`swizzle` is the one local that stays an enumeration fork.** `025_swizzle_blocks` carries a real `GROUP_M` knob with
ranked candidates the search benches — a genuine fork — and its object (a GRID block + its grid axes) is in the IR, so it
is a pre-`assemble` `Schedule.grid_swizzle` annotation. It is fixed-logic's opposite: a small but real choice, so it stays
in enumeration even though it is a "local."

**This maps straight onto the two-level MCTS.** The body-vs-schedule split is *orthogonal* to outer-vs-inner; the MCTS
boundary is *kernel-set-changing*. The kernel-set-changing enumeration passes (`partition_reduce` across CTAs, the
`005_split_demoted` cut at the partition head) branch the **outer** tree — one terminal per kernel set — while every
other enumeration pass branches the **inner** per-kernel tree. The deterministic materialization stage sits below both:
the inner search's reward is `assemble`→(`peel`/`mask_order`/`pad`/`unroll`)→cuda→bench of one leaf, summed per op for
the outer reward. Because the post-passes carry no fork, they add **zero** nodes to either tree — they are part of
evaluating a leaf, not part of the search. The inner search already chains `Fork`s across sequential per-kernel forks for
one kernel, so it consumes this pass list with no engine change.

**Knob-deltas are preserved.** Each *fork* still stamps its knob-delta onto the `Fork`
(`{RING:2}`, `{TMA:1}`, the `BM/BN/FM/FN` tile knobs) — the variant identity the perf DB and learned prior key on. What
changes is only the *realization*: the knob now drives a `Schedule` dict-write that the materialization stage reads,
instead of an eager tower rewrite. A deterministic post-pass writes its result into the same `Schedule` (`pad` →
`Schedule.pad`, `unroll` → `Schedule.unroll`) so it rides the variant identity like any other choice, but it is **not** a
search dimension — same value for the same materialized kernel, every time. The prior / DB / MCTS machinery is untouched;
`op_cache_key` becomes `canonical(bodies + edges) + Schedule` — the post-passes' `pad` / `unroll` are already inside that
`Schedule`, so no extra term — and the derived projections stay out of the key, same as `algebra_kind` today.

**Pass contracts — three kinds.**

- *Enumeration pass (genuine fork):* `rewrite(tile_node) -> Fork | tile_node | RuleSkipped`. Returns a `Fork` whose
  leaves are scheduled tile nodes **only when the family has ≥2 ranked offers**; the node unchanged (or `RuleSkipped`)
  for 0–1 — a one-child `Fork` is a bug, not a degenerate fork. **Never** a `KernelOp`. Offer legality is a pure function
  of derived projections + the current `Schedule`; the body is read-only except for the three body moves.
- *Materialization pass (`assemble`):* `rewrite(tile_node) -> KernelOp | Graph[KernelOp]`. No `Fork`, no `RuleSkipped` on
  a well-formed leaf. Deterministic and total. The load-bearing constraint: **if porting a regime tempts a tie-break
  heuristic, that decision belongs upstream as a fork + a `Schedule` field**, never inside the materialization stage.
- *Deterministic post-pass (fixed-logic):* `rewrite(kernel_op) -> KernelOp`. Runs after `assemble`. **No `Fork`** — a
  total function with no candidate to rank (a formula or a threshold). The test for landing here: *the rule has exactly
  one output for a given kernel.* If you find yourself wanting to bench two outcomes, it is a fork and belongs pre-`assemble`.

**Directory layout — two pass dirs.** The loader treats each `passes/<name>/` directory as one pipeline pass: it globs
that dir's numbered `*.py` rule files in order (non-recursively; `__init__.py` and `_`-prefixed files are helpers, not
rules — `pipeline.py::Pass.load`). The two sub-stages are therefore two pass dirs, split exactly at the `assemble`
boundary, and `TILE_PASSES` ends `…, "lowering/tile/enumeration", "lowering/tile/assembly"` in place of today's single
`"lowering/tile"`:

```
passes/lowering/tile/
  enumeration/                 # pass "lowering/tile/enumeration" — everything PRE-assemble: the seed + the forks
    000_build.py               #   deterministic: LoopOp → TileGraph algorithm DAG (no fork — seeds the search root)
    010_tensorize.py           #   fork (body):     atomize
    015_partition_reduce.py    #   fork (body):     split-K / cooperative-K
    020_register_tile.py       #   fork (schedule): bind REGISTER
    030_stage.py               #   fork (schedule): stage + hoist
    040_retime.py              #   fork (schedule): ring depth / distance
    050_transport.py           #   fork (schedule): promote CPASYNC / TMA
    060_warp_spec.py           #   fork (schedule): producer / consumer roles
    070_swizzle.py             #   fork (schedule): grid swizzle
    _moves.py  _knobs.py  …    #   shared offer/legality helpers (today's partition/moves.py, knobs.py)
  assembly/                    # pass "lowering/tile/assembly" — assemble + deterministic POST-processing (no fork)
    000_assemble.py            #   deterministic: (TileGraph + Schedule) → basic KernelOp tower
    010_peel.py                #   det. post: pipeline σ-peel from ring_depth
    020_mask_order.py          #   det. post: cooperative load above the mask Cond
    030_pad.py                 #   det. post: bank-conflict pad (formula)
    040_unroll.py              #   det. post: #pragma unroll (trip threshold)
    _slab.py  _synth.py  …     #   shared materialization helpers (today's materialize.py, _stage_expand, …)
```

`enumeration/` holds the entire search side — the deterministic `000_build` seed plus the genuine forks; `assembly/`
holds the deterministic materialization stage — `000_assemble` plus the fixed-logic post-passes, with no fork anywhere.
**The `assemble` boundary is the dir boundary**, so "is this pass allowed to return a `Fork`?" reduces to "which dir is
it in?" — a structural guardrail against fixed logic leaking into the search. Today's `partition/` package (the move
composer) dissolves into these two: `build` + forks + move helpers → `enumeration/`, `assemble` + slab synthesis →
`assembly/` (see "Relationship to existing code"). `000_build` is deterministic yet lives in `enumeration/` because it is
the pre-`assemble` front that produces the DAG the forks annotate — the dir is the *stage*, not the "has-a-fork" flag.

## Pass rewrites

Each current tile pass re-expressed against the IR above.

### stage — 020_stage_inputs

`stage(read)` writes `Schedule.staged[(src,dst,buffer)] = SYNC` for a reused gmem read; it inserts **no block** and
edits **no body**. The reuse-legality check reads the consumer's derived `reads` `Port` and `free_axes()` against the
enclosing `binding`: legal iff a parallel-bound axis inside the read is absent from `free_axes()` (fan-in) or ≥2 sibling
reads share one access (temporal reuse). `assemble` synthesizes the smem slab `Buffer`, the cooperative gmem→smem
producer (its `AccessMap` is the read's, projected to cache axes), and rewrites the consumer's gmem `Load` to the slab —
all at lowering. **Sibling-stage dedup (`026`) disappears entirely:** `assemble` keys slabs by `(buffer, access,
distance)`, so two reads of the same buffer at the same access share one slab and one producer by construction; the tree
could only ever de-stage, the DAG never creates the duplicate. Multi-source A+B share one `Schedule.cohort` (one
barrier). Masked-K: the consumer block's *derived* `Carrier.mask` forces `Transport.SYNC` (forbids CPASYNC). Overhang
clamp is the read `AccessMap.clamp`, itself derived from the gmem `Buffer.shape`.

### hoist — 030_hoist_invariant_compute + fused-prologue placement

`hoist(block)` sets `Schedule.scope[block]` to an enclosing nest; the *default* (no entry) is the max-hoist scope —
the outermost nest binding every axis in `block.reads[*].access.free_axes()` (pure LICM, derived). The recompute-vs-hoist
fork (`HOIST_COMPUTE`) is just default vs an inner override. The fused-prologue placement falls out with no special
case: an RMSNorm/softmax-stat reduce block reads only M-cache axes, so its default scope ends at `M_scope`, outside the
N register tier — `assemble` emits `SerialTile(M_scope) > {prologue; RegisterTile(N_reg){matmul}}`, prologue not
replicated per N cell. Masked rows ride through as a derived guard (`domain.real_extent` vs tile), coalesced by
`assemble` across co-scoped blocks. `030`'s invariant cone becomes an ordinary intermediate-buffer block; the edge to
its reduce consumer is derived from the buffer def-use, and `assemble` stages it like any other reused read.

### retime — 040_use_ring_buffers + 080_pipeline_stages

`retime(read, +k along serial axis S)` writes `Schedule.distance[edge] += ((S,k))` and raises
`Schedule.ring_depth[edge]` (invariant `≥ max(distance)+1`). No body edit. `assemble` is transport-parametric and
derives everything from `(distance, ring_depth, staged-transport)`: allocate `ring_depth` slabs, phase-index by
`S % ring_depth`, peel `ring_depth-1` prologue copies, emit the steady-state issue-ahead-by-`k`, drain the epilogue.
Waits follow transport — SYNC/CPASYNC get `AsyncWait(keep)` WAR fences; TMA gets mbarrier parity
`phase=(S/ring_depth)%2`, `slot=S % ring_depth`. `ring_depth` stays an independent field, not `distance+1`: distance
sets the correctness floor; extra depth is a free occupancy knob (depth 3 at distance 1). A multi-source K-loop shares
one `cohort`, hence one barrier/parity. The coupled-accum rejection (online-softmax running value) is a `retime`
*precondition* over `Block.compute`, never an IR field.

**Multi-level pipelining (gmem→smem→register).** Today `retime` parameterizes the **gmem→smem** stage only — pipelined
along the outer K loop (`K_o`) — and the steady body then loads smem→register (`ldmatrix` / `ld.shared`) and issues the
`Mma`/FMA **un-pipelined** (the legacy `080` behavior: an async `K_o` ring, then a direct smem load + mma on `K_i`). The
model generalizes to a **second pipeline level — smem→register double-buffering along the inner K loop (`K_i`)** —
because `distance` is **already per-serial-axis** (`tuple[(axis, offset), ...]`): a two-level mainloop is simply
`distance[(A→mm)] = ((K_o, 1), (K_i, 1))` — prefetch the next smem stage *and* the next register fragment. Three changes
realize it:

- **`ring_depth` becomes per-(edge, axis)** (today `dict[Edge, int]`): the smem ring (depth `N` on `K_o`) and the register
  double-buffer (depth `2` on `K_i`) carry independent depths. This is the one IR-schema touch.
- **The smem→register level is the *same* edge's inner-axis retime, not a new `Edge`.** The smem slab is an `assemble`
  artifact (not an IR object), so it can't be an `Edge` `src`; the register prefetch rides the gmem→mm edge's `K_i`
  distance, and `assemble` emits **both** levels from that one edge's `(multi-axis distance, per-axis ring_depth)` — the
  cp.async/TMA smem ring on `K_o` **and** the ping-pong `RegFragment` + lead `LdmatrixLoad` on `K_i`.
- **`assemble`'s warp-MMA synthesis gains the register-prefetch path**: two `RegFragment` sets ping-ponged, the next
  `LdmatrixLoad` issued before the current `MmaSyncPtx` — the real new codegen (the `atomize` section's single-buffered
  `RegFragment`/`LdmatrixLoad`/`MmaSyncPtx` chain becomes double-buffered). The `peel` post-pass already peels by serial
  axis, so it extends to peel `K_i` alongside `K_o`.

This lands with **R4/R5** (warp-MMA + transport), not RF — the `ring_depth`-per-axis change is the only schema edit; the
rest is `assemble`/`peel` codegen. A scalar-FMA mainloop generalizes the same way (`ld.shared` double-buffer on `K_i`),
so it is not warp-tier-specific.

### specialize_warps — 085_warp_specialize

`specialize_warps` writes `Schedule.role[block]` (PRODUCER on the TMA-staged loads, CONSUMER on the wait+reduce+`Write`
blocks) and `Schedule.reg_budget`. The cut-edges are producer→consumer reads over a TMA-staged buffer; their
`ring_depth` is the mbarrier ring depth. Legality: the role boundary is an antichain over the derived edge topology and
every crossing read is async (`CPASYNC`/`TMA`, never SYNC). `assemble` synthesizes everything else: the role axis
(extent = total warps, partition = #PRODUCER warps), the producer/consumer `Cond`, the mbarrier ring, the
consumer-scoped `bar.sync N,count`, the `SetMaxNReg`, and the `WarpSpecialize` Stmt. `consumer_is_warp` derives from
`binding[consumer_block]==WARP`; `n_producer_threads`/`tid_offset` from the partition.

### promote_transport — 050_use_tma + 060_use_async_copy

`promote_transport(read, →CPASYNC/TMA)` writes the `Schedule.staged[edge]` value. CPASYNC needs sm_80 + a pipelined read
(`distance≥1`); TMA needs sm_90 + a box-affine read. Eligibility reads only the derived `AccessMap` + `Buffer`:
`kind==AFFINE`, strictly-increasing `dims` with gap dims extent-1, every collapsed box extent (`dims`×`block`) ≤256, 16B
inner alignment. A `TEMPLATE` access (collapsed reshape) declines TMA. The swizzle (B64/B128) is derived at `assemble`
from the inner-box span × `dtype.nbytes` (`pick_swizzle_atom`), never stored. Promotion is cohort-atomic — every read
sharing a `cohort` flips together (mixed transports behind one barrier deadlock). A derived `Carrier.mask` (symbolic-K)
read takes SYNC or TMA, never CPASYNC. A hoisted-compute producer (its block's `compute` does more than load→write)
declines TMA. The re-entry gate reads `scope` + `domain` extents.

### partition_reduce — split-K (015/017) + cooperative-K

`partition_reduce(K, GRID|THREAD)` is a **body move**: it factors the reduce axis, binds the partition factor, and
inserts an explicit combine `Block`. Legality is algebraic — read `carrier.commutative` off the *derived*
`Block.carrier`, plus `carrier.has_identity` when the partition is uneven/masked (the derived `Carrier.mask`, applied
pre-clamp, zero-fills the tail). The `allow_split` veto (non-linear epilogue / multi-accum / fused-prologue) is a move
**precondition** over the candidate DAG, not a stored field. The combine block's realization is derived from its
`binding`/`Space`/`launch`: GRID + a write dropping the partition axis ⇒ `atomicAdd`; THREAD ⇒ warp-shuffle, its
broadcast `lane==0` guard derived from the binding; SERIAL + own `launch` group ⇒ second kernel (017's atomic-free
split-K — two launch groups in one `TileGraph`, joined by a partial intermediate `Buffer` whose edge is derived). 015's
residual-once is a derived `K_s==0` guard on the atomic arm, or a `K_s`-excluding finalize block on the atomic-free arm.

### atomize — 011_lower_atom_cell + warp tower

`atomize` is a **body move** on a warp-tier matmul block. The surrounding tower is `tile_axis` + `bind`: each output
axis splits four ways (`A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom`), bound GRID/WARP/REGISTER/ATOM; the reduce splits
`K_o`/`K_i`. `atomize` fuses the cell `[Load,Load,mul,Accum]` into one `Mma` — the only body edit — and `Block.atom`
then *derives* from that `Mma` (the `(M,N,K)` cell shape + operand dtypes). Legality: the derived SEMIRING `Carrier`
(`⊗` distributes over `⊕`) + atom eligibility. ATOM-bound axes are non-addressable (excluded from every `AccessMap`;
the per-lane `A_a` offset never enters σ). `assemble` synthesizes the rest from `(atom, role)` — `RegFragment`,
`LdmatrixLoad` (staged iff the operand read is `staged`), `MmaSyncPtx`, `RegStore`; lane→element maps and the
per-element store guard stay assemble-internal. The whole-tile masked skip is the derived `domain.real_extent` guard; a
symbolic reduce uses the derived `Carrier.mask` to zero-fill the partial slab past `seq_len`.

### register_tile — RegisterTile expansion

`register_tile` is `bind(axis, REGISTER)` — a `Schedule.binding` write; the F cells materialize only at `assemble`,
expanding the one block into F replicas with a per-cell `σ: axis→literal(i)` + SSA `_<i>` suffix. Per-cell `Port`
specialization is not eager: `assemble` runs the SSA def-use closure over `compute` first, and a `Port` specializes iff
its access reads a per-cell SSA name (the embedding-gather case); axis-free producers and slab cache axes stay shared.
The FK fold is a combine `Block` (a `partition_reduce` over the strip-mined K-register axis) whose default `scope` sits
inside THREAD but outside `K_o`/`K_i`. Fold-vs-unroll is derived: a REGISTER axis ∈ the block's *derived* carrier
reduce-axes ⇒ independent accumulators + fold; else plain unroll. Multiple REGISTER axes (FM×FN) compose; `assemble`
expands in `binding` order, intra-scope order by edge topo-sort.

### locals — 025 swizzle (fork, pre) + 070 pad & 090 unroll (det., post)

No body edit. The fork test splits them: one carries a ranked knob, two are deterministic. **025 (enumeration fork,
pre-`assemble`)** → `Schedule.grid_swizzle[block]` (L2 row-group): a real `GROUP_M` knob with candidates the search
benches; its object is a GRID block + its grid axes, both in the IR; gate = the GRID block's derived `Carrier` is matmul
and it has ≥2 GRID axes; `025` tags itself "Renderer-only," and `assemble` consumes the field. **070 (deterministic,
post-`assemble`)** → bank-conflict pad of a staged read's smem slab: the pad value is a *formula*
(`_max_conflict` / `lane_bank_distribution`), not a ranked choice, and the slab it reads is an `assemble` artifact (its
gate reads the realized `ld.shared.v4` width + the MMA atom-strided `AccessMap.block` stamp; skip on TMA), so it is a
deterministic `KernelOp`→`KernelOp` pass that widens the slab alloc (coords stay logical). **090 (deterministic,
post-`assemble`)** → `#pragma unroll` on every materialized SERIAL loop whose logical trip product is under a fixed
threshold (symbolic extents decline) — again no candidate to rank, so a deterministic post-pass, not an enumeration node.
The `Schedule.pad` / `Schedule.unroll` fields persist the chosen value for variant identity, written *by* the
deterministic post-pass, not consumed by `assemble`.

## Lowering: `assemble` (TileGraph → KernelOp | Graph[KernelOp])

The deterministic materialization stage turns (algorithm + Schedule) into the tower — the `assemble` core plus the
fixed-logic post-passes (the steps a post-pass owns are flagged inline). No step makes a search choice:

1. Partition blocks by `launch` group → one `KernelOp` per group (cross-group intermediate buffers become graph-node
   tensors); a single group yields one `KernelOp`.
2. Per block, take `Schedule.scope[block]` (or the derived max-hoist default); reconstruct the loop nest from those
   axes + `Schedule.binding` (GRID → SERIAL → THREAD/WARP → REGISTER/ATOM); merge blocks sharing a scope prefix;
   **intra-scope block order = the derived edge topo-sort**.
3. For each `staged` read, synthesize the smem slab `Buffer`, the cooperative producer, and the consumer's slab `Load`
   (the slab is emitted at its logical extent; the bank-conflict `pad` is applied later by the post-assemble `tile/pad`
   local pass, which can see the materialized byte span). **Coalesce slabs by `(buffer, access, distance)` — this is
   dedup, by construction.**
4. Expand `distance>0` reads into prologue/steady/epilogue + waits, transport-parametrically (factored into the `peel`
   deterministic post-pass).
5. Expand the `role` coloring into the warp-spec `Cond` + synthesized role axis + mbarrier ring + `SetMaxNReg` +
   consumer named-barrier.
6. Replicate REGISTER-bound blocks per cell (SSA def-use closure first); for ATOM blocks, synthesize fragments + the
   `ldmatrix`/`mma.sync`/`RegStore` chain from the derived `atom`.
7. Emit each block's derived guards (`domain.real_extent`, partition `K_s==0`/`lane==0`) as `Cond`s, coalesced across
   co-scoped blocks (the `mask_order` post-pass then lifts cooperative loads above the masked-tile guard).

Because 2–7 reproduce the current tower/`StageBundle`/`WarpSpecialize`/`AsyncWait` exactly, a "reference schedule" (the
annotations the composer picks today) must `assemble` to **byte-identical CUDA**. That is the safety contract.

### `assemble` & move contracts

- **Slabs, cooperative producers, pipeline peels, warp-spec `Cond`s, combine kernels, the role axis, named barriers,
  `phase`/`AsyncWait`/`WarpSpecialize`, and MMA fragments are all assemble OUTPUTS** — synthesized from the algorithm +
  Schedule, never stored in the IR.
- **Sibling-stage dedup is automatic** (slab coalescing by `(buffer, access, distance)`); there is no dedup move.
- **`allow_split` veto + loop-carried-scalar (coupled-accum) legality are move preconditions** over the candidate DAG,
  not IR fields.
- **Masked ordering:** the derived `Carrier.mask` predicate is evaluated on the **pre-clamp** index; `AccessMap.clamp`
  applies after.
- **Guard granularity:** the `domain.real_extent` guard governs the output `Write`; the MMA per-fragment-load clamp is
  the derived read `AccessMap.clamp`.
- **`cohort` invariant:** one transport + one barrier + one ring phase per cohort; a consumer's derived `Carrier.mask`
  ⇒ SYNC on its cohort's reads. **`ring_depth`** is independent, floored at `max(distance)+1`.
- **Multi-level-pipeline carve-out (the `K_i` register level has no barrier):** the cohort/barrier rules above govern the
  **gmem→smem** ring (the `K_o` distance) — its transport, `__syncthreads`/mbarrier parity, and one ring phase. The
  **smem→register** double-buffer (the inner `K_i` distance from "Multi-level pipelining") is a **compiler-scheduled
  register rotation** — its hazard is register WAR, resolved by instruction ordering (issue the next `LdmatrixLoad` before
  the current `MmaSyncPtx`), **not** a barrier. So `assemble` emits **no** sync / mbarrier / cohort phase for a `K_i`
  retime; `ring_depth[edge][K_i]` (= 2 for ping-pong) sizes the `RegFragment` rotation only. The `max(distance)+1` floor
  and the one-phase-per-cohort rule apply **per axis**: independently on the `K_o` (smem, barriered) and `K_i` (register,
  un-barriered) levels.

## Why moves compose (and pass-ordering lore dies)

Today's ordering constraints are implicit in matcher shapes. Here they are explicit facts on derived state: `retime`
requires a `staged` read; `promote_transport(TMA)` requires a box-affine *derived* `AccessMap`; `specialize_warps`
requires async crossing reads. A scheduler (the existing MCTS in `pipeline/fork.py` + `partition/tree.py`) explores
moves whose legal set is computed from the algorithm + current Schedule, not from "did the previous pass leave the tree
in the shape my matcher wants." The body is fixed (touched only by the three algebra moves); the Schedule is the search
space; each move is a fork with an algebra/dependency-gated offer set — the same `moves.py`/`tree.py` model, now over
the whole tile phase.

## Relationship to existing code

- **`partition/` (the move composer)** becomes the *builder* of the algorithm DAG + a reference Schedule, then the first
  consumer of the move set. `iter_dag` already derives a PARALLEL/REDUCE + carrier-tagged view — exactly the
  derived-projection style the new `Block` properties use. `legal_decomps` is already the algebra-trait gate.
- **`AlgebraKind` + carrier traits** (`ir/algebra.py`) are the algebraic legality oracle unchanged. `TWISTED_MONOID`
  (flash online-softmax) is the sharp case — its combine is coupled state (m/l/O), so its `partition_reduce` combine
  block carries that recurrence; restrict it in v1.
- **`Source`/`AffineAddressing`/`_stage_expand`** are precisely the assemble-time slab + cooperative-producer
  synthesis the `staged` annotation now drives — they move out of the IR into `assemble`.

## Recovery sequencing — build the foundation, recover tests tier by tier

This refactor was executed **demolition-first**: the legacy planner, the `005`–`009` structural passes, the `011`–`090`
scheduling passes, the warp / cooperative / flash builders, and the legacy materializers were deleted, and `partition/`
was reorganized into the `enumeration/` + `assembly/` subfolders this design calls for. So the work now is **recovery**,
not a forward port from a working pipeline: rebuild each tier as its block-DAG foundation — an enumeration builder
(`enumeration/_tree.py` etc.) **plus** its `assemble` synthesis (`assembly/_assemble.py`) — and recover that tier's
quarantined tests as it lands. This section supersedes the earlier "forward port" framing.

**Current state (verified empirically, branch `feature/tile-ir-block-dag`).** Structure already matches the target: one
rule `010_partition_loops.py` over the `enumeration/` + `assembly/` packages. `classify()` still tags all four regimes,
but `build_partition()` only *builds* **pointwise + scalar matmul**, which lower end-to-end (`LoopOp → TileOp → CudaOp`,
incl. K-chunk). Coop and flash return `None` → `010` raises → quarantined; warp-MMA / staging-synthesis / transport /
split-K combine / structural forks are deleted. `tests/compiler/` runs **~1175 passed · 28 failed · 166 xfailed · 89
XPASSED** — so the floor is healthy, but two numbers are the real signal: the **89 XPASS** are quarantined-but-now-green
(the registry is stale), and the **28 failed** are genuine, non-quarantined breakage.

**Two quarantine registries** in `tests/compiler/_composer_xfails.py`: the **`_DEMO`** set (34 funcs + 7 nodes + 16
files, "restored when assemble synthesizes the Schedule") is *this* plan's targets; the **`_REASON`** set (13 + 3,
`plans/composer-only-green-suite.md`) is an older symbolic/structural-gap plan that overlaps and folds in per phase.

The phases below each rebuild one tier and are **done** when its quarantined tests XPASS and are deleted from
`_composer_xfails.py` in the same commit (the registry docstring already mandates this), with the green floor unmoved.
Gate: **accuracy-vs-torch** for new synthesis; **byte-identical** where a legacy tower shape is reproduced.

(R0 — quarantine hygiene — has **landed**: the registry `tests/compiler/_composer_xfails.py` was rebuilt from the
empirical true-failure set so every red test is an `xfail(strict=False)` tagged with its recovery phase (R2–R7) and no
green test is masked; `make test` is green with 0 failed / 0 XPASS.)

**Foundation (RF) — split the single enumeration pass into per-family tree-expansion passes.** *Done before any tier.*
This is the Option-B groundwork that makes "The pass structure" / "Directory layout" real, refactoring **only** the
existing pointwise + scalar-matmul composer (no new regime), gated **byte-identical**.

**Landed (verified in the working tree; closes G1/G2/G5).** The two pass dirs are registered (`TILE_PASSES` ends
`…, "lowering/tile/enumeration", "lowering/tile/assembly"`); **`TileGraphOp`** is a real graph-node op (`ir/tile/ir.py`)
carrying `tilegraph` + `knobs` + `leading`; and the enumeration tree is now **split into per-family rule passes** —
`enumeration/{000_build, 010_reduce_tile, 020_thread_tile, 030_register_tile, 040_lower}` plus the extracted
`_classify.py`, with `010_enumerate.py` and the in-memory `_tree.py` (`_Choose*` nodes) deleted. `000_build` seeds the
`TileGraphOp`; the `*_tile` passes each match it, compute offers (`_moves.py`'s `*_offers` fns), and return a `Fork`
pinning one more knob group; `040_lower` builds the `TileGraph` and the separate `assembly/010_assemble` materializes it
`→ TileOp`. Pointwise + scalar matmul lower end-to-end through the chain.

**F3-a shipped; F3-b is the one deferred follow-up.** `040_lower` runs the unified `_build.build_dag` **once** at the end
of enumeration — the algorithm is still built all-at-once (`build_dag` σ-splits `Block.domain` by the full thread/reg/K
knob set; `Schedule` holds `binding`). The only remaining RF debt is **F3-b** — a knob-*invariant* algorithm where
`000_build` seeds logical axes, each tile pass applies an incremental `tile_axis` body move, and `assemble` does the
register/thread replication from `Schedule`. F3-b reworks both `build_dag` **and** `assemble`, so it is deferred until a
tier needs the invariant algorithm. The laziness guarantee (`pipeline/fork.py`) is unchanged: the per-family split
materializes one thin knob-carrying `TileGraphOp` per resolved pass (vs one at the leaf before), never the full
cartesian — the unexpanded frontier still materializes nothing.

**Known consequence — greedy compile is now level-greedy.** `greedy_decide` flattens each fork to complete leaves and
ranks via the prior; with one fork per search level it now ranks *partial* knob sets, and the cold `AnalyticPrior`
calibrates those poorly (a scalar matmul can land a degenerate `BN=BM=1` pick — accuracy holds within fp32 noise, but
the pick is no longer byte-identical to the legacy tower). The two-level MCTS / `tune` path enumerates the **identical**
leaf set and is unaffected; partial-config prior calibration is a follow-up. Two search tests that asserted the old
single-fork shape (`test_resolve.py::test_trace_records_partition_fork`,
`test_two_level.py::test_run_two_level_tune_single_terminal_assembles_bests`) are quarantined under R7.

*Gate: byte-identical `TileOp` for pointwise + scalar matmul; the green floor unmoved; the two-level MCTS enumerates the
**identical leaf set** — it now branches across passes instead of within one tree.* The `→ KernelOp` switch (audit **G3**)
and the deterministic post-passes (**G4**) remain later steps. Every tier below assumes this foundation: each **adds one
enumeration pass** (its move family) + grows `assemble`'s `Schedule`-driven synthesis — never
a tree-builder.

- **R1 — Staging** (deps: RF). **Landed (scalar tier).** Added the first `Schedule`-move enumeration fork
  `enumeration/050_stage.py` (it runs *after* `040_lower` on the built `TileGraphOp`, reads the derived `Block.reads` +
  `Schedule.binding`, and forks on a `STAGE` bitmask — writing the chosen `Edge`s straight into `Schedule.staged`, the
  source of truth `assemble` reads) + the candidate/legality helper `enumeration/_stage.py` (a read is stageable iff
  AFFINE + fan-in reuse across a parallel axis + a K-tower to stage through) + the `STAGE` BINMASK knob. `assemble`
  synthesizes the smem slab + cooperative producer from `Schedule.staged` in the new `assembly/_slab.py`
  (`synthesize_staging`): one SYNC `StageBundle` per K-tower, a `Source` per staged buffer whose affine `source_index`
  reconstructs the operand's original σ-rewritten gmem index (cache axes = THREAD/REGISTER tile + the within-stage K
  axes; GRID + serial-outer `K_o` fold into the slab origin; `block=()`, the composite stride rides the cache-axis
  extents). Handles both the multi-stage K loop (bundle reloads inside `serial_outer`) and the `BK == K` single-stage
  whole-K slab. `TileGraph.buffers` is now populated (logical gmem `Buffer`s from the source `LoopOp`'s I/O, carried on
  the `TileGraphOp` seed) so `assemble` can size slabs + stamp `Source.dtype`. The downstream kernel passes
  (`_stage_expand` / `100_materialize_tile`) — which were never deleted — expand the bundle untouched. *Gate:*
  accuracy-vs-torch (byte-exact, max err 0.0) for every stage subset (both / A-only / B-only / none) on `64³`, a masked
  non-divisor `64×47×64`, and `128³`; new test `tests/compiler/passes/test_stage_scalar.py` (16 cases). Green floor
  unmoved (28 baseline failures unchanged).
  *Recovery-test attribution corrected.* The two tests this bullet originally named are **not** recoverable by scalar
  staging and stay quarantined: `test_stage_inputs_mma_probe.py` pins `DEPLODOCK_MMA` and asserts a warp-tier `Mma` +
  `StageBundle` — that needs the `atomize` tier (**R4**); `…::test_hoist_refuses_lift_when_pipeline_reads_guarded_defs`
  `import`s the deleted `021_hoist_staged_loads_above_mask` and tests its legacy `_lift_if_match` internals — it must be
  rewritten against the new `assembly/020_mask_order` (the masked-load hoist, deferred with the masked tiers), not
  "recovered." Scalar masked staging itself works today (the bundle sits inside the boundary `Cond`, so masked threads
  skip the cooperative load — no OOB, no hoist needed; the hoist is a perf/transport requirement for cp.async/TMA).
  **Deferred follow-ups:** warp/atom + symbolic-K (masked-K zero-fill) staging ride R4; the `retime` / `transport` /
  `warp_spec` forks and the `pad` / `mask_order` post-passes are their own tiers.
- **R2 — Cooperative-reduce** (deps: RF; the biggest chunk). Add the coop `classify`→build path + an
  `enumeration/050_coop_reduce` pass (`(bk,fk,br)` offer) + `assemble`'s warp-shuffle / hierarchical combine synthesis.
  *Recovers ~22:* `test_monoid_reduce_kernel.py`,
  `test_cooperative_combine.py`, `test_masked_cooperative_reduce.py`, the six `test_reduction_rules.py` coop tests, the four
  `test_dtype_cuda.py` fp16 reduce/softmax/rmsnorm, the six `test_accuracy.py` e2e reduce/rmsnorm/softmax, the six
  `test_ops_vs_torch.py` reduce/mean/softmax/rmsnorm, `test_tile_naming.py::test_real_rms_norm_kernels_named_by_op`,
  `test_tune_accuracy.py::…[rmsnorm]`.
- **R3 — Split-K / `partition_reduce`** (deps: R2's combine synthesis). The `partition_reduce(K,GRID)` move + atomic /
  atomic-free combine block. *Recovers:* `test_mma_atomic_free_splitk.py`,
  `test_structural_push.py::test_atomic_free_splitk_fork_pushes_structural`.
- **R4 — Warp-tier MMA (`atomize`)** (deps: R1). Add `enumeration/060_tensorize` (atom-fold body move) + `assemble`'s
  `RegFragment`/`Ldmatrix`/`MmaSyncPtx`/`RegStore` synthesis. *Recovers:* `test_matmul_mma.py`, `…_residual.py`,
  `…_causal_epilogue.py`, `…_transposed_b.py` (+ its
  symbolic-MN nodes), `…_parity.py` (non-TMA params), `test_knob_pinning.py::test_unstaged_atom_lowers_gmem_direct`.
  *Folds in `_REASON` Phase 1–2 (symbolic masked warp / masked-K).*
- **R5 — Transport (cp.async / TMA)** (deps: R1, R4). `promote_transport` + descriptor/swizzle synthesis (`peel` post-pass
  does the pipeline expansion). *Recovers:* the `test_matmul_mma_parity.py` TMA nodes (static + dynamic),
  `test_knob_pinning.py::test_norm_linear_fp16_scalar_reduce_tma_alignment`. *Folds in `_REASON` Phase 3.*
- **R6 — Flash / attention** (deps: R2, R4). Add the flash build path (streaming `TWISTED_MONOID`, `FM=FN=1`) +
  `assemble`'s flash synthesis. *Recovers:*
  `test_flash_attention.py`, `test_flash_cooperative_kv.py`, `test_attention_chains.py`, the three
  `test_ops_vs_torch.py` sdpa, `test_tune_accuracy.py::…[sdpa]`.
- **R7 — e2e / CLI / structural-search / prior** (deps: R1–R6). Structural-fork outer search (`005_split_demoted`
  reborn), analytic/cold prior over the rebuilt enumeration, whole-program paths. *Recovers:* `test_run.py`,
  `test_block.py`, `test_program_rebind.py`, the two `test_compile.py`, the `test_two_level.py` / `test_structural_push.py`
  / `test_resolve.py` structural tests, the two `test_analytic.py::test_pick_matmul…`. *Folds in `_REASON` Phase 4 + 6.*

```
RF foundation ──┬─► R1 staging ──► R4 warp-MMA ──► R5 transport ─┐
(multi-pass split) ├─► R2 coop-reduce ──────────────────┐        ├─► R7 e2e / structural / prior
                └─► R3 split-K ──────────────────────────┘        │
                             R2 ──► R6 flash ◄── R4 ──────────────┘
```

RF (the per-family-pass split) gates every tier. R1 and R2 are independent (R2 is larger and unblocks R6); R3 rides R2's
combine work; R4→R5, R2+R4→R6, all→R7. As each
tier lands, retire the matching `composer-only-green-suite.md` phase into its R-phase here — that doc no longer tracks a
live process.

## Open design decisions

- **TWISTED_MONOID partition/retime.** Flash's coupled carrier makes `partition_reduce`/`retime` over the streaming KV
  axis subtler than a plain monoid (the rescale rides the combine block). Likely v1-restrict flash to no
  partition_reduce on the streaming axis (matches today's degenerate `FM=FN=1`), revisit with online-softmax-flash.
- **Canonical body form for caching.** `op_cache_key` is now `canonical(compute bodies + edge topology) + Schedule`.
  The bodies still need a canonical SSA/axis renaming (today's `normalize_body`) and the Schedule a canonical
  serialization — smaller than before (no derived projections), but still required before the DAG can key perf rows.
- **AccessMap quasi-affine escape hatch — decided.** Affine core (`dims`/`block`/`offset`) + a `TEMPLATE` verbatim
  fallback for collapsed-reshape views; matches today's `AffineAddressing`/`TemplateAddressing` split.
- **Masking as derived guard — decided.** Masked-ness is `domain.real_extent` vs tile, a derived `Cond` + clamped
  access, never a knob. Folds in `plans/drop-overhang-knob-structural-masked-feature.md`.

## Next step

The structural surface has converged: the algorithm is `name + domain + compute`, every projection is derived, every
choice is a `Schedule` annotation, and `assemble` is the one place the tower is built. **R1 (staging, scalar tier) has
landed** — the first `Schedule`-move enumeration fork (`enumeration/050_stage`) writes `Schedule.staged`, and
`assemble` synthesizes the smem slab + cooperative `StageBundle` from it (`assembly/_slab`), proven accuracy-vs-torch
on scalar matmul (static + masked). The pass structure is now validated end to end on a regime that touches smem (an
enumeration fork widens; the materialization stage lowers). Two independent tiers are unblocked: **R2** (cooperative
reduce — the biggest chunk, the coop `classify`→build path + `enumeration/050`-style `(bk,fk,br)` offer + warp-shuffle
combine synthesis) and **R4** (warp-tier `atomize` — which also brings warp/atom + symbolic-K staging, the deferred
half of R1). `retime` / `transport` / `warp_spec` forks and the `pad` / `mask_order` post-passes follow on top.
