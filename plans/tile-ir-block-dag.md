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

The IR below is the result of three co-design rounds that re-expressed all 15 current tile passes against it; each
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

The two headline consequences from the original design hold and are now literal: **prologue/epilogue are not special**
(a fused-prologue reduce is a block whose reads are {M,K}; its default `scope` ends at `M_scope`, outside the N register
tier — no special-case builder), and **pipelining / warp-spec stop being hand-peeled** (`assemble` emits the
peel / `Cond` / mbarrier ring from `distance` / `role`, not a pass).

## The pass structure: enumeration passes + one `assemble`

The model above resolves into exactly **two kinds of search-relevant pass**, bookended by two deterministic transforms.
The whole tile phase is: a deterministic `tile/build` seeds the root tile node, a pipeline of **enumeration passes**
widens the search tree move-family by move-family, and one deterministic **materialization pass** (`assemble`) lowers the
chosen leaf. This is the concrete realization of "the scheduling moves only edit the Schedule; `assemble` is the one
place the tower is built."

`assemble` is the *boundary*, and enumeration runs on **both sides** of it. Almost every move family enumerates
**pre-assemble** over the `TileGraph` (the dataflow / structural / scheduling moves, whose objects — edges, axes, blocks,
the Schedule — live in the IR). Exactly one family, **`pad`**, enumerates **post-assemble** over the materialized
`KernelOp`, because its decision reads an `assemble` *output* (the slab's byte layout + the realized vectorized-load
width) — see the note after the pass list. The decision rule is empirical: a family is post-`assemble` **iff its offer
legality reads an `assemble` artifact**. The pass *kind* is still binary (enumeration vs materialization); it is the
enumeration *phase* that the `assemble` boundary splits.

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

| pass                    | kind              | move family                          | edits                          | predecessor (deleted / legacy)        |
|-------------------------|-------------------|--------------------------------------|--------------------------------|---------------------------------------|
| `tile/build`            | deterministic     | — derive algorithm DAG from `LoopOp` | seeds node + reference `Schedule` | `010` front half (`build_dag.py`)  |
| `tile/tensorize`        | enum (body)       | `atomize`                            | `Block.compute`                | `011` + warp tower                    |
| `tile/partition_reduce` | enum (body)       | split-K / cooperative-K              | insert combine `Block`         | `015`/`017`                           |
| `tile/register_tile`    | enum (schedule)   | `register_tile` = `bind(REGISTER)`   | `binding`                      | `010` reg split                       |
| `tile/stage`            | enum (schedule)   | `stage`, `hoist`                     | `staged`, `scope`              | `020`/`021`/`026`/`030`               |
| `tile/retime`           | enum (schedule)   | `retime`                             | `distance`, `ring_depth`       | `040`/`080`                           |
| `tile/transport`        | enum (schedule)   | `promote_transport`                  | `staged` value                 | `050`/`060`                           |
| `tile/warp_spec`        | enum (schedule)   | `specialize_warps`                   | `role`, `reg_budget`           | `085`                                 |
| `tile/swizzle`          | enum (schedule)   | `swizzle`                            | `grid_swizzle`                 | `025`                                 |
| `tile/unroll`           | enum (schedule)   | `unroll`                             | `unroll`                       | `090`                                 |
| `tile/assemble`         | **materialize**   | —                                    | TileGraph → KernelOp           | the tower builders + `assemble_block`  |
| `tile/pad`              | enum (**local, post-assemble**) | `pad`                  | KernelOp smem-slab alloc       | `070`                                 |

(`tile_axis` is not its own pass — the free-axis σ-splits fold into `tile/build` and the reduce σ-split rides
`partition_reduce`/`tensorize`, exactly as `_split_free_axis` / `_replace_k_scalar` do in `build_dag.py` today.)

Three legacy passes are **not in the list at all** — a per-source-evidence sweep of the deleted files found they survive
only as `assemble`-internal *mechanics*, with no decision to enumerate: **`026`** (sibling-stage dedup) is automatic once
`assemble` coalesces slabs by `(buffer, access, distance)`; **`080`** (pipeline peel) is pure σ-shifted prologue/steady/
epilogue replication driven by `ring_depth` (the depth is `retime`'s knob, not `080`'s — `080` has no surviving search
choice); **`021`** (hoist staged loads above the mask `Cond`) is the correctness ordering `assemble` must produce when it
materializes a masked tile over a staged transport. None is a `Fork`; each was already a `RuleSkipped`-or-rewrite with no
knob. They move *into* `assemble`, not into the pass list.

**Exactly one local lives *after* `assemble`: `tile/pad`.** The per-source-evidence sweep was sharper than the first cut
here — only `pad` genuinely needs the materialized kernel. The smem slab is "only ever an assemble artifact" (the IR's
own words), and the pad's *safety gate* keys on `assemble`-time facts: the realized vectorized-load width (`ld.shared.v4`)
and the MMA atom-strided slab stamp (`AccessMap.block`). So `tile/pad` is a small `KernelOp`→`KernelOp` local `Fork`
(legacy `070`, essentially unchanged), running after `assemble`, reading the slab it produced. The `Schedule.pad` field
records the chosen knob for variant identity, but the choosing pass reads the materialized slab — `assemble` does not
consume it.

**`unroll` and `swizzle` stay *pre*-`assemble`** — the evidence corrected the intuition that grouped `unroll` with `pad`.
`090_mark_unroll` only ever reads **logical** `SERIAL` axis extents (`loop.axis.extent.as_static()`, a trip-count product
vs a threshold; symbolic extents decline) — the `K_o`/`K_i` and register-tile axes that all exist in the `TileGraph`
domain pre-`assemble`. It never consults the materialized nest, and a peeled ring loop's steady body equals one logical
iteration, so peeling does not change the per-iteration unroll decision. It is therefore a clean `Schedule.unroll`
annotation that `assemble` reads when it emits the loop. `swizzle` is likewise pre-`assemble`: its object — a GRID block
and its grid axes — *is* in the IR (a blockIdx remap, no materialized detail needed); `025` even tags itself
"Renderer-only," writing only the `grid_swizzle` field. The cut is empirical, not aesthetic: a family is post-`assemble`
**iff its decision reads an `assemble` artifact** — `pad` does, `unroll`/`swizzle` do not.

**This maps straight onto the two-level MCTS.** The body-vs-schedule split is *orthogonal* to outer-vs-inner; the MCTS
boundary is *kernel-set-changing*. The kernel-set-changing enumeration passes (`partition_reduce` across CTAs, the
`005_split_demoted` cut at the partition head) branch the **outer** tree — one terminal per kernel set — while every
other enumeration pass branches the **inner** per-kernel tree. `assemble` sits below both: the inner search's reward is
`assemble`→cuda→bench of one leaf, summed per op for the outer reward. The inner search already chains `Fork`s across
sequential per-kernel lowering passes for one kernel, so it consumes this pass list with no engine change.

**Knob-deltas are preserved.** Each enumeration move still stamps its knob-delta onto the `Fork`
(`{RING:2}`, `{TMA:1}`, the `BM/BN/FM/FN` tile knobs) — the variant identity the perf DB and learned prior key on. What
changes is only the *realization*: the knob now drives a `Schedule` dict-write that `assemble` reads, instead of an eager
tower rewrite. The prior / DB / MCTS machinery is untouched; `op_cache_key` becomes `canonical(bodies + edges) +
Schedule` (the derived projections stay out of the key, same as `algebra_kind` today).

**Pass contracts.**

- *Enumeration pass:* `rewrite(tile_node) -> Fork | tile_node | RuleSkipped`. Returns a `Fork` whose leaves are scheduled
  tile nodes when the family has ≥2 legal offers; the node unchanged (or `RuleSkipped`) for 0–1; **never** a `KernelOp`.
  Offer legality is a pure function of derived projections + the current `Schedule`; the body is read-only except for the
  three body moves.
- *Materialization pass:* `rewrite(tile_node) -> KernelOp | Graph[KernelOp]`. No `Fork`, no `RuleSkipped` on a
  well-formed leaf. Must be deterministic and total. The load-bearing constraint: **if porting a regime tempts a
  tie-break heuristic inside `assemble`, that decision belongs upstream as an enumeration move + a `Schedule` field** —
  keeping the leaf the sole source of variant identity is what makes it cacheable and byte-identical-verifiable.

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

### locals — 025 swizzle + 090 unroll (pre) + 070 pad (post)

No body edit. Per-source evidence places two pre-`assemble` and one post — the cut is *which IR holds the object the
offer legality reads*. **025 (pre-assemble)** → `Schedule.grid_swizzle[block]` (L2 row-group), a genuine `TileGraph`
annotation: its object is a GRID block + its grid axes, both in the IR; gate = the GRID block's derived `Carrier` is
matmul and it has ≥2 GRID axes; `025` tags itself "Renderer-only," and `assemble` consumes it. **090 (pre-assemble)**
→ `Schedule.unroll[axis]` on a SERIAL axis when the **logical** trip-count product (`loop.axis.extent.as_static()` over
`K_o`/`K_i` + register chains, symbolic extents decline) is under threshold. It reads only logical axis extents that
exist in the `TileGraph` domain — never the materialized nest — and peeling does not change the per-iteration unroll
decision, so it is a clean annotation `assemble` reads when it emits the loop. **070 (post-assemble)** → bank-conflict pad
of a staged read's smem slab: the slab is an `assemble` *artifact* and the safety gate reads materialized facts (the
realized `ld.shared.v4` vectorized width; the MMA atom-strided `AccessMap.block` stamp; skip on TMA transport), so it is a
local `KernelOp`→`KernelOp` `Fork` that widens the slab alloc (coords stay logical). The `Schedule.pad` field carries the
chosen knob for variant identity, recorded *by* the post-`assemble` pass, not *consumed by* `assemble`.

## Lowering: `assemble` (TileGraph → KernelOp | Graph[KernelOp])

One deterministic pass turns (algorithm + Schedule) into the tower:

1. Partition blocks by `launch` group → one `KernelOp` per group (cross-group intermediate buffers become graph-node
   tensors); a single group yields one `KernelOp`.
2. Per block, take `Schedule.scope[block]` (or the derived max-hoist default); reconstruct the loop nest from those
   axes + `Schedule.binding` (GRID → SERIAL → THREAD/WARP → REGISTER/ATOM); merge blocks sharing a scope prefix;
   **intra-scope block order = the derived edge topo-sort**.
3. For each `staged` read, synthesize the smem slab `Buffer`, the cooperative producer, and the consumer's slab `Load`
   (the slab is emitted at its logical extent; the bank-conflict `pad` is applied later by the post-assemble `tile/pad`
   local pass, which can see the materialized byte span). **Coalesce slabs by `(buffer, access, distance)` — this is
   dedup, by construction.**
4. Expand `distance>0` reads into prologue/steady/epilogue + waits, transport-parametrically.
5. Expand the `role` coloring into the warp-spec `Cond` + synthesized role axis + mbarrier ring + `SetMaxNReg` +
   consumer named-barrier.
6. Replicate REGISTER-bound blocks per cell (SSA def-use closure first); for ATOM blocks, synthesize fragments + the
   `ldmatrix`/`mma.sync`/`RegStore` chain from the derived `atom`.
7. Emit each block's derived guards (`domain.real_extent`, partition `K_s==0`/`lane==0`) as `Cond`s, coalesced across
   co-scoped blocks.

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

## Migration sequencing (each step gated on byte-identical CUDA + `make test`)

1. **Define the algorithm DAG + derived projections + `assemble`**, and make the composer emit a DAG + reference
   Schedule that `assemble`s to today's tower byte-identically for every golden (pointwise/coop/matmul/flash, static +
   `.dynM`). No move exists yet. *Highest-risk step; do it first.*
2. **Port `hoist`** (the `scope` default) + delete `_assemble_matmul_prologue` and `dag.prologue`. → byte-identical.
3. **Port `stage` + `partition_reduce`** — `020`/`026` collapse into the `staged` annotation + automatic slab
   coalescing; split-/cooperative-K become the combine-block move. → byte-identical.
4. **Port `retime` + `promote_transport` + `specialize_warps`** — `040`/`080`/`050`/`060`/`085` become Schedule
   annotations; their tree surgery moves into `assemble`. → byte-identical, then perf-equal under tune.
5. **Port the locals + `atomize`** and retire the tower builders. End state = the pass list in "The pass structure":
   deterministic `tile/build` → the enumeration passes (`tensorize` / `partition_reduce` / `register_tile` / `stage` /
   `retime` / `transport` / `warp_spec` / `locals`) → one `tile/assemble`.

The DAG and the tree can coexist through 1–4: the composer builds the DAG, `assemble` lowers to the *current* tower, and
the not-yet-ported tree passes run unchanged on it. Big-bang is avoided.

**Status (branch `feature/tile-ir-block-dag`).** Steps 1–2 are done: `assemble_block` lowers pointwise + scalar/warp
matmul + cooperative reduce byte-identically (`build_dag.py` → `assemble.py`), the legacy pointwise/matmul materializers
are deleted, and the 11 tower-rewrite scheduling passes (`021`–`090`) have been **deleted up front** to clear the deck —
they are the predecessors the `tile/stage` … `tile/locals` enumeration passes are reborn from, not yet reimplemented as
`Schedule` moves. The staging regimes (warp MMA / coop / flash) still ride `materialize.py`'s tower builders in the
interim. Next is steps 3–5: stand up the enumeration passes over the tile node and grow `assemble`'s `Schedule`-driven
synthesis to subsume them.

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
choice is a `Schedule` annotation, and `assemble` is the one place the tower is built. `assemble` is now proven
byte-identical on pointwise / scalar+warp matmul / coop reduce (steps 1–2). The natural next step is to **stand up the
first `Schedule`-move enumeration pass — `tile/stage`** — over the tile node (the simplest staging regime, reborn from
the deleted `020`/`026`), with `assemble` synthesizing the slab + cooperative producer from `staged`. That validates the
two-kind pass structure end to end (enumeration widens, `assemble` materializes) on a regime that actually touches smem,
before `retime` / `transport` / `warp_spec` follow.
