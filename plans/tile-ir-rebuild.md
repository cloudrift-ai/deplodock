# Tile IR rebuild

Status: **the scalar + cooperative + cross-CTA-split tiers have landed, plus the scalar register-tile and the
dynamic-grid tier.** `010_recognize` is the sole Loop-IR → Tile-IR boundary (it lifts every kernel to a `TileOp`
carrying one op-tree node — `Map` / `Monoid` / `Semiring` — plus a typed schedule); `020_schedule` maps the free axes
onto the grid and picks the reduce partition + the free-axis output tile; `030_split` consumes a cross-CTA `GRID` stage
as a graph rewrite; `lowering/kernel` materializes. Every elementwise, reduction, online-softmax, RMSNorm, matmul,
scalar-flash, and whole-transformer-block kernel is recovered, plus the **cooperative (BLOCK) reduce** tier (whole-CTA +
strided rows), the **register-fold ILP (REG)** tier (independent per-thread accumulator chains), the **scalar
register-tile (`TILE` codec)** tier (each thread owns a `reg_m·reg_n` block of output cells, operands reused across
them, the small inner reduce unrolled), the **dynamic-grid** tier (a symbolic free / output axis → a symbolic launch
grid sized from the runtime `Dim` arg; register-tiled symbolic axes mask their tail), and the **cross-CTA (GRID) split**
tier (`030_split`: the `g<n>[a|k]` codec — `atomicAdd` one-kernel finalize OR a deferred `__partial`-workspace +
sibling combine kernel; additive `sum` / split-K matmul AND the twisted flash `(m, l, O)` split-KV, carrier-generic via
`as_state_merge`). All of these work over **static AND symbolic** axes, plus the **warp / tensor-core tier** — the
`Semiring·Warp` gmem-direct `mma.sync` matmul (the `WARP` codec + the `_warp` materializer): canonical + transposed-B,
f16/f32 out, fused projection / causal epilogues, static + dynamic-grid (pin-only). **Remaining: operand staging (smem +
`ldmatrix`, cp.async / TMA), MMA split-K (the `Warp` contraction through `030_split`), symbolic-K masked tiles,
warp-tier flash (the `Monoid·Warp` fragment), and warp specialization.** Branch `refactoring/tile-ir-rebuild`.

This doc is now thin: the executed phases live in git history and the `ARCHITECTURE.md` files. What remains is the
**recovery contract** (the governing invariant), the **xfail mechanism** (still driving the rebuild), a one-paragraph
record of what landed, and — the substance — the **settled schedule design** that the warp-tier recovery builds on.

## The recovery contract

`tests/compiler/e2e/` is the **only** thing the rebuild must satisfy. Every file there is black-box: it builds a
graph, runs it through `CudaBackend`, and compares the output to a numpy/torch reference (a handful also assert on
generated CUDA source — part of the contract too). None assert on tile-IR Python objects, so they survive any internal
redesign. The rebuild is "done" when the whole `e2e/` suite is green with an **empty** xfail registry.

## Integration-test xfail mechanism

A single registry drives expected failures during the rebuild — no scattered `@pytest.mark.xfail` decorators.

- **Registry:** `tests/xfail_registry.py` exports `XFAIL: dict[str, str]` mapping a **test node-id substring** to a
  one-line reason. `"test_foo.py"` xfails a whole file; `"test_foo.py::test_bar"` xfails one case. The
  `pytest_collection_modifyitems` hook in `tests/conftest.py` marks every collected item whose `nodeid` contains a
  registered substring with `xfail(strict=False)`.
- **Recovery semantics:** `strict=False` means a test that starts passing reports as **XPASS**, not a failure — the
  signal a capability came back. Delete its entry and it reverts to a required test. **An empty `XFAIL` means the
  rebuild is fully recovered.** It shrinks as each warp-tier capability lands.
- **A file whose module-level import of a tile symbol breaks** fails at collection (before any item exists to mark) —
  guard the import (`try/except ModuleNotFoundError`) and register the nodeids, rather than letting a collection error
  return.

**Deleting unit tests while rebuilding:** a tile-IR **unit** test (imports `deplodock.compiler.ir.tile` or
`…passes.lowering.tile` and asserts on those objects — tile graphs, schedules, fragments, enumeration counts — instead
of running a graph and checking output) → **delete it**, don't port it; the new internals get new unit tests. An
**integration/accuracy** test (anything in `tests/compiler/e2e/`) → **never delete or weaken** it; xfail it if the
in-progress rebuild breaks it, and it flips back to a hard requirement when the capability returns.

## What has landed (scalar + cooperative + register-tile + dynamic-grid)

Recorded compactly; details in git + `ARCHITECTURE.md`:

- **The skeleton and the scalar tiers** — pointwise (`Map`), the per-cell reduce (`Monoid`, plain reduce + online
  softmax unified by the **twist**), scalar matmul (`Semiring`), and scalar flash (the nested `Monoid(Semiring)`), all
  on one generic `Tile` thread-decode + `lower` path.
- **The high-level op tree** (`ir/stmt/algebra.py` + `ir/tile/ops.py`) — the lift `Map`, the carrier `Monoid` + `Twist`,
  the `Semiring` contraction. A carrier **dissolves** into its fold `Accum`s; seeding is one path (`op.identity` via
  `Loop.render`), no explicit `Init`.
- **The op tree as flowing IR** — `010_recognize` is the sole Loop-IR → Tile-IR boundary; free axes ride the schedule's
  `Placement`, not the algebra node.
- **The typed schedule + the cooperative (BLOCK) reduce** — `TileOp` carries a typed `Kernel` (op + `*Schedule`); the
  reduce axis partitions across the BLOCK level (whole-CTA + strided-cooperative rows), the cross-thread combine is
  **derived** from the level, static AND symbolic reduce axis. This established the schedule type system below.
- **The REG (ILP) reduce level** — the `ReducePlan` `reg` width gives each thread `reg` independent register-accumulator
  chains (breaking the serial fold's loop-carried dependency), folded by a register tree (`carrier.as_state_merge`,
  carrier-generic) before the cross-thread combine. The materializer replicates the reduce body `reg`×, offset `r·coop`,
  in one `StridedLoop` of step `coop·reg`; a symbolic / non-divisible tail is **clamp-to-identity** (in-bounds `% extent`
  read + value masked to the fold identity). Composes with coop or stands alone (`coop = 1`). `reg = 1` default (ILP via
  the `REDUCE` `r<n>` pin / prior fork, not the cold path).
- **The scalar register-tile (`TILE` codec)** — the `Scalar`-fragment sibling of `REDUCE` (`TilePlan` on
  `SemiringSchedule`, decided in `020_schedule`, materialized by `_reg_tile`). The codec `n<N>[xm<M>]` (parallel
  thread-tile) `/ f<fn>[xf<fm>]` (register sub-tile) gives each thread a `reg_m·reg_n` block of output cells: the
  reduce-loop body is replicated per cell, **operand loads deduped** (an `A[m,k]` load is shared across the `n` cells,
  `B[k,n]` across the `m` cells — the arithmetic-intensity reuse), the small inner reduce `#pragma unroll`'d, the
  parallel widths set `block_threads` (`par·reg == extent` ⇒ one CTA). A non-divisible / symbolic tiled axis masks its
  tail (clamp-read `% extent` + a guarded store). The featurizer sources its `_free_slots` from the `TILE` codec.
- **The dynamic-grid tier** — a symbolic FREE / output axis rides a symbolic launch grid: `Tile` sizes the guard +
  decode from the runtime extents (`_gid < ∏extents`), `lowering/cuda` builds the `gridDim` as `ceil(∏extents /
  blockDim)` with the symbolic `Dim` resolved at launch (`runtime_args`). Recovered dynamic elementwise / linear /
  rmsnorm / sdpa and the dynamic Qwen layer + whole-model (+ capture-replay). Composes with the register-tile (a
  symbolic register-tiled axis masks its tail).

## The schedule design

The thesis: **the schedule is separate from the combine, and the schedule is defined by the *operation*, not by the
execution unit.** The combine (the ⊕) lives in the op tree (`ir/stmt/algebra.py`); the schedule — which axes are
parallel, how the reduce axis partitions, how a cell is realized — lives on a typed `*Schedule` paired with the op node
in a `*Kernel`. The pairing makes a kind/schedule mismatch unrepresentable; the variant is keyed by `type(op)`, no
`classify_algebra` tag.

### The two axes: kind × fragment

A schedule varies along two orthogonal axes:

1. **The algebra kind** (the *operation*) — `Map` (pointwise), `Monoid` (reduction / online softmax / flash), or
   `Semiring` (contraction). This is **primary**: it determines the schedule's *structure* (a `Monoid` schedule has the
   streaming-reduce axis + the carrier combine + the projection; a `Semiring` schedule has the contraction). Flash is a
   `Monoid` over a nested partial `Semiring`, so it is a **`Monoid` schedule** — the op tree nests, the schedule is flat,
   typed by the outermost kind.
2. **The fragment** (the per-output compute unit) — `Scalar` (one thread per cell, register accumulator) or `Warp`
   (a tensor-core mma tile: `WM·WN` warps per output tile, an mma C-fragment accumulator). This is **secondary**: it
   parameterizes the kind's schedule, picking the per-cell *realization* without changing the structure. It mirrors the
   kernel IR, where `FragmentApply` / `FragmentRowReduce` / `FragmentMask` are already the carrier-generic fragment-tier
   **siblings** of the scalar `Assign` / `WarpShuffle` / `Select`.

So scalar-vs-warp is **not** a different schedule class — it is the `fragment` field of the same kind's schedule. The
combine *mechanism* is derived from `(level, fragment)`: a `Warp` reduce row-reduces through `FragmentRowReduce`, a
`Scalar` reduce through `WarpShuffle` / `TreeHalve` — derived, never stored.

### The matrix

| `op` kind \ tier              | uniform · `Scalar`         | uniform · `Warp`         | `WarpSpec` |
|-------------------------------|----------------------------|--------------------------|------------|
| **`Map`** (pointwise)         | `MapSchedule`              | —                        | —          |
| **`Monoid`** (reduce / flash) | `MonoidSchedule(Scalar)`   | `MonoidSchedule(Warp)`   | `WarpSpec` |
| **`Semiring`** (contraction)  | `SemiringSchedule(Scalar)` | `SemiringSchedule(Warp)` | `WarpSpec` |

The Scalar / Warp columns are the **same class, different `fragment`** — "defined by the operation, not the block."
`Map` has no fragment axis (pointwise never accumulates, so never tensor-cores). The per-kernel schedule union is each
row:

```python
MapKernel.schedule      = MapSchedule
MonoidKernel.schedule   = MonoidSchedule  | WarpSpec
SemiringKernel.schedule = SemiringSchedule | WarpSpec
```

Five concepts total: three kind-schedules + two fragments + one `WarpSpec`. The Scalar half (`MapSchedule`,
`MonoidSchedule`, `SemiringSchedule`, the `ReducePlan` partition) is **built**; the `Warp` fragment and `WarpSpec` are
the **warp-tier recovery** (below).

### Why not a shared `WarpSchedule`

A tempting alternative — one kind-agnostic `WarpSchedule` reused across `Monoid` and `Semiring` — was **rejected**.
The two only *look* alike (same field names); their meaning differs: a `Semiring` warp tile's reduce axis and its mma
tile describe the **same** axis (K) at two granularities, while a flash `Monoid` warp tile's reduce axis (KV) and its
mma tile describe **different** axes (the tile is a *nested* contraction inside the streaming reduce). Deduping by
field-shape would force that kind-dependent relationship into an unwritten convention. Keeping the kind primary and the
warp tile as a `Warp` fragment of `MonoidSchedule` / `SemiringSchedule` makes each relationship intrinsic to its type.

`WarpSpec`, by contrast, **is** genuinely shared (one struct, no per-kind variants) — and for a principled reason: it
**delegates**. It holds `roles`, each carrying a `(kind, fragment)` sub-schedule, so it has no kind-specific structure
of its own. Sharing a container that delegates is sound; sharing a `WarpSchedule` whose fields encode kind-dependent
meaning is not.

### The structures

```python
# Kind-neutral free-axis → grid binding. on_grid() binds every free axis (scalar tier);
# the warp fragment reads place.grid at CTA-tile granularity instead of per-cell.
@dataclass(frozen=True)
class Placement:
    free: tuple[Axis, ...] = ()
    grid: tuple[Axis, ...] = ()

# The reduce-axis partition — tuned widths only, coarse→fine. ReduceStage(level, width) per
# level (GRID / BLOCK / REG / SERIAL); the combine MECHANISM (Fold) is derived from the
# level (+ fragment), not stored. serial = ceil(extent / parallel), derived. GRID = the
# cross-CTA split request (see "Kernel splits").
@dataclass(frozen=True)
class ReducePlan:
    stages: tuple[ReduceStage, ...] = ()
    # .of(cta=, coop=, reg=), .parse/.spell (the REDUCE codec g<n>/b<n>/r<n>), .needs_split, .coop, .cta, .reg

# The per-output compute-unit realization — the secondary (fragment) axis.
class Fragment: ...
@dataclass(frozen=True)
class Scalar(Fragment): ...                     # one thread / cell, register accumulator
@dataclass(frozen=True)
class Warp(Fragment):                           # WM·WN warps / tile, mma C-fragment accumulator
    tile: WarpTile

# The mma output tile (the Warp fragment's geometry): atom + warps + register sub-tiles + the
# K-chunk. tile_m = WM·FM·atom_m, tile_n = WN·FN·atom_n, block_threads = WM·WN·32.
@dataclass(frozen=True)
class WarpTile:
    atom: AtomKind                              # the mma cell (mma_m16n8k16_f16 today)
    warps: tuple[int, int]                      # (WM, WN)
    reg: tuple[int, int] = (1, 1)               # (FM, FN) register atom sub-tiles per warp
    bk: int = 1                                 # K-chunk per inner step, in atom_k units
    # .parse/.spell (the WARP codec a<atom>/w<m>x<n>/f<m>x<n>/k<n>)

# Operand transport over the reduce loop (built: the warp tier's smem pipeline + the scalar
# fused-prologue shared row). stage=None = gmem-direct; Stage(d1/sync) = real single-buffer
# staging (a plain __syncthreads fill, no prefetch); cp / tma add async / descriptor transport.
@dataclass(frozen=True)
class Stage:
    depth: int = 1
    transport: str = "sync"                     # sync | cp.async | tma
    smem: tuple[str, ...] = ()
    ring: bool = False

@dataclass(frozen=True)
class MapSchedule:                              # pointwise — no fragment, no reduce
    place: Placement

@dataclass(frozen=True)
class MonoidSchedule:                           # streaming reduce / online softmax / flash
    place: Placement
    fragment: Fragment = Scalar()
    block: tuple[Axis, ...] = ()                # free axes resident in the CTA (strided-coop rows)
    reduce: ReducePlan = ReducePlan()
    stage: Stage | None = None

@dataclass(frozen=True)
class SemiringSchedule:                         # contraction
    place: Placement
    fragment: Fragment = Scalar()
    reduce: ReducePlan = ReducePlan()           # the K (contraction) axis partition
    stage: Stage | None = None
```

`__post_init__` asserts the fragment's coherence with the dependent fields (e.g. a `Warp` fragment forbids a BLOCK
`ReducePlan` stage — the warp *is* the cooperation; only GRID split + serial remain) so an illegal combo fails loud
rather than miscompiling.

### Warp specialization (`WarpSpec`) — the third either-arm

The uniform tiers run **homogeneous** warps (every warp does the same job). `WarpSpec` runs **heterogeneous** warps —
the CTA's warps partition into producer / mma / reducer roles wired by shared smem rings. It is a generic container:

```python
@dataclass(frozen=True)
class Channel:                                  # a shared smem ring — the producer/consumer seam
    name: str
    depth: int                                  # ring slots (the shared fill↔drain depth)
    transport: str = "cp.async"                 # how the producer fills it: cp.async | tma

@dataclass(frozen=True)
class WarpRole:                                 # one warp group's job; its sub-schedule type NAMES the role
    stage_node: object                          # the op-tree node this role runs
    warps: int
    schedule: MapSchedule | MonoidSchedule | SemiringSchedule   # producer = Map; mma = Semiring/Monoid(Warp);
    reads: tuple[str, ...] = ()                                 #   reducer = Monoid(Scalar, coop ReducePlan)
    writes: tuple[str, ...] = ()
    stage: Stage | None = None                  # this role's LOCAL pipeline (e.g. consumer smem→reg double-buffer)

@dataclass(frozen=True)
class WarpSpec:
    place: Placement                            # the CTA-tile grid
    channels: tuple[Channel, ...] = ()
    roles: tuple[WarpRole, ...] = ()            # Σ role.warps = the CTA warp count
```

The single uniform `Stage` **splits** under warp-spec: the gmem→smem *fill* pipeline becomes the shared `Channel`
(depth + transport, since producer and consumer must agree), while each consumer's *local* register double-buffer stays
on its own `role.stage`. `WarpSpec` appears only at the top CTA-level schedule; roles bottom out in uniform schedules —
no nesting.

### Kernel splits — a schedule request, a graph rewrite, a materializer invariant

"Launch a separate kernel" is **not** a schedule field — a schedule describes exactly one launch. The split lives in
the **graph** (two `TileOp` nodes). What the schedule carries is the *request*: a `ReduceStage(GRID, n)` in the
`ReducePlan` (so `needs_split`). The lifecycle:

1. **`020_schedule` — decide:** a `GRID` stage lands in the `ReducePlan` (the partition's coarsest level — uniform
   across split-K on `Semiring`, split-reduce on `Monoid`, split-KV on flash).
2. **`030_split` — consume** (last tile pass): realize the `GRID` stage as either **atomics** (the partial kernel
   atomic-adds into the output — `ReduceStage(GRID).combine() → (ATOMIC,)`, the split axis folds into `place.grid`, **1
   node**) or **partial + finalize** (the partial kernel writes a `ws[split, *free]` workspace; the finalize kernel is an
   ordinary `Monoid` reduce over the split axis via `carrier.as_state_merge`, **2 nodes**). The finalize needs no new
   type — splitting a reduce yields a smaller reduce; the algebra is closed under it. **No `GRID` stage survives** (the
   partial consumed it into the grid; the finalize is a fresh `ReducePlan()`; the atomic case marks the store, not the
   partition).
3. **`010_materialize` — assume** (first kernel pass): `assert not schedule.reduce.needs_split`. The materializer only
   ever lowers a single-launch kernel — a `ReducePlan` of `{SERIAL, REG, BLOCK}` stages — so it carries no multi-kernel
   logic. Any cross-CTA combine was already realized upstream.

So the schedule carries the partition (including the split request); the **graph** carries the kernel count.

### The knob schema — shared orthogonal codes, not per-class

A schedule is **spelled** by a small set of **orthogonal codes**, one per tunable sub-component. A kernel's full
config is the **union** of the codes that apply to it — NOT one monolithic code per `*Schedule` class. The schedules
recompose the same sub-components (a `ReducePlan` lives on both `Monoid` and `Semiring`; a `WarpTile` on both `Warp`
fragments; a `Stage` on every reduce loop), so a per-class code would re-encode shared concepts and spell them
differently. The payoff is precisely two things — and **not** a third that's easy to over-claim:

- **No duplicated concept** — one vocabulary for the reduce partition, the tile, the warp fragment, the transport,
  wherever each appears. The knob schema mirrors the schedule's *fields* and the prior's *feature families*
  (`_reduce_decomp` / `_free_slots` / `_warp_tile_features`).
- **Learned-feature generalization across kinds** — because the featurizer reads ONE code schema-agnostically (today
  `_reduce_decomp` reads the raw `REDUCE` key), a `b32` reduce featurizes identically on a `Monoid` and a `Semiring`, so
  the *learned model* transfers what it knows about reduce width across kinds.
- **NOT cross-kind measured-evidence sharing.** `op_cache_key` digests the op-tree body + kind (`keys.py`), so a
  `Monoid` and a `Semiring` `perf`/reservoir row never share a key regardless of spelling — the *measured* `evidence_pick`
  is siloed by kind either way. Shared spelling buys the learned-feature transfer above, not measured-row reuse; don't
  expect the latter.

| code | sub-component (schedule field) | schedules | grammar (coarse→fine) | status |
|---|---|---|---|---|
| `REDUCE` | `ReducePlan` (reduce-axis partition) | `Monoid`, `Semiring` | `g<n>[a\|k]` / `b<n>` / `r<n>` · empty = serial | **built** |
| `TILE` | free-axis output tile (`TilePlan` — par + `Scalar` reg) | `Semiring` (`Map`/`Monoid` reserved) | `n<N>[xm<M>]` parallel · `f<fn>[xf<fm>]` reg · empty = per-cell | **built** (`Semiring·Scalar`) |
| `WARP` | the `Warp` fragment (`WarpTile`) — replaces `TILE`'s realization | `Monoid`, `Semiring` | `a:<atom>` · `w<WM>xw<WN>` · `f<FM>xf<FN>` · `k<bk>` | **built** (`Semiring·Warp`, gmem-direct + epilogues; pin-only) |
| `STAGE` | `Stage` (operand transport over the reduce loop) | `Monoid`, `Semiring` | `d<depth>` · `sync\|cp\|tma` · `[ring]` | **built** (warp tier: cp.async + TMA, static/masked-M/symbolic-K; scalar fused-prologue shared row; pin-only) |

**Delimiter hierarchy** (so codes survive the `DEPLODOCK_KNOBS` / `run --ab` parser, which splits on `,` and requires
`=` per entry — `knob.py::parse_knob_spec`): **`,` is reserved** as the knob-list separator and MUST NOT appear inside a
code value. Within a value: `/` separates fields, `x` pairs dims, `:` introduces a name (`a:<atom>`), `;` lists
(e.g. a multi-`Channel` `WarpSpec`). Sub-field order is fixed **m-then-n** everywhere (`f<FM>xf<FN>`, `w<WM>xw<WN>`) to
match the legacy `FM`/`FN`/`WM`/`WN` order — but note `_free_slots` re-canonicalizes the two free slots by parallel
width (wider = the `n`/coalesced slot), so the codec's `m`/`n` labels are nominal, not the feature's source of truth.

Three rules keep it unambiguous (the same separation the type design already draws):

1. **Vocabulary is shared; interpretation is per-class.** A code names a single-meaning sub-component; the `*Schedule`
   *type* resolves which axis it targets. `REDUCE b32` = "32-wide BLOCK partition of *the* reduce axis" — `Monoid`
   reads KV, `Semiring` reads K. This is why a shared knob does NOT reintroduce the ambiguity that a shared
   `WarpSchedule` *type* would: the K-vs-KV relationship stays in the type, never in the code.
2. **Fragment is implicit in which output code is present.** `TILE` ⇒ `Scalar` (`n`/`m` are threads); `WARP` ⇒ `Warp`
   (`n`/`m` are warps; `WARP` carries its own reg `f` + atom + `k`-chunk). Never both — the either-ness mirrors the
   `fragment` field.
3. **Grammar (validity) is per-(kind, fragment); vocabulary is not.** `Map` offers only `TILE`; `Monoid·Scalar` offers
   `REDUCE`+`TILE`+`STAGE`; `Semiring·Warp` offers `WARP`+`REDUCE`+`STAGE`. The enumeration (the `020_schedule` fork)
   decides which codes/values are valid candidates per cell; the codec vocabulary is the same everywhere.

`WarpSpec` reuses the role vocabulary (`REDUCE` reducer / `WARP` mma / `STAGE` producer) but DOES add one piece of
grammar: **role-namespacing** (a flat `TileOp.knobs` dict can't hold two roles each with a `STAGE` otherwise). A
role-prefixed key (`mma:WARP=…`) plus the `CHANNEL` ring is genuine new vocabulary — it must be registered so
`knob_features` / `apply_off_defaults` / `tuning_knob_items` handle role-prefixed keys, not silently drop them. Worked
examples (✅ built, 🔲 proposed; `;` lists, never `,` — see the delimiter hierarchy):

```
✅ Monoid·Scalar  REDUCE=b64/r4          # 64-way coop + 4 ILP register accumulators
✅ Monoid·Scalar  REDUCE=b16  TILE=n8    # strided-coop rows: 16 lanes, 8 output rows/CTA
✅ Monoid·Scalar  REDUCE=b32             # flash-decoding (cooperative-KV, scalar flash)
✅ Monoid·Scalar  REDUCE=g2a             # cross-CTA split-reduce, atomicAdd finalize (one kernel)
✅ Monoid·Scalar  REDUCE=g2k             # cross-CTA split-reduce / flash split-KV, deferred __partial + combine kernel
✅ Semiring·Scalar REDUCE=g2a            # split-K matmul, atomicAdd finalize (1-component additive carrier)
✅ Semiring·Scalar TILE=n128xm64/f4xf4   # register-blocked SGEMM: 4×4 reg cells/thread, operands reused, inner reduce unrolled
✅ Semiring·Warp  WARP=a:mma_m16n8k16_f16/w2xw2/f4xf8/k2   # gmem-direct mma.sync, 128×128 tile, 128 threads (+ fused epilogue / transposed-B / dynamic-grid)
✅ Semiring·Warp  WARP=a:mma_m16n8k16_f16/w2xw2/f2xf2/k2  STAGE=d2/tma   # + TMA operand staging (cp.async.bulk.tensor)
🔲 Semiring·Warp  WARP=a:m16n8k16_f16/w2xw2/f2xf2/k2  REDUCE=g4k  STAGE=d3/cp   # split-K=4 kernel-finalize (staging + split still proposed)
🔲 Monoid·Warp    WARP=a:m16n8k16_f16/w4xw1/f1xf1/k1  REDUCE=b2  STAGE=d2/tma   # warp-tier flash, coop-KV
🔲 WarpSpec       CHANNEL=K:d3/cp;V:d3/cp  mma:WARP=…/k2  reducer:REDUCE=b2  producer:STAGE=d3/cp
```

### Wiring cost & open gaps (from the design review)

The cost is **not** "only the source keys move" — featurization is **codec-aware** (as `_reduce_decomp` already parses
the raw `REDUCE` key). Concretely, before the warp tier can use this schema:

- **Codec-aware tier discriminator (blocking).** `is_warp` / `mma_atom` / `_free_slots` / `_warp_tile_features` key off
  scalar names (`WN`/`WM`/`ATOM@`) today; a packed `WARP=…` value has none, so a warp kernel would featurize through the
  *scalar* path. Either keep `WM`/`WN`/`ATOM@`/`FM`/`FN` as the on-dict representation and treat `WARP`/`TILE` as the
  pin/display spelling that explodes into those keys at stamp time (featurizers unchanged), **or** make all four entry
  points parse the codec. Pick one; the former is the smaller change and keeps `tile_signature` golden-matching working.
- **`REDUCE`'s finalize letter round-trips (landed).** `ReduceStage.finalize` carries `"atomic"` | `"kernel"`;
  `ReducePlan.parse`/`spell` round-trip the `g<n>[a|k]` letter and `ReducePlan.finalize` reads it (`030_split` picks the
  atomic vs deferred-kernel arm off it). `_Decomp.finalize` / `D_finalize_kernel` therefore featurize correctly. (The
  auto-fork still doesn't *propose* `g` — cross-CTA split is pin-/prior-driven, like `r`/ILP.)
- **OFF-sentinel & validity need a codec-aware home.** `apply_off_defaults` / the tier filters work per *named* knob; a
  packed code has no per-subfield OFF, so "decided: not a warp kernel" needs an explicit empty-codec OFF (`WARP=""`).
  And the per-(kind,fragment) validity must live as a **codec-dict validator** (where pins / the featurizer run), not
  only a schedule `__post_init__` — a pinned illegal union (`TILE`+`WARP`) otherwise constructs and mislowers silently.
- **Type reconciliation.** The "kind × fragment" section types the fragment as `Fragment = Scalar() | Warp(tile)`, but
  the *built* `schedule.py` carries `warp_tile: WarpTile | None` (no `Fragment` type yet) and has no `__post_init__`.
  Land the `Fragment` type (or restate the section against `warp_tile`) so the codec rules ("`TILE`⇒Scalar / `WARP`⇒Warp")
  have something to assert against.
- **Unspellable today (note, not block).** `>2` free axes (batched outputs) and the `block` strided-rows field beyond
  one packed axis have no dedicated code; `STAGE.depth` / `WarpTile.bk` / `Channel.depth` are three interacting depths,
  not fully orthogonal.

`REDUCE`, `TILE`, and `WARP` are wired (reachable via pin; `REDUCE` `r<n>` ILP, `TILE` reg-tile, `WARP` gmem-direct
mma.sync). Each one's auto-fork — the prior-ranked candidate set — is still a follow-up, so a cold greedy compile stays
scalar per-cell unless pinned. `STAGE` remains proposed.

## Remaining — the warp / tensor-core tier

The `Warp` fragment + `WarpSpec` are the recovery target. The kernel-IR tensor-core vocabulary **survived the
demolition** (`RegFragment`, `LdmatrixLoad`, `MmaSyncPtx`, the carrier-generic `FragmentApply` / `FragmentRowReduce` /
`FragmentMask`, `RegEpilogue` / `RegStore`, the `FragLayout` geometry seam, cp.async / TMA / mbarrier, the
`dpl_mma_m16n8k16` renderer), as did the matmul knob schema (`WM` / `WN` / `FM` / `FN` / `BK` / `SPLITK` / `ATOM@`) and
the prior featurization (`is_warp`, `_warp_tile_features`). What is gone is only the **tile-tier decision**
(`020_schedule` building a `Warp` fragment) and the **kernel-tier materializer** (`010_materialize` lowering a `Warp`
fragment / `WarpSpec` into those primitives). The phased recovery, feature by feature:

1. **Static fp16 mma matmul** — ✅ **LANDED.** The `Warp` fragment (`WarpTile` on `SemiringSchedule`) + the `WARP`
   codec (decided pin-only in `020_schedule`, exploded into the on-dict `ATOM@`/`WM`/`WN`/`FM`/`FN`/`BK` featurizer
   keys) + the `_warp` materializer (`010_materialize`): free axes split four ways (GRID/WARP/REGISTER/ATOM), the
   atom-lane offset decoded at render, a gmem-direct `LdmatrixLoad(staged=False)` a/b + `MmaSyncPtx` K loop per register
   cell, `RegStore` out. Canonical + transposed-B, f16/f32 out, static + dynamic-grid (symbolic M). The legacy-API
   `test_matmul_mma.py` / `test_matmul_mma_transposed_b.py` + the demolished rule tests were dropped; the capability is
   covered by the `WARP`-codec matrix in `test_matmul_tile_coverage`.
2. **Fused epilogues** — ✅ **LANDED.** The projection `Map` tail over the `Semiring` folds into the `RegStore`
   `RegEpilogue` (residual / bias / scale), the out-dtype cast on `RegStore` (`__floats2half2_rn`), a causal `Select` →
   per-element ternary. Covered by the epilogue matrix in `test_matmul_tile_coverage` (static + dynamic). Still gmem-direct
   (no staging). **Remaining phases below: operand staging, MMA split-K, symbolic-K edges, warp-tier flash.**
3. **Operand staging** — make `Stage` real: smem slabs + `ldmatrix`, double-buffered via cp.async (sm_80) / TMA (sm_90).
   Recovers `test_stage_scalar.py`, `test_matmul_mma_parity.py` (cp.async / tma), the blocked-prologue tests.
4. **Cross-CTA split** — `030_split` (the `GRID` stage → atomic / partial+finalize above) has **landed for the scalar
   tier**: additive `sum` split-reduce, split-K scalar matmul, and the twisted flash `(m, l, O)` split-KV
   (carrier-generic via `as_state_merge`), both atomic and deferred-kernel arms (the `test_cross_cta_finalize` matrix).
   What remains under the warp tier is **MMA split-K** — `test_mma_atomic_free_splitk.py` and the split-K rule tests —
   which needs the `Warp` fragment's contraction to feed the same `030_split` rewrite.
5. **Symbolic M / N / K edges** — masked mma tiles (per-element `RegStore` guards, clamped / zero-filled K slabs).
   Recovers `test_matmul_mma_masked.py`, the dynamic mma-parity / causal-epilogue / dynamic-shape matmul cases.
6. **Warp-tier flash** — the `Warp` fragment on a `MonoidSchedule` (flash's inner QK / PV), reusing the fragment
   machinery; cooperative-KV / split-KV composes the `ReducePlan` with the `Warp` fragment. Recovers
   `test_flash_tensorcore_generated.py`, the dynamic flash variants.

Then **warp specialization** (`WarpSpec`) and the remaining **operand-pipelining** depth tuning. The rebuild is complete
when `e2e/` is green and `XFAIL` is empty.
