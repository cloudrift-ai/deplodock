# Cooperative reduction in the rebuilt tile IR

## Context

The rebuilt tile IR (`refactoring/tile-ir-rebuild`) lowers every `MONOID`/`SEMIRING` reduce at the **scalar tier**:
`010_recognize` lifts a kernel to a `TileOp` carrying one op-tree node + a `Schedule(free, grid)`, `020_schedule`
binds `free→grid` (one thread per output cell), and the reduce axis stays a fully **serial** per-thread `Loop`. Results
are correct but a K=512 row is one thread doing 512 serial steps. This change introduces the **parametrization and
type system** for a cooperative reduction schedule (partition the reduce axis across the execution hierarchy; the
cross-thread combine is derived per level) and implements the **static `coop` (BLOCK) tier** end to end, leaving
register-fold (`reg`/ILP), cross-CTA split (`cta`), symbolic, flash, the tensor-core tile, warp specialization, and
operand pipelining as documented future steps / TODO slots.

Design was settled in discussion. Two assets already exist and are reused unchanged:

- **Combine algebra** — `Monoid.as_state_merge(other)` (`ir/stmt/algebra.py:296`) renders the cross-partition merge
  through the same `merge` machinery as a streaming step; `State.inits()` seeds partial accumulators. No algebra change.
- **Combine primitives** — `WarpShuffle` (`__shfl_xor_sync` butterfly, `__activemask`-correct, pow2 ≤ warp) and
  `TreeHalve` (smem tree, pow2 ≤ blockDim) in `ir/kernel/ir.py:599,526`. Their fields already match the carrier surface
  (`state`/`state_b`/`combine_states`/`length`/`dtype`/`tid_var`).

## The settled type system

A reduction's only freedom is **how the reduce axis is partitioned across hardware levels**; the combine *mechanism*
at each level is derived from the level, and the combine *algebra* rides the carrier. Schedule stays separate from
combine.

New module `deplodock/compiler/ir/tile/schedule.py` (split out of `ir.py`):

- `Level(Enum)` — `GRID, BLOCK, REG, SERIAL` (coarse→fine).
- `Fold(Enum)` — `SERIAL, REG, SHFL, SMEM, ATOMIC` (the per-level combine mechanism).
- `ReduceStage(level, width)` — one level's **tuned** partition. `combine(*, warp_size, segmented) -> tuple[Fold, …]`
  is the **derived** mechanism (the old `derive_combine_plan` logic, now a read on the stage): SERIAL/REG → `()`;
  BLOCK → `(SHFL,)` if `segmented or width ≤ warp`, `(SHFL, SMEM)` if `width % warp == 0`, else `(SMEM,)`; GRID →
  `(ATOMIC,)` (emitted by `030_split`, never by the in-kernel walk). Power-of-two `width` enforced for BLOCK.
- `ReducePlan(stages: tuple[ReduceStage, …])` — the kernel's single reduce partition. `stages=()` is the scalar serial
  fold. `parallel` = ∏ widths; `needs_split` = any GRID stage. Holds **only tuned widths** — no axis (1:1 and singular
  with the kernel's one reduce carrier; the carrier owns the axis), `serial` is derived by the materializer as
  `ceil(extent / parallel)`. Classmethod `ReducePlan.of(*, cta=1, coop=1, reg=1)`.
- `Placement(free, grid)` — kind-neutral free-axis→grid binding (the old `Schedule` two fields). `is_mapped`,
  `on_grid()` move here.
- Three **uniform** (SIMT) schedules — the one thread/block/warp mapping: `MapSchedule(place)`,
  `MonoidSchedule(place, block, reduce)`, `SemiringSchedule(place, block, reduce, warp_tile)`. `block` = free axes
  resident in the CTA alongside coop lanes (strided-cooperative); only on the reduce variants. `warp_tile` is the
  tensor-core tile, `# TODO(warp)` — and it is a **shared** `WarpTile`, not Semiring-only: a contraction is mma-tiled
  onto the one mapping whether it is the top node (matmul → `SemiringSchedule.warp_tile`) or nested (flash's inner
  QK/PV). **Schedule is flat, typed by the *outermost* kind** (flash = `Monoid` over a partial `Semiring`, so flash →
  `MonoidSchedule`); the `op` tree nests, the schedule does not.
- One **warp-spec** arm, shared / kind-agnostic, `# TODO(warp-spec)`: `WarpSpec(place, channels, roles)` — the
  warp-role pipeline (producer transport / mma / reducer), where each `WarpRole.schedule` is itself one of the uniform
  schedules above, **scoped to that role's warps** (reducer → `MonoidSchedule` carrying the cooperative `ReducePlan`,
  mma → `SemiringSchedule` carrying `warp_tile`). Roles bottom out in uniform schedules, so `WarpSpec` only appears at
  the top CTA-level schedule — no nesting, and **no `*WarpSpec*` per-kind variants** (that combinatorial blow-up is
  avoided; `WarpSpec` is one struct). See "Future steps — operand pipelining" for `channels` / the per-role `Stage`.
- Three op+schedule pairs + union: `MapKernel(op: Map, schedule: MapSchedule)`,
  `MonoidKernel(op: Monoid, schedule: MonoidSchedule | WarpSpec)`,
  `SemiringKernel(op: Semiring, schedule: SemiringSchedule | WarpSpec)`,
  `Kernel = MapKernel | MonoidKernel | SemiringKernel`. The **schedule is EITHER the kind's uniform schedule OR
  `WarpSpec`** — the union at the field *is* the either (no wrapper class, no invariant assert); `MapKernel` omits the
  warp-spec arm (pointwise never warp-specializes). The pairing makes a `Monoid`-with-`MapSchedule` mismatch
  **unrepresentable**; the variant is keyed by `type(op)` (no `classify_algebra`, which doesn't exist). This cut builds
  only the **uniform** arm.

## Scope of this cut

**Implemented:** the full type system above; `020_schedule` builds a cooperative `MonoidKernel`/`SemiringKernel` for
**static** reduces using **conservative module constants** (in place of knobs); the materializer lowers a cooperative
`ReducePlan` (strided per-lane loop → derived `WarpShuffle`/`TreeHalve` combine → projection/store) for whole-CTA **and**
strided-cooperative rows; the cuda lowering derives `blockDim = coop · (free cells per CTA)`.

**Stubbed / TODO (reserved slots, raise `NotImplementedError`):** `reg` (REG/ILP fold), `cta` (GRID split →
`030_split`), symbolic-axis cooperative, flash (the `Monoid`-over-nested-`Semiring`; cooperative-KV + warp-tier), the
tensor-core `warp_tile`, warp specialization (`WarpSpec`), and operand pipelining (`Stage`/`Channel`).

## Files to create / change

### New

- `deplodock/compiler/ir/tile/schedule.py` — `Level`, `Fold`, `ReduceStage`, `ReducePlan`, `Placement`, the three
  `*Schedule`, the three `*Kernel`, the unions. Export all via `ir/tile/__init__.py`.
- `deplodock/compiler/pipeline/passes/lowering/kernel/_combine.py` — re-port from the demolished
  `5cd6cb6e~1:.../kernel/_combine.py`: `emit_combine(carrier, tid_var, n_threads, *, warp_size, segmented)` and the
  per-stage walk, driving `WarpShuffle`/`TreeHalve` from `carrier.twist.combine_states` / `state_b` / `state.names`.
  The combine-plan branching now lives on `ReduceStage.combine`; this module is the **emit** half (stmt construction +
  the hierarchical-smem `Smem`/`Sync`/`Cond(lane==0)` slab). Leading `_` so the pass loader skips it.
- `deplodock/compiler/pipeline/passes/lowering/tile/030_split.py` — `PATTERN=[Pattern("root", TileOp)]`;
  `RuleSkipped` unless `schedule.needs_split`; otherwise `raise NotImplementedError` with the partial+finalize design
  in comments (partial kernel: `cta`→grid axis, writes carrier state to a `ws[cta, *free]` workspace; finalize kernel:
  an ordinary reduce over the split axis built from `carrier.as_state_merge(ws_partials)`, seeded by `State.inits`).
  Documents that **splits are expressible in the algebra tile IR** — the finalize is just another reduce node.

### Changed

- `deplodock/compiler/ir/tile/ir.py` — `TileOp` carries `kernel: Kernel | None` + `name`; keep `op`/`schedule` as
  **read-only properties** projecting `kernel.op`/`kernel.schedule` (preserves `keys.py::op_cache_key`,
  `dialect_of`, `pretty_body`, materialize call sites). `pretty_body` reads `self.op`. Move `Schedule` out (re-export
  for back-compat during transition).
- `tile/010_recognize.py` — wrap the lifted node + unmapped schedule in the matching `*Kernel` by `type(node)`:
  `TileOp(kernel=MapKernel(node, MapSchedule(Placement(free=free))), name=…)` etc. (deferred flash producers
  unchanged).
- `tile/020_schedule.py` — replace the one-line `on_grid()` with: `replace(tile, kernel=replace(tile.kernel,
  schedule=_schedule_for(tile.kernel)))`. `_schedule_for` maps `place.on_grid()` for `MapKernel`; for
  `Monoid`/`SemiringKernel` over a **static** reduce axis, build `ReducePlan.of(coop=_pick_coop(extent, free))` from
  **conservative constants** (e.g. `DEFAULT_COOP`, applied only when the reduce extent ≥ a threshold and the free
  product is small enough to leave the grid under-occupied; whole-CTA vs strided `block` derived from the free-cell
  packing) and set `block` accordingly. Symbolic axis → keep `ReducePlan()` (scalar) for now. `# TODO`: replace the
  constants with knob (`knob.py::_reduce_decomp`: BR→coop, BK→serial, FK→reg, SPLITK→cta) + prior-driven selection.
- `kernel/010_materialize.py` — dispatch on `tile.kernel`: `MapKernel` → today's path. `Monoid`/`SemiringKernel` with
  a non-trivial `ReducePlan` → emit the strided per-lane reduce (`StridedLoop(axis, start=lane, step=coop)` —
  `blocks.py:259` already exists), the masked tail when `parallel ∤ extent`, then walk `plan.stages` fine→coarse
  calling `emit_combine`; the projection `Map`/output `Write` run after (guarded to one lane for scalar output, strided
  across lanes for a full-row sweep). The `Tile` axes gain the coop lanes (+ any `block` free axes). Trivial plan →
  today's serial path unchanged.
- `cuda/010_lower_kernelop.py` + `ir/kernel/render.py` — derive `blockDim` from the cooperative `Tile`'s
  `coop · ∏block-cells` instead of fixed `_BLOCK_SIZE=256`, and `gridDim` from the **output**-cell count (not the
  reduce extent); update `_launch_bounds_for` to read the same. Scalar/`MapKernel` path keeps `_BLOCK_SIZE`.
- `ir/tile/__init__.py` — export the new names.
- `tests/xfail_registry.py` — remove entries that flip to XPASS (see Verification); leave symbolic/flash/split/`reg`
  entries registered.

## Verification

1. `make test` green (correctness lane, `-Xcicc -O1`), xfail registry honored. The cooperative **accuracy** tests that
   already pass via the serial fold must stay green and now exercise the real path:
   `test_accuracy.py::test_e2e_reduce_{sum,max,softmax}_cooperative`,
   `test_reduction_combine_coverage.py::test_cooperative_combine_accuracy[*]` (9),
   `test_strided_coop_rows.py::test_2d_coop_reduce_accuracy_cuda` (BR=16 strided).
2. Structural confirmation on a static reduce: `deplodock compile --code "<reduce over a large static axis>" --ir cuda`
   shows `__shfl_xor_sync` (whole-warp coop) or a `TreeHalve` smem slab (hierarchical), and a `blockDim` reflecting
   `coop`. Cross-check `deplodock run --code … --bench` accuracy vs eager.
3. Remove the now-XPASS registry entries and re-run; confirm no XPASS remains for the implemented scope. Entries
   expected to **stay xfailed** (out of scope): `test_masked_cooperative_reduce.py` (symbolic),
   `test_flash_cooperative_kv.py` (flash), `test_reduction_combine_coverage.py::test_cross_cta_finalize_*` (split, 5),
   `test_knob_pinning.py::test_norm_linear_fp16_scalar_reduce_tma_alignment[dynamic]` (symbolic). Check
   `test_knob_pinning.py::test_sgemm_inner_reduce_is_unrolled` and `test_monoid_reduce_kernel.py` once the static
   serial/coop path lands — flip if they pass.
4. `make lint`.

## REDUCE knob codec (agreed format; NOT wired this cut)

When knob-driven selection replaces the conservative constants, the reduce partition is a **single** `REDUCE` knob
(unqualified — there is one reduce carrier per kernel; even flash / online-softmax is one `Monoid`, so no `@axis` keys
and no `REDUCE_KV`/`_DD` namespace). Value = `/`-separated, **level-named** tokens, coarse→fine:

| token | level | meaning | notes |
|---|---|---|---|
| `g<n>[a\|k]` | GRID | cross-CTA split into `n`; `a`=atomic / `k`=kernel finalize | letter only when `n>1`; TODO split (`030_split`) |
| `b<n>` | BLOCK | `n` cooperative threads | shfl/smem mechanism **derived** from `n` (`ReduceStage.combine`) |
| `r<n>` | REG | `n` register-fold accumulators | TODO reg |
| — | SERIAL | derived `ceil(ext/parallel)` | never spelled |

Empty/absent = scalar serial fold. Each token maps 1:1 to a `ReduceStage`; `serial` is the derived remainder.

```
b32        -> ReducePlan.of(coop=32)                 one warp, pure shfl
b128       -> ReducePlan.of(coop=128)                hierarchical shfl→smem
r2/b32     -> ReducePlan.of(reg=2, coop=32)          ILP fold + coop
g4k/b32    -> ReducePlan.of(cta=4, coop=32), finalize=kernel   split-reduce, deferred finalize
```

Clean break from the demolished `s<serial>/f<fold>/c<cta>/t<coop>` codec — **no legacy `s/f/c/t` decoding**. The old
`DEPLODOCK_REDUCE="t32"/"c2k"` test pins won't parse under the new letters; those tests are out-of-scope
(symbolic/flash/split) and xfailed, so they get rewritten to `b…`/`g…` when their tier lands. Nothing reads `REDUCE`
until selection is wired (this cut uses conservative constants). A separate inner-contraction K-tiling pin, if ever
needed, is a `SemiringSchedule` (`ATOM@`/warp-tile) knob — never a `REDUCE@<axis>` resurrection.

## Future steps — operand pipelining (`Stage` / `Channel`)

Pipelining is a property of the **streaming (serial reduce) axis's transport**, not of a schedule kind — it is relevant
to uniform `MonoidSchedule` and `SemiringSchedule` alike (both have a reduce loop), and warp specialization is just the
cross-warp *realization* of the same pipeline, not a separate locus. Two levels, different prerequisites:

- **register pipelining** — prefetch the next operand (gmem→reg / smem→reg) ahead of the math. Needs only a serial
  loop with loads → present even in a plain reduction.
- **smem pipelining** — multi-stage smem buffers (cp.async/TMA fill stage `k+1` while compute drains `k`). Needs
  operand **reuse across cooperating threads** → present in a contraction (matmul) and flash, absent in a plain
  reduction (each lane reads its own strided slice — no cross-thread reuse).

So the reuse gate makes it per-schedule: `MapSchedule` — none (single pass); `MonoidSchedule` plain reduce — register
only; `MonoidSchedule` flash + `SemiringSchedule` — both levels.

### Uniform schedules — one `Stage` per reduce loop

A `Stage` descriptor attaches to the schedule's `reduce` (the streaming loop), shared by `MonoidSchedule` and
`SemiringSchedule`. One `Monoid` ⇒ one reduce axis ⇒ one pipeline ⇒ **one `Stage`**. Multiple streamed operands
(flash's K and V) ride inside it (`smem=("K","V")`, sharing the loop / depth / ring); smem + register levels are the
staircase of that one pipeline; a plain reduction degenerates to register-only (`smem=()`, depth from a simple
prefetch). The inner contraction's operand transport (flash's QK over `d`) is NOT a second `Stage` — it rides
`warp_tile` (the mma tile's smem operands), consistent with the flat-schedule rule.

```python
@dataclass(frozen=True)
class Stage:
    depth:     int = 1            # pipeline stages over the serial reduce loop (1 = no prefetch)
    transport: str = "sync"       # sync | cp.async | tma
    smem:      tuple[str, ...] = ()   # operands staged through smem (empty = register-only, e.g. plain reduce)
    ring:      bool = False       # ring buffer vs static double-buffer
```

Attached as `MonoidSchedule(place, block, reduce, stage)` and `SemiringSchedule(place, block, reduce, warp_tile,
stage)`. The reducing schedules then carry three **orthogonal** reduce-axis fields — `reduce` (partition) / `warp_tile`
(operand tile) / `stage` (transport) — to watch so they stay orthogonal and don't become a grab-bag.

### Warp-spec — the pipeline split across roles + a shared `Channel`

Warp specialization splits the uniform single `Stage` into the two halves of a decoupled producer/consumer pipeline,
connected by the smem ring `Channel`:

- **producer** role `Stage` — the gmem→smem fill (transport engine, run-ahead).
- **consumer/mma** role `Stage` — the smem→reg register double-buffer over the math loop.
- **reducer** role `Stage` — ≈ trivial (read the score from smem, combine).
- the `Channel` — owns the **shared** smem ring (buffer + depth), the connective tissue between fill and drain.

So each role's uniform sub-schedule carries its **own** `Stage` (its local pipelining), and the shared smem-ring depth
lives once on the `Channel` (producer and consumer must agree) — distinct from the consumer's register-double-buffer
depth on its own `Stage`.

```python
@dataclass(frozen=True)
class Channel:
    name:  str
    depth: int                    # smem ring slots — the SHARED pipeline depth (fill vs drain)
    # + buffer shape / dtype

@dataclass(frozen=True)
class WarpRole:
    stage_node: object            # which op-tree node this role runs
    warps:      int
    schedule:   object            # MapSchedule | MonoidSchedule | SemiringSchedule — carries this role's own Stage
    reads:      tuple[str, ...]    # channels drained
    writes:     tuple[str, ...]    # channels filled

@dataclass(frozen=True)
class WarpSpec:
    place:    Placement
    channels: tuple[Channel, ...]      # the shared smem rings + their depths
    roles:    tuple[WarpRole, ...]     # role.schedule.stage = that role's local pipelining
```

None of this is built in this cut (scalar/cooperative tier loads gmem-direct, `depth=1`, `transport="sync"`).
`Stage`/`Channel` are reserved slots alongside `warp_tile`/`warp_spec`. Lowering support already exists
(cp.async / TMA / ring buffers / named-barrier `Sync` / `SetMaxNReg` survived the demolition); the missing piece is the
tile-layer schedule that sizes the pipeline and assigns the roles.

## TODO ledger (reserved in the type system, not built here)

| slot | where | note |
|---|---|---|
| `reg` (ILP fold) | `ReduceStage(REG)` / `Fold.REG` | register tree; `combine()` returns `()` placeholder, materializer stub |
| `cta` (split-reduce) | `ReduceStage(GRID)` / `030_split` | partial+finalize via `as_state_merge` + workspace; algebra already supports it |
| symbolic cooperative | `020_schedule` | masked-tail ceil-div lane loop; stays scalar for now |
| flash cooperative-KV / warp-tier | the one `Monoid`'s `ReducePlan` + `warp_tile`/`WarpSpec` | flash is one `Monoid` (no second reduce axis); cooperative-KV scopes the `ReducePlan` to the reducer, inner QK/PV is `warp_tile` |
| warp/mma tier | shared `WarpTile` on `MonoidSchedule`+`SemiringSchedule` | a nested (flash) or top (matmul) contraction tiles onto the one mapping — never a nested/second schedule |
| warp specialization | `WarpSpec` arm of `Monoid`/`SemiringKernel.schedule` | the `<Kind>Schedule \| WarpSpec` either; roles bottom out in uniform schedules (reducer carries the cooperative `ReducePlan`); one shared struct, no `*WarpSpec*` per-kind variants |
| operand pipelining | `Stage` on `reduce` schedules; `Channel` on `WarpSpec` | one `Stage` per reduce loop (uniform); split into producer/consumer `Stage`s + a shared smem-ring `Channel` under warp-spec |
| knob/prior selection | `020_schedule` conservative constants | replace constants with `_reduce_decomp` + the learned/analytic prior |

## Follow-up (not this cut)

Update the `tile-ir-rebuild-design` memory and add a Phase 4 note to `plans/tile-ir-rebuild.md` capturing the
`Kernel`-pair design, the `ReducePlan→ReduceStage` model, the derived-combine rule, and the uniform-vs-`WarpSpec`
schedule either, so the settled design isn't re-litigated.
