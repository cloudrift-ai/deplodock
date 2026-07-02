# Warp specialization (`WarpSpec`) — the last unbuilt tier

Successor to the executed `tile-ir-rebuild.md`'s Phase 4. Heterogeneous warps: the CTA partitions into producer /
compute / reducer roles wired by shared smem rings, decoupling the operand fill from the mma consumers. The rebuild's
mandate applies: this lands as **config on the node that the one emitter already structurally supports** — a
`WarpSpec` whose roles each bottom out in a uniform `Map` / `Reduction` / `Contraction` sub-schedule — never a fourth
node kind or a bespoke emitter.

## Current state

- `TileOp.workers: WarpSpec | None` exists; the `WSPEC` codec + role registry (`RoleKind` / `ROLE_REGISTRY` in
  `ir/schedule.py`) parse and spell; `_schedule._wspec_workers` validates a pin (parse + `is_legal` — a producer needs
  a resolved `stage` to drive).
- **A structurally legal pin is currently REFUSED** (warning + no stamp + uniform SIMT,
  `test_warp_matmul_refuses_wspec_while_inert`): the materializer consumes nothing, and an accepted pin would record
  warp splits in the perf DB that no kernel ran. Un-refusing it is this plan's flip.
- The kernel IR already carries the transport vocabulary (cp.async ring, TMA + mbarrier phases) and the sm_90 register
  reallocation stmt (`setreg` — only a warp-specialized materialization emits it, TMA-gated to sm_90+).
- The pre-rebuild perf bar (recorded in `search/golden.py`): a 4-warp warp-specialized 64×64 CTA measured at/above
  cuBLAS on the sm_120 squares (2048²: 1.06×, 4096²: 1.03×).

## Design (carried from the rebuild plan)

```python
@dataclass(frozen=True)
class Channel:                       # a shared smem ring — the producer/consumer seam
    name: str
    depth: int
    transport: str = "cp.async"      # cp.async | tma

@dataclass(frozen=True)
class WarpRole:                      # one warp group's job; its sub-schedule node NAMES the role
    stage_node: object               # the Map / Reduction / Contraction this role runs
    warps: int
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    stage: Stage | None = None       # this role's LOCAL smem→register double-buffer

@dataclass(frozen=True)
class WarpSpec:
    place: Placement                 # the CTA-tile grid
    channels: tuple[Channel, ...] = ()
    roles: tuple[WarpRole, ...] = () # Σ role.warps = the CTA warp count
```

- **Delegation, not a union arm:** `WarpSpec` rides the orthogonal `TileOp.workers` field over the FIXED pipeline
  (tile / stage / reduce pins decide what runs; `WSPEC` only splits who runs it). Roles do not nest; `WarpSpec` lives
  only at the top CTA level.
- The uniform `Stage` splits: the gmem→smem *fill* becomes the shared `Channel`; each consumer's *local*
  smem→register double-buffer stays on `role.stage`.
- **Spelling** reuses the role vocabulary + role-namespacing:
  `CHANNEL=K:d3/cp;V:d3/cp mma:WARP=…/k2 reducer:REDUCE=b2 producer:STAGE=d3/cp` — register the role-prefixed grammar
  so `knob_features` / `apply_off_defaults` / `tuning_knob_items` handle the keys.

## Build steps

1. Materialize the producer/consumer split in the one emitter: the staged K-loop's fill half runs on the producer
   warps (driving the `Channel` ring), the drain + mma on the compute warps, `setreg` register reallocation between
   them; the fold's cross-role handoff rides the same placement-keyed move the fold consolidation defines
   (`tile-ir-cleanup-and-debt.md` §2 — coordinate, don't duplicate).
2. Flip `_schedule._wspec_workers` back to stamping (`return ws, pinned`) — the refusal comment marks the exact spot.
   Restore the honest-stamping rule: stamp only what materializes.
3. Featurization + legality (deferred from the knob-search restoration's Phase 2): a producer-warp-count / `q`-window
   featurizer, the `block_threads + 32·aux_warps ≤ 1024` legality gate, transport-aware `is_legal` (a producer over
   `sync` has nothing to drive).
4. Enumeration (after the pin path is proven): fold `WSPEC` moves into the schedule fork as a fourth level, gated on a
   warp `TILE` + a resolved async `Stage`; conservative option-0 = uniform SIMT (`""`).

## Purge (ships with the build)

- **`Channel` must not reimplement `Stage`'s transport** — one ring implementation; `Channel` holds only the shared
  variant.
- Delete `TileOp.workers`' "not yet built" language, the WSPEC refusal + its test (replaced by stamping tests), and
  the "materialization reserved (TODO)" help text in `search/space.py`.

## Verification

Bit-identity vs the uniform-SIMT staged baseline is NOT expected (warp spec changes scheduling), so: accuracy against
numpy/torch on the staged warp matmul matrix; a perf gate vs the recorded golden bar above (`make bench-kernels` on
the sm_120 squares); `emmy eval knobs` shows the role-prefixed keys; `make test` + `make lint`.
