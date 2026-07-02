# Flash attention enhancements — close the validation-gate exception

Successor to the executed `tile-ir-rebuild.md` / `knob-search-restore.md` plans. Flash is the one representative
kernel that failed the knob-search validation gate on the RTX 5090 (2026-07-01): the structure is right (the
two-`Contraction` `TWISTED` tree through the one emitter), the schedules are the problem — the search cannot explore
flash forms, and the conservative picks underperform eager.

## Measured state (RTX 5090, SDPA (1, 8, 128, 64), `emmy run --bench`)

| form | pick path | µs | vs eager |
|---|---|---|---|
| f32 scalar CHAIN (FA-2 shared-score) | deterministic (warp-ineligible f32) | 159 | 0.10× (eager 16) |
| f32 coop-KV (`REDUCE=b64` pin) | pin | 1138 | worse |
| fp16 warp (tensor-core) | deterministic conservative | 12.6 | 0.49× (eager 6.5) |

Root causes:

- **f32 chain**: one thread per (batch, head, m) row ⇒ 8·128 = 1024 threads = 4 CTAs on a 170-SM die — catastrophic
  under-occupancy. The per-cell serial form (65 536 threads, redundant score recompute) is plausibly faster here but
  is not reachable: `schedule()`'s TWISTED path early-returns the chain, and an empty `REDUCE` pin doesn't escape
  (only a non-empty partition pin does).
- **fp16 warp**: `_twisted_warp_option` stamps the fixed conservative geometry — one warp per CTA
  (`units=(1, 1)`, block 32, the `2·atom_n` key block) — ~48% occupancy and no K/V staging. There is no move grid
  over the flash `TilePlan`s.

## 1. The flash-form fork (blocked on the AnalyticPrior cold-start)

Offer the TWISTED schedules as prior-ranked siblings — warp (option-0 when eligible), chain, coop `b<n>`, per-cell
serial — instead of deterministic early returns. **Tried and reverted during the knob-search restoration**: with the
cold `AnalyticPrior`, a featureless serial row scores the neutral 1.0 against featured warp/chain rows, flipping the
cold unpinned pick per shape and breaking 29 e2e contract tests that pin warp-when-eligible / chain-when-not.
Sequencing:

1. Rebuild the offline fitter + refit `_W_A` (`tile-ir-cleanup-and-debt.md` §3) so cold ranking over
   structurally-different flash rows is sane — at minimum the warp row must dominate when eligible and the
   chain-vs-serial trade must follow occupancy (the chain's row identity is already stamped, `TILE@<k>=f<d>`).
2. Then land the fork (the reverted shape is preserved in `_schedule.schedule()`'s TWISTED comment); tune explores the
   siblings, `-O3` evidence steers greedy replay — the same mechanism that fixed the matmul replays.
3. Re-run the gate for flash only: tuned pick ≥ eager, greedy replay within ~10% of tuned best.

Also fix en route: **the fold decision is ordering-dependent under tune** — MCTS branches that lift the score
producer before its consumer degrade to `PLACE@fold=cut` accidentally, so tune wastes benches on cut fragments (seen:
199-bench flash tune whose "best" was a cut-side matmul). The offer should be stable per trajectory (the structural
replay machinery in `pipeline._replay_structural_decision` is the seam if fold ever becomes a real two-sided offer).

## 2. Warp-flash move grid (the fp16 0.49× fix)

Enumerate the flash `TilePlan` geometry instead of the fixed single-warp stamp: warps per CTA (query-row blocks per
warp / multiple warps over `m`), the score key-block width (`regs` on the QK node), and K/V operand staging (the
`Stage` seam — the warp matmul's cp.async/TMA transports apply; the streaming loop is the K-slab loop). Legality
stays with the option builder (atom eligibility, static/masked extents as today). Conservative option-0 = the current
single-warp geometry, so a cold greedy compile is unchanged until the fork above lands; pins (`TILE@<axis>` on the
QK/PV nodes) give manual access immediately — the multi-node `FAMILY@<axis>` keying and the per-node sum-pooled
featurizer already exist for exactly this.

## 3. Smaller items

- **Causal tile-skip** on the tensor-core tier (skip fully-masked key blocks; noted as a follow-up in `_flash.py`) —
  a schedule-derived loop bound, not a knob.
- **Cross-CTA flash split (`g<w>k` split-KV)** composes with the warp tier? Today the twisted split is scalar-only in
  practice; verify and extend once the warp move grid lands.
- **`PLACE@fold` auto default**: already `fuse` (today's behavior — the gate's "flip to fuse" clause is moot); the
  real follow-up is knobifying `fold`/`cone` through the `auto` seam once a shape where cut beats fuse shows up in
  tune data (decided on evidence, per the restoration's design note in `search/space.py`).

## Verification

The knob-search validation gate, flash column: `emmy tune --code "F.scaled_dot_product_attention(...)" --bench` with
a non-trivial explored space over FLASH schedules (not cut fragments); accuracy green; tuned pick ≥ eager for f32 AND
fp16; greedy prior-only replay within ~10% of tuned best; the 29 flash e2e contract tests stay green (or are
consciously re-pinned to the new cold contract in the same change that makes cold ranking trustworthy).
