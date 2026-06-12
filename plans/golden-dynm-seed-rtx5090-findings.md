# Dynamic (.dynM) golden seeding — RTX 5090, 2026-06-12

First batch of symbolic-M (masked-tile) goldens, seeded per `plans/dynamic-shape-goldens.md` §4 / order-of-work 4.
GPU: NVIDIA GeForce RTX 5090 (sm_120). Commands per shape: `deplodock tune -c "<matmul snippet>" --dynamic
seq_len@x0:0` (accumulating into the existing tune DB + global prior — no `--clean`), then two passes of
`deplodock run --bench -c "<snippet>" --dynamic seq_len@x0:0` (deployable -O3 A/B + noise-floor confirmation).
Approximate wall time: ~10 min for the 7 tunes, ~2×5 min for the two bench passes.

**Tally: 6 seeded (new), 1 dropped (square.1024.dynM — duplicate by hint, finding 1), 0 replaced.**

## Per-shape outcome (recorded = the greedy -O3 pick; both bench passes agreed within the noise band)

| shape                            | dep µs | eager µs | ratio | static twin dep µs | masked cost |
|----------------------------------|-------:|---------:|------:|-------------------:|------------:|
| square.512.dynM                  |   18.0 |       14 | 0.78x |                9.0 |       +100% |
| qwen3_06b.q_proj.s512.dynM       |   52.5 |       47 | 0.90x |               50.8 |        +3%  |
| qwen3_06b.kv_proj.s512.dynM      |   35.4 |       39 | 1.10x |               24.6 |       +44%  |
| qwen3_06b.o_proj.s512.dynM       |   62.6 |       57 | 0.91x |               50.5 |       +24%  |
| qwen3_06b.gate_up_proj.s512.dynM |   77.2 |       79 | 1.03x |               67.6 |       +14%  |
| qwen3_06b.down_proj.s512.dynM    |   81.2 |       76 | 0.94x |               80.8 |        +0%  |

"masked cost" compares against the static twin's recorded `deplodock_us` (a different sweep's numbers — directional,
not a same-run A/B). Eager references are this run's hint-shaped torch rows; they wander ~10% vs the static sweep's
recorded `cublas_us` (e.g. q_proj 47 vs 53.5), which is the known small-shape noise band.

## Finding 1 — the global `DEFAULT_SEQ_HINT` collapses every M≠512 dynamic shape (schema guard added)

`square.1024.dynM` (traced at M=1024) benched the **exact same kernel** as `qwen3_06b.kv_proj.s512.dynM` — same
content hash (`k_matmul_9afebf`), same 35 µs, table note `benched at seq_len=512`. The symbolic axis takes the
global `DEFAULT_SEQ_HINT=512` (`compiler/dim.py:49`) regardless of the traced extent, so a "1024-hint" dynamic
golden is a mislabeled duplicate of the (N=1024, K=1024, hint-512) shape. The YAML `M`-doubles-as-hint semantics
only hold at M=512 today.

- **Action taken**: dropped the duplicate; `MatmulGoldenConfig.__post_init__` now rejects a dynamic golden whose
  `M != DEFAULT_SEQ_HINT`, with a message naming this trap.
- **Recommendation**: plumb per-Dim hints (trace `--dynamic` spec → `Dim.hint` → tile sizing) so dynamic goldens at
  other hints (e.g. a 2048-hint long-context tile) become recordable; then relax the guard to `M == hint`.

## Finding 2 — masked-tile cost concentrates on small/skinny shapes (square.512 pays 2x)

The masked tier's overhead vs the static twins spans +0% (down_proj, K=3072 — reduce-bound, the guard amortizes)
to +100% (square.512 — the smallest tile, where the boundary guard + the symbolic-row restrictions on staged
prologues bite hardest). kv_proj (+44%) and o_proj (+24%) sit in between. Three of six shapes still land ≥0.95
vs the hint-shaped eager reference (kv 1.10x, gate_up 1.03x, down 0.94x), so masked tiles are already deployable
for the mid-size projections; the small-square gap is the optimization target.

- **Recommendation**: a `tune-model`-style drill on `square.512.dynM` (NCU compare vs the static twin's kernel) to
  attribute the 2x — guard overhead vs the locked-out staged pipeline vs occupancy. The structural-split path
  (`005_split_demoted` symbolic-row offers) is the designed escape; check whether the outer search ever offered it
  here.

## Finding 3 — the eval views compared the wrong artifact for dynamic goldens (fixed this sweep)

`eval golden` / `eval prior`'s greedy pick (`_emit_prior_golden_check.picked`) traced the snippet **statically**, so
a `.dynM` row's `found/golden` diff compared the static twin's pick against the dynamic golden's knobs — every row
looked like a multi-knob miss while the live `run --bench --golden` reproduced the goldens exactly. Two fixes:

- `picked()` now applies the golden's own `dynamic_specs()` to the trace (the same auto-spec rule as `tune`/`run`).
- `OVERHANG` recorded as YAML list vs pipeline tuple false-flagged every OVERHANG-carrying golden (static ones
  included) as non-reproduced — knob comparisons now normalize sequence representations (`_knob_eq`).

After both: all six `.dynM` rows show **11/11 knobs reproduced, TOTAL 6/6 exact**.

## Finding 4 — prior expectation: learned rank 0/1008 on all six; cold analytic median rank 55

Post-seed, the learned prior ranks every `.dynM` golden **0/1008** over the full enumeration (`eval prior --dataset
golden --kernel dynM`) — the seed tunes were absorbed immediately, and greedy deploys the goldens (verified live:
the kv_proj A/B reproduced the golden knob-for-knob at 34.7 vs 35.6 µs). The cold `AnalyticPrior` (`eval analytic
--kernel dynM`) lands median rank 55 (top10 0/6) with systematic misses — it under-sizes tiles on masked shapes
(`BM 8/16`, `BN 16/32|64`, `SPLITK 1/2`) because its weights were fit on static-tier goldens only.

- **Recommendation**: refit `scripts/golden_knob_heuristics.py` now that six dynamic goldens are recorded (the
  deferred §3 item); the stamped `S_ext_n_symbolic_axis` flag is already in the featurization, so the fit can
  price the masked tier separately.

## Finding 5 — `golden_deploy_perf`'s reservoir pick disagrees with the live deploy on two shapes (observation)

`eval prior`'s `vs gold` column shows q_proj 1.91x and kv_proj 1.36x — the prior's `mean_score` argmin over the
shape's **H_opt=3 reservoir rows** picks a measured config ~1.4–1.9x slower than the golden, while the live greedy
deploy (and the rank-0 enumeration scoring) pick the golden itself. The -O3 reservoir sample at a freshly-seeded
shape is thin (only the `DEPLODOCK_O3_TOL` band of one tune gets re-benched), so the argmin over it is noisy.
Re-check after the next full sweep; if it persists, the deploy_perf pick logic (argmin over stamped row features)
deserves its own look.

## Workflow notes

- **Seeding needed zero new CLI**: `tune -c ... --dynamic` + `run --bench -c ... --dynamic` for the unrecorded
  shapes, then the recorded entries flow through `tune --dataset golden` / `run --golden` automatically (the spec
  is part of the config). The kv_proj round-trip A/B confirmed the recorded-entry path end-to-end.
- The duplicate-hint trap (finding 1) cost one wasted tune+bench cycle; the schema guard turns it into an
  immediate load-time error.
- The comparison table prints eager µs integer-rounded, so `cublas_us` for new shapes records at integer precision
  (static entries carry one decimal). One decimal in the table would remove the asymmetry.
- The two bench passes agreed within ~1–8% on every shape (well inside the documented 10–13% band); for seeding,
  one confirmation pass was enough.
