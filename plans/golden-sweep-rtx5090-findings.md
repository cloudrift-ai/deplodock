# Golden sweep findings — RTX 5090 (sm_120), 2026-06-13 (fourth sweep; first multi-GPU run)

- Sweep: `deplodock tune --dataset golden --clean -q` — 29 matmul targets (23 static + 6 dynamic `.dynM`), ~28 min wall.
- A/B: a scripted `deplodock run --bench --golden NAME` pass per shape (`/tmp/harvest_goldens.py --mode ab`) — greedy
  pick benched live beside each recorded golden, all at -O3. Greedy knobs read programmatically from the compiled
  graph (`tuning_knob_items` over the matmul `CudaOp`), latencies from `60_bench_compare.json` (Deplodock) + the live
  golden table rows. The one win was re-benched 3× and is stable to the digit.
- Branch under test: `main` (the analytic-tilt `FallbackPrior.score` fp16-`BK=2` fix from the third sweep is present).
  Edits on `feature/golden-multi-gpu`.
- Tally: **1 replaced / 0 added / 6 exact reproductions / 22 left** (of the 22 left: 17 genuinely worse, 3 fp16 squares
  at parity-with-drift, 2 s128 RING-only adjacent configs).

## Headline: a clean re-tune regresses 17/29 shapes vs the recorded goldens — Finding 2 from the third sweep, unchanged

The recorded 5090 goldens were built up across three accumulating sweeps. `tune --dataset golden --clean` wipes the
DB + prior and re-searches each shape from a cold (5090-fit analytic) start under default patience; the fresh search
takes different trajectories and does not re-find most of the recorded configs. `evidence_pick` then faithfully
deploys the best *measured* config, which sits 6–44% above goldens whose region the clean search never sampled. This
is exactly the third sweep's Finding 2 ("clean-DB search variance unsamples the goldens"), and its recommendation
(seed the inner search with the recorded golden knobs) is **still the top action item** — it has not been implemented.

So the correct outcome for this sweep is *leave the goldens*: every "worse" shape is the search regressing against a
config that is already recorded and known-good, not a real latency regression. The recorded 5090 set stands.

## Per-shape outcome (live -O3 A/B; greedy µs vs best recorded-golden µs re-benched the same run)

| shape                            | greedy µs | best golden µs |   ratio | category                                  |
|----------------------------------|----------:|---------------:|--------:|-------------------------------------------|
| square.512                       |       9.0 |            9.1 |    0.99 | exact reproduction                        |
| square.1024                      |      49.0 |           43.6 |    1.12 | worse (clean-tune variance)               |
| square.2048                      |     346.7 |          265.9 |    1.30 | worse                                     |
| square.4096                      |    2594.8 |         2202.7 |    1.18 | worse                                     |
| square.512.fp16                  |       4.0 |            3.5 |    1.16 | worse                                     |
| square.1024.fp16                 |      14.8 |           14.9 |    0.99 | parity (drift; left — see Finding 3)      |
| square.2048.fp16                 |      90.8 |           91.5 |    0.99 | parity (drift; left)                      |
| square.4096.fp16                 |     636.2 |          649.9 |    0.98 | parity (drift; left)                      |
| qwen3_06b.q_proj.s32             |       7.2 |            6.4 |    1.12 | worse                                     |
| qwen3_06b.kv_proj.s32            |       4.9 |            4.9 |    1.00 | exact reproduction                        |
| qwen3_06b.o_proj.s32             |      11.2 |            8.0 |    1.40 | worse                                     |
| qwen3_06b.gate_up_proj.s32       |      10.2 |            9.5 |    1.08 | worse                                     |
| qwen3_06b.down_proj.s32          |      17.6 |           12.2 |    1.44 | worse                                     |
| qwen3_06b.q_proj.s128            |      16.2 |           16.3 |    0.99 | parity (RING-adjacent; left)              |
| qwen3_06b.kv_proj.s128           |       9.7 |            9.8 |    0.99 | parity (RING/SPLITK-adjacent; left)       |
| qwen3_06b.o_proj.s128            |      18.3 |           17.4 |    1.05 | worse-marginal                            |
| qwen3_06b.gate_up_proj.s128      |      21.0 |           21.0 |    1.00 | exact reproduction                        |
| qwen3_06b.down_proj.s128         |      24.5 |           24.2 |    1.01 | exact reproduction                        |
| qwen3_06b.q_proj.s512            |      46.6 |           44.0 |    1.06 | worse                                     |
| qwen3_06b.kv_proj.s512           |      29.3 |           24.8 |    1.18 | worse                                     |
| qwen3_06b.o_proj.s512            |      56.5 |           46.2 |    1.22 | worse                                     |
| qwen3_06b.gate_up_proj.s512      |      64.1 |           52.8 |    1.21 | worse                                     |
| qwen3_06b.down_proj.s512         |      84.6 |           69.3 |    1.22 | worse                                     |
| square.512.dynM                  |      11.0 |           13.6 |    0.81 | **replaced** (BN16 BM16 SPLITK1 RING3)    |
| qwen3_06b.q_proj.s512.dynM       |      71.5 |           49.6 |    1.44 | worse                                     |
| qwen3_06b.kv_proj.s512.dynM      |      30.6 |           26.8 |    1.14 | worse                                     |
| qwen3_06b.o_proj.s512.dynM       |      47.5 |           47.7 |    1.00 | exact reproduction                        |
| qwen3_06b.gate_up_proj.s512.dynM |      74.0 |           67.0 |    1.10 | worse                                     |
| qwen3_06b.down_proj.s512.dynM    |      68.1 |           67.6 |    1.01 | exact reproduction                        |

## Finding 1 — square.512.dynM: a genuinely faster masked-tile config (replaced, −14%)

The only real improvement. The clean search found `{BN:16, BM:16, FM:4, FN:2, FK:1, BK:64, SPLITK:1, BR:1,
OVERHANG:['a0'], STAGE:'11', RING:3}` at **11.0 µs**, beating the recorded `SPLITK:2 RING:4` config (12.8 µs) by 14%.
The win reproduces to the digit across three back-to-back A/B runs (greedy 11.0 / live-golden 13.6 / ratio 0.81 every
time), well above the dynM noise band. The difference is `SPLITK:1, RING:3` vs `SPLITK:2, RING:4` — a single-split,
3-stage ring beats the 2-split 4-stage one for this masked-tile shape. Recorded as the sole `square.512.dynM` entry
(old one deleted: it is now strictly slower).

## Finding 2 — the clean sweep un-finds recorded wins; seed the inner search with the golden knobs (P0, unchanged from third sweep)

17 shapes deploy 6–44% above goldens the search did not re-measure this run. The mechanism is unchanged from the third
sweep's Finding 2: `--clean` discards the accumulated reservoir that held those configs, and a single patience-bounded
re-search lands elsewhere; `evidence_pick` deploys its measured best, which is worse than the un-sampled golden. The
worst cases (down_proj.s32 1.44×, q_proj.s512.dynM 1.44×, o_proj.s32 1.40×, square.2048 1.30×) are all "golden absent
from the measured set," not a prior that ranks them deep.

**Recommendation (highest priority, carried over and now three sweeps old):** force-bench each recorded golden config
as the first variant(s) of its op's inner search in `tune --dataset golden` / `tune --golden NAME`. The tuner knows the
shape and the golden knobs; one bench per golden entry (~30 s across the sweep) guarantees `evidence_pick` can never
deploy worse than the recorded golden after a clean tune, and turns "worse" into a signal that means *only* real
regressions. This would also have prevented the 17 regressions above outright. Until it lands, **run routine sweeps
without `--clean`** (accumulate) and reserve `--clean` for prior-hygiene checks — the third sweep's closing note,
re-confirmed.

## Finding 3 — fp16 squares reproduce at parity; the stored deplodock_us are stale-by-drift, not improvable

square.1024/2048/4096.fp16 all land within 1–2% of the recorded golden's live re-bench (0.98–0.99), and the greedy
configs are the `BK=2` warp-tier region the third sweep's analytic-tilt fix made reachable — confirming that fix holds
on `main`. The greedy form drops the planner-derived `WARPSPEC` pin the older entries carry (per the third sweep's
Finding 4), but at the *search-knob* level it is the same region, so there is no new config to add. Note: the live
re-bench (14.9 / 91.5 / 650 µs) runs faster than the entries' stored `deplodock_us` (18.5 / 106.7 / 746 µs) — pure
measurement/codegen drift since recording, the same config. Left untouched: the deltas are within noise and conflate
drift; refreshing the stored µs would mean picking which of the two recorded entries to rewrite with no config change.

## Workflow notes

- **The harvest is now scripted end-to-end** (`/tmp/harvest_goldens.py`): per shape it reads the greedy pick's knobs
  programmatically (no fragile colored-table parsing) and the -O3 latencies from `run --bench`'s `60_bench_compare.json`
  + the live golden rows, emitting JSONL. Categorization is one pass over the JSONL against `goldens_by_name`. This
  replaced the by-hand table reading of the previous three sweeps and made the 29-shape A/B + noise re-runs a few
  scripted minutes. Worth folding into the CLI as a `tune --dataset golden --record` mode.
- **`tuning_knob_items` drops BOOL knobs**, so the harvested knob set already excludes `WARPSPEC`/`GROUP_M`-style
  planner-derived flags — exactly the third sweep's Finding 4(a) record-set, for free. The `--ab`/`--golden` *table*
  still hides those diffs (Finding 4(a) proper — surface a column for differing planner-derived knobs — remains open).
- **Multi-GPU goldens now coexist** (`rtx5090_sm120.yaml`, `rtx4090_sm89.yaml`, `rtxpro6000_sm120.yaml`). The 5090 and
  PRO 6000 share `compute_cap (12, 0)`, which the goldens layer assumed unique: `eval` / `tune --dataset golden`
  iterate the flat `GOLDEN_CONFIGS` with no live-GPU filter, and `ShapeKey` joins (shape-keyed, GPU-blind) merge
  same-shape entries across cards. `eval golden --kernel square.512` on this box now silently compares against
  whichever card's golden the grouping surfaces. Docstrings in `search/golden.py` were corrected to flag this; the
  real fix — filter the golden consumers to the live `(gpu_name, compute_cap)`, with an `--all-gpus` escape hatch — is
  a focused follow-up (touches `Dataset.from_golden` + the `eval` paths + tests on each GPU). See the 4090 / PRO 6000
  reports for the same note.
- **Fixed since the third report and held:** the fp16-`BK=2` analytic-tilt fix (Finding 1 there) — fp16 squares
  reproduce at parity here. **Not fixed (carried):** golden-seeding the inner search (Finding 2, now P0 across four
  sweeps); the planner-derived knob column in the A/B table (Finding 4a).
