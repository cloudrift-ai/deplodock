# Golden seed findings — RTX PRO 6000 Blackwell Max-Q (sm_120), 2026-06-13 (first-ever PRO 6000 golden file)

- New file: `deplodock/compiler/pipeline/search/goldens/rtxpro6000_sm120.yaml` (29 matmul shapes), seeded on
  `riftuser@38.108.83.78:60002` (RTX PRO 6000 Blackwell Max-Q Workstation Edition, sm_120, 96 GB, driver 580.65.06,
  CUDA 12.9, torch 2.12.0+cu130).
- Sweep: `deplodock tune --dataset golden --clean -q` (~25 min) → seed-mode harvest (`run --bench -c snippet`, greedy
  pick recorded directly). One shape re-tuned at higher patience (Finding 1). Prior + DB pulled to
  `/tmp/golden_artifacts/pro6000/` for offline analysis.
- Branch: `feature/golden-multi-gpu`.
- Headline: the PRO 6000 is **sm_120, same capability as the RTX 5090**, so the existing sm_120-fit prior transfers
  cleanly — unlike the 4090, the fresh seed is healthy out of the box. **20/29 shapes ≤ 1.1× eager** (median d/e
  **1.00**, max 1.27 after the one re-tune). fp16 correctly uses the warp/MMA tensor-core tier (no Hopper-gate problem,
  cap ≥ 9,0). Recorded the greedy picks directly as the seed.

## Per-shape outcome (greedy pick benched on the PRO 6000; d/e = deplodock / eager)

| shape                            | deplo µs | eager µs |  d/e | note                                  |
|----------------------------------|---------:|---------:|-----:|---------------------------------------|
| square.512                       |     11.0 |     12.6 | 0.87 |                                       |
| square.1024                      |     49.2 |     55.6 | 0.89 |                                       |
| square.2048                      |    357.6 |    324.8 | 1.10 |                                       |
| square.4096                      |   3021.4 |   2426.0 | 1.25 | big fp32 square (F2)                   |
| square.512.fp16                  |      3.8 |      6.1 | 0.61 | warp tier                             |
| square.1024.fp16                 |     13.9 |     12.5 | 1.11 | warp tier (F2)                        |
| square.2048.fp16                 |     78.6 |     69.9 | 1.12 | warp tier                             |
| square.4096.fp16                 |    524.1 |    467.2 | 1.12 | warp tier                             |
| qwen3_06b.q_proj.s32             |      7.8 |      8.7 | 0.90 |                                       |
| qwen3_06b.kv_proj.s32            |      6.2 |      8.5 | 0.74 |                                       |
| qwen3_06b.o_proj.s32             |     10.0 |     10.3 | 0.97 |                                       |
| qwen3_06b.gate_up_proj.s32       |     11.6 |     12.6 | 0.92 |                                       |
| qwen3_06b.down_proj.s32          |     13.8 |     14.4 | 0.96 |                                       |
| qwen3_06b.q_proj.s128            |     18.7 |     19.7 | 0.95 |                                       |
| qwen3_06b.kv_proj.s128           |     12.1 |     14.5 | 0.84 |                                       |
| qwen3_06b.o_proj.s128            |     20.9 |     22.6 | 0.92 |                                       |
| qwen3_06b.gate_up_proj.s128      |     30.9 |     26.4 | 1.17 | (F2)                                  |
| qwen3_06b.down_proj.s128         |     33.3 |     31.1 | 1.07 |                                       |
| qwen3_06b.q_proj.s512            |     55.8 |     55.4 | 1.01 |                                       |
| qwen3_06b.kv_proj.s512           |     30.3 |     39.5 | 0.77 |                                       |
| qwen3_06b.o_proj.s512            |     66.3 |     68.6 | 0.97 |                                       |
| qwen3_06b.gate_up_proj.s512      |     85.1 |     81.3 | 1.05 |                                       |
| qwen3_06b.down_proj.s512         |     97.8 |     92.0 | 1.06 |                                       |
| square.512.dynM                  |     13.6 |     12.6 | 1.08 |                                       |
| qwen3_06b.q_proj.s512.dynM       |     55.2 |     55.3 | 1.00 | **re-tuned** 89.3→55.2 (F1)           |
| qwen3_06b.kv_proj.s512.dynM      |     47.8 |     39.5 | 1.21 | dynM masked tile (F2)                 |
| qwen3_06b.o_proj.s512.dynM       |     86.9 |     68.4 | 1.27 | dynM masked tile (F2)                 |
| qwen3_06b.gate_up_proj.s512.dynM |     89.7 |     80.9 | 1.11 | dynM masked tile                      |
| qwen3_06b.down_proj.s512.dynM    |     87.5 |     92.1 | 0.95 |                                       |

## Finding 1 — q_proj.s512.dynM: fresh seed was degenerate (1.63×), one higher-patience re-tune fixed it (→1.00×)

The seed deployed `q_proj.s512.dynM` at 89.3 µs (1.63× eager). Unlike the 4090's degenerate picks, this one **did**
respond to patience: `tune --golden qwen3_06b.q_proj.s512.dynM --patience 100` (accumulate, no `--clean`) found
`{BM:8, BN:32, BK:32, FM:6, FN:4, FK:1, SPLITK:1, BR:1, OVERHANG:['a0'], STAGE:'11', RING:2}` at **55.2 µs / 1.00×**.
Recorded. So on sm_120 the prior reaches good regions with more search budget — the cold prior just stopped early on
this masked-tile shape. (Contrast the 4090, where patience did *not* help — Finding 2 there.)

## Finding 2 — a handful of shapes sit 1.1–1.27×; expected first-seed slack, not a defect

`square.4096` (1.25×), the fp16 squares (1.11–1.12×), `gate_up_proj.s128` (1.17×), and two dynM masked tiles
(kv/o_proj.s512.dynM at 1.21/1.27×) are above parity but not degenerate. These are the same shape families that sit
slightly off on the 5090 too; on a first clean seed they reflect search slack, not a tier/codegen lockout (fp16 is on
the correct warp tier; the big square is a known large-`FM` tuning target). They are honest deployable seeds and are
the natural targets for a second accumulating (non-`--clean`) sweep or the golden-seeding fix.

## Finding 3 — same-cap collision with the RTX 5090 (`compute_cap (12, 0)`) (P1 for the goldens layer)

The PRO 6000 and the RTX 5090 report the **identical** `compute_cap (12, 0)`. The goldens layer assumed `compute_cap`
uniquely identifies a card (`goldens_by_name`'s contract, the `_load_goldens` doc) — now false. With both files present
`GOLDEN_CONFIGS` carries two entries per name at the same cap, distinguishable only by `gpu_name`, and the `eval` /
`tune --dataset golden` paths plus every `ShapeKey`-keyed join are GPU-blind: they intermix the two cards. Nothing
crashes (the golden-config test passes; `eval golden` runs), and per-config evaluation still uses each config's own
`Context.from_target(compute_cap)` — but cross-shape grouping and the golden↔DB join silently merge 5090 and PRO 6000
data on the dev box. Docstrings in `search/golden.py` were corrected to state the `(gpu_name, compute_cap)` identity.

**Recommendation:** filter the golden consumers to the live `(gpu_name, compute_cap)` — add the pair to
`Dataset.from_golden` and the `eval` iterations, default to the live card, with an `--all-gpus` escape hatch for
cross-card comparison. Touches `Dataset.from_golden` + the `eval` command paths + tests; deferred from this task as a
focused follow-up rather than risk the eval tooling mid-sweep.

## Workflow notes

- **sm_120 seeds cleanly; sm_89 does not** (cf. the 4090 report). The deciding factor is whether the live capability
  matches the prior's training data (sm_120): the PRO 6000 needed *one* re-tune, the 4090 needed config transfer for 16
  shapes plus has no fp16 tensor-core path. A per-capability cold analytic prior (Finding 2 of the 4090 report) is the
  general fix.
- **The seed harvest is a single scripted pass** (`/tmp/harvest_goldens.py --mode seed`): greedy knobs read from the
  compiled graph, -O3 latencies from `60_bench_compare.json`; `/tmp/emit_golden_yaml.py` writes the hand-style flow
  YAML. Recording a brand-new GPU was mechanical once the prior transferred.
- **Prior + DB** pulled post-re-tune (949 MB DB / 77 MB prior) to `/tmp/golden_artifacts/pro6000/` for offline
  analysis; they are the clean sm_120 seed artifacts plus the one accumulated q_proj.s512.dynM re-tune.
