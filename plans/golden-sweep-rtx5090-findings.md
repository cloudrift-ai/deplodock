# Golden sweep findings — RTX 5090 (sm_120), 2026-06-12 (second sweep; first with .dynM shapes in the dataset)

- Sweep: `deplodock tune --dataset golden --clean` — 29 matmul targets (23 static + 6 dynamic `.dynM`), 52 min wall.
  The 6 dynM additions cost ~12 min over the previous 23-shape sweep on this box (40 min).
- A/B: `deplodock run --bench --golden NAME` per shape (29 names), then a second confirmation pass over the 16 shapes
  whose category depended on the margin. Absolute µs swung ~10% between the two passes (GPU thermal state — pass 2 ran
  hot), but the within-run greedy/golden **ratios** were stable to ≤3%, so categorization is ratio-based and the
  recorded µs are the cool-state pass-1 numbers.
- Branch under test: `feature/dynamic-golden` (post dynM-seeding, same masked-MMA codegen as the previous sweep).
- Tally: **4 replaced / 9 added / 2 exact reproductions / 14 worse / 0 re-records** (the 5 reduce/pointwise goldens
  remain unsweepable, as last time).

## Per-shape outcome (all numbers -O3 live A/B pass 1; pass-2 ratio in parens where it decided the category)

| shape                            | greedy µs | best golden µs | greedy/golden | category |
|----------------------------------|----------:|---------------:|--------------:|----------|
| square.512                       |       9.8 |            9.0 |  1.09 (1.06)  | worse (finding 3) |
| square.1024                      |      44.1 |           45.8 |  0.96 (0.98)  | **added** (parity, diff knobs) |
| square.2048                      |     296.8 |          272.3 |          1.09 | worse (finding 2) |
| square.4096                      |    2475.9 |         2218.1 |          1.12 | worse (finding 2) |
| square.512.fp16                  |       5.1 |            3.5 |          1.46 | worse (finding 1) |
| square.1024.fp16                 |      16.8 |           14.9 |          1.13 | worse (finding 1) |
| square.2048.fp16                 |     109.6 |           92.0 |          1.19 | worse (finding 1) |
| square.4096.fp16                 |     838.8 |          644.4 |          1.30 | worse (finding 1) |
| qwen3_06b.q_proj.s32             |       7.2 |            7.5 |  0.96 (0.97)  | **added** (parity, diff knobs) |
| qwen3_06b.kv_proj.s32            |       4.9 |            5.3 |  0.92 (0.95)  | **replaced** |
| qwen3_06b.o_proj.s32             |       8.0 |            8.7 |  0.92 (0.93)  | **replaced** (duplicate pair collapsed) |
| qwen3_06b.gate_up_proj.s32       |       9.6 |            9.6 |          1.00 | same knobs — exact reproduction |
| qwen3_06b.down_proj.s32          |      13.4 |           12.2 |          1.10 | worse (finding 5) |
| qwen3_06b.q_proj.s128            |      20.3 |           16.3 |          1.25 | worse (finding 4) |
| qwen3_06b.kv_proj.s128           |      10.3 |            9.8 |  1.05 (1.03)  | worse-marginal (within noise; left) |
| qwen3_06b.o_proj.s128            |      17.4 |           17.7 |  0.98 (0.98)  | **added** (parity, diff knobs) |
| qwen3_06b.gate_up_proj.s128      |      29.8 |           24.5 |          1.22 | worse (finding 4) |
| qwen3_06b.down_proj.s128         |      25.1 |           24.7 |  1.02 (1.00)  | **added** (parity, diff knobs) |
| qwen3_06b.q_proj.s512            |      44.2 |           48.1 |  0.92 (0.89)  | **replaced** |
| qwen3_06b.kv_proj.s512           |      26.6 |           24.9 |          1.07 | worse (finding 2) |
| qwen3_06b.o_proj.s512            |      50.6 |           46.5 |          1.09 | worse (finding 2) |
| qwen3_06b.gate_up_proj.s512      |      52.4 |           82.9 |  0.63 (0.59)  | **replaced** (−37%, the sweep's big win) |
| qwen3_06b.down_proj.s512         |      72.4 |           72.3 |  1.00 (0.99)  | **added** (parity, diff knobs) |
| square.512.dynM                  |      15.9 |           16.0 |          0.99 | same knobs — exact reproduction |
| qwen3_06b.q_proj.s512.dynM       |      50.4 |           50.8 |  0.99 (0.97)  | **added** (parity, RING 3 vs 2) |
| qwen3_06b.kv_proj.s512.dynM      |      30.7 |           31.2 |  0.98 (0.97)  | **added** (parity, RING 3 vs 2) |
| qwen3_06b.o_proj.s512.dynM       |      55.0 |           55.6 |  0.99 (0.98)  | **added** (parity, RING 3 vs 2) |
| qwen3_06b.gate_up_proj.s512.dynM |      70.0 |           70.0 |  1.00 (1.00)  | **added** (parity, RING 3 vs 2) |
| qwen3_06b.down_proj.s512.dynM    |      80.5 |           72.7 |          1.11 | worse (finding 6, first dynM shortfall) |

The wins continue the previous sweep's `BM16 + RING4` and `RING2`-at-s32 directions: kv/o_proj.s32 win on a single RING
4→2 flip, q_proj.s512 moves to `BM16 BK32 RING4`, and gate_up_proj.s512 jumps −37% onto the `FM10 FN4 SPLITK1
OVERHANG` family (its old `SPLITK2 FN2` golden also live-benched 23% over its recorded number, so the old entry was both
beaten and stale). All four wins reproduced on the confirmation pass with margins well above the ≤3% ratio spread. The
five dynM seeds from this morning survive a clean-DB retune: four reproduce at parity modulo a RING 3-vs-2 flip (added
as twins), square.512.dynM reproduces exactly, and only down_proj.s512.dynM regressed (finding 6).

## Finding 1 — fp16 warp tier: the TMA+WARPSPEC launch crash persists; all four fp16 squares still locked out (P0 repeat)

All four fp16 squares remain worse: +13% (1024) to +46% (512). Same signature as the previous sweep's Finding 1:

- `eval failures`: 6 of the sweep's 9 bench_fail rows are `CUDA_ERROR_INVALID_VALUE` with shared knobs
  `TMA=True, WARPSPEC=True, MMA=mma_m16n8k16_f16` (k_matmul_262948 ×4, 207791 ×2, 180e20 ×1).
- With the WARPSPEC class dead at tune time, the greedy locks onto the surviving `WM8 WN2 FM1` class on every fp16
  square; the goldens rank 58–288 under the learned prior (no healthy data near them) while the cold analytic ranks
  them 0–20 — the heuristic is fine, the data is poisoned.
- One improvement over last sweep, and it's data-driven: square.4096.fp16 narrowed from 8.4× to 1.30× because the
  greedy no longer falls into the `MMA=0 FK-split` timeout class — the prior learned from that class's bench_fail rows.
  The crash class itself is unchanged.

**Recommendation (P0, unchanged from last sweep — not yet actioned):** debug the `TMA+WARPSPEC+mma_f16` launch failure
(CUDA_ERROR_INVALID_VALUE at launch; introduced by the masked warp-tier MMA work). Add a compile+launch regression test
pinning the golden `square.1024.fp16` knobs with the planner-derived TMA transport. Until fixed, every fp16 golden is
unreachable by construction and re-sweeping fp16 is wasted GPU time — consider `--kernel`-excluding fp16 from routine
sweeps.

## Finding 2 — the prior deploys FN4 over the measured-faster FN2 + tall-FM family (kv/o_proj.s512, square.2048/4096)

kv_proj.s512 (+7%), o_proj.s512 (+9%), square.2048 (+9%), square.4096 (+12%). Two distinct mechanisms:

- **kv/o_proj.s512 — calibration miss on measured data.** `eval variants` (k_matmul_1262e3 / 8781f5): the golden
  `FM10–14, FN2` family holds measured ranks 1–3 **with -O3 re-bench rows** (kv: 29.6–30.7 µs; o: 54.8–55.5 µs), while
  the prior's pick is an `FN4` config at rank 5/43 (1.25x, flagged `misses best`) and 4/43 (1.20x) — both with no -O3
  row of their own. The prior had deployable-truth evidence for the golden family and deployed an unmeasured
  extrapolation instead. Not an -O1/-O3 inversion and not reachability.
- **square.2048 — reachability.** The `FM26` golden is absent from the kernel's 69 measured variants; learned rank 394,
  analytic rank 121. Patience cannot reach rank 394. square.4096 sits between: its golden family was measured (rank 1
  in the DB view) but the live deploy chose a `SPLITK2 BM8 BN16 FM6` config (grid 11008) that isn't even the DB view's
  pick — see finding 4's deploy-vs-eval-pick discrepancy.

**Recommendation:** two halves. (a) The cheap, high-leverage fix: make the deploy-time argmin respect measured -O3
reservoir rows — e.g. restrict the argmin to measured configs when any exist within the prior's top-K predicted, or
add a measured-evidence bonus. Findings 2, 4, 5 and 6 are all this one bug wearing different knobs. (b) The repeat
fix from last sweep (still unactioned): refit the analytic weights with the big-FM regime represented and add a `D_*`
register-tile-intensity feature so FM26-class configs stop ranking 100+ deep cold.

## Finding 3 — square.512: greedy drifts to BM16/BK32/RING4 (+6–9%)

`eval golden`: misses are `BM 16/8, BK 32/64, RING 4/3`. Learned rank 4/1008, analytic rank 2 — both shallow; the
argmin misorders within its own top-5. The same `BM16 + RING4` family that genuinely wins on q_proj.s512 and the s128
parity adds loses here, so the prior is over-generalizing the family across free-dim sizes. Same class as finding 5
(shallow-rank misorder); covered by recommendation 2(a)'s measured-evidence weighting — the golden family has DB rows.

## Finding 4 — gate_up_proj.s128 / q_proj.s128: the DB-view pick IS the golden, but greedy deploys something else (+22% / +25%)

The strangest result this sweep. For gate_up_proj.s128, `eval variants` (k_matmul_2f6858) shows the golden
`BN16 FM6 FN4 OVERHANG` config as measured rank 1/16 **with the best -O3 row (24.1 µs)** and marks it as the prior's
own pick (`◄ pick: rank 1/16, 1.00x`). Yet the live greedy deploys `FM8`, no OVERHANG, at 29.8 µs live. `eval prior`
agrees with the deploy side: golden rank 6/1008 over the full gated enumeration — so the prior prefers six other
(unmeasured) configs when ranking the full enumeration, but prefers the golden when restricted to measured rows. The
previous sweep blamed the OVERHANG offer gate; this sweep disproves that — the config was offered and measured. The
gap between "argmin over full enumeration" (deploy) and "argmin over measured variants" (eval variants) is the bug
surface. q_proj.s128 is the same shape of failure (golden learned ranks 5 and 28; greedy lands `BK32 FM8 RING2`
at +25%).

**Recommendation:** same as 2(a) — the deploy argmin must not prefer an unmeasured extrapolation over a config with a
winning -O3 reservoir row. This finding is the cleanest reproducer for that work: one shape, 16 measured variants, the
golden is measured-rank-1 with -O3 truth and still loses the deploy. Start here.

## Finding 5 — down_proj.s32: FM1 over FM2 again (+10%, exact repeat of last sweep's finding 3)

`eval golden`: misses `FM 1/2, FN 4/2`. Learned rank 1/1008 — the argmin misorders its own top-2. The `vs gold` column
reads 0.85x but that's the stale-denominator artifact (recorded 16.4 µs is a pre-capture number; the golden lives at
12.2 µs today — fixed by this sweep's YAML edits for the replaced shapes, but this entry was left). The previous
sweep's recommendation (a `D_*` term separating per-thread work at degenerate M) stands, unactioned.

## Finding 6 — down_proj.s512.dynM: first dynamic-golden shortfall (+11%); prior had better -O3 evidence and ignored it

The only one of six dynM seeds that regressed under a clean-DB retune. Greedy deploys `BM16 FM4 RING3`; golden is
`BM8 FM8 RING2`. `eval variants` (k_matmul_2189d8, 49 measured configs): the golden's `BM8 FM8 SPLITK2` family is
measured rank 1 with **-O3 75.7 µs vs the pick's -O3 80.9 µs** — again, measured deployable truth beaten by the
argmin's preference. Meanwhile `eval prior` ranks this golden 0/1008, i.e. the enumeration scoring agrees with the
golden, and still the live deploy lands elsewhere — the deploy-side pick and the eval-side rank disagree (the same
discrepancy as finding 4, seen from the other side). The masked-tier analytic ranks are improving but still cold-deep
(20–125 across dynM shapes, median ~28 vs the seed report's 55) — `scripts/golden_knob_heuristics.py` still hasn't
been refit with the dynM goldens (seed report's finding 4 recommendation, unactioned).

**Recommendation:** fold into 2(a)'s measured-evidence work, and refit the analytic weights now that the dynM goldens
are recorded. Also worth a 30-minute look: why `eval prior`'s rank-0 and the live deploy disagree on this shape —
if the deploy path stamps different `S_*`/`D_*` features than `Sample.from_golden`, every eval rank this report cites
is an approximation of the thing that actually picks kernels.

## Workflow notes

- **Tune wall time**: 52 min for 29 targets (previous: 40 min for 23). The dynM shapes are mid-cost. The fp16 squares
  remain pure waste until finding 1 is fixed (their goldens are unreachable by construction) — a
  `--kernel`-exclusion would save ~10 min per sweep.
- **Thermal drift dominates absolute µs; ratios are solid.** Pass 1 vs pass 2 absolute numbers differ ~10% uniformly
  (pass 2 hot), while within-run greedy/golden ratios agree to ≤3% on all 16 re-run shapes. Two consequences: (a) the
  recorded `deplodock_us` of any entry is a ±10% thermal lottery — this morning's dynM seeds turned out to be hot-state
  numbers (their pass-2 live re-bench matches the YAML to 0.2 µs); (b) the previous sweep's "≤1.5% spread" claim
  held within a thermal state, not across invocations. Proposal: categorize on ratios only (this sweep did), and have
  the A/B table print the same-run eager µs beside each row so a recorded ratio-to-reference survives thermal drift.
- **`eval variants` kernel-name collisions persist** (repeat note, unfixed): the k_matmul_207791 group shows fp32
  OVERHANG rows at 14–20 µs that belong to neither square.512 (9.8 µs live) nor square.512.fp16 (5.1 µs); 1cd565's
  group doesn't match q_proj.s128's live numbers either. Findings 2/4/6 leaned on the groups that were verifiably
  clean (latencies match the live A/B). The view still needs a ShapeKey subgrouping or `--shape` filter.
- **`eval prior`'s `vs gold` stale denominators persist** (repeat note): entries not touched this sweep keep
  pre-capture / hot-state numbers (down_proj.s32's 0.85x, q_proj.s512.dynM's 1.66x are artifacts). The thermal note
  above makes a one-shot re-record pass less attractive than switching the column to a same-run ratio.
- **reduce/pointwise goldens still unsweepable** (repeat note, unfixed): `tune --dataset golden` and `run --golden`
  cover only matmul shapes; the 5 reduce/pointwise entries are silently outside the loop.
- **A/B remains 29 + 16 cold process launches** (~45 min total). The previous note proposing an in-process
  `run --bench --dataset golden` loop with machine-readable per-shape output stands — this sweep again assembled the
  outcome table by awk-parsing 45 logs.
- **Fixed since the previous reports and held**: (a) the dynM eval-view fixes from the seed report (static-trace
  mismatch + OVERHANG list/tuple false-flags) — all six dynM rows now show exact or single-knob diffs in
  `eval golden`, matching the live A/Bs; (b) fp16 shapes now appear in `eval prior --dataset golden`'s rank list
  instead of being silently dropped (they rank 58–288, which is finding 1's knock-on made visible — exactly what the
  previous sweep's note asked for).
