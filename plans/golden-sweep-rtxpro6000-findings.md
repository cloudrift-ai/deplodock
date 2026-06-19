# Golden sweep findings — RTX PRO 6000 Blackwell Max-Q (sm_120), 2026-06-19

- **GPU**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, `compute_cap (12, 0)` (sm_120), driver/runtime
  CUDA 12.9 (nvcc 12.9), torch 2.12.1+cu130. Golden file:
  `deplodock/compiler/pipeline/search/goldens/rtxpro6000_sm120.yaml` (29 matmul shapes).
- **Sweep command**: `deplodock tune --dataset golden --clean` (re-fit the prior from scratch on the 29-shape
  reservoir), then per-shape `deplodock run --bench --golden NAME` A/B, each confirmed over **3 runs** for the 18
  non-identical-knob shapes.
- **Wall time**: tune ≈ 72 min (sum of per-shape inner search, serial single-GPU); A/B + 2× confirmation re-runs
  ≈ 18 min; `eval` evidence ≈ 5 min. Plus a one-time ≈ 20 min environment repair (see Workflow notes — the venv shipped
  with no pip/packages, missing `python3.12-dev`, and `nvcc` off PATH).
- **Branch**: `feature/golden-sweep-rtxpro6000-2026-06-19`.
- **Tally**: **8 replaced / 1 added / 15 unchanged / 5 left (worse vs recorded)**.
- **Headline**: the warm, full-dataset re-tune found materially better masked-tile configs for exactly the `.dynM`
  shapes the 2026-06-13 **seed** report flagged as slack (its Finding 2) — `kv/o/gate_up/down_proj.s512.dynM` improve
  **6–38 %**, and `square.512.dynM` **25 %**. The flip side: re-fitting the prior **from `--clean` on only the 29-shape
  reservoir regressed four static squares/projections** (`square.1024` +44 %, `square.1024.fp16` +26 %, `square.4096`
  +21 %, `kv_proj.s512` +11 %) whose golden configs the seed had recorded at parity — those are the findings below.
  Run-to-run variance was **< 1 %** on every shape (CUDA-graph capture is very stable on this box), so the noise floor
  never gated a decision.

## Method note — within-run A/B is sound; some recorded configs no longer reproduce

The live eager/cuBLAS reference matched each shape's recorded `cublas_us` within ~1–2 % across the board (e.g.
`square.2048` 321 vs 324.8, `square.4096` 2399 vs 2426, `kv_proj.s512` 38 vs 39.5), so the environment is consistent
and greedy's live -O3 number is directly comparable to the **recorded** `deplodock_us` — not just to the live golden
row. That cross-check also exposed a wrinkle: a few recorded golden **configs** no longer bench at their recorded
latency when re-run with the same knobs (cuBLAS is stable, so this is deplodock-side codegen/flag drift since the
2026-06-13 seed on torch 2.12.0): `square.2048` 357.6 → **445** (+24 %), `square.4096.fp16` 524 → **655** (+25 %),
`o_proj.s512.dynM` 86.9 → 68 (−22 %), `gate_up_proj.s512.dynM` 89.7 → 110 (+23 %). Recording decisions used the
recorded `deplodock_us` as the bar and never wrote a number worse than what was already on file.

## Per-shape outcome (greedy = deployed -O3 pick; golden = recorded knobs re-benched live this run)

| shape                            | greedy µs | golden-live µs | recorded µs |  ratio | category                       |
|----------------------------------|----------:|---------------:|------------:|-------:|--------------------------------|
| square.512                       |      11.0 |           11.2 |        11.0 |  0.98  | same (identical knobs)         |
| square.1024                      |      70.0 |           48.5 |        49.2 |  1.44  | **worse → left (Finding A)**   |
| square.2048                      |     392.4 |          445.1 |       357.6 |  1.10† | **worse → left (Finding E)**   |
| square.4096                      |    3603.6 |         2978.9 |      3021.4 |  1.21  | **worse → left (Finding B)**   |
| square.512.fp16                  |       3.7 |            3.8  |         3.8 |  0.97  | **added** (parity, diff knobs) |
| square.1024.fp16                 |      17.0 |           13.5 |        13.9 |  1.26  | **worse → left (Finding C)**   |
| square.2048.fp16                 |      77.3 |           77.8 |        78.6 |  0.99  | same (identical knobs)         |
| square.4096.fp16                 |     515.9 |          655.5 |       524.1 |  0.79‡ | **replaced** 524.1 → 515.9     |
| qwen3_06b.q_proj.s32             |       7.8 |            7.8  |         7.8 |  1.00  | same (identical knobs)         |
| qwen3_06b.kv_proj.s32            |       6.2 |            6.2  |         6.2 |  1.00  | same (identical knobs)         |
| qwen3_06b.o_proj.s32             |      10.0 |           10.0  |        10.0 |  1.00  | same (identical knobs)         |
| qwen3_06b.gate_up_proj.s32       |      11.7 |           11.9  |        11.6 |  0.98  | same (identical knobs)         |
| qwen3_06b.down_proj.s32          |      13.8 |           13.9  |        13.8 |  0.99  | same (identical knobs)         |
| qwen3_06b.q_proj.s128            |      18.5 |           18.9  |        18.7 |  0.98  | same (identical knobs)         |
| qwen3_06b.kv_proj.s128           |      12.1 |           12.2  |        12.1 |  0.99  | same (identical knobs)         |
| qwen3_06b.o_proj.s128            |      21.3 |           21.1  |        20.9 |  1.01  | same (tie, diff knobs)         |
| qwen3_06b.gate_up_proj.s128      |      31.3 |           30.9  |        30.9 |  1.01  | same (tie, diff knobs)         |
| qwen3_06b.down_proj.s128         |      30.0 |           33.9  |        33.3 |  0.88  | **replaced** 33.3 → 30.0       |
| qwen3_06b.q_proj.s512            |      50.5 |           55.2  |        55.8 |  0.92  | **replaced** 55.8 → 50.5       |
| qwen3_06b.kv_proj.s512           |      33.3 |           30.1  |        30.3 |  1.11  | **worse → left (Finding D)**   |
| qwen3_06b.o_proj.s512            |      65.9 |           65.8  |        66.3 |  1.00  | same (identical knobs)         |
| qwen3_06b.gate_up_proj.s512      |      84.8 |           83.8  |        85.1 |  1.01  | same (tie, diff knobs)         |
| qwen3_06b.down_proj.s512         |      97.4 |           97.5  |        97.8 |  1.00  | same (identical knobs)         |
| square.512.dynM                  |      10.2 |           13.6  |        13.6 |  0.75  | **replaced** 13.6 → 10.2       |
| qwen3_06b.q_proj.s512.dynM       |      57.6 |           57.5  |        55.2 |  1.00  | same (tie, diff knobs)         |
| qwen3_06b.kv_proj.s512.dynM      |      29.8 |           47.0  |        47.8 |  0.63  | **replaced** 47.8 → 29.8       |
| qwen3_06b.o_proj.s512.dynM       |      53.7 |           68.1  |        86.9 |  0.79  | **replaced** 86.9 → 53.7       |
| qwen3_06b.gate_up_proj.s512.dynM |      83.9 |          110.2  |        89.7 |  0.76  | **replaced** 89.7 → 83.9       |
| qwen3_06b.down_proj.s512.dynM    |      78.3 |           81.5  |        87.5 |  0.96  | **replaced** 87.5 → 78.3       |

† `square.2048` greedy (392) beats the live golden re-bench (445) but is **worse than the recorded 357.6**, and the
recorded golden config itself no longer reproduces (445 ≠ 357.6) — left untouched, see Finding E.
‡ `square.4096.fp16` greedy (516) beats both the recorded 524 and the now-regressed golden config (655) — a small win
that also retires a stale config.

The 8 replaces + 1 add are committed to the YAML; their knobs were read from the greedy `k_matmul` row of the -O3
`run --bench` A/B (search knobs only; transport/codegen flags the planner re-derives were dropped).

## Finding A — `square.1024`: prior deploys a large-`BK` config 44 % slower than a measured near-golden one

Greedy deploys `{BM:8, BN:32, BK:64, FM:4, FN:2, …}` at **70.0 µs**; the golden `{BM:16, BN:16, BK:32, FM:6, FN:4}`
re-benches at **48.5 µs** (≈ its recorded 49.2, so it reproduces). This is the worst pick-miss of the sweep (1.44×).

- `eval analytic`: golden ranks **16/1008** (cold prior mis-prices it — it prefers `BK:64`).
- `eval prior --dataset golden`: golden ranks **100/1008** under the *learned* prior, yet the `vs gold` reservoir
  column reads **1.02×** — i.e. the inner search **did measure** a config within 2 % of golden. So this is **not** a
  reachability gap: a golden-class config exists in the reservoir, but the re-fit learned prior buried the golden at
  100th and the greedy pipeline deployed an over-valued large-`BK` config instead.
- **Recommendation (P1, prior model)**: the learned prior over-weights `BK=64` on mid-size fp32 squares. Either the
  evidence pick (`Prior.pick`'s measured -O3 reservoir branch) should fire here — it already holds the 1.02× config —
  or refit the analytic/learned weights with a `D_*` term that penalises large `BK` when `M·N` is small enough that
  `BK=32` keeps more CTAs resident. `scripts/golden_knob_heuristics.py` over the static-square goldens.

## Finding B — `square.4096`: golden ranks shallow but greedy still lands 21 % slow, no -O3 reservoir anchor

Greedy `{BM:8, FM:8, RING:2}` at **3603 µs** vs golden `{BM:16, FM:6, RING:3}` re-benched **2979 µs** (≈ recorded 3021).

- `eval analytic` rank **17/1008**; `eval prior --dataset golden` rank **19/1008** (shallow under both priors) — yet the
  deployed pick is still 21 % slow, and the `vs gold` column is **—** (no `H_opt=3` reservoir row for this shape).
- Read: the prior ranks golden near the top, but the **greedy enumeration argmin** over the full (largely unmeasured)
  space still diverges, and the -O3 re-bench tolerance band never sampled a golden-class config for the 4096³ square, so
  there is no deployable evidence to anchor the pick.
- **Recommendation (P2)**: widen the -O3 re-bench band (`DEPLODOCK_O3_TOL`) or bump patience for the largest fp32
  square so the tolerance sweep actually re-benches the `BM:16/FM:6` neighbourhood at -O3 and gives `evidence_pick`
  something to deploy.

## Finding C — `square.1024.fp16`: learned prior contradicts the analytic, and reachability fails (1.25× in reservoir)

Greedy at **17.0 µs**, golden `{FM:4, FN:2, WM:1, WN:4, RING:4}` at **13.5 µs** (1.26×); only **4/9** knobs match —
the deployed warp tiling is essentially the golden's transposed (`WM:4/WN:1` vs golden `WM:1/WN:4`, `FM:1/FN:4` vs
`FM:4/FN:2`).

- `eval analytic`: golden ranks **0/2415** — the *cold* prior knows this config is best.
- `eval prior --dataset golden`: golden ranks **141/2415** under the *learned* prior, and `vs gold` reads **1.25×** —
  the reservoir's best measured config is itself 25 % off golden. So the inner search **never measured** a golden-class
  warp tiling, and the learned prior actively buried a config the analytic prior ranked #1.
- **Recommendation (P1, search + prior)**: this is the clearest "the learned re-fit over-fit away from a known-good
  config" case. The fp16 warp-tile search needs more patience to reach the golden `WM:1/WN:4` region, and the prior fit
  should not be allowed to rank a config the analytic prior places #1 down to 141 — consider seeding the learned fit
  with the analytic top-1 per shape, or a `D_*` warp-aspect feature so `WM·WN` orientation is visible to the regressor.

## Finding D — `kv_proj.s512`: near-miss, `SPLITK=2`/`BN=32` under-ranked (11 %)

Greedy `{BN:16, SPLITK:1, …}` at **33.3 µs** vs golden `{BN:32, SPLITK:2}` at **30.1 µs** (1.11×); 9/11 knobs match.

- `eval analytic` rank **5/1008**; `eval prior --dataset golden` rank **35/1008**, `vs gold` **1.00×** (a golden-equal
  config was measured). A modest mis-rank, not a reachability gap.
- **Recommendation (P3)**: smallest of the findings. `SPLITK=2` with `BN=32` is under-valued for this
  `M=512,N=1024,K=1024` shape; a patience bump or a `SPLITK×occupancy` interaction feature should recover it.

## Finding E — `square.2048`: deepest prior mis-rank **and** the recorded golden config no longer reproduces

Two problems compound here. (1) `eval analytic` ranks golden **72/1008** and `eval prior --dataset golden` ranks it
**551/1008** — the deepest mis-rank in the entire set; the prior simply does not know this shape. (2) The recorded
golden config (`BM:16, BN:32, BK:32, FM:12`) re-benches at **445 µs**, +24 % over its recorded **357.6** (stable across
3 runs), while cuBLAS is unchanged — a deplodock-side config-reproduction drift since the 2026-06-13 seed. Greedy (392)
beats the live golden (445) but is still 10 % over the recorded number, so recording it would write a regression.

- **Decision**: left untouched (the recorded 357.6 stays as the aspirational best; no current config reproduces it).
- **Recommendation (P2)**: re-seed `square.2048` from a fresh higher-patience tune and, separately, investigate the
  large-square / masked-tile codegen drift between torch 2.12.0 (seed) and 2.12.1 (now) — `square.4096.fp16` shows the
  same +25 % config drift. The drift is real (3-run-stable, cuBLAS-anchored), so a golden's stored `deplodock_us` is not
  a durable contract for these shapes across toolchain bumps.

## Status of the 2026-06-13 seed report's findings

- **Seed Finding 2 (1.1–1.27× slack on the `.dynM` masked tiles + big squares)** — **largely resolved by this sweep.**
  The warm full-dataset prior found genuinely better masked-tile configs: `kv_proj.s512.dynM` 47.8 → 29.8, `o_proj`
  86.9 → 53.7, `gate_up` 89.7 → 83.9, `down_proj` 87.5 → 78.3, `square.512.dynM` 13.6 → 10.2. The dynM family is no
  longer the slack frontier. The *static* fp32/fp16 squares (`square.1024`, `square.4096`, `square.1024.fp16`) took its
  place — see Findings A–C.
- **Seed Finding 3 (5090/PRO 6000 same-cap `(12,0)` collision)** — its fix **held**: `run --bench --golden` benched
  exactly one golden row per shape (the live PRO 6000 entry, `ng=1` everywhere), and `eval golden` / `eval prior
  --dataset golden` enumerated exactly the 29 PRO 6000 shapes — the live-GPU scoping (`goldens_for_live_gpu` /
  SM-count distinction) is working.
- **Seed Finding 1 (`q_proj.s512.dynM` needed a higher-patience re-tune)** — this sweep's default-patience pick
  reproduced the seed's 55.2 within noise (57.6, 1.00× the live golden), so the recorded config is stable; left as-is.

## Workflow notes

- **Environment was not provisioned** — the `venv/` existed but had no pip and zero packages, so `make setup`'s
  `[ ! -d venv ]` guard skipped it silently; bootstrapping needed `ensurepip`, then the `cppyy` build failed on a
  missing `Python.h` (`python3.12-dev`), and `nvcc` was off PATH (system CUDA at `/usr/local/cuda/bin`). ≈ 20 min lost.
  *Improvement*: add a step-0 preflight to the skill — `deplodock run -c "<tiny matmul>"` must succeed (it transitively
  checks torch/CUDA/cppyy/nvcc) before tuning, with the `ensurepip` / `python3.12-dev` / `PATH=/usr/local/cuda/bin`
  fixes noted as the common failure modes.
- **`eval variants --kernel <golden-name>` can't drill into a single shape** — the `--kernel` filter matches the
  kernel's C-identity (`k_matmul`), which all matmul shapes share, so `--kernel square.1024` returns "No measured
  variants". The per-shape reachability the Finding template asks for had to come from `eval prior --dataset golden`'s
  `vs gold` column instead. *Improvement*: let `eval variants` accept a golden NAME or `ShapeKey` signature and resolve
  it to the matching `S_*` group, or document that matmul variants must be sliced by shape signature, not name.
- **Recorded golden config not reproducing its `deplodock_us`** made greedy-vs-recorded ambiguous for `square.2048` /
  `square.4096.fp16` / two dynM shapes; disambiguating env-shift from config-drift required a hand-rolled cuBLAS-vs-
  `cublas_us` cross-check across all shapes. *Improvement*: `run --bench --golden` should print the recorded
  `deplodock_us` beside the live golden row and flag a >5 % divergence ("golden no longer reproduces") so the operator
  doesn't reconstruct it by hand.
- **The step-4 noise-floor re-runs were unnecessary here** — 3-run variance was < 1 % on every shape (vs the skill's
  warned 10–13 %), so two extra passes over 18 shapes (≈ 12 min) changed no category. *Improvement*: gate the
  confirmation re-run on the first-pass ratio — only re-bench shapes landing in the ambiguous band (≈ 0.90–1.10); the
  clear wins (0.63, 0.75) and clear losses (1.44) don't need it.
- **All A/B data lived in 29×3 per-shape logs** — extracting greedy/golden/eager µs and per-knob diffs needed a custom
  parser. *Improvement*: a `run --bench --json` (or a `tune --dataset golden --ab-report`) that emits the whole
  greedy-vs-golden table — latencies, knob diffs, category — in one machine-readable pass would remove the entire
  parsing step from this loop.
