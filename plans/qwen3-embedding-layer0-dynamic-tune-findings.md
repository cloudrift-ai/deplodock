# Qwen3-Embedding-0.6B layer 0 — DYNAMIC-shape tune findings (2026-06-13)

Status: **clean dynamic tune (`--dynamic seq_len@x:1`) with the M9 symbolic-axis tier landed — the standalone /
split matmuls now reach the masked warp/MMA tier (a large win over the pre-M9 dynamic run), but the three
SDPA-fused kernels (QK^T+softmax, o_proj, attn@V) stay scalar by three distinct gates and carry ~90% of the
deployed cost, leaving the layer ~10x behind torch.compile at seq_len 512.** This supersedes the pre-M9 dynamic
report whose findings 1–4 are all addressed on `main` now (`41f64eaf` masked warp-tier MMA + symbolic-row splits +
masked prologue tiles; `4347ee7a` strided-cooperative rows; the full-model-table shape-mismatch fix). The gap is no
longer "everything is scalar" — it is three specific SDPA gates, two known-by-design and one a clean,
actionable eligibility lockout (finding 2).

- Command: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir
  /tmp/tune-model-qwen3-emb-l0/dump`
- Hardware: RTX 5090 (sm_120), driver 580.x, nvcc 13.0, ncu 2025.3.1 (perf counters permitted)
- Run stats: wall **~40 min** (tune phase 2300.7 s + the -O3 `--bench`), **2 fused terminals** (the gated-MLP
  structural split offered — finding-2 of the pre-M9 report addressed; vs 1 terminal pre-M9, 16 static),
  **1880 ok `perf` rows / 0 `bench_fail`**, all CUDA-graph captured. Prior: 2308 benches (warmup 158 / post 2150),
  post Spearman **+1.00**, 98% of post benches ≥2x the running best.
- DB / prior: **isolated** to the work dir (`DEPLODOCK_TUNE_DB=/tmp/tune-model-qwen3-emb-l0/autotune.db`,
  `DEPLODOCK_PRIOR_FILE=.../prior.json`) so this scalar-heavy symbolic tune does not clobber the global default
  prior (the pre-M9 report's finding 5 hazard — avoided here by construction).
- **Number families**: bench tables / reproducer / `--ab` rows below are -O3 (deployable, CUDA-graph captured);
  tune-DB `us` quoted for ranking context are `-Xcicc -O1` (ranking signal only). NCU durations run at locked base
  clock — compare ratios / counters, not absolutes.
- **Dynamic measurement semantics**: the trace is symbolic in `seq_len`, hint = `DEFAULT_SEQ_HINT` 512 (`--seq-len`
  does NOT move it; it only sizes trace example tensors). Every per-op tune bench, every reproducer bench, and —
  now — the full-model table run at the 512 hint (the pre-M9 finding-4 mismatch is fixed: the table prints
  `benched at seq_len=512 (symbolic hint; torch inputs tiled to match)`). Masked-tile boundary guards
  (`if (coord < seq_len)`) are part of the measured cost.

## Bench results (-O3, CUDA-graph captured, seq_len=512 hint)

Full-model table as printed by the tune's `--bench` (now a valid like-for-like comparison — torch inputs tiled to
the 512 hint):

```
Backend        Latency (us)  vs Eager
-------------------------------------
Eager PyTorch           248     1.00x
torch.compile           167     1.49x
Deplodock              1635     0.15x
benched at seq_len=512 (symbolic hint; torch inputs tiled to match)
```

Per-kernel reproducer bench at the 512 hint (sorted by deplodock µs; layer-op labels from each kernel's
`.torch.json` provenance; `vs eager`/`vs tcompile` from the tune's per-kernel table; machine-readable rows in
`62_kernel_bench.json`):

| Kernel                  | Layer op                                                  | eager | tcompile | deplodock | vs tcompile |         tier |
|-------------------------|-----------------------------------------------------------|------:|---------:|----------:|------------:|-------------:|
| `k_sdpa_reduce`         | RoPE(q,k) + QK^T + softmax (SDPA numerator)               |   160 |       27 |      1052 |       0.03x |       scalar |
| `k_linear_sdpa_reduce`  | attn-out reshape + o_proj (linear_3) + residual           |    45 |       43 |       461 |       0.09x |       scalar |
| `k_sdpa_linear_reduce`  | SDPA attn@V (+ in-kernel V producer)                      |    34 |       33 |       347 |       0.10x |       scalar |
| `k_linear_mean_reduce`  | post-attn RMSNorm + MLP gate+up (linear_4/5) + SiLU·up    |   119 |       57 |        68 |       0.84x | warp (split) |
| `k_linear`              | MLP down (linear_6) + residual                            |    27 |       27 |        50 |       0.54x |         warp |
| `k_linear_reduce`       | q_proj matmul                                             |    16 |       16 |        29 |       0.55x |         warp |
| `k_mean_linear_reduce`  | q_norm RMSNorm (+ q_proj slice)                           |   105 |       20 |        29 |       0.69x |  coop-reduce |
| `k_mean_linear_reduce`  | k_norm RMSNorm (+ k_proj slice)                           |    76 |       14 |        18 |       0.78x |  coop-reduce |
| `k_linear_reduce`       | v_proj matmul                                             |    10 |       10 |        18 |       0.56x |         warp |
| `k_mean`                | input RMSNorm                                             |    65 |        4 |         2 |       2.00x |    pointwise |
| **Σ (per-kernel solo)** |                                                           |   657 |      251 |      2074 |       0.12x |              |

The deployed whole-program e2e is **1635 µs** (full-model table) vs the 2074 µs sum of per-kernel solo windows
(the solo windows over-count — no cross-kernel cache reuse). The three SDPA-fused kernels (`k_sdpa_reduce`,
`k_linear_sdpa_reduce`, `k_sdpa_linear_reduce`) carry **1860 µs — 90% of the per-kernel Σ** and all three are
scalar-tier; they get the deep dives below. Everything else either wins (input RMSNorm 2x eager, q/k-norm, the
split gated-MLP) or is a small warp-tier matmul trailing cuBLAS by <2x (finding 4). There were **zero bench
failures** and prior reachability is clean (mean 1.21x / median 1.06x over the DB) — **nothing here is a search
shortfall**; the whole gap is what the lowering tier *enumerates* for SDPA-fused matmuls.

## Finding 1 — QK^T + softmax + RoPE (`k_sdpa_reduce`, 1052 µs, 51% of the per-kernel Σ) — scalar by two gates

> **Status: largely addressed (indirectly)** on `feature/split-demoted-symbolic-kn`. The QK^T matmul itself stays
> scalar (the softmax-over-N prologue + transposed-B are intrinsic — the operand-cone split can't peel the softmax;
> true flash is still future). But admitting symbolic-N splits (below) branched the structural tree from 2 → 8
> terminals, forcing far deeper exploration of this previously-3-config kernel: a clean re-tune deploys the QK^T
> region at **155 µs (0.96x eager)**, down from 1052 µs — the RoPE gather split off and a much better attention
> config found. The deployable matmul-on-tensor-cores flash kernel remains the only path to beat torch.compile's 25 µs.

**Symptom.** The single biggest kernel deploys scalar tier with only **3 measured configs** in its leaderboard —
no `MMA` / `WM` / `WN` column at all (`eval variants --kernel k_sdpa_reduce`: rank 1/3, all `BM/BN/FM/FN`). Both
its free axes (M = q-rows = `seq_len`, N = k-cols = `seq_len`) are symbolic and stay degenerate; torch runs this as
one fused flash kernel. deplodock 1052 µs vs torch.compile 27 µs (39x).

**Evidence.**

- NCU compare on the reproducer (`run --ir k_sdpa_reduce_77b0f0.torch.json --bench --profile`; it re-fuses into two
  kernels — see workflow notes — locked base clock, compare counters not absolutes):

  ```
  side  kernel                            occ%   sm%  dram%  fma%    lsu.inst    ld.cnflct  regs
  dep   k_sdpa_reduce_6e4bd6 (QK^T+sm)    48.6  44.6    2.8  42.3   8,396,800            0    40
  dep   k_sdpa_reduce_042770 (RoPE tile)  62.5  29.5    0.2   7.1  60,817,408  155,223,141    64
  ref   pytorch_flash::flash_fwd_splitkv   8.2  16.1   22.2   2.9     300,800            0   206
  ```

  The QK^T+softmax half (`6e4bd6`, 923 µs of the 1052) is **compute-bound on the scalar FMA pipe** (42.3% fma,
  2.8% dram) — healthy occupancy for its tier, wrong pipe. The RoPE half (`042770`, 129 µs) is **bank-conflict /
  LSU bound**: 155M load bank conflicts, 60.8M LSU instructions (200x the flash kernel's) from the masked
  slice/cat/neg rotate gather.
- The two gates, both in `010_partition_loops.py`. (a) The warp-tier offer requires `not prologue`
  (`010_partition_loops.py:685`) and the symbolic-axis masking is disabled under a fused prologue
  (`mask_ok`/`prologue_mask_ok` at `:636-645`) — QK^T carries the softmax max/sum prologue, so symbolic M/N stay
  degenerate and MMA is never offered. (b) Even without the prologue, QK^T is **transposed-B** (Q@K^T — both
  operands carry the K reduce axis = head_dim in their last dim), which `classify_matmul_operands` can't split into
  A/B, so `is_atom_eligible` rejects it (`_atom.py:164-173`).

**Root cause.** Known limitation, correct-by-design today: flash-style attention with a symbolic seq axis on both
the matmul M and N (and softmax over the symbolic N) is the symbolic-K warp-tier follow-up the M9 work explicitly
deferred (`_enumeration` "cooperative work over symbolic axes is M5+"; the prologue-mask comment at
`010_partition_loops.py:626-644`). This report's contribution is the price tag: **1052 µs — half the layer** — and
the NCU split showing the RoPE gather is a *separate* bank-conflict problem layered on top of the scalar matmul.

**Repro.** `deplodock run --ir /tmp/tune-model-qwen3-emb-l0/dump/07_lowering_cuda.kernels/k_sdpa_reduce_77b0f0.torch.json
--bench --profile` (no `--dynamic` — the reproducer keeps its symbolic dims, benches at the 512 hint).

**Suggested fix (highest priority of everything here, but it is the flash-attention milestone — large).** A
symbolic-seq flash kernel (online-softmax tiling over the symbolic N with a masked K=head_dim warp tile) is the
real fix and is out of scope for incremental work. **Cheaper, independent win available now**: the RoPE half's
155M bank conflicts are a smem-layout problem in the masked rotate-gather, not a tier issue — `PAD_SMEM` /
`PERMUTE_LANES` on `042770` is worth an A/B (its 129 µs is bigger than several whole winning kernels).

## Finding 2 — o_proj locked out of the warp tier by its collapsed-reshape operand (`k_linear_sdpa_reduce`, 461 µs)

> **Status: addressed** on `feature/split-demoted-symbolic-kn` (the fix this finding's drill-down motivated). The
> structural split (`005_split_demoted`) now offers on this kernel dynamically and the consumer reaches the warp
> tier: a clean re-tune deploys it via an `_xn` contiguizing producer + an `mma.sync` consumer at **231 µs**, down
> from 461 µs. Root cause of the non-offer was NOT the eligibility gate below (that gate is correct) but a separate
> cut bail: the split treated the symbolic dim var `seq_len` in the collapsed-reshape stride arithmetic as an
> unmodeled-scope read (`_split_demoted.dim_names` now admits it). The split materializes a contiguous `(seq, 2048)`
> activation so the consumer's A operand is a single-K-dim Load — exactly the option (a) suggested below.

**Symptom.** The attn-out o_proj (`reshape(attn) → linear_3 → +residual`) is a symbolic-M matmul with **static
K=2048, N=1024** — structurally identical to the MLP down-proj (`k_linear`, M=seq, K=3072, N=1024, +residual),
which *does* deploy `mma.sync` and benches 50 µs. Yet o_proj emits **zero `mma.sync`** (139 configs, all scalar,
`eval variants` shows no `MMA`/`WM`/`WN` column) and benches 461 µs — 10x torch.compile's 43 µs.

**Evidence.**

- Tier confirmation by grep on the emitted CUDA: `k_linear_0837e7` (down-proj) and `k_linear_reduce_*` (q/v_proj)
  each carry `mma.sync`/`ldmatrix`; `k_linear_sdpa_reduce_43208b` carries **none**.
- The gate, `_atom.py:122-135`. The o_proj's A-operand is the attention output reaching the matmul through a
  **collapsed reshape** `(seq, 16, 128) → (seq, 2048)`, so the K axis is split across two index dims
  (`[(a/128)%16, …, a%128]`). `020_stage_inputs._classify` only stages a load whose cache var lands in a *single*
  index dim, so a collapsed-K operand is left gmem-direct — and `ldmatrix` is smem→register only. The eligibility
  predicate mirrors that rejection (`if len(k_dims) > 1 … return False`, `_atom.py:134`), with the in-code comment
  naming this exact case: *"a collapsed-reshape operand (e.g. an attention output reaching the o_proj via
  `[(a/128)%16, …, a%128]` — K split across two dims) is rejected there."*
- `--ab MMA=1` confirms it is a *structural* lockout, not a knob/search miss: pinning MMA on the reproducer still
  deploys scalar (the K-reduce `('a0',)` row stays `BM=16,BN=16,FM=1,FN=8`, no warp columns) — `is_atom_eligible`
  returns False before any warp row is enumerated, so the pin has nothing to force.

**Root cause.** A real, addressable eligibility lockout (distinct from finding 1's known limitation). The matmul
itself is warp-tier-eligible (static K/N, maskable symbolic M); only its *operand layout* — the un-materialized
attention reshape — blocks staging.

**Repro.** `deplodock run --ir .../k_linear_sdpa_reduce_43208b.torch.json --bench --ab "MMA=1"` (scalar rows
persist). Compile-only, no GPU: `deplodock compile .../k_linear_sdpa_reduce_43208b.torch.json --ir cuda` (no
`mma.sync`, gmem-direct inner loop).

**Suggested fix (medium-high — 461 µs, and the cleanest actionable item).** Either (a) extend the structural split
(`005_split_demoted`) to peel the o_proj into its own kernel reading a materialized contiguous
`(seq, 2048)` activation (the split already does exactly this for the gated-MLP `xn` buffer — see finding 4 — so
the machinery exists), or (b) teach `020_stage_inputs._classify` + `is_atom_eligible` to stage a collapsed-K
operand by materializing the reshape into an smem staging tile with the two-dim K addressing folded into the copy.
Option (a) reuses proven code and is lower-risk.

## Finding 3 — attn@V keeps the symbolic K degenerate (`k_sdpa_linear_reduce`, 347 µs) — by design; -O1 flag is a false alarm

> **Status: confirmed by-design** on `feature/split-demoted-symbolic-kn`. The split was deliberately NOT extended
> to symbolic K (P@V's reduce is `seq_len`): the warp tier needs a static reduce for `ldmatrix`, so a symbolic-K
> matmul is scalar fused or split — splitting would only add a materialization with no tier upgrade. attn@V's path
> to the warp tier is the same flash work as Finding 1, not the split. (It re-benched to 297 µs in the re-tune —
> within run-to-run scalar-pick variance.)

**Symptom.** SDPA attn@V (P@V, with the in-kernel V producer) deploys a masked thread tile `FM=1, FN=8` (scalar),
347 µs vs torch.compile 33 µs. `eval variants` flags the pick as **rank 11/180, 1.37x of best** — superficially a
search shortfall.

**Evidence.** The `-O3 us` column in the same `eval variants` output inverts the flag: the pick (rank 11 at -O1)
re-benches to **240.1 µs -O3**, *below* the -O1 rank-1 config's **255.7 µs -O3**. The -O1 ranking ties configs
that differ at -O3 (a known effect — see the `o1-pick-miss-flags-usually-invert` lesson); the deployed pick is in
fact the best -O3 config. So this is **not** a search shortfall. The scalar tier is structural: P@V has K = `seq_len`
(symbolic), and symbolic-K matmuls get masked thread tiles at `FM=FN=1` and are explicitly out of the warp tier
(`010_partition_loops.py:685` gates on `not k_symbolic`; symbolic-K flash tiling is the same future milestone as
finding 1).

**Root cause.** Correctness/scope bail, documented in-code (symbolic-K warp tier is future work). The pick is
optimal within the enumerated (scalar) space.

**Repro.** `deplodock run --ir .../k_sdpa_linear_reduce_a76a28.torch.json --bench` (reproduces 347 µs); the
`-O3 us` column is in `eval variants --kernel k_sdpa_linear_reduce`.

**Suggested fix (low priority — folds into finding 1's flash milestone).** Same symbolic-K warp/flash path as
finding 1; no independent cheap win. Do **not** spend patience here — the pick is already -O3-optimal.

## Finding 4 — M9 wins to record, and the warp-tier matmuls that still trail cuBLAS (known-good / low µs)

Recorded because the contrast is the headline of this run, and because the warp-tier matmuls are a small
codegen-quality item worth one A/B someday:

- **M9 wins (vs the pre-M9 dynamic run, which had *every* matmul scalar):** the down-proj `k_linear` (235→50 µs),
  q_proj/v_proj `k_linear_reduce` (now warp tier), and the **gated-MLP** `k_linear_mean_reduce` all deploy
  `mma.sync` now; the gated-MLP got there via the symbolic-row structural split (`005_split_demoted` offered at
  `mul_16` → `xn`/`mm0`/`mm1`/SiLU-combine) and **wins at 68 µs / 1.74x eager**. The cooperative-reduce q/k-norm
  kernels and the input RMSNorm also beat or match eager (input RMSNorm 2 µs / 2x tcompile). These are the
  finding-1/2/3 of the pre-M9 report, now closed.
- **Warp-tier matmuls still ~2x cuBLAS (class-3 codegen, low µs):** down-proj 50 µs vs cuBLAS 27 (0.54x), q_proj
  29 vs 16, v_proj 18 vs 10. Right tier, still behind cuBLAS — a codegen-quality gap (epilogue/staging) on tiny
  symbolic-M tiles. Combined ~97 µs; not worth a deep dive while findings 1–2 stand, but an NCU compare + emitted-
  CUDA read on `k_linear_0837e7` is the entry point when matmul codegen is the focus.

## Repro / artifacts

- Work dir: `/tmp/tune-model-qwen3-emb-l0/` — `tune.log`, dump at `dump/` (reproducers under
  `07_lowering_cuda.kernels/`, machine-readable bench `dump/62_kernel_bench.json`, chart `dump/kernels.html`;
  `.png` skipped — Playwright flake again), NCU under `ncu-sdpa/61_ncu_metrics.{csv,json}`.
- Isolated DB / prior (this run only): `/tmp/tune-model-qwen3-emb-l0/autotune.db` (1880 ok rows),
  `/tmp/tune-model-qwen3-emb-l0/prior.json`. The global default prior was **not** touched.
- Re-run the whole pass: the Command line at the top (set the two `DEPLODOCK_*` paths first to keep it isolated).
- Triage one-liners (set `DEPLODOCK_TUNE_DB`/`DEPLODOCK_PRIOR_FILE` to the work-dir paths):
  `deplodock eval variants --kernel k_sdpa_reduce` (3 scalar configs = the tier story),
  `deplodock eval prior --dataset db` (reachability clean), `deplodock eval failures` (0 rows).
- Finding-1 NCU: `DEPLODOCK_DUMP_DIR=<dir>/ncu-sdpa deplodock run --ir
  .../k_sdpa_reduce_77b0f0.torch.json --bench --profile`.
- Finding-2 lockout (no GPU): `deplodock compile .../k_linear_sdpa_reduce_43208b.torch.json --ir cuda` (no
  `mma.sync`); the gate is `_atom.py:134`. With GPU: `... run --ir ... --bench --ab "MMA=1"` (stays scalar).

## Workflow notes

Audit of the pre-M9 report's notes (most got fixed — checking whether the fixes held):

- **Full-model dynamic table shape-mismatch (its finding 4 / the half-hour-of-triage bug): FIXED and held.** The
  table now prints `benched at seq_len=512 (symbolic hint; torch inputs tiled to match)` and the eager row reads a
  sane 248 µs (the static-512 ballpark), not the old seq-32 96 µs. No wasted triage this run — the headline 0.15x
  was trustworthy from the first table. `_hint_sized_inputs` did its job.
- **`-O1` pick-miss false alarm: the `-O3 us` column resolved it in one view** (finding 3) — no `run --ab` round-
  trip needed to refute the rank-11 flag. The column added in a prior report paid off exactly as intended.
- **Default-prior pollution (its finding 5): avoided by construction** — isolating both `DEPLODOCK_TUNE_DB` and
  `DEPLODOCK_PRIOR_FILE` to the work dir means this scalar-heavy symbolic tune never reached the global prior. The
  skill should make this the default for findings runs, not an afterthought (it currently says `--clean` nukes the
  *default* paths).
- **Reproducer re-fusion: reproduced** — `k_sdpa_reduce_77b0f0` re-fuses into `6e4bd6` + `042770` under `run --ir`,
  so the NCU table shows two kernels for one deployed kernel. Harmless here (their Σ = the deployed 1052 µs) but
  still a footgun; the per-kernel reproducer total stayed the stable signal as predicted.
- **Chart PNG Playwright flake: reproduced verbatim** (`png skipped: Target page … closed`). The HTML chart is
  fine; the PNG step has been flaky across every findings run — worth either fixing the headless launch or dropping
  the PNG attempt and only emitting HTML.

New friction this run:

- **The dynamic tune is now 3x slower than pre-M9 (2300 s vs 730 s)** — exactly *because* M9 works: the warp tier
  now enumerates and the structural split offers, so the dynamic tree no longer collapses to one cheap terminal. A
  findings tune of one layer is now a ~40-minute commitment, not ~12. Not a bug, but the skill's "~10–20 min"
  estimate is stale for post-M9 single-layer dynamic tunes — bump it.
- **No progress signal under `tee`/non-tty for 38 minutes.** The live progress bar only renders on a tty, so the
  tee'd log and the background-task output were silent from "model loaded" to "done" — I had to poll the dump dir
  for kernel files to know it was alive. A periodic non-tty heartbeat line (e.g. one INFO line per completed op
  leaf, gated on `not isatty`) would make long tunes observable without `-v`'s full firehose. This was also raised
  in the pre-M9 report's notes and is **not yet fixed**.
- **Tier is not a column in any `eval`/bench view.** I inferred each kernel's tier by `grep -c mma.sync` over the
  emitted CUDA — the single most-used signal of this whole analysis. A `tier` column (scalar / warp / coop-reduce /
  pointwise) in the per-kernel bench table and in `eval variants` headers would replace that manual grep and make
  "which kernels reached the warp tier" a one-glance read. Strongly recommend.
- **`vs tcompile` is computed but not printed in the tune's per-kernel table** (it prints `vs eager`). For a
  findings report tcompile is the more telling baseline (it's the optimizing-compiler bar, eager is the floor). I
  reconstructed `vs tcompile` by hand from the eager/tcompile/deplodock columns. Add the column.
