# Qwen3-Embedding-0.6B layer 0 (serving shape, dynamic seq_len) — tune findings

**Status:** clean dynamic tune complete and -O3 deployable-benched. The serving-shape kernels are **5–8× slower than
eager / cuBLAS on every matmul**; root-caused to a **total tensor-core (mma.sync) lockout** with **two independent,
empirically-confirmed blockers**. Reductions/pointwise kernels are at or ahead of eager. This is the per-kernel
attribution behind the serving A/B in [`qwen3-embedding-serving-perf-findings.md`](qwen3-embedding-serving-perf-findings.md)
(deplodock plugin 24×→103× slower per request than stock vLLM).

**Run command** (RTX 4080, sm_89, HEAD `72a65e1a`, fp16 trunk, ncu present; isolated DB/prior/cubin under
`_tune/tune-model-qwen3-emb-serving/`):

```bash
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench \
  --dump-dir _tune/tune-model-qwen3-emb-serving/dump
```

**Date:** 2026-06-17. **Scope:** one layer, **dynamic** (symbolic `seq_len`), **benched at seq_len=512 (symbolic
hint** — `--seq-len` only sizes trace inputs; masked-tile boundary guards `if (coord < seq_len)` are part of the
measured cost). Whole-model not run — the 28 trunk layers are identical, so layer 0 holds every distinct kernel.

**Run stats:** 16 fused terminals, **14 178 s (3.94 h)**, **7 003 perf rows (6 986 ok / 17 bench_fail)**, prior
19 855 training benches, ranking calibration Spearman **+0.99**.

**Number-family disclaimer:** headline latencies are the deployable **-O3** `--bench` re-bench (CUDA-graph captured).
Tune-DB / `eval variants` latencies are **-O1 ranking only** (reduction/attention run 1.5–3× slower at -O1) and are
flagged as such where quoted.

---

## Bench results

**Layer-0 end-to-end** (eager / torch.compile / deplodock, benched at seq_len=512 symbolic hint):

| Backend | Latency (µs) | vs Eager |
|---|---|---|
| Eager PyTorch | 341 | 1.00× |
| torch.compile | 245 | 1.39× |
| **Deplodock** | **2246** | **0.15×** (6.6× slower than eager, 9.2× slower than tcompile) |

**Per-kernel** (-O3, sorted by deplodock µs; `vs eager` < 1.0 = deplodock slower). Layer-op from each kernel's
`.torch.json` reproducer:

| Kernel | Layer op | eager | tcompile | deplodock | vs eager | tier |
|---|---|---|---|---|---|---|
| k_linear_mean_reduce | q/k-norm + a projection (matmul+mean) | 171 | 92 | 453 | 0.38× | scalar |
| k_linear_sdpa_reduce | QKᵀ→softmax fused (matmul + sdpa) | 55 | 53 | 422 | 0.13× | scalar |
| k_sdpa_linear_reduce | P@V + o_proj (sdpa + matmul) | 38 | 38 | 316 | 0.12× | scalar |
| k_linear | MLP `down_proj` (`x@Wᵀ`) + residual | 44 | 44 | 237 | 0.19× | scalar |
| k_sdpa_reduce | attention softmax/reduce | 200 | 27 | 209 | 0.96× | scalar |
| k_linear_reduce | a projection (matmul + reduce) | 30 | 30 | 182 | 0.17× | scalar |
| k_mean_linear_reduce | norm + projection | 187 | 34 | 168 | 1.11× | scalar |
| k_mean_linear_reduce | norm + projection | 100 | 20 | 97 | 1.03× | scalar |
| k_linear_reduce | a projection (matmul + reduce) | 15 | 15 | 94 | 0.16× | scalar |
| k_mean | RMSNorm mean | 81 | 5 | 2 | 37.6× | scalar |

**Dominating kernels:** the top 6 (all `k_linear*` / `k_sdpa*` matmul/attention) are **1819 µs ≈ 83%** of the
deplodock total. Every one loses 5–8× to *both* eager and torch.compile. The memory-bound reductions (`k_mean`,
both `k_mean_linear_reduce`) match or beat eager — deplodock's scalar codegen is fine where tensor cores don't
matter; the loss is entirely in the matmul/attention kernels.

**Tier:** **0 of 7003 benched variants carry `MMA=1` / `WARPSPEC` / `TMA`** (verified by direct DB scan). The whole
dynamic-seq layer tuned in **scalar FMA tier**. cuBLAS (eager) and torch.compile use tensor cores for these
matmuls; deplodock cannot — the entire gap.

---

## Finding 1 — total tensor-core lockout, two independent blockers (≈83% of layer time)

**Symptom.** Every matmul/attention kernel is scalar and 5–8× off cuBLAS; no warp-tier variant exists anywhere in
the tune DB.

**Blocker A — sm_89 auto-enum gate (correct-by-design).** The mma.sync (s16816) family **only auto-enumerates on
sm_90+**: [`010_partition_loops.py:727-732`](deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py#L727)

```python
if not pin_is_mma_sync and ctx.compute_capability < (9, 0):
    # on sm_80-89 it's pin-only, so drop every atom.
    eligible = ()
```

The swizzled-TMA fast path it relies on needs Hopper; on sm_80–89 mma.sync falls back to slower cp.async staging,
so it's deliberately **pin-only** (`DEPLODOCK_MMA=<kind>`). RTX 4080 is sm_89 → `(8,9) < (9,0)` → dropped.
**Confirmed:** pinning `DEPLODOCK_MMA=mma_m16n8k16_f16` on a clean **NN** matmul (`a@b`, 512×3072×1024 fp16) *did*
enumerate a warp tile — and it **hung** (`k_matmul_… did not complete within 1000 ms — bench_fail`), exactly the
cp.async-fallback instability the gate guards against. So even with the pin, tensor cores are not deployable on this
Ada card.

**Blocker B — transposed-B (`x@Wᵀ`) layout is mma-ineligible on *every* GPU (the dominant one).** Every `nn.Linear`
(q/k/v/o + gate/up/down projections — all the transformer's matmul content) computes `x@Wᵀ` with weight stored
`[N,K]`, so **both operands carry the contraction K in their last dim**. `classify_matmul_operands`
([`_atom.py:42`](deplodock/compiler/pipeline/passes/lowering/tile/_atom.py#L42)) cannot tag A vs B for that layout
and returns `None`; `is_atom_eligible` mirrors the tagger, so the cell is **never offered the mma tier** — it falls
to scalar on sm_90 too. **Confirmed:** pinning `DEPLODOCK_MMA=mma_m16n8k16_f16` on an **NT** linear (same
512×3072×1024 shape via `F.linear`) deployed **scalar** (`k_linear_8c6ce2`, no `WM`/`WN`, 236 µs) — the pin had
zero effect, in contrast to the NN matmul above which reached the warp tier. cuBLAS handles NT natively on tensor
cores (it's the preferred GEMM layout); deplodock's tagger does not.

**Net:** Blocker B alone keeps every projection scalar on all hardware; Blocker A additionally keeps the few NN
matmuls (QKᵀ, P@V after a split) scalar on this Ada card.

**Fix priority (high).** Two orthogonal levers, in impact order:
1. **Make NT (`x@Wᵀ`) matmuls mma-eligible** — teach `classify_matmul_operands` / the tagger to recover A/B for
   transposed-B (K-in-last on both), or have the frontend present linears as NN by transposing the weight constant
   at trace/contiguize time. This unlocks tensor cores for the projections on **all** GPUs and is the bulk of the
   83%. Biggest single win.
2. **Stabilize the sm_80–89 mma.sync path** (cp.async-staged, no Hopper TMA) so it can auto-enumerate on Ada
   instead of pin-only — but the pinned NN matmul *hung* here, so this needs the hang root-caused first; lower
   priority than (1), and Ada-specific.

---

## Finding 2 — SDPA/attention kernels: scalar even on sm_90, plus symbolic-K bail

**Symptom.** `k_linear_sdpa_reduce` (422 µs, 0.13×), `k_sdpa_linear_reduce` (316 µs, 0.12×), `k_sdpa_reduce`
(209 µs; ~par eager but **7.7× slower than tcompile's 27 µs**).

**Root cause.** Two compounding, documented gates:
- **Fused softmax prologue blocks MMA.** The warp tier requires `not prologue`
  ([`010_partition_loops.py:711`](deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py#L711)) —
  an SDPA prologue's softmax stats can't share the zero-filled masked-K slab. So even on sm_90 these stay scalar
  unless the demoted-matmul split (`005_split_demoted`) un-fuses them into a clean consumer.
- **Symbolic-K P@V bails to scalar thread tiles.** `K = seq_len` is symbolic; the warp tier needs a static reduce,
  so P@V deploys masked thread tiles at `FM=FN=1`
  ([`010_partition_loops.py:606-613`](deplodock/compiler/pipeline/passes/lowering/tile/010_partition_loops.py#L606)).
  Flash-style symbolic-K attention is **future work** (CLAUDE.md / `serving/ARCHITECTURE.md`).

The split *is* offered (the leaderboards show `_xn` / `_mm1` / `_xnb` consumer terminals), but every split consumer
is **also scalar** — because its clean matmul is still either NT (Blocker B) or NN-on-sm_89 (Blocker A). The split
doesn't help here until Finding 1 is fixed.

**Fix priority (medium, gated on Finding 1).** Flash-style symbolic-K warp attention is the real fix and is already
recorded as future work; it only pays off once Finding 1's blockers are lifted (otherwise the consumer stays
scalar).

---

## Finding 3 — search/prior is NOT the problem (the -O1 vs -O3 inversion is handled)

**Symptom (apparent).** `eval variants` flags several picks as far from rank 1, e.g. `k_linear_mean_reduce_6a97e0`
`pick: rank 212/429 … misses best`, `k_linear_sdpa_reduce_158a0f` `rank 126/495`.

**Not a shortfall.** Those ranks are **-O1 ranking**; the prior re-benched at -O3 and picked the **-O3 winner**: the
rank-212 pick's `-O3 us` is **240.1** vs the -O1 rank-1's `-O3 us` of **591.9** (and rank-126's pick is 199.7 -O3
vs rank-1's 436.9). So the prior correctly recovers the deployable-best config despite the -O1 ranking inversion —
exactly what the `H_opt=3` reservoir is for. Spearman +0.99 overall. **No action**: the ceiling here is the tier
(Finding 1), not the search. Tuning harder cannot beat a scalar kernel into tensor-core territory.

---

## Finding 4 — class-4 bench failures: 17 oversized scalar tiles blow the nvcc compile budget (minor)

**Symptom.** All 17 `bench_fail` rows are **one kernel, `k_linear_fc9a1d`** (the MLP `down_proj`), every one
`RuntimeError: compile stage exceeded 2.0s budget (2.0–6.0 s)`. Shared knobs across the cluster: `MMA=0`, big
unrolled scalar tiles (`FM` up to 26, `FK=64`, `STAGE=11`, `SPLITK=2`) — `eval failures` clusters them cleanly.

**Root cause.** Large unrolled scalar register-tile variants hit the cicc/LLVM compile blowup the tune's `-O1`
budget exists to dodge; the biggest tiles exceed even the 2 s ceiling and are pinned `bench_fail`. Only 0.3% of
benches, and the kernel's final pick is fine — but it's a *symptom* of the scalar tier reaching for huge tiles to
compensate for no tensor cores. Lifting Finding 1 removes the need for these tiles.

**Fix priority (low).** Cosmetic until Finding 1 lands; the failures waste a few search slots but don't affect the
deployed pick.

---

## Repro / artifacts

- Tune log: `_tune/tune-model-qwen3-emb-serving/tune.log`; dump: `_tune/tune-model-qwen3-emb-serving/dump/`
  (per-kernel `.torch.json` reproducers under `dump/07_lowering_cuda.kernels/`, `kernels.html`,
  `62_kernel_bench.json`). Isolated DB/prior/cubin in the same dir (your global caches untouched).
- Tier-lockout confirmation (no GPU needed for the DB scan; `sqlite3` CLI isn't installed, use the venv):
  ```bash
  # 0 of 7003 perf rows name a warp-tier mma atom → fully scalar tune
  ./venv/bin/python -c "import sqlite3; c=sqlite3.connect('_tune/tune-model-qwen3-emb-serving/autotune.db'); \
  print('warp:', c.execute(\"select count(*) from perf where knobs like '%mma_m16n8k16%'\").fetchone()[0], \
  'of', c.execute('select count(*) from perf').fetchone()[0])"
  ```
- Blocker A (sm_89 pin → cp.async fallback hangs) vs Blocker B (NT linear pin → still scalar):
  ```bash
  REPRO=_tune/tune-model-qwen3-emb-serving/dump/07_lowering_cuda.kernels/k_linear_fc9a1d.torch.json
  DEPLODOCK_MMA=mma_m16n8k16_f16 deplodock run \
    --code "torch.matmul(torch.randn(512,3072,dtype=torch.float16,device='cuda'),torch.randn(3072,1024,dtype=torch.float16,device='cuda'))" --bench  # NN → warp tile, HANGS on sm_89
  DEPLODOCK_MMA=mma_m16n8k16_f16 deplodock run \
    --code "torch.nn.functional.linear(torch.randn(512,3072,dtype=torch.float16,device='cuda'),torch.randn(1024,3072,dtype=torch.float16,device='cuda'))" --bench  # NT → still scalar, pin no-op
  ```
- Per-kernel leaderboards: `deplodock eval variants --kernel k_linear_mean_reduce` (etc.); failures:
  `deplodock eval failures`. (Set the three `DEPLODOCK_*` cache env vars to the work dir first.)

---

## Workflow notes

For whoever maintains the deplodock CLI and this skill:

- **`--ab "MMA=…"` silently no-ops when the tier wasn't enumerated.** On sm_89 the warp tier is never enumerated,
  so `run --ir … --ab "MMA=mma_m16n8k16_f16"` returned a row byte-identical to the scalar pick — no error, no note.
  The pin only takes via the **env** (`DEPLODOCK_MMA`), because the enumeration gate reads `mma_mode()` (global
  env), not the per-variant `--ab` dict. *Proposed:* when an `--ab` knob names a tier/atom absent from the
  enumerated set, print a warning (`ab MMA=… had no effect: warp tier not enumerated for this kernel/arch`) instead
  of silently collapsing to scalar. Cost me two runs to notice.
- **"All scalar" is invisible in `eval variants`.** The view drops constant knob columns, so a 100%-scalar kernel
  shows no `MMA`/`WM`/`WN` column at all — indistinguishable from "warp columns omitted." I had to scan the DB by
  hand to confirm 0 MMA rows. *Proposed:* a one-line tier banner per kernel, e.g.
  `tier: scalar-only (no MMA enumerated — sm_89 pin-only)` or `tier: 12 warp / 417 scalar`.
- **Pinning mma.sync on sm < 90 can hang, not just slow down.** The NN-matmul pin produced a 1 s-timeout
  `bench_fail` (cp.async fallback). *Proposed:* a guard/warning at pin time when `cc < (9,0)` and the atom is
  mma.sync (`DEPLODOCK_MMA on sm_89 uses the cp.async fallback — may hang; intended for bring-up only`).
- **3.94 h for one dynamic layer (7 003 perf rows).** Long for an attribution pass whose conclusion is "tier is
  locked, search is fine." A `--no-o3-rebench` / ranking-only fast mode (skip the -O3 reservoir re-bench) would cut
  wall-clock substantially when the goal is structural attribution rather than deployable numbers — the -O3 numbers
  didn't change any finding here (the tier ceiling did).
- **No "why was MMA declined for this kernel?" introspection.** Root-causing the lockout meant reading three passes
  (`010_partition_loops`, `_atom`, `_enumeration`) and running two synthetic `--code` A/Bs. *Proposed:* a
  `deplodock eval eligibility --kernel <k>` (or a `--explain-tier` flag on `compile`) that prints, per matmul cell,
  the `is_atom_eligible` verdict + the first failing predicate (`transposed-B: classify_matmul_operands → None`,
  `cc 8.9 < 9.0 (pin-only)`, `K=… not divisible`, …). That single view would have replaced this entire
  investigation.
