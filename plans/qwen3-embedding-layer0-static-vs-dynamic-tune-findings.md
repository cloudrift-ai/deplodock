# Qwen3-Embedding-0.6B layer 0 — STATIC vs DYNAMIC side-by-side tune findings (2026-06-16, main @ 513a5df5 / #244)

Status: **clean per-layer tune of Qwen3-Embedding-0.6B layer 0 in two configurations — shape-specialised STATIC (no
`--dynamic`) and the deployable DYNAMIC (`--dynamic seq_len@x:1`, symbolic-`seq_len` masked tiles) — benched
side-by-side per kernel and end-to-end at a matched seq 512.** Headline: at seq 512 the static config runs **184 µs /
1.21x eager / 0.81x torch.compile**, the dynamic config **340 µs / 0.66x eager / 0.44x torch.compile** — static is
**~1.85x faster** than the deployable dynamic artifact. The gap is the masked-K attention split (standalone softmax +
conflict-heavy masked-K-mma P@V, ~50 µs of locked-clock NCU time over static's fused SDPA) plus the masked-tile
boundary-guard overhead on every symbolic-`seq_len` reduce/MLP kernel. **Neither config reaches PyTorch's fused flash
attention** (deplodock attention 125 µs static / 176 µs dynamic vs flash ~37 µs) — attention is the shared bottleneck.
Two process findings dominate the run: a **reproducible #244 regression that wedges the dynamic tune** (a TMA
cooperative-reduce variant hangs the GPU and the tuner parent never recovers), and a **~2x thermal inflation** of the
static@512 tune-`--bench` e2e (363 µs hot vs 184 µs cooled).

- Commands (each in its own isolated DB / prior / cubin cache under the work dir):
  - Static@512: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512 --clean --bench --dump-dir
    static512-dump` → cooled `deplodock run … --seq-len 512 --bench`
  - Static@32 (secondary, trace default): `deplodock tune … --layer 0 --clean --bench` → cooled `run … --bench`
  - Dynamic@512: tune `deplodock tune … --layer 0 --dynamic seq_len@x:1 --clean --bench` **wedged** (see Finding 2);
    the deployable numbers come from a fresh `deplodock run … --dynamic seq_len@x:1 --bench` against the **default
    fully-tuned dynamic prior** (`~/.cache/deplodock/prior.json`, the prior the previous dynamic report trained at
    `3abe9959`; `run` reads the prior, never writes it).
- Hardware: RTX 5090 (sm_120), ncu 2025.3.1 (perf counters permitted). `main` @ `513a5df5` (#244 "Enable TMA on the
  dynamic path + per-variant tune containment").
- Run stats:
  - Static@32 tune **7,939 s** (~2.2 h), 16 fused terminals, 12,838 benches, post-warmup Spearman +0.99, **3,829 ok /
    0 bench_fail**.
  - Static@512 tune **12,812 s** (~3.6 h), 16 fused terminals, 16,457 benches, Spearman +0.99, **5,437 ok / 7
    bench_fail** (all benign compile-time — Finding 6).
  - Dynamic@512 tune **wedged** after ~1,678 ok / 2 bench_fail (Finding 2); deployed numbers from the default prior.
- **Number families**: every e2e / per-kernel / `--ab` number below is **-O3** (deployable, CUDA-graph captured) unless
  tagged `-O1` (tune-DB ranking signal — reduction/attention kernels run 1.5–3x slower at -O1, so -O1 latencies rank,
  they don't deploy). NCU durations are at **locked base clock** — compare ratios, not absolutes, and NCU is the only
  clean per-kernel attribution (the deployed launch table mis-attributes attention — Finding 4).
- **Dynamic measurement semantics**: the dynamic trace is symbolic in `seq_len` (`Dim` hint `DEFAULT_SEQ_HINT` = 512 —
  `--seq-len` does NOT move it). All dynamic tune/bench/reproducer measurements run at the 512 hint; the full-model
  table tiles the torch closures' inputs to the hint and prints `benched at seq_len=512 (symbolic hint; torch inputs
  tiled to match)`, so its eager / deplodock rows are one shape. The masked-tile boundary guards (`if (coord <
  seq_len)`) are part of the measured dynamic cost — the overhead vs a shape-specialised kernel is itself a finding,
  not noise.
- **Scope note (single layer, no servable artifact)**: this is one transformer layer, not the whole model — there is
  no `deplodock serve` A/B here (Step 2b is whole-model-only). The static config is shape-specialised (no masked-tile
  guards) — **not** the deployable configuration; the dynamic config is what ships. The static side is included
  precisely to price what the deployable masked-tile lowering costs.

## Bench results (-O3, CUDA-graph captured, cooled fresh runs)

End-to-end, matched seq 512 (and static@32 for the seq-scaling point):

| Config        |  seq |  Eager (us) |  torch.compile |  Deplodock (us) |  vs eager |  vs tcompile | source                                     |
|---------------|-----:|------------:|---------------:|----------------:|----------:|-------------:|--------------------------------------------|
| Static        |   32 |          98 |       (45 hot) |              43 |     2.27x |        ~1.0x | `static-run.log` (e2e 43.1)                |
| **Static**    |  512 |         222 |      (149 hot) |         **184** | **1.21x** |    **0.81x** | `static512-run.log` (e2e 184.3)            |
| **Dynamic**   |  512 |         225 |      (149 hot) |         **340** | **0.66x** |    **0.44x** | `dynamic-run-defaultprior.log` (e2e 340.3) |

(torch.compile columns are from the tune-`--bench` tables, run on a thermally-loaded GPU — quoted only for context;
the cooled fresh `run --bench` reports eager + deplodock. The static@512 tune-`--bench` reported Deplodock **363 µs**;
the cooled fresh run reports **184 µs** — a ~2x thermal spread, Finding 5. Trust 184.)

Per-kernel **deployed launch tables** (cooled fresh `run --bench`, sorted by deployed µs; layer-op labels from each
kernel's `.torch.json` provenance). **These tables mis-attribute the attention kernels (Finding 4) — the clean
per-kernel signal is the NCU tables below.** Knob columns elided.

Static@512 (TOTAL 181.6, e2e 184.3):

| Kernel                           | Layer op                                   |      dep us |     % |
|----------------------------------|--------------------------------------------|------------:|------:|
| `k_sdpa_linear_reduce_3d2635_xn` | SDPA P@V producer (softmax)                |        25.4 | 14.0% |
| `k_linear_mean_reduce_f1b55d`    | post-attn RMSNorm + MLP gate/up + SiLU     |        22.6 | 12.5% |
| `k_mean_linear_reduce_1f1bec`    | k_norm RMSNorm + rotated-k                 |        22.3 | 12.3% |
| `k_mean_linear_reduce_125c9c`    | q_norm RMSNorm + rotated-q                 |        21.3 | 11.7% |
| `k_mean_5c2ff9`                  | input RMSNorm                              |        20.6 | 11.3% |
| `k_linear_mean_reduce_940035_xn` | post-attn RMSNorm producer                 |        15.9 |  8.7% |
| `k_linear_368d28`                | MLP down (linear_6) + residual             |        15.1 |  8.3% |
| `k_linear_reduce_*` (q/v proj)   | q_proj / v_proj matmul                     | 2.1/8.5/8.5 |       |
| `k_sdpa_linear_reduce_3d2635`    | SDPA P@V mma consumer                      |         5.4 |  3.0% |
| `k_linear_sdpa_reduce_94ab75*`   | attn-out + o_proj + residual               |     8.4/1.3 |       |
| `k_sdpa_reduce_82f310`           | RoPE + QK^T scores                         |         2.1 |  1.2% |

Dynamic@512 (TOTAL 336.6, e2e 340.3):

| Kernel                            | Layer op                                   |  dep us |     % |
|-----------------------------------|--------------------------------------------|--------:|------:|
| `k_sdpa_linear_reduce_a76a28`     | SDPA P@V masked-K mma consumer             |   102.8 | 30.5% |
| `k_linear_mean_reduce_05d34c`     | post-attn RMSNorm + MLP gate/up + SiLU     |    35.7 | 10.6% |
| `k_sdpa_linear_reduce_a76a28_xnb` | P@V producer (V contiguify)                |    30.8 |  9.1% |
| `k_mean_linear_reduce_5ceb87`     | k_norm RMSNorm + rotated-k                 |    27.9 |  8.3% |
| `k_mean_linear_reduce_67f4cf`     | q_norm RMSNorm + rotated-q                 |    26.0 |  7.7% |
| `k_linear_reduce_716194` (q/v)    | q_proj / v_proj matmul                     |    22.5 |  6.7% |
| `k_linear_sdpa_reduce_43208b_xn`  | attn-out contiguify                        |    21.5 |  6.4% |
| `k_sdpa_linear_reduce_a76a28_xna` | **scalar softmax producer** (0% occ)       |    20.3 |  6.0% |
| `k_linear_mean_reduce_05d34c_xn`  | post-attn RMSNorm producer                 |    18.4 |  5.5% |
| `k_linear_0837e7`                 | MLP down (linear_6) + residual             |    10.4 |  3.1% |

The dynamic launch table fingers the P@V consumer at 30.5% — but its solo window absorbs the softmax/QK^T latency
(Finding 4); NCU clocks the P@V at 38 µs, not 103 µs. Trust the NCU attribution below.

## Bench results — NCU compare (locked base clock, the clean per-kernel signal)

Both captured from the o_proj-chain reproducer (`run --ir …k_linear_sdpa_reduce_*.torch.json --bench --profile`),
which re-fuses the whole attention→o_proj cone, so deplodock + the torch flash/cutlass references sit in one capture.

**Dynamic@512 attention path** (`ncu-dyn.log`):

```
side  kernel                                  dur (ns)  occ%   sm%  dram%  fma%   lsu.inst  ld.cnflct  st.cnflct  regs
dep   k_sdpa_reduce (QK^T scores, scalar)        70,592  70.8  58.5   10.8  40.8  6,111,232      3,872     28,042    47
dep   k_sdpa_reduce_22a7a0 (P@V masked-K mma)    38,208  26.8  14.7   19.9   2.9    935,936  3,670,148        176    70
dep   k_linear_sdpa_reduce_43208b (o_proj mma)   34,432  15.6  19.2   17.3   0.8    681,216          0          0   125
dep   k_sdpa_reduce_22a7a0_xn (softmax producer) 22,976  37.8  33.3   46.4  17.8  1,101,824          0          0    80
dep   k_linear_sdpa_reduce_43208b_xn (contig)     9,728  31.7   2.6   22.9   3.5     40,960          0          0    24
ref   flash_fwd_splitkv_kernel                   25,088   8.3  16.2   23.1   2.9    300,800          0          0   206
ref   cutlass_80_tensorop_f16_s16816gemm(o_proj) 23,168   8.3  29.1   18.0   0.2    569,856          0          0    88
ref   flash_fwd_splitkv_combine_kernel           12,416  65.3  30.9   50.9   8.5     88,064          4          0    44
```

**Static@512 attention path** (`ncu-static512.log`):

```
side  kernel                                  dur (ns)  occ%   sm%  dram%  fma%  lsu.inst  ld.cnflct  st.cnflct  regs
dep   k_sdpa_reduce_c0c97e (fused scores+sm)     58,976  28.3  34.6    5.5  40.5   921,600  7,378,831          0    97
dep   k_linear_sdpa_reduce_94ab75 (o_proj mma)   26,880  18.5  24.4   18.1   4.0   550,016          0          0    72
dep   k_sdpa_reduce_ba65cb (P@V)                 15,808  26.6  20.6   39.9   0.8   346,368          0          0    89
dep   k_linear_sdpa_reduce_94ab75_xn (contig)    12,608  39.9   3.3   24.9   1.3    69,632          0          0    26
dep   k_sdpa_reduce_ba65cb_xn (softmax producer) 11,168  83.4  49.6   72.2  17.5   851,968        585        460    33
ref   flash_fwd_splitkv_kernel                   25,376   8.3  16.3   18.0   3.0   300,800          0          0   206
ref   cutlass_80_tensorop_f16_s16816gemm(o_proj) 22,816   8.3  29.2   17.8   0.2   569,856          0          0    88
ref   flash_fwd_splitkv_combine_kernel           12,160  58.7  31.8   48.4   7.7    88,064          0          0    44
```

By NCU locked-clock the **attention path is ~125 µs static vs ~176 µs dynamic** (QK^T 70.6 + P@V 38.2 + softmax 23.0 +
o_proj 34.4 + contig 9.7) — a ~50 µs split tax. Both are ~3–5x PyTorch's fused flash (`flash_fwd` 25 µs +
`flash_combine` 12 µs ≈ 37 µs, plus the 23 µs cutlass o_proj gemm). The dominating kernels (to ~80% of the deplodock
e2e) are: the attention chain (static 125 µs / dynamic 176 µs) + the three cooperative-reduce RMSNorm kernels
(input/q-norm/k-norm, ~20 µs each deployed) + the post-attn RMSNorm+MLP (`k_linear_mean_reduce`, static 22.6 /
dynamic 35.7 µs). The matmul-only linears (q/v/o proj, MLP-down) are at parity-to-2x of eager in both and get one line
each.

## Finding 1 — at seq 512 the masked-tile dynamic config costs ~1.85x the shape-specialised static config (184→340 µs), split between the masked-K attention demotion and per-kernel boundary-guard overhead

**Symptom.** Cooled fresh e2e: static@512 **184 µs / 1.21x eager**, dynamic@512 **340 µs / 0.66x eager** — a 156 µs
deployable gap on the same layer at the same seq. At seq 32 the same static config is **43 µs / 2.27x eager** (attention
is O(seq²), so its share is small there); the static win shrinks 2.27x→1.21x as seq grows because attention — the part
static does NOT speed up much — dominates at 512.

**Evidence — where the 156 µs goes.**
- **The masked-K attention split (~50 µs, NCU).** Dynamic's symbolic-K SDPA un-fuses (via `005_split_demoted` on
  symbolic K) into a **standalone scalar softmax producer** (`…_22a7a0_xn`, NCU 23.0 µs) + a **masked-K-mma P@V
  consumer** (`…_22a7a0`, NCU 38.2 µs with 3.67M smem load bank-conflicts), on top of a scalar QK^T (70.6 µs). Static
  keeps SDPA fused: its scores+softmax is one 59 µs kernel and its P@V is a 15.8 µs kernel — no materialised softmax,
  no masked-K-mma conflict storm. NCU attention totals: static ~125 µs vs dynamic ~176 µs.
- **Masked-tile boundary-guard overhead (~the remainder).** Every symbolic-`seq_len` kernel is a ceil-div grid + an
  `if (coord < seq_len)` guard sized for the 512 hint. The deployed reduce/MLP kernels run materially slower in
  dynamic: q/k-norm `k_mean_linear_reduce` 26.0/27.9 µs (dynamic) vs 21.3/22.3 µs (static); post-attn
  `k_linear_mean_reduce` 35.7 µs (dynamic) vs 22.6 µs (static); q_proj `k_linear_reduce` 22.5 µs (dynamic) vs 8.5 µs
  (static). These are the same ops with the same reduce extents (hidden 1024, seq-independent) — the delta is the
  masked-tile guard + the symbolic-axis runtime-arg handling, which the specialised static kernel doesn't pay.

**Root cause.** Two structural, by-design costs of the deployable masked-tile lowering: (a) symbolic K forces the SDPA
demotion split (`tile-level split, not a fusion guard` — the kernel-set decision lives in `005_split_demoted`), which
materialises the softmax and routes P@V through the conflict-heavy masked-K-mma slab; (b) every symbolic free axis is a
guarded masked tile. Neither is a defect — it is what "one kernel runs at any seq_len" buys. The finding prices it:
**~1.85x at seq 512.**

**Repro.** Static: `deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512 --bench` (work-dir prior). Dynamic:
`DEPLODOCK_PRIOR_FILE=~/.cache/deplodock/prior.json deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic
seq_len@x:1 --bench`. NCU side-by-side: `run --ir <dump>/*lowering_cuda.kernels/k_linear_sdpa_reduce_*.torch.json
--bench --profile` (both dumps).

**Suggested fix (large, known).** Flash-style symbolic-seq attention (the future-work item CLAUDE.md already names):
a scheduled online-softmax warp loop over the symbolic N axis (QK^T sub-tile → running max/sum rescale → P@V mma
accumulate) so the softmax never materialises and P@V never needs the masked-K slab — this closes most of the static↔
dynamic attention gap *and* most of the gap to flash for both configs.

## Finding 2 — #244 regression: a TMA cooperative-reduce variant hangs the GPU during the dynamic tune and wedges the tuner parent (the dynamic tune cannot complete clean on this commit)

**Symptom.** The clean dynamic tune (`tune … --dynamic seq_len@x:1 --clean`) **wedged twice** (original run + a
no-`--clean` resume): after benching ~1,678 ok variants the tuner process went to `state=S`, **0 CPU ticks / 20 s**, GPU
idle, no bench-worker / nvcc children, log frozen ~50 min — a hard hang, not slow progress. The last log line both times
was a handled `HungKernelError` on `k_linear_mean_reduce_63573a`. The reference dynamic tune at `3abe9959` (one commit
earlier) had **0 bench_fail** and completed in 4.4 h, so this is a **#244 regression** (#244 = "Enable TMA on the
dynamic path").

**Evidence — the hang config (`eval failures`, `dynamic.db`):**

```
k_linear_mean_reduce_63573a — 2 row(s)
  error: HungKernelError("kernel 'k_linear_mean_reduce_63573a' did not complete within 1000 ms")
  shared knobs: …, FM=1, FN=12, MMA=0, RING=2, TMA=True, PIPELINE_STAGES=True, ASYNC_COPY=False, …
```

The hang is **TMA=True on a scalar (MMA=0) cooperative-reduce fusion** — exactly the kernel #244's TMA-on-dynamic
enablement newly reaches.

**Root cause (confirmed by runtime, FIXED on this branch).** Not the mbarrier re-entry the `_reenters_pipeline` guard
covers — at `FM=1` the K pipeline runs exactly once. Running the variant under the watchdog surfaces the real fault:
`CUDA_ERROR_MISALIGNED_ADDRESS` on `cp.async.bulk.tensor` (then the in-flight TMA never completes → `mbarrier.wait`
spins → 1 s `HungKernelError`). The scalar reduce stages its fp16 input in a `BK=32` slab whose **per-slot box is a
single 32-elem axis = 64 B**, *not* a 128 B multiple. The 128 B ring-slot alignment check — both the eligibility gate
(`050_use_tma._source_eligible`) and the materializer's slot pad (`100_materialize_tile`) — sized its threshold off the
**fp32 `BYTES_PER_ELEM` constant** (`128 // 4 = 32` elems), so the 64 B fp16 slab read as already-128 B-aligned
(`32 % 32 == 0`), stayed unpadded, and the second ring slot (`RING=2`) landed at a 64 B offset from the 128 B base →
misaligned TMA store. A matmul slab avoids this because its box collapses `BK·BN·FN` into a ≥128 B footprint; only the
pure-reduction single-axis box is sub-128 B. (The earlier `FM=12`/`_reenters_pipeline` hypothesis was wrong — the
emitted CUDA shows the K pipeline runs once and the fault is a misaligned address, not stale parity.)

**Fix (landed).** `050_use_tma` now sizes the ring-slot 128 B check off the **true element width** for a
double-buffered NONE-swizzle bundle (`strict_slot_align`): the sub-128 B fp16 reduction slab is declined to cp.async
(which has no 128 B requirement and was already the correct fallback), while matmul boxes (≥128 B), swizzled/mma slabs
(aligned via their swizzle atom), and single-slot bundles are untouched. Regression-locked by three tests:
`test_tma_smem_alignment.py::test_fp16_subaligned_ring_slot_declines_tma_fp32_keeps_it` (compile-only — fp16 declines,
fp32 keeps TMA) and `test_knob_pinning.py::test_norm_linear_fp16_scalar_reduce_tma_alignment` (static **and** dynamic
accuracy through the formerly-wedging config — the parity check). Full compiler suite green (1566 passed).

**It is a tune-only hazard.** The greedy deploy never hit it: only the specific fused `_63573a` (sub-128 B reduction
slab) variant the *search* explores reached the misaligned-TMA path, so `run --dynamic --bench` completed fine (340 µs)
while `tune` wedged on benching that variant.

**Second bug, still open (tuner robustness, high).** A single hung bench variant should never wedge a multi-hour tune.
The `HungKernelError` *was* caught and pinned `bench_fail`, but the next operation hung — a dirty parent CUDA stream
after the worker SIGKILL (CLAUDE.md describes the intended behaviour as a clean saturated-queue abort; here it neither
aborted nor recovered). With the codegen fix above the misaligned variant is no longer emitted, so this run's wedge is
resolved, but the parent should still fence/reset the stream after any worker hang so a future bad variant degrades to
`bench_fail` instead of a wedge. Not addressed on this branch.

**Repro.** `DEPLODOCK_TUNE_DB=<dir>/dynamic.db deplodock eval failures` (the hang config, no GPU). Pre-fix the
misalignment reproduced compile-only via `DEPLODOCK_KNOBS="TMA=1,MMA=0,FN=12,FM=1,BK=32,BN=128,RING=2,SPLIT_CONE=0"
deplodock compile <dump>/…/k_linear_mean_reduce_05d34c.torch.json --ir cuda` (emits `cp.async.bulk.tensor` on a 64 B
slab); post-fix the same pins decline to cp.async. Gate: `050_use_tma.py` `_source_eligible` (`strict_slot_align`).
The deployable dynamic numbers in this report predate the fix and come from the default fully-tuned prior; a fresh
clean dynamic tune is now expected to complete (the misaligned variant is no longer emitted), modulo the still-open
tuner-robustness bug above.

## Finding 3 — attention is the shared bottleneck; #244 cut the dynamic softmax 147→23 µs, so the scalar QK^T (70 µs) is now the dynamic attention dominator — neither config reaches flash

**Symptom.** Both configs' attention is scalar/conflict-heavy and well behind flash. By NCU: dynamic QK^T scores
**70.6 µs** (scalar, 70.8% occ, 40.8% fma, 28K store bank-conflicts), P@V **38.2 µs** (masked-K mma, 3.67M load
conflicts), softmax **23.0 µs**; static fused-scores+softmax **59.0 µs** (28.3% occ, **7.38M load conflicts**, 40.5%
fma), P@V **15.8 µs**. PyTorch fuses the whole thing into flash at ~37 µs.

**Evidence — the #244 softmax win.** The prior dynamic report (`3abe9959`) clocked the standalone softmax producer at
**147 µs** (16-thread, 6% occ, 3-pass materialised). On this commit the same kernel deploys at **23.0 µs** (37.8% occ) —
a ~6x improvement, so the dynamic attention dominator has shifted from the softmax to the **scalar QK^T** (70.6 µs).
The QK^T stays scalar because its accumulator feeds the softmax max/sum reduce over the symbolic N (key) axis — a
mid-reduction use, so the mma fragment-store fold is rejected (`_atom.py` `classify_fragment_epilogue`,
`is_atom_eligible` → scalar tier). The static scores kernel is on the same scalar online-softmax pattern (40.5% fma but
no mma) and is dominated by **7.38M smem load bank-conflicts**.

**Root cause.** Class 2/3 by-design: the online-softmax data dependency (a warp QK^T sub-tile needs the full row before
P@V) blocks a plain masked warp tile — it needs the scheduled flash phase. The static and dynamic attention kernels are
different lowerings of the same un-flashed scalar attention, so both lose to flash.

**Suggested fix.** Same as Finding 1 (flash-style scheduled attention). Until then the QK^T's 28K store-conflicts
(dynamic) and the scores kernel's 7.38M load-conflicts (static) are cheap PAD_SMEM/PERMUTE_LANES targets *within* the
scalar kernels — low priority, superseded by the flash work.

## Finding 4 — the deployed launch table mis-attributes the dynamic P@V by ~2.7x (103 µs window vs 38 µs NCU); only NCU + e2e are trustworthy for attention (recurring)

**Symptom.** The dynamic deployed launch table reads `k_sdpa_linear_reduce_a76a28` (the masked-K-mma P@V) at **102.8 µs
/ 30.5%** — but NCU clocks the same kernel at **38.2 µs**. Its per-launch solo window absorbs the latency of the
softmax + QK^T that feed it, so the `%` column fingers the P@V consumer as the dominator when NCU says QK^T (70.6 µs) is
larger. Symmetrically, the scalar softmax producer reads only 20.3 µs deployed (NCU 23.0 µs — closer here, but the prior
report saw a 7x under-read on the old 147 µs softmax). This is the recurring attribution problem
(`plans/bench-attribution-by-slicing.md`), now the third tune report in a row to hit it.

**Root cause.** Not a kernel defect — a measurement artifact of per-launch solo windows on a fused chain. The clean
signal is the single-capture NCU table (both sides, one clock) and the whole-program e2e.

**Suggested fix (high — recurring tooling gap).** Land per-launch attribution by slicing so the deployed `%` column
matches NCU. Short term, the skill already says trust NCU + e2e for attention.

## Finding 5 — static@512 tune-`--bench` e2e (363 µs) is ~2x the cooled fresh-run e2e (184 µs) — thermal, larger than ever

**Symptom.** The static@512 tune's `--bench` full-model table reported Deplodock **363 µs / 0.62x eager**; an immediate
cooled fresh `run --bench` of the identical deployed kernels reported **184 µs / 1.21x eager** (TOTAL 181.6 ≈ e2e
184.3, internally consistent). A ~2x spread on the same code — far beyond the prior reports' 7%/30%.

**Root cause.** Thermal. The tune's `--bench` ran on a GPU loaded by the 3.6 h tune that just finished; the fresh run is
on a cooled card. The longer the tune, the hotter the card and the bigger the spread — and at 3.6 h this was the longest
yet. Consistent with the memory note that absolute µs drift ±10%+ while ratios stay stable (the cooled fresh
single-shape ratios are the reliable signal). **Trust the cooled fresh-run e2e (184 µs static / 340 µs dynamic).**

**Suggested fix (measurement hygiene, medium).** The tune-`--bench` headline e2e is unusable after a multi-hour tune.
Either clock-lock before the final `--bench`, or have the skill/CLI always take the reported e2e from a separate cooled
`run --bench` (as this report did). A `tune --bench --cooldown N` flag (sleep before the final bench) would make the
tune's own headline deployable.

## Finding 6 — static@512's 7 bench_fail rows are all benign compile-time aborts (contrast: dynamic's one runtime hang wedged the tune)

**Symptom.** `eval failures` on `static512.db`: 7 bench_fail, two clusters — 3× `k_sdpa_linear_reduce_3d2635_xn`
(`compile stage exceeded 2.0s budget (3.7s)` — big-unroll `SPLIT_CONE` producers, `FM=16/8/4`) and 4×
`k_mean_linear_reduce_125c9c` (`nvcc compile failed … warning #177-D variable "warp" declared but not used`, odd
`STAGE=110/101/111`). All are **compile-time**: the variant is cleanly marked bench_fail and the search continues —
the static tune completed normally. This is the intended containment, and the sharp contrast with Finding 2's dynamic
**runtime** HungKernel that wedged the parent: compile-time failures abort cleanly; a runtime device hang does not.

**Root cause / fix (low).** The `STAGE=110/101/111` nvcc failures on the scalar cooperative-reduce are a codegen-quality
nit (an emitted-but-unused `warp` variable tripping a warning-as-error or a downstream parse) — worth a one-line
emit-side fix so those configs become benchable, but they aren't the deployed pick (the kernel's greedy pick is fine)
so the latency impact is nil. The 2 s compile-budget timeouts on `FM=16` producers are expected (the budget exists to
dodge the cicc blowup) — no action.

## Repro / artifacts

- Work dir: `_tune/tune-model-qwen3-l0-staticdyn/` (gitignored, persists). Logs: `static-tune.log` / `static-run.log`
  (seq32), `static512-tune.log` / `static512-run.log`, `dynamic-tune-wedged.log` (the original wedge),
  `dynamic-tune.log` (the wedged resume), `dynamic-run-defaultprior.log` (the deployable 340 µs), `runner*.log`,
  `RESULTS.md` (scratch). NCU: `ncu-dyn.log` + `ncu-dyn/61_ncu_metrics.{csv,json}`, `ncu-static512.log` +
  `ncu-static512/…`. Dumps: `static512-dump/`, `dynamic-dump/` (reproducers under `07_lowering_cuda.kernels/`,
  `kernels.html`; `.png` skipped — Playwright `Target page … closed` flake again).
- Isolated DB/prior/cubin per run (never touched the default 2.7 GB DB / 91 MB prior except read-only for the dynamic
  deploy): `static.db` (seq32), `static512.db` + `static512-prior.json`, `dynamic.db` + `dynamic-prior.json` (partial,
  1,678 rows).
- Copy-paste, no GPU: `DEPLODOCK_TUNE_DB=<dir>/dynamic.db deplodock eval failures` (Finding 2 hang config);
  `DEPLODOCK_TUNE_DB=<dir>/static512.db deplodock eval failures` (Finding 6); `DEPLODOCK_KNOBS="TMA=1" deplodock
  compile <dump>/…/k_linear_mean_reduce_05d34c.torch.json --ir cuda` (Finding 2 clean-decline); gate reads
  `050_use_tma.py:224` (`_reenters_pipeline`) + `:273` (FM/FN docstring), `_atom.py` `classify_fragment_epilogue`
  (Finding 3 scalar-QK^T bail).
- GPU: NCU side-by-side `DEPLODOCK_DUMP_DIR=<dir>/ncu-<cfg> deplodock run --ir
  <dump>/07_lowering_cuda.kernels/k_linear_sdpa_reduce_*.torch.json --bench --profile`.

## Workflow notes

Audit of the prior dynamic report's notes (`3abe9959`):

- **Per-launch mis-attribution (recurring):** reproduced again (Finding 4) — the dynamic P@V deployed window over-reads
  by ~2.7x. Third report running; `plans/bench-attribution-by-slicing.md` is now the top recurring tooling gap.
- **Reproducer re-fusion:** reproduced — both attention reproducers re-fuse the cone, so every attention number came
  from NCU, not the reproducer table. A `run --ir … --bench --no-refuse` (bench only the named provenance kernel) would
  still save the most triage time.
- **Tune-`--bench` thermal inflation:** reproduced and **worse** (Finding 5) — 2x here vs 30% in the prior report,
  because this tune was the longest yet (3.6 h). The reported e2e again had to come from a separate cooled run.
- **Chart PNG Playwright flake:** reproduced verbatim on both tunes.
- **Stable per-op identity across views:** reproduced — the same P@V op wears `a76a28` (deployed) and `22a7a0`
  (NCU) because each re-lowering re-hashes; I hand-cross-referenced. A stable provenance name in the NCU / leaderboard /
  deployed tables would remove the cross-referencing.

New friction this run:

- **The dynamic tune cannot complete on #244 (Finding 2) — biggest wall-clock loss by far.** ~3.7 h of GPU time
  (original wedge ~50 min frozen + resume ~1 h that re-wedged) produced no usable dynamic tune; I recovered only by
  bench-detecting the wedge (CPU-tick sampling: `state=S` + 0 ticks/20 s, no children) and falling back to the default
  prior. **Two concrete asks:** (a) a tuner watchdog — if no new `perf` row lands for N minutes while the bench worker
  is dead, abort with the partial DB instead of hanging forever; (b) a `tune --bench-isolation strict` that fences the
  parent CUDA stream after every worker hang. Without either, a single bad TMA variant kills a 4 h tune silently.
- **Static vs dynamic at matched seq needs an explicit flag.** Static traces at `--seq-len` (default 32); dynamic
  benches at the hardcoded 512 hint. I tuned static at 32 first (43 µs — meaningless against dynamic@512) before
  realising the mismatch, then re-tuned static at `--seq-len 512` (+3.6 h). The skill should call out: **for a
  static-vs-dynamic side-by-side, tune static at `--seq-len 512` to match the dynamic hint** (or a CLI guard that warns
  when a static tune's seq ≠ `DEFAULT_SEQ_HINT` and a dynamic comparison is intended).
- **Wedge detection is a manual ritual.** Confirming the hang took `nvidia-smi` + `ps`/`/proc/<pid>/stat` CPU-tick
  sampling + child-process greps + `py-spy` (ptrace-blocked without sudo) across several commands. A `deplodock tune
  --heartbeat-file PATH` (touch a file per completed terminal) or a non-tty per-terminal stderr heartbeat (`[tune]
  terminal k/16, best Σ …`) would make liveness a one-line check instead of a forensic dig — the tty progress bar does
  not survive `tee`, so a 3.6 h tune logs ~10 lines.
- **`eval failures` shared-knobs is excellent and did its job twice** — it pinned `TMA=True` as the dynamic wedge's
  common knob (Finding 2) and clustered static's benign compile failures (Finding 6) in one command, no log grepping.
  One gap: it does not distinguish **runtime hang** from **compile-time abort** — both show as `bench_fail`, but they
  are operationally opposite (one wedges the tune, one is contained). A `kind` column (hang / compile / runtime-error)
  would surface the dangerous class immediately.
- **Reading the deployed dynamic e2e off the default prior was the salvage** — `run` being read-only on the prior meant
  the previous report's fully-tuned dynamic prior was reusable on the new commit without a re-tune. Worth documenting in
  the skill as the fallback when a dynamic tune wedges: `DEPLODOCK_PRIOR_FILE=~/.cache/deplodock/prior.json run
  --dynamic --bench` gives a deployable number from the last good tune (caveat: kernels #244 re-lowered may fall back
  to analytic picks — here the 340 µs matched the reference's 359 µs, so the transfer held).
