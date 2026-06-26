# Qwen3-Embedding-0.6B layer-0 tune findings — analytic-prior → CatBoost guidance

**Status:** complete (single-layer, dynamic). Run on an **RTX 5090** (`sm_120`, driver 580.159.03) on 2026-06-25,
immediately after the `prior: featurizer reads native MOVE@element knobs + refit _W_A/_W_A_DYN` commit (`a4bff929`) —
the point of this run is to watch the **refit cold `AnalyticPrior`** guide the learned `CatBoostPrior` on a real model.

**Run command**

```bash
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench \
  --dump-dir _tune/tune-qwen3-emb-06b/dump 2>&1 | tee _tune/tune-qwen3-emb-06b/tune.log
```

**Run stats:** 1 fused terminal, inner search **1010.9 s** (~17 min). Prior trained on **972 benches** (warmup 76 /
post 896), reservoir dataset 22 263 rows. **802 ok / 33 bench_fail** measured configs. Post-training calibration
(Spearman, predicted vs latency) **+0.51**; best inner latency 2.533 µs @ bench #885.

**Measurement semantics.** `--clean` reset the tune DB + prior + cubin caches, so this is a genuine cold start: the
**refit `AnalyticPrior` drove the warmup picks**, `CatBoostPrior` took over post-training. Dynamic (`--dynamic
seq_len@x:1`): all kernels are symbolic-`seq_len` masked tiles **benched at the `Dim` hint `seq_len=512`**; the boundary
guards (`if (coord < seq_len)`) are part of the measured cost. **Number families:** the per-kernel `--bench` table below
is **-O3, deployable, CUDA-graph captured**; tune-DB `us` columns quoted for ranking context are **-O1** (1.5–3× slower
on reduction/attention kernels) and the `-O3 us` column is the in-band -O3 re-bench. Never compare across the two.

---

## Bench results

**Full-model end-to-end bench did not complete** — it aborted with
`HungKernelError("kernel 'k_sdpa_linear_reduce_5d67bc' did not complete within 1000 ms")` (Finding 1), so there is no
eager / torch.compile / deplodock e2e row for the layer. The per-kernel -O3 table is the deployable result; the
attention kernel is the reason the model-level number is missing — itself the top finding.

**Per-kernel -O3 `--bench` table** (deplodock vs eager / torch.compile, µs; `vs eager` = eager/deplodock, >1 = win):

| Kernel                      | Layer op (provenance)                  | eager | tcompile | deplodock | vs eager |
|-----------------------------|----------------------------------------|------:|---------:|----------:|---------:|
| k_sdpa_linear_reduce        | SDPA (attn) 14/15 + linear_2           |    31 |       30 |       216 |  0.14x   |
| k_linear_mean_loop_reduce   | linear + mean-pool loop reduce         |   119 |       57 |       176 |  0.68x   |
| k_mean_linear_reduce        | mean-pool + linear reduce              |   104 |       20 |        43 |  2.43x   |
| k_linear                    | linear_6 (proj) + add_7                |    23 |       23 |        41 |  0.56x   |
| k_linear_reduce             | linear (q/k/v proj, partial)           |    14 |       14 |        35 |  0.41x   |
| k_mean_linear_reduce        | mean-pool + linear reduce              |    76 |       14 |        29 |  2.64x   |
| k_linear_reduce             | linear (proj, partial)                 |    10 |       10 |        24 |  0.43x   |
| k_mean                      | mean-pool reduce                       |    72 |        6 |        14 |  5.12x   |

(`k_linear_sdpa_reduce` was skipped — its reproducer bench exceeded the 60 s GPU-time cap.)

**Dominating slices.** deplodock **wins the reduction-shaped kernels** — `k_mean` 5.12×, the two `k_mean_linear_reduce`
2.43–2.64× — the mean-pool reductions the carrier-algebra reduce tier handles well. It **loses the matmul-shaped and
attention kernels**: `k_sdpa_linear_reduce` (0.14×, 216 µs) alone dwarfs the rest and is bench-failure-dominated; the
`k_linear`/`k_linear_reduce` projections sit at 0.41–0.56× of cuBLAS on tiny (M=512-hint) GEMMs.

---

## Prior guidance — does the refit `AnalyticPrior` steer `CatBoost` onto the best configs?

This is the headline question. Three views, and they tell a layered story.

**1. Aggregate pick reachability over the measured DB** (`deplodock eval prior --dataset db`): the prior's *model*
*argmin* recovers each op's measured-best to **mean 2.52× / median 1.06× / worst 12.05×**. The median is the story — for
the typical op the prior lands within **6 %** of the measured best. The tail is a handful of matmuls:

```
matmul free=1024 red=1024  best  57.38us  pick 691.36us  (12.05x, 32 configs)  <-- model argmin misses
matmul free=1024 red=2048  best  73.35us  pick 196.78us  (2.68x, 31 configs)
matmul free=3072 red=1024  best 106.39us  pick 171.18us  (1.61x, 23 configs)
```

**2. But the *deployed* pick is rescued by the -O3 evidence reservoir.** `eval prior --dataset db` scores the learned
model's raw argmin over **-O1** rows; the actual `Prior.pick` consults the **-O3 reservoir first**. `eval variants`
shows the deployed pick (◄) is frequently a poor -O1 rank yet the **-O3-deployable best**:

```
k_linear_reduce_716194  pick rank 3/32 (1.25x -O1)  ── but pick -O3 = 48.9us is the LOWEST -O3 of any config
k_mean_linear_reduce    pick rank 13/45 (1.32x -O1) ── but pick -O3 = 2.5us  beats the -O1-fastest (2.6us -O3)
k_mean_linear_reduce_mm1 pick rank 17/23 (1.61x -O1)── but pick -O3 = 100.7us beats rank-3's -O3 (102.6us)
```

So the same matmuls that look like a 12× "miss" under the -O1 model argmin are, after the -O3 re-bench of the
tolerance-band contenders, deployed at their **-O3 optimum**. The refit `AnalyticPrior`'s job in this loop is exactly
that: its -O1 ranking surfaces the right configs into the `DEPLODOCK_O3_TOL` band, they get -O3-sampled, and the
evidence-first `Prior.pick` then deploys the -O3 winner regardless of its noisy -O1 rank. The 12× model-argmin gap is a
**learned-model calibration** limitation (Spearman +0.51 — moderate), not a deployment regression, *as long as the
reservoir has an -O3 sample*. Where it has none (small static-K projections with `-O3 us = —`), the pick falls back to
the -O1 rank and the 1.21–1.25× shortfalls are real (Finding 5).

**3. Cold-start sanity.** The warmup 76 benches were driven by the refit `AnalyticPrior` alone (no `prior.json` after
`--clean`). The post-training "silly (≥2× best)" rate is 783/846 = 93 % — high, but expected for a single-layer cold
tune whose enumeration is dominated by the attention kernel's failing configs (Finding 1); the reduction kernels the
prior is well-calibrated on (the golden regimes) are the ones that deploy at ≤1.06×.

---

## Finding 1 — `k_sdpa_linear_reduce` attention: 26/27 configs bench_fail, blocks the e2e bench (priority: high)

**Symptom.** The attention kernel is 0.14× eager (216 µs vs 31) and its tune enumeration is almost entirely dead: `eval
variants --kernel k_sdpa_linear_reduce` shows **1 measured config / 26 bench_fail**, the lone survivor at a catastrophic
334 712 µs (-O1). It hangs the GPU past the 1 s watchdog, which is what aborted the **full-model e2e bench**.

**Evidence.** `eval failures` clusters all 26 under `k_sdpa_linear_reduce_5d67bc`:

```
14 rows  RuntimeError 'benchmark run stage exceeded 2.0s of GPU time'   shared: REDUCE@a3=s1/f1/c1/t1, SPLIT@a1=…
 6 rows  HungKernelError 'did not complete within 1000 ms'              shared: REDUCE@a3=s1/f1/c1/t1, SPLIT@a1=1x1
 …       bench worker EOF (timeout 1.0s)
```

Every failing config shares `REDUCE@a3=s1/f1/c1/t1` (a degenerate reduce on the attention reduce axis) and small
`SPLIT@a1`. Provenance: `scaled_dot_product_attention (SdpaOp): 14/15 — partial` — this is the **non-flash** SDPA reduce
path (the `FLASH` knob fuses SDPA into the streaming online-softmax MONOID nest; this kernel is the un-fused reduce).

**Root cause (hypothesis).** The symbolic-`seq_len` masked SDPA reduce, on the non-flash path, generates kernels that
loop the full hint-K with a degenerate per-row reduce and no convergent tiling — they run unboundedly long at small
`SPLIT@a1`. The dominant question is whether `FLASH=True` (the streaming flash nest) was enumerated for this symbolic-K
SDPA at all; CLAUDE.md notes flash-style **fused symbolic-K attention remains future work** and the prologue P@V stays
degenerate at `FM=FN=1`. If the flash fork is gated off for symbolic K, the search is stuck on the slow reduce path —
a **class-2 (tier/structural lockout)** on top of the **class-4 (bench failures)**.

**Next diagnostic.** `DEPLODOCK_KNOBS="FLASH=True" deplodock compile <k_sdpa_linear_reduce>.torch.json --ir cuda`
(compile only, no GPU) to see whether the flash nest is even reachable for this op, then find the gate in
`passes/lowering/tile/` that declines it on symbolic K. **Do not** re-bench this kernel live without the 1 s watchdog.

**Fix.** Gate the degenerate `REDUCE@a3=s1/...` + tiny-`SPLIT@a1` SDPA-reduce configs out of enumeration (they only ever
hang), and/or unlock the symbolic-K flash fork. Until then the watchdog correctly fences them, but they burn 26 search
slots and block the e2e bench.

**Resolution (the class-2 lockout — landed).** Root cause: a **symbolic** streaming-flash KV axis is serial-locked
(`streaming_br_offers` → `BR=1`, `_streaming_bk` → `BK=1`), so `enumeration/070_coop_reduce._streaming_leaves` fell to
`monoid_build` — the scalar streaming nest that recomputes the QK^T score **per P@V output `d`** (the `d` loop wraps the
KV stream), O(`d_v`=64) redundant, so every config ran unboundedly long (the 26 dead leaves were that one
per-`d`-recompute kernel under different free tiles). The fix is the **FA-2 shared-score restructuring**: route the
symbolic streaming flash through **`chain_build`** by default — the score is computed ONCE per KV step and shared across
`d` (the P@V output rides a register vector `O[d]`, the score edge placed INLINE). `_chain_applicable` was relaxed to
admit a **symbolic hinge** (only the inner QK^T must be static — the KV stream stays a serial runtime-bounded loop, no
tiling → no masking, every `kv < seq_len` valid); `_streaming_leaves` makes `chain_build` the symbolic default
(`monoid_build` + the one-leaf collapse remains the fallback for a chainless symbolic stream). A static stream keeps
`chain_build` a `DEPLODOCK_CHAIN=1` opt-in (unchanged).

**Measured (RTX 5090, symbolic SDPA `1×8×512×64` at the hint, `run --bench`, -O3 captured):** **334 712 µs (-O1 hang) →
515 µs** cold-analytic default (correct to max_diff 1e-6 vs eager) — a ~650× kernel-level improvement; the cold pick
threads the query (`block=32`, 25 % occ) with `d` in registers (`FM=64`, `PLACE@v1=inline`). The remaining gap to eager
(53 µs cuBLAS/flash) is the scalar-FMA P@V + serial KV stream + 167 regs (25 % occ ceiling) — closing it needs the
**tensor-core warp chain for symbolic streaming** (`warp_chain_eligible` is static-only today), the genuine follow-up.
NOTE the **learned prior** carried from this run is stale for the new chain regime (it was trained on the old
all-failing `monoid_build` kernels) and mis-ranks the chain leaves to `block=1` (1203 µs); a re-tune relearns it — the
cold `AnalyticPrior` already picks the 515 µs config. Accuracy is guarded e2e by the `*_dynamic_matches_torch` flash
tests (SDPA / GQA+causal / additive-mask over symbolic `seq_len`, now exercising the chain path); routing by
`tests/compiler/passes/test_streaming_symbolic_chain.py`. The **e2e-abort retry** (workflow note) is the only Finding-1
follow-up left.

## Finding 2 — learned-model argmin misranks small matmuls 2.7–12× (rescued by -O3 evidence) (priority: medium)

**Symptom / evidence.** See the Prior-guidance section: `eval prior --dataset db` worst 12.05× on `matmul free=1024
red=1024`, yet `eval variants` shows the deployed pick is the -O3-best for those same kernels. **Root cause:** the
`CatBoostPrior` Spearman is +0.51 — it ranks the broad strokes but not the fine -O1 ordering of near-best matmul
configs, so its argmin drifts onto a slow tile. The evidence-first `Prior.pick` corrects this **only where an -O3
reservoir sample exists** for a tolerance-band config.

**Fix / watch.** This is acceptable by design (the -O3 band is the safety net) but argues for (a) more tune patience on
the matmul ops so more contenders enter the -O3 band, and (b) feeding the -O3 rows back as `H_opt=3` training rows to
lift the model's matmul calibration over successive tunes. Re-running `tune` without `--clean` (accumulate) should
tighten the 12× tail — worth a follow-up measurement.

## Finding 3 — bf16 MMA codegen emits an unused helper → nvcc compile failure (priority: medium)

**Symptom.** `eval failures` cluster on `k_linear_reduce_716194` (3 rows):

```
nvcc compile failed: a_m16n8k16_bf16" was declared but never referenced
  static __attribute__((device)) … void dpl_mma_m16n8k16_bf16(float* d, const unsigned* a, …)
shared knobs: ATOM@out=mma_m16n8k16_f16, FM=2, SPLIT@a0=4x1, SPLIT@a1=8x2, REDUCE@a2=s1/f1/c1/t1
```

**Root cause.** With `ATOM@out=mma_m16n8k16_f16` selected, codegen still **emits the `dpl_mma_m16n8k16_bf16` helper**
(and a `a_m16n8k16_bf16` operand decl) that nothing references → the compile fails under the strict diagnostic. A
dead-helper emission in the warp-tier CUDA lowering — the f16 and bf16 atom paths aren't cleanly separated. **Class 4**
(bench failure): these are pure wasted search slots.

**Repro (no GPU).** `DEPLODOCK_KNOBS="ATOM@out=mma_m16n8k16_f16,FM=2,SPLIT@a0=4x1,SPLIT@a1=8x2"
deplodock compile <k_linear_reduce_716194>.torch.json --ir cuda` and grep for `bf16` in the emitted source.

**Fix.** Emit the atom helper for the *selected* kind only (gate the bf16 helper on `ATOM@out` being a bf16 atom).

## Finding 4 — `KeyError('mul_N_smem')` when staging a pointwise `mul` edge in smem (priority: medium)

**Symptom.** `eval failures` clusters `k_linear_0837e7` (`KeyError('mul_16_smem')`, 3 rows) and `k_linear_reduce_716194`
(`KeyError('mul_1_smem')`, 1 row), each sharing `PLACE@mul_N=smem:sync` (+ `PLACE@linear_N_wt=smem:sync`).

**Root cause.** When the placement fork stages the activation `mul_N` edge into smem, assembly looks up a `mul_N_smem`
buffer key that was never created — an assembly/schedule gap for staging a **pointwise (non-weight) edge** in smem.
**Class 4.** The weight edge (`linear_N_wt`) stages fine; the `mul` edge is the one missing its smem buffer.

**Repro (no GPU).** `DEPLODOCK_KNOBS="PLACE@mul_16=smem:sync,PLACE@linear_6_wt=smem:sync,ATOM@out=mma_m16n8k16_f16"
deplodock compile <k_linear_0837e7>.torch.json --ir cuda`.

**Fix.** Register the smem buffer for a staged pointwise edge in the assembly pass (or decline `PLACE@<pointwise>=smem`
in the offer if it's not meant to be stageable).

## Finding 5 — small-K projection GEMMs trail cuBLAS 2–2.5× (priority: low)

**Symptom.** `k_linear` 0.56×, `k_linear_reduce` 0.41×/0.43× vs eager (= cuBLAS) on tiny (M=512-hint, K=1024–3072)
projections. The deployed picks are reasonable (warp-tier `mma_m16n8k16_f16`, `smem:tma`/`smem:sync`) but these are
small GEMMs where cuBLAS's hand-tuned kernels dominate and some picks have **no -O3 reservoir sample** (`-O3 us =`
`—`), so they deploy on the -O1 rank (1.21–1.25× of -O1-best). **Class 3** + the Finding-2 reservoir gap.

**Next step.** NCU compare (`run --ir <k>.torch.json --bench --profile`) to quantify the cuBLAS gap (occupancy / fma% /
dram%); not done here to keep this run focused on the prior-guidance question and avoid the hanging attention kernel.

---

## What deplodock already wins

`k_mean` (5.12×), both `k_mean_linear_reduce` (2.43×, 2.64×) — the mean-pool reduction kernels, the regime the refit
prior is best calibrated on (the reduce goldens). These deploy at ≤1.06× of their measured best per `eval prior`. The
embedding model's pooling tail is deplodock's strength; the projections and attention are the gap.

## Repro / artifacts

- Tune log: `_tune/tune-qwen3-emb-06b/tune.log`
- Dump dir: `_tune/tune-qwen3-emb-06b/dump/` (per-kernel `.torch.json` reproducers, `kernels.html` / `.png`,
  `62_kernel_bench.json`)
- Tune DB: `~/.cache/deplodock/autotune.db`; prior: `~/.cache/deplodock/prior.json`
- Prior reachability: `deplodock eval prior --dataset db`
- Failures: `deplodock eval failures`
- Per-kernel leaderboards: `deplodock eval variants --kernel <k_sdpa_linear_reduce|k_linear_reduce|k_linear>`
- Compile-only repros (no GPU) for Findings 3/4: pin the cluster's shared knobs via `DEPLODOCK_KNOBS` and
  `deplodock compile <reproducer>.torch.json --ir cuda`.

## Workflow notes

- **Slow / blocked step — the hanging attention kernel.** `k_sdpa_linear_reduce`'s 26 hanging variants cost ~minutes of
  watchdog timeouts during the inner search and **aborted the full-model e2e bench entirely**, so this run has no
  model-level eager/tcompile/deplodock number. Proposal: when the full-model bench aborts on a `HungKernelError`, retry
  it once with the offending kernel pinned to its fastest *measured* config (skip the greedy re-pick that may re-hang),
  so a single bad kernel doesn't cost the whole e2e table.
- **Two-number-family cross-referencing was manual.** Answering "is the pick actually good?" needed reading the `-O3 us`
  column in `eval variants` against the `eval prior --dataset db` -O1 reachability by hand — the 12× "miss" is benign
  once you see the -O3 column, but nothing says so. Proposal: have `eval prior --dataset db` annotate each miss with
  "(deployed -O3 = X, rank R)" so the evidence-first rescue is visible in the same view, not two commands apart.
- **`eval failures` was excellent** — the `(kernel, error) + shared knobs` clustering pinned Findings 3 and 4 to exact
  knob sets with zero log grepping; this is the view that made the codegen bugs actionable. No change needed.
- **`pgrep -f` self-matched the monitor.** A polling loop containing the tune command string matched its own argv, so
  "is the tune still running?" falsely reported YES after exit; had to cross-check with `ps -C python`. Minor, but worth
  a note for anyone scripting around `deplodock tune`.
- **No NCU this run** — deliberately skipped to avoid re-launching the hanging attention kernel under `ncu` and to keep
  the run focused on prior guidance. The codegen findings (3–5) would each benefit from an `ncu compare` follow-up.

---

## Autonomous e2e-perf session (2026-06-26, RTX 5090) — model now runs; SDPA is the wall

Goal: get the layer-0 e2e to run and perform reasonably. Two fixes landed (committed) and the bottleneck was
root-caused with **nsys ground truth** (the deplodock bench harness mis-attributes per-kernel time — see below).

**Fixes committed**

1. `tile: symbolic streaming flash deploys FA-2 shared-score (chain_build) by default` — the symbolic SDPA no longer
   hangs the watchdog / aborts the e2e bench (Finding 1's real fix); the model now runs end to end with correct output
   (`max_diff 0.0039` vs eager at the hint).
2. `kernel: cooperative WarpShuffle uses __activemask()` — a whole-CTA cooperative reduce with `BR < warp_size` launches
   a partial warp; the hard-coded `0xffffffff` mask named absent lanes (UB). Correctness hygiene.

**nsys ground truth at the deployment seq (512).** The deplodock `run --bench` per-kernel TABLE mis-attributes (it
blamed `k_mean_linear_reduce` / k-norm at 7864 µs and the SDPA at 14 µs) — nsys shows the opposite. The whole-program
number is right (~8.8 ms ≈ Eager 224 µs → **0.02×**); only the per-launch "solo window" attribution is scrambled. nsys
per-kernel (uncaptured, seq=512):

```
k_sdpa_linear_reduce   8.16 ms/call   <-- ~90% — the wall (the streaming flash)
k_linear_0837e7        0.21 ms/call   <-- the second cost (a real linear)
k_mean_linear_reduce   2.4 µs         <-- the "28 ms catastrophe" was pure harness mis-attribution
… all other kernels    2–40 µs each
```

**No capture penalty — it's seq, not capture.** An earlier read suggested the SDPA ballooned 325 µs → 8 ms under
CUDA-graph capture; that was a confound. The 325 µs was the forward at the **trace seq (32)**; the bench runs at the
**hint (512)**. nsys of an **uncaptured** forward at seq=512 shows the SDPA at **8.16 ms** — identical captured and
uncaptured. The SDPA is genuinely ~8 ms at the deployment length, period (O(seq²) and the wall regardless of capture).

**Why it can't be cheap (yet).** The symbolic-`seq_len` streaming flash is a **scalar serial-KV** nest: each query
thread streams all `seq` keys, **re-reading K/V from gmem with no smem tile sharing across queries**, at low occupancy
(block 8 / 168 regs). Eager's flash shares K/V tiles in smem (and uses tensor cores) and runs the **whole layer** in
224 µs. So eager-competitive symbolic attention needs the **smem-tiled / tensor-core streaming flash for a symbolic
axis** — `warp_chain` / `warp_chain_eligible` is **static-only** today, and `chain_build` + cooperative-KV (shared score
AND parallel KV) is the explicitly-"not wired yet" combination (`_chain_applicable` refuses `BR > 1`). This is the large
remaining piece — real architecture work, checkpointed rather than half-built autonomously.

**Dead end checked — the materialized SDPA is INCORRECT for dynamic seq.** The tempting shortcut is to route symbolic
SDPA to the score-materializing decomposition (QK^T → softmax → P@V, which reaches the tensor-core matmul tier; the
bare-SDPA bench showed 861 µs vs 6663 µs streaming). It is **wrong** at runtime `seq < hint`: the masked-N QK^T
**zero-fills** the masked keys, but softmax needs **−inf** (`exp(0)=1` spuriously adds the masked keys to the
denominator). It only matched torch because every bench is at `seq=hint=512` (no masked region) — a focused test at
seq=8/16/37 fails. So `010_recognize_flash`'s forcing of flash for symbolic SDPA is a **correctness** guard (the
streaming flash masks the score to −inf in the stream, `_mask_carrier`), not just routing; making the materialized path
correct needs the −inf mask propagated into the score before softmax (non-trivial). **The streaming flash stays the
only correct symbolic SDPA today** — which is exactly why its speed is the gating item.

**Smaller, tractable next steps (do not reach eager parity, but real):**

- **Chain thread-budget mis-allocation.** `070_coop_reduce._streaming_leaves` fans out `thread_offers` as a balanced
  `(t_n, t_m)` over `(inner_n, outer_m)`, but `chain_build` forces the P@V output `d` to a REGISTER axis — the threads
  spent on `d` are **wasted**, leaving the query (`m`) under-threaded (block 8). The chain free-tile fork should put the
  whole budget on `m_axis`. Improves occupancy; still memory-bound until K/V smem tiling lands.
- **Per-op tune benches symbolic kernels at the trace seq, not the hint** — the prior ranks the SDPA on cheap seq≈32
  numbers while deployment is seq=512 (O(seq²)), so its SDPA pick is mis-ranked. Bench symbolic per-op slices at
  `DEFAULT_SEQ_HINT`.
- **Bench-harness per-kernel attribution is wrong** (blames k-norm; nsys says SDPA). The whole-program number is fine;
  the per-launch solo window is not. Trust nsys until fixed.

**Status:** model **works** (was a hard hang); the k-norm "28 ms catastrophe" was a harness mis-attribution (real
2.4 µs); no capture penalty (the SDPA is ~8 ms at seq=512 captured AND uncaptured). Reasonable (eager-class) e2e perf is
gated on the **smem-tiled / tensor-core symbolic streaming flash** — sharing K/V tiles across queries (today's scalar
serial-KV nest re-reads K/V per query, the memory-bound wall) and using tensor cores. That is the next major task: real
architecture work, checkpointed rather than half-built autonomously.
