# Qwen3-Embedding-0.6B — layer-0 tune findings (static + dynamic, e2e + per-kernel + serve A/B)

**Status:** complete. Clean autotune of layer 0 in both static (shape-specialised) and dynamic (symbolic `seq_len`,
masked-tile) modes on an RTX 5090, with per-kernel triage, a whole-model serve A/B vs vanilla vLLM, and a direct
static-vs-dynamic softmax comparison. **Headline: the dynamic layer-0 forward is ~par with eager (1.03×) but 1.4×
behind torch.compile, and the gap is concentrated entirely in the SDPA softmax reduce (`k_sdpa_reduce`). At A/B time the
serve test was a hard negative: the tuned plugin **crashed at compile** and the cold-prior plugin ran 34× slower than
vanilla vLLM, so the per-kernel wins did not reach serving. A follow-up whole-model tune (12 terminals, ~4.5 h)
confirmed *tuning* doesn't fix the crash — the deployed kernel (an MMA-tier masked-K P@V) is an unfinished lowering
path the tuner never benches. **UPDATE: that crash was a real compiler bug, fixed in
[PR #260](https://github.com/cloudrift-ai/deplodock/pull/260)** (four root causes — xn operand dtype, `Source.dtype`
validate-sizing, `_is_transposed_b` collapsed-reshape, `k_zero` replication; `tests/compiler/` stays green). The tuned
plugin now **compiles and serves at 97.53 req/s** (14.3× the cold fallback, the number the bug had blocked), but the
masked-K P@V MMA is **not yet bit-correct** (cosine 0.23 vs torch — PR #260 is WIP), so that throughput is a perf
ceiling, not a deployable result. See the Serve A/B table and Finding 1.**

**Date:** 2026-06-18. **GPU:** NVIDIA GeForce RTX 5090 (sm_120), driver 580.159.03. **ncu:** 2025.3.1 (worked; no
permission gate). **vLLM:** 0.22.1.

**Run commands**

```bash
# dynamic (deployable: symbolic seq_len, masked-tile kernels)
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench \
    --dump-dir _tune/tune-model-qwen3-emb-l0-report/dump-dynamic     # DEPLODOCK_TUNE_DB/PRIOR_FILE → dynamic.{db,prior}
# static (contrast: shape-specialised at trace-default seq=32)
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --clean --bench \
    --dump-dir _tune/tune-model-qwen3-emb-l0-report/dump-static       # …→ static.{db,prior}
# serve A/B (200 reqs, conc 32, random-input-len 512, seed 0)
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --num-prompts 200 --random-input-len 512 \
    --max-concurrency 32 --bench-seed 0 -- --gpu-memory-utilization 0.8           # plugin (dynamic prior) → CRASHES
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --stock --num-prompts 200 --random-input-len 512 \
    --max-concurrency 32 --bench-seed 0 -- --gpu-memory-utilization 0.8           # vanilla vLLM baseline
```

**Run stats**

| run                   | search wall           | variants  | terminals       | notes                                                                                                                                            |
|-----------------------|-----------------------|-----------|-----------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| dynamic, layer 0      | 6276.5 s (~104.6 min) | 6611      | 8 fused         | 5 bench_fail; prior 7798 benches, Spearman +0.99                                                                                                 |
| static, layer 0       | 3948.7 s (~65.8 min)  | 5655      | 8 fused         | 0 bench_fail; prior 6485 benches, Spearman +0.99                                                                                                 |
| dynamic, whole model  | ~4.5 h (interrupted)  | 14686     | 12 (no `done`)  | accumulated into `dynamic.db` (→10335 perf rows); 18 unique kernels / 422 positions; stopped after 12 terminals — see Finding 1 / workflow notes |

**Number-family disclaimer (do not mix).** Every latency below that comes from a `--bench` table or an `eval` `-O3 us`
column is **-O3, deployable, CUDA-graph captured**. Tune-DB latencies quoted for ranking context (e.g. `eval variants`
`us` column, `pick: rank R/N`) are **-Xcicc -O1** — a ranking signal only; reduction/attention kernels run 1.5–3×
slower at -O1 than -O3.

**Mode note.** The dynamic run is the deployable artifact: symbolic `seq_len`, masked-tile kernels, all tune/bench
measurements taken at the `Dim` hint **seq_len = 512** (`DEFAULT_SEQ_HINT`; `--seq-len` only sizes trace tensors). The
static run is shape-specialised at the **trace-default seq_len = 32** — fast, no masked-tile guards, **not** the
deployable configuration, and not the same shape as the dynamic run (see the seq-confound caveat in Finding 5).

---

## Bench results

### Dynamic layer-0 end-to-end (seq_len = 512 hint; torch inputs tiled to match)

| Backend       | Latency (µs) | vs Eager  |
|---------------|--------------|-----------|
| Eager PyTorch | 222          | 1.00×     |
| torch.compile | 151          | 1.47×     |
| **Deplodock** | **216**      | **1.03×** |

Deplodock is ~par with eager and **1.4× behind torch.compile**. (A single layer's e2e, not the whole model — see the
serve A/B for the deployable whole-model answer.)

### Dynamic per-kernel (-O3 reproducer table), sorted by deplodock µs

> **POST-FIX re-bench ([PR #260](https://github.com/cloudrift-ai/deplodock/pull/260)).** Re-run with the masked-K
> MMA fixes applied. The non-SDPA rows are unchanged (within run-to-run noise); the two **SDPA matmul** reproducers
> got **slower** — QK^T 80→**96 µs**, P@V 58→**74 µs**. That regression is the fix *working*: the `Source.dtype`
> validate-sizing bug previously rejected every masked-K MMA config, so the greedy prior fell back to scalar; with
> the slab now sized at fp16 the masked-K MMA split-consumer is enumerable and the prior deploys it — but at the
> seq=512 hint for these GQA shapes it is **not** faster than the scalar fallback it replaced (P@V re-lowers to
> softmax 50 µs + xn 11 µs + MMA consumer 11 µs + proj 11 µs). And it is still numerically WIP (the served P@V is
> cosine 0.23, Finding 1). So the fix **unblocks the deployment path** (which is what serving needed) but does not
> by itself make these reproducers a win. Pre-fix numbers in parentheses.

| kernel                        | layer op                           | eager | tcompile | deplodock     | vs eager                              |
|-------------------------------|------------------------------------|-------|----------|---------------|---------------------------------------|
| `k_linear_sdpa_reduce_6bb11e` | q_proj + QK^T scores (Linear+Sdpa) | 39    | 37       | **96** (←80)  | 0.41× ✗                               |
| `k_sdpa_linear_reduce_0045cd` | P@V + proj (Sdpa+Linear)           | 30    | 30       | **74** (←58)  | 0.40× ✗                               |
| `k_linear_mean_reduce_05d34c` | q/k-norm: 2×Linear + Mean          | 122   | 57       | 47            | 2.58× ✓                               |
| `k_sdpa_reduce_042770`        | rotary+mask+softmax (Sdpa reduce)  | 152   | 25       | **45**        | 3.36× vs eager, **loses to tcompile** |
| `k_linear_0837e7`             | linear + activation                | 23    | 23       | 23            | 1.00× ≈                               |
| `k_mean_linear_reduce_34d0f3` | RMSNorm + proj                     | 105   | 20       | 18            | 5.68× ✓                               |
| `k_linear_reduce_f94dd0`      | linear (o_proj/down)               | 14    | 14       | 15            | 0.94× ≈                               |
| `k_mean_linear_reduce_ca9764` | RMSNorm + proj                     | 76    | 14       | 12            | 6.17× ✓                               |
| `k_linear_reduce_716194`      | linear (o_proj/down)               | 10    | 10       | 9             | 1.09× ✓                               |
| `k_mean_d65726`               | RMSNorm (Mean)                     | 66    | 4        | 2             | 34.68× ✓                              |

The fused RMSNorm+matmul and pure-matmul kernels win or tie; the three SDPA-bearing rows lose. **The "96/74/45 µs"
SDPA numbers are multi-kernel re-lowering totals, not single kernels** — see Finding 2, which is essential to reading
this table correctly. The reproducer solo-sum over-counts vs the whole-program e2e (the SDPA reproducers re-lower the
softmax + projections alongside the matmul).

### Static per-kernel (-O3, seq_len = 32 — shape-specialised), sorted by deplodock µs

| kernel                        | eager | tcompile | deplodock | vs eager                  |
|-------------------------------|-------|----------|-----------|---------------------------|
| `k_linear_sdpa_reduce` (QK^T) | 14    | 12       | 12        | 1.17× ✓ (ties tcompile)   |
| `k_linear_mean_reduce`        | 65    | 14       | 12        | 5.25× ✓                   |
| `k_sdpa_linear_reduce` (P@V)  | 10    | 10       | 8         | 1.25× ✓                   |
| `k_linear`                    | 8     | 8        | 7         | 1.17× ✓                   |
| `k_sdpa_reduce` (softmax)     | 86    | 8        | 7         | 12.31× ✓ (beats tcompile) |
| `k_mean_linear_reduce` ×2     | 57/59 | 8/8      | 6/6       | ~9.5× ✓                   |
| `k_linear_reduce` ×2          | 6/6   | 6/6      | 4/3       | 1.68×/2.10× ✓             |
| `k_mean`                      | 52    | 4        | 1         | 40.18× ✓                  |

At seq=32 the specialised kernels meet or beat eager **and** torch.compile across the board — including the SDPA
kernels that lose in the dynamic run. The regression is the masked-tile path, not the kernels' fundamentals (Finding 5).

**Post-fix:** unchanged. PR #260 only touches the **masked-K** (symbolic reduce axis) path; static specialised kernels
have a concrete K and no masking, so re-benching them with the fix reproduces these numbers exactly (QK^T 12.2 µs, P@V
7.9 µs, softmax 5.4 µs — within noise).

### Serve A/B vs vanilla vLLM (200 reqs, concurrency 32, random-input-len 512, seed 0, enforce-eager)

| serve config                                  | req/s       | tok/s    | mean E2EL (ms)  | median   | P99      | outcome                                                                             |
|-----------------------------------------------|-------------|----------|-----------------|----------|----------|-------------------------------------------------------------------------------------|
| **vanilla vLLM** (`--stock`)                  | **229.05**  | 117,273  | 129.85          | 128.15   | 196.75   | baseline ✓                                                                          |
| deplodock **tuned** prior — at A/B time       | —           | —        | —               | —        | —        | **CRASHED** (Finding 1)                                                             |
| deplodock **tuned** prior — PR #260 (fixed)   | 97.53       | 49,935   | 307.89          | 317.72   | 382.73   | compiles + serves; **NUMERICALLY WRONG** (cosine 0.23 vs torch) — perf ceiling only |
| deplodock **cold** AnalyticPrior              | 6.81        | 3,489    | 4410.92         | 4674.76  | 4952.87  | compiles, correct, **~34× slower**                                                  |

**At A/B time the deplodock-tuned plugin could not be benched — it crashed at compile** (masked-K `LoweringError`,
Finding 1), so the only producible deplodock serving number was the cold/scalar fallback (6.81 req/s, ~34× slower than
vanilla vLLM). **The crash was a real compiler bug, fixed in [PR #260](https://github.com/cloudrift-ai/deplodock/pull/260)**
(four root causes: xn operand dtype, `Source.dtype` validate-sizing, `_is_transposed_b` collapsed-reshape, `k_zero`
replication). With those fixes the tuned plugin **compiles and serves at 97.53 req/s** — **14.3× the cold fallback** and
the throughput the bug had blocked — but the masked-K P@V MMA kernel is **not yet bit-correct** (cosine 0.23 vs torch;
PR #260 is WIP), so this row is a **perf ceiling, not a deployable quality result**. A valid quality A/B awaits the
P@V bit-correctness fix; even then it is ~2.35× behind vanilla vLLM, so the layer-0 per-kernel wins still do not (yet)
translate into a served win on this model.

---

## Finding 1 — the tuned plugin can't compile: the masked-K MMA P@V was rejected at smem-validate (RESOLVED, [PR #260](https://github.com/cloudrift-ai/deplodock/pull/260))

> **RESOLUTION ([PR #260](https://github.com/cloudrift-ai/deplodock/pull/260)).** This was a **real compiler bug**, not
> the "unfinished MMA path" the original analysis below concluded. Four root causes, all fixed; `tests/compiler/` stays
> green (1629 passed); the tuned plugin now compiles and **serves at 97.53 req/s** (Serve A/B table):
> 1. **`020_stage_inputs` didn't stamp `Source.dtype` at creation** — so the post-`020` `TileOp.validate` smem-budget
>    gate sized the fp16 masked-K slab at the fp32 `BYTES_PER_ELEM` fallback (132 KB vs the real 66 KB) and rejected it
>    against the 99 KB cap. **This is why the tune showed "0 MMA configs" — every masked-K MMA variant failed validate,
>    not because the path is unfinished.** With the operand un-stageable AND un-lowerable gmem-direct, the `005` gate
>    crashed greedy deploy.
> 2. **`_split_demoted` materialized the `xn` intermediate at the fp32 softmax-cone dtype**, not the matmul operand
>    dtype (V = fp16) — 2× oversizing the slab and mis-typing the `ldmatrix.b16` read.
> 3. **`011_lower_atom_cell._is_transposed_b`** mis-flagged the collapsed-reshape V (`[seq, kv*hd]`) as transposed.
> 4. **`LdmatrixLoad.k_zero`** was dropped through register-tile replication.
>
> So "tuning can't fix it" was *correct* (the whole-model tune confirmed it — the fix is in the compiler, not the
> search), but "no MMA config exists / the path is unfinished" was **wrong**: the MMA configs were being silently
> rejected at validate. **Caveat — PR #260 is WIP:** the deployed masked-K P@V MMA is **not yet bit-correct** (cosine
> 0.23 vs torch; a residual fragment-layout bug, see the PR), so the 97.53 req/s is a *perf ceiling*, not deployable
> quality. The original analysis is kept below for the record.

**Symptom.** `deplodock serve … --bench` with the tuned dynamic prior fails vLLM EngineCore init during the plugin's
model compile (`serving/runner.py:130` → `backend.compile` → `Pipeline.run`):

```
LoweringError: masked-K (symbolic reduce) mma operand can't lower gmem-direct — no K zero-fill
(only the staged _stage_expand path zero-fills the partial slab); needs staging
```

**Root cause (`compiler/pipeline/passes/lowering/kernel/005_lower_atom_tile.py:519-530`).** The gate is **by design** a
search-time signal:

```python
# Masked-K (symbolic reduce) correctness gate: the gmem-direct fragment load has no K zero-fill, so an unstaged
# masked-K operand reads past the padded extent ... Bail so the search stages it or falls to the scalar path.
if (not a_staged and _unstaged_masked_k(a_load, "a", graph)) or (
    not b_staged and not b_trans and _unstaged_masked_k(b_load, "b", graph)):
    # LoweringError (not RuleSkipped): drop this candidate so the search falls back to a staged-mma or scalar variant.
    raise LoweringError("masked-K (symbolic reduce) mma operand can't lower gmem-direct …")
```

During **tuning** the SP-MCTS catches this `LoweringError` and drops the candidate. During **greedy deployment** (serve
plugin / `compile` / `run`) there is no search and no fallback — the error propagates and crashes the engine. This is
the hard-error twin of the QK^T `dpl_mma_load_b_gmem` perf smell (Finding 3): an unstaged MMA B-operand is a
slow-but-correct fallback for non-masked-K, but a **correctness crash** for masked-K.

**The deeper cause — and why tuning cannot fix it (verified).** It is tempting to call this prior-dependent and
fixable by tuning the deployed graph. It is not. The fused masked-K P@V is an **unfinished MMA path**: CLAUDE.md states
"flash-style fused symbolic-K attention remains future work," and the tuner deliberately keeps the fused masked-K P@V
**degenerate scalar-tier**. Direct evidence from the tune DB: the whole-model P@V `k_sdpa_linear_reduce_95f2c6` has
**857 measured configs and 0 of them are MMA** (`eval variants --kernel k_sdpa_linear_reduce_95f2c6 | grep -c
mma_m16n8k16` → 0) — all scalar, ~1055 µs, and the prior's pick (rank 5) is scalar too. The **serving** greedy compile,
on its own graph (`serving/runner.py` traces the `AutoModel` trunk), reaches the **MMA-tier** masked-K P@V — the
unfinished path the tuner never takes — and hits the unguarded `LoweringError`.

**Experiment that proves it (the key result of this run).** I ran a full **whole-model dynamic tune accumulating into
the same `dynamic.db`/prior** (12 terminals, ~4.5 h, the served kernels benched directly — e.g. 857 P@V configs,
`k_linear_sdpa_reduce_6db6c8` 193, `k_sdpa_reduce_5661da` 89), then re-ran `serve` with that whole-model-trained prior.
**It crashes with the byte-identical masked-K `LoweringError`.** So no amount of tuning helps: the kernel serving
deploys (MMA masked-K fused P@V) is one the tuner *never produces* (it stays scalar) — tuning the deployment graph
gives the prior evidence for the *scalar* P@V, not the *MMA* one the serving lowering reaches. The cold `AnalyticPrior`
"compiles" only because it stays conservative scalar (→ Finding 1b's 34× slowdown), not because it found a good masked-K
MMA config — there isn't one.

**Repro (compile-only, no bench — fails once weights load; identical under the layer-0 or whole-model prior):**

```bash
DEPLODOCK_PRIOR_FILE=…/dynamic-prior.json deplodock serve Qwen/Qwen3-Embedding-0.6B --dry-run
DEPLODOCK_TUNE_DB=…/dynamic.db deplodock eval variants --kernel k_sdpa_linear_reduce_95f2c6 | grep -c mma   # → 0
```

**Fix (priority: HIGH — this blocks the deployable artifact; tuning is NOT the fix).** Two real options, neither of
which is "tune more": (1) **guard the deployment path** — when `005_lower_atom_tile` would raise `LoweringError` for an
unstaged masked-K MMA operand, the greedy compile must force-stage it or fall to the (lowerable) scalar/split variant
the tuner uses, instead of propagating; equivalently, `Prior.pick`/enumeration eligibility must exclude unstageable
masked-K MMA configs for symbolic-K ops so they can never be deployed. (2) **finish the fused masked-K MMA path**
(flash-style fused symbolic-K attention) so it lowers and is fast — the real performance fix. Until one of these lands,
the embedding plugin cannot serve any model whose whole-graph attention reaches a masked-K MMA, regardless of tuning.

## Finding 1b — cold-prior serving is 34× slower than vanilla vLLM

**Symptom.** The cold-`AnalyticPrior` plugin serves correctly but at **6.81 req/s vs vanilla vLLM's 229.05 req/s**
(mean E2EL 4411 ms vs 130 ms). Both run under `enforce-eager` (CUDA graphs off on both sides — apples-to-apples on that
axis).

**Root cause.** The cold prior deploys un-tuned kernels, and for the masked-K attention it stays scalar-tier (the
conservative pick), i.e. scalar attention against vLLM's flash kernels — ~1 ms per P@V × 28 layers, no CUDA graphs.

> **UPDATE ([PR #260](https://github.com/cloudrift-ai/deplodock/pull/260)).** The original claim "there is no fast
> masked-K MMA config to deploy" was wrong — the MMA configs were rejected at smem-validate (Finding 1's resolution).
> With the fix the **tuned** plugin deploys the masked-K MMA P@V and serves at **97.53 req/s — 14.3× this cold
> fallback** (Serve A/B table). It is still ~2.35× behind vanilla vLLM (229 req/s) and **not yet bit-correct** (cosine
> 0.23), so it is not a deployable win — but the *deployment* path the cold prior worked around is no longer a wall.

**Fix (priority: HIGH).** ~~Blocked on finishing the fused masked-K MMA attention.~~ The compiler crash is fixed
([PR #260](https://github.com/cloudrift-ai/deplodock/pull/260)); what remains is (a) the P@V MMA's bit-correctness
(WIP in the PR) and (b) the perf gap to vLLM's fused flash attention — the per-kernel layer wins (RMSNorm/matmul
fusions) only translate to a served win once both land.

## Finding 2 — per-kernel reproducer totals over-count SDPA kernels (the matmuls are fast; the softmax is the cost)

**Symptom.** The per-kernel table attributes 80 µs to `k_linear_sdpa_reduce` and 58 µs to `k_sdpa_linear_reduce`
(both 0.49× eager), which reads as "the attention matmuls are slow." They are not.

**Evidence.** Re-lowering each SDPA reproducer standalone (`run --ir … --bench`) decomposes it into several kernels,
because the sliced reproducer re-lowers the full SDPA (QK^T + softmax + P@V) with no surrounding graph to fuse into:

- **QK^T reproducer (80 µs total) = 4 kernels:** the QK^T matmul `k_linear_sdpa_reduce_6bb11e` is **6.5 µs (8.3%)**;
  two softmax reduces `k_sdpa_reduce_*` are 31.3 + 30.6 = **61.9 µs (79%)**; split producer 10.1 µs.
- **P@V reproducer (58 µs total) = 4 kernels:** softmax `k_sdpa_reduce_6874a2` **31.3 µs (55%)**; P@V matmul
  `k_sdpa_linear_reduce_2fafc0` 10.0 µs; V-proj `k_linear_reduce` 8.9 µs; split producer 6.5 µs.
- **softmax reproducer (48 µs total) = 5 kernels:** two MMA reduces 13.8 + 10.1 µs + three scalar `_xn/_xna/_xnb`
  split producers 8.2 + 7.8 + 6.9 µs.

**Conclusion.** Every matmul is fast and competitive (QK^T 6.5 µs, P@V 10 µs, V-proj 8.9 µs). The recurring softmax
reduce `k_sdpa_reduce` (the [16, 512, 512] rotary+mask+softmax) is the real bottleneck and the entire gap to
torch.compile. This is also why solo-sum (313 µs) ≫ whole-program (216 µs): the SDPA reproducers re-decompose without
fusion. **Read the whole-program e2e and the named-kernel solo µs, never the reproducer total attributed to one name.**

**Fix (priority: MEDIUM — a reporting/tooling fix).** The per-kernel `--bench` table should label multi-kernel
re-lowerings as such (it already prints the sub-kernel breakdown in the `run --ir` knob table, but
`62_kernel_bench.json` and `kernels.html` collapse it to one row keyed by the reproducer name). See workflow notes.

## Finding 3 — the SDPA softmax reduce is bank-conflict- and occupancy-bound, and the conflict-breaking knobs are no-ops

**Symptom.** Standalone, deplodock's softmax is 48 µs vs torch.compile's 25 µs (1.9× behind). It is MMA-tier (so not a
tier lockout) and the prior's pick is rank 2/275 — so neither a search shortfall nor a tier problem (`eval variants
--kernel k_sdpa_reduce`).

**Evidence (NCU, `ncu-softmax/61_ncu_metrics.json`).**

| kernel                                                 | dur     | occ%     | sm%  | dram% | fma% | ld.cnflct   | regs   |
|--------------------------------------------------------|---------|----------|------|-------|------|-------------|--------|
| `k_sdpa_reduce_77b0f0` (main MMA reduce)               | 20.7 µs | 45.6     | 26.5 | 23.1  | 5.2  | **107,480** | 56     |
| `k_sdpa_reduce_6e4bd6` (2nd MMA reduce)                | 18.6 µs | **24.1** | 25.5 | 37.6  | 0.2  | 0           | **72** |
| torch ref `flash_fwd_splitkv_kernel` (whole attention) | 25.3 µs | 8.4      | 16.0 | 19.8  | 2.9  | 0           | 206    |

The main reduce is stalled on **107,480 shared-load bank conflicts** (SM 26.5%, FMA 5.2% — not compute-bound); the
second reduce is **occupancy-limited at 24.1% by 72 regs/thread** (DRAM 37.6%, FMA 0.2% — latency-bound). torch.compile
fuses the whole attention into **one** flash kernel; deplodock spreads it across 5 masked split kernels.

**The standard fix knobs do not reach this kernel.** A/B of `PAD_SMEM=1` and `PAD_SMEM=1,PERMUTE_LANES=1` on the
softmax reproducer (`run --ir … --ab …`) is a **no-op**: smem stays 32 K and the main reduce stays 13.8 µs. The pad
fork only emits "when at least one Source actually benefits" (`tile/070_pad_smem`), and the conflicts here come from the
MMA reduce's fixed `ldmatrix` smem staging, which the `+1` pad does not touch.

**Repro:**

```bash
deplodock run --ir dump-dynamic/07_lowering_cuda.kernels/k_sdpa_reduce_042770.torch.json --bench --profile \
    --bench-backends eager,deplodock                                   # NCU compare table
deplodock run --ir …/k_sdpa_reduce_042770.torch.json --bench --ab "PAD_SMEM=1" --ab "PAD_SMEM=1,PERMUTE_LANES=1"
```

**Fix (priority: HIGH — this is the entire e2e gap to tcompile).** A conflict-free smem layout for the MMA softmax
reduce (swizzle the score-tile staging so the `ldmatrix` reads don't collide), and/or a fused flash-style attention so
the scores never round-trip through gmem across 5 kernels (CLAUDE.md: "flash-style fused symbolic-K attention remains
future work"). Lower the second reduce's register pressure (72 → occupancy is 24%).

## Finding 4 — prior pick-reachability shortfall on the large (gated-MLP) matmuls

**Symptom.** `eval prior --dataset db` reachability: mean 1.27×, median 1.16×, **worst 2.13×**. The misses are the big
matmuls:

```
matmul free=3072 red=1024  best 31.05us  pick 66.16us  (2.13x, 141 configs)  <-- misses best
matmul free=3072 red=1024  best 33.71us  pick 56.94us  (1.69x, 123 configs)  <-- misses best
matmul free=1024 red=2048  best 46.17us  pick 67.77us  (1.47x,  79 configs)  <-- misses best
```

**Evidence.** `eval knobs --dataset db` ranks **`FM`** (per-thread M cells) as the highest-leverage tunable
(geomean regret 12.63×, p90 71×), then `SPLITK` 12.25×, `MMA` 11.61×, `BR` 9.74×, `FK` 8.93×, `WM`/`WN` ~8.6×. The
learned prior is mis-ordering `FM`/`SPLITK` on the large reduce shapes — the deployed config is a tensor-core tile but
not the fastest one.

**Fix (priority: MEDIUM).** More patience on the gated-MLP shapes, or better `FM`/`SPLITK` features in the prior.
Confirm by re-tuning the offending reproducer with 2–4× patience (no `--clean`); if the rank-1 config becomes the pick,
it's prior/patience, not codegen.

## Finding 5 — the masked-tile tax: dynamic vs static softmax (user-requested comparison)

**Caveat first:** static is seq=32, dynamic is seq=512 (16× more data) — compare the **deplodock-vs-tcompile ratio at
each shape**, not absolute µs.

| aspect                             | static @ seq=32                                  | dynamic @ seq=512                                                                                 |
|------------------------------------|--------------------------------------------------|---------------------------------------------------------------------------------------------------|
| reproducer decomposition           | 4 kernels                                        | 5 kernels (extra split producer)                                                                  |
| emitted CUDA (`compile --ir cuda`) | 383 lines, **3** if-guards, **no `seq_len` arg** | 1093 lines (**2.9×**), **54** if-guards, `int seq_len` in every signature                         |
| deplodock total                    | 5.4 µs (wp 6.4)                                  | 46.7 µs (wp 48.5)                                                                                 |
| eager / tcompile / **deplodock**   | 84 / 8 / **6** → **beats tcompile**              | 151 / 25 / **48** → **1.9× behind tcompile**                                                      |
| main MMA reduce                    | `k_sdpa_reduce_39b7dc` 2.2 µs, occ 50%, 32 regs  | `k_sdpa_reduce_77b0f0` 13.8 µs, occ 45.6%, **107 K conflicts** + a second reduce occ 24%, 72 regs |

The masking tax is concrete in the dynamic CUDA (absent in static, which bakes every shape as a constant): runtime
ceil-div grid decode `blockIdx.x / (((seq_len+31)/32)*16)`; nested boundary guards on **both** axes of the [seq×seq]
scores `if (a1*64+a4*64 < seq_len) { if (a2*64+a5*8 < seq_len) { if (…+_g < seq_len && …< seq_len) … } }`; a
padded-to-64 output slab `((seq_len+63)/64*64)`; TMA descriptors. The softmax's **posture flips** from beating
torch.compile (specialised) to 1.9× behind it (masked). The masking guards and the MMA-reduce codegen limits (Finding 3)
compound.

**Repro:**

```bash
deplodock compile dump-static/07_lowering_cuda.kernels/k_sdpa_reduce_39b7dc.torch.json  --ir cuda   # 3 guards, no seq_len
deplodock compile dump-dynamic/07_lowering_cuda.kernels/k_sdpa_reduce_042770.torch.json --ir cuda   # 54 guards, seq_len
```

**Known-limitation / future work (priority: MEDIUM).** A clean same-shape comparison (static **at seq=512** vs dynamic
at 512) would isolate masking from data-size; I skipped that 70-min re-tune because the dynamic-vs-tcompile@512 gap
(48 vs 25 µs) already isolates the regression without the seq confound. The real fix is the same as Finding 3 (fused
flash-style masked attention).

## Finding 6 — compile-budget bench failures (wasted search slots)

**Symptom.** `eval failures` (dynamic.db): 5 `bench_fail` rows, **all** "compile stage exceeded 2.0s budget" nvcc
timeouts (3.96 s–5.76 s); 4 of 5 are on `k_sdpa_linear_reduce_0045cd_xn` (the split P@V producer) with `SPLIT_CONE=True`
and large `FK`. Static.db had **0** bench_fail.

**Root cause.** The split-producer's large `FK` unroll blows past the tuner's 2 s per-variant compile budget
(`_tune_backend`'s `bench_wall_timeout`). These are wasted search slots, not deployment problems (the final pick is
fine). The dynamic graph hits them because the masked split producers carry more unrolled code than the static ones.

**Fix (priority: LOW).** Cap `FK` for split-producer enumeration, or raise the compile budget for the `_xn` family.

## Finding 7 — TMA transport declines the symbolic SDPA split producer (known limitation)

**Symptom.** The dynamic tune log repeats: "dropped un-lowerable candidate (StageBundle TMA: source
`scaled_dot_product_attention_reduce__xn_smem` pad must be empty, got (0, 0, 8)) — pruning branch, continuing search."

**Root cause.** The TMA staging pass (`tile/050_use_tma`) can't promote the masked SDPA split producer's padded smem
slab, so it correctly prunes those candidates. Search-time only (pruned, not crashed) — recorded because it narrows the
transport options for exactly the SDPA kernels that are already the bottleneck.

**Fix (priority: LOW, correct-by-design).** Noted for context; folds into the flash-attention work (Finding 3).

---

## Repro / artifacts

Work dir: `_tune/tune-model-qwen3-emb-l0-report/` (2.0 G, under gitignored `_tune/`).

- Tune logs: `tune-dynamic.log`, `tune-static.log`. Dumps: `dump-dynamic/`, `dump-static/` (each has the
  `*_lowering_cuda.kernels/*.torch.json` reproducers, `62_kernel_bench.json`, `kernels.html`).
- Tune DBs / priors: `dynamic.{db,prior.json}`, `static.{db,prior.json}` (point `DEPLODOCK_TUNE_DB` /
  `DEPLODOCK_PRIOR_FILE` at these to re-read).
- Triage: `triage/{variants_qkt,variants_pv,variants_softmax,prior_reach_db,knobs,failures}.txt`,
  `triage/{qkt_ab,pv_bench,softmax_bench,softmax_ncu,softmax_pad_ab,softmax_static_bench}.log`,
  `triage/softmax_{static,dynamic}.cu`. NCU CSV/JSON: `ncu-softmax/61_ncu_metrics.{csv,json}`.
- Serve logs: `serve-stock.log` (works), `serve-deplodock.log` (layer-0-prior crash),
  `serve-deplodock-wholemodel.log` (whole-model-prior crash — identical error), `serve-deplodock-coldprior.log`
  (34× slower). Whole-model tune: `tune-wholemodel-dynamic.log`, `dump-wholemodel-dynamic/`.

Compile-only repros (no GPU):

```bash
DEPLODOCK_PRIOR_FILE=…/dynamic-prior.json deplodock serve Qwen/Qwen3-Embedding-0.6B --dry-run     # Finding 1 crash
deplodock compile dump-dynamic/…/k_sdpa_reduce_042770.torch.json --ir cuda                        # Finding 5 guards
DEPLODOCK_TUNE_DB=…/dynamic.db deplodock eval variants --kernel k_sdpa_reduce                     # Finding 3 rank
DEPLODOCK_TUNE_DB=…/dynamic.db deplodock eval prior --dataset db                                  # Finding 4 reachability
```

GPU repros:

```bash
deplodock run --ir dump-dynamic/…/k_sdpa_reduce_042770.torch.json --bench --profile               # Finding 3 NCU
deplodock run --ir dump-dynamic/…/k_linear_sdpa_reduce_6bb11e.torch.json --bench                  # Finding 2 decomposition
deplodock compare _tune/.../dump-static _tune/.../dump-dynamic                                     # cross-mode diff
```

---

## Workflow notes

**Slow steps.**
- The two-level tune is **far** past the skill's "~10–20 min single-layer" estimate: **104.6 min dynamic, 65.8 min
  static** (6611 / 5655 variants). The cost is the SP-MCTS exploring structural forks — each terminal re-runs the full
  per-kernel inner search, and the SDPA split forks (`005_split_demoted` + `_xn` producers) multiply the kernel set
  (8 fused terminals, 10→13 kernels). *Proposal:* document a realistic per-layer wall on a transformer block (~1–2 h),
  and consider an outer-tree iteration cap or a `--max-terminals` flag so a "quick look" tune is possible. The 5-min
  estimate in the skill is wrong for any real decoder layer.
- The **whole-model** tune is worse: it ran **~4.5 h to 12 terminals without reaching `done`** and I stopped it with
  SIGINT (it checkpoints the prior incrementally + writes per-op rows on interrupt, so the accumulated evidence is
  intact). The whole model has more structural-fork sites than one layer, so the outer SP-MCTS tree is far larger (8
  terminals for layer-0 vs 12-and-counting whole-model). A `--max-terminals` / wall-clock cap is effectively mandatory
  for whole-model tuning; without it the run does not terminate in a practical window. (op-key dedup *does* work — 18
  unique kernels cover 422 positions — so the cost is outer-tree breadth, not 28× the kernels.)
- The whole-model dynamic `run --bench` (for the e2e headline) **failed**: "benchmark run stage exceeded 10.0s of GPU
  time — variant marked bench_fail", almost certainly the dispatch-bound op-by-op torch *reference* replay for the full
  310-weight model, not a deplodock blow-up. *Proposal:* skip / time-cap the torch reference for whole-model `--bench`,
  or bench deplodock-only with a note, so a whole-model e2e number is obtainable. (We fell back to the serve A/B + the
  layer-0 e2e.)

**Many-step detours.**
- **Reproducer attribution is misleading for SDPA (Finding 2).** It took a `run --ir … --bench` per kernel to discover
  the "80 µs" rows are 4–5-kernel re-lowerings whose matmul is 6.5 µs. *Proposal:* `62_kernel_bench.json` and
  `kernels.html` should carry the sub-kernel breakdown the `run --ir` knob table already prints, or flag rows where the
  reproducer re-lowers to >1 kernel. This is the single biggest "data existed only after manual drilling" item.
- **Reservoir -O3 vs reproducer -O3 looked like a 2–6× contradiction** until the decomposition explained it. An
  `eval variants` note like "reproducer re-lowers to N kernels; this -O3 row is the named kernel only" would have saved
  the confusion.

**Flakiness / environment.**
- The serve A/B needed `-- --gpu-memory-utilization 0.8`: vLLM's default 0.9 util wanted 28.19 GiB but only ~27 GiB was
  free (the ~3.7 GiB display baseline on a workstation 5090, plus a leaked ~4 GiB CUDA context from an earlier drill).
  *Proposal:* `deplodock serve` could default `gpu_memory_utilization` lower, or surface a clear "free X, need Y" hint
  instead of a raw vLLM `RuntimeError`. Also: the per-kernel drills leak GPU memory (orphan contexts) — worth a cleanup
  between GPU steps.

**Output friction.**
- `--ab` **silently skips un-promotable pins**: pinning the reservoir's `RING=2` configs printed "compile/bench of the
  pinned config failed (DEPLODOCK_RING=2 pinned but cannot fire: BUFFER_COUNT=2 not promotable …) — skipping its row"
  and continued. Easy to miss in a long table; the skip should be louder (or surfaced in a summary line).
- The plugin compile crash (Finding 1) only shows the real cause in the *separate* vLLM server log
  (`/tmp/deplodock-serve-*.log`), not in the `deplodock serve` stdout (which shows only "Engine core initialization
  failed"). *Proposal:* `deplodock serve` should tail the root-cause exception (it already prints "Log tail" but
  truncates above the `LoweringError`).

**Prior reports.** None existed (the reference `plans/qwen3-embedding-layer0-*-tune-findings.md` the skill cites were
deleted in `8727fbb5 delete executed plans`); this report was written fresh. The analysis CLI the skill leans on
(`eval variants/failures/prior/knobs`, `run --ab`, the `ncu compare` table, `compare`) all worked and carried the
triage — the main gap is the SDPA per-kernel attribution above.
