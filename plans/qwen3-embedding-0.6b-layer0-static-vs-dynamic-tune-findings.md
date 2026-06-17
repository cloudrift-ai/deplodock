# Qwen3-Embedding-0.6B layer 0 — static vs dynamic tune findings

**Status:** static tune complete and deployable-benched; dynamic tune **wedged** mid-search (reproducible
`HungKernelError → parent-wedge`) and was salvaged via the greedy deploy path. Both halves benched end-to-end and
per-kernel at -O3; static and dynamic compared on e2e and per-kernel; plus a whole-model **serving A/B** (deplodock
plugin vs vanilla vLLM).

**Run commands** (RTX 5090, sm_120, HEAD `a5a5690b`, ncu 2025.3.1, isolated DB/prior/cubin per run under
`_tune/tune-model-qwen3-l0-v2/`):

```bash
# static (shape-specialised, seq 512)
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512 --clean --bench --dump-dir static-dump
# dynamic (symbolic seq_len, benched at hint 512) — WEDGED; deployable numbers from greedy run instead:
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench --dump-dir dynamic-dump
deplodock run  Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench --dump-dir dyn-run-dump  # salvage
```

**Date:** 2026-06-16.

**Run stats**
- **Static:** 16 fused terminals, **16 070 benches**, 12 470 s (**3.46 h**), Spearman +0.99, **5670 ok / 4 bench_fail**.
`--bench` -O3 tables + `static-dump/62_kernel_bench.json` produced.
- **Dynamic:** **wedged** after ~65 min at **2938 ok / 1 bench_fail** — the bench process dead-locked (frozen DB, 0 %
CPU, idle GPU, no children) right after recording a `HungKernelError`. No `--bench` tables, no
`62_kernel_bench.json`. Deployable numbers come from a greedy `run --bench` (the deploy path does not hang — it
avoids the bench_fail variant), using the **partial** dynamic prior the wedged tune had trained (~2900 benches).

**Number-family disclaimer:** every latency below is the deployable **-O3** re-bench (CUDA-graph captured), *not*
the tune-DB `-O1` ranking numbers. Where `-O1` DB ranks are quoted (e.g. `eval variants`) they are flagged as
ranking-only.

**Dynamic measurement semantics:** the dynamic run is symbolic `seq_len`, **benched at the `DEFAULT_SEQ_HINT=512`**
(`--seq-len` only sizes trace inputs; it does not move the hint). The full-model table tiles the torch closures'
inputs to 512, so eager / tcompile / deplodock are one shape (`benched at seq_len=512 (symbolic hint)`). The
masked-tile boundary guards (`if (coord < seq_len)`) are part of the measured cost — that overhead vs a
shape-specialised static kernel is itself a finding (see Finding 4), not noise.

---

## Bench results

### End-to-end (seq 512, -O3, CUDA-graph captured, vs the real torch module)

| Config                         | eager | torch.compile | deplodock | vs eager | vs tcompile |
|--------------------------------|------:|--------------:|----------:|---------:|------------:|
| **Static** (shape-specialised) |   219 |           147 |   **267** |    0.82× |       0.55× |
| **Dynamic** (masked, hint 512) |   219 |           148 |   **217** |    1.01× |       0.68× |

Deplodock loses to torch.compile in **both** configs and only reaches eager parity in the dynamic case. Counter-
intuitively the **dynamic build is faster than the static one (217 vs 267 µs)** — the masked-K SDPA P@V *split*
deploys a cheaper attention tail than static's fork (Finding 3), outweighing the masked-tile overhead the reduces
pay (Finding 4).

### Per-kernel (-O3, reproducer-based: each kernel re-lowered from its `.torch.json` and benched vs eager / tcompile)

Sorted by deplodock µs; both columns are reproducer benches at seq/hint 512. `static` =
`static-dump/62_kernel_bench.json`; `dynamic` = greedy `run --ir <repro> --bench`. Layer op from each reproducer's
`.torch.json` coverage header.

| Layer op                          | kernel                   | eager | tc  |  **static depl** |  **dyn depl** | verdict                       |
|-----------------------------------|--------------------------|------:|----:|-----------------:|--------------:|-------------------------------|
| attn-out + o_proj + residual      | `k_linear_sdpa_reduce`   |    39 |  37 |         **98.0** |        **80** | **lose** 0.40× / 0.49× eager  |
| v_proj tail + SDPA P@V            | `k_sdpa_linear_reduce`   |    34 |  34 |         **96.6** |        **74** | **lose** 0.35× / 0.39× eager  |
| RoPE + QK^T scores (+mask)        | `k_sdpa_reduce`          |   148 |  25 |             53.1 |        **64** | beat eager, **lose to tc**    |
| post-attn RMSNorm + MLP gate/up   | `k_linear_mean_reduce`   |   120 |  57 |             47.1 |            57 | beat eager 2×, tie/lose tc    |
| k_norm (RMSNorm + rotated k)      | `k_mean_linear_reduce`   |   104 |  20 |             20.0 |            18 | **win** ~5×                   |
| q_norm (RMSNorm + rotated q)      | `k_mean_linear_reduce`   |    76 |  14 |             12.3 |            12 | **win** ~6×                   |
| MLP down (linear_6) + residual    | `k_linear`               |    27 |  25 |             27.2 |            26 | tie                           |
| q/v_proj (partial linear)         | `k_linear_reduce`        |    17 |  16 |             18.7 |            15 | ~tie                          |
| q/v_proj (partial linear)         | `k_linear_reduce`        |    10 |  10 |              9.8 |             9 | tie/win                       |
| input RMSNorm                     | `k_mean`                 |    64 |   4 |              1.9 |             2 | **win** ~33×                  |

**Dominators of the deplodock total** (where the loss lives): the **attention tail** — `k_linear_sdpa_reduce`
(o_proj) and `k_sdpa_linear_reduce` (P@V) — is ~150–195 µs of solo-window time across the two configs and is the
only place deplodock loses to *eager*. The norm/reduce kernels (`k_mean`, `k_mean_linear_reduce`,
`k_linear_mean_reduce`) win 2–33×. So Deplodock wins every reduction and loses the attention compute; the attention
tail is what holds the e2e below torch.compile.

> Note on the dynamic P@V split: the reproducer benches `k_sdpa_linear_reduce` as a single fused op (74 µs), but in the
> full model `005_split_demoted` splits it into `_xna` (softmax producer) + `_xnb` + masked-K mma consumer that together
> deploy at **~44 µs** (`dyn-run-dump` kernel table). The deployed split is cheaper than the fused reproducer — the split
> is the real artifact.

### Serving A/B — deplodock plugin vs vanilla vLLM (whole model)

`deplodock serve Qwen/Qwen3-Embedding-0.6B --bench` (deplodock kernels behind vLLM `/v1/embeddings`) vs the same with
`--stock` (vanilla vLLM). Both: `vllm bench serve --backend openai-embeddings`, **200 prompts, random-input-len 512,
max-concurrency 32, seed 0**. The plugin greedy-picks from the dynamic prior this tune trained (layer-0 configs, applied
structurally to all 28 layers).

| Backend (serve)              | req/s     | tok/s       | Mean E2EL | Median E2EL | P99 E2EL | duration |
|------------------------------|----------:|------------:|----------:|------------:|---------:|---------:|
| **Stock vLLM**               | **232.2** |  118 871    |  128.5 ms |    124.3 ms | 208.1 ms |   0.86 s |
| **Deplodock plugin**         | **103.6** |   53 058    |  290.1 ms |    299.1 ms | 367.4 ms |   1.93 s |

**The deplodock plugin serves embeddings ~2.24× slower than vanilla vLLM** (103.6 vs 232.2 req/s; 290 vs 128 ms mean
latency). **The per-kernel norm/reduce wins did *not* translate to served throughput** — consistent with the layer-0
picture: the attention tail (P@V + o_proj, Finding 2) dominates and loses ~2–2.5×, and vanilla vLLM serves it with flash
attention + cuBLAS + a compiled/CUDA-graph path the plugin cannot use.

**Comparison caveats (state when citing this number — it is not fully apples-to-apples):**
- **Mode asymmetry (inherent):** the plugin *requires* `--enforce-eager`, so vLLM's per-layer scaffolding runs eager
  (the deplodock kernels themselves run under the plugin's own captured graph); **stock ran vLLM's default
  `VLLM_COMPILE` + CUDA-graph path**. This is the honest "what you would actually deploy each way" comparison, but part
  of the 2.24× is the eager-scaffolding dispatch, not the deplodock kernels alone.
- **Memory asymmetry (forced):** an orphaned ~3.6 GB CUDA context (leaked by the SIGKILL of the wedged tune, Finding 1)
  shrank the GPU budget. To fit the plugin's separate cupy buffers I served it at `--gpu-memory-utilization 0.60` +
  `DEPLODOCK_SERVING_CAPTURE_CAP=512`; stock ran at 0.80. For prefill-only embeddings (no decode KV pressure) at 200
  reqs this barely affects throughput, but it is an asymmetry. A clean re-run on a freshly-reset GPU would tighten it.
- **Scope:** the plugin serves the **whole 28-layer model** with **layer-0-tuned** configs (structural transfer); the
  tune only saw layer 0. The served result is therefore the full-model consequence of the per-kernel picture above.

Repro:
```bash
DEPLODOCK_PRIOR_FILE=…/dynamic-prior.json DEPLODOCK_SERVING_CAPTURE_CAP=512 \
  deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --num-prompts 200 --random-input-len 512 \
    --max-concurrency 32 --bench-seed 0 -- --gpu-memory-utilization 0.60
deplodock serve Qwen/Qwen3-Embedding-0.6B --bench --num-prompts 200 --random-input-len 512 \
    --max-concurrency 32 --bench-seed 0 --stock -- --gpu-memory-utilization 0.60
```

---

## Finding 1 — Dynamic tune wedges: `HungKernelError → parent dead-lock` (blocks the clean dynamic tune)

**Severity: blocker.** This is why there is no clean dynamic `--bench`.

**Symptom.** ~65 min into the dynamic two-level tune the process froze: DB row count stuck at 2939, DB mtime frozen,
instantaneous CPU 0.0 %, GPU idle (0 %/20 W), no bench-worker children, parent in `Sl` sleep — no recovery (observed
frozen 30 min, deterministic across the prior session too).

**Evidence** (`deplodock eval failures` on `dynamic.db`):
```
1 bench_fail rows (beside 2917 ok):
  k_sdpa_linear_reduce_a76a28 — HungKernelError("kernel 'k_sdpa_linear_reduce_a76a28' did not complete within 1000 ms")
    shared knobs: MMA=mma_m16n8k16_f16, BK=64, OVERHANG=['a1','a3'], WM=2, WN=1, SPLIT_CONE=True, FN=4, GROUP_M=8
```
The hung variant is the **warp-tier masked-K mma P@V** (`MMA=mma_m16n8k16_f16`, masked `OVERHANG=['a1','a3']`,
`BK=64`). The per-variant containment (`bench_wall_timeout_s` → SIGKILL the worker, pin `bench_fail @ 2e6 µs`) fires
correctly — the variant *is* recorded as bench_fail — but the **parent process wedges immediately afterward**: the
worker SIGKILL leaves the parent's CUDA stream / asyncio bench loop in a state it never returns from.

**Root cause / hypothesis.** This is the prior session's wedge (RESULTS.md, commit 513a5df5) **recurring on HEAD
`a5a5690b`** — `#245` ("dtype-aware TMA ring-slot alignment — fix #244 fp16 reduction-slab wedge") and `#246` ("kill
the masked-K consumer's ldmatrix bank-conflict storm") did **not** fully fix it. The bug is not the slow kernel per
se (the 1000 ms timeout catches it); it is that **post-SIGKILL recovery doesn't restore the parent's bench stream**,
so the next `await` never completes. The deploy path is unaffected — greedy `run`/`compile` never benches the pinned
variant, so `run --bench` produced the full dynamic table without hanging.

**Repro (deterministic).**
```bash
deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --clean --bench   # wedges ~once the
                                                                                           # masked warp P@V variant
                                                                                           # with BK=64 is benched
```

**Suggested fix (priority: high).** Make HungKernel containment *recover the parent*: after SIGKILL-ing a hung bench
worker, tear down and recreate the parent's CUDA bench stream / context (or run every bench in a worker that owns
its own context so a SIGKILL can never poison the parent). Separately, the warp masked-K P@V enumerator should
**gate out** `BK=64` for the masked-`OVERHANG` mma P@V (the hang is a codegen pathology in that exact config) — find
the enumeration in `lowering/tile/_atom.py` / `010_partition_loops.py::_build_split_body_warp` and the masked-K
branch. A `bench_fail` variant that *also* wedges the tuner is far worse than one cleanly skipped.

---

## Finding 2 — SDPA attention tail loses ~2–2.5× to eager (cuBLAS + flash) — codegen quality, both configs

**Severity: high (the e2e gap).** This is the only place deplodock loses to *eager*, and it is what keeps both
configs below torch.compile.

**Symptom.** P@V (`k_sdpa_linear_reduce`) 96.6 µs static / 74 µs dynamic vs eager 29–34 µs (0.35–0.39×); o_proj
(`k_linear_sdpa_reduce`) 98 µs static / 80 µs dynamic vs eager 39 µs (0.40–0.49×).

**Evidence** (`ncu compare`, dynamic P@V split reproducer, `ncu-dyn-pv/61_ncu_metrics.json`):

| side       | kernel                                      |  dur µs |  occ% |   sm% |    dram% |     fma% |  ld bank-cnflct |  regs |
|------------|---------------------------------------------|--------:|------:|------:|---------:|---------:|----------------:|------:|
| deplodock  | `k_sdpa_linear_reduce_*_xna` (softmax→gmem) |    15.7 |  83.5 |  34.4 | **60.6** |     16.9 |               0 |    39 |
| deplodock  | `k_sdpa_linear_reduce_*_xnb`                |     4.8 |  23.8 |  12.8 |     15.3 |      6.6 |               0 |    32 |
| deplodock  | `k_sdpa_linear_reduce_*` (mma consumer)     |    27.9 |  49.2 |  37.1 |     27.4 | **14.3** |             360 |    56 |
| **torch**  | `cutlass_80_tensorop_f16` (gemm)            |    13.3 |   8.3 |  25.1 |     14.2 |      0.3 |               0 |    88 |
| **torch**  | `flash_fwd_splitkv_kernel`                  |    24.6 |   8.4 |  16.3 |     15.9 |      2.9 |               0 |   206 |
| **torch**  | `flash_fwd_splitkv_combine`                 |    11.5 |  65.7 |  34.7 |     47.2 |      8.3 |               4 |    44 |

**Root cause.** Two reads: (a) the **`005_split_demoted` materializes the softmax/P to gmem** — `_xna` is **dram
60.6 %**, memory-bound, writing the attention probabilities out for the consumer to re-read. Flash attention keeps
softmax+P@V **on-chip** (the split-kv kernels never spill P). The gmem round-trip is pure overhead deplodock pays
and flash does not. (b) The mma consumer runs at **fma 14.3 %** — the tensor cores are <15 % utilised; the kernel is
not compute-bound on the mma, it is stalling (on the gmem-materialized P read and the masked-K serial loop). Net:
deplodock's split (~48 µs) vs flash+gemm (~36 µs + the gemm overlaps) ≈ 1.3–2×.

**Repro.**
```bash
deplodock run --ir dyn-run-dump/07_lowering_cuda.kernels/k_sdpa_linear_reduce_a76a28.torch.json --bench --profile
```

**Suggested fix (priority: high, hard).** A flash-style **fused** symbolic-K attention (softmax + P@V on-chip, no
gmem spill) is the real fix and is called out as future work in CLAUDE.md ("flash-style fused symbolic-K attention
remains future work"). Shorter term: keep P in smem across the split (avoid the `_xna` gmem write) where the
head/seq tile fits.

---

## Finding 3 — Static's attention tail is *slower* than dynamic's (fork-selection, not intrinsic)

**Severity: medium (explains the 267-vs-217 inversion); variance caveat applies.**

**Symptom.** Same op, static deploys slower: P@V 96.6 (static) vs 74 (dynamic); o_proj 98 (static) vs 80 (dynamic).
The masked-K mma *split* the dynamic path is forced into (symbolic seq ⇒ masked tile ⇒ `005_split_demoted`) happens
to land a better-pipelined consumer than the fork static's greedy pick chose.

**Evidence** (`eval variants --kernel k_sdpa_linear_reduce` on `static.db`): the base P@V consumer picks **rank
1/2** and the o_proj base picks **rank 1/15** — so static is *not* losing to a search miss on those; it is deploying
the rank-1 fork and that fork is still ~97 µs at -O3. The one genuine search miss is the **softmax `_xn` producer**:
`pick: rank 74/321, 1.38× of best <-- misses best` (greedy picks a `BR=64` cooperative reduce; the measured best is
a `BN=32, BR=1` thread tile). That is the only prior-reachability shortfall in the tail.

**Diagnostic that separates the hypotheses.** If re-tuning the static P@V reproducer with higher patience moves the
pick to a sub-90 µs fork, it is search/prior; if the rank-1 fork stays ~97 µs, it is codegen (Finding 2) and the
static/dynamic gap is just which masked-vs-clean fork each path is allowed to pick. The deployed-split note above
(dynamic ~44 µs in the full model) says the split is genuinely the better structure.

**Caveat.** Scalar-/fork-tier picks swing run-to-run with prior reachability. The per-kernel reproducer rows are the
stable signal; re-run `deplodock compare static-dump dyn-run-dump` before treating 267-vs-217 as a fixed delta.

---

## Finding 4 — Masked-tile guard overhead: dynamic reduces pay ~10–20 % vs shape-specialised static

**Severity: low–medium (the cost of being deployable).**

**Symptom.** The reduce/QK^T kernels are *slower* dynamic than static: RoPE+QK^T `k_sdpa_reduce` 53→64 µs; post-attn
`k_linear_mean_reduce` 47→57 µs. These run a `ceil_div` grid over the symbolic extent plus per-element `if (coord <
seq_len)` boundary guards that the static seq-512-specialised kernel omits.

**Root cause.** Expected and correct-by-design: a symbolic-axis masked tile cannot assume divisibility, so it
carries the boundary `Cond`. The overhead (~10–20 % on these reduces) is the price of one kernel that runs at any
seq_len vs a kernel hard-specialised to 512. Recorded here because it is a real component of the dynamic e2e, and
because it is *outweighed* by the attention-tail win (Finding 3) — net dynamic still beats static.

`k_sdpa_reduce` additionally loses to **torch.compile** (64 vs 25 µs) for a separate reason — see Finding 5.

---

## Finding 5 — `k_sdpa_reduce` (RoPE + QK^T): 4.2 M smem bank conflicts → loses to torch.compile

**Severity: medium.**

**Symptom.** `k_sdpa_reduce` beats eager (53–64 vs 148 µs) but loses badly to torch.compile (25 µs).

**Evidence** (`ncu`, `ncu-dyn-pv/61_ncu_metrics.json`): `k_sdpa_reduce_6874a2` — dur 66.8 µs, occ 61 %, sm 60 %, fma
43.7 %, **`l1tex…shared_op_ld` bank conflicts = 4 207 840** (4.2 M shared-load bank conflicts; the next-worst kernel
has 360). A bank-conflict storm on the QK^T smem reads — the score-tile smem layout collides across the warp.

**Root cause / fix (priority: medium).** smem layout on the QK^T consumer. Try `PAD_SMEM` / `PERMUTE_LANES` on the
masked QK^T tile; the conflict count is the smoking gun (cf. the `#246` "ldmatrix bank-conflict storm" work on the
*other* SDPA kernel — the same class of bug lives here on `k_sdpa_reduce`). A/B:
```bash
deplodock run --ir dyn-run-dump/07_lowering_cuda.kernels/k_sdpa_reduce_77b0f0.torch.json --bench \
    --ab "PAD_SMEM=1" --ab "PERMUTE_LANES=1"
```

---

## Finding 6 — nvcc "unused `warp` variable" bench_fail cluster (static scalar cooperative-reduce)

**Severity: low (contained, but wastes search slots).**

**Evidence** (`deplodock eval failures` on `static.db`): 4 bench_fail rows, all `k_mean_linear_reduce_125c9c`,
identical error — nvcc rejects the kernel because `int warp = threadIdx.x >> 5;` is declared but never referenced
(warning `#177-D` escalated to error). Shared knobs across all four: `MMA=0` (scalar tier), `SPLIT_CONE=True`,
`BN=8, BM=1, FM=2`, `BR ∈ {8,16,32}` (cooperative reduce). So: the **scalar cooperative-reduce codegen emits a dead
`warp` declaration** for `SPLIT_CONE` variants where the warp index is unused, and the build treats unused-variable
as an error.

**Fix (priority: low).** Either don't emit the `warp` decl when unused, or add `(void)warp;` / `-diag-suppress 177`
to the cooperative-reduce codegen. Reproducible; compile-only repro (pin the cluster's shared knobs):
```bash
DEPLODOCK_KNOBS="MMA=0,SPLIT_CONE=1,BN=8,BM=1,FM=2,BR=16,BK=4" \
  deplodock compile static-dump/.../k_mean_linear_reduce_125c9c.torch.json --ir cuda
```

---

## Repro / artifacts

Work dir: `_tune/tune-model-qwen3-l0-v2/` (gitignored). HEAD `a5a5690b`, RTX 5090 sm_120.

- Static tune log + dump: `static-tune.log`, `static-dump/` (`62_kernel_bench.json`, `kernels.html`, reproducers under
`*_lowering_cuda.kernels/`).
- Dynamic: `dynamic-tune.log` (wedged), `dynamic.db` (partial, 2938 ok / 1 fail), `dynamic-prior.json` (partial prior),
`dynamic-run-deploy.log` (greedy e2e + kernel table), `dyn-run-dump/` (reproducers), `dyn-perkernel2.log`
(per-kernel eager/tc/deplodock), `ncu-dyn-pv/61_ncu_metrics.{csv,json}` (NCU compare).

```bash
# e2e + per-kernel, both configs
deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512 --bench --bench-backends eager,tcompile,deplodock
deplodock run Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench --bench-backends eager,tcompile,deplodock
# triage
DEPLODOCK_TUNE_DB=…/static.db  deplodock eval failures
DEPLODOCK_TUNE_DB=…/dynamic.db deplodock eval failures      # the HungKernel wedge variant
DEPLODOCK_TUNE_DB=…/static.db  deplodock eval variants --kernel k_sdpa_linear_reduce --top 6
# cross-run per-kernel diff (after any fix/re-tune)
deplodock compare static-dump dyn-run-dump
```

---

## Workflow notes

For whoever maintains the deplodock CLI and the `tune-model` skill. Friction logged as it happened.

1. **Stale half-applied working tree poisoned a 1.5 h tune (the biggest time sink — ~4 h lost).** The first static tune
crashed at 15:44 with `AttributeError: 'Dim' object has no attribute 'ceil_div'`. This was **not** a code bug: the
tune launched at 14:07 against a working tree where `#247` (`Dim.ceil_div`) was being authored —
`partition_loops.py` already *called* `.ceil_div` but `dim.py` did not yet *define* it. The long-lived process
cached that inconsistent import and died when it first hit a masked tile. Confirmed via git reflog + file mtimes
(`dim.py` written 15:52, after the 14:07 launch) and 263 passing masked-tile/partition tests on the consistent HEAD.
*Proposed:* `tune`/`run` should **stamp the git SHA + dirty flag into the run log header and the dump**, and warn if
the tree is dirty at launch — a long tune against a half-edited tree is a silent trap. (Cost: ~4 h — the crashed
static tune + diagnosis + a full re-run.)

2. **The dynamic-tune wedge has no automatic detection (Finding 1).** It froze for 30 min before I caught it by polling
DB mtime. *Proposed:* a watchdog in the tuner — if no `perf` row commits for N× the bench timeout, abort with a
clear "tuner wedged after HungKernel on <kernel>" instead of hanging forever. This would have turned a 30-min frozen
poll into an immediate, actionable failure.

3. **Progress is invisible in a redirected log.** The live progress bar is tty-gated, so a backgrounded `tune > log`
writes essentially nothing for hours; I had to poll `sqlite3 … count(*) FROM perf` + DB mtime to tell progress from
a wedge. *Proposed:* a `--progress-file PATH` (or `DEPLODOCK_PROGRESS_JSON`) that appends `{ts, benches, best_us,
current_kernel}` every K benches, so non-tty runs are monitorable without poking the DB.

4. **`run --ir … --bench` has no machine-readable per-kernel output.** Static's per-kernel eager/tc/deplodock came free
from `tune --bench`'s `62_kernel_bench.json`, but the dynamic tune wedged before that file is written, so I had to
loop `run --ir <repro> --bench` over 10 reproducers and scrape stdout — and my first scrape was lossy
(case-sensitive `Eager` vs `eager`, data rows lack the keywords), forcing a second ~15-min bench pass. *Proposed:*
give `run --bench --dump-dir` the same `62_kernel_bench.json` writer `tune --bench` has (or a `--bench-json PATH`).
It would have made `deplodock compare static-dump dyn-run-dump` work directly and saved a re-run.

5. **Slow cold tune.** Static took **3.46 h** for 16 070 benches from an empty prior (`--clean`). Mostly nvcc-compile-
bound (GPU often <5 % util while cicc churns on big unrolled tiles). Not a bug, but a single-layer "clean tune" is
~3.5 h here, not the "~10–20 min" the skill prerequisites suggest — worth correcting the estimate in the skill for
this model class. A warm-prior re-tune (no `--clean`) would be far faster.

6. **`eval failures` shared-knobs output is excellent** — it pinned the nvcc `warp` cluster (`MMA=0, SPLIT_CONE`) and the
HungKernel variant (`MMA=mma_…, BK=64, OVERHANG=['a1','a3']`) in one line each, no log grepping. Kept as-is.

7. **Serving the plugin needed two manual workarounds, both traceable to other findings.** (a) The orphaned ~3.6 GB
CUDA context leaked by the wedge's SIGKILL (Finding 1) pushed vLLM's default `gpu-memory-utilization 0.92` over the
free-memory budget — both deplodock *and* `--stock` failed engine init with `ValueError: Free memory … less than
desired`. (b) Once past that, the plugin OOM'd in `program.py::_allocate` (cupy `zeros`) because it pre-allocates its
captured-graph buffers for the **default `--max-model-len 4096`** (~29.6 GB), separate from vLLM's pool. Fix was
`DEPLODOCK_SERVING_CAPTURE_CAP=512` + `--gpu-memory-utilization 0.60`. *Proposed:* (i) `deplodock serve` should default
`CAPTURE_CAP` to the bench's `--random-input-len` (or warn when `max-model-len` × buffers exceeds free VRAM with a
"lower CAPTURE_CAP / max-model-len" hint) instead of OOMing deep in cupy; (ii) the wedge fix (Finding 1) also removes
the leak that started this — a SIGKILLed worker should not strand a context. Cost: 2 failed serve attempts (~10 min)
before the cap/util fix.

(No prior `*-tune-findings.md` workflow-notes section existed to check against — this is the first for this model on
this work dir; the older `_tune/tune-model-qwen3-l0-staticdyn/RESULTS.md` is scratch, not a report.)
