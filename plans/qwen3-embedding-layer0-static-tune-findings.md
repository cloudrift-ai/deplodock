# Qwen3-Embedding-0.6B layer 0 — STATIC-shape tune findings + static-vs-dynamic side-by-side (2026-06-15)

Status: **clean static (shape-specialised, seq_len=512) tune of layer 0 on branch `fix/hoist-compute-literal-index`
(off `main` @ `f87098e8`). The clean tune first crashed on a lowering-contract bug — a sibling-cell-fused
slab the single-Write materializer can't represent — that aborted the WHOLE tune with no per-variant
containment
(finding 1, fixed on this branch: a descriptive `LoweringError` + a tune-only `Run.drive` containment that drops the
un-lowerable fork and keeps searching). With the fix the tune completes (16 terminals, 3 un-lowerable forks cleanly
dropped) and the deployed static layer runs `168 µs / 1.29x eager` at seq 512 — it BEATS eager, where the deployable
dynamic (`--dynamic seq_len@x:1`) twin runs `361 µs / 0.62x eager` on the SAME cooled card. Static is 2.15x faster
end-to-end. The kernel SET is identical between the two (same 10 kernels, same SDPA P@V split), so the gap is NOT
structural — it is the per-kernel masked-tile tax the symbolic path pays: a standalone 148 µs materialised softmax
(static: 14 µs), the fp16-half2-window lockout on symbolic-K scalar matmuls, and the masked-K zero-fill + SYNC-transport
pin on the P@V mma. Every one of these is gated on `*_symbolic` in `010_partition_loops.py` — they fire only under
`--dynamic`.**

> **Update (#245).** One transport item above has since been fixed: the masked-tile tax included routing the dynamic
> matmuls onto cp.async (TMA was wholesale-declined for symbolic graphs). #245 enables **TMA for dynamic M-masked
> matmuls with a static innermost dim** (o_proj, q/v_proj, MLP-down), closing the o_proj NCU gap (47.0 → 29.2 µs, ≈
> static 28.6 µs; 6.8M → 0 bank conflicts) and moving whole-layer dynamic e2e **361 → 337 µs (0.62 → 0.65x eager)**.
> The attention path (softmax materialisation, QK^T, masked-K P@V) is unchanged and remains the gap. Tables below carry
> the post-#245 numbers inline (marked); see the finding-2 and finding-4 notes.

- Command: `deplodock tune Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512 --clean --bench --dump-dir <dir>/dump`
  (note: **`--seq-len 512`** — static kernels are shape-specialised, so they are tuned/benched at the SAME 512 the
  dynamic report's symbolic hint uses; tuning at the default seq 32 would shrink attention ~256x and make the
  side-by-side meaningless).
- Hardware: RTX 5090 (sm_120), ncu 2025.3.1 (perf counters permitted). Branch `fix/hoist-compute-literal-index`.
- Run stats: tune wall **12,732.5 s** (~3.5 h), **16 fused terminals**, **3 un-lowerable candidates dropped** (the
  finding-1 containment — `[tune] dropped un-lowerable candidate (LoweringError: …)`), **5,478 ok / 47 `bench_fail`**
  `perf` rows, all CUDA-graph captured; prior 16,001 benches, post-warmup Spearman **+0.99**, 97% of post benches ≥2x
  the running best.
- DB / prior: **isolated** in the work dir (`_tune/tune-model-qwen3-emb-l0-static/static.db`, `…/static-prior.json`) so
  the clean static tune does NOT clobber the dynamic tune's DB/prior at the default path — the dynamic-side `eval` /
  `run` comparison reads `~/.cache/deplodock/{autotune.db,prior.json}` (7,215 dynamic rows, intact).
- **Number families**: bench tables / reproducer runs / NCU below are **-O3** (deployable, CUDA-graph captured);
  tune-DB latencies quoted for ranking context are `-Xcicc -O1` (ranking signal only). NCU durations run at locked base
  clock — **compare ratios, not absolutes**.
- **Shape semantics**: this is the **static / shape-specialised** configuration (no masked-tile boundary guards, full
  static K, fp16 windows allowed) — NOT the deployable dynamic-serving artifact. It exists here as the apples-to-apples
  baseline for "what does the symbolic path cost", benched at the same seq 512 the dynamic twin uses. The deployable
  dynamic numbers are `plans/qwen3-embedding-layer0-dynamic-tune-findings.md` (2026-06-14, post-#243).
- **Single-layer scope**: no servable artifact, so no `deplodock serve` A/B (per the skill — single-layer has no served
  e2e). The deliverable is the per-kernel table + the static-vs-dynamic contrast.

## Bench results (-O3, CUDA-graph captured)

End-to-end (layer 0, seq 512), both from a fresh `run --bench` on the **same cooled card** this session (eager ~220 µs
both ways — the honest static-512 torch number):

```
Backend          static (this run)        dynamic (--dynamic seq_len@x:1)
---------------------------------------------------------------------------
Eager PyTorch     217 us   1.00x           223 us   1.00x
Deplodock         168 us   1.29x           361 us   0.62x      (2.15x slower than static)
                  (whole-program e2e 168.2 / TOTAL 165.0)   (e2e 361.4 / TOTAL 359.8)
```

The tune's own `--bench` full-model table did NOT print — its captured full-model bench failed with a TMA descriptor
rank mismatch (finding 5); the deployable e2e above is the separate cooled `run --bench`, which picks non-TMA variants
and succeeds. Use 168 µs as the deployed static number.

Per-kernel reproducer bench at seq 512 (both from each report's `tune --bench`, sorted by static deplodock µs;
layer-op labels read off each kernel's `.torch.json` provenance). The attention reproducers (`k_*sdpa*`) re-fuse the
upstream cone, so their `deplodock` µs is a slice-set total, not the deployed kernel alone — the clean per-kernel
attention attribution is the NCU table in finding 2.

| Kernel                 | Layer op                                                  | eager | tcompile | **static** | dynamic |  static vs dyn |
|------------------------|-----------------------------------------------------------|------:|---------:|-----------:|--------:|---------------:|
| `k_linear_sdpa_reduce` | attn-out reshape + o_proj (linear_3) + residual           |    39 |       37 |     **78** |     213 |    2.7x faster |
| `k_sdpa_linear_reduce` | SDPA P@V split (v_proj + softmax-normalise + P@V mma)     |    29 |       29 |     **70** |     203 |    2.9x faster |
| `k_sdpa_reduce`        | RoPE(q,k) + QK^T scores + softmax-stats                   |   148 |       25 |     **45** |     162 |    3.6x faster |
| `k_linear_mean_reduce` | post-attn RMSNorm + MLP gate+up (linear_4/5) + SiLU·up    |   119 |       57 |     **47** |      58 |    1.2x faster |
| `k_linear`             | MLP down (linear_6) + residual                            |    26 |       26 |     **26** |      44 |    1.7x faster |
| `k_mean_linear_reduce` | q_norm RMSNorm + rotated-q producer (xna)                 |   105 |       20 |     **18** |      27 |    1.5x faster |
| `k_linear_reduce`      | v_proj matmul (linear_1)                                  |    14 |       14 |     **15** |      27 |    1.8x faster |
| `k_mean_linear_reduce` | k_norm RMSNorm + rotated-k producer (xnb)                 |    76 |       14 |     **12** |      18 |    1.5x faster |
| `k_linear_reduce`      | q_proj matmul (linear)                                    |    10 |       10 |      **9** |      17 |    1.9x faster |
| `k_mean`               | input RMSNorm                                             |    64 |        4 |      **2** |       2 |           same |

Same 10 kernels in the same fusion/split structure as the dynamic report — the SDPA P@V split (`005_split_demoted`)
fires for static-K too. The dominators by static deplodock µs are the three attention reproducers
(`k_linear_sdpa_reduce` 78 + `k_sdpa_linear_reduce` 70 + `k_sdpa_reduce` 45 ≈ 193 µs of slice-set total, ~80% once
de-duplicated against the deployed total) — and every one is 2.7–3.6x faster than its dynamic twin. The linears are
1.5–1.9x faster; the RMSNorms are at parity (already memory-bound, no masked-tile content).

> **`k_linear_sdpa_reduce` — updated post-#245 (TMA for dynamic M-masked matmuls).** This dynamic column is the
> *original* cp.async
> reproducer totals (re-fused slice-set, so the row is dominated by the unchanged upstream softmax — that is why the
> total stays ~2.7x even though the o_proj kernel itself was fixed). The o_proj *kernel* now stages via TMA on the
> dynamic path: its solo reproducer window is **34.4 µs (cp.async) → 19.6 µs (TMA)** and its clean NCU is **47.0 →
> 29.2 µs, ≈ the static 28.6 µs** (finding 2 note). The other two attention rows (`k_sdpa_linear_reduce` P@V,
> `k_sdpa_reduce` QK^T) are unchanged — still cp.async (P@V is masked-K; QK^T has a symbolic-N inner axis), so their
> 2.9x / 3.6x gaps stand. The matmul-only kernels below (`k_linear` MLP-down, `k_linear_reduce` q/v_proj) also moved to
> TMA on the dynamic path (per-kernel ~1.6–1.8x), shrinking their dynamic numbers toward the static ones.

## Finding 1 — a sibling-cell-fused hoisted-compute slab aborted the whole static tune; fixed with a descriptive `LoweringError` + per-variant `Run.drive` containment (fixed on this branch)

**Symptom.** The first clean static seq-512 tune died ~30 min in with no traceback containment — one search-explored
fork raised mid-lowering and took the entire tune with it:

```
AttributeError: 'Literal' object has no attribute 'name'
  …/lowering/kernel/_stage_expand.py:398, in compute_phase_info: cache_axes = tuple(axis_map[v.name] for v in write.index)
```

**Evidence.** Instrumenting `compute_phase_info` caught the exact bundle (and a second/third like it):

```
output='linear_1_reduce_smem_position_embeddings_0_smem_fused'
index=(Literal(value=3), Var('a2'), Var('a4'))   axis_map=('a3','a2','a4')
```

The hoisted-compute Write's index — built at `tile/030_hoist_invariant_compute.py:288` as
`tuple(Var(ax.name) for ax in fused_cache_axes)`, i.e. all `Var`s — has had its **first** entry σ-collapsed to a
constant `Literal(3)`. `StageBundle.nested()` (`ir/tile/ir.py:1348`) exposes the `compute` template to generic index
substitution, so `012_fuse_sibling_register_cells` (which co-fills one smem slab from N sibling register cells) pins
the cell's `a3` coordinate to a constant in each sibling's self-describing Write. `compute_phase_info` /
`emit_compute_phase` (`_stage_expand.py`) derive ONE iteration domain + slab shape from a single Write, so they cannot
represent that multi-sibling fill (the slab needs `a3`'s full extent for the decl, but this bundle must iterate only
the non-collapsed `a2,a4` and write its pinned `a3` slice). Greedy `compile`/`run` never pick this fork (greedy at seq
512 compiles fine), so it is a tuner-only dead end.

**Root cause — two stacked defects (both the static-reference report's finding 1 pattern, recurring):**

1. **Proximate**: `compute_phase_info` assumed every Write-index entry was a cache-axis `Var` and indexed `axis_map`
   by `.name`, throwing an opaque `AttributeError` on the collapsed `Literal`.
2. **Containment**: a *rewrite/lowering* exception in the inner search had no per-variant containment — bench failures
   become `bench_fail` rows, but an exception while ADVANCING the lowering of one candidate propagated out of
   `Run.drive`'s generator and aborted the whole tune. `LoweringError`'s own docstring already says the tune
   fork-pruning path treats an un-lowerable branch as a legitimate dead end — `drive` just didn't apply that to
   *exceptions*, only to the `validate(ctx)`-filtered case.

**Fix (this branch).**

- `_stage_expand.py:compute_phase_info` now detects a non-cache-axis index entry and raises a descriptive
  `LoweringError` naming the slab, the index, and the cause ("a sibling-cell-fused slab fill the single-Write
  materializer can't represent") instead of an opaque `AttributeError`.
- `pipeline.py:Run.drive` wraps the per-candidate `_step` (the tune-only search driver; greedy uses the deterministic
  `Run.resolve`, untouched and still loud): an un-lowerable lowering exception drops the candidate's subtree, logs a
  `[tune] dropped un-lowerable candidate (…)` warning, increments `Run._dropped_candidates`, and continues the search.
  The `pop()` already decremented the node's `live` count, so a dropped candidate is bookkeeping-identical to a
  dead-end terminal — no wedge, and if every fork of an op is un-lowerable the run still ends cleanly. The terminal
  count line now reports `N un-lowerable candidate(s) dropped`.
- Tests: `tests/compiler/pipeline/test_lowering_error_guardrail.py` gains 3 — `compute_phase_info` raises a
  `LoweringError` on a collapsed index (and still recovers a clean all-`Var` index); a *raising* lowering pass
  propagates loudly under greedy `Pipeline.run`; and is contained (no raise, branch pruned, warning logged) under tune.

**Repro.** Greedy is unaffected: `deplodock compile Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512` succeeds.
The crash was tuner-only and needed default patience (50) to reach the fork; with the fix the tune logs three
`dropped un-lowerable candidate` lines (slabs `…position_embeddings_0…`, `…_1…`, `…_0…_1…`, all
`index=(Literal(3), Var('a2'), Var('a4'))`) and completes.

**Suggested follow-up (low — the fork is not deployable).** If the sibling-cell-fused hoisted-compute producer is ever
wanted as a real tuned variant, the materializer needs to separate "slab shape" (full extents, all siblings) from
"this bundle's iteration domain" (non-collapsed axes only) and co-emit the N sibling Writes into one slab. Until then,
pruning it is correct: greedy never picks it and the tuner has cheaper equivalents.

## Finding 2 — WHY the dynamic path is slower (the headline): the masked-K split forces a standalone 148 µs materialised softmax; static's is a 14 µs normalise producer

**Symptom.** The single largest contributor to the 361→168 µs static-vs-dynamic e2e gap is the **softmax**. NCU
attribution of the static attention chain (one `ncu` capture, deplodock `k_*` + torch reference kernels side by side,
locked base clock — `run --ir k_linear_sdpa_reduce_94ab75.torch.json --bench --profile`) vs the dynamic report's
finding-1 NCU:

```
attention kernel (NCU, locked clock)                         static      dynamic
-----------------------------------------------------------------------------------
QK^T scores (+ softmax max/sum stats; scalar)                 68.1 us     71.6 us  (QK^T only — softmax split out)
standalone softmax producer (…_xn)                            14.3 us    148.5 us  ← 10.4x
P@V (mma_m16n8k16_f16)                                         15.4 us     42.4 us
o_proj (mma_m16n8k16_f16)                                      28.6 us     47.0 → 29.2 us  (dynamic TMA, #245 — see note)
attn-out contiguify (…_xn)                                      4.4 us      4.3 us
-----------------------------------------------------------------------------------
torch ref: flash_fwd_splitkv + combine                        ~36 us      ~37 us
torch ref: cutlass_80_tensorop_f16_s16816gemm (o_proj)        22.7 us     22.2 us
```

Static raw NCU rows (this run): `k_sdpa_reduce_c0c97e` (QK^T+softmax-stats, scalar, regs 114) **68,096 ns**;
`k_sdpa_reduce_ba65cb` (P@V mma) **15,360 ns**; `k_sdpa_reduce_ba65cb_xn` (softmax-normalise producer, 67% dram —
memory-bound but small) **14,336 ns**; `k_linear_sdpa_reduce_94ab75` (o_proj mma) **28,608 ns**;
`k_linear_sdpa_reduce_94ab75_xn` (contiguify) **4,416 ns**.

> **Update (#245 — TMA for dynamic M-masked matmuls).** The o_proj `47.0 µs` dynamic number above was the *original*
> cp.async dynamic path. #245 enables TMA staging on the dynamic path for M-masked matmuls with a static innermost dim
> (the descriptor's globalDim is encoded per launch from the runtime shape; TMA zero-fills the masked overhang), so the
> deployed dynamic o_proj `k_linear_sdpa_reduce_43208b` now NCU-clocks **29,216 ns (29.2 µs), 0 smem bank conflicts**
> (vs the cp.async 47.0 µs / 6.8M conflicts) — essentially **at parity with the static o_proj (28.6 µs)**. The o_proj
> transport gap is closed; the other dynamic attention rows (softmax 148 µs, QK^T 71 µs, P@V 42 µs) are unchanged — they
> stay on cp.async (QK^T's symbolic-N inner axis breaks TMA's 16 B stride alignment; P@V is masked-K). Whole-layer
> dynamic e2e moved **361 → 337 µs (0.62 → 0.65x eager)**. Repro: `deplodock run --ir
> dynamic.k_linear_sdpa_reduce.repro.torch.json --bench --profile` (TMA default) vs `--ab "TMA=0"`.

**Root cause.** In the **dynamic** path, the masked-K demotion split (`005_split_demoted` on symbolic K) un-fuses the
SDPA into a **standalone softmax-normalising producer** that must MATERIALISE the full normalised seq×seq P matrix to
HBM so the masked-K mma consumer can read it — a scalar 3-pass (`max → exp-sum → normalise`) at 6% occupancy over the
symbolic seq×seq matrix, NCU **148 µs** (dynamic finding 1). In the **static** path the same P@V split fires, but the
normalise producer stays a cheap **14 µs** `_xn` slab (static K, no symbolic 3-pass materialisation), and the softmax
max/sum stats stay fused into the QK^T kernel (`k_sdpa_reduce`, the 68 µs scalar). Net softmax-path cost: ~82 µs static
vs ~219 µs dynamic. CLAUDE.md still records *flash-style fused symbolic-K attention remains future work* — that future
work is exactly what removes the dynamic 148 µs kernel.

**Repro.** Static NCU: `DEPLODOCK_DUMP_DIR=<dir>/ncu-static-oproj deplodock run --ir
<dump>/07_lowering_cuda.kernels/k_linear_sdpa_reduce_94ab75.torch.json --bench --profile` (raw CSV/JSON in
`ncu-static-oproj/61_ncu_metrics.{csv,json}`). Dynamic side: the dynamic report's finding-1 NCU. The masked-K mma gate
that creates the dynamic split: `010_partition_loops.py:737` (`k_forced_mask=k_symbolic`); the masked-K zero-fill +
SYNC-transport pin that makes the dynamic P@V consumer expensive: `_stage_expand.py:179` (`has_kmask` → SYNC) and
`:256`/`:305` (zero-fill, no cp.async).

**Suggested fix (highest priority for dynamic serving, large + known).** Flash-style symbolic-seq attention: a
scheduled online-softmax warp loop over the symbolic N (QK^T sub-tile → running max/sum rescale → P@V mma accumulate),
so the softmax never materialises to HBM. The static path shows the ceiling: with the softmax cheap, the deployed layer
beats eager (1.29x); closing the symbolic softmax gap is most of the dynamic 361→~200 µs runway.

## Finding 3 — fp16-half2-window lockout: static scalar matmuls get 2× packed `__hfma2`, symbolic-K never does

**Symptom.** The QK^T kernel is **scalar** (0 `mma.sync`) in BOTH static and dynamic — the online-softmax epilogue
forbids the warp-tier fold (the accumulator feeds the softmax max/sum reduce, a mid-reduction use), `_atom.py:324`:

> *the accumulator is consumed inside a reduce loop (mid-reduction use, not a store-time fold)*

This bail is shape-agnostic, so it is NOT the static-vs-dynamic differentiator. The differentiator is what the *scalar*
kernel is allowed to do: the static QK^T emits **48 `__half2` packed ops** (the fp16 half2 window, 2× `__hfma2`
throughput) and NCU-clocks **68 µs / 36.6% fma**; the dynamic QK^T has the window **locked out** and clocks **71 µs**
NCU / **162 µs** reproducer. Across the matmuls the same lockout costs the symbolic path: v_proj 15→27 µs, q_proj
9→17 µs, o_proj on the static path reaches mma (the linears below).

**Root cause.** `010_partition_loops.py:660`:

```python
fp16_window = (not prologue) and not k_symbolic and _is_fp16_matmul(matmul_reduces, graph)
```

`not k_symbolic` — the half2 window is gated OFF for any symbolic-K matmul (the window's bounded fp16-accumulation
flush can't interleave with the masked-K schedule). Static K → window ON. Same for the masked-tile per-element store
guards: symbolic M/N/K stamp `*_forced_mask` (`:670`, `:734`, `:737`) → an `if (coord < seq_len)` boundary cond on
every store; static tiles are clean (`prologue_mask_ok`/`E_M`/`E_N` degeneration at `:653`/`:662`-`:663` only bites the
symbolic axes).

**Repro.** Tier fingerprint: `grep -c mma.sync` + `grep -c __half2` over the emitted `.txt` —
`<dump>/07_lowering_cuda.kernels/k_sdpa_reduce_*.txt` shows `mma=0 half2=48` (static, window on). The locked-out
dynamic twin: the dynamic dump's `k_sdpa_reduce_*` (window off). Gate probe (no GPU): `010_partition_loops.py:660`.

**Suggested fix (medium, bounded by finding 2).** A masked-K-compatible fp16 window (flush the half2 accumulator at
the masked-K tile boundary) would recover the 2× on the symbolic scalar matmuls. But the QK^T scalar tier itself is
subsumed by the flash work (finding 2), which puts QK^T on tensor cores; the window matters mainly for the symbolic
linears (v_proj/q_proj), worth ~10–15 µs.

## Finding 4 — the masked-K P@V mma runs at 2.8× the static mma (42 vs 15 µs): zero-fill + SYNC-transport pin, not a tier or pick miss

**Symptom.** The P@V consumer reaches `mma_m16n8k16_f16` in BOTH paths (static `k_sdpa_reduce_ba65cb` mma=11; dynamic
masked-K mma), but the dynamic one runs **42.4 µs** (dynamic finding 2, 26% occ, 3.67M smem bank conflicts) vs static
**15.4 µs** (50% occ). (o_proj used to belong here too — dynamic 47 µs vs static 28.6 µs — but **#245 closed it**: the
o_proj matmul is M-masked with a static innermost dim, so it now stages via TMA on the dynamic path → 29.2 µs / 0 bank
conflicts, ≈ static. **P@V is the remaining gap**, because it's masked-K (symbolic reduce) and stays on cp.async.) The
static prior recovers its measured best cleanly (`eval prior --dataset db`: static mean **1.25x** / median 1.09x /
worst 1.97x, all -O1 inversions — vs the dynamic DB's mean 2.52x / worst 17.64x), so this is NOT a search shortfall or
a tier lockout.

**Root cause.** The dynamic masked-K P@V pays three symbolic-only taxes the static-K mma skips, all from the masked-K
staging: (1) the K reduce is tiled at the hint with a partial final slab **zero-filled in smem** (`_stage_expand.py:256`
— `(k < seq_len) ? v : 0`, a clamped read), (2) the source carrying the kmask is **pinned to the SYNC transport**
(`_stage_expand.py:179` — cp.async can't ternary a copied value, so no ring-buffered cp.async pipeline), and (3)
per-element store guards from `n_forced_mask`/`m_forced_mask`. The dynamic report's finding 2 proved `PAD_SMEM` /
`PERMUTE_LANES` are no-ops on this masked-tile smem layout (the bank conflicts are inherent to the masked-K slab), so
the levers the tuner has don't reach it. Static K → none of this: clean staging, cp.async eligible, no guards.

**Repro.** Static P@V tier: `<dump>/07_lowering_cuda.kernels/k_sdpa_reduce_ba65cb.txt` (mma=11). Gate probes:
`_stage_expand.py:179` (SYNC pin), `:256` (zero-fill), `010_partition_loops.py:737` (`k_forced_mask`). The dynamic A/B
proving PAD_SMEM/PERMUTE_LANES are no-ops: dynamic report finding 2.

**Suggested fix (medium — ~27 µs at stake on P@V; the o_proj half was already landed in #245).** The cleanest path is
**TMA for masked-K**: TMA's hardware OOB zero-fill replaces the manual `(k < seq_len) ? v : 0` ternary AND removes the
SYNC pin (so the async pipeline is usable again), exactly as #245 did for the M-masked matmuls — but P@V first needs
the 4-D leading-batch box derivation fixed (finding 5's `_collapse_inert_dims` / `box can't collapse arr` bench-fail).
Failing that, a conflict-free masked-K smem layout (swizzle inside the zero-filled slab) would close most of the
42→15 µs P@V gap. Folds into the flash work (finding 2) since that owns the symbolic-K schedule.

## Finding 5 — TMA descriptor rank mismatch: 20 `bench_fail`s on the P@V mma + the tune's full-model `--bench` capture (static-512 codegen bug)

**Symptom.** 47 of 5,525 `perf` rows are `bench_fail`. The dominant cluster (`eval failures`):

```
k_sdpa_linear_reduce_3d2635 — 20 row(s)
  error: ValueError('TMA descriptor rank mismatch: arr_shape=(1, 512, 1, 1024) cannot be collapsed to match
                     box_extents=(1, 64, 2, 64)')  [and (1, 32, 2, 64)]
  shared knobs: MMA=mma_m16n8k16_f16, TMA=True, SPLIT_CONE=True, STAGE=11, …
```

Every failing row has `TMA=True` on the P@V mma consumer at seq 512. The same error killed the tune's full-model
`--bench` captured bench (`[tune] full-model bench failed (… TMA descriptor rank mismatch …); continuing to
per-kernel`), which is why the deployed e2e (168 µs) had to come from a separate `run --bench` (greedy picks a non-TMA
P@V variant, so it succeeds).

**Root cause hypothesis.** Class 4 / class 3: the TMA descriptor builder for the static-512 P@V mma tries to collapse a
`(1, 512, 1, 1024)` array onto a `(1, 64, 2, 64)` box and fails the rank/collapse check — a real codegen bug in the TMA
box-extent derivation for this kernel's operand shape (the 4-D V slab with a leading batch + the s16816 box). It only
fires for `TMA=True` variants; non-TMA cp.async/sync staging works (the deployed pick and the 70 µs reproducer are
non-TMA). 20 wasted search slots, and it blocks the captured full-model bench. The remaining `bench_fail`s are
**compile-budget timeouts** (`compile stage exceeded 2.0s budget (2.3–3.2s)`) on big unrolled FK-window scalar kernels
at -O3 — the known cicc/LLVM blow-up, a budget not a correctness bug.

**Repro.** `deplodock eval failures` (clusters + shared knobs). Compile-only TMA repro (no GPU, source inspection):
`DEPLODOCK_KNOBS="MMA=mma_m16n8k16_f16,TMA=1,SPLIT_CONE=1,STAGE=11" deplodock compile
<dump>/07_lowering_cuda.kernels/k_sdpa_linear_reduce_3d2635.torch.json --ir cuda` (expect the descriptor builder to
raise). The TMA path: `lowering/tile/050_use_tma.py` + the descriptor box derivation.

**Suggested fix (medium — blocks the captured full-model bench + wastes 20 slots).** Fix the TMA box-extent
collapse for the leading-batch 4-D operand (or gate `TMA` off for this operand shape so the tuner doesn't enumerate a
guaranteed-fail variant). Until then the full-model `--bench` capture is unreliable on static-512 — use a separate
greedy `run --bench` for the deployed e2e (as this report does).

## Finding 6 — bench attribution: the deployed launch table mis-attributes attention; only NCU + e2e are trustworthy (recurring, 4th report)

**Symptom.** The deployed static `run --bench` launch table fingers the q_norm/k_norm RMSNorm reductions
(`k_mean_linear_reduce_125c9c` 21.5 µs / 13%, `…_1f1bec` 22.2 µs / 13.5%) and the softmax `_xn` (22.8 µs / 13.8%) as
the layer's biggest kernels, while NCU (finding 2) clocks the QK^T+softmax-stats at 68 µs — the per-launch solo windows
absorb cross-kernel latency and the `%` column mislabels the dominator, exactly as the dynamic report's finding 5 and
the two before it. The `--bench` reproducer table also re-fuses the upstream cone, so the attention reproducers'
78/70 µs are slice-set totals, not the named kernel.

Only the **NCU single-capture table** and the **whole-program e2e** (168 µs) are trustworthy for attention. This is the
recurring attribution problem (`plans/bench-attribution-by-slicing.md`) — the static report reproduces it on the same
kernels.

**Suggested fix (high — the recurring tooling gap).** Land per-launch attribution by slicing so the deployed table
matches NCU; until then the skill's "trust NCU + e2e for attention" holds, and this run is one more datapoint.

## Repro / artifacts

- Work dir: `_tune/tune-model-qwen3-emb-l0-static/` (gitignored, survives reboots) — `tune.log` (3.5 h tune tee, incl.
  the 3 `dropped un-lowerable candidate` lines + the per-kernel -O3 table), `run-static.log` (deployed e2e 168 µs +
  launch table), `run-dynamic-cooled.log` (cooled dynamic e2e 361 µs, same card), `ncu-static-oproj.log` +
  `ncu-static-oproj/61_ncu_metrics.{csv,json}` (finding-2 NCU), dump at `dump/` (reproducers under
  `07_lowering_cuda.kernels/`, `62_kernel_bench.json`, `kernels.html`; `.png` skipped — Playwright `Target page closed`
  flake), `PRIOR-static-ref-2026-06-10.md` (the recovered static reference report for tone/structure).
- Tune DB / prior (isolated, this run only): `_tune/tune-model-qwen3-emb-l0-static/{static.db,static-prior.json}`.
  Dynamic-side comparison reads the preserved default `~/.cache/deplodock/{autotune.db,prior.json}`.
- Static e2e (deployed): `DEPLODOCK_TUNE_DB=…/static.db DEPLODOCK_PRIOR_FILE=…/static-prior.json deplodock run
  Qwen/Qwen3-Embedding-0.6B --layer 0 --seq-len 512 --bench`. Dynamic e2e (same card): `deplodock run
  Qwen/Qwen3-Embedding-0.6B --layer 0 --dynamic seq_len@x:1 --bench` (default paths = dynamic prior).
- Finding-1 fix (this branch): `_stage_expand.py:compute_phase_info` (descriptive `LoweringError`),
  `pipeline.py:Run.drive` (per-variant containment) + `Run._dropped_candidates`,
  `tests/compiler/pipeline/test_lowering_error_guardrail.py` (+3 tests). Gate probes (no GPU):
  `_atom.py:324` (online-softmax fold bail), `010_partition_loops.py:660` (fp16-window
  lockout), `:737` (`k_forced_mask`), `_stage_expand.py:179`/`:256` (masked-K SYNC pin + zero-fill).
- Finding-5 failures: `deplodock eval failures` (static DB).

## Workflow notes

Audit of the dynamic report's notes first:

- **Per-launch mis-attribution (dynamic finding 5)**: **reproduced** (finding 6). The static deployed table mislabels
  the RMSNorms / softmax `_xn` as the dominators; only NCU revealed the 68 µs QK^T. Now 4 reports in a row — strongly
  reinforces `plans/bench-attribution-by-slicing.md`.
- **Reproducer re-fusion (dynamic finding 5)**: **reproduced.** The static attention reproducers re-fuse the cone
  (78/70 µs slice-set totals), so I fell back to NCU for every attention number. A `run --ir … --bench --no-refuse`
  (bench only the named provenance kernel) would still save the most triage time.
- **Stable per-op identity across views**: **reproduced.** The static QK^T wears `82f310` (deployed), `3c1b01` (dump),
  `c0c97e` (NCU re-lower) — three hashes for one op, hand-cross-referenced. A stable provenance name in the NCU /
  leaderboard / deployed tables would remove the cross-referencing.
- **Chart PNG Playwright flake**: reproduced verbatim (`png skipped: Target page … closed`).

New friction this run:

- **A single un-lowerable search fork aborted the whole 3.5 h tune (finding 1).** This is the static-reference report's
  finding 1 recurring — "rewrite-time exceptions in the inner search have no per-variant containment." I fixed it here
  (`Run.drive` containment), but the lesson for the skill: **a clean tune on a new shape/commit can crash on a
  search-only lowering bug greedy never hits** — the memory note "clean-tune is the real test gate" held exactly (unit
  tests + greedy compile both passed while the tune crashed). Budget for a fix-and-rerun on the first clean tune of a
  new configuration.
- **The crash took ~30 min to reach under default patience and `-q` is near-silent under `tee`** (the tty progress bar
  doesn't survive the pipe). A non-tty per-terminal heartbeat (`[tune] terminal k/16 done, best Σ …`) would show
  liveness without `-v`'s firehose, and would have told me the tune was alive vs wedged during the 30-min CPU-bound
  nvcc grind (GPU sat at 5%). Biggest wall-clock uncertainty.
- **The tune's full-model `--bench` capture is unreliable on static-512 (finding 5)** — it failed on a TMA descriptor
  bug, so the headline e2e had to come from a separate greedy `run --bench`. The skill already prefers a cooled fresh
  `run --bench` for the reported e2e; this run shows the tune-`--bench` full-model row can be *absent*, not just
  thermally inflated. A `tune --bench` that falls back to the greedy (non-TMA) pick for the full-model row when the
  captured pick fails would keep the headline number in the tune output.
- **`--seq-len` is load-bearing for a static-vs-dynamic comparison and easy to miss.** The dynamic path benches at the
  512 hint regardless of `--seq-len`; the static path specialises to `--seq-len` (default 32). Tuning static at the
  default would have produced a meaningless comparison (attention ~256x smaller). The skill should call out
  "match `--seq-len` to the dynamic hint (512) when comparing static vs dynamic."

What worked well: the triage loop was tight once the tune completed — `eval failures` clustered the 47 fails into
TMA-vs-compile-budget in one line (finding 5); the single `--profile` NCU run on the o_proj-chain reproducer gave the
whole static attention path + the torch flash/cutlass references in one aligned table (the clean signal for finding 2,
directly comparable to the dynamic report's NCU); `eval prior --dataset db` settled "search shortfall vs masked-tile
tax" (static 1.25x mean reachability) in one command; and the `grep -c mma.sync` / `grep -c __half2` over the emitted
`.txt` was the fastest tier + fp16-window fingerprint (static QK^T `mma=0 half2=48` → window on). A `tier` (scalar/mma)
+ `half2` column in `eval variants` would make findings 3–4 a zero-step read.
