# Qwen3-Embedding-0.6B whole-model tune findings — RTX 5090, 2026-07-02

**Status:** whole-model dynamic tune complete; the full-model -O3 e2e table is UNOBTAINABLE (the fused
final-norm → lm_head kernel exceeds the 10 s bench budget — finding 1); per-kernel -O3 table below; serving
A/B recorded (see the serving section).

- **Command:** `emmy tune Qwen/Qwen3-Embedding-0.6B --dynamic seq_len@input_ids:1 --dynamic
  seq_len@attention_mask:2 --dynamic seq_len@attention_mask:3 --dynamic seq_len@position_ids:1 --clean
  --dump-dir _tune/tune-qwen3-embed/dump`, followed by `rm prior.json` (deploys are the **cold AnalyticPrior's**
  — the learned checkpoint is mis-calibrated, carried finding from `plans/golden-sweep-rtx5090-findings.md`)
  and per-reproducer `emmy run --ir <k>.torch.json --bench --bench-backends eager,tcompile,emmy`.
- **Run stats:** tune wall ~4.1 min (1 fused terminal, 243.6 s), 398 ok / 2 `bench_fail` DB rows (the two
  fused-tail hangs, finding 1); zero un-lowerable candidate drops after the distributivity gate landed (the
  first attempt had 23+ — finding 3). **Dynamic run: symbolic `seq_len`, all numbers at the 512 hint.**
- Numbers below are the -O3 reproducer re-bench; tune-DB latencies quoted for ranking context are -O1.
- The trace is the CAUSAL-LM wrapper (includes `lm_head`, vocab 151669); the serving trace (AutoModel trunk)
  excludes it, so finding 1's worst case does not ship in the served model.

## Per-kernel -O3 bench (14 unique kernels, dedup'd; benched at seq_len=512, symbolic hint)

| kernel | layer op | eager µs | tcompile µs | emmy µs | vs eager |
|---|---|---|---|---|---|
| k_linear_sdpa_reduce_6bb11e | attn linear + SDPA chain | 163 | 162 | 18707 | 0.009× |
| k_linear_sdpa_reduce_6db6c8 | embed-gather + linear + SDPA chain | — | — | 18721 | — |
| k_linear_mean_reduce_63573a | post-attn rmsnorm → gate/up proj (N=3072) | 233 | 151 | 5635 | 0.04× |
| k_linear_mean_reduce_7aeec0 | final rmsnorm → lm_head (N=151669) | — | — | **>10 s, bench_fail** | — |
| k_sdpa_linear_reduce_95f2c6 | SDPA → o_proj chain | — | — | **>7 min timeout** | — |
| k_sdpa_reduce_5661da | rotary + SDPA | 206 | 88 | 2401 | 0.09× |
| k_mean_d65726 | rmsnorm stats | 65 | 6 | 313 | 0.21× |
| k_mean_9cf1ca | q/k-norm stats | — | — | 190 | — |
| k_mean_linear_reduce_efb40d | fused mean→linear (small N) | 151 | 62 | 92 | 1.64× |
| k_mean_linear_reduce_39ba47 | fused mean→linear (small N) | 102 | 44 | 67 | 1.52× |
| k_linear_0837e7 | plain linear | 111 | 110 | 71 | 1.55× |
| k_linear_reduce_f94dd0 | linear (+reduce epilogue) | 56 | 55 | 50 | 1.12× |
| k_linear_reduce_716194 | linear (+reduce epilogue) | 41 | 41 | 25 | 1.62× |
| k_cat_slice_transpose_unsqueeze_pointwise_1f8c16 | rotary layout | 59 | 2 | 2 | 29× |

Dominating kernels: the four fused chains (two `linear_sdpa`, the N=3072 `linear_mean`, `sdpa_reduce`) are
~45 ms of a ~46 ms emmy forward — >97% of the total, all 10–200× behind eager. Every non-mega-fused kernel is at
parity or ahead of eager (and the plain linears beat it 1.1–1.6×).

## Finding 1 — mega-fused kernels are structurally pathological, and the tree has no un-fuse escape

**µs at stake: ~45 ms/forward (≈97% of the emmy total).** The loop-fusion pass merges whole chains — embedding
gather + QKV linear + SDPA (`k_linear_sdpa_reduce_*`), rmsnorm → 3072-wide gate/up projection
(`k_linear_mean_reduce_63573a`), and worst, final rmsnorm → the 151669-wide lm_head
(`k_linear_mean_reduce_7aeec0`) — into single kernels whose contraction rides the projection TAIL of a
cooperative reduce. That shape destroys matmul tiling: each row-CTA re-streams the full weight matrix with
uncoalesced per-lane access (the lm_head case reads 621 MB per row — minutes per launch; even the coop
partition can't fix bandwidth). Evidence: the tune DB's per-op slices of the same chains measure 86–107 µs each
(-O1, `eval variants k_linear_sdpa_reduce_6bb11e`: 22 configs, pick rank 2/22), while the fused wholes re-bench
at 5.6–18.7 ms (-O3) — the pieces are fine, the fusion is the defect. The demolition removed the structural
un-fuse fork (`010_split_demoted` — "PLACE@cone is pin-only" in the rebuilt tree), so there is NO escape: not a
knob, not a pin. **Fix (highest priority): restore the structural CUT producer (the PLACE auto-knobification
follow-up), with a fusion cost gate — don't fuse a reduce producer into a contraction whose free-column extent
exceeds what the tail can stream (a bandwidth-model gate, not a shape name).** Repro:
`emmy run --ir _tune/tune-qwen3-embed/dump/08_lowering_cuda.kernels/k_linear_mean_reduce_63573a.torch.json --bench`.

## Finding 2 — the cooperative free-grid cap starved tailed reduces (FIXED this run)

`_pick_coop` refused cooperation whenever the output grid exceeded `_FREE_CAP=256` cells, so every
fused norm→linear kernel at the 512 hint ran ONE THREAD PER ROW: `k_linear_mean_reduce_63573a` at ~250 ms/launch
(observed as the "vLLM serving hang" — 45+ min of warmup at 100% GPU), the lm_head variant in the minutes.
Fixed: a reduce with a fused contraction tail (`_has_contraction_tail`, the same structural read the shared-row
stage uses) always offers the cooperative partition — 63573a dropped 250 ms → 5.6 ms (44×). Still 24× behind
eager (finding 1 owns the rest).

## Finding 3 — atomic split-K rows flooded the search with un-lowerable candidates (FIXED this run)

The first tune attempt burned 23+ search slots on `g<w>a` rows whose fused non-distributive projections raise at
`030_split` materialize. Fixed: `_reduce_candidates` gates atomic-finalize splits on `projection_distributes`
(moved to `_carrier.py`); the re-run had zero drops and 398 ok rows. The pin path keeps its loud raise.

## Finding 4 — the whole-model tune is one 4-minute fused terminal

`tune` treats the whole model as ONE fused terminal (243.6 s, single outer terminal); per-op inner searches share
that budget, so per-op coverage is thin (e.g. `k_sdpa_reduce_5661da`: 1 measured config). With deploys now
analytic-only (the learned checkpoint deleted — golden-sweep finding 2 carried), tune depth matters less for
deploys, but DB coverage for `eval`/future prior training is minimal. Recommendation: a per-op patience floor for
whole-model scope, or document `--layer` scope as the tuning workhorse.

## Serving A/B (emmy plugin vs stock vLLM)

Command pair (matched params, `--num-prompts 64 --random-input-len 256 --max-concurrency 8 --bench-seed 0`,
`--max-model-len 4096`, gpu-mem 0.5): `emmy serve Qwen/Qwen3-Embedding-0.6B --bench [--stock]`.

| metric | emmy plugin | stock vLLM | emmy/stock |
|---|---|---|---|
| Request throughput (req/s) | 4.93 | 352.97 | 0.014× |
| Mean E2EL (ms) | 1599.40 | 21.75 | 73.5× |
| Median E2EL (ms) | 1592.58 | 15.29 | 104× |
| P99 E2EL (ms) | 1808.60 | 68.51 | 26× |
| Benchmark duration (s), 64 reqs | 12.97 | 0.18 | — |

The per-kernel wins (plain linears 1.1–1.6× ahead of eager) do NOT translate into served numbers: the four
mega-fused chains own >97% of the forward (finding 1), so the served model is ~72× behind stock vLLM. Until the
structural un-fuse lands, serving this model with the emmy plugin is not competitive — the honest deployable
gate for the plugin is finding 1's fix, not more per-kernel tuning.

## Repro / artifacts

- Tune log: `_tune/tune-qwen3-embed/tune.log`; dump: `_tune/tune-qwen3-embed/dump` (reproducers under
  `08_lowering_cuda.kernels/`); per-kernel bench logs: `_tune/tune-qwen3-embed/kbench/`.
- Compile-only repro of the lm_head pathology (no GPU):
  `emmy compile --ir tile _tune/tune-qwen3-embed/dump/08_lowering_cuda.kernels/k_linear_mean_reduce_7aeec0.torch.json`
  — the tail `for a2 in 0..151669 { for a3 in 0..1024 … }` inside a per-row kernel is the whole story.
- Serving logs: `_tune/tune-qwen3-embed/serve_emmy.log` / `serve_stock.log`.

## Workflow notes

- **The -O3 full-model bench dies with the model** when any one kernel exceeds the 10 s budget — a per-kernel
  skip-and-continue (like the tune's bench_fail pinning) would have salvaged the table; instead the per-kernel
  numbers were re-assembled from 14 separate `run --ir` invocations (~40 min).
- **`eval variants` leaderboards are per-op slices, not per-kernel** — cross-referencing a fused kernel's -O3
  latency against its slices' -O1 rows takes care (nearly mis-read as a 200× -O1/-O3 gap); a fused-kernel-level
  view naming its slices would prevent that.
- **Background-run hygiene**: three environment kills across the day's runs; `setsid` + PID-file + log-tail
  waiters was the pattern that survived. The golden-sweep note about `--clean` restarts applies here too.
- Serving benches compile the whole model at -O3 inside vLLM startup (~10–20 min before the first request) —
  a persisted kernel cache across serve runs would cut the A/B's wall time in half.
