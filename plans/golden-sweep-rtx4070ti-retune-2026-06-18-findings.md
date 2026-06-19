# Golden re-tune — RTX 4070 Ti (sm_89) — post-feasibility-filter sweep

**Date:** 2026-06-18
**GPU:** NVIDIA GeForce RTX 4070 Ti (sm_89, Ada, 60 SMs, 12 GB, `MaxSharedMemoryPerBlockOptin = 101376 B` ≈ 99 KB)
**Sweep command:** `deplodock tune --dataset golden --clean` (live-GPU scoped → 29 shapes), then
`deplodock run --bench --golden NAME` per shape for the deployable A/B.
**Tune wall:** ~69 min (29 shapes incl. 6 `.dynM`, sum of per-shape `[tune] done`; the 4096 squares dominate at
318–364 s each) + ~18 min for the 29 `run --bench` A/Bs + confirmation re-runs.
**Tally:** **12 replaced / 0 added / 3 unchanged / 14 worse** (29 shapes).

This is the **first refresh** of `rtx4070ti_sm89.yaml` after the seed
(`plans/golden-sweep-rtx4070ti-findings.md`, 2026-06-16), run specifically to validate the in-pass smem-feasibility
filter (`greedy: in-pass smem-feasibility filter, revert retry cap 64->8`). **Seed Finding 1 is resolved** — see the
headline below.

## Headline — the feasibility filter eliminated the Ada smem crash

The seed sweep's **Finding 1** was a hard blocker: greedy deploy ranked >99 KB tiles first (Blackwell-trained prior on
Ada), lowering failed at `KernelOp.validate`'s smem gate, and `Pipeline.run` re-resolved the whole graph once per
infeasible tile — exhausting the original `_MAX_GREEDY_RETRIES = 8` and **hard-crashing** rectangular projections. The
seed shipped a stopgap (cap 8 → 64).

This sweep ran end-to-end with the durable fix (cap reverted to 8): `greedy_decide` now probes each prior-ranked
partition leaf by lowering it through `KERNEL_PASSES` and deploys the first that produces a valid `KernelOp`. **All
29/29 shapes tuned and deployed in-process with zero `LoweringError` / illegal-address / saturated-queue aborts.** The
only tune errors were 44 per-variant `bench_fail` timeouts (normal compile-budget / GPU-time containment). The fix holds
on real Ada deploys, not just unit tests.

## Per-shape outcome (all numbers live -O3 `run --bench` on the 4070 Ti, CUDA-graph captured)

| shape | greedy µs | best golden µs | ratio | category |
|---|---|---|---|---|
| square.512 | 24.4 | 24.5 | 1.00 | unchanged (same knobs) |
| square.1024 | 144.4 | 135.6 | 1.06 | unchanged (same knobs; golden re-bench crashes — Finding 3) |
| square.2048 | 849.9 | 820.2 | 1.04 | worse (leave) |
| square.4096 | 6733.8 | 6008.8 | 1.12 | worse (leave) |
| square.512.fp16 | 27.6 | 28.9 | 0.96 | **replaced** |
| square.1024.fp16 | 136.9 | 128.4 | 1.07 | worse (leave) |
| square.2048.fp16 | 899.1 | 995.3 | 0.90 | **replaced** |
| square.4096.fp16 | 7276.5 | 7975.9 | 0.91 | **replaced** |
| qwen3_06b.q_proj.s32 | 23.2 | 17.2 | 1.35 | worse (Finding 1) |
| qwen3_06b.kv_proj.s32 | 24.9 | 12.7 | 1.96 | worse (Finding 1) |
| qwen3_06b.o_proj.s32 | 46.7 | 23.7 | 1.97 | worse (Finding 1) |
| qwen3_06b.gate_up_proj.s32 | 34.7 | 28.2 | 1.23 | worse (Finding 1) |
| qwen3_06b.down_proj.s32 | 70.5 | 49.8 | 1.42 | worse (Finding 1) |
| qwen3_06b.q_proj.s128 | 42.9 | 65.5 | 0.65 | **replaced** |
| qwen3_06b.kv_proj.s128 | 25.6 | 34.0 | 0.75 | **replaced** |
| qwen3_06b.o_proj.s128 | 75.0 | 82.0 | 0.91 | **replaced** |
| qwen3_06b.gate_up_proj.s128 | 58.9 | CRASH | — | **replaced** (golden config misaligned — Finding 3) |
| qwen3_06b.down_proj.s128 | 116.8 | 77.8 | 1.50 | worse (Finding 1) |
| qwen3_06b.q_proj.s512 | 149.2 | 186.2 | 0.80 | **replaced** |
| qwen3_06b.kv_proj.s512 | 67.4 | 92.0 | 0.73 | **replaced** |
| qwen3_06b.o_proj.s512 | 162.3 | 154.5 | 1.05 | worse (leave) |
| qwen3_06b.gate_up_proj.s512 | 357.7 | 190.7 | 1.88 | worse (Finding 1) |
| qwen3_06b.down_proj.s512 | 322.6 | 583.7 | 0.55 | **replaced** |
| square.512.dynM | 23.9 | 24.0 | 1.00 | unchanged (same knobs) |
| qwen3_06b.q_proj.s512.dynM | 141.3 | 191.7 | 0.74 | **replaced** |
| qwen3_06b.kv_proj.s512.dynM | 95.7 | 68.9 | 1.39 | worse (Finding 1) |
| qwen3_06b.o_proj.s512.dynM | 121.5 | 155.5 | 0.78 | **replaced** |
| qwen3_06b.gate_up_proj.s512.dynM | 242.4 | 172.0 | 1.41 | worse (Finding 1) |
| qwen3_06b.down_proj.s512.dynM | 246.3 | 223.2 | 1.10 | worse (leave) |

The 12 replacements are real, reproducible wins (the four marginal ones — `square.512.fp16` 4.5%, `square.2048.fp16`
10%, `square.4096.fp16` 9%, `o_proj.s128` 9% — were re-run twice; CUDA-graph capture makes the A/B deterministic to
<1%, so they hold). Eight of the twelve are large (>20%) wins on the s128/s512 projections, where the warm prior now
deploys a tighter `BM8/BN32/BK32, SPLITK2, RING2` family instead of the cross-arch 4090 tile.

## Finding 1 — learned prior deploys `FM1`/`BN64` wide tiles that win at -O1 but lose ~2× at -O3 (9 shapes)

The worst regressions all share a shape: greedy deploys `BM16 BN64 BK32 FM1 FN2` (block 1024) — a wide-N tile with
**`FM=1`** (no per-thread M register reuse). Live -O3 it is ~2× slower than the golden's `BN16/FM2-4`, which **lowers
fine** (the golden row benches clean):

| shape | deployed (greedy) | golden | live ratio |
|---|---|---|---|
| qwen3_06b.o_proj.s32 | BM16 BN64 FM1 FN2 | BM8 BN16 FM2 FN2 | 1.97× |
| qwen3_06b.kv_proj.s32 | BM16 BN64 FM1 FN2 | BM8 BN16 FM2 FN2 | 1.96× |
| qwen3_06b.gate_up_proj.s512 | BM16 BN64 FM2 FN2 | BM8 BN32 FM10 FN4 | 1.88× |
| qwen3_06b.down_proj.s128 | BM16 BN16 FM4 FN8 | BM8 BN16 FM4 FN2 | 1.50× |
| qwen3_06b.down_proj.s32 | BM16 BN64 FM1 FN2 | BM8 BN16 FM2 FN2 | 1.42× |
| qwen3_06b.gate_up_proj.s512.dynM | BM16 BN64 FM16 FN2 | BM8 BN16 FM8 FN4 | 1.41× |
| qwen3_06b.kv_proj.s512.dynM | BM8 BN64 FM10 FN2 | BM8 BN16 FM8 FN4 | 1.39× |
| qwen3_06b.q_proj.s32 | BM8 BN16 FM6 FN4 | BM8 BN16 FM4 FN2 | 1.35× |
| qwen3_06b.gate_up_proj.s32 | BM8 BN16 FM8 FN8 | BM16 BN32 FM2 FN2 | 1.23× |

**This is not the feasibility filter.** These are tiny/mid shapes with no smem pressure — every candidate tile lowers,
so `_leaf_feasible` returns `True` for all and the filter never engages. The mis-pick is the prior's ranking. Evidence:

- `eval analytic` ranks the golden **shallow** (o_proj.s32 rank 17/1008, kv_proj.s32 17, q_proj.s32 4, gate_up_proj.s512
  10) — the cold hand-coded heuristic *can* reach these configs.
- `eval prior --dataset golden` ranks the golden much deeper under the **learned** prior (o_proj.s32 30, kv_proj.s32 47,
  q_proj.s32 363) and its `vs gold` column — read off the **-O1 reservoir** — even claims o_proj.s32 is `0.82×` (a
  win), directly contradicting the live `1.97×`. That is the **-O1/-O3 inversion** the skill warns about: the prior is
  trained on -O1 tune latencies, where the `FM1`/`BN64` tile ranks fastest; at -O3 deploy it is 2× slower, and the -O3
  re-bench reservoir did not correct the ranking for these shapes.

**Recommendation (priority 1):** close the -O1/-O3 gap for the deployed config. The `DEPLODOCK_O3_TOL` tolerance-band
re-bench only samples configs within 10% of the best **-O1** latency; the `FM1`/`BN64` tile is the -O1 best, so it *is*
sampled — but a single `H_opt=3` row isn't enough to outrank it in the trained model. Either (a) weight `H_opt=3` rows
more heavily in `Prior.fit`, or (b) raise `DEPLODOCK_ANALYTIC_TILT` for small-M shapes (the analytic, which ranks these
goldens shallow, should pull the learned prior off its -O1 overfit). A focused experiment:
`run --bench --golden o_proj.s32 --ab "<golden knobs>"` already shows the golden config is 2× faster live — the
data to retrain on exists; the fit/tilt is the lever.

## Finding 2 — fp32/fp16 large squares mildly worse (3 shapes)

`square.4096` 1.12×, `square.1024.fp16` 1.07×, `square.2048` 1.04×. Greedy picks `BK64` where the golden has `BK32`
(square.4096: also `FM8` vs `FM12`). `eval analytic` ranks these goldens shallow (square.4096 rank 40, square.2048 68);
same family as Finding 1 (the learned prior's -O1 ranking favors the deeper-`BK` tile). Lower priority — the gaps are
4–12%, and the same `H_opt=3`-weighting / analytic-tilt fix from Finding 1 should pull them in. Left unchanged.

## Finding 3 — golden A/B re-bench crashes with `CUDA_ERROR_MISALIGNED_ADDRESS` on two shapes (tooling bug)

`run --bench --golden square.1024` and `…gate_up_proj.s128` both abort the **`golden NAME` row** with
`cudaErrorMisalignedAddress` (reproduced twice), while the greedy deploy row still benches. Two distinct causes:

- **gate_up_proj.s128** — the recorded golden was a 4090-derived `BN64 BK64 FM10` config (seed Finding 3 class:
  4090 configs that fault on the 4070 Ti). It is genuinely unrunnable here → **replaced** with the working greedy pick
  (`BN16 FM8 FN8`, clean 58.9 µs via plain `run --bench` with no golden variants).
- **square.1024** — the recorded golden knobs are **identical** to the greedy pick (`BM16 BN32 BK64 FM10 FN4`), and
  the greedy deploy of exactly those knobs runs clean at 144.4 µs. So the crash is in the **`_bench_golden_variants`
  fresh re-trace / codegen path**, not the config. Left unchanged (the deployable config is fine); the crash is a
  diagnostic-tool bug.

Side effect: the misaligned error zeroes the greedy row's NCU profiling (`regs 0 occ --`) even though its latency is
valid. **Recommendation (priority 2):** isolate `_bench_golden_variants` in a SIGKILL-able subprocess like tune's
`_bench_worker` (`commands/run.py` golden A/B path), so a crashing golden variant can neither abort the row nor pollute
the greedy row's profiling; and root-cause the misaligned-address in the fresh-trace codegen for these N/K.

## Workflow notes (retrospective for the CLI / skill maintainer)

- **`eval` deployable signal disagrees with the live A/B.** `eval prior --dataset golden`'s `vs gold` column and `eval
  golden`'s "found" pick read the **-O1 reservoir**, so they contradict the live -O3 `run --bench` on many shapes
  (q_proj.s128: `vs gold 2.35×` but live `0.65×`; o_proj.s32: `0.82×` but live `1.97×`). *Symptom:* the one-glance
  eval view points the opposite way from the ground-truth A/B. *Fix:* compute `vs gold` from `H_opt=3` reservoir
  rows only, or label the column `-O1` so it isn't read as deployable.
- **`eval variants --kernel` can't filter by shape.** `--kernel qwen3_06b.o_proj.s32` returns "no measured variants" —
  the filter matches the CUDA kernel name (`k_matmul_<hash>`), not the golden/shape name, so the per-shape reachability
  drill-down the findings need is unreachable by name. *Fix:* accept a golden/shape-name filter on `eval variants`.
- **Golden re-bench isn't crash-isolated.** A misaligned/hanging golden config (Finding 3) aborts the `golden NAME` row
  and corrupts the greedy row's profiling in the same process. *Fix:* subprocess-isolate `_bench_golden_variants` (the
  tune path already does this for variant benches).
- **Noise floor is smaller than the skill assumes — a positive.** The skill's step 4 warns of ~10–13% run-to-run
  swing; with CUDA-graph capture the A/B rows are deterministic to <1% (confirmation reps were byte-identical for the
  fp16
  squares). The noise re-runs were near-unnecessary. *Fix:* relax the skill's step-4 caveat to "re-run only sub-5%
  margins" — capture made the rest stable.
- **Tune wall ~69 min for 29 shapes** (skill estimate ~30 min / 23 shapes; this card is slower and has 6 extra `.dynM`).
  The 4096 squares (fp32 + fp16) dominate at ~320–365 s each. `--kernel SUBSTR` narrowing works for iterating on a
  family without the full sweep.

### Seed report's workflow notes — status

The seed (`golden-sweep-rtx4070ti-findings.md`) flagged three workflow gaps; this sweep confirms which held:

- **"smem crash aborted the entire sweep"** — **FIXED** (the feasibility filter; this whole sweep ran clean). Held.
- **"no `deplodock doctor` preflight (driver/nvcc/smem)"** — still absent; CUDA env still set by hand (`CUDA_HOME` via
  venv activate). Unchanged.
- **"`eval`/A/B data only in logs, hand cross-referenced"** — still true (this report's tables were assembled by
  grepping per-shape `run --bench` logs). A machine-readable per-shape A/B dump (like `tune --bench`'s
  `62_kernel_bench.json`) would remove the hand cross-referencing.
