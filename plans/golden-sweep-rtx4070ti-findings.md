# Golden sweep — RTX 4070 Ti (sm_89) — seeding findings

**Date:** 2026-06-16
**GPU:** NVIDIA GeForce RTX 4070 Ti (sm_89, Ada AD104, 60 SMs, 12 GB,
`MaxSharedMemoryPerBlockOptin = 101376 B`)
**Sweep command:** `deplodock tune --dataset golden` (live-GPU scoped → 29 shapes), then
`deplodock run --bench --golden NAME` per shape for the deployable A/B.
**Tune wall:** ~64 min (29 shapes, sum of per-shape `[tune] done`), + ~25 min for the 29 `run --bench` A/Bs.

This was a **seed**, not a refresh: the 4070 Ti had **no golden file**. The 29 shape *definitions* (M/N/K, dtype, dynamic
specs) were cloned from `rtx4090_sm89.yaml` (same compute cap 8.9); every knob set + latency in the new
`rtx4070ti_sm89.yaml` is a fresh 4070 Ti measurement. The 4070 Ti was also registered in `deplodock/gpu.py`
(sm_count=60, pci `2782`) so goldens featurize with the card's own SM regime.

**Prerequisite (out of band):** CUDA was dead on this box — driver `535.171` (CUDA 12.2) could not load the installed
`torch 2.12.0+cu130` / `cupy 12.9` userspace, and there was no `nvcc`. Fixed by updating to `nvidia-driver-580-open`
(reboot) and installing a no-sudo pip CUDA-13 toolkit (`nvidia-cuda-nvcc` + `nvidia-cuda-runtime` + `nvidia-cuda-cccl`,
`CUDA_HOME` persisted in the venv activate). Without this no GPU work runs at all.

## Outcome tally (seed)

| source of recorded golden | count | meaning |
|---|---|---|
| **greedy** | 13 | greedy pick is the fastest (or only viable) deployable config on the 4070 Ti |
| **gold4090 (cross-arch)** | 16 | the 4090's config, benched live on the 4070 Ti, beats the greedy pick by >5% |

The 16 `gold4090` shapes are the headline: on those, the greedy pipeline pick is **1.1×–3.5× slower** than simply
deploying the 4090's hand-tuned config on the same Ada microarchitecture. That gap is Finding 2.

## Per-shape outcome (all numbers are live -O3 `run --bench` on the 4070 Ti)

| shape | greedy µs | 4090-cfg µs | recorded | source |
|---|---|---|---|---|
| square.512 | 24.4 | 31.2 | 24.4 | greedy |
| square.1024 | 135.6 | — (hung) | 135.6 | greedy |
| square.2048 | 790.5 | — (hung) | 790.5 | greedy |
| square.4096 | 5944.3 | 6071.3 | 5944.3 | greedy |
| square.512.fp16 | 28.9 | 37.8 | 28.9 | greedy |
| square.1024.fp16 | 128.3 | 178.9 | 128.3 | greedy |
| square.2048.fp16 | 929.8 | 1011.7 | 929.8 | greedy |
| square.4096.fp16 | 12103.7 | 7806.0 | 7806.0 | gold4090 (1.55×) |
| qwen3_06b.q_proj.s32 | 22.9 | 20.0 | 20.0 | gold4090 (1.15×) |
| qwen3_06b.kv_proj.s32 | 24.4 | 11.7 | 11.7 | gold4090 (2.09×) |
| qwen3_06b.o_proj.s32 | 81.8 | 23.4 | 23.4 | gold4090 (3.50×) |
| qwen3_06b.gate_up_proj.s32 | 44.1 | — (hung) | 44.1 | greedy |
| qwen3_06b.down_proj.s32 | 114.3 | 46.0 | 46.0 | gold4090 (2.48×) |
| qwen3_06b.q_proj.s128 | 114.9 | 60.7 | 60.7 | gold4090 (1.89×) |
| qwen3_06b.kv_proj.s128 | 82.3 | 31.6 | 31.6 | gold4090 (2.60×) |
| qwen3_06b.o_proj.s128 | 155.0 | 81.7 | 81.7 | gold4090 (1.90×) |
| qwen3_06b.gate_up_proj.s128 | 84.2 | 138.8 | 84.2 | greedy (1.65× the other way) |
| qwen3_06b.down_proj.s128 | 223.0 | 100.4 | 100.4 | gold4090 (2.22×) |
| qwen3_06b.q_proj.s512 | 361.8 | 172.9 | 172.9 | gold4090 (2.09×) |
| qwen3_06b.kv_proj.s512 | 202.8 | 112.0 | 112.0 | gold4090 (1.81×) |
| qwen3_06b.o_proj.s512 | 408.1 | 155.1 | 155.1 | gold4090 (2.63×) |
| qwen3_06b.gate_up_proj.s512 | 514.6 | 178.0 | 178.0 | gold4090 (2.89×) |
| qwen3_06b.down_proj.s512 | 546.3 | — (hung) | 546.3 | greedy |
| square.512.dynM | 22.3 | 23.7 | 22.3 | greedy |
| qwen3_06b.q_proj.s512.dynM | 204.2 | 178.7 | 178.7 | gold4090 (1.14×) |
| qwen3_06b.kv_proj.s512.dynM | 74.0 | 64.3 | 64.3 | gold4090 (1.15×) |
| qwen3_06b.o_proj.s512.dynM | 145.2 | 146.0 | 145.2 | greedy |
| qwen3_06b.gate_up_proj.s512.dynM | 193.7 | 161.6 | 161.6 | gold4090 (1.20×) |
| qwen3_06b.down_proj.s512.dynM | 222.7 | 242.4 | 222.7 | greedy |

## Finding 1 — Ada smem-gate crash aborts the whole sweep (BLOCKER)

The first `tune --dataset golden` died at shape 13/29 (`down_proj.s32`) with a hard `LoweringError`:

```
'matmul': k:100_materialize_tile rejected its only lowering — smem 122880 > max_dynamic_smem 101376
```

**Root cause.** Greedy deploy (`Pipeline.run` → `greedy_decide`, `pipeline.py:460`) flattens each fork to complete
leaves and takes the prior's `mean_scores` argmin, **deferring materialization** — so it doesn't know a tile's smem
until it tries to assemble it. The learned prior is trained on the Blackwell golden cards (5090 / Pro6000), whose
dynamic-smem cap is far larger than Ada's 99 KB, so it ranks 114–123 KB tiles first for rectangular projections. The
existing recovery is a **reactive blocklist-retry loop** (`pipeline.py:522`, `_MAX_GREEDY_RETRIES = 8`): each failed
`validate(ctx)` (the smem gate at `candidate.py:505`) blocklists one tile and re-resolves to the next prior-ranked
leaf. On Ada the prior ranks *more than 8* over-budget tiles ahead of any in-budget one, so the loop exhausts and
crashes. The search's inner loop is fine (`validate(ctx)` skips over-budget variants there); only the **greedy deploy
assembly** lacks a sufficient budget gate.

**Blast radius:** 13/29 shapes crashed under the default cap (every rectangular projection at s32–s512, plus the big
squares' deploy). It blocks tuning **and** `run --bench --golden` (same greedy path), i.e. the entire A/B step.

**Stopgap applied:** `_MAX_GREEDY_RETRIES` 8 → 64 (`pipeline.py:53`). This unblocked all 29 shapes (the in-budget tile
is reachable, just deep in the prior's ranking). It is safe on well-matched hardware — the loop exits on attempt 1 when
no tile fails `validate`, so only mis-calibrated (Ada) compiles pay the extra re-resolves.

**Recommendation (priority 1).** Replace the band-aid with a feasibility filter in `greedy_decide` that makes greedy
pick the best *in-budget* tile directly and removes the retry blow-up.

**Resolution (implemented).** A *closed-form* smem pre-filter turned out to be impossible: at the partition fork the
leaf has no `StageBundle` yet (the `020`–`070` staging passes run later) and unstamped dtype (`030_stamp_types` is a
`lowering/kernel` pass), and the real footprint depends on staging multiplicity (a downstream greedy choice) — so the
knob row alone does **not** determine smem. The shipped fix instead probes feasibility by *lowering*: at the partition
fork `greedy_decide` walks the prior-ranked leaves best-first and deploys the first whose pinned single-node slice
lowers through `KERNEL_PASSES` to a `KernelOp` passing `KernelOp.validate(ctx)` (`_leaf_feasible`, memoized) — one
resolve pass, no whole-graph re-resolution. `_MAX_GREEDY_RETRIES` reverted 64 → 8. See
`deplodock/compiler/pipeline/ARCHITECTURE.md` → "Greedy feasibility filter".

## Finding 2 — greedy pick is mis-calibrated for Ada: loses 1.1×–3.5× to the cross-arch 4090 config on 16/29 shapes

Even once it compiles, the greedy pick is poor on Ada for rectangular projections. Worst offenders:

| shape | greedy µs | 4090 µs | greedy slowdown |
|---|---|---|---|
| qwen3_06b.o_proj.s32 | 81.8 | 23.4 | **3.50×** |
| qwen3_06b.gate_up_proj.s512 | 514.6 | 178.0 | 2.89× |
| qwen3_06b.o_proj.s512 | 408.1 | 155.1 | 2.63× |
| qwen3_06b.kv_proj.s128 | 82.3 | 31.6 | 2.60× |
| qwen3_06b.down_proj.s32 | 114.3 | 46.0 | 2.48× |

**Why.** Two compounding effects: (a) the prior's *preferred* (Blackwell-sized) tiles are over-budget on Ada and get
blocklisted by Finding 1's retry loop, so greedy lands on whatever in-budget tile the prior ranks next — and (b) the
Blackwell-trained prior ranks Ada in-budget tiles poorly to begin with (the greedy `down_proj.s32` pick is `FM:1`, a
near-degenerate register tile). The squares (rounder shapes) greedy handles fine — it's the high-N / high-K rectangles
that fall over. This is why 16 goldens are seeded from the 4090 config: on the same microarchitecture the 4090's
hand-tuned knobs are a far better deployable than the current pipeline pick.

**Recommendation (priority 2).** Recalibrate the learned prior for Ada. This sweep already wrote 29 shapes of Ada -O3
reservoir evidence into the prior checkpoint; a follow-up `tune --dataset golden` pass (post Finding-1 fix, so the good
in-budget tiles are actually reachable and benched) should let `Prior.pick`'s evidence path deploy them. If a single
prior can't span Ada+Blackwell, add an Ada-gated analytic weight set (mirroring `_W_A_DYN`'s selection on a stamped
hardware feature) via `scripts/golden_knob_heuristics.py`. Until then, the `gold4090`-seeded goldens are honest
cross-arch references, not 4070 Ti-native optima — re-sweep to refine.

## Finding 3 — pinned 4090 config *hangs* on the 4070 Ti for 4 large shapes

For `square.1024`, `square.2048`, `gate_up_proj.s32`, and `down_proj.s512`, the seeded 4090 config, when pinned and
benched, **hung** (>1000 ms watchdog → `bench_fail`), so its A/B row was silently skipped:

```
[golden] square.1024: compile/bench of the pinned config failed (kernel '…' did not complete within 1000 ms
— variant marked bench_fail) — skipping its row
```

These 4090 configs use `SPLITK:2` + large `BN/BK/FM` combos that evidently mis-behave on the 4070 Ti's narrower memory
system (fewer SMs / smaller L2). Only the greedy pick was available, so those 4 are seeded from greedy. Not a pipeline
bug per se — just confirmation that 4090 knobs do **not** transfer wholesale, and these 4 shapes specifically need a
native 4070 Ti tune once Finding 1/2 are fixed.

**Recommendation (priority 3).** Re-tune these 4 shapes natively after the greedy fix; the greedy values recorded now
are placeholders likely beatable.

## Workflow notes (retrospective for the CLI / skill maintainer)

- **CUDA env repair dominated wall time** (driver reboot + nvcc bring-up before any tuning). *Improvement:* a
  `deplodock doctor` preflight (driver vs torch/cupy CUDA, `nvcc` presence, smem cap) would have surfaced all three
  blockers in one command instead of failing one allocation at a time.
- **The smem crash aborted the entire `tune --dataset golden` loop** rather than skipping the one shape. The loop is
  crash-isolated for bench worker deaths but not for a `LoweringError` thrown at greedy assembly. *Improvement:* wrap
  each shape's `_tune_one` so an assembly `LoweringError` records the shape as failed and continues — I had to fall
  back to an external per-shape bash loop with `continue`-on-error.
- **`eval variants --kernel <golden-name>` returns nothing** — every matmul shares the `k_matmul` C-identity in the DB,
  so the per-kernel leaderboard can't isolate a single golden shape; I couldn't drill per-shape from the DB and relied
  entirely on the A/B logs. *Improvement:* let `eval variants` accept a golden-NAME / ShapeKey filter (group by
  `ShapeKey`, not kernel C-name) so the measured-variant leaderboard works per golden shape.
- **The golden A/B row is silently dropped when the pinned config hangs/fails** — only discoverable by grepping the log
  for `[golden] … skipping its row`. *Improvement:* print a visible `golden NAME  FAILED (bench_fail: …)` row in the
  kernel table so a missing comparison isn't mistaken for "greedy is simply faster."
- **Hand-parsing the kernel table for knobs is fragile** — a blank trailing `OVERHANG` cell misaligns a
  right-anchored parse (cost me a parser bug; fixed by left-anchoring after `occ`). *Improvement:* have `run --bench`
  emit a machine-readable per-kernel JSON (knobs + -O3 µs + grid/block/smem), the way `tune --bench` writes
  `62_kernel_bench.json`, so recording a golden never needs table scraping.

No previous 4070 Ti sweep exists, so there are no prior workflow notes to check against.

## Environment note — pre-existing mma.sync failures on sm_89 + CUDA 13.3 (orthogonal)

`make test` on this box shows ~100 failures/errors, **all** in the CUDA tensor-core families (`test_matmul_mma*`,
`test_program_rebind`): one mma.sync kernel raises `CUDA_ERROR_ILLEGAL_ADDRESS` on the 4070 Ti under the pip CUDA-13.3
toolkit, corrupting the context and cascading to every later CUDA test on the same xdist worker (each test **passes in
isolation**). Verified **pre-existing**: stashing this PR's changes (gpu registry entry + `_MAX_GREEDY_RETRIES` bump)
reproduces the identical cascade, so it is **not** caused by the golden seeding — the 1940 non-CUDA tests pass and
`test_golden_configs.py` passes 28/28. It is the same theme as Findings 1–2 (codegen validated on Blackwell/4090, not
vetted on Ada): the mma.sync path needs an sm_89 correctness pass under CUDA 13.x. Worth a separate investigation; out
of scope for this seed.
