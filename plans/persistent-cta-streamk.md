# Persistent CTA + Stream-K for matmul — kill the wave tail at compute-bound sizes

**Branch:** `feature/persistent-cta-streamk`
**Origin:** `plans/matmul-cublas-gap-2026-05-30.md` § "Five new optimization kinds to feature";
discussion 2026-05-30 ranking against `.reuse` hints and CTA cluster + DSMEM.
**Effort:** 1–2 weeks
**Risk:** medium-high — intrusive change to grid scheduling, but the data says this is the
highest-ceiling lever on 2048³.

## Problem

The matmul kernel launches **160 CTAs on 170 SMs** at 2048³ fp32 (`10 × 16` grid from
`208 × 128` block tile). That's **0.94 waves**: 160 SMs busy for one wave duration `T`,
10 SMs idle for the same `T`. Wasted compute = `10 / 170 = 5.9 %` of wall clock — **larger
than the 4 % gap to cuBLAS**.

Cross-check from the ncu data (`plans/matmul-cublas-gap-2026-05-30.md`):

| metric | golden (ours) | cuBLAS |
|:---|---:|---:|
| Grid size | 160 CTAs | 640 CTAs |
| Waves per SM | 0.94 | 3.76 |
| Tail fraction | 5.9 % | 5.3 % |
| Latency at 2048³ | 275 µs | 265 µs |

cuBLAS amortizes the same proportional tail (`(640 mod 170) / (4·170) = 5.3 %`) across 4 waves,
but the *absolute* waste is identical — both lose ~10 SM-T of compute. The only way to
recover is to land on an integer-multiple-of-170 CTA count, or to keep all 170 SMs busy for
exactly the same time via Stream-K's per-CTA work-range distribution.

We confirmed via deplodock knobs that **Split-K=5 on top of the wide tile measures equal to
golden** (`275 µs` either way) — Split-K's atomicAdd cost cancels its wave-amortization win on
this shape. Stream-K with the **atomic-free combine** breaks that tie because the per-tile
output cell is written exactly once, by exactly one CTA.

## What this delivers

Theoretical ceiling: **save the entire 5.9 % tail = ~16 µs**, landing at ~259 µs vs cuBLAS's
265 µs. That would be the *first* deplodock matmul that crosses 100 % of cuBLAS at this size.

Practical estimate after Stream-K bookkeeping overhead:

- Work-range computation per CTA (~10 cycles)
- Scratch buffer write for boundary tiles (~1 % of cells)
- Combine kernel for boundary tiles only (~5 µs at 2048³ memory bandwidth)

Net realistic: **3–5 % wall-clock improvement on 2048³** → 261–267 µs → 98–100 % of cuBLAS.
The variance vs the theoretical ceiling lives in M2 (wave quantization free-money) and M4
(combine kernel cost).

## Why this is the right TMA-native lever

TMA descriptors decouple tile addressing from the launch grid. A persistent CTA can update
its `cuTensorMapEncodeTiled` parameters per iteration — or equivalently use the same
descriptor with different `(c0, c1)` coordinates per `cp.async.bulk.tensor.2d` call — for
**zero per-iteration setup cost**. This is what cuBLAS's sm_80 forwardCompat kernel *can't*
exploit, because cp.async.commit_group has a depth-8 ceiling and per-tile bookkeeping is
synchronous. Persistent CTAs pair naturally with TMA in a way they don't with cp.async.

## Design — two-phase rollout

The plan ships in two independent pieces because the first piece is free money and the
second piece is the larger structural change.

### Phase A — Wave-quantized matmul tile (concrete A/B sweep, no compiler changes)

Without changing the kernel structure at all: find a tile whose CTA count gets us closer to
an exact multiple of `num_sms` than the golden 160 CTAs do. The waste curve is sawtooth, not
monotonic — getting *closer* to `num_sms` from above (e.g. 176 CTAs at 1.04 waves) is
catastrophic, not free. Concretely on 170 SMs:

| CTAs | waves | wasted SM-T  | waste of total | comment                                  |
|-----:|------:|-------------:|---------------:|:-----------------------------------------|
|  160 |  1    | 10           |       **5.9 %**| **golden** — 10 SMs idle for the wave    |
|  165 |  1    | 5            |          2.9 % | better                                   |
|  168 |  1    | 2            |          1.2 % | better                                   |
|  169 |  1    | 1            |        **0.6 %** | best single-wave                        |
|  170 |  1    | 0            |          0.0 % | perfect, hard to land exactly            |
|  171 |  2    | 169          |         49.7 % | **catastrophic** — wave 2 has 1 CTA      |
|  176 |  2    | 164          |         48.2 % | **catastrophic** — original plan was wrong here |
|  256 |  2    |  84          |         24.7 % | bad                                      |
|  320 |  2    |  20          |          5.9 % | same as golden                           |
|  336 |  2    |   4          |          1.2 % | better — 2 waves with tight last         |
|  340 |  2    |   0          |          0.0 % | perfect 2-wave                           |
|  510 |  3    |   0          |          0.0 % | perfect 3-wave                           |

The sweet spots are CTA counts **just under** a multiple of 170. On 2048×2048 with reasonable
factor pairs (BM·BN = 256 = 8 warps; BM·FM and BN·FN reasonably square per-block) the only
sweet-spot landings are at **169 CTAs** (13 × 13 grid, 160 × 160 block) and **338 CTAs**
(13 × 26 grid, 160 × 80 block, but block size only 64 threads — too thin).

#### Exact knob sets to test

Run each via `deplodock run -c "torch.matmul(torch.randn(2048,2048),torch.randn(2048,2048))"
--bench --warmup 50 --iters 200`. Methodology matches `plans/matmul-cublas-gap-2026-05-30.md`
§ "Hand-optimized kernel variants".

| # | knobs                                          | block tile | grid  | CTAs | waste | cells/thr | LDS/iter | hypothesis                |
|--:|:-----------------------------------------------|:-----------|:------|-----:|------:|----------:|---------:|:--------------------------|
| 0 | `BM=8,FM=26,BN=32,FN=4` (golden)               | 208 × 128  | 10×16 |  160 | 5.9 % |       104 |       30 | baseline                  |
| 1 | `BM=8,FM=20,BN=32,FN=5`                        | 160 × 160  | 13×13 |  169 | 0.6 % |       100 |       25 | **headline candidate** — wave-quantized, fewer LDS/iter |
| 2 | `BM=16,FM=10,BN=16,FN=10`                      | 160 × 160  | 13×13 |  169 | 0.6 % |       100 |       20 | square per-thread tile, lowest LDS/iter |
| 3 | `BM=32,FM=5,BN=8,FN=20`                        | 160 × 160  | 13×13 |  169 | 0.6 % |       100 |       25 | swap M/N threading; tests if 8-row warp layout matters |
| 4 | `BM=8,FM=26,BN=16,FN=4`                        | 208 × 64   | 10×32 |  320 | 5.9 % |       104 |       30 | same tail as golden but 2× more blocks → better L2 reuse on B |
| 5 | `BM=8,FM=13,BN=32,FN=8`                        | 104 × 256  |  20×8 |  160 | 5.9 % |       104 |       21 | same CTAs as golden, but B-strip wider (lower LDS/iter, possibly better .reuse pattern) |
| 6 | `BM=8,FM=21,BN=32,FN=4` (anti-pattern probe)   | 168 × 128  | 13×16 |  208 | 38.8 %|        84 |       25 | **expect regression** — confirms the sawtooth |

Variants 1–3 all land at the **169-CTA single-wave** sweet spot (0.6 % waste, vs golden's
5.9 %); they differ only in the per-thread tile shape, which is the live variable. Variant 1
keeps the BM=8, BN=32 threading from golden so the diff is purely tile dimensions. Variant 2
goes square to cut LDS/iter from 25 to 20. Variant 3 swaps M and N threading to see if warp
layout matters at this tile.

Variant 4 keeps golden's per-thread tile but goes thin-in-N to double the block count without
moving the tail percentage — pure L2-reuse probe. Variant 5 widens the B strip (FN=8 instead
of 4) to test whether more B-cells per A-load helps the FFMA cluster's `.reuse` density.

Variant 6 is the **negative control**: confirms the sawtooth. We expect it to be worse than
golden because it lands at 208 CTAs (1.22 waves with 132/340 = 38.8 % waste). If variant 6
ties or wins, the wave-quantization theory is wrong and we should re-examine the
bottleneck. (Strongly expect regression.)

#### Acceptance criteria

- **Variant 1 or 2 beats golden by ≥ 3 %** (target: ~267 µs vs golden's 275 µs) → wave
  quantization is the real lever; proceed to Phase B with confidence. Pin the winner as the
  new default tile for this shape.
- **All variants tie golden within ± 1 %** → the 5.9 % tail isn't actually on the critical
  path (likely because LDS-pressure or FFMA-pipe dominates and tail just slots into idle
  cycles). Phase B unlikely to help; reconsider before investing.
- **Variant 1 or 2 *regresses* vs golden** → the smaller per-thread tile (100 vs 104 cells)
  costs more than the tail saves; means the LDS-per-FFMA delta matters more than tail.
  Phase B still viable but expected gain shrinks.
- **Variant 6 ties or beats golden** → wave-quantization theory wrong; pause Phase B until
  the actual bottleneck is re-identified.

All four outcomes are informative. The decision to invest in Phase B should be driven by the
variant 1/2 result; if it ties or regresses, the 5.9 % "tail waste" was nominal and Stream-K's
ceiling is correspondingly smaller than the back-of-envelope 16 µs.

Phase A lands as: **commit pins the winning tile as the priority candidate in
`_default_tile` for `K_blocks ≥ 12, M_blocks ≥ 12, N_blocks ≥ 12` shapes on this hardware**.
No new pass, no autotune-priority weight tweaks — just a fixed tile preference at this shape
range, gated by the measured numbers from the variant sweep.

### Phase B — Stream-K with atomic-free combine (the structural change)

Launch exactly `num_sms` CTAs (170 on RTX 5090). Each CTA processes a contiguous range of
`[start_mac, end_mac)` MAC iterations from the total `M × N × K` work, computed at launch
time:

```
total_macs = (M / BM) × (N / BN) × (K / BK)          // in K-block units
per_cta    = ceil(total_macs / num_sms)
start[i]   = i × per_cta
end[i]     = min((i+1) × per_cta, total_macs)
```

Inside each CTA, walk the assigned range tile by tile. The vast majority of tiles fall
entirely within one CTA's range (full K-loop); boundary tiles get **partial K-loops** at
either the head or tail of the range. Three cases per output tile:

- **Owner CTA** (`start_mac` and `end_mac` both inside this tile's K-range): full K-loop,
  result goes directly to `output[m, n]`. The 1-CTA-per-tile common case.
- **Head partial** (`start_mac` mid-tile, range extends past tile end): K-loop from mid to
  end, partial sum goes to `scratch[partial_idx, m, n]` — `partial_idx` is the boundary
  index.
- **Tail partial** (`start_mac` at tile start, `end_mac` mid-tile): K-loop from start to mid,
  partial to scratch.

After all 170 CTAs complete, a tiny **combine kernel** sweeps the scratch buffer and reduces
boundary tile partials into the output. With per-CTA work ~94 % of one tile, there are at
most `num_sms - 1` boundary tiles → ≤169 partials × `BM × BN` cells = 169 × 26624 cells = 4.5
M floats = 18 MB scratch. Single-kernel bandwidth-bound reduce at ~1.5 TB/s → ~12 µs combine
cost.

The combine kernel can reuse the existing `017_atomic_free_splitk.py` infrastructure
(workspace-then-reduce pattern is already in the codebase, just for the SPLITK partials axis).

### Where each piece lives

**Phase A** — variant sweep, then pin the winner:

- No compiler changes during the sweep itself. Run the six knob configurations from
  "Exact knob sets to test" above via `DEPLODOCK_KNOBS=...` on the standard `deplodock run
  -c "torch.matmul(torch.randn(2048,2048), torch.randn(2048,2048))" --bench` invocation.
- Once a winning tile is identified, pin it in `deplodock/compiler/tuning.py:_default_tile`
  as the preferred candidate when `(M_blocks, N_blocks) × waves` lands near the sweet-spot
  region for the detected `num_sms`. One concrete `if num_sms == 170 and M >= 2048 and
  N >= 2048: return (winning_BM, winning_FM, winning_BN, winning_FN)` clause, not a generic
  scoring weight.

**Phase B** — new tile pass + IR primitive + runtime support:

- New IR primitive in `deplodock/compiler/ir/tile/ir.py` — `PersistentTile(num_ctas, body)`
  wrapping the work-range loop. Lowered by a new pass to a CUDA `for` loop reading two new
  kernel args `cta_work_start`, `cta_work_end` (one per CTA, supplied at launch).
- New pass `deplodock/compiler/pipeline/passes/lowering/tile/018_persistent_streamk.py`
  (runs after `017_atomic_free_splitk`, before staging). Rewrites a `GridTile` over `(m_b,
  n_b, k_o)` into:
  ```
  PersistentTile(num_ctas, body=
      for mac_idx in cta_work_start..cta_work_end:
          (m_b, n_b, k_chunk_start, k_chunk_end) = unravel(mac_idx, M_blocks, N_blocks)
          if (k_chunk_start == 0 and k_chunk_end == K_blocks):
              <full K-loop, write to output>
          else:
              <partial K-loop, write to scratch[boundary_idx]>
  )
  ```
- Reuse `017_atomic_free_splitk.py`'s combine TileOp construction for the post-pass reduce
  kernel — sum scratch partials into the boundary tiles' output cells.
- New knob `STREAMK = Knob("STREAMK", KnobType.BOOL, default=False)` — off until validated.
  Autotune-selectable later.

### Runtime support

- `deplodock/compiler/backend/cuda/program.py` — at launch time, query
  `cp.cuda.Device().attributes["MultiProcessorCount"]`, compute the per-CTA work range
  `(start, end)`, push as two `int32[num_ctas]` arrays in gmem (cheap — 1.4 KB for 170 CTAs).
  Pass the arrays as kernel args; the kernel indexes by `blockIdx.x`.
- Allocate the scratch buffer at compile time (size = `min(num_ctas - 1, total_tiles) ×
  BM × BN × 4` bytes). For 2048³ at 170 CTAs and 208 × 128 tile = ~18 MB. Reusable across
  invocations of the same kernel.

## Milestones (single branch, commit after each `make test` passes)

Per `feedback_single_branch_milestones`.

**M1 — Variant sweep (Phase A — measurement only, no code change)**

- Run the six knob configurations from "Exact knob sets to test" with
  `DEPLODOCK_KNOBS=...` against the standard 2048³ benchmark. Capture median over 200
  iterations × 5 trials per variant.
- Record the table in the commit message (or an intermediate doc); identify the winner.
- **No `make test` / `make lint` for M1** — it's pure benchmarking, no code touched.
- **Success criterion**: variant 1 or 2 beats golden by ≥ 3 % → proceed to M2. Variants
  tie golden within ± 1 % → halt Phase B (the tail wasn't on the critical path; revisit
  bottleneck). Variant 6 ties or beats golden → wave-quantization theory is wrong, escalate.

**M2 — Pin the winning tile (Phase A — code change)**

- If M1 produced a winner, hardcode it in `deplodock/compiler/tuning.py:_default_tile` as
  the preferred tile for `M ≥ 2048 ∧ N ≥ 2048 ∧ num_sms == 170` matmul shapes. Single
  `if` clause, not a generic scoring function.
- Add `num_sms` to the compile context (read once via
  `cp.cuda.Device().attributes["MultiProcessorCount"]`); used by the new `if` and by Phase B.
- Validate: `make test`, `make lint`, re-run the bench to confirm the new default ties the
  M1 winner.

**M3 — `PersistentTile` IR primitive + lowering**

- Add `PersistentTile` to `deplodock/compiler/ir/tile/ir.py` (frozen dataclass, `num_ctas:
  int`, `body: Body`, `work_range_axis: Axis`). Hashable per `feedback_stmt_hashable`.
- Add lowering in the materializer (`100_materialize_tile.py`) — emit a `for` loop reading
  `cta_work_start[blockIdx.x]` / `cta_work_end[blockIdx.x]`.
- Add the new pass `018_persistent_streamk.py` — for now, only handles the trivial case
  where every CTA gets exactly one tile (no K-splitting). Off behind `STREAMK=0` knob.
- Unit tests: tile-IR round-trip, lowering produces compilable CUDA.
- **Success criterion**: `STREAMK=1` with the trivial case launches `num_sms` CTAs each
  doing 1 tile (or 0). On 2048³ this is identical workload to golden; perf neutral. Confirms
  the scaffolding works end-to-end.

**M4 — Stream-K work distribution with scratch + combine**

- Extend `018_persistent_streamk.py` to compute per-CTA `(start_mac, end_mac)` ranges
  spanning the full `M × N × K` work.
- Emit head/tail partial K-loop variants in the kernel body. Boundary tile partials write
  to scratch.
- Add the combine TileOp (reusing `017_atomic_free_splitk.py`'s reduce skeleton).
- Launch-time runtime: compute per-CTA work ranges, allocate scratch, push arrays as kernel
  args.
- Validate: accuracy unchanged on TinyLlama block test, 2048³ matmul, and Qwen3-Embedding
  layer-0.
- **Success criterion**: end-to-end run produces correct output; latency on 2048³ within
  ±10 % of golden (the structural win is in M5 after the combine kernel is tuned).

**M5 — Combine kernel tuning + scratch reuse**

- Tune the combine reduce kernel: vectorize loads, pick BM/BN for the reduce, autotune.
- Cache the scratch buffer between invocations (per-shape, per-`num_sms`).
- Validate on 2048³: should now show the structural improvement.
- **Success criterion**: 2048³ at ≥ 99 % of cuBLAS (`≤ 268 µs` on RTX 5090).

**M6 — Autotune integration**

- Add `STREAMK` to the autotune fork search.
- Lock the autotuner so STREAMK forks only at the right shapes (compute-bound matmul, grid
  count < 2 × num_sms, atomic-free-splitk path already eligible).
- Validate: a clean autotune on 2048³ picks STREAMK; on 8192³ it should not (already
  saturated waves).

**M7 — Larger shape validation**

- Run on 1024³, 4096³, 8192³, 16384³. Compare against
  `project_tma_perf_findings.md`'s baseline. Expect:
  - 1024³: similar or slight win (164 / 170 = 0.96 waves, same regime)
  - 2048³: 3–5 % win (the target)
  - 4096³: neutral (already 3.7 waves, tail negligible)
  - 8192³: neutral or slight regression from scratch-buffer overhead (already wave-saturated)

**M8 — Docs**

- Update `deplodock/compiler/ARCHITECTURE.md` and the relevant pass list.
- Update `plans/matmul-cublas-gap-2026-05-30.md` to note that the persistent-CTA + Stream-K
  line item has shipped, with measured numbers.
- Add a new article section if the user wants it: "Persistent CTAs and Stream-K: leaving
  the launch grid behind" with the before/after kernel diff.

## Validation checklist (per `CLAUDE.md` § "Before committing")

After every milestone:

1. `make test` — all tests pass
2. `make lint` (or `make format` then `make lint`)
3. Update `ARCHITECTURE.md` in any directory touched

After M2, M4, M5, M7 specifically:

4. Capture before/after `deplodock run --bench` numbers in the commit message
5. Per `feedback_perf_eval_scope`: one or two shapes per milestone, isolated DB, no full
   sweeps unless asked. The shape matrix in M7 is the *one* full sweep.
6. Re-check the `--ir cuda` dump after M4 to confirm the persistent-CTA structure is what
   we expected (one `for` loop over work range, three branches for full / head / tail).

## Risks and edge cases

- **Cross-CTA work range computation accuracy.** The unravel `(mac_idx → m_b, n_b,
  k_chunk_range)` must agree across CTAs that share a tile. Off-by-one here silently
  corrupts boundary tile results. Mitigation: a self-test in the kernel (each CTA computes
  the next CTA's `start_mac` and asserts equality with its own `end_mac`) during M4.
- **Scratch buffer size at small shapes.** At 128×128×16384 (the Split-K beneficiary shape
  in the article), `BM × BN = 64 × 64 = 4096` cells × 170 partials = 2.8 MB scratch.
  Negligible. At 16384³ with the golden tile, 16K^2 / (208 × 128) = 9856 tiles, partials
  bounded by `min(num_ctas - 1, tiles) = 169`, scratch = 18 MB — same as 2048³. Bounded by
  CTA count, not problem size. Safe.
- **TMA descriptor reuse.** Each persistent CTA needs to swap the descriptor target tile
  per iteration. `cuTensorMapEncodeTiled` is a host call; we must instead use the **same
  descriptor with different `(c0, c1)` coordinates** per `cp.async.bulk.tensor.2d`. The
  existing `cp_async_bulk_tensor_2d` wrapper already takes the coordinates as args; the
  per-iteration update is one register update per coordinate. Free.
- **`num_sms` is device-specific.** Compile-time `num_sms` baked into kernel args breaks
  portability across GPUs in a deployment. Mitigation: pass `num_sms` as a runtime kernel
  arg; the work-range arrays are also runtime. The cached kernel is portable across SM
  counts.
- **Interaction with autotuner's existing `SPLITK` fork.** Stream-K and Split-K both split
  the K axis; they're mutually exclusive on the same TileOp. Gate at the fork point: if
  `STREAMK=1` is selected, force `SPLITK=1`.
- **Interaction with `ATOMIC_FREE_SPLITK`.** The combine reduce kernel is built on the same
  workspace pattern. Reuse the helper from `017_atomic_free_splitk.py` to avoid duplicating
  the construction. Stream-K's scratch shape is `(boundary_count, M, N)` vs Split-K's `(S,
  M, N)` — the dimension that changes is the outer one; the combine logic is identical.
- **Phase A might already extract most of the win.** The 169-CTA single-wave landings
  (variants 1–3) take the tail from 5.9 % to 0.6 % — capturing essentially all the
  recoverable tail in one wave. If they deliver the expected ~3 % wall-clock improvement,
  Phase B's marginal value shrinks to "covering the shapes Phase A's pinned tile doesn't
  reach". Re-evaluate after M2 — if Phase A alone delivers ≥ 98 % of cuBLAS at 2048³, Phase
  B becomes optional polish rather than the headline lever.
- **Persistent CTA blocks SM-level dynamic scheduling.** Each SM is locked to its assigned
  work range; if one CTA finishes early (e.g. boundary-tile path is shorter than full-tile),
  that SM sits idle until the slowest CTA finishes. Mitigation: compute work ranges in
  *MAC count*, not in *tile count* — that's what Stream-K already does, and it's the reason
  Stream-K beats a naive "give each SM N/170 tiles" scheme.
- **Compile time impact.** The new pass and combine TileOp construction add ~1 second per
  matmul compile. Acceptable given the cache; deplodock's nvcc-cubin caching means the
  pass runs once per unique shape.

## Out-of-scope (separate follow-ups)

- **Stream-K on non-matmul kernels.** SDPA's score×value has the same structure; the same
  technique applies but the per-tile reduction is along a different axis. Defer until the
  matmul version is shipped and measured.
- **Multi-cluster Stream-K** (Stream-K within a CTA cluster). Combines this plan with the
  CTA cluster + DSMEM lever from `plans/matmul-cublas-gap-2026-05-30.md`. Productive on
  larger shapes but requires both features to land first.
- **Dynamic work-stealing.** Stream-K assigns work ranges statically at launch. Dynamic
  work-stealing (where idle CTAs pull work from a global queue) handles input-dependent
  load imbalance better — irrelevant for fixed-shape SGEMM, useful for sparse / dynamic
  workloads.
- **Combine kernel fusion.** The combine reduce kernel is a separate launch. CUTLASS-style
  Stream-K issues the combine from inside the last CTA via a global memory fence. Saves one
  launch (~5 µs). Worth doing only after the kernel-launch cost shows up as a measured
  bottleneck.

## Expected outcome

- **Phase A alone** (M1–M2, ~1 day): if a 169-CTA variant wins, ~3 % improvement on 2048³
  (~267 µs vs golden 275 µs, ~99 % of cuBLAS). No IR changes; one `if num_sms == 170 …`
  clause in `_default_tile`. Phase A also produces an unambiguous **go/no-go signal** for
  Phase B — if 169-CTA tiles tie golden, the tail wasn't on the critical path and Stream-K's
  ceiling shrinks below the engineering cost.
- **Phase A + B** (M1–M5, ~1–2 weeks): 3–5 % improvement on 2048³ (~261–268 µs) — at the
  low end ties cuBLAS at 265 µs; at the high end beats it by 4 µs. First deplodock matmul
  to cross 100 % of cuBLAS on the article's headline shape.
- **No regression** on shapes that already saturate waves (4096³+).
- **Stream-K is shape-selective** — autotune learns to fork it on at the right shapes
  (compute-bound matmul with grid count < 2 × num_sms).

The honest hedge: the theoretical 16 µs ceiling assumes zero combine-kernel cost and zero
work-range overhead. Realistic delivery is 8–12 µs, which lands at 98–100 % of cuBLAS. If
Phase A surprises us by extracting more than expected, Phase B becomes polish — that's a
better outcome, not a worse one.
