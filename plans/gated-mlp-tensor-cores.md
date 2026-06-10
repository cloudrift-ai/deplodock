# Gated MLP on tensor cores: fusion split + dual-Mma + SwiGLU fragment epilogue

**Branch:** TBD (follow-up to `fix/tma-boxdim-and-pipeline-hang` / PR #218)
**Status:** planned — design agreed, not started. Closes finding 4 of `plans/qwen3-embedding-layer0-tune-findings.md`:
the fused RMSNorm + gate/up + SwiGLU kernel (`k_linear_mean_reduce`, out (32, 3072), K=1024) runs ~76–80 µs scalar vs
torch.compile's 14 µs — the largest single contributor to the layer's 0.5x-vs-eager deficit.

## Problem

The loop-fusion pass merges the whole chain RMSNorm → gate_proj & up_proj → silu·mul into ONE `LoopOp`
(`04_loop_fusion` dump, annotated):

```
for a0 in 0..32                             # M: rows
    for a1 in 0..1024                       # norm reduce (full row)
        in3 = load add_5[0, a0, a1]
        acc0 <- add(acc0, in3·in3)
    v4 = rsqrt(acc0/1024 + eps)             # row stat
    for a2 in 0..3072                       # N
        for a3 in 0..1024                   # matmul reduce
            in4 = load add_5[0, a0, a3]     # x re-read
            v6  = nw[a3] · (in4 · v4)       # ← computed A operand
            acc1 <- acc1 + Wg[a3,a2] · v6   # gate
            acc2 <- acc2 + Wu[a3,a2] · v6   # up
        mul_12[0, a0, a2] = silu(acc1) · acc2
```

Four compounding consequences, each mapped to a line above:

1. **Computed A operand** (`v6`): the matmul multiplies a weight Load against a 3-op cone, not a Load. `ldmatrix`
   feeds MMA fragments from staged smem; a computed operand has no buffer to stage → the warp tier is structurally
   unreachable, independent of every other rule. This is the root blocker.
2. **Sequential whole-row dependency** (`v4` consumed inside `a3`): not one normalized value exists until the `a1`
   reduce completes — the kernel is inherently two-pass over K. This is also what nests the K pipeline inside the
   row-cell loop (the finding-2 TMA hang shape). Warp specialization / `setmaxnreg` cannot remove this dependency:
   WS schedules data movement against compute; the producer would stall at the same barrier. (Assessed and
   rejected as the primary fix — they re-enter the picture only as the existing transport optimizations on the
   final gemm kernel.)
3. **Dual accumulator** (`acc1`/`acc2`): the reduce body is 4 Loads / 2 Accums, failing the canonical
   2-Load/1-Assign/1-Accum purity cell (`tile/_atom.py`), and the SwiGLU tail consumes both accumulators — blocked
   by the multi-accumulator rule of `classify_fragment_epilogue` (everything else about that epilogue already
   folds: all ops are `op_to_expr`-renderable, single Write, nothing escapes).
4. **Redundant traffic**: `x` is re-read from gmem per K-tile, and the per-thread scalar lowering computes the
   full 1024-element row stat redundantly in EVERY thread sharing the row.

Why the fusion happened: `loop/fusion/010_merge_loop_ops.py` is greedy with two blowup guards (`_total_work`,
`_total_reads`, factor 8) — a pure cost model over arithmetic and traffic. Fusing the norm in looks mildly
*positive* on reads (saves the xn round trip) while silently destroying a ~10× ALU-tier upgrade the cost model
cannot see. Note the model's INPUT norm did not fuse — but only by accident: it has 3 consumers (q/k/v) and the
rule only fuses sole-consumer producers; the post-attn norm ends up sole-consumer after gate/up/silu·mul merge
into one consumer first.

## Design decision: split now, smem-resident fused kernel later

Two viable designs make the A operand stageable:

- **A. Fusion split**: the norm materializes `xn = x·rsqrt(mean(x²)+eps)·nw` to gmem ((32, 1024) fp16 = 64 KB,
  L2-resident); the MLP kernel consumes `xn` as a plain Load. Cost: one 64 KB round trip + one launch (≈ free
  under CUDA graphs). This is torch.compile's design, and the input-norm → q/k/v path already demonstrates it in
  this very graph.
- **B. Compute-staged fused kernel**: keep one kernel; stage the `x` row tile into smem once (2 KB/row — K=1024
  is the whole reduce extent, so the A slab is CTA-resident, no ring), compute the row stat cooperatively from
  smem, run the normalize cone as a `StageBundle.compute` phase writing the normalized slab, `ldmatrix` from it.
  The `HOIST_COMPUTE` machinery (`030_hoist_invariant_compute` + `emit_compute_phase`) is the seed of exactly
  this, but: its cone rule requires deps to stay inside stage-source Loads (the norm cone depends on a
  *reduction-derived* scalar), the row stat needs a cooperative in-kernel prologue, and the mma + TMA paths
  reject compute bundles outright ("emitted SYNC and never reach this rule").

B strictly dominates A on the steady state (no round trip; reads `x` once) but its ceiling advantage at these
shapes is ~1–2 µs, while it needs three new features beyond what A needs. Both need dual-Mma. **Decision: ship A
(M1–M3), keep B as M4** — and long-term, fuse-vs-split is precisely the kind of decision that should be a
structural fork the outer tuner compares as terminals (`plans/structural-forks-in-two-level.md`) rather than
either hard policy.

Expected after M1–M3: norm kernel ~1–2 µs (the coop-reduce kernels in this layer run 1–2 µs) + one gated-pair MMA
kernel ≈ 2× the down_proj-class gemm sharing one A staging ≈ 12–16 µs, SwiGLU folded free → **~15–18 µs total vs
76–80 today** (tcompile: 14). M4 closes the remaining gap and should beat tcompile.

## M1 — fusion guard: no reduction-derived cones inside a matmul-reduce body

> **Superseded — shipped as a partition-level structural split instead** (branch `feature/fusion-structural-fork`).
> A fusion-site guard/oracle can't work: decomposed chains assemble the matmul multiply-last, so no standalone node is
> ever atom-eligible mid-fusion (verified by probe — the scale→matmul chain merges through 5 steps with neither side
> eligible at any step). By partition time the fused body is final and the demotion is visible order-independently, so
> the split lives in `lowering/tile/010_partition_loops`: when `_split_demoted.try_split_demoted` finds a demoted
> matmul whose CLEAN gemm would reach the warp tier, the planner offers `[fused tree, OptionFork(split Graph,
> {UNFUSE: True})]` — a producer materializing the cone (with its prologue deps) to an `xn` intermediate + the clean
> gemm loading it. Greedy `compile`/`run` never pick the structural option (kernel sets unchanged cold); `tune`
> explores both inside the op's slice; `DEPLODOCK_UNFUSE` pins either branch. The M1 guard text below is kept for the
> record. NOTE for M2/M3: the split's gate is `is_atom_eligible` on the clean gemm, so the gated-MLP kernel (dual
> accum) starts offering the split exactly when M2 generalizes the purity cell — no extra wiring needed. Also found
> while landing this: the mma cell tagger rejects Linear-derived (transposed-B) operands and leaks an un-consumed
> `AtomTile` to render (crash, not a graceful scalar fallback) — fixing that is a prerequisite for the M3 perf goal.

In `010_merge_loop_ops.rewrite`, after `splice_graph` builds `merged` (the rule already constructs it before the
blowup checks), add a structural guard:

> Refuse the fusion if, in `merged`, any **matmul-reduce** loop body (`is_matmul_reduce`) contains an `Assign`
> whose value derives (transitively, through the SSA chain) from an `Accum` defined **outside** that loop.

This is the direct statement of "the splice would put a reduction-derived cone inside the canonical matmul cell"
— the exact property that kills warp-tier eligibility (the gate's operand-dtype walk and purity rule both trip on
it). Checking the *merged* body makes the guard order-independent: it doesn't matter whether the norm fuses into
the matmul or the matmul pair forms first.

Notes / scope:

- Matmul→elementwise stays fuseable (down_proj + residual: the producer reduce is consumed *post*-reduce in the
  consumer — lands outside any matmul-reduce body). Matmul→norm stays fuseable (consumer reduce is not
  matmul-reduce). Elementwise→matmul stays fuseable (no producer Accum). Only reduction→inside-matmul-K dies.
- **Side effect on finding 5 (review during M3):** the softmax-stats → P@V fusion (`38c877`) has the same shape
  (stats consumed inside the P@V reduce), so the guard would split it too — P@V becomes a *pure* matmul (warp-tier
  eligible, no prologue!) at the cost of materializing the post-softmax probabilities. The scores matrix is
  ALREADY materialized between `b2ab33` and `38c877` today, so this changes memory asymptotics not at all at the
  current no-online-softmax design — plausibly a win (P@V on MMA), but it must be measured, not assumed. If it
  regresses, narrow the guard (e.g. require the producer reduce extent == consumer matmul K extent AND the
  producer to be normalization-shaped) or gate it on a knob until the structural-fork machinery can A/B it.
- Regression safety: re-run the TinyLlama block bench (`scripts/bench_block.py`) that calibrated
  `_BLOWUP_FACTOR`, plus the full layer-0 tune. The guard must not unlock the harmful silu→down_proj /
  up_proj→down_proj nestings documented in the factor sweep.
- Tests: loop-fusion unit test asserting the norm→matmul pair stays split and the merged kernel set for the MLP
  slice is {norm, gated-matmul}; the down_proj+residual and matmul→norm fusions still merge.

## M2 — dual-Mma: two matmuls sharing A, two C fragments, multi-fragment epilogue

After M1 the MLP kernel's reduce body is `[Load xn, Load Wg, mul, Accum, (Load xn'), Load Wu, mul, Accum]`
(020_dedup_loads should collapse the xn re-load; verify) with the SwiGLU tail consuming both accumulators. Extend
the mma path from one matmul cell to N cells sharing the A operand (N=2 here; write it N-ary, it's no harder):

1. **Purity rule** (`tile/_atom.py` predicate): generalize the canonical cell to "one shared A Load + N B Loads +
   N multiplies (each `(A, B_i)`) + N Accums, nothing else". Keep rejecting cells where the multiplies don't all
   share the same A name (a true 2-independent-matmul body has no shared staging win and doubles the fragment
   pressure for nothing — let it stay scalar until measured).
2. **Multi-accumulator blocker** (`classify_fragment_epilogue`): allow `accs_used` > 1 iff every acc is one of
   the matmul accumulators passed in (they all become fragments). `EpilogueSlice.acc` → tuple.
3. **Cell tagging** (`tile/011_lower_atom_cell._try_tag_here`): emit N `Mma` stmts sharing `a_load`'s SSA name,
   one per (B_i, Accum_i) pair, atom spec identical.
4. **Lowering** (`kernel/005_lower_atom_tile`): `_scan_cell` collects all Mmas (shared `a` seed, distinct `b`/`c`
   seeds); emit one A `RegFragment` + N B fragments + N C fragments; per reduce site `ldmatrix A` once then
   `(ldmatrix B_i, MmaSyncPtx_i)` per i. `RegEpilogue` gains the acc→frag name mapping (today it assumes the
   single `RegStore.frag`); the `_rewrite.register` handler renames every referenced fragment so per-cell
   replication keeps working.
5. **Staging / enumeration**: nothing structural — `STAGE` is already a bitmask over ranked candidate buffers
   (3 sources: xn, Wg, Wu) and the smem budget check already prices the slabs. Expect the B slab footprint to
   double; the RING reorder and `validate` handle it. Verify the WS producer split handles two B sources (it
   fans TMA issues per source already).
6. **Tests**: extend `test_matmul_mma_residual.py`'s harness with a hand-built dual-matmul LoopOp + SwiGLU tail —
   compile-only gating (admitted; 3-accum and non-shared-A variants stay blocked) + CUDA accuracy vs numpy
   (`silu(x@Wg)·(x@Wu)`, FM=1 and FM>1), plus the real `--code` MLP slice end-to-end.

Risks: register pressure (2 B + 2 C fragments per cell — watch `regs`/occupancy in the bench table; FM/FN hints
may need a tighter cap for N=2 via `_MAX_CELLS_PER_WARP_CELL`); `dedup_replicated` must not fold the two B
chains (they differ by buffer — content-keyed CSE is safe, but assert it); `place_inits` with two C fragments
(zero-init at declaration should make it a no-op — verify).

## M3 — verification

- `tune --code` the MLP slice (the `_build_mlp_slice` graph from `test_use_tma_gates.py`, minus the norm after
  M1 splits it) and `run --bench` A/B vs eager + tcompile; target ≤ 20 µs for norm + gated pair.
- Full layer: clean `tune Qwen/Qwen3-Embedding-0.6B --layer 0 --clean --bench`; expect the per-kernel table to
  show the norm kernel + an MMA gated-pair kernel; zero bench_fails stays mandatory.
- Finding-5 side effect: compare the SDPA kernel rows before/after M1 (see the guard note); record the outcome
  in the findings file either way.
- TinyLlama block bench unchanged-or-better (fusion-guard collateral check).
- `make test` / `make lint`; update `pipeline/ARCHITECTURE.md` (fusion guard note in the `loop/fusion` row,
  dual-Mma in the `lowering/kernel` row + purity description), `ir/ARCHITECTURE.md` (RegStore/RegEpilogue
  multi-fragment), and the findings file.

## M4 (follow-up) — compute-staged fused kernel, as a structural fork

The smem-resident single-kernel design, sequenced as three independently testable features:

1. **Cooperative stat prologue**: per-row sum-of-squares computed once per row tile from the staged `x` slab
   (the `BR`/warp-shuffle combine machinery exists for reduce kernels; wire it as a stage-compute prologue).
2. **Reduction-aware compute cones**: extend `030_hoist_invariant_compute`'s cone rule to admit
   reduction-derived scalars (`v4`) as compute-phase inputs, sequencing stat → barrier → normalize-into-slab.
3. **MMA/TMA over compute bundles**: let the warp tier consume a compute-written slab (`ldmatrix` with
   `swizzle=NONE` first; swizzled compute-writes later) and lift `050_use_tma`'s blanket SYNC demotion for
   compute bundles where the compute phase is a prologue (not per-K_i).

Ship it as the fuse-vs-split **structural fork** once `plans/structural-forks-in-two-level.md` lands, so the
tuner measures A vs B per shape instead of trusting either policy — at decode-time shapes (M=1) or long-seq
prefill the calculus between materialization traffic and kernel count genuinely flips.

## Non-goals

- Online-rescaled (flash-style) norm fusion — mathematically possible (rescale matmul partials as the running
  stat improves), numerically delicate, and worth at most the M4 ceiling; revisit only if M4's two-pass-over-smem
  shows up in profiles.
- `wgmma` / `setmaxnreg` warpgroup work — different atom family, different hardware tier (sm_90a+); the gated
  pair deploys on the existing `mma.sync` + WS path.
- Unfusing gate/up from each other (4-kernel split) — strictly worse than dual-Mma; only a fallback if M2 stalls.
