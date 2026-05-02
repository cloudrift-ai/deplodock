# LoopIR Refactoring Plan: Eliminate Raw C Strings

## Current State

22 raw code generation sites totaling ~180 lines of generated C/CUDA strings.
Pointwise, single-reduce, multi-reduce, and contraction epilogue+write are
already proper LoopIR. What remains is ranked below by complexity.

## Ranked by Complexity

### Already Done (ranks 1-2)

| # | Pattern | Raw C | Notes |
|---|---------|-------|-------|
| 1 | `fmaxf` accumulate | 1 line | `Accumulate("max")` in codegen |
| 2 | `return;` guards | 1 line | `Guard` + early return |

### Easy: Warp Shuffle Reduction (ranks 3-5)

**~6 lines total across 5 sites. Duplicated in tiled.py and loop_codegen.py.**

| # | Pattern | Raw C | Location |
|---|---------|-------|----------|
| 3 | Shared write/read broadcast | 2 lines | loop_codegen.py:310,314 |
| 4 | Cross-warp conditional load | 2 lines | loop_codegen.py:299 |
| 5 | `__shfl_down_sync` loop | 1 line x2 | loop_codegen.py:283,303 |

**What's needed:**
- New `WarpShuffle` LoopOp for `__shfl_down_sync` with `offset >>= 1` loop
- `Alloc("smem")` + `Load`/`Store` for the shared scalar broadcast
- Eliminates duplicate `_emit_warp_reduce` in both tiled.py and loop_codegen.py

**Estimate:** ~50 new lines of LoopIR/codegen code.

### Medium: CTA-Swizzle + Naive K-Loop + Smem Strategy (ranks 6-10)

**~80 lines of raw C. All standard constructs.**

| # | Pattern | Raw C | Location |
|---|---------|-------|----------|
| 6 | CTA-swizzle grid setup | 14 lines | loop_lower.py:654-668 |
| 7 | Naive K-loop (global loads + FMA) | 16 lines | loop_lower.py:687-697 |
| 8 | Smem K-tile loop | 23 lines | tiled.py:348-370 |
| 9 | Smem grid setup + shared decl | 10 lines | tiled.py:320-329 |
| 10 | Smem write phase | 16 lines | tiled.py:390 |

**What's needed:**
- #6: Series of `Compute` ops for integer arithmetic. No special CUDA constructs.
- #7: `LoopNest("k")` + guarded `Load` (`(row<M) ? A[...] : 0`) + `Accumulate` on `RegAccess`. Highest-value target — last RawLoopOp in naive contraction.
- #8-10: Same patterns as naive but with smem staging. `Alloc("smem")` + `Load` to smem + `Barrier` + inner FMA loop.

**Estimate:** ~150 new lines. Eliminates last raw code from naive contraction path.

### Hard: Softmax Epilogue + TMA (ranks 11-12)

**~140 lines of raw C. Needs new LoopIR ops.**

| # | Pattern | Raw C | Location |
|---|---------|-------|----------|
| 11 | Contraction+softmax epilogue | ~90 lines | tiled.py:157-244 |
| 12 | TMA double-buffer K-loop | 53 lines | tiled.py:773-825 |

**What's needed:**
- #11: Per-row `WarpShuffle` with `__shfl_xor_sync` (horizontal reduce across register tile columns). Different from `__shfl_down_sync` used in row-reduce. Needs new `WarpShuffleXor` LoopOp.
- #12: 5 inline PTX asm blocks for mbarrier init/arrive/wait + cp.async.bulk. Needs entirely new LoopIR ops: `TMALoad`, `MBarrierInit`, `MBarrierArrive`, `MBarrierWait`.

**Estimate:** ~300+ new lines. Eliminates last two legacy-only paths.

## Recommended Order

1. **Warp shuffle** (easy, eliminates duplication between tiled.py and loop_codegen.py)
2. **Naive K-loop** (#7, makes naive contraction fully LoopIR)
3. **CTA-swizzle** (#6, completes naive contraction — zero RawLoopOp)
4. **Smem strategy** (#8-10, ports _lower_smem to LoopIR, can delete legacy)
5. **Softmax epilogue** (#11, removes last multi-reduce fallback)
6. **TMA K-loop** (#12, final boss — makes TMA strategy fully LoopIR)

After steps 1-4, the naive and smem contraction paths are fully LoopIR.
After step 5, the multi-reduce contraction fallback is eliminated.
After step 6, `_lower_naive` and `_lower_smem` in tiled.py can be deleted entirely.
