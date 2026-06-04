# Partition-planner compile performance (whole-model compile ~5 min → seconds)

## Context

`deplodock run/compile <whole HF model>` spent ~80% of its ~5-minute wall time (4m56s measured on
`Qwen/Qwen3-Embedding-0.6B`, RTX 5090 host) in `lowering/tile/010_partition_loops`. Profiled under cProfile
(687 s total, 337 LoopOps):

- planner `rewrite`: 591 s cumulative (86% of the compile)
- `TileOp.lazy_score` called 14,364,385× (298 s) — ~42k scored variants per matmul kernel (the architecture
  doc's "~200 per matmul-class kernel" estimate predated the widened FM/FN candidate set)
- `enumerate_cartesian`: 126 s; eager Fork-tree construction + sorting: ~160 s
- `_materialize` was already properly lazy (3,781 calls, 5.6 s); `-vv` was innocent (19.4 vs 19.7 s on a
  single layer — it just makes the stall visible)

Three compounding causes, three fixes:

- **A.** `lazy_score` redid shape-constant body walks per variant (`_count_loop_input_buffers` 79 s, the
  per-Write inner-stride `free_vars()` inside `_coalescing_inner_extent_from_writes` 76 s)
- **B.** 28 structurally identical transformer layers re-enumerated + re-scored identical kernel shapes
  from scratch
- **C.** `build_fork_tree` eagerly constructed the entire Fork tree (~42k leaf Forks per matmul kernel)
  although greedy descends one path (`GreedySearch.push` drops `children[1:]`; no sibling fallback on
  validation failure — it raises `LoweringError`) and MCTS expands level-by-level on pop

## Phase A — hoist shape-constant scoring inputs

`TileOp.lazy_score` gained optional `n_staged_inputs` / `write_inner_free_vars` kwargs (default `None` →
recompute, preserving the lazy↔eager parity contract). The new `TileOp._write_inner_free_vars(shape)`
helper does the one body walk; `_coalescing_inner_extent_from_writes` accepts the precomputed sets and
reduces per-variant work to a set intersection. `_Plan` carries `score_n_staged` / `score_write_inner_fv`,
computed once in `_plan_kernel` and forwarded by `_score_variant`.

## Phase B — structural enumeration memo

`_ENUM_MEMO` in `010_partition_loops` maps a structural key → `(params tuple, score_cache)` shared across
structurally identical LoopOps. Key components: `Body.structural_key()` (canonical SSA / buffer / axis
renames), carry-forward `loop_op.knobs`, the sorted multiset of `Load.input` dtypes (the fp16 half2 window
gates on operand dtype via `_is_fp16_matmul`), `ctx` hardware fields (compute capability, warp size, smem
cap, max threads/CTA), and `planner_pin_snapshot()` (live `DEPLODOCK_<KNOB>` env
values — a pin flipped mid-process lands on a fresh key). Classification (chain walk, `shape`, `leading`)
still runs per-LoopOp since materialization needs the layer's real buffer names; only the
`enumerate_cartesian` output and the lazily filled `id(param) → score` cache are shared. The memo is only
written after a successful non-empty enumeration, so skipped shapes never poison it.

Invariant: the memo's shared `params` tuple object is what keeps the `id()`-keyed score cache valid —
plans must never copy or rebuild it.

## Phase C — lazy fork-tree construction

`build_fork_tree` keeps its signature; internals changed. `leaf_score` stays eager at builder entry (the
"score called exactly once per param" contract + exact max-propagation; cheap dict lookups after A+B), but
branch Forks now carry `expand` thunks that build the next level on demand, and a branch's score is `max`
over the leaf scores of its param subgroup — provably equal to eager `max(child scores)` propagation
without instantiating the subtree. Greedy descent builds O(path) Forks instead of ~42k; MCTS pays one
level per pop. Engine contracts hold: `_best_fork` reads branch `fork.knobs` without firing expand, leaf
`expand` still materializes one op for `op_cache_key`.

## Tests

`tests/compiler/passes/test_partition_planner_memo.py` — call-count based: enumeration memoized across
identical LoopOps with different SSA names; total `lazy_score` calls == `len(params)` across two plans;
dtype signature separates memo entries; pin flip invalidates; lazy tree creates < `len(params)` Forks on a
single-path walk with branch scores matching `max` over expanded children; `_score_variant` parity with
from-scratch `lazy_score`.
