
## Status addendum (2026-06-04)

Phases A (hoisted score inputs) and B (structural cross-LoopOp memo) were later dropped for simplicity:
measured cost on whole-model Qwen3-0.6B compile is 1m10s → 3m03s (the lazy fork tree and the op-metadata
plan stamp keep the rest of the original 4m56s win). `TileOp.lazy_score` was subsequently narrowed to the
`(knobs, shapes)` contract — the planner feeds it the exact stamped knob dict via `_variant_knobs` — and the
plan now rides the LoopOp as `loop_op.meta["plan"]`. Revisit phases A/B via git history if compile latency
becomes a pain point again.
