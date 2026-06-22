"""Hierarchical move composer for the partition stage.

A from-scratch reimplementation of the partition planner as a stack of
algebraically-justified *moves* (each gated by a carrier trait) composed
bottom-up into the existing ``TileOp`` tower, with a generative
``Fork`` tree the two-level MCTS branches on move-by-move. See
``plans/melodic-giggling-gem.md``.

It is the **sole** partitioner — ``010_partition_loops`` dispatches every
``LoopOp`` through ``compose.try_compose``; a kernel it can't lower raises (no
legacy planner, no fallback). Covers pointwise (``MAP``), matmul (``SEMIRING``,
scalar + tensor-core), cooperative reduce (``MONOID``), and fused flash
(``TWISTED_MONOID``); see ``plans/algebra-licensed-decomposition-moves.md``.
"""
