"""Hierarchical move composer for the partition stage.

A from-scratch reimplementation of the partition planner as a stack of
algebraically-justified *moves* (each gated by a carrier trait) composed
bottom-up into the existing ``TileOp`` tower, with a generative
``Fork`` tree the two-level MCTS branches on move-by-move. See
``plans/melodic-giggling-gem.md``.

Brought up regime by regime behind ``config.move_composer_enabled()``
(``DEPLODOCK_MOVE_COMPOSER``); ``010_partition_loops`` dispatches the
regimes the composer covers and falls through to the legacy planner for
the rest. Phase 1 covers the pointwise (``MAP``) regime only.
"""
