"""Alias of ``loop/stamp/010_stamp_loop_names`` for the post-split re-fusion window.

Kernels merged by ``006_merge_split_glue`` come out of the splicer with an empty
``name``; ``010_partition_loops`` forwards ``loop_op.name`` to the TileOp, so the
provenance-derived label must be restamped before partition. Idempotent — skips ops
that already carry a name, so split products named by the cut builder are untouched.
"""

from __future__ import annotations

import importlib

_m = importlib.import_module("deplodock.compiler.pipeline.passes.loop.stamp.010_stamp_loop_names")

PATTERN = _m.PATTERN
rewrite = _m.rewrite
