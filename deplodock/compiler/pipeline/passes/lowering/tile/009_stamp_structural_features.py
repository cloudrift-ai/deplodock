"""Alias of ``loop/stamp/020_stamp_structural_features`` for the post-split re-fusion window.

Kernels merged by ``006_merge_split_glue`` carry no ``S_*`` features (the splicer builds a
fresh op; 006 stamps only the decision knobs). They must be restamped before
``010_partition_loops`` so ``op_cache_key`` and the prior's feature vector see the merged
body — after ``007_dedup_loads`` has settled it. Aliasing the rewrite keeps ``992`` the
sole producer of structural features; its idempotence (skip when any ``S_`` knob is
present) leaves every other op untouched.
"""

from __future__ import annotations

import importlib

_m = importlib.import_module("deplodock.compiler.pipeline.passes.loop.stamp.020_stamp_structural_features")

PATTERN = _m.PATTERN
rewrite = _m.rewrite
