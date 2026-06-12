"""Alias of ``loop/fusion/020_dedup_loads`` for the post-split re-fusion window.

``006_merge_split_glue``'s spliced bodies can duplicate Loads the same way loop-tier
fusion does; the dedup rule is idempotent (fires only on actual ``(input, index)``
duplicates), so aliasing it verbatim is safe — inert on already-deduped ops and in the
inner search's single-node slices.
"""

from __future__ import annotations

import importlib

_m = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.020_dedup_loads")

PATTERN = _m.PATTERN
rewrite = _m.rewrite
