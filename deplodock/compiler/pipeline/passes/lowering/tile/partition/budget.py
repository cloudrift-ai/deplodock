"""Incremental resource accounting threaded through the move ``offers``, so a
composition that would blow a hardware ceiling is never offered (vs the legacy
enumerate-then-filter). Phase 1 tracks per-CTA thread count and per-thread
register cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import MAX_CELLS_PER_THREAD, MAX_THREADS_PER_CTA


@dataclass(frozen=True)
class Budget:
    max_threads: int = MAX_THREADS_PER_CTA
    max_cells: int = MAX_CELLS_PER_THREAD

    def threads_ok(self, threads: int) -> bool:
        return 1 <= threads <= self.max_threads

    def cells_ok(self, cells: int) -> bool:
        return 1 <= cells <= self.max_cells
