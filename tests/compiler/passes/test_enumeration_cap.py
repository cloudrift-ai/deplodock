"""Pin the per-thread cell-product cap on ``enumerate_cartesian``.

``_priority_matmul`` / ``_priority_pointwise`` both cap the ranked
cells/thread at 32 (matmul: ``min(fm*fn, 32)``; pointwise:
``-(fm*fn)``), so any tied / preferred variant already lives under 32.
Historically ``_MAX_CELLS_PER_THREAD`` admitted up to 128 — those
larger-than-preferred variants were only reachable after MCTS exhausted
patience on better ones, and they blew past the autotune's 2 s
compile-budget on fused matmul kernels (Qwen3-Embedding's linear+mean
reduce and SDPA-prologue matmul both hit this on real tunes).

The fix aligns the enumeration cap with the priority cap. These tests
codify the invariant so we don't silently re-introduce the compile-bomb
tail. If a future change wants to raise the cap, it has to update both
this test and the matmul / pointwise priority docstrings.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    _MAX_CELLS_PER_THREAD,
    enumerate_cartesian,
)

# Pinned sm_80 context — matches the rest of the planner tests; the cap is
# device-independent so a real GPU isn't required.
_CTX = Context(compute_capability="sm_80")


def _max_cells(extents: tuple[int, int, int], mode: str) -> int:
    """Run the cartesian and return the largest fm·fn seen across variants."""
    e_m, e_n, e_k = extents
    variants = enumerate_cartesian(E_M=e_m, E_N=e_n, E_K=e_k, ctx=_CTX, priority_mode=mode)
    assert variants, f"empty enumeration for mode={mode!r} extents={extents!r}"
    return max(v.fm * v.fn for v in variants)


def test_matmul_enumeration_respects_cells_cap() -> None:
    """A real Qwen3-Embedding-style matmul shape (M=32, K=1024, N=3072) used
    to surface FM·FN=64 / 128 variants that the autotune then pushed past
    the 2 s compile budget. No matmul variant must exceed the cap."""
    assert _max_cells((32, 3072, 1024), "matmul") <= _MAX_CELLS_PER_THREAD


def test_matmul_enumeration_cap_is_thirty_two() -> None:
    """If the cap is ever raised, this test fails loudly — the matmul priority
    docstring (``capped at 32 (NVRTC compile time)``) needs updating at the
    same time. Keeping the literal pinned prevents accidental drift."""
    assert _MAX_CELLS_PER_THREAD == 32


def test_pointwise_enumeration_respects_cells_cap() -> None:
    """Pointwise priority prefers *fewer* cells/thread (better occupancy), so
    the cap mostly never binds — but a tiny output extent can still emit
    variants near the cap. Pin the invariant anyway."""
    assert _max_cells((4, 4096, 1), "pointwise") <= _MAX_CELLS_PER_THREAD


def test_reduce_enumeration_respects_cells_cap() -> None:
    """Reduce uses BR rather than FM·FN as its cell driver, so FM·FN stays
    tiny — verify nothing leaks above the cap regardless."""
    assert _max_cells((32, 1024, 128), "reduce") <= _MAX_CELLS_PER_THREAD
