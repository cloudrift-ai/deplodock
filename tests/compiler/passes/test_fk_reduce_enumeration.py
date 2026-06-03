"""Unit tests for the ``FK`` (reduce-axis multiple-accumulator) enumeration.

``FK`` strip-mines the per-thread K serial loop into ``FK`` independent
accumulators for ILP — see ``plans/fk-register-tile-reductions.md``. These
tests pin the pure ``enumerate_cartesian`` generator (no CUDA, no lowering)
and assert the sweep is reduce-only, divisor-clean, register-budget-bounded,
and that the priority keeps ``FK=1`` first so the greedy default is unchanged.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    _MAX_CELLS_PER_THREAD,
    ScalarTileParams,
    enumerate_cartesian,
)

_CTX = Context(compute_capability=(12, 0))


def _reduce(e_k: int, e_m: int = 128):
    return enumerate_cartesian(E_M=e_m, E_N=1, E_K=e_k, ctx=_CTX, priority_mode="reduce")


def test_reduce_sweeps_fk_above_one():
    """A reduce with a chunkable K (per-thread K-chunk count > 1) surfaces
    FK > 1 variants in the enumeration."""
    combos = _reduce(e_k=2048)
    fks = {p.fk for p in combos if isinstance(p, ScalarTileParams)}
    assert fks - {1}, f"reduce enumeration produced no FK>1 variant; saw {sorted(fks)}"


def test_fk_divides_k_o_ext():
    """Every emitted FK divides the per-thread serial-loop extent
    (K_o_ext = E_K // (splitk·br·bk·fk) must be a whole number)."""
    for p in _reduce(e_k=2048):
        assert isinstance(p, ScalarTileParams)
        k_chunks_per_thread = (2048 // p.br) // p.bk  # the extent FK strip-mines
        assert k_chunks_per_thread % p.fk == 0, f"FK={p.fk} does not divide K_o_ext for {p}"


def test_fk_respects_cell_budget():
    """FK · FM · FN stays within the per-thread register-cell cap."""
    for p in _reduce(e_k=2048):
        assert p.fm * p.fn * p.fk <= _MAX_CELLS_PER_THREAD, p


def test_matmul_and_pointwise_force_fk_one():
    """Only the reduce mode sweeps FK; matmul / pointwise keep FK=1."""
    mm = enumerate_cartesian(E_M=128, E_N=128, E_K=128, ctx=_CTX, priority_mode="matmul")
    pw = enumerate_cartesian(E_M=256, E_N=256, E_K=1, ctx=_CTX, priority_mode="pointwise")
    assert all(p.fk == 1 for p in mm), "matmul must not sweep FK"
    assert all(p.fk == 1 for p in pw), "pointwise must not sweep FK"


def test_fk_one_ranks_first_in_priority():
    """For the greedy (non-tuned) default the FK=1 variant must outrank its
    FK>1 siblings sharing the same BR / thread geometry, so ``compile`` stays
    byte-identical to the pre-FK planner (golden criterion)."""
    combos = _reduce(e_k=2048)
    # The enumeration is sorted by priority DESC. Find the first variant for a
    # representative (br, bn, bm, bk, splitk) group and confirm it's FK=1.
    seen_group: set[tuple] = set()
    for p in combos:
        key = (p.br, p.bn, p.bm, p.bk, p.splitk)
        if key in seen_group:
            continue
        seen_group.add(key)
        # First-seen variant per group is the highest-priority one.
        assert p.fk == 1, f"FK={p.fk} outranked FK=1 for group {key}: greedy default would change"
