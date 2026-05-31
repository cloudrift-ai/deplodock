"""Stream-K MAC-unit work distribution (Phase B2).

Pins the reference walk in ``kernel/_streamk.py``: every ``(tile, k_chunk)`` MAC
unit is computed exactly once across all CTAs, each output tile's K range is
fully and disjointly covered, full-tile vs boundary-partial classification is
correct, and the boundary count is bounded. These properties are what make the
adaptive split correct — the kernel codegen (B3) implements this exact walk.
"""

from __future__ import annotations

import importlib

import pytest

sk = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel._streamk")


# (m_blocks, n_blocks, k_blocks, num_sms) — under-occupied, multi-wave, ragged,
# single-tile, K=1 (no split possible), and a prime num_sms.
_SHAPES = [
    (13, 13, 4, 170),  # 169 tiles × 4 = 676 units / 170 SMs
    (2, 2, 4, 3),  # tiny, forces mid-tile boundaries
    (4, 4, 1, 170),  # K_blocks=1 → no tile is ever split
    (1, 1, 16, 170),  # one tile, 16 K-chunks spread across many CTAs
    (10, 16, 8, 170),  # multi-wave
    (7, 5, 3, 13),  # ragged, prime SM count
]


def _all_segments(m, n, k, sms):
    return [(c, s) for c in range(sms) for s in sk.cta_segments(c, m, n, k, sms)]


@pytest.mark.parametrize("m,n,k,sms", _SHAPES)
def test_every_mac_unit_computed_exactly_once(m, n, k, sms):
    """The union of all CTA segments covers every (m_b, n_b, k_chunk) cell once
    and only once — no gaps (wrong result), no overlaps (double-counted)."""
    seen: dict[tuple[int, int, int], int] = {}
    for _, seg in _all_segments(m, n, k, sms):
        for kc in range(seg.k_lo, seg.k_hi):
            seen[(seg.m_b, seg.n_b, kc)] = seen.get((seg.m_b, seg.n_b, kc), 0) + 1
    expected = {(mb, nb, kc) for mb in range(m) for nb in range(n) for kc in range(k)}
    assert set(seen) == expected, "coverage gap or stray cell"
    assert all(v == 1 for v in seen.values()), "a MAC unit was computed by >1 CTA"


@pytest.mark.parametrize("m,n,k,sms", _SHAPES)
def test_each_tile_k_range_partitioned(m, n, k, sms):
    """For every output tile, the K-sub-ranges that touch it (across all CTAs)
    partition [0, k_blocks) contiguously — so summing the partials reconstructs
    the full K reduction."""
    per_tile: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for _, seg in _all_segments(m, n, k, sms):
        per_tile.setdefault((seg.m_b, seg.n_b), []).append((seg.k_lo, seg.k_hi))
    for tile, ranges in per_tile.items():
        ranges.sort()
        covered = []
        for lo, hi in ranges:
            covered.extend(range(lo, hi))
        assert covered == list(range(k)), f"tile {tile} K-range not a clean partition: {ranges}"


@pytest.mark.parametrize("m,n,k,sms", _SHAPES)
def test_is_full_matches_whole_tile_coverage(m, n, k, sms):
    """A segment is flagged full iff it alone covers the tile's entire K range —
    the signal the body uses to pick direct-write vs scratch."""
    for _, seg in _all_segments(m, n, k, sms):
        assert seg.is_full == (seg.k_lo == 0 and seg.k_hi == k)


@pytest.mark.parametrize("m,n,k,sms", _SHAPES)
def test_boundary_partials_bounded(m, n, k, sms):
    """Each CTA contributes at most one head + one tail boundary partial, so the
    combine workspace is bounded by 2·num_sms (independent of problem size)."""
    for c in range(sms):
        partials = [s for s in sk.cta_segments(c, m, n, k, sms) if not s.is_full]
        assert len(partials) <= 2


def test_k_blocks_one_never_splits():
    """K_blocks=1 means a tile is a single MAC unit — never split, so every
    segment is full and there are zero boundary partials (degenerates to the
    perf-neutral tile-parallel walk)."""
    segs = [s for c in range(170) for s in sk.cta_segments(c, 4, 4, 1, 170)]
    assert len(segs) == 16  # 4×4 tiles, one segment each
    assert all(s.is_full for s in segs)


def test_cta_range_partitions_unit_space():
    """The per-CTA ranges tile [0, total) with no overlap; trailing CTAs idle."""
    total = sk.total_mac_units(13, 13, 4)  # 676
    covered = []
    for c in range(170):
        lo, hi = sk.cta_range(c, total, 170)
        covered.extend(range(lo, hi))
    assert covered == list(range(total))
    # 676 / 170 → per_cta=4; 169 CTAs do 4, last does 0 (676 = 169·4).
    assert sk.per_cta(total, 170) == 4
