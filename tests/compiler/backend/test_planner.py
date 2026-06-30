"""Unit tests for the liveness-based scratch-buffer memory planner.

Pure CPU — no cupy, no GPU. Validates the two helpers that decide how scratch
buffers share one slab: :func:`compute_live_intervals` (def-use over the topo
launch order, incl. TMA-source reads) and :func:`plan_offsets` (greedy packing
where only overlapping live intervals get distinct memory).
"""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda._planner import compute_live_intervals, plan_offsets


class _Launch:
    """Duck-typed stand-in for ``program._Launch`` (node_id, arg_names, tma)."""

    def __init__(self, node_id: str, arg_names: list[str], tma_src: list[str] | None = None) -> None:
        self.node_id = node_id
        self.arg_names = tuple(arg_names)
        self.tma_descriptors = tuple(_Tma(s) for s in (tma_src or []))


class _Tma:
    def __init__(self, src_buf: str) -> None:
        self.src_buf = src_buf


def _no_overlap_shares_no_bytes(intervals, sizes, offsets):
    """Assert: any two buffers whose byte ranges overlap have disjoint intervals."""
    names = list(offsets)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ao, bo = offsets[a], offsets[b]
            bytes_overlap = ao < bo + sizes[b] and bo < ao + sizes[a]
            fa, fra = intervals[a]
            fb, frb = intervals[b]
            ivl_overlap = fa < frb and fb < fra
            if bytes_overlap:
                assert not ivl_overlap, f"{a} and {b} share bytes but their live intervals overlap"


def test_intervals_basic_chain():
    # in -> a (L0) -> b (L1) -> c=output (L2). a read at 1, b read at 2.
    launches = [_Launch("a", ["in", "a"]), _Launch("b", ["a", "b"]), _Launch("c", ["b", "c"])]
    iv = compute_live_intervals(["a", "b"], launches)
    assert iv == {"a": (0, 2), "b": (1, 3)}


def test_tma_source_counts_as_read():
    # b is loaded by L2 via a TMA descriptor (its name is NOT in arg_names).
    launches = [
        _Launch("a", ["in", "a"]),
        _Launch("b", ["a", "b"]),
        _Launch("c", ["c"], tma_src=["b"]),  # reads b through TMA only
    ]
    iv = compute_live_intervals(["a", "b"], launches)
    assert iv["b"] == (1, 3), "TMA-source read must extend b's live range to launch 2"


def test_no_producer_or_consumer_raises():
    launches = [_Launch("a", ["in", "a"]), _Launch("b", ["a", "b"])]
    with pytest.raises(ValueError, match="no producing launch"):
        compute_live_intervals(["ghost"], launches)
    # 'b' is produced but never read -> dead scratch.
    with pytest.raises(ValueError, match="no consuming launch"):
        compute_live_intervals(["b"], launches)


def test_disjoint_intervals_reuse_offset():
    # a:[0,2) and d:[2,4) do not overlap -> may share offset 0.
    intervals = {"a": (0, 2), "d": (2, 4)}
    sizes = {"a": 512, "d": 512}
    offsets, total = plan_offsets(intervals, sizes, {"a": 1, "d": 1})
    assert offsets["a"] == offsets["d"] == 0
    assert total == 512  # one slot reused, not summed
    _no_overlap_shares_no_bytes(intervals, sizes, offsets)


def test_overlapping_intervals_distinct_memory():
    intervals = {"a": (0, 2), "b": (1, 3)}  # overlap at index 1
    sizes = {"a": 512, "b": 256}
    offsets, total = plan_offsets(intervals, sizes, {"a": 1, "b": 1})
    assert offsets["a"] != offsets["b"]
    assert total == 768
    _no_overlap_shares_no_bytes(intervals, sizes, offsets)


def test_output_never_aliases_own_input():
    # Single launch L1 reads a (input) and writes b (output): a:[0,2), b:[1,?).
    # With the half-open [first_write, last_read+1) convention they overlap at 1,
    # so a and b must never share memory.
    launches = [_Launch("a", ["in", "a"]), _Launch("b", ["a", "b"]), _Launch("c", ["b", "c"])]
    iv = compute_live_intervals(["a", "b"], launches)
    sizes = {"a": 128, "b": 128}
    offsets, _ = plan_offsets(iv, sizes, {"a": 1, "b": 1})
    assert offsets["a"] != offsets["b"]
    _no_overlap_shares_no_bytes(iv, sizes, offsets)


def test_determinism():
    intervals = {"x": (0, 2), "y": (0, 2), "z": (3, 5)}
    sizes = {"x": 100, "y": 100, "z": 100}
    aligns = {"x": 1, "y": 1, "z": 1}
    r1 = plan_offsets(intervals, sizes, aligns)
    r2 = plan_offsets(dict(reversed(list(intervals.items()))), sizes, aligns)
    assert r1 == r2, "layout must be independent of dict insertion order"


def test_total_below_naive_sum_on_chain():
    # A long chain of same-size buffers, each live only across one launch gap,
    # should pack into ~2 slots, far below the naive sum.
    n = 20
    launches = [_Launch(f"b{i}", [f"b{i - 1}" if i else "in", f"b{i}"]) for i in range(n)]
    names = [f"b{i}" for i in range(n - 1)]  # last is an output, exclude
    iv = compute_live_intervals(names, launches)
    sizes = {nm: 1000 for nm in names}
    aligns = {nm: 1 for nm in names}
    offsets, total = plan_offsets(iv, sizes, aligns)
    assert total < sum(sizes.values()) // 5, "adjacent-only liveness should pack into a few slots"
    _no_overlap_shares_no_bytes(iv, sizes, offsets)


def test_alignment_respected():
    intervals = {"a": (0, 2), "b": (1, 3)}
    sizes = {"a": 100, "b": 100}
    aligns = {"a": 256, "b": 256}
    offsets, _ = plan_offsets(intervals, sizes, aligns)
    assert offsets["b"] % 256 == 0
