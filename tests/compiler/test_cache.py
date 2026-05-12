"""Tests for the tree-shaped tuning cache (``deplodock.compiler.pipeline.search.cache``).

The schema is three tables — ``cuda_perf`` for terminal measurements,
``expansions`` for parent→child edges, ``nodes`` for the maintained
``expected_terminals`` / ``seen_terminals`` counters. These tests verify
the online maintenance of the counters under expansion and terminal
recording, plus the helper functions ``op_cache_key`` / ``record_terminal``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.pipeline.search.cache import (
    TuningCache,
    count_unmeasured_ops,
    op_cache_key,
    record_terminal,
)


def _make_cuda_graph(*, kernel_source: str = "__global__ void k() {}") -> Graph:
    g = Graph()
    g.add_node(InputOp(), inputs=[], output=Tensor(name="x", shape=(4,), dtype="float32"), node_id="x")
    g.add_node(
        CudaOp(kernel_source=kernel_source, kernel_name="k", arg_order=("x",), grid=(1, 1, 1), block=(32, 1, 1)),
        inputs=["x"],
        output=Tensor(name="y", shape=(4,), dtype="float32"),
        node_id="k",
    )
    g.inputs = ["x"]
    g.outputs = ["k"]
    return g


# ---------------------------------------------------------------------------
# cuda_perf
# ---------------------------------------------------------------------------


def test_cuda_perf_miss_returns_none() -> None:
    cache = TuningCache()
    assert cache.cuda_perf("ctx", "cuda") is None


def test_record_cuda_perf_then_lookup() -> None:
    cache = TuningCache()
    cache.record_cuda_perf("ctx", "cuda", latency_us=12.0)
    row = cache.cuda_perf("ctx", "cuda")
    assert row is not None and row.latency_us == 12.0 and row.status == "ok"


def test_record_cuda_perf_keeps_best() -> None:
    cache = TuningCache()
    cache.record_cuda_perf("ctx", "cuda", latency_us=2.5)
    cache.record_cuda_perf("ctx", "cuda", latency_us=1.0)
    assert cache.cuda_perf("ctx", "cuda").latency_us == 1.0
    cache.record_cuda_perf("ctx", "cuda", latency_us=5.0)
    assert cache.cuda_perf("ctx", "cuda").latency_us == 1.0


def test_status_change_overrides_keep_best() -> None:
    cache = TuningCache()
    cache.record_cuda_perf("ctx", "cuda", latency_us=0.0, status="bench_fail")
    cache.record_cuda_perf("ctx", "cuda", latency_us=10.0, status="ok")
    row = cache.cuda_perf("ctx", "cuda")
    assert row.status == "ok" and row.latency_us == 10.0


# ---------------------------------------------------------------------------
# Tree expansion / counters
# ---------------------------------------------------------------------------


def test_root_placeholder_has_expected_one() -> None:
    cache = TuningCache()
    cache.ensure_root("ctx", "root")
    node = cache.node("ctx", "root")
    assert node.expected_terminals == 1 and node.seen_terminals == 0


def test_first_expansion_propagates_delta() -> None:
    """First expansion of a parent consumes its placeholder ``1``, so
    the delta to ancestors is ``n_new - 1``."""
    cache = TuningCache()
    cache.expand("ctx", "root", ["a", "b", "c"])
    root = cache.node("ctx", "root")
    assert root.expected_terminals == 3  # placeholder 1 + delta 2
    for ck in ("a", "b", "c"):
        child = cache.node("ctx", ck)
        assert child.expected_terminals == 1
        assert child.parent_key == "root"


def test_second_expansion_is_pure_addition() -> None:
    """Once a parent already has children, every new edge adds +1."""
    cache = TuningCache()
    cache.expand("ctx", "root", ["a"])
    assert cache.node("ctx", "root").expected_terminals == 1
    cache.expand("ctx", "root", ["b", "c"])
    assert cache.node("ctx", "root").expected_terminals == 3  # 1 + 2 new


def test_expansion_is_idempotent() -> None:
    """Re-running the same expansion is a no-op (PRIMARY KEY rejects
    duplicates; ``INSERT OR IGNORE`` makes it silent)."""
    cache = TuningCache()
    cache.expand("ctx", "root", ["a", "b"])
    cache.expand("ctx", "root", ["a", "b"])
    assert cache.node("ctx", "root").expected_terminals == 2


def test_deep_expansion_propagates_to_root() -> None:
    cache = TuningCache()
    cache.expand("ctx", "root", ["a"])  # root: 1, a: 1
    cache.expand("ctx", "a", ["a1", "a2", "a3"])  # a: 3 (+2), root: 3 (+2)
    assert cache.node("ctx", "root").expected_terminals == 3
    assert cache.node("ctx", "a").expected_terminals == 3


def test_terminal_propagates_seen_upward() -> None:
    cache = TuningCache()
    cache.expand("ctx", "root", ["a", "b"])
    cache.record_cuda_perf("ctx", "a", latency_us=10.0)
    assert cache.node("ctx", "a").seen_terminals == 1
    assert cache.node("ctx", "root").seen_terminals == 1
    cache.record_cuda_perf("ctx", "b", latency_us=20.0)
    assert cache.node("ctx", "root").seen_terminals == 2


def test_re_recording_same_terminal_doesnt_double_count() -> None:
    cache = TuningCache()
    cache.expand("ctx", "root", ["a"])
    cache.record_cuda_perf("ctx", "a", latency_us=10.0)
    cache.record_cuda_perf("ctx", "a", latency_us=5.0)  # better — overwrites latency
    assert cache.node("ctx", "a").seen_terminals == 1
    assert cache.node("ctx", "root").seen_terminals == 1


def test_is_fully_explored_tracks_seen_vs_expected() -> None:
    cache = TuningCache()
    cache.expand("ctx", "root", ["a", "b"])
    assert not cache.is_fully_explored("ctx", "root")
    cache.record_cuda_perf("ctx", "a", latency_us=1.0)
    assert not cache.is_fully_explored("ctx", "root")
    cache.record_cuda_perf("ctx", "b", latency_us=2.0)
    assert cache.is_fully_explored("ctx", "root")


def test_expansion_grows_denominator_mid_run() -> None:
    """If a previously-terminal-looking node gets expanded, ``expected``
    grows. The semantics for 'fully explored' tighten accordingly."""
    cache = TuningCache()
    cache.expand("ctx", "root", ["a"])
    cache.record_cuda_perf("ctx", "a", latency_us=1.0)
    assert cache.is_fully_explored("ctx", "root")  # a was a leaf
    # Now suppose a gets expanded into two sub-options later.
    cache.expand("ctx", "a", ["a1", "a2"])
    # root.expected: 1 + delta(+1) = 2 ; root.seen still 1
    assert not cache.is_fully_explored("ctx", "root")


# ---------------------------------------------------------------------------
# op_cache_key / record_terminal
# ---------------------------------------------------------------------------


def test_op_cache_key_only_for_cuda_op() -> None:
    g = _make_cuda_graph()
    assert op_cache_key(g.nodes["k"].op) is not None
    assert op_cache_key(g.nodes["x"].op) is None  # InputOp


def test_op_cache_key_distinguishes_kernel_source() -> None:
    a = _make_cuda_graph(kernel_source="// A")
    b = _make_cuda_graph(kernel_source="// B")
    assert op_cache_key(a.nodes["k"].op) != op_cache_key(b.nodes["k"].op)


def test_record_terminal_writes_cuda_perf_with_stub_backend() -> None:
    cache = TuningCache()
    g = _make_cuda_graph()
    record_terminal(g, cache, context_key="ctx")  # no backend → stub latency 1.0
    key = op_cache_key(g.nodes["k"].op)
    row = cache.cuda_perf("ctx", key)
    assert row is not None and row.latency_us == 1.0


def test_count_unmeasured_drops_after_record_terminal() -> None:
    cache = TuningCache()
    g = _make_cuda_graph()
    assert count_unmeasured_ops(g, cache, "ctx") == 1
    record_terminal(g, cache, "ctx")
    assert count_unmeasured_ops(g, cache, "ctx") == 0


def test_context_structural_key_segregates() -> None:
    a = Context.from_target((12, 0))
    b = Context.from_target((9, 0))
    assert a.structural_key() != b.structural_key()
