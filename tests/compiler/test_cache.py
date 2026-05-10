"""Tests for the autotune tuning cache (``deplodock.compiler.cache``).

Covers the SQLite layer (lookup / has / record / replace) and the helpers
that bridge it to the search policy (``op_cache_key`` for CudaOp,
``count_unmeasured_ops``, ``record_terminal``).
"""

from __future__ import annotations

from deplodock.compiler.cache import (
    TuningCache,
    count_unmeasured_ops,
    op_cache_key,
    record_terminal,
)
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp


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


def test_cache_lookup_miss_returns_none() -> None:
    cache = TuningCache()
    assert cache.lookup("ctx", "op") is None
    assert not cache.has("ctx", "op")


def test_cache_record_then_lookup() -> None:
    cache = TuningCache()
    cache.record("ctx", "op", latency_us=1.0)
    entry = cache.lookup("ctx", "op")
    assert entry is not None
    assert entry.latency_us == 1.0
    assert entry.status == "ok"
    assert cache.has("ctx", "op")


def test_cache_record_keeps_best() -> None:
    """Re-recording an ``ok`` entry only updates if the new latency is
    lower — autotune attribution writes the same ancestor key multiple
    times; we want the best variant ever seen, not the most recent."""
    cache = TuningCache()
    cache.record("ctx", "op", latency_us=2.5)
    cache.record("ctx", "op", latency_us=1.0)
    assert cache.lookup("ctx", "op").latency_us == 1.0  # improved
    cache.record("ctx", "op", latency_us=5.0)
    assert cache.lookup("ctx", "op").latency_us == 1.0  # kept best


def test_cache_record_status_change_overwrites() -> None:
    """A successful measurement supersedes a prior ``bench_fail`` row
    even if its latency would otherwise lose the keep-best check."""
    cache = TuningCache()
    cache.record("ctx", "op", latency_us=0.0, status="bench_fail")
    cache.record("ctx", "op", latency_us=10.0, status="ok")
    entry = cache.lookup("ctx", "op")
    assert entry.status == "ok" and entry.latency_us == 10.0


def test_cache_segregates_by_context() -> None:
    cache = TuningCache()
    cache.record("ctx_a", "op", latency_us=1.0)
    assert cache.has("ctx_a", "op")
    assert not cache.has("ctx_b", "op")


def test_op_cache_key_only_for_cuda_op() -> None:
    g = _make_cuda_graph()
    assert op_cache_key(g.nodes["k"].op) is not None
    assert op_cache_key(g.nodes["x"].op) is None  # InputOp


def test_op_cache_key_distinguishes_kernel_source() -> None:
    a = _make_cuda_graph(kernel_source="// A")
    b = _make_cuda_graph(kernel_source="// B")
    assert op_cache_key(a.nodes["k"].op) != op_cache_key(b.nodes["k"].op)


def test_record_terminal_writes_each_cuda_op() -> None:
    cache = TuningCache()
    g = _make_cuda_graph()
    record_terminal(g, cache, context_key="ctx")
    key = op_cache_key(g.nodes["k"].op)
    assert key is not None
    entry = cache.lookup("ctx", key)
    assert entry is not None and entry.latency_us == 1.0


def test_count_unmeasured_drops_after_record_terminal() -> None:
    cache = TuningCache()
    g = _make_cuda_graph()
    assert count_unmeasured_ops(g, cache, "ctx") == 1
    record_terminal(g, cache, "ctx")
    assert count_unmeasured_ops(g, cache, "ctx") == 0


def test_context_structural_key_is_stable_and_segregates_by_cap() -> None:
    a = Context.from_target((12, 0))
    b = Context.from_target((12, 0))
    c = Context.from_target((9, 0))
    assert a.structural_key() == b.structural_key()
    assert a.structural_key() != c.structural_key()
