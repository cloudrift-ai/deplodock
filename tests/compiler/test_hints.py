"""Tests for the hint system."""

from deplodock.compiler.graph import Graph, Hints, Tensor, resolve_hints
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp


def _matmul_graph() -> Graph:
    """Build a minimal matmul graph: C = Reduce{sum}(Elementwise{mul}(A, B)).

    Uses matching-shape elementwise and keepdim reduction to stay on the
    rank-preserving Tensor IR invariant.
    """
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8, 4)))
    b = g.add_node(InputOp(), [], Tensor("B", (4, 8, 4)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(op="multiply"), [a, b], Tensor("AB", (4, 8, 4)))
    c = g.add_node(ReduceOp(op="sum", axis=1), [ew], Tensor("C", (4, 1, 4)))
    g.outputs = [c]
    return g


# --- Hints class ---


def test_hints_get_set():
    h = Hints()
    assert h.get("foo") is None
    assert h.get("foo", 42) == 42
    h.set("foo", 99)
    assert h.get("foo") == 99


def test_hints_has_remove():
    h = Hints()
    h.set("a", 1)
    assert h.has("a")
    h.remove("a")
    assert not h.has("a")
    h.remove("nonexistent")  # no error


def test_hints_merge():
    a = Hints()
    a.set("x", 1)
    a.set("y", 2)
    b = Hints()
    b.set("y", 99)
    b.set("z", 3)
    a.merge(b)
    assert a.get("x") == 1
    assert a.get("y") == 99  # b wins
    assert a.get("z") == 3


def test_hints_prefix():
    h = Hints()
    h.set("cuda.matmul.strategy", "naive")
    h.set("cuda.matmul.block_k", 32)
    h.set("cuda.other.flag", True)
    sub = h.prefix("cuda.matmul")
    assert sub == {"strategy": "naive", "block_k": 32}


def test_hints_bool():
    assert not Hints()
    h = Hints()
    h.set("a", 1)
    assert h


def test_hints_serialization():
    h = Hints()
    h.set("cuda.matmul.strategy", "tma_db")
    h.set("cuda.matmul.block_k", 64)
    d = h.to_dict()
    h2 = Hints.from_dict(d)
    assert h2.get("cuda.matmul.strategy") == "tma_db"
    assert h2.get("cuda.matmul.block_k") == 64


# --- resolve_hints ---


def test_resolve_hints_node_overrides_graph():
    g = _matmul_graph()
    g.hints.set("cuda.matmul.strategy", "naive")
    reduce_id = g.outputs[0]
    g.nodes[reduce_id].hints.set("cuda.matmul.strategy", "tma_db")

    resolved = resolve_hints(g, reduce_id)
    assert resolved.get("cuda.matmul.strategy") == "tma_db"  # node wins


def test_resolve_hints_graph_only():
    g = _matmul_graph()
    g.hints.set("cuda.matmul.block_k", 32)
    reduce_id = g.outputs[0]

    resolved = resolve_hints(g, reduce_id)
    assert resolved.get("cuda.matmul.block_k") == 32


def test_resolve_hints_empty():
    g = _matmul_graph()
    reduce_id = g.outputs[0]
    resolved = resolve_hints(g, reduce_id)
    assert not resolved


# --- Graph serialization round-trip with hints ---


def test_graph_to_from_dict_with_hints():
    g = _matmul_graph()
    g.hints.set("cuda.matmul.strategy", "naive")
    reduce_id = g.outputs[0]
    g.nodes[reduce_id].hints.set("cuda.matmul.block_k", 64)

    d = g.to_dict()
    assert d["hints"] == {"cuda.matmul.strategy": "naive"}
    assert d["nodes"][reduce_id]["hints"] == {"cuda.matmul.block_k": 64}

    g2 = Graph.from_dict(d)
    assert g2.hints.get("cuda.matmul.strategy") == "naive"
    assert g2.nodes[reduce_id].hints.get("cuda.matmul.block_k") == 64


def test_graph_to_from_dict_no_hints():
    """Old graphs without hints deserialize correctly."""
    g = _matmul_graph()
    d = g.to_dict()
    # Simulate old format: no hints keys
    assert "hints" not in d
    for ndata in d["nodes"].values():
        assert "hints" not in ndata

    g2 = Graph.from_dict(d)
    assert not g2.hints
    for node in g2.nodes.values():
        assert not node.hints


def test_graph_copy_preserves_hints():
    g = _matmul_graph()
    g.hints.set("x", 1)
    nid = g.outputs[0]
    g.nodes[nid].hints.set("y", 2)

    g2 = g.copy()
    assert g2.hints.get("x") == 1
    assert g2.nodes[nid].hints.get("y") == 2

    # Ensure deep copy — mutating copy doesn't affect original.
    g2.hints.set("x", 99)
    assert g.hints.get("x") == 1


# --- Integration: hints on the lowered KernelOps ---


def test_hints_flow_through_lower():
    """Node hints from consumed graph nodes are promoted to the LoopOp's graph node.

    After fusion, the matmul pair (mul + sum) becomes one LoopOp graph
    node whose ``.hints`` carries the merged hints from all consumed nodes.
    """
    from deplodock.compiler.pipeline.engine import run_pass

    g = _matmul_graph()
    ew_id = next(nid for nid, n in g.nodes.items() if isinstance(n.op, ElementwiseOp))
    g.nodes[ew_id].hints.set("cuda.matmul.strategy", "tma_db")

    from pathlib import Path

    from deplodock.compiler.pipeline import LOOP_PASSES

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "pipeline" / "passes"
    fused = g
    for name in LOOP_PASSES:
        fused = run_pass(fused, rules_dir / name)

    from deplodock.compiler.ir.loop import LoopOp

    kernel_nodes = [n for n in fused.nodes.values() if isinstance(n.op, LoopOp)]
    assert len(kernel_nodes) == 1
    assert kernel_nodes[0].hints.get("cuda.matmul.strategy") == "tma_db"
