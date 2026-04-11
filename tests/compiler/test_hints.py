"""Tests for the hint system."""

from deplodock.compiler.hints import Hints, resolve_hints
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp


def _matmul_graph() -> Graph:
    """Build a minimal matmul graph: C = Reduce{sum}(Elementwise{mul}(A, B))."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)))
    b = g.add_node(InputOp(), [], Tensor("B", (8, 4)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(fn="mul"), [a, b], Tensor("AB", (4, 8, 4)))
    c = g.add_node(ReduceOp(fn="sum", axis=1), [ew], Tensor("C", (4, 4)))
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


# --- MatmulConfig.from_hints ---


def test_matmul_config_from_hints_partial():
    from deplodock.compiler.backend.cuda.lower import MatmulConfig

    config = MatmulConfig.from_hints({"strategy": "tma_db", "block_k": 64})
    assert config.strategy == "tma_db"
    assert config.block_k == 64
    assert config.threads_y == 16  # default


def test_matmul_config_from_hints_empty():
    from deplodock.compiler.backend.cuda.lower import MatmulConfig

    config = MatmulConfig.from_hints({})
    assert config == MatmulConfig()


def test_matmul_config_from_hints_ignores_unknown():
    from deplodock.compiler.backend.cuda.lower import MatmulConfig

    config = MatmulConfig.from_hints({"strategy": "naive", "bogus_key": 999})
    assert config.strategy == "naive"
    assert not hasattr(config, "bogus_key")


# --- Integration: hints flow through plan_graph ---


def test_hints_flow_through_plan():
    from deplodock.compiler.plan import plan_graph

    g = _matmul_graph()
    g.hints.set("cuda.matmul.strategy", "tma_db")

    plan = plan_graph(g)
    # Find the reduce op kernel.
    reduce_ops = [op for op in plan.ops if op.op.startswith("reduce_")]
    assert len(reduce_ops) == 1
    hints = reduce_ops[0].params.get("_hints", {})
    assert hints.get("cuda.matmul.strategy") == "tma_db"


# --- Integration: hints flow through fusion ---


def test_hints_survive_fusion():
    from deplodock.compiler.fusion import auto_fuse

    g = _matmul_graph()
    # Set a hint on the elementwise (mul) node.
    ew_id = [nid for nid, n in g.nodes.items() if isinstance(n.op, ElementwiseOp)][0]
    g.nodes[ew_id].hints.set("cuda.matmul.strategy", "tma_db")

    fused = auto_fuse(g)
    # The fused node should have inherited the hint.
    from deplodock.compiler.ops import FusedRegionOp

    fused_nodes = [n for n in fused.nodes.values() if isinstance(n.op, FusedRegionOp)]
    assert len(fused_nodes) == 1
    assert fused_nodes[0].hints.get("cuda.matmul.strategy") == "tma_db"
