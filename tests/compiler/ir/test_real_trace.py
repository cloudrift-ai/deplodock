"""Tests against real ATen traces from TinyLlama (fixture-based, no torch needed).

Fixture coverage only — verifies the tracer produced the expected op
types. Compilation / decomposition tests moved into ``test_lower.py``
and ``test_emit.py`` against synthetic inputs; E2E flows through
``test_pipeline.py``.
"""

import json
from pathlib import Path

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.tensor.ir import ElementwiseOp

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures"


def _load_fixture(name: str) -> Graph:
    with open(FIXTURE_DIR / name) as f:
        return Graph.from_dict(json.load(f))


def _count_ops(graph: Graph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for n in graph.nodes.values():
        name = type(n.op).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def test_tinyllama_fixture_loads():
    g = _load_fixture("tinyllama_layer0.json")
    assert len(g.nodes) > 0
    assert len(g.inputs) > 0
    assert len(g.outputs) > 0


def test_tinyllama_has_expected_ops():
    """The traced graph has the expected op types from torch.export."""
    g = _load_fixture("tinyllama_layer0.json")
    ops = _count_ops(g)

    # torch.export produces these for a Llama block.
    assert ops.get("ConstantOp", 0) >= 9  # weight matrices + layernorm weights + scalar constants
    assert ops.get("InputOp", 0) == 3  # hidden_states + cos + sin
    assert ops.get("ElementwiseOp", 0) > 0
    # Faithful tracer keeps aten.mean.dim as MeanOp (lowering handles decomposition).
    assert ops.get("MeanOp", 0) > 0


def test_tinyllama_has_7_linear_patterns():
    """The graph should have 7 linear projections (Q, K, V, O, gate, up, down)."""
    from deplodock.compiler.ir.frontend.ir import LinearOp

    g = _load_fixture("tinyllama_layer0.json")
    linear_count = sum(1 for n in g.nodes.values() if isinstance(n.op, LinearOp))
    assert linear_count == 7, f"Expected 7 linear projections, found {linear_count}"


def test_tinyllama_has_sdpa():
    """torch.export keeps scaled_dot_product_attention as a single op."""
    from deplodock.compiler.ir.frontend.ir import SdpaOp

    g = _load_fixture("tinyllama_layer0.json")
    sdpa_count = sum(1 for n in g.nodes.values() if isinstance(n.op, SdpaOp))
    assert sdpa_count == 1


def test_tinyllama_has_silu():
    """torch.export keeps silu as a single op (not decomposed)."""
    g = _load_fixture("tinyllama_layer0.json")
    silu_count = sum(1 for n in g.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.name == "silu")
    assert silu_count == 1


def test_tinyllama_roundtrip_serialization():
    """Load fixture → serialize → deserialize → same ops."""
    g = _load_fixture("tinyllama_layer0.json")
    data = g.to_dict()
    reloaded = Graph.from_dict(data)

    assert len(reloaded.nodes) == len(g.nodes)
    assert _count_ops(reloaded) == _count_ops(g)
    assert reloaded.inputs == g.inputs
    assert reloaded.outputs == g.outputs
