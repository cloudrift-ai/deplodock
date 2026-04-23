"""Tests for the Graphviz DOT dump emitter in ``compiler.dump``.

Validates that ``_graph_to_dot`` produces well-formed, deterministic DOT
output with the expected node styling conventions. Rendering to SVG/PNG is
left to the user (via the ``dot`` binary) — these tests only check the DOT
source itself.
"""

from __future__ import annotations

from deplodock.compiler.dump import _graph_to_dot
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.expr import placeholder
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource, ReduceOp


def _small_graph() -> Graph:
    """Build a 5-node DAG: (X input) + (W const) → mul → sum → output."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4, 8)), node_id="x")
    w = g.add_node(op=ConstantOp(name="W"), inputs=[], output=Tensor("W", (4, 8)), node_id="w")
    g.inputs = [x]
    m = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[x, w],
        output=Tensor("m", (4, 8)),
        node_id="m",
    )
    r = g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=[m],
        output=Tensor("r", (4, 1)),
        node_id="r",
    )
    g.outputs = [r]
    return g


def test_dot_is_well_formed():
    dot = _graph_to_dot(_small_graph())
    assert dot.startswith("digraph G {")
    assert dot.rstrip().endswith("}")
    # Balanced braces — no stray subgraph emissions.
    assert dot.count("{") == dot.count("}") == 1


def test_dot_has_one_node_line_per_graph_node():
    g = _small_graph()
    dot = _graph_to_dot(g)
    for nid in g.nodes:
        # Each node appears as a node declaration: "nid" [label=...]
        assert f'"{nid}" [label=' in dot, f"missing node line for {nid!r}"


def test_dot_has_one_edge_per_input_pair():
    g = _small_graph()
    dot = _graph_to_dot(g)
    expected_edges = [(src, dst) for dst in g.nodes for src in g.nodes[dst].inputs]
    for src, dst in expected_edges:
        assert f'"{src}" -> "{dst}"' in dot


def test_input_nodes_are_ellipse_with_green_fill():
    dot = _graph_to_dot(_small_graph())
    # x is an InputOp
    assert '"x" [label=' in dot
    line = next(line for line in dot.splitlines() if line.lstrip().startswith('"x" '))
    assert "shape=ellipse" in line
    assert "#d0f0c0" in line


def test_constant_nodes_are_ellipse_with_gray_fill():
    dot = _graph_to_dot(_small_graph())
    line = next(line for line in dot.splitlines() if line.lstrip().startswith('"w" '))
    assert "shape=ellipse" in line
    assert "#e0e0e0" in line


def test_output_node_gets_output_fill():
    dot = _graph_to_dot(_small_graph())
    line = next(line for line in dot.splitlines() if line.lstrip().startswith('"r" '))
    assert "#c0d8f0" in line


def test_indexmap_node_renders_as_parallelogram():
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4, 8)), node_id="x")
    g.inputs = [x]
    # Transpose (i, j) → (j, i)
    idx = g.add_node(
        op=IndexMapOp(
            out_shape=(8, 4),
            sources=(IndexSource(input_idx=0, coord_map=(placeholder(1), placeholder(0))),),
        ),
        inputs=[x],
        output=Tensor("xt", (8, 4)),
        node_id="xt",
    )
    g.outputs = [idx]
    dot = _graph_to_dot(g)
    line = next(line for line in dot.splitlines() if line.lstrip().startswith('"xt" '))
    assert "shape=parallelogram" in line


def test_dot_output_is_deterministic():
    g = _small_graph()
    assert _graph_to_dot(g) == _graph_to_dot(g)


def test_label_contains_op_name_and_shape():
    dot = _graph_to_dot(_small_graph())
    # ElementwiseOp(mul) should label the op line as "mul"
    mul_line = next(line for line in dot.splitlines() if line.lstrip().startswith('"m" '))
    assert "mul" in mul_line
    assert "(4,8)" in mul_line
    # ReduceOp(sum) — label shows "sum"
    sum_line = next(line for line in dot.splitlines() if line.lstrip().startswith('"r" '))
    assert "sum" in sum_line
