"""Tests for ``emmy.compiler.pipeline.dump``.

Two surfaces in one file: the Graphviz DOT emitter (``_graph_to_dot``,
checked at the source-string level — rendering to SVG/PNG is left to the
``dot`` binary), and the per-kernel Torch reproducer dump
(``CompilerDump`` writes ``<kname>.torch.json`` + ``.torch.txt`` slices
of the pristine pre-decomposition graph keyed by each kernel's prov
origins).
"""

from __future__ import annotations

import json

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.expr import placeholder
from emmy.compiler.ir.frontend.ir import RmsNormOp
from emmy.compiler.ir.tensor.ir import ElementwiseOp, IndexMapOp, IndexSource, ReduceOp
from emmy.compiler.pipeline import TILE_PASSES, Pipeline
from emmy.compiler.pipeline.dump import CompilerDump, _graph_to_dot

# ---------- _graph_to_dot ----------


def _small_graph() -> Graph:
    """Build a 5-node DAG: (X input) + (W const) → mul → sum → output."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4, 8)), node_id="x")
    w = g.add_node(op=ConstantOp(name="W"), inputs=[], output=Tensor("W", (4, 8)), node_id="w")
    g.inputs = [x]
    m = g.add_node(
        op=ElementwiseOp(op="multiply"),
        inputs=[x, w],
        output=Tensor("m", (4, 8)),
        node_id="m",
    )
    r = g.add_node(
        op=ReduceOp(op="sum", axis=-1),
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
    # ElementwiseOp(mul) should label the op line as "multiply"
    mul_line = next(line for line in dot.splitlines() if line.lstrip().startswith('"m" '))
    assert "multiply" in mul_line
    assert "(4,8)" in mul_line
    # ReduceOp(sum) — label shows "sum"
    sum_line = next(line for line in dot.splitlines() if line.lstrip().startswith('"r" '))
    assert "sum" in sum_line


# ---------- CompilerDump: per-kernel Torch reproducer ----------


def _rms_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("rms_norm_0", (1, 4, 8)), node_id="rms_norm_0")
    g.inputs, g.outputs = ["x", "w"], ["rms_norm_0"]
    return g


def test_torch_repro_is_whole_op_and_loadable(tmp_path):
    g = _rms_graph()
    dump = CompilerDump(dir=tmp_path)
    dump.dump_input_graph(g)
    Pipeline.build(TILE_PASSES).run(g, dump=dump)

    repros = sorted(tmp_path.glob("*.kernels/*.torch.json"))
    assert repros, "expected a per-kernel .torch.json reproducer"

    # Sliced from the pristine graph → contains the whole original RmsNormOp,
    # not its decomposed primitives.
    sub = Graph.from_dict(json.loads(repros[-1].read_text()))
    kinds = {type(n.op).__name__ for n in sub.nodes.values()}
    assert "RmsNormOp" in kinds
    assert sub.outputs == ["rms_norm_0"]
    assert set(sub.inputs) == {"x", "w"}  # the rms_norm's own inputs become boundaries


def test_torch_repro_coverage_header(tmp_path):
    g = _rms_graph()
    dump = CompilerDump(dir=tmp_path)
    dump.dump_input_graph(g)
    Pipeline.build(TILE_PASSES).run(g, dump=dump)

    txts = sorted(tmp_path.glob("*.kernels/*.torch.txt"))
    assert txts
    body = txts[-1].read_text()
    assert "rms_norm_0 (RmsNormOp):" in body
    # rms_norm fully fuses at this shape → full coverage.
    assert "— full" in body


def test_no_repro_without_input_graph(tmp_path):
    """A dump that never captured the input graph writes no reproducers."""
    g = _rms_graph()
    dump = CompilerDump(dir=tmp_path)  # no dump_input_graph call
    Pipeline.build(TILE_PASSES).run(g, dump=dump)
    assert not list(tmp_path.glob("*.kernels/*.torch.json"))


def test_repro_keeps_constant_derived_boundaries(tmp_path):
    """A non-origin feed that is a pure function of constants (e.g. the
    broadcast of a pow-exponent scalar) keeps its constant chain in the slice;
    demoting it to an ``InputOp`` would feed it random bench data — out of
    domain (``pow(x, random)`` → NaN). A feed computed from a real input still
    becomes a synthetic boundary."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    # Constant-derived feed: scalar exponent → broadcast.
    g.add_node(ConstantOp(name="exp_c", value=2.0), [], Tensor("exp_c", (1,), value=2.0, constant=True), node_id="exp_c")
    g.add_node(
        IndexMapOp(out_shape=(4, 8), sources=(IndexSource(input_idx=0, coord_map=(placeholder(0),)),)),
        ["exp_c"],
        Tensor("exp_bc", (4, 8)),
        node_id="exp_bc",
    )
    # Input-derived feed: must stay a synthetic boundary.
    g.add_node(ElementwiseOp(op="silu"), ["x"], Tensor("act", (4, 8)), node_id="act")
    g.add_node(ElementwiseOp(op="pow"), ["act", "exp_bc"], Tensor("p", (4, 8)), node_id="p")
    g.inputs, g.outputs = ["x"], ["p"]

    dump = CompilerDump(dir=tmp_path)
    dump.dump_input_graph(g)
    sub = dump._torch_repro_subgraph({"p"})  # slice to the pow op only

    kinds = {nid: type(n.op).__name__ for nid, n in sub.nodes.items()}
    assert kinds["exp_c"] == "ConstantOp", "constant leaf must be preserved"
    assert kinds["exp_bc"] == "IndexMapOp", "constant-derived broadcast must be kept, not demoted"
    assert kinds["act"] == "InputOp", "input-derived feed must become a synthetic boundary"
    assert sub.inputs == ["act"]
    assert sub.outputs == ["p"]
