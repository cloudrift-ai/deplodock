"""Tests for optimization rules: structural checks and numerical correctness."""

from pathlib import Path

import numpy as np

from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, IndexMapOp, ReduceOp
from deplodock.compiler.rewriter import Pass, Rule

RULES_DIR = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "rules" / "optimization"

rng = np.random.default_rng(42)
_backend = NumpyBackend()


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs).outputs


def _assert_close(before: dict, after: dict, *, rtol=1e-5, atol=1e-6):
    bvals = list(before.values())
    avals = list(after.values())
    assert len(bvals) == len(avals)
    for i, (b, a) in enumerate(zip(bvals, avals, strict=True)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output[{i}]")


def _load(name: str) -> Rule:
    return Rule.from_file(RULES_DIR / name)


def _apply(graph: Graph, rule: Rule) -> Graph:
    return Pass(name="opt", rules=[rule]).apply(graph)


# ===================================================================
# insert_broadcast_indexmap: add explicit IndexMapOp for broadcast reads
# ===================================================================

_RULE = "002_insert_broadcast_indexmap.py"


def _make_broadcast_add_graph():
    """(4, 8) + (8,) → broadcast add."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (8,)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    return g


def test_broadcast_inserts_indexmap():
    g = _make_broadcast_add_graph()
    result = _apply(g, _load(_RULE))
    has_indexmap = any(isinstance(n.op, IndexMapOp) for n in result.nodes.values())
    assert has_indexmap, "Should insert IndexMapOp for broadcast input"


def test_broadcast_preserves_elementwise():
    g = _make_broadcast_add_graph()
    result = _apply(g, _load(_RULE))
    adds = [n for n in result.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.fn == "add"]
    assert len(adds) == 1


def test_broadcast_no_insert_when_shapes_match():
    """Equal shapes → no IndexMapOp inserted."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 8)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    result = _apply(g, _load(_RULE))
    has_indexmap = any(isinstance(n.op, IndexMapOp) for n in result.nodes.values())
    assert not has_indexmap


def test_broadcast_idempotent():
    g = _make_broadcast_add_graph()
    p = Pass(name="opt", rules=[_load(_RULE)])
    once = p.apply(g)
    twice = p.apply(once)
    assert len(twice.nodes) == len(once.nodes)


def test_broadcast_add_correctness():
    g = _make_broadcast_add_graph()
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = rng.standard_normal(8).astype(np.float32)
    inputs = {"x": x, "y": y}
    before = _run(g, inputs)
    after = _run(_apply(g, _load(_RULE)), inputs)
    _assert_close(before, after)


def test_broadcast_mul_scalar_correctness():
    """(4, 8) * (1,) scalar broadcast."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ConstantOp(name="s", value=2.0), [], Tensor("s", (1,)), node_id="s")
    g.add_node(ElementwiseOp("mul"), ["x", "s"], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x"], ["z"]
    x = rng.standard_normal((4, 8)).astype(np.float32)
    before = _run(g, {"x": x})
    after = _run(_apply(g, _load(_RULE)), {"x": x})
    _assert_close(before, after)


def test_broadcast_3d_correctness():
    """(2, 4, 8) + (4, 1) → broadcast on two axes."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 1)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (2, 4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    x = rng.standard_normal((2, 4, 8)).astype(np.float32)
    y = rng.standard_normal((4, 1)).astype(np.float32)
    before = _run(g, {"x": x, "y": y})
    after = _run(_apply(g, _load(_RULE)), {"x": x, "y": y})
    _assert_close(before, after)


def test_broadcast_rmsnorm_chain_correctness():
    """Broadcast appears in RMSNorm: (rows, 1) * (rows, dim) and (dim,) * (rows, dim)."""
    rows, dim = 4, 16
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (rows, dim)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (dim,)), node_id="w")
    g.add_node(ConstantOp(name="eps", value=1e-6), [], Tensor("eps", (1,)), node_id="eps")
    g.add_node(ElementwiseOp("mul"), ["x", "x"], Tensor("sq", (rows, dim)), node_id="sq")
    g.add_node(ReduceOp("sum", -1), ["sq"], Tensor("red", (rows, 1)), node_id="red")
    g.add_node(ElementwiseOp("add"), ["red", "eps"], Tensor("ae", (rows, 1)), node_id="ae")
    g.add_node(ElementwiseOp("rsqrt"), ["ae"], Tensor("rsq", (rows, 1)), node_id="rsq")
    g.add_node(ElementwiseOp("mul"), ["x", "rsq"], Tensor("norm", (rows, dim)), node_id="norm")
    g.add_node(ElementwiseOp("mul"), ["norm", "w"], Tensor("out", (rows, dim)), node_id="out")
    g.inputs, g.outputs = ["x", "w"], ["out"]

    x = rng.standard_normal((rows, dim)).astype(np.float32)
    w = rng.standard_normal(dim).astype(np.float32)
    before = _run(g, {"x": x, "w": w})
    after = _run(_apply(g, _load(_RULE)), {"x": x, "w": w})
    _assert_close(before, after, rtol=1e-4)


# ===================================================================
# compose_indexmap: collapse adjacent single-source IndexMapOps
# ===================================================================


def test_compose_indexmap_collapses_chain():
    """Adjacent IndexMapOps fold into one with a substituted coord_map."""
    from deplodock.compiler.ir.expr import Literal, placeholder
    from deplodock.compiler.ir.tensor import IndexSource

    # Chain: a(4,8) --unsqueeze--> (4,8,1) --broadcast--> (4,8,3)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(
        IndexMapOp(
            out_shape=(4, 8, 1),
            sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1))),),
        ),
        ["a"],
        Tensor("u", (4, 8, 1)),
        node_id="u",
    )
    g.add_node(
        IndexMapOp(
            out_shape=(4, 8, 3),
            sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1), Literal(0, "int"))),),
        ),
        ["u"],
        Tensor("b", (4, 8, 3)),
        node_id="b",
    )
    g.inputs, g.outputs = ["a"], ["b"]

    x = rng.standard_normal((4, 8)).astype(np.float32)
    before = _run(g, {"a": x})
    result = _apply(g, _load("001_compose_indexmap.py"))
    after = _run(result, {"a": x})
    _assert_close(before, after)

    # After composition: one IndexMapOp reading `a` directly (the intermediate is gone).
    im_nodes = [n for n in result.nodes.values() if isinstance(n.op, IndexMapOp)]
    assert len(im_nodes) == 1
    composed = im_nodes[0]
    assert composed.op.out_shape == (4, 8, 3)
    assert composed.inputs == ["a"]


def test_matmul_with_transpose_fuses_to_one_kernel():
    """``A @ B.T`` fuses to a single kernel with the transpose absorbed into Port.index."""
    from deplodock.compiler.ir.frontend import MatmulOp, TransposeOp
    from deplodock.compiler.ir.loop import LoopOp
    from deplodock.compiler.pipeline import compile_graph

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (3, 8)), node_id="b")
    g.add_node(TransposeOp(axes=(1, 0)), ["b"], Tensor("bt", (8, 3)), node_id="bt")
    g.add_node(MatmulOp(), ["a", "bt"], Tensor("c", (4, 3)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]

    lp = compile_graph(g)
    assert len(lp.launches) == 1, f"expected 1 launch, got {len(lp.launches)}: {[l.output_name for l in lp.launches]}"
    launch = lp.launches[0]
    assert set(launch.input_names) == {"a", "b"}, f"transpose should be absorbed; inputs={launch.input_names}"
    assert isinstance(launch.loop, LoopOp)
