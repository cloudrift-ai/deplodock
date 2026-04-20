"""Tests for the compile_graph pipeline: decomposition → optimization → fusion → extract.

After compile_graph, every primitive op is inside a LoopOp. Reductions
live as ``Accum + Accum`` on the LoopOp; elementwise ops as
``Assign``; final output as a ``Write``.

Structural fixtures also have a matching ``*_correctness`` test that runs
the pre-pipeline graph and the post-pipeline graph through ``NumpyBackend``
(now that ``LoopOp.forward`` can execute fused kernels) and compares the
outputs — this validates the full decomposition+optimization+fusion chain
preserves semantics without needing a GPU.
"""

from pathlib import Path

import numpy as np

from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Accum, Assign, Write
from deplodock.compiler.ir.tensor_ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import compile_graph
from deplodock.compiler.rewriter import Rewriter

_RULES_DIR = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
_backend = NumpyBackend()
rng = np.random.default_rng(0)


def _fully_rewrite(graph: Graph) -> Graph:
    """Apply the full rewriter chain (decomposition → optimization → fusion)."""
    return Rewriter.from_directory(_RULES_DIR).apply(graph)


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs).outputs


def _assert_pipeline_preserves_semantics(make_graph, inputs, *, rtol=1e-5, atol=1e-5):
    """Numpy-execute the original graph and the rewritten graph; outputs must match."""
    before = _run(make_graph(), inputs)
    after = _run(_fully_rewrite(make_graph()), inputs)
    bvals, avals = list(before.values()), list(after.values())
    assert len(bvals) == len(avals)
    for i, (b, a) in enumerate(zip(bvals, avals, strict=True)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output[{i}]")


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def _elementwise_fns(body) -> list[str]:
    from deplodock.compiler.ir.loop import flatten_body

    return [s.op.fn for s in flatten_body(body) if isinstance(s, Assign)]


def _has_update(body) -> bool:
    from deplodock.compiler.ir.loop import flatten_body

    return any(isinstance(s, Accum) for s in flatten_body(body))


def _has_write(body) -> bool:
    from deplodock.compiler.ir.loop import flatten_body

    return any(isinstance(s, Write) for s in flatten_body(body))


def test_pointwise_add():
    g = Graph()
    _input(g, "x", (4,))
    _input(g, "y", (4,))
    g.add_node(op=ElementwiseOp(fn="add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 1
    assert "add" in _elementwise_fns(launches[0].loop.body)
    assert _has_write(launches[0].loop.body)


def test_chained_pointwise_fuses_into_one():
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 1
    fns = _elementwise_fns(launches[0].loop.body)
    assert "exp" in fns and "neg" in fns


def test_reduce_sum():
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["x"], output=Tensor("r", (4, 1)), node_id="r")
    g.inputs = ["x"]
    g.outputs = ["r"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 1
    loop = launches[0].loop
    assert _has_update(loop.body)
    # Reduction target is a Accum with combine=add (sum).
    assert any(lb.combine.fn == "add" for lb in loop.accums)


def test_matmul():
    from deplodock.compiler.ir.frontend_ir import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    result = compile_graph(g)
    launches = result.launches
    # Expect a mul Assign and a sum Accum somewhere across the launches.
    has_mul = any("mul" in _elementwise_fns(k.loop.body) for k in launches)
    has_sum = any(any(lb.combine.fn == "add" for lb in k.loop.accums) for k in launches)
    assert has_mul
    assert has_sum


def test_no_matmul_when_mul_fans_out():
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (4, 8))
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8)), node_id="m")
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["m"], output=Tensor("d", (4, 1)), node_id="d")
    g.add_node(op=ElementwiseOp("neg"), inputs=["m"], output=Tensor("n", (4, 8)), node_id="n")
    g.inputs = ["a", "b"]
    g.outputs = ["d", "n"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 3


def test_matmul_op_decomposes_and_fuses():
    from deplodock.compiler.ir.frontend_ir import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("m", (4, 4)), node_id="m")
    g.inputs = ["a", "b"]
    g.outputs = ["m"]

    result = compile_graph(g)
    launches = result.launches
    has_reduce = any(_has_update(k.loop.body) for k in launches)
    assert has_reduce


def test_compile_graph_produces_kernel_ops():
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.inputs = ["x"]
    g.outputs = ["e"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 1


# ===================================================================
# Correctness: full rewriter (decomp + opt + fusion) preserves semantics.
# ===================================================================


def test_pointwise_add_correctness():
    def _make():
        g = Graph()
        _input(g, "x", (4,))
        _input(g, "y", (4,))
        g.add_node(op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
        g.inputs, g.outputs = ["x", "y"], ["z"]
        return g

    x = rng.standard_normal(4).astype(np.float32)
    y = rng.standard_normal(4).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"x": x, "y": y})


def test_chained_pointwise_correctness():
    def _make():
        g = Graph()
        _input(g, "x", (4,))
        g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
        g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
        g.inputs, g.outputs = ["x"], ["n"]
        return g

    x = rng.standard_normal(4).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"x": x})


def test_reduce_sum_correctness():
    def _make():
        g = Graph()
        _input(g, "x", (4, 8))
        g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["x"], output=Tensor("r", (4, 1)), node_id="r")
        g.inputs, g.outputs = ["x"], ["r"]
        return g

    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"x": x})


def test_matmul_correctness():
    from deplodock.compiler.ir.frontend_ir import MatmulOp

    def _make():
        g = Graph()
        _input(g, "a", (4, 8))
        _input(g, "b", (8, 4))
        g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
        g.inputs, g.outputs = ["a", "b"], ["o"]
        return g

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((8, 4)).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"a": a, "b": b}, rtol=1e-4)


def test_mul_fan_out_correctness():
    def _make():
        g = Graph()
        _input(g, "a", (4, 8))
        _input(g, "b", (4, 8))
        g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8)), node_id="m")
        g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["m"], output=Tensor("d", (4, 1)), node_id="d")
        g.add_node(op=ElementwiseOp("neg"), inputs=["m"], output=Tensor("n", (4, 8)), node_id="n")
        g.inputs, g.outputs = ["a", "b"], ["d", "n"]
        return g

    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_pipeline_preserves_semantics(_make, {"a": a, "b": b})
