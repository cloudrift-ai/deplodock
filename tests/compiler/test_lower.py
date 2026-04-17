"""Tests for the compile_graph pipeline: decomposition → optimization → fusion → extract.

After compile_graph, every primitive op is inside a LoopOp. Tests verify
the structural shape of the resulting KernelOps (SSA Assign body).
"""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import compile_graph


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


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
    assert all(isinstance(a.op, ElementwiseOp) for a in launches[0].loop.body)


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
    assert len(launches[0].loop.body) == 2


def test_reduce_sum():
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["x"], output=Tensor("r", (4,)), node_id="r")
    g.inputs = ["x"]
    g.outputs = ["r"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 1
    assert any(isinstance(a.op, ReduceOp) for a in launches[0].loop.body)


def test_matmul():
    from deplodock.compiler.ir.frontend import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    result = compile_graph(g)
    launches = result.launches
    assert any(isinstance(a.op, ElementwiseOp) and a.op.fn == "mul" for k in launches for a in k.loop.body)
    assert any(isinstance(a.op, ReduceOp) and a.op.fn == "sum" for k in launches for a in k.loop.body)


def test_no_matmul_when_mul_fans_out():
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (4, 8))
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8)), node_id="m")
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["m"], output=Tensor("d", (4,)), node_id="d")
    g.add_node(op=ElementwiseOp("neg"), inputs=["m"], output=Tensor("n", (4, 8)), node_id="n")
    g.inputs = ["a", "b"]
    g.outputs = ["d", "n"]

    result = compile_graph(g)
    launches = result.launches
    assert len(launches) == 3


def test_matmul_op_decomposes_and_fuses():
    from deplodock.compiler.ir.frontend import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("m", (4, 4)), node_id="m")
    g.inputs = ["a", "b"]
    g.outputs = ["m"]

    result = compile_graph(g)
    launches = result.launches
    # Should have at least one kernel with a mul+sum pattern
    has_reduce = any(any(isinstance(a.op, ReduceOp) for a in k.loop.body) for k in launches)
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
