"""Tests for the compile_graph pipeline: decomposition → optimization → fusion → extract.

After compile_graph, every primitive op is inside a KernelOp. Tests verify
the structural shape of the resulting KernelOps.
"""

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    InputOp,
    ReduceOp,
)
from deplodock.compiler.pipeline import compile_graph


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


def test_pointwise_add():
    g = Graph()
    _input(g, "x", (4,))
    _input(g, "y", (4,))
    g.add_node(op=ElementwiseOp(fn="add"), inputs=["x", "y"], output=Tensor("z", (4,)), node_id="z")
    g.inputs = ["x", "y"]
    g.outputs = ["z"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
    k = kernels[0]
    assert k.contraction is None
    assert k.reduce_stages == ()


def test_chained_pointwise_fuses_into_one():
    """exp → neg with fan-out 1 fuses into one kernel."""
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
    assert kernels[0].contraction is None
    assert kernels[0].reduce_stages == ()


# ---------------------------------------------------------------------------
# Reduce
# ---------------------------------------------------------------------------


def test_reduce_sum():
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["x"], output=Tensor("r", (4,)), node_id="r")
    g.inputs = ["x"]
    g.outputs = ["r"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
    assert len(kernels[0].reduce_stages) == 1
    assert kernels[0].reduce_stages[0].reduce.fn == "sum"


# ---------------------------------------------------------------------------
# Matmul (mul + sum pair)
# ---------------------------------------------------------------------------


def test_matmul_mul_sum():
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8, 4)), node_id="m")
    g.add_node(op=ReduceOp(fn="sum", axis=1), inputs=["m"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
    assert isinstance(kernels[0].contraction, ContractionCore)
    assert kernels[0].contraction.reduce.fn == "sum"


def test_no_matmul_when_mul_fans_out():
    """mul with fan-out > 1 can't be paired with sum as a contraction."""
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (4, 8))
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8)), node_id="m")
    g.add_node(op=ReduceOp(fn="sum", axis=-1), inputs=["m"], output=Tensor("d", (4,)), node_id="d")
    g.add_node(op=ElementwiseOp("neg"), inputs=["m"], output=Tensor("n", (4, 8)), node_id="n")
    g.inputs = ["a", "b"]
    g.outputs = ["d", "n"]

    kernels = compile_graph(g)
    # mul has fan-out > 1 → separate kernel. sum and neg are also separate.
    assert len(kernels) == 3
    assert all(k.contraction is None for k in kernels)


# ---------------------------------------------------------------------------
# MatmulOp (high-level) gets decomposed then fused
# ---------------------------------------------------------------------------


def test_matmul_op_decomposes_and_fuses():
    """MatmulOp → decompose to mul+sum → fuse into ContractionCore."""
    from deplodock.compiler.ops import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("m", (4, 4)), node_id="m")
    g.inputs = ["a", "b"]
    g.outputs = ["m"]

    kernels = compile_graph(g)
    contractions = [k for k in kernels if k.contraction is not None]
    assert len(contractions) >= 1


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def test_compile_graph_produces_kernel_ops():
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.inputs = ["x"]
    g.outputs = ["e"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
