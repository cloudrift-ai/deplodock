"""Tests for Graph -> list[KernelOp] structural lowering."""

import pytest

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.lower import lower
from deplodock.compiler.ops import (
    Combine,
    ContractionCore,
    ElementwiseOp,
    InputOp,
    ReduceOp,
)


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# ---------------------------------------------------------------------------
# Pointwise: single ElementwiseOp becomes one Combine-rooted KernelOp.
# ---------------------------------------------------------------------------


def test_lower_pointwise_add():
    g = Graph()
    _input(g, "x", (4,))
    _input(g, "y", (4,))
    g.add_node(
        op=ElementwiseOp(fn="add"),
        inputs=["x", "y"],
        output=Tensor("add0", (4,)),
        node_id="add0",
    )
    g.inputs = ["x", "y"]
    g.outputs = ["add0"]

    kernels = lower(g)
    assert len(kernels) == 1
    k = kernels[0]

    # Pointwise: body lives in inputs[0] as a Combine; no core stages.
    assert k.contraction is None
    assert k.reduce_stages == ()
    assert k.epilogue == ()
    assert len(k.inputs) == 1
    assert isinstance(k.inputs[0], Combine)
    assert k.inputs[0].ops[0].op.fn == "add"
    assert [s.buffer_id for s in k.inputs[0].sources] == ["x", "y"]
    assert [o.buffer_id for o in k.outputs] == ["add0"]


# ---------------------------------------------------------------------------
# Reduce: single ReduceOp (non-matmul) becomes one reduce_stages kernel.
# ---------------------------------------------------------------------------


def test_lower_reduce_sum():
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=["x"],
        output=Tensor("r0", (4,)),
        node_id="r0",
    )
    g.inputs = ["x"]
    g.outputs = ["r0"]

    kernels = lower(g)
    assert len(kernels) == 1
    k = kernels[0]
    assert k.contraction is None
    assert len(k.reduce_stages) == 1
    stage = k.reduce_stages[0]
    assert stage.pre_ops == ()
    assert stage.reduce.op.fn == "sum"


# ---------------------------------------------------------------------------
# Matmul: mul + sum (fan-out=1 on mul) collapses into one ContractionCore.
# ---------------------------------------------------------------------------


def test_lower_matmul_mul_sum_pair():
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (4, 8))
    g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=["a", "b"],
        output=Tensor("mul0", (4, 8)),
        node_id="mul0",
    )
    g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=["mul0"],
        output=Tensor("dot0", (4,)),
        node_id="dot0",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["dot0"]

    kernels = lower(g)
    assert len(kernels) == 1
    k = kernels[0]
    assert isinstance(k.contraction, ContractionCore)
    assert k.reduce_stages == ()
    assert k.contraction.reduce.op.fn == "sum"
    assert k.contraction.k_axis == -1
    operand = k.contraction.operand
    assert isinstance(operand, Combine)
    assert operand.ops[0].op.fn == "mul"
    assert [s.buffer_id for s in operand.sources] == ["a", "b"]


# ---------------------------------------------------------------------------
# Fan-out guard: mul with >1 consumer should NOT collapse into matmul.
# ---------------------------------------------------------------------------


def test_lower_no_matmul_when_mul_fans_out():
    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (4, 8))
    g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=["a", "b"],
        output=Tensor("mul0", (4, 8)),
        node_id="mul0",
    )
    g.add_node(
        op=ReduceOp(fn="sum", axis=-1),
        inputs=["mul0"],
        output=Tensor("dot0", (4,)),
        node_id="dot0",
    )
    # Second consumer of mul0 — pointwise copy.
    g.add_node(
        op=ElementwiseOp(fn="neg"),
        inputs=["mul0"],
        output=Tensor("neg0", (4, 8)),
        node_id="neg0",
    )
    g.inputs = ["a", "b"]
    g.outputs = ["dot0", "neg0"]

    kernels = lower(g)
    # Three kernels: mul0 (pointwise), dot0 (pure reduce), neg0 (pointwise).
    assert len(kernels) == 3
    assert kernels[0].contraction is None  # mul is now a pointwise kernel
    assert kernels[0].reduce_stages == ()
    assert kernels[1].contraction is None
    assert len(kernels[1].reduce_stages) == 1


# ---------------------------------------------------------------------------
# Chained primitives: each node becomes its own singleton kernel.
# ---------------------------------------------------------------------------


def test_lower_chain_of_pointwise_all_singletons():
    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n", (4,)), node_id="n")
    g.inputs = ["x"]
    g.outputs = ["n"]

    kernels = lower(g)
    assert len(kernels) == 2
    assert [k.inputs[0].ops[0].op.fn for k in kernels] == ["exp", "neg"]
    # Second kernel consumes the first's output by buffer_id.
    assert kernels[1].inputs[0].sources[0].buffer_id == "e"


# ---------------------------------------------------------------------------
# Unsupported op types should raise until follow-up commits add support.
# ---------------------------------------------------------------------------


def test_lower_rejects_unknown_op_type():
    from deplodock.compiler.ops import MatmulOp

    g = Graph()
    _input(g, "a", (4, 8))
    _input(g, "b", (8, 4))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("m", (4, 4)), node_id="m")
    g.inputs = ["a", "b"]
    g.outputs = ["m"]

    with pytest.raises(NotImplementedError, match="MatmulOp"):
        lower(g)


# ---------------------------------------------------------------------------
# Pipeline entry point mirrors lower() 1:1.
# ---------------------------------------------------------------------------


def test_compile_graph_entry_point():
    from deplodock.compiler.pipeline import compile_graph

    g = Graph()
    _input(g, "x", (4,))
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.inputs = ["x"]
    g.outputs = ["e"]

    kernels = compile_graph(g)
    assert len(kernels) == 1
    assert kernels[0].inputs[0].ops[0].op.fn == "exp"
