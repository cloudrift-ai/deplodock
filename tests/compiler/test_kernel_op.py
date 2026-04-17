"""Tests for the structural LoopOp IR with SSA Assign body."""

import pytest

from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import Assign, Axis, LoopOp, Port
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp

# ---------------------------------------------------------------------------
# Axis / Port
# ---------------------------------------------------------------------------


def test_axis_free():
    a = Axis("a0", 8, "free")
    assert a.name == "a0"
    assert a.extent == 8
    assert a.kind == "free"


def test_axis_reduce():
    a = Axis("a1", 4, "reduce")
    assert a.kind == "reduce"


def test_port_default_is_empty_index():
    assert Port().index == ()


def test_port_with_index():
    p = Port(index=(Var("a0"), Var("a1")))
    assert len(p.index) == 2


def test_port_broadcast_literal():
    p = Port(index=(Literal(0, "int"), Var("a0")))
    assert isinstance(p.index[0], Literal)


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", ElementwiseOp("add"), args=("$0", "$1"))
    assert a.name == "out"
    assert a.op.fn == "add"
    assert a.args == ("$0", "$1")


# ---------------------------------------------------------------------------
# LoopOp construction
# ---------------------------------------------------------------------------


def _pointwise_axes(n: int) -> tuple[Axis, ...]:
    return tuple(Axis(f"a{i}", 4, "free") for i in range(n))


def test_kernel_pointwise():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p, p),
        body=(Assign("z", ElementwiseOp("add"), args=("$0", "$1")),),
        outputs=(Port(index=(Var("a0"),)),),
    )
    assert len(k.body) == 1


def test_kernel_reduce():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        body=(Assign("s", ReduceOp("sum", -1), args=("$0",)),),
        outputs=(Port(index=(Var("a0"),)),),
    )
    assert isinstance(k.body[0].op, ReduceOp)


def test_kernel_matmul():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = LoopOp(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Assign("dot", ReduceOp("sum", -1), args=("mul",)),
        ),
        outputs=(Port(index=(Var("a0"),)),),
    )
    assert len(k.body) == 2


def test_kernel_matmul_bias():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p_mk = Port(index=(Var("a0"), Var("a1")))
    p_bias = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p_mk, p_mk, p_bias),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Assign("dot", ReduceOp("sum", -1), args=("mul",)),
            Assign("out", ElementwiseOp("add"), args=("dot", "$2")),
        ),
        outputs=(Port(index=(Var("a0"),)),),
    )
    assert len(k.body) == 3


def test_kernel_softmax():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        body=(
            Assign("max", ReduceOp("max", -1), args=("$0",)),
            Assign("sub", ElementwiseOp("sub"), args=("$0", "max")),
            Assign("exp", ElementwiseOp("exp"), args=("sub",)),
            Assign("sum", ReduceOp("sum", -1), args=("exp",)),
            Assign("div", ElementwiseOp("div"), args=("exp", "sum")),
        ),
        outputs=(Port(index=(Var("a0"), Var("a1"))),),
    )
    assert len(k.body) == 5


def test_kernel_unary_chain():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        body=(
            Assign("neg", ElementwiseOp("neg"), args=("$0",)),
            Assign("exp", ElementwiseOp("exp"), args=("neg",)),
        ),
        outputs=(Port(index=(Var("a0"),)),),
    )
    assert len(k.body) == 2


# ---------------------------------------------------------------------------
# Axis / SSA validation
# ---------------------------------------------------------------------------


def test_rejects_duplicate_axis_name():
    with pytest.raises(ValueError, match="duplicate axis"):
        LoopOp(axes=(Axis("a0", 4, "free"), Axis("a0", 8, "free")))


def test_ssa_rejects_undefined_arg():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="not defined"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            body=(Assign("y", ElementwiseOp("exp"), args=("z",)),),
            outputs=(p,),
        )


def test_ssa_rejects_duplicate_name():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="already defined"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            body=(
                Assign("y", ElementwiseOp("exp"), args=("$0",)),
                Assign("y", ElementwiseOp("neg"), args=("y",)),
            ),
            outputs=(p,),
        )


def test_ssa_rejects_forward_reference():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="not defined"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            body=(
                Assign("a", ElementwiseOp("add"), args=("$0", "b")),
                Assign("b", ElementwiseOp("exp"), args=("$0",)),
            ),
            outputs=(p,),
        )


def test_ssa_allows_input_name_reuse_in_multiple_args():
    axes = _pointwise_axes(1)
    p = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        body=(
            Assign("a", ElementwiseOp("exp"), args=("$0",)),
            Assign("b", ElementwiseOp("neg"), args=("$0",)),
            Assign("c", ElementwiseOp("add"), args=("a", "b")),
        ),
        outputs=(p,),
    )
    assert len(k.body) == 3
