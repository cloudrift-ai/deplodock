"""Tests for the structural LoopOp IR with SSA body (Assign/Update/Write/Select)."""

import pytest

from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import (
    Assign,
    Axis,
    LocalBuffer,
    LoopOp,
    Port,
    Select,
    SelectBranch,
    Update,
    Write,
)
from deplodock.compiler.ir.tensor import ElementwiseOp

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
        body=(
            Assign("z", ElementwiseOp("add"), args=("$0", "$1")),
            Write(output=0, index=(Var("a0"),), value="z"),
        ),
    )
    assert len(k.body) == 2


def test_kernel_reduce():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        locals=(LocalBuffer(name="s", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Update(target="s", value="$0"),
            Write(output=0, index=(Var("a0"),), value="s"),
        ),
    )
    assert any(isinstance(lb.combine, ElementwiseOp) for lb in k.locals)


def test_kernel_matmul():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = LoopOp(
        axes=axes,
        inputs=(p, p),
        locals=(LocalBuffer(name="dot", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Update(target="dot", value="mul"),
            Write(output=0, index=(Var("a0"),), value="dot"),
        ),
    )
    assert len(k.body) == 3


def test_kernel_matmul_bias():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p_mk = Port(index=(Var("a0"), Var("a1")))
    p_bias = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p_mk, p_mk, p_bias),
        locals=(LocalBuffer(name="dot", combine=ElementwiseOp("add"), init=Literal(0.0)),),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Update(target="dot", value="mul"),
            Assign("out", ElementwiseOp("add"), args=("dot", "$2")),
            Write(output=0, index=(Var("a0"),), value="out"),
        ),
    )
    assert len(k.body) == 4


def test_kernel_softmax():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        locals=(
            LocalBuffer(name="mx", combine=ElementwiseOp("max"), init=Literal(-1e30)),
            LocalBuffer(name="sm", combine=ElementwiseOp("add"), init=Literal(0.0)),
        ),
        body=(
            Update(target="mx", value="$0"),
            Assign("sub", ElementwiseOp("sub"), args=("$0", "mx")),
            Assign("ex", ElementwiseOp("exp"), args=("sub",)),
            Update(target="sm", value="ex"),
            Assign("div", ElementwiseOp("div"), args=("ex", "sm")),
            Write(output=0, index=(Var("a0"), Var("a1")), value="div"),
        ),
    )
    assert any(lb.name == "mx" for lb in k.locals)
    assert any(lb.name == "sm" for lb in k.locals)


def test_kernel_unary_chain():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p,),
        body=(
            Assign("neg", ElementwiseOp("neg"), args=("$0",)),
            Assign("exp", ElementwiseOp("exp"), args=("neg",)),
            Write(output=0, index=(Var("a0"),), value="exp"),
        ),
    )
    assert len(k.body) == 3


def test_kernel_scatter_output_via_select():
    """Select replaces Mux for coord-predicated dispatch on output."""
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    k = LoopOp(
        axes=axes,
        inputs=(p, p),
        body=(
            Assign("z", ElementwiseOp("add"), args=("$0", "$1")),
            Select(
                name="v",
                branches=(
                    SelectBranch(value="$0", select=Var("c1")),
                    SelectBranch(value="$1", select=Var("c2")),
                ),
            ),
            Write(output=0, index=(Var("a0"),), value="v"),
        ),
    )
    assert isinstance(k.body[1], Select)


# ---------------------------------------------------------------------------
# Validator — SSA / accumulator / v1 pins
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
    )
    assert len(k.body) == 3


def test_rejects_update_without_matching_local():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="does not name a LocalBuffer"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_accumulator_without_reduce_axis():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="no reduce axis"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0)),),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_reduce_axis_without_accumulator():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="no accumulator"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            body=(Assign("a", ElementwiseOp("exp"), args=("$0",)),),
        )


def test_rejects_shaped_localbuffer_in_v1():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="shape=\\(\\)"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0), shape=(4,)),),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_nonthread_scope_in_v1():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="scope='thread'"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            locals=(LocalBuffer(name="acc", combine=ElementwiseOp("add"), init=Literal(0.0), scope="block"),),
            body=(Update(target="acc", value="$0"),),
        )


def test_rejects_nondense_output_indices():
    axes = (Axis("a0", 4, "free"),)
    p = Port(index=(Var("a0"),))
    with pytest.raises(ValueError, match="dense"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            body=(
                Assign("z", ElementwiseOp("neg"), args=("$0",)),
                Write(output=2, index=(Var("a0"),), value="z"),
            ),
        )


def test_rejects_unused_accumulator():
    axes = (Axis("a0", 4, "free"), Axis("a1", 8, "reduce"))
    p = Port(index=(Var("a0"), Var("a1")))
    with pytest.raises(ValueError, match="never Updated"):
        LoopOp(
            axes=axes,
            inputs=(p,),
            locals=(
                LocalBuffer(name="a", combine=ElementwiseOp("add"), init=Literal(0.0)),
                LocalBuffer(name="b", combine=ElementwiseOp("add"), init=Literal(0.0)),
            ),
            body=(Update(target="a", value="$0"),),
        )
