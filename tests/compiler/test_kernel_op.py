"""Tests for the structural KernelOp IR with SSA Assign body."""

import pytest

from deplodock.compiler.backend.ir.expr import Var
from deplodock.compiler.ops import (
    Assign,
    Combine,
    ElementwiseOp,
    KernelOp,
    Mux,
    MuxBranch,
    Port,
    ReduceOp,
)

# ---------------------------------------------------------------------------
# Port / Combine / Mux
# ---------------------------------------------------------------------------


def test_port_plain():
    assert Port("x").buffer_id == "x"


def test_combine_binary():
    c = Combine(sources=(Port("a"), Port("b")), ops=(ElementwiseOp("add"),))
    assert len(c.sources) == 2


def test_combine_rejects_noop():
    with pytest.raises(ValueError):
        Combine(sources=(Port("a"),), ops=())


def test_mux():
    m = Mux(branches=(MuxBranch(Port("a"), Var("c1")), MuxBranch(Port("b"), Var("c2"))))
    assert len(m.branches) == 2


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", ElementwiseOp("add"), args=("x", "y"))
    assert a.name == "out"
    assert a.op.fn == "add"
    assert a.args == ("x", "y")


# ---------------------------------------------------------------------------
# KernelOp with SSA body
# ---------------------------------------------------------------------------


def test_kernel_pointwise():
    k = KernelOp(
        inputs=(Port("x"), Port("y")),
        body=(Assign("z", ElementwiseOp("add"), args=("x", "y")),),
        outputs=(Port("z"),),
    )
    assert len(k.body) == 1


def test_kernel_reduce():
    k = KernelOp(
        inputs=(Port("x"),),
        body=(Assign("s", ReduceOp("sum", -1), args=("x",)),),
        outputs=(Port("s"),),
    )
    assert isinstance(k.body[0].op, ReduceOp)


def test_kernel_matmul():
    k = KernelOp(
        inputs=(Port("a"), Port("b")),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("a", "b")),
            Assign("dot", ReduceOp("sum", -1), args=("mul",)),
        ),
        outputs=(Port("dot"),),
    )
    assert len(k.body) == 2


def test_kernel_matmul_bias():
    k = KernelOp(
        inputs=(Port("a"), Port("b"), Port("bias")),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("a", "b")),
            Assign("dot", ReduceOp("sum", -1), args=("mul",)),
            Assign("out", ElementwiseOp("add"), args=("dot", "bias")),
        ),
        outputs=(Port("out"),),
    )
    assert len(k.body) == 3


def test_kernel_softmax():
    k = KernelOp(
        inputs=(Port("x"),),
        body=(
            Assign("max", ReduceOp("max", -1), args=("x",)),
            Assign("sub", ElementwiseOp("sub"), args=("x", "max")),
            Assign("exp", ElementwiseOp("exp"), args=("sub",)),
            Assign("sum", ReduceOp("sum", -1), args=("exp",)),
            Assign("div", ElementwiseOp("div"), args=("exp", "sum")),
        ),
        outputs=(Port("div"),),
    )
    assert len(k.body) == 5


def test_kernel_unary_chain():
    k = KernelOp(
        inputs=(Port("x"),),
        body=(
            Assign("neg", ElementwiseOp("neg"), args=("x",)),
            Assign("exp", ElementwiseOp("exp"), args=("neg",)),
        ),
        outputs=(Port("exp"),),
    )
    assert len(k.body) == 2


def test_kernel_scatter_output():
    scatter = Mux(
        branches=(
            MuxBranch(input=Port("out_a"), select=Var("c1")),
            MuxBranch(input=Port("out_b"), select=Var("c2")),
        )
    )
    k = KernelOp(
        inputs=(Port("x"), Port("y")),
        body=(Assign("z", ElementwiseOp("add"), args=("x", "y")),),
        outputs=(scatter,),
    )
    assert isinstance(k.outputs[0], Mux)


# ---------------------------------------------------------------------------
# SSA validation
# ---------------------------------------------------------------------------


def test_ssa_rejects_undefined_arg():
    with pytest.raises(ValueError, match="not defined"):
        KernelOp(
            inputs=(Port("x"),),
            body=(Assign("y", ElementwiseOp("exp"), args=("z",)),),
            outputs=(Port("y"),),
        )


def test_ssa_rejects_duplicate_name():
    with pytest.raises(ValueError, match="already defined"):
        KernelOp(
            inputs=(Port("x"),),
            body=(
                Assign("y", ElementwiseOp("exp"), args=("x",)),
                Assign("y", ElementwiseOp("neg"), args=("y",)),
            ),
            outputs=(Port("y"),),
        )


def test_ssa_rejects_forward_reference():
    with pytest.raises(ValueError, match="not defined"):
        KernelOp(
            inputs=(Port("x"),),
            body=(
                Assign("a", ElementwiseOp("add"), args=("x", "b")),
                Assign("b", ElementwiseOp("exp"), args=("x",)),
            ),
            outputs=(Port("a"),),
        )


def test_ssa_allows_input_name_reuse_in_multiple_args():
    """Same input referenced by multiple Assigns is valid SSA."""
    k = KernelOp(
        inputs=(Port("x"),),
        body=(
            Assign("a", ElementwiseOp("exp"), args=("x",)),
            Assign("b", ElementwiseOp("neg"), args=("x",)),
            Assign("c", ElementwiseOp("add"), args=("a", "b")),
        ),
        outputs=(Port("c"),),
    )
    assert len(k.body) == 3
