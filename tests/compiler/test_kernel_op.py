"""Tests for the structural LoopOp IR with SSA Assign body."""

import pytest

from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Assign, Combine, LoopOp, Mux, MuxBranch, Port
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp

# ---------------------------------------------------------------------------
# Port / Combine / Mux
# ---------------------------------------------------------------------------


def test_port_plain():
    assert Port().indexmap is None


def test_combine_binary():
    c = Combine(sources=(Port(), Port()), ops=(ElementwiseOp("add"),))
    assert len(c.sources) == 2


def test_combine_rejects_noop():
    with pytest.raises(ValueError):
        Combine(sources=(Port(),), ops=())


def test_mux():
    m = Mux(branches=(MuxBranch(Port(), Var("c1")), MuxBranch(Port(), Var("c2"))))
    assert len(m.branches) == 2


# ---------------------------------------------------------------------------
# Assign
# ---------------------------------------------------------------------------


def test_assign_construction():
    a = Assign("out", ElementwiseOp("add"), args=("$0", "$1"))
    assert a.name == "out"
    assert a.op.fn == "add"
    assert a.args == ("$0", "$1")


# ---------------------------------------------------------------------------
# LoopOp with SSA body
# ---------------------------------------------------------------------------


def test_kernel_pointwise():
    k = LoopOp(
        inputs=(Port(), Port()),
        body=(Assign("z", ElementwiseOp("add"), args=("$0", "$1")),),
        outputs=(Port(),),
    )
    assert len(k.body) == 1


def test_kernel_reduce():
    k = LoopOp(
        inputs=(Port(),),
        body=(Assign("s", ReduceOp("sum", -1), args=("$0",)),),
        outputs=(Port(),),
    )
    assert isinstance(k.body[0].op, ReduceOp)


def test_kernel_matmul():
    k = LoopOp(
        inputs=(Port(), Port()),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Assign("dot", ReduceOp("sum", -1), args=("mul",)),
        ),
        outputs=(Port(),),
    )
    assert len(k.body) == 2


def test_kernel_matmul_bias():
    k = LoopOp(
        inputs=(Port(), Port(), Port()),
        body=(
            Assign("mul", ElementwiseOp("mul"), args=("$0", "$1")),
            Assign("dot", ReduceOp("sum", -1), args=("mul",)),
            Assign("out", ElementwiseOp("add"), args=("dot", "$2")),
        ),
        outputs=(Port(),),
    )
    assert len(k.body) == 3


def test_kernel_softmax():
    k = LoopOp(
        inputs=(Port(),),
        body=(
            Assign("max", ReduceOp("max", -1), args=("$0",)),
            Assign("sub", ElementwiseOp("sub"), args=("$0", "max")),
            Assign("exp", ElementwiseOp("exp"), args=("sub",)),
            Assign("sum", ReduceOp("sum", -1), args=("exp",)),
            Assign("div", ElementwiseOp("div"), args=("exp", "sum")),
        ),
        outputs=(Port(),),
    )
    assert len(k.body) == 5


def test_kernel_unary_chain():
    k = LoopOp(
        inputs=(Port(),),
        body=(
            Assign("neg", ElementwiseOp("neg"), args=("$0",)),
            Assign("exp", ElementwiseOp("exp"), args=("neg",)),
        ),
        outputs=(Port(),),
    )
    assert len(k.body) == 2


def test_kernel_scatter_output():
    scatter = Mux(
        branches=(
            MuxBranch(input=Port(), select=Var("c1")),
            MuxBranch(input=Port(), select=Var("c2")),
        )
    )
    k = LoopOp(
        inputs=(Port(), Port()),
        body=(Assign("z", ElementwiseOp("add"), args=("$0", "$1")),),
        outputs=(scatter,),
    )
    assert isinstance(k.outputs[0], Mux)


# ---------------------------------------------------------------------------
# SSA validation
# ---------------------------------------------------------------------------


def test_ssa_rejects_undefined_arg():
    with pytest.raises(ValueError, match="not defined"):
        LoopOp(
            inputs=(Port(),),
            body=(Assign("y", ElementwiseOp("exp"), args=("z",)),),
            outputs=(Port(),),
        )


def test_ssa_rejects_duplicate_name():
    with pytest.raises(ValueError, match="already defined"):
        LoopOp(
            inputs=(Port(),),
            body=(
                Assign("y", ElementwiseOp("exp"), args=("$0",)),
                Assign("y", ElementwiseOp("neg"), args=("y",)),
            ),
            outputs=(Port(),),
        )


def test_ssa_rejects_forward_reference():
    with pytest.raises(ValueError, match="not defined"):
        LoopOp(
            inputs=(Port(),),
            body=(
                Assign("a", ElementwiseOp("add"), args=("$0", "b")),
                Assign("b", ElementwiseOp("exp"), args=("$0",)),
            ),
            outputs=(Port(),),
        )


def test_ssa_allows_input_name_reuse_in_multiple_args():
    k = LoopOp(
        inputs=(Port(),),
        body=(
            Assign("a", ElementwiseOp("exp"), args=("$0",)),
            Assign("b", ElementwiseOp("neg"), args=("$0",)),
            Assign("c", ElementwiseOp("add"), args=("a", "b")),
        ),
        outputs=(Port(),),
    )
    assert len(k.body) == 3
