"""Tests for the structural KernelOp IR."""

import pytest

from deplodock.compiler.backend.ir.expr import Var
from deplodock.compiler.ops import (
    Combine,
    ContractionCore,
    ElementwiseOp,
    IndexMapOp,
    IndexSource,
    KernelOp,
    Mux,
    MuxBranch,
    Port,
    ReduceOp,
)


def test_port_plain():
    assert Port("x").buffer_id == "x"


def test_port_indexmap():
    idx = IndexMapOp(out_shape=(4, 3), sources=(IndexSource(input_idx=0, coord_map=(Var("out_coord_0"),)),))
    assert Port("x", indexmap=idx).indexmap is idx


def test_combine_binary():
    c = Combine(sources=(Port("a"), Port("b")), ops=(ElementwiseOp("add"),))
    assert len(c.sources) == 2


def test_combine_rejects_noop():
    with pytest.raises(ValueError):
        Combine(sources=(Port("a"),), ops=())


def test_combine_rejects_reduce_in_ops():
    with pytest.raises(TypeError):
        Combine(sources=(Port("a"),), ops=(ReduceOp("sum", -1),))


def test_mux():
    m = Mux(branches=(MuxBranch(Port("a"), Var("c1")), MuxBranch(Port("b"), Var("c2"))))
    assert len(m.branches) == 2


def test_contraction():
    cc = ContractionCore(
        operand=Combine(sources=(Port("a"), Port("b")), ops=(ElementwiseOp("mul"),)),
        reduce=ReduceOp("sum", -1),
    )
    assert isinstance(cc.operand, Combine)


def test_contraction_rejects_elementwise():
    with pytest.raises(TypeError):
        ContractionCore(operand=Port("x"), reduce=ElementwiseOp("mul"))


def test_kernel_pointwise():
    k = KernelOp(inputs=(Combine(sources=(Port("x"),), ops=(ElementwiseOp("exp"),)),), outputs=(Port("y"),))
    assert k.body == ()


def test_kernel_reduce():
    k = KernelOp(inputs=(Port("x"),), outputs=(Port("y"),), body=(ReduceOp("sum", -1),))
    assert isinstance(k.body[0], ReduceOp)


def test_kernel_softmax_body():
    k = KernelOp(
        inputs=(Port("x"),),
        outputs=(Port("y"),),
        body=(ReduceOp("max", -1), ElementwiseOp("sub"), ElementwiseOp("exp"), ReduceOp("sum", -1), ElementwiseOp("div")),
    )
    assert len(k.body) == 5


def test_kernel_matmul_with_body():
    cc = ContractionCore(
        operand=Combine(sources=(Port("a"), Port("b")), ops=(ElementwiseOp("mul"),)),
        reduce=ReduceOp("sum", -1),
    )
    k = KernelOp(inputs=(Port("bias"),), outputs=(Port("y"),), contraction=cc, body=(ElementwiseOp("add"),))
    assert len(k.body) == 1
