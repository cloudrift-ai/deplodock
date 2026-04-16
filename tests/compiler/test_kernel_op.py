"""Tests for the structural KernelOp IR.

Covers construction of each KernelInput variant (Port / Mux / Combine),
the contraction / reduce-chain / epilogue slots on KernelOp, and the
runtime invariants enforced by ``__post_init__``.
"""

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
    ReduceStage,
)

# ---------------------------------------------------------------------------
# Port — signal-flow leaf
# ---------------------------------------------------------------------------


def test_port_plain_read():
    p = Port("x")
    assert p.buffer_id == "x"
    assert p.indexmap is None


def test_port_with_indexmap():
    idx = IndexMapOp(out_shape=(4, 3), sources=(IndexSource(input_idx=0, coord_map=(Var("out_coord_0"),)),))
    p = Port("x", indexmap=idx)
    assert p.indexmap is idx


# ---------------------------------------------------------------------------
# Combine — operadic composition
# ---------------------------------------------------------------------------


def test_combine_cross_input_add():
    add = ElementwiseOp("add")
    c = Combine(sources=(Port("a"), Port("b")), ops=(add,))
    assert len(c.sources) == 2
    assert c.ops == (add,)


def test_combine_rejects_no_op_wrapper():
    with pytest.raises(ValueError, match="no-op wrapper"):
        Combine(sources=(Port("a"),), ops=())


def test_combine_rejects_empty_sources():
    with pytest.raises(ValueError, match="non-empty"):
        Combine(sources=(), ops=())


def test_combine_rejects_reduce_in_ops():
    red = ReduceOp("sum", -1)
    with pytest.raises(TypeError, match="expected ElementwiseOp"):
        Combine(sources=(Port("a"),), ops=(red,))


# ---------------------------------------------------------------------------
# Mux — hardware multiplexer
# ---------------------------------------------------------------------------


def test_mux_kv_cache_pattern():
    m = Mux(
        branches=(
            MuxBranch(input=Port("k_cache"), select=Var("t_lt_past")),
            MuxBranch(input=Port("k_new"), select=Var("t_ge_past")),
        )
    )
    assert len(m.branches) == 2
    assert isinstance(m.branches[0].input, Port)


def test_mux_rejects_empty_branches():
    with pytest.raises(ValueError, match="non-empty"):
        Mux(branches=())


def test_mux_nests_into_combine():
    dequant = ElementwiseOp("mul")
    muxed = Mux(
        branches=(
            MuxBranch(input=Port("src_a"), select=Var("cond_a")),
            MuxBranch(input=Port("src_b"), select=Var("cond_b")),
        )
    )
    c = Combine(sources=(muxed, Port("scale")), ops=(dequant,))
    assert c.sources[0] is muxed


# ---------------------------------------------------------------------------
# ContractionCore — systolic core
# ---------------------------------------------------------------------------


def test_contraction_matmul_shape():
    mul = ElementwiseOp("mul")
    red = ReduceOp("sum", -1)
    operand = Combine(sources=(Port("a"), Port("b")), ops=(mul,))
    cc = ContractionCore(operand=operand, reduce=red)
    assert isinstance(cc.operand, Combine)


def test_contraction_rejects_elementwise_as_reduce():
    mul = ElementwiseOp("mul")
    with pytest.raises(TypeError, match="expected ReduceOp"):
        ContractionCore(operand=Port("x"), reduce=mul)


# ---------------------------------------------------------------------------
# ReduceStage — one reduction in a chain
# ---------------------------------------------------------------------------


def test_reduce_stage_empty_pre_ops():
    red = ReduceOp("max", -1)
    stage = ReduceStage(pre_ops=(), reduce=red)
    assert stage.pre_ops == ()


def test_reduce_stage_softmax_sum():
    sub = ElementwiseOp("sub")
    exp = ElementwiseOp("exp")
    red = ReduceOp("sum", -1)
    stage = ReduceStage(pre_ops=(sub, exp), reduce=red)
    assert stage.pre_ops == (sub, exp)


def test_reduce_stage_rejects_reduce_in_pre_ops():
    red = ReduceOp("sum", -1)
    with pytest.raises(TypeError, match="expected ElementwiseOp"):
        ReduceStage(pre_ops=(red,), reduce=red)


def test_reduce_stage_rejects_elementwise_as_reduce():
    mul = ElementwiseOp("mul")
    with pytest.raises(TypeError, match="expected ReduceOp"):
        ReduceStage(pre_ops=(), reduce=mul)


# ---------------------------------------------------------------------------
# KernelOp — tiled dataflow pipeline
# ---------------------------------------------------------------------------


def test_kernel_pointwise():
    add = ElementwiseOp("add")
    body = Combine(sources=(Port("x"), Port("y")), ops=(add,))
    k = KernelOp(inputs=(body,), outputs=(Port("z"),))
    assert k.contraction is None
    assert k.reduce_stages == ()
    assert k.epilogue == ()


def test_kernel_matmul():
    mul = ElementwiseOp("mul")
    red = ReduceOp("sum", -1)
    operand = Combine(sources=(Port("a"), Port("b")), ops=(mul,))
    cc = ContractionCore(operand=operand, reduce=red)
    k = KernelOp(inputs=(Port("a"), Port("b")), outputs=(Port("y"),), contraction=cc)
    assert k.contraction is cc


def test_kernel_matmul_plus_softmax():
    mul = ElementwiseOp("mul")
    contraction_reduce = ReduceOp("sum", -1)
    operand = Combine(sources=(Port("Q"), Port("K")), ops=(mul,))
    cc = ContractionCore(operand=operand, reduce=contraction_reduce)

    rmax = ReduceOp("max", -1)
    sub = ElementwiseOp("sub")
    exp = ElementwiseOp("exp")
    rsum = ReduceOp("sum", -1)
    div = ElementwiseOp("div")

    k = KernelOp(
        inputs=(Port("Q"), Port("K")),
        outputs=(Port("S"),),
        contraction=cc,
        reduce_stages=(
            ReduceStage(pre_ops=(), reduce=rmax),
            ReduceStage(pre_ops=(sub, exp), reduce=rsum),
        ),
        epilogue=(div,),
    )
    assert len(k.reduce_stages) == 2
    assert k.epilogue == (div,)


def test_kernel_pure_reduce_chain():
    rmax = ReduceOp("max", -1)
    sub = ElementwiseOp("sub")
    exp = ElementwiseOp("exp")
    rsum = ReduceOp("sum", -1)
    div = ElementwiseOp("div")

    k = KernelOp(
        inputs=(Port("x"),),
        outputs=(Port("y"),),
        reduce_stages=(
            ReduceStage(pre_ops=(), reduce=rmax),
            ReduceStage(pre_ops=(sub, exp), reduce=rsum),
        ),
        epilogue=(div,),
    )
    assert k.contraction is None
    assert len(k.reduce_stages) == 2


def test_kernel_scatter_output():
    add = ElementwiseOp("add")
    body = Combine(sources=(Port("x"), Port("y")), ops=(add,))
    scatter = Mux(
        branches=(
            MuxBranch(input=Port("out_a"), select=Var("cond_a")),
            MuxBranch(input=Port("out_b"), select=Var("cond_b")),
        )
    )
    k = KernelOp(inputs=(body,), outputs=(scatter,))
    assert isinstance(k.outputs[0], Mux)


def test_kernel_epilogue_rejects_reduce():
    red = ReduceOp("sum", -1)
    with pytest.raises(TypeError, match="expected ElementwiseOp"):
        KernelOp(inputs=(Port("x"),), outputs=(Port("y"),), epilogue=(red,))
