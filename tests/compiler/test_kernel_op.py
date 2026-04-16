"""Tests for the structural KernelOp IR.

Covers construction of each KernelInput variant (Port / Mux / Combine),
the contraction / reduce-chain / epilogue slots on KernelOp, and the
runtime invariants enforced by ``__post_init__``.
"""

import pytest

from deplodock.compiler.backend.ir.expr import Var
from deplodock.compiler.ir import Node, Tensor
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


def _ew(nid: str, fn: str, inputs: list[str], shape: tuple) -> Node:
    return Node(id=nid, op=ElementwiseOp(fn=fn), inputs=inputs, output=Tensor(name=nid, shape=shape))


def _red(nid: str, fn: str, axis, inputs: list[str], shape: tuple) -> Node:
    return Node(id=nid, op=ReduceOp(fn=fn, axis=axis), inputs=inputs, output=Tensor(name=nid, shape=shape))


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
    add = _ew("n0", "add", ["a", "b"], (4,))
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
    red = _red("n0", "sum", -1, ["a"], ())
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
    """A Combine can have a Mux as one of its sources."""
    dequant = _ew("n0", "mul", ["src", "scale"], (4,))
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
    mul = _ew("mul", "mul", ["a", "b"], (4, 3, 5))  # per-K product
    red = _red("red", "sum", -1, ["mul"], (4, 3))
    operand = Combine(sources=(Port("a"), Port("b")), ops=(mul,))
    cc = ContractionCore(operand=operand, k_axis=-1, reduce=red)
    assert cc.k_axis == -1
    assert isinstance(cc.operand, Combine)


def test_contraction_rejects_elementwise_as_reduce():
    mul = _ew("mul", "mul", ["a", "b"], (4, 3, 5))
    with pytest.raises(TypeError, match="expected ReduceOp"):
        ContractionCore(operand=Port("x"), k_axis=-1, reduce=mul)


# ---------------------------------------------------------------------------
# ReduceStage — one reduction in a chain
# ---------------------------------------------------------------------------


def test_reduce_stage_empty_pre_ops():
    red = _red("r0", "max", -1, ["x"], (4,))
    stage = ReduceStage(pre_ops=(), reduce=red)
    assert stage.pre_ops == ()


def test_reduce_stage_softmax_sum():
    sub = _ew("sub", "sub", ["x", "rmax"], (4, 8))
    exp = _ew("exp", "exp", ["sub"], (4, 8))
    red = _red("rsum", "sum", -1, ["exp"], (4,))
    stage = ReduceStage(pre_ops=(sub, exp), reduce=red)
    assert stage.pre_ops == (sub, exp)


def test_reduce_stage_rejects_reduce_in_pre_ops():
    red = _red("r0", "sum", -1, ["x"], (4,))
    with pytest.raises(TypeError, match="expected ElementwiseOp"):
        ReduceStage(pre_ops=(red,), reduce=red)


def test_reduce_stage_rejects_elementwise_as_reduce():
    mul = _ew("mul", "mul", ["x", "y"], (4,))
    with pytest.raises(TypeError, match="expected ReduceOp"):
        ReduceStage(pre_ops=(), reduce=mul)


# ---------------------------------------------------------------------------
# KernelOp — tiled dataflow pipeline
# ---------------------------------------------------------------------------


def test_kernel_pointwise():
    """Pointwise kernel: no contraction, no reduces, all work in inputs[0]."""
    add = _ew("add", "add", ["x", "y"], (4,))
    body = Combine(sources=(Port("x"), Port("y")), ops=(add,))
    k = KernelOp(inputs=(body,), outputs=(Port("z"),))
    assert k.contraction is None
    assert k.reduce_stages == ()
    assert k.epilogue == ()


def test_kernel_matmul():
    """Plain matmul: contraction only, no reduce_stages, no epilogue."""
    mul = _ew("mul", "mul", ["a", "b"], (4, 3, 5))
    red = _red("red", "sum", -1, ["mul"], (4, 3))
    operand = Combine(sources=(Port("a"), Port("b")), ops=(mul,))
    cc = ContractionCore(operand=operand, k_axis=-1, reduce=red)
    k = KernelOp(
        inputs=(Port("a"), Port("b")),
        outputs=(Port("y"),),
        contraction=cc,
    )
    assert k.contraction is cc
    assert k.reduce_stages == ()


def test_kernel_matmul_plus_softmax():
    """Contraction + reduce chain: matmul feeding softmax (max, sub+exp+sum, div)."""
    mul = _ew("mul", "mul", ["q", "k"], (4, 8, 16))
    contraction_reduce = _red("cred", "sum", -1, ["mul"], (4, 8))
    operand = Combine(sources=(Port("Q"), Port("K")), ops=(mul,))
    cc = ContractionCore(operand=operand, k_axis=-1, reduce=contraction_reduce)

    rmax = _red("rmax", "max", -1, ["cred"], (4,))
    sub = _ew("sub", "sub", ["cred", "rmax"], (4, 8))
    exp = _ew("exp", "exp", ["sub"], (4, 8))
    rsum = _red("rsum", "sum", -1, ["exp"], (4,))
    div = _ew("div", "div", ["exp", "rsum"], (4, 8))

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
    """Pure softmax: reduce_stages without a contraction."""
    rmax = _red("rmax", "max", -1, ["x"], (4,))
    sub = _ew("sub", "sub", ["x", "rmax"], (4, 8))
    exp = _ew("exp", "exp", ["sub"], (4, 8))
    rsum = _red("rsum", "sum", -1, ["exp"], (4,))
    div = _ew("div", "div", ["exp", "rsum"], (4, 8))

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
    """Mux on the output side models scatter / masked writeout."""
    add = _ew("add", "add", ["x", "y"], (4,))
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
    red = _red("r0", "sum", -1, ["x"], (4,))
    with pytest.raises(TypeError, match="expected ElementwiseOp"):
        KernelOp(
            inputs=(Port("x"),),
            outputs=(Port("y"),),
            epilogue=(red,),
        )
