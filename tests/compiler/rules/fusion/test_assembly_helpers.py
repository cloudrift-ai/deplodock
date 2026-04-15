"""Unit tests for shared assembly helpers (ported _can_merge logic)."""

from __future__ import annotations

from deplodock.compiler.ir import Node, Tensor
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    KernelOp,
    Port,
    ReduceOp,
    ReduceStage,
)
from deplodock.compiler.rules.fusion._assembly_helpers import (
    is_row_reduce,
    kernel_has_contraction,
    kernel_kind,
    kernel_last_node_id,
    kernel_reduces_with_input_shapes,
    merged_external_inputs_compat,
    reduces_compatible,
    rewire_node_input,
)

# ---------------------------------------------------------------------------
# is_row_reduce
# ---------------------------------------------------------------------------


def test_is_row_reduce_negative_axis():
    assert is_row_reduce(ReduceOp("sum", -1), input_ndim=3)


def test_is_row_reduce_explicit_last_axis():
    assert is_row_reduce(ReduceOp("max", 2), input_ndim=3)


def test_is_row_reduce_non_last_axis():
    assert not is_row_reduce(ReduceOp("sum", 0), input_ndim=3)


def test_is_row_reduce_symbolic_axis():
    assert not is_row_reduce(ReduceOp("sum", "K"), input_ndim=3)


# ---------------------------------------------------------------------------
# reduces_compatible
# ---------------------------------------------------------------------------


def test_reduces_compatible_softmax_pair():
    # softmax: max + sum, both row-reduces over same shape
    r1 = ReduceOp("max", -1)
    r2 = ReduceOp("sum", -1)
    assert reduces_compatible(r1, (4, 16), r2, (4, 16))


def test_reduces_incompatible_mixed_axes():
    r1 = ReduceOp("sum", -1)  # row reduce
    r2 = ReduceOp("sum", 0)  # not row reduce
    assert not reduces_compatible(r1, (4, 16), r2, (4, 16))


def test_reduces_compatible_same_rank_same_trailing():
    # Per ported logic: same rank + same trailing dim → compatible.
    r1 = ReduceOp("sum", -1)
    r2 = ReduceOp("max", -1)
    assert reduces_compatible(r1, (4, 16), r2, (8, 16))


def test_reduces_incompatible_different_rank():
    r1 = ReduceOp("sum", -1)
    r2 = ReduceOp("max", -1)
    assert not reduces_compatible(r1, (4, 16), r2, (4, 4, 16))


def test_reduces_incompatible_different_trailing():
    r1 = ReduceOp("sum", -1)
    r2 = ReduceOp("max", -1)
    assert not reduces_compatible(r1, (4, 16), r2, (4, 32))


def test_reduces_incompatible_symbolic_axes():
    r1 = ReduceOp("sum", "K")
    r2 = ReduceOp("sum", "K")
    assert not reduces_compatible(r1, (4, 8), r2, (4, 8))


# ---------------------------------------------------------------------------
# kernel_kind / kernel_has_contraction
# ---------------------------------------------------------------------------


def _pointwise_kernel():
    add = Node(id="add", op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("z", (4,)))
    return KernelOp(
        inputs=[Port("x"), Port("y")],
        outputs=[Port("add")],
        prologue=(add,),
        core=None,
        external_shapes={"x": (4,), "y": (4,)},
    )


def _reduce_kernel():
    red = Node(id="r", op=ReduceOp("sum", -1), inputs=["x"], output=Tensor("s", (4,)))
    return KernelOp(
        inputs=[Port("x")],
        outputs=[Port("r")],
        core=(ReduceStage(pre_ops=(), reduce=red),),
        external_shapes={"x": (4, 16)},
    )


def _contraction_kernel():
    mul = Node(id="m", op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8, 16)))
    red = Node(id="r", op=ReduceOp("sum", 1), inputs=["m"], output=Tensor("c", (4, 16)))
    return KernelOp(
        inputs=[Port("a"), Port("b")],
        outputs=[Port("r")],
        core=ContractionCore(a=Port("a"), b=Port("b"), k_axis=1, mul=mul, reduce=red),
        external_shapes={"a": (4, 8), "b": (8, 16)},
    )


def test_kernel_kind_pointwise():
    assert kernel_kind(_pointwise_kernel()) == "pointwise"


def test_kernel_kind_reduce():
    assert kernel_kind(_reduce_kernel()) == "reduce"


def test_kernel_kind_contraction():
    assert kernel_kind(_contraction_kernel()) == "contraction"


def test_kernel_has_contraction():
    assert kernel_has_contraction(_contraction_kernel())
    assert not kernel_has_contraction(_pointwise_kernel())
    assert not kernel_has_contraction(_reduce_kernel())


# ---------------------------------------------------------------------------
# kernel_reduces_with_input_shapes
# ---------------------------------------------------------------------------


def test_kernel_reduces_pointwise_empty():
    assert kernel_reduces_with_input_shapes(_pointwise_kernel()) == []


def test_kernel_reduces_single_reduce():
    k = _reduce_kernel()
    pairs = kernel_reduces_with_input_shapes(k)
    assert len(pairs) == 1
    op, shape = pairs[0]
    assert op.fn == "sum" and op.axis == -1
    # Reduce input is the external x buffer; shape is x's full shape.
    assert shape == (4, 16)


def test_kernel_reduces_contraction():
    k = _contraction_kernel()
    pairs = kernel_reduces_with_input_shapes(k)
    assert len(pairs) == 1
    op, shape = pairs[0]
    assert op.fn == "sum"
    # Reduce input is the mul output shape.
    assert shape == (4, 8, 16)


# ---------------------------------------------------------------------------
# kernel_last_node_id
# ---------------------------------------------------------------------------


def test_kernel_last_node_id_pointwise():
    assert kernel_last_node_id(_pointwise_kernel()) == "add"


def test_kernel_last_node_id_reduce():
    assert kernel_last_node_id(_reduce_kernel()) == "r"


def test_kernel_last_node_id_contraction():
    assert kernel_last_node_id(_contraction_kernel()) == "r"


def test_kernel_last_node_id_with_epilogue():
    k = _contraction_kernel()
    relu = Node(id="relu", op=ElementwiseOp("relu"), inputs=["r"], output=Tensor("o", (4, 16)))
    k = KernelOp(
        inputs=k.inputs,
        outputs=[Port("relu")],
        prologue=k.prologue,
        core=k.core,
        epilogue=(relu,),
        external_shapes=k.external_shapes,
    )
    assert kernel_last_node_id(k) == "relu"


# ---------------------------------------------------------------------------
# rewire_node_input
# ---------------------------------------------------------------------------


def test_rewire_node_input_replaces_only_target():
    n = Node(id="n", op=ElementwiseOp("add"), inputs=["a", "b", "a"], output=Tensor("o", (4,)))
    rewired = rewire_node_input(n, "a", "z")
    assert rewired.inputs == ["z", "b", "z"]
    # Original untouched.
    assert n.inputs == ["a", "b", "a"]


# ---------------------------------------------------------------------------
# merged_external_inputs_compat
# ---------------------------------------------------------------------------


def test_merge_compat_single_input():
    assert merged_external_inputs_compat([(4, 16)])


def test_merge_compat_two_2d_same_size():
    assert merged_external_inputs_compat([(4, 16), (4, 16)])


def test_merge_compat_2d_plus_1d():
    # 1D input is exempt — uses [j] indexing.
    assert merged_external_inputs_compat([(4, 16), (16,)])


def test_merge_compat_broadcast_compatible():
    # Smaller broadcasts to larger.
    assert merged_external_inputs_compat([(4, 16), (1, 16)])


def test_merge_compat_2d_size_mismatch_rejected():
    # (8, 16) vs (4, 16) — same trailing dim but neither broadcasts.
    assert not merged_external_inputs_compat([(8, 16), (4, 16)])


def test_merge_compat_pure_contraction_skipped():
    # Pure contraction kernels use A/B indexing — shape check skipped.
    assert merged_external_inputs_compat([(4, 8), (8, 16)], is_pure_contraction=True)
