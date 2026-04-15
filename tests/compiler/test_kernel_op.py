"""Tests for KernelOp: structural fused op with prologue/core/epilogue."""

from deplodock.compiler.backend.ir.expr import Var
from deplodock.compiler.coord_expr import placeholder
from deplodock.compiler.ir import Node, Tensor
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    IndexMapOp,
    IndexSource,
    KernelOp,
    Port,
    ReduceOp,
    ReduceStage,
)


def _node(nid: str, op, inputs: list[str], shape: tuple, dtype: str = "f32") -> Node:
    return Node(id=nid, op=op, inputs=inputs, output=Tensor(name=nid, shape=shape, dtype=dtype))


def test_pointwise_kernel_shape():
    """Pure pointwise: prologue only, core=None."""
    mul = _node("mul", ElementwiseOp("mul"), ["x", "y"], (4, 4))
    add = _node("add", ElementwiseOp("add"), ["mul", "bias"], (4, 4))

    k = KernelOp(
        inputs=[Port("x"), Port("y"), Port("bias")],
        outputs=[Port("add")],
        prologue=(mul, add),
        core=None,
        epilogue=(),
    )
    assert k.infer_output_shape([(4, 4), (4, 4), (4, 4)]) == (4, 4)
    assert k.core is None


def test_reduce_kernel_shape():
    """Reduce-chain: core is a tuple of ReduceStages."""
    square = _node("sq", ElementwiseOp("mul"), ["x", "x"], (8, 16))
    sum_node = _node("sum", ReduceOp("sum", axis=-1), ["sq"], (8,))
    stage = ReduceStage(pre_ops=(square,), reduce=sum_node)

    k = KernelOp(
        inputs=[Port("x")],
        outputs=[Port("sum")],
        prologue=(),
        core=(stage,),
        epilogue=(),
    )
    assert k.infer_output_shape([(8, 16)]) == (8,)
    assert isinstance(k.core, tuple)
    assert len(k.core) == 1


def test_multireduce_softmax_shape():
    """Softmax: two-stage ReduceCore plus an epilogue."""
    max_node = _node("max", ReduceOp("max", axis=-1), ["x"], (4,))
    sub = _node("sub", ElementwiseOp("sub"), ["x", "max"], (4, 8))
    exp = _node("exp", ElementwiseOp("exp"), ["sub"], (4, 8))
    sum_node = _node("sum", ReduceOp("sum", axis=-1), ["exp"], (4,))
    div = _node("div", ElementwiseOp("div"), ["exp", "sum"], (4, 8))

    stages = (
        ReduceStage(pre_ops=(), reduce=max_node),
        ReduceStage(pre_ops=(sub, exp), reduce=sum_node),
    )
    k = KernelOp(
        inputs=[Port("x")],
        outputs=[Port("div")],
        prologue=(),
        core=stages,
        epilogue=(div,),
    )
    # Epilogue wins — final shape = div's shape.
    assert k.infer_output_shape([(4, 8)]) == (4, 8)


def test_contraction_kernel_shape():
    """Contraction: core is ContractionCore; output shape derived from A/B ports."""
    core = ContractionCore(a=Port("A"), b=Port("B"), k_axis=1)
    k = KernelOp(
        inputs=[Port("A"), Port("B")],
        outputs=[Port("Y")],
        prologue=(),
        core=core,
        epilogue=(),
    )
    # A(M,K) @ B(K,N) → (M,N).
    assert k.infer_output_shape([(128, 64), (64, 256)]) == (128, 256)


def test_contraction_with_epilogue():
    """Epilogue overrides shape inference — epilogue's last node is the final output."""
    bias_add = _node("add", ElementwiseOp("add"), ["matmul", "bias"], (128, 256))
    core = ContractionCore(a=Port("A"), b=Port("B"), k_axis=1)
    k = KernelOp(
        inputs=[Port("A"), Port("B"), Port("bias")],
        outputs=[Port("add")],
        prologue=(),
        core=core,
        epilogue=(bias_add,),
    )
    assert k.infer_output_shape([(128, 64), (64, 256), (256,)]) == (128, 256)


def test_port_identity_vs_indexmap():
    """Identity Port has no IndexMap; IndexMap Port carries access-pattern info."""
    p1 = Port("x")
    assert p1.indexmap is None

    im = IndexMapOp(
        out_shape=(8, 4),
        sources=(
            IndexSource(
                input_idx=0,
                coord_map=(placeholder(1), placeholder(0)),
                select=None,
            ),
        ),
    )
    p2 = Port("x", indexmap=im)
    assert p2.indexmap is im
    assert p2.indexmap.out_shape == (8, 4)


def test_contraction_uses_indexmap_shape():
    """When A has an IndexMap (e.g. transpose-into-matmul), output derives from the
    IndexMap's out_shape, not the raw buffer shape."""
    # A_raw is (K,M); IndexMap presents it as (M,K) via transpose.
    transpose = IndexMapOp(
        out_shape=(128, 64),
        sources=(
            IndexSource(
                input_idx=0,
                coord_map=(placeholder(1), placeholder(0)),  # swap
                select=None,
            ),
        ),
    )
    core = ContractionCore(a=Port("A", indexmap=transpose), b=Port("B"), k_axis=1)
    k = KernelOp(
        inputs=[Port("A", indexmap=transpose), Port("B")],
        outputs=[Port("Y")],
        prologue=(),
        core=core,
        epilogue=(),
    )
    # Raw A is (64, 128); transpose presents (128, 64); matmul with B (64, 256) → (128, 256).
    assert k.infer_output_shape([(64, 128), (64, 256)]) == (128, 256)


def test_empty_kernel_returns_empty_shape():
    """Degenerate: no stages → empty shape tuple."""
    k = KernelOp(inputs=[], outputs=[], prologue=(), core=None, epilogue=())
    assert k.infer_output_shape([]) == ()


def test_kernel_op_name_registered_in_pattern_matcher():
    """The 'Kernel' short name maps to KernelOp in the pattern matcher."""
    from deplodock.compiler.pattern import resolve_op_class

    assert resolve_op_class("Kernel") == "KernelOp"


def test_reference_var_in_indexmap():
    """Sanity check: coord_expr.placeholder round-trips through Var."""
    p = placeholder(2)
    assert isinstance(p, Var)
    assert p.name.endswith("_2")
