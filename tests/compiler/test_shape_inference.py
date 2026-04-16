"""Unit tests for Op.infer_output_shape and shape_utils."""

import pytest

from deplodock.compiler.ops import (
    CatOp,
    ElementwiseOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReduceOp,
    ReshapeOp,
    SdpaOp,
    SliceOp,
    TransposeOp,
    UnsqueezeOp,
)
from deplodock.compiler.shape_utils import broadcast_shapes


def test_broadcast_shapes_scalar_with_tensor():
    assert broadcast_shapes((1,), (4, 3)) == (4, 3)


def test_broadcast_shapes_left_padding():
    assert broadcast_shapes((3,), (4, 3)) == (4, 3)


def test_broadcast_shapes_size_one_dims():
    assert broadcast_shapes((1, 1, 8, 128), (1, 28, 8, 128)) == (1, 28, 8, 128)


def test_broadcast_shapes_incompatible_raises():
    with pytest.raises(ValueError):
        broadcast_shapes((4, 3), (5, 3))


def test_broadcast_shapes_divisible_only_when_opted_in():
    with pytest.raises(ValueError):
        broadcast_shapes((28, 8, 128), (4, 8, 128))
    assert broadcast_shapes((28, 8, 128), (4, 8, 128), allow_divisible=True) == (28, 8, 128)


def test_elementwise_infer():
    op = ElementwiseOp("mul")
    assert op.infer_output_shape([(1, 28, 8, 128), (1, 1, 8, 128)]) == (1, 28, 8, 128)


def test_reduce_drops_axis():
    assert ReduceOp("sum", axis=-1).infer_output_shape([(1, 8, 128)]) == (1, 8)
    assert ReduceOp("max", axis=1).infer_output_shape([(4, 8, 128)]) == (4, 128)


def test_mean_drops_axis():
    assert MeanOp(axis=-1).infer_output_shape([(1, 8, 3584)]) == (1, 8)


def test_transpose_swap():
    assert TransposeOp(axes=(1, 2)).infer_output_shape([(1, 8, 28, 128)]) == (1, 28, 8, 128)


def test_transpose_full_permute():
    assert TransposeOp(axes=(0, 2, 1, 3)).infer_output_shape([(1, 8, 28, 128)]) == (1, 28, 8, 128)


def test_reshape_returns_op_shape():
    assert ReshapeOp(shape=(1, 8, 3584)).infer_output_shape([(1, 28, 8, 128)]) == (1, 8, 3584)


def test_unsqueeze_inserts_dim():
    assert UnsqueezeOp(dim=1).infer_output_shape([(1, 8, 128)]) == (1, 1, 8, 128)
    assert UnsqueezeOp(dim=-1).infer_output_shape([(1, 8, 128)]) == (1, 8, 128, 1)


def test_slice_returns_op_shape():
    assert SliceOp(shape=(1, 28, 8, 64)).infer_output_shape([(1, 28, 8, 128)]) == (1, 28, 8, 64)


def test_cat_sums_last_dim():
    assert CatOp().infer_output_shape([(1, 28, 8, 64), (1, 28, 8, 64)]) == (1, 28, 8, 128)


def test_linear_replaces_last_with_out_features():
    assert LinearOp().infer_output_shape([(1, 8, 3584), (3584, 3584)]) == (1, 8, 3584)
    # K projection: out_features=512 (the -2 dim of weight)
    assert LinearOp().infer_output_shape([(1, 8, 3584), (512, 3584)]) == (1, 8, 512)


def test_matmul_standard():
    assert MatmulOp().infer_output_shape([(8, 16), (16, 32)]) == (8, 32)


def test_sdpa_uses_v_last_dim():
    q = (1, 28, 8, 64)
    k = (1, 4, 8, 64)
    v = (1, 4, 8, 64)
    assert SdpaOp().infer_output_shape([q, k, v]) == (1, 28, 8, 64)
