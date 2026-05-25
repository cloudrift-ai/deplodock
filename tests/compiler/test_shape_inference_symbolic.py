"""Symbolic-shape coverage for ``infer_output_shape`` on arithmetic ops.

After the Dim-carries-Expr refactor, the arithmetic ops (Matmul, Linear, Cat,
Reshape) compose shapes via ``Dim`` operators with no static/symbolic branch.
These tests pin down the resulting shapes for the patterns that matter for
whole-model HF compile: matmul-on-symbolic-M, cat-of-symbolic-axis,
reshape that turns a symbolic dim into a composite (``S * H``).
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.frontend.ir import CatOp, LinearOp, MatmulOp, ReshapeOp


def test_matmul_symbolic_m():
    # (1, S, K) @ (K, N) -> (1, S, N)
    out = MatmulOp().infer_output_shape([(Dim(1), Dim("S"), Dim(64)), (Dim(64), Dim(128))])
    assert out == (Dim(1), Dim("S"), Dim(128))


def test_linear_symbolic_m_lm_head_pattern():
    # (1, S, hidden=3584) @ (vocab=152064, hidden=3584).T -> (1, S, vocab)
    out = LinearOp().infer_output_shape([(Dim(1), Dim("S"), Dim(3584)), (Dim(152064), Dim(3584))])
    assert out == (Dim(1), Dim("S"), Dim(152064))


def test_cat_two_symbolic_halves_doubles():
    # RoPE half-rotation: concat two (1, H, S, D/2) tensors back to (1, H, S, D/2 + D/2)
    out = CatOp().infer_output_shape([(Dim(1), Dim(8), Dim("S"), Dim(32)), (Dim(1), Dim(8), Dim("S"), Dim(32))])
    assert out == (Dim(1), Dim(8), Dim("S"), Dim(64))


def test_cat_symbolic_axis_stays_symbolic():
    # cat along a symbolic axis produces 2*S as a composite Dim
    out = CatOp().infer_output_shape([(Dim(1), Dim("S")), (Dim(1), Dim("S"))])
    expected_last = Dim("S") + Dim("S")
    assert out == (Dim(1), expected_last)
    assert out[1].expr == BinaryExpr("+", Var("S"), Var("S"))


def test_reshape_minus_one_inferred_static():
    # Same as the legacy static case: -1 resolves to 3584 via numel arithmetic.
    out = ReshapeOp(shape=(1, 8, -1)).infer_output_shape([(Dim(1), Dim(28), Dim(8), Dim(128))])
    assert out == (Dim(1), Dim(8), Dim(3584))
    assert out[2].expr == Literal(3584, "int")


def test_reshape_minus_one_cancels_symbolic_factor():
    # (1, S, 2048).reshape(1, S, H, -1) with H=8 -> last dim = (S*2048) // (S*8).
    # The simplifier cancels the shared positive ``S`` factor (Dim arithmetic
    # injects a >=1 range for shape vars) and folds the int gcd: 2048/8 = 256.
    out = ReshapeOp(shape=(1, "S", 8, -1)).infer_output_shape([(Dim(1), Dim("S"), Dim(2048))])
    assert out == (Dim(1), Dim("S"), Dim(8), Dim(256))


def test_reshape_minus_one_drops_when_only_static_remains():
    # (1, S, 2048).reshape(1, S, -1): the shared ``S`` factor cancels and
    # 2048 / 1 = 2048 — the -1 resolves cleanly.
    out = ReshapeOp(shape=(1, "S", -1)).infer_output_shape([(Dim(1), Dim("S"), Dim(2048))])
    assert out == (Dim(1), Dim("S"), Dim(2048))


def test_reshape_minus_one_keeps_symbolic_when_unmatched():
    # (1, S, T).reshape(1, T, -1): the ``T`` factor cancels but ``S`` does not,
    # so -1 = S. Composite Dim that resolves at launch.
    out = ReshapeOp(shape=(1, "T", -1)).infer_output_shape([(Dim(1), Dim("S"), Dim("T"))])
    assert out[0] == Dim(1)
    assert out[1] == Dim("T")
    assert out[2] == Dim("S")
