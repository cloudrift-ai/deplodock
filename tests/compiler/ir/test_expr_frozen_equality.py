"""Lock in that ``Expr`` nodes are frozen + structurally hashable.

Required so ``Dim`` (which will carry an ``Expr``) keeps a stable
``__hash__`` across construction paths and survives use as a dict key /
set member in ``Axis`` and graph structural keys.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from deplodock.compiler.ir.expr import (
    BinaryExpr,
    Builtin,
    CastExpr,
    FuncCallExpr,
    Literal,
    TernaryExpr,
    Var,
)


def test_var_structural_equality():
    assert Var("x") == Var("x")
    assert Var("x") != Var("y")
    assert hash(Var("x")) == hash(Var("x"))


def test_literal_structural_equality():
    assert Literal(1, "int") == Literal(1, "int")
    assert Literal(1, "int") != Literal(1, "float")
    assert hash(Literal(1, "int")) == hash(Literal(1, "int"))


def test_binary_expr_structural_equality():
    e1 = BinaryExpr("+", Var("a"), Literal(2, "int"))
    e2 = BinaryExpr("+", Var("a"), Literal(2, "int"))
    assert e1 == e2
    assert hash(e1) == hash(e2)


def test_funccallexpr_structural_equality():
    f1 = FuncCallExpr("exp", (Var("x"),))
    f2 = FuncCallExpr("exp", (Var("x"),))
    assert f1 == f2
    assert hash(f1) == hash(f2)


def test_ternary_and_cast_hashable():
    t = TernaryExpr(Var("c"), Var("a"), Var("b"))
    assert hash(t) == hash(TernaryExpr(Var("c"), Var("a"), Var("b")))
    c = CastExpr("int", Var("x"))
    assert hash(c) == hash(CastExpr("int", Var("x")))


def test_builtin_hashable():
    assert Builtin("tid.x") == Builtin("tid.x")
    assert hash(Builtin("tid.x")) == hash(Builtin("tid.x"))


def test_var_is_frozen():
    v = Var("x")
    with pytest.raises(FrozenInstanceError):
        v.name = "y"  # type: ignore[misc]


def test_binary_expr_is_frozen():
    e = BinaryExpr("+", Var("a"), Literal(1, "int"))
    with pytest.raises(FrozenInstanceError):
        e.op = "-"  # type: ignore[misc]


def test_funccallexpr_args_is_tuple():
    f = FuncCallExpr("exp", (Var("x"),))
    assert isinstance(f.args, tuple)


def test_expr_works_as_dict_key():
    """Used to key ``SimplifyCtx.ranges``, hash-cache equivalent subexpressions, etc."""
    d: dict[object, int] = {}
    d[BinaryExpr("+", Var("a"), Literal(1, "int"))] = 7
    assert d[BinaryExpr("+", Var("a"), Literal(1, "int"))] == 7
