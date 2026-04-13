"""Shared expression types for GPU kernel IRs.

Backend-agnostic expression AST shared by LoopIR and KernelIR.
Includes operator overloading for readable expression building.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Operator overloading mixin
# ---------------------------------------------------------------------------


class _ExprOps:
    """Mixin that adds arithmetic and comparison operators to Expr nodes.

    Returns BinOp nodes, enabling::

        Var("row") * Var("cols") + Var("j")   # → BinOp("+", BinOp("*", ...), ...)
        Var("i").lt(Var("n"))                   # → BinOp("<", ...)
    """

    def __add__(self, other: Expr) -> BinOp:
        return BinOp("+", self, _coerce(other))

    def __radd__(self, other: Expr) -> BinOp:
        return BinOp("+", _coerce(other), self)

    def __sub__(self, other: Expr) -> BinOp:
        return BinOp("-", self, _coerce(other))

    def __rsub__(self, other: Expr) -> BinOp:
        return BinOp("-", _coerce(other), self)

    def __mul__(self, other: Expr) -> BinOp:
        return BinOp("*", self, _coerce(other))

    def __rmul__(self, other: Expr) -> BinOp:
        return BinOp("*", _coerce(other), self)

    def __truediv__(self, other: Expr) -> BinOp:
        return BinOp("/", self, _coerce(other))

    def __mod__(self, other: Expr) -> BinOp:
        return BinOp("%", self, _coerce(other))

    def __neg__(self) -> BinOp:
        return BinOp("-", Literal(0, "int"), self)

    def lt(self, other: Expr) -> BinOp:
        """Less-than (``<``)."""
        return BinOp("<", self, _coerce(other))

    def ge(self, other: Expr) -> BinOp:
        """Greater-or-equal (``>=``)."""
        return BinOp(">=", self, _coerce(other))

    def eq(self, other: Expr) -> BinOp:
        """Equal (``==``)."""
        return BinOp("==", self, _coerce(other))

    def and_(self, other: Expr) -> BinOp:
        """Logical AND (``&&``)."""
        return BinOp("&&", self, _coerce(other))

    def or_(self, other: Expr) -> BinOp:
        """Logical OR (``||``)."""
        return BinOp("||", self, _coerce(other))


def _coerce(v: Expr | int | float) -> Expr:
    """Coerce Python int/float to Literal for operator overloading."""
    if isinstance(v, int):
        return Literal(v, "int")
    if isinstance(v, float):
        return Literal(v)
    return v


# ---------------------------------------------------------------------------
# Expression types
# ---------------------------------------------------------------------------


@dataclass
class Var(_ExprOps):
    """Variable reference."""

    name: str


@dataclass
class Literal(_ExprOps):
    """Numeric constant."""

    value: int | float
    dtype: str = "float"


@dataclass
class BinOp(_ExprOps):
    """Binary operation."""

    op: str  # "+", "-", "*", "/", "%", "<", ">", "<=", ">=", "==", "&&", "||"
    left: Expr
    right: Expr


@dataclass
class Builtin(_ExprOps):
    """GPU built-in variable (threadIdx.x, blockIdx.y, blockDim.x, etc.)."""

    name: str


@dataclass
class FuncCall(_ExprOps):
    """Intrinsic / math function call (C-level: expf, fmaxf, etc.)."""

    name: str
    args: list[Expr]


@dataclass
class Ternary(_ExprOps):
    """Ternary expression: cond ? if_true : if_false."""

    cond: Expr
    if_true: Expr
    if_false: Expr


Expr = Var | Literal | BinOp | Builtin | FuncCall | Ternary
