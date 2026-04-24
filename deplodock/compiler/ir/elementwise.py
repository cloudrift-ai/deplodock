"""Elementwise op metadata — named scalar operations with numpy backing.

Used as the ``op`` field on Tensor IR / Loop IR op classes
(``ElementwiseOp``, ``ReduceOp``, ``ScanOp``, ``Assign``, ``Accum``).
Carries the op's name (for codegen and serialization) plus its numpy
callable (for the interpreter backend) and — when meaningful — reducer
metadata (``commutative``, ``identity``).

Construction resolves the callable from the op's ``name`` via
``_NAME_TO_FN`` (for non-numpy intrinsics like ``rsqrt`` / ``relu``) or
``getattr(np, name)`` otherwise. Unknown names raise. No registry, no
singletons — ``coerce_elementwise_impl(name)`` just tries
``BinaryElementwiseImpl`` then ``UnaryElementwiseImpl`` with numpy's
ufunc ``nin`` self-selecting the arity.

This module intentionally doesn't depend on the ``Expr`` AST in
``ir/expr.py`` — that's the coordinate / predicate sublanguage for
indices, a separate layer.
"""

from __future__ import annotations

import numpy as np

# Names whose callable isn't a plain ``getattr(np, name)`` — non-numpy
# intrinsics. Every other op name matches a numpy attribute, and
# ``__init__`` falls through to ``getattr(np, name)`` for them.
_NAME_TO_FN: dict[str, object] = {
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "relu": lambda x: np.maximum(0.0, x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "copy": lambda x: x,
}


class ElementwiseImpl:
    """Base class for named elementwise / combine ops.

    Subclasses (``UnaryElementwiseImpl`` / ``BinaryElementwiseImpl``) fix
    the ``arity``. Construction resolves the numpy callable from the op's
    ``name`` — either via ``_NAME_TO_FN`` (for non-numpy intrinsics) or
    ``getattr(np, name)`` for numpy-aligned names. Unknown names raise.
    ``commutative`` / ``identity`` are computed properties reading from
    class-level tables keyed by name.
    """

    arity: int = 1

    # Commutative ops — binary combines where ``op(a, b) == op(b, a)``.
    _COMMUTATIVE: frozenset[str] = frozenset({"add", "multiply", "maximum", "minimum", "amax", "sum", "prod"})
    # Reducer neutral elements — only meaningful when used as an Accum
    # combine or a ReduceOp.
    _IDENTITY: dict[str, float] = {
        "add": 0.0,
        "sum": 0.0,
        "multiply": 1.0,
        "prod": 1.0,
        "maximum": -1e30,
        "amax": -1e30,
        "minimum": 1e30,
    }

    def __init__(self, name: str) -> None:
        fn = _NAME_TO_FN.get(name)
        if fn is None:
            fn = getattr(np, name, None)
        if fn is None:
            raise ValueError(f"unknown elementwise op name: {name!r} (not in numpy or _NAME_TO_FN)")
        # Validate arity against numpy's ufunc metadata when available.
        nin = getattr(fn, "nin", None)
        if nin is not None and nin != self.arity:
            raise ValueError(f"arity mismatch for {name!r}: {type(self).__name__} expects {self.arity}, numpy callable has nin={nin}")
        self.name = name
        self.fn = fn

    @property
    def commutative(self) -> bool:
        return self.name in self._COMMUTATIVE

    @property
    def identity(self) -> float | None:
        return self._IDENTITY.get(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ElementwiseImpl) and type(self) is type(other) and self.name == other.name

    def __hash__(self) -> int:
        return hash((type(self), self.name))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.name!r})"


class UnaryElementwiseImpl(ElementwiseImpl):
    arity = 1


class BinaryElementwiseImpl(ElementwiseImpl):
    arity = 2


def coerce_elementwise_impl(v: str | ElementwiseImpl) -> ElementwiseImpl:
    """Accept a string name or ``ElementwiseImpl`` instance; return an impl.

    Tries ``BinaryElementwiseImpl(name)`` first, falls back to
    ``UnaryElementwiseImpl(name)``. numpy's ufunc ``nin`` metadata makes
    the attempt self-selecting — binary ops like ``add`` (nin=2) pass
    the binary check, unary ops like ``exp`` (nin=1) raise in the binary
    constructor and land in the unary one.
    """
    if isinstance(v, ElementwiseImpl):
        return v
    try:
        return BinaryElementwiseImpl(v)
    except ValueError:
        pass
    return UnaryElementwiseImpl(v)
