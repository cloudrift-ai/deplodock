"""Elementwise op metadata — named scalar operations with numpy backing.

Used as the ``op`` field on Tensor IR / Loop IR op classes
(``ElementwiseOp``, ``ReduceOp``, ``ScanOp``, ``Assign``, ``Accum``).
Carries the op's name (for codegen and serialization) plus its numpy
callable (for the interpreter backend) and — when meaningful — reducer
metadata (``commutative``, ``identity``).

Construction resolves the callable from the op's ``name`` via
``_NAME_TO_FN`` (for non-numpy intrinsics like ``rsqrt`` / ``relu``) or
``getattr(np, name)`` otherwise. Unknown names raise. Arity is read
from the callable's ufunc ``nin`` (non-ufunc intrinsics are all unary).

This module intentionally doesn't depend on the ``Expr`` AST in
``ir/expr.py`` — that's the coordinate / predicate sublanguage for
indices, a separate layer.
"""

from __future__ import annotations

import numpy as np

# Names whose callable isn't a plain ``getattr(np, name)`` — non-numpy
# intrinsics (all unary). Every other op name matches a numpy attribute,
# and ``__init__`` falls through to ``getattr(np, name)`` for them.
def _erf(x):  # numpy lacks an erf ufunc; scipy ships one and is a torch dep.
    from scipy.special import erf

    return erf(x)


_NAME_TO_FN: dict[str, object] = {
    "rsqrt": lambda x: 1.0 / np.sqrt(x),
    "relu": lambda x: np.maximum(0.0, x),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    "silu": lambda x: x / (1.0 + np.exp(-x)),
    "erf": _erf,
    "gelu": lambda x: 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0))),
    "gelu_tanh": lambda x: 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3))),
    "copy": lambda x: x,
}


class ElementwiseImpl:
    """Named scalar op — name + numpy callable + arity + reducer metadata.

    Construction resolves the callable from ``_NAME_TO_FN`` (non-numpy
    intrinsics) or ``getattr(np, name)`` for numpy-aligned names, and
    reads arity from the ufunc's ``nin`` (non-ufunc intrinsics are
    unary). Unknown names raise. ``commutative`` / ``identity`` are
    computed properties reading from class-level tables keyed by name.
    """

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
        self.name = name
        self._fn = fn
        self.arity = getattr(fn, "nin", 1)

    def __call__(self, *args):
        return self._fn(*args)

    @property
    def commutative(self) -> bool:
        return self.name in self._COMMUTATIVE

    @property
    def identity(self) -> float | None:
        return self._IDENTITY.get(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ElementwiseImpl) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"ElementwiseImpl({self.name!r})"
