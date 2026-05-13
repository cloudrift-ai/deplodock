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


# ---------------------------------------------------------------------------
# Op clustering — used by ``Body.structural_key()`` (opt-in) to collapse
# ops that share a GPU functional unit so two kernels that differ only by
# the *kind* of cheap-FMA op (or expensive-SFU op) at the same position
# hash equal. The cluster representative is just one ``ElementwiseImpl``
# name per group — the choice is arbitrary, only equality matters.
# ---------------------------------------------------------------------------


# Maps each known op name → its cluster representative.
#
# Clusters are picked by the GPU compute unit that issues the op:
#
# - **fma** (rep ``add``) — cheap ALU (~1-2 cycle): add / sub / multiply /
#   negative / abs / fma. ``sum`` and ``prod`` are reduce aliases of
#   add / multiply and land here too.
# - **compare** (rep ``maximum``) — predicate / select ALU: min / max /
#   amax / sign.
# - **sfu_div** (rep ``divide``) — integer / float division SFU path:
#   divide / true_divide / floor_divide / remainder / mod / reciprocal.
# - **sfu_trans** (rep ``exp``) — MUFU transcendental path (~10-30x cycle
#   cost): sqrt / rsqrt / exp / log / sin / cos / tanh / sigmoid /
#   silu / erf / gelu* / pow / relu. (relu joins the SFU bucket only
#   because composite activations live here and a position that *might*
#   carry one of them dominates the perf signal; bucketing the cheap
#   max(0, x) implementation alongside doesn't lose meaningful
#   information for the search.)
# - **copy** (rep ``copy``) — passthrough; its own bucket so a no-op
#   doesn't get collapsed with arithmetic.
_OP_CLUSTERS: dict[str, str] = {
    # fma
    "add": "add",
    "sum": "add",
    "subtract": "add",
    "sub": "add",
    "negative": "add",
    "multiply": "add",
    "prod": "add",
    "abs": "add",
    "fma": "add",
    # compare
    "maximum": "maximum",
    "minimum": "maximum",
    "amax": "maximum",
    "sign": "maximum",
    # sfu_div
    "divide": "divide",
    "true_divide": "divide",
    "floor_divide": "divide",
    "remainder": "divide",
    "mod": "divide",
    "reciprocal": "divide",
    # sfu_trans
    "sqrt": "exp",
    "rsqrt": "exp",
    "exp": "exp",
    "log": "exp",
    "log2": "exp",
    "log10": "exp",
    "sin": "exp",
    "cos": "exp",
    "tan": "exp",
    "tanh": "exp",
    "sigmoid": "exp",
    "silu": "exp",
    "erf": "exp",
    "gelu": "exp",
    "gelu_tanh": "exp",
    "pow": "exp",
    "relu": "exp",
    # passthrough
    "copy": "copy",
}


def cluster_representative(op: ElementwiseImpl) -> ElementwiseImpl:
    """Return the canonical op for ``op``'s compute-unit cluster.

    Unknown op names pass through unchanged — the mapping is best-effort
    and a missing entry is treated as "stand-alone cluster". Callers
    that want to detect coverage gaps can ``assert op.name in
    _OP_CLUSTERS``."""
    rep = _OP_CLUSTERS.get(op.name)
    if rep is None or rep == op.name:
        return op
    return ElementwiseImpl(rep)
