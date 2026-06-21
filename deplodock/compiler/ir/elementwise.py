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

from collections.abc import Callable
from typing import NamedTuple

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
    unary). Unknown names raise. ``commutative`` / ``associative`` /
    ``identity`` / ``has_identity`` are computed properties reading from
    class-level tables keyed by name — the algebraic traits reassociation
    gates (split-K, cooperative tree-combine) query instead of matching op
    names.
    """

    # Commutative ops — binary combines where ``op(a, b) == op(b, a)``.
    _COMMUTATIVE: frozenset[str] = frozenset({"add", "multiply", "maximum", "minimum", "amax", "sum", "prod"})
    # Associative ops — binary combines where ``op(op(a, b), c) == op(a, op(b, c))``.
    # The reassociable reduce combines: a reduction over one of these may be
    # split / reordered (split-K, cooperative tree-combine) without changing
    # the result. ``subtract`` / ``divide`` are deliberately absent.
    _ASSOCIATIVE: frozenset[str] = frozenset({"add", "multiply", "maximum", "minimum", "amax", "sum", "prod"})
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
    # Selecting ops *choose* an existing input value (max/min family) rather
    # than accumulate magnitude — so an Accum over one may stay in the input
    # dtype, and the flash recognizer's rowmax keys off it. The single source
    # for the per-op ``selecting`` trait (was a ``_SELECTING_OPS`` frozenset in
    # ``020_place_inits``).
    _SELECTING: frozenset[str] = frozenset({"maximum", "amax", "minimum", "max", "min"})
    # Semiring pairing — a reduce combine ``⊕`` mapped to the products ``⊗``
    # that distribute over it (``a·(b⊕c) == a·b ⊕ a·c``), so a contraction
    # ``Σ_k a⊗b`` is a matmul over ``⊕``. Only ``(+, ×)`` is exercised today;
    # the table is *data* so tropical ``(min, +)`` etc. is a one-line add when a
    # consumer exists — but DO NOT add unused semirings (simplicity-first).
    _SEMIRING: dict[str, frozenset[str]] = {"add": frozenset({"multiply"})}

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

    def __reduce__(self):
        # The resolved ``_fn`` is a numpy ufunc or one of the lambdas in
        # ``_NAME_TO_FN`` — neither pickles cleanly. Serialize the name
        # and re-resolve on unpickle by going through ``__init__``.
        return (self.__class__, (self.name,))

    @property
    def commutative(self) -> bool:
        return self.name in self._COMMUTATIVE

    @property
    def associative(self) -> bool:
        return self.name in self._ASSOCIATIVE

    @property
    def identity(self) -> float | None:
        return self._IDENTITY.get(self.name)

    @property
    def has_identity(self) -> bool:
        """True iff this op has a neutral element — i.e. it can seed an
        accumulator. The reassociation gates pair this with ``associative``
        / ``commutative`` to admit a reduce for split-K / tree-combine."""
        return self.identity is not None

    @property
    def selecting(self) -> bool:
        """True for ops that *select* an existing input value (the max/min
        family) instead of accumulating magnitude — an Accum over one may keep
        the input dtype rather than promote to the accumulating dtype."""
        return self.name in self._SELECTING

    @property
    def reduce_canon(self) -> str:
        """This op's canonical reduce-combine identity (``sum`` → ``add``,
        ``prod`` → ``multiply``, ``amax`` → ``maximum`` …); aliasless names map
        to themselves. The op-name-free key the reduce/scan render + numpy
        sites share."""
        return reduce_canon(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ElementwiseImpl) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"ElementwiseImpl({self.name!r})"


# ---------------------------------------------------------------------------
# Algebraic-role queries — the op-name-free helpers the partition planner,
# atom-cell matchers, and flash recognizer ask instead of string-matching
# ``"multiply"`` / ``"add"`` / ``"maximum"``.
# ---------------------------------------------------------------------------

# Reduce/scan op aliases → their canonical combine identity. The single map
# behind ``ElementwiseImpl.reduce_canon`` and the lift-reduce tensor→loop
# alias; an unlisted name canonicalizes to itself.
_REDUCE_CANON: dict[str, str] = {
    "add": "add",
    "sum": "add",
    "multiply": "multiply",
    "prod": "multiply",
    "maximum": "maximum",
    "amax": "maximum",
    "fmax": "maximum",
    "minimum": "minimum",
    "fmin": "minimum",
}


def reduce_canon(name: str) -> str:
    """Canonicalize a reduce/scan op name to its base combine identity
    (``sum`` → ``add`` …). Names without an alias map to themselves."""
    return _REDUCE_CANON.get(name, name)


def _op_name(op) -> str:
    return op.name if isinstance(op, ElementwiseImpl) else op


def distributes_over(product, reduce) -> bool:
    """True iff op ``product`` (``⊗``) distributes over the reduce combine
    ``reduce`` (``⊕``) — i.e. ``Σ_k a⊗b`` is a contraction/matmul over ``⊕``.
    Accepts op names or ``ElementwiseImpl``."""
    return _op_name(product) in ElementwiseImpl._SEMIRING.get(reduce_canon(_op_name(reduce)), frozenset())


def is_semiring_product(op) -> bool:
    """True iff ``op`` is a ``⊗`` in some semiring (today only ``multiply``) —
    the op-name-free 'is this a matmul / square product' query. Accepts an op
    name or ``ElementwiseImpl``."""
    name = _op_name(op)
    return any(name in prods for prods in ElementwiseImpl._SEMIRING.values())


# ---------------------------------------------------------------------------
# Reduce render-spelling registry — the single op-keyed table behind the four
# sites that used to switch on the reduce op name: ``Accum.render`` (CUDA
# ``+=`` / ``*=`` / ``fmax`` / ``fmin``), ``kernel/ir._binary_combine_expr``
# (the tree-combine binary expr), and ``ReduceOp.forward`` / ``ScanOp.forward``
# (the numpy interpreter reductions). Keyed by canonical combine name.
# ---------------------------------------------------------------------------


class ReduceSpelling(NamedTuple):
    infix: str | None  # binary-expr operator (``a + b``), or None for a call form
    compound: str | None  # compound assignment (``name += rhs``), or None
    intrinsic: str | None  # CUDA/abstract intrinsic (``fmax`` / ``fmin``), or None
    np_reduce: Callable  # numpy axis reduction (keepdims)
    np_scan: Callable | None  # numpy cumulative (scan), or None if scan is undefined


_REDUCE_SPELLING: dict[str, ReduceSpelling] = {
    "add": ReduceSpelling("+", "+=", None, np.sum, np.cumsum),
    "multiply": ReduceSpelling("*", "*=", None, np.prod, np.cumprod),
    "maximum": ReduceSpelling(None, None, "fmax", np.max, np.maximum.accumulate),
    "minimum": ReduceSpelling(None, None, "fmin", np.min, np.minimum.accumulate),
}


def reduce_spelling(op) -> ReduceSpelling:
    """The render data for a reduce combine, defaulting to additive (``+=``)
    for non-reduce / unknown ops — matches ``Accum.render``'s legacy fallback.
    Accepts an op name or ``ElementwiseImpl``."""
    return _REDUCE_SPELLING.get(reduce_canon(_op_name(op)), _REDUCE_SPELLING["add"])


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
