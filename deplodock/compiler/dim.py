"""``Dim`` — one tensor / axis extent, backed by an ``Expr``.

Every Dim carries a shape expression:

- ``Dim(32)`` wraps ``Literal(32, "int")`` — a static extent.
- ``Dim("seq_len")`` wraps ``Var("seq_len")`` — an atomic symbolic extent.
- ``Dim(seq) * Dim(2)`` wraps ``BinaryExpr("*", Var("seq"), Literal(2))`` —
  a composite symbolic extent. Operators on Dim dispatch to ``Expr`` and
  eagerly fold via ``Expr.simplify``: ``Dim(32) * Dim(64) → Dim(Literal(2048))``,
  ``Dim("s") * Dim(1) → Dim(Var("s"))``.

Reads:

- ``d.expr`` always works — returns the underlying ``Expr``.
- ``d.as_static()`` returns the int for ``Literal``-backed dims, raises otherwise.
- ``d.value`` returns the int (``Literal``) or the name (``Var``); raises on
  composite. Kept for back-compat with existing call sites that key on the
  symbolic name string (e.g. launch-time ``sym_env`` dicts).
- ``d == 32`` / ``d == "seq_len"`` / ``d == Dim(...)`` all work — the first
  two unwrap to ``Literal.value`` / ``Var.name``; composite dims only compare
  structurally to other Dims.

No ``__int__`` / ``__index__``: ``int(d)`` and ``range(d)`` deliberately fail.
Sites that need a static int must say so via ``.as_static()`` — that way
introducing a symbolic dim later breaks loudly at the sites that can't yet
handle it, rather than silently casting through.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.expr import BinaryExpr, Expr, Interval, Literal, SimplifyCtx, Var


def _coerce_expr(value: int | str | Expr | Dim) -> Expr:
    """Coerce a Dim constructor argument to an ``Expr``.

    - bare ``int`` → ``Literal(value, "int")``
    - bare ``str`` → ``Var(value)``
    - ``Expr``     → passed through
    - ``Dim``      → unwrapped to its ``expr``
    """
    if isinstance(value, Dim):
        return value.expr
    if isinstance(value, int):
        return Literal(value, "int")
    if isinstance(value, str):
        return Var(value)
    # Expr is a Union — check by membership in the known node classes
    # via duck-typing on the AST API (``eval`` is on every concrete Expr).
    if hasattr(value, "eval") and hasattr(value, "substitute"):
        return value
    raise TypeError(f"Dim: cannot wrap {type(value).__name__}: {value!r}")


def _simplify(expr: Expr) -> Expr:
    # Shape vars (every free ``Var`` in a Dim expression) are positive by
    # definition: a tensor extent can't be zero or negative. Expose that
    # via ``SimplifyCtx.ranges`` so the simplifier's ``//`` cancellation
    # path can safely cancel matching factors (otherwise ``(s*128) // (s*4)``
    # has no way to reduce to ``32``).
    ranges = {name: Interval(1, 1 << 30) for name in expr.free_vars()}
    return expr.simplify(SimplifyCtx(ranges))


@dataclass(frozen=True, init=False, eq=False)
class Dim:
    expr: Expr

    def __init__(self, value: int | str | Expr | Dim) -> None:
        # ``frozen=True`` blocks normal assignment; route through object.__setattr__.
        object.__setattr__(self, "expr", _coerce_expr(value))

    # ---- inspection ------------------------------------------------------

    @property
    def is_static(self) -> bool:
        return isinstance(self.expr, Literal) and isinstance(self.expr.value, int)

    def as_static(self) -> int:
        if not (isinstance(self.expr, Literal) and isinstance(self.expr.value, int)):
            raise TypeError(f"Dim({self.expr!r}) is not a static int — cannot resolve to int statically")
        return self.expr.value

    @property
    def value(self) -> int | str:
        """Back-compat shim: ``Literal.value`` for static, ``Var.name`` for atomic
        symbolic. Raises on composite. Prefer ``.expr`` at new call sites."""
        if isinstance(self.expr, Literal) and isinstance(self.expr.value, int):
            return self.expr.value
        if isinstance(self.expr, Var):
            return self.expr.name
        raise TypeError(f"Dim({self.expr!r}).value: composite dim has no scalar value; use .expr")

    # ---- arithmetic — eager-fold via Expr.simplify -----------------------

    def __add__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("+", self.expr, _coerce_expr(other))))

    def __radd__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("+", _coerce_expr(other), self.expr)))

    def __sub__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("-", self.expr, _coerce_expr(other))))

    def __rsub__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("-", _coerce_expr(other), self.expr)))

    def __mul__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("*", self.expr, _coerce_expr(other))))

    def __rmul__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("*", _coerce_expr(other), self.expr)))

    def __floordiv__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("//", self.expr, _coerce_expr(other))))

    def __rfloordiv__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("//", _coerce_expr(other), self.expr)))

    def __mod__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("%", self.expr, _coerce_expr(other))))

    def __rmod__(self, other: int | Dim) -> Dim:
        return Dim(_simplify(BinaryExpr("%", _coerce_expr(other), self.expr)))

    # ---- equality + hashing ---------------------------------------------

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Dim):
            return self.expr == other.expr
        # Back-compat ergonomics: ``Dim(32) == 32`` and ``Dim("s") == "s"``
        # keep working so test assertions and existing pass code don't churn.
        # Composite dims (``BinaryExpr``-backed) never compare equal to bare
        # int/str — they only compare structurally to other Dims.
        if isinstance(other, bool):
            return NotImplemented
        if isinstance(other, int):
            return isinstance(self.expr, Literal) and self.expr.value == other
        if isinstance(other, str):
            return isinstance(self.expr, Var) and self.expr.name == other
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        eq = self.__eq__(other)
        if eq is NotImplemented:
            return NotImplemented
        return not eq

    def __hash__(self) -> int:
        # Hash on the canonical Expr so Dim(32) and Dim(Literal(32, "int"))
        # hash identically. Atomic Vars and Literals are themselves frozen
        # and hashable; composite BinaryExpr is hashable post-M0.
        return hash(self.expr)

    def __repr__(self) -> str:
        if isinstance(self.expr, Literal) and isinstance(self.expr.value, int):
            return f"Dim({self.expr.value!r})"
        if isinstance(self.expr, Var):
            return f"Dim({self.expr.name!r})"
        return f"Dim({self.expr!r})"

    def __str__(self) -> str:
        # Used by f-strings / ``str()`` — render transparently so pretty IR
        # output (``for i in 0..32``) and ``Body.structural_key`` digests
        # don't change shape just because the type wrapper exists.
        if isinstance(self.expr, Literal) and isinstance(self.expr.value, int):
            return str(self.expr.value)
        if isinstance(self.expr, Var):
            return self.expr.name
        return self.expr.pretty()


def to_dim(value: int | str | Dim | Expr) -> Dim:
    """Coerce ``int`` / ``str`` / ``Expr`` to ``Dim``; pass ``Dim`` through unchanged."""
    return value if isinstance(value, Dim) else Dim(value)


def as_static(d: object) -> int:
    """Coerce a shape element (``Dim``, ``int``, or ``str``) to a static int.

    Use at boundaries where the upstream type isn't yet uniformly ``Dim`` —
    e.g. ``infer_output_shape`` returns that mix ints/strings from constructor
    args with ``Dim`` from another Tensor's shape.

    Raises if ``d`` is symbolic (``Dim('seq_len')`` or ``str``)."""
    if isinstance(d, Dim):
        return d.as_static()
    if isinstance(d, int):
        return d
    raise TypeError(f"as_static: cannot convert {d!r} to int (symbolic shape element?)")


__all__ = ["Dim", "to_dim", "as_static"]
