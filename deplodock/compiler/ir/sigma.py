"""Sigma — Expr-substitution helper used across all IR layers.

Wraps the bare ``dict[str, Expr]`` substitution form so call sites can use
``.apply(e)`` / ``.extend(name, expr)`` / ``.restrict(names)`` instead of
open-coding ``substitute`` and key flattening. Used by ``Stmt.rewrite``
(every IR layer) and the Loop-IR fusion splicer.

Lives at the top of ``ir/`` rather than under any one IR package because
``Stmt.rewrite`` is shared infrastructure — same reasoning as
``ir/axis.py`` and ``ir/stmt.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.expr import Expr, render, substitute


@dataclass(frozen=True, eq=False)
class Sigma:
    """Axis substitution: axis name → replacement ``Expr``.

    Carried by the fusion splicer as it walks from the consumer into the
    producer's expression chain — each producer axis gets rewritten into
    the consumer's namespace.

    Equality and hashing are by canonical form — sorted ``(name,
    render(expr))`` pairs — so two Sigmas built from different dicts that
    denote the same substitution compare equal and share a hash bucket.
    """

    mapping: dict[str, Expr] = field(default_factory=dict)

    def __post_init__(self) -> None:
        key = tuple(sorted((k, render(v)) for k, v in self.mapping.items()))
        object.__setattr__(self, "_key", key)

    def apply(self, e: Expr) -> Expr:
        return substitute(e, self.mapping)

    def extend(self, name: str, expr: Expr) -> Sigma:
        return Sigma({**self.mapping, name: expr})

    def restrict(self, names: set[str]) -> Sigma:
        """Return a new Sigma keeping only bindings whose axis name is in ``names``."""
        return Sigma({k: v for k, v in self.mapping.items() if k in names})

    def get(self, name: str) -> Expr | None:
        return self.mapping.get(name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Sigma):
            return NotImplemented
        return self._key == other._key  # type: ignore[attr-defined]

    def __hash__(self) -> int:
        return hash(self._key)  # type: ignore[attr-defined]


Sigma.IDENTITY = Sigma({})

__all__ = ["Sigma"]
