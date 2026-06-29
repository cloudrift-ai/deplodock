"""Warp-specialization **roles** — the worker bands a CTA's warps split into.

A :class:`RoleKind` is one warp-specialized worker role (producer / sfu / ...): its ``WSPEC`` codec
token, its per-role param schema, and a legality predicate over the (uniform) schedule it would
specialize. The COMPUTE / mma-consumer role is **implicit** — it is sized by ``WarpTile.warps`` and
never appears in the ``WSPEC`` codec, so the worker split never restates the tile (the orthogonality
litmus). Every other role is a band split off the fixed pipeline and spelled by its token in the
``WSPEC`` codec (``<token><np>[:<param>,...]`` — :class:`~deplodock.compiler.ir.tile.schedule.WarpSpec`).

Adding a role is one registration here (plus, eventually, its emission in ``010_materialize``); the
codec engine and ``WarpSpec`` pick up the new token for free. Mirrors :mod:`atom` — a small registry
the ``ir/tile`` layer imports; kept dependency-light so ``schedule.py`` can build the ``WSPEC``
schema from it without a cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from deplodock.compiler.ir.tile.codec import Field, FieldKind


def _always(_sched: object) -> bool:
    return True


def _has_stage(sched: object) -> bool:
    """The producer band is only meaningful when the pipeline actually stages operands."""
    return getattr(sched, "stage", None) is not None


@dataclass(frozen=True)
class RoleKind:
    """One warp-specialized worker role.

    ``token`` is the ``WSPEC`` codec letter (``p`` producer, ``s`` sfu, ...); ``params`` is the
    per-role param schema (extra :class:`~deplodock.compiler.ir.tile.codec.Field`\\s after the
    warp-count value — e.g. the producer's in-flight op window ``q``); ``legal`` decides whether
    the role is meaningful for a given uniform schedule (the producer needs a ``stage`` to drive).
    Frozen + hashable so it rides on a frozen ``WarpSpec``."""

    token: str
    help: str = ""
    params: tuple[Field, ...] = ()
    legal: Callable[[object], bool] = field(default=_always)


#: The registered warp-spec roles, keyed by the ``WSPEC`` codec token. COMPUTE (the mma consumer)
#: is implicit — sized by ``WarpTile.warps``, never registered. PRODUCER drives the ``Stage`` load
#: half; SFU is a stub example of the role-extension path (a transcendental epilogue band).
ROLE_REGISTRY: dict[str, RoleKind] = {
    "p": RoleKind(
        "p",
        help="producer warps — drive the Stage gmem→smem load half",
        params=(Field("q", FieldKind.TUPLE),),  # in-flight op window (producer-local; not STAGE.depth)
        legal=_has_stage,
    ),
    "s": RoleKind("s", help="sfu / transcendental combine warps — reserved example role"),
}


def role_for(token: str) -> RoleKind:
    """The registered :class:`RoleKind` for ``token`` (a ``WSPEC`` codec role token)."""
    try:
        return ROLE_REGISTRY[token]
    except KeyError:
        raise ValueError(f"unknown warp-spec role {token!r} (have {sorted(ROLE_REGISTRY)})") from None


__all__ = ["ROLE_REGISTRY", "RoleKind", "role_for"]
