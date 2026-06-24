"""Knob-pin validation — refuse a force-pinned env knob foreign to the op's tier.

A ``DEPLODOCK_<KNOB>`` env pin is global, but every kernel is lowered on exactly
ONE *tier* (the codegen regime its body + pins resolve to): a pointwise ``MAP``, a
scalar ``SEMIRING`` reduce, a tensor-core ``WARP`` matmul, a cooperative ``MONOID``
reduce, or a streaming ``TWISTED_MONOID`` flash. Each tier owns a disjoint slice of
the knob schema (a warp tile has no ``BN``/``BM`` THREAD width; a cooperative reduce
has no ``WM``/``WN`` warp count; a pointwise nest has no ``BK`` K-chunk). Pinning a
knob the resolved tier never reads used to be **silently dropped** (or overwritten by
an OFF sentinel) — the user pinned a config and got a different one with no warning.

This module makes that a **hard error** (strict per-op policy): given the op's
:class:`~deplodock.compiler.ir.algebra.AlgebraKind` and the live env pins, it computes
the set of tiers that could satisfy *every* force-pinned knob and raises
:class:`KnobPinError` when that set is empty. It is called once, deterministically, at
the enumeration seed (``000_build``), so a contradictory pin fails before any tile
search rather than disappearing into a quietly-wrong kernel.

The legality table is value-aware: a knob at its universal / OFF value (``SPLITK=1``,
``FK=1``, ``BR=1``, ``FM=1``, ``STAGE=`` empty, ``TMA=0``) constrains nothing — only a
*meaningful* pin (``SPLITK>1`` cross-CTA split-K, ``BR>1`` cooperative lanes,
``MMA=<kind>`` tensor core, ``STAGE`` selecting a read-site, ``TMA=1``) narrows the
tier. The staging / transport knobs ``STAGE`` / ``TMA`` are legal only on the **staged**
tiers (``SCALAR`` / ``WARP``): a cooperative ``MONOID`` reduce and a streaming
``TWISTED_MONOID`` flash are smem-free, and a pointwise ``MAP`` nest has no K-tower, so
neither stages anything. (Scalar-tile ``TMA`` transport is not yet wired — see
``052_transport`` — but it is not *senseless*, so it stays allowed rather than refused;
the gap is tracked by an xfail in ``test_knob_pinning.py``.)
"""

from __future__ import annotations

from enum import Enum

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.pipeline.knob import mma_decode
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    BK,
    BM,
    BN,
    BR,
    FK,
    FM,
    FN,
    MMA,
    SPLITK,
    STAGE,
    TMA,
    WM,
    WN,
)


class KnobPinError(ValueError):
    """A force-pinned ``DEPLODOCK_<KNOB>`` is foreign to the tier the op lowers on
    (no codegen regime reads it for this op's algebra). Subclasses ``ValueError`` so
    callers already catching config errors keep working."""


class Tier(Enum):
    """The codegen regime a kernel materializes on — each owns a disjoint knob slice."""

    MAP = "map"  # pointwise functor (free-axis thread + register tile only)
    SCALAR = "scalar"  # scalar SEMIRING reduce / matmul (thread tile + K-chunk + split-K)
    WARP = "warp"  # tensor-core MMA matmul (warp counts + register cells + atom-K chunk)
    COOP = "coop"  # cooperative MONOID reduce (free thread + BR lanes + K-chunk, reg forced 1)
    STREAMING = "streaming"  # streaming TWISTED_MONOID flash (free thread + BR lanes)


_NON_WARP = frozenset({Tier.MAP, Tier.SCALAR, Tier.COOP, Tier.STREAMING})
_STAGED = frozenset({Tier.SCALAR, Tier.WARP})  # the tiers that synthesize an smem slab

# The tiers an op's algebra can resolve to (before any pin narrows further). SEMIRING
# is the only fork — scalar register-tile FMA vs the tensor-core warp atom.
_CANDIDATES: dict[AlgebraKind, frozenset[Tier]] = {
    AlgebraKind.MAP: frozenset({Tier.MAP}),
    AlgebraKind.SEMIRING: frozenset({Tier.SCALAR, Tier.WARP}),
    AlgebraKind.MONOID: frozenset({Tier.COOP}),
    AlgebraKind.TWISTED_MONOID: frozenset({Tier.STREAMING}),
}


def _legal_tiers(name: str, raw: str) -> frozenset[Tier] | None:
    """The tiers in which a force-pin of knob ``name`` to ``raw`` is meaningful, or
    ``None`` when the pin is the universal / OFF value (constrains nothing). A return
    of ``frozenset()`` means "no tier reads this" (an unconditional foreign pin)."""
    if name == MMA.name:
        enabled, kind = mma_decode(raw)
        if kind is not None:
            return frozenset({Tier.WARP})  # a pinned atom kind forces the warp tier
        if not enabled:
            return _NON_WARP  # MMA=0 — scalar/coop/streaming/map, never warp
        return None  # truthy / auto — no constraint
    if name in (WM.name, WN.name):
        return frozenset({Tier.WARP}) if _as_int(raw) >= 1 else None
    if name in (BN.name, BM.name):
        return _NON_WARP if _as_int(raw) >= 1 else None  # THREAD width — every tier but warp
    if name in (FM.name, FN.name):
        # Register cells: coop / streaming force the free-axis register tile to 1, so a
        # cell pin > 1 is foreign to them; = 1 is the universal value.
        return frozenset({Tier.MAP, Tier.SCALAR, Tier.WARP}) if _as_int(raw) > 1 else None
    if name == BK.name:
        # K-chunk: the reduce tiers + warp. A pointwise MAP has no K; streaming forces BK=1.
        return frozenset({Tier.SCALAR, Tier.COOP, Tier.WARP}) if _as_int(raw) > 1 else None
    if name == FK.name:
        # Strip-mine accumulators: the scalar / cooperative reduce only.
        return frozenset({Tier.SCALAR, Tier.COOP}) if _as_int(raw) > 1 else None
    if name == SPLITK.name:
        # Cross-CTA split-K: the scalar reduce AND the warp tier (the atomic-free
        # split-K combine, ``055_atomic_free_splitk``). A non-linear epilogue downgrades
        # it to 1 at the shape level — that is the pipeline's job, not a pin contradiction.
        return frozenset({Tier.SCALAR, Tier.WARP}) if _as_int(raw) > 1 else None
    if name == BR.name:
        return frozenset({Tier.COOP, Tier.STREAMING}) if _as_int(raw) > 1 else None  # cooperative lanes
    if name == STAGE.name:
        # Staging a read-site: only the scalar reduce / warp matmul stage (COOP / STREAMING
        # are smem-free, MAP has no K-tower). A mask with no selected bit is inert.
        return _STAGED if "1" in raw else None
    if name == TMA.name:
        # TMA transport rides the staging schedule, so it shares STAGE's staged tiers. Only
        # WARP is wired today (052_transport); scalar-tile TMA is unimplemented, not senseless.
        return _STAGED if raw.strip().lower() in {"1", "true", "yes", "on"} else None
    return None


def _as_int(raw: str) -> int:
    try:
        return int(raw, 0)
    except (TypeError, ValueError):
        return 0


# The knobs whose env pin this validator audits — tile geometry + the staging /
# transport schedule (STAGE / TMA), each legal only on the tiers in ``_legal_tiers``.
_AUDITED = (MMA, WM, WN, BN, BM, FM, FN, BK, FK, SPLITK, BR, STAGE, TMA)


def validate_pins(algebra: AlgebraKind) -> None:
    """Raise :class:`KnobPinError` when the live ``DEPLODOCK_<KNOB>`` pins cannot all be
    satisfied by any tier an op of ``algebra`` resolves to. No-op when no geometry knob
    is pinned, or every pin is a universal / OFF value, or the pins agree on a tier."""
    candidates = _CANDIDATES.get(algebra)
    if candidates is None:
        return
    feasible = candidates
    pinned: list[tuple[str, str, frozenset[Tier]]] = []
    for knob in _AUDITED:
        raw = knob.raw()
        if raw is None or raw == "":
            continue
        tiers = _legal_tiers(knob.name, raw)
        if tiers is None:
            continue
        pinned.append((knob.name, raw, tiers))
        next_feasible = feasible & tiers
        if not next_feasible:
            raise KnobPinError(_message(algebra, candidates, pinned))
        feasible = next_feasible


def _message(algebra: AlgebraKind, candidates: frozenset[Tier], pinned: list[tuple[str, str, frozenset[Tier]]]) -> str:
    cand = ", ".join(sorted(t.value for t in candidates))
    parts = [f"{name}={raw} (needs {', '.join(sorted(t.value for t in tiers)) or 'no tier'})" for name, raw, tiers in pinned]
    return (
        f"contradictory knob pins for a {algebra.value} kernel (tiers it can lower on: {cand}): "
        + "; ".join(parts)
        + " — no single tier serves every pin. Drop the tier-foreign pin(s) (e.g. BN/BM/BR/FK on a "
        "warp MMA kernel, or WM/WN on a scalar kernel)."
    )
