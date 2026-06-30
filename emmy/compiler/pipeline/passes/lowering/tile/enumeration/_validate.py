"""Knob-pin validation — refuse a force-pinned env knob foreign to the op's tier.

A ``DEPLODOCK_<KNOB>`` env pin is global, but every kernel is lowered on exactly
ONE *tier* (the codegen regime its body + pins resolve to): a pointwise ``MAP``, a
scalar ``SEMIRING`` reduce, a tensor-core ``WARP`` matmul, or a ``MONOID`` reduce (the
flat cooperative reduce **and** the streaming flash — one tier, since both share the
``build_monoid`` move and knob slice; the streaming schedule is a derived structural
property, not a separate tier). Each tier owns a disjoint slice of the knob schema (a
warp tile has no ``BN``/``BM`` THREAD width; a ``MONOID`` reduce has no ``WM``/``WN``
warp count; a pointwise nest has no ``BK`` K-chunk). Pinning a knob the resolved tier
never reads used to be **silently dropped** (or overwritten by an OFF sentinel) — the
user pinned a config and got a different one with no warning.

This module makes that a **hard error** (strict per-op policy): given the op's
:class:`~deplodock.compiler.ir.algebra.AlgebraKind` and the live env pins, it computes
the set of tiers that could satisfy *every* force-pinned knob and raises
:class:`KnobPinError` when that set is empty. It is called once, deterministically, at
the enumeration seed (``010_build``), so a contradictory pin fails before any tile
search rather than disappearing into a quietly-wrong kernel.

The legality table is value-aware: a knob at its universal / OFF value (``SPLITK=1``,
``FK=1``, ``BR=1``, ``FM=1``, ``STAGE=`` empty, ``TMA=0``) constrains nothing — only a
*meaningful* pin (``SPLITK>1`` cross-CTA split-K, ``BR>1`` cooperative lanes,
``MMA=<kind>`` tensor core, ``STAGE`` selecting a read-site, ``TMA=1``) narrows the
tier. The K-chunk knobs ``BK``/``FK`` are legal on the ``MONOID`` tier (split-KV / serial
re-bracketing is associativity-licensed on the nested monoid too — the streaming flash
re-brackets its KV stream the same way the cooperative reduce chunks its K). The staging
/ transport knobs ``STAGE`` / ``TMA`` are legal only on the **staged** tiers (``SCALAR`` /
``WARP``): a ``MONOID`` reduce is smem-free and a pointwise ``MAP`` nest has no K-tower,
so neither stages anything. ``TMA`` is wired on both staged tiers — the warp-tier
``ldmatrix`` matmul and the scalar register-tiled SGEMM (``130_transport`` promotes any
staged matmul with a ringable K loop).
"""

from __future__ import annotations

from enum import Enum

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.pipeline.knob import mma_decode
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import (
    CHAIN,
    MMA,
    STAGE,
    TMA,
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
    MONOID = "monoid"  # MONOID reduce — flat cooperative AND streaming flash (free thread + BR lanes + K-chunk, reg 1)


_NON_WARP = frozenset({Tier.MAP, Tier.SCALAR, Tier.MONOID})
_STAGED = frozenset({Tier.SCALAR, Tier.WARP})  # the tiers that synthesize an smem slab

# The tiers an op's algebra can resolve to (before any pin narrows further). SEMIRING
# is the only fork — scalar register-tile FMA vs the tensor-core warp atom. MONOID is one
# tier: the cooperative reduce and the streaming flash share the same ``build_monoid`` move
# and knob slice (a twisted monoid is a monoid — the streaming schedule is derived, not a
# separate tier), so the K-chunk knobs (``BK``/``FK``) are legal on it (split-KV / serial
# re-bracketing is associativity-licensed on the nested monoid too).
_CANDIDATES: dict[AlgebraKind, frozenset[Tier]] = {
    AlgebraKind.MAP: frozenset({Tier.MAP}),
    AlgebraKind.SEMIRING: frozenset({Tier.SCALAR, Tier.WARP}),
    AlgebraKind.MONOID: frozenset({Tier.MONOID}),
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
    if name == STAGE.name:
        # Staging a read-site: only the scalar reduce / warp matmul stage (COOP / STREAMING
        # are smem-free, MAP has no K-tower). A mask with no selected bit is inert.
        return _STAGED if "1" in raw else None
    if name == TMA.name:
        # TMA transport rides the staging schedule, so it shares STAGE's staged tiers
        # (scalar SGEMM + warp matmul — both wired in 130_transport).
        return _STAGED if raw.strip().lower() in {"1", "true", "yes", "on"} else None
    if name == CHAIN.name:
        # The FA-2 shared-score restructuring is a MONOID(SEMIRING) streaming-flash move.
        return frozenset({Tier.MONOID}) if raw.strip().lower() in {"1", "true", "yes", "on"} else None
    return None


def _as_int(raw: str) -> int:
    try:
        return int(raw, 0)
    except (TypeError, ValueError):
        return 0


# The knobs whose env pin this validator still audits — the atom-tier selector (MMA) and
# the staging / transport schedule (STAGE / TMA / CHAIN). The geometry families
# (``SPLIT@<axis>`` / ``REDUCE@<axis>``) are NO LONGER policed by tier-foreignness — a
# ``SPLIT`` value carries no tier (it is the cell's ``ATOM``), and the reduce levers'
# legality is the carrier-trait value-domain gate. This is the plan's step-3 retirement.
_AUDITED = (MMA, STAGE, TMA, CHAIN)


def validate_pins(algebra: AlgebraKind) -> None:
    """Raise :class:`KnobPinError` when the live ``DEPLODOCK_<KNOB>`` pins cannot all be
    satisfied by any tier an op of ``algebra`` resolves to. No-op when no geometry knob
    is pinned, or every pin is a universal / OFF value, or the pins agree on a tier.

    A ``MONOID`` nest — the flat cooperative reduce **and** the streaming flash — resolves
    to the single ``MONOID`` tier (the streaming schedule is a derived structural property,
    not a separate tier; both share ``build_monoid``), so ``BK``/``FK``/``BR`` are legal on
    it. Whether a particular shape realizes a pinned ``BK`` is the pipeline's job (like a
    non-linear matmul downgrading ``SPLITK``), not a pin contradiction."""
    candidates = _CANDIDATES.get(algebra)
    if candidates is None:
        return
    feasible = candidates
    pinned: list[tuple[str, str, frozenset[Tier]]] = []
    for knob in _AUDITED:
        # The atom selector is the native ``ATOM@<cell>`` family now — audit its pin
        # (``DEPLODOCK_ATOM`` / legacy ``DEPLODOCK_MMA`` via ingest); the message keeps the
        # ``MMA`` label for continuity.
        raw = fam.atom_raw(fam.MATMUL_CELL) if knob is MMA else knob.raw()
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
