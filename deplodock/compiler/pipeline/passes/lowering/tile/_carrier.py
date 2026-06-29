"""Carrier builders for the tile-lowering passes — the spec-mode ``Monoid`` constructors that
``_flash`` / ``_softmax`` use. The carrier *algebra* (channel spec, inverse-ψ generator,
per-family stabilizer, stability certificate) lives one layer down in
``deplodock.compiler.ir.stmt.carrier``; this module only assembles ``Monoid``\\ s from it.
"""

from __future__ import annotations

from deplodock.compiler.ir.stmt import Monoid, State, Twist
from deplodock.compiler.ir.stmt.carrier import (  # noqa: F401 — re-exported for the passes/tests
    Channel,
    UnstableCarrierError,
    denom,
    exp_channels,
    expect,
    pivot,
)


def exp_family_twist(score: str, accumulators: list[Channel], state: tuple[str, ...]) -> Monoid:
    """The exp/LSE-family carrier ``Monoid`` over ``state`` (pivot first), as a name-free **spec**
    ``Twist`` — ``merge`` / ``combine_states`` / ``state_b`` are derived on demand from the
    channel spec. ``accumulators`` are the non-pivot channels (``denom()`` / ``expect(v)``)."""
    return Monoid(state=State(names=state), partial=(), twist=Twist(family="exp", channels=exp_channels(score, accumulators)))
