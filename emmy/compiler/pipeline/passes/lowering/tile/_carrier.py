"""Carrier builders for the tile-lowering passes — the spec-mode :class:`Carrier` constructors that
``_flash`` / ``_softmax`` use. The carrier *algebra* (channel spec, inverse-ψ generator,
per-family stabilizer, stability certificate) lives one layer down in
``emmy.compiler.ir.stmt.carrier``; this module only assembles :class:`Carrier`\\ s from it.
"""

from __future__ import annotations

from emmy.compiler.ir.stmt import Carrier, State, Twist
from emmy.compiler.ir.stmt.carrier import (  # noqa: F401 — re-exported for the passes/tests
    Channel,
    UnstableCarrierError,
    denom,
    exp_channels,
    expect,
    pivot,
)


def exp_family_twist(score: str, accumulators: list[Channel], state: tuple[str, ...]) -> Carrier:
    """The exp/LSE-family :class:`Carrier` over ``state`` (pivot first), as a name-free **spec**
    ``Twist`` — ``merge`` / ``combine_states`` / ``state_b`` are derived on demand from the
    channel spec. ``accumulators`` are the non-pivot channels (``denom()`` / ``expect(v)``)."""
    return Carrier(state=State(names=state), twist=Twist(family="exp", channels=exp_channels(score, accumulators)))
