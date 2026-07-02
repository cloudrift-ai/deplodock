"""Carrier builders for the tile-lowering passes — the spec-mode :class:`Carrier` constructors that
``_flash`` / ``_softmax`` use. The carrier *algebra* (channel spec, inverse-ψ generator,
per-family stabilizer, stability certificate) lives one layer down in
``emmy.compiler.ir.stmt.carrier``; this module only assembles :class:`Carrier`\\ s from it.
"""

from __future__ import annotations

from emmy.compiler.ir.stmt import Assign, Carrier, Load, State, Twist, Write
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


def projection_distributes(body, states: tuple[str, ...]) -> bool:
    """True if the kernel's projection epilogue is a **linear-homogeneous** map of the carrier
    state(s) — i.e. it distributes over the atomic-add combine, so applying it to each CTA's
    partition before the ``atomicAdd`` equals applying it once after the cross-CTA sum
    (``Σ c·xₛ = c·(Σ xₛ)``). A bare state write (``proj = id``) trivially distributes; a constant
    *scale* — ``mean``'s ``×1/N`` — does; an additive offset (a fused bias), a nonlinear unary
    (``relu`` / ``reciprocal`` of the *state*), or a product of two state-derived values do NOT.

    Conservative forward dataflow: ``linear`` is the set of SSA names that are a
    linear-homogeneous function of the state. A value is grown into it only by ``multiply`` with
    a state-independent operand (an arg not itself in ``linear``); any other op that consumes a
    ``linear`` value — or any projection stmt we can't reason about — refuses. The final ``Write``
    must store only ``linear`` values."""
    linear = set(states)
    for s in body:
        if isinstance(s, Write):
            return all(v in linear for v in s.values)
        if isinstance(s, Load):
            continue  # reads memory (the count / a per-output operand) — state-independent
        if not isinstance(s, Assign):
            return False  # an unfamiliar projection stmt — can't prove distributivity
        hot = [a for a in s.args if a in linear]
        if not hot:
            continue  # state-independent — a constant w.r.t. the split
        if s.op.name == "multiply" and len(hot) == 1:
            linear.add(s.name)  # state · constant — still linear-homogeneous
            continue
        return False  # add / divide / nonlinear of a state value breaks distributivity
    return False  # no Write reached
