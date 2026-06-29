"""The generated exp-family carrier (``_carrier.exp_twist``) must reproduce the hand-written
``flash_combine`` / ``online_softmax_combine`` programs — the safety net that lets the
hand-authored bodies be deleted. Compared after structural normalization (alpha-rename of
non-state temps + commutative-arg sort), since the generator chooses its own temp names/order.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.stmt import carrier as _carrier
from deplodock.compiler.ir.stmt.carrier import UnstableCarrierError
from deplodock.compiler.pipeline.passes.lowering.tile._carrier import denom, exp_family_twist, expect
from deplodock.compiler.pipeline.passes.lowering.tile._flash import flash_combine
from deplodock.compiler.pipeline.passes.lowering.tile._softmax import online_softmax_combine

_COMMUTATIVE = {"add", "multiply", "maximum"}


def _canon(prog: tuple, state: tuple[str, ...]) -> dict:
    """Value-numbering equivalence: inline every temp to express each STATE-OUTPUT write as a
    canonical expression tree over the fixed input names (state = old carried value, plus
    state_b/score/value), so two programs that compute the same values compare equal regardless
    of temp names or statement order. ``copy`` is transparent; an ``Accum`` is ``op(base|name,
    value)`` with its commutative args sorted."""
    fixed = set(state)
    temp: dict[str, object] = {}  # temp name -> its Assign
    for s in prog:
        if isinstance(s, Assign) and s.name not in fixed:
            temp[s.name] = s

    def resolve(name: str):
        s = temp.get(name)
        if s is None:  # a fixed input leaf (state old-value / state_b / score / value)
            return name
        if s.op.name == "copy":
            return resolve(s.args[0])
        args = [resolve(a) for a in s.args]
        if s.op.name in _COMMUTATIVE:
            args = sorted(args, key=repr)
        return (s.op.name, tuple(args))

    out: dict[str, object] = {}
    for s in prog:
        if s.name not in fixed:
            continue
        if isinstance(s, Accum):
            base = s.base if s.base is not None else s.name
            args = sorted([resolve(base), resolve(s.value)], key=repr)
            out[s.name] = (s.op.name, tuple(args))
        elif s.op.name == "copy":
            out[s.name] = resolve(s.args[0])
        else:
            args = [resolve(a) for a in s.args]
            if s.op.name in _COMMUTATIVE:
                args = sorted(args, key=repr)
            out[s.name] = (s.op.name, tuple(args))
    return out


def _merge_is_seedable(prog: tuple, state: tuple[str, ...]) -> bool:
    """Each accumulator channel is written by a ``base``-``Accum`` (seed rides ``op.identity``)
    and the pivot by a ``maximum`` ``Accum`` — the streaming-fold shape the loop seeds."""
    writes = {s.name: s for s in prog if s.name in set(state)}
    pivot = writes[state[0]]
    if not (isinstance(pivot, Accum) and pivot.op.name == "maximum"):
        return False
    return all(
        isinstance(writes[n], Accum) and writes[n].op.name == "add" and writes[n].base is not None and writes[n].dtype is not None
        for n in state[1:]
    )


def test_generated_flash_matches_handwritten():
    state = ("m_i", "l_i", "O_i")
    gen = exp_family_twist("s_e", [denom(), expect("v_e")], state)  # spec-mode Monoid
    hand = flash_combine("m_i", "l_i", "O_i", "s_e", "v_e")  # bound-mode (hand-written) Monoid
    assert _canon(gen.merge, state) == _canon(hand.merge, state)
    assert _canon(gen.combine_states, state) == _canon(hand.combine_states, state)
    assert gen.state_b == hand.state_b
    assert _merge_is_seedable(gen.merge, state)


def test_generated_online_softmax_matches_handwritten():
    state = ("m", "d")
    gen = exp_family_twist("s", [denom()], state)
    hand = online_softmax_combine("m", "d", "s")
    assert _canon(gen.merge, state) == _canon(hand.merge, state)
    assert _canon(gen.combine_states, state) == _canon(hand.combine_states, state)
    assert gen.state_b == hand.state_b
    assert _merge_is_seedable(gen.merge, state)


def test_certificate_rejects_unstable_combine():
    # A mis-specced carrier whose rescale subtracts the WRONG max (a bare maximum that does not
    # include the numerator) must be rejected, not silently emitted.
    bad = (
        Assign("mx", "maximum", ("a", "b")),
        Assign("dm", "subtract", ("z", "mx")),  # z ∉ {a, b} → exp(z − max(a,b)) not provably ≤ 0
        Assign("e", "exp", ("dm",)),
    )
    with pytest.raises(UnstableCarrierError):
        _carrier._certify(bad)


def test_every_exp_arg_is_nonpositive_by_construction():
    # The certificate runs inside the builders; if it passed, every exp is x − max(…, x, …).
    state = ("m_i", "l_i", "O_i")
    tw = exp_family_twist("s_e", [denom(), expect("v_e")], state)
    for prog in (tw.merge, tw.combine_states):
        defs = {s.name: s for s in prog if isinstance(s, Assign)}
        exps = [s for s in prog if isinstance(s, Assign) and s.op.name == "exp"]
        assert exps  # the LSE carrier has rescales
        for e in exps:
            sub = defs[e.args[0]]
            assert sub.op.name == "subtract"


def test_softmax_is_flash_minus_the_expectation_channel():
    # Structural: the (m, d) softmax combine_states is the (m, l) projection of flash's — same
    # generator, one fewer accumulator channel.
    flash = exp_family_twist("s", [denom(), expect("v")], ("m", "l", "O"))
    soft = exp_family_twist("s", [denom()], ("m", "d"))
    # both have exactly one exp-rescale pair (a, b) in combine_states
    n_exp = lambda p: sum(isinstance(s, Assign) and s.op.name == "exp" for s in p)  # noqa: E731
    assert n_exp(soft.combine_states) == n_exp(flash.combine_states) == 2
