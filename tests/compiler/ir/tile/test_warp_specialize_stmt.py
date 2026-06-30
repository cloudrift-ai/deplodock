"""Stmt-protocol tests for ``WarpSpecialize`` (the Tile-IR carrier for the
warp-specialization producer/consumer split).

The materializer's lowering of ``WarpSpecialize`` into mbarrier
primitives lives in ``tests/compiler/passes/test_materialize_warp_specialize.py``;
this file only checks that the dataclass plays nicely with the generic
body-walker protocol (``nested()`` / ``with_bodies()``) and the
rewrite/simplify dispatch.
"""

from __future__ import annotations

import pytest

from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Body, Cond
from emmy.compiler.ir.tile.ir import AsyncWait, WarpSpecialize


def _ws(**overrides):
    """Build a minimal WarpSpecialize for protocol tests."""
    kwargs = dict(
        producer_body=Body((AsyncWait(keep=0),)),
        consumer_body=Body((AsyncWait(keep=1),)),
        ring_depth=2,
        n_producer_threads=32,
        consumer_thread_axes=(),
    )
    kwargs.update(overrides)
    return WarpSpecialize(**kwargs)


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def test_construct_basic():
    ws = _ws()
    assert ws.ring_depth == 2
    assert ws.n_producer_threads == 32


def test_body_coercion_from_tuple():
    """tuple → Body coercion mirrors StageBundle / Cond."""
    ws = WarpSpecialize(
        producer_body=(AsyncWait(keep=0),),  # type: ignore[arg-type]
        consumer_body=(AsyncWait(keep=1),),  # type: ignore[arg-type]
        ring_depth=2,
        n_producer_threads=32,
        consumer_thread_axes=(),
    )
    assert isinstance(ws.producer_body, Body)
    assert isinstance(ws.consumer_body, Body)


def test_rejects_zero_ring_depth():
    with pytest.raises(ValueError, match="ring_depth"):
        _ws(ring_depth=0)


def test_rejects_zero_n_producer_threads():
    with pytest.raises(ValueError, match="n_producer_threads"):
        _ws(n_producer_threads=0)


def test_rejects_nested_warp_specialize_in_producer():
    inner = _ws()
    with pytest.raises(ValueError, match="cannot nest"):
        _ws(producer_body=Body((inner,)))


def test_rejects_nested_warp_specialize_in_consumer():
    inner = _ws()
    with pytest.raises(ValueError, match="cannot nest"):
        _ws(consumer_body=Body((inner,)))


def test_rejects_nested_warp_specialize_deep_in_consumer():
    """Nesting check looks through Cond wrappers — not just top-level."""
    inner = _ws()
    wrapper = Cond(cond=Var("p"), body=(inner,))
    with pytest.raises(ValueError, match="cannot nest"):
        _ws(consumer_body=Body((wrapper,)))


# ---------------------------------------------------------------------------
# Body-walker protocol — nested() / with_bodies()
# ---------------------------------------------------------------------------


def test_nested_returns_two_bodies():
    ws = _ws()
    assert ws.nested() == (ws.producer_body, ws.consumer_body)


def test_with_bodies_round_trip():
    ws = _ws()
    same = ws.with_bodies(ws.nested())
    assert same == ws


def test_with_bodies_replaces_both():
    ws = _ws()
    new_prod = Body((AsyncWait(keep=2),))
    new_cons = Body((AsyncWait(keep=3),))
    ws2 = ws.with_bodies((new_prod, new_cons))
    assert isinstance(ws2, WarpSpecialize)
    assert ws2.producer_body == new_prod
    assert ws2.consumer_body == new_cons
    # Scalar fields preserved.
    assert ws2.ring_depth == ws.ring_depth
    assert ws2.n_producer_threads == ws.n_producer_threads


def test_with_bodies_wrong_arity_raises():
    ws = _ws()
    with pytest.raises(ValueError, match="expected 2 bodies"):
        ws.with_bodies((Body(()),))


def test_body_iter_recurses_into_both_branches():
    """Body.iter walks nested bodies via Stmt.nested(); a WarpSpecialize in
    a parent Body should expose both branches' stmts."""
    ws = _ws()
    outer = Body((ws,))
    seen = list(outer.iter())
    # The wrapper itself + the AsyncWait from each branch.
    assert ws in seen
    assert AsyncWait(keep=0) in seen
    assert AsyncWait(keep=1) in seen


# ---------------------------------------------------------------------------
# rewrite / simplify — both bodies are visited
# ---------------------------------------------------------------------------


def test_rewrite_substitutes_into_both_bodies():
    """A Sigma substitution on ``x`` reaches AsyncWait.phase inside both
    producer and consumer bodies."""
    ws = WarpSpecialize(
        producer_body=Body((AsyncWait(keep=0, phase=Var("x")),)),
        consumer_body=Body((AsyncWait(keep=1, phase=Var("x")),)),
        ring_depth=2,
        n_producer_threads=32,
        consumer_thread_axes=(),
    )
    sigma = Sigma({"x": Literal(7, "int")})
    out = ws.rewrite(lambda n: n, sigma)
    assert isinstance(out, WarpSpecialize)
    # Both phases are now the literal 7.
    prod_wait = out.producer_body[0]
    cons_wait = out.consumer_body[0]
    assert isinstance(prod_wait, AsyncWait)
    assert isinstance(cons_wait, AsyncWait)
    assert prod_wait.phase == Literal(7, "int")
    assert cons_wait.phase == Literal(7, "int")


def test_rewrite_preserves_scalar_fields():
    ws = _ws()
    out = ws.rewrite(lambda n: n, Sigma.IDENTITY)
    assert isinstance(out, WarpSpecialize)
    assert out.ring_depth == ws.ring_depth
    assert out.n_producer_threads == ws.n_producer_threads


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------


def test_pretty_contains_role_labels():
    ws = _ws()
    text = "\n".join(ws.pretty())
    assert "warp_specialize" in text
    assert "ring=2" in text
    assert "n_prod=32" in text
    assert "producer:" in text
    assert "consumer:" in text


# ---------------------------------------------------------------------------
# Hashability — required by Stmt invariant
# ---------------------------------------------------------------------------


def test_hashable_and_equal():
    a = _ws()
    b = _ws()
    assert hash(a) == hash(b)
    assert a == b
    assert {a, b} == {a}
