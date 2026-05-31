"""Write.atomic — force an atomicAdd store (Stream-K boundary-partial write).

The structural ``Body.coordination`` atomic-axes rule can't see a Stream-K
boundary partial (it's gated by a *runtime* full-vs-partial branch, not a missing
block axis), so the adaptive rewrite sets ``atomic=True`` on that Write directly.
"""

from __future__ import annotations

from deplodock.compiler.graph import _eval_stmt
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Write
from deplodock.compiler.ir.stmt.base import RenderCtx


def test_scalar_atomic_renders_atomicadd():
    w = Write(output="C", index=(Var("m"),), value="acc", atomic=True)
    assert w.render(RenderCtx(indent=0)) == ["atomicAdd(&C[m], acc);"]


def test_default_is_plain_store():
    w = Write(output="C", index=(Var("m"),), value="acc")
    assert w.render(RenderCtx(indent=0)) == ["C[m] = acc;"]
    assert w.atomic is False


def test_vector_atomic_emits_per_element_atomicadd():
    """A packed store can't be atomic in one transaction — each element gets its
    own atomicAdd into the shared output cell."""
    w = Write(output="C", index=(Var("m"),), values=("a0", "a1"), atomic=True)
    assert w.render(RenderCtx(indent=0)) == ["atomicAdd(&C[m], a0);", "atomicAdd(&C[m + 1], a1);"]


def test_atomic_participates_in_identity_and_round_trips():
    a = Write(output="C", index=(Var("m"),), value="acc", atomic=True)
    plain = Write(output="C", index=(Var("m"),), value="acc")
    assert a != plain  # the flag is part of structural identity
    assert _eval_stmt(repr(a)) == a  # JSON repr round-trip preserves it
