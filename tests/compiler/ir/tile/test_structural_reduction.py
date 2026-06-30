"""The :class:`Reduction` structural tile-IR node — its algebra/structure split and the round-trip
invariant the materializer relies on.

A ``Reduction`` is the typed successor of the bare annotated reduce ``Loop``: ``ops.lower`` must
flatten it back to the *exact* loop nest (``[loop, *projection]``), so ``op_cache_key`` and the
``_reduce`` expander stay byte-identical to the bare-loop form. These pin that contract plus the
structural reads (``axis_role`` / ``reduce_loop`` / ``out``) dispatching on the node.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from deplodock.compiler.ir.stmt.algebra import Map
from deplodock.compiler.ir.tile import Reduction
from deplodock.compiler.ir.tile.ops import axis_role, lower, reduce_loop


def _sum_loop(role: AxisRole = AxisRole.PLANAR) -> Loop:
    """A minimal annotated reduce ``Loop`` — ``acc += x[m, k]`` over ``k``, the way recognition
    stamps it (its degenerate ``add`` carrier read off the fold ``Accum``)."""
    acc = Accum(name="acc", value="x_e", op="add")
    body = Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), acc))
    return Loop(axis=Axis("k", 1024), body=body, role=role, carrier=acc.as_carrier())


def test_from_loop_reconstructs_the_loop_exactly() -> None:
    loop = _sum_loop()
    red = Reduction.from_loop(loop, ())
    # The synthesized loop is byte-identical to the captured one (axis / role / carrier / body).
    assert red.loop == loop
    assert reduce_loop(red) == loop
    assert axis_role(red) is AxisRole.PLANAR


def test_bare_reduction_lowers_to_just_the_loop() -> None:
    loop = _sum_loop()
    red = Reduction.from_loop(loop, ())
    assert lower(red) == [loop]
    # A bare reduce's grid ``Write`` is glue — ``out`` is the carrier state's primary component.
    assert red.out == loop.carrier.out == "acc"


def test_reduction_with_projection_round_trips() -> None:
    loop = _sum_loop()
    proj = (Assign(name="rms", op="sqrt", args=("acc",)),)
    red = Reduction.from_loop(loop, proj)
    assert lower(red) == [loop, *proj]
    assert red.out == "rms"  # the projection's last def
    assert axis_role(red) is AxisRole.PLANAR


def test_reduction_matches_the_equivalent_map_form() -> None:
    """``lower(Reduction)`` equals the bare-loop ``Map`` body — the parity guarantee that keeps
    ``op_cache_key`` stable across the lift."""
    loop = _sum_loop()
    proj = (Write(output="out", index=(Var("m"),), value="acc"),)
    red = Reduction.from_loop(loop, proj)
    map_form = Map(body=(loop, *proj))
    assert lower(red) == list(map_form.body)
    assert reduce_loop(red) == reduce_loop(map_form)
    assert axis_role(red) == axis_role(map_form)


def test_twisted_role_propagates() -> None:
    red = Reduction.from_loop(_sum_loop(role=AxisRole.TWISTED), ())
    assert red.role is AxisRole.TWISTED
    assert axis_role(red) is AxisRole.TWISTED
