"""The :class:`Reduction` / :class:`Map` structural tile-IR nodes — their algebra/structure split
and the round-trip invariant the materializer relies on.

A ``Reduction`` is the typed successor of the bare annotated reduce ``Loop`` (holding no projection);
a projected reduce (softmax / RMSNorm) is a ``Map`` whose body IS the projection over a ``Reduction``
``source``. ``ops.lower`` must flatten either back to the *exact* loop nest, so ``op_cache_key`` and
the ``_reduce`` expander stay byte-identical to the bare-loop form. These pin that contract plus the
structural reads (``axis_role`` / ``reduce_loop`` / ``out``) dispatching on the nodes.
"""

from __future__ import annotations

from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from deplodock.compiler.ir.tile import Map, Reduction
from deplodock.compiler.ir.tile.ops import axis_role, lower, reduce_loop


def _sum_loop(role: AxisRole = AxisRole.PLANAR) -> Loop:
    """A minimal annotated reduce ``Loop`` — ``acc += x[m, k]`` over ``k``, the way recognition
    stamps it (its degenerate ``add`` carrier read off the fold ``Accum``)."""
    acc = Accum(name="acc", value="x_e", op="add")
    body = Body((Load(name="x_e", input="x", index=(Var("m"), Var("k"))), acc))
    return Loop(axis=Axis("k", 1024), body=body, role=role, carrier=acc.as_carrier())


def test_from_loop_reconstructs_the_loop_exactly() -> None:
    loop = _sum_loop()
    red = Reduction.from_loop(loop)
    # The synthesized loop is byte-identical to the captured one (axis / role / carrier / body).
    assert red.loop == loop
    assert reduce_loop(red) == loop
    assert axis_role(red) is AxisRole.PLANAR


def test_bare_reduction_lowers_to_just_the_loop() -> None:
    loop = _sum_loop()
    red = Reduction.from_loop(loop)
    assert lower(red) == [loop]
    # A bare reduce's grid ``Write`` is glue — ``out`` is the carrier state's primary component.
    assert red.out == loop.carrier.out == "acc"


def test_projected_reduce_is_a_map_over_the_reduction() -> None:
    loop = _sum_loop()
    proj = (Assign(name="rms", op="sqrt", args=("acc",)),)
    node = Map(body=Body(proj), source=Reduction.from_loop(loop))
    # lower flattens the source's loop then the projection body — the bare-loop ``Map`` body.
    assert lower(node) == [loop, *proj]
    assert node.out == "rms"  # the projection's last def
    # The structural reads see straight through to the source's reduce.
    assert reduce_loop(node) == loop
    assert axis_role(node) is AxisRole.PLANAR


def test_map_over_reduction_matches_the_legacy_loop_in_body_form() -> None:
    """``lower(Map(source=Reduction))`` equals the bare-loop ``Map(body=(loop, *proj))`` — the parity
    guarantee that keeps ``op_cache_key`` stable across the lift."""
    loop = _sum_loop()
    proj = (Write(output="out", index=(Var("m"),), value="acc"),)
    node = Map(body=Body(proj), source=Reduction.from_loop(loop))
    legacy = Map(body=(loop, *proj))
    assert lower(node) == lower(legacy)
    assert reduce_loop(node) == reduce_loop(legacy)
    assert axis_role(node) == axis_role(legacy)


def test_pure_pointwise_map_has_no_reduce() -> None:
    node = Map(body=(Load(name="x_e", input="x", index=(Var("m"),)), Assign(name="y", op="relu", args=("x_e",))))
    assert node.source is None
    assert reduce_loop(node) is None
    assert axis_role(node) is AxisRole.FREE
    assert node.out == "y"


def test_twisted_role_propagates() -> None:
    red = Reduction.from_loop(_sum_loop(role=AxisRole.TWISTED))
    assert red.role is AxisRole.TWISTED
    assert axis_role(red) is AxisRole.TWISTED
    assert axis_role(Map(body=Body(()), source=red)) is AxisRole.TWISTED
