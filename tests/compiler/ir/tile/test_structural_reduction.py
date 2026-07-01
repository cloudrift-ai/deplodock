"""The :class:`Reduction` / :class:`Contraction` / :class:`Map` structural tile-IR nodes — their
algebra/structure split and the round-trip invariant the materializer relies on.

A ``Reduction`` is the typed successor of the bare annotated reduce ``Loop`` (holding no projection);
a projected reduce (softmax / RMSNorm) is a ``Map`` whose body IS the projection over a ``Reduction``
``source``. A ``Contraction`` is the tiled matmul node (built recognize-side at fork-emit), holding
its operands + ``tile`` + projection ``epilogue``; it synthesizes the canonical mul-add ``CONTRACTION``
loop on demand so ``ops.lower`` / ``reduce_loop`` flatten it back to the loop nest the materializer
expands (via ``_factor.factorize``). These pin that contract plus the structural reads (``axis_role`` /
``reduce_loop`` / ``out``) dispatching on the nodes.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.schedule import TilePlan
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from deplodock.compiler.ir.tile import Contraction, Map, ReducePlan, Reduction, TileOp
from deplodock.compiler.ir.tile.ops import axis_role, lower, reduce_loop, reduce_plan


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


def _tile(op, schedule_reduce: ReducePlan | None = None) -> TileOp:
    """An unmapped :class:`TileOp` — its ``reduce`` field is the residual root partition (the
    fallback ``reduce_plan`` reads for a not-yet-nodified reduce)."""
    return TileOp(op=op, reduce=schedule_reduce or ReducePlan())


def test_reduce_plan_reads_the_partition_off_the_reduction_node() -> None:
    plan = ReducePlan.of(coop=128)
    red = replace(Reduction.from_loop(_sum_loop()), reduce=plan)
    # A bare reduce root and a projecting Map both surface the node's partition.
    assert reduce_plan(_tile(red)) is plan
    wrapped = Map(body=Body((Assign(name="rms", op="sqrt", args=("acc",)),)), source=red)
    assert reduce_plan(_tile(wrapped)) is plan
    # The partition rides the node, not the ``TileOp``'s residual ``reduce`` field.
    assert _tile(red).reduce == ReducePlan()


def test_reduce_plan_falls_back_to_the_residual_reduce_for_a_legacy_loop_in_body_map() -> None:
    residual = ReducePlan.of(coop=64)
    legacy = Map(body=(_sum_loop(),))  # loop in the body, no Reduction source (flash's form)
    assert reduce_plan(_tile(legacy, residual)) is residual


def test_twisted_role_propagates() -> None:
    red = Reduction.from_loop(_sum_loop(role=AxisRole.TWISTED))
    assert red.role is AxisRole.TWISTED
    assert axis_role(red) is AxisRole.TWISTED
    assert axis_role(Map(body=Body(()), source=red)) is AxisRole.TWISTED


def _contraction(epilogue: Body | None = None) -> Contraction:
    """A minimal tiled contraction node — ``acc = Σ_k A[m, k]·B[k, n]`` over a scalar tile."""
    a = Load(name="a_e", input="A", index=(Var("m"), Var("k")))
    b = Load(name="b_e", input="B", index=(Var("k"), Var("n")))
    return Contraction(
        axes=(Axis("m", 128), Axis("n", 128)),
        k_axis=Axis("k", 256),
        a_load=a,
        b_load=b,
        acc="acc",
        tile=TilePlan.parse("n2/f2"),
        epilogue=epilogue or Body(()),
    )


def test_contraction_synthesizes_the_mul_add_loop() -> None:
    c = _contraction()
    loop = c.loop
    assert loop.role is AxisRole.CONTRACTION
    assert loop.axis == c.k_axis
    # The shared ``contraction_loop`` builder: B, A loads + the ⊗ lift + the additive fold.
    assert isinstance(loop.body[-1], Accum) and loop.body[-1].name == "acc"
    assert isinstance(loop.body[-2], Assign) and loop.body[-2].op.name == "multiply"


def test_contraction_dispatches_through_ops() -> None:
    c = _contraction()
    assert axis_role(c) is AxisRole.CONTRACTION
    assert reduce_loop(c) == c.loop
    assert lower(c) == [c.loop]  # bare: just the synthesized loop (the grid Write is materialize glue)
    assert c.out == "acc"


def test_contraction_lower_appends_the_fused_epilogue() -> None:
    proj = (Assign(name="y", op="relu", args=("acc",)), Write(output="out", index=(Var("m"), Var("n")), value="y"))
    c = _contraction(epilogue=Body(proj))
    assert lower(c) == [c.loop, *proj]
    assert reduce_loop(c).role is AxisRole.CONTRACTION  # the projection doesn't hide the contraction


# --- split-K: Reduction ⊃ Contraction (E1) --------------------------------------------------- #


def test_factor_k_splits_the_axis_with_distinct_names() -> None:
    """``_factor_k`` factors a static ``k`` into ``ksplit × kslice`` — distinct names, the σ
    reconstructing the absolute index ``ksplit·(K/w) + kslice``."""
    from deplodock.compiler.pipeline.passes.lowering.tile._schedule import _factor_k

    ksplit, kslice, sigma = _factor_k(Axis("k", 512), 2)
    assert (ksplit.name, ksplit.extent.as_static()) == ("k_ks", 2)
    assert (kslice.name, kslice.extent.as_static()) == ("k", 256)  # original name, K/w extent
    assert sigma.apply(Var("k")).pretty() == "((k_ks * 256) + k)"


def test_splitk_reduction_over_contraction_is_no_double_reduce() -> None:
    """Split-K is ``Reduction(axis=ksplit, source=Contraction(k_axis=kslice))``: the outer additive
    reduce sums partials across CTAs, the inner contraction folds its slice. ``lower`` is a SINGLE
    ``for ksplit:[for kslice: mul-add]`` with DISTINCT axis names (not ``for k:[for k:]``), and it
    still classifies as a ``CONTRACTION`` carrying the GRID (cta) partition."""
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.pipeline.passes.lowering.tile._schedule import _factor_k

    c = _contraction()  # k_axis = k(256)
    ksplit, kslice, sigma = _factor_k(c.k_axis, 2)
    inner = replace(
        c,
        k_axis=kslice,
        a_load=replace(c.a_load, index=tuple(sigma.apply(e) for e in c.a_load.index)),
        b_load=replace(c.b_load, index=tuple(sigma.apply(e) for e in c.b_load.index)),
    )
    carrier = Accum(name="acc", value="acc__v", op=ElementwiseImpl("add")).as_carrier()
    red = Reduction(carrier=carrier, axis=ksplit, role=AxisRole.CONTRACTION, source=inner, reduce=ReducePlan.of(cta=2, finalize="atomic"))

    assert axis_role(red) is AxisRole.CONTRACTION
    assert reduce_plan(_tile(red)).cta == 2
    lo = lower(red)
    assert len(lo) == 1 and isinstance(lo[0], Loop) and lo[0].axis.name == "k_ks"
    inner_loops = [s for s in lo[0].body if isinstance(s, Loop)]
    assert len(inner_loops) == 1 and inner_loops[0].axis.name == "k"  # distinct from ksplit — no double-reduce
    assert isinstance(inner_loops[0].body[-1], Accum) and inner_loops[0].body[-1].name == "acc"


# --- flash: a reduce partial composing a nested PV Contraction (tensor-core-flash seam) --------- #


def test_reduce_partial_flattens_a_nested_pv_contraction() -> None:
    """Flash composes TWO contractions — QK on ``source`` and PV **inside the partial**. The QK loop
    splices ahead of the partial and the nested PV ``Contraction`` (a ``Stmt``) flattens to its own
    loop in place — one recursion rule, so the scalar tier expands ``for kv:[QK loop; P; PV loop;
    fold]``. This is the structural seam warp-flash rides."""
    qk = _contraction()  # Σ_k A·B -> acc (the score S)
    pv = replace(_contraction(), axes=(Axis("m", 128), Axis("d", 64)), k_axis=Axis("j", 32), acc="oblk")
    prob = Assign(name="p", op="exp", args=("acc",))  # softmax weight between the two contractions
    fold = Accum(name="O_i", value="oblk", op="add")
    red = Reduction(carrier=fold.as_carrier(), axis=Axis("kv", 128), partial=Body((prob, pv, fold)), role=AxisRole.TWISTED, source=qk)

    (kv_loop,) = lower(red)
    assert kv_loop.axis.name == "kv" and kv_loop.role is AxisRole.TWISTED
    body = list(kv_loop.body)
    assert not any(isinstance(s, Contraction) for s in body), "the nested PV contraction must be flattened, not left raw"
    (qk_loop,) = [s for s in body if isinstance(s, Loop) and s.axis.name == "k"]
    (pv_loop,) = [s for s in body if isinstance(s, Loop) and s.axis.name == "j"]
    # QK (source) first, then the pre-PV probability, then the flattened PV loop, then the carrier fold.
    assert body.index(qk_loop) < body.index(prob) < body.index(pv_loop) < body.index(fold)
    assert pv_loop.role is AxisRole.CONTRACTION and isinstance(pv_loop.body[-1], Accum) and pv_loop.body[-1].name == "oblk"
