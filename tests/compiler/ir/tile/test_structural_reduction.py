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

from emmy.compiler.ir.axis import Axis, AxisRole
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import TilePlan
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import Contraction, Map, ReducePlan, Reduction, TileOp
from emmy.compiler.ir.tile.ops import axis_role, lower, reduce_loop, reduce_plan


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


def _tile(op) -> TileOp:
    """An unmapped :class:`TileOp` wrapping ``op`` — the reduce partition rides ``op``'s
    :class:`Reduction` node (there is no residual ``TileOp.reduce`` field)."""
    return TileOp(op=op)


def test_reduce_plan_reads_the_partition_off_the_reduction_node() -> None:
    plan = ReducePlan.of(coop=128)
    red = replace(Reduction.from_loop(_sum_loop()), reduce=plan)
    # A bare reduce root and a projecting Map both surface the node's partition.
    assert reduce_plan(_tile(red)) is plan
    wrapped = Map(body=Body((Assign(name="rms", op="sqrt", args=("acc",)),)), source=red)
    assert reduce_plan(_tile(wrapped)) is plan


def test_reduce_plan_is_none_for_a_flat_map_without_a_reduction_node() -> None:
    """A flat ``Map`` (pointwise, or a scalar per-cell contraction holding a loop-in-body with no
    ``Reduction`` source) has NO partition — ``reduce_plan`` reads ``None``, not a residual field.
    Every partitioned reduce is nodified (``nodify_reduce``), so the fallback is gone."""
    legacy = Map(body=(_sum_loop(),))  # loop in the body, no Reduction source
    assert reduce_plan(_tile(legacy)) is None
    pointwise = Map(body=(Load(name="x_e", input="x", index=(Var("m"),)),))
    assert reduce_plan(_tile(pointwise)) is None


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
        a_operand=a,
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


# --- nodify_reduce: the coop-K / split partial flat-Map → Reduction node lift ------------------ #


def test_nodify_reduce_lifts_a_bare_loop_in_body_map_to_a_reduction() -> None:
    """A flat ``Map`` holding just the annotated reduce loop nodifies to a **bare** ``Reduction`` node
    carrying the partition — ``lower`` byte-identical, ``reduce_plan`` reading the node."""
    from emmy.compiler.ir.tile.ops import nodify_reduce

    loop = _sum_loop()
    flat = Map(body=(loop,))
    plan = ReducePlan.of(coop=4)
    node = nodify_reduce(flat, plan)
    assert isinstance(node, Reduction) and node.reduce is plan
    assert lower(node) == lower(flat)  # bit-identical lowering
    assert reduce_plan(_tile(node)) is plan


def test_nodify_reduce_keeps_a_projection_tail_as_a_wrapping_map() -> None:
    """A fused-epilogue contraction (loop then projection) nodifies to ``Map(body=proj,
    source=Reduction)`` — the tail rides the wrapping ``Map``, the partition the ``Reduction``."""
    from emmy.compiler.ir.tile.ops import nodify_reduce

    loop = _sum_loop(role=AxisRole.CONTRACTION)
    proj = (Assign(name="y", op="relu", args=("acc",)), Write(output="out", index=(Var("m"),), value="y"))
    flat = Map(body=(loop, *proj))
    node = nodify_reduce(flat, ReducePlan.of(reg=2))
    assert isinstance(node, Map) and isinstance(node.source, Reduction)
    assert tuple(node.body) == proj
    assert lower(node) == lower(flat)  # bit-identical
    assert axis_role(node) is AxisRole.CONTRACTION


# --- split-K: Reduction ⊃ Contraction (E1) --------------------------------------------------- #


def test_factor_k_splits_the_axis_with_distinct_names() -> None:
    """``_factor_k`` factors a static ``k`` into ``ksplit × kslice`` — distinct names, the σ
    reconstructing the absolute index ``ksplit·(K/w) + kslice``."""
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import _factor_k

    ksplit, kslice, sigma = _factor_k(Axis("k", 512), 2)
    assert (ksplit.name, ksplit.extent.as_static()) == ("k_ks", 2)
    assert (kslice.name, kslice.extent.as_static()) == ("k", 256)  # original name, K/w extent
    assert sigma.apply(Var("k")).pretty() == "((k_ks * 256) + k)"


def test_splitk_reduction_over_contraction_is_no_double_reduce() -> None:
    """Split-K is ``Reduction(axis=ksplit, source=Contraction(k_axis=kslice))``: the outer additive
    reduce sums partials across CTAs, the inner contraction folds its slice. ``lower`` is a SINGLE
    ``for ksplit:[for kslice: mul-add]`` with DISTINCT axis names (not ``for k:[for k:]``), and it
    still classifies as a ``CONTRACTION`` carrying the GRID (cta) partition."""
    from emmy.compiler.ir.elementwise import ElementwiseImpl
    from emmy.compiler.pipeline.passes.lowering.tile._schedule import _factor_k

    c = _contraction()  # k_axis = k(256)
    ksplit, kslice, sigma = _factor_k(c.k_axis, 2)
    inner = replace(
        c,
        k_axis=kslice,
        a_operand=replace(c.a_operand, index=tuple(sigma.apply(e) for e in c.a_operand.index)),
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


def test_flash_op_is_a_two_contraction_tree() -> None:
    """``_flash_op`` builds the blocked two-``Contraction`` tree: BOTH contractions ride the single
    walked edge — ``partial`` — with QK (``Σ_dd Q·K`` score) at its head and PV — a
    **register-resident-A** contraction (``A = P``, the exp weight) — spliced later (block=1: a
    singleton ``pj`` reduce). No ``source`` asymmetry: the streaming reduce's ``source`` is ``None``
    and the walk reaches both QK and PV as nodes on ``partial``. Its A is computed, not a gmem load;
    its O-fold consumes the PV output ``O_i__pv``, so the ⊗ is a contraction node, not an inline FMA."""
    from emmy.compiler.dim import Dim
    from emmy.compiler.ir.tile.ir import Contraction as _C
    from emmy.compiler.pipeline.passes.lowering.tile._flash import _flash_op

    op = _flash_op("Q", "K", "V", [1, 2], Dim(16), Dim(16), 8, 8)  # (batch, s_q, s_k, head_dim, d_v)
    red = op.source  # Map(body=[O/l proj], source=Reduction(TWISTED, partial=[QK, ..., PV, ...]))
    assert red.source is None, "flash's streaming reduce has no source — both contractions ride partial"
    contractions = [s for s in red.partial if isinstance(s, _C)]
    assert len(contractions) == 2 and contractions[0].acc == "sacc", "QK score contraction is the partial's head node"
    pv = contractions[1]
    assert pv.a_computed and pv.a_name == "O_i__p", "PV's A operand is the register-resident exp weight P"
    assert pv.acc == "O_i__pv"
    # The reduce loop flattens BOTH contractions; the O-fold reads the PV output (no inline v·P).
    (kv_loop,) = lower(red)
    o_fold = next(s for s in kv_loop.body if isinstance(s, Accum) and s.name == "O_i")
    assert o_fold.value == "O_i__pv", "the O-fold consumes the PV contraction's output, not an inline product"


# --- computed (register-resident) A operand: the tensor-core-flash PV crux ---------------------- #


def _pv_contraction(tile: str = "") -> Contraction:
    """A PV-style contraction whose **A operand is computed**, not a gmem ``Load``:
    ``O[m, d] = Σ_j P[m, j]·V[j, d]`` with ``P = exp(S[m, j])`` produced from an in-register score
    (the flash PV shape — its A is register-resident, so ``a_operand`` is a ``Body``, not a ``Load``)."""
    a_body = Body((Load(name="s_e", input="S", index=(Var("m"), Var("j"))), Assign(name="p", op="exp", args=("s_e",))))
    return Contraction(
        axes=(Axis("m", 8), Axis("d", 8)),
        k_axis=Axis("j", 8),
        a_operand=a_body,
        b_load=Load(name="v_e", input="V", index=(Var("j"), Var("d"))),
        acc="oblk",
        tile=TilePlan.parse(tile) if tile else TilePlan(),
        epilogue=Body((Write(output="out", index=(Var("m"), Var("d")), value="oblk"),)),
    )


def test_contraction_computed_a_operand_exposes_its_body() -> None:
    c = _pv_contraction()
    assert c.a_computed and c.a_name == "p"
    assert isinstance(c.a_body[0], Load) and c.a_body[0].input == "S" and c.a_body[-1].op.name == "exp"
    # external reads are the A body's LOADED buffers + B — the computed ``p`` is an internal temp, not a read.
    assert set(c.external_reads()) == {"S", "V"}


def test_contraction_computed_a_lowers_into_the_k_loop() -> None:
    """The computed A body is spliced into the synthesized ``CONTRACTION`` loop AHEAD of the ⊗ multiply:
    ``for j: s_e = S[m, j]; p = exp(s_e); v_e = V[j, d]; oblk__v = v_e·p; oblk += oblk__v`` — the
    register-resident P produced per K-step, then multiplied by V and folded. Same builder a gmem-A
    contraction uses; the operand is just a body, not a leaf load."""
    loop = _pv_contraction().loop
    assert loop.role is AxisRole.CONTRACTION and loop.axis.name == "j"
    body = list(loop.body)
    exp_i = next(i for i, s in enumerate(body) if isinstance(s, Assign) and s.op.name == "exp")
    mul_i = next(i for i, s in enumerate(body) if isinstance(s, Assign) and s.op.name == "multiply")
    acc_i = next(i for i, s in enumerate(body) if isinstance(s, Accum))
    assert exp_i < mul_i < acc_i  # P computed, then P·V, then the additive fold
    assert "p" in body[mul_i].args  # the ⊗ multiplies the register-resident P, no gmem A load


def test_contraction_computed_a_factorizes_at_the_scalar_tier() -> None:
    """The scalar tier expands a computed-A contraction with **no gmem A address**: the register-tile
    replication treats ``P = exp(S)`` as ordinary K-loop body, so the emitted kernel carries the ``exp``
    inside the reduce loop feeding the accumulator. This is the standalone P@V the tensor-core-flash
    rebuild rests on (proved at the scalar tier; the mma tier reads the same operand as a fragment)."""
    from emmy.compiler.pipeline.passes.lowering.kernel._factor import factorize

    tile = factorize(TileOp(op=_pv_contraction()), root=None)
    exps = [s for s in tile.body.iter_of_type(Assign) if s.op.name == "exp"]
    assert exps, "the computed A operand (exp of the score) must survive into the scalar kernel body"
