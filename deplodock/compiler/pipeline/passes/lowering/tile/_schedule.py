"""Schedule a lifted kernel onto the thread grid (+ pick the reduce partition / output tile).

The scheduling **half** of the merged ``010_recognize`` tile-lowering pass — recognition
builds an UNMAPPED :class:`~deplodock.compiler.ir.tile.ir.TileOp` (the structural-IR root ``op`` +
a ``place`` carrying just the free axes) and calls :func:`schedule` here in the same rewrite (no
separate ``020`` pass). Scheduling binds the placement's ``free`` axes onto the grid
(``Placement.on_grid``) and offers the per-axis
scheduling forks — the reduce-axis **partition** (:class:`~...schedule.ReducePlan`, the
``REDUCE`` codec) for a reduce axis and the output **tile** (:class:`~...schedule.TilePlan`,
the ``TILE`` codec) for a contraction — read off the axes' :class:`~...axis.AxisRole`, never a
kernel kind. This is a helper module (``_``-prefixed, not a standalone rule); its knob
constants still register (``knob._walk_modules`` walks every imported module under the package).

This cut picks a **whole-CTA cooperative** partition for a **static, scalar-output,
degenerate-monoid** reduce (plain ``sum`` / ``max`` / ``mean``) when the reduce axis is
wide and the output grid is small enough to leave the GPU under-occupied — one CTA per
output cell, ``coop`` threads cooperatively folding the reduce axis (the combine is
materialized in ``lowering/kernel``). Everything else (pointwise ``Map``, twisted /
full-row reductions like online-softmax & RMSNorm, contractions, symbolic axes) keeps the
**scalar serial** fold (``ReducePlan()`` — one thread per output cell).

The selection here is **conservative module constants** standing in for the eventual
``REDUCE`` knob + prior-driven choice. ``# TODO``: replace the constants with
``knob.py::_reduce_decomp`` (BR→coop, BK→serial, FK→reg, SPLITK→cta) + the learned /
analytic prior. The cross-CTA ``g<n>`` split (``030_split``) and the ``r<n>`` (ILP) reg
fold are built and honored for an additive carrier via an explicit ``REDUCE`` pin (the
split emits the partial + finalize kernels / atomicAdd; the reg fold emits the ILP
accumulators). Strided-cooperative rows (a small whole free axis packed alongside the coop
lanes), the symbolic-axis cooperative tier, the twisted-carrier (flash) cross-CTA split,
and flash cooperative-KV remain future steps.
"""

from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace

from deplodock.compiler.dim import DEFAULT_SEQ_HINT, Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.schedule import Stage, WarpSpec, is_warp_codec
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum
from deplodock.compiler.ir.tile import Contraction, Map, ReducePlan, Reduction, TileOp, TilePlan
from deplodock.compiler.ir.tile.ops import axis_role, reduce_loop
from deplodock.compiler.pipeline.forks import REDUCE, STAGE, TILE, WSPEC
from deplodock.compiler.pipeline.passes.lowering.tile._atomize import semiring_binding
from deplodock.compiler.pipeline.passes.lowering.tile._catalog import scalar_tile_moves
from deplodock.compiler.pipeline.pipeline import LoweringError

# The schedule codec knobs (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC``) are declared in
# ``pipeline/forks.py`` (split out of ``knob.py``) and imported here, where they are
# resolved into the schedule slices. The decision hierarchy for each is the env pin (via
# ``Knob.narrow``) > the search/prior fork > the conservative default below.


def _is_mma_flash(op) -> bool:
    """True iff ``op`` is the mma-flash tree — a ``TWISTED`` :class:`Reduction` (bare or under a
    projecting :class:`Map`) whose ``source`` :class:`Contraction` carries a tensor-core
    :class:`TilePlan` (the ``DEPLODOCK_CHAIN`` build). It is fully scheduled recognize-side, so
    :func:`schedule` passes it through untouched to the flash-warp emitter."""
    red = op.source if isinstance(op, Map) else op
    return isinstance(red, Reduction) and isinstance(red.source, Contraction) and red.source.tile.is_warp


def _at(knob, axis_name: str) -> str:
    """The axis-named knob key ``FAMILY@<axis>`` (e.g. ``TILE@d``) — the per-node schedule codec keyed
    by the reduce/contraction axis it schedules, so a multi-node kernel addresses each node."""
    return f"{knob.name}@{axis_name}"


# Conservative cooperative-reduce selection constants (the default when REDUCE is unpinned).
_COOP_MIN_EXTENT = 128  # only cooperate when the reduce axis is at least this wide
_SERIAL_TARGET = 8  # aim for ~this many serial steps per cooperating thread
_MAX_COOP = 256  # cap on cooperative threads per CTA (power of two)
_FREE_CAP = 256  # only cooperate when the output grid is at most this many cells (under-occupied)
_MAX_BLOCK_THREADS = 1024  # CUDA hardware limit on threads per CTA (guards an oversized TILE parallel tile)


def _hint_extent(ax) -> int:
    """An axis's static extent, or its ``Dim`` hint when symbolic (the occupancy heuristic
    sizes a dynamic axis by its hint; the kernel still deploys over the runtime extent)."""
    e = ax.extent
    return e.as_static() if e.is_static else (e.hint or DEFAULT_SEQ_HINT)


def _prevpow2(n: int) -> int:
    """The largest power of two ≤ ``n`` (≥ 1)."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _pick_coop(extent: int, free: int) -> int:
    """The conservative whole-CTA cooperative-thread count for a reduce of static
    ``extent`` over ``free`` output cells, or ``1`` (stay scalar/serial). Cooperate only on
    a wide reduce (``extent ≥ _COOP_MIN_EXTENT``) feeding a small grid (``free ≤
    _FREE_CAP`` — otherwise the scalar tier already saturates the GPU); the count targets
    ``_SERIAL_TARGET`` serial steps, capped at ``_MAX_COOP``, rounded to a power of two (the
    butterfly / tree reorder)."""
    if extent < _COOP_MIN_EXTENT or free > _FREE_CAP:
        return 1
    coop = min(_prevpow2(extent // _SERIAL_TARGET), _MAX_COOP)
    return coop if coop >= 2 else 1


def _coop_carrier(kernel):
    """The cooperative-eligible reduce ``Loop`` of ``kernel`` (read for its ``axis``), or ``None``
    (keep serial).

    Eligible: a ``PLANAR`` / ``TWISTED`` reduce loop — **degenerate** (plain ``sum`` / ``max`` /
    ``mean``) AND **twisted** (online-softmax ``(m, d)``, flash ``(m, l, O)``) alike, since the
    cross-thread combine is carrier-generic (it drives off the carrier's ``combine_states``, which
    a twisted carrier authors). Both **scalar** outputs (flash's ``O/l`` per ``(m, d)`` cell — ``d``
    is a grid axis) and **full-row** outputs (softmax / RMSNorm — the post-reduce sweep is
    distributed across the coop lanes by the materializer) are handled. The reduce axis may be
    **symbolic** (dynamic ``seq_len``): each lane strides it to the runtime extent (the ``< seq_len``
    bound is the masked tail). A ``CONTRACTION`` (its output tile is ``_tile_option`` / ``_warp_option``;
    a cross-CTA split-K is the ``_splitk_option`` fork) or a flat-``Map`` fallback (multi /
    nested-non-flash reduce — no annotated reduce loop) is not eligible here and keeps the serial fold."""
    rl = reduce_loop(kernel.op)
    if rl is None or rl.role not in (AxisRole.PLANAR, AxisRole.TWISTED):
        return None
    return rl


def _reduce_specs(kernel, place) -> list[str]:
    """The candidate ``REDUCE`` codec strings for ``kernel``, applying the decision
    hierarchy. A kernel the cooperative tier can't partition (pointwise, or a twisted /
    full-row / contraction reduce) is the lone scalar fold ``[""]`` — the ``REDUCE`` pin is
    ignored there, since it only governs the cooperative reduce tier. An eligible reduce
    offers ``[conservative coop, scalar]`` (a fork the search / prior ranks, option-0 = the
    conservative pick so a cold greedy compile keeps cooperating), with an env pin
    (``DEPLODOCK_REDUCE``) authoritative over the candidates (``Knob.narrow``)."""
    carrier = _coop_carrier(kernel)
    if carrier is None:
        return [""]  # not cooperative-eligible — scalar serial fold; the pin doesn't apply
    # A symbolic reduce axis is sized by its ``Dim`` hint for the conservative pick (the
    # kernel deploys at the hint and strides to the runtime extent); a pin overrides it.
    extent = _hint_extent(carrier.axis)
    # A symbolic free axis (dynamic-grid tier) is sized by its ``Dim`` hint for the occupancy
    # heuristic — the kernel still deploys over the runtime grid.
    free = prod(_hint_extent(a) for a in place.free) if place.free else 1
    coop = _pick_coop(extent, free)
    cands = [f"b{coop}", ""] if coop > 1 else [""]  # conservative coop first (cold greedy → option-0)
    return list(REDUCE.narrow(cands))


def _with_reduce(op, plan: ReducePlan):
    """Stamp the chosen ``ReducePlan`` onto the op's :class:`Reduction` node (bare, or wrapped under a
    projecting :class:`Map`). The reduce partition lives **on the node**, not the ``TileSchedule`` —
    read back via ``ops.reduce_plan``. ``_option`` only schedules a PLANAR / TWISTED reduce, whose op
    recognition always emits as a bare ``Reduction`` or a projecting ``Map(source=Reduction)``."""
    if isinstance(op, Reduction):
        return replace(op, reduce=plan)
    assert isinstance(op, Map) and isinstance(op.source, Reduction), f"reduce op must nodify to Reduction, got {type(op).__name__}"
    return replace(op, source=replace(op.source, reduce=plan))


def _option(tile, place, spec: str, name: str, knobs: dict) -> TileOp:
    """One scheduled ``TileOp``: ``place`` mapped onto the grid + the ``REDUCE`` spec resolved into the
    :class:`Reduction` node's ``ReducePlan`` (the ephemeral knob → materialized plan stamped **on the
    node**), with the spec stamped on ``knobs`` for the prior. The spec is keyed ``REDUCE@<axis>``
    (the reduce axis this node partitions), so a multi-node kernel addresses each reduce."""
    plan = ReducePlan.parse(spec)
    op = _with_reduce(tile.op, plan)
    raxis = reduce_loop(tile.op).axis.name
    return TileOp(op=op, name=name, place=place, knobs={**knobs, _at(REDUCE, raxis): spec})


def _tile_specs(kernel) -> list[str]:
    """Candidate ``TILE`` codec strings for ``kernel`` — only a ``CONTRACTION`` contraction tiles
    its output; everything else is the per-cell tier (``[""]``, the pin doesn't apply). The env
    pin ``DEPLODOCK_TILE`` is authoritative (``Knob.narrow``); unpinned, the default is the
    **permitted-move catalog** (:func:`_catalog.scalar_tile_moves` — per-cell option-0 then the
    legality-guarded scalar register-tile grid), so an unpinned ``compile`` / ``tune`` explores the
    tile space ranked by the prior. Warp (tensor-core) tiles stay pin-driven (a pinned ``a:<atom>``
    codec routes to ``_warp_option``); folding the warp / reduce / stage moves into the catalog is the
    next slice."""
    if axis_role(kernel.op) is not AxisRole.CONTRACTION:
        return [""]
    return list(TILE.narrow(scalar_tile_moves()))


def _splitk_pin() -> str:
    """The pinned ``g<w>[a|k]`` split-K spec (or ``""``) — the cross-CTA K partition a
    ``CONTRACTION`` honors through the structural ``Reduction ⊃ Contraction`` fork
    (:func:`_splitk_option`), consumed by ``030_split``. Reads the ``REDUCE`` pin and returns it
    only when it parses to a **GRID split** (``needs_split``); a non-split ``b`` / ``r`` pin or
    another codec is not a split-K request — ignore it rather than fail."""
    pinned = REDUCE.narrow([""])[0]
    try:
        plan = ReducePlan.parse(pinned)
    except ValueError:
        return ""
    return pinned if plan.needs_split else ""


def _coop_reduce_spec() -> str:
    """The pinned cooperative (``b``) / ILP (``r``) K partition a **non-output-tiled** ``CONTRACTION``
    honors — folded through ``_factor._factorize_reduce`` (a contraction is the degenerate carrier of
    its additive fold), riding the residual ``reduce`` field on the still-``Map`` scalar tier. Returns
    the ``REDUCE`` pin iff it parses to a coop / reg partition WITHOUT a GRID split (the split-K ``g``
    takes the structural :func:`_splitk_option` fork instead); ``""`` otherwise (a foreign codec is
    not ours — ignore it rather than fail)."""
    pinned = REDUCE.narrow([""])[0]
    try:
        plan = ReducePlan.parse(pinned)
    except ValueError:
        return ""
    return pinned if (not plan.needs_split and (plan.coop > 1 or plan.reg > 1)) else ""


def _stage_spec(kernel) -> str:
    """The pinned ``STAGE`` codec for ``kernel`` — only a ``CONTRACTION`` contraction stages its
    operands today (everything else is ``""``, the pin doesn't apply). Pin-only this cut:
    returns the authoritative ``DEPLODOCK_STAGE`` pin (``Knob.narrow``) or ``""`` (gmem-direct,
    ``stage=None``). A pin that doesn't parse as the ``STAGE`` codec (e.g. a bare operand
    binmask ``"11"``) is **structurally invalid** for this tier, so it degrades to ``""``
    (gmem-direct) rather than failing the lowering — the same pin-validity rule the other
    codecs follow."""
    if axis_role(kernel.op) is not AxisRole.CONTRACTION:
        return ""
    pinned = STAGE.narrow([""])[0]
    if not pinned:
        return ""
    try:
        Stage.parse(pinned)
    except ValueError:
        return ""
    return pinned


def _wspec_workers(stage) -> tuple[WarpSpec | None, str]:
    """The pinned ``WSPEC`` worker split for a pipeline with the given ``stage``, or ``(None, "")`` —
    uniform SIMT. Pin-only this cut: returns the authoritative ``DEPLODOCK_WSPEC`` pin (``Knob.narrow``)
    when it parses AND every role is legal (a producer needs a ``stage`` to drive); a pin that doesn't
    parse, names no role, or whose roles are illegal degrades to uniform — the same pin-validity rule the
    other codecs follow. The second element is the spec to restamp on ``knobs`` (``""`` when uniform)."""
    pinned = WSPEC.narrow([""])[0]
    if not pinned:
        return None, ""
    try:
        ws = WarpSpec.parse(pinned)
    except ValueError:
        return None, ""
    # ``is_legal`` reads only ``.stage`` off its arg (the producer-needs-a-stage rule) — pass a probe.
    if not ws.roles or not ws.is_legal(SimpleNamespace(stage=stage)):
        return None, ""
    return ws, pinned


def _check_warp_static_k(kernel, wt) -> None:
    """Reject a warp pin whose **static** contraction K is not a multiple of the inner mma
    K-step (``atom_k · bk``). The warp K-loop has no static-K tail handling — a partial final
    K-step reads past the operand and silently corrupts the result (max error ≫ tol, yet the
    output's *mean* error stays small so the accuracy gate passes it). A **symbolic** K is
    fine: it reaches the masked tier (ceil-div grid + boundary ``Cond`` + zero-filled partial
    slab), so guard only the static case. Raising here surfaces a clean compile error instead
    of a numerically-wrong kernel."""
    ext = reduce_loop(kernel.op).axis.extent
    if not ext.is_static:
        return
    k = ext.as_static()
    step = wt.atom.atom_k * wt.bk
    if k % step:
        raise ValueError(
            f"warp TILE pin K-step {step} (atom_k={wt.atom.atom_k}·bk={wt.bk}) does not divide the "
            f"static contraction K={k}; the warp K-loop has no static-K tail masking yet, so a "
            f"partial final step corrupts the result. Pin a K that is a multiple of {step}, or "
            f"drop the a:<atom> token to use the scalar tier."
        )


def _contraction_node(node, place, tile_plan: TilePlan) -> Contraction:
    """The high-level :class:`Contraction` structural node for a tiled ``CONTRACTION`` leaf, built
    here at fork-emit (seam #1 — the node must exist recognize-side so its ``tile`` rides the node,
    not a root schedule field; the build moved off ``010_materialize``'s retired
    ``_build_contraction``). Resolves the ``(a_load, b_load, acc, epilogue)`` operand→role facts
    structurally (:func:`semiring_binding`) — raising ``LoweringError`` on an unbindable atom — plus
    the resolved ``tile_plan`` from the schedule fork, and the (m, n) output / K axes off the
    still-``Map`` ``node``. The projection ``epilogue`` is the binding's body verbatim — the
    synthesized grid-``Write`` for a bare contraction stays a materialize concern (it needs
    ``root.output``), appended there when the epilogue carries no ``Write``."""
    grid = list(place.grid)
    a_load, b_load, acc, epilogue = semiring_binding(node, place.grid)
    return Contraction(
        axes=(grid[-2], grid[-1]),
        k_axis=reduce_loop(node).axis,
        a_operand=a_load,
        b_load=b_load,
        acc=acc,
        tile=tile_plan,
        lead_axes=tuple(grid[:-2]),
        epilogue=epilogue,
    )


def _factor_k(k_axis: Axis, w: int) -> tuple[Axis, Axis, Sigma]:
    """Factor a **static** contraction axis ``k`` into ``ksplit × kslice`` for split-K.

    ``ksplit`` (extent ``w``, name ``<k>_ks``) is the outer *partition index* — becomes the
    :class:`Reduction`'s reduce axis, parallelized across CTAs and summed in the finalize; ``kslice``
    (extent ``K/w``, the **original** name) is the per-partition chunk — stays the inner
    :class:`Contraction`'s ``k_axis``. The returned ``sigma`` maps the original ``k`` var to
    ``ksplit·(K/w) + kslice`` so the operand loads reconstruct the absolute index. Distinct names
    (``k`` vs ``<k>_ks``) are what avoid a double-reduce ``for k:[for k:]`` — every original ``k`` is
    visited once (``kslice`` folded into a partial, ``ksplit`` summed across partials)."""
    big_k = k_axis.extent.as_static()
    b = big_k // w
    ksplit = Axis(name=f"{k_axis.name}_ks", extent=Dim(w))
    kslice = replace(k_axis, extent=Dim(b))
    sigma = Sigma({k_axis.name: BinaryExpr("+", BinaryExpr("*", Var(ksplit.name), Literal(b, "int")), Var(k_axis.name))})
    return ksplit, kslice, sigma


def _splitk_option(tile, place, tile_spec: str, split_spec: str, name: str, knobs: dict, stage_spec: str = "") -> TileOp:
    """One scheduled **split-K** contraction ``TileOp``: the structural ``Reduction(axis=ksplit,
    source=Contraction(k_axis=kslice))``. The inner :class:`Contraction` is the **same** node a
    non-split matmul builds (:func:`_contraction_node`, so it factorizes through ``_factor`` to mma or
    scalar per the ``tile_spec`` atom) but over ``kslice`` with operands reindexed to
    ``ksplit·(K/w) + kslice``; the outer additive :class:`Reduction` carries the ``g<w>[a|k]`` GRID
    partition (:class:`ReducePlan`) that ``030_split`` consumes into the cross-CTA partial + finalize.

    The additive carrier is built exactly as ``contraction_loop`` / a plain-sum reduce does — an
    ``Accum(op="add").as_carrier()`` (identity ``0.0``, 1 component) — so ``030_split``'s finalize
    (which reads the carrier's identity + ``as_state_merge``) needs no change. The output tile
    (``tier``) rides the inner ``Contraction``; the ``Reduction`` holds only the K partition.

    Knob keying: ``TILE`` / ``REDUCE`` are stamped on the **original** k-axis name (not
    ``ksplit`` / ``kslice``), keeping the kernel single-eligible-axis so golden bare-collapse + the
    prior featurizer stay invariant vs the residual/golden spelling."""
    wt = TilePlan.parse(tile_spec)
    inner = _contraction_node(tile.op, place, wt)
    w = ReducePlan.parse(split_spec).cta
    # A warp (mma) slice must keep the inner K-step dividing K/w — the warp K-loop has no static-K
    # tail masking (same guard as ``_check_warp_static_k``, but on the post-split slice).
    if wt.is_warp:
        step = wt.atom.atom_k * wt.bk
        ks = inner.k_axis.extent.as_static() // w
        if ks % step:
            raise ValueError(
                f"split-K slice K={ks} (K/{w}) is not a multiple of the mma K-step {step} "
                f"(atom_k={wt.atom.atom_k}·bk={wt.bk}); pick a split width whose slice is divisible."
            )
    ksplit, kslice, sigma = _factor_k(inner.k_axis, w)
    inner = replace(
        inner,
        k_axis=kslice,
        a_operand=replace(inner.a_operand, index=tuple(sigma.apply(e) for e in inner.a_operand.index)),
        b_load=replace(inner.b_load, index=tuple(sigma.apply(e) for e in inner.b_load.index)),
    )
    carrier = Accum(name=inner.acc, value=f"{inner.acc}__v", op=ElementwiseImpl("add"), dtype=F32).as_carrier()
    op = Reduction(carrier=carrier, axis=ksplit, role=AxisRole.CONTRACTION, source=inner, reduce=ReducePlan.parse(split_spec))
    kaxis = reduce_loop(tile.op).axis.name  # the ORIGINAL k-axis name — single-eligible-axis keying
    stage = Stage.parse(stage_spec) if stage_spec else None
    stamped = {**knobs, _at(TILE, kaxis): tile_spec, _at(REDUCE, kaxis): split_spec}
    if stage_spec:
        stamped[_at(STAGE, kaxis)] = stage_spec
    return TileOp(op=op, name=name, place=place, tier=inner.tile, stage=stage, knobs=stamped)


def _warp_option(tile, place, spec: str, name: str, knobs: dict, stage_spec: str = "") -> TileOp:
    """One scheduled warp-tier contraction ``TileOp``: ``place`` mapped onto the grid + the warp
    form of the ``TILE`` spec resolved into the warp-atom :class:`TilePlan`, plus an optional operand
    ``STAGE`` resolved into a :class:`Stage`. The tiled :class:`Contraction` leaf is built here (``op``),
    so materialize only ``factorize``\\ s. The packed ``TILE`` codec is the sole on-dict spelling — the
    learned-prior featurizer parses it directly (one codec, not a per-knob ``WM``/``WN``/``MMA`` explosion)."""
    wt = TilePlan.parse(spec)
    _check_warp_static_k(tile, wt)
    stage = Stage.parse(stage_spec) if stage_spec else None
    # Build the tiled Contraction node here — it resolves the operand→role facts internally, so an
    # unbindable atom (a non-Load operand: a computed-cone / demoted matmul) raises and is rejected
    # at fork construction, like the static-K check.
    op = _contraction_node(tile.op, place, wt)
    # Warp specialization rides ORTHOGONAL to the tile/stage just resolved: an optional WSPEC pin
    # splits the warps into roles over this fixed pipeline (gated on the ``stage``).
    workers, wspec_spec = _wspec_workers(stage)
    # The per-node schedule codecs key ``@<k_axis>`` (the contraction axis this node schedules), so a
    # multi-node kernel can address each node; ``WSPEC`` stays root-global (bare).
    kaxis = op.k_axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec}
    if stage_spec:
        stamped[_at(STAGE, kaxis)] = stage_spec
    if wspec_spec:
        stamped[WSPEC.name] = wspec_spec
    return TileOp(op=op, name=name, place=place, tier=wt, stage=stage, workers=workers, knobs=stamped)


def _tile_option(tile, place, spec: str, name: str, knobs: dict, reduce_spec: str = "", stage_spec: str = "") -> TileOp:
    """One scheduled scalar-tier contraction ``TileOp``: ``place`` mapped onto the grid + the ``TILE``
    spec resolved into the ``TilePlan`` (an optional cooperative / ILP ``REDUCE`` spec into the
    orthogonal residual ``ReducePlan``, an optional operand ``STAGE`` into the :class:`Stage`), the
    specs stamped on ``knobs`` for the prior. ``reduce_spec`` is the ``b`` / ``r`` K partition only —
    the cross-CTA split-K ``g`` rides the separate structural :func:`_splitk_option` fork."""
    stage = Stage.parse(stage_spec) if stage_spec else None
    plan = TilePlan.parse(spec)
    # The scalar tile's CTA launches ``par_n · par_m`` threads (one per parallel output cell,
    # each owning a ``reg_n · reg_m`` register sub-tile). Reject a parallel tile over the
    # 1024-thread/CTA hardware limit — otherwise the launch fails late with an opaque
    # ``CUDA_ERROR_INVALID_VALUE`` instead of a clear compile-time error.
    block = plan.block_threads
    if block > _MAX_BLOCK_THREADS:
        raise ValueError(
            f"TILE parallel block {plan.units_n}×{plan.units_m}={block} threads exceeds the "
            f"{_MAX_BLOCK_THREADS}-thread/CTA limit; shrink n/m or move work to the f register sub-tile."
        )
    # A tiled register-tile leaf (a ``TILE`` pin) becomes a :class:`Contraction` node here, so
    # materialize only ``factorize``\\ s. An unbindable contraction (a non-``Load`` operand) keeps the
    # ``Map`` form — materialize's per-cell scalar tier lowers it. A coop / ILP ``reduce_spec`` keeps
    # the ``Map`` too (the K partition rides the residual ``reduce``, folded by ``_factorize_reduce``).
    op = tile.op
    if plan.is_tiled and not reduce_spec:
        try:
            op = _contraction_node(tile.op, place, plan)
        except LoweringError:
            pass  # an unbindable contraction (a non-Load operand) keeps the Map form
    # ``TILE`` / ``REDUCE`` / ``STAGE`` key ``@<k_axis>`` (the contraction axis this node schedules),
    # unifying the schedule onto the axis-named family.
    kaxis = reduce_loop(tile.op).axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec}
    if reduce_spec:
        stamped[_at(REDUCE, kaxis)] = reduce_spec
    if stage_spec:
        stamped[_at(STAGE, kaxis)] = stage_spec
    return TileOp(op=op, name=name, place=place, tier=plan, reduce=ReducePlan.parse(reduce_spec), stage=stage, knobs=stamped)


def schedule(tile: TileOp, name: str, knobs: dict) -> list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling forks —
    the scheduling half of ``010_recognize``, called inline once recognition has built the tile op.
    ``tile`` is an unmapped :class:`TileOp` (its ``op`` set, ``place`` carrying just the free axes).
    Returns a single scheduled ``TileOp`` (no fork) or a list of candidate ``TileOp``\\ s (the search /
    prior ranks them). ``knobs`` is the recognized kernel's knob base (empty for a fresh kernel)."""
    place = tile.place.on_grid()
    # The mma-flash tree (the ``DEPLODOCK_CHAIN`` warp chain) is already fully scheduled
    # recognize-side — both contractions carry their warp ``TilePlan`` and the query-tile grid
    # rides ``place`` — so it bypasses the generic reduce/contraction fork and lowers straight
    # through the flash-warp emitter (``_factor`` dispatches on ``is_mma_flash``).
    if _is_mma_flash(tile.op):
        return TileOp(op=tile.op, name=name, place=place, knobs=knobs)
    # Dispatch on the axes' role, not a kernel kind: a pointwise (FREE) kernel has no reduce
    # decision — just map the grid (the off-default stamps ``REDUCE=""``). A reduction offers its
    # ``REDUCE`` candidate(s); a contraction offers its output ``TILE``. One candidate applies
    # directly; multiple fork for the search / prior to rank.
    role = axis_role(tile.op)
    if role is AxisRole.FREE:
        return TileOp(op=tile.op, name=name, place=place)
    # A contraction picks its free-axis output tile (``TILE``); a reduction picks its reduce
    # partition (``REDUCE``). Each offers its candidate(s): one applies directly, multiple fork.
    # A contraction ALSO honors a cross-CTA split-K (``g``) / cooperative (``b``/``r``) ``REDUCE``
    # pin — orthogonal to the output tile (``reduce`` = the K partition; ``g`` is consumed by
    # ``030_split``, ``b``/``r`` by ``_factor._factorize_reduce`` on the non-tiled scalar tier).
    # ``TILE`` is the unified output-fragment knob: a candidate whose codec names an atom
    # (``a:<atom>`` — :func:`is_warp_codec`) builds the tensor-core warp option, otherwise the
    # scalar register-tile option (the either-ness — a kernel is one fragment or the other).
    if role is AxisRole.CONTRACTION:
        stage_spec = _stage_spec(tile)
        # A pinned cross-CTA split-K (``g<w>[a|k]``) routes EVERY tile candidate (scalar or mma)
        # through the structural ``Reduction ⊃ Contraction`` fork — one split-K path, consumed by
        # ``030_split`` (the partial is a bare ``Contraction`` that factorizes to mma / scalar).
        split_spec = _splitk_pin()
        if split_spec:
            return [_splitk_option(tile, place, spec, split_spec, name, knobs, stage_spec) for spec in _tile_specs(tile)]
        # A non-split cooperative / ILP (``b`` / ``r``) K partition rides the residual ``reduce`` on the
        # scalar tier (``_factorize_reduce``); orthogonal to the output tile.
        reduce_spec = _coop_reduce_spec()
        return [
            _warp_option(tile, place, spec, name, knobs, stage_spec)
            if is_warp_codec(spec)
            else _tile_option(tile, place, spec, name, knobs, reduce_spec, stage_spec)
            for spec in _tile_specs(tile)
        ]
    specs = _reduce_specs(tile, place)
    return [_option(tile, place, spec, name, knobs) for spec in specs]
