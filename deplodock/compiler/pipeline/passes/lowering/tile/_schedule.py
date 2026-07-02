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
from deplodock.compiler.ir.atom import ATOM_REGISTRY
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.schedule import Stage, WarpSpec, is_warp_codec
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt
from deplodock.compiler.ir.tile import Contraction, Map, Placement, ReducePlan, Reduction, TileOp, TilePlan
from deplodock.compiler.ir.tile.ops import axis_role, nodify_reduce, reduce_loop
from deplodock.compiler.pipeline.forks import REDUCE, STAGE, TILE, WSPEC
from deplodock.compiler.pipeline.passes.lowering.tile._atomize import semiring_binding
from deplodock.compiler.pipeline.passes.lowering.tile._catalog import scalar_tile_moves
from deplodock.compiler.pipeline.pipeline import LoweringError

# The schedule codec knobs (``REDUCE`` / ``TILE`` / ``STAGE`` / ``WSPEC``) are declared in
# ``pipeline/forks.py`` (split out of ``knob.py``) and imported here, where they are
# resolved into the schedule slices. The decision hierarchy for each is the env pin (via
# ``Knob.narrow``) > the search/prior fork > the conservative default below.


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


# ---- shared-row operand staging (the fused norm→linear prologue) -------------------------------- #
# The reduce tier's one staging move: when an input row is folded by the cooperative reduce AND
# re-read per output column of a contraction tail (the fused RMSNorm→linear shape), stage it into
# smem once and share it across both readers. The DETECTION lives here — stamped as a first-class
# ``sync`` :class:`Stage` whose ``smem`` names the row buffer — and ``_factor._tile_reduce_axis`` only
# APPLIES it (fill + load-rewrite), the same Stage → apply path the contraction tiers follow. Not a
# knob: it fires whenever the cooperative partition is chosen and the shape qualifies (a pure perf
# transform), so nothing is spelled on ``knobs`` and the prior featurization is untouched.


def _scalar_loads(stmts: list[Stmt]) -> list[Load]:
    """Every scalar ``Load`` reachable in ``stmts`` (deep)."""
    out: list[Load] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            out.append(s)
        for b in s.nested():
            out.extend(_scalar_loads(list(b)))
    return out


def _has_accum(stmts: list[Stmt]) -> bool:
    return any(isinstance(s, Accum) or any(_has_accum(list(b)) for b in s.nested()) for s in stmts)


def _has_contraction_tail(stmts: list[Stmt]) -> bool:
    """The post-reduce tail contracts over a NEW free axis — a ``Loop`` (the free output
    axis) whose body holds an inner reduce ``Loop`` (an ``Accum``). This is the fused
    norm→linear shape (``for n: for k: acc += …``), and it distinguishes it from a plain
    softmax tail (a single ``for k`` sum over the SAME reduce axis, no nested contraction).
    Only the former benefits from staging the shared input row — and only it is staged."""
    for s in stmts:
        if isinstance(s, Loop) and any(isinstance(c, Loop) and _has_accum(list(c.body)) for c in s.body):
            return True
        if any(_has_contraction_tail(list(b)) for b in s.nested()):
            return True
    return False


def _shared_row_buf(carrier_body, tail: list[Stmt], grid_vars: tuple, raxis: Axis, inputs: dict) -> str | None:
    """The input buffer reused as a CTA-shared ROW across the reduce + a contraction tail — an
    input read in the carrier reduce at ``(grid…, raxis)`` AND in the tail at ``(grid…, k)``,
    whose trailing dim is the (static) reduce extent. That row (e.g. RMSNorm's ``x[m, :]``,
    folded by the mean reduce then re-read per output column of the fused linear) is the one
    operand worth staging into smem. ``None`` ⇒ no eligible operand (stay gmem-direct)."""
    if not raxis.extent.is_static or not _has_contraction_tail(tail):
        return None
    n = len(grid_vars)
    carrier_bufs = {
        s.input
        for s in _scalar_loads(list(carrier_body))
        if len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars and s.index[-1] == Var(raxis.name)
    }
    for s in _scalar_loads(tail):
        if s.input in carrier_bufs and len(s.index) == n + 1 and tuple(s.index[:n]) == grid_vars:
            t = inputs.get(s.input)
            if t is not None and t.shape and t.shape[-1].is_static and t.shape[-1].as_static() == raxis.extent.as_static():
                return s.input
    return None


def _row_stage(tile, place) -> Stage | None:
    """The shared-row :class:`Stage` for a **cooperative** reduce ``tile``, or ``None`` (no eligible
    row — gmem-direct). Reads the reduce loop / projection tail off the node tree (the same stmts the
    materializer emits) and the operand shapes off ``tile.inputs`` (seeded from the recognized
    ``LoopOp``); the stamped stage is the depth-1 ``sync`` transport with ``smem`` naming the row."""
    rloop = reduce_loop(tile.op)
    tail = list(tile.op.body) if isinstance(tile.op, Map) else []
    grid_vars = tuple(Var(a.name) for a in place.grid)
    buf = _shared_row_buf(rloop.body, tail, grid_vars, rloop.axis, tile.inputs)
    return Stage(transport="sync", smem=(buf,)) if buf is not None else None


def _option(tile, place, spec: str, name: str, knobs: dict) -> TileOp:
    """One scheduled ``TileOp``: ``place`` mapped onto the grid + the ``REDUCE`` spec resolved into the
    :class:`Reduction` node's ``ReducePlan`` (the ephemeral knob → materialized plan stamped **on the
    node**), with the spec stamped on ``knobs`` for the prior. The spec is keyed ``REDUCE@<axis>``
    (the reduce axis this node partitions), so a multi-node kernel addresses each reduce. A
    cooperative partition also derives the shared-row operand :class:`Stage` (:func:`_row_stage`,
    stamped on the schedule field only — a derived perf transform, never a knob)."""
    plan = ReducePlan.parse(spec)
    op = _with_reduce(tile.op, plan)
    raxis = reduce_loop(tile.op).axis.name
    stage = _row_stage(tile, place) if plan.coop > 1 else None
    return TileOp(op=op, name=name, place=place, stage=stage, knobs={**knobs, _at(REDUCE, raxis): spec})


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
    honors — folded through ``_factor._tile_reduce_axis`` (a contraction is the degenerate carrier of
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
    codecs follow. The returned spec is only the *spelling* (stamped on ``knobs``); each option
    builder RESOLVES it against its built node (:func:`_resolve_warp_stage` /
    :func:`_resolve_scalar_stage`) into the ``Stage`` it stamps."""
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


# ---- contraction operand-stage RESOLUTION (eligibility + sizing, once, scheduler-side) ---------- #
# A ``STAGE`` pin on a contraction is resolved HERE against the built :class:`Contraction` node —
# transport eligibility, the slab K-chunk (``bk_elems``), and the depth clamps — and the RESOLVED
# :class:`Stage` (or ``None``, gmem-direct) is stamped on the ``TileOp``. The materializer
# (``_atom._staged``) applies it verbatim, deciding nothing — the same stamp-then-apply shape as the
# reduce tier's shared-row stage (:func:`_row_stage`). The raw pin string still rides ``knobs`` for
# the prior, so featurization is untouched by resolution.


def _can_stage_warp(stage, k_axis: Axis, tile_m: int, tile_n: int, bk: int, atom_k: int, mask_m: bool, mask_n: bool, b_trans: bool) -> bool:
    """cp.async staging eligibility: a ``cp.async`` stage over a contraction with a STATIC,
    tile-divisible K axis and a canonical (non-transposed) B operand. A masked / symbolic **M**
    (output rows) is fine — the A-slab fill clamp-reads the overhanging rows in-bounds and the
    ``RegStore`` guards their store. A masked **N** (the B-slab inner dim) and a symbolic /
    non-divisible **K** stay gmem-direct (K zero-fill is a follow-up). Staging only ever *adds* a
    faster lowering, so an ineligible kernel silently falls back to gmem-direct."""
    if stage is None or stage.transport != "cp.async" or b_trans or mask_n:
        return False
    if not k_axis.extent.is_static:
        return False
    bk_elems = bk * atom_k
    if k_axis.extent.as_static() % bk_elems != 0:
        return False
    # cp.async needs a ≥4-byte contiguous chunk; the 16-bit mma operands give 2 B/elem, so the
    # inner slab dim must be even (A's BK, B's tile_n). Odd ⇒ fall back.
    return (bk_elems % 2 == 0) and (tile_n % 2 == 0)


def _can_stage_warp_tma(
    stage, k_axis: Axis, n_axis: Axis, tile_n: int, bk: int, atom_k: int, elem_bytes: int, mask_n: bool, b_trans: bool
) -> bool:
    """TMA (``cp.async.bulk.tensor``) staging eligibility: a ``tma`` stage over a contraction with a
    STATIC, tile-divisible K and a canonical B. A masked / symbolic **M** is fine — the descriptor's
    globalDim is the runtime M and TMA zero-fills the box overhang past it (no fill clamp needed). A
    masked **N** and a symbolic / non-divisible **K** stay gmem-direct. The box's inner dim (A's BK,
    B's tile_n) and the source's inner global stride (A's K, B's N) must be 16 B-aligned (the
    NONE-swizzle TMA box-copy rule)."""
    if stage is None or stage.transport != "tma" or b_trans or mask_n:
        return False
    if not (k_axis.extent.is_static and n_axis.extent.is_static):
        return False
    bk_elems = bk * atom_k
    k, n = k_axis.extent.as_static(), n_axis.extent.as_static()
    if k % bk_elems != 0:
        return False
    return all((x * elem_bytes) % 16 == 0 for x in (bk_elems, tile_n, k, n))


def _resolve_warp_stage(c: Contraction, stage: Stage) -> Stage | None:
    """Resolve a pinned operand ``Stage`` against the warp (mma) contraction ``c`` — TMA > cp.async >
    gmem-direct (``None``). The resolved stage carries ``bk_elems`` (the codec-spelled ``TilePlan.bk``
    in elements), ``depth`` clamped so the ring's slots fit the 48 KiB smem budget (dropping ``ring``
    when the clamp leaves nothing to cycle), and ``reg_depth`` clamped to ``bk`` (nothing to ping-pong
    past the resident chunk)."""
    atom = c.atom
    a_nbytes = atom.operand_dtype("a").nbytes
    bk = c.tile.bk
    m, n = c.m, c.n
    tma_ok = _can_stage_warp_tma(stage, c.k_axis, n.axis, n.tile, bk, atom.atom_k, a_nbytes, n.mask, c.b_trans)
    cp_ok = (not tma_ok) and _can_stage_warp(stage, c.k_axis, m.tile, n.tile, bk, atom.atom_k, m.mask, n.mask, c.b_trans)
    if not (tma_ok or cp_ok):
        return None
    bk_elems = bk * atom.atom_k
    slot_bytes = (m.tile + n.tile) * bk_elems * a_nbytes
    depth = min(stage.depth, max(1, (48 * 1024) // slot_bytes))
    return replace(stage, depth=depth, ring=stage.ring and depth >= 2, reg_depth=min(stage.reg_depth, bk), bk_elems=bk_elems)


def _resolve_scalar_stage(c: Contraction, stage: Stage, inputs) -> Stage | None:
    """Resolve a pinned operand ``Stage`` against the scalar register-tile contraction ``c``, or
    ``None`` (gmem-direct). Staging is **opt-in behind a ``STAGE`` pin**: eligible when the transport
    is ``tma`` / ``cp.async`` and K is static (a computed-A contraction never reaches here — it keeps
    the ``Map`` form). A masked (overhanging) M / N is fine — the drain reads the slab by LOCAL tile
    coords and the overhanging store is guarded, so TMA zero-fills the box overhang and cp.async
    clamps the gmem read. The slab K-chunk ``bk_elems`` is **derived** to fit a single
    ``tile_m×bk + bk×tile_n`` operand slab in 48 KiB (largest power-of-two dividing K; ``inputs``
    supplies the element dtype) — not spelled by a codec, so no schema change. The resolved stage is
    single-buffer (``depth == 1``; the scalar gmem→smem ring is a follow-on)."""
    if not c.k_axis.extent.is_static or stage.transport not in ("tma", "cp.async"):
        return None
    if not inputs or c.a_operand.input not in inputs:
        return None
    K = c.k_axis.extent.as_static()
    elem_bytes = inputs[c.a_operand.input].dtype.nbytes
    cap = (48 * 1024) // (max(1, c.m.tile + c.n.tile) * elem_bytes)
    bk_elems = next((v for v in (128, 64, 32, 16, 8, 4) if v <= cap and K % v == 0), 0)
    if bk_elems < 4:
        return None
    return replace(stage, depth=1, ring=False, reg_depth=1, bk_elems=bk_elems)


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
    # Resolve the STAGE pin against the SLICED inner contraction (K/w — the K the partial kernel
    # sees). NOTE: ``030_split`` does not thread ``stage`` onto its partial ``TileOp``s today, so a
    # split-K partial always materializes gmem-direct; the resolved stamp here is what a future
    # threading consumes.
    stage = None
    if stage_spec:
        parsed = Stage.parse(stage_spec)
        stage = _resolve_warp_stage(inner, parsed) if wt.is_warp else _resolve_scalar_stage(inner, parsed, tile.inputs)
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
    # Build the tiled Contraction node here — it resolves the operand→role facts internally, so an
    # unbindable atom (a non-Load operand: a computed-cone / demoted matmul) raises and is rejected
    # at fork construction, like the static-K check.
    op = _contraction_node(tile.op, place, wt)
    stage = _resolve_warp_stage(op, Stage.parse(stage_spec)) if stage_spec else None
    # Warp specialization rides ORTHOGONAL to the tile/stage just resolved: an optional WSPEC pin
    # splits the warps into roles over this fixed pipeline (gated on the RESOLVED ``stage`` — an
    # ineligible pin leaves no pipeline for a producer to drive, so WSPEC degrades to uniform).
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
    spec resolved into the ``TilePlan`` (an optional cooperative / ILP ``REDUCE`` spec **nodifying** the
    contraction to a :class:`Reduction` node carrying the K partition, an optional operand ``STAGE`` into
    the :class:`Stage`), the specs stamped on ``knobs`` for the prior. ``reduce_spec`` is the ``b`` / ``r``
    K partition only — the cross-CTA split-K ``g`` rides the separate structural :func:`_splitk_option`
    fork."""
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
    # ``Map`` form — materialize's per-cell scalar tier lowers it. A coop / ILP ``reduce_spec``
    # **nodifies** the flat ``Map`` contraction to a :class:`Reduction` node carrying the K partition
    # (:func:`nodify_reduce`), so the plan rides the node — not a residual ``TileOp.reduce`` field —
    # and ``_factor._tile_reduce_axis`` folds it off the node.
    op = tile.op
    stage = None
    if plan.is_tiled and not reduce_spec:
        try:
            op = _contraction_node(tile.op, place, plan)
        except LoweringError:
            pass  # an unbindable contraction (a non-Load operand) keeps the Map form
        else:
            # Only a built Contraction node can engage operand staging — resolve the pin against it
            # (per-cell / coop-K / unbindable forms stamp None: nothing downstream would read a stage).
            if stage_spec:
                stage = _resolve_scalar_stage(op, Stage.parse(stage_spec), tile.inputs)
    elif reduce_spec:
        op = nodify_reduce(tile.op, ReducePlan.parse(reduce_spec))
    # ``TILE`` / ``REDUCE`` / ``STAGE`` key ``@<k_axis>`` (the contraction axis this node schedules),
    # unifying the schedule onto the axis-named family.
    kaxis = reduce_loop(tile.op).axis.name
    stamped = {**knobs, _at(TILE, kaxis): spec}
    if reduce_spec:
        stamped[_at(REDUCE, kaxis)] = reduce_spec
    if stage_spec:
        stamped[_at(STAGE, kaxis)] = stage_spec
    return TileOp(op=op, name=name, place=place, tier=plan, stage=stage, knobs=stamped)


def schedule(tile: TileOp, name: str, knobs: dict) -> list[TileOp] | TileOp:
    """Map a freshly-recognized (UNMAPPED) ``tile`` onto the grid and offer its scheduling forks —
    the scheduling half of ``010_recognize``, called inline once recognition has built the tile op.
    ``tile`` is an unmapped :class:`TileOp` (its ``op`` set, ``place`` carrying just the free axes).
    Returns a single scheduled ``TileOp`` (no fork) or a list of candidate ``TileOp``\\ s (the search /
    prior ranks them). ``knobs`` is the recognized kernel's knob base (empty for a fresh kernel)."""
    place = tile.place.on_grid()
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
    # ``030_split``, ``b``/``r`` by ``_factor._tile_reduce_axis`` on the non-tiled scalar tier).
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
        # scalar tier (``_factor._tile_reduce_axis``); orthogonal to the output tile.
        reduce_spec = _coop_reduce_spec()
        return [
            _warp_option(tile, place, spec, name, knobs, stage_spec)
            if is_warp_codec(spec)
            else _tile_option(tile, place, spec, name, knobs, reduce_spec, stage_spec)
            for spec in _tile_specs(tile)
        ]
    # A TWISTED streaming reduce whose per-step partial is a contraction pair takes the WARP
    # (fragment-resident) tier when the mma atom is eligible — the conservative deterministic pick,
    # like the coop constants above (the tensor-core stream strictly dominates the redundant
    # per-cell scalar recompute). An explicit ``REDUCE`` pin is the scalar escape: it asks for a
    # reduce partition, which only the scalar tiers honor.
    if not REDUCE.narrow([""])[0]:
        warp = _twisted_warp_option(tile, name, knobs)
        if warp is not None:
            return warp
        # A PLANAR ⊗-fold over a computed MAP cone — the fused producer → matmul edge — honors a
        # warp ``TILE`` pin (pin-driven, like the matmul warp tier): the demoted contraction
        # nodifies with its computed A and the sync compute-fill stage.
        demoted = _demoted_warp_option(tile, place, name, knobs)
        if demoted is not None:
            return demoted
    specs = _reduce_specs(tile, place)
    return [_option(tile, place, spec, name, knobs) for spec in specs]


def _map_cone(body: list, root: str) -> list | None:
    """The backward cone of SSA ``root`` within ``body`` — the fused producer's compute, in body
    order. ``None`` unless every cone stmt is a scalar ``Load`` or a pointwise ``Assign`` (a pure
    MAP cone — a reduce-bearing cone, e.g. an rmsnorm scale, is not compute-fillable per cell)."""
    defs: dict[str, Stmt] = {}
    for st in body:
        for d in st.defines():
            defs[d] = st
    need, cone, seen = [root], [], set()
    while need:
        nm = need.pop()
        st = defs.get(nm)
        if st is None or id(st) in seen:
            continue
        seen.add(id(st))
        if isinstance(st, Load):
            cone.append(st)
            continue
        if isinstance(st, Assign):
            cone.append(st)
            need.extend(st.args)
            continue
        return None  # an Accum / Loop / Select in the cone — not a pure MAP producer
    order = {id(st): i for i, st in enumerate(body)}
    return sorted(cone, key=lambda st: order[id(st)])


def _demoted_warp_option(tile: TileOp, place, name: str, knobs: dict) -> TileOp | None:
    """The warp (mma) candidate for a **demoted-cone contraction** — a ``PLANAR`` ⊗-fold whose
    lift multiplies a gmem ``Load`` B with a computed pure-MAP cone A (the fused producer → matmul
    edge: ``f(x, …) @ w``), or ``None`` (stay scalar). PIN-DRIVEN like the matmul warp tier: fires
    only under a warp ``TILE`` pin. Nodifies the fold to a computed-A :class:`Contraction` (the
    same ``a_operand = Body`` the flash P@V rides) and stamps the ``sync`` compute-fill
    :class:`Stage` — the producer cone materializes the A tile straight into the smem slab the
    ``ldmatrix`` drain reads (the fused edge IS the mma tier's ``sync`` transport). First cut:
    exact-cover geometry only (static M/N/K divisible by the tile / K-chunk — no masked overhang),
    and the cone may read the ``(m, k)`` axes only."""
    spec = TILE.narrow([""])[0]
    if not is_warp_codec(spec):
        return None
    op = tile.op
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.PLANAR or red.source is not None or red.carrier.twist.family != "id":
        return None
    body = list(red.partial)
    accums = [st for st in body if isinstance(st, Accum)]
    if len(accums) != 1 or accums[0].op.name != "add":
        return None
    acc = accums[0]
    defs = {st.name: st for st in body if isinstance(st, Assign)}
    lift = defs.get(acc.value)
    if lift is None or lift.op.name != "multiply" or len(lift.args) != 2:
        return None
    grid = list(place.grid)
    if len(grid) < 2:
        return None
    m_ax, n_ax, k_ax = grid[-2], grid[-1], red.axis
    loads = {st.names[0]: st for st in body if isinstance(st, Load)}

    def _load_vars(nm: str) -> set | None:
        ld = loads.get(nm)
        return {v for e in ld.index for v in e.free_vars()} if ld is not None else None

    b_name = next((a for a in lift.args if (vs := _load_vars(a)) and n_ax.name in vs and k_ax.name in vs), None)
    if b_name is None:
        return None
    a_name = next(a for a in lift.args if a != b_name)
    cone = _map_cone(body, a_name)
    if cone is None or not cone:
        return None
    for st in cone:
        if isinstance(st, Load) and n_ax.name in {v for e in st.index for v in e.free_vars()}:
            return None  # the cone must be (m, k)-indexed — an n-dependent producer isn't the A tile
    wt = TilePlan.parse(spec)
    atom = wt.atom
    atom_m, atom_n, atom_k = atom.shape
    q_tensor = tile.inputs.get(next(st.input for st in cone if isinstance(st, Load))) if tile.inputs else None
    if getattr(getattr(q_tensor, "dtype", None), "name", None) != atom.ab_dtype:
        return None
    exts = (m_ax.extent, n_ax.extent, k_ax.extent)
    if not all(e.is_static for e in exts):
        return None
    M, N, K = (e.as_static() for e in exts)
    bk_elems = wt.bk * atom_k
    if K % bk_elems or M % (wt.units_m * wt.reg_m * atom_m) or N % (wt.units_n * wt.reg_n * atom_n):
        return None
    epilogue = Body(tuple(op.body)) if isinstance(op, Map) else Body(())
    node = Contraction(
        axes=(m_ax, n_ax),
        k_axis=k_ax,
        a_operand=Body(tuple(cone)),
        b_load=loads[b_name],
        acc=acc.name,
        tile=wt,
        lead_axes=tuple(grid[:-2]),
        epilogue=epilogue,
    )
    stage = Stage(transport="sync", smem=(node.a_name,), bk_elems=bk_elems)
    return TileOp(op=node, name=name, place=place, tier=wt, stage=stage, knobs={**knobs, _at(TILE, k_ax.name): spec})


def _twisted_warp_option(tile: TileOp, name: str, knobs: dict) -> TileOp | None:
    """The fragment-resident (tensor-core) candidate for a ``TWISTED`` streaming reduce, or ``None``
    (not eligible — the scalar options stand alone). Eligible when the tree is the streaming
    contraction pair — a head :class:`Contraction` with gmem ``Load`` operands producing the score
    and an expect :class:`Contraction` consuming a computed (register-resident) weight, under an
    exp-family carrier — and the mma atom's own demands hold (a 16-bit operand dtype; the head's
    contraction axis and the expect's output axis divisible by the atom; a static stream / query
    extent divisible by the block, since a static ragged tail has no fragment mask — the symbolic
    path masks at the fragment and guards the gmem reads). The same-per-node stamping rule as
    ``_warp_option``: the two contractions get their mma :class:`TilePlan`\\ s (one warp, the score
    block ``2·atom_n`` keys wide, the value dim folded into the expect tile), and the placement maps
    one warp per ``atom_m`` query rows — the value axis leaves the grid. An additive ``(m, kv)``
    score bias is not realizable at the fragment tier → ``None``."""
    op = tile.op
    red = op.source if isinstance(op, Map) and isinstance(op.source, Reduction) else (op if isinstance(op, Reduction) else None)
    if red is None or red.role is not AxisRole.TWISTED or red.carrier.twist.family != "exp" or len(red.partial) == 0:
        return None
    head = red.partial[0]
    if not isinstance(head, Contraction) or not isinstance(head.a_operand, Load):
        return None
    tail_contractions = [s for s in list(red.partial)[1:] if isinstance(s, Contraction)]
    if len(tail_contractions) != 1 or not tail_contractions[0].a_computed:
        return None
    pv = tail_contractions[0]
    channels = red.carrier.twist.channels
    if len(channels) != 3 or channels[1].lift is not None or channels[2].lift is None:
        return None
    q_tensor = tile.inputs.get(head.a_operand.input) if tile.inputs else None
    atom_name = {"f16": "mma_m16n8k16_f16", "bf16": "mma_m16n8k16_bf16"}.get(getattr(getattr(q_tensor, "dtype", None), "name", None))
    if atom_name is None:
        return None
    atom = ATOM_REGISTRY[atom_name]
    atom_m, atom_n, atom_k = atom.shape
    head_dim, d_v = head.k_axis.extent, pv.n_axis.extent
    if not (head_dim.is_static and head_dim.as_static() % atom_k == 0 and d_v.is_static and d_v.as_static() % atom_n == 0):
        return None
    bn = 2 * atom_n  # the streaming block: one double-atom key step
    kv_ext, m_ext = red.axis.extent, head.m_axis.extent
    if (kv_ext.is_static and kv_ext.as_static() % bn != 0) or (m_ext.is_static and m_ext.as_static() % atom_m != 0):
        return None
    m_name, kv_name = head.m_axis.name, red.axis.name
    for s in list(red.partial)[1:]:
        if isinstance(s, Load) and s.index and {m_name, kv_name} <= {v for e in s.index for v in e.free_vars()}:
            return None  # an additive (m, kv) score bias — fragment-unrealizable, stay scalar
    qk_plan = TilePlan(atom=atom, units=(1, 1), regs=(1, bn // atom_n), bk=head_dim.as_static() // atom_k)
    pv_plan = TilePlan(atom=atom, units=(1, 1), regs=(1, d_v.as_static() // atom_n), bk=1)
    partial = tuple(replace(s, tile=qk_plan) if s is head else (replace(s, tile=pv_plan) if s is pv else s) for s in red.partial)
    red2 = replace(red, partial=type(red.partial)(partial))
    op2 = replace(op, source=red2) if isinstance(op, Map) else red2
    # One warp per atom_m query rows: the query axis shrinks to its block count; the value (expect
    # output) axis folds into the fragment tile and leaves the grid.
    grid = tuple(
        Axis(name=ax.name, extent=ax.extent.ceil_div(atom_m), source_axis=ax.source_axis or ax) if ax.name == m_name else ax
        for ax in tile.place.free
        if ax.name != pv.n_axis.name
    )
    place = Placement(free=tile.place.free, grid=grid)
    return TileOp(op=op2, name=name, place=place, knobs={**knobs, _at(TILE, head.k_axis.name): qk_plan.spell()})
