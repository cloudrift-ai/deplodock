"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, etc.) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes (``001_tileify``, ``006a_register_tile_planned``,
``007_stage_inputs``, ...) read the tags and skip their own equivalent
decisions, doing only the leftover rewrites (lift to ``Tile.axes``,
replicate stmts, build stages).

**Matmul register tile** — detect matmul-shaped LoopOps, pre-split the
outer M / N output Loops by ``(FM, FN)`` from
:func:`tuning.register_tile_shape`, tag the inner halves
``Role.REGISTER``, and σ-substitute the body. ``001_tileify`` lifts
M_o / N_o to ``Tile.axes`` and stops at the REGISTER tags;
``006a_register_tile_planned`` replicates the per-cell bodies *before*
``007_stage_inputs`` runs.

**Matmul K chunking** — after the M/N register-tile decision, locate
the matmul K reduce and pre-split it into
``Loop(K_o, SERIAL_OUTER) → Loop(K_i, reduce, STAGE_INNER)`` with
σ: K → K_o*BK + K_i. The planner forks over a ``BK`` knob; greedy
callers pick variant 0 (the heuristic ``forced_bk`` value).

**Non-matmul chunk-reduce** — for each non-matmul reduce whose
K-indexed Loads project a thread-fanin slab over the 16 KB
``007_stage_inputs`` cap, split K → ``Loop(K_o, SERIAL_OUTER) →
Loop(K_i, reduce, STAGE_INNER)``.

**Matmul (BN, BM) stamp + SPLITK** — pure knob enumeration so
``004_launch_geometry`` / ``003_split_matmul_k`` can run deterministic.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    MAX_CELLS_PER_THREAD,
    TUNE_F_CHOICES,
    is_matmul_reduce,
)
from deplodock.compiler.tuning import (
    BodyInfo,
    auto_splitk,
    forced_bk,
    register_tile_shape,
    thread_tile_shape,
)

PATTERN = [Pattern("root", LoopOp)]

# Knob stamp signalling the planner produced output (for planner-side
# idempotence — re-firing on a planned LoopOp is a no-op). Downstream
# 006a uses ``Role.REGISTER`` presence + ``FN`` absence as its trigger.
_PLANNER_KNOB = "PLANNER"

# Matches the legacy 002_chunk_matmul_k BK candidate set.
_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)

BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")

# Mirrors ``004_launch_geometry._TUNE_AXIS_CHOICES`` so planner-driven
# matmul BN/BM forks enumerate the same candidate space as the legacy
# 004 fork did.
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_WARP_SIZE = 32

# Mirrors ``003_split_matmul_k._SPLITK_CANDIDATES``.
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    loop_op: LoopOp = root.op
    if loop_op.knobs.get(_PLANNER_KNOB):
        raise RuleSkipped("already planned")

    body_info = BodyInfo.of(loop_op.body)

    reg_variants = _try_matmul_register_tile(loop_op, body_info)
    base_variants: list[LoopOp] = list(reg_variants) if reg_variants is not None else [loop_op]

    after_k: list[LoopOp] = []
    any_k_chunk = False
    for base in base_variants:
        k_variants = _try_matmul_k_chunk(ctx, base, body_info)
        if k_variants is not None:
            any_k_chunk = True
            after_k.extend(k_variants)
        else:
            after_k.append(base)

    after_splitk: list[LoopOp] = []
    any_splitk = False
    for cand in after_k:
        sk_variants = _try_splitk(cand, body_info)
        if sk_variants is not None:
            any_splitk = True
            after_splitk.extend(sk_variants)
        else:
            after_splitk.append(cand)

    chunked: list[LoopOp] = []
    any_chunk_reduce = False
    for cand in after_splitk:
        after_cr = _try_chunk_reduce(cand, body_info)
        if after_cr is not None:
            any_chunk_reduce = True
            chunked.append(after_cr)
        else:
            chunked.append(cand)

    bn_bm_results: list[LoopOp] = []
    any_bn_bm = False
    for cand in chunked:
        bn_bm_variants = _try_matmul_bn_bm_fork(cand, body_info)
        if bn_bm_variants is not None:
            any_bn_bm = True
            bn_bm_results.extend(bn_bm_variants)
        else:
            bn_bm_results.append(cand)

    fired = reg_variants is not None or any_k_chunk or any_splitk or any_chunk_reduce or any_bn_bm
    if not fired:
        raise RuleSkipped("no planner branch matched")

    chunked = bn_bm_results

    if len(chunked) == 1:
        return _stamp_planned(chunked[0])
    return [_stamp_planned(v) for v in chunked]


def _stamp_planned(op: LoopOp) -> LoopOp:
    knobs = dict(op.knobs)
    knobs[_PLANNER_KNOB] = True
    return LoopOp(body=op.body, knobs=knobs)


# --- chain extent helpers --------------------------------------------


def _outer_free_loop_chain(body) -> tuple[Loop, ...]:
    """Walk the outer single-stmt chain of untagged free Loops outermost-
    first. Mirrors ``001_tileify._strip_outer_free_chain``."""
    out: list[Loop] = []
    cur = tuple(body)
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        out.append(cur[0])
        cur = tuple(cur[0].body)
    return tuple(out)


def _chain_extents_desc(chain: tuple[Loop, ...]) -> tuple[int, ...]:
    """Extents of the chain, sorted descending. Mirrors what
    ``_logical_output_extents`` produced for the planner's synthetic
    Tiles (all-THREAD axes, no folding)."""
    return tuple(sorted((int(lp.axis.extent) for lp in chain), reverse=True))


# --- matmul register-tile branch -------------------------------------


def _try_matmul_register_tile(loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
    """Detect a matmul-shape LoopOp; if eligible, fork over ``(FM, FN)``
    candidates from :data:`TUNE_F_CHOICES`. For each viable pair, emit
    a LoopOp variant whose outer M / N output Loops are pre-split by
    ``(FM, FN)`` with ``Role.REGISTER`` on the inner halves. Stamps
    ``knobs={"FM": fm, "FN": fn}``.

    Heuristic shape (from :func:`tuning.register_tile_shape`) is emitted
    as variant 0. ``(1, 1)`` is included so the autotuner can elect the
    no-register-tile shape too.

    Returns ``None`` when the kernel isn't matmul-shaped or has fewer
    than two outer free Loops."""
    if not body_info.has_matmul:
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None
    outer_m, outer_n = chain[0], chain[1]
    ext_m, ext_n = int(outer_m.axis.extent), int(outer_n.axis.extent)

    output_extents = _chain_extents_desc(chain)
    thread_extents = tuple(int(lp.axis.extent) for lp in chain)
    h_fm, h_fn = (int(x) for x in register_tile_shape(output_extents, thread_extents, body_info))

    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    def _add(fm: int, fn: int) -> None:
        if (fm, fn) in seen:
            return
        if fm < 1 or fn < 1:
            return
        if ext_m % fm != 0 or ext_n % fn != 0:
            return
        if fm * fn > MAX_CELLS_PER_THREAD:
            return
        seen.add((fm, fn))
        ordered.append((fm, fn))

    _add(h_fm, h_fn)
    for fm in TUNE_F_CHOICES:
        for fn in TUNE_F_CHOICES:
            _add(fm, fn)

    variants: list[LoopOp] = []
    for fm, fn in ordered:
        knobs = {**loop_op.knobs, "FM": fm, "FN": fn}
        if fm == 1 and fn == 1:
            variants.append(LoopOp(body=loop_op.body, knobs=knobs))
            continue
        new_body = _split_register_outer_two(loop_op.body, outer_m.axis.name, outer_n.axis.name, fm, fn)
        variants.append(LoopOp(body=new_body, knobs=knobs))
    return variants or None


def _split_register_outer_two(body, m_name: str, n_name: str, fm: int, fn: int):
    """Pre-split the outer M and N output Loops by ``(FM, FN)``.

    For each axis with ``F > 1``: split into ``F_o`` (outer free,
    extent ``E/F``) over ``F_i`` (inner ``Role.REGISTER``, extent ``F``)
    with ``σ: axis → F_o*F + F_i``. When ``F == 1`` the axis is left
    untouched."""

    def _identity_rename(name: str) -> str:
        return name

    stmts = tuple(body)
    assert len(stmts) == 1 and isinstance(stmts[0], Loop) and stmts[0].axis.name == m_name
    m_loop = stmts[0]
    m_body = tuple(m_loop.body)
    assert len(m_body) == 1 and isinstance(m_body[0], Loop) and m_body[0].axis.name == n_name
    n_loop = m_body[0]
    inner = tuple(n_loop.body)

    sigma_map: dict = {}
    m_o = m_i = None
    n_o = n_i = None
    if fm > 1:
        m_o = Axis(f"{m_name}_o", int(m_loop.axis.extent) // fm)
        m_i = Axis(f"{m_name}_i", fm)
        sigma_map[m_name] = Var(m_o.name) * Literal(fm, "int") + Var(m_i.name)
    if fn > 1:
        n_o = Axis(f"{n_name}_o", int(n_loop.axis.extent) // fn)
        n_i = Axis(f"{n_name}_i", fn)
        sigma_map[n_name] = Var(n_o.name) * Literal(fn, "int") + Var(n_i.name)

    if sigma_map:
        sigma = Sigma(sigma_map)
        inner_rewritten = tuple(s.rewrite(_identity_rename, sigma) for s in inner)
    else:
        inner_rewritten = inner

    current = inner_rewritten
    if fn > 1:
        current = (Loop(axis=n_i, role=Role.REGISTER, body=current),)
    if fm > 1:
        current = (Loop(axis=m_i, role=Role.REGISTER, body=current),)
    outer_n_axis = n_o if fn > 1 else n_loop.axis
    current = (Loop(axis=outer_n_axis, body=current),)
    outer_m_axis = m_o if fm > 1 else m_loop.axis
    current = (Loop(axis=outer_m_axis, body=current),)
    return current


# --- matmul K-chunk branch ------------------------------------------


def _try_matmul_k_chunk(ctx: Context, loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
    """Fork over BK for the matmul K reduce. Splits K → K_o
    (``Role.SERIAL_OUTER``) × K_i (``Role.STAGE_INNER``)."""
    site = _find_first_matmul_reduce(loop_op.body)
    if site is None:
        return None
    if site.role is Role.STAGE_INNER:
        return None  # already chunked
    K = int(site.axis.extent)
    cands = _bk_candidates_for(ctx, loop_op, body_info, K)
    if not cands:
        return None
    variants: list[LoopOp] = []
    for bk in cands:
        new_body, changed = _chunk_first_matmul_k(loop_op.body, bk)
        if not changed:
            continue
        knobs = dict(loop_op.knobs)
        knobs[BK.name] = bk
        variants.append(LoopOp(body=new_body, knobs=knobs))
    return variants or None


def _bk_candidates_for(ctx: Context, loop_op: LoopOp, body_info: BodyInfo, K: int) -> tuple[int, ...]:
    """Filter ``_BK_CANDIDATES`` by ``K % bk == 0`` ∧ ``K > bk``, with
    the heuristic ``forced_bk`` value first when it qualifies."""
    # Post-register-tile chain: walks ``loop_op.body`` outer free chain
    # (chain breaks at REGISTER tags), so extents reflect the split.
    chain = _outer_free_loop_chain(loop_op.body)
    if chain:
        output_extents = _chain_extents_desc(chain)
    else:
        output_extents = ()
    forced = forced_bk(output_extents, body_info, ctx.static_smem_cap)
    base = [c for c in _BK_CANDIDATES if K % c == 0 and K > c]
    if forced is not None and K % forced == 0 and K > forced and forced not in base:
        base = [forced, *base]
    elif forced is not None and forced in base:
        base = [forced, *(c for c in base if c != forced)]
    return tuple(base)


def _find_first_matmul_reduce(stmts) -> Loop | None:
    """Locate the first matmul-shaped reduce ``Loop`` reachable by
    descent through Loops / StridedLoops / Conds."""
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            return s
        if isinstance(s, (Loop, StridedLoop)):
            found = _find_first_matmul_reduce(s.body)
            if found is not None:
                return found
        if isinstance(s, Cond):
            found = _find_first_matmul_reduce(s.body) or _find_first_matmul_reduce(s.else_body)
            if found is not None:
                return found
    return None


def _chunk_first_matmul_k(stmts, bk: int) -> tuple[tuple, bool]:
    """Walk ``stmts``, replacing the first matmul-shaped reduce
    ``Loop(K, …)`` with ``Loop(K_o, SERIAL_OUTER, Loop(K_i, reduce,
    STAGE_INNER, σ(body)))``."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if changed:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            chunked = _chunk_k_loop(s, bk)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_first_matmul_k(s.body, bk)
            if inner_changed:
                out.append(replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_first_matmul_k(s.body, bk)
            inner_e, ce = _chunk_first_matmul_k(s.else_body, bk)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _chunk_k_loop(loop: Loop, bk: int) -> Loop | None:
    K = int(loop.axis.extent)
    if K % bk != 0 or K <= bk:
        return None
    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // bk)
    K_i = Axis(f"{K_name}_i", bk)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(bk, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_identity_rename_k, sigma) for s in loop.body)
    return Loop(
        axis=K_o,
        role=Role.SERIAL_OUTER,
        body=(Loop(axis=K_i, role=Role.STAGE_INNER, body=inner_body),),
    )


def _identity_rename_k(name: str) -> str:
    return name


# --- non-matmul chunk-reduce branch ---------------------------------


_REDUCE_BK_CANDIDATES = (128, 64, 32, 16, 8)
_REDUCE_SLAB_CAP_BYTES = 16 * 1024
_REDUCE_SLAB_HEADROOM_BYTES = 8 * 1024
_BYTES_PER_ELEM = 4


def _try_chunk_reduce(loop_op: LoopOp, body_info: BodyInfo) -> LoopOp | None:
    """Chunk every non-matmul reduce Loop whose K-indexed Loads would
    otherwise produce a smem slab over the ``007_stage_inputs`` cap."""
    chain = _outer_free_loop_chain(loop_op.body)
    if not chain:
        return None
    lifted = _lifted_output_loops(tuple(chain[-1].body))
    combined = chain + lifted
    thread_targets = _predict_thread_extents(body_info, chain, lifted)
    if not thread_targets:
        return None
    new_body, changed = _chunk_reduces_in_body(loop_op.body, combined, thread_targets)
    if not changed:
        return None
    return LoopOp(body=new_body, knobs=dict(loop_op.knobs))


def _predict_thread_extents(body_info: BodyInfo, chain: tuple[Loop, ...], lifted: tuple[Loop, ...] = ()) -> tuple[int, ...] | None:
    """Predicted per-axis thread extent for each outer-chain axis
    followed by each lifted axis."""
    output_extents = _chain_extents_desc(chain)
    shape = thread_tile_shape(output_extents, body_info)  # innermost-first
    targets_outer_first = tuple(reversed(shape))
    extents: list[int] = []
    for i, lp in enumerate(chain):
        ext = int(lp.axis.extent)
        if i < len(chain) - len(targets_outer_first):
            extents.append(1)
            continue
        j = i - (len(chain) - len(targets_outer_first))
        target = targets_outer_first[j]
        extents.append(min(ext, target))
    for lp in lifted:
        extents.append(int(lp.axis.extent))
    return tuple(extents)


def _lifted_output_loops(stmts: tuple) -> tuple[Loop, ...]:
    """Mirror ``001_tileify._lift_output_loops``."""
    out: list[Loop] = []
    for s in stmts:
        if isinstance(s, Loop) and not s.is_reduce and s.role is not Role.REGISTER and _writes_with_axis(s.body, s.axis.name):
            out.append(s)
    return tuple(out)


def _writes_with_axis(stmts: tuple, axis_name: str) -> bool:
    for s in stmts:
        if isinstance(s, Write):
            free: set[str] = set()
            for e in s.index:
                free |= e.free_vars()
            if axis_name in free:
                return True
        if isinstance(s, (Loop, StridedLoop)) and _writes_with_axis(s.body, axis_name):
            return True
        if isinstance(s, Cond):
            if _writes_with_axis(s.body, axis_name) or _writes_with_axis(s.else_body, axis_name):
                return True
    return False


def _chunk_reduces_in_body(stmts, chain: tuple[Loop, ...], thread_extents: tuple[int, ...]) -> tuple[tuple, bool]:
    """Walk the body, chunking every qualifying non-matmul reduce."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and _reduce_qualifies(s, chain, thread_extents):
            chunked = _chunk_reduce_loop(s, chain, thread_extents)
            if chunked is not None:
                out.append(chunked)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, inner_changed = _chunk_reduces_in_body(s.body, chain, thread_extents)
            if inner_changed:
                out.append(replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _chunk_reduces_in_body(s.body, chain, thread_extents)
            inner_e, ce = _chunk_reduces_in_body(s.else_body, chain, thread_extents)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _reduce_qualifies(loop: Loop, chain: tuple[Loop, ...], thread_extents: tuple[int, ...]) -> bool:
    if is_matmul_reduce(loop):
        return False
    if loop.role is Role.STAGE_INNER:
        return False
    if any(inner.is_reduce for inner in loop.body.of_type(Loop)):
        return False
    has_fanin, max_in_ext, k_indexed_count = _slab_geometry(loop, chain, thread_extents)
    if k_indexed_count == 0:
        return False
    K_extent = int(loop.axis.extent)
    return has_fanin and K_extent * max_in_ext * _BYTES_PER_ELEM > _REDUCE_SLAB_CAP_BYTES


def _slab_geometry(loop: Loop, chain: tuple[Loop, ...], thread_extents: tuple[int, ...]):
    from deplodock.compiler.ir.stmt import Load

    K_name = loop.axis.name
    has_fanin = False
    max_in_ext = 1
    count = 0
    for ld in loop.body.iter_of_type(Load):
        ld_vars = {v for e in ld.index for v in e.free_vars()}
        if K_name not in ld_vars:
            continue
        count += 1
        in_ext = 1
        for lp, t_ext in zip(chain, thread_extents, strict=True):
            if t_ext <= 1:
                continue
            if lp.axis.name in ld_vars:
                in_ext *= t_ext
            else:
                has_fanin = True
        max_in_ext = max(max_in_ext, in_ext)
    return has_fanin, max_in_ext, count


def _chunk_reduce_loop(loop: Loop, chain: tuple[Loop, ...], thread_extents: tuple[int, ...]) -> Loop | None:
    K = int(loop.axis.extent)
    _, max_in_ext, _ = _slab_geometry(loop, chain, thread_extents)
    bk = _pick_reduce_bk(K, max_in_ext)
    if bk is None:
        return None
    K_name = loop.axis.name
    K_o = Axis(f"{K_name}_o", K // bk)
    K_i = Axis(f"{K_name}_i", bk)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(bk, "int") + Var(K_i.name)})
    inner = tuple(s.rewrite(_identity_rename_k, sigma) for s in loop.body)
    return Loop(
        axis=K_o,
        role=Role.SERIAL_OUTER,
        body=(Loop(axis=K_i, role=Role.STAGE_INNER, body=inner),),
    )


def _pick_reduce_bk(K: int, in_extent_product: int) -> int | None:
    if in_extent_product < 1:
        in_extent_product = 1
    for c in _REDUCE_BK_CANDIDATES:
        if K % c != 0 or K <= c:
            continue
        if in_extent_product * c * _BYTES_PER_ELEM > _REDUCE_SLAB_HEADROOM_BYTES:
            continue
        return c
    return None


# --- matmul (BN, BM) stamp fork --------------------------------------


def _try_matmul_bn_bm_fork(loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
    """Enumerate ``(BN, BM)`` candidates and stamp them as knobs (no
    body change). ``004_launch_geometry`` reads BN/BM from knobs and
    does the deterministic axis split."""
    if not body_info.has_matmul:
        return None
    if "BN" in loop_op.knobs:
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None

    ext_outer = int(chain[-2].axis.extent)
    ext_inner = int(chain[-1].axis.extent)

    output_extents = _chain_extents_desc(chain)
    heuristic = thread_tile_shape(output_extents, body_info)
    h_bn = int(heuristic[0]) if len(heuristic) >= 1 else _TUNE_AXIS_CHOICES[2]
    h_bm = int(heuristic[1]) if len(heuristic) >= 2 else _TUNE_AXIS_CHOICES[2]

    seen: set[tuple[int, int]] = set()
    ordered: list[tuple[int, int]] = []

    def _add(bn: int, bm: int) -> None:
        bn = min(bn, ext_inner)
        bm = min(bm, ext_outer)
        if ext_inner > bn and ext_inner % bn != 0:
            return
        if ext_outer > bm and ext_outer % bm != 0:
            return
        if bn == ext_inner and bm == ext_outer:
            return
        if bn * bm > 1024:
            return
        shape = (bn, bm)
        if shape in seen:
            return
        seen.add(shape)
        ordered.append(shape)

    _add(h_bn, h_bm)
    for bn in _TUNE_AXIS_CHOICES:
        for bm in _TUNE_AXIS_CHOICES:
            _add(bn, bm)

    variants: list[LoopOp] = []
    for bn, bm in ordered:
        knobs = {**loop_op.knobs, "BN": bn, "BM": bm}
        variants.append(LoopOp(body=loop_op.body, knobs=knobs))
    return variants or None


# --- SPLITK stamp fork ----------------------------------------------


def _try_splitk(loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
    """Enumerate ``SPLITK`` candidates and stamp them as knobs."""
    if not body_info.has_matmul:
        return None
    if "SPLITK" in loop_op.knobs:
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    inner_body = tuple(chain[-1].body) if chain else tuple(loop_op.body)
    k_o = _find_serial_outer_k(inner_body)
    if k_o is None:
        return None
    K_o_extent = int(k_o.axis.extent)

    output_extents = _chain_extents_desc(chain) if chain else ()
    thread_extents = tuple(int(lp.axis.extent) for lp in chain)
    heuristic = int(auto_splitk(output_extents, body_info, K_o_extent, thread_extents))
    if heuristic <= 1:
        return None
    knobs = {**loop_op.knobs, "SPLITK": heuristic}
    return [LoopOp(body=loop_op.body, knobs=knobs)]


def _find_serial_outer_k(stmts) -> Loop | None:
    for s in stmts:
        if isinstance(s, Loop) and s.role is Role.SERIAL_OUTER:
            return s
        if isinstance(s, (Loop, StridedLoop)):
            found = _find_serial_outer_k(s.body)
            if found is not None:
                return found
        if isinstance(s, Cond):
            found = _find_serial_outer_k(s.body) or _find_serial_outer_k(s.else_body)
            if found is not None:
                return found
    return None
