"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, etc.) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes (``001_tileify``, ``006a_register_tile_planned``,
``007_stage_inputs``, ...) read the tags and skip their own equivalent
decisions, doing only the leftover rewrites (lift to ``Tile.axes``,
replicate stmts, build stages).

**M4 scope** — matmul register tile. Detect matmul-shaped LoopOps,
pre-split the outer M / N output Loops by ``(FM, FN)`` from
:func:`tuning.register_tile_shape`, tag the inner halves
``Role.REGISTER``, and σ-substitute the body. ``001_tileify`` lifts
M_o / N_o to ``Tile.axes`` and stops at the REGISTER tags;
``006a_register_tile_planned`` replicates the per-cell bodies *before*
``007_stage_inputs`` runs.

**M7 scope** — matmul K chunking. After the M/N register-tile decision
(if any), locate the matmul K reduce and pre-split it into
``Loop(K_o, SERIAL_OUTER) → Loop(K_i, reduce, STAGE_INNER)`` with
σ: K → K_o*BK + K_i. The planner forks over a ``BK`` knob (same
candidates as ``002_chunk_matmul_k``) so ``deplodock tune`` still
walks the same variant space; greedy callers pick variant 0
(the heuristic ``forced_bk`` value). ``002_chunk_matmul_k`` carries an
idempotence guard that no-ops once the planner has stamped
``Role.STAGE_INNER`` on the matmul reduce.

**M8 scope** — non-matmul chunk-reduce. For each non-matmul reduce
whose K-indexed Loads project a thread-fanin slab over the 16 KB
``007_stage_inputs`` cap, split K → ``Loop(K_o, SERIAL_OUTER) →
Loop(K_i, reduce, STAGE_INNER)`` using the same BK picker as
``006_chunk_reduce``. Predicts post-blockify thread extents via
``min(extent, thread_tile_shape_target)`` against the outer free-Loop
chain; skips cooperative-viable kernels (their synthetic ``t`` axis
never matches a Load's index). ``006_chunk_reduce``'s M3 STAGE_INNER
guard handles idempotence.

**M8 limitation**: the predictor only covers the single-stmt outer
free chain; it misses output-write Loops that ``001_tileify``'s
``_lift_output_loops`` step lifts from multi-stmt bodies (SDPA's
``head_dim`` etc.). Kernels with lifted output axes still rely on the
legacy ``006_chunk_reduce`` post-tileify (which is correctly preserved
through the M3 guard).

Gated by ``DEPLODOCK_PLANNER`` env var so each milestone can test the
new path against the legacy default (=0) for structural equivalence.
"""

from __future__ import annotations

import os
from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop
from deplodock.compiler.ir.tile.ir import Tile
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

PATTERN = [Pattern("root", LoopOp)]

_ENABLE_ENV = "DEPLODOCK_PLANNER"
# Knob stamp signalling the planner produced output (for planner-side
# idempotence — re-firing on a planned LoopOp is a no-op). Downstream
# 006a uses ``Role.REGISTER`` presence + ``FN`` absence as its trigger.
_PLANNER_KNOB = "PLANNER"

# Matches ``002_chunk_matmul_k._BK_CANDIDATES`` so the planner-driven
# fork enumerates the same variant space as the legacy 002 fork did.
_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)

BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    if not os.environ.get(_ENABLE_ENV):
        raise RuleSkipped(f"{_ENABLE_ENV} not set")
    loop_op: LoopOp = root.op
    if loop_op.knobs.get(_PLANNER_KNOB):
        raise RuleSkipped("already planned")

    after_reg = _try_matmul_register_tile(ctx, loop_op)
    base = after_reg if after_reg is not None else loop_op

    k_variants = _try_matmul_k_chunk(ctx, base)
    candidates: list[LoopOp] = list(k_variants) if k_variants is not None else [base]

    chunked: list[LoopOp] = []
    any_chunk_reduce = False
    for cand in candidates:
        after_cr = _try_chunk_reduce(cand)
        if after_cr is not None:
            any_chunk_reduce = True
            chunked.append(after_cr)
        else:
            chunked.append(cand)

    fired = after_reg is not None or k_variants is not None or any_chunk_reduce
    if not fired:
        raise RuleSkipped("no planner branch matched")

    if len(chunked) == 1:
        return _stamp_planned(chunked[0])
    return [_stamp_planned(v) for v in chunked]


def _stamp_planned(op: LoopOp) -> LoopOp:
    knobs = dict(op.knobs)
    knobs[_PLANNER_KNOB] = True
    return LoopOp(body=op.body, knobs=knobs)


# --- matmul register-tile branch -------------------------------------


def _try_matmul_register_tile(ctx: Context, loop_op: LoopOp) -> LoopOp | None:
    """Detect a matmul-shape LoopOp; if eligible, pre-split the outer two
    output Loops by ``(FM, FN)`` with ``Role.REGISTER`` on the inner
    halves and σ-substitute the body. Returns ``None`` when no
    transformation applies (legacy 008 then handles it post-staging)."""
    fm, fn = _pick_register_factors(loop_op)
    if fm <= 1 and fn <= 1:
        return None

    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None
    outer_m, outer_n = chain[0], chain[1]
    if int(outer_m.axis.extent) % fm != 0 or int(outer_n.axis.extent) % fn != 0:
        return None

    new_body = _split_register_outer_two(loop_op.body, outer_m.axis.name, outer_n.axis.name, fm, fn)
    return LoopOp(body=new_body, knobs=dict(loop_op.knobs))


def _pick_register_factors(loop_op: LoopOp) -> tuple[int, int]:
    """Heuristic ``(FM, FN)`` from :func:`tuning.register_tile_shape`
    against a synthetic Tile carrying the outer-chain axes as THREAD —
    same classification + small-tile guard 008 applies post-tileify."""
    from deplodock.compiler.tuning import _has_matmul_reduce, register_tile_shape  # noqa: PLC0415

    if not _has_matmul_reduce(loop_op.body):
        return (1, 1)
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return (1, 1)
    synthetic = Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )
    fm, fn = register_tile_shape(synthetic)
    return int(fm), int(fn)


def _outer_free_loop_chain(body) -> tuple[Loop, ...]:
    """Walk the outer single-stmt chain of untagged free Loops outermost-
    first. Mirrors ``001_tileify._strip_outer_free_chain``."""
    out: list[Loop] = []
    cur = tuple(body)
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        out.append(cur[0])
        cur = tuple(cur[0].body)
    return tuple(out)


def _split_register_outer_two(body, m_name: str, n_name: str, fm: int, fn: int):
    """``Loop(M:E) → Loop(M_o:E/FM) → Loop(N_o:E'/FN) → Loop(M_i:FM, REG)
    → Loop(N_i:FN, REG) → σ(body)`` where σ maps M → M_o*FM+M_i and
    N → N_o*FN+N_i in one pass over the innermost body."""

    def _identity_rename(name: str) -> str:
        return name

    stmts = tuple(body)
    assert len(stmts) == 1 and isinstance(stmts[0], Loop) and stmts[0].axis.name == m_name
    m_loop = stmts[0]
    m_body = tuple(m_loop.body)
    assert len(m_body) == 1 and isinstance(m_body[0], Loop) and m_body[0].axis.name == n_name
    n_loop = m_body[0]
    inner = tuple(n_loop.body)

    m_o = Axis(f"{m_name}_o", int(m_loop.axis.extent) // fm)
    m_i = Axis(f"{m_name}_i", fm)
    n_o = Axis(f"{n_name}_o", int(n_loop.axis.extent) // fn)
    n_i = Axis(f"{n_name}_i", fn)

    sigma = Sigma(
        {
            m_name: Var(m_o.name) * Literal(fm, "int") + Var(m_i.name),
            n_name: Var(n_o.name) * Literal(fn, "int") + Var(n_i.name),
        }
    )
    inner_rewritten = tuple(s.rewrite(_identity_rename, sigma) for s in inner)

    rebuilt = Loop(
        axis=m_o,
        body=(
            Loop(
                axis=n_o,
                body=(
                    Loop(
                        axis=m_i,
                        role=Role.REGISTER,
                        body=(Loop(axis=n_i, role=Role.REGISTER, body=inner_rewritten),),
                    ),
                ),
            ),
        ),
    )
    return (rebuilt,)


# --- matmul K-chunk branch ------------------------------------------


def _try_matmul_k_chunk(ctx: Context, loop_op: LoopOp) -> list[LoopOp] | None:
    """Fork over BK for the matmul K reduce. Splits K → K_o
    (``Role.SERIAL_OUTER``) × K_i (``Role.STAGE_INNER``) with
    σ: K → K_o*BK + K_i. Returns one variant per BK candidate that
    divides K (with K > BK); variant 0 carries the heuristic
    ``forced_bk`` value so greedy callers (no autotune DB) reproduce
    the legacy 002 default. Returns ``None`` when no matmul K reduce
    is reachable or it's already chunked."""
    site = _find_first_matmul_reduce(loop_op.body)
    if site is None:
        return None
    if site.role is Role.STAGE_INNER:
        return None  # already chunked
    K = int(site.axis.extent)
    cands = _bk_candidates_for(ctx, loop_op, K)
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


def _bk_candidates_for(ctx: Context, loop_op: LoopOp, K: int) -> tuple[int, ...]:
    """Filter ``_BK_CANDIDATES`` by ``K % bk == 0`` ∧ ``K > bk``, with
    the heuristic ``forced_bk`` value first when it qualifies (so
    variant 0 reproduces the legacy 002 default for greedy callers)."""
    from deplodock.compiler.tuning import forced_bk

    synthetic = _synthetic_tile_post_register(loop_op)
    forced = forced_bk(synthetic, ctx.static_smem_cap)
    base = [c for c in _BK_CANDIDATES if K % c == 0 and K > c]
    if forced is not None and K % forced == 0 and K > forced and forced not in base:
        base = [forced, *base]
    elif forced is not None and forced in base:
        base = [forced, *(c for c in base if c != forced)]
    return tuple(base)


def _synthetic_tile_post_register(loop_op: LoopOp) -> Tile | None:
    """Build a synthetic ``Tile`` reflecting the launch shape that
    ``001_tileify`` will produce — outer free Loops above the first
    REGISTER tag become ``Tile.axes`` (BIND_THREAD). Used to feed
    :func:`tuning.forced_bk` so the picker sees the post-register-tile
    extents."""
    chain: list[Loop] = []
    cur = tuple(loop_op.body)
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        chain.append(cur[0])
        cur = tuple(cur[0].body)
    if not chain:
        return None
    return Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )


def _find_first_matmul_reduce(stmts) -> Loop | None:
    """Locate the first matmul-shaped reduce ``Loop`` reachable by
    descent through Loops / StridedLoops / Conds. Mirrors 002's
    walker so the K we measure is the K the rewrite will hit."""
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
    STAGE_INNER, σ(body)))``. Recurses through wrapper Loops /
    StridedLoops / Conds so a matmul nested under register-tile REG
    loops (the typical post-M4 shape) is reachable. Returns
    ``(new_body, changed)``."""
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


# Mirrors 006_chunk_reduce's BK candidate set + slab budgets.
_REDUCE_BK_CANDIDATES = (128, 64, 32, 16, 8)
_REDUCE_SLAB_CAP_BYTES = 16 * 1024
_REDUCE_SLAB_HEADROOM_BYTES = 8 * 1024
_BYTES_PER_ELEM = 4


def _try_chunk_reduce(loop_op: LoopOp) -> LoopOp | None:
    """Chunk every non-matmul reduce Loop whose K-indexed Loads would
    otherwise produce a smem slab over the ``007_stage_inputs`` cap.

    Predicts post-blockify thread extents (``min(extent, target)``
    where target comes from :func:`tuning.thread_tile_shape`) so the
    slab math matches what 006 sees post-launch_geometry. Cooperative-
    viable bodies (reduce extent ≥ 32 with independent Accums) are
    skipped — their synthetic ``t`` axis never matches a Load's index,
    so the slab degenerates to ``K × 4`` and never exceeds the cap.
    """
    chain = _outer_free_loop_chain(loop_op.body)
    if not chain:
        return None
    # launch_geometry dispatches matmul-path before cooperative-path,
    # so cooperative-shape only matters when no matmul reduce is
    # present. With a matmul reduce, we run chunk_reduce on the
    # non-matmul reduces alongside the matmul-path.
    from deplodock.compiler.tuning import _has_matmul_reduce  # noqa: PLC0415

    if not _has_matmul_reduce(loop_op.body) and _cooperative_shape(loop_op.body, chain):
        return None
    thread_targets = _predict_thread_extents(loop_op, chain)
    if not thread_targets:
        return None
    new_body, changed = _chunk_reduces_in_body(loop_op.body, chain, thread_targets)
    if not changed:
        return None
    return LoopOp(body=new_body, knobs=dict(loop_op.knobs))


def _cooperative_shape(body, chain: tuple[Loop, ...]) -> bool:
    """Mirror ``004_launch_geometry._cooperative_viable`` at the LoopOp
    level: any reduce Loop with extent ≥ 32 + every reduce has an Accum
    + Accums independent. The reduce sits at ``chain[-1].body`` (or
    deeper) once the outer chain is stripped."""
    from deplodock.compiler.ir.stmt import Accum

    inner_body = chain[-1].body if chain else body
    inner_stmts = tuple(inner_body) if not isinstance(inner_body, tuple) else inner_body
    body_tuple = tuple(inner_stmts) if hasattr(inner_stmts, "__iter__") else ()
    # Search the inner body for reduce Loops; cooperative path is keyed
    # off ≥ WARP_SIZE reduce extent + Accum independence.
    from deplodock.compiler.ir.stmt.body import Body

    coerced = Body.coerce(body_tuple)
    reduce_loops = [lp for lp in coerced.of_type(Loop) if lp.is_reduce]
    if not reduce_loops:
        return False
    if int(reduce_loops[0].axis.extent) < 32:
        return False
    for rl in reduce_loops:
        if not any(isinstance(s, Accum) for s in rl.body):
            return False
        accum_names = {s.name for s in rl.body if isinstance(s, Accum)}
        for s in rl.body:
            if isinstance(s, Accum) and rl.body.depends_on(s.value, accum_names - {s.name}):
                return False
    return True


def _predict_thread_extents(loop_op: LoopOp, chain: tuple[Loop, ...]) -> tuple[int, ...] | None:
    """Predicted (axis_name, thread_extent) for each outer-chain axis,
    mirroring ``004_launch_geometry._plan_partition``: an axis with
    ``ext ≤ target`` stays at full extent; ``ext > target`` shrinks to
    ``target``. Returns the per-axis predicted thread extent in the
    same order as ``chain`` (outermost-first).

    ``thread_tile_shape`` returns innermost-first targets; we reverse
    so we can zip with ``chain``."""
    from deplodock.compiler.tuning import thread_tile_shape

    synthetic = Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )
    shape = thread_tile_shape(synthetic)  # innermost-first
    targets_outer_first = tuple(reversed(shape))
    extents: list[int] = []
    for i, lp in enumerate(chain):
        ext = int(lp.axis.extent)
        # Outer axes beyond ``len(shape)`` become BIND_BLOCK only — they
        # contribute no thread fan-in. Use extent 1 so they don't enter
        # the slab product.
        if i < len(chain) - len(targets_outer_first):
            extents.append(1)
            continue
        j = i - (len(chain) - len(targets_outer_first))
        target = targets_outer_first[j]
        extents.append(min(ext, target))
    return tuple(extents)


def _chunk_reduces_in_body(stmts, chain: tuple[Loop, ...], thread_extents: tuple[int, ...]) -> tuple[tuple, bool]:
    """Walk the body, chunking every qualifying non-matmul reduce
    Loop (mirroring 006's "fire on every qualifying reduce" behavior,
    not 002's first-match-only). Returns ``(new_body, changed)``."""
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
    """True iff ``loop`` is a non-matmul reduce, not already chunked
    (by role tag or nested-reduce), with thread-axis fan-in over at
    least one K-indexed Load whose candidate slab exceeds the
    ``007_stage_inputs`` 16 KB cap."""
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
    """Per-Load slab analysis mirroring 006's ``_slab_geometry`` but
    against *predicted* thread extents. A thread axis derived from
    ``chain[i]`` contributes its predicted extent to ``in_ext`` iff
    the chain axis's original name appears in the Load's index."""
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
    """Largest BK from ``_REDUCE_BK_CANDIDATES`` that divides K (with
    K > BK) and keeps the per-Load slab within the headroom budget.
    Matches ``006_chunk_reduce._pick_bk``."""
    if in_extent_product < 1:
        in_extent_product = 1
    for c in _REDUCE_BK_CANDIDATES:
        if K % c != 0 or K <= c:
            continue
        if in_extent_product * c * _BYTES_PER_ELEM > _REDUCE_SLAB_HEADROOM_BYTES:
            continue
        return c
    return None
