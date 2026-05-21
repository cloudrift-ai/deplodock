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

The predictor also mirrors ``001_tileify._lift_output_loops``: after
the outer chain breaks, top-level free Loops in the remaining body
whose subtree writes through their axis (SDPA's ``head_dim`` etc.)
join the predicted thread-axes list at full extent. This catches the
SDPA-class kernels that previously fell through to legacy
``006_chunk_reduce`` post-tileify.

M10: ``DEPLODOCK_PLANNER`` env flag has been dropped — the planner
always runs. Legacy fallback passes (``002_chunk_matmul_k``,
``006_chunk_reduce``, ``008_register_tile``'s matmul fork) carry
idempotence guards that no-op once the planner has produced their
output and stay reachable for the small subset of kernels the
planner short-circuits (e.g. matmul-shaped LoopOps whose pre-
blockify register_tile_shape heuristic returns ``(1, 1)``).
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis, Role
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import Tile
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    MAX_CELLS_PER_THREAD,
    TUNE_F_CHOICES,
    is_matmul_reduce,
)

PATTERN = [Pattern("root", LoopOp)]

# Knob stamp signalling the planner produced output (for planner-side
# idempotence — re-firing on a planned LoopOp is a no-op). Downstream
# 006a uses ``Role.REGISTER`` presence + ``FN`` absence as its trigger.
_PLANNER_KNOB = "PLANNER"

# Matches ``002_chunk_matmul_k._BK_CANDIDATES`` so the planner-driven
# fork enumerates the same variant space as the legacy 002 fork did.
_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)

BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")

# Mirrors ``004_launch_geometry._TUNE_AXIS_CHOICES`` and ``_WARP_SIZE``
# so planner-driven forks (cooperative-BN, matmul BN/BM) enumerate the
# same candidates the legacy 004 fork did.
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_WARP_SIZE = 32


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    loop_op: LoopOp = root.op
    if loop_op.knobs.get(_PLANNER_KNOB):
        raise RuleSkipped("already planned")

    reg_variants = _try_matmul_register_tile(ctx, loop_op)
    base_variants: list[LoopOp] = list(reg_variants) if reg_variants is not None else [loop_op]

    after_k: list[LoopOp] = []
    any_k_chunk = False
    for base in base_variants:
        k_variants = _try_matmul_k_chunk(ctx, base)
        if k_variants is not None:
            any_k_chunk = True
            after_k.extend(k_variants)
        else:
            after_k.append(base)

    after_splitk: list[LoopOp] = []
    any_splitk = False
    for cand in after_k:
        sk_variants = _try_splitk(ctx, cand)
        if sk_variants is not None:
            any_splitk = True
            after_splitk.extend(sk_variants)
        else:
            after_splitk.append(cand)

    coop_results: list[LoopOp] = []
    any_coop = False
    for cand in after_splitk:
        coop_variants = _try_cooperative_reduce(cand)
        if coop_variants is not None:
            any_coop = True
            coop_results.extend(coop_variants)
        else:
            coop_results.append(cand)

    chunked: list[LoopOp] = []
    any_chunk_reduce = False
    for cand in coop_results:
        # Cooperative-tagged kernels: their reduce axes are mapped to a
        # strided thread axis; the per-load slab degenerates and
        # chunk-reduce doesn't apply.
        if _has_cooperative_stride(cand.body):
            chunked.append(cand)
            continue
        after_cr = _try_chunk_reduce(cand)
        if after_cr is not None:
            any_chunk_reduce = True
            chunked.append(after_cr)
        else:
            chunked.append(cand)

    # Matmul (BN, BM) stamp fork — pure knob enumeration, no body
    # change. The planner is the source of truth; ``004_launch_geometry``
    # reads BN/BM from knobs and emits one deterministic TileOp.
    bn_bm_results: list[LoopOp] = []
    any_bn_bm = False
    for cand in chunked:
        bn_bm_variants = _try_matmul_bn_bm_fork(cand)
        if bn_bm_variants is not None:
            any_bn_bm = True
            bn_bm_results.extend(bn_bm_variants)
        else:
            bn_bm_results.append(cand)

    fired = reg_variants is not None or any_k_chunk or any_splitk or any_coop or any_chunk_reduce or any_bn_bm
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


# --- matmul register-tile branch -------------------------------------


def _try_matmul_register_tile(ctx: Context, loop_op: LoopOp) -> list[LoopOp] | None:
    """Detect a matmul-shape LoopOp; if eligible, fork over ``(FM, FN)``
    candidates from :data:`TUNE_F_CHOICES`. For each viable pair, emit
    a LoopOp variant whose outer M / N output Loops are pre-split by
    ``(FM, FN)`` with ``Role.REGISTER`` on the inner halves. Stamps
    ``knobs={"FM": fm, "FN": fn}``.

    Heuristic shape (from :func:`tuning.register_tile_shape`) is emitted
    as variant 0 so deterministic compiles match the legacy 008
    behavior. ``(1, 1)`` is included so the autotuner can elect the
    no-register-tile shape too — it emits the body unchanged with
    stamped knobs.

    Returns ``None`` when the kernel isn't matmul-shaped or has fewer
    than two outer free Loops."""
    from deplodock.compiler.tuning import _has_matmul_reduce, register_tile_shape  # noqa: PLC0415

    if not _has_matmul_reduce(loop_op.body):
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None
    outer_m, outer_n = chain[0], chain[1]
    ext_m, ext_n = int(outer_m.axis.extent), int(outer_n.axis.extent)

    synthetic = Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )
    h_fm, h_fn = (int(x) for x in register_tile_shape(synthetic))
    if h_fm == 1 and h_fn == 1:
        # ``register_tile_shape`` is pre-blockify here — without the
        # (BN, BM) split (Step 4) the synthetic Tile has full extents
        # and the heuristic is too pessimistic for small / fused matmuls
        # (e.g. SDPA's 32×64 inner matmul). Defer to ``008_register_tile``
        # post-staging, which sees the actual blockified shape. Once
        # Step 4 hoists (BN, BM) into the planner, this short-circuit
        # goes away and the full fork emits.
        return None

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
    """Pre-split the outer M and N output Loops by ``(FM, FN)``.

    For each axis with ``F > 1``: split into ``F_o`` (outer free,
    extent ``E/F``) over ``F_i`` (inner ``Role.REGISTER``, extent ``F``)
    with ``σ: axis → F_o*F + F_i``. When ``F == 1`` the axis is left
    untouched (no split, no REGISTER tag) — this happens for variants
    like ``(FM=1, FN=k)`` that only register-tile one of the two axes.

    Resulting nesting (showing both >1):
    ``Loop(M_o) → Loop(N_o) → Loop(M_i, REG) → Loop(N_i, REG) → σ(body)``.
    ``F==1`` cases drop the corresponding ``F_o`` / ``F_i`` pair and
    keep the original axis."""

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

    # Build inside-out: register-tile inner halves, then outer halves.
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
    lifted = _lifted_output_loops(tuple(chain[-1].body))
    combined = chain + lifted
    thread_targets = _predict_thread_extents(loop_op, chain, lifted)
    if not thread_targets:
        return None
    new_body, changed = _chunk_reduces_in_body(loop_op.body, combined, thread_targets)
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


def _predict_thread_extents(loop_op: LoopOp, chain: tuple[Loop, ...], lifted: tuple[Loop, ...] = ()) -> tuple[int, ...] | None:
    """Predicted (axis_name, thread_extent) for each outer-chain axis
    followed by each lifted axis, mirroring ``004_launch_geometry.
    _plan_partition`` for the chain (``ext ≤ target`` stays full;
    ``ext > target`` shrinks to ``target``) and ``001_tileify.
    _lift_output_loops`` for the lifted axes (always full extent —
    tileify lifts but doesn't split them).

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
    for lp in lifted:
        extents.append(int(lp.axis.extent))
    return tuple(extents)


def _lifted_output_loops(stmts: tuple) -> tuple[Loop, ...]:
    """Mirror ``001_tileify._lift_output_loops``: collect top-level
    free Loops in ``stmts`` (the body remaining after the outer free
    chain is stripped) whose subtree contains a ``Write`` whose index
    references the loop's axis. These become additional thread axes
    once tileify runs."""
    out: list[Loop] = []
    for s in stmts:
        if isinstance(s, Loop) and not s.is_reduce and s.role is not Role.REGISTER and _writes_with_axis(s.body, s.axis.name):
            out.append(s)
    return tuple(out)


def _writes_with_axis(stmts: tuple, axis_name: str) -> bool:
    """Mirror ``001_tileify._writes_with_axis``: any Write under
    ``stmts`` whose index references ``axis_name``."""
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


# --- cooperative-reduce branch ---------------------------------------


def _try_cooperative_reduce(loop_op: LoopOp) -> list[LoopOp] | None:
    """Tag cooperative-viable reduce Loops with ``Role.COOPERATIVE_STRIDE``
    and fork over the cooperative thread count ``BN``.

    Mirrors ``004_launch_geometry._cooperative_viable`` at the LoopOp
    level: the inner body (post-chain) has at least one reduce Loop
    with extent ≥ WARP_SIZE, every reduce has an ``Accum``, and Accums
    are independent. For each viable kernel:

    1. Tag every top-level reduce Loop in the inner body with
       ``Role.COOPERATIVE_STRIDE``.
    2. Fork over BN candidates (heuristic ``_effective_block_size``
       first; powers-of-two in ``_TUNE_AXIS_CHOICES`` ≥ WARP_SIZE next).
    3. Stamp ``knobs={"BN": bn}`` per variant.

    Returns ``None`` when not cooperative-viable (matmul-shape, no
    reduce, or short reduce axis)."""
    from deplodock.compiler.tuning import _has_matmul_reduce  # noqa: PLC0415

    if _has_matmul_reduce(loop_op.body):
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    if not _cooperative_shape(loop_op.body, chain):
        return None
    inner_body = tuple(chain[-1].body) if chain else tuple(loop_op.body)
    reduce_loops = [s for s in inner_body if isinstance(s, Loop) and s.is_reduce]
    if not reduce_loops:
        return None
    reduce_extent = int(reduce_loops[0].axis.extent)

    tagged_inner = tuple(replace(s, role=Role.COOPERATIVE_STRIDE) if (isinstance(s, Loop) and s.is_reduce) else s for s in inner_body)
    new_body = _rebuild_outer_chain(chain, tagged_inner) if chain else tagged_inner

    heuristic_bn = _effective_block_size(reduce_extent)
    bn_set = {bn for bn in _TUNE_AXIS_CHOICES if _WARP_SIZE <= bn <= heuristic_bn}
    bn_set.add(heuristic_bn)
    ordered = [heuristic_bn] + sorted([bn for bn in bn_set if bn != heuristic_bn], reverse=True)

    variants: list[LoopOp] = []
    for bn in ordered:
        knobs = dict(loop_op.knobs)
        knobs["BN"] = bn
        variants.append(LoopOp(body=new_body, knobs=knobs))
    return variants


def _effective_block_size(reduce_extent: int) -> int:
    """Threads/CTA for the cooperative reduce — mirrors
    ``004_launch_geometry._effective_block_size``."""
    from deplodock.compiler.ir.tile.ir import BLOCK_SIZE  # noqa: PLC0415

    p = 1
    while p < reduce_extent:
        p <<= 1
    return max(_WARP_SIZE, min(BLOCK_SIZE, p))


def _rebuild_outer_chain(chain: tuple[Loop, ...], new_inner: tuple) -> tuple:
    """Re-wrap ``new_inner`` with the outer free-Loop chain unchanged."""
    result = new_inner
    for lp in reversed(chain):
        result = (Loop(axis=lp.axis, body=result, unroll=lp.unroll, role=lp.role),)
    return result


def _has_cooperative_stride(stmts) -> bool:
    """True iff any reachable reduce Loop carries
    ``Role.COOPERATIVE_STRIDE``."""
    for s in stmts:
        if isinstance(s, Loop) and s.role is Role.COOPERATIVE_STRIDE:
            return True
        if isinstance(s, (Loop, StridedLoop)) and _has_cooperative_stride(s.body):
            return True
        if isinstance(s, Cond):
            if _has_cooperative_stride(s.body) or _has_cooperative_stride(s.else_body):
                return True
    return False


# --- matmul (BN, BM) stamp fork --------------------------------------


def _try_matmul_bn_bm_fork(loop_op: LoopOp) -> list[LoopOp] | None:
    """Enumerate ``(BN, BM)`` candidates for a matmul kernel and stamp
    them as knobs (no body change). ``004_launch_geometry`` reads BN/BM
    from knobs and does the deterministic axis split.

    Mirrors ``004_launch_geometry._matmul_variants`` clamping logic:
    candidates from ``_TUNE_AXIS_CHOICES``, clamped to the chain's
    outer two extents, dropped on non-divisibility, deduped after
    clamp. Heuristic shape (from :func:`tuning.thread_tile_shape`) is
    variant 0.

    Returns ``None`` for non-matmul kernels, kernels with < 2 outer
    chain Loops, or kernels that already carry ``BN`` in knobs (e.g.
    cooperative kernels — they stamp BN via the cooperative branch)."""
    from deplodock.compiler.tuning import _has_matmul_reduce, thread_tile_shape  # noqa: PLC0415

    if not _has_matmul_reduce(loop_op.body):
        return None
    if "BN" in loop_op.knobs:
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None

    ext_outer = int(chain[-2].axis.extent)
    ext_inner = int(chain[-1].axis.extent)

    synthetic = Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain),
        body=chain[-1].body,
    )
    heuristic = thread_tile_shape(synthetic)
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
        # Skip noop partitions — both axes staying full THREAD means
        # 004's ``_plan_partition`` produces no split. Greedy would
        # then have no launch geometry to use.
        if bn == ext_inner and bm == ext_outer:
            return
        # Reject candidates whose THREAD axis product exceeds the CTA
        # limit (1024). ``TileOp.validate`` rejects them downstream once
        # FM is stamped, and greedy has no fallback if every variant
        # fails validate. Mirrors the post-clamp thread layout: the two
        # innermost axes become BN × BM THREAD; outer chain axes go
        # BLOCK (no thread contribution).
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


# Mirrors ``003_split_matmul_k._SPLITK_CANDIDATES``. Planner stamps one
# SPLITK value per variant; ``003_split_matmul_k`` reads the stamp and
# does the σ-split + epilogue rewrite deterministically.
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)


def _try_splitk(ctx: Context, loop_op: LoopOp) -> list[LoopOp] | None:
    """Enumerate ``SPLITK`` candidates for a K-chunked matmul kernel
    and stamp them as knobs (no body change). The heuristic from
    :func:`tuning.auto_splitk` against the synthetic Tile (post-register-
    tile, post-K-chunk) becomes variant 0 — when it's ``1`` the fork
    skips because no candidate would split the K_o loop.

    Returns ``None`` when the kernel isn't matmul-shaped, hasn't been
    K-chunked (no ``Role.SERIAL_OUTER`` K loop), or already carries
    ``SPLITK`` in knobs."""
    from deplodock.compiler.tuning import _has_matmul_reduce, auto_splitk  # noqa: PLC0415

    if not _has_matmul_reduce(loop_op.body):
        return None
    if "SPLITK" in loop_op.knobs:
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    inner_body = tuple(chain[-1].body) if chain else tuple(loop_op.body)
    k_o = _find_serial_outer_k(inner_body)
    if k_o is None:
        return None
    K_o_extent = int(k_o.axis.extent)

    synthetic = Tile(
        axes=tuple(BoundAxis(axis=lp.axis, bind=BIND_THREAD) for lp in chain) if chain else (),
        body=inner_body,
    )
    heuristic = int(auto_splitk(synthetic, K_o_extent))
    if heuristic <= 1:
        return None
    # No fork — one-shot stamp matching today's 003 behavior. The
    # autotune layer can search SPLITK by overriding via env knobs;
    # multi-variant SPLITK fork is out of scope for now.
    knobs = {**loop_op.knobs, "SPLITK": heuristic}
    return [LoopOp(body=loop_op.body, knobs=knobs)]


def _find_serial_outer_k(stmts) -> Loop | None:
    """Find the first ``Loop`` with ``Role.SERIAL_OUTER`` reachable in
    ``stmts`` (the M7 K-chunk marker)."""
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
