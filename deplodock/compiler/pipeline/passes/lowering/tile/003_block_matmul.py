"""Generic blockify rule — splits each thread axis into outer-BLOCK +
inner-THREAD pair, with an optional free sub-tile loop that
``003a_unroll_free_loops`` then unrolls into per-thread register cells.

Pre-rewrite::

    Tile(axes=(t1 THREAD, ..., tk THREAD)):
        ... body ...

Post-rewrite (per blocked axis with sub-tile factor ``T``)::

    Tile(axes=(t1_i THREAD, ..., t1_o BLOCK, ...)):
        Loop(t1_t, free, extent=T):
            Loop(t2_t, free, extent=T):
                ... body with t<i> -> t<i>_o * (BLOCK_TG*T) + t<i>_t * BLOCK_TG + t<i>_i ...

The inner free Loops (``t<i>_t``) are picked up by the unroll pass:
each cell becomes a literal-substituted copy with SSA names tagged
``_uN``, producing ``T1·T2`` per-thread register accumulators. The
staging pass (``004_stage_inputs``) then sees the unrolled body's
literal-spread Loads and folds them into wider smem caches.

Sub-tile factors are picked from ``_SUBTILE_CHOICES`` to land in a
target block-count window (~2-4 blocks per SM on a 5090). If only one
axis is blocked, sub-tiling is disabled there (T=1).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import (
    BIND_BLOCK,
    BIND_THREAD,
    Axis,
    BoundAxis,
)
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_BLOCK_TG = 16
_MAX_THREADS_PER_BLOCK = 1024
# Cap on the number of axes we split into outer-BLOCK + inner-THREAD.
# Above 2 we'd push thread count past 256 / risk worse warp-coalescing.
_MAX_BLOCKED_AXES = 2

# Per-axis sub-tile candidates (free-loop wrapper extent), preferred
# first. Picked greedily to keep total grid block_count above
# ``_TARGET_MIN_BLOCKS``. Values match the historical matmul-rule's
# choices (microbenched on 5090 SGEMM, see scripts/sweep_subtile.py).
_SUBTILE_CHOICES = (4, 2, 1)
_TARGET_MIN_BLOCKS = 340  # ~2 blocks per SM on a 5090
# Max distinct Loads in the body for sub-tile to be enabled. Matmul
# (Load·Load·Mul·Accum) = 2; fused activations / residual adds bump it
# slightly. Bodies with more Loads (SDPA QK^T+RoPE) have too much
# per-cell state to replicate without register pressure spills.
_SUBTILE_MAX_LOADS = 4


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if tile.block_axes:
        return None  # idempotence
    new_tile = _blockify(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _blockify(tile: Tile) -> Tile | None:
    splittable = [a for a in tile.thread_axes if int(a.extent) % _BLOCK_TG == 0 and int(a.extent) >= _BLOCK_TG]
    if not splittable:
        return None

    # Pick up to _MAX_BLOCKED_AXES axes to split, ranked by cross-axis
    # reuse score — number of body Loads invariant in this axis (and
    # hence shared across threads varying it). Ties broken by larger
    # extent (more parallelism). Axes with score 0 still split but
    # come last (no smem-reuse benefit but they still tile cleanly).
    loads = list(_walk_loads(tile.body))
    thread_axis_names = {a.name for a in tile.thread_axes}
    scores: dict[str, int] = {}
    for axis in splittable:
        score = 0
        for load in loads:
            free = _index_free_vars(load.index)
            if axis.name not in free and (free & thread_axis_names):
                score += 1
        scores[axis.name] = score
    ranked = sorted(splittable, key=lambda a: (-scores[a.name], -int(a.extent)))
    blocked_axes = ranked[:_MAX_BLOCKED_AXES]
    blocked_names_set = {a.name for a in blocked_axes}

    # Sub-tile is only profitable when the body has cross-cell operand
    # reuse — i.e., for each candidate sub-tile axis, some Load in the
    # body is invariant in that axis (TM cells share that load). For
    # matmul: M-axis sub-tile reuses the B-load (which has only N, K),
    # N-axis sub-tile reuses the A-load. For non-matmul bodies (e.g.
    # SDPA's QK^T+RoPE kernel) too many Loads in the body make
    # replication too expensive → don't sub-tile.
    blocked_extents = {a.name: int(a.extent) for a in blocked_axes}
    passthrough_extents: dict[str, int] = {}
    for a in tile.thread_axes:
        if a.name in blocked_names_set:
            continue
        passthrough_extents[a.name] = int(a.extent)
    if _has_cross_axis_reuse(tile.body, blocked_names_set):
        sub_tiles = _pick_sub_tiles(blocked_extents, passthrough_extents)
    else:
        sub_tiles = {name: 1 for name in blocked_extents}

    new_axes: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}
    sub_axes: list[Axis] = []
    threads_so_far = 1

    for axis in tile.thread_axes:
        if axis.name in blocked_names_set and threads_so_far * _BLOCK_TG <= _MAX_THREADS_PER_BLOCK:
            t = sub_tiles.get(axis.name, 1)
            outer = Axis(f"{axis.name}_o", int(axis.extent) // (_BLOCK_TG * t))
            inner = Axis(f"{axis.name}_i", _BLOCK_TG)
            new_axes.append(BoundAxis(axis=inner, bind=BIND_THREAD))
            new_axes.append(BoundAxis(axis=outer, bind=BIND_BLOCK))
            if t > 1:
                sub = Axis(f"{axis.name}_t", t)
                sub_axes.append(sub)
                sigma_map[axis.name] = (
                    Var(outer.name) * Literal(_BLOCK_TG * t, "int")
                    + Var(sub.name) * Literal(_BLOCK_TG, "int")
                    + Var(inner.name)
                )
            else:
                sigma_map[axis.name] = Var(outer.name) * Literal(_BLOCK_TG, "int") + Var(inner.name)
            threads_so_far *= _BLOCK_TG
        elif threads_so_far * int(axis.extent) <= _MAX_THREADS_PER_BLOCK:
            new_axes.append(BoundAxis(axis=axis, bind=BIND_THREAD))
            threads_so_far *= int(axis.extent)
        else:
            new_axes.append(BoundAxis(axis=axis, bind=BIND_BLOCK))

    sigma = Sigma(sigma_map)
    new_body: tuple[Stmt, ...] = tuple(s.rewrite(_id, sigma) for s in tile.body)

    # Wrap the body in free Loops for each sub-tile axis, outermost first.
    # ``003a_unroll_free_loops`` will unroll these, replicating cells
    # with literal axis substitution. Subsequent staging then folds the
    # literal-spread Loads into wider cache axes via the existing
    # ``_cache_extent_across_loads`` path in ``004_stage_inputs``.
    for sub in reversed(sub_axes):
        new_body = (Loop(axis=sub, body=new_body),)

    return Tile(axes=tuple(new_axes), body=new_body)


def _has_cross_axis_reuse(body: tuple, blocked_names: set[str]) -> bool:
    """True iff (a) for EVERY blocked axis there's some Load invariant
    in it, AND (b) the body is small enough that sub-tile replication
    is profitable. Bigger bodies (e.g. SDPA's QK^T+RoPE: 6 loads + many
    elementwise ops) replicate so much per-cell work that register
    pressure outweighs any per-load reuse gain. The threshold matches
    the canonical matmul body (Load·Load·Mul·Accum-add = 4 stmts) plus
    a small margin for fused activations / residual adds.
    """
    if len(blocked_names) < 2:
        return False
    loads = list(_walk_loads(body))
    if not loads or len(loads) > _SUBTILE_MAX_LOADS:
        return False
    for axis in blocked_names:
        has_invariant = False
        for load in loads:
            free = _index_free_vars(load.index)
            if axis not in free and (free & blocked_names):
                has_invariant = True
                break
        if not has_invariant:
            return False
    return True


def _walk_loads(stmts: tuple) -> list[Load]:
    out: list[Load] = []
    for s in stmts:
        if isinstance(s, Load):
            out.append(s)
        elif isinstance(s, (Loop, StridedLoop)):
            out.extend(_walk_loads(s.body))
        elif isinstance(s, Cond):
            out.extend(_walk_loads(s.body))
            out.extend(_walk_loads(s.else_body))
    return out


def _index_free_vars(index) -> set[str]:
    out: set[str] = set()
    for e in index:
        out |= e.free_vars()
    return out


def _pick_sub_tiles(blocked_extents: dict[str, int], passthrough_extents: dict[str, int]) -> dict[str, int]:
    """Per-axis sub-tile pick. Each axis independently tries the largest
    factor in ``_SUBTILE_CHOICES`` that divides its extent and keeps the
    total grid block_count above ``_TARGET_MIN_BLOCKS``.

    Total grid = product of (blocked_axis.extent / (BLOCK_TG·T)) over
    blocked axes × product of passthrough_extents (those become BLOCK
    axes whole when they don't fit the thread budget).
    """
    if len(blocked_extents) < 2:
        # With only one blocked axis the sub-tile factor just shrinks
        # the grid without extra arithmetic intensity to amortize
        # against (no second operand to share across cells).
        return {name: 1 for name in blocked_extents}

    out: dict[str, int] = dict.fromkeys(blocked_extents, 1)
    pass_factor = 1
    for ext in passthrough_extents.values():
        pass_factor *= ext

    # Greedy in extent-ASCENDING order so smaller axes get priority for
    # higher sub-tile factors. For typical matmul shapes (M < N), this
    # produces TM ≥ TN — the historical preference, microbenched on
    # SGEMM as preferring "tall" tiles. Equal extents tie-break by
    # axis name for determinism.
    for name in sorted(blocked_extents, key=lambda n: (blocked_extents[n], n)):
        for t in _SUBTILE_CHOICES:
            if t == 1:
                continue
            if blocked_extents[name] % (_BLOCK_TG * t) != 0:
                continue
            trial = {**out, name: t}
            grid = pass_factor
            for n, ext in blocked_extents.items():
                grid *= ext // (_BLOCK_TG * trial[n])
            if grid >= _TARGET_MIN_BLOCKS:
                out[name] = t
                break
    return out


def _id(name: str) -> str:
    return name
