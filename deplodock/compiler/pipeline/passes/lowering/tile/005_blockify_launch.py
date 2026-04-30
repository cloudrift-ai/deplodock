"""Launch-geometry pass.

Decides the THREAD/BLOCK partition of every parallel axis on a
``TileOp`` containing one ``Tile`` whose ``block_axes`` is still empty.
Replaces the legacy ``003_block_matmul`` rule with a single
launch-geometry decision informed by the actual per-block thread
budget. Output-dim free-Loop lifting happens earlier in
``001_tileify`` — by the time this rule runs, every parallel axis is
already in ``Tile.axes``.

**Apportion THREAD/BLOCK.** Every axis in ``Tile.axes`` starts as
THREAD (from ``001_tileify``). Walk innermost-to-outermost, accumulating
``threads_used`` against the per-block budget ``thread_budget()``
(``tuning.thread_budget``, default 256, env ``DEPLODOCK_TB``).

For each axis with extent ``ext`` and per-axis tile width
``pat = per_axis_threads(tile)`` (typically 16):

- ``threads_used >= budget`` — axis goes BLOCK whole.
- ``ext < pat`` — THREAD whole if it still fits the budget, else BLOCK.
- ``ext == pat`` — same fit-or-BLOCK check.
- ``ext > pat`` and ``ext % pat == 0`` — split into ``axis_i:pat``
  (THREAD) and ``axis_o:ext/pat`` (BLOCK), with body indices
  σ-rewritten ``axis → axis_o*pat + axis_i``.
- ``ext > pat`` and not divisible — THREAD whole if it fits, else BLOCK.

The split factor is always ``pat`` (the per-axis tile width), not the
remaining budget. The budget governs *whether* an axis goes THREAD
at all, while ``pat`` governs *how* a THREAD-eligible axis with
oversized extent is sliced.

Idempotent: if no axis was split and every axis kept its original
bind, returns None.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.tuning import per_axis_threads, thread_budget

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        raise RuleSkipped(f"need exactly one Tile in TileOp.body, found {len(tiles)}")
    idx, tile = tiles[0]
    if tile.block_axes:
        raise RuleSkipped("Tile already partitioned (block_axes non-empty)")

    partitioned = _partition_threads(tile)
    if partitioned is None:
        raise RuleSkipped("partition already fits within thread budget")
    return body[:idx] + (partitioned,) + body[idx + 1 :]


# ---------------------------------------------------------------------------
# Apportion THREAD / BLOCK
# ---------------------------------------------------------------------------


def _partition_threads(tile: Tile) -> Tile | None:
    """Walk axes innermost→outermost. Each axis with extent ≥
    ``_PER_AXIS_THREADS`` gets a ``_PER_AXIS_THREADS``-thread inner
    slice + remainder BLOCK; smaller axes go THREAD whole. Stop adding
    THREAD slices once the running product reaches ``_THREAD_BUDGET``;
    remaining outer axes go BLOCK whole.

    The per-axis 16-thread tile matches the legacy ``003`` semantics: a
    matmul (M, N both ≥16) gets a 16×16 thread tile, leaving each
    operand load with its own thread axis for cooperative reuse.
    Single-parallel-axis kernels (RMSNorm row-reduction) get a single
    16-thread axis — same as before."""
    pat = per_axis_threads(tile)
    tb = thread_budget()
    axes = list(tile.axes)
    new_axes_inner_first: list[BoundAxis] = []
    sigma_map: dict[str, object] = {}
    threads_used = 1

    for ba in reversed(axes):
        ext = int(ba.axis.extent)
        if threads_used >= tb:
            new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        if ext < pat:
            # Small axis — keep whole as THREAD if it'd fit, else BLOCK.
            if threads_used * ext <= tb:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
                threads_used *= ext
            else:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        if ext == pat:
            if threads_used * ext <= tb:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
                threads_used *= ext
            else:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        # Larger than per-axis tile → split.
        if ext % pat != 0:
            # Non-divisible — keep whole; if it'd overflow, BLOCK; else THREAD.
            if threads_used * ext <= tb:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_THREAD))
                threads_used *= ext
            else:
                new_axes_inner_first.append(BoundAxis(axis=ba.axis, bind=BIND_BLOCK))
            continue
        inner = Axis(f"{ba.axis.name}_i", pat)
        outer = Axis(f"{ba.axis.name}_o", ext // pat)
        new_axes_inner_first.append(BoundAxis(axis=inner, bind=BIND_THREAD))
        new_axes_inner_first.append(BoundAxis(axis=outer, bind=BIND_BLOCK))
        sigma_map[ba.axis.name] = Var(outer.name) * Literal(pat, "int") + Var(inner.name)
        threads_used *= pat

    new_axes_inner_first.reverse()
    new_axes = new_axes_inner_first

    # No-op short-circuit: identical axes, no split.
    if not sigma_map and len(new_axes) == len(axes):
        same = all(a.axis is b.axis and a.bind == b.bind for a, b in zip(new_axes, axes, strict=True))
        if same:
            return None

    sigma = Sigma(sigma_map) if sigma_map else Sigma.IDENTITY
    new_body = tuple(s.rewrite(_id, sigma) for s in tile.body) if sigma_map else tile.body
    return Tile(axes=tuple(new_axes), body=new_body)


def _id(name: str) -> str:
    return name
