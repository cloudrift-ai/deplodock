"""Top up the per-CTA thread count to ``_TARGET_THREADS`` by absorbing
inner slices of ``BIND_BLOCK`` axes into ``BIND_THREAD``.

After ``008_register_tile`` splits each of the two PAT-extent THREAD
axes by F (each becomes ``PAT/F``), matmul kernels drop to ``(PAT/F)²``
threads / CTA — typically 64 (PAT=16, F=2). This pass brings the count
back up to ``_TARGET_THREADS`` (256) by carving an inner slice off a
BLOCK axis and binding it to THREAD instead.

Order matters: this runs *before* ``010_stage_inputs`` so the staging
pass sees the final thread-axis set. The downside is that the new
thread axis may share a source-buffer dim with an existing thread
axis (e.g. for a matmul, splitting the N-block axis adds a thread Var
that combines with the existing N-thread Var in the W-Stage origin).
The current staging classifier rejects those multi-thread-axis-per-dim
cases, so register-tiled matmul kernels currently fall back to direct
DRAM Loads under this ordering — a known regression that motivates
extending the staging classifier.

The chosen axis is the largest BLOCK axis whose extent has a divisor
in ``[2, gap]`` where ``gap = _TARGET_THREADS / current_threads``.

Idempotence: skip when ``current_threads >= _TARGET_THREADS`` or no
suitable BLOCK axis exists.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_TARGET_THREADS = 256


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        raise RuleSkipped(f"need exactly one Tile in TileOp.body, found {len(tiles)}")
    idx, tile = tiles[0]

    threads_used = 1
    for ba in tile.axes:
        if ba.bind == BIND_THREAD:
            threads_used *= int(ba.axis.extent)
    if threads_used >= _TARGET_THREADS:
        raise RuleSkipped(f"already at thread budget: threads_used={threads_used} >= target={_TARGET_THREADS}")

    gap = _TARGET_THREADS // threads_used
    chosen: tuple[int, BoundAxis, int] | None = None
    for i, ba in enumerate(tile.axes):
        if ba.bind != BIND_BLOCK:
            continue
        ext = int(ba.axis.extent)
        slice_size = max((d for d in range(2, gap + 1) if ext % d == 0), default=1)
        if slice_size <= 1:
            continue
        if chosen is None or ext > int(chosen[1].axis.extent):
            chosen = (i, ba, slice_size)
    if chosen is None:
        raise RuleSkipped(f"no BLOCK axis with a divisor in [2,{gap}] to slice for more threads")

    axis_idx, ba, slice_size = chosen
    orig = ba.axis
    ext = int(orig.extent)
    outer = Axis(name=f"{orig.name}_o", extent=ext // slice_size)
    inner = Axis(name=f"{orig.name}_i", extent=slice_size)

    new_axes = (
        *tile.axes[:axis_idx],
        BoundAxis(axis=outer, bind=BIND_BLOCK),
        *tile.axes[axis_idx + 1 :],
        BoundAxis(axis=inner, bind=BIND_THREAD),
    )
    sigma = Sigma({orig.name: Var(outer.name) * Literal(slice_size, "int") + Var(inner.name)})
    new_body_inner = tuple(s.rewrite(_id, sigma) for s in tile.body)
    new_tile = Tile(axes=new_axes, body=new_body_inner)
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _id(name: str) -> str:
    return name
