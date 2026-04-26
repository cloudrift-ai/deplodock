"""Operand-staging strategy — caches input buffers read multiple times
in a cooperative ``Tile`` body.

Trigger: scan the body (transitively) for ``Load`` stmts. Any buffer
loaded ≥2 times is a candidate for caching. Cache axes are derived
from the load's index — every Var whose name matches an enclosing
``StridedLoop`` axis (the cooperative iteration axes). Stage if the
total smem footprint fits in ``STAGE_BYTES_LIMIT``.

Targets the softmax / RMSNorm / norm-style fusion shape, where one
input row is read several times across reduction + output phases.
Without staging, the same global memory is fetched repeatedly. With a
``Stage`` declaration at the top of the Tile body, materialization
loads the row once into shared memory cooperatively, then all phases
read from smem.

Pre-rewrite (post cooperative-reduce strategy)::

    Tile(axes=(t=THREAD, a0=BLOCK)):
      StridedLoop(a1 = t; < N; += BLOCK_SIZE):
        Load x[a0, a1]
        Accum
      Combine
      StridedLoop(a2 = t; < N; += BLOCK_SIZE):
        Load x[a0, a2]
        Write y[a0, a2]

Post-rewrite::

    Tile(axes=(t=THREAD, a0=BLOCK)):
      Stage(x[a0, a1], axes=(a1,))    # ← inserted
      StridedLoop(a1 = t; < N; += BLOCK_SIZE):
        Load x[a0, a1]    # ← redirected to x_stage[a1] at materialize time
        Accum
      Combine
      StridedLoop(a2 = t; < N; += BLOCK_SIZE): ...

Trigger conditions:

- ``Tile.block_axes`` non-empty (cooperative — staging needs smem).
- No existing ``Stage`` in the body (idempotence).
- Some buffer is loaded ≥2 times in the body.
- The Load's index references at least one cooperative-loop axis.
- Cache footprint ≤ ``STAGE_BYTES_LIMIT``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Load, StridedLoop, iter_body
from deplodock.compiler.ir.tile.ir import Stage, Tile, TileOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

# Cache size budget per staged buffer (per CUDA block). 16 KB ≈ one
# 4096-fp32 row, comfortably fits with smem headroom for accumulators.
STAGE_BYTES_LIMIT = 16 * 1024
DTYPE_BYTES = 4


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    new_body = _maybe_stage(tile_op.body)
    if new_body is None:
        return None
    node.op = TileOp(body=new_body, name=tile_op.name)
    return None


def _maybe_stage(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if not tile.block_axes:
        return None  # not cooperative — no smem for staging
    if any(isinstance(s, Stage) for s in tile.body):
        return None  # already staged

    stages = _collect_stages(tile)
    if not stages:
        return None

    new_tile = Tile(axes=tile.axes, body=tuple([*stages, *tile.body]))
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _collect_stages(tile: Tile) -> list[Stage]:
    """Scan the body for buffers loaded ≥2 times; return Stage Stmts to
    insert at the top of the Tile body."""
    # Cooperative-loop axes are the candidate cache axes — each one is
    # iterated cooperatively across the block's threads.
    coop_axes_by_name = {s.axis.name: s.axis for s in iter_body(tile.body) if isinstance(s, StridedLoop)}

    loads_per_buf: dict[str, list[Load]] = {}
    for s in iter_body(tile.body):
        if isinstance(s, Load):
            loads_per_buf.setdefault(s.input, []).append(s)

    stages: list[Stage] = []
    for buf, loads in loads_per_buf.items():
        if len(loads) < 2:
            continue
        ref = loads[0]
        cache_axes = []
        for e in ref.index:
            if isinstance(e, Var) and e.name in coop_axes_by_name:
                cache_axes.append(coop_axes_by_name[e.name])
        if not cache_axes:
            continue  # Load doesn't access cooperative axes — staging won't help

        cache_elems = 1
        for ax in cache_axes:
            cache_elems *= int(ax.extent)
        if cache_elems * DTYPE_BYTES > STAGE_BYTES_LIMIT:
            continue

        stages.append(Stage(buf=buf, index=ref.index, axes=tuple(cache_axes)))

    return stages
