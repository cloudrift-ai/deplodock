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
from deplodock.compiler.ir.stmt import Cond, Load, Loop, Stmt, StridedLoop, iter_body
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

    # Rewrite body Loads of staged buffers to target the staged name
    # with cache-local indices. ``cache_positions[buf]`` lists the
    # positions in the original load index that hold cache-axis Vars.
    redirects = {st.buf: (st.name, _cache_positions(st)) for st in stages}
    rewritten_body = tuple(_rewrite_loads(s, redirects) for s in tile.body)

    new_tile = Tile(axes=tile.axes, body=tuple([*stages, *rewritten_body]))
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

        stages.append(Stage(name=f"{buf}_stage", buf=buf, index=ref.index, axes=tuple(cache_axes)))

    return stages


def _cache_positions(stage: Stage) -> tuple[int, ...]:
    """Positions in ``stage.index`` whose Var name is a cache axis."""
    cache_names = {ax.name for ax in stage.axes}
    return tuple(i for i, e in enumerate(stage.index) if isinstance(e, Var) and e.name in cache_names)


def _rewrite_loads(stmt: Stmt, redirects: dict) -> Stmt:
    """Recursively rewrite ``Load(buf, ...)`` to ``Load(stage_name,
    cache-local index)`` for every staged buf. The cache-local index is
    the load's index restricted to the cache positions — at each Load
    site this picks up the loop's own iteration Var (which may be
    named differently than the Stage's reference Var, e.g. softmax has
    ``a1`` in the reduce loop and ``a2`` in the output loop)."""
    if isinstance(stmt, Load) and stmt.input in redirects:
        stage_name, positions = redirects[stmt.input]
        return Load(name=stmt.name, input=stage_name, index=tuple(stmt.index[i] for i in positions))
    if isinstance(stmt, Loop):
        return Loop(axis=stmt.axis, body=tuple(_rewrite_loads(c, redirects) for c in stmt.body))
    if isinstance(stmt, StridedLoop):
        return StridedLoop(
            axis=stmt.axis,
            start=stmt.start,
            step=stmt.step,
            body=tuple(_rewrite_loads(c, redirects) for c in stmt.body),
        )
    if isinstance(stmt, Cond):
        return Cond(
            cond=stmt.cond,
            body=tuple(_rewrite_loads(c, redirects) for c in stmt.body),
            else_body=tuple(_rewrite_loads(c, redirects) for c in stmt.else_body),
        )
    return stmt
