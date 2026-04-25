"""Operand-staging strategy — caches input buffers that are read by
multiple BoundLoops in a cooperative ``Tile`` body.

Targets the softmax / norm-style fusion shape, where one input row is
read three times (max-reduction, sum-of-exp reduction, output write).
Without staging, the same global memory is fetched three times. Adding
a ``Stage`` declaration at the top of the Tile body causes
materialization to load the row once into shared memory cooperatively,
then all three phases read from smem.

Pre-rewrite (post cooperative-reduce strategy)::

    Tile(axes=(a0=BLOCK, a2=BLOCK_STRIDED)):
      BoundLoop(a1, BLOCK_STRIDED):
        in0 = load input[a0, a1]
        ...
      Combine(acc0, max)
      BoundLoop(a1, BLOCK_STRIDED):
        in1 = load input[a0, a1]
        ...
      Combine(acc1, add)
      BoundLoop(a2, BLOCK_STRIDED):
        in2 = load input[a0, a2]
        ...

Post-rewrite::

    Tile(axes=(a0=BLOCK, a2=BLOCK_STRIDED)):
      Stage(input[a0, a1], axes=(a1,))    # ← inserted
      BoundLoop(a1, BLOCK_STRIDED): ...
      Combine(acc0, max)
      ...

Trigger conditions:

- ``Tile.block_axes`` non-empty (cooperative — staging needs smem).
- Body contains ≥2 BoundLoops loading the same input buffer.
- Each load's index has all dimensions classifiable as block-bound
  (Var name in ``Tile.axes`` ``BIND_BLOCK`` axes) or iterated (Var name
  matching a body BoundLoop's axis).
- Block-bound dimensions are *consistent* across all loads (same Var at
  the same position).
- Cache footprint ≤ ``STAGE_BYTES_LIMIT``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_BLOCK, Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.ir.tile.ir import BoundLoop, Stage, Tile, TileOp
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
    """Find input buffers loaded by multiple BoundLoops; return Stage
    Stmts to insert at the top of the Tile body."""
    block_axis_names = {ba.axis.name for ba in tile.axes if ba.bind == BIND_BLOCK}
    loop_axes_by_name: dict[str, Axis] = {}

    # Per buf, collect the (BoundLoop axis name, Load) pairs found in the body.
    loads_per_buf: dict[str, list[tuple[str, Load]]] = {}
    for s in tile.body:
        if not isinstance(s, BoundLoop):
            continue
        loop_axis = s.axis
        loop_axes_by_name[loop_axis.name] = loop_axis
        for inner in s.body:
            if isinstance(inner, Load):
                loads_per_buf.setdefault(inner.input, []).append((loop_axis.name, inner))

    stages: list[Stage] = []
    for buf, entries in loads_per_buf.items():
        # Need ≥2 distinct BoundLoops loading this buffer.
        if len({lname for lname, _ in entries}) < 2:
            continue

        ref_load = entries[0][1]
        # Classify each index position; require all positions to be
        # either block-bound or iterated by some body BoundLoop.
        cache_positions, classification_ok = _classify_index(ref_load.index, block_axis_names, loop_axes_by_name)
        if not classification_ok:
            continue

        # Verify other Loads have the same block-bound dim values.
        if not _block_dims_consistent(ref_load.index, [load for _, load in entries], cache_positions):
            continue

        # Build the Stage's axes from the cache positions.
        stage_axes = tuple(loop_axes_by_name[ref_load.index[i].name] for i in cache_positions)

        # Cache footprint check.
        cache_elems = 1
        for ax in stage_axes:
            cache_elems *= int(ax.extent)
        if cache_elems * DTYPE_BYTES > STAGE_BYTES_LIMIT:
            continue

        stages.append(Stage(buf=buf, index=ref_load.index, axes=stage_axes))

    return stages


def _classify_index(index: tuple, block_axis_names: set[str], loop_axes_by_name: dict[str, Axis]) -> tuple[tuple[int, ...], bool]:
    """Return (cache_positions, ok). cache_positions are positions that
    iterate (Var name matches some body BoundLoop's axis). ``ok`` is
    True iff every position is classifiable (block-bound or iterated)."""
    cache_positions: list[int] = []
    for i, e in enumerate(index):
        if not isinstance(e, Var):
            return (), False
        if e.name in block_axis_names:
            continue  # block-bound
        if e.name in loop_axes_by_name:
            cache_positions.append(i)
            continue
        return (), False  # unclassified (e.g. constant or unknown axis)
    return tuple(cache_positions), True


def _block_dims_consistent(ref_index: tuple, all_loads: list[Load], cache_positions: tuple) -> bool:
    """Every Load's non-cache (i.e. block-bound) positions must use the
    same Var as the reference."""
    cache_set = set(cache_positions)
    for load in all_loads:
        if len(load.index) != len(ref_index):
            return False
        for i, (a, b) in enumerate(zip(ref_index, load.index, strict=True)):
            if i in cache_set:
                continue  # cache positions can differ across phases
            if not isinstance(a, Var) or not isinstance(b, Var) or a.name != b.name:
                return False
    return True
