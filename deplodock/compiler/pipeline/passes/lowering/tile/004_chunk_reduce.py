"""K-chunk a matmul-shaped Tile's reduce loop into outer (K_o) + inner
(K_i) loops.

After ``003_block_matmul`` splits M / N output axes into BLOCK + THREAD
pairs, the K reduction is still a single serial Loop iterating one
element at a time per K step. This pass splits it::

    Loop(K, reduce, body=B)
    →
    Loop(K_o, reduce, body=Loop(K_i, reduce, body=B[K → K_o*BK + K_i]))

Effect downstream: ``005_stage_inputs`` picks up ``K_i`` as a cache axis
on each operand Load (along with the operand's THREAD axis), producing
``BM × BK`` and ``BK × BN`` smem stages refilled once per K_o iteration
instead of once per K element. Per-block ``__syncthreads`` count drops
by ``BK×``; per-thread compute density and load granularity rise
correspondingly.

Pattern guard — only fires on tiles that look like matmuls: ≥2 BLOCK
axes (M-block + N-block) and the body's outermost top-level reduce loop
isn't already chunked. Pure-elementwise + softmax / RMSNorm tiles have
1 BLOCK axis at most and aren't touched.

BK selection: largest divisor of K from ``_BK_CANDIDATES``. K must be
strictly greater than BK so the split actually fires.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_BLOCK, Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Loop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern

PATTERN = [Pattern("root", TileOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body):
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    new_tile = _chunk(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _chunk(tile: Tile) -> Tile | None:
    block_axes = [ba for ba in tile.axes if ba.bind == BIND_BLOCK]
    if len(block_axes) < 2:
        return None  # not matmul-shaped — single-axis reduce kernels stay unchunked

    reduce_idx = None
    for i, s in enumerate(tile.body):
        if isinstance(s, Loop) and s.is_reduce:
            reduce_idx = i
            break
    if reduce_idx is None:
        return None
    reduce_loop = tile.body[reduce_idx]

    # Idempotence: already chunked if first stmt of body is another reduce.
    for s in reduce_loop.body:
        if isinstance(s, Loop) and s.is_reduce:
            return None

    K = int(reduce_loop.axis.extent)
    BK = next((c for c in _BK_CANDIDATES if K % c == 0 and K > c), None)
    if BK is None:
        return None

    K_name = reduce_loop.axis.name
    K_o = Axis(f"{K_name}_o", K // BK)
    K_i = Axis(f"{K_name}_i", BK)
    sigma = Sigma({K_name: Var(K_o.name) * Literal(BK, "int") + Var(K_i.name)})
    inner_body = tuple(s.rewrite(_id, sigma) for s in reduce_loop.body)
    nested = Loop(axis=K_o, body=(Loop(axis=K_i, body=inner_body),))
    new_body = tile.body[:reduce_idx] + (nested,) + tile.body[reduce_idx + 1 :]
    return Tile(axes=tile.axes, body=new_body)


def _id(name: str) -> str:
    return name
