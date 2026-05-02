"""Split each matmul-shaped inner reduce ``Loop`` (``K_i``) into N
consecutive halves of equal extent, gated by ``DEPLODOCK_BK_SPLIT``.

Motivates the BK=64 + B128 swizzle case: ``cuTensorMapEncodeTiled``
caps inner box bytes at the swizzle width (128 B for B128 fp32 = 32
fp32 elements), so a BK=64 cache axis can't be loaded with a single
TMA box. Splitting ``K_i`` into ``N=2`` halves yields two consecutive
reduce loops, each with ``K_i.extent / 2 = 32``. Downstream:

- ``007_stage_inputs`` classifies each half's Loads independently,
  producing ``2 * num_buffers`` BufferedStages (e.g. weight_lo +
  weight_hi + input_lo + input_hi for a 2-input matmul).
- ``014a_tma_copy`` narrows each to ``TmaBufferedStage`` (inner=32 fp32,
  TMA-legal).
- ``014d_tma_swizzle`` selects B128 swizzle per stage (inner exactly 128 B).
- ``015_pipeline_async``'s ``len(stages) >= 2`` gate is trivially
  satisfied; pipelining covers all halves with one shared mbarrier.
- ``001_materialize_tile`` distributes issuer threads across the now-
  larger stage count (tid 0..N-1).

Default ``DEPLODOCK_BK_SPLIT=1`` is a no-op. ``=2`` halves every
matmul-shaped K_i whose extent is divisible by 2 *and* by 32 after
the split (so the post-split halves are valid TMA box widths). Higher
N is permitted for symmetry but only N ∈ {1, 2, 4} typically matter.

Runs immediately after ``002_split_matmul_k`` (which produces the
``K_o`` / ``K_i`` shape) so the rule sees a single inner reduce per
matmul. Idempotence: an already-split body has multiple consecutive
matmul reduces under one ``K_o`` and the rule re-fires per loop, but
the env-var-driven split count prevents runaway recursion (a halved
body with extent 32 fails the divisibility gate at N=2).
"""

from __future__ import annotations

import os
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, single_tile

PATTERN = [Pattern("root", TileOp)]

# Post-split halves below this width (in fp32) provide no benefit for
# the TMA+swizzle case (B128 swizzle requires inner=32 fp32 = 128 B
# exactly). Refuse to split if any half would be smaller.
_MIN_HALF_EXTENT = 32


def rewrite(graph: Graph, root: Node) -> Graph | None:
    n = _split_count()
    if n <= 1:
        raise RuleSkipped("DEPLODOCK_BK_SPLIT <= 1 — no-op")
    new_body = _maybe_rewrite(root.op.body, n)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _split_count() -> int:
    raw = os.environ.get("DEPLODOCK_BK_SPLIT", "1")
    try:
        n = int(raw)
    except ValueError as e:
        raise ValueError(f"DEPLODOCK_BK_SPLIT must be a positive integer, got {raw!r}") from e
    if n < 1:
        raise ValueError(f"DEPLODOCK_BK_SPLIT must be >= 1, got {n}")
    return n


def _maybe_rewrite(body, n: int):
    idx, tile = single_tile(body)
    new_tile_body, changed = _split_in_body(tile.body, n)
    if not changed:
        raise RuleSkipped(f"no matmul reduce divisible by {n} with halves >= {_MIN_HALF_EXTENT}")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _split_in_body(stmts: tuple[Stmt, ...], n: int) -> tuple[tuple[Stmt, ...], bool]:
    """Walk a body, splitting the first eligible matmul reduce found.
    Mirrors ``002_split_matmul_k._chunk_in_body``: recurses through
    wrapper Loops / StridedLoops / Conds so a matmul nested inside a
    free output-position loop is reachable."""
    out: list[Stmt] = []
    changed = False
    for s in stmts:
        if changed:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            split = _split_loop(s, n)
            if split is not None:
                out.extend(split)
                changed = True
                continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, ic = _split_in_body(s.body, n)
            if ic:
                out.append(dc_replace(s, body=inner))
                changed = True
                continue
        if isinstance(s, Cond):
            inner_b, cb = _split_in_body(s.body, n)
            inner_e, ce = _split_in_body(s.else_body, n)
            if cb or ce:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                changed = True
                continue
        out.append(s)
    return tuple(out), changed


def _split_loop(loop: Loop, n: int) -> tuple[Loop, ...] | None:
    """Split a matmul-shaped reduce Loop into ``n`` consecutive halves
    with sigma-shifted K indices. Returns ``None`` if the extent isn't
    cleanly divisible or the resulting half is too narrow."""
    K = int(loop.axis.extent)
    if K % n != 0:
        return None
    half = K // n
    if half < _MIN_HALF_EXTENT:
        return None

    K_name = loop.axis.name
    halves: list[Loop] = []
    for i in range(n):
        # Each half iterates ``half`` times with K_var rebound to
        # ``K_var + i*half``. Reusing ``K_name`` keeps Sigma rewrites
        # under downstream passes (007, 014x) symbolically identical
        # across halves.
        sigma = Sigma({K_name: Var(K_name) + Literal(i * half, "int")})
        new_body = tuple(s.rewrite(_id, sigma) for s in loop.body)
        halves.append(Loop(axis=Axis(K_name, half), body=new_body, unroll=loop.unroll))
    return tuple(halves)


def _id(name: str) -> str:
    return name
