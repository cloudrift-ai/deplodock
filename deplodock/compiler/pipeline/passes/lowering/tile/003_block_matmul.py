"""Block-tiled matmul strategy — splits M/N/K into outer/inner tiles
and reshapes the ``Tile`` so each CUDA block owns a BM·BN output region
walked cooperatively across threads, with the K reduction chunked into
BK-sized slices.

Mirrors the cooperative-reduce pattern from ``002_cooperative_reduce.py``:
the rule splits the M / N / K axes, builds the nested per-thread output
loop, and inserts ``Stage(A, axes=(m_i, k_i))`` /
``Stage(B, axes=(k_i, n_i))`` at the K_o loop head so each K-chunk's
tiles get cached in smem. Matmul-aware materialization
(``passes/lowering/kernel/001_materialize_tile.py``) expands the Stages
and consumes the ``BIND_THREAD`` output axes for the launch geometry.

Default tile sizes ``BM = BN = BK = 16`` give exactly one output per
thread (``BM·BN == BLOCK_SIZE = 256``); no per-thread sub-tiling
needed. Larger tiles (typical SGEMM) require a thread-tile extension
to ``materialize_tile``.

Pre-rewrite (post ``lower_naive`` of fused matmul ``C = A @ B``)::

    Tile(axes=(m THREAD, n THREAD)):
      Loop(k):
        a = Load("A", (m, k))
        b = Load("B", (k, n))
        t = Assign(a * b)
        Accum(acc, op=add, value=t)
      Write("C", (m, n), acc)

Post-rewrite::

    Tile(axes=(m_o BLOCK, n_o BLOCK, m_i THREAD, n_i THREAD)):
      Loop(m_i):
        Loop(n_i):
          Loop(k_o):
            Loop(k_i):
              a = Load("A", (m_o*BM + m_i, k_o*BK + k_i))
              b = Load("B", (k_o*BK + k_i, n_o*BN + n_i))
              t = Assign(a * b)
              Accum(acc, op=add, value=t)
          Write("C", (m_o*BM + m_i, n_o*BN + n_i), acc)

No ``Combine`` — every output element is owned by exactly one thread.

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.thread_axes`` is exactly 2D, ``block_axes`` is empty (idempotence).
- ``Tile.body`` is the canonical fused-matmul shape: one reduce ``Loop``
  (Load · Load · Assign-mul · Accum-add, with two distinct source bufs)
  followed by a ``Write`` over the two output axes.
- M, N, K extents are each divisible by the configured tile size.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import (
    BIND_BLOCK,
    BIND_THREAD,
    BoundAxis,
)
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import Tile, TileOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

# Default tile sizes. ``BM·BN == BLOCK_SIZE`` (16·16 = 256) is the
# "one output per thread" first cut — no per-thread output sub-tiling
# needed. Larger tiles (the real-SGEMM choice) need a thread-tile
# extension to materialize_tile.
BM = 16
BN = 16
BK = 16


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    tile_op: TileOp = node.op

    new_body = _maybe_rewrite(tile_op.body)
    if new_body is None:
        return None
    node.op = TileOp(body=new_body, name=tile_op.name)
    return None


def _maybe_rewrite(body: tuple) -> tuple | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        return None
    idx, tile = tiles[0]
    if tile.block_axes:
        return None  # idempotence — already block-bound
    new_tile = _rewrite_tile(tile)
    if new_tile is None:
        return None
    return body[:idx] + (new_tile,) + body[idx + 1 :]


def _rewrite_tile(tile: Tile) -> Tile | None:
    if len(tile.thread_axes) != 2:
        return None
    m, n = tile.thread_axes
    if m.extent % BM or n.extent % BN:
        return None

    parsed = _match_matmul_body(tile.body, m_name=m.name, n_name=n.name)
    if parsed is None:
        return None
    reduce_loop, write = parsed
    k = reduce_loop.axis
    if k.extent % BK:
        return None

    extracted = _extract_inner(reduce_loop)
    if extracted is None:
        return None
    accum, load_a, load_b, mul = extracted

    m_o, m_i = m.split(BM)
    n_o, n_i = n.split(BN)
    k_o, k_i = k.split(BK)

    sigma = Sigma(
        {
            m.name: Var(m_o.name) * Literal(BM, "int") + Var(m_i.name),
            n.name: Var(n_o.name) * Literal(BN, "int") + Var(n_i.name),
            k.name: Var(k_o.name) * Literal(BK, "int") + Var(k_i.name),
        }
    )

    new_load_a = load_a.rewrite(_id, sigma)
    new_load_b = load_b.rewrite(_id, sigma)
    inner_compute: tuple[Stmt, ...] = (new_load_a, new_load_b, mul, accum)

    # m_i / n_i are bound directly as Tile THREAD axes (their
    # extents·product equals BLOCK_SIZE — one output per thread). No
    # wrapping loops over them in the body — the thread decode at
    # render time provides Var(m_i) / Var(n_i) for the inner compute.
    # Operand staging is left to ``004_stage_inputs`` — the K_o > K_i
    # nesting plus the THREAD axes (a0_i, a1_i) in scope give the
    # staging rule everything it needs to spot the reuse.
    new_body: tuple[Stmt, ...] = (
        Loop(axis=k_o, body=(Loop(axis=k_i, body=inner_compute),)),
        write.rewrite(_id, sigma),
    )

    new_axes = (
        BoundAxis(axis=m_i, bind=BIND_THREAD),
        BoundAxis(axis=n_i, bind=BIND_THREAD),
        BoundAxis(axis=m_o, bind=BIND_BLOCK),
        BoundAxis(axis=n_o, bind=BIND_BLOCK),
    )
    return Tile(axes=new_axes, body=new_body)


def _match_matmul_body(body: tuple, m_name: str, n_name: str) -> tuple[Loop, Write] | None:
    """The body must be exactly ``(reduce_loop, write)`` where ``write``
    indexes both output axes ``m`` and ``n`` (other index positions may
    be Literals from collapsed leading dims, e.g. batch=1 in the Linear
    forward shape ``(1, 32, K)``)."""
    if len(body) != 2:
        return None
    reduce_loop, write = body
    if not (isinstance(reduce_loop, Loop) and isinstance(write, Write)):
        return None
    if not reduce_loop.is_reduce:
        return None
    var_names = {e.name for e in write.index if isinstance(e, Var)}
    if {m_name, n_name} - var_names:
        return None
    return reduce_loop, write


def _extract_inner(reduce_loop: Loop) -> tuple[Accum, Load, Load, Assign] | None:
    """Inner body must be exactly ``(Load, Load, Assign-mul, Accum-add)``
    with two distinct source buffers and the Assign feeding the Accum."""
    inner = reduce_loop.body
    if len(inner) != 4:
        return None
    load_a, load_b, mul, accum = inner
    if not (isinstance(load_a, Load) and isinstance(load_b, Load)):
        return None
    if load_a.input == load_b.input:
        return None
    if not isinstance(mul, Assign) or mul.op.name != "multiply":
        return None
    if set(mul.args) != {load_a.name, load_b.name}:
        return None
    if not isinstance(accum, Accum) or accum.op.name != "add":
        return None
    if accum.value != mul.name:
        return None
    return accum, load_a, load_b, mul


def _id(name: str) -> str:
    return name
