"""Block-tiled matmul strategy — splits M/N/K into outer/inner tiles
and reshapes the ``Tile`` so each CUDA block owns a BM·BN output region
walked cooperatively across threads, with the K reduction chunked into
BK-sized slices.

Mirrors the cooperative-reduce pattern from ``002_cooperative_reduce.py``:
this rule is the *structural* step (blockify); operand staging is left to
``003_stage_inputs.py`` (which sees A/B as reused operands once the
inner BoundLoops over m_i, n_i, k_i exist).

**Not yet auto-loaded** — file is prefixed with ``_`` so the engine's
``_load_rules`` glob skips it. Imported directly by unit tests via
``run_rule(graph, _block_matmul.py)``.

Activation status:

- ✅ Multi-axis ``Stage`` emission (flat-decode StridedLoop) — done.
- ✅ Matmul-style ``_materialize_cooperative`` branch (output axes
  promote to ``BIND_THREAD``, body BoundLoops stripped, no
  ``Cond on tid==0`` Write guard) — done.
- ✅ Nested-reduce Accum init via ``Init`` Stmt — done.
- ❌ This rule does not emit ``Stage`` itself. Without staging the
  blockified kernel reads A/B from global per K iteration — slower than
  the unblocked version. Either this rule needs to emit Stages, or the
  ``003_stage_inputs`` rule needs to be generalized to detect operand
  reuse across the matmul body's cooperative thread tile.
- ❌ ``_redirect_loads`` matches positions by Var name only; with
  matmul's affine index positions (``m_o*BM + m_i``) the cache-position
  matcher misses, so redirected smem reads use wrong indices.
- ❌ Default tile sizes ``BM=BN=64`` exceed ``BLOCK_SIZE=256`` per-block
  thread count, requiring per-thread output sub-tiles. Activation
  initially with ``BM=BN=16`` (one output per thread) is the simplest
  path.

Pre-rewrite (post ``lower_naive`` of fused matmul ``C = A @ B``)::

    Tile(thread_axes=(m, n)):
      BoundLoop(k, SERIAL):
        a = Load("A", (m, k))
        b = Load("B", (k, n))
        t = Assign(a * b)
        Accum(acc, op=add, value=t)
      Write("C", (m, n), acc)

Post-rewrite::

    Tile(axes=(m_o BLOCK, n_o BLOCK, m_i BLOCK_STRIDED, n_i BLOCK_STRIDED)):
      BoundLoop(m_i, BLOCK_STRIDED):
        BoundLoop(n_i, BLOCK_STRIDED):
          BoundLoop(k_o, SERIAL):
            BoundLoop(k_i, SERIAL):
              a = Load("A", (m_o*BM + m_i, k_o*BK + k_i))
              b = Load("B", (k_o*BK + k_i, n_o*BN + n_i))
              t = Assign(a * b)
              Accum(acc, op=add, value=t)
          Write("C", (m_o*BM + m_i, n_o*BN + n_i), acc)

No ``Combine`` — every output element is owned by exactly one thread.

Trigger conditions:

- ``TileOp.body`` contains exactly one ``Tile``.
- ``Tile.thread_axes`` is exactly 2D, ``block_axes`` is empty (idempotence).
- ``Tile.body`` is the canonical fused-matmul shape: one reduce
  ``BoundLoop`` (Load · Load · Assign-mul · Accum-add, with two distinct
  source bufs) followed by a ``Write`` over the two output axes.
- M, N, K extents are each divisible by the configured tile size.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import (
    BIND_BLOCK,
    BIND_BLOCK_STRIDED,
    BIND_SERIAL,
    BoundAxis,
    split_axis,
)
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Stmt, Write
from deplodock.compiler.ir.tile.ir import BoundLoop, Tile, TileOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

# Default tile sizes. BM·BN = output tile; BK = K-reduction chunk.
BM = 64
BN = 64
BK = 16


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
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
    k = reduce_loop.axis.axis
    if k.extent % BK:
        return None

    extracted = _extract_inner(reduce_loop)
    if extracted is None:
        return None
    accum, load_a, load_b, mul = extracted

    m_o, m_i = split_axis(m, BM)
    n_o, n_i = split_axis(n, BN)
    k_o, k_i = split_axis(k, BK)

    sigma = Sigma(
        {
            m.name: Var(m_o.name) * Literal(BM, "int") + Var(m_i.name),
            n.name: Var(n_o.name) * Literal(BN, "int") + Var(n_i.name),
            k.name: Var(k_o.name) * Literal(BK, "int") + Var(k_i.name),
        }
    )

    inner_compute: tuple[Stmt, ...] = (
        load_a.rewrite(_id, sigma),
        load_b.rewrite(_id, sigma),
        mul,
        accum,
    )

    new_body: tuple[Stmt, ...] = (
        BoundLoop(
            axis=BoundAxis(axis=m_i, bind=BIND_BLOCK_STRIDED),
            body=(
                BoundLoop(
                    axis=BoundAxis(axis=n_i, bind=BIND_BLOCK_STRIDED),
                    body=(
                        BoundLoop(
                            axis=BoundAxis(axis=k_o, bind=BIND_SERIAL),
                            body=(
                                BoundLoop(
                                    axis=BoundAxis(axis=k_i, bind=BIND_SERIAL),
                                    body=inner_compute,
                                ),
                            ),
                        ),
                        write.rewrite(_id, sigma),
                    ),
                ),
            ),
        ),
    )

    new_axes = (
        BoundAxis(axis=m_o, bind=BIND_BLOCK),
        BoundAxis(axis=n_o, bind=BIND_BLOCK),
        BoundAxis(axis=m_i, bind=BIND_BLOCK_STRIDED),
        BoundAxis(axis=n_i, bind=BIND_BLOCK_STRIDED),
    )
    return Tile(axes=new_axes, body=new_body)


def _match_matmul_body(body: tuple, m_name: str, n_name: str) -> tuple[BoundLoop, Write] | None:
    """The body must be exactly ``(reduce_loop, write)`` where ``write``
    indexes the two output axes by name."""
    if len(body) != 2:
        return None
    reduce_loop, write = body
    if not (isinstance(reduce_loop, BoundLoop) and isinstance(write, Write)):
        return None
    if not _is_reduce(reduce_loop):
        return None
    if reduce_loop.bind != BIND_SERIAL:
        return None
    if len(write.index) != 2:
        return None
    out_names = tuple(e.name if isinstance(e, Var) else None for e in write.index)
    if set(out_names) != {m_name, n_name}:
        return None
    return reduce_loop, write


def _extract_inner(reduce_loop: BoundLoop) -> tuple[Accum, Load, Load, Assign] | None:
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


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


def _id(name: str) -> str:
    return name
