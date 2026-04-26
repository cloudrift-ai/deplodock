"""Lower each ``LoopOp`` node to a ``TileOp``.

Mechanical translation: outer free-Loop chain becomes
``Tile(thread_axes=...)``, leaves and inner Loops pass through unchanged.
Strategy passes (``002_cooperative_reduce``, ``003_block_matmul``) may
later convert serial Loops to ``StridedLoop`` for cooperative iteration.

**Outer free-Loop chain → ``Tile.thread_axes``**. After stripping
leading non-Loop stmts (scalar Loads) into the TileOp body prefix,
lowering walks the outer free-Loop chain and strips it into a
``Tile(thread_axes=...)`` (default: one thread per output point — the
correct shape for pointwise kernels and for reductions the cooperative
strategy chooses not to rewrite). The chain ends at: a level with
multiple sibling stmts, a Loop with ``Accum`` in its immediate body
(reduce — can't strip), or no Loop at all.

The node's id, inputs, and output tensor are preserved — only the op
changes.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Loop
from deplodock.compiler.ir.stmt import Stmt as LoopStmt
from deplodock.compiler.ir.tile.ir import Stmt, Tile, TileOp
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", LoopOp)]


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, LoopOp):
        return None
    nid = match.root_node_id
    desired = node.output.name or nid
    if desired != nid and desired not in graph.nodes:
        graph.rename_node(nid, desired)
        nid = desired
    kname = _kernel_name_for(node.op, nid)
    graph.nodes[nid].op = lower_naive(graph.nodes[nid].op, kname)
    return None


def lower_naive(loop_op: LoopOp, kernel_name: str = "") -> TileOp:
    """Translate a ``LoopOp`` into a ``TileOp`` holding a logical ``Tile``.

    Steps:

    1. Pull leading non-Loop stmts (typically scalar Loads) off the LoopOp
       body — they sit at the start of ``TileOp.body``, above any Tile.
    2. Descend the outer free-Loop chain, collecting axes until the chain
       breaks (multi-stmt level, reduce Loop, or no more Loops).
    3. If any axes were collected, wrap the remaining inner body in a
       ``Tile(thread_axes=..., bind=BIND_THREAD)``. Otherwise, lower the
       inner body in place (single-thread serial — degenerate).

    Inner ``Loop``s pass through unchanged — strategy passes may
    convert select ones to ``StridedLoop`` for cooperative iteration.
    """
    leading: list[LoopStmt] = []
    rest: tuple[LoopStmt, ...] = loop_op.body
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]

    output_axes, inner = _strip_outer_free_chain(rest)

    body: list[Stmt] = list(leading)
    if output_axes:
        bound = tuple(BoundAxis(axis=ax, bind=BIND_THREAD) for ax in output_axes)
        body.append(Tile(axes=bound, body=tuple(inner)))
    else:
        body.extend(inner)

    return TileOp(body=tuple(body), name=kernel_name)


def _strip_outer_free_chain(stmts: tuple[LoopStmt, ...]) -> tuple[tuple[Axis, ...], tuple[LoopStmt, ...]]:
    """Walk the outer free-Loop chain and return ``(stripped_axes, remainder)``.

    Stops when the current level has more than one stmt, isn't a Loop, or
    is a Loop whose body contains an ``Accum`` at the immediate level (a
    reduce Loop — stripping it would lose the accumulator)."""
    axes: list[Axis] = []
    cur = stmts
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        axes.append(cur[0].axis)
        cur = cur[0].body
    return tuple(axes), cur


def _kernel_name_for(loop: LoopOp, base_name: str) -> str:
    suffix = "reduce" if any(isinstance(s, Accum) for s in loop) else "pointwise"
    return f"k_{_dedup_tokens(base_name)}_{suffix}"


def _dedup_tokens(name: str) -> str:
    """Drop consecutive duplicate ``_``-separated tokens.

    ``softmax_softmax_max`` → ``softmax_max``; ``rms_rms_norm`` → ``rms_norm``.
    Preserves order; only collapses adjacent duplicates so structurally
    distinct repeats (``add_mul_add``) survive.
    """
    out: list[str] = []
    for tok in name.split("_"):
        if not tok or (out and out[-1] == tok):
            continue
        out.append(tok)
    return "_".join(out) if out else name
