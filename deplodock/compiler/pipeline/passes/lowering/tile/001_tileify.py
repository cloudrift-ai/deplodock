"""Form a ``TileOp`` from each ``LoopOp`` — "tileification".

Mechanical translation: outer free-Loop chain becomes
``Tile(thread_axes=...)``, leaves and inner Loops pass through unchanged.
Strategy passes (``002_chunk_matmul_k``, ``004_cooperative_reduce``) may
later convert serial Loops to ``StridedLoop`` for cooperative iteration.

**Outer free-Loop chain → ``Tile.thread_axes``**. After stripping
leading non-Loop stmts (scalar Loads) into the TileOp body prefix,
lowering walks the outer free-Loop chain and strips it into a
``Tile(thread_axes=...)`` (default: one thread per output point — the
correct shape for pointwise kernels and for reductions the cooperative
strategy chooses not to rewrite). The chain ends at: a level with
multiple sibling stmts, a Loop with ``Accum`` in its immediate body
(reduce — can't strip), or no Loop at all.

**Body free Loops over output dims → ``Tile.thread_axes``**. After the
outer chain is stripped, top-level body stmts may still contain free
Loops whose iteration writes distinct output positions (e.g. fused
SDPA's head-dim loop sits as a sibling to two softmax reduces). Each
such Loop is lifted into ``Tile.axes`` (THREAD) and replaced by its
body, so the launch can spawn one thread per iteration instead of
serializing the writes. Detection: top-level body stmts only; the
loop's subtree must contain a ``Write`` whose index expression
references the loop's axis.

The node's id, inputs, and output tensor are preserved — only the op
changes.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, StridedLoop, Write
from deplodock.compiler.ir.stmt import Stmt as LoopStmt
from deplodock.compiler.ir.tile.ir import Stmt, Tile, TileOp
from deplodock.compiler.pipeline import Pattern

PATTERN = [Pattern("root", LoopOp)]


def rewrite(root: Node) -> Graph | None:
    kname = _kernel_name_for(root.op, root.id)
    return tileify(root.op, kname)


def tileify(loop_op: LoopOp, kernel_name: str = "") -> TileOp:
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
        body.append(_lift_output_loops(Tile(axes=bound, body=inner)))
    else:
        body.extend(inner)

    return TileOp(body=body, name=kernel_name)


def _lift_output_loops(tile: Tile) -> Tile:
    """Lift top-level free Loops that wrap a Write whose index varies
    with the loop's axis into ``Tile.axes`` (THREAD). Only top-level
    body stmts are considered — nested promotions would mix loop
    ordering decisions with reduction structure.

    SDPA's head-dim free loop sits at top level (after the two softmax
    reduces) so this catches the case we care about; without lifting,
    every thread re-runs the inner reduce per head-dim element.
    """
    new_axes = list(tile.axes)
    new_body: list[Stmt] = []
    changed = False
    for s in tile.body:
        if isinstance(s, Loop) and not s.is_reduce and _writes_with_axis(s.body, s.axis.name):
            new_axes.append(BoundAxis(axis=s.axis, bind=BIND_THREAD))
            new_body.extend(s.body)
            changed = True
        else:
            new_body.append(s)
    if not changed:
        return tile
    return Tile(axes=tuple(new_axes), body=new_body)


def _writes_with_axis(stmts: tuple, axis_name: str) -> bool:
    for s in stmts:
        if isinstance(s, Write):
            free: set[str] = set()
            for e in s.index:
                free |= e.free_vars()
            if axis_name in free:
                return True
        if isinstance(s, (Loop, StridedLoop)) and _writes_with_axis(s.body, axis_name):
            return True
        if isinstance(s, Cond):
            if _writes_with_axis(s.body, axis_name) or _writes_with_axis(s.else_body, axis_name):
                return True
    return False


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
