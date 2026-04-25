"""Materialize a ``Block`` into the lowered ``Enclosure`` + ``Tile`` form.

Reads the high-level ``Block`` (with or without strategy-placed
``Combine`` siblings) and emits the concrete kernel shape that
``lowering/cuda`` + ``render_tileop`` consume::

    Enclosure(thread_axes=…, block_axes=…,
      [Tile(live_axes=…, extents=…,
        Smem(...)*,
        StridedLoop | Loop | Accum | Load | Assign | Write,
        Sync, TreeHalve, Cond(t == 0, Write), …
      )] or body directly for pointwise)

Dispatch is driven by two fields:

- ``Block.output_bind`` — ``BIND_THREAD`` (pointwise or
  one-thread-per-output-slot reduction) vs ``BIND_BLOCK`` (cooperative,
  one CUDA block per output slot).
- ``BoundLoop.bind`` — ``BIND_SERIAL`` (emit ``loop.Loop``, per-thread
  walk) vs ``BIND_STRIDED`` (emit ``StridedLoop`` with ``start=Var(t)``,
  ``step=BLOCK``).

Combine siblings (present only in ``BIND_BLOCK`` mode) emit the smem
tree-halve phase and set up broadcast ``Load``s. Subsequent reads of
the combined Accum target are renamed to ``<name>_b`` via
``Stmt.rewrite``.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import Accum, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    BIND_BLOCK,
    BIND_SERIAL,
    BIND_STRIDED,
    BIND_THREAD,
    COMBINE_REGISTER,
    COMBINE_SMEM_TREE_HALVE,
    Axis,
    Block,
    BoundLoop,
    Combine,
    Enclosure,
    Smem,
    StridedLoop,
    Sync,
    Tile,
    TileOp,
    TreeHalve,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

BLOCK_SIZE = 256


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    if not any(isinstance(s, Block) for s in tile_op.body):
        return None  # already materialized or nothing to do

    new_body: list[Stmt] = []
    changed = False
    for s in tile_op.body:
        if isinstance(s, Block):
            new_body.append(_materialize(s))
            changed = True
        else:
            new_body.append(s)

    if not changed:
        return None
    node.op = TileOp(body=tuple(new_body), name=tile_op.name)
    return None


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: Block) -> Stmt:
    if blk.output_bind == BIND_THREAD:
        return _materialize_thread_per_output(blk.output_axes, blk.body)
    if blk.output_bind == BIND_BLOCK:
        return _materialize_block_per_output(blk.output_axes, blk.body)
    raise NotImplementedError(f"Block output_bind={blk.output_bind!r} not yet handled")


def _materialize_thread_per_output(output_axes: tuple, body: tuple) -> Stmt:
    """``BIND_THREAD`` — one thread per output element. Pointwise kernels
    have no inner Loops / Accums; reductions that opted out of cooperation
    stay here with serial per-thread ``Loop`` walks folding ``Accum``
    into per-thread registers (Tile wrapper preserves the register scope)."""
    lowered = tuple(_lower_uncooperative(s) for s in body)
    if _has_accum(body):
        extents = tuple(int(ax.extent) for ax in output_axes)
        wrapped: tuple = (Tile(live_axes=output_axes, extents=extents, body=lowered),)
    else:
        wrapped = lowered
    return Enclosure(thread_axes=output_axes, block_axes=(), body=wrapped)


def _lower_uncooperative(s: Stmt) -> Stmt:
    """Translate a ``BoundLoop(bind=SERIAL)`` tree back to Loop-IR ``Loop``.
    Leaves pass through. ``Combine`` must not appear here (no strategy
    places it in BIND_THREAD mode)."""
    if isinstance(s, BoundLoop):
        if s.bind != BIND_SERIAL:
            raise ValueError(f"BIND_THREAD Block cannot contain BoundLoop with bind={s.bind!r}")
        return Loop(axis=s.axis, body=tuple(_lower_uncooperative(c) for c in s.body))
    if isinstance(s, Combine):
        raise ValueError("Combine not allowed in BIND_THREAD Block")
    return s


def _materialize_block_per_output(output_axes: tuple, body: tuple) -> Stmt:
    """``BIND_BLOCK`` — one CUDA block per output slot, threads cooperate.

    Walks ``body`` left-to-right. Each ``BoundLoop(bind=STRIDED)`` becomes
    a ``StridedLoop`` reading/writing via a cooperative thread axis ``t``.
    A ``Combine`` triggers the per-Accum smem + tree-halve phase and
    records an SSA rename so subsequent reads of the Accum target
    resolve to the broadcast ``Load``.
    """
    t_axis = Axis(name="t", extent=BLOCK_SIZE)
    rename: dict[str, str] = {}

    def renamed(s: Stmt) -> Stmt:
        if not rename:
            return s
        return s.rewrite(lambda n: rename.get(n, n))

    new_body: list[Stmt] = []
    # Track the most-recently-emitted reduce BoundLoop + its Accum so the
    # following Combine knows which partial to collapse.
    pending_reduce: tuple[BoundLoop, Accum] | None = None

    for stmt in body:
        if isinstance(stmt, BoundLoop):
            pending_reduce = None  # a BoundLoop resets the pending slot
            if _is_reduce(stmt):
                accum = next(a for a in stmt.body if isinstance(a, Accum))
                new_body.append(_emit_strided(stmt, t_axis.name, renamed))
                pending_reduce = (stmt, accum)
            else:
                new_body.append(_emit_strided(stmt, t_axis.name, renamed))
        elif isinstance(stmt, Combine):
            if pending_reduce is None:
                raise ValueError(f"Combine({stmt.name!r}) without a preceding reduce BoundLoop")
            _, accum = pending_reduce
            if accum.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({accum.name!r})")
            phase = _emit_combine(stmt, accum, t_axis.name)
            new_body.extend(phase)
            if stmt.via == COMBINE_SMEM_TREE_HALVE:
                rename[accum.name] = f"{accum.name}_b"
            pending_reduce = None
        elif isinstance(stmt, Write):
            new_body.append(
                Cond(
                    cond=BinaryExpr("==", Var(t_axis.name), Literal(0, "int")),
                    body=(renamed(stmt),),
                    else_body=(),
                )
            )
        else:
            new_body.append(renamed(stmt))

    extents = tuple(int(ax.extent) for ax in output_axes)
    tile = Tile(live_axes=output_axes, extents=extents, body=tuple(new_body))
    return Enclosure(thread_axes=(t_axis,), block_axes=output_axes, body=(tile,))


def _emit_strided(loop: BoundLoop, t: str, renamed) -> Stmt:
    """Convert a ``BoundLoop`` to its post-materialization form per binding."""
    body = tuple(_lower_inner(c, renamed) for c in loop.body)
    if loop.bind == BIND_STRIDED:
        return StridedLoop(axis=loop.axis, start=Var(t), step=BLOCK_SIZE, body=body)
    if loop.bind == BIND_SERIAL:
        return Loop(axis=loop.axis, body=body)
    raise NotImplementedError(f"BoundLoop bind={loop.bind!r} inside BIND_BLOCK Block not yet handled")


def _lower_inner(s: Stmt, renamed) -> Stmt:
    """Inside a BoundLoop's body: nested BoundLoops recurse; leaves are
    SSA-renamed through the accumulated ``rename`` map."""
    if isinstance(s, BoundLoop):
        # Nested BoundLoops (not used by 1D cooperative today but safe).
        return Loop(axis=s.axis, body=tuple(_lower_inner(c, renamed) for c in s.body))
    return renamed(s)


def _emit_combine(combine: Combine, accum: Accum, t: str) -> list[Stmt]:
    """Emit the per-phase combine. ``REGISTER`` is a no-op (the Accum
    already lives in the per-thread register). ``SMEM_TREE_HALVE`` stages
    the partial through smem, halves, re-syncs, and broadcasts the
    combined value to every thread via a ``Load`` at index 0."""
    if combine.via == COMBINE_REGISTER:
        return []
    if combine.via == COMBINE_SMEM_TREE_HALVE:
        smem_name = f"{accum.name}_smem"
        broadcast_name = f"{accum.name}_b"
        return [
            Smem(name=smem_name, extents=(BLOCK_SIZE,)),
            Write(output=smem_name, index=(Var(t),), value=accum.name),
            Sync(),
            TreeHalve(buf=smem_name, op=accum.op, length=BLOCK_SIZE, tid_var=t),
            Sync(),
            Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
        ]
    raise NotImplementedError(f"Combine via={combine.via!r} not yet handled")


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


def _has_accum(stmts: tuple) -> bool:
    for s in stmts:
        if isinstance(s, Accum):
            return True
        if isinstance(s, BoundLoop) and _has_accum(s.body):
            return True
    return False
