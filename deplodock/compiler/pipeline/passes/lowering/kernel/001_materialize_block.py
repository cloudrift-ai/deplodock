"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

Reads each ``Block`` in the TileOp body and emits the concrete
hardware shape — ``Enclosure`` / ``Smem`` / ``Sync`` / ``TreeHalve`` /
``StridedLoop`` — that ``render_kernelop`` consumes.

The Block→Enclosure mapping is structural: the two nodes share the
``thread_axes`` / ``block_axes`` field shape. Materialization is then:

- ``Block(thread_axes=T, block_axes=())`` (pointwise / per-thread serial)
  → ``Enclosure(thread_axes=T, block_axes=())``. Inner ``BoundLoop``s
  fall back to serial Loop-IR ``Loop``s.
- ``Block(thread_axes=(), block_axes=B)`` (cooperative)
  → ``Enclosure(thread_axes=(t,), block_axes=B)``, where ``t`` is the
  synthetic cooperative thread axis introduced here. Inner
  ``BoundLoop(bind=BIND_STRIDED)`` becomes ``StridedLoop`` driven by ``t``;
  ``Combine`` siblings emit the smem tree-halve phase and broadcast
  loads; ``Stmt.rewrite`` renames subsequent Accum reads to ``<name>_b``.

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    KernelOp,
    Smem,
    StridedLoop,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.loop import Accum, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    BIND_SERIAL,
    BIND_STRIDED,
    COMBINE_REGISTER,
    COMBINE_SMEM_TREE_HALVE,
    Axis,
    Block,
    BoundLoop,
    Combine,
    TileOp,
)
from deplodock.compiler.pipeline.engine import Match, Pattern

PATTERN = [Pattern("root", TileOp)]

BLOCK_SIZE = 256


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    new_body: list[Stmt] = []
    for s in tile_op.body:
        if isinstance(s, Block):
            new_body.append(_materialize(s))
        else:
            new_body.append(s)

    node.op = KernelOp(body=tuple(new_body), name=tile_op.name)
    return None


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: Block) -> Stmt:
    if blk.block_axes:
        return _materialize_cooperative(blk.thread_axes, blk.block_axes, blk.body)
    return _materialize_thread_per_output(blk.thread_axes, blk.body)


def _materialize_thread_per_output(output_axes: tuple, body: tuple) -> Stmt:
    """One thread per output element. Pointwise kernels
    have no inner Loops / Accums; reductions that opted out of cooperation
    stay here with serial per-thread ``Loop`` walks folding ``Accum``
    into per-thread registers."""
    lowered = tuple(_lower_uncooperative(s) for s in body)
    return Enclosure(thread_axes=output_axes, block_axes=(), body=lowered)


def _lower_uncooperative(s: Stmt) -> Stmt:
    """Translate a ``BoundLoop(bind=SERIAL)`` tree to Loop-IR ``Loop``.
    Leaves pass through. ``Combine`` must not appear in a non-cooperative
    Block (no strategy places it without setting ``block_axes``)."""
    if isinstance(s, BoundLoop):
        if s.bind != BIND_SERIAL:
            raise ValueError(f"non-cooperative Block cannot contain BoundLoop with bind={s.bind!r}")
        return Loop(axis=s.axis, body=tuple(_lower_uncooperative(c) for c in s.body))
    if isinstance(s, Combine):
        raise ValueError("Combine not allowed in non-cooperative Block (block_axes must be populated)")
    return s


def _materialize_cooperative(thread_axes: tuple, block_axes: tuple, body: tuple) -> Stmt:
    """Cooperative materialization: one CUDA block per output point in
    ``block_axes``; a synthetic ``t`` thread axis drives cooperation."""
    t_axis = Axis(name="t", extent=BLOCK_SIZE)
    rename: dict[str, str] = {}

    def renamed(s: Stmt) -> Stmt:
        if not rename:
            return s
        return s.rewrite(lambda n: rename.get(n, n))

    new_body: list[Stmt] = []
    pending_reduce: tuple[BoundLoop, Accum] | None = None

    for stmt in body:
        if isinstance(stmt, BoundLoop):
            pending_reduce = None
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

    # Cooperative thread axis ``t`` plus any pre-existing thread_axes the
    # strategy left in place (none today, but the field shape supports it).
    return Enclosure(thread_axes=(t_axis, *thread_axes), block_axes=block_axes, body=tuple(new_body))


def _emit_strided(loop: BoundLoop, t: str, renamed) -> Stmt:
    body = tuple(_lower_inner(c, renamed) for c in loop.body)
    if loop.bind == BIND_STRIDED:
        return StridedLoop(axis=loop.axis, start=Var(t), step=BLOCK_SIZE, body=body)
    if loop.bind == BIND_SERIAL:
        return Loop(axis=loop.axis, body=body)
    raise NotImplementedError(f"BoundLoop bind={loop.bind!r} inside cooperative Block not yet handled")


def _lower_inner(s: Stmt, renamed) -> Stmt:
    if isinstance(s, BoundLoop):
        return Loop(axis=s.axis, body=tuple(_lower_inner(c, renamed) for c in s.body))
    return renamed(s)


def _emit_combine(combine: Combine, accum: Accum, t: str) -> list[Stmt]:
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
