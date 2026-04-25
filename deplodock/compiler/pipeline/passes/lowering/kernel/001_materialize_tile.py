"""Materialize a Tile-IR ``TileOp`` into a Kernel-IR ``KernelOp``.

Reads each ``Tile`` in the TileOp body and emits the concrete
hardware shape — ``Enclosure`` / ``Smem`` / ``Sync`` / ``TreeHalve`` /
``StridedLoop`` — that ``render_kernelop`` consumes.

The Tile→Enclosure mapping is structural: both nodes carry
``axes: tuple[BoundAxis, ...]``. Materialization is then:

- All ``BoundAxis`` in the Tile are ``BIND_THREAD`` (pointwise / per-
  thread serial) → ``Enclosure(axes=blk.axes)``. Inner ``BoundLoop``s
  fall back to serial Loop-IR ``Loop``s.
- Any ``BoundAxis`` is ``BIND_BLOCK`` (cooperative) →
  ``Enclosure(axes=(BoundAxis(t, BIND_THREAD), *blk.axes))``, where
  ``t`` is the synthetic cooperative thread axis introduced here.
  Inner ``BoundLoop`` with ``bind=BIND_BLOCK_STRIDED`` becomes ``StridedLoop`` driven
  by ``t``; ``Combine`` siblings emit the smem tree-halve phase and
  broadcast loads; ``Stmt.rewrite`` renames subsequent Accum reads to
  ``<name>_b``.

Produces a ``KernelOp`` — distinct type from ``TileOp``, so Kernel-IR
passes can pattern-match on it.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import BIND_BLOCK_STRIDED, BIND_SERIAL, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    Enclosure,
    KernelOp,
    Smem,
    StridedLoop,
    Sync,
    TreeHalve,
)
from deplodock.compiler.ir.stmt import Accum, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import (
    BoundLoop,
    Combine,
    Tile,
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
        if isinstance(s, Tile):
            new_body.append(_materialize(s))
        else:
            new_body.append(s)

    node.op = KernelOp(body=tuple(new_body), name=tile_op.name)
    return None


# ---------------------------------------------------------------------------
# Materialization
# ---------------------------------------------------------------------------


def _materialize(blk: Tile) -> Stmt:
    if blk.block_axes:
        return _materialize_cooperative(blk.axes, blk.body)
    return _materialize_thread_per_output(blk.axes, blk.body)


def _materialize_thread_per_output(axes: tuple, body: tuple) -> Stmt:
    """One thread per output element. ``axes`` is passed through
    unchanged — every BoundAxis is already ``BIND_THREAD``."""
    lowered = tuple(_lower_uncooperative(s) for s in body)
    return Enclosure(axes=axes, body=lowered)


def _lower_uncooperative(s: Stmt) -> Stmt:
    """Translate a ``BoundLoop(bind=SERIAL)`` tree to Loop-IR ``Loop``.
    Leaves pass through. ``Combine`` must not appear in a non-cooperative
    Tile (no strategy places it without setting ``block_axes``)."""
    if isinstance(s, BoundLoop):
        if s.bind != BIND_SERIAL:
            raise ValueError(f"non-cooperative Tile cannot contain BoundLoop with bind={s.bind!r}")
        return Loop(axis=s.axis, body=tuple(_lower_uncooperative(c) for c in s.body))
    if isinstance(s, Combine):
        raise ValueError("Combine not allowed in non-cooperative Tile (block_axes must be populated)")
    return s


def _materialize_cooperative(axes: tuple, body: tuple) -> Stmt:
    """Cooperative materialization: one CUDA block per output point;
    a synthetic ``t`` thread axis drives cooperation. ``axes`` carries
    the output BoundAxes (all ``BIND_BLOCK``); the synthetic ``t`` is
    prepended as ``BIND_THREAD``."""
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
            reduce_loop, accum = pending_reduce
            if accum.name != stmt.name:
                raise ValueError(f"Combine({stmt.name!r}) does not match preceding Accum({accum.name!r})")
            # Combine scope is derived from the surrounding BoundLoop's bind:
            # BIND_BLOCK_STRIDED → smem tree-halve at block scope.
            phase = _emit_combine(stmt, accum, reduce_loop.bind, t_axis.name)
            new_body.extend(phase)
            if reduce_loop.bind == BIND_BLOCK_STRIDED:
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

    # Cooperative thread axis ``t`` (BIND_THREAD) plus the original output
    # axes — but BIND_BLOCK_STRIDED axes are filtered out because they
    # don't contribute to launch geometry (the body's strided BoundLoops
    # handle their iteration).
    launch_axes = tuple(ba for ba in axes if ba.bind != BIND_BLOCK_STRIDED)
    new_axes = (BoundAxis(axis=t_axis, bind=BIND_THREAD), *launch_axes)
    return Enclosure(axes=new_axes, body=tuple(new_body))


def _emit_strided(loop: BoundLoop, t: str, renamed) -> Stmt:
    body = tuple(_lower_inner(c, renamed) for c in loop.body)
    if loop.bind == BIND_BLOCK_STRIDED:
        return StridedLoop(axis=loop.axis, start=Var(t), step=BLOCK_SIZE, body=body)
    if loop.bind == BIND_SERIAL:
        return Loop(axis=loop.axis, body=body)
    raise NotImplementedError(f"BoundLoop bind={loop.bind!r} inside cooperative Tile not yet handled")


def _lower_inner(s: Stmt, renamed) -> Stmt:
    if isinstance(s, BoundLoop):
        return Loop(axis=s.axis, body=tuple(_lower_inner(c, renamed) for c in s.body))
    return renamed(s)


def _emit_combine(combine: Combine, accum: Accum, scope: str, t: str) -> list[Stmt]:
    """Emit the cross-thread combine. ``scope`` is the surrounding reduce
    BoundLoop's bind value, which drives the combine mechanism:

    - ``BIND_BLOCK_STRIDED`` → smem tree-halve at block scope.
    - ``BIND_SERIAL`` → no combine (each thread's partial is already
      the final value; legal but unused today since strategies don't
      emit Combine after a serial loop).
    - Future: ``BIND_WARP_STRIDED`` → warp-shuffle (no smem).
    """
    if scope == BIND_SERIAL:
        return []
    if scope == BIND_BLOCK_STRIDED:
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
    raise NotImplementedError(f"Combine for surrounding bind={scope!r} not yet handled")


def _is_reduce(loop: BoundLoop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)
