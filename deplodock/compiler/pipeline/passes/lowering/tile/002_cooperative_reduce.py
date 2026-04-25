"""Cooperative-block multi-phase reduction strategy.

Rewrites a 1D-reduction ``TileOp`` so each CUDA block owns one output
slot and threads in the block cooperate on the reduction axis via a
``__shared__`` tree-halve. Generalizes over any sequence of "phases"
under one ``Tile``: each reduce-``Loop`` becomes a cooperative phase
(per-thread strided partial → smem store → tree-halve → broadcast
load); each free ``Loop`` becomes a strided write loop that lets all
threads in the block share the row-output work; any leaf stmts between
phases (e.g. RMSNorm's ``acc / N + eps; rsqrt(...)``) are carried
through with SSA rename so prior Accum targets resolve to the
broadcast load.

Pre-rewrite shape (any number of sibling Loops + leaves under the Tile)::

    Enclosure(thread_axes=(i,), block_axes=(),
      Tile(live_axes=(i,), extents=(M,),
        Loop(k1) { Load; Accum("acc_max", op=max) }       # reduction
        Loop(k2) { Load; Assign(v0=in-acc_max);
                   Assign(v1=exp(v0)); Accum("acc_sum") } # uses prior result
        Loop(k3) { Load; Assign(...); Write(out, ...) }   # free output
      ))

Post-rewrite (per-reduction smem buf + halve; free loops strided
across threads)::

    Enclosure(thread_axes=(t,), block_axes=(i,),
      Tile(live_axes=(i,), extents=(M,),
        Smem("acc_max_smem", (BLOCK,))
        StridedLoop(k1, start=t, step=BLOCK) { Load; Accum }
        Write(acc_max_smem, (t,), acc_max)
        Sync; TreeHalve(acc_max_smem, max, BLOCK, t); Sync
        Load(acc_max_b, "acc_max_smem", (0,))             # broadcast scalar

        Smem("acc_sum_smem", (BLOCK,))
        StridedLoop(k2, start=t, step=BLOCK) {
          Load; Assign(v0 = in - acc_max_b);              # ← rename
          Assign(v1=exp(v0)); Accum }
        Write(acc_sum_smem, (t,), acc_sum)
        Sync; TreeHalve(...); Sync
        Load(acc_sum_b, "acc_sum_smem", (0,))

        StridedLoop(k3, start=t, step=BLOCK) {
          Load; Assign(... acc_max_b ...);                # ← renames
          Assign(... acc_sum_b ...); Write(out, ...) }
      ))

Trigger conditions:

- TileOp body has exactly one ``Enclosure`` with one ``thread_axis`` and
  one inner ``Tile`` with one ``live_axis`` (``thread_axes == live_axes``).
- ``Tile.body`` contains at least one reduce ``Loop`` whose immediate
  body has exactly one ``Accum`` (multi-Accum-per-Loop / online
  algorithms — separate strategy).
- The first reduce Loop's axis extent ≥ ``COOP_THRESHOLD``.

Final ``Write`` handling:

- Inside a free ``Loop`` (softmax, RMSNorm) → strided write across
  threads, no Cond gating (each thread handles its own slab of the
  output axis).
- Bare at Tile top level (``y[i] = sum_k x[i,k]``) → guarded by
  ``Cond(t == 0, Write)``.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Cond, Load, Loop, Write
from deplodock.compiler.ir.tile.ir import (
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

BLOCK = 256
COOP_THRESHOLD = 128


def rewrite(graph: Graph, match: Match) -> Graph | None:
    node = graph.nodes[match.root_node_id]
    if not isinstance(node.op, TileOp):
        return None
    tile_op: TileOp = node.op

    new_body = _maybe_rewrite_body(tile_op.body)
    if new_body is None:
        return None
    node.op = TileOp(body=new_body, name=tile_op.name)
    return None


def _maybe_rewrite_body(body: tuple) -> tuple | None:
    enclosures = [(idx, s) for idx, s in enumerate(body) if isinstance(s, Enclosure)]
    if len(enclosures) != 1:
        return None
    idx, encl = enclosures[0]
    if encl.block_axes:
        return None  # already rewritten

    rewritten = _rewrite_enclosure(encl)
    if rewritten is None:
        return None
    return body[:idx] + (rewritten,) + body[idx + 1 :]


def _rewrite_enclosure(encl: Enclosure) -> Enclosure | None:
    if len(encl.thread_axes) != 1:
        return None
    if len(encl.body) != 1 or not isinstance(encl.body[0], Tile):
        return None
    tile: Tile = encl.body[0]
    if len(tile.live_axes) != 1 or tile.live_axes != encl.thread_axes:
        return None

    reduce_loops = [s for s in tile.body if isinstance(s, Loop) and _is_reduce_loop(s)]
    if not reduce_loops:
        return None
    if int(reduce_loops[0].axis.extent) < COOP_THRESHOLD:
        return None
    for rl in reduce_loops:
        if sum(1 for s in rl.body if isinstance(s, Accum)) != 1:
            return None  # multi-Accum reductions punted

    t_axis = Axis(name="t", extent=BLOCK)
    new_body = _build_cooperative_body(tile.body, t_axis.name)
    if new_body is None:
        return None

    new_tile = Tile(live_axes=tile.live_axes, extents=tile.extents, body=new_body)
    return Enclosure(thread_axes=(t_axis,), block_axes=encl.thread_axes, body=(new_tile,))


def _is_reduce_loop(loop: Loop) -> bool:
    return any(isinstance(s, Accum) for s in loop.body)


def _build_cooperative_body(stmts: tuple, t: str) -> tuple | None:
    """Walk ``Tile.body`` left-to-right, classifying each stmt:

    - reduce ``Loop`` → cooperative phase (smem + halve + broadcast Load).
    - free ``Loop`` → ``StridedLoop`` (all threads share the output work).
    - bare ``Write`` at Tile top level → ``Cond(t == 0, Write)``.
    - other leaf stmts → carried through unchanged.

    Maintains a ``rename`` map of prior Accum SSA names to their
    broadcast-load names; subsequent stmts have their reads remapped via
    ``_rename_stmt``.
    """
    out: list = []
    rename: dict[str, str] = {}

    for stmt in stmts:
        if isinstance(stmt, Loop) and _is_reduce_loop(stmt):
            phase = _emit_reduction_phase(stmt, t, rename)
            if phase is None:
                return None
            phase_stmts, accum = phase
            out.extend(phase_stmts)
            rename[accum.name] = f"{accum.name}_b"
        elif isinstance(stmt, Loop):
            renamed_body = tuple(_rename_stmt(s, rename) for s in stmt.body)
            out.append(StridedLoop(axis=stmt.axis, start=Var(t), step=BLOCK, body=renamed_body))
        elif isinstance(stmt, Write):
            renamed = _rename_stmt(stmt, rename)
            out.append(
                Cond(
                    cond=BinaryExpr("==", Var(t), Literal(0, "int")),
                    body=(renamed,),
                    else_body=(),
                )
            )
        else:
            out.append(_rename_stmt(stmt, rename))
    return tuple(out)


def _emit_reduction_phase(loop: Loop, t: str, rename: dict[str, str]) -> tuple[list, Accum] | None:
    """One cooperative reduction phase: smem alloc → strided per-thread
    partial → store partial → sync → tree-halve → sync → broadcast load."""
    accums = [s for s in loop.body if isinstance(s, Accum)]
    if len(accums) != 1:
        return None
    accum = accums[0]
    smem_name = f"{accum.name}_smem"
    broadcast_name = f"{accum.name}_b"
    body_renamed = tuple(_rename_stmt(s, rename) for s in loop.body)
    return (
        [
            Smem(name=smem_name, extents=(BLOCK,)),
            StridedLoop(axis=loop.axis, start=Var(t), step=BLOCK, body=body_renamed),
            Write(output=smem_name, index=(Var(t),), value=accum.name),
            Sync(),
            TreeHalve(buf=smem_name, op=accum.op, length=BLOCK, tid_var=t),
            Sync(),
            Load(name=broadcast_name, input=smem_name, index=(Literal(0, "int"),)),
        ],
        accum,
    )


# ---------------------------------------------------------------------------
# SSA rename — substitute prior Accum reads with their broadcast names
# ---------------------------------------------------------------------------


def _rename_name(name: str, rename: dict[str, str]) -> str:
    return rename.get(name, name)


def _rename_stmt(s: object, rename: dict[str, str]) -> object:
    if not rename:
        return s
    if isinstance(s, Loop):
        return Loop(axis=s.axis, body=tuple(_rename_stmt(c, rename) for c in s.body))
    if isinstance(s, StridedLoop):
        return StridedLoop(
            axis=s.axis,
            start=s.start,
            step=s.step,
            body=tuple(_rename_stmt(c, rename) for c in s.body),
        )
    if isinstance(s, Cond):
        return replace(
            s,
            body=tuple(_rename_stmt(c, rename) for c in s.body),
            else_body=tuple(_rename_stmt(c, rename) for c in s.else_body),
        )
    if isinstance(s, Assign):
        return Assign(name=s.name, op=s.op, args=tuple(_rename_name(a, rename) for a in s.args))
    if isinstance(s, Accum):
        return Accum(name=s.name, value=_rename_name(s.value, rename), op=s.op)
    if isinstance(s, Write):
        return Write(output=s.output, index=s.index, value=_rename_name(s.value, rename))
    return s
