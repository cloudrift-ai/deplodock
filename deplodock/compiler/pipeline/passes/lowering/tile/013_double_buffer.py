"""Ping-pong double-buffering for K-outer staged matmul kernels.

For a Tile body shaped like ``Loop(K_outer, body=[Stage*, reduce])``,
this pass marks each Stage with ``buffer_count = 2`` and a phase
expression ``Var(K_outer_axis) % 2``. The materializer then doubles the
smem allocation and prepends the phase to both the cooperative-load
write and every body Load that reads from the staged buffer.

The win comes from removing the leading ``__syncthreads`` between
prev-compute and next-load: with ping-pong, consecutive iterations
write to *different* physical smem regions, so the next iteration's
load can issue before the previous iteration's compute fully drains.
The trailing Sync (after the cooperative load, before compute reads)
is preserved.

Trigger:

- Tile body has exactly one ``Tile`` (the standard shape).
- The Tile body has a free Loop ``K_outer`` (extent ≥ 2) whose body
  contains ≥ 1 ``Stage`` and exactly one reduce ``Loop``.
- The reduce body matches the pure-matmul shape (Load / Assign / Accum
  only, no Selects / Conds, no reads of accumulators) — same gate
  ``register_tile`` uses.
- Smem budget: ``2 × sum(slab_floats) ≤ _SMEM_BUDGET`` (default 96 KB
  to fit the per-block dynamic smem cap on Ada / Hopper consumer
  parts).

Idempotence: a Stage with ``buffer_count > 1`` is left alone.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Stmt, Tile, map_body
from deplodock.compiler.ir.tile.ir import Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_BUFFER_COUNT = 2
_SMEM_BUDGET_FLOATS = 24 * 1024  # 96 KB at fp32


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: tuple[Stmt, ...]) -> tuple[Stmt, ...] | None:
    tiles = [(i, s) for i, s in enumerate(body) if isinstance(s, Tile)]
    if len(tiles) != 1:
        raise RuleSkipped(f"need exactly one Tile in TileOp.body, found {len(tiles)}")
    idx, tile = tiles[0]

    new_tile_body = _process_scope(tile.body)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no K-outer matmul Loop eligible for double-buffering within smem budget")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process_scope(body: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Walk the body looking for K-outer Loops eligible for double-buffering."""
    new_body: list[Stmt] = list(body)
    changed = False
    for i, s in enumerate(body):
        if not isinstance(s, Loop) or s.is_reduce:
            continue
        if not _is_kouter_matmul(s):
            continue
        if any(st.buffer_count > 1 for st in s.body if isinstance(st, Stage)):
            continue
        stages = [st for st in s.body if isinstance(st, Stage)]
        total_floats = sum(_slab_size(st) for st in stages)
        if _BUFFER_COUNT * total_floats > _SMEM_BUDGET_FLOATS:
            continue
        updated = _double_buffer(s)
        if updated is not None:
            new_body[i] = updated
            changed = True
    return tuple(new_body) if changed else body


def _is_kouter_matmul(loop: Loop) -> bool:
    """A Loop is K-outer-matmul iff its body has ≥ 1 Stage + exactly one
    reduce Loop with a pure-matmul body (Load/Assign/Accum only, no
    cross-iteration accumulator reads). Multiple Accums are fine — they
    occur after register-tile replicates cells."""
    if int(loop.axis.extent) < 2:
        return False
    stages = [s for s in loop.body if isinstance(s, Stage)]
    if not stages:
        return False
    reduce_loops = [s for s in loop.body if isinstance(s, Loop) and s.is_reduce]
    if len(reduce_loops) != 1:
        return False
    rl = reduce_loops[0]
    if not all(isinstance(c, (Load, Assign, Accum)) for c in rl.body):
        return False
    acc_names = {c.name for c in rl.body if isinstance(c, Accum)}
    if not acc_names:
        return False
    # No stmt may read an accumulator target (online-softmax-style fusion).
    for c in rl.body:
        if isinstance(c, Accum):
            continue
        if any(d in acc_names for d in c.deps()):
            return False
    return True


def _slab_size(stage: Stage) -> int:
    n = 1
    for ax in stage.axes:
        n *= int(ax.extent)
    return n


def _double_buffer(loop: Loop) -> Loop | None:
    """Set buffer_count + phase on each Stage in the loop body, and
    rewrite each body Load that reads from a staged buffer to prepend
    the phase index."""
    phase = Var(loop.axis.name) % Literal(_BUFFER_COUNT, "int")
    staged_names: set[str] = set()
    new_body: list[Stmt] = []
    for s in loop.body:
        if isinstance(s, Stage):
            staged_names.add(s.name)
            new_body.append(dc_replace(s, buffer_count=_BUFFER_COUNT, phase=phase))
        elif isinstance(s, Loop) and s.is_reduce:
            new_body.append(dc_replace(s, body=map_body(s.body, _make_load_rewriter(staged_names, phase))))
        else:
            new_body.append(s)
    return dc_replace(loop, body=tuple(new_body))


def _make_load_rewriter(staged_names: set[str], phase):
    def fn(s: Stmt) -> Stmt:
        if isinstance(s, Load) and s.input in staged_names:
            return Load(name=s.name, input=s.input, index=(phase, *s.index))
        return s

    return fn
