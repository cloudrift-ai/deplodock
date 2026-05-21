"""Emit ``Combine`` after the K_o reduce subtree for cooperative-K kernels.

Runs right after ``001_launch_geometry`` lifts the K_c axis into
``Tile.axes`` with ``Role.COOPERATIVE_STRIDE``. The planner stamped
the role; this rule materializes the cooperative reduction primitive:

1. Walk ``Tile.body`` to find the ``Loop(K_o, SERIAL_OUTER)`` that
   wraps a ``Loop(K_i, STAGE_INNER, reduce, [Accum])`` subtree.
2. Insert ``Combine(name=accum.name, op=accum.op)`` immediately after
   the K_o Loop closes, one per immediate Accum inside K_i.
3. Wrap trailing ``Write`` stmts whose index doesn't reference K_c in
   a ``Cond(K_c == 0)`` so only one K_c thread commits the post-Combine
   value. ``materialize_tile`` translates that to ``if (K_c_var == 0)
   { ... }``. Writes whose index *does* reference K_c (per-K-thread
   post-pointwise output, e.g. RMSNorm's normalized write) pass through
   unchanged — every thread writes its own cell.

The materializer's existing ``_emit_combine`` produces the smem
tree-halve / warp-shuffle and renames the Accum so subsequent reads
see the broadcast value.

Replaces the deleted ``005_cooperative_reduce.py`` (commit 66528c36).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import BoundAxis, Role
from deplodock.compiler.ir.elementwise import ElementwiseImpl  # noqa: F401  (Combine.op type)
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.tile.ir import Combine, Tile, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]


def rewrite(root: Node) -> Graph | None:
    body = root.op.body
    idx, tile = single_tile(body)

    coop_axes = tuple(ba for ba in tile.axes if ba.role is Role.COOPERATIVE_STRIDE)
    if not coop_axes:
        raise RuleSkipped("no Role.COOPERATIVE_STRIDE axis in Tile.axes")
    if len(coop_axes) > 1:
        raise RuleSkipped(f"v1 supports a single cooperative axis; got {len(coop_axes)}")
    coop_axis = coop_axes[0]

    new_tile_body = _rewrite_body(tuple(tile.body), coop_axis)
    if new_tile_body == tuple(tile.body):
        raise RuleSkipped("no K_o · K_i reduce subtree found (or already rewritten)")

    new_tile = Tile(axes=tile.axes, body=new_tile_body)
    return TileOp(body=body[:idx] + (new_tile,) + body[idx + 1 :], name=root.op.name, knobs=root.op.knobs)


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


def _rewrite_body(stmts: tuple[Stmt, ...], coop_axis: BoundAxis) -> tuple[Stmt, ...]:
    """Locate the cooperative-K reduce subtree, insert Combines after it,
    and guard trailing scalar Writes with ``Cond(K_c == 0)``.

    The reduce subtree shape is one of:

    - ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, reduce, [Accum]))``
      — the canonical multi-chunk form.
    - ``Loop(K_i, STAGE_INNER, reduce, [Accum])`` — same shape but K_o
      collapsed to extent 1 and inlined by ``drop_size_one_free_axes``.

    Idempotence: if any ``Combine`` sibling is already present in
    ``stmts``, skip — re-firing would emit duplicate tree-halves.
    """
    if any(isinstance(s, Combine) for s in stmts):
        return stmts

    k_idx = _find_reduce_subtree_index(stmts)
    if k_idx is None:
        # Try descending one wrapper level (an output THREAD layer
        # could still sit between Tile body and the reduce subtree if
        # v1's BN=BM=1 constraint didn't fully collapse it).
        for i, s in enumerate(stmts):
            if isinstance(s, (Loop, StridedLoop)):
                inner = _rewrite_body(tuple(s.body), coop_axis)
                if inner != tuple(s.body):
                    return stmts[:i] + (dc_replace(s, body=inner),) + stmts[i + 1 :]
        return stmts

    k_loop = stmts[k_idx]
    accums = _collect_immediate_accums_in_reduce(k_loop)
    if not accums:
        return stmts

    combines: list[Stmt] = [Combine(name=a.name, op=a.op) for a in accums]

    # Post-reduce stmts: Writes referencing the cooperative axis stay
    # per-thread; Writes whose index doesn't reference the cooperative
    # axis get Cond-guarded so only one thread commits.
    coop_name = coop_axis.axis.name
    tail: list[Stmt] = []
    for s in stmts[k_idx + 1 :]:
        tail.append(_guard_scalar_write(s, coop_name))

    return stmts[:k_idx] + (k_loop, *combines, *tail)


def _find_reduce_subtree_index(stmts: tuple[Stmt, ...]) -> int | None:
    """Find the index of the cooperative-K reduce subtree at this body
    level. Either:

    - A SERIAL_OUTER (non-reduce) Loop that wraps a STAGE_INNER reduce
      Loop with an immediate Accum.
    - A STAGE_INNER reduce Loop with an immediate Accum directly at
      this level (when K_o was extent-1 and inlined).

    Returns ``None`` when no such structure exists at this level."""
    for i, s in enumerate(stmts):
        if isinstance(s, Loop) and not s.is_reduce and s.role is Role.SERIAL_OUTER and _wraps_stage_inner_reduce(s.body):
            return i
        if (
            isinstance(s, Loop)
            and s.is_reduce
            and s.role is Role.STAGE_INNER
            and any(isinstance(c, Accum) for c in s.body)
        ):
            return i
    return None


def _wraps_stage_inner_reduce(stmts) -> bool:
    """True iff ``stmts`` contains a STAGE_INNER reduce Loop with an
    immediate Accum (allowing arbitrary single-stmt nesting)."""
    cur = tuple(stmts)
    while len(cur) == 1 and isinstance(cur[0], (Loop, StridedLoop)):
        s = cur[0]
        if s.is_reduce and s.role is Role.STAGE_INNER and any(isinstance(c, Accum) for c in s.body):
            return True
        cur = tuple(s.body)
    return False


def _collect_immediate_accums_in_reduce(k_loop) -> list[Accum]:
    """Walk down to the STAGE_INNER reduce Loop and collect every
    immediate Accum stmt inside it. Handles both the K_o-wrapped form
    and the bare K_i form (when K_o was inlined as extent-1)."""
    if k_loop.is_reduce and any(isinstance(c, Accum) for c in k_loop.body):
        return [c for c in k_loop.body if isinstance(c, Accum)]
    cur = tuple(k_loop.body)
    while len(cur) == 1 and isinstance(cur[0], (Loop, StridedLoop)):
        s = cur[0]
        if s.is_reduce and any(isinstance(c, Accum) for c in s.body):
            return [c for c in s.body if isinstance(c, Accum)]
        cur = tuple(s.body)
    return []


def _guard_scalar_write(s: Stmt, coop_name: str) -> Stmt:
    """Wrap ``s`` in ``Cond(coop == 0)`` when it's a Write whose index
    doesn't reference the cooperative axis. Otherwise (per-thread Write
    or non-Write stmt) returns ``s`` unchanged. Descends into block
    stmts so multi-line post-reduce epilogues get guarded uniformly."""
    if isinstance(s, Write):
        free = set()
        for e in s.index:
            free |= e.free_vars()
        if coop_name not in free:
            return Cond(
                cond=BinaryExpr("==", Var(coop_name), Literal(0, "int")),
                body=Body((s,)),
                else_body=Body(()),
            )
        return s
    if isinstance(s, (Loop, StridedLoop)):
        inner = tuple(_guard_scalar_write(c, coop_name) for c in s.body)
        if inner != tuple(s.body):
            return dc_replace(s, body=inner)
        return s
    if isinstance(s, Cond):
        b = tuple(_guard_scalar_write(c, coop_name) for c in s.body)
        e = tuple(_guard_scalar_write(c, coop_name) for c in s.else_body)
        if b != tuple(s.body) or e != tuple(s.else_body):
            return Cond(cond=s.cond, body=Body(b), else_body=Body(e))
        return s
    return s
