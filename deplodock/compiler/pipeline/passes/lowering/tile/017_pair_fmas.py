"""Pair each multiply with its consuming Accum inside matmul-shaped
reduce bodies.

A typical register-tiled (008) matmul reduce body emits, in order::

    [loads]   in0 = load A[..]
              in1 = load B[..]
              ...
    [mults]   v0 = in_a0 * in_b0
              v1 = in_a0 * in_b1
              ...
              v15 = in_a3 * in_b3
    [accs]    acc0 += v0
              acc1 += v1
              ...
              acc15 += v15

NVCC's instruction scheduler consumes this and almost always fuses
each ``v_i = a*b; acc_i += v_i`` pair into a single ``FFMA`` SASS
instruction. But the source-level grouping leaves all 16 ``v*``
registers simultaneously live at the boundary between mults and accs,
which can hurt the register allocator on dense inner-loop bodies
(specifically the BK=64 + ``014_bk_split`` merged variant — 32
``v*`` would be live at once before pairing). Reordering to::

    [loads]   in0..in7 = load ...
    [pairs]   v0  = in_a0 * in_b0;  acc0  += v0
              v1  = in_a0 * in_b1;  acc1  += v1
              ...
              v15 = in_a3 * in_b3;  acc15 += v15

shrinks each ``v_i``'s live range to two stmts. Same number of FFMAs
either way (the SASS is typically identical), but the IR-level live
ranges line up with what NVCC's allocator wants.

Pre-conditions for firing on a reduce-Loop body:

- All ``Accum`` stmts in the body have a unique ``value`` SSA name
  defined elsewhere in the body by an ``Assign``.
- That defining ``Assign`` is a binary multiply with two distinct
  load-defined args (the matmul-shape signature: ``v = a * b`` where
  both ``a`` and ``b`` are SSAs introduced earlier in the body).
- No other stmt reads any ``v_i`` between its definition and its
  consuming ``Accum`` (otherwise reordering changes semantics — bail).

Bodies that don't match the matmul pattern (softmax / RMSNorm / SDPA
with online-softmax merges) pass through unchanged.

Idempotence: a pre-paired body has every ``Accum`` immediately
following its defining ``Assign``; the pass re-emits the same shape.

Position: after ``014_bk_split`` (so the merged-body shape is
visible) and after ``016_unroll_small_loops`` (which only flips
``unroll`` flags, doesn't reshape the body).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Loop, Stmt, StridedLoop, Tile
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body, changed = _walk_body(root.op.body)
    if not changed:
        raise RuleSkipped("no matmul-shape reduce body needed FMA pairing")
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _walk_body(body: Body) -> tuple[Body, bool]:
    """Recurse into Tile / Loop / StridedLoop / Cond. Pair the FMA chain
    of each reduce Loop's immediate body."""
    new_stmts: list[Stmt] = []
    changed = False
    for s in body:
        new_s, c = _walk_stmt(s)
        new_stmts.append(new_s)
        changed = changed or c
    return Body(tuple(new_stmts)), changed


def _walk_stmt(s: Stmt) -> tuple[Stmt, bool]:
    if isinstance(s, (Loop, StridedLoop)):
        new_inner, c_inner = _walk_body(s.body)
        if s.is_reduce:
            paired = _pair_reduce_body(new_inner)
            if paired is not new_inner:
                return dc_replace(s, body=paired), True
        if c_inner:
            return dc_replace(s, body=new_inner), True
        return s, False
    if isinstance(s, Tile):
        new_inner, c = _walk_body(s.body)
        if c:
            return Tile(axes=s.axes, body=new_inner), True
        return s, False
    if isinstance(s, Cond):
        nb, cb = _walk_body(s.body)
        ne, ce = _walk_body(s.else_body)
        if cb or ce:
            return Cond(cond=s.cond, body=nb, else_body=ne), True
        return s, False
    return s, False


def _pair_reduce_body(body: Body) -> Body:
    """Reorder a matmul-shape reduce body so each multiply ``Assign`` is
    immediately followed by the ``Accum`` that consumes it. Bail (return
    the input unchanged) if the body doesn't match the pattern."""
    accums = [s for s in body if isinstance(s, Accum)]
    if not accums:
        return body

    # Map each Accum to the Assign that defines its ``value``.
    assign_by_name: dict[str, Assign] = {s.name: s for s in body if isinstance(s, Assign)}
    pairs: list[tuple[Assign, Accum]] = []
    for a in accums:
        defining = assign_by_name.get(a.value)
        if defining is None or defining.op.name != "multiply" or len(defining.args) != 2:
            return body  # not the matmul shape — bail
        pairs.append((defining, a))

    # Each ``v`` (Assign defining an Accum's value) must be consumed
    # only by its matching Accum — anyone else reading the v means
    # reordering changes semantics.
    v_names = {p[0].name for p in pairs}
    consumers: dict[str, list[Stmt]] = {n: [] for n in v_names}
    for s in body:
        for d in s.deps():
            if d in consumers:
                consumers[d].append(s)
    for cs in consumers.values():
        if len(cs) != 1:
            return body
        if not isinstance(cs[0], Accum):
            return body

    # Idempotence: if the body is already in (... loads ... pair pair pair)
    # shape, return as-is to keep the rule's ``changed`` signal honest.
    paired_set = {id(s) for p in pairs for s in p}
    prelude = tuple(s for s in body if id(s) not in paired_set)
    paired_seq = tuple(stmt for p in pairs for stmt in p)
    new_body = prelude + paired_seq
    if tuple(body) == new_body:
        return body
    return Body(new_body)
