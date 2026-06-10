"""Demoted-matmul split — un-fuse a computed A-operand cone into its own kernel.

Loop fusion can merge a producer chain INTO a matmul's reduce body (the gated-MLP norm, an
elementwise scale, softmax stats): the multiply feeding the ``Accum`` then reads a computed
SSA cone instead of a plain ``Load``, and the warp tier dies — ``ldmatrix`` feeds MMA
fragments from staged smem, and a computed operand has no buffer to stage
(``plans/gated-mlp-tensor-cores.md``). By partition time the fused body is final, so the
demotion is visible order-independently — which is why this lives here and not as a fusion
guard: only this tier knows whether the clean matmul would actually reach the warp tier.

:func:`try_split_demoted` inspects a ``LoopOp`` and, when profitable, builds a two-kernel
``Graph`` fragment ``010_partition_loops`` offers as a structural fork option:

- **producer** — computes the cone for every ``(row, k)`` and writes an ``xn`` intermediate
  (operand dtype, so the gemm's loads stay atom-eligible). It carries the cone's prologue
  dependencies too (e.g. the norm's row-stat reduce), nested back at row level.
- **consumer** — the original kernel with the cone replaced by ``Load xn[row, k]`` (reusing
  the cone root's SSA name, so the multiply and everything downstream are untouched).

The split is offered only when the clean consumer would actually enumerate warp rows: the
``MMA`` knob on, sm_90+ (or an explicit ``DEPLODOCK_MMA`` pin), and ``is_atom_eligible``
passing on the rebuilt consumer — splitting buys ~nothing when the gemm stays scalar.
Conservative bails (return ``None``, never raise) keep the fused path the default for any
shape the cut doesn't fully understand: multiple K loops, a multiply with zero or two
computed sides, accums with different cones, cone values escaping past the multiply,
symbolic extents, or a cone indexed by the output N axis.

No re-match marker is needed: the fused branch rebinds the root to a ``TileOp`` (pattern no
longer matches) and the split branch replaces it with two clean LoopOps — the consumer has
no cone (not demoted) and the producer has no matmul reduce, so neither re-splits.
"""

from __future__ import annotations

import importlib
from collections import deque
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY
from deplodock.compiler.pipeline.passes.lowering.tile._atom import is_atom_eligible
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import mma_mode
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import collect_invariant_names, is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context


def try_split_demoted(loop_op: LoopOp, ctx: Context, *, graph: Graph, node_id: str, out_tensor: Tensor) -> Graph | None:
    """Build the producer+consumer split fragment for a demoted matmul ``LoopOp``,
    or ``None`` when the kernel isn't a profitable, cleanly-cuttable demotion."""
    mma_on, pinned_atom = mma_mode()
    if not mma_on:
        return None
    if pinned_atom not in ATOM_REGISTRY and ctx.compute_capability < (9, 0):
        # Auto-enumerated mma.sync needs the Hopper+ swizzled-TMA fast path
        # (mirrors the 010 planner gate); a DEPLODOCK_MMA pin is authoritative.
        return None

    cut = _classify_cut(loop_op)
    if cut is None:
        return None
    leading, rows, prologue_level, outer_n, k_loop = cut

    # --- locate the computed A operand (the cone root) -----------------------
    top = tuple(k_loop.body)
    cell_def = {n: s for s in top for n in s.defines()}
    accums = [s for s in top if isinstance(s, Accum)]
    if not accums:
        return None
    cone_root: str | None = None
    weight_dtype = None
    for acc in accums:
        mul = cell_def.get(acc.value)
        if not isinstance(mul, Assign) or mul.op.name != "multiply" or len(mul.args) != 2:
            return None
        arg_defs = [cell_def.get(a) for a in mul.args]
        load_args = [a for a, d in zip(mul.args, arg_defs, strict=True) if isinstance(d, Load)]
        cone_args = [a for a, d in zip(mul.args, arg_defs, strict=True) if isinstance(d, Assign)]
        if len(load_args) != 1 or len(cone_args) != 1:
            return None  # pure cell (not demoted) or doubly-computed (ambiguous)
        weight = cell_def[load_args[0]]
        if k_loop.axis.name not in {v for e in weight.index for v in e.free_vars()}:
            return None
        wnode = graph.nodes.get(weight.input)
        if wnode is None:
            return None
        weight_dtype = wnode.output.dtype
        if cone_root is None:
            cone_root = cone_args[0]
        elif cone_root != cone_args[0]:
            return None  # accums with different cones — no shared operand to materialize
    assert cone_root is not None

    # --- backward-slice the cone over the cell / prologue / leading scopes ---
    axis_names = {a.name for a in loop_op.axes}
    pro_def: dict[str, Stmt] = {}
    for s in prologue_level:
        for n in collect_invariant_names(s):
            pro_def[n] = s
    lead_def = {n: s for s in leading for n in s.defines()}

    cell_used: set[int] = set()
    pro_used: set[int] = set()
    lead_used: set[int] = set()
    pending = deque([cone_root])
    seen: set[str] = set()
    while pending:
        n = pending.popleft()
        if n in seen or n in axis_names:
            continue
        seen.add(n)
        s = cell_def.get(n)
        if s is not None:
            if isinstance(s, Accum):
                return None  # cone reads the matmul's own running accumulator — not cuttable
            cell_used.add(id(s))
            pending.extend(s.deps())
            continue
        s = pro_def.get(n)
        if s is not None:
            pro_used.add(id(s))
            pending.extend(_external_reads(s, axis_names))
            continue
        s = lead_def.get(n)
        if s is not None:
            lead_used.add(id(s))
            pending.extend(s.deps())
            continue
        return None  # name from an unmodeled scope — bail conservatively

    cell_stmts = [s for s in top if id(s) in cell_used]
    pro_stmts = [s for s in prologue_level if id(s) in pro_used]
    lead_stmts = [s for s in leading if id(s) in lead_used]
    if not cell_stmts:
        return None  # cone root must live in the cell (a K-invariant operand isn't this pattern)

    # The cone may only address row axes and K — an N-indexed cone would
    # materialize an (M, N, K) buffer, defeating the split.
    row_names = [lp.axis.name for lp in rows]
    allowed = set(row_names) | {k_loop.axis.name}
    for s in cell_stmts:
        if isinstance(s, Load) and any(v not in allowed for e in s.index for v in e.free_vars()):
            return None

    # --- escape check: moved values must die at the multiply -----------------
    moved_ids = cell_used | pro_used
    moved_defs: set[str] = set()
    for s in (*cell_stmts, *pro_stmts):
        moved_defs |= collect_invariant_names(s)
    consumers_of_root = {id(cell_def[a.value]) for a in accums}  # the multiplies
    if not _moved_defs_die_at_cone(Body(loop_op.body), moved_ids, moved_defs, cone_root, consumers_of_root):
        return None

    # --- build the producer ---------------------------------------------------
    xn_id = f"{node_id}__xn"
    row_vars = tuple(Var(n) for n in row_names)
    k_free = Loop(
        axis=k_loop.axis,
        body=Body((*cell_stmts, Write(output=xn_id, index=(*row_vars, Var(k_loop.axis.name)), values=(cone_root,)))),
    )
    level: tuple[Stmt, ...] = (*pro_stmts, k_free)
    for lp in reversed(rows):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    producer_op = LoopOp(body=Body((*lead_stmts, *level)))
    producer_op.name = f"{loop_op.name}_xn" if loop_op.name else ""

    # --- build the consumer ---------------------------------------------------
    cone_def_stmt = cell_def[cone_root]
    xn_load = Load(names=(cone_root,), input=xn_id, index=(*row_vars, Var(k_loop.axis.name)))
    new_top: list[Stmt] = []
    for s in top:
        if s is cone_def_stmt:
            new_top.append(xn_load)
        elif id(s) in cell_used:
            continue
        else:
            new_top.append(s)
    new_k_loop = Loop(axis=k_loop.axis, body=Body(tuple(new_top)), unroll=k_loop.unroll)
    new_outer_n = Loop(
        axis=outer_n.axis,
        body=Body(tuple(new_k_loop if s is k_loop else s for s in outer_n.body)),
        unroll=outer_n.unroll,
    )
    level = tuple(new_outer_n if s is outer_n else s for s in prologue_level if id(s) not in pro_used or s is outer_n)
    for lp in reversed(rows):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    cons_id = f"{node_id}__mm"
    kept_lead = tuple(s for s in leading if id(s) not in lead_used)
    consumer_op = LoopOp(body=Body((*kept_lead, *level)))
    consumer_op = _rename_write_output(consumer_op, old=node_id, new=cons_id)
    consumer_op.name = loop_op.name
    # Leading stmts used by BOTH sides were copied to the producer; drop the
    # consumer copies that nothing reads anymore (a dangling def renders fine
    # but is noise). Conversely a shared leading stmt stays in both.
    used_in_consumer = {d for s in consumer_op.body.iter() for d in s.deps()}
    if any(set(s.defines()) - used_in_consumer for s in kept_lead):
        body = tuple(s for s in consumer_op.body if not (isinstance(s, (Load, Assign)) and not (set(s.defines()) & used_in_consumer)))
        name = consumer_op.name
        consumer_op = LoopOp(body=Body(body))
        consumer_op.name = name

    # --- assemble the fragment and gate on the clean gemm's warp tier --------
    frag = Graph()
    for op in (producer_op, consumer_op):
        for buf in op.inputs:
            if buf == xn_id or buf in frag.nodes:
                continue
            ext = graph.nodes.get(buf)
            if ext is None:
                return None
            frag.add_node(InputOp(), [], Tensor(buf, ext.output.shape, ext.output.dtype), node_id=buf)
    xn_shape = tuple(lp.axis.extent.as_static() for lp in rows) + (k_loop.axis.extent.as_static(),)
    frag.add_node(producer_op, list(producer_op.inputs), Tensor(xn_id, xn_shape, weight_dtype), node_id=xn_id)
    frag.add_node(consumer_op, list(consumer_op.inputs), Tensor(out_tensor.name, out_tensor.shape, out_tensor.dtype), node_id=cons_id)
    frag.outputs = [cons_id]

    if not any(is_atom_eligible(atom, consumer_op, ctx, graph=frag) for atom in ATOM_REGISTRY.values()):
        return None

    # Both new bodies differ from the fused one — restamp the structural
    # identity (992 ran at fusion end and never re-runs; stale S_* would make
    # the split kernels featurize as the fused kernel for the learned prior).
    feats = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.992_stamp_structural_features")
    for nid in (xn_id, cons_id):
        op = frag.nodes[nid].op
        op.knobs = {k: v for k, v in op.knobs.items() if not k.startswith("S_")}
        op.knobs.update(feats.structure_features(op.body, frag))
    return frag


def _classify_cut(loop_op: LoopOp):
    """Decompose the body into ``(leading, rows, prologue_level, outer_n, k_loop)``.

    Walks the single-stmt free-Loop chain (the row axes), stopping at either a
    prologue level (sibling reduces/assigns + exactly one non-reduce Loop that
    contains the matmul — the gated-MLP / P@V shape) or the matmul-bearing Loop
    itself (in-cell-cone-only shape, e.g. scale→matmul). Returns ``None`` for
    anything else: no matmul reduce, more than one K loop, symbolic extents, or
    no row axis (the planner requires ``outer_m`` for matmuls anyway).
    ``prologue_level`` is the full stmt tuple at the stopped level (``outer_n``
    included, original order) so the consumer rebuild preserves ordering.
    """
    leading: list[Stmt] = []
    rest = tuple(loop_op.body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]

    rows: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        rows.append(cur[0])
        cur = tuple(cur[0].body)

    inner = [s for s in cur if isinstance(s, Loop) and not s.is_reduce]
    if len(inner) == 1 and _contains_matmul_reduce_loop(inner[0]):
        outer_n = inner[0]
        prologue_level = cur
    elif rows and not inner:
        outer_n = rows.pop()
        prologue_level = (outer_n,)
    else:
        return None
    if not rows:
        return None

    k_loops = [s for s in outer_n.body if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s)]
    if len(k_loops) != 1:
        return None
    k_loop = k_loops[0]
    if not k_loop.axis.extent.is_static or not outer_n.axis.extent.is_static:
        return None
    if any(not lp.axis.extent.is_static for lp in rows):
        return None
    return tuple(leading), rows, prologue_level, outer_n, k_loop


def _contains_matmul_reduce_loop(stmt: Stmt) -> bool:
    if isinstance(stmt, Loop) and stmt.is_reduce and is_matmul_reduce(stmt):
        return True
    return any(_contains_matmul_reduce_loop(c) for body in stmt.nested() for c in body)


def _external_reads(stmt: Stmt, axis_names: set[str]) -> set[str]:
    """Names ``stmt`` (incl. a whole Loop subtree) reads from its enclosing
    scope — all nested deps minus internally-defined names and loop axes."""
    reads: set[str] = set()
    defs: set[str] = set()

    def walk(s: Stmt) -> None:
        reads.update(s.deps())
        defs.update(s.defines())
        for body in s.nested():
            if isinstance(s, Loop):
                defs.add(s.axis.name)
            for c in body:
                walk(c)

    walk(stmt)
    return reads - defs - axis_names


def _moved_defs_die_at_cone(body: Body, moved_ids: set[int], moved_defs: set[str], cone_root: str, allowed_ids: set[int]) -> bool:
    """True iff no stmt outside the moved set reads a moved def — except the
    multiplies (``allowed_ids``) reading exactly ``cone_root``."""

    def walk(stmts) -> bool:
        for s in stmts:
            if id(s) in moved_ids:
                continue  # the whole subtree moves; internal uses are fine
            reads = set(s.deps()) & moved_defs
            if reads and not (id(s) in allowed_ids and reads == {cone_root}):
                return False
            for sub in s.nested():
                if not walk(sub):
                    return False
        return True

    return walk(body)


def _rename_write_output(op: LoopOp, *, old: str, new: str) -> LoopOp:
    """Rebuild ``op`` with every ``Write(output=old)`` renamed to ``new``."""

    def walk(stmts) -> tuple[Stmt, ...]:
        out: list[Stmt] = []
        for s in stmts:
            if isinstance(s, Write) and s.output == old:
                from dataclasses import replace as dc_replace  # noqa: PLC0415

                out.append(dc_replace(s, output=new))
            elif isinstance(s, Loop):
                from dataclasses import replace as dc_replace  # noqa: PLC0415

                out.append(dc_replace(s, body=Body(walk(s.body))))
            else:
                out.append(s)
        return tuple(out)

    return LoopOp(body=Body(walk(op.body)))
