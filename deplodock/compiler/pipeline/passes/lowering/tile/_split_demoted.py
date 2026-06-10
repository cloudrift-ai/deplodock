"""Demoted-matmul split — un-fuse a computed A-operand cone into its own kernel.

Loop fusion can merge a producer chain INTO a matmul's reduce body (the gated-MLP norm, an
elementwise scale, softmax stats): the multiply feeding the ``Accum`` then reads a computed
SSA cone instead of a plain ``Load``, and the warp tier dies — ``ldmatrix`` feeds MMA
fragments from staged smem, and a computed operand has no buffer to stage
(``plans/gated-mlp-tensor-cores.md``). By partition time the fused body is final, so the
demotion is visible order-independently — which is why this lives here and not as a fusion
guard: only this tier knows whether the clean matmul would actually reach the warp tier.

:func:`try_split_demoted` inspects a ``LoopOp`` and, when the cut is expressible, builds a
``Graph`` fragment ``005_split_demoted`` offers as a structural fork option:

- **producer** — computes the cone for every ``(row, k)`` and writes an ``xn`` intermediate
  (operand dtype, so the gemm's loads keep their tier eligibility). It carries the cone's
  prologue dependencies too (e.g. the norm's row-stat reduce), nested back at row level.
- **consumer** — the original kernel with the cone replaced by ``Load xn[row, k]`` (reusing
  the cone root's SSA name, so the multiply and everything downstream are untouched).

When BOTH multiply operands are computed cones (rotary + QK^T: rotary chains on Q and K
feed the reduce directly, so neither side is a plain ``Load``), the cut generalizes to two
producers: the cone that avoids the output N axis materializes as ``xn_a[rows…, k]`` (the
A operand), the N-indexed cone as ``xn_b[rows…, k, n]`` — K deliberately NOT in the last
index dim, the canonical B layout the cell tagger / stager can serve, so the consumer gemm
keeps its warp-tier shape even though the original B access was transposed ``[n, k]``. A
B cone that reads a row axis (the GQA ``head / 2`` shared-KV access) keeps that axis as a
leading ``xn_b`` dim — duplicated across the sharing heads, simple over minimal.

The checks here are the cut's own WELL-FORMEDNESS conditions, not a profitability gate:
this module deliberately does not predict whether the clean gemm will reach the warp tier
(an earlier version simulated ``is_atom_eligible`` on the rebuilt consumer and immediately
drifted from what the cell tagger actually accepts). Whether the split pays is the search's
question — the tuner measures both branches, greedy never picks the structural option
(``policy/greedy._is_structural``), and a lowering failure on either side must surface as a
rejection. Conservative bails (return ``None``, never raise) keep the fused path the only
outcome for any shape the cut doesn't fully understand: multiple K loops, a multiply with
zero or two computed sides, accums with different cones, cone values escaping past the
multiply, symbolic extents, or a cone indexed by the output N axis.
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
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import collect_invariant_names, is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context


def try_split_demoted(loop_op: LoopOp, ctx: Context, *, graph: Graph, node_id: str, out_tensor: Tensor) -> Graph | None:
    """Build the producer+consumer split fragment for a demoted matmul ``LoopOp``,
    or ``None`` when the body isn't a cleanly-cuttable demotion."""
    cut = _classify_cut(loop_op)
    if cut is None:
        return None
    leading, rows, prologue_level, outer_n, k_loop = cut

    # --- locate the computed operand cone(s) ----------------------------------
    top = tuple(k_loop.body)
    cell_def = {n: s for s in top for n in s.defines()}
    accums = [s for s in top if isinstance(s, Accum)]
    if not accums:
        return None
    first_mul = cell_def.get(accums[0].value)
    if (
        len(accums) == 1
        and isinstance(first_mul, Assign)
        and first_mul.op.name == "multiply"
        and len(first_mul.args) == 2
        and all(isinstance(cell_def.get(a), Assign) for a in first_mul.args)
    ):
        # Both operands computed (rotary QK^T) — the two-producer cut.
        return _split_two_cones(loop_op, cut, graph=graph, node_id=node_id, out_tensor=out_tensor, mul=first_mul)
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
    pro_def = _prologue_defs(prologue_level)
    lead_def = {n: s for s in leading for n in s.defines()}

    sliced = _slice_cone(cone_root, cell_def=cell_def, pro_def=pro_def, lead_def=lead_def, axis_names=axis_names)
    if sliced is None:
        return None
    cell_used, pro_used, lead_used = sliced

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
    if not _moved_defs_die_at_cone(Body(loop_op.body), moved_ids, moved_defs, frozenset({cone_root}), consumers_of_root):
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
    consumer_op = _drop_dangling_leads(consumer_op, kept_lead)

    # --- assemble the fragment -------------------------------------------------
    xn_shape = tuple(lp.axis.extent.as_static() for lp in rows) + (k_loop.axis.extent.as_static(),)
    return _assemble_fragment(
        graph,
        producers=((producer_op, Tensor(xn_id, xn_shape, weight_dtype)),),
        consumer_op=consumer_op,
        cons_id=cons_id,
        out_tensor=out_tensor,
    )


def _split_two_cones(loop_op: LoopOp, cut, *, graph: Graph, node_id: str, out_tensor: Tensor, mul: Assign) -> Graph | None:
    """The two-producer cut for a multiply whose BOTH operands are computed
    cones (rotary QK^T). The cone avoiding the output N axis materializes as
    ``xn_a[rows…, k]`` (the A operand); the N-indexed cone as
    ``xn_b[rows…, k, n]`` — K kept out of the last index dim so the consumer's
    B load has the canonical layout the cell tagger / stager serve (the
    original transposed ``[n, k]`` access would be warp-tier-unclassifiable).
    Same conservative-bail contract as the one-sided cut."""
    leading, rows, prologue_level, outer_n, k_loop = cut
    top = tuple(k_loop.body)
    cell_def = {n: s for s in top for n in s.defines()}
    axis_names = {a.name for a in loop_op.axes}
    pro_def = _prologue_defs(prologue_level)
    lead_def = {n: s for s in leading for n in s.defines()}
    k_name = k_loop.axis.name
    n_name = outer_n.axis.name
    row_names = [lp.axis.name for lp in rows]

    # --- slice both cones; assign A/B by output-N usage -----------------------
    slices = []
    for root in mul.args:
        sliced = _slice_cone(root, cell_def=cell_def, pro_def=pro_def, lead_def=lead_def, axis_names=axis_names)
        if sliced is None:
            return None
        cell_used, pro_used, lead_used = sliced
        stmts = (
            [s for s in top if id(s) in cell_used],
            [s for s in prologue_level if id(s) in pro_used],
            [s for s in leading if id(s) in lead_used],
        )
        if not stmts[0]:
            return None  # cone root must live in the cell
        slices.append((root, sliced, stmts))
    (root_1, sliced_1, stmts_1), (root_2, sliced_2, stmts_2) = slices
    # A shared cell / prologue stmt would have to compute in both producers —
    # bail rather than duplicate work the original kernel did once.
    if (sliced_1[0] & sliced_2[0]) or (sliced_1[1] & sliced_2[1]):
        return None
    axes_1 = _axes_read(stmts_1[0], axis_names)
    axes_2 = _axes_read(stmts_2[0], axis_names)
    if n_name in axes_2 and n_name not in axes_1:
        (root_a, sliced_a, stmts_a), (root_b, sliced_b, stmts_b) = slices
        axes_a, axes_b = axes_1, axes_2
    elif n_name in axes_1 and n_name not in axes_2:
        (root_b, sliced_b, stmts_b), (root_a, sliced_a, stmts_a) = slices
        axes_a, axes_b = axes_2, axes_1
    else:
        return None  # both (or neither) cone reads the output N axis — no A/B assignment
    allowed = set(row_names) | {k_name}
    if not axes_a <= allowed or not axes_b <= allowed | {n_name}:
        return None
    dtype_a = _cone_dtype(stmts_a, graph)
    dtype_b = _cone_dtype(stmts_b, graph)
    if dtype_a is None or dtype_b is None:
        return None

    # --- escape check: moved values must die at the multiply ------------------
    moved_ids = sliced_a[0] | sliced_a[1] | sliced_b[0] | sliced_b[1]
    moved_defs: set[str] = set()
    for s in (*stmts_a[0], *stmts_a[1], *stmts_b[0], *stmts_b[1]):
        moved_defs |= collect_invariant_names(s)
    if not _moved_defs_die_at_cone(Body(loop_op.body), moved_ids, moved_defs, frozenset({root_a, root_b}), {id(mul)}):
        return None

    # --- build the producers ----------------------------------------------------
    # A: every row axis (matches the one-sided cut), K innermost.
    xn_a_id = f"{node_id}__xna"
    row_vars = tuple(Var(n) for n in row_names)
    k_free_a = Loop(
        axis=k_loop.axis,
        body=Body((*stmts_a[0], Write(output=xn_a_id, index=(*row_vars, Var(k_name)), values=(root_a,)))),
    )
    level: tuple[Stmt, ...] = (*stmts_a[1], k_free_a)
    for lp in reversed(rows):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    producer_a = LoopOp(body=Body((*stmts_a[2], *level)))
    producer_a.name = f"{loop_op.name}_xna" if loop_op.name else ""

    # B: only the row axes the cone actually reads (a row-free rotary K side
    # stays 2-D), then K, then N — N innermost so the Write is coalesced and
    # the K dim lands second-to-last (the canonical B layout).
    xn_b_id = f"{node_id}__xnb"
    rows_b = [lp for lp in rows if lp.axis.name in axes_b]
    row_b_vars = tuple(Var(lp.axis.name) for lp in rows_b)
    n_free = Loop(
        axis=outer_n.axis,
        body=Body((*stmts_b[0], Write(output=xn_b_id, index=(*row_b_vars, Var(k_name), Var(n_name)), values=(root_b,)))),
    )
    level = (*stmts_b[1], Loop(axis=k_loop.axis, body=Body((n_free,))))
    for lp in reversed(rows_b):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    producer_b = LoopOp(body=Body((*stmts_b[2], *level)))
    producer_b.name = f"{loop_op.name}_xnb" if loop_op.name else ""

    # --- build the consumer ------------------------------------------------------
    def_a, def_b = cell_def[root_a], cell_def[root_b]
    load_a = Load(names=(root_a,), input=xn_a_id, index=(*row_vars, Var(k_name)))
    load_b = Load(names=(root_b,), input=xn_b_id, index=(*row_b_vars, Var(k_name), Var(n_name)))
    moved_cell = sliced_a[0] | sliced_b[0]
    new_top: list[Stmt] = []
    for s in top:
        if s is def_a:
            new_top.append(load_a)
        elif s is def_b:
            new_top.append(load_b)
        elif id(s) in moved_cell:
            continue
        else:
            new_top.append(s)
    new_k_loop = Loop(axis=k_loop.axis, body=Body(tuple(new_top)), unroll=k_loop.unroll)
    new_outer_n = Loop(
        axis=outer_n.axis,
        body=Body(tuple(new_k_loop if s is k_loop else s for s in outer_n.body)),
        unroll=outer_n.unroll,
    )
    pro_used_all = sliced_a[1] | sliced_b[1]
    level = tuple(new_outer_n if s is outer_n else s for s in prologue_level if id(s) not in pro_used_all or s is outer_n)
    for lp in reversed(rows):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    cons_id = f"{node_id}__mm"
    lead_used_all = {id(s) for s in (*stmts_a[2], *stmts_b[2])}
    kept_lead = tuple(s for s in leading if id(s) not in lead_used_all)
    consumer_op = LoopOp(body=Body((*kept_lead, *level)))
    consumer_op = _rename_write_output(consumer_op, old=node_id, new=cons_id)
    consumer_op.name = loop_op.name
    consumer_op = _drop_dangling_leads(consumer_op, kept_lead)

    # --- assemble the fragment -----------------------------------------------------
    E_K = k_loop.axis.extent.as_static()
    xn_a_shape = tuple(lp.axis.extent.as_static() for lp in rows) + (E_K,)
    xn_b_shape = tuple(lp.axis.extent.as_static() for lp in rows_b) + (E_K, outer_n.axis.extent.as_static())
    return _assemble_fragment(
        graph,
        producers=((producer_a, Tensor(xn_a_id, xn_a_shape, dtype_a)), (producer_b, Tensor(xn_b_id, xn_b_shape, dtype_b))),
        consumer_op=consumer_op,
        cons_id=cons_id,
        out_tensor=out_tensor,
    )


def _assemble_fragment(graph: Graph, *, producers, consumer_op: LoopOp, cons_id: str, out_tensor: Tensor) -> Graph | None:
    """Wire the producer/consumer LoopOps into a ``Graph`` fragment: InputOps
    for every external buffer, one node per producer (its ``xn`` Tensor is the
    node id), the consumer as the fragment output. ``producers`` is a sequence
    of ``(LoopOp, Tensor)``. Restamps structural features on every new body."""
    frag = Graph()
    xn_ids = {t.name for _, t in producers}
    for op in (*(p for p, _ in producers), consumer_op):
        for buf in op.inputs:
            if buf in xn_ids or buf in frag.nodes:
                continue
            ext = graph.nodes.get(buf)
            if ext is None:
                return None
            frag.add_node(InputOp(), [], Tensor(buf, ext.output.shape, ext.output.dtype), node_id=buf)
    for op, tensor in producers:
        frag.add_node(op, list(op.inputs), tensor, node_id=tensor.name)
    frag.add_node(consumer_op, list(consumer_op.inputs), Tensor(out_tensor.name, out_tensor.shape, out_tensor.dtype), node_id=cons_id)
    frag.outputs = [cons_id]

    # Every new body differs from the fused one — restamp the structural
    # identity (992 ran at fusion end and never re-runs; stale S_* would make
    # the split kernels featurize as the fused kernel for the learned prior).
    feats = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.992_stamp_structural_features")
    for nid in (*xn_ids, cons_id):
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


def _moved_defs_die_at_cone(
    body: Body, moved_ids: set[int], moved_defs: set[str], cone_roots: frozenset[str], allowed_ids: set[int]
) -> bool:
    """True iff no stmt outside the moved set reads a moved def — except the
    multiplies (``allowed_ids``) reading only ``cone_roots``."""

    def walk(stmts) -> bool:
        for s in stmts:
            if id(s) in moved_ids:
                continue  # the whole subtree moves; internal uses are fine
            reads = set(s.deps()) & moved_defs
            if reads and not (id(s) in allowed_ids and reads <= cone_roots):
                return False
            for sub in s.nested():
                if not walk(sub):
                    return False
        return True

    return walk(body)


def _prologue_defs(prologue_level) -> dict[str, Stmt]:
    """Map every cross-scope SSA name a prologue-level stmt exposes to that
    stmt (a whole reduce Loop exposes its Accum names)."""
    pro_def: dict[str, Stmt] = {}
    for s in prologue_level:
        for n in collect_invariant_names(s):
            pro_def[n] = s
    return pro_def


def _slice_cone(
    root: str, *, cell_def: dict[str, Stmt], pro_def: dict[str, Stmt], lead_def: dict[str, Stmt], axis_names: set[str]
) -> tuple[set[int], set[int], set[int]] | None:
    """Backward-slice ``root`` over the cell / prologue / leading scopes.
    Returns the ``(cell, prologue, leading)`` used-stmt id-sets, or ``None``
    when the cone reads the matmul's own running accumulator or a name from
    an unmodeled scope."""
    cell_used: set[int] = set()
    pro_used: set[int] = set()
    lead_used: set[int] = set()
    pending = deque([root])
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
    return cell_used, pro_used, lead_used


def _axes_read(stmts, axis_names: set[str]) -> set[str]:
    """Axis names appearing in the stmts' carried Exprs (Load / Write indices,
    Select predicates) — the data the cut uses to tell the row-only A cone
    from the N-indexed B cone."""
    return {v for s in stmts for e in s.exprs() for v in e.free_vars()} & axis_names


def _cone_dtype(stmts_triple, graph: Graph):
    """The uniform dtype of every graph-resolvable Load in the cone's moved
    stmts — the dtype its ``xn`` buffer materializes at (mirrors the
    one-sided cut's operand-dtype rule). ``None`` (bail) when a Load source
    is unresolvable or the leaf dtypes disagree."""
    dtypes = set()
    for ld in Body(tuple(s for group in stmts_triple for s in group)).iter_of_type(Load):
        node = graph.nodes.get(ld.input)
        if node is None:
            return None
        dtypes.add(node.output.dtype)
    if len(dtypes) != 1:
        return None
    return dtypes.pop()


def _drop_dangling_leads(consumer_op: LoopOp, kept_lead: tuple[Stmt, ...]) -> LoopOp:
    """Leading stmts used by BOTH sides were copied to the producer; drop the
    consumer copies that nothing reads anymore (a dangling def renders fine
    but is noise). Conversely a shared leading stmt stays in both."""
    used_in_consumer = {d for s in consumer_op.body.iter() for d in s.deps()}
    if not any(set(s.defines()) - used_in_consumer for s in kept_lead):
        return consumer_op
    body = tuple(s for s in consumer_op.body if not (isinstance(s, (Load, Assign)) and not (set(s.defines()) & used_in_consumer)))
    name = consumer_op.name
    out = LoopOp(body=Body(body))
    out.name = name
    return out


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
