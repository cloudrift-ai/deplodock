"""Demoted-matmul split — un-fuse computed multiply-operand cones into producer kernels.

Loop fusion can merge a producer chain INTO a matmul's reduce body (the gated-MLP norm, an
elementwise scale, softmax stats, rotary): a multiply operand feeding the ``Accum`` then
reads a computed SSA cone instead of a plain ``Load``, and the warp tier dies —
``ldmatrix`` feeds MMA fragments from staged smem, and a computed operand has no buffer to
stage (``plans/gated-mlp-tensor-cores.md``). By partition time the fused body is final, so
the demotion is visible order-independently — which is why this lives here and not as a
fusion guard: only this tier knows whether the clean matmul would actually reach the warp
tier.

:func:`try_split_demoted` inspects a ``LoopOp`` and, when the cut is expressible, builds a
``Graph`` fragment ``005_split_demoted`` offers as a structural fork option. ONE rule, no
per-shape cases: each multiply operand is independently a plain ``Load`` (stays put) or a
computed cone (becomes a producer kernel); every distinct cone materializes an ``xn``
intermediate over exactly the axes it reads —

    ``xn[row axes read…, k]``        (a row cone — the A operand)
    ``xn[row axes read…, k, n]``     (an N-reading cone — the B operand)

— and the consumer is the original kernel with each cone-root def replaced by
``Load xn[…]`` under the same SSA name, so the multiply and everything downstream are
untouched. K deliberately lands second-to-last in an N-reading cone's buffer: the original
access may be transposed ``[n, k]`` (rotary QK^T), which the cell tagger / stager cannot
serve, but the producer's Write order is ours to choose, so the consumer's B load comes
out canonical. Each producer carries its cone's prologue dependencies (e.g. the norm's
row-stat reduce, P@V's softmax stats), nested back at row level, and its ``xn``
materializes at the cone's own (uniform) leaf-Load dtype — value-preserving, and identical
to the old "other operand's dtype" rule on every shape seen so far. The familiar shapes
are instances of the one rule: norm→linear / scale→matmul = one row cone beside a Load;
SDPA P@V = one row cone with prologue deps; rotary QK^T = a row cone + an N cone (the GQA
``head / 2`` shared-KV read keeps that row axis as a leading dim — duplicated across the
sharing heads, simple over minimal); a weight-side scale = one N cone beside a Load.

The checks here are the cut's own WELL-FORMEDNESS conditions, not a profitability gate:
this module deliberately does not predict whether the clean gemm will reach the warp tier
(an earlier version simulated ``is_atom_eligible`` on the rebuilt consumer and immediately
drifted from what the cell tagger actually accepts). Whether the split pays is the search's
question — the tuner measures both branches, greedy never picks the structural option
(``policy/greedy._is_structural``), and a lowering failure on either side must surface as a
rejection. Conservative bails (return ``None``, never raise) keep the fused path the only
outcome for any shape the cut doesn't fully understand: multiple K loops, no computed
operand, a K-invariant Load operand, accums not sharing one cone set, cones sharing stmts,
cone values escaping past the multiply, mixed-dtype cone leaves, symbolic extents, or more
than one cone reading the output N axis (two ``(…, K, N)`` buffers would re-do the
matmul's own volume — the materialization that defeats the split).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from deplodock.compiler.ir.stmt.body import Cone
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

if TYPE_CHECKING:
    from deplodock.compiler.context import Context


@dataclass
class _RootCone:
    """One computed multiply operand: its chained backward :class:`Cone`\\ s
    over the cell / prologue / leading scope levels, the axes it reads
    (recursively, prologue/leading deps included), and the uniform leaf
    dtype its ``xn`` buffer materializes at."""

    root: str
    cell: Cone
    pro: Cone
    lead: Cone
    axes: frozenset[str]
    dtype: object

    @property
    def moved(self) -> tuple[Stmt, ...]:
        """Members the cut MOVES into the producer — cell + prologue.
        Leading stmts are copied, not moved (a shared one stays in both
        kernels), so they're excluded from the escape check."""
        return (*self.cell.members, *self.pro.members)


def try_split_demoted(loop_op: LoopOp, ctx: Context, *, graph: Graph, node_id: str, out_tensor: Tensor) -> Graph | None:
    """Build the producer(s)+consumer split fragment for a demoted matmul
    ``LoopOp``, or ``None`` when the body isn't a cleanly-cuttable demotion."""
    cut = _classify_cut(loop_op)
    if cut is None:
        return None
    leading, rows, prologue_level, outer_n, k_loop = cut
    k_name = k_loop.axis.name
    n_name = outer_n.axis.name
    row_names = [lp.axis.name for lp in rows]

    # --- classify each accum's multiply: Load operands stay, cones split out --
    top = tuple(k_loop.body)
    cell_def = {n: s for s in top for n in s.defines()}
    accums = [s for s in top if isinstance(s, Accum)]
    if not accums:
        return None
    roots: tuple[str, ...] | None = None
    muls: list[Assign] = []
    for acc in accums:
        mul = cell_def.get(acc.value)
        if not isinstance(mul, Assign) or mul.op.name != "multiply" or len(mul.args) != 2:
            return None
        muls.append(mul)
        cone_args: dict[str, None] = {}  # ordered de-dup (a squared cone appears twice)
        for a in mul.args:
            d = cell_def.get(a)
            if isinstance(d, Assign):
                cone_args[a] = None
            elif isinstance(d, Load):
                # A Load operand must be the matmul's own K-indexed read — a
                # K-invariant Load means this multiply isn't the A×B cell.
                if k_name not in {v for e in d.index for v in e.free_vars()}:
                    return None
            else:
                return None
        if not cone_args:
            return None  # pure cell (not demoted)
        if roots is None:
            roots = tuple(cone_args)
        elif set(roots) != set(cone_args):
            return None  # accums with different cones — no shared operand set to materialize
    assert roots is not None

    # --- backward-slice each cone over the cell / prologue / leading scopes --
    # Three chained Body.backward_cone calls, one per scope level: each
    # level's unresolved external reads seed the next. After the last level
    # only axis vars may remain unresolved.
    axis_names = {a.name for a in loop_op.axes}
    cones: list[_RootCone] = []
    for root in roots:
        cell_cone = k_loop.body.backward_cone((root,))
        if not cell_cone.members:
            return None  # cone root must live in the cell (a K-invariant operand isn't this pattern)
        if any(isinstance(m, Accum) for m in cell_cone.members):
            return None  # cone reads the matmul's own running accumulator — not cuttable
        pro_cone = Body(prologue_level).backward_cone(cell_cone.external_reads)
        lead_cone = Body(leading).backward_cone(pro_cone.external_reads)
        if lead_cone.external_reads - axis_names:
            return None  # name from an unmodeled scope — bail conservatively
        # Axes the materialization must cover: everything the moved stmts read
        # (prologue deps included — their row loops rebuild around the xn Write).
        axes = lead_cone.external_reads & axis_names
        if not axes <= set(row_names) | {k_name, n_name}:
            return None
        dtype = _cone_dtype((*cell_cone.loads, *pro_cone.loads, *lead_cone.loads), graph)
        if dtype is None:
            return None
        cones.append(_RootCone(root, cell_cone, pro_cone, lead_cone, axes, dtype))
    if sum(1 for c in cones if n_name in c.axes) > 1:
        return None  # two (…, K, N) buffers would re-do the matmul's own volume
    for i, a in enumerate(cones):
        for b in cones[i + 1 :]:
            if {id(m) for m in a.moved} & {id(m) for m in b.moved}:
                return None  # cones sharing a stmt would compute it in both producers

    # --- escape check: moved values must die at the multiplies ----------------
    if not loop_op.body.defs_die_at((m for c in cones for m in c.moved), roots=roots, allowed=muls):
        return None

    # --- build one producer per cone -------------------------------------------
    # The N-reading cone sorts last so it is always the "b" suffix; a single
    # cone keeps the plain "__xn" name.
    cones.sort(key=lambda c: n_name in c.axes)
    suffixes = ("",) if len(cones) == 1 else ("a", "b")
    producers: list[tuple[LoopOp, Tensor]] = []
    cone_loads: dict[int, Load] = {}  # id(cone root's def stmt) → replacement Load
    for c, sfx in zip(cones, suffixes, strict=True):
        xn_id = f"{node_id}__xn{sfx}"
        rows_used = [lp for lp in rows if lp.axis.name in c.axes]
        row_vars = tuple(Var(lp.axis.name) for lp in rows_used)
        reads_n = n_name in c.axes
        index = (*row_vars, Var(k_name), *((Var(n_name),) if reads_n else ()))
        inner: tuple[Stmt, ...] = (*c.cell.members, Write(output=xn_id, index=index, values=(c.root,)))
        if reads_n:
            # N innermost: the Write walks the buffer's last dim (coalesced) and
            # K lands second-to-last — the canonical B layout, even when the
            # original access was transposed [n, k].
            inner = (Loop(axis=outer_n.axis, body=Body(inner)),)
        level: tuple[Stmt, ...] = (*c.pro.members, Loop(axis=k_loop.axis, body=Body(inner)))
        for lp in reversed(rows_used):
            level = (Loop(axis=lp.axis, body=Body(level)),)
        producer = LoopOp(body=Body((*c.lead.members, *level)))
        producer.name = f"{loop_op.name}_xn{sfx}" if loop_op.name else ""
        shape = tuple(lp.axis.extent.as_static() for lp in rows_used) + (k_loop.axis.extent.as_static(),)
        if reads_n:
            shape += (outer_n.axis.extent.as_static(),)
        producers.append((producer, Tensor(xn_id, shape, c.dtype)))
        cone_loads[id(cell_def[c.root])] = Load(names=(c.root,), input=xn_id, index=index)

    # --- build the consumer ----------------------------------------------------
    moved_cell = {id(m) for c in cones for m in c.cell.members}
    new_top: list[Stmt] = []
    for s in top:
        repl = cone_loads.get(id(s))
        if repl is not None:
            new_top.append(repl)
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
    pro_used_all = {id(m) for c in cones for m in c.pro.members}
    level = tuple(new_outer_n if s is outer_n else s for s in prologue_level if id(s) not in pro_used_all or s is outer_n)
    for lp in reversed(rows):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    cons_id = f"{node_id}__mm"
    lead_used_all = {id(m) for c in cones for m in c.lead.members}
    kept_lead = tuple(s for s in leading if id(s) not in lead_used_all)
    consumer_op = LoopOp(body=Body((*kept_lead, *level)))
    consumer_op = _rename_write_output(consumer_op, old=node_id, new=cons_id)
    consumer_op.name = loop_op.name
    consumer_op = _drop_dangling_leads(consumer_op, kept_lead)

    return _assemble_fragment(graph, producers=tuple(producers), consumer_op=consumer_op, cons_id=cons_id, out_tensor=out_tensor)


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


def _cone_dtype(loads, graph: Graph):
    """The uniform dtype of every graph-resolvable Load in the cone's moved
    stmts — the dtype its ``xn`` buffer materializes at (value-preserving;
    identical to the multiply's other-operand dtype on every shape seen so
    far). ``None`` (bail) when a Load source is unresolvable or the leaf
    dtypes disagree."""
    dtypes = set()
    for ld in loads:
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
