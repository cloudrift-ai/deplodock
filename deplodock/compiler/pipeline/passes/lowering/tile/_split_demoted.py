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
per-shape cases: each multiply operand is independently a stageable plain ``Load`` (K in
one index dim — stays put), a K-FOLDED ``Load`` (K across several dims, the collapsed
reshape/transpose o_proj attn-out read — a degenerate cone whose only member is the Load,
its producer the contiguizing copy), or a
computed cone (becomes a producer kernel); every distinct cone materializes an ``xn``
intermediate over exactly the axes it reads —

**Segmented-K exception (no producer).** A K-folded ``Load`` whose innermost (contiguous)
index dim is ``expr % C`` — the o_proj attn-out / P@V-V transpose layout, where K is a
``(strided-outer × contiguous-inner extent C)`` nest — is NOT demoted (see
``_helpers.segmentable_k_extent``). Partition C-aligns the K split (``k_per_block == C``)
so the delinearization folds to a clean ``[K_o, …, K_i]`` read (the range-aware
``Expr.simplify``); the matmul then reaches the mma tier reading the folded operand from
gmem directly — the ``K_o`` outer segment's per-step base is the head stride (``seq_len *
C``), the ``K_i`` inner run is contiguous and ``ldmatrix``-stageable. So segmented-K
supersedes the contiguizing producer for these reads; only a *non*-segmentable fold (no
literal-modulus inner run) still materializes the copy.

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
(o_proj's collapsed attn-out — once a degenerate Load cone — now takes the segmented-K path
above: no producer, mma reads it directly.)

Cones compare by VALUE, not SSA name: fusion inlines a shared producer chain once per
consuming matmul (the gated-MLP norm feeds gate AND up as two structurally identical
chains), so the cell is value-numbered and roots in the same class share one ``xn``
materialization. A MULTI-accum K loop (gate+up sharing the reduce) additionally extracts
each accum's matmul into its own clean single-matmul gemm producer — the mma cell gate
admits exactly one matmul per K loop, so the fused pair could never reach the warp tier —
writing the accumulator at ``mm_i[rows…, n]`` in f32 (its own precision), and the consumer
becomes the pointwise combine: each K loop replaced by ``Load``\\ s that re-read the
``mm_i`` buffers under the accums' SSA names, the epilogue (SiLU·up) untouched. The
single-accum consumer keeps its matmul inline as before.

The checks here are the cut's own WELL-FORMEDNESS conditions, not a profitability gate:
this module deliberately does not predict whether the clean gemm will reach the warp tier
(an earlier version simulated ``is_atom_eligible`` on the rebuilt consumer and immediately
drifted from what the cell tagger actually accepts). Whether the split pays is the search's
question — the tuner measures both branches, greedy deploys the structural option only when
the trained prior prices its kernel set cheaper (``policy/greedy._pick_structural``; never
cold), and a lowering failure on either side must surface as a rejection.
Conservative bails (return ``None``, never raise) keep the fused path the only
outcome for any shape the cut doesn't fully understand: multiple K loops, no computed
or K-folded operand, a K-invariant Load operand, distinct-class cones sharing stmts,
cone values escaping past the multiply, mixed-dtype cell leaves, more
than one cone reading the output N axis (two ``(…, K, N)`` buffers would re-do the
matmul's own volume — the materialization that defeats the split), or — multi-accum only —
a cell stmt claimed by no gemm / an operand Load doubling as a cone member, or a symbolic K
extent. Symbolic N / ROW extents ARE admitted: the cut tiles neither, so the ``xn`` /
``mm_i`` buffers just carry the symbolic Dim (allocated from the runtime extent) and the
clean gemm reaches the warp tier on a symbolic N as a masked tile — un-fusing the
rotary QK^T (symbolic N). Symbolic K is ALSO admitted now that the warp tier accepts a
masked reduce (the K is hint-tiled and the partial final slab is zero-filled in smem):
the SDPA P@V demotion un-fuses into a softmax-normalizing ``xn`` producer + a clean
symbolic-K consumer that reaches the tensor-core tier (matching its static twin), instead
of staying fused-scalar. A symbolic dimension VARIABLE in a cone's index
arithmetic (the o_proj collapsed attn-out, whose head stride is ``seq_len * 128``) is a
legitimate runtime read, not an unmodeled scope — see ``dim_names`` below.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from string import ascii_lowercase
from typing import TYPE_CHECKING

from deplodock.compiler import dtype as dtype_mod
from deplodock.compiler.dim import DEFAULT_SEQ_HINT, Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Stmt, Write
from deplodock.compiler.ir.stmt.body import Cone
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce, segmentable_k_extent

if TYPE_CHECKING:
    from deplodock.compiler.context import Context

# Element-count multiple a symbolic N (inner) extent is padded up to so the
# materialized B operand's above-inner gmem stride stays 16 B-aligned for TMA
# (the ``cuTensorMapEncodeTiled`` requirement ``050_use_tma`` gates on). 64
# elements is ≥ 128 B for any ≥ 2-byte dtype (fp16 → 128 B, fp32 → 256 B),
# clearing the descriptor's recommended box alignment too, while wasting at
# most 63 columns of scratch. The matching aligned-stride acceptance lives in
# ``050_use_tma._inner_stride_aligned``.
_TMA_INNER_PAD = 64


def _pad_inner_for_tma(extent: Dim) -> Dim:
    """Round a *symbolic* inner (N) extent up to a multiple of ``_TMA_INNER_PAD``
    so the materialized buffer's above-inner gmem stride is 16 B-aligned and the
    operand becomes TMA-eligible. Static extents pass through unchanged (a fixed
    inner extent already has a compile-checkable, aligned stride). The composite
    ``round_up`` Dim carries an explicit hint (the base hint rounded up the same
    way) because arithmetic-result Dims otherwise drop their hint, which
    ``050_use_tma._shape_for_tma`` needs to size the descriptor box."""
    if extent.is_static:
        return extent
    base_hint = extent.hint if extent.hint is not None else DEFAULT_SEQ_HINT
    pad_hint = -(-base_hint // _TMA_INNER_PAD) * _TMA_INNER_PAD
    return Dim((extent.ceil_div(_TMA_INNER_PAD) * _TMA_INNER_PAD).expr, hint=pad_hint)


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
    per_acc: list[tuple[Accum, Assign, tuple[str, ...]]] = []
    for acc in accums:
        mul = cell_def.get(acc.value)
        if not isinstance(mul, Assign) or mul.op.name != "multiply" or len(mul.args) != 2:
            return None
        cone_args: dict[str, None] = {}  # ordered de-dup (a squared cone appears twice)
        for a in mul.args:
            d = cell_def.get(a)
            if isinstance(d, Assign):
                cone_args[a] = None
            elif isinstance(d, Load):
                # A Load operand must be the matmul's own K-indexed read — a
                # K-invariant Load means this multiply isn't the A×B cell.
                k_dims = [i for i, e in enumerate(d.index) if k_name in e.free_vars()]
                if not k_dims:
                    return None
                # K folded across index dims (a collapsed reshape/transpose
                # layout — the o_proj attn-out read): ``020_stage_inputs`` can
                # only stage a single-K-dim slab, so the warp tier is
                # structurally unreachable. The Load is a DEGENERATE cone (its
                # only member is itself): the producer is the contiguizing
                # copy, materializing the operand as a plain ``xn[rows…, K]``.
                # A single-K-dim Load is already stageable and stays put.
                #
                # EXCEPT a *segmentable* fold (segmented-K): when K's innermost
                # index dim is ``expr % C`` the read is a (strided-outer ×
                # contiguous-inner) nest — a C-aligned K split in partition folds
                # the delinearization to a clean single-K-dim read and the matmul
                # reaches the mma tier reading gmem directly. Don't demote it;
                # let it flow to partition (no transpose producer needed).
                if len(k_dims) > 1 and segmentable_k_extent(d, k_name) is None:
                    cone_args[a] = None
            else:
                return None
        if not cone_args:
            return None  # pure cell (not demoted)
        per_acc.append((acc, mul, tuple(cone_args)))
    muls = [m for _, m, _ in per_acc]

    # Value-number the cell so SSA-duplicated cones count as ONE: fusion
    # inlines a shared producer chain once per consuming matmul (the gated-MLP
    # norm feeds gate AND up as two structurally identical chains), and roots
    # in the same value class share a single xn materialization. An external
    # name is its own class (same SSA name = same value).
    vn: dict[str, object] = {}
    for s in top:
        if isinstance(s, Load) and len(s.names) == 1:
            vn[s.names[0]] = ("load", s.input, tuple(e.pretty() for e in s.index))
        elif isinstance(s, Assign):
            vn[s.name] = (s.op.name, tuple(vn.get(a, a) for a in s.args))
    rep_of: dict[object, str] = {}  # value class → representative root, first in body order
    for _, _, rs in per_acc:
        for r in rs:
            rep_of.setdefault(vn[r], r)
    roots = tuple(dict.fromkeys(r for _, _, rs in per_acc for r in rs))

    # --- backward-slice each cone over the cell / prologue / leading scopes --
    # Three chained Body.backward_cone calls, one per scope level: each
    # level's unresolved external reads seed the next. After the last level
    # only axis vars may remain unresolved. Every root is sliced (the escape
    # check and the consumer rebuild need each duplicate chain's stmts); only
    # the class representatives materialize producers.
    axis_names = {a.name for a in loop_op.axes}
    # Symbolic dimension vars (e.g. ``seq_len``) are legitimate external reads,
    # not an unmodeled scope: a symbolic-extent buffer's index arithmetic
    # references the Dim in its strides (the collapsed-reshape o_proj attn-out
    # read, whose head stride is ``seq_len * 128``), and the backend resolves it
    # from the input shapes at launch like any runtime kernel arg. They are NOT
    # added to ``axis_names`` — they're scalar runtime quantities, not loop axes
    # to tile/materialize over; the producers re-emit the symbolic row loops and
    # the index arithmetic flows through unchanged.
    dim_names = {v for a in loop_op.axes for v in a.extent.expr.free_vars()}
    cone_of: dict[str, _RootCone] = {}
    for root in roots:
        cell_cone = k_loop.body.backward_cone((root,))
        if not cell_cone.members:
            return None  # cone root must live in the cell (a K-invariant operand isn't this pattern)
        if any(isinstance(m, Accum) for m in cell_cone.members):
            return None  # cone reads the matmul's own running accumulator — not cuttable
        pro_cone = Body(prologue_level).backward_cone(cell_cone.external_reads)
        lead_cone = Body(leading).backward_cone(pro_cone.external_reads)
        if lead_cone.external_reads - axis_names - dim_names:
            return None  # name from an unmodeled scope (axis vars + symbolic Dims allowed)
        # Axes the materialization must cover: everything the moved stmts read
        # (prologue deps included — their row loops rebuild around the xn Write).
        axes = lead_cone.external_reads & axis_names
        if not axes <= set(row_names) | {k_name, n_name}:
            return None
        # The xn materializes at the matmul's OPERAND dtype — the dtype the
        # consuming multiply reads this operand as, i.e. the *other* multiply
        # operand's dtype (a matmul's two operands share an element type, and the
        # mma tier requires it). For the SDPA P@V the cone is the fp32 softmax
        # weights but the other operand (V) is fp16, so xn downcasts to fp16 —
        # matching eager fp16 attention AND halving the staged slab so the
        # masked-K mma tile fits the smem budget (an fp32 xn over-allocates 2× and
        # is misread by ``ldmatrix.b16``). Falls back to the CELL leaves' own
        # dtype when the other operand is itself a computed cone (rotary QK^T:
        # both operands already share the cone dtype) — prologue/lead loads only
        # feed row stats (an f32 mean-count beside an f16 chain) and need only
        # resolve, not match.
        dtype = _matmul_operand_dtype(root, per_acc, cell_def, graph)
        if dtype is None:
            dtype = _cone_dtype(cell_cone.loads, graph)
        if dtype is None:
            return None
        if any(graph.nodes.get(ld.input) is None for ld in (*pro_cone.loads, *lead_cone.loads)):
            return None  # prologue/lead load from a buffer the graph doesn't know
        cone_of[root] = _RootCone(root, cell_cone, pro_cone, lead_cone, axes, dtype)
    cones = [cone_of[r] for r in rep_of.values()]
    if sum(1 for c in cones if n_name in c.axes) > 1:
        return None  # two (…, K, N) buffers would re-do the matmul's own volume
    for i, a in enumerate(cones):
        for b in cones[i + 1 :]:
            if {id(m) for m in a.moved} & {id(m) for m in b.moved}:
                return None  # distinct-class cones sharing a stmt would compute it in both producers

    # --- escape check: moved values must die at the multiplies ----------------
    if not loop_op.body.defs_die_at((m for r in roots for m in cone_of[r].moved), roots=roots, allowed=muls):
        return None

    # A multi-accum cell (the gated-MLP gate+up) cannot stay one kernel and
    # reach the warp tier (the mma cell gate admits exactly one matmul per K
    # loop), so the split also extracts each accum's matmul into its own clean
    # gemm producer and rebuilds the consumer as the pointwise combine. Every
    # cell stmt must then have a home in some gemm: bail on strays, and on an
    # operand Load doubling as a cone member (it would be both kept and moved).
    moved_cell = {id(m) for r in roots for m in cone_of[r].cell.members}
    if len(accums) > 1:
        claimed = moved_cell | {id(m) for m in muls} | {id(a) for a in accums}
        for _, mul, rs in per_acc:
            for a in mul.args:
                if a in rs:
                    continue  # a cone root (incl. a K-folded Load) — moved and replaced
                d = cell_def[a]
                if isinstance(d, Load):
                    if id(d) in moved_cell:
                        return None
                    claimed.add(id(d))
        if any(id(s) not in claimed for s in top):
            return None

    # --- build one producer per cone class --------------------------------------
    # The N-reading cone sorts last so it is always the "b" suffix; a single
    # cone keeps the plain "__xn" name. Every root of the class — duplicates
    # included — re-reads the one shared buffer under its own SSA name.
    cones.sort(key=lambda c: n_name in c.axes)
    suffixes = ("",) if len(cones) == 1 else tuple(ascii_lowercase[: len(cones)])
    if len(suffixes) != len(cones):
        return None  # more cone classes than suffix letters — not a real shape
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
        # Every extent passes through as a Dim — rows, K, and N alike allocate
        # from the runtime extent when symbolic (``_classify_cut`` admits all).
        # A row cone (the A operand) puts the reduce K innermost; when K is
        # symbolic (the masked-K P@V probs cone) pad it like the N cone so the
        # above-inner stride stays 16 B-aligned and the operand is TMA-eligible.
        # The reduce overhang here is NOT self-zeroed (this buffer's TMA globalDim
        # is the padded extent), but it doesn't need to be: the matmul's *other*
        # operand — the middle-K B cone (V), allocated at the real ``seq_len`` — is
        # TMA-OOB-zero-filled past ``seq_len``, so every overhang product is
        # ``A_overhang × 0``. The scratch slab is zero-initialised and only ever
        # holds finite kernel outputs, so ``A_overhang`` is finite (never NaN/Inf)
        # and the product is a true 0 — the masked-K reduction stays exact. (On the
        # SYNC fallback the per-value ternary zeroes it directly.)
        k_inner = _pad_inner_for_tma(k_loop.axis.extent) if not reads_n else k_loop.axis.extent
        shape = tuple(lp.axis.extent for lp in rows_used) + (k_inner,)
        if reads_n:
            # An N-reading cone materializes the canonical B operand ``xnb[…, K, N]``.
            # When N is symbolic (the rotary QK^T's ``seq_len`` key cone), the raw
            # ``seq_len`` inner extent gives the K dim a gmem stride of ``seq_len``
            # elements — runtime values like 31 / 700 aren't 16 B-aligned, so
            # ``cuTensorMapEncodeTiled`` rejects the descriptor and ``050_use_tma``
            # declines TMA (and warp-spec with it). Pad the inner N up to a multiple
            # of ``_TMA_INNER_PAD`` so the above-inner stride is always 16 B-aligned
            # while the buffer stays runtime-sized (correct at any seq, unlike a
            # fixed static width that would overflow past the hint). The extra
            # ``[seq_len, round_up)`` columns are never stored — the consumer's
            # per-element boundary guard masks ``n >= seq_len`` — so the pad is
            # value-neutral; it only enlarges the scratch alloc by ``< _TMA_INNER_PAD``
            # columns. The garbage columns DO feed the mma, but only into
            # store-masked output positions, so they can't contaminate a live score.
            shape += (_pad_inner_for_tma(outer_n.axis.extent),)
        producers.append((producer, Tensor(xn_id, shape, c.dtype)))
        for root in roots:
            if vn[root] == vn[c.root]:
                cone_loads[id(cell_def[root])] = Load(names=(root,), input=xn_id, index=index)

    # --- multi-accum: one clean gemm producer per accum -------------------------
    # Each gemm materializes its accumulator at (rows…, N) in f32 — the
    # accumulator's own precision, value-preserving — and the consumer re-reads
    # it under the accum's SSA name, so the epilogue is untouched.
    acc_loads: list[Load] = []
    if len(accums) > 1:
        mm_index = (*(Var(lp.axis.name) for lp in rows), Var(n_name))
        # Every Dim passes through (symbolic rows / N allocate at runtime).
        mm_shape = tuple(lp.axis.extent for lp in rows) + (outer_n.axis.extent,)
        for i, (acc, mul, _) in enumerate(per_acc):
            mm_id = f"{node_id}__mm{i}"
            cell: list[Stmt] = []
            for a in mul.args:
                d = cell_def[a]
                repl = cone_loads.get(id(d))  # any cone root — Assign or K-folded Load
                cell.append(repl if repl is not None else d)
            inner_mm: tuple[Stmt, ...] = (
                Loop(axis=k_loop.axis, body=Body((*cell, mul, acc)), unroll=k_loop.unroll),
                Write(output=mm_id, index=mm_index, values=(acc.name,)),
            )
            level_mm: tuple[Stmt, ...] = (Loop(axis=outer_n.axis, body=Body(inner_mm), unroll=outer_n.unroll),)
            for lp in reversed(rows):
                level_mm = (Loop(axis=lp.axis, body=Body(level_mm)),)
            gemm = LoopOp(body=Body(level_mm))
            gemm.name = f"{loop_op.name}_mm{i}" if loop_op.name else ""
            producers.append((gemm, Tensor(mm_id, mm_shape, dtype_mod.get("f32"))))
            acc_loads.append(Load(names=(acc.name,), input=mm_id, index=mm_index))

    # --- build the consumer ----------------------------------------------------
    if len(accums) == 1:
        new_top: list[Stmt] = []
        for s in top:
            repl = cone_loads.get(id(s))
            if repl is not None:
                new_top.append(repl)
            elif id(s) in moved_cell:
                continue
            else:
                new_top.append(s)
        cell_repl: tuple[Stmt, ...] = (Loop(axis=k_loop.axis, body=Body(tuple(new_top)), unroll=k_loop.unroll),)
    else:
        cell_repl = tuple(acc_loads)
    cons_inner: list[Stmt] = []
    for s in outer_n.body:
        if s is k_loop:
            cons_inner.extend(cell_repl)
        else:
            cons_inner.append(s)
    new_outer_n = Loop(axis=outer_n.axis, body=Body(tuple(cons_inner)), unroll=outer_n.unroll)
    pro_used_all = {id(m) for r in roots for m in cone_of[r].pro.members}
    level = tuple(new_outer_n if s is outer_n else s for s in prologue_level if id(s) not in pro_used_all or s is outer_n)
    for lp in reversed(rows):
        level = (Loop(axis=lp.axis, body=Body(level)),)
    cons_id = f"{node_id}__mm"
    lead_used_all = {id(m) for r in roots for m in cone_of[r].lead.members}
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
    # Symbolic N / ROW / K extents are all admitted. The cut tiles none of these
    # axes — the producers and consumer re-emit them verbatim and every ``xn`` /
    # ``mm_i`` buffer dim carries the symbolic Dim (allocated from the runtime
    # extent; nothing here reasons about the numeric volume). A symbolic N is
    # offered because the clean gemm reaches the WARP tier as a masked tile; a
    # symbolic K is offered for the same reason now that the warp tier admits a
    # MASKED reduce (``mma.sync`` over a hint-tiled K with the partial final slab
    # zero-filled in smem — see ``010_partition_loops`` + ``_stage_expand``). So
    # the SDPA P@V demotion (softmax prologue + a symbolic-K matmul) un-fuses
    # into a softmax-normalizing ``xn`` producer + a clean symbolic-K consumer
    # that reaches the tensor-core tier, matching its static twin — the search
    # prices the split against the fused-scalar form.
    return tuple(leading), rows, prologue_level, outer_n, k_loop


def _contains_matmul_reduce_loop(stmt: Stmt) -> bool:
    if isinstance(stmt, Loop) and stmt.is_reduce and is_matmul_reduce(stmt):
        return True
    return any(_contains_matmul_reduce_loop(c) for body in stmt.nested() for c in body)


def _matmul_operand_dtype(root: str, per_acc, cell_def, graph: Graph):
    """The dtype the consuming matmul reads this cone operand AS — the *other*
    multiply operand's gmem dtype. A matmul's two operands share an element type
    (and the warp/mma tier requires it), so the materialized ``xn`` must match
    the operand it multiplies against, NOT the (possibly higher-precision)
    softmax/compute dtype the cone is calculated in. For the SDPA P@V this turns
    the fp32 softmax cone into an fp16 ``xn`` (V is fp16). ``None`` when the other
    operand is itself a computed cone (an Assign — both sides already share the
    cone dtype, e.g. rotary QK^T), a squared cone (``a*a`` — no distinct other
    operand), or the source is unresolvable; the caller then falls back to the
    cone leaf dtype."""
    for _acc, mul, rs in per_acc:
        if root not in rs:
            continue
        others = [a for a in mul.args if a != root]
        if not others:
            return None  # squared cone (a*a) — no distinct other operand
        d = cell_def.get(others[0])
        if isinstance(d, Load):
            node = graph.nodes.get(d.input)
            return node.output.dtype if node is not None else None
        return None  # other operand is a computed cone — share the cone dtype
    return None


def _cone_dtype(loads, graph: Graph):
    """The uniform dtype of every graph-resolvable Load in the cone's CELL
    stmts — the fallback dtype its ``xn`` buffer materializes at when the
    matmul-operand dtype can't be read (``_matmul_operand_dtype``; the two
    rules coincide on most shapes — the SDPA P@V fp32-softmax cone is the
    exception). Prologue/lead loads don't vote: they feed row stats (an f32
    mean-count scalar must not block an f16 norm chain). ``None`` (bail) when
    a Load source is unresolvable or the leaf dtypes disagree."""
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
