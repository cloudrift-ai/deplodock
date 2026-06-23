"""Per-kernel atom *eligibility* — the planner-side gate for each matmul atom.

The :class:`~deplodock.compiler.ir.tile.ir.Atom` spec (cell shape, per-operand
dtypes, group size) + the ``ATOM_REGISTRY`` live in ``ir/tile/ir.py`` (next to
the other tile-IR types, and carried directly on the ``Mma`` op). They are
re-exported here so existing ``from ...tile._atom import ATOM_REGISTRY`` call
sites keep working.

This module owns only the part that *can't* live in the IR layer: the per-kind
**eligibility** predicate (does a given ``LoopOp`` admit this atom on this
device?). It depends on the loop body / graph / context and on
``is_matmul_reduce`` — pipeline-layer concerns — so it stays in the planner.
:func:`is_atom_eligible` dispatches via the ``_ELIGIBILITY`` map; adding a kind
(future: wgmma, NVFP4) registers an ``Atom`` in ``ir/tile/ir.py`` and a
predicate here.

Prefixed ``_`` so the pipeline rule loader (``_load_rules``) skips it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, Atom
from deplodock.compiler.pipeline.passes.lowering.kernel._helpers import is_matmul_reduce, segmentable_k_extent

if TYPE_CHECKING:
    from deplodock.compiler.context import Context
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.ir.loop import LoopOp

# Back-compat re-exports (canonical definitions live in ``ir/tile/ir.py``).
__all__ = [
    "ATOM_REGISTRY",
    "Atom",
    "is_atom_eligible",
]


def classify_matmul_operands(loads, k_name: str, *, out_index=None):
    """Identify the A (M×K) / B (K×N) operand ``Load``s of a canonical matmul
    cell by where the reduce axis ``k_name`` sits in each load's index.

    Primary tests (the historical rule): K in the LAST index dim (and not the
    first) ⇒ A; K in the FIRST dim (and not the last) ⇒ B. Fallback for K in a
    *middle* dim — e.g. the SDPA cone-split's 4-D V slab ``(0, k, 0, n)``,
    where K is dim 1 of 4: a load whose single K dim sits AFTER every other
    var-carrying dim ⇒ A, BEFORE every one ⇒ B.

    **Transposed-B (e.g. Q @ K^T):** when BOTH operands carry K in their last
    dim the primary test tags both as A and leaves B empty. This is the native
    ``mma.row.col`` layout — B stored N×K *is* the col-major K×N operand the
    instruction wants — so with the output coordinates supplied via ``out_index``
    we disambiguate: the operand sharing the M (row) output var is A, the one
    sharing N (col) is B (loaded ``ldmatrix`` without ``.trans``; see
    ``kernel/005_lower_atom_tile``). Without ``out_index`` it stays unclassified
    (returns ``None`` for B), as before. Other ambiguous layouts (K folded
    across dims, K-only indices) also classify neither side.

    This is the ONE A/B layout decision: ``011_lower_atom_cell._classify_ab``
    tags cells with it, and the ``is_atom_eligible`` mma gate calls the same
    function so the gate mirrors the tagger by construction — a cell the
    tagger can't classify is never offered the mma tier (an untagged
    ``AtomTile`` would survive to render and crash there).

    Returns ``(a_load, b_load)``, either possibly ``None``.
    """
    a_load = None
    b_load = None
    k_last: list = []  # transposed-B candidates (K in the LAST dim, like A)
    for ld in loads:
        if not ld.index:
            continue
        k_in_first = k_name in ld.index[0].free_vars()
        k_in_last = k_name in ld.index[-1].free_vars()
        if k_in_last and not k_in_first:
            if a_load is None:
                a_load = ld
            k_last.append(ld)
            continue
        if k_in_first and not k_in_last:
            b_load = ld
            continue
        k_dims = [d for d, e in enumerate(ld.index) if k_name in e.free_vars()]
        var_dims = [d for d, e in enumerate(ld.index) if d not in k_dims and e.free_vars()]
        if len(k_dims) > 1:
            # Folded K across >1 dim. Segmentable (innermost ``% C``, the o_proj
            # attn-out read) ⇒ A: a C-aligned K split in partition folds it to a
            # clean single-K-dim read, so the consumer reads gmem directly with
            # no transpose producer. Other folds stay unclassified (no mma).
            if segmentable_k_extent(ld, k_name) is not None and a_load is None:
                a_load = ld
            continue
        if len(k_dims) != 1 or not var_dims:
            continue
        after_k = [d for d in var_dims if d > k_dims[0]]
        if not after_k:
            # K is the innermost var-carrying dim (every other var dim precedes
            # it) ⇒ A.
            a_load = ld
        elif len(after_k) == 1:
            # Exactly ONE var dim follows K (the N output); any var dims BEFORE K
            # are leading BATCH axes (shared by both operands — the batched SDPA
            # P@V split-consumer ``xnb[head, k, n]``), so K immediately precedes N
            # ⇒ B. The old ``all(k < d)`` test mis-rejected this whenever a batch
            # dim sat before K, leaving the consumer scalar.
            b_load = ld
    # Transposed-B: two K-in-last operands and no canonical B. Recover A/B from
    # the output coordinates (the M / N output vars are the last two var-bearing
    # dims — the same convention as ``classify_fragment_epilogue`` / the
    # ``_boundary_guards`` M/N split).
    if b_load is None and len(k_last) == 2 and out_index is not None:
        a2, b2 = _classify_transposed_b(k_last, k_name, out_index)
        if a2 is not None and b2 is not None:
            return a2, b2
    return a_load, b_load


def _classify_transposed_b(loads, k_name: str, out_index):
    """Split two K-in-last operands into (A, B) by which output coordinate each
    shares: A carries the M (row) output var, B carries N (col)."""
    out_vars = [e for e in out_index if e.free_vars()]
    if len(out_vars) < 2:
        return None, None
    m_vars = set(out_vars[-2].free_vars()) - {k_name}
    n_vars = set(out_vars[-1].free_vars()) - {k_name}
    a_load = None
    b_load = None
    for ld in loads:
        non_k: set = set()
        for d, e in enumerate(ld.index):
            if d == len(ld.index) - 1 and k_name in e.free_vars():
                continue  # the trailing K dim
            non_k |= set(e.free_vars())
        non_k.discard(k_name)
        if (non_k & m_vars) and not (non_k & n_vars):
            a_load = ld
        elif (non_k & n_vars) and not (non_k & m_vars):
            b_load = ld
    return a_load, b_load


def _mma_eligible_factory(atom: Atom, *, min_cc: tuple[int, int] = (8, 0)) -> Callable[[LoopOp, Context, Graph], bool]:
    """Build an mma.sync eligibility predicate for ``atom`` — its cell shape +
    A-operand dtype come straight off the spec; ``min_cc`` is the device gate.

    The predicate checks:

    - At least one matmul-reduce in the body (``is_matmul_reduce``).
    - Every K-indexed Load resolves to the atom's operand dtype via
      ``graph.nodes[buf].output.dtype``.
    - ``ctx.compute_capability >= min_cc``.
    - K extent divisible by the cell's K dim.
    - Each output extent divisible by the corresponding cell dim (M / N).
    """
    cell_m, cell_n, cell_k = atom.shape
    operand_dtype = atom.operand_dtype("a")

    def predicate(loop_op: LoopOp, ctx: Context, graph: Graph) -> bool:
        from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, StridedLoop, Write  # noqa: PLC0415

        if ctx.compute_capability < min_cc:
            return False
        matmul_reduces = [lp for lp in loop_op.body.iter_of_type(Loop, StridedLoop) if lp.is_reduce and is_matmul_reduce(lp)]
        if not matmul_reduces:
            return False
        # Output coordinates for the transposed-B (Q @ K^T) A/B disambiguation:
        # the matmul output Write (≥2 var-bearing index dims). Threaded into the
        # shared classifier so the gate mirrors the tagger on this layout too.
        out_writes = [w for w in loop_op.body.iter_of_type(Write) if sum(1 for e in w.index if e.free_vars()) >= 2]
        out_index = out_writes[0].index if out_writes else None
        # Buffers produced *inside* this fused kernel (e.g. an attention output
        # feeding the o_proj matmul). ``ldmatrix`` is smem→register only and
        # ``020_stage_inputs`` stages external gmem inputs, not register-resident
        # intermediates — so a matmul whose operand is produced here can't have
        # that operand staged, and the mma path would crash in ``kernel/005``.
        # Gate it off so the scalar register-tile path picks it up.
        produced = {w.output for w in loop_op.body.iter_of_type(Write)}
        accum_names: set[str] = set()
        for k_loop in matmul_reduces:
            K_name = k_loop.axis.name
            if k_loop.axis.extent.is_static and k_loop.axis.extent.as_static() % cell_k != 0:
                return False
            for load in k_loop.body.iter_of_type(Load):
                k_dims = [d for d, e in enumerate(load.index) if K_name in e.free_vars()]
                if not k_dims:
                    continue
                # The mma path needs both operands staged for ``ldmatrix``.
                # ``020_stage_inputs._classify`` only stages a load whose cache
                # var (the K axis here) lands in a *single* index dim. A
                # collapsed-reshape operand (attention output → o_proj via
                # ``[(a/128)%16, …, a%128]`` — K split across two dims) is
                # normally rejected — EXCEPT a *segmentable* fold (innermost
                # ``% C``) whose contiguous extent ``C`` is a multiple of the
                # atom's K (so partition can C-align the split: ``k_per_block ==
                # C``). Post-split the staged read is single-K-dim (the
                # delinearization folds away), so the stager serves it. Gating on
                # ``C % cell_k`` keeps eligibility in lock-step with partition's
                # C-alignment — never offer mma on a read partition leaves
                # delinearized. A fused in-kernel intermediate still can't be
                # staged (``ldmatrix`` is smem→reg only).
                seg_c = segmentable_k_extent(load, K_name)
                segmentable = seg_c is not None and seg_c % cell_k == 0
                if (len(k_dims) > 1 and not segmentable) or load.input in produced:
                    return False
                node = graph.nodes.get(load.input)
                if node is None or node.output.dtype != operand_dtype:
                    return False
            # Body purity: ``kernel/005_lower_atom_tile`` only handles the
            # canonical ``[Load a, Load b, Assign(multiply, a, b), Accum]``
            # shape — extra Assigns (e.g. constant-folded pre-scaling like
            # ``a * 0.1`` from torch's per-input scale) would be silently
            # dropped by the mma.sync emit. Until the MMA emit handles
            # in-cell scaling, gate those kernels off so the scalar path
            # picks them up correctly.
            top_level = list(k_loop.body)
            n_loads = sum(1 for s in top_level if isinstance(s, Load))
            n_assigns = sum(1 for s in top_level if isinstance(s, Assign))
            n_accums = sum(1 for s in top_level if isinstance(s, Accum))
            # Pure matmul cell: 2 Loads, 1 Assign (the multiply), 1 Accum.
            if not (n_loads == 2 and n_assigns == 1 and n_accums == 1):
                return False
            # Verify the Assign feeds the Accum and reads only the two Loads.
            loads = [s for s in top_level if isinstance(s, Load)]
            assigns = [s for s in top_level if isinstance(s, Assign)]
            accums = [s for s in top_level if isinstance(s, Accum)]
            multiply = assigns[0]
            accum = accums[0]
            load_names = {ld.names[0] for ld in loads if ld.names}
            if set(multiply.args) != load_names:
                return False
            if accum.value != multiply.name:
                return False
            # The cell must be a semiring contraction — the product distributes
            # over the Accum's reduce (the matmul ``(×, +)``). This matches the
            # check the lowering it feeds enforces (``011_lower_atom_cell`` /
            # ``_split_demoted``: ``mul.op.distributes_over(acc.op)``), so the
            # eligibility gate and the tagger agree on what reaches the mma tier.
            if not multiply.op.distributes_over(accum.op):
                return False
            # Operand layout: the gate and the tagger share ONE classifier
            # (:func:`classify_matmul_operands`), so a cell the tagger can't
            # recover A/B for is never offered the mma tier (an untagged
            # ``AtomTile`` would survive to render unconsumed and crash); it
            # falls to the scalar register-tile path instead. A transposed-B
            # Q @ K^T (both operands K-in-last) IS recovered here via the output
            # coordinates (``out_index``) — the native ``mma.row.col`` layout,
            # B loaded ``ldmatrix`` without ``.trans`` in ``kernel/005``.
            a_ld, b_ld = classify_matmul_operands(loads, K_name, out_index=out_index)
            if a_ld is None or b_ld is None:
                return False
            accum_names.add(accum.name)
        # Post-reduce epilogue: ``kernel/005_lower_atom_tile`` stores the mma
        # accumulator *fragment* straight to the output, so scalar epilogue
        # Assigns have no accumulator SSA name to read (each lane holds 4
        # elements of the C tile, not one ``acc0``). The fragment store can
        # fold any PURE POINTWISE epilogue — each fragment element owns a
        # known (row, col) of the output, so ``RegStore`` evaluates the chain
        # per element in f32, loading leaf operands at the element's own
        # coordinates (the CUTLASS epilogue-visitor pattern). Eligibility is
        # checked in the negative — :func:`classify_fragment_epilogue` walks
        # the backward slice from the Write to the accumulator and reports the
        # first ineligible operation / dependency; anything blocked gates to
        # the scalar register-tile path, which threads the scalar accumulator
        # through arbitrary epilogues correctly.
        _slice, blocker = classify_fragment_epilogue(
            loop_op.body,
            accum_names,
            produced=produced,
            leaf_dtype=lambda buf: graph.nodes[buf].output.dtype.name if buf in graph.nodes else None,
        )
        if blocker is not None:
            return False
        # Each output free-axis extent must divide cleanly. The body's outer
        # free Loops contribute the M (outer) / N (inner) extents the planner
        # will partition into output cells; gate the outermost-to-inner
        # static free axes against (cell_m, cell_n) in their order of
        # appearance.
        free_extents = [
            lp.axis.extent.as_static()
            for lp in loop_op.body.iter_of_type(Loop, StridedLoop)
            if not lp.is_reduce and lp.axis.extent.is_static
        ]
        cell_dims = (cell_m, cell_n)
        for ext, cell in zip(reversed(free_extents), reversed(cell_dims), strict=False):
            if ext > 1 and ext % cell != 0:
                return False
        return True

    return predicate


# Leaf-load dtypes the fragment epilogue can convert to f32 at render time
# (``RegStore._epilogue_lines`` — ``__half2float`` / ``__bfloat162float`` /
# identity). Anything else blocks the fold.
_EPILOGUE_LEAF_DTYPES = frozenset({"f16", "bf16", "f32"})


@dataclass(frozen=True)
class EpilogueSlice:
    """The foldable post-reduce epilogue of a matmul body: the backward slice
    from the (single) ``Write`` to the accumulator. ``assigns`` are in body
    (SSA/topological) order; ``loads`` are the leaf operand Loads with
    ``load_roles`` classifying each load's index dims as ``"m"`` / ``"n"``
    (struct-equal to the Write's M / N index expr — the fragment element's row
    / col offset applies there at that buffer's own stride) or ``"fixed"``
    (uniform across the cell: literals, batch/grid vars, broadcasts)."""

    acc: str
    assigns: tuple
    loads: tuple
    load_roles: tuple[tuple[str, ...], ...]
    write: object
    # Coord-predicated Selects folded into the store (the causal attention mask):
    # the raw ``Select`` stmts. ``kernel/005_lower_atom_tile`` turns each into a
    # per-element ternary, rewriting its predicate's M/N coordinate vars to the
    # fragment element's own (row, col).
    selects: tuple = ()


def classify_fragment_epilogue(
    body, accum_names: set[str], *, produced: set[str], leaf_dtype, outer_loads: dict[str, object] | None = None
) -> tuple[EpilogueSlice | None, str | None]:
    """Classify the post-reduce epilogue for the mma fragment-store fold.

    Returns ``(slice, None)`` when a foldable epilogue exists, ``(None, None)``
    when there is no epilogue at all (the Write stores the accumulator
    directly), and ``(None, reason)`` when the epilogue contains an ineligible
    operation or dependency.

    The rule is written in the NEGATIVE: the fold can render any pure
    pointwise SSA chain from the accumulator to the Write whose leaf Loads are
    addressable at a fragment element's own (row, col) — so instead of
    pattern-matching admissible shapes, this walks the slice and reports the
    first thing the fold fundamentally cannot do:

    - the accumulator is consumed *inside* a reduce loop (a mid-reduction use,
      e.g. online softmax rescale — needs a scheduled phase, not a store fold);
    - the slice depends on more than one accumulator (multi-fragment epilogue
      — the doorway to the gated-MLP combine, not wired yet);
    - the slice feeds zero / multiple Writes, a vector Write, or the
      accumulator is also written directly alongside the epilogue;
    - a slice value escapes (consumed by a statement outside the slice) —
      stripping the scalar stmts would break the consumer;
    - a leaf operand is not a (scalar, gmem, not-produced-in-kernel) Load with
      an f32-convertible dtype — e.g. a cooperative-reduce scalar, a Select
      result, or an in-kernel intermediate;
    - a leaf Load index dim is not addressable per element: indexed by a
      reduce axis, or mixing the output cell axes in an expression that isn't
      struct-equal to the Write's own M / N index expr (the lane arithmetic
      can only reproduce those two motions, plus cell-uniform terms).

    Dialect-agnostic: works on Loop-IR (the eligibility gate) and Tile-IR (the
    ``005_lower_atom_tile`` fold) bodies — both carry ``Assign`` / ``Load`` /
    ``Write`` leaves and reduce loops duck-typed on ``.is_reduce`` / ``.axis``.
    ``leaf_dtype`` maps a buffer name to its dtype name (graph lookup).

    ``outer_loads`` (fold side only) maps SSA names to ``Load``s defined in the
    enclosing kernel body ABOVE this slice — splice/hoist passes park
    loop-invariant scalar loads (a real trace's f32 constants, e.g. SiLU's
    ``1``) at the TileOp root, outside the ``AtomTile`` the fold scans. The
    Loop-IR gate sees the whole body and admits them as leaves, so without
    this fallback the fold would disagree with the gate on exactly those
    kernels. An outer leaf passes the same produced / dtype / per-dim role
    checks as an in-slice one (its indices carry no cell axes by construction
    — they're not in scope at the root — so it folds as a ``fixed`` load)."""
    from deplodock.compiler.ir.stmt import Assign, Load, Write  # noqa: PLC0415

    # One deep walk in body order: stmts with their inside-a-reduce-loop flag,
    # plus the reduce axis names (for the index check).
    stmts: list[tuple[object, bool]] = []
    reduce_axes: set[str] = set()

    def _walk(stmt_iter, in_reduce: bool) -> None:
        for s in stmt_iter:
            stmts.append((s, in_reduce))
            is_red = bool(getattr(s, "is_reduce", False))
            if is_red and hasattr(s, "axis"):
                reduce_axes.add(s.axis.name)
            for sub in s.nested():
                _walk(sub, in_reduce or is_red)

    _walk(body, False)

    # Backward slice = forward closure from the accumulators over Assigns
    # (body order is SSA order, so one pass converges).
    closure: set[str] = set()
    slice_assigns: list = []
    slice_in_reduce = False
    for s, in_reduce in stmts:
        if isinstance(s, Assign) and set(s.args) & (accum_names | closure):
            closure.add(s.name)
            slice_assigns.append(s)
            slice_in_reduce = slice_in_reduce or in_reduce
    if not slice_assigns:
        return None, None

    if slice_in_reduce:
        return None, "the accumulator is consumed inside a reduce loop (mid-reduction use, not a store-time fold)"
    # Renderability probe: the fold translates each chain op exactly like the
    # scalar Assign render does (``op_to_expr``) — an op that translation
    # doesn't cover (or with the wrong arity) is an ineligible operation.
    from deplodock.compiler.ir.expr import Var  # noqa: PLC0415
    from deplodock.compiler.ir.stmt.base import op_to_expr  # noqa: PLC0415

    for a in slice_assigns:
        try:
            op_to_expr(a.op.name, [Var(x) for x in a.args])
        except (NotImplementedError, IndexError):
            return None, f"epilogue op {a.op.name!r} has no fragment-store rendering"
    accs_used = set()
    for a in slice_assigns:
        accs_used |= set(a.args) & accum_names
    if len(accs_used) > 1:
        return None, f"the epilogue depends on {len(accs_used)} accumulators (multi-fragment fold not supported)"

    writes = [s for s, _ in stmts if isinstance(s, Write)]
    fed_writes = [w for w in writes if w.value in closure]
    if len(fed_writes) != 1:
        return None, f"the epilogue feeds {len(fed_writes)} Writes (need exactly 1)"
    write = fed_writes[0]
    if write.is_vector:
        return None, "the epilogue feeds a vector Write"
    if any(w is not write and w.value in accs_used for w in writes):
        return None, "the accumulator is also written directly alongside the epilogue"

    slice_set = set(map(id, slice_assigns))

    # Per-element-addressable cell coordinates: the Write's var-bearing index
    # dims give the M (second-to-last) / N (last) motions. Computed here (before
    # leaf classification) because a folded Select's predicate is checked
    # against these coordinate vars.
    from deplodock.compiler.ir.stmt import Select  # noqa: PLC0415

    w_var_dims = [d for d, e in enumerate(write.index) if e.free_vars()]
    if not w_var_dims:
        return None, "the Write index has no free axes (not a matmul output)"
    n_expr = write.index[w_var_dims[-1]]
    m_expr = write.index[w_var_dims[-2]] if len(w_var_dims) >= 2 else None
    m_coord_vars = frozenset(m_expr.free_vars()) if m_expr is not None else frozenset()
    n_coord_vars = frozenset(n_expr.free_vars())
    cell_vars = n_coord_vars | m_coord_vars

    # A folded Select (the causal attention mask) is itself a slice statement —
    # collect its id so the escape check doesn't treat its consumed-by-slice use
    # as an escape, and so it is stripped from the consumer body.
    selects_by_name: dict[str, object] = {s.name: s for s, _ in stmts if isinstance(s, Select)}
    folded_selects: list = []

    for s, _ in stmts:
        if id(s) in slice_set or s is write or isinstance(s, Select):
            continue
        if set(s.deps()) & closure:
            return None, f"an epilogue value escapes the slice (consumed by {type(s).__name__})"

    # Leaf operands: every arg that is neither an accumulator nor a slice value.
    # A leaf may resolve to a Select (the coord-predicated causal mask) instead
    # of a Load — fold it: its predicate must be addressable per element (only
    # the M/N coordinate vars) and its branch values must themselves be leaf
    # Loads. Branch values join the leaf worklist.
    loads_by_name: dict[str, object] = {}
    loads_in_reduce: set[str] = set()
    for s, in_reduce in stmts:
        if isinstance(s, Load) and len(s.names) == 1:
            loads_by_name.setdefault(s.names[0], s)
            if in_reduce:
                loads_in_reduce.add(s.names[0])
    leaf_names: list[str] = []
    for a in slice_assigns:
        for arg in a.args:
            if arg not in accum_names and arg not in closure and arg not in leaf_names:
                leaf_names.append(arg)
    leaf_loads = []
    seen_selects: set[str] = set()
    i = 0
    while i < len(leaf_names):
        name = leaf_names[i]
        i += 1
        sel = selects_by_name.get(name)
        if sel is not None:
            if name in seen_selects:
                continue
            seen_selects.add(name)
            for br in sel.branches:
                pred_vars = set(br.select.free_vars())
                if not pred_vars <= cell_vars:
                    return None, f"epilogue Select {name!r} predicate {br.select.pretty()} references non-coordinate axes"
                if br.value not in leaf_names:
                    leaf_names.append(br.value)
            folded_selects.append(sel)
            continue
        ld = loads_by_name.get(name) or (outer_loads or {}).get(name)
        if ld is None:
            return None, f"epilogue operand {name!r} is not a scalar Load or the accumulator"
        if name in loads_in_reduce:
            return None, f"epilogue operand {name!r} is loaded inside a reduce loop"
        if ld.input in produced:
            return None, f"epilogue operand buffer {ld.input!r} is produced in-kernel (register-resident intermediate)"
        dt = leaf_dtype(ld.input)
        if dt is not None and dt not in _EPILOGUE_LEAF_DTYPES:
            return None, f"epilogue operand buffer {ld.input!r} dtype {dt!r} has no f32 conversion"
        leaf_loads.append(ld)
    load_roles: list[tuple[str, ...]] = []
    for ld in leaf_loads:
        roles: list[str] = []
        for e in ld.index:
            if m_expr is not None and e == m_expr:
                roles.append("m")
                continue
            if e == n_expr:
                roles.append("n")
                continue
            fv = e.free_vars()
            if fv & reduce_axes:
                return None, f"epilogue operand {ld.input!r} is indexed by a reduce axis ({e.pretty()})"
            if fv & cell_vars:
                return None, f"epilogue operand {ld.input!r} index {e.pretty()} mixes output cell axes (lane arithmetic can't reproduce it)"
            roles.append("fixed")
        load_roles.append(tuple(roles))

    acc = next(iter(accs_used))
    return (
        EpilogueSlice(
            acc=acc,
            assigns=tuple(slice_assigns),
            loads=tuple(leaf_loads),
            load_roles=tuple(load_roles),
            write=write,
            selects=tuple(folded_selects),
        ),
        None,
    )


# Per-atom eligibility predicates, keyed by the ``Atom`` itself (cell shape +
# operand dtype come from the spec). Kept parallel to the registry rather than
# on ``Atom`` so the spec stays a pure data record in ``ir/tile/ir.py`` with no
# loop/graph/context dependency.
_ELIGIBILITY: dict[Atom, Callable[[LoopOp, Context, Graph], bool]] = {atom: _mma_eligible_factory(atom) for atom in ATOM_REGISTRY.values()}


def is_atom_eligible(atom: Atom, loop_op: LoopOp, ctx: Context, *, graph: Graph) -> bool:
    """Dispatch the per-atom eligibility predicate. ``graph`` is the
    :class:`Graph` the ``loop_op`` lives in — the predicate uses it to resolve
    Load source-buffer dtypes (Loop-IR Loads don't carry ``.dtype`` until the
    Kernel-IR ``030_stamp_types`` pass)."""
    return _ELIGIBILITY[atom](loop_op, ctx, graph)
