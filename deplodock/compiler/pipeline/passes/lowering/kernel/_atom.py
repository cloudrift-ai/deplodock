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

from dataclasses import dataclass

from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, Atom

# Back-compat re-exports (canonical definitions live in ``ir/tile/ir.py``).
__all__ = [
    "ATOM_REGISTRY",
    "Atom",
]

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


