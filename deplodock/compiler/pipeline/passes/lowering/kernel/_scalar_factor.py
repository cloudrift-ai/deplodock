"""Scalar (register-tile) contraction tier — the scalar leaf codegen for the shared contraction
factorizer.

:func:`scalar_codegen` returns the ``(state_decls, reduce_region, store)`` callables the generic
tiling layer (``_tiling`` / ``_factor.factorize``) splices through the SAME
``atomize → register_tile → unit_tile → grid_tile`` pipeline the mma tier uses — only the atom
differs: a ``1×1`` scalar fma cell with ``lanes == 1`` (one thread per unit), so the UNIT level is
the parallel thread-tile and there is no ``_lane`` axis. Each thread owns a ``reg_m × reg_n`` block
of output cells; the reduce-loop body is replicated per cell (its operand loads deduped — the
arithmetic-intensity reuse), the small inner reduce unrolled, and each cell writes its own (guarded)
output.

The body comes from the contraction's lowered per-cell body (``lower(op)``, captured in
``005_contract`` on the ``ScalarLeaf``), split into a ``pre`` region / the reduce ``Loop`` / a
projection ``tail``. Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.expr import BinaryExpr
from deplodock.compiler.ir.kernel.ir import Contraction, ScalarLeaf
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr


def _unroll_inner(axis) -> bool:
    """Mark the inner contraction loop for ``#pragma unroll`` when it's a small static reduce
    (≤ 64 trips) — register-resident operand reuse + ILP, the scalar-SGEMM lever."""
    return axis.extent.is_static and axis.extent.as_static() <= 64


def _dedup_loads(stmts: list[Stmt]) -> list[Stmt]:
    """Collapse syntactically-identical scalar ``Load``s (same buffer + index) to one binding,
    rewriting the dropped names to the survivor — the operand reuse a register tile exists for (a
    load not referencing the ``m`` cell axis is shared across the ``n`` cells, and vice versa)."""
    seen: dict = {}
    rename: dict[str, str] = {}
    kept: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Load) and s.is_scalar:
            sig = (s.input, tuple(e.pretty() for e in s.index))
            if sig in seen:
                rename[s.names[0]] = seen[sig]
                continue
            seen[sig] = s.names[0]
        kept.append(s)
    if rename:
        kept = [s.rewrite(lambda nm: rename.get(nm, nm)) for s in kept]
    return kept


def _guard_writes(stmts: list[Stmt], cond) -> list[Stmt]:
    """Wrap each output ``Write`` in ``Cond(cond, …)`` — the masked tail cell computes (with
    clamp-read operands) but only stores when in bounds. Non-``Write`` stmts pass through."""
    if cond is None:
        return stmts
    return [Cond(cond=cond, body=(s,)) if isinstance(s, Write) else s for s in stmts]


def _scalar_sigma(m_axis, n_axis, offset, i: int, j: int, masks) -> Sigma:
    """σ mapping the output axes to register cell ``(i, j)``'s real coordinate (the offset's
    block·tile + unit·reg + r), a **masked** axis wrapped in-bounds (``% extent``)."""
    mask_m, mask_n, m_ext, n_ext = masks
    smap: dict = {}
    if m_axis is not None:
        bm = offset.base("m", i)
        smap[m_axis.name] = BinaryExpr("%", bm, m_ext) if mask_m else bm
    bn = offset.base("n", j)
    smap[n_axis.name] = BinaryExpr("%", bn, n_ext) if mask_n else bn
    return Sigma(smap)


def _scalar_bound(m_axis, n_axis, offset, i: int, j: int, masks):
    """The in-bounds predicate for cell ``(i, j)`` — ``base < extent`` for each masked axis (anded),
    or ``None`` when nothing overhangs."""
    mask_m, mask_n, m_ext, n_ext = masks
    conds = []
    if mask_m and m_axis is not None:
        conds.append(BinaryExpr("<", offset.base("m", i), m_ext))
    if mask_n:
        conds.append(BinaryExpr("<", offset.base("n", j), n_ext))
    if not conds:
        return None
    cond = conds[0]
    for c in conds[1:]:
        cond = BinaryExpr("&&", cond, c)
    return cond


def scalar_codegen(c: Contraction):
    """The scalar leaf codegen — returns the ``(state_decls, reduce_region, store)`` callables
    ``_factor.factorize`` hands to ``grid_tile``. Splits the captured per-cell body **once** (a
    ``pre`` region / the reduce ``Loop`` over the contraction axis / a projection ``tail``) and
    replicates it per register cell (operand loads deduped — the arithmetic-intensity reuse); the
    tiling geometry (tile / mask / axis names) is read off ``c``.

    (This is the scalar tier's whole codegen — the body split + protected-name set that used to live
    on a ``ScalarUnit`` object now live in this closure scope, so there is no per-atom object beside
    the ``Contraction``.)"""
    leaf: ScalarLeaf = c.leaf
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
    # Split the lowered per-cell body: the ``pre`` region, the reduce ``Loop``, and the ``tail``.
    full = list(leaf.body)
    ridx = next(i for i, s in enumerate(full) if isinstance(s, Loop) and s.axis.name == k_axis.name)
    pre, rloop, tail = full[:ridx], full[ridx], full[ridx + 1 :]
    # The shared iteration coordinates — excluded from the per-cell SSA rename.
    prot = {c.n_b, c.n_uvar, k_axis.name}
    if m_axis is not None:
        prot |= {c.m_b, c.m_uvar}
    axes_for_ext = [n_axis, k_axis, *c.lead_axes]
    if m_axis is not None:
        axes_for_ext.append(m_axis)
    for a in c.lead_axes:
        prot.add(a.name)
    for a in axes_for_ext:
        prot |= set(_extent_expr(a).free_vars())
    protected = frozenset(prot)

    def _cells(region, cells, offset, masks, *, guard: bool) -> list[Stmt]:
        """Replicate ``region`` over every register cell — σ-offset the free indices, suffix the
        per-cell SSA names, optionally guard the writes — then collapse shared operand loads."""
        out: list[Stmt] = []
        for i, j in cells:
            sigma = _scalar_sigma(m_axis, n_axis, offset, i, j, masks)
            rename = lambda nm, i=i, j=j: nm if nm in protected else f"{nm}__c{i}_{j}"  # noqa: E731
            cell = [s.rewrite(rename, sigma) for s in region]
            if guard:
                cell = _guard_writes(cell, _scalar_bound(m_axis, n_axis, offset, i, j, masks))
            out.extend(cell)
        return _dedup_loads(out)

    def state_decls(cells) -> list[Stmt]:
        # The scalar accumulators are seeded inside the reduce ``Loop`` (the dissolved fold
        # ``Accum``\\ s + ``Loop.render``), so there are no separate state decls.
        return []

    def reduce_region(cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
        pre_cells = _cells(pre, cells, offset, masks, guard=False) if pre else []
        loop_body = _cells(rloop.body, cells, offset, masks, guard=False)
        new_loop = Loop(axis=k_axis, body=Body(tuple(loop_body)), unroll=rloop.unroll or _unroll_inner(k_axis))
        return [], [*pre_cells, new_loop]

    def store(i, j, offset, masks) -> list[Stmt]:
        sigma = _scalar_sigma(m_axis, n_axis, offset, i, j, masks)
        rename = lambda nm: nm if nm in protected else f"{nm}__c{i}_{j}"  # noqa: E731
        cell = [s.rewrite(rename, sigma) for s in tail]
        cell = _guard_writes(cell, _scalar_bound(m_axis, n_axis, offset, i, j, masks))
        return _dedup_loads(cell)

    return state_decls, reduce_region, store
