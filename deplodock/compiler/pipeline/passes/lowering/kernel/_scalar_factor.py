"""Scalar (register-tile) contraction tier — the scalar :class:`ScalarUnit` leaf for the shared
contraction factorizer.

:class:`ScalarUnit` is the scalar-atom ``Unit`` the generic tiling layer (``_tiling`` /
``_factor.factorize``) tiles through the SAME ``atomize → register_tile → unit_tile → grid_tile``
pipeline the mma tier uses — only the atom differs: a ``1×1`` scalar fma cell with ``lanes == 1``
(one thread per unit), so the UNIT level is the parallel thread-tile and there is no ``_lane`` axis.
Each thread owns a ``reg_m × reg_n`` block of output cells; the reduce-loop body is replicated per
cell (its operand loads deduped — the arithmetic-intensity reuse), the small inner reduce unrolled,
and each cell writes its own (guarded) output.

The leaf's body comes from the contraction's lowered per-cell body (``lower(op)``, captured in
``005_contract``), split into a ``pre`` region / the reduce ``Loop`` / a projection ``tail``.
Leading ``_`` so the pass loader skips this module."""

from __future__ import annotations

from deplodock.compiler.ir.expr import BinaryExpr
from deplodock.compiler.ir.kernel.ir import Contraction, ScalarLeaf
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import OffsetFn, Unit


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


class ScalarUnit(Unit):
    """The scalar-fma leaf — wraps the contraction's lowered per-cell body so the generic tiling
    layer assembles the register tile. The shared tiling geometry (tile / mask / axis-name / block)
    is derived by the base from the scalar ``1×1`` atom (``lanes == 1`` — the UNIT is a single
    thread, block / unit axes spelled ``_b`` / ``_u``); this class only splits the captured body
    into a ``pre`` region, the reduce ``Loop`` over the contraction axis, and a projection ``tail``,
    and supplies the four leaf methods."""

    def __init__(self, c: Contraction):
        leaf: ScalarLeaf = c.leaf
        super().__init__(
            atom=c.atom,
            m_axis=c.m_axis,
            n_axis=c.n_axis,
            reg_m=leaf.reg_m,
            reg_n=leaf.reg_n,
            units_m=leaf.par_m,  # the scalar parallel thread-tile IS the UNIT grid
            units_n=leaf.par_n,
            b_suffix="_b",
            u_suffix="_u",
            lead_axes=c.lead_axes,
        )
        self.k_axis = c.k_axis
        # Split the lowered per-cell body: everything before the reduce loop (``pre``), the reduce
        # ``Loop`` itself, and the projection ``tail`` (the finalize + output ``Write``).
        full = list(leaf.body)
        ridx = next(i for i, s in enumerate(full) if isinstance(s, Loop) and s.axis.name == self.k_axis.name)
        self.pre, self.rloop, self.tail = full[:ridx], full[ridx], full[ridx + 1 :]
        # The shared iteration coordinates — excluded from the per-cell SSA rename.
        prot = {self.n_b, self.n_uvar, self.k_axis.name}
        if self.m_axis is not None:
            prot |= {self.m_b, self.m_uvar}
        axes_for_ext = [self.n_axis, self.k_axis, *self.lead_axes]
        if self.m_axis is not None:
            axes_for_ext.append(self.m_axis)
        for a in (*self.lead_axes,):
            prot.add(a.name)
        for a in axes_for_ext:
            prot |= set(_extent_expr(a).free_vars())
        self.protected = frozenset(prot)

    # -- the tiling-layer ``Unit`` protocol --------------------------------------------------- #

    def state_decls(self, cells) -> list[Stmt]:
        # The scalar accumulators are seeded inside the reduce ``Loop`` (the dissolved fold
        # ``Accum``\\ s + ``Loop.render``), so there are no separate state decls. (``operands`` is
        # the base default ``[]`` — the scalar operands ride inside the lowered body, not as leaves.)
        return []

    def reduce_region(self, cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
        pre_cells = self._cells(self.pre, cells, offset, masks, guard=False) if self.pre else []
        loop_body = self._cells(self.rloop.body, cells, offset, masks, guard=False)
        new_loop = Loop(axis=self.k_axis, body=Body(tuple(loop_body)), unroll=self.rloop.unroll or _unroll_inner(self.k_axis))
        return [], [*pre_cells, new_loop]

    def store(self, i, j, offset, masks) -> list[Stmt]:
        sigma = self._sigma(offset, i, j, masks)
        rename = lambda nm: nm if nm in self.protected else f"{nm}__c{i}_{j}"  # noqa: E731
        cell = [s.rewrite(rename, sigma) for s in self.tail]
        cell = _guard_writes(cell, self._bound(offset, i, j, masks))
        return _dedup_loads(cell)

    # -- per-cell σ / guard over the offset --------------------------------------------------- #

    def _sigma(self, offset: OffsetFn, i: int, j: int, masks) -> Sigma:
        """σ mapping the output axes to register cell ``(i, j)``'s real coordinate (the offset's
        block·tile + unit·reg + r), a **masked** axis wrapped in-bounds (``% extent``)."""
        mask_m, mask_n, m_ext, n_ext = masks
        smap: dict = {}
        if self.m_axis is not None:
            bm = offset.base("m", i)
            smap[self.m_axis.name] = BinaryExpr("%", bm, m_ext) if mask_m else bm
        bn = offset.base("n", j)
        smap[self.n_axis.name] = BinaryExpr("%", bn, n_ext) if mask_n else bn
        return Sigma(smap)

    def _bound(self, offset: OffsetFn, i: int, j: int, masks):
        """The in-bounds predicate for cell ``(i, j)`` — ``base < extent`` for each masked axis
        (anded), or ``None`` when nothing overhangs."""
        mask_m, mask_n, m_ext, n_ext = masks
        conds = []
        if mask_m and self.m_axis is not None:
            conds.append(BinaryExpr("<", offset.base("m", i), m_ext))
        if mask_n:
            conds.append(BinaryExpr("<", offset.base("n", j), n_ext))
        if not conds:
            return None
        cond = conds[0]
        for c in conds[1:]:
            cond = BinaryExpr("&&", cond, c)
        return cond

    def _cells(self, region, cells, offset, masks, *, guard: bool) -> list[Stmt]:
        """Replicate ``region`` over every register cell — σ-offset the free indices, suffix the
        per-cell SSA names, optionally guard the writes — then collapse shared operand loads."""
        out: list[Stmt] = []
        for i, j in cells:
            sigma = self._sigma(offset, i, j, masks)
            rename = lambda nm, i=i, j=j: nm if nm in self.protected else f"{nm}__c{i}_{j}"  # noqa: E731
            cell = [s.rewrite(rename, sigma) for s in region]
            if guard:
                cell = _guard_writes(cell, self._bound(offset, i, j, masks))
            out.extend(cell)
        return _dedup_loads(out)
