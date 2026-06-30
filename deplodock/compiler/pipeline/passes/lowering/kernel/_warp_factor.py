"""Warp/mma tier — the mma leaf codegen for the shared contraction factorizer.

:func:`mma_codegen` returns the ``(state_decls, reduce_region, store)`` callables the generic tiling
layer (``_tiling`` / ``_factor.factorize``) splices into the ``RegFragment`` / ``LdmatrixLoad`` /
``MmaSyncPtx`` / ``RegStore`` four-way GRID/UNIT/REGISTER/ATOM split — they own the K-loop and the
per-cell projection epilogue. Operands are loaded **gmem-direct**; the smem operand-staging pipeline
(cp.async / TMA) was dropped to restore symmetry with the scalar tier — a symmetric staging mechanism
for both tiers is reserved (the ``STAGE`` codec + ``schedule.Stage`` still land). The tiling geometry
is read off the ``Contraction`` node; only the tensor-core codegen lives here, out of the IR layer
(``005_contract`` emits a ``Contraction`` with an ``MmaLeaf``). Leading ``_`` so the pass loader skips
this module."""

from __future__ import annotations

from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.kernel.ir import (
    Contraction,
    EpilogueLoad,
    LdmatrixLoad,
    MmaLeaf,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
)
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Assign, Body, Load, Select, Stmt, StridedLoop, Write
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.pipeline import LoweringError


def _warp_roles(index, m_name: str, n_name: str) -> tuple[str, ...]:
    """Per-dim epilogue-load role: ``"m"`` / ``"n"`` for a dim varying with the output row /
    col axis, else ``"fixed"`` (batch / grid literal — uniform across the fragment cell)."""
    roles = []
    for e in index:
        fv = e.free_vars()
        roles.append("m" if m_name in fv else "n" if n_name in fv else "fixed")
    return tuple(roles)


def _warp_epilogue(pre: list[Stmt], tail: list[Stmt], acc: str, m_name: str, n_name: str, sigma: Sigma) -> RegEpilogue | None:
    """Fold the projection ``Map`` into a :class:`RegEpilogue` for cell ``sigma``. ``None`` when
    there is no projection (a bare ``Write`` of the accumulator).

    The projection is the ``lower`` stmts straddling the K reduce loop: ``pre`` — the
    loop-invariant scalar leaf ``Load``s the lift parks above the loop (a fused matmul's scale /
    mask constants) — plus ``tail`` — the post-reduce leaf ``Load``s + pointwise ``Assign``s +
    an optional causal ``Select``. Each leaf ``Load`` becomes an :class:`EpilogueLoad` at the
    cell-base coordinate (σ-applied; the render adds the per-element row/col motion on the
    ``m``/``n`` dims); each ``Assign`` becomes an ``(name, op, args)`` op; a coord-predicated
    ``Select`` (causal mask) rewrites its ``m``/``n`` coordinate vars to the ``__M__`` / ``__N__``
    placeholders the store substitutes with the element's own (row, col)."""
    loads, ops, selects = [], [], []
    write = None
    ph = {m_name: Var("__M__"), n_name: Var("__N__")}
    for s in (*pre, *tail):
        if isinstance(s, Load):
            loads.append(
                EpilogueLoad(
                    name=s.names[0],
                    buffer=s.input,
                    index=tuple(sigma.apply(e) for e in s.index),
                    roles=_warp_roles(s.index, m_name, n_name),
                )
            )
        elif isinstance(s, Assign):
            ops.append((s.name, s.op.name, tuple(s.args)))
        elif isinstance(s, Select):
            selects.append((s.name, tuple((br.select.substitute(ph), br.value) for br in s.branches)))
        elif isinstance(s, Write):
            write = s
    if write is None or (not ops and not selects):
        return None
    return RegEpilogue(acc=acc, loads=tuple(loads), ops=tuple(ops), result=write.value, selects=tuple(selects))


def mma_codegen(c: Contraction):
    """The mma leaf codegen — returns the ``(state_decls, reduce_region, store)`` callables
    ``_factor.factorize`` hands to ``grid_tile``. The operand-fragment naming + the epilogue split
    are computed **once** here and captured by the three closures; the tiling geometry (tile / mask /
    axis names / units) is read off ``c``.

    Operands are loaded **gmem-direct** (``LdmatrixLoad`` straight from global memory; the reuse
    comes from the register tile). The smem operand-staging pipeline (cp.async / TMA) was dropped to
    restore symmetry with the scalar tier — a symmetric staging mechanism for both tiers is reserved
    (the ``STAGE`` codec + ``schedule.Stage`` still land; see ``ir/tile/schedule``)."""
    leaf: MmaLeaf = c.leaf
    atom = c.atom
    atom_k = atom.atom_k
    a_load, b_load, b_trans, acc = leaf.a_load, leaf.b_load, leaf.b_trans, leaf.acc
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
    reg_m, reg_n = c.reg_m, c.reg_n
    pre: list[Stmt] = []
    tail = list(leaf.epilogue)
    write = next(s for s in tail if isinstance(s, Write))
    a_frags = [f"_a{i}" for i in range(reg_m)]
    b_frags = [f"_b{j}" for j in range(reg_n)]

    def state_decls(cells) -> list[Stmt]:
        decls: list[Stmt] = []
        for name in a_frags:
            decls.append(RegFragment(name=name, role="a", shape=atom.shape, dtype=atom.operand_dtype("a")))
        for name in b_frags:
            decls.append(RegFragment(name=name, role="b", shape=atom.shape, dtype=atom.operand_dtype("b")))
        for i in range(reg_m):
            for j in range(reg_n):
                decls.append(RegFragment(name=f"_c{i}_{j}", role="c", shape=atom.shape, dtype=atom.operand_dtype("c")))
        return decls

    def reduce_region(cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
        # Gmem-direct. A symbolic / non-divisible K zero-fills the masked-K tail via the ``k_zero``
        # helper variants — a duplicate read would corrupt the reduction (unlike a masked M/N row,
        # whose store is just guarded). Transposed-B has no gmem-direct K zero-fill helper, so it bails.
        mask_m, mask_n, m_ext, n_ext = masks
        k_static = k_axis.extent.is_static
        if not k_static and b_trans:
            raise LoweringError("warp tier: transposed-B symbolic-K mma not supported (no gmem-direct K zero-fill)")
        k_zero = None if k_static else (Var(k_axis.name), _extent_expr(k_axis))
        chain: list[Stmt] = []
        for i in range(reg_m):
            idx = tuple(Sigma({m_axis.name: offset.base("m", i)}).apply(e) for e in a_load.index)
            guard = (offset.base("m", i), m_ext) if mask_m else None
            chain.append(
                LdmatrixLoad(frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=guard, k_zero=k_zero)
            )
        for j in range(reg_n):
            idx = tuple(Sigma({n_axis.name: offset.base("n", j)}).apply(e) for e in b_load.index)
            guard = (offset.base("n", j), n_ext) if mask_n else None
            chain.append(
                LdmatrixLoad(
                    frag=f"_b{j}",
                    src_buffer=b_load.input,
                    src_index=idx,
                    role="b",
                    staged=False,
                    b_trans=b_trans,
                    gmem_guard=guard,
                    k_zero=k_zero,
                )
            )
        for i in range(reg_m):
            for j in range(reg_n):
                chain.append(MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype))
        kstmts = [StridedLoop(axis=k_axis, start=Literal(0, "int"), step=Literal(atom_k, "int"), body=Body(tuple(chain)), unroll=k_static)]
        return [], kstmts

    def store(i, j, offset, masks) -> list[Stmt]:
        mask_m, mask_n, m_ext, n_ext = masks
        sigma = Sigma({m_axis.name: offset.base("m", i), n_axis.name: offset.base("n", j)})
        return [
            RegStore(
                dst_buffer=write.output,
                dst_index=tuple(sigma.apply(e) for e in write.index),
                frag=f"_c{i}_{j}",
                shape=atom.shape,
                epilogue=_warp_epilogue(pre, tail, acc, m_axis.name, n_axis.name, sigma),
                m_guard=(offset.base("m", i), m_ext) if mask_m else None,
                n_guard=(offset.base("n", j), n_ext) if mask_n else None,
            )
        ]

    return state_decls, reduce_region, store
