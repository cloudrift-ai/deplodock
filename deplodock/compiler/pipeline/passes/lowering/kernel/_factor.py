"""The one contraction factorizer — atom-generic, both codegens in one place.

Both atoms of a :class:`~...ir.Contraction` (a tensor-core :class:`AtomKind` or the scalar
:class:`ScalarAtom`) expand through the *same* four-level tiling pipeline (``atomize →
register_tile → unit_tile → grid_tile``). :func:`factorize` reads the tiling **geometry straight off
the** ``Contraction`` **node** (``tile_m`` / ``mask_m`` / ``m_b`` / ``m_uvar`` / ``units_m`` /
``block_threads`` / …, derived there from the ``tile`` schedule + the output axes) and splices two
codegen halves into ``grid_tile``:

- :func:`reduce_codegen` — the reusable, **sink-agnostic** ``(state_decls, reduce_region)``: the
  operand fragments + the contraction K-loop, dispatched off the atom (the tensor-core mma pair
  :func:`_mma_state` / :func:`_mma_reduce` vs the scalar fma pair :func:`_scalar_state` /
  :func:`_scalar_reduce`). The mma tier loads operands **gmem-direct** (``LdmatrixLoad`` +
  ``MmaSyncPtx``; reuse from the register tile); the scalar tier synthesizes ``for k: acc += a*b``
  (:func:`_synth_reduce`) replicated per register cell (loads deduped). Both leave the accumulator
  (mma ``_c{i}_{j}`` fragments / scalar ``acc__c{i}_{j}``) for the sink.
- the **sink** ``store(i, j, offset, masks)`` — the per-cell consumer of that accumulator.
  :func:`store_sink` is the default **matmul** sink (an mma ``RegStore`` / the replicated scalar
  ``epilogue`` tail, projecting to the output). ``factorize(c, store=…)`` swaps it — the flash inner
  QK/PV pass a sink that bridges the accumulator into the streaming-softmax twist, reusing the same
  :func:`reduce_codegen`.

The smem operand-staging pipeline (cp.async / TMA) was dropped to keep the two tiers symmetric (the
``STAGE`` codec + ``schedule.Stage`` still land; see ``ir/schedule``). Leading ``_`` so the pass
loader skips this module."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

from deplodock.compiler.ir.atom import AtomKind
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.kernel import Tile
from deplodock.compiler.ir.kernel.ir import (
    EpilogueLoad,
    LdmatrixLoad,
    MmaSyncPtx,
    RegEpilogue,
    RegFragment,
    RegStore,
)
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Load, Loop, Select, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import Contraction
from deplodock.compiler.ir.tile.ops import contraction_loop, lower
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import copy_cell
from deplodock.compiler.pipeline.passes.lowering.kernel._geom import extent_expr as _extent_expr
from deplodock.compiler.pipeline.passes.lowering.kernel._store import has_write, with_store
from deplodock.compiler.pipeline.passes.lowering.kernel._tiling import atomize, grid_tile, register_tile, unit_tile
from deplodock.compiler.pipeline.pipeline import LoweringError

#: The contraction semiring — multiply ⊗ then accumulate ⊕ (add). The same multiply-add ``mma.sync``
#: realizes; in the scalar tier it is a plain scalar fma loop.
_MUL = ElementwiseImpl("multiply")
_ADD = ElementwiseImpl("add")


# ---- warp/mma tier ----------------------------------------------------------------------------- #
def _warp_roles(index, m_name: str, n_name: str) -> tuple[str, ...]:
    """Per-dim epilogue-load role: ``"m"`` / ``"n"`` for a dim varying with the output row /
    col axis, else ``"fixed"`` (batch / grid literal — uniform across the fragment cell)."""
    roles = []
    for e in index:
        fv = e.free_vars()
        roles.append("m" if m_name in fv else "n" if n_name in fv else "fixed")
    return tuple(roles)


def _warp_epilogue(tail: list[Stmt], acc: str, m_name: str, n_name: str, sigma: Sigma) -> RegEpilogue | None:
    """Fold the projection ``Map`` into a :class:`RegEpilogue` for cell ``sigma``. ``None`` when
    there is no projection (a bare ``Write`` of the accumulator).

    The projection is the post-reduce ``tail`` stmts: the leaf ``Load``s + pointwise ``Assign``s +
    an optional causal ``Select``. Each leaf ``Load`` becomes an :class:`EpilogueLoad` at the
    cell-base coordinate (σ-applied; the render adds the per-element row/col motion on the
    ``m``/``n`` dims); each ``Assign`` becomes an ``(name, op, args)`` op; a coord-predicated
    ``Select`` (causal mask) rewrites its ``m``/``n`` coordinate vars to the ``__M__`` / ``__N__``
    placeholders the store substitutes with the element's own (row, col)."""
    loads, ops, selects = [], [], []
    write = None
    ph = {m_name: Var("__M__"), n_name: Var("__N__")}
    for s in tail:
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


def _mma_state(c: Contraction, cells) -> list[Stmt]:
    """The mma operand/accumulator register fragments — one ``_a``/``_b`` per register row/col and
    one ``_c`` accumulator per cell (held across the K-loop)."""
    atom = c.atom
    decls: list[Stmt] = []
    for i in range(c.reg_m):
        decls.append(RegFragment(name=f"_a{i}", role="a", shape=atom.shape, dtype=atom.operand_dtype("a")))
    for j in range(c.reg_n):
        decls.append(RegFragment(name=f"_b{j}", role="b", shape=atom.shape, dtype=atom.operand_dtype("b")))
    for i in range(c.reg_m):
        for j in range(c.reg_n):
            decls.append(RegFragment(name=f"_c{i}_{j}", role="c", shape=atom.shape, dtype=atom.operand_dtype("c")))
    return decls


def _mma_reduce(c: Contraction, cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
    """The mma K-loop: ``ldmatrix`` each operand fragment **gmem-direct**, then ``mma.sync`` every
    cell. A symbolic / non-divisible K zero-fills the masked-K tail via the ``k_zero`` helper variants
    — a duplicate read would corrupt the reduction (unlike a masked M/N row, whose store is just
    guarded). Transposed-B has no gmem-direct K zero-fill helper, so it bails."""
    atom = c.atom
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
    a_load, b_load, b_trans = c.a_load, c.b_load, c.b_trans
    mask_m, mask_n, m_ext, n_ext = masks
    k_static = k_axis.extent.is_static
    if not k_static and b_trans:
        raise LoweringError("warp tier: transposed-B symbolic-K mma not supported (no gmem-direct K zero-fill)")
    k_zero = None if k_static else (Var(k_axis.name), _extent_expr(k_axis))
    chain: list[Stmt] = []
    for i in range(c.reg_m):
        idx = tuple(Sigma({m_axis.name: offset.base("m", i)}).apply(e) for e in a_load.index)
        guard = (offset.base("m", i), m_ext) if mask_m else None
        chain.append(
            LdmatrixLoad(frag=f"_a{i}", src_buffer=a_load.input, src_index=idx, role="a", staged=False, gmem_guard=guard, k_zero=k_zero)
        )
    for j in range(c.reg_n):
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
    for i in range(c.reg_m):
        for j in range(c.reg_n):
            chain.append(MmaSyncPtx(c_frag=f"_c{i}_{j}", a_frag=f"_a{i}", b_frag=f"_b{j}", shape=atom.shape, ab_dtype=atom.ab_dtype))
    kstmts = [StridedLoop(axis=k_axis, start=Literal(0, "int"), step=Literal(atom.atom_k, "int"), body=Body(tuple(chain)), unroll=k_static)]
    return [], kstmts


def _mma_store(c: Contraction, i: int, j: int, offset, masks) -> list[Stmt]:
    """Store cell ``(i, j)``'s ``_c`` fragment to the output, folding the projection ``tail`` into a
    :class:`RegEpilogue` and guarding overhanging M/N rows."""
    atom = c.atom
    m_axis, n_axis = c.m_axis, c.n_axis
    mask_m, mask_n, m_ext, n_ext = masks
    tail = list(c.epilogue)
    write = next(s for s in tail if isinstance(s, Write))
    sigma = Sigma({m_axis.name: offset.base("m", i), n_axis.name: offset.base("n", j)})
    return [
        RegStore(
            dst_buffer=write.output,
            dst_index=tuple(sigma.apply(e) for e in write.index),
            frag=f"_c{i}_{j}",
            shape=atom.shape,
            epilogue=_warp_epilogue(tail, c.acc, m_axis.name, n_axis.name, sigma),
            m_guard=(offset.base("m", i), m_ext) if mask_m else None,
            n_guard=(offset.base("n", j), n_ext) if mask_n else None,
        )
    ]


# ---- scalar (register-tile) tier --------------------------------------------------------------- #
def _unroll_inner(axis) -> bool:
    """Mark the inner contraction loop for ``#pragma unroll`` when it's a small static reduce
    (≤ 64 trips) — register-resident operand reuse + ILP, the scalar-SGEMM lever."""
    return axis.extent.is_static and axis.extent.as_static() <= 64


def _synth_reduce(c: Contraction) -> Loop:
    """The scalar contraction reduce loop ``for k: v = a*b; acc += v`` — built by the shared
    ``ops.contraction_loop`` builder (the **same** ``CONTRACTION`` loop generation the flash score
    producer uses, one source of truth, no register-tile special case), then stamping the
    small-static ``unroll``. The :class:`Contraction` node carries A/B as plain leaf ``Load``\\ s (their
    indices carry the cell ``m`` / ``n`` + the loop ``k``); the operands keep B-then-A order for the
    load reuse."""
    k = c.k_axis
    loop = contraction_loop(
        lift=_MUL,
        fold=Accum(name=c.acc, value=f"{c.acc}__v", op=_ADD, axes=(k.name,)),
        operand_bodies=([c.b_load], [c.a_load]),  # B[k, n], A[m, k]
        reduce_axis=k,
    )
    return replace(loop, unroll=_unroll_inner(k))


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


def _scalar_protected(c: Contraction) -> frozenset[str]:
    """The shared iteration coordinates — the block / unit / loop / extent vars excluded from the
    per-cell SSA rename (everything else is suffixed ``__c{i}_{j}`` so each cell owns its names)."""
    m_axis, n_axis, k_axis = c.m_axis, c.n_axis, c.k_axis
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
    return frozenset(prot)


def _scalar_cells(c: Contraction, region: list[Stmt], cells, offset, masks, protected: frozenset[str], *, guard: bool) -> list[Stmt]:
    """Replicate ``region`` over every register cell — σ-offset the free indices, suffix the
    per-cell SSA names, optionally guard the writes — then collapse shared operand loads."""
    m_axis, n_axis = c.m_axis, c.n_axis
    out: list[Stmt] = []
    for i, j in cells:
        sigma = _scalar_sigma(m_axis, n_axis, offset, i, j, masks)
        cell = copy_cell(region, sigma, f"__c{i}_{j}", protected)
        if guard:
            cell = _guard_writes(cell, _scalar_bound(m_axis, n_axis, offset, i, j, masks))
        out.extend(cell)
    return _dedup_loads(out)


def _scalar_state(c: Contraction, cells) -> list[Stmt]:
    """No separate state decls — the scalar accumulators are seeded inside the reduce ``Loop``
    (the dissolved fold ``Accum``\\ s + ``Loop.render``)."""
    return []


def _scalar_reduce(c: Contraction, cells, offset, masks) -> tuple[list[Stmt], list[Stmt]]:
    """**Synthesize** the scalar reduce loop (:func:`_synth_reduce`) and replicate its body per
    register cell (loads deduped). There is no pre-loop region — any loop-invariant operand reads
    ride in the projection ``tail`` (the store's epilogue)."""
    k_axis = c.k_axis
    rloop = _synth_reduce(c)
    loop_body = _scalar_cells(c, rloop.body, cells, offset, masks, _scalar_protected(c), guard=False)
    new_loop = Loop(axis=k_axis, body=Body(tuple(loop_body)), unroll=rloop.unroll or _unroll_inner(k_axis))
    return [], [new_loop]


def _scalar_store(c: Contraction, i: int, j: int, offset, masks) -> list[Stmt]:
    """Replicate the projection ``tail`` for cell ``(i, j)`` — σ-offset, suffix the SSA names, guard
    the (overhanging) write, dedup shared operand loads."""
    m_axis, n_axis = c.m_axis, c.n_axis
    sigma = _scalar_sigma(m_axis, n_axis, offset, i, j, masks)
    cell = copy_cell(c.epilogue, sigma, f"__c{i}_{j}", _scalar_protected(c))
    cell = _guard_writes(cell, _scalar_bound(m_axis, n_axis, offset, i, j, masks))
    return _dedup_loads(cell)


#: The reusable ``(state_decls, reduce_region)`` pair, keyed by atom kind — the operand fragments +
#: the K-loop, **sink-agnostic**: both leave the accumulator a sink consumes (mma ``_c{i}_{j}``
#: register fragments / scalar ``acc__c{i}_{j}`` per cell). :func:`reduce_codegen` binds the node.
_MMA_REDUCE = (_mma_state, _mma_reduce)
_SCALAR_REDUCE = (_scalar_state, _scalar_reduce)


def reduce_codegen(c: Contraction):
    """The reusable ``(state_decls, reduce_region)`` — operand fragments + the contraction K-loop
    (``ldmatrix`` + ``mma.sync`` / the synthesized scalar fma), dispatched off the atom and bound to
    ``c``. **Sink-agnostic**: it leaves the accumulator the :func:`store_sink` (or a flash sink) then
    consumes, so the same K-loop emission is reused wherever a contraction is tiled."""
    state, reduce_region = _MMA_REDUCE if isinstance(c.atom, AtomKind) else _SCALAR_REDUCE
    return partial(state, c), partial(reduce_region, c)


def store_sink(c: Contraction):
    """The default **matmul sink** — the per-cell ``store(i, j, offset, masks)`` callable that writes
    each accumulator cell to the output through the projection ``epilogue`` (an mma ``RegStore`` /
    the replicated scalar epilogue tail), dispatched off the atom. The flash branch swaps a sink that
    instead feeds the accumulator fragments into the streaming-softmax twist, reusing
    :func:`reduce_codegen`."""
    store = _mma_store if isinstance(c.atom, AtomKind) else _scalar_store
    return partial(store, c)


def factorize(tile, root, store=None) -> Tile:
    """The single node-kind dispatcher — expand a ``TileOp``'s ``op`` into its bound ``Tile``.

    Reads the structural node off ``tile.op`` and picks the emitter:

    - a :class:`Contraction` (warp / register tile) → :func:`_factorize_contraction`, the atom-generic
      four-level pipeline. The bare grid-``Write`` is synthesized here (it needs ``root.output``, so it
      can't ride the node) into the projection ``epilogue`` before the tiling.
    - anything else (a pointwise ``Map``, or a reduction with a trivial :class:`ReducePlan`) → the
      **scalar tier**: one thread per output cell. ``lower(op)`` emits the per-cell body (a serial
      reduce ``Loop`` sits inside it), the output-store glue is appended if the body has none, and the
      body is wrapped in a single :class:`Tile` bound to ``place.grid``."""
    op = tile.op
    if isinstance(op, Contraction):
        tail = list(op.epilogue)
        if not has_write(tail):
            op = replace(op, epilogue=with_store(tail, root.output.name, tile.place.grid, op))
        return _factorize_contraction(op, store)
    stmts = with_store(lower(op), root.output.name, tile.place.grid, op)
    return Tile(axes=tuple(tile.place.grid), body=Body(tuple(stmts)))


def _factorize_contraction(c: Contraction, store=None) -> Tile:
    """Expand a :class:`Contraction` into its tiled ``Tile`` — the one pipeline for both atoms. The
    node supplies the per-level geometry; :func:`reduce_codegen` synthesizes the operand load + K-loop
    and ``store`` is the **per-cell sink** (default: the matmul :func:`store_sink`; the flash inner
    QK/PV pass a sink that bridges the accumulator into the softmax twist); the layer owns the offset,
    the axes, and the splice."""
    state_decls, reduce_region = reduce_codegen(c)
    if store is None:
        store = store_sink(c)
    masks = (c.mask_m, c.mask_n, c.m_ext, c.n_ext)
    t = atomize(c.atom.atom_m, c.atom.atom_n)
    t = register_tile(t, c.reg_m, c.reg_n)
    t = unit_tile(t, c.units_m, c.units_n, c.m_uvar, c.n_uvar)
    return grid_tile(
        t,
        masks,
        n_axis=c.n_axis,
        n_b=c.n_b,
        tile_n=c.tile_n,
        m_axis=c.m_axis,
        m_b=c.m_b,
        tile_m=c.tile_m,
        lead_axes=c.lead_axes,
        block_threads=c.block_threads,
        lanes=c.atom.lanes,
        state_decls=state_decls,
        reduce_region=reduce_region,
        store=store,
    )
