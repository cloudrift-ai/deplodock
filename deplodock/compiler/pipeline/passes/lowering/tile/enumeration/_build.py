"""Build the block-DAG ``TileGraph`` from the iteration DAG + a move choice.

The composer's front half: instead of
materializing the ``TileOp`` tower, ``build_dag`` emits the invariant algorithm (a
:class:`Block`) + a reference :class:`Schedule` (the binding), and ``lower`` wraps
it in the ``TileGraphOp`` the enumeration pass returns. One ``build_dag`` serves
every regime — a ``MAP`` nest tiles its free axes; a ``SEMIRING`` / ``MONOID``
reduce additionally re-brackets its contraction axis (``target_names``). There is
no per-shape code: the moves are algebra-licensed, applied uniformly.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import BF16, F32, DataType
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Monoid, Select, SelectBranch, Stmt, Write
from deplodock.compiler.ir.stmt.carrier_algebra import split_carrier
from deplodock.compiler.ir.tile.ir import (
    ATOM_REGISTRY,
    Atom,
    AtomTile,
    Binding,
    Block,
    Edge,
    RegisterTile,
    Schedule,
    SerialTile,
    TileGraph,
    Transport,
)
from deplodock.compiler.ir.twist import ScalarCombiner
from deplodock.compiler.pipeline.passes.lowering._addr import add as _fadd
from deplodock.compiler.pipeline.passes.lowering._addr import mul as _fmul
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _identity_rename, _wrap_tower
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, chain_free_axes


def _free_axes(dag: IterDag) -> tuple[Axis, Axis | None, tuple[Loop, ...]]:
    """The free (PARALLEL) axes to tile, read off the DAG's parallel chain:
    ``(inner_n axis, outer_m axis | None, extra-outer loops)``."""
    parallel = dag.parallel
    inner_n = parallel[-1].loop.axis
    outer_m = parallel[-2].loop.axis if len(parallel) >= 2 else None
    extra_outer = tuple(n.loop for n in parallel[:-2])
    return inner_n, outer_m, extra_outer


def _scaled(name: str, weight: int):
    """A σ term ``Var·weight`` — a bare ``Var`` at unit weight 1, scaled otherwise.
    The least-significant axis in a mixed-radix split (weight 1) renders bare; an
    inner lane / cell axis (``br`` / ``atom_k`` / ``reg``·… stride) renders scaled."""
    return Var(name) if weight == 1 else Var(name) * Literal(weight, "int")


def _split_axis(axis: Axis, specs: list[tuple[str, int, bool]], *, interleave_when_masked: bool = False):
    """σ-split one axis into an outer block axis + the inner factor axes in ``specs``
    (``(suffix, extent, emit_sigma)``, most-significant-first) — the ONE free-axis split
    behind the scalar thread tile (``A → A_b·(T·R) + A_t·R + A_r``) and the warp tile
    (``A → A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom``, the ``A_a`` ATOM lane carrying the
    cell extent but emitting NO σ term — its per-lane offset is owned by ``mma.sync``).

    ``per_block = ∏ extent``; the block axis ``A_b`` ceil-divides + carries ``real_extent``
    + returns the boundary ``bound`` when masked (non-divisible / symbolic), else ``//``.
    σ is the mixed-radix sum ``A_b·per_block + Σ A_i·(∏ extents to its right)`` over the
    ``emit_sigma`` axes (ATOM extents still count toward the other axes' weights). A masked
    inner axis may **interleave** — reverse the inner significance order so the thread lane
    varies fastest (the masked-tile store-coalescing layout). Returns
    ``(a_b, inner_axes, expr, bound)``; the caller lays the domain + bindings."""
    src = axis.source_axis or axis
    per_block = 1
    for _, ext, _ in specs:
        per_block *= ext
    masked = (not axis.extent.is_static) or (axis.extent.as_static() % per_block != 0)
    a_b = Axis(
        f"{axis.name}_b",
        axis.extent.ceil_div(per_block) if masked else axis.extent // per_block,
        source_axis=src,
        real_extent=axis.extent.as_static() if (masked and axis.extent.is_static) else None,
    )
    inner = tuple(Axis(f"{axis.name}_{sfx}", ext, source_axis=src) for sfx, ext, _ in specs)
    order = list(range(len(specs)))
    if masked and interleave_when_masked:
        order = order[::-1]
    weight: dict[int, int] = {}
    suffix = 1
    for i in reversed(order):
        weight[i] = suffix
        suffix *= specs[i][1]
    expr = Var(a_b.name) * Literal(per_block, "int")
    for i in order:
        if specs[i][2]:  # emit_sigma (ATOM lanes excluded)
            expr = expr + _scaled(inner[i].name, weight[i])
    bound = axis.extent.expr if masked else None
    return a_b, inner, expr, bound


def _rebracket_k(
    stmts: tuple[Stmt, ...],
    target_names: frozenset[str],
    *,
    bk: int,
    fk: int = 1,
    unit: int = 1,
    coop_axis: str | None = None,
    grid: tuple[Axis, int] | None = None,
    thread: Axis | None = None,
    masking: str = "none",
) -> tuple[Stmt, ...]:
    """Re-bracket every contraction loop named in ``target_names`` into the ``K_o``
    (serial-outer) / ``K_i`` (stage-inner) tower — the ONE K decomposition behind the
    scalar, MONOID, and warp tiers. **Recursive**: a nested contraction is towered
    before its enclosing loop. The σ-map is

        K → K_o·(u·bk·fk) + K_f·(u·bk) + K_i·u + K_c     ( + K_s·(k_o_ext·u·bk·fk) )

    where ``u`` is the inner ``unit`` on the PRIMARY axis (``coop_axis``) and ``1``
    elsewhere; ``K_f`` only when ``fk > 1``; the thread partition lane ``K_c`` only on
    ``coop_axis`` when ``thread`` is set; the grid split-K axis ``K_s`` prepended when
    ``grid`` is set. Params select the tier:

    - **scalar** (matmul / flat SEMIRING): ``unit=1``, ``grid=(K_s, splitk)`` for
      split-K, ``thread=None``, ``masking="carrier"`` (a no-op on the static path — it
      only fires on a symbolic axis, closing the latent symbolic-scalar gap the legacy
      ``_replace_k_scalar`` left: a plain ``//`` with no mask).
    - **MONOID** (cooperative reduce / streaming flash): ``unit=br`` on ``coop_axis``,
      ``thread=K_c`` (the ``br``-lane cooperative partition), ``masking="carrier"``.
    - **warp** (tensor-core): ``unit=atom_k``, ``fk=1``, no partition,
      ``masking="downstream"`` (a symbolic K ceil-divides but is NOT body-masked —
      ``kernel/005`` / ``_stage_expand`` zero-fill the partial slab instead).

    ``masking``: ``"carrier"`` clamps a symbolic reduce's load index for a safe read and
    folds the carrier identity past the bound (``_mask_carrier``), and guards a symbolic
    map loop's store with a boundary ``Cond``; ``"downstream"`` ceil-divides ``K_o`` but
    leaves the body unmasked; ``"none"`` is the static-only ``//`` (no ceil-div)."""
    out: list[Stmt] = []
    for s in stmts:
        if not (isinstance(s, Loop) and s.axis.name in target_names):
            out.append(s)
            continue
        kn = s.axis.name
        src = s.axis.source_axis or s.axis
        is_primary = coop_axis is None or kn == coop_axis
        u = unit if is_primary else 1
        kc = thread if (thread is not None and is_primary) else None
        # The cross-CTA split-K ``K_s`` GRID partition rides the PRIMARY axis only — the
        # one cross-execution-unit partition (the matmul's single K, the flash KV stream),
        # never an inner contraction (the QK^T D-reduce of a streaming flash, which is the
        # carrier's hinge, not a partitionable axis).
        kg = grid if (grid is not None and is_primary) else None
        stride = u * bk * fk
        ext = s.axis.extent
        symbolic = not ext.is_static
        if symbolic:
            k_o_ext = ext.ceil_div(stride)
        elif kg is not None:
            k_o_ext = ext.as_static() // (kg[1] * stride)
        else:
            k_o_ext = ext.as_static() // stride
        k_o = Axis(f"{kn}_o", k_o_ext, source_axis=src)
        k_i = Axis(f"{kn}_i", bk, source_axis=src)
        expr = Var(k_o.name) * Literal(stride, "int")
        k_f = None
        if fk > 1:
            k_f = Axis(f"{kn}_f", fk, source_axis=src)
            expr = expr + Var(k_f.name) * Literal(u * bk, "int")
        expr = expr + _scaled(k_i.name, u)
        if kc is not None:
            expr = expr + Var(kc.name)
        if kg is not None:
            expr = Var(kg[0].name) * Literal(k_o_ext * stride, "int") + expr
        sigma_k = Sigma({kn: expr})
        inner = _rebracket_k(
            tuple(s.body), target_names, bk=bk, fk=fk, unit=unit, coop_axis=coop_axis, grid=grid, thread=thread, masking=masking
        )
        if symbolic and masking == "carrier":
            bound = ext.expr
            decoded_k = sigma_k.apply(Var(kn))
            pred = BinaryExpr("<", decoded_k, bound)
            if s.is_reduce:
                clamp = TernaryExpr(cond=pred, if_true=decoded_k, if_false=BinaryExpr("-", bound, Literal(1, "int")))
                body_c = tuple(c.rewrite(_identity_rename, Sigma({kn: clamp})) for c in inner)
                new_body = _mask_carrier(body_c, pred)
            else:
                body_u = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
                new_body = (Cond(cond=pred, body=Body(body_u)),)
        else:
            new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in inner)
        if k_f is not None:
            new_body = (RegisterTile(axes=(k_f,), body=Body(new_body), reduce=s.is_reduce),)
        out.extend(_wrap_tower([(k_i, Role.STAGE_INNER), (k_o, Role.SERIAL_OUTER)], new_body))
    return tuple(out)


def _apply_masked_guards(body: tuple, bounds: list, sigma_outer: Sigma) -> tuple:
    """Wrap ``body`` in a boundary ``Cond`` per masked free axis — the derived
    store guard (masked-ness is ``real_extent`` vs
    tile, a derived ``Cond``). Mirrors ``materialize._assemble`` exactly, so the
    σ-split + guarded body stays byte-identical."""
    for name, bound in bounds:
        pred = sigma_outer.reduce(Var(name), SimplifyCtx({}))
        body = (Cond(cond=BinaryExpr("<", pred, bound), body=Body(body)),)
    return body


def _k_s_axis(dag: IterDag, knobs: dict, target_names: frozenset[str]) -> Axis | None:
    """The split-K GRID axis ``K_s`` (the cross-CTA partition factor), or ``None``
    when there is no contraction (``MAP``) or ``SPLITK == 1``. Computed identically
    by the reduce-decomp body move (it σ-maps ``K`` through ``K_s``) and the
    free-tile move (it binds ``K_s`` GRID) — a pure function of the knobs, so the
    two agree without sharing state."""
    if not target_names:
        return None
    splitk = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)]).cta
    if splitk <= 1:
        return None
    kax = dag.k_node.loop.axis
    return Axis(f"{kax.name}_s", splitk, source_axis=kax.source_axis or kax)


# === The per-pass body moves. ===
# ``build_dag`` is no longer a monolith called at one site: it is the COMPOSITION of
# the incremental moves the enumeration passes apply one at a time to the stored,
# knob-invariant algorithm. ``010_build`` seeds the logical block; ``060_reduce_tile``
# applies ``reduce_decomp`` (the K re-bracket); ``100_register_tile`` applies
# ``free_tile`` (the free-axis σ-split — needs both the thread + register knobs, so it
# is the one free-axis body move, at the last free fork). The composition order
# (reduce **then** free) is the reverse of the old monolith (free then reduce), but
# the two σ-rewrites touch disjoint axis sets (K vs the free N/M), so they commute and
# the built block is byte-identical — the migration oracle.


def seed_graph(dag: IterDag, *, kernel_name: str, buffers: dict | None = None) -> TileGraph:
    """The logical (un-tiled) algorithm: one ``Block`` whose ``compute`` is the DAG's
    inner body verbatim and whose ``domain`` / ``Schedule`` are empty. The moves below
    refine it in place; nothing is built all-at-once."""
    block = Block(name=kernel_name, domain=(), compute=Body(tuple(dag.inner_body)))
    return TileGraph(name=kernel_name, buffers=dict(buffers or {}), blocks=(block,), schedule=Schedule(binding={}))


def reduce_decomp(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The reduce-decomposition body move (``060_reduce_tile``): re-bracket each
    contraction axis named in ``target_names`` into the ``K_o`` (serial-outer) /
    ``K_i`` (stage-inner) tower with an optional ``K_f`` strip-mine, σ-mapped through
    the split-K factor ``K_s`` — embedded directly in ``compute``. No-op for a ``MAP``
    nest. Touches only the body (the ``K_s`` GRID binding is added by ``free_tile``)."""
    if not target_names:
        return graph
    k_s = _k_s_axis(dag, knobs, target_names)
    block = graph.blocks[0]
    d = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)])
    compute = _rebracket_k(
        tuple(block.compute),
        target_names,
        bk=d.serial,
        fk=d.fold,
        unit=1,
        grid=(k_s, d.cta) if k_s is not None else None,
        masking="carrier",
    )
    return replace(graph, blocks=(replace(block, compute=Body(compute)), *graph.blocks[1:]))


def free_tile(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The free-axis ``tile_axis`` body move (``100_register_tile`` — the last free
    fork, where both the thread + register knobs are pinned): split each free
    (``PARALLEL`` / ``MAP``) axis ``A → A_b·(T·R) + A_t·R + A_r``, σ-rewriting the
    body, laying ``A_b, A_t, A_r`` into ``Block.domain`` (bound GRID / THREAD /
    REGISTER) with the split-K ``K_s`` + extra-outer axes trailing GRID, and wrapping
    each masked free axis in its boundary ``Cond``. This is the only free-axis split;
    ``090_thread_tile`` just pins the thread knob (the layout needs the register knob
    too, so a byte-identical split can't be staged across the two forks)."""
    inner_n, outer_m, extra_outer = _free_axes(dag)
    n_par, n_reg = fam.dec_split(knobs[fam.split_key(inner_n.name)])
    free_specs = [(inner_n, n_par, n_reg, True)]
    if outer_m is not None:
        m_par, m_reg = fam.dec_split(knobs[fam.split_key(outer_m.name)])
        free_specs.append((outer_m, m_par, m_reg, False))

    sigma_map: dict = {}
    bounds: list = []
    domain: list = []
    binding: dict[str, Binding] = {}
    for axis, thread, reg, interleave in free_specs:
        a_b, (a_t, a_r), expr, bound = _split_axis(axis, [("t", thread, True), ("r", reg, True)], interleave_when_masked=interleave)
        sigma_map[axis.name] = expr
        if bound is not None:
            bounds.append((axis.name, bound))
        domain.extend((a_b, a_t, a_r))
        binding[a_b.name] = Binding.GRID
        binding[a_t.name] = Binding.THREAD
        binding[a_r.name] = Binding.REGISTER

    k_s = _k_s_axis(dag, knobs, target_names)
    if k_s is not None:
        domain.append(k_s)
        binding[k_s.name] = Binding.GRID
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    sigma_outer = Sigma(sigma_map)
    block = graph.blocks[0]
    compute = tuple(s.rewrite(_identity_rename, sigma_outer) for s in block.compute)
    compute = _apply_masked_guards(compute, bounds, sigma_outer)
    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(compute))
    return replace(
        graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding={**graph.schedule.binding, **binding})
    )


# === MONOID build move — one move for the cooperative reduce AND the streaming flash ===
# A ``MONOID`` nest — a plain reduce (softmax LSE / rmsnorm stat / mean / max) **or** a
# streaming-flash nest (online-softmax over a nested QK^T contraction) — lowers through
# the SAME body move: σ-split the free axes (``free_tile``, register forced to 1) + apply
# the reduce-decomposition tower to **each** contraction axis the DAG exposes. A flat
# monoid exposes one contraction; a nested (streaming) monoid exposes the outer KV stream
# plus the inner QK^T reduce — the same move per axis, the cooperative ``K_c`` THREAD lane
# (the carrier's commutative-licensed partition, ``BR`` lanes per row) placed on the
# PRIMARY axis (``dag.k_node``) and the rest serial. The cross-thread / cross-lane combine
# is NOT in the body — the carrier's ``axes`` (``Accum.axes`` / ``Monoid.axes``) carry
# ``K_c`` through σ and ``kernel/100_materialize_tile`` + ``kernel/_combine`` synthesize
# the warp-shuffle / tree / online-softmax-rescale from ``carrier.axes ∩ ThreadTile``.


def _mask_partial(partial: str, op: ElementwiseImpl, dtype: DataType, pred: object) -> tuple[str, list[Stmt]]:
    """Neutralize one folded ``partial`` past a masked-K boundary: ``Init(kid, op)``
    (the op's neutral element — ``0`` for add, ``-inf`` for max) +
    ``Select(km, partial if pred else kid)``. Returns the masked name + the two
    stmts to splice before the carrier (the Load was already index-clamped for a
    safe read; this only neutralizes the contribution)."""
    ident, masked = f"{partial}_kid", f"{partial}_km"
    return masked, [
        Init(name=ident, op=op, dtype=dtype),
        Select(
            name=masked,
            branches=(SelectBranch(value=partial, select=pred), SelectBranch(value=ident, select=Literal(1, "int"))),
        ),
    ]


def _mask_carrier(body: tuple[Stmt, ...], pred: object) -> tuple[Stmt, ...]:
    """Mask a symbolic-K reduce loop's body past the runtime bound, dispatching on
    the carrier the loop folds. A scalar ``Accum`` masks its folded ``value`` to the
    op's identity. A streaming ``Monoid`` (flash online-softmax) masks its score
    (``partial[0]``) to ``-inf`` (maximum's identity) so the masked key contributes
    nothing (``max(m, -inf) = m``, ``exp(-inf) = 0``) while the in-place state update
    stays unconditional; ``partial[1:]`` (the value) is left untouched. Both share
    the :func:`_mask_partial` skeleton; the carrier stays a direct child of the
    reduce loop (``is_reduce`` + the cross-thread combine intact)."""
    out: list[Stmt] = []
    for c in body:
        if isinstance(c, Accum):
            masked, stmts = _mask_partial(c.value, c.op, c.dtype or F32, pred)
            out.extend([*stmts, replace(c, value=masked)])
        elif isinstance(c, Monoid):
            masked, stmts = _mask_partial(c.partial[0], ElementwiseImpl("maximum"), F32, pred)
            out.extend([*stmts, replace(c, partial=(masked, *c.partial[1:]))])
        else:
            out.append(c)
    return tuple(out)


def _realize_serial_monoid(stmts: tuple[Stmt, ...], carrier: Monoid) -> tuple[Stmt, ...]:
    """Replace the ``carrier`` ``Monoid`` (matched by its carried state) wherever it sits in ``stmts``
    — recursing into ``Loop`` bodies — with the ``ScalarCombiner``-projected serial merge (the fold as
    ``Assign`` / ``Reassign`` statements). The carried-state ``Init`` seeds live above the reduce loop
    already (the recognizer placed them), so only the per-iteration merge is realized here."""
    merge = ScalarCombiner().combine(carrier).merge

    def walk(body: tuple[Stmt, ...]) -> list[Stmt]:
        out: list[Stmt] = []
        for s in body:
            if isinstance(s, Monoid) and s.state == carrier.state:
                out.extend(merge)
            elif isinstance(s, Loop):
                out.append(replace(s, body=Body(tuple(walk(tuple(s.body))))))
            else:
                out.append(s)
        return out

    return tuple(walk(stmts))


def monoid_build(graph: TileGraph, dag: IterDag, knobs: dict, *, target_names: frozenset[str]) -> TileGraph:
    """The MONOID build move — one move for the cooperative reduce (softmax / rmsnorm
    / mean / max) AND the streaming flash (online-softmax over a nested QK^T). Apply
    the reduce-decomposition tower to **each** contraction axis the DAG exposes
    (``_rebracket_k`` with ``masking="carrier"``, recursive for a nested stream) with the cooperative ``K_c``
    THREAD lane on the primary axis, then σ-split the free axes (``free_tile``, register
    forced to 1 by the caller). The ``K_c`` axis is laid FIRST in ``Block.domain`` so it
    sits innermost in the THREAD tier (fastest threadIdx bits); ``assemble`` reconstructs
    the tower from ``domain`` + ``Schedule.binding`` and the downstream combine rides the
    carrier's ``axes``.

    The caller pins the knobs per regime: a flat cooperative reduce searches ``(bk, fk,
    br)``; a streaming flash defaults ``bk = fk = 1`` and ``br`` over the static KV axis
    (cooperative-KV). ``BR > 1`` lays the cooperative lane — for the flat reduce the
    carrier's commutative partition, for the stream each lane's strided slice into its own
    online-softmax partial, merged at materialize via the carrier's ``combine_states``."""
    d = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)])
    bk, fk, br = d.serial, d.fold, d.coop
    targets = target_names or frozenset({dag.k_node.loop.axis.name})

    kax = dag.k_node.loop.axis
    # The cooperative lane rides the primary axis only. A symbolic streaming axis stays
    # serial (the offer set already returns ``br = 1`` for it); a flat symbolic reduce
    # tiles the masked K at the hint and may carry ``br`` (ceil-div, masked fill).
    k_c = Axis(f"{kax.name}_c", br, source_axis=kax.source_axis or kax) if br > 1 else None

    # Cross-CTA split-K of the cooperative reduce: thread the ``K_s`` GRID partition through
    # the K re-bracket when ``cta > 1`` (the additive ``Accum`` carrier — the offer set gates
    # the legality). The same ``grid=(K_s, cta)`` the matmul ``reduce_decomp`` uses; ``free_tile``
    # binds the ``K_s`` GRID axis (it reads ``cta`` off the same knobs). The cross-CTA producer
    # mirrors the intra-CTA cooperative ``K_c`` lane one partition level up.
    k_s = _k_s_axis(dag, knobs, targets)
    block = graph.blocks[0]
    compute_in = tuple(block.compute)

    # A serial (br=1, single-CTA), non-twisted ``Monoid`` reduce — the fused online-softmax ``(m, d)``
    # carrier — is realized through the carrier-generic ``ScalarCombiner`` (the scalar tier of the
    # combiner, sibling of the fragment ``MmaTwist``), the SAME ``combine()`` the flash tiers drive,
    # instead of being left a ``Monoid`` for ``render_merge_program``. Its ``Init`` seeds are already
    # placed (the recognizer emits them) and a serial reduce has no cross-thread combine. Cooperative
    # (``br > 1``) / split-K ``Monoid``s keep the carrier for ``emit_combine``; flat ``Accum`` reduces
    # and twisted carriers (flash → ``chain_build`` / ``warp_chain_build``) are untouched.
    carrier = dag.reduction.carrier if dag.reduction is not None else None
    if br == 1 and k_s is None and isinstance(carrier, Monoid) and len(carrier.partial) == 1:
        compute_in = _realize_serial_monoid(compute_in, carrier)

    compute = _rebracket_k(
        compute_in,
        targets,
        bk=bk,
        fk=fk,
        unit=br,
        coop_axis=kax.name,
        thread=k_c,
        grid=(k_s, d.cta) if k_s is not None else None,
        masking="carrier",
    )
    g = free_tile(replace(graph, blocks=(replace(block, compute=Body(compute)),)), dag, knobs, target_names=target_names)
    if k_c is not None:
        # Lay K_c FIRST in domain (innermost THREAD bits), so the segmented cross-lane
        # combine stays an aligned intra-warp segment.
        nb = g.blocks[0]
        g = replace(
            g,
            blocks=(replace(nb, domain=(k_c, *nb.domain)),),
            schedule=replace(g.schedule, binding={**g.schedule.binding, k_c.name: Binding.THREAD}),
        )
    return g


# === The carried-contraction-chain build move (the shared-axis reduce_decomp) ===
# A ``MONOID(SEMIRING)``
# nest — a twisted carrier streaming over a nested contraction whose combine embeds a
# SECOND contraction (flash: online softmax over QK^T, with P@V embedded in
# ``O = O·α + p·v``) — is restructured so the **P@V free output ``d``** rides a register
# vector ``O[BM, D]`` *inside* the stream and the score is computed ONCE per KV step and
# **shared** across ``d`` (the INLINE score edge), instead of being recomputed per ``d``
# block. This is the FA-2 nest the warp tier needs: the score becomes a register
# fragment, the inner QK^T and the embedded P@V become two cells, and the twisted
# carrier splits into a **scalar stats carrier** (row max / denom, ``d``-invariant) +
# a **register-tiled accumulation carrier** (``O[d]``, reading the stats' rescale ``α``
# and probability ``p``). Not flash-specific: it dispatches on the compositional
# ``MONOID(SEMIRING)`` algebra the chain view exposes (``IterDag.chain``), the
# ``MONOID(SEMIRING)`` analog of ``monoid_build``.


def chain_build(graph: TileGraph, dag: IterDag, knobs: dict) -> TileGraph:
    """The shared-axis reduce_decomp (Phase 1c): restructure a ``MONOID(SEMIRING)``
    chain nest into the FA-2 form — the P@V output ``d`` as a register vector ``O[d]``
    inside the stream, the score computed once per KV step (the INLINE score edge,
    shared across ``d``), the twisted carrier split into a scalar stats carrier + a
    register-tiled accumulation carrier.

    ``d`` becomes a **REGISTER domain axis** (so the tower wraps the whole block once);
    the register-replication pass (``kernel/010_split_register_axes``) then replicates
    ONLY the statements that depend on the ``d`` var — the value load + the accumulation
    carrier ``O[d]`` — while the score + the scalar stats carrier (which the split made
    ``d``-independent) pass through shared, once per KV step. ``m`` binds THREAD, the
    shared axes GRID. The reduce axes stay serial (``BN=1`` per step — the per-KV online
    fold)."""
    reduction = dag.reduction
    if reduction is None or reduction.inner is None:
        raise ValueError("chain_build requires a streaming MONOID(SEMIRING) reduction (inner contraction present)")
    block = graph.blocks[0]
    body = tuple(block.compute)
    carrier = reduction.carrier
    value_name = carrier.partial[1]

    # Locate the KV stream loop + its carrier / value load.
    kv_loop = next(s for s in body if isinstance(s, Loop) and s.axis.name == reduction.hinge_name)
    kv_body = tuple(kv_loop.body)
    value_load = next(s for s in kv_body if isinstance(s, Load) and s.name == value_name)
    monoid = next(s for s in kv_body if isinstance(s, Monoid))
    prefix = tuple(s for s in kv_body if s is not value_load and s is not monoid)  # d-invariant score chain

    # Realize the carrier's online-softmax combine at the scalar tier through the shared
    # ``ScalarCombiner`` — the same ``combine()`` orchestration ``MmaTwist`` drives at the fragment
    # tier (``carry_scope_from_graph``) — instead of leaving raw split stats/accum ``Monoid``s for
    # ``render_merge_program``. The FA-2 scalar flash and the warp flash now share one carrier-combine
    # realizer (the user-visible split: stats fold → rescale ``O·α`` → consume ``p·v`` → normalize
    # ``O/l`` is generated once, in the combiner). The kv reduce stays serial (no cross-thread
    # combine), and ``010_split_register_axes`` replicates the d-dependent rescale/consume ``Assign``s
    # over ``O[d]`` exactly as it did the accum ``Monoid`` (generic ``rewrite`` replication).
    fs = ScalarCombiner().combine(carrier)
    m_axis, d_axis, grid = chain_free_axes(reduction, dag)  # walk the composition for the geometry — no stored view

    # ``d`` -> a REGISTER domain axis; ``m`` -> a THREAD tile (reg forced to 1).
    d_r = Axis(f"{d_axis.name}_r", d_axis.extent, source_axis=d_axis.source_axis or d_axis)
    m_split = knobs.get(fam.split_key(m_axis.name))
    m_thread = fam.dec_split(m_split)[0] if m_split is not None else 1
    m_b, (m_t, m_r), m_expr, m_bound = _split_axis(m_axis, [("t", m_thread, True), ("r", 1, True)], interleave_when_masked=True)

    # The carrier-state Inits + epilogue stay where they are; the register-replication
    # pass derives which (the ``O`` init / normalize / write) depend on ``d``.
    head_inits = tuple(s for s in body if isinstance(s, Init))
    epilogue = tuple(s for s in body if isinstance(s, (Assign, Write)))
    new_kv = replace(kv_loop, body=Body((*prefix, *fs.merge, value_load, *fs.rescale, *fs.consume)))
    compute: tuple[Stmt, ...] = (*head_inits, new_kv, *epilogue)

    # σ-rewrite: ``d`` -> the register axis var; ``m`` -> its block/thread split.
    sigma = Sigma({d_axis.name: Var(d_r.name), m_axis.name: m_expr})
    compute = tuple(s.rewrite(_identity_rename, sigma) for s in compute)
    if m_bound is not None:
        compute = _apply_masked_guards(compute, [(m_axis.name, m_bound)], sigma)

    # Domain ordered register..thread..grid (inner→outer) so the tower groups
    # GridTile > ThreadTile > RegisterTile cleanly; ``d_r`` is the innermost register.
    domain: list[Axis] = [d_r, m_r, m_t, m_b]
    binding: dict[str, Binding] = {
        d_r.name: Binding.REGISTER,
        m_r.name: Binding.REGISTER,
        m_t.name: Binding.THREAD,
        m_b.name: Binding.GRID,
    }
    for lp in reversed(grid):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID
    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(compute))
    return replace(graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding=binding))


# === Warp-tier flash build move (σ-tile + atomize the two chained contractions). ===
# ``warp_chain_build`` produces the warp-tier flash as an **atomized streaming TileGraph** the
# generic ``assembly._assemble.carry_scope_from_graph`` walk realizes (the fragment-tier softmax /
# scale / mask / C→A handoff / epilogue). It σ-tiles + atomizes the QK^T cell (``out_index``
# fragment output, transposed-B) and the P@V cell (``frag_a``, canonical-B) via the generic
# ``atomize_cell`` — no hand-authored ``Mma``, no flat 1-D authoring (the render flattens the 4D
# loads via the buffer strides). The kv-stream is the ``Schedule.carry`` axis; the score→P→A handoff
# is a staged ``flash_pv_smem`` edge. This is the deployed warp-flash path (it replaced the
# hand-assembled ``realize_flash``).


def warp_chain_build(op) -> TileGraph:  # noqa: ANN001 — op: TileGraphOp (avoid the ir↔passes import)
    """σ-tile the warp-tier flash's two chained contractions into an **atomized streaming**
    ``TileGraph`` the generic ``carry_scope_from_graph`` walk realizes.

    The seed's logical 4D cells are σ-tiled to the warp geometry (16 query rows / warp, the kv
    stream a 16-key serial-outer carry, D re-bracketed at ``atom_k``) and fused via the generic
    ``atomize_cell`` (the render flattens the 4D loads via the buffer strides — no flat 1-D
    authoring). The QK^T inner D-reduce → ``2`` transposed-B ``Mma`` cells (the 16-col score =
    ``16/atom_n`` N-atoms, ``out_index`` = the INLINE score ``(m, kv)``); the split-carrier P@V →
    ``D/atom_n`` ``frag_a`` canonical-B ``Mma`` cells (``A`` = the probability fragment, ``B`` = V).
    The kv stream is ``Schedule.carry``; the score→A handoff is a staged ``flash_pv_smem`` edge. The
    block ``domain`` lays the warp tile (query block GRID, the grid axes GRID);
    ``carry_scope_from_graph`` reads these AtomTiles + the carrier and assembles the ``CarryScope``
    (softmax / scale / mask / handoff / epilogue) — the path that replaced ``realize_flash``."""
    tg = op.tilegraph
    block = tg.blocks[0]
    reduction = op.dag.reduction
    kv = reduction.hinge_name
    kv_loop = next(s for s in block.compute if isinstance(s, Loop) and s.axis.name == kv)
    qkt_loop = next(s for s in kv_loop.body if isinstance(s, Loop))  # the QK^T D-reduce loop
    qkt_body, a3_name = tuple(qkt_loop.body), qkt_loop.axis.name

    # Geometry off the graph, no flash descriptor: the head dim ``D`` is the QK^T reduce extent; the
    # operand dtype (→ which 16-bit mma atom) is read off a QK^T load's gmem buffer.
    D = qkt_loop.axis.extent.as_static()
    dtype = op.buffers[next(s for s in qkt_body if isinstance(s, Load)).input].dtype
    atom = ATOM_REGISTRY["mma_m16n8k16_bf16" if dtype == BF16 else "mma_m16n8k16_f16"]
    atom_m, atom_n, atom_k = atom.shape

    value_load = next(s for s in kv_loop.body if isinstance(s, Load) and s.names[0] == reduction.carrier.partial[1])
    m_axis, d_axis, grid = chain_free_axes(reduction, op.dag)  # walk the composition for the geometry — no stored view

    # σ-tile geometry. m (query row) → m_b·atom_m (GRID block, atom owns the 16 lane); kv (stream)
    # → kv_b·16 (serial-outer carry, the 16-key tile owned by atom_n/atom_k); the QK^T 16-col score
    # = ``16/atom_n`` N-atoms (nt); the P@V output D = ``D/atom_n`` N-atoms (n).
    m_b, _m_a, m_expr, m_bound = _split_axis(m_axis, [("a", atom_m, False)])
    kv_b, _kv_a, kv_expr, kv_bound = _split_axis(kv_loop.axis, [("a", 16, False)])
    nt_count, nd, kt = 16 // atom_n, D // atom_n, D // atom_k
    # The score-masking ``Select`` the recognizer placed (``v2 = select(col<=row, score, -inf)``) —
    # the causal mask's structural representation. Carried THROUGH to the graph (below) so assembly
    # reads causality off its presence, not off a flag. It is the score mask's analog of the carrier
    # ``Monoid`` (the softmax's structural representation).
    score_mask = next((s for s in kv_loop.body if isinstance(s, Select)), None)

    # produce — the QK^T D-reduce, σ-tiled per N-atom (nt) → a transposed-B Mma over the INLINE
    # score (m, kv·16 + nt·atom_n). The D reduce becomes a ``ko`` K-tile loop (the load's K base is
    # ``ko·atom_k``, the atom spans the atom_k lane) — a degenerate 1-trip loop at D==atom_k keeps
    # ``ko`` a *defined* runtime var (vs a raw D var the render can't strip into the fragment).
    ko = "ko"
    produce: list[Stmt] = []
    for nt in range(nt_count):
        n_base = _fadd(kv_expr, nt * atom_n)
        sig = Sigma({m_axis.name: m_expr, kv: n_base, a3_name: _fmul(Var(ko), atom_k)})
        cell = tuple(s.rewrite(_identity_rename, sig) for s in qkt_body)
        fused = atomize_cell(cell, atom=atom, k_name=ko, write=None, out_index=(m_expr, n_base))
        # kt>1: a real K_o serial loop (the atom spans atom_k per step). kt==1: the whole D is one
        # atom — strip ``ko`` to 0 and emit the cell directly (the form 005_lower_atom_tile lowers).
        if kt > 1:
            body: tuple[Stmt, ...] = (SerialTile(axis=Axis(ko, Dim(kt)), body=Body(fused), kind="plain"),)
        else:
            body = tuple(s.rewrite(_identity_rename, Sigma({ko: Literal(0, "int")})) for s in fused)
        produce.append(AtomTile(axes=(Axis("qm", Dim(atom_m)), Axis("qn", Dim(atom_n))), body=Body(body), atom=atom))

    # consume — the split-carrier P@V cell, σ-tiled per output N-atom (n) → a frag_a canonical-B Mma
    # (A = the probability fragment, B = V), accumulating O[d]. The kv tile (16 keys) is one atom_k:
    # a 1-trip ``kpv`` K loop (load K base ``a3_b·16 + kpv·atom_k``, the atom spans the 16 keys).
    _stats, accum, d_state = split_carrier(reduction.carrier, reduction.carrier.partial[1])
    prob = next(a.args[0] for a in accum.merge if a.op.name == "multiply" and d_state not in a.args)  # p in p·v
    kpv = "kpv"
    consume: list[Stmt] = []
    for n in range(nd):
        # ``kpv`` is only the K var atomize needs to fuse the frag_a cell; the 16 keys are one atom,
        # so strip it to 0 post-fuse (the atom spans them) → the Mma+V-load form 005 lowers.
        sig = Sigma({d_axis.name: Literal(n * atom_n, "int"), kv: _fadd(kv_expr, _fmul(Var(kpv), atom_k))})
        vload = replace(value_load.rewrite(_identity_rename, sig), names=("vv",))
        cell = (vload, Assign(name="pv", op=ElementwiseImpl("multiply"), args=(prob, "vv")), Accum(name=d_state, value="pv"))
        fused = atomize_cell(cell, atom=atom, k_name=kpv, write=None, frag_a=True)
        body = tuple(s.rewrite(_identity_rename, Sigma({kpv: Literal(0, "int")})) for s in fused)
        consume.append(AtomTile(axes=(Axis("am", Dim(atom_m)), Axis("an", Dim(atom_n))), body=Body(body), atom=atom))

    # The streaming carry body: produce QK^T cells → (the score-mask ``Select``, when causal) → the
    # full twisted carrier (the online-softmax Monoid — the walk splits it via
    # realize_fragment_softmax) → consume P@V cells, wrapped in the kv-stream serial-outer carry.
    # carry_scope_from_graph realizes the fragment phases (softmax / scale / mask / C→A handoff /
    # epilogue) around these AtomTiles, reading causality off the ``Select``'s presence.
    mask = (score_mask,) if score_mask is not None else ()
    new_kv = SerialTile(axis=kv_b, body=Body((*produce, *mask, reduction.carrier, *consume)), kind="serial_outer")
    head_inits = tuple(s for s in block.compute if isinstance(s, Init))
    epilogue = tuple(s for s in block.compute if isinstance(s, (Assign, Write)))
    compute = (*head_inits, new_kv, *epilogue)

    domain = (m_b, *(lp.axis for lp in reversed(grid)))
    binding = {m_b.name: Binding.GRID, **{lp.axis.name: Binding.GRID for lp in grid}}
    handoff_edge = Edge(src=block.name, dst=block.name, buffer="flash_pv_smem")
    schedule = replace(
        tg.schedule,
        binding={**tg.schedule.binding, **binding},
        carry=frozenset({kv_b.name}),
        staged={**tg.schedule.staged, handoff_edge: Transport.SYNC},
    )
    new_block = replace(block, domain=domain, compute=Body(compute))
    return replace(tg, blocks=(new_block, *tg.blocks[1:]), schedule=schedule)


# === Warp-tier (tensor-core ``atomize``) build move. ===
# The warp tower is the same kind of body move as the scalar ``free_tile`` /
# ``reduce_decomp``, but it splits each output axis FOUR ways
# (``A → A_b·(W·R·atom) + A_w·(R·atom) + A_r·atom``, bound GRID/WARP/REGISTER/ATOM)
# and re-brackets K at ``atom_k`` granularity, then FUSES the cell
# ``[Load,Load,mul,Accum]`` into one ``Mma`` via ``_atom.atomize_cell`` (the atom
# layer's body edit — provenance-agnostic, naming A/B by SSA value; ``Block.atom``
# then derives from that ``Mma``). The staging geometry below (free-axis σ-split,
# K re-bracket, gmem I/O) is the matmul-staging layer ``warp_build`` composes with
# the atom layer. ``assemble`` materializes the AtomTile/WarpTile tower around it
# via the shared ``_free_layers`` + ``_wrap_tower``.


def warp_build(graph: TileGraph, dag: IterDag, knobs: dict, *, atom: Atom) -> TileGraph:
    """The warp-tier ``atomize`` build move: take the logical seed graph and σ-split
    the free axes four ways (GRID/WARP/REGISTER/ATOM) + re-bracket K at ``atom_k``
    granularity + fuse the cell into an ``Mma``, laying the domain axes + bindings.
    ``assemble`` reconstructs the AtomTile/WarpTile tower from ``domain`` + ``Mma``."""
    atom_m, atom_n, atom_k = atom.shape
    bk = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)]).serial

    inner_n, outer_m, extra_outer = _free_axes(dag)
    wn, fn = fam.dec_split(knobs[fam.split_key(inner_n.name)])
    wm, fm = fam.dec_split(knobs[fam.split_key(outer_m.name)])
    # Four-way warp split: the ATOM lane carries the cell extent but emits no σ term.
    n_b, (n_w, n_r, n_a), n_expr, n_bound = _split_axis(inner_n, [("w", wn, True), ("r", fn, True), ("a", atom_n, False)])
    m_b, (m_w, m_r, m_a), m_expr, m_bound = _split_axis(outer_m, [("w", wm, True), ("r", fm, True), ("a", atom_m, False)])
    sigma_outer = Sigma({inner_n.name: n_expr, outer_m.name: m_expr})

    block = graph.blocks[0]
    new_inner = tuple(s.rewrite(_identity_rename, sigma_outer) for s in block.compute)
    kn = dag.k_node.loop.axis.name
    new_inner = _rebracket_k(new_inner, frozenset({kn}), bk=bk, fk=1, unit=atom_k, coop_axis=kn, masking="downstream")
    new_inner = atomize_cell(new_inner, atom=atom, k_name=None, write=None)
    bounds = [(n, b) for n, b in ((inner_n.name, n_bound), (outer_m.name, m_bound)) if b is not None]
    new_inner = _apply_masked_guards(new_inner, bounds, sigma_outer)

    # Domain: N then M, each split b/w/r/a. ``_free_layers`` orders the tiers
    # (ATOM innermost … GRID outermost) for ``assemble``; extra-outer trail GRID.
    domain: list[Axis] = []
    binding: dict[str, Binding] = {}
    for a_b, a_w, a_r, a_a in ((n_b, n_w, n_r, n_a), (m_b, m_w, m_r, m_a)):
        domain.extend((a_b, a_w, a_r, a_a))
        binding[a_b.name] = Binding.GRID
        binding[a_w.name] = Binding.WARP
        binding[a_r.name] = Binding.REGISTER
        binding[a_a.name] = Binding.ATOM
    for lp in reversed(extra_outer):
        domain.append(lp.axis)
        binding[lp.axis.name] = Binding.GRID

    new_block = Block(name=block.name, domain=tuple(domain), compute=Body(new_inner))
    return replace(
        graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding={**graph.schedule.binding, **binding})
    )


def build_dag(
    dag: IterDag,
    knobs: dict,
    *,
    kernel_name: str,
    target_names: frozenset[str] = frozenset(),
    buffers: dict | None = None,
) -> TileGraph:
    """The full single-block ``TileGraph`` for a knob choice — now the COMPOSITION of
    the per-pass moves (``seed_graph`` → ``reduce_decomp`` → ``free_tile``), kept as
    one entry point for unit / equivalence callers. The enumeration passes apply the
    same three moves incrementally (F3-b); this wrapper makes the composition explicit
    and is the byte-identity oracle for that distribution."""
    graph = seed_graph(dag, kernel_name=kernel_name, buffers=buffers)
    graph = reduce_decomp(graph, dag, knobs, target_names=target_names)
    return free_tile(graph, dag, knobs, target_names=target_names)
