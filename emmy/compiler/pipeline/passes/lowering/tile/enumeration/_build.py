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

from emmy.compiler.dim import Dim
from emmy.compiler.dtype import BF16, F32, DataType
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, TernaryExpr, Var
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Monoid, Select, SelectBranch, Stmt, Write
from emmy.compiler.ir.stmt.carrier_algebra import split_carrier
from emmy.compiler.ir.tile.ir import (
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
from emmy.compiler.ir.twist import MmaTwist
from emmy.compiler.pipeline.passes.lowering._addr import add as _fadd
from emmy.compiler.pipeline.passes.lowering._addr import mul as _fmul
from emmy.compiler.pipeline.passes.lowering.tile.assembly._tower import Role, _identity_rename, _wrap_tower
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._atom import atomize_cell
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import IterDag, chain_free_axes


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


def _lay_domain(layers: list[tuple[Axis, Binding]]) -> tuple[tuple[Axis, ...], dict[str, Binding]]:
    """Build a block's ``(domain, binding)`` from innermost-first ``(axis, binding)`` layers — the
    shared domain construction of ``warp_build`` / ``build_monoid`` (each tier lays
    its own σ-split axes + a GRID trail). ``domain`` order is load-bearing (the tower nests
    inner→outer); the ``binding`` dict is keyed by name (``structural_key`` sorts it)."""
    return tuple(a for a, _ in layers), {a.name: b for a, b in layers}


def _stream_compute(compute, new_kv: Stmt) -> Body:
    """Wrap a kv-stream carry loop between the carrier-state ``Init`` seeds (above) and the epilogue
    ``Assign`` / ``Write`` (below) — the identical postamble of both chain geometries of
    ``build_monoid`` (the scalar FA-2 and warp branches). The register-replication / fragment realizer derives
    which seeds / epilogue depend on ``d``; the chain restructure only relocates the stream loop."""
    head_inits = tuple(s for s in compute if isinstance(s, Init))
    epilogue = tuple(s for s in compute if isinstance(s, (Assign, Write)))
    return Body((*head_inits, new_kv, *epilogue))


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


def _realize_serial_monoid(stmts: tuple[Stmt, ...], carrier: Monoid, combiner) -> tuple[Stmt, ...]:  # noqa: ANN001
    """Replace the ``carrier`` ``Monoid`` (matched by its carried state) wherever it sits in ``stmts``
    — recursing into ``Loop`` bodies — with the ``ScalarCombiner``-projected serial merge (the fold as
    ``Assign`` / ``Reassign`` statements). The carried-state ``Init`` seeds live above the reduce loop
    already (the recognizer placed them), so only the per-iteration merge is realized here."""
    merge = combiner.combine(carrier).merge

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


def build_monoid(op, knobs: dict, *, combiner: type) -> TileGraph:  # noqa: ANN001 — op: TileGraphOp (avoid the ir↔passes import)
    """The ONE MONOID build move — cooperative reduce, scalar FA-2 stream, OR warp tensor-core flash —
    parametrized by the carrier ``Combiner`` *tier* (the existing ``ir/twist`` combiners, never a
    per-case subclass). ``combiner`` is the only thing that varies between the tiers:

    - ``MmaTwist`` → the **warp tier**: σ-tile + atomize the two chained contractions; the fragment
      online-softmax combine is realized later by assembly's ``carry_scope_from_graph`` FROM the cells
      emitted here (so this move never calls ``combine`` — the tier defers it).
    - ``ScalarCombiner`` → the **scalar tier**: the carrier's ``combine`` is realized in-body. The
      score's placement knob then picks the geometry — an ``INLINE`` score is the FA-2 shared-score
      restructuring (``d`` rides a register vector ``O[d]``, the score computed ONCE per KV step and
      shared across ``d``); otherwise the cooperative ``_rebracket_k`` tower over each contraction axis
      (a flat reduce, or a serial stream).

    The three are the honest 2×2 of (chained pair vs single contraction) × (scalar vs warp), dispatched
    off ``MonoidReduction`` + the combiner — one composition, not three hand-written moves.

    The caller pins the knobs per regime: a flat cooperative reduce searches ``(bk, fk, br)``; a
    streaming flash defaults ``bk = fk = 1`` and ``br`` over the static KV axis (cooperative-KV).
    ``BR > 1`` lays the cooperative ``K_c`` lane (the carrier's commutative partition for the flat
    reduce; each lane's strided online-softmax partial for the stream, merged via ``combine_states``)."""
    dag = op.dag
    reduction = dag.reduction
    carrier = reduction.carrier if reduction is not None else None

    # === warp tier (``combiner is MmaTwist``): σ-tile + atomize the two chained contractions to the
    # warp geometry (16 query rows / warp, the kv stream a 16-key serial-outer carry, D re-bracketed at
    # ``atom_k``) via the generic ``atomize_cell``. The QK^T D-reduce → transposed-B ``Mma`` cells over
    # the INLINE score; the split-carrier P@V → ``frag_a`` canonical-B cells. The fragment combine is
    # realized later by ``carry_scope_from_graph`` FROM these cells (the MmaTwist tier defers it). ===
    if combiner is MmaTwist:
        tg = op.tilegraph
        block = tg.blocks[0]
        kv = reduction.hinge_name
        kv_loop = next(s for s in block.compute if isinstance(s, Loop) and s.axis.name == kv)
        qkt_loop = next(s for s in kv_loop.body if isinstance(s, Loop))  # the QK^T D-reduce loop
        qkt_body, a3_name = tuple(qkt_loop.body), qkt_loop.axis.name
        # Geometry off the graph, no flash descriptor: ``D`` is the QK^T reduce extent; the operand
        # dtype (→ which 16-bit mma atom) is read off a QK^T load's gmem buffer.
        D = qkt_loop.axis.extent.as_static()
        dtype = op.buffers[next(s for s in qkt_body if isinstance(s, Load)).input].dtype
        atom = ATOM_REGISTRY["mma_m16n8k16_bf16" if dtype == BF16 else "mma_m16n8k16_f16"]
        atom_m, atom_n, atom_k = atom.shape
        value_load = next(s for s in kv_loop.body if isinstance(s, Load) and s.names[0] == reduction.carrier.partial[1])
        m_axis, d_axis, grid = chain_free_axes(reduction, dag)  # walk the composition for the geometry
        m_b, _m_a, m_expr, m_bound = _split_axis(m_axis, [("a", atom_m, False)])
        kv_b, _kv_a, kv_expr, kv_bound = _split_axis(kv_loop.axis, [("a", 16, False)])
        nt_count, nd, kt = 16 // atom_n, D // atom_n, D // atom_k
        # The score-masking ``Select`` the recognizer placed (the causal mask's structural form) is
        # carried THROUGH so assembly reads causality off its presence, not off a flag.
        score_mask = next((s for s in kv_loop.body if isinstance(s, Select)), None)
        # produce — the QK^T D-reduce, σ-tiled per N-atom (nt) → a transposed-B Mma over the INLINE
        # score (m, kv·16 + nt·atom_n). The D reduce becomes a ``ko`` K-tile loop of ``kt`` atom_k steps.
        ko = "ko"
        produce: list[Stmt] = []
        for nt in range(nt_count):
            n_base = _fadd(kv_expr, nt * atom_n)
            sig = Sigma({m_axis.name: m_expr, kv: n_base, a3_name: _fmul(Var(ko), atom_k)})
            cell = tuple(s.rewrite(_identity_rename, sig) for s in qkt_body)
            axes = (Axis("qm", Dim(atom_m)), Axis("qn", Dim(atom_n)))
            produce.append(_atom_cell(cell, atom=atom, k_name=ko, kt=kt, axes=axes, out_index=(m_expr, n_base)))
        # consume — the split-carrier P@V cell, σ-tiled per output N-atom (n) → a frag_a canonical-B Mma
        # (A = the probability fragment, B = V), accumulating O[d]. The kv tile (16 keys) is one atom_k.
        _stats, accum, d_state = split_carrier(reduction.carrier, reduction.carrier.partial[1])
        prob = next(a.args[0] for a in accum.merge if a.op.name == "multiply" and d_state not in a.args)  # p in p·v
        kpv = "kpv"
        consume: list[Stmt] = []
        for n in range(nd):
            sig = Sigma({d_axis.name: Literal(n * atom_n, "int"), kv: _fadd(kv_expr, _fmul(Var(kpv), atom_k))})
            vload = replace(value_load.rewrite(_identity_rename, sig), names=("vv",))
            cell = (vload, Assign(name="pv", op=ElementwiseImpl("multiply"), args=(prob, "vv")), Accum(name=d_state, value="pv"))
            axes = (Axis("am", Dim(atom_m)), Axis("an", Dim(atom_n)))
            consume.append(_atom_cell(cell, atom=atom, k_name=kpv, kt=1, axes=axes, frag_a=True))
        # The streaming carry body: produce cells → (the score-mask, when causal) → the full twisted
        # carrier (the walk splits it via realize_fragment_softmax) → consume cells, in the kv carry.
        mask = (score_mask,) if score_mask is not None else ()
        new_kv = SerialTile(axis=kv_b, body=Body((*produce, *mask, reduction.carrier, *consume)), kind="serial_outer")
        compute = _stream_compute(block.compute, new_kv)  # the shared chain postamble (also the scalar FA-2 path)
        domain, binding = _lay_domain([(m_b, Binding.GRID), *[(lp.axis, Binding.GRID) for lp in reversed(grid)]])
        handoff_edge = Edge(src=block.name, dst=block.name, buffer="flash_pv_smem")
        schedule = replace(
            tg.schedule,
            binding={**tg.schedule.binding, **binding},
            carry=frozenset({kv_b.name}),
            staged={**tg.schedule.staged, handoff_edge: Transport.SYNC},
        )
        return replace(tg, blocks=(replace(block, domain=domain, compute=compute), *tg.blocks[1:]), schedule=schedule)

    # === scalar tier (``combiner is ScalarCombiner``): realize the carrier's combine in-body. ===
    graph, target_names = op.tilegraph, op.target_names
    sc = combiner()

    # FA-2 shared-score stream: a real carried-contraction chain whose score is placed INLINE — ``d``
    # rides a REGISTER vector ``O[d]`` and the score is computed ONCE per KV step (shared across ``d``).
    # ``ScalarCombiner.combine`` realizes the online-softmax split (stats fold → rescale ``O·α`` →
    # consume ``p·v`` → normalize ``O/l``) — the SAME combine the warp tier drives at the fragment tier.
    if reduction is not None and reduction.inner is not None and knobs.get(fam.place_key(reduction.score)) == fam.INLINE:
        block = graph.blocks[0]
        body = tuple(block.compute)
        value_name = carrier.partial[1]
        kv_loop = next(s for s in body if isinstance(s, Loop) and s.axis.name == reduction.hinge_name)
        kv_body = tuple(kv_loop.body)
        value_load = next(s for s in kv_body if isinstance(s, Load) and s.name == value_name)
        monoid = next(s for s in kv_body if isinstance(s, Monoid))
        prefix = tuple(s for s in kv_body if s is not value_load and s is not monoid)  # d-invariant score chain
        fs = sc.combine(carrier)
        m_axis, d_axis, grid = chain_free_axes(reduction, dag)  # walk the composition for the geometry
        # ``d`` -> a REGISTER domain axis; ``m`` -> a THREAD tile (reg forced to 1).
        d_r = Axis(f"{d_axis.name}_r", d_axis.extent, source_axis=d_axis.source_axis or d_axis)
        m_split = knobs.get(fam.split_key(m_axis.name))
        m_thread = fam.dec_split(m_split)[0] if m_split is not None else 1
        m_b, (m_t, m_r), m_expr, m_bound = _split_axis(m_axis, [("t", m_thread, True), ("r", 1, True)], interleave_when_masked=True)
        # ``010_split_register_axes`` replicates the d-dependent rescale/consume over ``O[d]``.
        new_kv = replace(kv_loop, body=Body((*prefix, *fs.merge, value_load, *fs.rescale, *fs.consume)))
        compute = tuple(_stream_compute(body, new_kv))
        sigma = Sigma({d_axis.name: Var(d_r.name), m_axis.name: m_expr})
        compute = tuple(s.rewrite(_identity_rename, sigma) for s in compute)
        if m_bound is not None:
            compute = _apply_masked_guards(compute, [(m_axis.name, m_bound)], sigma)
        # Domain register..thread..grid (inner→outer); ``d_r`` is the innermost register.
        domain, binding = _lay_domain(
            [
                (d_r, Binding.REGISTER),
                (m_r, Binding.REGISTER),
                (m_t, Binding.THREAD),
                (m_b, Binding.GRID),
                *[(lp.axis, Binding.GRID) for lp in reversed(grid)],
            ]
        )
        new_block = Block(name=block.name, domain=domain, compute=Body(compute))
        return replace(graph, blocks=(new_block, *graph.blocks[1:]), schedule=replace(graph.schedule, binding=binding))

    # === cooperative reduce / serial stream: the ``_rebracket_k`` tower over each contraction axis,
    # with the cooperative ``K_c`` THREAD lane (``BR`` lanes per row) on the primary axis, then the
    # free-axis σ-split. ``K_c`` is laid FIRST in ``Block.domain`` (innermost THREAD bits). ===
    d = fam.dec_reduce(knobs[fam.reduce_key(dag.k_node.loop.axis.name)])
    bk, fk, br = d.serial, d.fold, d.coop
    targets = target_names or frozenset({dag.k_node.loop.axis.name})
    kax = dag.k_node.loop.axis
    k_c = Axis(f"{kax.name}_c", br, source_axis=kax.source_axis or kax) if br > 1 else None
    # Cross-CTA split-K: thread the ``K_s`` GRID partition through the re-bracket when ``cta > 1``.
    k_s = _k_s_axis(dag, knobs, targets)
    block = graph.blocks[0]
    compute_in = tuple(block.compute)
    # A serial (br=1, single-CTA), non-twisted ``Monoid`` reduce realizes through the scalar combine
    # (the SAME ``combine()`` the flash tiers drive); cooperative / split-K Monoids keep the carrier.
    if br == 1 and k_s is None and isinstance(carrier, Monoid) and len(carrier.partial) == 1:
        compute_in = _realize_serial_monoid(compute_in, carrier, sc)
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
        nb = g.blocks[0]
        g = replace(
            g,
            blocks=(replace(nb, domain=(k_c, *nb.domain)),),
            schedule=replace(g.schedule, binding={**g.schedule.binding, k_c.name: Binding.THREAD}),
        )
    return g


def _atom_cell(cell, *, atom, k_name, kt, axes, out_index=None, frag_a=False) -> AtomTile:
    """Build ONE warp-tier contraction cell — the shared per-cell atomization the flash produce
    (QK^T) and consume (P@V) both perform: fuse the σ-tiled ``cell`` into an ``Mma`` via the generic
    ``atomize_cell``, then wrap its K in a ``ko`` serial loop (``kt`` atom_k steps) or, at ``kt == 1``
    (the whole reduce is one atom), strip the K var to 0 so ``005_lower_atom_tile`` lowers the bare
    ``Mma``. ``out_index`` (the produce's INLINE score coords) / ``frag_a`` (the consume's
    probability-fragment A operand) select the produce vs consume shape; the cell is wrapped in an
    ``AtomTile`` over ``axes``."""
    fused = atomize_cell(cell, atom=atom, k_name=k_name, write=None, out_index=out_index, frag_a=frag_a)
    if kt > 1:
        body: tuple[Stmt, ...] = (SerialTile(axis=Axis(k_name, Dim(kt)), body=Body(fused), kind="plain"),)
    else:
        body = tuple(s.rewrite(_identity_rename, Sigma({k_name: Literal(0, "int")})) for s in fused)
    return AtomTile(axes=axes, body=Body(body), atom=atom)


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

    # Domain: N then M, each split b/w/r/a (GRID/WARP/REGISTER/ATOM); extra-outer trail GRID.
    # ``_free_layers`` orders the tiers (ATOM innermost … GRID outermost) for ``assemble``.
    tiers = (Binding.GRID, Binding.WARP, Binding.REGISTER, Binding.ATOM)
    domain, binding = _lay_domain(
        [
            *[(ax, b) for axes in ((n_b, n_w, n_r, n_a), (m_b, m_w, m_r, m_a)) for ax, b in zip(axes, tiers, strict=True)],
            *[(lp.axis, Binding.GRID) for lp in reversed(extra_outer)],
        ]
    )
    new_block = Block(name=block.name, domain=domain, compute=Body(new_inner))
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
