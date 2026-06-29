"""Cross-CTA split-reduce (the ``cta`` tier) — consume a ``GRID`` ``ReduceStage``.

A reduce partition with a ``GRID`` stage (``ReducePlan.needs_split``) splits the reduce
axis across CTAs. This pass realizes that split as a **graph rewrite** — the schedule
carries the partition, the graph carries the kernel count:

- **partial kernel** — the ``cta`` stage becomes an extra grid axis (``_ksplit``); each CTA
  reduces its **contiguous slice** ``[s·B, (s+1)·B)`` of the reduce axis (``B =
  extent / cta``) and contributes its carrier *state* (not the projected output).
- **finalize** — two arms, picked by the ``GRID`` stage's finalize letter
  (``ReducePlan.finalize``):
  - ``"kernel"`` — the partial writes its state to a ``ws[cta, *free]`` ``__partial``
    workspace; a sibling **finalize kernel** reduces the workspace over the split axis via
    ``carrier.as_state_merge`` (the cross-partition combine) and projects the output. **2
    nodes.** The only legal arm for a twisted carrier (flash's ``e^{Δm}`` rescale can't be
    an atomic).
  - ``"atomic"`` — the partial ``atomicAdd``\\ s its (additive) state straight into the
    output; the output is zero-init'd per launch. **1 node.** Additive carriers only.

So the schedule's GRID stage is **consumed** here (the partial's plan is stripped of it,
the finalize is a fresh ``ReducePlan``); ``lowering/kernel`` only ever sees single-launch
kernels (``assert not needs_split``).

This cut handles **additive** carriers — a degenerate ``Monoid`` reduce (``sum``) and a
``Semiring`` contraction (split-K matmul), one carrier-state component each. A twisted
multi-component carrier (flash split-KV) is the remaining step.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Body, Init, Load, Loop, Monoid, Semiring, Write
from deplodock.compiler.ir.stmt.algebra import Map
from deplodock.compiler.ir.tile import (
    Placement,
    ReducePlan,
    SemiringKernel,
    TileOp,
    kernel_for,
)
from deplodock.compiler.ir.tile.schedule import Level
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


def _reduce_axis(carrier) -> Axis:
    """The carrier's reduce :class:`Axis` — ``Monoid.axis`` or ``Semiring.reduce_axis``."""
    return carrier.axis if isinstance(carrier, Monoid) else carrier.reduce_axis


def _state_names(carrier) -> tuple[str, ...]:
    """The carrier-state component names — a ``Monoid``'s ``state.names``, a ``Semiring``'s
    single accumulator (``out``)."""
    return carrier.state.names if isinstance(carrier, Monoid) else (carrier.out,)


def _apply_sigma(node, sigma: Sigma):
    """Rewrite an op-tree ``node`` (``Map`` / ``Monoid`` / ``Semiring``) under ``sigma`` —
    substitutes the reduce-axis var inside every operand / partial load index (recurse into
    sources; the partial bodies' nested loop-IR is handled by ``Stmt.rewrite``)."""
    ident = lambda n: n  # noqa: E731
    if isinstance(node, Map):
        src = _apply_sigma(node.source, sigma) if node.source is not None else None
        return replace(node, source=src, body=Body(tuple(s.rewrite(ident, sigma) for s in node.body)))
    if isinstance(node, Monoid):
        return replace(node, partial=tuple(_apply_sigma(p, sigma) for p in node.partial))
    if isinstance(node, Semiring):
        return replace(node, operands=tuple(_apply_sigma(o, sigma) for o in node.operands))
    raise TypeError(f"_apply_sigma: unexpected node {type(node).__name__}")


def _slice_carrier(carrier, b: int):
    """Slice the carrier's reduce axis to a CTA's contiguous block: offset every reduce-axis
    load by ``_ksplit · B`` and shrink the axis extent to ``B`` (so the loop walks ``[0, B)``
    while reading ``[s·B, (s+1)·B)``)."""
    rax = _reduce_axis(carrier)
    offset = BinaryExpr("+", Var(rax.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sliced = _apply_sigma(carrier, Sigma({rax.name: offset}))
    new_ax = replace(rax, extent=Dim(b))
    return replace(sliced, axis=new_ax) if isinstance(sliced, Monoid) else replace(sliced, reduce_axis=new_ax)


def _cell_index(op, grid) -> tuple:
    """The output-cell index the original kernel writes (the projection ``Write``'s index,
    or — for a bare carrier — the grid-axis vars)."""
    if isinstance(op, Map):
        for s in op.body:
            if isinstance(s, Write):
                return s.index
    return tuple(Var(ax.name) for ax in grid)


def _strip_grid(plan: ReducePlan) -> ReducePlan:
    """``plan`` with the ``GRID`` stage removed — the partial is a single-launch reduce over
    its slice (the cross-CTA combine is realized by the finalize, not in the partial)."""
    return ReducePlan(tuple(s for s in plan.stages if s.level is not Level.GRID))


def _carrier_identities(carrier) -> dict[str, float]:
    """The per-state-component neutral element (the finalize seeds the state with it before
    folding the partitions). A degenerate carrier reads it off its dissolved ``Accum``\\ s; a
    twisted carrier (flash) off the ``Accum`` folds in its streaming ``merge`` (``l``/``O`` →
    add → 0, ``m`` → maximum → −inf)."""
    base = carrier if isinstance(carrier, Monoid) else carrier.fold.as_monoid()
    accums = base.as_accums()
    if accums is not None:
        return {a.name: a.op.identity for a in accums}
    return {s.name: s.op.identity for s in base.merge if isinstance(s, Accum)}


def _mapped(op, grid, *, reduce: ReducePlan | None = None, tile=None):
    """A ``*Kernel`` wrapping ``op`` with a **mapped** placement over ``grid`` (so
    ``020_schedule`` skips it). ``reduce`` sets the reduce partition where the schedule has
    one (``Monoid`` / ``Semiring``; a flat ``Map`` finalize has none); ``tile`` preserves a
    ``Semiring``'s output tile."""
    k = kernel_for(op, Placement(free=tuple(grid), grid=tuple(grid)))
    sched = k.schedule
    if reduce is not None and hasattr(sched, "reduce"):
        sched = replace(sched, reduce=reduce)
    if isinstance(k, SemiringKernel) and tile is not None:
        sched = replace(sched, tile=tile)
    return replace(k, schedule=sched)


def rewrite(match: Match, root: Node) -> TileOp | Graph | None:
    tile: TileOp = root.op
    sched = tile.kernel.schedule if tile.kernel is not None else None
    plan = getattr(sched, "reduce", None)
    if plan is None or not plan.needs_split:
        raise RuleSkipped("no cross-CTA split stage — nothing to split")

    op = tile.op
    carrier = op.reduce_node
    cta = plan.cta
    rax = _reduce_axis(carrier)
    if not rax.extent.is_static:
        raise NotImplementedError("cross-CTA split of a symbolic reduce axis is not built yet")
    extent = rax.extent.as_static()
    if extent % cta != 0:
        raise NotImplementedError(f"cross-CTA split needs a divisible reduce axis (extent {extent} % cta {cta})")
    b = extent // cta
    states = _state_names(carrier)
    n_comp = len(states)

    out = root.output
    grid = sched.place.grid
    cell = _cell_index(op, grid)
    split = Axis(name=_SPLIT, extent=Dim(cta))
    sliced = _slice_carrier(carrier, b)
    tile_plan = getattr(sched, "tile", None)

    # --- atomic finalize: ONE kernel — each CTA atomicAdds its slice's state into the output
    # (zero-init'd per launch). Additive (single-component) carriers only; the GRID stage is
    # consumed into the grid (the split becomes a grid axis), no second node.
    if plan.finalize == "atomic":
        if n_comp != 1:
            raise NotImplementedError("atomic finalize needs an additive (1-component) carrier; the twisted carrier is kernel-only")
        atomic_op = Map(source=sliced, body=Body((Write(output=out.name, index=cell, values=states, atomic=True),)))
        atomic_kernel = _mapped(atomic_op, (split, *grid), reduce=_strip_grid(plan), tile=tile_plan)
        return TileOp(kernel=atomic_kernel, name=tile.name)

    # The ``__partial`` workspace packs every carrier-state component: ``ws[comp, cta, *free]``
    # (the ``comp`` leading axis dropped for a single-component additive carrier, so the
    # additive workspace stays ``ws[cta, *free]``). A multi-component (twisted flash) carrier
    # writes its ``(m, l, O)`` state to the three ``comp`` slices, no multi-output kernel.
    ws_name = f"{out.name}__partial"
    ws_shape = (Dim(n_comp), Dim(cta), *out.shape) if n_comp > 1 else (Dim(cta), *out.shape)

    def ws_index(i: int) -> tuple:
        lead = (Literal(i, "int"), Var(_SPLIT)) if n_comp > 1 else (Var(_SPLIT),)
        return (*lead, *cell)

    # --- partial kernel: reduce a CTA's slice, write its carrier state to the workspace -----
    partial_op = Map(source=sliced, body=Body(tuple(Write(output=ws_name, index=ws_index(i), value=states[i]) for i in range(n_comp))))
    partial_kernel = _mapped(partial_op, (split, *grid), reduce=_strip_grid(plan), tile=tile_plan)
    partial_tile = TileOp(kernel=partial_kernel, name=f"{tile.name}__partial")

    # --- finalize kernel: seed the carrier state, then fold each partition's state from the
    # workspace over the split axis via the carrier's ``combine_states`` (the ``partial=()``
    # state-merge Monoid stays a stmt inside the split loop, rendered by ``render_merge_program``
    # — the same realizer the cooperative combine uses). A flat ``Map`` of loop-IR: ``Init``
    # seeds, the split ``Loop`` (loads + the combine), then the original projection + store.
    fin_base = carrier if isinstance(carrier, Monoid) else carrier.fold.as_monoid()
    state = fin_base.state.names
    other = tuple(f"{nm}__p" for nm in state)
    combine = fin_base.as_state_merge(other)
    # A twisted state-merge (kept as a Monoid stmt, rendered by render_merge_program) must seed
    # the state explicitly; a degenerate one dissolves to ``Accum``\\ s whose seed ``Loop.render``
    # derives — an ``Init`` there would double-declare the accumulator.
    kept = combine.as_accums() is None and not any(isinstance(s, Accum) for s in combine.merge)
    ids = _carrier_identities(carrier)
    seeds = tuple(Init(name=state[i], identity=ids[state[i]], dtype=F32) for i in range(n_comp)) if kept else ()
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    loop = Loop(axis=split, body=Body((*loads, combine)))
    proj = list(op.body) if isinstance(op, Map) else []
    if not any(isinstance(s, Write) for s in proj):
        out_val = proj[-1].defines()[-1] if proj else state[0]
        proj.append(Write(output=out.name, index=cell, value=out_val))
    fin_op = Map(body=Body((*seeds, loop, *proj)))
    fin_kernel = _mapped(fin_op, grid)
    fin_tile = TileOp(kernel=fin_kernel, name=tile.name)

    # --- splice the two-kernel fragment in place of the single split TileOp ----------------
    frag = Graph()
    for inp in root.inputs:
        n = match.graph.nodes[inp]
        frag.add_node(op=InputOp(), inputs=[], output=n.output, node_id=inp)
    frag.add_node(op=partial_tile, inputs=list(root.inputs), output=Tensor(ws_name, ws_shape, out.dtype), node_id=ws_name)
    frag.add_node(op=fin_tile, inputs=[ws_name], output=Tensor(out.name, out.shape, out.dtype), node_id=out.name)
    frag.outputs = [out.name]
    return frag
