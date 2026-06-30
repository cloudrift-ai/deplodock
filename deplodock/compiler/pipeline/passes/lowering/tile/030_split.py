"""Cross-CTA split-reduce (the ``cta`` tier) — consume a ``GRID`` ``ReduceStage``.

A reduce partition with a ``GRID`` stage (``ReducePlan.needs_split``) splits the reduce
axis across CTAs. This pass realizes that split as a **graph rewrite** — the schedule
carries the partition, the graph carries the kernel count. It reads the reduce structure
off the kernel's annotated reduce ``Loop`` (``loop.carrier`` / ``loop.axis``), never an
op-tree node:

- **partial kernel** — the ``cta`` stage becomes an extra grid axis (``_ksplit``); each CTA
  reduces its **contiguous slice** ``[s·B, (s+1)·B)`` of the reduce axis (``B =
  extent / cta``) and contributes its carrier *state* (not the projected output).
- **finalize** — two arms, picked by the ``GRID`` stage's finalize letter
  (``ReducePlan.finalize``):
  - ``"kernel"`` — the partial writes its state to a ``ws[cta, *free]`` ``__partial``
    workspace; a sibling **finalize kernel** seeds the carrier state then folds the
    workspace over the split axis via ``carrier.as_state_merge`` (the cross-partition
    combine, a renderable :class:`StateMerge`) and projects the output. **2 nodes.** The only
    legal arm for a twisted carrier (flash's ``e^{Δm}`` rescale can't be an atomic).
  - ``"atomic"`` — the partial ``atomicAdd``\\ s its (additive) state into the output (applying
    the kernel's projection epilogue per-partition first, when that epilogue *distributes* over
    the add — ``mean``'s ``×1/N``; a non-distributive one like ``l2``'s ``sqrt`` is refused, use
    ``"kernel"``); the output is zero-init'd per launch. **1 node.** Additive carriers only.

So the schedule's GRID stage is **consumed** here (the partial's plan is stripped of it,
the finalize is a fresh ``ReducePlan``); ``lowering/kernel`` only ever sees single-launch
kernels (``assert not needs_split``).

This cut handles **additive** carriers — a degenerate ``PLANAR`` reduce (``sum``) and a
``CONTRACTION`` contraction (split-K matmul), one carrier-state component each. A twisted
multi-component carrier (flash split-KV) is the remaining step.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Node, Tensor
from deplodock.compiler.ir.axis import Axis, AxisRole
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Write
from deplodock.compiler.ir.tile import (
    Map,
    Placement,
    ReducePlan,
    TileOp,
    TilePlan,
    kernel_for,
)
from deplodock.compiler.ir.tile.ops import axis_role, lower, reduce_loop
from deplodock.compiler.ir.tile.schedule import Level
from deplodock.compiler.pipeline import Match, Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_SPLIT = "_ksplit"  # the cross-CTA split grid axis


def _slice_loop(rloop: Loop, b: int) -> Loop:
    """Slice the reduce ``Loop`` to a CTA's contiguous block: offset every reduce-axis load by
    ``_ksplit · B`` (σ on the loop body) and shrink the axis extent to ``B`` (so the loop walks
    ``[0, B)`` while reading ``[s·B, (s+1)·B)``). The carrier (state algebra) rides through
    unchanged — only the operand load indices move."""
    rax = rloop.axis
    offset = BinaryExpr("+", Var(rax.name), BinaryExpr("*", Var(_SPLIT), Literal(b, "int")))
    sigma = Sigma({rax.name: offset})
    ident = lambda n: n  # noqa: E731
    new_body = tuple(s.rewrite(ident, sigma) for s in rloop.body)
    new_ax = replace(rax, extent=Dim(b))
    return Loop(axis=new_ax, body=Body(new_body), unroll=rloop.unroll, role=rloop.role, carrier=rloop.carrier)


def _cell_index(stmts, grid) -> tuple:
    """The output-cell index the original kernel writes (the projection ``Write``'s index,
    or — for a bare carrier whose grid-cell store is glue — the grid-axis vars). Read off the
    kernel's lowered body (``ops.lower`` — the annotated loop nest, ``Map`` / ``Reduction`` alike)."""
    for s in stmts:
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
    accums = carrier.as_accums()
    if accums is not None:
        return {a.name: a.op.identity for a in accums}
    return {s.name: s.op.identity for s in carrier.merge if isinstance(s, Accum)}


def _mapped(op, grid, *, reduce: ReducePlan | None = None, tier=None):
    """A ``Kernel`` wrapping ``op`` with a **mapped** placement over ``grid`` (so ``_schedule``
    skips it). ``reduce`` sets the reduce partition; ``tier`` preserves a contraction's scalar
    output tier (:class:`TilePlan`)."""
    k = kernel_for(op, Placement(free=tuple(grid), grid=tuple(grid)))
    sched = k.schedule
    if reduce is not None and hasattr(sched, "reduce"):
        sched = replace(sched, reduce=reduce)
    if axis_role(op) is AxisRole.CONTRACTION and tier is not None:
        sched = replace(sched, tier=tier)
    return replace(k, schedule=sched)


def _projection_distributes(body, states: tuple[str, ...]) -> bool:
    """True if the kernel's projection epilogue is a **linear-homogeneous** map of the carrier
    state(s) — i.e. it distributes over the atomic-add combine, so applying it to each CTA's
    partition before the ``atomicAdd`` equals applying it once after the cross-CTA sum
    (``Σ c·xₛ = c·(Σ xₛ)``). A bare state write (``proj = id``) trivially distributes; a constant
    *scale* — ``mean``'s ``×1/N`` — does; an additive offset (a fused bias), a nonlinear unary
    (``relu`` / ``reciprocal`` of the *state*), or a product of two state-derived values do NOT.

    Conservative forward dataflow: ``linear`` is the set of SSA names that are a
    linear-homogeneous function of the state. A value is grown into it only by ``multiply`` with
    a state-independent operand (an arg not itself in ``linear``); any other op that consumes a
    ``linear`` value — or any projection stmt we can't reason about — refuses. The final ``Write``
    must store only ``linear`` values."""
    linear = set(states)
    for s in body:
        if isinstance(s, Write):
            return all(v in linear for v in s.values)
        if isinstance(s, Load):
            continue  # reads memory (the count / a per-output operand) — state-independent
        if not isinstance(s, Assign):
            return False  # an unfamiliar projection stmt — can't prove distributivity
        hot = [a for a in s.args if a in linear]
        if not hot:
            continue  # state-independent — a constant w.r.t. the split
        if s.op.name == "multiply" and len(hot) == 1:
            linear.add(s.name)  # state · constant — still linear-homogeneous
            continue
        return False  # add / divide / nonlinear of a state value breaks distributivity
    return False  # no Write reached


def rewrite(match: Match, root: Node) -> TileOp | Graph | None:
    tile: TileOp = root.op
    sched = tile.kernel.schedule if tile.kernel is not None else None
    plan = getattr(sched, "reduce", None)
    if plan is None or not plan.needs_split:
        raise RuleSkipped("no cross-CTA split stage — nothing to split")

    op = tile.op
    rloop = reduce_loop(op)
    carrier = rloop.carrier
    cta = plan.cta
    rax = rloop.axis
    if not rax.extent.is_static:
        raise NotImplementedError("cross-CTA split of a symbolic reduce axis is not built yet")
    extent = rax.extent.as_static()
    if extent % cta != 0:
        raise NotImplementedError(f"cross-CTA split needs a divisible reduce axis (extent {extent} % cta {cta})")
    b = extent // cta
    states = carrier.state.names
    n_comp = len(states)

    out = root.output
    grid = sched.place.grid
    # The lowered loop nest (``Map`` / ``Reduction`` alike) — find the carrier-bearing reduce loop
    # in it by position (``reduce_loop`` returns a fresh synthesized loop for a ``Reduction``, so key
    # off the lowered list, not object identity).
    stmts = lower(op)
    cell = _cell_index(stmts, grid)
    split = Axis(name=_SPLIT, extent=Dim(cta))
    idx = next(i for i, s in enumerate(stmts) if isinstance(s, Loop) and s.carrier is not None)
    sliced_loop = _slice_loop(stmts[idx], b)
    # The stmts before / after the reduce loop: ``before`` is the (typically empty) prologue, ``after``
    # the projection epilogue (its own loads + computes + the output ``Write``).
    before = tuple(stmts[:idx])
    after = list(stmts[idx + 1 :])
    # Only the scalar output tier survives a split (a warp/None tier doesn't ride the partial
    # kernels — they materialize scalar); ``getattr`` covers a non-contraction split (no ``tier``).
    tier = getattr(sched, "tier", None)
    tile_plan = tier if isinstance(tier, TilePlan) and not tier.is_warp else None

    # --- atomic finalize: ONE kernel — each CTA atomicAdds its slice's state into the output
    # (zero-init'd per launch). Additive (single-component) carriers only; the GRID stage is
    # consumed into the grid (the split becomes a grid axis), no second node.
    if plan.finalize == "atomic":
        if n_comp != 1:
            raise NotImplementedError("atomic finalize needs an additive (1-component) carrier; the twisted carrier is kernel-only")
        # The kernel's projection epilogue (``mean``'s ``×1/N``, a fused bias/activation, …) rides
        # on ``after``; a bare carrier (``sum`` / a contraction matmul) has just the output ``Write``.
        # Atomic finalize applies the projection PER-PARTITION before the ``atomicAdd``, so it must
        # distribute over the add — else each CTA's contribution is mis-scaled. When it doesn't
        # distribute, refuse loudly: the deferred-kernel finalize (``g<n>k``) projects once after the
        # combine and is always correct.
        if after:
            if not _projection_distributes(after, states):
                raise NotImplementedError(
                    "atomic finalize can't carry a non-distributive projection epilogue "
                    "(e.g. a fused bias / activation on a split reduce); pin the deferred-kernel "
                    "finalize instead (REDUCE=…g<n>k)"
                )
            atomic_proj = tuple(replace(s, atomic=True) if isinstance(s, Write) else s for s in after)
        else:
            # A bare carrier (``sum`` / a contraction matmul) — its grid-cell store is glue; synthesize
            # the atomic ``Write`` of the carrier state directly.
            atomic_proj = (Write(output=out.name, index=cell, values=states, atomic=True),)
        atomic_op = Map(body=Body((*before, sliced_loop, *atomic_proj)))
        atomic_kernel = _mapped(atomic_op, (split, *grid), reduce=_strip_grid(plan), tier=tile_plan)
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
    ws_writes = tuple(Write(output=ws_name, index=ws_index(i), value=states[i]) for i in range(n_comp))
    partial_op = Map(body=Body((*before, sliced_loop, *ws_writes)))
    partial_kernel = _mapped(partial_op, (split, *grid), reduce=_strip_grid(plan), tier=tile_plan)
    partial_tile = TileOp(kernel=partial_kernel, name=f"{tile.name}__partial")

    # --- finalize kernel: seed the carrier state, then fold each partition's state from the
    # workspace over the split axis via the carrier's ``as_state_merge`` (a renderable
    # :class:`StateMerge`, the same combine the cooperative tier uses). A flat ``Map`` of loop-IR:
    # ``Init`` seeds, the split ``Loop`` (loads + the combine), then the original projection + store.
    other = tuple(f"{nm}__p" for nm in states)
    combine = carrier.as_state_merge(other)
    ids = _carrier_identities(carrier)
    seeds = tuple(Init(name=states[i], identity=ids[states[i]], dtype=F32) for i in range(n_comp))
    loads = tuple(Load(name=other[i], input=ws_name, index=ws_index(i)) for i in range(n_comp))
    fin_loop = Loop(axis=split, body=Body((*loads, combine)))
    fin_proj = list(after)
    if not any(isinstance(s, Write) for s in fin_proj):
        out_val = fin_proj[-1].defines()[-1] if fin_proj else states[0]
        fin_proj.append(Write(output=out.name, index=cell, value=out_val))
    fin_op = Map(body=Body((*seeds, fin_loop, *fin_proj)))
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
