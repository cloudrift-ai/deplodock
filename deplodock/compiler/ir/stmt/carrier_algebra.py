"""Pure carrier-algebra transforms over a ``Monoid`` — no IR-dialect or pass dependencies.

A twisted ``MONOID(SEMIRING)`` carrier (flash attention's online softmax: ``state=(m, l, O)``,
``partial=(score, value)``) splits into a ``d``-invariant softmax-stats monoid ``(m, l)`` and a
``d``-varying accumulation monoid ``(O)``. The split is plain algebra over the carrier's ``merge``
program (a fixpoint over SSA reads), so it lives here — next to ``Monoid`` in ``ir/stmt`` — rather
than inside any one pass, and is shared by the enumeration ``build_monoid`` (the cooperative /
cross-thread realization) and the assembly fragment realizer (the m16n8 tensor-core realization).

One carrier algebra, three **realizations** (each consumes the same ``Monoid`` surface — the
``merge`` / ``combine_states`` programs over ``carried_names`` / ``partial`` — and differs only in
how state and partials are distributed):

- **streaming** (one thread, serial) — ``Monoid.render`` → ``render_merge_program`` (the per-KV-tile
  fold, inline in the loop body).
- **cross-thread** (lanes / smem) — ``lowering/kernel/_combine.emit_combine`` → ``WarpShuffle`` /
  ``TreeHalve`` (the cooperative reduce of a partial split across the CTA's threads).
- **fragment** (m16n8 tensor-core registers) — ``ir/twist.MmaTwist.combine`` (the ``Combiner`` base's
  generic ``combine`` over the fragment backend) → ``FragmentRowReduce`` (the cross-column fold) + the
  carrier-generic ``FragmentApply`` (every pointwise step — ``exp`` / scale / any op, the warp-chain
  flash softmax and beyond).

The cross-thread and fragment realizers are deliberate siblings — same algebra source, mirrored
structure — not two hand-authored transcriptions.

**The combiner does ALGEBRA; the distribution is geometry.** A reduce **carrier** (``Accum`` —
the degenerate 1-component ``Monoid`` — or a tuple ``Monoid``) exposes ONE combine surface
(``merge`` / ``combine_states`` over ``carried_names`` / ``partial``). That surface is realized by the
ONE combiner, ``ScalarCombiner`` (``ir/twist`` — fragment registers are the ``MmaTwist`` sibling),
which projects it op-by-op via :meth:`Monoid.project` (this module's :func:`interpret`) over a
:class:`Distribution` backend — a reduce over the distributed axis → the backend's ``fold``
(``_reduce``), an elementwise op → ``pointwise``. **The distribution tier — cross-thread / cross-CTA —
is NOT a combiner**: ``emit_combine`` (lanes/smem ``WarpShuffle`` / ``TreeHalve``) and
``_partition.deferred_combine_tilegraph`` (the cross-CTA workspace fold) are pure *geometry* that carry
the carrier's ``combine_states`` straight onto their nodes, which realize the algebra via the SAME
``render_merge_program`` ``ScalarCombiner`` / ``Monoid.render`` use. So a regular reduction lowers
through ``ScalarCombiner`` at any tier; ``dist`` is orthogonal to the carrier (Bird's Third
Homomorphism Theorem keeps ``◁``/``⊙`` distinct operators — ``Monoid`` carries both). No softmax
knowledge in any driver: each dispatches by the op's *role under the distribution*.
"""

from __future__ import annotations

from typing import Protocol

from deplodock.compiler.ir.stmt.leaves import Assign, Monoid, Twist

# Associative+commutative ops whose presence over a distributed operand marks a reduction
# (rowmax / rowsum). Disambiguates ``max(m, s)`` (a FOLD, ``s`` distributed) from a purely
# scalar ``max(m, r)`` — only a distributed operand under one of these is a fold.
_REDUCE_OPS = frozenset({"add", "maximum", "minimum"})


def distributed_taint(merge: tuple[Assign, ...], seeds) -> set[str]:
    """SSA names that are **distributed** (e.g. fragment-valued): those transitively derived from
    ``seeds`` (the distributed inputs — flash's score partial) via NON-reducing ops. A
    reduce-eligible op (``add``/``maximum``/``minimum``) with exactly one distributed operand is a
    FOLD — its result is a per-row SCALAR, so it stops the taint. Geometry-independent — the
    analysis half of every distribution backend (the backend supplies how the tainted values are
    distributed). ``seeds`` is a single name or an iterable."""
    frag = {seeds} if isinstance(seeds, str) else set(seeds)
    changed = True
    while changed:
        changed = False
        for a in merge:
            if a.name in frag:
                continue
            frag_args = [x for x in a.args if x in frag]
            if not frag_args:
                continue
            is_fold = a.op.name in _REDUCE_OPS and len(frag_args) == 1 and len(a.args) == 2
            if not is_fold:
                frag.add(a.name)
                changed = True
    return frag


class Distribution(Protocol):
    """A **projection target** — how a carrier's program realizes once its partials are
    distributed along some axis (across lanes, across C-fragment registers, …). The generic
    driver :func:`interpret` (a.k.a. :meth:`Monoid.project`) walks a merge program and calls
    exactly one method per ``Assign`` by its *role under the distribution*:

    - :meth:`fold` — a reduce over the distributed axis (a reduce-op with one distributed operand) →
      the backend's cross-partition combine (a butterfly / tree / ``FragmentRowReduce``), producing
      a per-partition scalar; ``is_state`` marks a carried-state update (``l``) vs a temp (``mx``).
    - :meth:`pointwise` — an elementwise op producing a distributed value → a per-element map
      (``FragmentApply`` …); ``distributed`` is the taint set so the backend tags each operand.
    - :meth:`scalar` — a pure replicated scalar temp.
    - :meth:`state` — a carried-state reassign that isn't a fold (a copy or a scalar update).

    A backend is stateful (it accumulates the emitted stmts + its name bindings); ``interpret``
    is the pure control flow. No softmax / shape knowledge in the driver."""

    def fold(self, name: str, op, src: str, scalar: str, *, is_state: bool) -> None: ...
    def pointwise(self, name: str, op, args: tuple[str, ...], distributed: set[str]) -> None: ...
    def scalar(self, name: str, op, args: tuple[str, ...]) -> None: ...
    def state(self, name: str, op, args: tuple[str, ...]) -> None: ...


def interpret(merge: tuple[Assign, ...], *, distributed_inputs, state_names, dist: Distribution) -> None:
    """The carrier-generic projection driver behind :meth:`Monoid.project`: taint the distributed
    values (seeded by ``distributed_inputs``), then dispatch each ``Assign`` to ``dist``'s
    fold / pointwise / scalar / state by its role under the distribution — the distribution law
    applied uniformly, no op cap, no shape knowledge. Mutates ``dist`` (the stateful backend);
    returns nothing."""
    frag = distributed_taint(merge, distributed_inputs)
    states = set(state_names)
    for a in merge:
        dargs = [x for x in a.args if x in frag]
        if a.op.name in _REDUCE_OPS and len(dargs) == 1 and len(a.args) == 2:
            scalarnm = a.args[0] if a.args[1] == dargs[0] else a.args[1]
            dist.fold(a.name, a.op, dargs[0], scalarnm, is_state=a.name in states)
        elif a.name in frag:  # a distributed-producing elementwise op
            dist.pointwise(a.name, a.op, a.args, frag)
        elif a.name in states:  # a carried-state reassign (copy / scalar)
            dist.state(a.name, a.op, a.args)
        else:  # a replicated scalar temp
            dist.scalar(a.name, a.op, a.args)


def value_dependent(merge: tuple[Assign, ...], value_name: str) -> set[str]:
    """Fixpoint over a twisted carrier's ``merge`` program: the SSA names that
    transitively read the value partial (``value_name`` — flash's V load). These are
    the P@V accumulation (``O·α + p·v`` and its temps + the ``O`` state update); the
    rest is the ``d``-invariant softmax-stats update. A fixpoint (not one pass) because
    the state's own update (``O = om + pv``) precedes a temp that reads it (``om = O·α``)
    in program order."""
    dvar = {value_name}
    changed = True
    while changed:
        changed = False
        for a in merge:
            if a.name not in dvar and any(arg in dvar for arg in a.args):
                dvar.add(a.name)
                changed = True
    return dvar


def split_carrier(carrier: Monoid, value_name: str) -> tuple[Monoid, Monoid, str]:
    """Split a twisted ``MONOID(SEMIRING)`` carrier into ``(stats, accum, d_state)``:
    the ``d``-invariant softmax-stats ``Monoid`` (state minus the accumulator, partial =
    the score) and the ``d``-varying accumulation ``Monoid`` (the accumulator state,
    partial = the value), which reads the stats carrier's rescale/probability temps by
    name (they render inline, visible to the sibling carrier). ``d_state`` is the
    accumulator state component (flash's ``O``)."""
    dvar = value_dependent(carrier.twist.merge, value_name)
    accum_merge = tuple(a for a in carrier.twist.merge if a.name in dvar)
    stats_merge = tuple(a for a in carrier.twist.merge if a.name not in dvar)
    d_states = [s for s in carrier.state if s in dvar]
    stats_states = [s for s in carrier.state if s not in dvar]
    if len(d_states) != 1:
        raise ValueError(f"split_carrier: expected exactly one accumulator state, got {d_states}")
    d_state = d_states[0]
    ident = dict(zip(carrier.state, carrier.identity, strict=True)) if carrier.identity else {}
    stats = Monoid(
        state=tuple(stats_states),
        partial=(carrier.partial[0],),
        twist=Twist(merge=stats_merge),
        identity=tuple(ident[s] for s in stats_states) if ident else (),
        commutative=carrier.commutative,
        axes=carrier.axes,
    )
    accum = Monoid(
        state=(d_state,),
        partial=(value_name,),
        twist=Twist(merge=accum_merge),
        identity=(ident[d_state],) if ident else (),
        commutative=carrier.commutative,
        axes=carrier.axes,
    )
    return stats, accum, d_state
