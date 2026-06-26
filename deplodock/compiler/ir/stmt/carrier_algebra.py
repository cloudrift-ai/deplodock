"""Pure carrier-algebra transforms over a ``Monoid`` — no IR-dialect or pass dependencies.

A twisted ``MONOID(SEMIRING)`` carrier (flash attention's online softmax: ``state=(m, l, O)``,
``partial=(score, value)``) splits into a ``d``-invariant softmax-stats monoid ``(m, l)`` and a
``d``-varying accumulation monoid ``(O)``. The split is plain algebra over the carrier's ``merge``
program (a fixpoint over SSA reads), so it lives here — next to ``Monoid`` in ``ir/stmt`` — rather
than inside any one pass, and is shared by the enumeration ``chain_build`` (the cooperative /
cross-thread realization) and the assembly fragment realizer (the m16n8 tensor-core realization).

One carrier algebra, three **realizations** (each consumes the same ``Monoid`` surface — the
``merge`` / ``combine_states`` programs over ``carried_names`` / ``partial`` — and differs only in
how state and partials are distributed):

- **streaming** (one thread, serial) — ``Monoid.render`` → ``render_merge_program`` (the per-KV-tile
  fold, inline in the loop body).
- **cross-thread** (lanes / smem) — ``lowering/kernel/_combine.emit_combine`` → ``WarpShuffle`` /
  ``TreeHalve`` (the cooperative reduce of a partial split across the CTA's threads).
- **fragment** (m16n8 tensor-core registers) — ``assembly/_frag_softmax.realize_fragment_softmax``
  → ``FragmentRowReduce`` / ``FragmentExp`` / ``FragmentScale`` (the warp-chain flash softmax).

The cross-thread and fragment realizers are deliberate siblings — same algebra source, mirrored
structure — not two hand-authored transcriptions.
"""

from __future__ import annotations

from deplodock.compiler.ir.stmt.leaves import Assign, Monoid


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
    dvar = value_dependent(carrier.merge, value_name)
    accum_merge = tuple(a for a in carrier.merge if a.name in dvar)
    stats_merge = tuple(a for a in carrier.merge if a.name not in dvar)
    d_states = [s for s in carrier.state if s in dvar]
    stats_states = [s for s in carrier.state if s not in dvar]
    if len(d_states) != 1:
        raise ValueError(f"split_carrier: expected exactly one accumulator state, got {d_states}")
    d_state = d_states[0]
    ident = dict(zip(carrier.state, carrier.identity, strict=True)) if carrier.identity else {}
    stats = Monoid(
        state=tuple(stats_states),
        partial=(carrier.partial[0],),
        merge=stats_merge,
        identity=tuple(ident[s] for s in stats_states) if ident else (),
        commutative=carrier.commutative,
        axes=carrier.axes,
    )
    accum = Monoid(
        state=(d_state,),
        partial=(value_name,),
        merge=accum_merge,
        identity=(ident[d_state],) if ident else (),
        commutative=carrier.commutative,
        axes=carrier.axes,
    )
    return stats, accum, d_state
