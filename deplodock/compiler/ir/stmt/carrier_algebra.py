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

The per-key online-softmax ``merge`` program is classified ONCE here — :func:`classify_merge_program`
(over :func:`fragment_taint`) tags each ``Assign`` with its geometry-independent carrier role
(``fold`` / ``exp`` / ``state_copy`` / ``state_scalar`` / ``scalar``, as :class:`MergeStep`s) — and the
fragment realizer consumes that classification as a thin geometry emitter (the streaming + cross-thread
realizers consume the carrier's ``merge`` / ``combine_states`` surface directly via
``render_merge_program``, so they need no role split). One analysis, geometry per realizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.stmt.leaves import Assign, Monoid

# Associative+commutative ops whose presence over a fragment operand marks a reduction
# (rowmax / rowsum). Disambiguates ``max(m, s)`` (a FOLD, ``s`` fragment) from a purely
# scalar ``max(m, r)`` — only a fragment operand under one of these is a fold.
_REDUCE_OPS = frozenset({"add", "maximum", "minimum"})


def fragment_taint(merge: tuple[Assign, ...], score_partial: str) -> set[str]:
    """SSA names that are fragment-valued: those transitively derived from the score
    partial via NON-reducing ops. A reduce-eligible op (``add``/``maximum``/``minimum``)
    with exactly one fragment operand is a FOLD — its result is a per-row SCALAR, so it
    stops the taint. Geometry-independent — the analysis half of the fragment / cooperative
    realizers (the realizer supplies how the tainted values are distributed)."""
    frag = {score_partial}
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


@dataclass(frozen=True)
class MergeStep:
    """One classified ``Assign`` of an online-softmax ``merge`` program, tagged with its
    carrier role (geometry-independent). A realizer maps each role onto its distribution:

    - ``"fold"`` — a reduction of a fragment operand to a per-row scalar (rowmax / rowsum);
      ``frag_src`` is the reduced fragment SSA, ``scalar`` the running scalar operand,
      ``is_state`` whether the result updates a carried state (``l``) vs a temp (``mx``).
    - ``"exp"`` — a fused ``exp(frag_src − scalar)`` producing a new fragment (the
      probabilities ``p``); ``frag_src`` the source fragment, ``scalar`` the subtracted row scalar.
    - ``"state_copy"`` — a carried-state reassign from a single source (``m = copy(mx)``);
      ``args = (src,)``.
    - ``"state_scalar"`` / ``"scalar"`` — a scalar state reassign / a pure scalar temp;
      ``op`` + ``args`` are the source program's."""

    role: str
    name: str
    op: object = None
    args: tuple = ()
    frag_src: str | None = None
    scalar: str | None = None
    is_state: bool = False


def classify_merge_program(merge: tuple[Assign, ...], score_partial: str, state_names) -> tuple[list[MergeStep], set[str]]:
    """Classify an online-softmax stats ``merge`` program into role-tagged :class:`MergeStep`s
    (the analysis the fragment realizer ``assembly/_frag_softmax`` consumes), returning
    ``(steps, frag)`` where ``frag`` is the :func:`fragment_taint` set. The ``subtract`` feeding
    an ``exp`` is fused into the ``exp`` step (it emits no step of its own). Pure carrier algebra:
    no IR-dialect / geometry dependency — the realizer supplies the fragment / lane distribution."""
    frag = fragment_taint(merge, score_partial)
    states = set(state_names)
    sub_fuse: dict[str, tuple[str, str]] = {}  # subtract name -> (src fragment, sub scalar)
    steps: list[MergeStep] = []
    for a in merge:
        frag_args = [x for x in a.args if x in frag]
        is_fold = a.op.name in _REDUCE_OPS and len(frag_args) == 1 and len(a.args) == 2
        if is_fold:
            fragnm = frag_args[0]
            scalarnm = a.args[0] if a.args[1] == fragnm else a.args[1]
            steps.append(MergeStep("fold", a.name, op=a.op, frag_src=fragnm, scalar=scalarnm, is_state=a.name in states))
        elif frag_args:
            if a.op.name == "subtract" and len(a.args) == 2 and a.args[0] in frag and a.args[1] not in frag:
                sub_fuse[a.name] = (a.args[0], a.args[1])  # `src - scalar`, fused into the next exp
            elif a.op.name == "exp" and len(a.args) == 1 and a.args[0] in sub_fuse:
                src, sub = sub_fuse[a.args[0]]
                steps.append(MergeStep("exp", a.name, frag_src=src, scalar=sub))
            else:
                raise NotImplementedError(f"v1 fragment softmax: unhandled fragment op {a.op.name!r} on {a.name!r}")
        elif a.name in states:
            if a.op.name == "copy" and len(a.args) == 1:
                steps.append(MergeStep("state_copy", a.name, args=(a.args[0],)))
            else:
                steps.append(MergeStep("state_scalar", a.name, op=a.op, args=a.args))
        else:
            steps.append(MergeStep("scalar", a.name, op=a.op, args=a.args))
    return steps, frag


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
