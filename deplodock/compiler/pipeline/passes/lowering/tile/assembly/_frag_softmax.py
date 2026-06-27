"""Fragment-tier realization of a streaming flash ``Monoid`` — the m16n8 sibling of the
cross-thread ``emit_combine`` (``lowering/kernel/_combine.py``).

``emit_combine`` realizes a carrier's reduce *across lanes / smem* (``WarpShuffle`` /
``TreeHalve``). This module realizes the SAME ``flash_combine`` ``Monoid`` (online softmax,
``state=(m, l, O)``, ``partial=(score, value)``) *across the m16n8 C-fragment registers*:
the per-row stats ``(m, l)`` are 2 scalars per lane (rows ``g`` / ``g+8`` → suffixes 0/1),
the score partial lives fragment-distributed in the C-fragment, and the reduction over the
tile's columns is a :class:`FragmentRowReduce` (the fragment-tier analog of the lane reduce).

The split (``ir/stmt/carrier_algebra.split_carrier``) separates the ``d``-invariant stats
monoid from the ``d``-varying accumulation monoid; the stats ``merge`` program is then
classified by the shared carrier algebra (``carrier_algebra.classify_merge_program`` →
role-tagged ``MergeStep``s, via the fragment/scalar **taint** analysis: the score partial is
fragment-valued; a reduce-eligible op consuming it is a FOLD whose result is a per-row scalar).
This module is a **thin geometry emitter** over that classification: the carrier supplies the
*algebra* (which ops, what order, which reductions, the normalizer); the caller supplies the
*fragment geometry* (N-atom / D-atom frag names) via :class:`FragGeom` — that geometry is a
tiling decision, not recoverable from the scalar program. v1 targets the m16n8 layout
(2 rows/lane) exercised by the warp-chain flash.

Pure functions; leading-underscore module name keeps the pass loader from treating it as a
rule. Imports only ``ir.*`` + the shared carrier algebra — never ``enumeration``.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.kernel.ir import (
    FRAG,
    ROW,
    FragmentApply,
    FragmentBoundaryMask,
    FragmentCausalMask,
    FragmentRowReduce,
    Reassign,
)
from deplodock.compiler.ir.stmt import Assign, Init, Monoid, Stmt
from deplodock.compiler.ir.stmt.carrier_algebra import classify_merge_program, split_carrier

_SUBTRACT = ElementwiseImpl("subtract")
_EXP = ElementwiseImpl("exp")
_MULTIPLY = ElementwiseImpl("multiply")
_DIVIDE = ElementwiseImpl("divide")


@dataclass(frozen=True)
class FragGeom:
    """The fragment geometry the realizer can't derive from the scalar program — the tiling
    decision. ``atom_m``/``atom_n`` fix the m16n8 layout (``atom_m==16`` ⇒ 2 rows/lane).
    ``score_frags`` are the live QK^T C-fragments (named by ``kernel/005`` off the produce
    ``Mma``); ``prob_frags`` the P fragments this realizer produces (consumed by the C→A
    handoff); ``accum_frags`` the streaming O accumulators (one per D-atom)."""

    atom_m: int
    atom_n: int
    score_frags: tuple[str, ...]
    prob_frags: tuple[str, ...]
    accum_frags: tuple[str, ...]


@dataclass(frozen=True)
class FragmentSoftmax:
    """The generated phase contents for the warp-chain ``CarryScope`` — what replaces the
    hand-listed softmax. ``epilogue`` is the per-D-atom normalize (in-place ``FragmentApply`` divides); the
    caller interleaves the ``RegStore``s. ``update`` is empty (state reassigned in ``merge``)."""

    init: tuple[Stmt, ...]
    merge: tuple[Stmt, ...]
    rescale: tuple[Stmt, ...]
    update: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]


def realize_fragment_softmax(carrier: Monoid, *, geom: FragGeom) -> FragmentSoftmax:
    """Generate the m16n8 fragment realization of a streaming flash ``Monoid``.

    Splits the twisted carrier into stats ``(m, l)`` + accum ``(O)`` (``split_carrier``), then
    consumes the carrier-algebra classification of the stats ``merge`` (``classify_merge_program``
    → role-tagged :class:`MergeStep`s) as a **thin geometry emitter**: each role maps to the
    m16n8 fragment ops (fold → ``FragmentRowReduce`` + the row-scalar update; exp → a ``subtract``
    then ``exp`` ``FragmentApply`` over the prob fragments; a generic ``frag_apply`` → one
    ``FragmentApply`` per N-atom; state/scalar → row-distributed ``Assign`` / ``Reassign``). The
    accum ``merge``'s ``O·α`` rescale is an in-place ``FragmentApply`` multiply; ``O = O·α + p·v``
    is left to the SEMIRING P@V ``Mma`` (the consume cell), not emitted here. Every pointwise
    fragment op is the ONE ``FragmentApply`` node (it subsumed the former ``FragmentExp`` /
    ``FragmentScale``), so any carrier's vocabulary — not just softmax's — reaches the tier."""
    if geom.atom_m != 16 or geom.atom_n != 8:
        raise NotImplementedError(f"v1 fragment softmax targets m16n8; got atom ({geom.atom_m}, {geom.atom_n})")
    stats, accum, d_state = split_carrier(carrier, carrier.partial[1])
    score = stats.partial[0]
    steps, _frag = classify_merge_program(stats.merge, score, stats.state)

    frag_binding: dict[str, tuple[str, ...]] = {score: geom.score_frags}  # fragment SSA -> frag-array
    state_fold_op: dict[str, object] = {}  # state -> its reduce op (drives Init identity + the denominator)
    fold_op_of: dict[str, object] = {}  # temp -> reduce op (for copy chains, e.g. m = copy(mx))
    prob_emitted = False

    merge_out: list[Stmt] = []
    for st in steps:
        if st.role == "fold":
            r0, r1 = f"{st.name}_r0", f"{st.name}_r1"
            merge_out.append(FragmentRowReduce(top=r0, bot=r1, frags=frag_binding[st.frag_src], op=st.op))
            fold_op_of[st.name] = st.op
            if st.is_state:  # state-update fold: l = lm + rowsum(p)
                n0, n1 = f"{st.name}_n0", f"{st.name}_n1"
                merge_out += [Assign(n0, st.op, (f"{st.scalar}0", r0)), Assign(n1, st.op, (f"{st.scalar}1", r1))]
                merge_out += [Reassign(f"{st.name}0", n0), Reassign(f"{st.name}1", n1)]
                state_fold_op[st.name] = st.op
            else:  # temp fold: mx = max(m, rowmax(s))
                merge_out += [Assign(f"{st.name}0", st.op, (f"{st.scalar}0", r0)), Assign(f"{st.name}1", st.op, (f"{st.scalar}1", r1))]
        elif st.role == "exp":
            if prob_emitted:
                raise NotImplementedError("v1 fragment softmax supports a single probability fragment map")
            # ``p = exp(s − m)`` = a per-row ``subtract`` then an ``exp`` (the former fused
            # ``FragmentExp``, now two generic ``FragmentApply``s). The exp lands in the geometry's
            # probability fragments (consumed by the P@V ``Mma`` / the C→A handoff).
            sub = (f"{st.scalar}0", f"{st.scalar}1")  # the per-row new-max, explicit pair
            for j, (sf, pf) in enumerate(zip(frag_binding[st.frag_src], geom.prob_frags, strict=True)):
                ds = f"{st.name}_ds_{j}"
                merge_out.append(FragmentApply(out=ds, op=_SUBTRACT, args=(sf, sub), kinds=(FRAG, ROW)))
                merge_out.append(FragmentApply(out=pf, op=_EXP, args=(ds,), kinds=(FRAG,)))
            frag_binding[st.name] = geom.prob_frags
            prob_emitted = True
        elif st.role == "state_copy":  # scalar state reassign (m = copy(mx))
            src = st.args[0]
            merge_out += [Reassign(f"{st.name}0", f"{src}0"), Reassign(f"{st.name}1", f"{src}1")]
            if src in fold_op_of:
                state_fold_op[st.name] = fold_op_of[src]
        elif st.role == "state_scalar":
            n0, n1 = f"{st.name}_n0", f"{st.name}_n1"
            merge_out += [Assign(n0, st.op, tuple(f"{x}0" for x in st.args)), Assign(n1, st.op, tuple(f"{x}1" for x in st.args))]
            merge_out += [Reassign(f"{st.name}0", n0), Reassign(f"{st.name}1", n1)]
        elif st.role == "frag_apply":  # a generic fragment-producing op (any non-exp activation, mul/div, …)
            # One FragmentApply per N-atom; a fragment arg uses that atom's fragment, a per-row
            # scalar arg is broadcast by row (the FragmentApply render adds the 0/1 suffix). The
            # result is a fresh per-N-atom fragment array, registered for downstream steps.
            new_frags = tuple(f"{st.name}_{j}" for j in range(len(geom.score_frags)))
            for j, out_fr in enumerate(new_frags):
                args, kinds = [], []
                for a in st.args:
                    bound = frag_binding.get(a)
                    args.append(bound[j] if bound is not None else (f"{a}0", f"{a}1"))
                    kinds.append(FRAG if bound is not None else ROW)
                merge_out.append(FragmentApply(out=out_fr, op=st.op, args=tuple(args), kinds=tuple(kinds)))
            frag_binding[st.name] = new_frags
        else:  # pure scalar temp, row-distributed
            for sfx in ("0", "1"):
                merge_out.append(Assign(f"{st.name}{sfx}", st.op, tuple(f"{x}{sfx}" for x in st.args)))

    # accum.merge: only the `O·α` rescale is realized here; `p·v` + `O = O·α + p·v` are the
    # SEMIRING P@V Mma (the consume cell), accumulated in place into the O fragments.
    rescale_out: list[Stmt] = []
    for a in accum.merge:
        if a.name == d_state:
            continue  # O = om + pv — the Mma accumulate
        if a.op.name == "multiply" and d_state in a.args:
            scalar = a.args[0] if a.args[1] == d_state else a.args[1]
            # O *= α — in-place per-row multiply (the former FragmentScale).
            alpha = (f"{scalar}0", f"{scalar}1")
            rescale_out += [
                FragmentApply(out=fr, op=_MULTIPLY, args=(fr, alpha), kinds=(FRAG, ROW), in_place=True) for fr in geom.accum_frags
            ]
        # else: pv = p·v (reads the value partial) — part of the consume Mma, not emitted.

    init_out: list[Stmt] = []
    for st in stats.state:
        op = state_fold_op[st]
        init_out += [Init(name=f"{st}0", op=op, dtype=F32), Init(name=f"{st}1", op=op, dtype=F32)]

    # The denominator is the add-fold stats state (flash's l); the max-fold state (m) isn't
    # read in the epilogue. Normalize each O accumulator by the per-row denom (in-place divide).
    denom = next(st for st in stats.state if state_fold_op[st].name == "add")
    denom_pair = (f"{denom}0", f"{denom}1")
    epilogue_out: list[Stmt] = [
        FragmentApply(out=fr, op=_DIVIDE, args=(fr, denom_pair), kinds=(FRAG, ROW), in_place=True) for fr in geom.accum_frags
    ]

    return FragmentSoftmax(
        init=tuple(init_out), merge=tuple(merge_out), rescale=tuple(rescale_out), update=(), epilogue=tuple(epilogue_out)
    )


def realize_score_mask(geom: FragGeom, *, q_row_base: Expr, kv_col_bases: tuple[Expr, ...]) -> list[Stmt]:
    """The fragment-tier score-partial mask — the same "neutralize ``partial[0]`` to the fold
    identity past a bound" operation Part D's ``_mask_carrier`` does cooperatively, one tier
    down. Masks each score C-fragment to ``-1e30`` (the carrier's ``m`` identity, the soft -inf
    that avoids ``-inf − -inf = nan``) over the strict upper triangle, before the rowmax fold.
    ``kv_col_bases`` is the absolute column origin per N-atom (caller adds the ``nt·atom_n``
    offset — an expr/geometry concern)."""
    return [
        FragmentCausalMask(frag=sf, q_row_base=q_row_base, kv_col_base=cb) for sf, cb in zip(geom.score_frags, kv_col_bases, strict=True)
    ]


def realize_boundary_mask(geom: FragGeom, *, kv_col_bases: tuple[Expr, ...], bound: Expr) -> list[Stmt]:
    """The fragment-tier symbolic-``seq_len`` boundary mask — the column-only sibling of
    :func:`realize_score_mask`. Masks each score C-fragment to ``-1e30`` (the carrier's ``m``
    identity) where the element's absolute key column ``>= bound`` (the partial final KV
    tile's padding keys), before the rowmax fold — so the online-softmax denominator excludes
    them (``exp(0) = 1`` would corrupt it). Composes with the causal mask by sequencing both
    (each writes ``-1e30``, the AND of the keep predicates). ``kv_col_bases`` is the absolute
    column origin per N-atom."""
    return [FragmentBoundaryMask(frag=sf, kv_col_base=cb, bound=bound) for sf, cb in zip(geom.score_frags, kv_col_bases, strict=True)]
