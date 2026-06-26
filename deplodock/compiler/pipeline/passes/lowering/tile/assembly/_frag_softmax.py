"""Fragment-tier realization of a streaming flash ``Monoid`` — the m16n8 sibling of the
cross-thread ``emit_combine`` (``lowering/kernel/_combine.py``).

``emit_combine`` realizes a carrier's reduce *across lanes / smem* (``WarpShuffle`` /
``TreeHalve``). This module realizes the SAME ``flash_combine`` ``Monoid`` (online softmax,
``state=(m, l, O)``, ``partial=(score, value)``) *across the m16n8 C-fragment registers*:
the per-row stats ``(m, l)`` are 2 scalars per lane (rows ``g`` / ``g+8`` → suffixes 0/1),
the score partial lives fragment-distributed in the C-fragment, and the reduction over the
tile's columns is a :class:`FragmentRowReduce` (the fragment-tier analog of the lane reduce).

The split (``ir/stmt/carrier_algebra.split_carrier``) separates the ``d``-invariant stats
monoid from the ``d``-varying accumulation monoid; this realizer then maps the stats
``merge`` program onto fragment ops by a fragment/scalar **taint** analysis (the score
partial is fragment-valued; a reduce-eligible op consuming it is a FOLD whose result is a
per-row scalar). The carrier supplies the *algebra* (which ops, what order, which reductions,
the normalizer); the caller supplies the *fragment geometry* (N-atom / D-atom frag names) via
:class:`FragGeom` — that geometry is a tiling decision, not recoverable from the scalar
program. v1 targets the m16n8 layout (2 rows/lane) exercised by the warp-chain flash.

Pure functions; leading-underscore module name keeps the pass loader from treating it as a
rule. Imports only ``ir.*`` + the shared carrier algebra — never ``enumeration``.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.expr import Expr
from deplodock.compiler.ir.kernel.ir import (
    FragmentBoundaryMask,
    FragmentCausalMask,
    FragmentExp,
    FragmentRowReduce,
    FragmentScale,
    Reassign,
)
from deplodock.compiler.ir.stmt import Assign, Init, Monoid, Stmt
from deplodock.compiler.ir.stmt.carrier_algebra import split_carrier

# Associative+commutative ops whose presence over a fragment operand marks a reduction
# (rowmax / rowsum). Disambiguates ``max(m, s)`` (a fold, ``s`` fragment) from a purely
# scalar ``max(m, r)`` — only a fragment operand under one of these is a FOLD.
_REDUCE_OPS = frozenset({"add", "maximum", "minimum"})


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
    hand-listed softmax. ``epilogue`` is the per-D-atom normalize ``FragmentScale``s; the
    caller interleaves the ``RegStore``s. ``update`` is empty (state reassigned in ``merge``)."""

    init: tuple[Stmt, ...]
    merge: tuple[Stmt, ...]
    rescale: tuple[Stmt, ...]
    update: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]


def _fragment_taint(merge: tuple[Assign, ...], score_partial: str) -> set[str]:
    """SSA names that are fragment-valued: transitively derive from the score partial via
    NON-reducing ops. A reduce-eligible op (``add``/``maximum``/``minimum``) with exactly one
    fragment operand is a FOLD — its result is a per-row SCALAR, so it stops the taint."""
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


def realize_fragment_softmax(carrier: Monoid, *, geom: FragGeom) -> FragmentSoftmax:
    """Generate the m16n8 fragment realization of a streaming flash ``Monoid``.

    Splits the twisted carrier into stats ``(m, l)`` + accum ``(O)`` (``split_carrier``), then
    walks the stats ``merge`` mapping each ``Assign`` to fragment ops / row-distributed scalars
    by taint, and the accum ``merge``'s ``O·α`` rescale to ``FragmentScale``. ``O = O·α + p·v``
    is left to the SEMIRING P@V ``Mma`` (the consume cell), not emitted here."""
    if geom.atom_m != 16 or geom.atom_n != 8:
        raise NotImplementedError(f"v1 fragment softmax targets m16n8; got atom ({geom.atom_m}, {geom.atom_n})")
    stats, accum, d_state = split_carrier(carrier, carrier.partial[1])
    score = stats.partial[0]
    frag = _fragment_taint(stats.merge, score)
    states = set(stats.state)

    frag_binding: dict[str, tuple[str, ...]] = {score: geom.score_frags}  # fragment SSA -> frag-array
    frag_subexpr: dict[str, tuple[str, str]] = {}  # name -> (src fragment, sub scalar) for a fused `src - scalar`
    state_fold_op: dict[str, object] = {}  # state -> its reduce op (drives Init identity + the denominator)
    fold_op_of: dict[str, object] = {}  # temp -> reduce op (for copy chains, e.g. m = copy(mx))
    prob_emitted = False

    merge_out: list[Stmt] = []
    for a in stats.merge:
        frag_args = [x for x in a.args if x in frag]
        is_fold = a.op.name in _REDUCE_OPS and len(frag_args) == 1 and len(a.args) == 2
        if is_fold:
            fragnm = frag_args[0]
            scalarnm = a.args[0] if a.args[1] == fragnm else a.args[1]
            r0, r1 = f"{a.name}_r0", f"{a.name}_r1"
            merge_out.append(FragmentRowReduce(top=r0, bot=r1, frags=frag_binding[fragnm], op=a.op))
            fold_op_of[a.name] = a.op
            if a.name in states:  # state-update fold: l = lm + rowsum(p)
                n0, n1 = f"{a.name}_n0", f"{a.name}_n1"
                merge_out += [Assign(n0, a.op, (f"{scalarnm}0", r0)), Assign(n1, a.op, (f"{scalarnm}1", r1))]
                merge_out += [Reassign(f"{a.name}0", n0), Reassign(f"{a.name}1", n1)]
                state_fold_op[a.name] = a.op
            else:  # temp fold: mx = max(m, rowmax(s))
                merge_out += [Assign(f"{a.name}0", a.op, (f"{scalarnm}0", r0)), Assign(f"{a.name}1", a.op, (f"{scalarnm}1", r1))]
        elif frag_args:
            if a.op.name == "subtract" and len(a.args) == 2 and a.args[0] in frag and a.args[1] not in frag:
                frag_subexpr[a.name] = (a.args[0], a.args[1])  # `src - scalar`, fused into the next exp
            elif a.op.name == "exp" and len(a.args) == 1 and a.args[0] in frag_subexpr:
                if prob_emitted:
                    raise NotImplementedError("v1 fragment softmax supports a single probability fragment map")
                src, sub = frag_subexpr[a.args[0]]
                src_frags = frag_binding[src]
                for sf, pf in zip(src_frags, geom.prob_frags, strict=True):
                    merge_out.append(FragmentExp(out=pf, src=sf, top_sub=f"{sub}0", bot_sub=f"{sub}1"))
                frag_binding[a.name] = geom.prob_frags
                prob_emitted = True
            else:
                raise NotImplementedError(f"v1 fragment softmax: unhandled fragment op {a.op.name!r} on {a.name!r}")
        elif a.name in states:  # scalar state reassign (m = copy(mx))
            if a.op.name == "copy" and len(a.args) == 1:
                src = a.args[0]
                merge_out += [Reassign(f"{a.name}0", f"{src}0"), Reassign(f"{a.name}1", f"{src}1")]
                if src in fold_op_of:
                    state_fold_op[a.name] = fold_op_of[src]
            else:
                n0, n1 = f"{a.name}_n0", f"{a.name}_n1"
                merge_out += [Assign(n0, a.op, tuple(f"{x}0" for x in a.args)), Assign(n1, a.op, tuple(f"{x}1" for x in a.args))]
                merge_out += [Reassign(f"{a.name}0", n0), Reassign(f"{a.name}1", n1)]
        else:  # pure scalar temp, row-distributed
            for sfx in ("0", "1"):
                merge_out.append(Assign(f"{a.name}{sfx}", a.op, tuple(f"{x}{sfx}" for x in a.args)))

    # accum.merge: only the `O·α` rescale is realized here; `p·v` + `O = O·α + p·v` are the
    # SEMIRING P@V Mma (the consume cell), accumulated in place into the O fragments.
    rescale_out: list[Stmt] = []
    for a in accum.merge:
        if a.name == d_state:
            continue  # O = om + pv — the Mma accumulate
        if a.op.name == "multiply" and d_state in a.args:
            scalar = a.args[0] if a.args[1] == d_state else a.args[1]
            rescale_out += [FragmentScale(frag=fr, top=f"{scalar}0", bot=f"{scalar}1") for fr in geom.accum_frags]
        # else: pv = p·v (reads the value partial) — part of the consume Mma, not emitted.

    init_out: list[Stmt] = []
    for st in stats.state:
        op = state_fold_op[st]
        init_out += [Init(name=f"{st}0", op=op, dtype=F32), Init(name=f"{st}1", op=op, dtype=F32)]

    # The denominator is the add-fold stats state (flash's l); the max-fold state (m) isn't
    # read in the epilogue. Normalize each O accumulator by 1/denom per row.
    denom = next(st for st in stats.state if state_fold_op[st].name == "add")
    epilogue_out: list[Stmt] = [FragmentScale(frag=fr, top=f"(1.0f/{denom}0)", bot=f"(1.0f/{denom}1)") for fr in geom.accum_frags]

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
