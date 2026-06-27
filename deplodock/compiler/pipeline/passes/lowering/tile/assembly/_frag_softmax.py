"""Fragment-tier realization of a streaming flash ``Monoid`` — the m16n8 sibling of the
cross-thread ``emit_combine`` (``lowering/kernel/_combine.py``).

``emit_combine`` realizes a carrier's reduce *across lanes / smem* (``WarpShuffle`` /
``TreeHalve``). This module realizes the SAME ``flash_combine`` ``Monoid`` (online softmax,
``state=(m, l, O)``, ``partial=(score, value)``) *across the m16n8 C-fragment registers*:
the per-row stats ``(m, l)`` are 2 scalars per lane (rows ``g`` / ``g+8`` → suffixes 0/1),
the score partial lives fragment-distributed in the C-fragment, and the reduction over the
tile's columns is a :class:`FragmentRowReduce` (the fragment-tier analog of the lane reduce).

The split (``ir/stmt/carrier_algebra.split_carrier``) separates the ``d``-invariant stats monoid
from the ``d``-varying accumulation monoid; :class:`FragmentDist` is then a
:class:`~deplodock.compiler.ir.stmt.carrier_algebra.Distribution` **backend** that ``Monoid.project``
drives over the stats ``merge`` — each ``Assign`` dispatches by its role under the distribution
(fold → ``FragmentRowReduce``, pointwise → ``FragmentApply``, scalar / carried-state → row
``Assign`` / ``Reassign``). No softmax knowledge in the driver; this module supplies only the
*geometry* (which fragment registers, the Mma coupling) — the ``FragmentGeom`` register names + the
**per-atom C-fragment layout** ``FragLayout`` (``frag_layout(atom_m, atom_n)``) the emitted nodes
carry, so a second atom plugs in by adding a descriptor, not editing the nodes. Only m16n8 is
modeled today; the per-row scalar distribution (the ``0`` / ``1`` suffixes) is still its 2-rows/lane
form (generalizing it to ``rows_per_lane > 2`` needs a real second atom to design + test against).

Leading-underscore module name keeps the pass loader from treating it as a rule. Imports only
``ir.*`` + the shared carrier algebra — never ``enumeration``.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Var
from deplodock.compiler.ir.kernel.ir import (
    FRAG,
    FRAG_COL,
    FRAG_ROW,
    ROW,
    FragmentApply,
    FragmentMask,
    FragmentRowReduce,
    Reassign,
    frag_layout,
)
from deplodock.compiler.ir.stmt import Assign, Init, Monoid, Stmt
from deplodock.compiler.ir.stmt.carrier_algebra import split_carrier

_SUBTRACT = ElementwiseImpl("subtract")
_EXP = ElementwiseImpl("exp")
_MULTIPLY = ElementwiseImpl("multiply")
_DIVIDE = ElementwiseImpl("divide")


@dataclass(frozen=True)
class FragmentGeom:
    """The fragment-combiner geometry the carrier projection can't derive from the program — the
    tiling decision (the register roles + the atom). ``atom_m``/``atom_n`` select the per-atom
    ``FragLayout``. ``score_frags`` are the live QK^T C-fragments (named by ``kernel/005`` off the
    produce ``Mma``); ``prob_frags`` the probability fragments the projection writes (consumed by
    the P@V ``Mma`` — the C→A handoff); ``accum_frags`` the streaming output accumulators (one per
    D-atom)."""

    atom_m: int
    atom_n: int
    score_frags: tuple[str, ...]
    prob_frags: tuple[str, ...]
    accum_frags: tuple[str, ...]


@dataclass(frozen=True)
class FragmentPhases:
    """The realized per-iteration carrier phases for the warp-chain ``CarryScope`` (the fragment-tier
    analog of the streaming ``CarryScope``'s phases) — what the projection produces. ``epilogue`` is
    the per-D-atom normalize (in-place ``FragmentApply`` divides); the caller interleaves the
    ``RegStore``s. ``update`` is empty (state reassigned in ``merge``). Not softmax-specific — any
    streaming carrier projected onto the fragment tier yields these phases."""

    init: tuple[Stmt, ...]
    merge: tuple[Stmt, ...]
    rescale: tuple[Stmt, ...]
    update: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]


def _prob_name(accum_merge: tuple, value: str) -> str | None:
    """The distributed SSA that feeds the P@V ``Mma`` — the fragment operand of the accum merge's
    value-multiply (``pv = p·v``). It must land in the geometry's ``prob_frags`` (the registers the
    consume ``Mma`` reads). Read structurally off the carrier, not from an ``exp`` role."""
    for a in accum_merge:
        if a.op.name == "multiply" and value in a.args and len(a.args) == 2:
            return a.args[0] if a.args[1] == value else a.args[1]
    return None


class FragmentDist:
    """The m16n8 C-fragment :class:`~deplodock.compiler.ir.stmt.carrier_algebra.Distribution`
    backend — realizes a carrier's projected merge over the tensor-core register tile. The geometry
    (``FragmentGeom`` + ``FragLayout``) and the Mma coupling (which fragment feeds P@V) live here; the
    carrier algebra (``Monoid.project``) drives it generically.

    Distribution roles → fragment ops: ``fold`` → ``FragmentRowReduce`` (cross-column reduce) + the
    per-row scalar update; ``pointwise`` → one ``FragmentApply`` per N-atom (its result lands in the
    geometry's ``prob_frags`` when it's the P@V probability, else a fresh fragment); ``scalar`` /
    ``state`` → row-distributed ``Assign`` / ``Reassign``."""

    def __init__(self, geom: FragmentGeom):
        self.geom = geom
        self.layout = frag_layout(geom.atom_m, geom.atom_n)  # per-atom C-fragment geometry (raises if unmodeled)
        self.out: list[Stmt] = []  # the emitted merge stmts
        self.frag_binding: dict[str, tuple[str, ...]] = {}  # distributed SSA -> per-N-atom fragment arrays
        self.state_fold_op: dict[str, object] = {}  # state -> fold op (drives Init identity + the denominator)
        self.fold_op_of: dict[str, object] = {}  # temp -> fold op (for copy chains, e.g. m = copy(mx))
        self.prob_name: str | None = None  # the distributed SSA bound to geom.prob_frags (feeds P@V)

    # --- Distribution protocol (driven by Monoid.project) ---

    def fold(self, name, op, src, scalar, *, is_state) -> None:
        r0, r1 = f"{name}_r0", f"{name}_r1"
        self.out.append(FragmentRowReduce(top=r0, bot=r1, frags=self.frag_binding[src], op=op, group=self.layout.reduce_group))
        self.fold_op_of[name] = op
        if is_state:  # state-update fold: l = lm + rowsum(p)
            n0, n1 = f"{name}_n0", f"{name}_n1"
            self.out += [Assign(n0, op, (f"{scalar}0", r0)), Assign(n1, op, (f"{scalar}1", r1))]
            self.out += [Reassign(f"{name}0", n0), Reassign(f"{name}1", n1)]
            self.state_fold_op[name] = op
        else:  # temp fold: mx = max(m, rowmax(s))
            self.out += [Assign(f"{name}0", op, (f"{scalar}0", r0)), Assign(f"{name}1", op, (f"{scalar}1", r1))]

    def pointwise(self, name, op, args, distributed) -> None:  # noqa: ARG002 — taint via frag_binding
        # The P@V probability lands in the geometry's prob fragments; any other distributed result
        # gets a fresh per-N-atom fragment. One FragmentApply per N-atom: a fragment arg uses that
        # atom's fragment, a per-row scalar arg is the (row0, row1) pair.
        frags = self.geom.prob_frags if name == self.prob_name else tuple(f"{name}_{j}" for j in range(len(self.geom.score_frags)))
        for j, out_fr in enumerate(frags):
            a_args, kinds = [], []
            for x in args:
                bound = self.frag_binding.get(x)
                a_args.append(bound[j] if bound is not None else (f"{x}0", f"{x}1"))
                kinds.append(FRAG if bound is not None else ROW)
            self.out.append(FragmentApply(out=out_fr, op=op, args=tuple(a_args), kinds=tuple(kinds), layout=self.layout))
        self.frag_binding[name] = frags

    def scalar(self, name, op, args) -> None:  # a replicated scalar temp, per row
        for sfx in ("0", "1"):
            self.out.append(Assign(f"{name}{sfx}", op, tuple(f"{x}{sfx}" for x in args)))

    def state(self, name, op, args) -> None:  # a carried-state reassign (copy / scalar)
        if op.name == "copy" and len(args) == 1:
            src = args[0]
            self.out += [Reassign(f"{name}0", f"{src}0"), Reassign(f"{name}1", f"{src}1")]
            if src in self.fold_op_of:
                self.state_fold_op[name] = self.fold_op_of[src]
        else:
            n0, n1 = f"{name}_n0", f"{name}_n1"
            self.out += [Assign(n0, op, tuple(f"{x}0" for x in args)), Assign(n1, op, tuple(f"{x}1" for x in args))]
            self.out += [Reassign(f"{name}0", n0), Reassign(f"{name}1", n1)]

    # --- the full flash realization built around the projection ---

    def realize(self, carrier: Monoid) -> FragmentPhases:
        """Realize the m16n8 fragment flash from a streaming ``Monoid``: split the twisted carrier
        (``split_carrier``), **project** the stats merge onto this backend (``stats.project`` →
        ``Monoid.project``), then add the accum ``O·α`` rescale, the per-state ``Init`` seeds, and
        the ``O /= l`` epilogue. The merge body is pure carrier projection (no softmax knowledge);
        only the Mma coupling (the prob/accum fragment bindings) lives here."""
        stats, accum, d_state = split_carrier(carrier, carrier.partial[1])
        score = stats.partial[0]
        self.prob_name = _prob_name(accum.merge, carrier.partial[1])
        self.frag_binding = {score: self.geom.score_frags}
        stats.project(stats.merge, distributed_inputs={score}, dist=self)

        # accum.merge: only the O·α rescale is realized here (in-place per-row multiply); p·v +
        # O = O·α + p·v are the SEMIRING P@V Mma (the consume cell), accumulated into the O fragments.
        rescale_out: list[Stmt] = []
        for a in accum.merge:
            if a.name != d_state and a.op.name == "multiply" and d_state in a.args:
                scalar = a.args[0] if a.args[1] == d_state else a.args[1]
                alpha = (f"{scalar}0", f"{scalar}1")
                rescale_out += [
                    FragmentApply(out=fr, op=_MULTIPLY, args=(fr, alpha), kinds=(FRAG, ROW), in_place=True, layout=self.layout)
                    for fr in self.geom.accum_frags
                ]

        init_out: list[Stmt] = []
        for st in stats.state:
            op = self.state_fold_op[st]
            init_out += [Init(name=f"{st}0", op=op, dtype=F32), Init(name=f"{st}1", op=op, dtype=F32)]

        # The denominator is the add-fold stats state (flash's l); normalize each O by it per row.
        denom = next(st for st in stats.state if self.state_fold_op[st].name == "add")
        denom_pair = (f"{denom}0", f"{denom}1")
        epilogue_out: list[Stmt] = [
            FragmentApply(out=fr, op=_DIVIDE, args=(fr, denom_pair), kinds=(FRAG, ROW), in_place=True, layout=self.layout)
            for fr in self.geom.accum_frags
        ]
        return FragmentPhases(
            init=tuple(init_out), merge=tuple(self.out), rescale=tuple(rescale_out), update=(), epilogue=tuple(epilogue_out)
        )


def realize_fragment_softmax(carrier: Monoid, *, geom: FragmentGeom) -> FragmentPhases:
    """Realize the m16n8 fragment flash from a streaming ``Monoid`` — the thin entry the assembler
    calls: build the :class:`FragmentDist` backend and project the carrier onto it. The merge body
    is generated by ``Monoid.project`` (carrier-generic, no softmax knowledge); ``FragmentDist``
    supplies the geometry. (The former hand-rolled classify+realize; the carrier vocabulary + the
    geometry are now the distribution backend, driven by the one projection method.)"""
    return FragmentDist(geom).realize(carrier)


def realize_score_mask(geom: FragmentGeom, *, q_row_base: Expr, kv_col_bases: tuple[Expr, ...]) -> list[Stmt]:
    """The fragment-tier score-partial mask — the same "neutralize ``partial[0]`` to the fold
    identity past a bound" operation Part D's ``_mask_carrier`` does cooperatively, one tier
    down. Masks each score C-fragment to ``-1e30`` (the carrier's ``m`` identity, the soft -inf
    that avoids ``-inf − -inf = nan``) over the strict upper triangle, before the rowmax fold.
    ``kv_col_bases`` is the absolute column origin per N-atom (caller adds the ``nt·atom_n``
    offset — an expr/geometry concern). The causal predicate is the generic ``FragmentMask``'s
    ``mask_when`` = ``key_col > query_row`` (the strict upper triangle)."""
    layout = frag_layout(geom.atom_m, geom.atom_n)
    causal = BinaryExpr(">", Var(FRAG_COL), Var(FRAG_ROW))  # mask where key col > query row
    return [
        FragmentMask(frag=sf, mask_when=causal, col_base=cb, row_base=q_row_base, layout=layout)
        for sf, cb in zip(geom.score_frags, kv_col_bases, strict=True)
    ]


def realize_boundary_mask(geom: FragmentGeom, *, kv_col_bases: tuple[Expr, ...], bound: Expr) -> list[Stmt]:
    """The fragment-tier symbolic-``seq_len`` boundary mask — the column-only sibling of
    :func:`realize_score_mask`. Masks each score C-fragment to ``-1e30`` (the carrier's ``m``
    identity) where the element's absolute key column ``>= bound`` (the partial final KV
    tile's padding keys), before the rowmax fold — so the online-softmax denominator excludes
    them (``exp(0) = 1`` would corrupt it). Composes with the causal mask by sequencing both
    (each writes ``-1e30``, the AND of the keep predicates). ``kv_col_bases`` is the absolute
    column origin per N-atom. The boundary predicate is the generic ``FragmentMask``'s
    ``mask_when`` = ``key_col >= bound`` (column-only — no ``row_base``)."""
    layout = frag_layout(geom.atom_m, geom.atom_n)
    beyond = BinaryExpr(">=", Var(FRAG_COL), bound)  # mask where key col >= seq_len (padding keys)
    return [
        FragmentMask(frag=sf, mask_when=beyond, col_base=cb, layout=layout) for sf, cb in zip(geom.score_frags, kv_col_bases, strict=True)
    ]
