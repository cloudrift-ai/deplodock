"""Fragment-tier realization of a streaming ``Monoid`` carrier onto the mma C-fragment registers —
the m16n8 sibling of the cross-thread ``emit_combine`` (``lowering/kernel/_combine.py``).

``emit_combine`` realizes a carrier's reduce *across lanes / smem* (``WarpShuffle`` / ``TreeHalve``).
:class:`Fragment` realizes a carrier *across the m16n8 C-fragment registers* (flash online softmax is
the motivating carrier — ``state=(m, l, O)``, ``partial=(score, value)`` — but the realizer is
carrier-generic): the per-row stats are scalars per lane (rows ``g`` / ``g+8`` → suffixes 0/1), the
partial lives fragment-distributed in the C-fragment, and the reduction over its columns is a
:class:`FragmentRowReduce` (the fragment-tier analog of the lane reduce).

:class:`Fragment` IS the :class:`~deplodock.compiler.ir.stmt.carrier_algebra.Distribution` **backend**
``Monoid.project`` drives: it holds the geometry (the MMA register roles + the per-atom ``FragLayout``)
and **generates the combine + mask ops for a carrier** — ``Fragment.combine(carrier)`` projects the
split-off stats ``merge`` (each ``Assign`` dispatched by its role under the distribution: fold →
``FragmentRowReduce``, pointwise → ``FragmentApply``, scalar / carried-state → row ``Assign`` /
``Reassign``) + adds the accum rescale / init / normalize; ``Fragment.mask(mask_when, …)`` builds the
coordinate-predicated ``FragmentMask``s. No softmax / causal / boundary knowledge here — the caller
supplies the mask predicate; the carrier supplies the merge. A second atom plugs in by adding a
``FragLayout`` (``frag_layout``), not editing the nodes; only m16n8 is modeled today (the per-row
scalar distribution is still its 2-rows/lane form — ``rows_per_lane > 2`` awaits a real second atom).

Leading-underscore module name keeps the pass loader from treating it as a rule. Imports only
``ir.*`` + the shared carrier algebra — never ``enumeration``.
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


def _weight_name(accum_merge: tuple, value: str) -> str | None:
    """The distributed SSA the embedded consume contraction multiplies the value by — the fragment
    operand of the accum merge's value-multiply (``coeff · value``). It must land in the geometry's
    ``weight_frags`` (the registers the consume ``Mma``'s A operand reads). Read structurally off the
    carrier — no softmax / ``exp`` knowledge (flash's ``p`` is one instance)."""
    for a in accum_merge:
        if a.op.name == "multiply" and value in a.args and len(a.args) == 2:
            return a.args[0] if a.args[1] == value else a.args[1]
    return None


class Fragment:
    """The mma C-fragment combiner for one warp-chain kernel — the carrier's projection target,
    holding the geometry (the atom + the MMA register roles) and **generating the combine + mask
    ops for a given carrier**:

    - :meth:`combine` projects the carrier's merge onto the fragment tier and assembles the
      per-iteration :class:`FragmentPhases`. ``Fragment`` IS the
      :class:`~deplodock.compiler.ir.stmt.carrier_algebra.Distribution` backend ``Monoid.project``
      drives — fold → ``FragmentRowReduce`` (cross-column reduce) + the per-row scalar update;
      pointwise → one ``FragmentApply`` per N-atom (landing in ``weight_frags`` when it's the
      consume contraction's value-coefficient, else a fresh fragment); scalar / carried-state →
      row-distributed ``Assign`` / ``Reassign``.
    - :meth:`mask` neutralizes each distributed partial fragment to the carrier fold identity where
      a coordinate predicate holds (one ``FragmentMask`` per fragment).

    Carrier-generic — no softmax / causal / boundary knowledge. The register roles: ``partial_frags``
    (the produce contraction's distributed partial), ``weight_frags`` (the consume contraction's
    value-coefficient — flash's probability; the C→A handoff), ``accum_frags`` (the output
    accumulators); ``atom_m`` / ``atom_n`` select the per-atom ``FragLayout``."""

    def __init__(
        self, *, atom_m: int, atom_n: int, partial_frags: tuple[str, ...], weight_frags: tuple[str, ...], accum_frags: tuple[str, ...]
    ):
        self.atom_m = atom_m
        self.atom_n = atom_n
        self.partial_frags = partial_frags
        self.weight_frags = weight_frags
        self.accum_frags = accum_frags
        self.layout = frag_layout(atom_m, atom_n)  # per-atom C-fragment geometry (raises if unmodeled)
        self._reset()

    def _reset(self) -> None:
        # Per-``combine`` projection scratch (so a Fragment is reusable across calls).
        self.out: list[Stmt] = []  # the emitted merge stmts
        self.frag_binding: dict[str, tuple[str, ...]] = {}  # distributed SSA -> per-N-atom fragment arrays
        self.state_fold_op: dict[str, object] = {}  # state -> fold op (drives Init identity + the denominator)
        self.fold_op_of: dict[str, object] = {}  # temp -> fold op (for copy chains, e.g. m = copy(mx))
        self.weight_name: str | None = None  # the distributed SSA bound to weight_frags (feeds the consume Mma)

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
        # The consume contraction's value-coefficient lands in the weight fragments; any other
        # distributed result gets a fresh per-N-atom fragment. One FragmentApply per N-atom: a
        # fragment arg uses that atom's fragment, a per-row scalar arg is the (row0, row1) pair.
        frags = self.weight_frags if name == self.weight_name else tuple(f"{name}_{j}" for j in range(len(self.partial_frags)))
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

    # --- generators (combine + mask) given the carrier ---

    def combine(self, carrier: Monoid) -> FragmentPhases:
        """Generate the per-iteration fragment phases for a streaming ``Monoid``: split the twisted
        carrier (``split_carrier``), **project** the stats merge onto this Fragment
        (``stats.project`` → ``Monoid.project``), then add the accum rescale (``O·α``), the per-state
        ``Init`` seeds, and the normalize epilogue (``O / l``). The merge body is pure carrier
        projection (no softmax knowledge); only the Mma coupling (the weight/accum fragment
        bindings) lives here."""
        self._reset()
        stats, accum, d_state = split_carrier(carrier, carrier.partial[1])
        score = stats.partial[0]
        self.weight_name = _weight_name(accum.merge, carrier.partial[1])
        self.frag_binding = {score: self.partial_frags}
        stats.project(stats.merge, distributed_inputs={score}, dist=self)

        # accum.merge: only the O·α rescale is realized here (in-place per-row multiply); p·v +
        # O = O·α + p·v are the SEMIRING consume Mma, accumulated into the accum fragments.
        rescale_out: list[Stmt] = []
        for a in accum.merge:
            if a.name != d_state and a.op.name == "multiply" and d_state in a.args:
                scalar = a.args[0] if a.args[1] == d_state else a.args[1]
                alpha = (f"{scalar}0", f"{scalar}1")
                rescale_out += [
                    FragmentApply(out=fr, op=_MULTIPLY, args=(fr, alpha), kinds=(FRAG, ROW), in_place=True, layout=self.layout)
                    for fr in self.accum_frags
                ]

        init_out: list[Stmt] = []
        for st in stats.state:
            op = self.state_fold_op[st]
            init_out += [Init(name=f"{st}0", op=op, dtype=F32), Init(name=f"{st}1", op=op, dtype=F32)]

        # The denominator is the add-fold stats state (flash's l); normalize each accum by it per row.
        denom = next(st for st in stats.state if self.state_fold_op[st].name == "add")
        denom_pair = (f"{denom}0", f"{denom}1")
        epilogue_out: list[Stmt] = [
            FragmentApply(out=fr, op=_DIVIDE, args=(fr, denom_pair), kinds=(FRAG, ROW), in_place=True, layout=self.layout)
            for fr in self.accum_frags
        ]
        return FragmentPhases(
            init=tuple(init_out), merge=tuple(self.out), rescale=tuple(rescale_out), update=(), epilogue=tuple(epilogue_out)
        )

    def mask(self, *, mask_when: Expr, col_bases: tuple[Expr, ...], row_base: Expr | None = None) -> list[Stmt]:
        """Generate the fragment mask — neutralize each distributed partial fragment to the carrier
        fold identity where ``mask_when`` holds, before the fold. ONE method for any coordinate
        predicate: ``mask_when`` is a predicate ``Expr`` over the reserved
        :data:`~deplodock.compiler.ir.kernel.ir.FRAG_COL` / :data:`~deplodock.compiler.ir.kernel.ir.FRAG_ROW`
        coordinate vars; ``col_bases`` the per-N-atom column origins (+ ``row_base`` when the
        predicate references the row). Knows nothing about causal / boundary / softmax — those are
        coordinate predicates the caller builds (``__fcol > __frow`` / ``__fcol >= seq_len`` / a
        windowed band / …). Sequencing two masks ANDs their keep-predicates."""
        return [
            FragmentMask(frag=sf, mask_when=mask_when, col_base=cb, row_base=row_base, layout=self.layout)
            for sf, cb in zip(self.partial_frags, col_bases, strict=True)
        ]
