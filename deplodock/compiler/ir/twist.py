"""``Twist`` — the MONOID twist between two SEMIRING contractions, realized at the mma fragment tier.

A twisted ``MONOID(SEMIRING)`` carrier (flash attention's online softmax) is a MONOID fold over the
fragments an embedded SEMIRING contraction produces. ``Twist`` is that fold's realizer at the
tensor-core register tier: built FROM the two ``Mma`` cells (the QK^T *produce* + the P@V *consume*),
it derives the C-fragment register roles off them and **generates the combine + mask ops** for a
``Monoid`` carrier (it IS the ``Distribution`` backend ``Monoid.project`` drives — see
``ir/stmt/carrier_algebra``).

It lives at ``ir/`` rather than under a single IR layer because it spans them: it consumes the
``Monoid`` carrier (``ir/stmt``) and emits the fragment ops + storage nodes (``ir/kernel/ir``:
``FragmentApply`` / ``FragmentRowReduce`` / ``FragmentMask`` / ``RegFragment``). ``Mma`` owns the
contraction; ``Twist`` owns the twist over the fragments it produces — so the geometry/registers are
derived off the cells, never duplicated.

Carrier-generic — no softmax / causal / boundary knowledge: the carrier supplies the merge, the
caller (the attention assembler) supplies the mask predicate. The only attention name that survives
is in the *caller*, not here.
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
    RegFragment,
    frag_layout,
)
from deplodock.compiler.ir.stmt import Assign, Init, Mma, Monoid, Stmt

_SUBTRACT = ElementwiseImpl("subtract")
_EXP = ElementwiseImpl("exp")
_MULTIPLY = ElementwiseImpl("multiply")
_DIVIDE = ElementwiseImpl("divide")


@dataclass(frozen=True)
class FragmentPhases:
    """The realized per-iteration carrier phases for the warp-chain ``CarryScope`` (the fragment-tier
    analog of the streaming ``CarryScope``'s phases) — what :meth:`Twist.combine` produces. ``init``
    seeds the carried-state identities + declares the accum ``RegFragment``s; ``epilogue`` is the
    per-output-atom normalize (in-place ``FragmentApply`` divides); the caller interleaves the
    ``RegStore``s. ``update`` is empty (state reassigned in ``merge``). Not softmax-specific — any
    streaming carrier projected onto the fragment tier yields these phases."""

    init: tuple[Stmt, ...]
    merge: tuple[Stmt, ...]
    rescale: tuple[Stmt, ...]
    update: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]


def _weight_name(accum_merge: tuple, value: str) -> str | None:
    """The distributed SSA the embedded consume contraction multiplies the value by — the fragment
    operand of the accum merge's value-multiply (``coeff · value``). It must land in ``weight_frags``
    (the registers the consume ``Mma``'s A operand reads). Read structurally off the carrier — no
    softmax / ``exp`` knowledge (flash's ``p`` is one instance)."""
    for a in accum_merge:
        if a.op.name == "multiply" and value in a.args and len(a.args) == 2:
            return a.args[0] if a.args[1] == value else a.args[1]
    return None


class Twist:
    """The MONOID **twist** between two SEMIRING contractions — flash's online softmax sitting between
    the QK^T (produce) and P@V (consume) ``Mma`` cells. Built FROM those cells: the geometry (the atom
    + the C-fragment register roles) is **derived** off them, not re-specified, so ``Twist`` and
    ``Mma`` don't duplicate the register/atom info (``Mma`` owns the contraction, ``Twist`` the twist
    over the fragments it produces).

    - :meth:`combine` projects the carrier's merge onto the fragment tier and assembles the
      per-iteration :class:`FragmentPhases`. ``Twist`` IS the
      :class:`~deplodock.compiler.ir.stmt.carrier_algebra.Distribution` backend ``Monoid.project``
      drives — fold → ``FragmentRowReduce`` (cross-column reduce) + the per-row scalar update;
      pointwise → one ``FragmentApply`` per N-atom (landing in ``weight_frags`` when it's the consume
      contraction's value-coefficient, else a fresh fragment); scalar / carried-state →
      row-distributed ``Assign`` / ``Reassign``. It also declares the accum ``RegFragment``s.
    - :meth:`mask` neutralizes each distributed partial fragment to the carrier fold identity where a
      coordinate predicate holds (one ``FragmentMask`` per fragment).

    Carrier-generic — no softmax / causal / boundary knowledge. The register roles derived off the
    cells: ``partial_frags`` = each produce cell's C-fragment (``f"{m.c}_frag"`` — the distributed
    partial / scores); ``accum_frags`` = each consume cell's C (the output accumulators); ``atom`` =
    the produce atom (→ the per-atom ``FragLayout``). ``weight_frags`` (the consume contraction's
    value-coefficient — flash's probability, the C→A handoff) is the one role NOT a cell field (the
    consume reads the post-handoff fragment), so it is minted, one per produce cell."""

    def __init__(self, *, produce: tuple[Mma, ...], consume: tuple[Mma, ...]):
        self.produce = produce
        self.consume = consume
        self.atom = produce[0].atom
        self.layout = frag_layout(self.atom.shape[0], self.atom.shape[1])  # per-atom C-fragment geometry
        self.partial_frags = tuple(f"{m.c}_frag" for m in produce)  # produce cells' C-fragments (kernel/005's `<c>_frag`)
        self.accum_frags = tuple(m.c for m in consume)  # consume cells' C = the output accumulators
        self.weight_frags = tuple(f"Pf{j}" for j in range(len(produce)))  # the probability — minted (not a cell field)
        self._reset()

    def _reset(self) -> None:
        # Per-``combine`` projection scratch (so a Twist is reusable across calls).
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
        carrier (``split_carrier``), **project** the stats merge onto this Twist (``stats.project`` →
        ``Monoid.project``), then add the accum rescale (``O·α``), the per-state ``Init`` seeds + the
        accum ``RegFragment`` decls, and the normalize epilogue (``O / l``). The merge body is pure
        carrier projection (no softmax knowledge); only the Mma coupling (the weight/accum fragment
        bindings) lives here."""
        from deplodock.compiler.ir.stmt.carrier_algebra import split_carrier  # noqa: PLC0415

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

        # init: the carried-state identity seeds, then the accum C-fragment declarations.
        init_out: list[Stmt] = []
        for st in stats.state:
            op = self.state_fold_op[st]
            init_out += [Init(name=f"{st}0", op=op, dtype=F32), Init(name=f"{st}1", op=op, dtype=F32)]
        init_out += [RegFragment(name=fr, role="c", shape=self.atom.shape, dtype=F32) for fr in self.accum_frags]

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
        coordinate vars; ``col_bases`` the per-N-atom column origins (+ ``row_base`` when the predicate
        references the row). Knows nothing about causal / boundary / softmax — those are coordinate
        predicates the caller builds (``__fcol > __frow`` / ``__fcol >= seq_len`` / a windowed band /
        …). Sequencing two masks ANDs their keep-predicates."""
        return [
            FragmentMask(frag=sf, mask_when=mask_when, col_base=cb, row_base=row_base, layout=self.layout)
            for sf, cb in zip(self.partial_frags, col_bases, strict=True)
        ]
