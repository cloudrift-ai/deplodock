"""Carrier **combiners** — the realizers that project a streaming ``Monoid`` carrier's combine onto a
concrete tier (fragment registers, scalars, …) as a set of per-iteration :class:`CombinePhases`.

One carrier algebra, many realizations (see ``ir/stmt/carrier_algebra``). A combiner is the
``Distribution`` backend ``Monoid.project`` drives PLUS the surrounding phase orchestration (the
carried-state seeds, the accumulator rescale ``O·α`` / declare / normalize ``O/l``). The orchestration
is carrier-generic and lives once, in :meth:`Combiner.combine`; only the tier-specific *emission*
(which IR nodes a fold / pointwise / rescale becomes) differs per backend:

- :class:`MmaTwist` — the genuine **twist** between two SEMIRING contractions (flash's online softmax
  between the QK^T *produce* and P@V *consume* ``Mma`` cells), realized at the tensor-core register
  tier: folds → ``FragmentRowReduce``, pointwise → ``FragmentApply``, accum → ``RegFragment``. Built
  FROM the two ``Mma`` cells so the geometry/registers are derived off them, never duplicated.

``Combiner`` lives at ``ir/`` rather than under a single IR layer because it spans them: it consumes
the ``Monoid`` carrier (``ir/stmt``) and (for :class:`MmaTwist`) emits the fragment ops + storage
nodes (``ir/kernel/ir``: ``FragmentApply`` / ``FragmentRowReduce`` / ``FragmentMask`` /
``RegFragment``).

Carrier-generic — no softmax / causal / boundary knowledge: the carrier supplies the merge, the
caller (the attention assembler) supplies the mask predicate. The only attention name that survives
is in the *caller*, not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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
class CombinePhases:
    """The realized per-iteration carrier phases for a streaming ``CarryScope`` — what
    :meth:`Combiner.combine` produces, at whatever tier the backend realizes. ``init`` seeds the
    carried-state identities + declares the accumulators; ``epilogue`` is the per-output normalize
    (``O / l``); the caller interleaves the stores. ``update`` is empty (state reassigned in
    ``merge``). Not softmax- or tier-specific — any streaming carrier projected onto any backend
    yields these phases."""

    init: tuple[Stmt, ...]
    merge: tuple[Stmt, ...]
    rescale: tuple[Stmt, ...]
    update: tuple[Stmt, ...]
    epilogue: tuple[Stmt, ...]


def _weight_name(accum_merge: tuple, value: str) -> str | None:
    """The distributed SSA the embedded consume contraction multiplies the value by — the operand of
    the accum merge's value-multiply (``coeff · value``). For :class:`MmaTwist` it must land in
    ``weight_frags`` (the registers the consume ``Mma``'s A operand reads). Read structurally off the
    carrier — no softmax / ``exp`` knowledge (flash's ``p`` is one instance)."""
    for a in accum_merge:
        if a.op.name == "multiply" and value in a.args and len(a.args) == 2:
            return a.args[0] if a.args[1] == value else a.args[1]
    return None


class Combiner(ABC):
    """A carrier **combiner** — the ``Distribution`` backend ``Monoid.project`` drives plus the
    phase orchestration that wraps it. :meth:`combine` is the one carrier-generic driver shared by
    every tier; subclasses supply only the tier-specific emission:

    - the ``Distribution`` protocol (:meth:`fold` / :meth:`pointwise` / :meth:`scalar` /
      :meth:`state`) — how a fold / elementwise / replicated-scalar / carried-state step is emitted,
    - the phase hooks (:meth:`bind_score` / :meth:`seed_state` / :meth:`declare_accum` /
      :meth:`rescale_accum` / :meth:`normalize_accum`) — how the accumulator is seeded, declared,
      rescaled by the twist, and normalized.

    The scratch fields (``out`` / ``frag_binding`` / ``state_fold_op`` / ``fold_op_of`` /
    ``weight_name``) are populated during projection and read by :meth:`combine`; :meth:`_reset` is
    called at the top of every :meth:`combine` so a combiner is reusable across calls."""

    def _reset(self) -> None:
        # Per-``combine`` projection scratch (so a combiner is reusable across calls).
        self.out: list[Stmt] = []  # the emitted merge stmts
        self.frag_binding: dict[str, tuple[str, ...]] = {}  # distributed SSA -> backend value(s)
        self.state_fold_op: dict[str, object] = {}  # state -> fold op (drives Init identity + the denominator)
        self.fold_op_of: dict[str, object] = {}  # temp -> fold op (for copy chains, e.g. m = copy(mx))
        self.weight_name: str | None = None  # the distributed SSA bound to the consume contraction's value coeff

    # --- Distribution protocol (driven by Monoid.project) — tier-specific emission ---

    @abstractmethod
    def fold(self, name, op, src, scalar, *, is_state) -> None: ...

    @abstractmethod
    def pointwise(self, name, op, args, distributed) -> None: ...

    @abstractmethod
    def scalar(self, name, op, args) -> None: ...

    @abstractmethod
    def state(self, name, op, args) -> None: ...

    # --- phase hooks — tier-specific accumulator handling ---

    @abstractmethod
    def bind_score(self, score: str) -> None:
        """Bind the distributed input (the score partial) to this backend's representation, before
        projection — for :class:`MmaTwist` the per-N-atom score fragments."""

    @abstractmethod
    def seed_state(self, name: str, op) -> list[Stmt]:
        """The carried-state identity seed(s) for state ``name`` folded by ``op`` (e.g. an ``Init``
        per distributed row)."""

    @abstractmethod
    def declare_accum(self) -> list[Stmt]:
        """Declare the output accumulator(s) (e.g. the accum C ``RegFragment``s)."""

    @abstractmethod
    def rescale_accum(self, scalar: str) -> list[Stmt]:
        """Rescale each accumulator by the per-row carrier scalar ``scalar`` (the twist's ``O·α``)."""

    @abstractmethod
    def normalize_accum(self, denom: str) -> list[Stmt]:
        """Normalize each accumulator by the per-row denominator state ``denom`` (``O / l``)."""

    # --- the one carrier-generic combine driver ---

    def combine(self, carrier: Monoid) -> CombinePhases:
        """Generate the per-iteration phases for a streaming ``Monoid``: split the twisted carrier
        (``split_carrier``), **project** the stats merge onto this backend (``stats.project`` →
        ``Monoid.project``), then add the accum rescale (``O·α``), the per-state seeds + the accum
        declarations, and the normalize epilogue (``O / l``). The merge body is pure carrier
        projection (no softmax knowledge); only the tier-specific emission lives in the backend
        hooks. ``combine(carrier, backend)`` is spelled ``backend.combine(carrier)``."""
        from deplodock.compiler.ir.stmt.carrier_algebra import split_carrier  # noqa: PLC0415

        self._reset()
        stats, accum, d_state = split_carrier(carrier, carrier.partial[1])
        score = stats.partial[0]
        self.weight_name = _weight_name(accum.merge, carrier.partial[1])
        self.bind_score(score)
        stats.project(stats.merge, distributed_inputs={score}, dist=self)

        # accum.merge: only the O·α rescale is realized here (in-place per-row multiply); p·v +
        # O = O·α + p·v are the SEMIRING consume contraction, accumulated into the accumulators.
        rescale_out: list[Stmt] = []
        for a in accum.merge:
            if a.name != d_state and a.op.name == "multiply" and d_state in a.args:
                scalar = a.args[0] if a.args[1] == d_state else a.args[1]
                rescale_out += self.rescale_accum(scalar)

        # init: the carried-state identity seeds, then the accumulator declarations.
        init_out: list[Stmt] = []
        for st in stats.state:
            init_out += self.seed_state(st, self.state_fold_op[st])
        init_out += self.declare_accum()

        # The denominator is the add-fold stats state (flash's l); normalize each accum by it per row.
        denom = next(st for st in stats.state if self.state_fold_op[st].name == "add")
        epilogue_out = self.normalize_accum(denom)
        return CombinePhases(
            init=tuple(init_out), merge=tuple(self.out), rescale=tuple(rescale_out), update=(), epilogue=tuple(epilogue_out)
        )


class MmaTwist(Combiner):
    """The MONOID **twist** between two SEMIRING contractions — flash's online softmax sitting between
    the QK^T (produce) and P@V (consume) ``Mma`` cells, realized at the tensor-core register tier.
    Built FROM those cells: the geometry (the atom + the C-fragment register roles) is **derived** off
    them, not re-specified, so ``MmaTwist`` and ``Mma`` don't duplicate the register/atom info
    (``Mma`` owns the contraction, ``MmaTwist`` the twist over the fragments it produces).

    - :meth:`combine` (inherited) projects the carrier's merge onto the fragment tier: fold →
      ``FragmentRowReduce`` (cross-column reduce) + the per-row scalar update; pointwise → one
      ``FragmentApply`` per N-atom (landing in ``weight_frags`` when it's the consume contraction's
      value-coefficient, else a fresh fragment); scalar / carried-state → row-distributed ``Assign``
      / ``Reassign``. It also declares the accum ``RegFragment``s.
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

    # --- phase hooks (fragment-tier emission) ---

    def bind_score(self, score: str) -> None:
        self.frag_binding = {score: self.partial_frags}

    def seed_state(self, name: str, op) -> list[Stmt]:
        return [Init(name=f"{name}0", op=op, dtype=F32), Init(name=f"{name}1", op=op, dtype=F32)]

    def declare_accum(self) -> list[Stmt]:
        return [RegFragment(name=fr, role="c", shape=self.atom.shape, dtype=F32) for fr in self.accum_frags]

    def rescale_accum(self, scalar: str) -> list[Stmt]:
        alpha = (f"{scalar}0", f"{scalar}1")
        return [
            FragmentApply(out=fr, op=_MULTIPLY, args=(fr, alpha), kinds=(FRAG, ROW), in_place=True, layout=self.layout)
            for fr in self.accum_frags
        ]

    def normalize_accum(self, denom: str) -> list[Stmt]:
        denom_pair = (f"{denom}0", f"{denom}1")
        return [
            FragmentApply(out=fr, op=_DIVIDE, args=(fr, denom_pair), kinds=(FRAG, ROW), in_place=True, layout=self.layout)
            for fr in self.accum_frags
        ]

    # --- mask generator (given the carrier) ---

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
