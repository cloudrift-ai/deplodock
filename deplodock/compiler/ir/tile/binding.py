"""Atom bindings — the structural algebra→hardware-atom resolution stamped on a schedule.

The ``040_atomize`` tile pass resolves, *once and structurally*, the facts the kernel
materializers used to re-recognize from lowered loop-IR: which contracted operand is the
mma ``a`` vs ``b`` (by which grid output axis its index carries), whether ``b`` is
transposed, the fold accumulator, and the fused-projection epilogue. The result rides the
**schedule** (a sibling of :class:`~.schedule.WarpTile`), NOT the op tree — so the
``Semiring`` combine stays the single source of truth and ``op_cache_key`` (which digests
``lower(op.op)``, not the schedule) is untouched.

These are tile-tier, abstract: they reference the algebra's leaf :class:`Load` exprs, never
kernel-IR realization nodes (``MmaSyncPtx`` / ``LdmatrixLoad``)."""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Load
from deplodock.compiler.ir.tile.atom import MonoidAtom


@dataclass(frozen=True)
class Operand:
    """One contracted operand bound to its tensor-core role. ``load`` is the operand
    ``Map``'s leaf read (``body[-1]`` — its ``.input`` buffer + ``.index`` exprs, what the
    staged kloops σ-apply); ``role`` is ``"a"`` / ``"b"``, resolved by which grid output
    axis the index carries."""

    load: Load
    role: str  # "a" | "b"


@dataclass(frozen=True)
class AtomBinding:
    """The atom-tier operand→role binding ``040_atomize`` resolves off the ``Semiring`` —
    the structural facts the warp materializer used to re-discover from lowered loop-IR.
    The forward sibling of :class:`~.schedule.WarpTile` (which carries the *atom geometry*
    decision); read only by ``_warp``. Carries NO ``atom`` (that's ``WarpTile.atom``) and no
    ``reduce_axis`` / ``m_axis`` / ``n_axis`` (those stay on the ``Semiring`` / the grid) —
    only what isn't already on the schedule or the still-present combine."""

    a: Operand  # the m-bearing operand (A)
    b: Operand  # the n-bearing operand (B)
    b_trans: bool  # B[n,k] (K last in index) vs canonical B[k,n]
    acc: str  # the fold accumulator name (== Semiring.out / fold.name)
    epilogue: Body = field(default_factory=Body)  # the projection Map body (scale/bias/relu/
    # residual + the output Write + any loop-invariant scalar Loads); empty Body = a bare
    # contraction (_warp emits the accumulator Write itself).

    def pretty(self) -> str:
        """One-line dump summary: ``bind: a:<buf>@m b:<buf>@n[ trans][ +epi] -> <acc>``."""
        trans = " trans" if self.b_trans else ""
        epi = " +epi" if len(self.epilogue) else ""
        return f"bind: a:{self.a.load.input}@m b:{self.b.load.input}@n{trans}{epi} -> {self.acc}"


@dataclass(frozen=True)
class ReduceBinding:
    """The cooperative-combine binding ``040_atomize`` resolves off a ``Monoid`` carrier + its
    ``ReducePlan`` — the ``Monoid``-kind sibling of :class:`AtomBinding`. Carries the
    :class:`~.atom.MonoidAtom` (the per-component accumulator dtype) and the partition widths;
    the fold-mechanism sequence (shuffle / tree) is **derived** on demand (:meth:`folds`),
    never stored — ``ReduceStage.combine`` owns that decision (level + width), so a stored copy
    would be the recovered tag the rebuild forbids."""

    atom: MonoidAtom
    coop: int  # cooperating BLOCK threads (1 = ILP-only / scalar lane, no cross-thread combine)
    reg: int = 1  # ILP register-fold copies
    segmented: bool = False  # strided-row segmented combine (the plain cooperative path is False)

    def folds(self, warp_size: int = 32) -> tuple:
        """The derived per-level combine mechanism (a tuple of ``Fold``) — ``ReduceStage.combine``,
        not a stored field. Empty when there is no cross-thread combine (``coop == 1``)."""
        if self.coop <= 1:
            return ()
        from deplodock.compiler.ir.tile.schedule import Level, ReduceStage  # noqa: PLC0415

        return ReduceStage(Level.BLOCK, self.coop).combine(warp_size=warp_size, segmented=self.segmented)

    def pretty(self) -> str:
        """One-line dump summary: ``reduce: coop:<n> reg:<n> <dtype> [<mech>]``."""
        mech = "/".join(f.name.lower() for f in self.folds()) or "serial"
        return f"reduce: coop:{self.coop} reg:{self.reg} {self.atom.dtype.name} [{mech}]"


__all__ = ["AtomBinding", "Operand", "ReduceBinding"]
