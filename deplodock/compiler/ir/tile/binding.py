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


__all__ = ["AtomBinding", "Operand"]
