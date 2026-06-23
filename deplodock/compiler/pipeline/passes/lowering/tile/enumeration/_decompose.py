"""The decomposition move — factor a reduce axis, recombine via the carrier.

A single carrier-parameterized move replaces the per-regime offer functions'
hand-coded legality: ``legal_decomps`` enumerates the factorizations the reduce
node's carrier *algebra* licenses (``associative`` → split at all;
``commutative`` → reorder / cross-CTA combine; ``has_identity`` → mask a
non-divisible / symbolic axis with the carrier identity), bounded by the cell /
thread budget. The recombination operator is derived (``carrier.combine_partials``)
and the hardware realization (atomic / shuffle / tree / mma) is chosen downstream
from the placement — neither is stored here. See
``plans/algebra-licensed-decomposition-moves.md`` (phase 3).

Split-K (strip + split + chunk over a matmul ``SEMIRING`` reduce) and cooperative
reduce (chunk + strip + cooperative-thread over a ``MONOID`` reduce) are two
instances of this ONE move, differing only in the placement of the factored
pieces and how the recombine is realized — both cost / hardware choices kept
OUT of the legality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.stmt import ReduceCarrier
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._tower import Role


@dataclass(frozen=True)
class AxisDecomp:
    """A factorization of one reduce ``axis`` into ``factors`` (extent product
    ``== axis.extent`` unless masked), each piece placed by ``placement``.

    The recombine operator and the realization are NOT stored — they are derived:
    the recombine is ``carrier.combine_partials()`` (algebra); the realization
    (atomic / shuffle / tree / mma / serial) is the cost+hardware choice keyed off
    ``placement``."""

    axis: Axis
    factors: tuple[int, ...]
    placement: tuple[Role, ...]


def legal_decomps(
    carrier: ReduceCarrier | None,
    axis: Axis,
    extent: int,
    *,
    factor_menus: Sequence[Sequence[int]],
    placement: Sequence[Role],
    masked: bool,
    allow_split: bool = True,
) -> list[AxisDecomp]:
    """The factorizations the ``carrier`` algebra licenses over ``axis``.

    ``factor_menus`` is one candidate menu per factored piece (e.g. split / chunk
    / strip for matmul, or cooperative / chunk / strip for coop reduce);
    ``placement`` names where each piece lands (the same length). The legality is
    a carrier-trait query:

    - **associative** licenses splitting the axis at all. A non-associative
      carrier admits only the trivial all-``1`` factorization (no recombine).
    - **commutative** licenses a *partitioning* factor > 1 whose recombine
      reorders partials (split-K cross-CTA / cooperative-tree) — the FIRST factor
      by convention (``placement[0]`` is the partition: ``BLOCK`` split-K /
      ``THREAD`` cooperative). ``allow_split`` is the orthogonal cost/soundness
      gate the caller supplies (a non-linear epilogue / multi-accumulator matmul
      forbids split-K regardless of algebra).
    - **has_identity** licenses a ``masked`` (ceil-div + identity-fill)
      factorization of a non-divisible / symbolic axis; without it the product
      must divide ``extent`` exactly.

    A **PARALLEL** axis (``carrier is None``) is the degenerate, no-recombine case
    (phase 7): free-axis tiling (block × thread × register) and the tensorize
    atom-block are product decompositions of *independent* work, so every
    factorization is legal (no associativity needed — there is nothing to
    recombine), masking is a plain boundary store-guard (no carrier identity), and
    a partition factor needs no commutativity.

    Returns the legal :class:`AxisDecomp`s unranked — pruning / best-first
    ordering stays with the caller (cost, not algebra)."""
    parallel = carrier is None
    if masked and not parallel and not carrier.has_identity:
        return []  # can't identity-fill a fill-less reduce carrier (a parallel axis masks via a store guard)
    splittable = parallel or carrier.associative
    can_partition = parallel or (allow_split and carrier.commutative)
    placement_t = tuple(placement)

    out: list[AxisDecomp] = []

    def _emit(combo: tuple[int, ...]) -> None:
        product = 1
        for f in combo:
            product *= f
        if product > extent:
            return
        if not masked and extent % product != 0:
            return
        if not splittable and product != 1:
            return
        if combo[0] != 1 and not can_partition:
            return
        out.append(AxisDecomp(axis=axis, factors=combo, placement=placement_t))

    _enumerate(factor_menus, (), _emit)
    return out


def _enumerate(menus: Sequence[Sequence[int]], prefix: tuple[int, ...], emit) -> None:
    """Cartesian product of the per-factor menus, calling ``emit`` per combo."""
    if not menus:
        emit(prefix)
        return
    for v in menus[0]:
        _enumerate(menus[1:], (*prefix, v), emit)
