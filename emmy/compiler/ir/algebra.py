"""Bottom-up algebra analysis — derive each reduce loop's algebraic *kind*
from its carrier and loads (Part B of ``plans/algebraic-carrier-analysis.md``).

This is the inverse of the bespoke top-down recognizers (``loop/recognize``):
rather than hand-match a known pattern, *read back* the algebra that already
lives in the loop body and expose a uniform tag the scheduler can dispatch on.

The tag is a **derived cache, not a second source of truth**: it is computed
from the loop body on demand (`classify_algebra` / the `Loop.algebra_kind`
property), so it can never contradict the carrier's own traits — the
`MonoidNode(associative=False)` class of bug is unrepresentable. Because it is
computed, not stored, it never enters equality / `op_cache_key`, and it is
always consistent with the current body (re-derived after every rewrite). The
*expensive* match — turning a raw coupled-accumulator cluster into a verified
twisted monoid — is done ONCE by the recognizer (``loop/recognize``, which
emits a `Monoid` carrier); this classification is then a cheap read of that
carrier.

A twisted monoid **is** a monoid (transport of structure — the rescale-by-max
bijection conjugates a plain direct-product monoid), so a `Monoid` carrier
classifies as ``MONOID`` here, the same kind as a scalar `Accum` reduce. The
``MONOID`` algebra is not where flash's streaming schedule is chosen: that is a
*structural* property (a tuple `Monoid` carrier streaming over a nested
contraction), read off the iteration DAG by the tile classifier
(``lowering/tile/enumeration/_classify``), one layer below the algebra. See
``plans/twisted-monoid-carrier-design.md``.

The kind is well-defined where the **carrier is present** — i.e. on a `LoopOp`
body's reduce loops, before the partition planner tiles the carrier away (a
matmul becomes `Mma`/`ldmatrix` fragments; a `Monoid` combine becomes an
explicit rescale + `Accum`). Post-partition nothing re-derives it.
"""

from __future__ import annotations

from enum import Enum

from emmy.compiler.ir.stmt.base import ReduceCarrier
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Mma


class AlgebraKind(Enum):
    """The algebraic kind of a loop scope, derived bottom-up from its carrier.

    A non-reduce scope is ``MAP`` (the default — a pointwise functor); a reduce
    loop is one of the fold kinds below.
    """

    MAP = "map"  # pointwise / functor — a non-reduce scope (the default)
    MONOID = "monoid"  # an associative reduce (carrier: Accum, or a tuple `Monoid` — flash / Welford)
    SEMIRING = "semiring"  # a contraction / matmul           (carrier: Mma, or matmul-shaped Accum)
    SCAN = "scan"  # prefix / causal — reserved, out of scope v1


def matmul_reduce(loop) -> bool:
    """True iff ``loop`` is a reduce loop whose body matches the matmul
    signature: ≥ 2 distinct buffers with K-indexed Loads (K = ``loop.axis.name``)
    plus at least one :class:`ReduceCarrier` (`Accum`, or its fused `Mma`).

    The ir-level structural core of ``lowering/tile/_helpers.is_matmul_reduce``;
    duck-typed on ``.is_reduce`` / ``.axis`` / ``.body`` so it serves Loop-IR
    ``Loop`` / ``StridedLoop`` and Tile-IR ``SerialTile`` / ``StridedTile``
    alike (the caller restricts the type)."""
    if not getattr(loop, "is_reduce", False):
        return False
    k_name = loop.axis.name
    bufs = {ld.input for ld in loop.body.of_type(Load) if k_name in {v for e in ld.index for v in e.free_vars()}}
    if len(bufs) < 2:
        return False
    return any(isinstance(s, ReduceCarrier) for s in loop.body)


def contains_matmul_reduce(stmt) -> bool:
    """True iff ``stmt`` is or transitively contains a matmul-shape reduce
    ``Loop`` (``matmul_reduce``). Recurses through every nested body — the
    shared core behind the planner's fused-prologue probe and the
    demoted-split cut's K-loop search."""
    if isinstance(stmt, Loop) and matmul_reduce(stmt):
        return True
    return any(contains_matmul_reduce(c) for body in stmt.nested() for c in body)


def _is_semiring_contraction(loop) -> bool:
    """A matmul-shaped reduce whose product ``⊗`` distributes over the reduce
    combine ``⊕`` — a genuine semiring contraction, not just any 2-load reduce.

    The fused `Mma` carrier *is* the ``(×, +)`` tensor-core fold (the product is
    folded into the instruction); a scalar matmul cell is an `Accum` fed by a
    distributing product `Assign`."""
    if not matmul_reduce(loop):
        return False
    body = loop.body
    if any(isinstance(s, Mma) for s in body):
        return True
    assigns = {s.name: s for s in body if isinstance(s, Assign)}
    for acc in body:
        if isinstance(acc, Accum):
            prod = assigns.get(acc.value)
            if prod is not None and prod.op.distributes_over(acc.op):
                return True
    return False


def classify_algebra(loop) -> AlgebraKind:
    """Classify a loop scope by its carrier — the cheap bottom-up read.

    - ``MAP`` — a non-reduce scope (no `ReduceCarrier` in the immediate body).
    - ``SEMIRING`` — a matmul-shaped reduce whose product distributes over its
      reduce (`Mma`, or an `Accum` fed by a distributing product).
    - ``MONOID`` — an associative reduce: a scalar `Accum` (a plain reduce), OR a
      recognized tuple `Monoid` carrier (flash's online softmax / Welford). A
      twisted monoid is a monoid (transport of structure), so both land here;
      `has_identity` makes either maskable. The streaming-flash *schedule* is
      selected structurally one layer below, not by a distinct algebra kind.
    - ``MAP`` (fallback) — a reduce the analysis doesn't recognize (no
      regression: callers keep their current path)."""
    if not getattr(loop, "is_reduce", False):
        return AlgebraKind.MAP
    if _is_semiring_contraction(loop):
        return AlgebraKind.SEMIRING
    body = loop.body
    carriers = [s for s in body if isinstance(s, ReduceCarrier)]
    if carriers and all(c.associative for c in carriers):
        return AlgebraKind.MONOID
    return AlgebraKind.MAP
