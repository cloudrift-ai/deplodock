"""Generative schedule tree — one algebra-uniform builder.

The tree generates children move-by-move; the search (two-level MCTS / greedy)
branches on each decision. There is **no per-shape code** — the regime is just
the reduce axes' ``Loop.algebra_kind`` and the moves are algebra-licensed:

- a ``MAP`` nest (no contraction) offers only the free-axis ``tile_axis`` move:
  thread → register.
- a ``SEMIRING`` reduce additionally offers the carrier-licensed reduce
  decomposition (``_moves.reduce_offers`` over the contraction axis) at the root:
  reduce → thread → register.

(``MONOID`` cooperative-reduce and the ``TWISTED_MONOID`` flash stream are
recognised by ``classify`` but not yet built — they re-enter on the same uniform
tree once their decomposition placement + assembly land.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.stmt import Loop, Write
from deplodock.compiler.ir.tile.ir import TileGraphOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._build import lower
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import AxisRole, IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._knobs import RED_FK
from deplodock.compiler.pipeline.passes.lowering.tile.enumeration._moves import (
    Budget,
    map_reg_offers,
    reduce_knobs,
    reduce_offers,
    reduce_reg_offers,
    reg_knobs,
    thread_knobs,
    thread_offers,
)


@dataclass(frozen=True)
class _Ctx:
    """Per-LoopOp state shared by every node of one kernel's tree. ``target_names``
    are the contraction-axis names a reduce decomposition rewrites — empty for a
    ``MAP`` nest, which makes :attr:`has_reduce` ``False`` and skips the reduce
    move + its compute-bound register menu."""

    dag: IterDag
    budget: Budget
    base_knobs: dict
    kernel_name: str
    target_names: frozenset[str] = frozenset()

    @property
    def has_reduce(self) -> bool:
        return bool(self.target_names)


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete move stack; ``expand`` emits the ``TileGraphOp``."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [
            lower(
                self.ctx.dag,
                self.knobs,
                kernel_name=self.ctx.kernel_name,
                base_knobs=self.ctx.base_knobs,
                target_names=self.ctx.target_names,
            )
        ]


@dataclass(frozen=True)
class _ThreadChosen(Fork):
    """Thread tile pinned; ``expand`` offers the register tiles legal under the
    cell budget (the compute-bound menu for a reduce regime, else bandwidth)."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        if self.ctx.has_reduce:
            regs = reduce_reg_offers(self.ctx.dag, self.ctx.budget, self.knobs[RED_FK.name])
        else:
            regs = map_reg_offers(self.ctx.dag, self.ctx.budget)
        return [_Leaf(ctx=self.ctx, knobs={**self.knobs, **reg_knobs(self.ctx.dag, reg)}) for reg in regs]


@dataclass(frozen=True)
class _ChooseThread(Fork):
    """``expand`` offers the legal thread tiles (the free-axis ``tile_axis`` move).
    Carries any already-pinned knobs (the reduce decomposition, for a reduce
    regime)."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [
            _ThreadChosen(ctx=self.ctx, knobs={**self.knobs, **thread_knobs(self.ctx.dag, t)})
            for t in thread_offers(self.ctx.dag, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _ChooseReduce(Fork):
    """Root for a reduce regime: ``expand`` offers the carrier-licensed reduce
    decompositions; each chains into the free-axis ``_ChooseThread`` subtree."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [_ChooseThread(ctx=self.ctx, knobs={**self.knobs, **reduce_knobs(r)}) for r in reduce_offers(self.ctx.dag)]


# --- Regime classification (the reduce axes' algebra) + the partition entry. ---


@dataclass(frozen=True)
class _Regime:
    """The classification handoff: the nest's algebra + the contraction-axis names
    a reduce decomposition rewrites (``target_names``), plus the flash streaming
    ``k_bounds``."""

    algebra: AlgebraKind  # MAP | SEMIRING | MONOID | TWISTED_MONOID
    target_names: frozenset[str] = frozenset()
    k_bounds: dict = field(default_factory=dict)


def classify(dag: IterDag) -> _Regime | None:
    """Tag the nest's regime off the derived DAG — purely the reduce axes'
    ``Loop.algebra_kind`` (``MAP`` no contraction, ``SEMIRING`` a contraction,
    ``MONOID`` plain reduce, ``TWISTED_MONOID`` online stream), or ``None`` for a
    shape the moves don't yet cover. No shape matching."""
    if not dag.parallel:
        return None
    reduce_loops = [n.loop for n in dag.reduce]
    algebras = dag.algebras
    inner_body = dag.inner_body
    nested_reduce = any(n.parent is not None and n.parent.role is AxisRole.REDUCE for n in dag.reduce)
    coop_monoid = AlgebraKind.TWISTED_MONOID in algebras and not nested_reduce

    if AlgebraKind.TWISTED_MONOID in algebras and nested_reduce:
        if len(dag.parallel) < 2:
            return None
        twisted = [lp for lp in reduce_loops if lp.algebra_kind == AlgebraKind.TWISTED_MONOID]
        if any(not lp.axis.extent.is_static for lp in reduce_loops if lp.algebra_kind != AlgebraKind.TWISTED_MONOID):
            return None
        k_bounds = {lp.axis.name: lp.axis.extent.expr for lp in twisted if not lp.axis.extent.is_static}
        return _Regime(AlgebraKind.TWISTED_MONOID, frozenset(lp.axis.name for lp in reduce_loops), k_bounds)

    if not reduce_loops:  # no contraction — a MAP nest.
        return _Regime(AlgebraKind.MAP)

    k_dim = reduce_loops[0].axis.extent
    body_loops = [s for s in inner_body if isinstance(s, Loop)]

    if algebras == {AlgebraKind.SEMIRING}:
        if not k_dim.is_static or len(dag.parallel) < 2:
            return None
        if {lp.axis.extent for lp in reduce_loops} != {k_dim}:
            return None
        if not body_loops or any(lp.axis.extent != k_dim or not lp.is_reduce for lp in body_loops):
            return None
        if not any(isinstance(s, Write) for s in inner_body):
            return None
        return _Regime(AlgebraKind.SEMIRING, frozenset(lp.axis.name for lp in body_loops))

    if algebras == {AlgebraKind.MONOID} or coop_monoid:
        if coop_monoid and not k_dim.is_static:
            return None
        k_extent = k_dim.as_static() if k_dim.is_static else (k_dim.hint or 0)
        if k_extent < 1:
            return None
        if {lp.axis.extent for lp in reduce_loops} != {k_dim}:
            return None
        if any(lp.axis.extent != k_dim for lp in body_loops):
            return None
        return _Regime(AlgebraKind.MONOID, frozenset(lp.axis.name for lp in body_loops))

    return None


def build_partition(*, dag: IterDag, loop_op, context, graph, base_knobs: dict, kernel_name: str) -> Fork | TileGraphOp | None:  # noqa: ARG001
    """The single partition entry. ``classify`` tags the algebra; the SAME uniform
    tree builds it — a ``MAP`` nest from the free-axis root, a ``SEMIRING`` reduce
    from the reduce root. ``MONOID`` / ``TWISTED_MONOID`` return ``None`` (their
    decomposition placement + assembly are not yet built). ``loop_op`` /
    ``context`` / ``graph`` are kept for the tensor-core atom eligibility a future
    ``SEMIRING`` warp-tier move reads."""
    r = classify(dag)
    if r is None:
        return None
    ctx = _Ctx(dag=dag, budget=Budget(), base_knobs=base_knobs, kernel_name=kernel_name, target_names=r.target_names)
    if r.algebra is AlgebraKind.MAP:
        threads, regs = thread_offers(dag, ctx.budget), map_reg_offers(dag, ctx.budget)
        if not threads or not regs:
            return None
        if len(threads) == 1 and len(regs) == 1:  # single legal variant — skip the fork
            full = {**thread_knobs(dag, threads[0]), **reg_knobs(dag, regs[0])}
            return lower(dag, full, kernel_name=kernel_name, base_knobs=base_knobs)
        return _ChooseThread(ctx=ctx, knobs={})
    if r.algebra is AlgebraKind.SEMIRING:
        reduces, threads = reduce_offers(dag), thread_offers(dag, ctx.budget)
        if not reduces or not threads:
            return None
        regs0 = reduce_reg_offers(dag, ctx.budget, reduces[0][1])
        if len(reduces) == 1 and len(threads) == 1 and len(regs0) == 1:
            full = {**reduce_knobs(reduces[0]), **thread_knobs(dag, threads[0]), **reg_knobs(dag, regs0[0])}
            return lower(dag, full, kernel_name=kernel_name, base_knobs=base_knobs, target_names=r.target_names)
        return _ChooseReduce(ctx=ctx, knobs={})
    return None

