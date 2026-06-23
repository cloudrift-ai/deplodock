"""Generative schedule tree — pointwise + scalar matmul (block-DAG path).

The tree generates children move-by-move: the root offers legal thread tiles,
each thread branch offers the register tiles legal *given that thread tile*, and
each leaf lowers via the block-DAG path (``build_dag`` → ``assemble``).

The warp-tier MMA, cooperative-reduce, and fused-flash subtrees were deleted in
the tile-ir-block-dag demolition (``plans/tile-ir-block-dag.md``) — they will be
re-added on the rebuilt ``assemble → KernelOp`` path. ``classify`` still tags all
four regimes (``005_split_demoted`` consults it), but ``build_partition`` only
builds pointwise + scalar matmul; coop / flash hit a hard error (010 raises),
quarantined at the test layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.stmt import Loop, Write
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.build_dag import lower_matmul, lower_pointwise
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import AxisRole, IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import RED_FK
from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import (
    matmul_reduce_offers,
    matmul_reg_offers,
    matmul_thread_offers,
    reduce_knobs,
    reg_knobs,
    reg_offers,
    thread_knobs,
    thread_offers,
)


@dataclass(frozen=True)
class _Ctx:
    """Per-LoopOp state shared by every node of one kernel's tree."""

    dag: IterDag
    budget: Budget
    base_knobs: dict
    kernel_name: str
    target_names: frozenset[str] = frozenset()
    k_bounds: dict = field(default_factory=dict)


# --- Pointwise (MAP) generative tree: thread → register. ---


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete pointwise move stack; ``expand`` lowers via the block-DAG path."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [lower_pointwise(self.ctx.dag, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)]


@dataclass(frozen=True)
class _ThreadChosen(Fork):
    """Thread tile pinned; ``expand`` offers the register tiles legal under the
    cell budget."""

    ctx: _Ctx
    thread: tuple[int, int]
    knobs: dict

    def expand(self) -> list:
        return [
            _Leaf(ctx=self.ctx, knobs={**self.knobs, **reg_knobs(self.ctx.dag, reg)}) for reg in reg_offers(self.ctx.dag, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _ChooseThread(Fork):
    """Root: ``expand`` offers the legal thread tiles."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [
            _ThreadChosen(ctx=self.ctx, thread=t, knobs=thread_knobs(self.ctx.dag, t)) for t in thread_offers(self.ctx.dag, self.ctx.budget)
        ]


def build_pointwise_tree(*, dag: IterDag, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Root ``Fork`` of the pointwise tree, a bare ``TileOp`` for a single legal
    variant, or ``None`` when nothing is legal."""
    budget = Budget()
    threads = thread_offers(dag, budget)
    regs = reg_offers(dag, budget)
    if not threads or not regs:
        return None
    ctx = _Ctx(dag=dag, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name)
    if len(threads) == 1 and len(regs) == 1:
        full = {**thread_knobs(dag, threads[0]), **reg_knobs(dag, regs[0])}
        return lower_pointwise(dag, full, kernel_name=kernel_name, base_knobs=base_knobs)
    return _ChooseThread(ctx=ctx, knobs={})


# --- Scalar matmul (SEMIRING) generative tree: reduce → thread → register. ---


@dataclass(frozen=True)
class _MmLeaf(Fork):
    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [
            lower_matmul(
                self.ctx.dag,
                self.knobs,
                kernel_name=self.ctx.kernel_name,
                base_knobs=self.ctx.base_knobs,
                target_names=self.ctx.target_names,
            )
        ]


@dataclass(frozen=True)
class _MmThreadChosen(Fork):
    """Reduce + thread tiles pinned; ``expand`` offers register tiles."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        fk = self.knobs[RED_FK.name]
        return [
            _MmLeaf(ctx=self.ctx, knobs={**self.knobs, **reg_knobs(self.ctx.dag, reg)})
            for reg in matmul_reg_offers(self.ctx.dag, self.ctx.budget, fk)
        ]


@dataclass(frozen=True)
class _MmReduceChosen(Fork):
    """Reduce tile pinned; ``expand`` offers thread tiles."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [
            _MmThreadChosen(ctx=self.ctx, knobs={**self.knobs, **thread_knobs(self.ctx.dag, t)})
            for t in matmul_thread_offers(self.ctx.dag, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _MmChooseReduce(Fork):
    """Root: ``expand`` offers the legal K-tilings (``bk``, ``fk``)."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [_MmReduceChosen(ctx=self.ctx, knobs=reduce_knobs(r)) for r in matmul_reduce_offers(self.ctx.dag)]


def _scalar_subtree(ctx: _Ctx) -> Fork | TileOp | None:
    """The scalar-tier matmul subtree (reduce → thread → register), a bare
    ``TileOp`` for a single variant, or ``None`` if nothing is legal."""
    dag = ctx.dag
    reduces = matmul_reduce_offers(dag)
    threads = matmul_thread_offers(dag, ctx.budget)
    if not reduces or not threads:
        return None
    regs0 = matmul_reg_offers(dag, ctx.budget, reduces[0][1])
    if len(reduces) == 1 and len(threads) == 1 and len(regs0) == 1:
        full = {**reduce_knobs(reduces[0]), **thread_knobs(dag, threads[0]), **reg_knobs(dag, regs0[0])}
        return lower_matmul(dag, full, kernel_name=ctx.kernel_name, base_knobs=ctx.base_knobs, target_names=ctx.target_names)
    return _MmChooseReduce(ctx=ctx, knobs={})


# --- Regime classification + the single partition entry. ---


@dataclass(frozen=True)
class _Regime:
    """The transient classification handoff: the regime tag plus the K-info
    derived off the DAG — ``target_names`` (the K-extent loop names a reduce
    transform rewrites) and the flash streaming ``k_bounds``."""

    kind: str  # "pointwise" | "matmul" | "coop" | "flash"
    target_names: frozenset[str] = frozenset()
    k_bounds: dict = field(default_factory=dict)


def classify(dag: IterDag) -> _Regime | None:
    """Tag the nest's regime off the derived DAG (``MAP`` → pointwise, ``SEMIRING``
    → matmul, ``MONOID`` → cooperative reduce, ``TWISTED_MONOID`` → flash if it
    nests a reduce, else a cooperative monoid), or ``None`` for an unrecognized
    shape. The recognition predicate ``005_split_demoted`` consults."""
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
        return _Regime("flash", frozenset(lp.axis.name for lp in reduce_loops), k_bounds)

    if not reduce_loops:  # PARALLEL-only — pointwise.
        return _Regime("pointwise")

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
        return _Regime("matmul", frozenset(lp.axis.name for lp in body_loops))

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
        return _Regime("coop", frozenset(lp.axis.name for lp in body_loops))

    return None


def build_partition(*, dag: IterDag, loop_op, context, graph, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:  # noqa: ARG001
    """The single partition entry. ``classify`` tags the regime; only pointwise +
    scalar matmul are built (warp / coop / flash deleted in the demolition — they
    return ``None`` → ``010`` raises, quarantined). ``loop_op`` / ``context`` /
    ``graph`` are kept for the rebuilt warp-tier's atom eligibility."""
    r = classify(dag)
    if r is None:
        return None
    if r.kind == "pointwise":
        return build_pointwise_tree(dag=dag, base_knobs=base_knobs, kernel_name=kernel_name)
    if r.kind == "matmul":
        ctx = _Ctx(dag=dag, budget=Budget(), base_knobs=base_knobs, kernel_name=kernel_name, target_names=r.target_names)
        return _scalar_subtree(ctx)
    return None
