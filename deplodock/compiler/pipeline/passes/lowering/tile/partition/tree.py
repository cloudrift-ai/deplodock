"""Generative schedule tree for the pointwise regime.

Unlike the legacy ``build_fork_tree`` (which groups a pre-enumerated flat
cartesian post-hoc), this tree generates children move-by-move: the root
offers legal thread tiles, each thread branch offers the register tiles legal
*given that thread tile* (incremental budget pruning), and each leaf
materializes. The two-level MCTS branches on these genuine move decisions —
coarse (thread fan-out) near the root, fine (register cells) at the leaves —
reusing the ``Fork`` / ``LazyCandidate`` engine unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import RED_FK
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_matmul_tile, build_pointwise_tile
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
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import MatmulSkeleton, PointwiseSkeleton


@dataclass(frozen=True)
class _Ctx:
    """Per-LoopOp state shared by every node of one kernel's tree."""

    skel: PointwiseSkeleton | MatmulSkeleton
    budget: Budget
    base_knobs: dict
    kernel_name: str


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete move stack (thread + register tiles); ``expand`` materializes
    the ``TileOp``."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [build_pointwise_tile(self.ctx.skel, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)]


@dataclass(frozen=True)
class _ThreadChosen(Fork):
    """Thread tile pinned; ``expand`` offers the register tiles legal under the
    cell budget."""

    ctx: _Ctx
    thread: tuple[int, int]
    knobs: dict

    def expand(self) -> list:
        return [
            _Leaf(ctx=self.ctx, knobs={**self.knobs, **reg_knobs(self.ctx.skel, reg)}) for reg in reg_offers(self.ctx.skel, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _ChooseThread(Fork):
    """Root: ``expand`` offers the legal thread tiles."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [
            _ThreadChosen(ctx=self.ctx, thread=t, knobs=thread_knobs(self.ctx.skel, t))
            for t in thread_offers(self.ctx.skel, self.ctx.budget)
        ]


def build_pointwise_tree(skel: PointwiseSkeleton, *, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Return the root ``Fork`` of the generative tree, a bare ``TileOp`` when
    only one variant is legal (mirrors the legacy single-variant short-circuit),
    or ``None`` when nothing is legal (the dispatcher falls through to legacy)."""
    budget = Budget()
    threads = thread_offers(skel, budget)
    regs = reg_offers(skel, budget)
    if not threads or not regs:
        return None
    ctx = _Ctx(skel=skel, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name)
    if len(threads) == 1 and len(regs) == 1:
        full = {**thread_knobs(skel, threads[0]), **reg_knobs(skel, regs[0])}
        return build_pointwise_tile(skel, full, kernel_name=kernel_name, base_knobs=base_knobs)
    return _ChooseThread(ctx=ctx, knobs={})


# --- Matmul (SEMIRING) generative tree: reduce → thread → register. ---


@dataclass(frozen=True)
class _MmLeaf(Fork):
    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [build_matmul_tile(self.ctx.skel, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)]


@dataclass(frozen=True)
class _MmThreadChosen(Fork):
    """Reduce + thread tiles pinned; ``expand`` offers register tiles (the cell
    budget depends on the pinned ``fk`` strip-mine)."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        fk = self.knobs[RED_FK.name]
        return [
            _MmLeaf(ctx=self.ctx, knobs={**self.knobs, **reg_knobs(self.ctx.skel, reg)})
            for reg in matmul_reg_offers(self.ctx.skel, self.ctx.budget, fk)
        ]


@dataclass(frozen=True)
class _MmReduceChosen(Fork):
    """Reduce tile pinned; ``expand`` offers thread tiles."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [
            _MmThreadChosen(ctx=self.ctx, knobs={**self.knobs, **thread_knobs(self.ctx.skel, t)})
            for t in matmul_thread_offers(self.ctx.skel, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _MmChooseReduce(Fork):
    """Root: ``expand`` offers the legal K-tilings (``bk``, ``fk``)."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [_MmReduceChosen(ctx=self.ctx, knobs=reduce_knobs(r)) for r in matmul_reduce_offers(self.ctx.skel)]


def build_matmul_tree(skel: MatmulSkeleton, *, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Root ``Fork`` of the generative matmul tree (reduce → thread → register),
    a bare ``TileOp`` when only one variant is legal, or ``None`` when nothing
    is legal (legacy fallthrough)."""
    budget = Budget()
    reduces = matmul_reduce_offers(skel)
    threads = matmul_thread_offers(skel, budget)
    if not reduces or not threads:
        return None
    ctx = _Ctx(skel=skel, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name)
    regs0 = matmul_reg_offers(skel, budget, reduces[0][1])
    if len(reduces) == 1 and len(threads) == 1 and len(regs0) == 1:
        full = {**reduce_knobs(reduces[0]), **thread_knobs(skel, threads[0]), **reg_knobs(skel, regs0[0])}
        return build_matmul_tile(skel, full, kernel_name=kernel_name, base_knobs=base_knobs)
    return _MmChooseReduce(ctx=ctx, knobs={})
