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

from collections.abc import Callable
from dataclasses import dataclass

from deplodock.compiler.ir.stmt import Loop
from deplodock.compiler.ir.tile.ir import Atom, TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import RED_FK, TC_ATOM, WARP_M, WARP_N
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import (
    build_coop_reduce_tile,
    build_flash_tile,
    build_matmul_tile,
    build_pointwise_tile,
    build_warp_matmul_tile,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import (
    coop_reduce_knobs,
    coop_reduce_offers,
    eligible_atoms,
    matmul_reduce_offers,
    matmul_reg_offers,
    matmul_thread_offers,
    reduce_knobs,
    reg_knobs,
    reg_offers,
    thread_knobs,
    thread_offers,
    warp_bk_knobs,
    warp_bk_offers,
    warp_knobs,
    warp_offers,
    warp_reg_knobs,
    warp_reg_offers,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import (
    CoopReduceSkeleton,
    FlashSkeleton,
    MatmulSkeleton,
    PointwiseSkeleton,
)


@dataclass(frozen=True)
class _Ctx:
    """Per-LoopOp state shared by every node of one kernel's tree. ``tile_builder``
    materializes a leaf — `build_pointwise_tile` by default, `build_flash_tile`
    for the flash nest (both free-axis-tiled, differing only in the K transform)."""

    skel: PointwiseSkeleton | MatmulSkeleton | FlashSkeleton
    budget: Budget
    base_knobs: dict
    kernel_name: str
    tile_builder: Callable = build_pointwise_tile


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete move stack (thread + register tiles); ``expand`` materializes
    the ``TileOp``."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [self.ctx.tile_builder(self.ctx.skel, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)]


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


@dataclass(frozen=True)
class _ChooseThreadFlash(Fork):
    """Flash root: a thread-tile-only fork. The streaming online-softmax carrier's
    state (m/l/O) is seeded by an ``Init`` OUTSIDE the per-cell ``RegisterTile``,
    so it can't span register cells — flash forces ``FM=FN=1`` (the same constraint
    legacy's ``mask_f1`` enforces on a fused prologue). Register tiling is not a
    search dimension; each thread choice is a terminal leaf."""

    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        reg1 = reg_knobs(self.ctx.skel, (1, 1))
        return [
            _Leaf(ctx=self.ctx, knobs={**thread_knobs(self.ctx.skel, t), **reg1}) for t in thread_offers(self.ctx.skel, self.ctx.budget)
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


def build_flash_tree(skel: FlashSkeleton, *, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Flash nest: the same free-axis (thread → register) generative tree as
    pointwise, but each leaf materializes via `build_flash_tile` (which serial-
    transforms the streaming reduces). The reduces aren't a search dimension —
    the `FlashCombine` carrier owns the KV recurrence."""
    budget = Budget()
    threads = thread_offers(skel, budget)
    if not threads:
        return None
    # FM=FN=1 only (the streaming carrier can't span register cells), so the
    # register tile is not a search dimension — just the thread tile.
    ctx = _Ctx(skel=skel, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name, tile_builder=build_flash_tile)
    reg1 = reg_knobs(skel, (1, 1))
    if len(threads) == 1:
        full = {**thread_knobs(skel, threads[0]), **reg1}
        return build_flash_tile(skel, full, kernel_name=kernel_name, base_knobs=base_knobs)
    return _ChooseThreadFlash(ctx=ctx, knobs={})


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


def _scalar_subtree(ctx: _Ctx) -> Fork | TileOp | None:
    """The scalar-tier matmul subtree (reduce → thread → register), or a bare
    ``TileOp`` for a single variant, or ``None`` if nothing is legal."""
    skel = ctx.skel
    reduces = matmul_reduce_offers(skel)
    threads = matmul_thread_offers(skel, ctx.budget)
    if not reduces or not threads:
        return None
    regs0 = matmul_reg_offers(skel, ctx.budget, reduces[0][1])
    if len(reduces) == 1 and len(threads) == 1 and len(regs0) == 1:
        full = {**reduce_knobs(reduces[0]), **thread_knobs(skel, threads[0]), **reg_knobs(skel, regs0[0])}
        return build_matmul_tile(skel, full, kernel_name=ctx.kernel_name, base_knobs=ctx.base_knobs)
    return _MmChooseReduce(ctx=ctx, knobs={})


# --- Tensorize: warp-tier (tensor-core MMA) subtree, atom → warp → reg → bk. ---


@dataclass(frozen=True)
class _WarpLeaf(Fork):
    ctx: _Ctx
    atom: Atom
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [
            build_warp_matmul_tile(self.ctx.skel, self.atom, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)
        ]


@dataclass(frozen=True)
class _WarpChooseBk(Fork):
    ctx: _Ctx
    atom: Atom
    knobs: dict

    def expand(self) -> list:
        return [
            _WarpLeaf(ctx=self.ctx, atom=self.atom, knobs={**self.knobs, **warp_bk_knobs(bk)})
            for bk in warp_bk_offers(self.ctx.skel, self.atom)
        ]


@dataclass(frozen=True)
class _WarpChooseReg(Fork):
    ctx: _Ctx
    atom: Atom
    knobs: dict

    def expand(self) -> list:
        warp = (self.knobs[WARP_M.name], self.knobs[WARP_N.name])
        return [
            _WarpChooseBk(ctx=self.ctx, atom=self.atom, knobs={**self.knobs, **warp_reg_knobs(reg)})
            for reg in warp_reg_offers(self.ctx.skel, self.atom, warp)
        ]


@dataclass(frozen=True)
class _WarpChooseWarp(Fork):
    ctx: _Ctx
    atom: Atom
    knobs: dict

    def expand(self) -> list:
        return [
            _WarpChooseReg(ctx=self.ctx, atom=self.atom, knobs={**self.knobs, **warp_knobs(self.atom, w)})
            for w in warp_offers(self.ctx.skel, self.atom, self.ctx.budget)
        ]


@dataclass(frozen=True)
class _MmTensorize(Fork):
    """Root matmul choice: the scalar subtree + one warp subtree per eligible
    atom (the ``Tensorize`` decision that gates whether warp knobs exist)."""

    ctx: _Ctx
    scalar: Fork | TileOp | None
    atoms: tuple[Atom, ...]
    knobs: dict

    def expand(self) -> list:
        # Warp subtrees first: for an eligible (fp16/bf16) matmul the tensor-core
        # tier is the default pick (the cold prior can't yet rank the greenfield
        # knobs, so emission order decides — Phase 4 retrain takes over). The
        # scalar subtree stays as the fallback the search can still reach.
        opts: list = [_WarpChooseWarp(ctx=self.ctx, atom=atom, knobs={TC_ATOM.name: atom.name}) for atom in self.atoms]
        if self.scalar is not None:
            opts.append(self.scalar)
        return opts


def build_matmul_tree(skel: MatmulSkeleton, *, loop_op, context, graph, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Root of the generative matmul tree. The scalar subtree is always
    available; eligible tensor-core atoms add warp subtrees under a top-level
    ``Tensorize`` choice. Returns the scalar subtree directly when no atom is
    eligible, or ``None`` when even the scalar tier has nothing legal."""
    ctx = _Ctx(skel=skel, budget=Budget(), base_knobs=base_knobs, kernel_name=kernel_name)
    scalar = _scalar_subtree(ctx)
    # The warp (tensor-core) path handles one clean-tile contraction. A symbolic
    # free axis (masked ceil-div) or a multi-accumulator matmul (>1 same-K reduce
    # — gated MLP) stays on the scalar tier (the warp builder takes one reduce;
    # `is_atom_eligible` would still pass on the first, so gate explicitly).
    n_reduce = sum(1 for s in skel.inner_body if isinstance(s, Loop) and s.is_reduce)
    # The warp (tensor-core) path handles one clean-tile contraction. A symbolic /
    # non-divisible free axis stays on the SCALAR tier: the masked warp tile staged
    # via real_extent isn't yet a verified-correct substitute for the legacy masked
    # mma.sync (it faults at some runtime sizes / under a TMA pin), so masked free
    # axes route to scalar (correct, slower) until the masked-warp + TMA-descriptor
    # path is fully landed. A multi-accum matmul (gated MLP) also stays scalar.
    scalar_only = skel.inner_n.symbolic or skel.outer_m.symbolic or n_reduce > 1
    # Honor the legacy ``MMA`` pin (the aliased ``TC_ATOM`` knob): ``MMA=0`` forces
    # the scalar tier (e.g. the FK half2 window); ``MMA=<kind>`` restricts to that
    # atom; unset / truthy auto-enumerates every eligible atom.
    from deplodock.compiler.pipeline import knob as _knob  # noqa: PLC0415

    mma_enabled, mma_kind = _knob.mma_decode(TC_ATOM.raw())
    if scalar_only or not mma_enabled:
        atoms: list = []
    else:
        atoms = [a for a in eligible_atoms(loop_op, context, graph) if warp_offers(skel, a, ctx.budget)]
        if mma_kind is not None:
            atoms = [a for a in atoms if a.name == mma_kind]
    if not atoms:
        return scalar
    # A specific pinned atom (``MMA=<kind>``) forces the warp tier — drop the
    # scalar fallback so the prior can't rank it back to scalar.
    return _MmTensorize(ctx=ctx, scalar=(None if mma_kind is not None else scalar), atoms=tuple(atoms), knobs={})


# --- Cooperative-reduce (MONOID) tree: one reduce (bk, fk, br) decision. ---


@dataclass(frozen=True)
class _CoopLeaf(Fork):
    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [build_coop_reduce_tile(self.ctx.skel, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs)]


@dataclass(frozen=True)
class _CoopChooseReduce(Fork):
    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        return [_CoopLeaf(ctx=self.ctx, knobs=coop_reduce_knobs(r)) for r in coop_reduce_offers(self.ctx.skel)]


def build_coop_reduce_tree(skel: CoopReduceSkeleton, *, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Root of the cooperative-reduce tree (one ``(bk, fk, br)`` decision); a
    bare ``TileOp`` for a single legal variant, or ``None`` if none is legal."""
    offers = coop_reduce_offers(skel)
    if not offers:
        return None
    ctx = _Ctx(skel=skel, budget=Budget(), base_knobs=base_knobs, kernel_name=kernel_name)
    if len(offers) == 1:
        return build_coop_reduce_tile(skel, coop_reduce_knobs(offers[0]), kernel_name=kernel_name, base_knobs=base_knobs)
    return _CoopChooseReduce(ctx=ctx, knobs={})
