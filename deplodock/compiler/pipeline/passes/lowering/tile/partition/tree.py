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

from dataclasses import dataclass, field

from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.stmt import Loop, Write
from deplodock.compiler.ir.tile.ir import Atom, TileOp
from deplodock.compiler.pipeline.fork import Fork
from deplodock.compiler.pipeline.passes.lowering.tile.partition.budget import Budget
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import AxisRole, IterDag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.knobs import RED_FK, TC_ATOM, WARP_M, WARP_N
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import (
    build_coop_reduce_tile,
    build_flash_tile,
    build_matmul_tile,
    build_pointwise_tile,
    build_warp_matmul_tile,
)
from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import (
    coop_free_thread_knobs,
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


@dataclass(frozen=True)
class _Ctx:
    """Per-LoopOp state shared by every node of one kernel's tree. ``dag`` is the
    iteration-DAG view the builders read free axes / inner body / K-info off;
    ``target_names`` are the K-extent loop names a reduce transform rewrites
    (regime-derived in :func:`build_partition`), ``k_bounds`` the symbolic flash
    streaming bounds."""

    dag: IterDag
    budget: Budget
    base_knobs: dict
    kernel_name: str
    target_names: frozenset[str] = frozenset()
    k_bounds: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _Leaf(Fork):
    """A complete pointwise move stack (thread + register tiles); ``expand``
    materializes the ``TileOp``."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [build_pointwise_tile(self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs, dag=self.ctx.dag)]


@dataclass(frozen=True)
class _FlashLeaf(Fork):
    """A complete flash move stack; ``expand`` materializes via ``build_flash_tile``
    (serial-transforms the streaming reduces; the carrier owns the recurrence)."""

    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [
            build_flash_tile(
                self.knobs,
                kernel_name=self.ctx.kernel_name,
                base_knobs=self.ctx.base_knobs,
                dag=self.ctx.dag,
                target_names=self.ctx.target_names,
                k_bounds=self.ctx.k_bounds,
            )
        ]


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
        reg1 = reg_knobs(self.ctx.dag, (1, 1))
        return [
            _FlashLeaf(ctx=self.ctx, knobs={**thread_knobs(self.ctx.dag, t), **reg1}) for t in thread_offers(self.ctx.dag, self.ctx.budget)
        ]


def build_pointwise_tree(*, dag: IterDag, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Return the root ``Fork`` of the generative tree, a bare ``TileOp`` when
    only one variant is legal (mirrors the legacy single-variant short-circuit),
    or ``None`` when nothing is legal (the dispatcher falls through to legacy)."""
    budget = Budget()
    threads = thread_offers(dag, budget)
    regs = reg_offers(dag, budget)
    if not threads or not regs:
        return None
    ctx = _Ctx(dag=dag, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name)
    if len(threads) == 1 and len(regs) == 1:
        full = {**thread_knobs(dag, threads[0]), **reg_knobs(dag, regs[0])}
        return build_pointwise_tile(full, kernel_name=kernel_name, base_knobs=base_knobs, dag=dag)
    return _ChooseThread(ctx=ctx, knobs={})


def build_flash_tree(
    *, dag: IterDag, target_names: frozenset[str], k_bounds: dict, base_knobs: dict, kernel_name: str
) -> Fork | TileOp | None:
    """Flash nest: the same free-axis (thread → register) generative tree as
    pointwise, but each leaf materializes via `build_flash_tile` (which serial-
    transforms the streaming reduces). The reduces aren't a search dimension —
    the `FlashCombine` carrier owns the KV recurrence."""
    budget = Budget()
    threads = thread_offers(dag, budget)
    if not threads:
        return None
    # FM=FN=1 only (the streaming carrier can't span register cells), so the
    # register tile is not a search dimension — just the thread tile.
    ctx = _Ctx(dag=dag, budget=budget, base_knobs=base_knobs, kernel_name=kernel_name, target_names=target_names, k_bounds=k_bounds)
    reg1 = reg_knobs(dag, (1, 1))
    if len(threads) == 1:
        full = {**thread_knobs(dag, threads[0]), **reg1}
        return build_flash_tile(full, kernel_name=kernel_name, base_knobs=base_knobs, dag=dag, target_names=target_names, k_bounds=k_bounds)
    return _ChooseThreadFlash(ctx=ctx, knobs={})


# --- Matmul (SEMIRING) generative tree: reduce → thread → register. ---


@dataclass(frozen=True)
class _MmLeaf(Fork):
    ctx: _Ctx
    knobs: dict
    is_leaf = True

    def expand(self) -> list:
        return [
            build_matmul_tile(
                self.knobs,
                kernel_name=self.ctx.kernel_name,
                base_knobs=self.ctx.base_knobs,
                dag=self.ctx.dag,
                target_names=self.ctx.target_names,
            )
        ]


@dataclass(frozen=True)
class _MmThreadChosen(Fork):
    """Reduce + thread tiles pinned; ``expand`` offers register tiles (the cell
    budget depends on the pinned ``fk`` strip-mine)."""

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
    """The scalar-tier matmul subtree (reduce → thread → register), or a bare
    ``TileOp`` for a single variant, or ``None`` if nothing is legal."""
    dag = ctx.dag
    reduces = matmul_reduce_offers(dag)
    threads = matmul_thread_offers(dag, ctx.budget)
    if not reduces or not threads:
        return None
    regs0 = matmul_reg_offers(dag, ctx.budget, reduces[0][1])
    if len(reduces) == 1 and len(threads) == 1 and len(regs0) == 1:
        full = {**reduce_knobs(reduces[0]), **thread_knobs(dag, threads[0]), **reg_knobs(dag, regs0[0])}
        return build_matmul_tile(full, kernel_name=ctx.kernel_name, base_knobs=ctx.base_knobs, dag=dag, target_names=ctx.target_names)
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
            build_warp_matmul_tile(
                self.atom, self.knobs, kernel_name=self.ctx.kernel_name, base_knobs=self.ctx.base_knobs, dag=self.ctx.dag
            )
        ]


@dataclass(frozen=True)
class _WarpChooseBk(Fork):
    ctx: _Ctx
    atom: Atom
    knobs: dict

    def expand(self) -> list:
        return [
            _WarpLeaf(ctx=self.ctx, atom=self.atom, knobs={**self.knobs, **warp_bk_knobs(bk)})
            for bk in warp_bk_offers(self.ctx.dag, self.atom)
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
            for reg in warp_reg_offers(self.ctx.dag, self.atom, warp)
        ]


@dataclass(frozen=True)
class _WarpChooseWarp(Fork):
    ctx: _Ctx
    atom: Atom
    knobs: dict

    def expand(self) -> list:
        return [
            _WarpChooseReg(ctx=self.ctx, atom=self.atom, knobs={**self.knobs, **warp_knobs(self.atom, w)})
            for w in warp_offers(self.ctx.dag, self.atom, self.ctx.budget)
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


def build_matmul_tree(
    *, dag: IterDag, target_names: frozenset[str], loop_op, context, graph, base_knobs: dict, kernel_name: str
) -> Fork | TileOp | None:
    """Root of the generative matmul tree. The scalar subtree is always
    available; eligible tensor-core atoms add warp subtrees under a top-level
    ``Tensorize`` choice. Returns the scalar subtree directly when no atom is
    eligible, or ``None`` when even the scalar tier has nothing legal."""
    ctx = _Ctx(dag=dag, budget=Budget(), base_knobs=base_knobs, kernel_name=kernel_name, target_names=target_names)
    scalar = _scalar_subtree(ctx)
    n_reduce = sum(1 for s in dag.inner_body if isinstance(s, Loop) and s.is_reduce)
    # A symbolic **M** (outer, row) **or N** (inner, contiguous) axis reaches the
    # warp tier as a masked mma.sync tile (`_warp_axis` ceil-divides + emits the
    # boundary `Cond`; `020`/`021` clamp the staged slab fill, `005` stamps the
    # per-cell `RegStore` row / col guard, and `RegStore.render` falls back to a
    # per-element scalar store when `ldm` is symbolic-odd so the `__half2` site
    # stays aligned). Verified-correct across runtime sizes (accuracy at off-tile
    # 130 / 200). A multi-accum matmul (gated MLP) stays scalar (the warp builder
    # takes one reduce; `is_atom_eligible` would still pass on the first, so gate
    # explicitly).
    scalar_only = n_reduce > 1
    # Honor the legacy ``MMA`` pin (the aliased ``TC_ATOM`` knob): ``MMA=0`` forces
    # the scalar tier (e.g. the FK half2 window); ``MMA=<kind>`` restricts to that
    # atom; unset / truthy auto-enumerates every eligible atom.
    from deplodock.compiler.pipeline import knob as _knob  # noqa: PLC0415

    mma_enabled, mma_kind = _knob.mma_decode(TC_ATOM.raw())
    if scalar_only or not mma_enabled:
        atoms: list = []
    else:
        atoms = [a for a in eligible_atoms(loop_op, context, graph) if warp_offers(dag, a, ctx.budget)]
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
        return [
            build_coop_reduce_tile(
                self.knobs,
                kernel_name=self.ctx.kernel_name,
                base_knobs=self.ctx.base_knobs,
                dag=self.ctx.dag,
                target_names=self.ctx.target_names,
            )
        ]


@dataclass(frozen=True)
class _CoopChooseReduce(Fork):
    ctx: _Ctx
    knobs: dict

    def expand(self) -> list:
        # ``self.knobs`` carries the free-axis THREAD tiles (BN/BM); each reduce
        # leaf adds its (bk, fk, br).
        return [_CoopLeaf(ctx=self.ctx, knobs={**self.knobs, **coop_reduce_knobs(r)}) for r in coop_reduce_offers(self.ctx.dag)]


def build_coop_reduce_tree(*, dag: IterDag, target_names: frozenset[str], base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """Root of the cooperative-reduce tree (one ``(bk, fk, br)`` decision); a
    bare ``TileOp`` for a single legal variant, or ``None`` if none is legal. The
    free-axis THREAD tiles (``BN``/``BM``, the strided-cooperative form) are pinned
    once up front and ride every leaf."""
    offers = coop_reduce_offers(dag)
    if not offers:
        return None
    free_knobs = coop_free_thread_knobs(dag)
    ctx = _Ctx(dag=dag, budget=Budget(), base_knobs=base_knobs, kernel_name=kernel_name, target_names=target_names)
    if len(offers) == 1:
        full = {**free_knobs, **coop_reduce_knobs(offers[0])}
        return build_coop_reduce_tile(full, kernel_name=kernel_name, base_knobs=base_knobs, dag=dag, target_names=target_names)
    return _CoopChooseReduce(ctx=ctx, knobs=free_knobs)


@dataclass(frozen=True)
class _Regime:
    """The transient classification handoff (not a typed skeleton): the regime tag
    plus the K-info derived off the DAG — ``target_names`` (the K-extent loop names
    a reduce transform rewrites) and the flash streaming ``k_bounds``."""

    kind: str  # "pointwise" | "matmul" | "coop" | "flash"
    target_names: frozenset[str] = frozenset()
    k_bounds: dict = field(default_factory=dict)


def classify(dag: IterDag) -> _Regime | None:
    """Tag the nest's regime off the derived DAG (no typed-skeleton layer), or
    ``None`` for a shape the composer leaves to the legacy planner. The regime is
    the reduce axes' ``Loop.algebra_kind`` (``MAP`` → pointwise, ``SEMIRING`` →
    matmul, ``MONOID`` → cooperative reduce, ``TWISTED_MONOID`` → flash if it nests
    a reduce, else a cooperative monoid). This is the recognition predicate
    ``005_split_demoted`` consults to decide whether a fused cone composes."""
    if not dag.parallel:
        return None
    reduce_loops = [n.loop for n in dag.reduce]
    algebras = dag.algebras
    inner_body = dag.inner_body
    # Flash = a TWISTED_MONOID stream wrapping a NESTED reduce (the QK^T SEMIRING
    # inside the KV stream). A non-nested single monoid stream is a cooperative
    # MONOID reduce (softmax LSE), handled on the coop path below.
    nested_reduce = any(n.parent is not None and n.parent.role is AxisRole.REDUCE for n in dag.reduce)
    coop_monoid = AlgebraKind.TWISTED_MONOID in algebras and not nested_reduce

    if AlgebraKind.TWISTED_MONOID in algebras and nested_reduce:
        # Fused flash: a symbolic ``seq_len`` lands on the free q-rows axis (masked
        # tile) AND the streaming KV reduce (masked stream — ``k_bounds``); the
        # nested QK^T reduce (head_dim) must stay static.
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
        # Matmul. A MAP epilogue (QK^T scale, matmul_add) rides the output tile;
        # require static K, an M + N free pair, all body loops to BE reduces over
        # the one K extent (reject a map-loop prologue), and a Write present.
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
        # Cooperative reduce. The reduce(s) + any second-pass map loop iterate the
        # SAME K axis. ``coop_monoid`` (a non-nested softmax LSE Monoid) is static-K
        # only (the masked-K fill isn't wired for the tuple-state carrier).
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


def build_partition(*, dag: IterDag, loop_op, context, graph, base_knobs: dict, kernel_name: str) -> Fork | TileOp | None:
    """The single partition entry — one ``build_partition(dag)`` over the derived
    iteration-DAG view (no typed-skeleton layer). :func:`classify` tags the regime
    + derives the K-info off the DAG; every regime offers the same ``legal_decomps``
    family filtered by carrier traits. Returns a ``Fork`` / ``TileOp`` for a covered
    regime, or ``None`` to fall through. See
    ``plans/algebra-licensed-decomposition-moves.md`` (phase 6)."""
    r = classify(dag)
    if r is None:
        return None
    if r.kind == "pointwise":
        return build_pointwise_tree(dag=dag, base_knobs=base_knobs, kernel_name=kernel_name)
    if r.kind == "matmul":
        return build_matmul_tree(
            dag=dag,
            target_names=r.target_names,
            loop_op=loop_op,
            context=context,
            graph=graph,
            base_knobs=base_knobs,
            kernel_name=kernel_name,
        )
    if r.kind == "coop":
        return build_coop_reduce_tree(dag=dag, target_names=r.target_names, base_knobs=base_knobs, kernel_name=kernel_name)
    return build_flash_tree(dag=dag, target_names=r.target_names, k_bounds=r.k_bounds, base_knobs=base_knobs, kernel_name=kernel_name)
