"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain. Stamps ``Role`` tags on body Loops;
``001_launch_geometry`` and other downstream rules read the tags and skip
their own decisions.

Output axes split as ``A → A_b·(T·R) + A_t·R + A_r`` (T = BN or BM, R = FN or
FM). K splits as ``K → K_s·(K_o·br·bk) + K_o·(br·bk) + K_i·br + K_c`` (K_s for
SPLITK > 1, K_c for cooperative-K BR > 1). Resulting nesting:

    K_s BLOCK (split-K) → M_b BLOCK → N_b BLOCK → K_c THREAD (coop-K) →
      M_t THREAD → N_t THREAD → M_r REGISTER → N_r REGISTER →
        prelude → K_o SERIAL_OUTER → K_i STAGE_INNER (reduce σ(body)) →
        (helper-driven cross-thread combine when K_c is cooperative) →
        post-K tower → Write

Example transformation (matmul A[M=64,K=32] @ B[K=32,N=64] with BN=BM=16,
FM=FN=1, BK=16, SPLITK=1, BR=1):

    Input LoopOp body:
        for m in 0..64:
            for n in 0..64:
                Init(acc)
                for k in 0..32 reduce:
                    a = load A[m, k]; b = load B[k, n]
                    Accum(acc, a*b)
                Write(C[m, n], acc)

    Output (σ_outer rewrites m, n; σ_k rewrites k; tower wraps):
        for m_b in 0..4 BLOCK:
            for n_b in 0..4 BLOCK:
                for m_t in 0..16 THREAD:
                    for n_t in 0..16 THREAD:
                        for m_r in 0..1 REGISTER:    # inlined by normalize_body
                            for n_r in 0..1 REGISTER:    # inlined
                                Init(acc)
                                for k_o in 0..2 SERIAL_OUTER:
                                    for k_i in 0..16 STAGE_INNER reduce:
                                        a = load A[m_b·16 + m_t, k_o·16 + k_i]
                                        b = load B[k_o·16 + k_i, n_b·16 + n_t]
                                        Accum(acc, a*b)
                                Write(C[m_b·16 + m_t, n_b·16 + n_t], acc)

For cooperative-K reduce (e.g. sum K=512 with BR=256, BK=2), K_c appears as
a THREAD axis above the BLOCK level and σ_k extends to
``k = k_o·512 + k_i·256 + k_c``; the materializer emits the cross-thread
combine after the reduce subtree based on the escape-analysis helper
(``ir/tile/escape_analysis.py``) reading ``ThreadTile.cooperative_axes``.

For the fused-prologue matmul (SDPA P@V — softmax max/sum/reciprocal sitting
as siblings of an inner output Loop that holds the actual matmul), the chain
walker extends through the inner Loop (``_classify_fused_prologue``), stashes
the sibling reduces + assigns as ``KernelShape.prologue``, and the body builder
places the σ-rewritten prologue inside the ``M_r`` REGISTER scope but outside
the ``N_r`` register tower. SPLITK clamps to 1 in this branch — the prologue
feeds the matmul, so each K_s CTA would consume a partial softmax stat.

Pointwise collapses to BM = FM = FN = 1; extent-1 sub-axes get inlined by
normalize_body. SPLITK + Write atomicity is derived at codegen time from
``escape_analysis.atomic_axes`` (Write index vs enclosing block axes).
Cooperative-K combine emission similarly happens in the materializer from
the helper's ``accum_cooperative_axes``.

Priority keys: matmul prefers high cells/thread (amortize K-loop overhead);
pointwise prefers low cells/thread (memory-bandwidth bound); cooperative
reduce prefers warp-sized-or-larger BR. All target ~256 threads/CTA.
"""

from __future__ import annotations

import enum
from collections.abc import Callable
from dataclasses import dataclass, replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Expr, Literal, SimplifyCtx, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Init, Load, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import (
    GridTile,
    RegisterTile,
    SerialTile,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.fork_tree import Level, build_fork_tree
from deplodock.compiler.pipeline.passes.lowering.tile._enumeration import (
    BK,
    BM,
    BN,
    BR,
    FM,
    FN,
    SPLITK,
    TileParams,
    enumerate_cartesian,
)
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce
from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import has_nonlinear_post_reduce_epilogue
from deplodock.compiler.pipeline.pipeline import Fork


class Role(enum.Enum):
    """Planner-internal label for ``_wrap_tower`` layers.

    Drives which tile-flavor the layer becomes when the planner builds the
    tower. Not part of the IR — never reaches downstream passes (which
    discriminate on tile-flavor type instead).
    """

    BLOCK = "block"
    THREAD = "thread"
    REGISTER = "register"
    STAGE_INNER = "stage_inner"
    SERIAL_OUTER = "serial_outer"
    PIPELINE = "pipeline"


PATTERN = [Pattern("root", LoopOp)]


@dataclass(frozen=True)
class KernelShape:
    """Per-LoopOp shape info that stays constant across every ``TileParams``
    variant of a single kernel: the output axis Loops (innermost-N, optional
    M, extra outer chain), the K reduce Loop (None for pointwise), and the
    set of axis names ``_replace_k_loops`` should rewrite (collected once
    upfront by ``_plan_kernel`` instead of re-classified per variant).
    """

    outer_n: Loop
    outer_m: Loop | None
    extra_outer: tuple[Loop, ...]
    k_loop: Loop | None
    target_names: frozenset[str]
    # Sibling stmts of ``outer_n`` at the inner level where the chain stopped.
    # Non-empty only for the fused-prologue matmul (softmax max/sum/reciprocal
    # → SDPA P@V); empty for plain matmul / pointwise / cooperative-reduce.
    prologue: tuple[Stmt, ...] = ()


@dataclass(frozen=True)
class _Plan:
    """Cheap-to-build planning state shared across every variant of one
    LoopOp. Holds the inputs ``_materialize`` needs to build any single
    TileOp on demand: the per-LoopOp ``shape``, the leading non-Loop body
    stmts, the LoopOp's carry-forward knobs, the rendered kernel name, and
    the enumerated ``TileParams`` sorted by score. The planner runs every
    classification + enumeration step once and stops short of body
    construction — ``_materialize(plan, params)`` runs the expensive
    ``_build_split_body`` + ``TileOp.__post_init__`` work for one
    variant only, called lazily from the chosen Fork leaf's ``expand``.
    """

    shape: KernelShape
    leading: tuple[Stmt, ...]
    base_knobs: dict
    kernel_name: str
    loop_op: LoopOp
    params: tuple[TileParams, ...]


def rewrite(ctx: Context, root: Node, match) -> Graph | None | TileOp | Fork | list[Fork]:
    """Emit a hierarchical Fork tree over knob bundles:
    ``BR → (BM,BN) → (FM,FN) → (BK,SPLITK) → TileOp leaf``.

    Single-variant kernels short-circuit to a bare ``TileOp`` (the engine
    applies it inline without a fork point). Single-value Fork levels are
    collapsed so a tree with N effective branching levels emits only N
    Fork wrappers — most matmul kernels have BR fixed at 1 so they
    actually emit a 3-level tree.

    Variant *materialization* (``_build_split_body`` + ``TileOp(...)``
    which runs the body-normalize pipeline) is deferred to the leaf
    Fork's ``expand`` thunk, so greedy compile only builds the one
    chosen variant per LoopOp. ``_plan_kernel`` runs the cheap up-front
    classification + enumeration and produces a ``_Plan`` with bare
    ``TileParams`` tuples; sibling sorting uses :meth:`TileOp.lazy_score`
    (mirrors :meth:`TileOp.score`) so the ordering matches what the
    fully-built TileOps would have produced."""
    loop_op: LoopOp = root.op
    # Name was stamped onto the LoopOp by ``loop/fusion/030_stamp_loop_names``
    # (the last loop-dialect pass), so we just forward it onto the TileOp.
    kernel_name = loop_op.name
    # Idempotence is structural: once the planner has built a TileOp, the
    # rule pattern (LoopOp) no longer matches.
    plan = _plan_kernel(loop_op, ctx, kernel_name=kernel_name)
    if plan is None:
        raise RuleSkipped("kernel shape not handled by planner (or already planned)")

    if len(plan.params) == 1:
        return _materialize(plan, plan.params[0])

    # Hierarchical Fork tree over (BR, (BM,BN), (FM,FN), (BK,SPLITK)). The
    # builder collapses single-key levels (e.g. pointwise's BR=1 layer),
    # sorts siblings by max-propagated score descending, and defers
    # ``_materialize`` to each leaf's ``expand`` thunk so greedy compile
    # only builds one variant per LoopOp. See ``pipeline/fork_tree.py``.
    return build_fork_tree(
        params=plan.params,
        levels=[
            Level((BR.name,), lambda p: (p.br,)),
            Level((BM.name, BN.name), lambda p: (p.bm, p.bn)),
            Level((FM.name, FN.name), lambda p: (p.fm, p.fn)),
            Level((BK.name, SPLITK.name), lambda p: (p.bk, p.splitk)),
        ],
        materialize=lambda p: _materialize(plan, p),
        score=lambda p: _score_variant(plan, p, ctx),
    )


def _score_variant(plan: _Plan, params: TileParams, ctx: Context) -> float:
    """Score one variant from the plan + params. Prefers
    :meth:`TileOp.lazy_score` (cheap, params/shape-only) and falls back to
    materializing + :meth:`TileOp.score` if a future TileOp drops its lazy
    implementation. Lets the planner rank siblings before paying the
    body-construction cost."""
    lazy = TileOp.lazy_score(ctx, shapes=plan.shape, params=params)
    if lazy is not None:
        return lazy
    return _materialize(plan, params).score(ctx)


def _split_leading_non_loops(body) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Split body into ``(leading non-Loop stmts, rest)``. Mirrors
    ``001_launch_geometry``'s prefix handling."""
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


def _outer_free_loop_chain(body) -> tuple[tuple[Loop, ...], tuple[Stmt, ...]]:
    """Walk the outer single-stmt chain of untagged free Loops, outermost-first.

    Returns ``(chain, prologue)``. ``prologue`` is empty for the plain
    matmul / pointwise / cooperative-reduce paths. The fused-prologue
    matmul (SDPA P@V — softmax max/sum/reciprocal sitting as siblings
    of the head_dim Loop) is detected via :func:`_classify_fused_prologue`:
    when the chain stops because the body has siblings split as
    ``[reduces..., assigns..., one non-reduce Loop containing a Write]``,
    pull the inner Loop into the chain (so ``outer_n`` becomes the buried
    output-N axis) and surface the siblings as ``prologue``."""
    _, rest = _split_leading_non_loops(body)
    out: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce:
        out.append(cur[0])
        cur = tuple(cur[0].body)

    prologue, inner_loop = _classify_fused_prologue(cur)
    if inner_loop is not None:
        out.append(inner_loop)
        return tuple(out), prologue
    return tuple(out), ()


def _classify_fused_prologue(stmts: tuple[Stmt, ...]) -> tuple[tuple[Stmt, ...], Loop | None]:
    """Match the fused-prologue matmul shape: ``[reduce Loops..., leaf
    stmts..., exactly one non-reduce Loop whose body transitively contains
    a matmul-shape reduce]``. Returns ``(prologue, inner_loop)`` on
    match, ``((), None)`` otherwise.

    The matmul-reduce requirement on the inner Loop is what keeps this
    helper from over-matching non-matmul kernels with sibling shapes:
    RMSNorm / softmax / mean follow ``[reduce..., assigns, Loop(post-
    pointwise)]`` too, but their post-pointwise Loop has no matmul-shape
    reduce — those fall through to the cooperative-K path the planner
    already handles.

    Bail conditions: a second non-reduce Loop at this level, an
    unexpected wrapper (``StridedLoop``), or a naked Write."""
    prologue: list[Stmt] = []
    inner: Loop | None = None
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce:
            prologue.append(s)
        elif isinstance(s, Loop) and not s.is_reduce and inner is None and _contains_matmul_reduce(s):
            inner = s
        elif isinstance(s, (Loop, StridedLoop)):
            return (), None
        elif isinstance(s, Write):
            return (), None
        else:
            prologue.append(s)
    return (tuple(prologue), inner) if inner is not None else ((), None)


def _contains_matmul_reduce(stmt: Stmt) -> bool:
    """True if ``stmt`` is or transitively contains a matmul-shape reduce
    Loop (``is_matmul_reduce``: reduce body has ≥ 2 K-indexed Loads + an
    Accum)."""
    if isinstance(stmt, Loop) and stmt.is_reduce and is_matmul_reduce(stmt):
        return True
    for sub in stmt.nested():
        for c in sub:
            if _contains_matmul_reduce(c):
                return True
    return False


def _identity_rename(name: str) -> str:
    return name


def _wrap_tower(layers: list[tuple[Axis, Role | None]], inner: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Wrap ``inner`` in nested typed tile flavors, innermost layer first.

    ``layers`` is innermost-first: ``[(K_i, STAGE_INNER), (K_o, SERIAL_OUTER)]``
    walks outer ``K_o`` outermost. Consecutive parallel-binding axes group
    into one tile (so ``[BLOCK, BLOCK, THREAD, THREAD, REGISTER]`` yields
    ``GridTile(BLOCK,BLOCK) > ThreadTile(THREAD,THREAD) > RegisterTile(REGISTER)``).
    Each serial-binding axis becomes its own ``SerialTile`` with the
    matching ``kind``.

    Role → flavor mapping:

    - ``BLOCK`` → ``GridTile.axes``. Split-K vs. regular output-partition
      is derived at codegen time from ``escape_analysis.atomic_axes``.
    - ``THREAD`` → ``ThreadTile.axes``. Cooperative-K cooperativity is
      recovered at materialize time from ``Accum.axes ∩ ThreadTile.axes``
      (see ``escape_analysis``).
    - ``REGISTER`` → ``RegisterTile.axes``.
    - ``SERIAL_OUTER`` / ``STAGE_INNER`` / ``PIPELINE`` → ``SerialTile(kind=…)``.
    - Untagged (``None``) → ``SerialTile(kind="plain")``.

    **Size-1 axis filtering.** ``Loop`` IR's ``drop_size_one_free_axes``
    inlines extent-1 free Loops by substituting their var with 0. The
    planner's σ-split sometimes generates such axes (e.g. cooperative
    softmax with BN=BM=1 makes N_t / M_t extent-1 THREAD axes). Mirror
    the same drop here — except for ``BLOCK`` axes,
    which signal launch geometry to the CUDA backend and must survive
    even at extent 1 (single-CTA cooperative kernels rely on this).
    """
    inner_body = tuple(inner)
    # Drop size-1 axes that aren't launch-geometry markers; substitute
    # ``Var(axis.name) → Literal(0, "int")`` in the inner body.
    filtered: list[tuple[Axis, Role | None]] = []
    for axis, role in layers:
        if axis.extent.is_static and axis.extent.as_static() == 1 and role is not Role.BLOCK:
            sub = Sigma({axis.name: Literal(0, "int")})
            inner_body = tuple(c.rewrite(_identity_rename, sub) for c in inner_body)
            continue
        filtered.append((axis, role))

    # Walk outermost-first so consecutive parallel axes group naturally.
    outermost_first = list(reversed(filtered))
    # Group: list of (group_kind, [axes], [roles])
    groups: list[tuple[str, list[Axis], list[Role | None]]] = []
    for axis, role in outermost_first:
        kind = _layer_kind_for(role)
        # Parallel kinds group consecutive same-kind axes; serial kinds always
        # start a fresh group (each serial layer is its own SerialTile).
        if groups and groups[-1][0] == kind and kind in ("grid", "thread", "register"):
            groups[-1][1].append(axis)
            groups[-1][2].append(role)
        else:
            groups.append((kind, [axis], [role]))

    # Build the tree innermost-first by wrapping ``inner`` with each group
    # in reverse order.
    current: tuple[Stmt, ...] = inner_body
    for kind, axes, roles in reversed(groups):
        if kind == "grid":
            # Split-K block axes (K_s) need no tag — codegen derives
            # atomic-add from ``escape_analysis.atomic_axes`` (block axis
            # missing from Write.index).
            current = (GridTile(axes=tuple(axes), body=Body(current)),)
        elif kind == "thread":
            # Cooperative-K thread axes (K_c) get into ``Accum.axes``
            # via the planner's σ-split; the materializer recovers
            # cooperativity from ``Accum.axes ∩ ThreadTile.axes``.
            current = (ThreadTile(axes=tuple(axes), body=Body(current)),)
        elif kind == "register":
            current = (RegisterTile(axes=tuple(axes), body=Body(current)),)
        else:  # serial — one axis per layer
            ax = axes[0]
            role = roles[0]
            serial_kind: str = "plain"
            if role is Role.SERIAL_OUTER:
                serial_kind = "serial_outer"
            elif role is Role.STAGE_INNER:
                serial_kind = "stage_inner"
            elif role is Role.PIPELINE:
                serial_kind = "pipeline"
            current = (SerialTile(axis=ax, body=Body(current), kind=serial_kind),)
    return current


def _layer_kind_for(role: Role | None) -> str:
    if role is Role.BLOCK:
        return "grid"
    if role is Role.THREAD:
        return "thread"
    if role is Role.REGISTER:
        return "register"
    return "serial"


class _BuildSkipped(Exception):
    """Raised by ``_build_split_body`` when the body's shape doesn't match."""


def _classify_n_dep(body: Body, n_axis_name: str) -> tuple[list[Stmt], list[Stmt]]:
    """Partition ``body`` stmts into (N-invariant, N-dependent) by transitive
    free-var analysis over Exprs + SSA def-use. ``n_axis_name`` is the post-σ
    N register axis (typically ``N_r.name``) — a stmt is N-dependent iff its
    transitive Expr ``free_vars`` ∪ ``deps``-chain references ``n_axis_name``.
    Order within each group is preserved (source order = SSA topo order, so
    N-invariant stmts feeding N-dependent ones come first).

    Used by :func:`_build_register_blocked_body` to split a matmul K-reduce
    body into the N-invariant cone (RMSNorm prologue, M-axis Loads — shared
    across F_N cells) and the N-dependent tail (weight Load + ``Accum`` —
    replicated per cell).
    """

    def fn(s: Stmt, child_T: tuple[frozenset[str] | None, ...], bound: frozenset[str]) -> frozenset[str]:
        own: frozenset[str] = frozenset()
        for e in s.exprs():
            own = own | frozenset(v for v in e.free_vars() if v not in bound)
        for c in child_T:
            if c is not None:
                own = own | c
        return own

    deps = body.fold(fn)
    invariant: list[Stmt] = []
    dependent: list[Stmt] = []
    for s in body:
        if n_axis_name in deps[id(s)]:
            dependent.append(s)
        else:
            invariant.append(s)
    return invariant, dependent


def _split_k_tower_for_block(
    stmt: Stmt,
    n_axis_name: str,
    n_r: Axis,
    leaf_cond: Callable[[tuple[Stmt, ...]], tuple[Stmt, ...]] | None,
) -> tuple[Stmt, int]:
    """Walk into a SerialTile K-tower; at the innermost ``stage_inner`` layer
    (K_i, the reduce), split its body into ``[N-invariant cone...,
    RegisterTile(N_r, [N-dep tail])]``. ``leaf_cond`` (when set) wraps the
    N-dep tail in a per-cell Cond — used for masked N tiles where each cell's
    work must be guarded. Returns ``(new_stmt, n_replaced)``.
    """
    if isinstance(stmt, SerialTile) and stmt.kind == "stage_inner":
        invariant, dependent = _classify_n_dep(stmt.body, n_axis_name)
        if not dependent:
            return stmt, 0
        tail_body = tuple(dependent)
        if leaf_cond is not None:
            tail_body = leaf_cond(tail_body)
        new_body = Body(tuple(invariant) + (RegisterTile(axes=(n_r,), body=Body(tail_body)),))
        return replace(stmt, body=new_body), 1
    if isinstance(stmt, SerialTile):
        new_body_stmts: list[Stmt] = []
        replaced = 0
        for c in stmt.body:
            new_c, r = _split_k_tower_for_block(c, n_axis_name, n_r, leaf_cond)
            new_body_stmts.append(new_c)
            replaced += r
        if replaced:
            return replace(stmt, body=Body(new_body_stmts)), replaced
    return stmt, 0


def _build_register_blocked_body(
    new_inner: tuple[Stmt, ...],
    n_axis_name: str,
    n_r: Axis,
    leaf_cond: Callable[[tuple[Stmt, ...]], tuple[Stmt, ...]] | None,
) -> tuple[Stmt, ...] | None:
    """Apply the register-blocked GEMM nest to ``new_inner`` (the σ-rewritten
    matmul body sitting inside ``M_r`` REGISTER, before the outer N_r wrap).
    Replaces the existing per-cell ``[Init, K-tower, ..., Write]`` flat
    structure with sibling RegisterTile(N_r) towers separated by the
    (now shared) K-loop:

    - ``Init(acc)`` → ``RegisterTile(N_r, [Init(acc)])`` (per-cell accumulators).
    - K-tower (SerialTile K_o > SerialTile K_i) → K_i body split into N-invariant
      cone + ``RegisterTile(N_r, [N-dep tail])`` — the cone (e.g. fused RMSNorm
      prologue, M-axis Loads) runs once per K iteration, the tail runs F_N×.
    - ``Write(C, acc)`` → ``RegisterTile(N_r, [Write(C, acc)])`` (per-cell writes).
    - Adjacent ``Load`` / ``Assign`` / ``Write`` siblings (matmul_add's
      ``[Load(r), Assign(v=acc+r), Write(v)]`` residual epilogue) are
      grouped into ONE ``RegisterTile(N_r)`` so the replicator emits a
      contiguous per-cell ``[Load r_i, Assign v_i, Write _i]`` block —
      same code shape as the per-cell legacy path. Grouping keeps the
      epilogue's intermediate SSA values (``r_i``, ``v_i``) within one
      per-cell scope; wrapping each stmt in its own RegisterTile would
      still be correct, but the replicator would emit a wide register
      window with every ``r_i`` live until its ``Assign v_i`` consumer.

    ``leaf_cond`` wraps each tower's leaf body in a per-cell Cond (the masked-N
    boundary guard). The replicator (010_split_register_axes) replicates each
    Cond per cell using σ N_r → i, producing F_N concrete-i guarded leaves.

    Returns ``None`` when the shape doesn't fit (no K_i found, empty N-dep
    tail, or unexpected outer-scope stmts — vectorized Writes, unknown
    block stmts, etc.) so the caller falls back to per-cell.
    """
    out: list[Stmt] = []
    n_replaced = 0
    pending: list[Stmt] = []

    def flush_pending() -> None:
        """Emit any accumulated Load / Assign chain as one epilogue tower.

        Conservatively requires the chain to terminate in a Write — a
        dangling Load/Assign group with no consumer Write is the
        unrecognised-stmt path and signals the per-cell fallback."""
        if not pending:
            return
        leaf_body: tuple[Stmt, ...] = tuple(pending)
        if leaf_cond is not None:
            leaf_body = leaf_cond(leaf_body)
        out.append(RegisterTile(axes=(n_r,), body=Body(leaf_body)))
        pending.clear()

    for s in new_inner:
        if isinstance(s, Init):
            # Init must be unconditional: it declares the per-cell accumulator
            # variable at thread scope. Wrapping it in the masked-N Cond would
            # scope ``acc_i`` to the if-block; the K-tower's Accum and the
            # Write tower (both Cond-wrapped per cell) would reference an
            # ``acc_i`` that's no longer in scope. OOB cells initialize a
            # dead register, which is harmless.
            flush_pending()
            out.append(RegisterTile(axes=(n_r,), body=Body((s,))))
        elif isinstance(s, (Load, Assign)):
            # Part of an epilogue / prologue chain (matmul_add residual:
            # ``Load(r)``, ``Assign(v=acc+r)``). Defer until the closing
            # ``Write`` so the entire chain lands in one RegisterTile.
            pending.append(s)
        elif isinstance(s, Write):
            pending.append(s)
            flush_pending()
        elif isinstance(s, SerialTile):
            flush_pending()
            new_s, r = _split_k_tower_for_block(s, n_axis_name, n_r, leaf_cond)
            out.append(new_s)
            n_replaced += r
        elif isinstance(s, Cond):
            # Post-K epilogue Cond — matmul_add's K_s==0 gate, or other
            # N-uniform predicate wrapping a per-cell Write/epilogue. Wrap
            # the whole Cond in RegisterTile(N_r); the replicator descends
            # into Cond.body / Cond.else_body and replicates the inner Loads
            # / Assigns / Writes per cell (Cond's own predicate is K_s-only
            # so it doesn't itself get replicated, just its content).
            flush_pending()
            out.append(RegisterTile(axes=(n_r,), body=Body((s,))))
        else:
            # Unrecognised outer-scope stmt. Fall back to the per-cell legacy
            # path so the kernel still lowers (the planner-fallback branch in
            # ``_build_split_body`` picks up).
            return None
    if pending:
        # Trailing Load / Assign without a closing Write — likely a partial
        # epilogue we don't yet recognise. Bail to per-cell.
        return None
    if n_replaced == 0:
        return None
    return tuple(out)


def _plan_kernel(loop_op: LoopOp, ctx: Context, *, kernel_name: str = "") -> _Plan | None:
    """Unified σ-split planning for matmul, pointwise, and cooperative-reduce
    kernels. Returns a :class:`_Plan` whose ``params`` enumerates every
    candidate ``TileParams`` but doesn't materialize any TileOp — the
    expensive ``_build_split_body`` + ``TileOp.__post_init__`` work is
    deferred to :func:`_materialize`, invoked from the chosen Fork leaf's
    ``expand`` thunk in :func:`_build_fork_tree_lazy`.

    Detection is predicate-driven: ``is_matmul_reduce`` (≥ 2 K-indexed Loads +
    Accum) picks the matmul knob set; any other reduce with extent ≥ warp_size
    picks the cooperative-K set; no qualifying reduce falls to pointwise.

    **Phantom outer axis for chain-less kernels.** Body shapes without an
    outer free-Loop chain — e.g. global reductions ``[Loop(k, reduce),
    Write(o[0])]`` — would otherwise have no axis to lift into
    ``GridTile`` / ``ThreadTile``. We synthesize a size-1 ``__phantom__``
    axis and wrap the LoopOp body in one outer free Loop so the rest of
    the partitioner runs unchanged. ``_wrap_tower``'s size-1-axis filter
    drops the phantom before it reaches the IR.
    """
    chain, prologue = _outer_free_loop_chain(loop_op.body)
    if not chain:
        # Synthesize a single size-1 outer Loop so the rest of the
        # partitioner has an axis to lift. We can't actually wrap it in a
        # new LoopOp — ``LoopOp.__post_init__`` runs ``drop_size_one_free_axes``
        # which would inline the phantom back out. Build the phantom Loop
        # as a chain entry directly; downstream consumers (``shape.outer_n``,
        # ``_split_leading_non_loops``) only need the Loop object and its
        # body, not a containing LoopOp.
        phantom_axis = Axis("__phantom__", 1)
        phantom = Loop(axis=phantom_axis, body=Body(loop_op.body))
        chain = (phantom,)
        prologue = ()

    outer_n: Loop = chain[-1]
    outer_m: Loop | None = chain[-2] if len(chain) >= 2 else None
    extra_outer: tuple[Loop, ...] = chain[:-2] if outer_m is not None else chain[:-1]
    # Symbolic free axes (Dim("seq_len") etc.) use their hint as the expected
    # size and always emit a MASKED (overhang) tile: the enumerator picks tile
    # sizes for the hint, ``_build_split_body`` ceil-divs the symbolic extent
    # for the grid, and a boundary Cond gates lanes past the runtime value.
    # ``hint`` is set at trace from ``--seq-len`` (assumed present on any
    # symbolic axis the planner reaches).
    n_symbolic = not outer_n.axis.extent.is_static
    m_symbolic = outer_m is not None and not outer_m.axis.extent.is_static
    E_N = outer_n.axis.extent.hint if n_symbolic else outer_n.axis.extent.as_static()
    E_M = (outer_m.axis.extent.hint if m_symbolic else outer_m.axis.extent.as_static()) if outer_m is not None else 1

    # Single walk: classify body + collect every axis name _replace_k_loops
    # should rewrite. ``target_names`` survives σ_outer (only axis NAMES are
    # used downstream, not Loop identity — names don't change under σ).
    all_loops: tuple[Loop, ...] = outer_n.body.iter_of_type(Loop)
    matmul_reduces = [lp for lp in all_loops if lp.is_reduce and is_matmul_reduce(lp)]
    nonmatmul_reduces = [lp for lp in all_loops if lp.is_reduce and not is_matmul_reduce(lp)]

    k_loop: Loop | None
    target_names: frozenset[str]
    if matmul_reduces:
        if outer_m is None:
            return None
        k_loop = matmul_reduces[0]
        # Symbolic M / N is allowed: planner forces BM/BN/FM/FN=1 so each output
        # element runs on its own CTA with a serial K loop. Symbolic K also runs
        # via the same path (BK=SPLITK=BR=1 — whole K stays as one serial
        # iteration inside the per-output-element CTA). Both are inefficient
        # but correct; perf follow-up uses strided cooperative threads.
        k_symbolic = not k_loop.axis.extent.is_static
        E_K = 1 if k_symbolic else k_loop.axis.extent.as_static()
        # target_names unions over outer_n.body (the matmul K reduces) and
        # the prologue (softmax max/sum reduces sharing the matmul K extent,
        # axis-name-unified by unify_sibling_reduce_axes upstream). For
        # plain matmul / matmul_add / gated_mlp the prologue is empty and
        # this collapses to ``{lp.axis.name for lp in matmul_reduces}``.
        prologue_reduces = tuple(lp for lp in Body(prologue).iter_of_type(Loop) if lp.is_reduce and lp.axis.extent == k_loop.axis.extent)
        target_names = frozenset((*(lp.axis.name for lp in matmul_reduces), *(lp.axis.name for lp in prologue_reduces)))
        # SPLITK > 1 only works when each Write's atomic-add is mathematically
        # equivalent to the unsplit reduce. SPLITK is forced off for two
        # patterns:
        #
        # 1. Non-linear post-reduce combines like ``silu(acc_g) * acc_u``
        #    (gated_mlp) or softmax (sdpa). Conservative proxy: any matmul-
        #    reduce loop with > 1 Accum fuses multiple K-sums into one
        #    output cell — must be non-linear (else fusion would have
        #    merged them upstream).
        # 2. A non-empty prologue (SDPA P@V) — softmax max/sum/reciprocal
        #    are *consumed* by the matmul reduce, so each K_s CTA's
        #    partial softmax stat would feed a partial matmul.
        #
        # A post-reduce *linear* epilogue (``matmul_add``: ``Load(r)`` +
        # ``Assign(v = acc + r)`` → ``Write(v)``) IS allowed at SPLITK > 1.
        # ``_build_split_body`` rewrites the body so the residual add is
        # hoisted into a ``Cond(K_s == 0)`` — the K_s == 0 CTA atomic-adds
        # ``acc + r`` while every other K_s CTA atomic-adds just ``acc``,
        # so ``sum_i acc_i + r = c·sum_k a_k + r``. Non-linear epilogues
        # (e.g. ``v = acc * r``) fall through here and force splitk=1.
        multi_accum = any(sum(1 for s in lp.body if isinstance(s, Accum)) > 1 for lp in matmul_reduces)
        has_nonlinear_epilogue = has_nonlinear_post_reduce_epilogue(outer_n.body)
        force_splitk_one = multi_accum or bool(prologue) or has_nonlinear_epilogue
        # A fused prologue (SDPA P@V: softmax max/sum) carries a per-M-row
        # reduction whose accumulators must reset per register cell. Masking a
        # symbolic M/N axis admits FM/FN > 1, register-tiling the row and
        # sharing one accumulator across cells — wrong. So a prologue matmul
        # keeps symbolic axes degenerate (E=1, no mask): correct via the
        # symbolic grid, one output element per thread.
        mask_ok = not prologue
        param_combos = enumerate_cartesian(
            E_M=E_M if (mask_ok or not m_symbolic) else 1,
            E_N=E_N if (mask_ok or not n_symbolic) else 1,
            E_K=E_K,
            ctx=ctx,
            priority_mode="matmul",
            force_splitk_one=force_splitk_one,
            m_axis_name=outer_m.axis.name if outer_m is not None else None,
            n_axis_name=outer_n.axis.name,
            m_forced_mask=m_symbolic and mask_ok,
            n_forced_mask=n_symbolic and mask_ok,
        )
    elif nonmatmul_reduces and nonmatmul_reduces[0].axis.extent.is_static and nonmatmul_reduces[0].axis.extent.as_static() >= ctx.warp_size:
        # Cooperative-K: BR>1 requires the sole THREAD axis (materializer's
        # _single_thread_var) — bn/bm_choices prepend 1 to enable BN=BM=1.
        # E_K ≥ warp_size: smaller reduces don't justify a warp-shuffle.
        # target_names includes both K-reduce axes AND per-K post-pointwise
        # axes (non-reduce free Loops sharing E_K), since both get rewritten.
        #
        # SPLITK is restricted to 1 here: cross-CTA reduce for cooperative-K
        # would need atomic accumulation of the partial sums (the per-CTA
        # Combine only reduces *within* a CTA), plus a barrier before the
        # post-reduce pointwise epilogue reads the final value. Neither is
        # wired up today — the K_s=0 CTA would race with K_s>0 CTAs that
        # are still writing partial sums, and only K_s=0 writes the output
        # using its own (half-data) reduction. Forcing SPLITK=1 keeps the
        # search space honest.
        k_loop = nonmatmul_reduces[0]
        E_K = k_loop.axis.extent.as_static()
        target_names = frozenset(lp.axis.name for lp in all_loops if lp.axis.extent.as_static() == E_K and not is_matmul_reduce(lp))
        # Cooperative reduce binds the free axis whole-to-grid (BR>1 forces
        # BN=BM=1, so the grid covers seq exactly — no overhang). A masked
        # register-tile (FN>1) would wrap the reduce body in the boundary Cond
        # and hide it from the cooperative-reduce + smem-staging passes,
        # breaking the cross-thread combine. So a symbolic free axis stays
        # degenerate here (E=1, no forced mask): correct at any seq_len via the
        # symbolic grid, just not register-tiled.
        param_combos = enumerate_cartesian(
            E_M=1 if m_symbolic else E_M,
            E_N=1 if n_symbolic else E_N,
            E_K=E_K,
            ctx=ctx,
            priority_mode="reduce",
        )
    else:
        # Pointwise — no qualifying reduce.
        k_loop = None
        target_names = frozenset()
        param_combos = enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=1,
            ctx=ctx,
            priority_mode="pointwise",
            m_axis_name=outer_m.axis.name if outer_m is not None else None,
            n_axis_name=outer_n.axis.name,
            m_forced_mask=m_symbolic,
            n_forced_mask=n_symbolic,
        )

    shape = KernelShape(
        outer_n=outer_n,
        outer_m=outer_m,
        extra_outer=extra_outer,
        k_loop=k_loop,
        target_names=target_names,
        prologue=prologue,
    )
    leading, _ = _split_leading_non_loops(loop_op.body)
    if not param_combos:
        return None
    return _Plan(
        shape=shape,
        leading=leading,
        base_knobs=dict(loop_op.knobs),
        kernel_name=kernel_name,
        loop_op=loop_op,
        params=tuple(param_combos),
    )


def _materialize(plan: _Plan, params: TileParams) -> TileOp:
    """Build one ``TileOp`` for a single ``TileParams`` against the
    planner's pre-computed shape. The expensive bits — ``_build_split_body``
    and ``TileOp.__post_init__`` (which runs ``normalize_body`` over the
    fresh body) — happen here, lazily, from the chosen Fork leaf's
    ``expand`` thunk.

    ``_BuildSkipped`` is shape-determined (raised only when
    ``_replace_k_loops`` can't find any K-axis Loops in the body, which
    depends on ``shape.target_names`` matching the body's axis names, not
    on params). If it ever fires here, the shape was misclassified at
    plan time — the assertion surfaces the bug instead of silently
    dropping a leaf.
    """
    try:
        chain_body = _build_split_body(plan.shape, params)
    except _BuildSkipped as exc:
        raise AssertionError(f"shape-level _BuildSkipped fired at materialize time for kernel {plan.kernel_name!r}: {exc}") from exc
    knobs = {
        **plan.base_knobs,
        BN.name: params.bn,
        BM.name: params.bm,
        FM.name: params.fm,
        FN.name: params.fn,
        BK.name: params.bk,
        SPLITK.name: params.splitk,
        BR.name: params.br,
    }
    # Drop leading stmts whose SSA name is *also* defined inside ``chain_body``.
    # A pre-loop invariant (e.g. ``v0 = reciprocal(1024)``) used only by a
    # post-reduce stmt gets pulled into the thread scope by the body builder as
    # that stmt's dependency; prepending it here too would put two defs of the
    # same name in nested scopes — invalid CUDA ("v0 already declared"). The
    # outer copy is dead (nothing at the outer scope uses it), so it's safe to
    # drop. Leading stmts genuinely used at the outer scope aren't re-defined
    # inside, so they're kept.
    inner_defs = {name for s in Body.coerce(chain_body).iter() for name in s.defines()}
    leading = tuple(s for s in plan.leading if not (set(s.defines()) & inner_defs))
    return TileOp(body=leading + chain_body, name=plan.kernel_name, knobs=knobs)


def _build_split_body(shape: KernelShape, params: TileParams) -> tuple[Stmt, ...]:
    """σ-split ``shape.outer_n``'s body and wrap in the output
    BLOCK/THREAD/REGISTER tower. ``shape.outer_m`` / ``shape.k_loop`` are
    None for 1D pointwise / non-reduce kernels.

    K_s and K_c (when present) are shared across all reduces in the kernel;
    K_o / K_i are per-K-Loop, built inside ``_replace_k_loops``. SPLITK
    atomic-Write is deferred to ``001_launch_geometry``.

    Fused-prologue matmul (``shape.prologue`` non-empty, SDPA P@V): the
    prologue runs inside the ``M_r`` REGISTER scope but *outside* the
    ``N_r`` register tower — softmax stats are per-seq-q-row (per M_r),
    independent of the output N. σ_outer + σ_k are applied uniformly so
    the prologue's M-axis references resolve to the in-scope ``M_r``."""
    sigma_map: dict[str, object] = {}

    # source_axis: every sub-axis points back to the original Axis it was
    # carved out of (= ``shape.outer_n.axis`` for N, etc.). Downstream
    # passes use this to group surrounding axes by source identity (e.g.
    # the MMA factorization plan's BLOCK·GROUP·CELL·ATOM enumeration along
    # each output axis) without name-suffix string matching.
    overhang = frozenset(params.overhang)
    N_axis = shape.outer_n.axis
    N_name = N_axis.name
    N_src = N_axis.source_axis or N_axis
    n_bnfn = params.bn * params.fn
    # ``n_bound`` is the masked-boundary Cond RHS (an Expr): ``Literal(E_N)`` for
    # a static overhang axis, the symbolic ``Var`` for a hint-driven one, None
    # when N isn't masked.
    n_bound: object | None = None
    if N_axis.extent.is_static:
        E_N = N_axis.extent.as_static()
        # Masked tiles (N in overhang): use ceil_div so the boundary CTA covers
        # the partial tile. ``real_extent`` on N_b tells the materializer how
        # to gate boundary lanes (``if decoded < real_extent``).
        if N_name in overhang:
            N_b = Axis(f"{N_name}_b", -(-E_N // n_bnfn), source_axis=N_src, real_extent=E_N)
            n_bound = Literal(E_N, "int")
        else:
            N_b = Axis(f"{N_name}_b", E_N // n_bnfn, source_axis=N_src)
    elif N_name in overhang:
        # Symbolic + hint-driven masked: ceil-div the SYMBOLIC extent so the grid
        # covers the partial last tile at any runtime size; the boundary Cond
        # gates lanes past the runtime value. ``Dim`` arithmetic builds the
        # composite ceil-div Expr; the launch resolver evals it from sym_values.
        N_b = Axis(f"{N_name}_b", (N_axis.extent + (n_bnfn - 1)) // n_bnfn, source_axis=N_src)
        n_bound = N_axis.extent.expr
    else:
        # Symbolic, no hint (degenerate): bn=fn=1 by construction, bind the
        # whole axis to GridTile; the size-1 normalizer drops N_t / N_r.
        N_b = Axis(f"{N_name}_b", N_axis.extent, source_axis=N_src)
    N_t = Axis(f"{N_name}_t", params.bn, source_axis=N_src)
    N_r = Axis(f"{N_name}_r", params.fn, source_axis=N_src)
    # N register-tile decode — two layouts:
    #
    #   blocked      ``N = N_b·(BN·FN) + N_t·FN + N_r``  (thread-major)
    #   interleaved  ``N = N_b·(BN·FN) + N_r·BN + N_t``  (thread-minor)
    #
    # blocked keeps a thread's FN cells in contiguous columns. On a clean-
    # divisor tile the FN cells share one K-loop, so those contiguous
    # per-thread loads vectorize and the warp fully uses each cache line —
    # blocked wins, and it's also what staged-smem reads want (conflict-free
    # after permute_lane_accesses). A MASKED tile (N not a multiple of BN·FN
    # — e.g. the lm_head vocab=151669 projection) instead wraps each cell in
    # its own ``if (col < N)`` guard, splitting the FN cells into separate
    # K-loops: no cross-cell vectorization, and consecutive threads land
    # FN·elem bytes apart → fully uncoalesced global weight loads (~25×
    # wasted bandwidth; 94 ms → 3.5 ms on Qwen3 lm_head). Interleaved strides
    # the cell by BN so consecutive threads map to consecutive columns — the
    # warp reads BN contiguous columns per step (coalesced) despite the
    # per-cell guards. The Stage-fill (if any) and the register read share
    # this σ, so the column order stays self-consistent either way.
    n_block = Var(N_b.name) * Literal(params.bn * params.fn, "int")
    if N_name in overhang:
        sigma_map[N_name] = n_block + Var(N_r.name) * Literal(params.bn, "int") + Var(N_t.name)
    else:
        sigma_map[N_name] = n_block + Var(N_t.name) * Literal(params.fn, "int") + Var(N_r.name)

    M_b = M_t = M_r = None
    m_bound: object | None = None
    if shape.outer_m is not None:
        M_axis = shape.outer_m.axis
        M_name = M_axis.name
        M_src = M_axis.source_axis or M_axis
        m_bnfm = params.bm * params.fm
        if M_axis.extent.is_static:
            E_M = M_axis.extent.as_static()
            if M_name in overhang:
                M_b = Axis(f"{M_name}_b", -(-E_M // m_bnfm), source_axis=M_src, real_extent=E_M)
                m_bound = Literal(E_M, "int")
            else:
                M_b = Axis(f"{M_name}_b", E_M // m_bnfm, source_axis=M_src)
        elif M_name in overhang:
            M_b = Axis(f"{M_name}_b", (M_axis.extent + (m_bnfm - 1)) // m_bnfm, source_axis=M_src)
            m_bound = M_axis.extent.expr
        else:
            M_b = Axis(f"{M_name}_b", M_axis.extent, source_axis=M_src)
        M_t = Axis(f"{M_name}_t", params.bm, source_axis=M_src)
        M_r = Axis(f"{M_name}_r", params.fm, source_axis=M_src)
        sigma_map[M_name] = (
            Var(M_b.name) * Literal(params.bm * params.fm, "int") + Var(M_t.name) * Literal(params.fm, "int") + Var(M_r.name)
        )

    sigma_outer = Sigma(sigma_map)

    # K axes: K_s / K_c are kernel-wide (single SPLITK / single cooperative
    # thread direction); K_o / K_i are per-K-Loop, built inside _replace_k_loops.
    K_s = K_c = None
    K_o_ext: object = 0
    K_src: Axis | None = None
    if shape.k_loop is not None:
        K_axis = shape.k_loop.axis
        K_name = K_axis.name
        K_src = K_axis.source_axis or K_axis
        if K_axis.extent.is_static:
            E_K = K_axis.extent.as_static()
            K_o_ext = E_K // (params.splitk * params.br * params.bk)
        else:
            # Symbolic K (params.bk=splitk=br=1 by planner construction): whole
            # axis stays as one serial K_o iteration. ``Axis.__post_init__``
            # coerces the ``Dim`` straight back into the K_o axis.
            K_o_ext = K_axis.extent
        K_s = Axis(f"{K_name}_s", params.splitk, source_axis=K_src) if params.splitk > 1 else None
        K_c = Axis(f"{K_name}_c", params.br, source_axis=K_src) if params.br > 1 else None

    # σ-rewrite outer_n's body (M/N axes), then replace every K-iter Loop with
    # a K_o · K_i tower. Both paths use shared canonical K_o / K_i names so
    # 020_stage_inputs row-cache can merge structurally-equivalent Loads.
    inner_after_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in shape.outer_n.body)

    if shape.k_loop is not None:
        new_inner, n_replaced = _replace_k_loops(
            inner_after_outer,
            target_names=shape.target_names,
            K_canonical_name=shape.k_loop.axis.name,
            K_s=K_s,
            K_c=K_c,
            br=params.br,
            bk=params.bk,
            K_o_ext=K_o_ext,
        )
        if n_replaced == 0:
            raise _BuildSkipped("K reduce not found in body")
        # SPLITK > 1 with a linear residual epilogue (matmul_add) is gated
        # post-planner by ``015_gate_splitk_residual``, which finds the
        # K_s axis in the wrapped GridTile and hoists the linear epilogue
        # under ``Cond(K_s == 0)`` so the residual is added exactly once
        # across the K_s CTAs. We only have to keep ``force_splitk_one``
        # at enumeration time so non-linear epilogues don't even reach
        # the gate pass.
    else:
        new_inner = inner_after_outer

    # Register-blocked GEMM nest (default for matmul shapes the blocked
    # builder accepts): split the N_r register-tile around the K-reduce so
    # N-invariant compute runs once per K step and is shared across F_N
    # cells. Emits sibling RegisterTile(N_r) towers (Init / K-reduce N-dep
    # tail / Write) inside the M_r scope. Prologue kernels are in-scope
    # too: the fused RMSNorm + lm_head matmul (linear_196) has K-dependent
    # N-invariant compute (``v_k = x[m, k]·v4·norm_weight[k]``) inside the
    # matmul K-loop — exactly the blocked-nest target. The prologue's own
    # K-reduce (mean / softmax stats) still runs once at the M_r scope; the
    # blocked transform only touches the matmul body's K_i.
    # The blocked builder declines (returns None) on shapes it can't yet
    # handle — BR > 1's cooperative-K combine, M-mask — the per-cell legacy
    # path below picks up those cases.
    #
    # FN == 1 is no longer disqualified: the blocked builder degenerates
    # to three sibling ``RegisterTile(N_r=1)`` towers; the replicator
    # (``lowering/kernel/010_split_register_axes``) unwraps each (one cell,
    # σ ``N_r → 0``), so the end-state IR is structurally identical to
    # the legacy per-cell layout. ``SerialTileBase.is_reduce`` walks
    # through the per-tower ``RegisterTile`` wrappers so the staging
    # passes see K_i as a reduce; ``025_unify_sibling_stages`` then drops
    # any redundant matmul Stage that would re-stage a buffer already
    # staged in a prior sibling scope (the fused-RMSNorm + linear case).
    #
    # SPLITK > 1 is no longer disqualified: the blocked builder now groups
    # the matmul_add residual epilogue (Load(r) + Assign(v=acc+r) + Write)
    # into one ``RegisterTile(N_r)`` Write tower; ``015_gate_splitk_residual``
    # detects this shape and emits the ``Cond(K_s == 0)`` wrap inside the
    # Write tower body. The 020 staging walk (Phase 2) sees the K-tower's
    # nested Accum via ``is_reduce`` and the residual ``Load(r)`` via
    # ``_iter_loads_through_register`` so neither input goes unstaged.
    reg_blocked = False
    if params.br == 1 and shape.k_loop is not None and m_bound is None:
        if n_bound is not None:
            # Masked N: wrap each tower's leaf body in its own Cond. The
            # replicator (010_split_register_axes) replicates the Cond per
            # cell using σ N_r → i, so each replica gets its own σ-folded
            # predicate. A single top-level Cond wrapping all three towers
            # would reference N_r in its predicate but expose nested
            # RegisterTile(N_r) inside — the replicator can't reconcile
            # both.
            n_pred_for_cond = sigma_outer.reduce(Var(N_name), SimplifyCtx({}))

            def _wrap_in_n_cond(stmts: tuple[Stmt, ...], pred: Expr = n_pred_for_cond, bound: Expr = n_bound) -> tuple[Stmt, ...]:  # type: ignore[assignment]
                return (Cond(cond=BinaryExpr("<", pred, bound), body=Body(stmts)),)

            leaf_cond = _wrap_in_n_cond
        else:
            leaf_cond = None
        blocked_inner = _build_register_blocked_body(new_inner, N_r.name, N_r, leaf_cond)
        if blocked_inner is not None:
            new_inner = blocked_inner
            reg_blocked = True

    # Masked tiles: when ceil-div has rounded a block-axis extent up past the
    # axis bound, wrap the σ-rewritten body in ``Cond(decoded_axis_coord <
    # bound)``. ``bound`` is the static ``real_extent`` (lm_head vocab) or the
    # symbolic ``Var`` (hint-driven dynamic axis — resolved to the runtime
    # value at launch). The Cond sits INSIDE the register tower (N_r is in
    # scope for the predicate). The replicator in ``010_split_register_axes``
    # handles Conds whose predicate depends on the replicated axis by
    # σ-substituting per replica so each replicated body gets a partly-
    # constant-folded predicate (NVRTC drops always-true copies).
    #
    # Under REG_BLOCK the N-cond was already applied per-tower above, so skip
    # the top-level N-cond here. The M-cond still always wraps (M_r is outside
    # the N_r towers, regardless of reg_block).
    if n_bound is not None and not reg_blocked:
        n_pred = sigma_outer.reduce(Var(N_name), SimplifyCtx({}))
        new_inner = (Cond(cond=BinaryExpr("<", n_pred, n_bound), body=Body(new_inner)),)
    if m_bound is not None:
        m_pred = sigma_outer.reduce(Var(M_name), SimplifyCtx({}))
        new_inner = (Cond(cond=BinaryExpr("<", m_pred, m_bound), body=Body(new_inner)),)

    # σ-rewrite + K-replace the prologue when present. The prologue sits
    # inside M_r (so M_r is in scope for σ_outer's M-axis mapping) and
    # outside N_r (softmax stats are N-invariant).
    prologue_rewritten: tuple[Stmt, ...] = ()
    if shape.prologue:
        prologue_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in shape.prologue)
        if shape.k_loop is not None:
            prologue_rewritten, _ = _replace_k_loops(
                prologue_outer,
                target_names=shape.target_names,
                K_canonical_name=shape.k_loop.axis.name,
                K_s=K_s,
                K_c=K_c,
                br=params.br,
                bk=params.bk,
                K_o_ext=K_o_ext,
            )
        else:
            prologue_rewritten = prologue_outer

    # Wrap tower, innermost first. extent-1 layers (e.g. M_t / N_t under v1
    # cooperative BN=BM=1) are inlined later by normalize_body.
    #
    # Without prologue: REGISTER layers (N_r, M_r) join the same tower as
    # THREAD/BLOCK, wrapped innermost-first by _wrap_tower.
    #
    # With prologue: N_r wraps the matmul body alone; the prologue stmts
    # are prepended at the M_r scope, so the M_r Loop body becomes
    # ``[prologue..., N_r tower]``. Outer layers (THREAD/BLOCK/SPLITK)
    # then wrap that combined body. When the blocked nest has already
    # emitted the per-tower RegisterTile(N_r) wrappers inside new_inner,
    # skip the outer ``_wrap_tower([(N_r, ...)], ...)`` — the N_r tile is
    # already present per-tower.
    if shape.prologue:
        if reg_blocked:
            matmul_tower = new_inner
        else:
            matmul_tower = _wrap_tower([(N_r, Role.REGISTER)], new_inner)
        body_inside_mr = prologue_rewritten + matmul_tower
        if M_r is not None:
            # M_r stays serial (not RegisterTile) — the SDPA prologue
            # (softmax max/sum/reciprocal) computes once per output row
            # and must NOT be replicated per register cell. Only N_r is
            # the register-tile axis here.
            body_after_register = (SerialTile(axis=M_r, body=Body(body_inside_mr), kind="plain"),)
        else:
            body_after_register = body_inside_mr
        outer_layers: list[tuple[Axis, Role | None]] = [(N_t, Role.THREAD)]
        if M_t is not None:
            outer_layers.append((M_t, Role.THREAD))
        if K_c is not None:
            outer_layers.append((K_c, Role.THREAD))  # cooperative-K stride
        outer_layers.append((N_b, Role.BLOCK))
        if M_b is not None:
            outer_layers.append((M_b, Role.BLOCK))
        if K_s is not None:
            outer_layers.append((K_s, Role.BLOCK))  # split-K
        outer_layers.extend((lp.axis, Role.BLOCK) for lp in reversed(shape.extra_outer))
        return _wrap_tower(outer_layers, body_after_register)

    # When REG_BLOCK has wrapped the three sibling RegisterTile(N_r) towers
    # already, the outer layers list skips N_r — the towers each carry their
    # own RegisterTile(N_r) wrapper inside the M_r/THREAD scope.
    layers: list[tuple[Axis, Role | None]] = []
    if not reg_blocked:
        layers.append((N_r, Role.REGISTER))
    if M_r is not None:
        layers.append((M_r, Role.REGISTER))
    layers.append((N_t, Role.THREAD))
    if M_t is not None:
        layers.append((M_t, Role.THREAD))
    if K_c is not None:
        layers.append((K_c, Role.THREAD))  # cooperative-K stride
    layers.append((N_b, Role.BLOCK))
    if M_b is not None:
        layers.append((M_b, Role.BLOCK))
    if K_s is not None:
        layers.append((K_s, Role.BLOCK))  # split-K
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(shape.extra_outer))
    return _wrap_tower(layers, new_inner)


def _replace_k_loops(
    stmts: tuple[Stmt, ...],
    *,
    target_names: frozenset[str],
    K_canonical_name: str,
    K_s: Axis | None,
    K_c: Axis | None,
    br: int,
    bk: int,
    K_o_ext: int,
) -> tuple[tuple[Stmt, ...], int]:
    """Replace every ``Loop`` whose axis name is in ``target_names`` with a
    ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, σ(body)))`` tower.
    Returns ``(new_stmts, n_replaced)``.

    ``target_names`` is built once in ``_split_kernel_fully``: the set of
    K-iteration axes that should be rewritten (matmul-shape reduces, or for
    cooperative-K both the reduces AND the per-K post-pointwise loops sharing
    the same K extent). ``Loop.is_reduce`` is derived from body Accum
    presence, so K_i inherits the right status automatically.

    Non-reduce match + SPLITK > 1: wrap the K_o tower in ``Cond(K_s == 0)`` —
    every K_s CTA would otherwise re-execute the post. Reduce-K skips the
    Cond; launch_geometry's atomic-Write rewrite handles cross-CTA SPLITK."""
    out: list[Stmt] = []
    n_replaced = 0
    for s in stmts:
        if isinstance(s, Loop) and s.axis.name in target_names:
            K_name = s.axis.name
            K_src = s.axis.source_axis or s.axis
            K_o = Axis(f"{K_canonical_name}_o", K_o_ext, source_axis=K_src)
            K_i = Axis(f"{K_canonical_name}_i", bk, source_axis=K_src)
            sigma_k = _build_k_sigma(K_name, K_s, K_o, K_c, K_i, K_o_ext, br, bk)
            new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
            tower = _wrap_tower([(K_i, Role.STAGE_INNER), (K_o, Role.SERIAL_OUTER)], new_body)
            if not s.is_reduce and K_s is not None:
                out.append(
                    Cond(
                        cond=BinaryExpr("==", Var(K_s.name), Literal(0, "int")),
                        body=tower,
                        else_body=(),
                    )
                )
            else:
                out.extend(tower)
            n_replaced += 1
            continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, r = _replace_k_loops(
                s.body, target_names=target_names, K_canonical_name=K_canonical_name, K_s=K_s, K_c=K_c, br=br, bk=bk, K_o_ext=K_o_ext
            )
            if r:
                out.append(replace(s, body=inner))
                n_replaced += r
                continue
        if isinstance(s, Cond):
            inner_b, rb = _replace_k_loops(
                s.body, target_names=target_names, K_canonical_name=K_canonical_name, K_s=K_s, K_c=K_c, br=br, bk=bk, K_o_ext=K_o_ext
            )
            inner_e, re_ = _replace_k_loops(
                s.else_body,
                target_names=target_names,
                K_canonical_name=K_canonical_name,
                K_s=K_s,
                K_c=K_c,
                br=br,
                bk=bk,
                K_o_ext=K_o_ext,
            )
            if rb or re_:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                n_replaced += rb + re_
                continue
        out.append(s)
    return tuple(out), n_replaced


def _build_k_sigma(
    K_name: str,
    K_s: Axis | None,
    K_o: Axis,
    K_c: Axis | None,
    K_i: Axis,
    K_o_ext: int,
    br: int,
    bk: int,
) -> Sigma:
    """σ for ``K = K_s·(K_o_ext·br·bk) + K_o·(br·bk) + K_i·br + K_c``.
    K_s / K_c terms collapse when those axes are None (SPLITK=1 / BR=1);
    when K_c is absent, K_i loses its ``·br`` stride."""
    inner_expr = Var(K_o.name) * Literal(br * bk, "int")
    if K_c is not None:
        inner_expr = inner_expr + Var(K_i.name) * Literal(br, "int") + Var(K_c.name)
    else:
        inner_expr = inner_expr + Var(K_i.name)
    if K_s is not None:
        inner_expr = Var(K_s.name) * Literal(K_o_ext * br * bk, "int") + inner_expr
    return Sigma({K_name: inner_expr})
