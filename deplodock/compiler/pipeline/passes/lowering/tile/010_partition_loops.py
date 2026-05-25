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
import os
from collections.abc import Callable
from dataclasses import dataclass, replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Cond, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.ir.tile.ir import (
    GridTile,
    RegisterTile,
    SerialTile,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce
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

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2, 1)
_TUNE_AXIS_CHOICES: tuple[int, ...] = (1, 16, 32, 64, 128, 256)
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)
# Cooperative-K thread count. v1: BR > 1 requires BN = BM = 1 (single THREAD
# axis for materializer's _single_thread_var).
_BR_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Cap on per-thread cell-product. NVRTC compile time explodes past this.
_MAX_CELLS_PER_THREAD: int = 128

BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (matmul output N tile)")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul output M tile)")
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul N (output) axis")
BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")
SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")
BR = Knob("BR", KnobType.INT, hints=_BR_CANDIDATES, help="Cooperative-K thread count (1 = pure serial chunked reduce)")

_PLANNER_KNOBS: tuple[Knob, ...] = (BN, BM, FM, FN, BK, SPLITK, BR)


def _planner_pin_set() -> bool:
    """True if any planner knob has its ``DEPLODOCK_<NAME>`` env pin set.
    Used by ``_enumerate_cartesian`` to gate the peer-kernel fallback."""
    return any(os.environ.get(k.env) is not None for k in _PLANNER_KNOBS)


@dataclass(frozen=True)
class TileParams:
    """One ``(BN, BM, FM, FN, BK, SPLITK, BR)`` variant. Frozen for de-dup in
    the cartesian's ``seen`` set; ``br=1`` default keeps matmul / pointwise
    sites terse."""

    bn: int
    bm: int
    fm: int
    fn: int
    bk: int
    splitk: int
    br: int = 1


@dataclass(frozen=True)
class KernelShape:
    """Per-LoopOp shape info that stays constant across every ``TileParams``
    variant of a single kernel: the output axis Loops (innermost-N, optional
    M, extra outer chain), the K reduce Loop (None for pointwise), and the
    set of axis names ``_replace_k_loops`` should rewrite (collected once
    upfront by ``_split_kernel_fully`` instead of re-classified per variant).
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


def rewrite(ctx: Context, root: Node) -> Graph | None | TileOp | Fork | list[Fork]:
    """Emit a hierarchical Fork tree over knob bundles:
    ``BR → (BM,BN) → (FM,FN) → (BK,SPLITK) → TileOp leaf``.

    Single-variant kernels short-circuit to a bare ``TileOp`` (the engine
    applies it inline without a fork point). Single-value Fork levels are
    collapsed so a tree with N effective branching levels emits only N
    Fork wrappers — most matmul kernels have BR fixed at 1 so they
    actually emit a 3-level tree.

    Each branch Fork's ``score`` is the max ``TileOp.score(ctx)`` of any
    leaf reachable under it — the planner-side equivalent of the max-Q
    propagation MCTS does on measured rewards. Siblings at each level
    are sorted by score descending so the highest-scoring branch is
    option-0 (the greedy primary, the +∞-tiebreak winner)."""
    loop_op: LoopOp = root.op
    kernel_name = _kernel_name_for(loop_op, root.id)
    # Idempotence is structural: once the planner has built a TileOp, the
    # rule pattern (LoopOp) no longer matches.
    variants = _split_kernel_fully(loop_op, ctx, kernel_name=kernel_name)
    if variants is None:
        raise RuleSkipped("kernel shape not handled by planner (or already planned)")

    if len(variants) == 1:
        return variants[0]

    return _build_fork_tree(variants, ctx)


def _build_fork_tree(variants: list[TileOp], ctx: Context) -> Fork | list[Fork]:
    """Convert a flat ``list[TileOp]`` into the hierarchical Fork tree.

    Grouping order (outermost first): BR → (BM,BN) → (FM,FN) → (BK,SPLITK).
    A level is skipped (Fork wrapper omitted) when every variant in the
    enclosing group shares the same knob values at that level, so the
    common case (all variants have BR=1) doesn't add a useless 1-child
    Fork layer.

    **Sibling ordering** is by max-propagated ``TileOp.score`` descending:
    option-0 at each level is the branch containing the best-scoring
    reachable leaf. ``TileOp.score`` is the single source of truth for
    both greedy primary selection AND the MCTS prior — keep them aligned
    so a high-score variant the search will exploit later is also the
    one greedy picks today.

    Returns a single ``Fork`` when the top-level group collapses to one
    branch (engine still routes through the fork-spawn path since
    ``isinstance(option, Fork)``), otherwise the ``list[Fork]`` of
    siblings.
    """
    leaf_score: dict[int, float] = {id(v): float(v.score(ctx)) for v in variants}

    def _sorted(forks: list[Fork]) -> list[Fork]:
        return sorted(forks, key=lambda f: -f.score)

    def _leaf_forks(group: list[TileOp]) -> list[Fork]:
        out = [
            Fork(
                knobs={BK.name: v.knobs[BK.name], SPLITK.name: v.knobs[SPLITK.name]},
                expand=(lambda op=v: [op]),
                score=leaf_score[id(v)],
                is_leaf=True,
            )
            for v in group
        ]
        return _sorted(out)

    def _group_level(
        group: list[TileOp],
        knob_names: tuple[str, ...],
        child_builder,
    ) -> list[Fork]:
        """Build sibling Forks for one level, keyed by ``knob_names``,
        sorted by max-propagated score descending."""
        keyed: dict[tuple, list[TileOp]] = {}
        for v in group:
            key = tuple(v.knobs[n] for n in knob_names)
            keyed.setdefault(key, []).append(v)
        if len(keyed) == 1:
            return child_builder(next(iter(keyed.values())))
        siblings: list[Fork] = []
        for key, subgroup in keyed.items():
            children = child_builder(subgroup)
            if not children:
                continue
            siblings.append(
                Fork(
                    knobs=dict(zip(knob_names, key, strict=True)),
                    expand=(lambda c=children: list(c)),
                    score=max(c.score for c in children),
                    is_leaf=False,
                )
            )
        return _sorted(siblings)

    def _fmfn_level(group: list[TileOp]) -> list[Fork]:
        return _group_level(group, (FM.name, FN.name), _leaf_forks)

    def _bmbn_level(group: list[TileOp]) -> list[Fork]:
        return _group_level(group, (BM.name, BN.name), _fmfn_level)

    def _br_level(group: list[TileOp]) -> list[Fork]:
        return _group_level(group, (BR.name,), _bmbn_level)

    top = _br_level(variants)
    if len(top) == 1:
        return top[0]
    return top


def _kernel_name_for(loop: LoopOp, base_name: str) -> str:
    """Build the rendered kernel-function name from the LoopOp shape and
    the graph node id. ``k_<dedup_base>_<reduce|pointwise>``."""
    suffix = "reduce" if any(isinstance(s, Accum) for s in loop) else "pointwise"
    return f"k_{_dedup_tokens(base_name)}_{suffix}"


def _dedup_tokens(name: str) -> str:
    """Drop consecutive duplicate ``_``-separated tokens.

    ``softmax_softmax_max`` → ``softmax_max``; ``rms_rms_norm`` → ``rms_norm``.
    Preserves order; only collapses adjacent duplicates so structurally
    distinct repeats (``add_mul_add``) survive.
    """
    out: list[str] = []
    for tok in name.split("_"):
        if not tok or (out and out[-1] == tok):
            continue
        out.append(tok)
    return "_".join(out) if out else name


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


def _divisors_up_to(n: int, cap: int) -> tuple[int, ...]:
    """Divisors of ``n`` ≤ ``cap``, ascending. FM / FN candidate set — a
    divisor of ``E / bm_c`` automatically satisfies the divisibility check."""
    if n < 1 or cap < 1:
        return ()
    return tuple(d for d in range(1, min(n, cap) + 1) if n % d == 0)


class _BuildSkipped(Exception):
    """Raised by ``_build_split_body`` when the body's shape doesn't match."""


def _stmt_contains_accum(stmt: Stmt) -> bool:
    """True if ``stmt`` is or transitively contains an ``Accum``."""
    if isinstance(stmt, Accum):
        return True
    for body in stmt.nested():
        for s in body:
            if _stmt_contains_accum(s):
                return True
    return False


def _find_first_accum_name(stmts: tuple[Stmt, ...]) -> str | None:
    """Return the first Accum's ``name`` found anywhere in ``stmts``,
    or None if no Accum exists. Used by the SPLITK matmul-add lowering
    to spell the else-branch ``Write(out, acc_name)``."""
    for s in stmts:
        if isinstance(s, Accum):
            return s.name
        for body in s.nested():
            n = _find_first_accum_name(tuple(body))
            if n is not None:
                return n
    return None


def _is_linear_in_accum(value_name: str, acc_name: str, assigns_by_name: dict[str, Assign]) -> bool:
    """True iff ``value_name`` is computed from ``acc_name`` via a chain of
    ``add`` Assigns (any number of external Load arguments allowed).

    Recurses through the SSA chain of Assigns that produces ``value_name``.
    At each ``Assign``, requires ``op == "add"`` and that at least one arg
    transitively traces back to ``acc_name``. Bare Var args that are not
    in ``assigns_by_name`` and not ``acc_name`` are treated as external
    loads (residuals) — allowed under linearity. A non-``add`` Assign or
    a chain that never reaches ``acc_name`` returns False."""
    if value_name == acc_name:
        return True
    a = assigns_by_name.get(value_name)
    if a is None:
        return False
    if a.op.name != "add":
        return False
    return any(arg == acc_name or _is_linear_in_accum(arg, acc_name, assigns_by_name) for arg in a.args)


def _has_nonlinear_post_reduce_epilogue(stmts: tuple[Stmt, ...]) -> bool:
    """True iff ``stmts`` has any post-reduce epilogue that is NOT a linear
    add chain over the Accum.

    Used by ``_split_kernel_fully`` to decide ``force_splitk_one`` for
    matmul-shape kernels: linear epilogues (``matmul_add``) are handled by
    ``_gate_linear_epilogue_on_k_s_zero`` in ``_build_split_body``;
    non-linear ones (e.g. ``v = acc * r``) must run at SPLITK=1."""
    epilogue = tuple(s for s in stmts if not _stmt_contains_accum(s))
    if not epilogue:
        return False
    writes = [s for s in epilogue if isinstance(s, Write)]
    if len(writes) != 1:
        return True
    write = writes[0]
    if write.is_vector:
        return True
    acc_name = _find_first_accum_name(stmts)
    if acc_name is None:
        return True
    assigns_by_name = {s.name: s for s in epilogue if isinstance(s, Assign)}
    return not _is_linear_in_accum(write.value, acc_name, assigns_by_name)


def _gate_linear_epilogue_on_k_s_zero(stmts: tuple[Stmt, ...], k_s_name: str) -> tuple[Stmt, ...]:
    """Rewrite ``stmts`` so the linear post-reduce epilogue runs only on
    the ``K_s == 0`` CTA, with the other CTAs atomic-adding the bare
    Accum.

    Input shape (matmul_add after ``_replace_k_loops``)::

        [Load(r), <reduce tower>, Assign(v=acc+r), Write(out, v)]

    Output shape::

        [<reduce tower>,
         Cond(K_s == 0,
              body=[Load(r), Assign(v=acc+r), Write(out, v)],
              else_body=[Write(out, acc)])]

    Both Writes lower to ``atomicAdd`` under SPLITK > 1, so the final
    output is ``sum_i acc_i + r`` — the residual added exactly once.
    Returns ``stmts`` unchanged when no linear epilogue is present.

    **Partition is positional, not set-based.** When the K-loop is fully
    unrolled (BK=1 + K_o_ext=1, i.e. ``_wrap_tower`` drops both K_o and
    K_i as size-1) the Loads / Assigns that *feed* the Accum end up as
    siblings of the Accum at this level. Those are reduce-body stmts,
    not epilogue — they must stay with the Accum, not get moved into
    the Cond (which would leave the Accum referencing values defined
    inside a scope it no longer dominates). Split at the position of
    the last Accum-bearing stmt: ``[:cut+1]`` is the reduce body that
    the Cond skips over; ``[cut+1:]`` is the true post-reduce epilogue.
    """
    last_accum_idx = -1
    for i, s in enumerate(stmts):
        if _stmt_contains_accum(s):
            last_accum_idx = i
    if last_accum_idx < 0:
        return stmts
    reduce_part = stmts[: last_accum_idx + 1]
    epilogue = stmts[last_accum_idx + 1 :]
    if not epilogue:
        return stmts
    writes = [s for s in epilogue if isinstance(s, Write)]
    if len(writes) != 1:
        return stmts
    write = writes[0]
    if write.is_vector:
        return stmts
    acc_name = _find_first_accum_name(stmts)
    if acc_name is None:
        return stmts
    assigns_by_name = {s.name: s for s in epilogue if isinstance(s, Assign)}
    if not _is_linear_in_accum(write.value, acc_name, assigns_by_name):
        return stmts

    write_acc = Write(
        output=write.output,
        index=write.index,
        value=acc_name,
        value_dtype=write.value_dtype,
    )
    cond = Cond(
        cond=BinaryExpr("==", Var(k_s_name), Literal(0, "int")),
        body=Body(epilogue),
        else_body=Body((write_acc,)),
    )
    return reduce_part + (cond,)


def _split_kernel_fully(loop_op: LoopOp, ctx: Context, *, kernel_name: str = "") -> list[TileOp] | None:
    """Unified σ-split for matmul, pointwise, and cooperative-reduce kernels.

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
    # Symbolic free axes (Dim("seq_len") etc.) bind whole-to-block: no
    # split, BN/BM forced to 1. Pass E=1 to the enumerator so divisibility
    # filters pass vacuously; ``_build_split_body`` reads the real symbolic
    # extent back from the Loop axis when stamping the ``*_b`` block axis.
    n_symbolic = not outer_n.axis.extent.is_static
    m_symbolic = outer_m is not None and not outer_m.axis.extent.is_static
    E_N = 1 if n_symbolic else outer_n.axis.extent.as_static()
    E_M = 1 if m_symbolic else (outer_m.axis.extent.as_static() if outer_m is not None else 1)

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
        # Matmul on symbolic K or free axes is M2+ territory (BK/BN/BM tiling
        # all need a concrete extent to split against). Bail out cleanly.
        if not k_loop.axis.extent.is_static or n_symbolic or m_symbolic:
            raise RuleSkipped(
                f"matmul kernel has symbolic axis (K={k_loop.axis.extent!r}, "
                f"M={outer_m.axis.extent!r}, N={outer_n.axis.extent!r}); "
                f"matmul split requires static extents (extend partition_loops to support this)"
            )
        E_K = k_loop.axis.extent.as_static()
        # target_names unions over outer_n.body (the matmul K reduces) and
        # the prologue (softmax max/sum reduces sharing the matmul K extent,
        # axis-name-unified by unify_sibling_reduce_axes upstream). For
        # plain matmul / matmul_add / gated_mlp the prologue is empty and
        # this collapses to ``{lp.axis.name for lp in matmul_reduces}``.
        prologue_reduces = tuple(lp for lp in Body(prologue).iter_of_type(Loop) if lp.is_reduce and lp.axis.extent.as_static() == E_K)
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
        has_nonlinear_epilogue = _has_nonlinear_post_reduce_epilogue(outer_n.body)
        force_splitk_one = multi_accum or bool(prologue) or has_nonlinear_epilogue
        param_combos = _enumerate_cartesian(E_M=E_M, E_N=E_N, E_K=E_K, ctx=ctx, priority_mode="matmul", force_splitk_one=force_splitk_one)
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
        param_combos = _enumerate_cartesian(E_M=E_M, E_N=E_N, E_K=E_K, ctx=ctx, priority_mode="reduce")
    else:
        # Pointwise — no qualifying reduce.
        k_loop = None
        target_names = frozenset()
        param_combos = _enumerate_cartesian(E_M=E_M, E_N=E_N, E_K=1, ctx=ctx, priority_mode="pointwise")

    shape = KernelShape(
        outer_n=outer_n,
        outer_m=outer_m,
        extra_outer=extra_outer,
        k_loop=k_loop,
        target_names=target_names,
        prologue=prologue,
    )
    leading, _ = _split_leading_non_loops(loop_op.body)
    variants: list[TileOp] = []
    for params in param_combos:
        try:
            chain_body = _build_split_body(shape, params)
        except _BuildSkipped:
            continue
        new_body = leading + chain_body
        knobs = {
            **loop_op.knobs,
            BN.name: params.bn,
            BM.name: params.bm,
            FM.name: params.fm,
            FN.name: params.fn,
            BK.name: params.bk,
            SPLITK.name: params.splitk,
            BR.name: params.br,
        }
        variants.append(TileOp(body=new_body, name=kernel_name, knobs=knobs))
    return variants or None


def _priority_matmul(p: TileParams) -> tuple[int, ...]:
    # High cells/thread (amortize K-loop) capped at 32 (NVRTC compile time),
    # threads near 256, larger BK, smaller SPLITK.
    threads = p.bn * p.bm
    return (min(p.fm * p.fn, 32), -abs(256 - threads), p.bk, -p.splitk)


def _priority_pointwise(p: TileParams) -> tuple[int, ...]:
    # Memory-bandwidth bound — fewer cells/thread → more CTAs → better
    # SM occupancy. Threads near 256.
    threads = p.bn * p.bm
    return (-(p.fm * p.fn), -abs(256 - threads))


def _priority_reduce(p: TileParams) -> tuple[int, ...]:
    # Warp-sized BR enables warp-shuffle Combine; threads near 256.
    threads = p.bn * p.bm * p.br
    return (min(p.br, 256), -abs(256 - threads), p.bk, -p.splitk)


_NarrowFn = Callable[[tuple[int, ...]], tuple[int, ...]]


def _enumerate_cartesian(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    ctx: Context,
    priority_mode: str,
    force_splitk_one: bool = False,
) -> list[TileParams]:
    """Pruned cartesian over ``(BN, BM, FM, FN, BK, SPLITK, BR)``, sorted by
    priority.

    Picks the canonical candidate tuples for ``priority_mode`` (``matmul`` /
    ``reduce`` / ``pointwise``); the choice sets are tightly coupled to the
    kernel class so each one lives here, not at the call site:

        matmul   : BN/BM = _TUNE_AXIS_CHOICES; BK = _BK_CANDIDATES; BR = (1,);
                   SPLITK = _SPLITK_CANDIDATES — clipped to (1,) when
                   ``force_splitk_one`` is set (caller passes True for
                   non-linear post-reduce combines like gated_mlp /
                   sdpa where ``sum_i (c·a_i + r) = c·sum_i a_i + r``
                   doesn't hold, or for the matmul-with-prologue case
                   where the prologue feeds the matmul). min_k_chunks=2.
        reduce   : BN/BM = (1, *_TUNE_AXIS_CHOICES) — the leading 1 enables
                   the cooperative-K v1 constraint (BR>1 ⇒ BN=BM=1, single
                   THREAD axis for the materializer). SPLITK = (1,) — atomic
                   cross-CTA reduce + barrier for the post-reduce epilogue
                   isn't wired up. BR = _BR_CANDIDATES.
        pointwise: BN/BM = _TUNE_AXIS_CHOICES; BK = SPLITK = BR = (1,) — no
                   K loop to chunk or split.

    Env pins (``DEPLODOCK_<KNOB>``, set directly or splatted from
    ``DEPLODOCK_KNOBS`` at ``knob.apply_knobs_env``) are folded into the
    candidate lists via ``Knob.narrow`` here in the wrapper for the five
    static choices, and via ``fm_narrow`` / ``fn_narrow`` callables passed
    into the impl for the per-iteration FM/FN divisor lists. When the
    pinned enumeration is empty *and* any planner pin is set we retry with
    every narrow disabled (raw tuples, ``None`` callables): pins are meant
    to scope the kernel under test, but a graph that fuses peer kernels
    (SDPA = QK^T + P@V; gated MLP at full-model scale) may have peers where
    the pin is invalid by divisibility. Without the fallback those peers
    would ``RuleSkipped`` the planner and leave a ``LoopOp`` in the lowered
    graph, tripping ``CudaBackend``.

    BN/BM clamped to extent + divisibility-checked. FM/FN as divisors of the
    per-thread remainder (auto-divisibility), capped by ``_MAX_CELLS_PER_THREAD``.
    BK/SPLITK divisor-checked against ``per_thread_K = E_K // BR``.
    ``BN·BM·BR ≤ ctx.max_threads_per_cta`` (typically 1024).

    Single-K-iter (per_thread_K == bk) is allowed for pointwise and
    cooperative-reduce, rejected for matmul (``min_k_chunks=2`` — needs ≥ 2
    chunks to amortize K-loop overhead)."""
    # ``_TUNE_AXIS_CHOICES`` already includes ``1`` so tiny output extents
    # (e.g. ``torch.matmul`` of 4×3×2) and global-reduce kernels with a
    # phantom size-1 outer axis survive enumeration. ``_wrap_tower`` drops
    # the resulting size-1 THREAD axes before they reach the IR, so the
    # broader search space only changes behavior for tiny shapes.
    if priority_mode == "matmul":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = _BK_CANDIDATES
        splitk_choices = (1,) if force_splitk_one else _SPLITK_CANDIDATES
        br_choices: tuple[int, ...] = (1,)
        min_k_chunks = 1
        priority_fn: Callable[[TileParams], tuple[int, ...]] = _priority_matmul
    elif priority_mode == "reduce":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = _BK_CANDIDATES
        splitk_choices = (1,)
        br_choices = _BR_CANDIDATES
        min_k_chunks = 1
        priority_fn = _priority_reduce
    elif priority_mode == "pointwise":
        bn_choices = _TUNE_AXIS_CHOICES
        bm_choices = _TUNE_AXIS_CHOICES
        bk_choices = (1,)
        splitk_choices = (1,)
        br_choices = (1,)
        min_k_chunks = 1
        priority_fn = _priority_pointwise
    else:
        raise ValueError(f"unknown priority_mode {priority_mode!r}")

    def _run(apply_pins: bool) -> list[TileParams]:
        return _enumerate_cartesian_impl(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            bn_choices=BN.narrow(bn_choices) if apply_pins else bn_choices,
            bm_choices=BM.narrow(bm_choices) if apply_pins else bm_choices,
            bk_choices=BK.narrow(bk_choices) if apply_pins else bk_choices,
            splitk_choices=SPLITK.narrow(splitk_choices) if apply_pins else splitk_choices,
            br_choices=BR.narrow(br_choices) if apply_pins else br_choices,
            fm_narrow=FM.narrow if apply_pins else None,
            fn_narrow=FN.narrow if apply_pins else None,
            max_threads_per_cta=ctx.max_threads_per_cta,
            min_k_chunks=min_k_chunks,
            priority_fn=priority_fn,
        )

    result = _run(apply_pins=True)
    if result or not _planner_pin_set():
        return result
    return _run(apply_pins=False)


def _enumerate_cartesian_impl(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    bn_choices: tuple[int, ...],
    bm_choices: tuple[int, ...],
    bk_choices: tuple[int, ...],
    splitk_choices: tuple[int, ...],
    br_choices: tuple[int, ...],
    fm_narrow: _NarrowFn | None,
    fn_narrow: _NarrowFn | None,
    max_threads_per_cta: int,
    min_k_chunks: int,
    priority_fn: Callable[[TileParams], tuple[int, ...]],
) -> list[TileParams]:
    """Pure cartesian enumeration: caller supplies the (possibly already
    pin-narrowed) choice tuples, the per-iteration FM/FN narrow callables
    (``None`` to skip), the per-thread K-chunk floor, and the sort key.
    No env reads, no mode dispatch."""
    seen: set[TileParams] = set()
    ordered: list[TileParams] = []
    for bn in bn_choices:
        bn_c = min(bn, E_N)
        if bn_c < 1 or E_N % bn_c != 0:
            continue
        for bm in bm_choices:
            bm_c = min(bm, E_M)
            if bm_c < 1 or E_M % bm_c != 0:
                continue
            if bn_c * bm_c > max_threads_per_cta:
                continue
            # v1 cooperative constraint: BR > 1 ⇒ BN = BM = 1.
            br_eligible: tuple[int, ...] = br_choices if (bn_c == 1 and bm_c == 1) else (1,)
            for br in br_eligible:
                if br < 1 or E_K % br != 0:
                    continue
                if bn_c * bm_c * br > max_threads_per_cta:
                    continue
                # Lowering requires at least one BIND_THREAD axis on the
                # Tile (materializer's _materialize raises otherwise).
                # With bn = bm = br = 1 every output axis lands in BLOCK
                # / REGISTER and the THREAD set is empty — skip.
                if bn_c * bm_c * br == 1:
                    continue
                per_thread_K = E_K // br
                fm_candidates = _divisors_up_to(E_M // bm_c, _MAX_CELLS_PER_THREAD)
                if fm_narrow is not None:
                    fm_candidates = fm_narrow(fm_candidates)
                for fm in fm_candidates:
                    fn_candidates = _divisors_up_to(E_N // bn_c, _MAX_CELLS_PER_THREAD // fm)
                    if fn_narrow is not None:
                        fn_candidates = fn_narrow(fn_candidates)
                    for fn in fn_candidates:
                        for bk in bk_choices:
                            if per_thread_K % bk != 0:
                                continue
                            # Skip when this (bk, per_thread_K) yields fewer than
                            # ``min_k_chunks`` K chunks — matmul uses 2 to amortize
                            # K-loop overhead; reduce/pointwise pass 1 (a no-op
                            # given the divisor check above).
                            if per_thread_K > 1 and per_thread_K < bk * min_k_chunks:
                                continue
                            if bk > per_thread_K:
                                continue
                            k_o_total = per_thread_K // bk
                            for splitk in splitk_choices:
                                if k_o_total % splitk != 0:
                                    continue
                                params = TileParams(bn=bn_c, bm=bm_c, fm=fm, fn=fn, bk=bk, splitk=splitk, br=br)
                                if params in seen:
                                    continue
                                seen.add(params)
                                ordered.append(params)

    ordered.sort(key=priority_fn, reverse=True)
    return ordered


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
    N_axis = shape.outer_n.axis
    N_name = N_axis.name
    N_src = N_axis.source_axis or N_axis
    if N_axis.extent.is_static:
        E_N = N_axis.extent.as_static()
        N_b_ext = E_N // (params.bn * params.fn)
        N_b = Axis(f"{N_name}_b", N_b_ext, source_axis=N_src)
    else:
        # Symbolic N (bn=fn=1 by planner construction): bind whole axis to GridTile;
        # the size-1 normalizer drops N_t / N_r.
        N_b = Axis(f"{N_name}_b", N_axis.extent, source_axis=N_src)
    N_t = Axis(f"{N_name}_t", params.bn, source_axis=N_src)
    N_r = Axis(f"{N_name}_r", params.fn, source_axis=N_src)
    sigma_map[N_name] = Var(N_b.name) * Literal(params.bn * params.fn, "int") + Var(N_t.name) * Literal(params.fn, "int") + Var(N_r.name)

    M_b = M_t = M_r = None
    if shape.outer_m is not None:
        M_axis = shape.outer_m.axis
        M_name = M_axis.name
        M_src = M_axis.source_axis or M_axis
        if M_axis.extent.is_static:
            E_M = M_axis.extent.as_static()
            M_b_ext = E_M // (params.bm * params.fm)
            M_b = Axis(f"{M_name}_b", M_b_ext, source_axis=M_src)
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
    K_o_ext = 0
    K_src: Axis | None = None
    if shape.k_loop is not None:
        K_axis = shape.k_loop.axis
        K_name = K_axis.name
        E_K = K_axis.extent.as_static()
        K_o_ext = E_K // (params.splitk * params.br * params.bk)
        K_src = K_axis.source_axis or K_axis
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
        # SPLITK > 1 with a linear residual epilogue (matmul_add):
        # hoist ``Load(r) + Assign(v=acc+r) + Write(v)`` behind
        # ``Cond(K_s == 0)`` so the residual is added exactly once across
        # the K_s CTAs (other K_s atomic-add the bare Accum). No-op for
        # plain matmul / non-SPLITK / non-linear epilogues.
        if K_s is not None:
            new_inner = _gate_linear_epilogue_on_k_s_zero(new_inner, K_s.name)
    else:
        new_inner = inner_after_outer

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
    # then wrap that combined body.
    if shape.prologue:
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

    layers: list[tuple[Axis, Role | None]] = [(N_r, Role.REGISTER)]
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
