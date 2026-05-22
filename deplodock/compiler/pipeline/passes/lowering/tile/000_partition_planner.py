"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain. Stamps ``Role`` tags on body Loops;
``001_launch_geometry`` and other downstream rules read the tags and skip
their own decisions.

Output axes split as ``A → A_b·(T·R) + A_t·R + A_r`` (T = BN or BM, R = FN or
FM). K splits as ``K → K_s·(K_o·br·bk) + K_o·(br·bk) + K_i·br + K_c`` (K_s for
SPLITK > 1, K_c for cooperative-K BR > 1). Resulting nesting:

    K_s SPLITK_BLOCK → M_b BLOCK → N_b BLOCK → K_c COOPERATIVE_STRIDE →
      M_t THREAD → N_t THREAD → M_r REGISTER → N_r REGISTER →
        prelude → K_o SERIAL_OUTER → K_i STAGE_INNER (reduce σ(body)) →
        Combine (cross-thread, when K_c is present) → post-K tower → Write

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
a COOPERATIVE_STRIDE thread axis above the BLOCK level and σ_k extends to
``k = k_o·512 + k_i·256 + k_c``; launch_geometry emits a ``Combine`` after
the reduce subtree.

Pointwise collapses to BM = FM = FN = 1; extent-1 sub-axes get inlined by
normalize_body. SPLITK + Write atomicity is handled by launch_geometry's
generic BLOCK-lift rewrite, not here. Cooperative-K's Combine emission lives
in ``001_launch_geometry`` (folded from the deleted ``002_cooperative_reduce``).

Priority keys: matmul prefers high cells/thread (amortize K-loop overhead);
pointwise prefers low cells/thread (memory-bandwidth bound); cooperative
reduce prefers warp-sized-or-larger BR. All target ~256 threads/CTA.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis, Role
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Cond, Loop, Stmt, StridedLoop
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce

PATTERN = [Pattern("root", LoopOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
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


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    loop_op: LoopOp = root.op
    # Idempotence is structural: once roles are stamped, _outer_free_loop_chain
    # returns empty (it requires role=None) and _split_kernel_fully → None.
    variants = _split_kernel_fully(loop_op, ctx)
    if variants is None:
        raise RuleSkipped("kernel shape not handled by planner (or already planned)")

    if len(variants) == 1:
        return variants[0]
    return variants


def _split_leading_non_loops(body) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Split body into ``(leading non-Loop stmts, rest)``. Mirrors
    ``001_launch_geometry``'s prefix handling."""
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


def _outer_free_loop_chain(body) -> tuple[Loop, ...]:
    """Walk the outer single-stmt chain of untagged free Loops, outermost-first."""
    _, rest = _split_leading_non_loops(body)
    out: list[Loop] = []
    cur = rest
    while len(cur) == 1 and isinstance(cur[0], Loop) and not cur[0].is_reduce and cur[0].role is None:
        out.append(cur[0])
        cur = tuple(cur[0].body)
    return tuple(out)


def _identity_rename(name: str) -> str:
    return name


def _wrap_tower(layers: list[tuple[Axis, Role | None]], inner: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Wrap ``inner`` in nested Loops, innermost layer first.
    ``[(K_i, STAGE_INNER), (K_o, SERIAL_OUTER)]`` →
    ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, inner))``."""
    current = inner
    for axis, role in layers:
        current = (Loop(axis=axis, role=role, body=current),)
    return current


def _divisors_up_to(n: int, cap: int) -> tuple[int, ...]:
    """Divisors of ``n`` ≤ ``cap``, ascending. FM / FN candidate set — a
    divisor of ``E / bm_c`` automatically satisfies the divisibility check."""
    if n < 1 or cap < 1:
        return ()
    return tuple(d for d in range(1, min(n, cap) + 1) if n % d == 0)


class _BuildSkipped(Exception):
    """Raised by ``_build_split_body`` when the body's shape doesn't match."""


def _split_kernel_fully(loop_op: LoopOp, ctx: Context) -> list[LoopOp] | None:
    """Unified σ-split for matmul, pointwise, and cooperative-reduce kernels.

    Detection is predicate-driven: ``is_matmul_reduce`` (≥ 2 K-indexed Loads +
    Accum) picks the matmul knob set; any other reduce with extent ≥ warp_size
    picks the cooperative-K set; no qualifying reduce falls to pointwise.
    ``None`` only when there's no outer chain at all."""
    chain = _outer_free_loop_chain(loop_op.body)
    if not chain:
        return None

    outer_n: Loop = chain[-1]
    outer_m: Loop | None = chain[-2] if len(chain) >= 2 else None
    extra_outer: tuple[Loop, ...] = chain[:-2] if outer_m is not None else chain[:-1]
    E_N = int(outer_n.axis.extent)
    E_M = int(outer_m.axis.extent) if outer_m is not None else 1

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
        target_names = frozenset(lp.axis.name for lp in matmul_reduces)
        E_K = int(k_loop.axis.extent)
        # SPLITK > 1 only works when each Write's atomic-add is mathematically
        # equivalent to the unsplit reduce. That requires a linear-in-Accum
        # chain from Accum to Write: ``sum_i (c·a_i + r) = c·sum_i a_i + r``
        # holds for matmul (Write=acc) and matmul_add (Write=add(acc, r))
        # but breaks for non-linear post-reduce combines like
        # ``silu(acc_g) * acc_u`` (gated_mlp) or softmax (sdpa). A simple,
        # conservative proxy: any matmul-reduce loop with > 1 Accum is fusing
        # multiple K-sums into one output cell — must be a non-linear combine
        # (else the fusion would have merged them upstream). Force SPLITK=1.
        multi_accum = any(sum(1 for s in lp.body if isinstance(s, Accum)) > 1 for lp in matmul_reduces)
        splitk_choices = (1,) if multi_accum else _SPLITK_CANDIDATES
        param_combos = _enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            bn_choices=_TUNE_AXIS_CHOICES,
            bm_choices=_TUNE_AXIS_CHOICES,
            bk_choices=_BK_CANDIDATES,
            splitk_choices=splitk_choices,
            max_cells_per_thread=_MAX_CELLS_PER_THREAD,
            max_threads_per_cta=ctx.max_threads_per_cta,
            priority_mode="matmul",
        )
    elif nonmatmul_reduces and int(nonmatmul_reduces[0].axis.extent) >= ctx.warp_size:
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
        E_K = int(k_loop.axis.extent)
        target_names = frozenset(lp.axis.name for lp in all_loops if int(lp.axis.extent) == E_K and not is_matmul_reduce(lp))
        param_combos = _enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            bn_choices=(1, *_TUNE_AXIS_CHOICES),
            bm_choices=(1, *_TUNE_AXIS_CHOICES),
            bk_choices=_BK_CANDIDATES,
            splitk_choices=(1,),
            br_choices=_BR_CANDIDATES,
            max_cells_per_thread=_MAX_CELLS_PER_THREAD,
            max_threads_per_cta=ctx.max_threads_per_cta,
            priority_mode="reduce",
        )
    else:
        # Pointwise — no qualifying reduce.
        k_loop = None
        target_names = frozenset()
        param_combos = _enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=1,
            bn_choices=_TUNE_AXIS_CHOICES,
            bm_choices=_TUNE_AXIS_CHOICES,
            bk_choices=(1,),
            splitk_choices=(1,),
            max_cells_per_thread=_MAX_CELLS_PER_THREAD,
            max_threads_per_cta=ctx.max_threads_per_cta,
            priority_mode="pointwise",
        )

    param_combos = _filter_by_env(param_combos)
    shape = KernelShape(outer_n=outer_n, outer_m=outer_m, extra_outer=extra_outer, k_loop=k_loop, target_names=target_names)
    leading, _ = _split_leading_non_loops(loop_op.body)
    variants: list[LoopOp] = []
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
        variants.append(LoopOp(body=new_body, knobs=knobs))
    return variants or None


def _filter_by_env(params: list[TileParams]) -> list[TileParams]:
    """Drop variants that don't match the env-pinned knob values.

    ``DEPLODOCK_<KNOB>=V`` (or the aggregate ``DEPLODOCK_KNOBS="K1=V1,...``,
    already splatted to per-knob env vars at ``knob.apply_knobs_env``)
    pins the planner to a specific (BN, BM, FM, FN, BK, SPLITK, BR)
    tuple — handy for reproducing a single autotune variant in tests
    or one-off ``deplodock compile`` runs. Unset knobs aren't filtered
    (i.e. ``DEPLODOCK_BN=16`` alone keeps every variant with BN=16
    regardless of the other knobs).

    Falls back to the unfiltered list when the pinned tuple doesn't
    intersect any enumerated variant for *this* kernel's shape — pins
    are meant to scope the matmul-style kernel under test, but a graph
    that fuses several kernels with disparate shapes (SDPA = QK^T +
    P@V; gated MLP at full-model scale) may have peers where the pin
    is invalid by divisibility. Filtering them all out would
    ``RuleSkipped`` the planner on the peer and leave a ``LoopOp`` in
    the lowered graph, which trips ``CudaBackend``. The fallback keeps
    those peer kernels lowering through rule defaults while the
    pinned kernel still gets its forced variant."""
    pins: dict[str, int] = {}
    for knob, attr in ((BN, "bn"), (BM, "bm"), (FM, "fm"), (FN, "fn"), (BK, "bk"), (SPLITK, "splitk"), (BR, "br")):
        raw = os.environ.get(knob.env)
        if raw is None:
            continue
        try:
            pins[attr] = int(raw)
        except ValueError:
            continue
    if not pins:
        return params
    filtered = [p for p in params if all(getattr(p, attr) == v for attr, v in pins.items())]
    return filtered if filtered else params


def _enumerate_cartesian(
    *,
    E_M: int,
    E_N: int,
    E_K: int,
    bn_choices: tuple[int, ...],
    bm_choices: tuple[int, ...],
    bk_choices: tuple[int, ...],
    splitk_choices: tuple[int, ...],
    br_choices: tuple[int, ...] = (1,),
    max_cells_per_thread: int,
    max_threads_per_cta: int,
    priority_mode: str,
) -> list[TileParams]:
    """Pruned cartesian over ``(BN, BM, FM, FN, BK, SPLITK, BR)``, sorted by
    priority.

    BN/BM clamped to extent + divisibility-checked. FM/FN as divisors of the
    per-thread remainder (auto-divisibility), capped by ``max_cells_per_thread``.
    BK/SPLITK divisor-checked against ``per_thread_K = E_K // BR``.
    ``BN·BM·BR ≤ max_threads_per_cta`` (typically 1024, from ``ctx``).

    Single-K-iter (per_thread_K == bk) is allowed for pointwise and
    cooperative-reduce, rejected for matmul (≥ 2 chunks needed to amortize
    K-loop overhead).

    v1 cooperative constraint: BR > 1 forces BN = BM = 1 — the materializer's
    Combine path requires a single THREAD axis."""
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
                for fm in _divisors_up_to(E_M // bm_c, max_cells_per_thread):
                    for fn in _divisors_up_to(E_N // bn_c, max_cells_per_thread // fm):
                        for bk in bk_choices:
                            if per_thread_K % bk != 0:
                                continue
                            # Matmul needs ≥ 2 K chunks per thread; reduce/pointwise OK with 1.
                            if priority_mode == "matmul" and per_thread_K > 1 and per_thread_K <= bk:
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

    priority_fn = {
        "matmul": _priority_matmul,
        "pointwise": _priority_pointwise,
        "reduce": _priority_reduce,
    }[priority_mode]
    ordered.sort(key=priority_fn, reverse=True)
    return ordered


def _build_split_body(shape: KernelShape, params: TileParams) -> tuple[Stmt, ...]:
    """σ-split ``shape.outer_n``'s body and wrap in the output
    BLOCK/THREAD/REGISTER tower. ``shape.outer_m`` / ``shape.k_loop`` are
    None for 1D pointwise / non-reduce kernels.

    K_s and K_c (when present) are shared across all reduces in the kernel;
    K_o / K_i are per-K-Loop, built inside ``_replace_k_loops``. SPLITK
    atomic-Write is deferred to ``001_launch_geometry``."""
    sigma_map: dict[str, object] = {}

    N_name = shape.outer_n.axis.name
    E_N = int(shape.outer_n.axis.extent)
    N_b_ext = E_N // (params.bn * params.fn)
    N_b = Axis(f"{N_name}_b", N_b_ext)
    N_t = Axis(f"{N_name}_t", params.bn)
    N_r = Axis(f"{N_name}_r", params.fn)
    sigma_map[N_name] = Var(N_b.name) * Literal(params.bn * params.fn, "int") + Var(N_t.name) * Literal(params.fn, "int") + Var(N_r.name)

    M_b = M_t = M_r = None
    if shape.outer_m is not None:
        M_name = shape.outer_m.axis.name
        E_M = int(shape.outer_m.axis.extent)
        M_b_ext = E_M // (params.bm * params.fm)
        M_b = Axis(f"{M_name}_b", M_b_ext)
        M_t = Axis(f"{M_name}_t", params.bm)
        M_r = Axis(f"{M_name}_r", params.fm)
        sigma_map[M_name] = (
            Var(M_b.name) * Literal(params.bm * params.fm, "int") + Var(M_t.name) * Literal(params.fm, "int") + Var(M_r.name)
        )

    sigma_outer = Sigma(sigma_map)

    # K axes: K_s / K_c are kernel-wide (single SPLITK / single cooperative
    # thread direction); K_o / K_i are per-K-Loop, built inside _replace_k_loops.
    K_s = K_c = None
    K_o_ext = 0
    if shape.k_loop is not None:
        K_name = shape.k_loop.axis.name
        E_K = int(shape.k_loop.axis.extent)
        K_o_ext = E_K // (params.splitk * params.br * params.bk)
        K_s = Axis(f"{K_name}_s", params.splitk) if params.splitk > 1 else None
        K_c = Axis(f"{K_name}_c", params.br) if params.br > 1 else None

    # σ-rewrite outer_n's body (M/N axes), then replace every K-iter Loop with
    # a K_o · K_i tower. Both paths use shared canonical K_o / K_i names so
    # 007_stage_inputs row-cache can merge structurally-equivalent Loads.
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
    else:
        new_inner = inner_after_outer

    # Wrap tower, innermost first. extent-1 layers (e.g. M_t / N_t under v1
    # cooperative BN=BM=1) are inlined later by normalize_body.
    layers: list[tuple[Axis, Role | None]] = [(N_r, Role.REGISTER)]
    if M_r is not None:
        layers.append((M_r, Role.REGISTER))
    layers.append((N_t, Role.THREAD))
    if M_t is not None:
        layers.append((M_t, Role.THREAD))
    if K_c is not None:
        layers.append((K_c, Role.COOPERATIVE_STRIDE))
    layers.append((N_b, Role.BLOCK))
    if M_b is not None:
        layers.append((M_b, Role.BLOCK))
    if K_s is not None:
        layers.append((K_s, Role.SPLITK_BLOCK))
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
            K_o = Axis(f"{K_canonical_name}_o", K_o_ext)
            K_i = Axis(f"{K_canonical_name}_i", bk)
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
