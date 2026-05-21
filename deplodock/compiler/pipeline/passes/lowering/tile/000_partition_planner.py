"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_launch_geometry``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, split-K) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes read the tags and skip their own equivalent
decisions.

``_split_kernel_fully`` handles both matmul and pointwise via the same
σ-split machinery. Each output axis splits into ``A → A_b·(T·R) + A_t·R + A_r``
where ``(T, R) = (BN, FN)`` for the innermost N axis and ``(BM, FM)``
for the second-innermost M axis (matmul; or degenerate ``(1, 1)`` for
pointwise). Anything further out in the chain becomes ``Role.BLOCK``
whole. For matmul, the reduce axis K splits as
``K → K_s·(K_o_count·BK) + K_o·BK + K_i`` (K_s omitted when SPLITK=1).

The resulting nesting (matmul, SPLITK > 1):

    Loop(K_s SPLITK_BLOCK) →
      Loop(M_b BLOCK) → Loop(N_b BLOCK) →
        Loop(M_t THREAD) → Loop(N_t THREAD) →
          Loop(M_r REGISTER) → Loop(N_r REGISTER) →
            <σ-substituted prelude> →
            Loop(K_o SERIAL_OUTER) →
              Loop(K_i STAGE_INNER, reduce, σ(K body)) →
            <σ-substituted post — Write is non-atomic here>

For pointwise the same structure applies with K-related levels absent
and ``BM = FM = FN = 1`` so the M_t / M_r / N_r sub-axes have extent
1 and are removed by the downstream normalize pass; only ``M_b BLOCK``
(whole-extent) and ``N_b BLOCK + N_t THREAD`` remain.

When ``SPLITK > 1`` the planner just produces the σ-split structure
with a non-atomic Write — ``001_launch_geometry`` does the generic
"BIND_BLOCK lift → atomic Write" rewrite once it commits the binding,
including the ``Cond(K_s == 0, …)`` decomposition for the residual
case (where ``Write.value = add(K_s-indep, K_s-dep)``).

Matmul cartesian iterates only structurally valid candidates: BN/BM
from a pow-2 preset (post-clamp to extent + divisibility check),
FM/FN as divisors of the per-thread remainder, BK as divisors of E_K,
SPLITK as divisors of K_o_total. Thread (≤ 1024) and register
(FM·FN ≤ ``_MAX_CELLS_PER_THREAD``) budgets gate the inner loops.
Variant 0 (what greedy compiles pick) comes from a priority sort:
highest cells/thread (capped at 32), threads closest to 256/CTA,
larger BK, smaller SPLITK.

Pointwise enumerates the same ``(BN, BM, FM, FN)`` cartesian with
``BK = SPLITK = 1``. The priority key for pointwise prefers FEWER
cells/thread (memory-bandwidth-bound — extra register tiling hurts
SM occupancy), still targeting threads close to 256/CTA. Autotune
can explore higher-cells variants where they help (~2.5× win
measured on rotary-style 4D shapes via ``FM=32``).

Chunk-reduce for large-K non-matmul reductions is not yet brought
back into the planner — current tests don't depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis, Role
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Cond, Loop, Stmt, StridedLoop
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce
from deplodock.compiler.tuning import BodyInfo

PATTERN = [Pattern("root", LoopOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)
# Cooperative-K thread count (non-matmul reductions). BR=1 collapses to
# a serial chunked reduce; BR>1 distributes K across `BR` threads with a
# cross-thread Combine after the K_o reduce. v1 constraint: BR>1 requires
# BN=BM=1 (single THREAD axis for materializer's _single_thread_var).
_BR_CANDIDATES = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# Per-axis register-tile factor choices (FM, FN candidates).
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Cap on total per-thread replication (∏ factors). NVRTC compile time
# explodes on more-unrolled bodies.
_MAX_CELLS_PER_THREAD: int = 128

# Knob declarations. The planner is the source of truth for matmul
# axis-structure tuning (BN/BM CTA tile, FM/FN per-thread cells, BK
# K-chunk, SPLITK cross-CTA, BR cooperative-K threads). Each enumerates
# its own candidate space in ``_enumerate_cartesian``; ``hints`` is
# autotune metadata + the ``DEPLODOCK_<NAME>`` env-var registry binding.
BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (matmul output N tile)")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul output M tile)")
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul N (output) axis")
BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")
SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")
BR = Knob("BR", KnobType.INT, hints=_BR_CANDIDATES, help="Cooperative-K thread count (1 = pure serial chunked reduce)")


@dataclass(frozen=True)
class TileParams:
    """One concrete (BN, BM, FM, FN, BK, SPLITK, BR) variant produced by
    ``_enumerate_cartesian``. Carries the axis-structure factors that
    flow from the cartesian into ``_build_split_body`` (σ-split tower
    construction) and into the per-variant knob stamps on the emitted
    ``LoopOp``. ``frozen`` for hashability + de-dup in the cartesian's
    ``seen`` set; default ``br=1`` keeps matmul / pointwise call sites
    that ignore cooperative-K terse."""

    bn: int
    bm: int
    fm: int
    fn: int
    bk: int
    splitk: int
    br: int = 1


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    loop_op: LoopOp = root.op
    body_info = BodyInfo.of(loop_op.body)

    # Idempotence: once the planner has stamped roles on the body's
    # outer chain, ``_outer_free_loop_chain`` (which requires
    # ``role is None``) returns an empty chain and ``_split_kernel_fully``
    # returns ``None``. No explicit "already planned" marker needed.
    variants = _split_kernel_fully(loop_op, body_info, ctx)
    if variants is None:
        raise RuleSkipped("kernel shape not handled by planner (or already planned)")

    if len(variants) == 1:
        return variants[0]
    return variants


# --- chain helpers ----------------------------------------------------


def _split_leading_non_loops(body) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Mirror ``001_launch_geometry``: strip leading non-Loop stmts off the body
    (typically hoisted Loads / Stages). Returns ``(leading, rest)``."""
    leading: list[Stmt] = []
    rest = tuple(body)
    while rest and not isinstance(rest[0], Loop):
        leading.append(rest[0])
        rest = rest[1:]
    return tuple(leading), rest


def _outer_free_loop_chain(body) -> tuple[Loop, ...]:
    """Walk the outer single-stmt chain of untagged free Loops outermost-
    first, after skipping leading non-Loop stmts."""
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
    """Wrap ``inner`` in nested ``Loop``s — innermost layer first.
    ``[(K_i, STAGE_INNER), (K_o, SERIAL_OUTER)]`` produces
    ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, inner))``.

    Used by ``_build_split_body`` for the output BLOCK/THREAD/REGISTER
    wrap tower and by ``_replace_k_loops`` for the per-reduce K_o · K_i
    tower. Returns a 1-tuple containing the outermost Loop (or ``inner``
    unchanged when ``layers`` is empty)."""
    current = inner
    for axis, role in layers:
        current = (Loop(axis=axis, role=role, body=current),)
    return current


def _divisors_up_to(n: int, cap: int) -> tuple[int, ...]:
    """All divisors of ``n`` ≤ ``cap``, ascending. Used as the FM / FN
    candidate set: a divisor of ``E_M / bm_c`` automatically satisfies
    the ``E_M % (bm_c * fm) == 0`` constraint, and for pow-2 ``n`` the
    result is the same pow-2 set the legacy preset enumerated."""
    if n < 1 or cap < 1:
        return ()
    return tuple(d for d in range(1, min(n, cap) + 1) if n % d == 0)


# --- unified kernel split ---------------------------------------------


class _BuildSkipped(Exception):
    """Raised by ``_build_split_body`` when the body has a shape we
    don't know how to rewrite (e.g. matmul K reduce not where we
    expect)."""


def _split_kernel_fully(loop_op: LoopOp, body_info: BodyInfo, ctx: Context) -> list[LoopOp] | None:
    """Unified σ-split for matmul, pointwise, and non-matmul-reduce
    kernels.

    Single-pass detection: extract the outer chain (innermost = N, next
    out = M if present), then look inside the innermost body for reduce
    Loops. Classify by predicate, not branch — ``is_matmul_reduce``
    (≥ 2 K-indexed Loads + Accum) selects the matmul knob set;
    any other reduce selects the cooperative-reduce knob set (BR > 1
    allowed); no reduce at all selects the pointwise knob set
    (degenerate K).

    For v1, kernels must have at most one reduce extent (multi-reduce
    fused SDPA-class kernels are out of scope). When multiple reduces
    are present with matching extents, the first matched reduce drives
    the σ-split; downstream rules consume the cooperative role on the
    lifted K_c axis to emit Combine.

    Returns ``None`` when the kernel has no outer chain at all."""
    chain = _outer_free_loop_chain(loop_op.body)
    if not chain:
        return None

    # Output axes — innermost is N, next-out is M (when ≥ 2 chain
    # axes), everything further out is extra_outer (BLOCK-whole).
    outer_n: Loop = chain[-1]
    outer_m: Loop | None = chain[-2] if len(chain) >= 2 else None
    extra_outer: tuple[Loop, ...] = chain[:-2] if outer_m is not None else chain[:-1]
    E_N = int(outer_n.axis.extent)
    E_M = int(outer_m.axis.extent) if outer_m is not None else 1

    # Reduce detection — predicate-driven, not branch-driven. The
    # matmul-shape predicate uses ``is_matmul_reduce``; the cooperative
    # path uses the negation. ``body_info.has_matmul`` is used as a
    # fast preflight to avoid scanning the body twice.
    k_matmul: Loop | None = None
    k_nonmatmul: Loop | None = None
    if body_info.has_matmul:
        k_matmul = _find_first_reduce(tuple(outer_n.body), match=is_matmul_reduce)
    else:
        k_nonmatmul = _find_first_reduce(tuple(outer_n.body), match=lambda lp: lp.is_reduce and not is_matmul_reduce(lp))

    if k_matmul is not None:
        # Matmul branch — innermost two chain axes are M/N (require ≥ 2).
        if outer_m is None:
            return None
        k_loop: Loop | None = k_matmul
        k_is_matmul = True
        E_K = int(k_loop.axis.extent)
        param_combos = _enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            bn_choices=_TUNE_AXIS_CHOICES,
            bm_choices=_TUNE_AXIS_CHOICES,
            bk_choices=_BK_CANDIDATES,
            splitk_choices=_SPLITK_CANDIDATES,
            max_cells_per_thread=_MAX_CELLS_PER_THREAD,
            priority_mode="matmul",
        )
    elif k_nonmatmul is not None and int(k_nonmatmul.axis.extent) >= ctx.warp_size:
        # Non-matmul reduce branch — cooperative-K enabled via BR.
        # Output axes M / N (M optional) go to BLOCK/THREAD as usual;
        # K splits as K_s · K_o · K_c · K_i with K_c at COOPERATIVE_STRIDE.
        # ``bn_choices`` / ``bm_choices`` prepend 1 so the cartesian can
        # explore (BN=1, BM=1) — the v1 BR>1 constraint requires the
        # output thread axes to be extent-1 (sole THREAD axis is K_c).
        # Per-K post-pointwise loops (RMSNorm, softmax) are σ-rewritten
        # in ``_build_split_body`` via ``_replace_post_k_loops`` so they
        # share K_c with the reduce side — no need to gate them out.
        # Gate:
        #   - ``E_K >= ctx.warp_size``: small reduces aren't worth a
        #     cross-thread Combine (a single thread walks the row
        #     faster than a warp-shuffle / tree-halve setup).
        k_loop = k_nonmatmul
        k_is_matmul = False
        E_K = int(k_loop.axis.extent)
        param_combos = _enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=E_K,
            bn_choices=(1, *_TUNE_AXIS_CHOICES),
            bm_choices=(1, *_TUNE_AXIS_CHOICES),
            bk_choices=_BK_CANDIDATES,
            splitk_choices=_SPLITK_CANDIDATES,
            br_choices=_BR_CANDIDATES,
            max_cells_per_thread=_MAX_CELLS_PER_THREAD,
            priority_mode="reduce",
        )
    else:
        # Pointwise — no reduce in body. Degenerate K (E_K = bk = splitk = 1).
        k_loop = None
        k_is_matmul = True  # unused (no K loop), keep default
        param_combos = _enumerate_cartesian(
            E_M=E_M,
            E_N=E_N,
            E_K=1,
            bn_choices=_TUNE_AXIS_CHOICES,
            bm_choices=_TUNE_AXIS_CHOICES,
            bk_choices=(1,),
            splitk_choices=(1,),
            max_cells_per_thread=_MAX_CELLS_PER_THREAD,
            priority_mode="pointwise",
        )

    leading, _ = _split_leading_non_loops(loop_op.body)
    variants: list[LoopOp] = []
    for params in param_combos:
        try:
            chain_body = _build_split_body(extra_outer, outer_m, outer_n, k_loop, params, k_is_matmul=k_is_matmul)
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
    priority_mode: str,
) -> list[TileParams]:
    """Pruned cartesian over ``(BN, BM, FM, FN, BK, SPLITK, BR)``.

    BN / BM iterate the supplied candidate sets, clamped to extent and
    divisibility-checked. FM / FN iterate as divisors of the per-thread
    remainder so divisibility is automatic, with ``FM·FN`` capped at
    ``max_cells_per_thread``. BK / SPLITK iterate the supplied
    candidate sets, divisor-checked against per-thread K / K_o_total.
    BR iterates the supplied cooperative-K thread-count set; ``per_thread_K
    = E_K // BR`` replaces the raw ``E_K`` in the K divisibility checks.
    Total thread budget ``BN · BM · BR ≤ 1024`` gates the inner loops.

    Same algorithm for matmul, pointwise, and reduce — callers supply
    appropriate degenerate sets to collapse unused axes (e.g. pointwise
    passes ``bk_choices=(1,)``, ``splitk_choices=(1,)``, ``br_choices=(1,)``;
    matmul passes ``br_choices=(1,)``). The single-K-iter case
    (``per_thread_K == bk``) is allowed for ``E_K == 1`` (pointwise) and
    for the cooperative-reduce path (``priority_mode == "reduce"``), but
    rejected for matmul (``priority_mode == "matmul"``) which wants ≥ 2
    K chunks per thread to amortize K-loop overhead.

    **v1 cooperative constraint:** BR > 1 requires the sole THREAD axis
    (materializer's ``_single_thread_var``). When ``BN_c > 1`` or
    ``BM_c > 1``, BR is forced to 1 — the M_t / N_t output thread axes
    are not extent-1 and can't coexist with a cooperative-K thread axis
    in v1.

    Returns survivors sorted by priority. ``priority_mode`` selects the
    key — matmul wants high cells/thread (amortize K-loop overhead),
    pointwise wants low cells/thread (memory-bandwidth-bound, more CTAs
    help SM occupancy), reduce wants warp-sized-or-larger cooperative
    groups. All prefer threads close to 256/CTA."""
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
            if bn_c * bm_c > 1024:
                continue
            # v1: BR > 1 requires the sole THREAD axis. When output
            # thread fan-out exists, force BR = 1.
            br_eligible: tuple[int, ...] = br_choices if (bn_c == 1 and bm_c == 1) else (1,)
            for br in br_eligible:
                if br < 1 or E_K % br != 0:
                    continue
                if bn_c * bm_c * br > 1024:
                    continue
                per_thread_K = E_K // br
                for fm in _divisors_up_to(E_M // bm_c, max_cells_per_thread):
                    for fn in _divisors_up_to(E_N // bn_c, max_cells_per_thread // fm):
                        for bk in bk_choices:
                            if per_thread_K % bk != 0:
                                continue
                            # Matmul: require ≥ 2 K chunks per thread.
                            # Reduce/pointwise: single chunk per thread is fine.
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
        threads = p.bn * p.bm
        cells = p.fm * p.fn
        return (
            min(cells, 32),  # high cells/thread (capped — NVRTC compile time)
            -abs(256 - threads),  # threads close to 256
            p.bk,  # bigger BK (fewer K iters)
            -p.splitk,  # smaller SPLITK (less atomic contention)
        )

    def _priority_pointwise(p: TileParams) -> tuple[int, ...]:
        threads = p.bn * p.bm
        cells = p.fm * p.fn
        # Pointwise is memory-bandwidth-bound: prefer FEW cells/thread
        # (each cell = one load+op+store; no K-loop arithmetic to
        # amortize register pressure) and threads close to 256/CTA.
        return (
            -cells,  # fewer cells/thread (negate → ascending preference)
            -abs(256 - threads),  # threads close to 256
        )

    def _priority_reduce(p: TileParams) -> tuple[int, ...]:
        threads = p.bn * p.bm * p.br
        # Cooperative reduce: prefer warp-sized-or-larger cooperative
        # groups (BR ≥ 32 lets the materializer use warp-shuffle), threads
        # close to 256/CTA, larger BK (fewer K iters), smaller SPLITK
        # (less atomic contention).
        return (
            min(p.br, 256),
            -abs(256 - threads),
            p.bk,
            -p.splitk,
        )

    priority_fn = {
        "matmul": _priority_matmul,
        "pointwise": _priority_pointwise,
        "reduce": _priority_reduce,
    }[priority_mode]
    ordered.sort(key=priority_fn, reverse=True)
    return ordered


def _build_split_body(
    extra_outer: tuple[Loop, ...],
    outer_m: Loop | None,
    outer_n: Loop,
    k_loop: Loop | None,
    params: TileParams,
    *,
    k_is_matmul: bool = True,
) -> tuple[Stmt, ...]:
    """Unified σ-split body construction. The N axis is always present;
    ``outer_m`` and ``k_loop`` are ``None`` for 1D pointwise and
    non-matmul kernels respectively, in which case the corresponding
    σ-substitution + Loop tower levels are skipped.

    Each output axis splits as ``A → A_b·(T·R) + A_t·R + A_r`` (with
    ``T = BN | BM``, ``R = FN | FM``); the extent-1 sub-axes generated
    when ``T = R = 1`` (pointwise M and outer slots) are removed
    downstream by ``normalize_extent_one_loops``. K splits as
    ``K → K_s·(K_o_count·BK) + K_o·BK + K_i`` (K_s omitted when
    ``SPLITK == 1``).

    The atomic-Write rewrite for the K_s case is deferred to
    ``001_launch_geometry`` — it's the generic "BIND_BLOCK lift →
    atomic Write" rewrite triggered when K_s lifts to BIND_BLOCK."""
    sigma_map: dict[str, object] = {}

    # N axis split (always present).
    N_name = outer_n.axis.name
    E_N = int(outer_n.axis.extent)
    N_b_ext = E_N // (params.bn * params.fn)
    N_b = Axis(f"{N_name}_b", N_b_ext)
    N_t = Axis(f"{N_name}_t", params.bn)
    N_r = Axis(f"{N_name}_r", params.fn)
    sigma_map[N_name] = Var(N_b.name) * Literal(params.bn * params.fn, "int") + Var(N_t.name) * Literal(params.fn, "int") + Var(N_r.name)

    # M axis split (optional — None for 1D pointwise).
    M_b = M_t = M_r = None
    if outer_m is not None:
        M_name = outer_m.axis.name
        E_M = int(outer_m.axis.extent)
        M_b_ext = E_M // (params.bm * params.fm)
        M_b = Axis(f"{M_name}_b", M_b_ext)
        M_t = Axis(f"{M_name}_t", params.bm)
        M_r = Axis(f"{M_name}_r", params.fm)
        sigma_map[M_name] = (
            Var(M_b.name) * Literal(params.bm * params.fm, "int") + Var(M_t.name) * Literal(params.fm, "int") + Var(M_r.name)
        )

    sigma_outer = Sigma(sigma_map)

    # K axis split (optional — None for non-matmul).
    #
    # Cooperative-K (br > 1) introduces a K_c THREAD axis between K_o
    # SERIAL_OUTER and K_i STAGE_INNER, distributing each K_o chunk's
    # ``br · bk`` K-values across ``br`` threads. K_c sits innermost in
    # the σ-stride so adjacent threads read adjacent K (coalesced
    # cooperative loads). When br == 1, K_c is absent and the σ
    # collapses to the existing matmul pattern ``K_o · bk + K_i``.
    #
    # K_s (cross-CTA) and K_c (cross-thread) are *shared* across all
    # reduces in the kernel — single SPLITK and single cooperative
    # thread axis. K_o and K_i are per-reduce (each reduce gets its own
    # inner-body tower with axis names derived from its source K axis
    # name), built lazily inside ``_replace_k_loops``.
    K_s = K_c = None
    E_K = K_o_ext = 0
    if k_loop is not None:
        K_name = k_loop.axis.name
        E_K = int(k_loop.axis.extent)
        K_o_ext = E_K // (params.splitk * params.br * params.bk)
        K_s = Axis(f"{K_name}_s", params.splitk) if params.splitk > 1 else None
        K_c = Axis(f"{K_name}_c", params.br) if params.br > 1 else None

    # Step 1: apply σ_outer over all stmts inside outer_n's body. K
    # loop's body Loads pick up M/N substitutions; K iter var is
    # untouched.
    inner_stmts = tuple(outer_n.body)
    inner_after_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in inner_stmts)

    # Step 2: rewrite every K-iteration Loop into a K_o · K_i tower
    # σ-rewriting its body. One unified walk handles both:
    #
    #   - reduce K iterators (matmul, cooperative-K max/sum/...): body
    #     carries an Accum, so the K_i wrapper becomes reduce-tagged
    #     automatically.
    #   - per-K post-pointwise iterators (RMSNorm, softmax post-divide):
    #     body has no Accum, K_i stays non-reduce. When SPLITK > 1, the
    #     K_o tower wraps in ``Cond(K_s == 0)`` — every K_s CTA would
    #     otherwise re-execute the post and clobber the gmem cell.
    #     Reduce K skips the Cond; launch_geometry's atomic-Write
    #     rewrite handles cross-CTA SPLITK.
    #
    # All matched Loops share K_o / K_i axis NAMES (derived from the
    # first reduce's K_name). Multi-reduce kernels (softmax max + sum)
    # and reduce + post (RMSNorm) emit structurally identical σ-Loads,
    # which 007_stage_inputs row-cache exploits to merge into one Stage.
    #
    # Matmul calls the helper with ``is_matmul_reduce`` — only matmul-
    # shape reduces match; matmul kernels have no per-K post-pointwise
    # loop, so the cooperative-only Cond branch never fires.
    # Cooperative non-matmul calls with the K-extent predicate so both
    # the reduce K and the post K hit the same rewrite.
    if k_loop is not None:
        K_canonical_name = k_loop.axis.name
        if k_is_matmul:
            match = is_matmul_reduce
        else:
            match = lambda lp: int(lp.axis.extent) == E_K and not is_matmul_reduce(lp)  # noqa: E731
        new_inner, n_replaced = _replace_k_loops(
            inner_after_outer,
            match=match,
            K_canonical_name=K_canonical_name,
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

    # Step 3: wrap inside-out with REGISTER → THREAD → BLOCK loops.
    # K_c (cooperative-K) slots between the output THREAD layer
    # (M_t / N_t) and the BLOCK layer; launch_geometry's
    # _strip_outer_free_chain lifts it as BIND_THREAD. With v1's
    # BR>1 ⇒ BN=BM=1 constraint, the M_t/N_t Loops are extent-1 and get
    # inlined by normalize_body, leaving K_c as the sole THREAD axis in
    # Tile.axes — which the materializer's _single_thread_var requires
    # for Combine emission.
    #
    # Extra outer chain axes (e.g. head_idx in multi-head SDPA; or any
    # axis further out than the second-innermost in pointwise) become
    # BLOCK directly — they were already iteration axes in the original
    # body; we just re-stamp them so launch_geometry binds BIND_BLOCK.
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
    layers.extend((lp.axis, Role.BLOCK) for lp in reversed(extra_outer))
    return _wrap_tower(layers, new_inner)


def _replace_k_loops(
    stmts: tuple[Stmt, ...],
    *,
    match,
    K_canonical_name: str,
    K_s: Axis | None,
    K_c: Axis | None,
    br: int,
    bk: int,
    K_o_ext: int,
) -> tuple[tuple[Stmt, ...], int]:
    """Walk ``stmts`` and rewrite every ``Loop`` matching ``match`` into
    a ``Loop(K_o, SERIAL_OUTER, Loop(K_i, STAGE_INNER, σ(body)))`` tower.
    Returns ``(new_stmts, n_replaced)``.

    Handles both reduce K iterators (matmul + cooperative-K reduce) and
    per-K post-pointwise iterators (RMSNorm / softmax post) uniformly —
    they share the same σ, the same SERIAL_OUTER / STAGE_INNER role
    pair, and the same canonical K_o / K_i axis names. ``Loop.is_reduce``
    is a computed property of the body's Accum presence, so the K_i
    wrapper inherits the correct reduce-status automatically: reduce-K
    bodies (with Accum) produce a reduce-tagged K_i; post-K bodies (no
    Accum) produce a non-reduce K_i.

    The shared canonical naming
    (``{K_canonical_name}_o`` / ``{K_canonical_name}_i``) is what lets
    007_stage_inputs' row-cache merge structurally-equivalent Loads
    across reduce + post into a single Stage.

    SPLITK + non-reduce: when the matched Loop is non-reduce AND
    ``K_s`` is present, wrap the K_o tower in ``Cond(K_s == 0, …)`` so
    only the K_s=0 CTA executes the post — every K_s CTA otherwise
    repeats the same broadcast-value write. Reduce-K skips the Cond;
    launch_geometry's atomic-Write rewrite handles cross-CTA SPLITK
    via the symmetric ``Cond(K_s == 0, atomic(indep))`` decomposition.

    Combine emission lives in
    ``001_launch_geometry._insert_combines_after_reduces``, which keys
    off STAGE_INNER + Accum — post-K's non-reduce STAGE_INNER produces
    no Combine, exactly correct.

    Match composition:

    - Matmul: ``match=is_matmul_reduce`` — only the matmul-shape
      reduce matches; matmul has no per-K post-pointwise.
    - Cooperative non-matmul:
      ``match=lambda lp: lp.axis.extent == E_K and not is_matmul_reduce(lp)``
      — both the cooperative-K reduce and the per-K post-pointwise
      iterate the same E_K and get rewritten in one walk.
    """
    out: list[Stmt] = []
    n_replaced = 0
    for s in stmts:
        if isinstance(s, Loop) and match(s):
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
                s.body, match=match, K_canonical_name=K_canonical_name, K_s=K_s, K_c=K_c, br=br, bk=bk, K_o_ext=K_o_ext
            )
            if r:
                out.append(replace(s, body=inner))
                n_replaced += r
                continue
        if isinstance(s, Cond):
            inner_b, rb = _replace_k_loops(
                s.body, match=match, K_canonical_name=K_canonical_name, K_s=K_s, K_c=K_c, br=br, bk=bk, K_o_ext=K_o_ext
            )
            inner_e, re_ = _replace_k_loops(
                s.else_body,
                match=match,
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
    """Build the σ for a single K-axis split:
    ``K = K_s · (K_o_ext · br · bk) + K_o · (br · bk) + K_i · br + K_c``.
    K_s and K_c terms collapse when those axes are None (SPLITK=1 and
    BR=1 respectively); the K_i · br term loses the ``· br`` when K_c
    is absent, matching the existing matmul σ shape."""
    inner_expr = Var(K_o.name) * Literal(br * bk, "int")
    if K_c is not None:
        inner_expr = inner_expr + Var(K_i.name) * Literal(br, "int") + Var(K_c.name)
    else:
        inner_expr = inner_expr + Var(K_i.name)
    if K_s is not None:
        inner_expr = Var(K_s.name) * Literal(K_o_ext * br * bk, "int") + inner_expr
    return Sigma({K_name: inner_expr})


def _find_first_reduce(stmts, *, match) -> Loop | None:
    """Find the first reduce Loop satisfying ``match`` anywhere in the
    subtree. ``match`` is ``is_matmul_reduce`` for matmul, the negation
    for cooperative-K reduces."""
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and match(s):
            return s
        if isinstance(s, (Loop, StridedLoop)):
            found = _find_first_reduce(s.body, match=match)
            if found is not None:
                return found
        if isinstance(s, Cond):
            found = _find_first_reduce(s.body, match=match) or _find_first_reduce(s.else_body, match=match)
            if found is not None:
                return found
    return None
