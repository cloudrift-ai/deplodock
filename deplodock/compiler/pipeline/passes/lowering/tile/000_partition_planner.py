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

from dataclasses import replace

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
    for bn, bm, fm, fn, bk, splitk, br in param_combos:
        try:
            chain_body = _build_split_body(extra_outer, outer_m, outer_n, k_loop, bn, bm, fm, fn, bk, splitk, br, k_is_matmul=k_is_matmul)
        except _BuildSkipped:
            continue
        new_body = leading + chain_body
        knobs = {
            **loop_op.knobs,
            BN.name: bn,
            BM.name: bm,
            FM.name: fm,
            FN.name: fn,
            BK.name: bk,
            SPLITK.name: splitk,
            BR.name: br,
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
) -> list[tuple[int, int, int, int, int, int, int]]:
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
    seen: set[tuple[int, int, int, int, int, int, int]] = set()
    ordered: list[tuple[int, int, int, int, int, int, int]] = []
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
                                key = (bn_c, bm_c, fm, fn, bk, splitk, br)
                                if key in seen:
                                    continue
                                seen.add(key)
                                ordered.append(key)

    def _priority_matmul(combo: tuple[int, int, int, int, int, int, int]) -> tuple[int, ...]:
        bn, bm, fm, fn, bk, splitk, _br = combo
        threads = bn * bm
        cells = fm * fn
        return (
            min(cells, 32),  # high cells/thread (capped — NVRTC compile time)
            -abs(256 - threads),  # threads close to 256
            bk,  # bigger BK (fewer K iters)
            -splitk,  # smaller SPLITK (less atomic contention)
        )

    def _priority_pointwise(combo: tuple[int, int, int, int, int, int, int]) -> tuple[int, ...]:
        bn, bm, fm, fn, bk, splitk, _br = combo
        threads = bn * bm
        cells = fm * fn
        # Pointwise is memory-bandwidth-bound: prefer FEW cells/thread
        # (each cell = one load+op+store; no K-loop arithmetic to
        # amortize register pressure) and threads close to 256/CTA.
        return (
            -cells,  # fewer cells/thread (negate → ascending preference)
            -abs(256 - threads),  # threads close to 256
        )

    def _priority_reduce(combo: tuple[int, int, int, int, int, int, int]) -> tuple[int, ...]:
        bn, bm, _fm, _fn, bk, splitk, br = combo
        threads = bn * bm * br
        # Cooperative reduce: prefer warp-sized-or-larger cooperative
        # groups (BR ≥ 32 lets the materializer use warp-shuffle), threads
        # close to 256/CTA, larger BK (fewer K iters), smaller SPLITK
        # (less atomic contention).
        return (
            min(br, 256),
            -abs(256 - threads),
            bk,
            -splitk,
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
    bn: int,
    bm: int,
    fm: int,
    fn: int,
    bk: int,
    splitk: int,
    br: int = 1,
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
    N_b_ext = E_N // (bn * fn)
    N_b = Axis(f"{N_name}_b", N_b_ext)
    N_t = Axis(f"{N_name}_t", bn)
    N_r = Axis(f"{N_name}_r", fn)
    sigma_map[N_name] = Var(N_b.name) * Literal(bn * fn, "int") + Var(N_t.name) * Literal(fn, "int") + Var(N_r.name)

    # M axis split (optional — None for 1D pointwise).
    M_b = M_t = M_r = None
    if outer_m is not None:
        M_name = outer_m.axis.name
        E_M = int(outer_m.axis.extent)
        M_b_ext = E_M // (bm * fm)
        M_b = Axis(f"{M_name}_b", M_b_ext)
        M_t = Axis(f"{M_name}_t", bm)
        M_r = Axis(f"{M_name}_r", fm)
        sigma_map[M_name] = Var(M_b.name) * Literal(bm * fm, "int") + Var(M_t.name) * Literal(fm, "int") + Var(M_r.name)

    sigma_outer = Sigma(sigma_map)

    # K axis split (optional — None for non-matmul).
    #
    # Cooperative-K (br > 1) introduces a K_c THREAD axis between K_o
    # SERIAL_OUTER and K_i STAGE_INNER, distributing each K_o chunk's
    # ``br · bk`` K-values across ``br`` threads. K_c sits innermost in
    # the σ-stride so adjacent threads read adjacent K (coalesced
    # cooperative loads). When br == 1, K_c is absent and the σ
    # collapses to the existing matmul pattern ``K_o · bk + K_i``.
    K_s = K_c = K_o = K_i = None
    sigma_k: Sigma | None = None
    if k_loop is not None:
        K_name = k_loop.axis.name
        E_K = int(k_loop.axis.extent)
        K_o_ext = E_K // (splitk * br * bk)
        K_s = Axis(f"{K_name}_s", splitk) if splitk > 1 else None
        K_c = Axis(f"{K_name}_c", br) if br > 1 else None
        K_o = Axis(f"{K_name}_o", K_o_ext)
        K_i = Axis(f"{K_name}_i", bk)
        # σ: K = K_s·(K_o_ext·br·bk) + K_o·(br·bk) + K_i·br + K_c
        # K_c innermost (consecutive threads → consecutive K, coalesced).
        # K_i·br stride keeps K_o · K_i · K_c contiguous within a stage.
        inner_expr = Var(K_o.name) * Literal(br * bk, "int")
        if K_c is not None:
            inner_expr = inner_expr + Var(K_i.name) * Literal(br, "int") + Var(K_c.name)
        else:
            inner_expr = inner_expr + Var(K_i.name)
        if K_s is not None:
            inner_expr = Var(K_s.name) * Literal(K_o_ext * br * bk, "int") + inner_expr
        sigma_k = Sigma({K_name: inner_expr})

    # Step 1: apply σ_outer over all stmts inside outer_n's body. K
    # loop's body Loads pick up M/N substitutions; K iter var is
    # untouched.
    inner_stmts = tuple(outer_n.body)
    inner_after_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in inner_stmts)

    # Step 2: replace the K reduce Loop with K_o → K_i if K is present.
    # The matmul branch uses ``is_matmul_reduce`` to identify the K
    # reduce; the cooperative-reduce branch uses the complementary
    # "reduce but not matmul-shape" predicate. Both are structural
    # checks that survive σ-rewrite.
    if k_loop is not None:
        assert sigma_k is not None and K_o is not None and K_i is not None
        if k_is_matmul:
            match = is_matmul_reduce
        else:
            match = lambda lp: lp.is_reduce and not is_matmul_reduce(lp)  # noqa: E731
        new_inner, replaced = _replace_inner_reduce(inner_after_outer, K_o, K_i, sigma_k, match=match)
        if not replaced:
            raise _BuildSkipped("K reduce not found in body")

        # Cooperative-K post-K loop rewrite: when br > 1 and the body has
        # additional non-reduce free Loops iterating the same K extent
        # (RMSNorm's per-K post-pointwise, softmax's final divide loop),
        # rewrite them into a K_o' · K_i' tower under the cooperative
        # K_c axis. Without this, _lift_output_loops would lift the
        # post-K Loop as a second BIND_THREAD axis and break the
        # materializer's single-THREAD-axis Combine assumption.
        if br > 1 and K_c is not None:
            E_K = int(k_loop.axis.extent)
            new_inner = _replace_post_k_loops(new_inner, K_extent=E_K, K_s=K_s, K_c=K_c, br=br, bk=bk)
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
    current: tuple[Stmt, ...] = new_inner
    current = (Loop(axis=N_r, role=Role.REGISTER, body=current),)
    if M_r is not None:
        current = (Loop(axis=M_r, role=Role.REGISTER, body=current),)
    current = (Loop(axis=N_t, role=Role.THREAD, body=current),)
    if M_t is not None:
        current = (Loop(axis=M_t, role=Role.THREAD, body=current),)
    if K_c is not None:
        current = (Loop(axis=K_c, role=Role.COOPERATIVE_STRIDE, body=current),)
    current = (Loop(axis=N_b, role=Role.BLOCK, body=current),)
    if M_b is not None:
        current = (Loop(axis=M_b, role=Role.BLOCK, body=current),)
    if K_s is not None:
        current = (Loop(axis=K_s, role=Role.SPLITK_BLOCK, body=current),)
    # Extra outer chain axes (e.g. head_idx in multi-head SDPA; or any
    # axis further out than the second-innermost in pointwise) become
    # BLOCK directly — they were already iteration axes in the original
    # body; we just re-stamp them so launch_geometry binds BIND_BLOCK.
    for outer_lp in reversed(extra_outer):
        current = (Loop(axis=outer_lp.axis, role=Role.BLOCK, body=current),)
    return current


def _replace_inner_reduce(
    stmts: tuple[Stmt, ...],
    K_o: Axis,
    K_i: Axis,
    sigma_k: Sigma,
    *,
    match,
) -> tuple[tuple[Stmt, ...], bool]:
    """Walk ``stmts`` and replace the first reduce Loop matching ``match``
    with ``Loop(K_o, SERIAL_OUTER, Loop(K_i, reduce, STAGE_INNER, σ_k(body)))``.
    Returns ``(new_stmts, replaced_flag)``.

    ``match(loop)`` is a callable applied to each reduce Loop encountered
    during the walk: ``is_matmul_reduce`` for matmul kernels, the
    complementary "reduce but not matmul" predicate for non-matmul
    cooperative-K reduces. No Combine emission here — that lives in
    ``002_cooperative_reduce``, which reads ``Role.COOPERATIVE_STRIDE`` off the
    lifted ``BoundAxis``."""
    out: list[Stmt] = []
    replaced = False
    for s in stmts:
        if replaced:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and match(s):
            new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
            k_i_loop = Loop(axis=K_i, role=Role.STAGE_INNER, body=new_body)
            k_o_loop = Loop(axis=K_o, role=Role.SERIAL_OUTER, body=(k_i_loop,))
            out.append(k_o_loop)
            replaced = True
            continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, r = _replace_inner_reduce(s.body, K_o, K_i, sigma_k, match=match)
            if r:
                out.append(replace(s, body=inner))
                replaced = True
                continue
        if isinstance(s, Cond):
            inner_b, rb = _replace_inner_reduce(s.body, K_o, K_i, sigma_k, match=match)
            inner_e, re_ = _replace_inner_reduce(s.else_body, K_o, K_i, sigma_k, match=match)
            if rb or re_:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                replaced = True
                continue
        out.append(s)
    return tuple(out), replaced


def _replace_post_k_loops(
    stmts: tuple[Stmt, ...],
    *,
    K_extent: int,
    K_s: Axis | None,
    K_c: Axis,
    br: int,
    bk: int,
) -> tuple[Stmt, ...]:
    """Rewrite every non-reduce free Loop ``Loop(a_post=K_extent, body=...)``
    in ``stmts`` into a K_o' · K_i' tower that σ-rewrites the post-iter
    axis to ``K_o'·(br·bk) + K_i'·br + K_c.name``, matching the σ used
    by the cooperative reduce side. The post Loop's iteration is now
    distributed across the ``br`` K_c threads — each thread writes its
    own K-slice instead of every thread redundantly iterating all K.

    K_o' carries ``Role.SERIAL_OUTER`` (launch_geometry won't lift it).
    K_i' carries ``Role.STAGE_INNER``, semantically "inner slab dim,
    regardless of reduce-status," so M11's ``007_stage_inputs`` tweak
    can pick it up as a Load-collection site alongside the reduce K_i.

    When ``K_s`` is present (SPLITK > 1), wrap the K_o' tower in
    ``Cond(K_s == 0, body=[…])`` so only the K_s=0 CTA executes the
    post — every K_s CTA otherwise repeats the same broadcast-value
    write. Mirrors the existing atomic-Write Cond pattern.

    Naming convention: the per-post-loop axes derive from the source
    axis name (``a_post.name``) — ``{name}_o`` for K_o' and
    ``{name}_i`` for K_i'. This keeps the IR readable when multiple
    post-K loops exist (e.g. softmax) and avoids axis-name collisions
    across siblings."""
    out: list[Stmt] = []
    for s in stmts:
        if isinstance(s, Loop) and not s.is_reduce and int(s.axis.extent) == K_extent:
            a_post = s.axis
            K_op = Axis(f"{a_post.name}_o", K_extent // (br * bk))
            K_ip = Axis(f"{a_post.name}_i", bk)
            # σ: a_post = K_o' · (br · bk) + K_i' · br + K_c
            post_expr = Var(K_op.name) * Literal(br * bk, "int") + Var(K_ip.name) * Literal(br, "int") + Var(K_c.name)
            sigma_post = Sigma({a_post.name: post_expr})
            new_body = tuple(c.rewrite(_identity_rename, sigma_post) for c in s.body)
            k_ip_loop = Loop(axis=K_ip, role=Role.STAGE_INNER, body=new_body)
            k_op_loop: Stmt = Loop(axis=K_op, role=Role.SERIAL_OUTER, body=(k_ip_loop,))
            if K_s is not None:
                k_op_loop = Cond(
                    cond=BinaryExpr("==", Var(K_s.name), Literal(0, "int")),
                    body=(k_op_loop,),
                    else_body=(),
                )
            out.append(k_op_loop)
            continue
        if isinstance(s, (Loop, StridedLoop)):
            inner = _replace_post_k_loops(tuple(s.body), K_extent=K_extent, K_s=K_s, K_c=K_c, br=br, bk=bk)
            if inner != tuple(s.body):
                out.append(replace(s, body=inner))
                continue
        if isinstance(s, Cond):
            b = _replace_post_k_loops(tuple(s.body), K_extent=K_extent, K_s=K_s, K_c=K_c, br=br, bk=bk)
            e = _replace_post_k_loops(tuple(s.else_body), K_extent=K_extent, K_s=K_s, K_c=K_c, br=br, bk=bk)
            if b != tuple(s.body) or e != tuple(s.else_body):
                out.append(Cond(cond=s.cond, body=b, else_body=e))
                continue
        out.append(s)
    return tuple(out)


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
