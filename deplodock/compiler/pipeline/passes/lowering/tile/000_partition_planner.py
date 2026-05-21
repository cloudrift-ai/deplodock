"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_launch_geometry``. The
planner is the source of truth for launch-axis structure: it decides
splits (output partition, K chunking, register tile, split-K) and tags
the resulting axes with ``Role`` values (see :class:`Role`). Downstream
materialization passes read the tags and skip their own equivalent
decisions.

**Matmul — joint enumeration.** ``_split_matmul_fully`` enumerates the
full pruned cartesian over ``(BN, BM, FM, FN, BK, SPLITK)`` candidates
and emits one variant per surviving combination. Each variant carries
the σ-split body produced by ``_build_matmul_body``:

    Loop(K_s SPLITK_BLOCK) →
      Loop(M_b BLOCK) → Loop(N_b BLOCK) →
        Loop(M_t THREAD) → Loop(N_t THREAD) →
          Loop(M_r REGISTER) → Loop(N_r REGISTER) →
            <σ-substituted prelude> →
            Loop(K_o SERIAL_OUTER) →
              Loop(K_i STAGE_INNER, reduce, σ(K body)) →
            <σ-substituted post (epilogue rewritten when SPLITK > 1)>

σ-maps:

- M → M_b*(BM*FM) + M_t*FM + M_r
- N → N_b*(BN*FN) + N_t*FN + N_r
- K → K_s*(K_o_count*BK) + K_o*BK + K_i  (K_s omitted when SPLITK=1)

When ``SPLITK > 1`` the planner rewrites the trailing Write to be an
atomic-add (``reduce_op=add``) and wraps any acc-independent residual
contribution in ``Cond(K_s == 0, ...)`` so only one CTA per output cell
contributes the residual.

Each cartesian level iterates only structurally valid candidates:
BN/BM from a pow-2 preset (post-clamp to extent + divisibility check),
FM/FN as divisors of the per-thread remainder, BK as divisors of E_K,
SPLITK as divisors of K_o_total. Thread (≤ 1024) and register
(FM·FN ≤ ``_MAX_CELLS_PER_THREAD``) budgets gate the inner loops.

Variant 0 (what greedy compiles pick) comes from a priority sort over
the surviving cartesian: highest cells/thread (capped at 32), threads
closest to 256/CTA, larger BK, smaller SPLITK. The literal class-tuned
heuristic isn't emitted explicitly — its constituent values are
already in the cartesian, and the priority captures the same intent.

**Pointwise** (no matmul reduce in body) is handled by
``_split_pointwise_fully``: each chain axis is stamped ``Role.THREAD``
(when ext ≤ target) or σ-split into ``A_b BLOCK`` over ``A_t THREAD``
(when ext > target and divides). Outer chain axes beyond the THREAD
slot become ``Role.BLOCK`` whole.

Chunk-reduce for large-K non-matmul reductions is not yet brought
back into the planner — current tests don't depend on it.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis, Role
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Stmt, StridedLoop, Write
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_reduce
from deplodock.compiler.tuning import BodyInfo, thread_tile_shape

PATTERN = [Pattern("root", LoopOp)]

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)
# Per-axis register-tile factor choices (FM, FN candidates).
_TUNE_F_CHOICES: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
# Cap on total per-thread replication (∏ factors). NVRTC compile time
# explodes on more-unrolled bodies.
_MAX_CELLS_PER_THREAD: int = 128

# Knob declarations. The planner is the source of truth for matmul
# axis-structure tuning (BN/BM CTA tile, FM/FN per-thread cells, BK
# K-chunk, SPLITK cross-CTA). Each enumerates its own candidate space
# in ``_split_matmul_fully``; ``hints`` is autotune metadata + the
# ``DEPLODOCK_<NAME>`` env-var registry binding.
BN = Knob("BN", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA innermost THREAD width (matmul output N tile)")
BM = Knob("BM", KnobType.INT, hints=_TUNE_AXIS_CHOICES, help="CTA outer THREAD width (matmul output M tile)")
FM = Knob("FM", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul M (output) axis")
FN = Knob("FN", KnobType.INT, hints=_TUNE_F_CHOICES, help="Per-thread cells along the matmul N (output) axis")
BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")
SPLITK = Knob("SPLITK", KnobType.INT, hints=_SPLITK_CANDIDATES, help="Cross-CTA K-split factor (1 = no split)")


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    loop_op: LoopOp = root.op
    body_info = BodyInfo.of(loop_op.body)

    # Idempotence: once the planner has stamped roles on the body's
    # outer chain, ``_outer_free_loop_chain`` (which requires
    # ``role is None``) returns an empty chain and the branches below
    # return ``None``. No explicit "already planned" marker needed.
    variants: list[LoopOp] | None
    if body_info.has_matmul:
        variants = _split_matmul_fully(loop_op, body_info)
    else:
        variants = _split_pointwise_fully(loop_op, body_info)
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


# --- matmul: joint enumeration ---------------------------------------


def _split_matmul_fully(loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
    """Enumerate the pruned cartesian and emit one variant per
    surviving ``(BN, BM, FM, FN, BK, SPLITK)`` combination."""
    if not body_info.has_matmul:
        return None
    chain = _outer_free_loop_chain(loop_op.body)
    if len(chain) < 2:
        return None
    # Innermost two axes are the matmul output dims (M, N). Anything
    # further outer in the chain (e.g. head_idx in multi-head SDPA) is
    # an extra output axis that becomes BLOCK directly.
    m_idx = len(chain) - 2
    outer_m = chain[m_idx]
    outer_n = chain[m_idx + 1]
    extra_outer = chain[:m_idx]
    M_name = outer_m.axis.name
    N_name = outer_n.axis.name
    E_M = int(outer_m.axis.extent)
    E_N = int(outer_n.axis.extent)

    inner_stmts = tuple(outer_n.body)
    k_loop = _find_first_matmul_reduce(inner_stmts)
    if k_loop is None:
        return None
    K_name = k_loop.axis.name
    E_K = int(k_loop.axis.extent)

    seen: set[tuple[int, int, int, int, int, int]] = set()
    ordered: list[tuple[int, int, int, int, int, int]] = []

    def _add(bn: int, bm: int, fm: int, fn: int, bk: int, splitk: int) -> None:
        key = (bn, bm, fm, fn, bk, splitk)
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)

    # Pruned cartesian over (BN, BM, FM, FN, BK, SPLITK). Every loop
    # iterates only over structurally valid candidates (BN/BM from the
    # pow-2 preset, FM/FN as divisors of the per-thread remainder so
    # divisibility is automatic, BK divides E_K, SPLITK divides K_o);
    # ``_priority`` below sorts the survivors so variant 0 carries the
    # heuristic-spirit shape (high cells/thread, ~256 threads/CTA).
    for bn in _TUNE_AXIS_CHOICES:
        bn_c = min(bn, E_N)
        if bn_c < 1 or E_N % bn_c != 0:
            continue
        for bm in _TUNE_AXIS_CHOICES:
            bm_c = min(bm, E_M)
            if bm_c < 1 or E_M % bm_c != 0:
                continue
            if bn_c * bm_c > 1024:
                continue
            # FM / FN iterate over divisors of the remaining per-thread
            # cell counts. When E_M / bm_c is pow-2 (the common case)
            # these match the legacy pow-2 preset exactly; non-pow-2
            # extents pick up their structural divisors (e.g. E_N=192,
            # bn_c=64 → FN ∈ {1, 3} where the preset only had {1}).
            for fm in _divisors_up_to(E_M // bm_c, _MAX_CELLS_PER_THREAD):
                for fn in _divisors_up_to(E_N // bn_c, _MAX_CELLS_PER_THREAD // fm):
                    for bk in _BK_CANDIDATES:
                        if E_K % bk != 0 or E_K <= bk:
                            continue
                        k_o_total = E_K // bk
                        for splitk in _SPLITK_CANDIDATES:
                            if k_o_total % splitk != 0:
                                continue
                            _add(bn_c, bm_c, fm, fn, bk, splitk)

    # Sort by heuristic-spirit priority so variant 0 carries the
    # class-tuned shape (the explicit literal heuristic combo isn't
    # needed — its constituent values are already in the cartesian, and
    # the priority key captures the same intent the heuristic was
    # designed around).
    def _priority(combo: tuple[int, int, int, int, int, int]) -> tuple[int, ...]:
        bn, bm, fm, fn, bk, splitk = combo
        threads = bn * bm
        cells = fm * fn
        # High cells/thread (capped to 32 — beyond hurts NVRTC).
        cell_score = min(cells, 32)
        # Threads close to 256.
        thread_score = -abs(256 - threads)
        # Bigger BK (fewer K iters).
        bk_score = bk
        # Smaller SPLITK (less atomic contention).
        splitk_score = -splitk
        return (cell_score, thread_score, bk_score, splitk_score)

    ordered.sort(key=_priority, reverse=True)

    leading, _ = _split_leading_non_loops(loop_op.body)
    variants: list[LoopOp] = []
    for bn, bm, fm, fn, bk, splitk in ordered:
        try:
            chain_body = _build_matmul_body(extra_outer, outer_m, outer_n, M_name, N_name, K_name, k_loop, bn, bm, fm, fn, bk, splitk)
        except _BuildSkipped:
            continue
        new_body = leading + chain_body
        knobs = {**loop_op.knobs, BN.name: bn, BM.name: bm, FM.name: fm, FN.name: fn, BK.name: bk, SPLITK.name: splitk}
        variants.append(LoopOp(body=new_body, knobs=knobs))
    return variants or None


class _BuildSkipped(Exception):
    """Raised by ``_build_matmul_body`` when the body has a shape we
    don't know how to rewrite (e.g. epilogue isn't a known split-K
    pattern)."""


def _build_matmul_body(
    extra_outer: tuple[Loop, ...],
    outer_m: Loop,
    outer_n: Loop,
    M_name: str,
    N_name: str,
    K_name: str,
    k_loop_ref: Loop,
    bn: int,
    bm: int,
    fm: int,
    fn: int,
    bk: int,
    splitk: int,
) -> tuple[Stmt, ...]:
    E_M = int(outer_m.axis.extent)
    E_N = int(outer_n.axis.extent)
    E_K = int(k_loop_ref.axis.extent)

    M_b_ext = E_M // (bm * fm)
    N_b_ext = E_N // (bn * fn)
    K_o_ext = E_K // (splitk * bk)

    M_b = Axis(f"{M_name}_b", M_b_ext)
    M_t = Axis(f"{M_name}_t", bm)
    M_r = Axis(f"{M_name}_r", fm)
    N_b = Axis(f"{N_name}_b", N_b_ext)
    N_t = Axis(f"{N_name}_t", bn)
    N_r = Axis(f"{N_name}_r", fn)
    K_s = Axis(f"{K_name}_s", splitk)
    K_o = Axis(f"{K_name}_o", K_o_ext)
    K_i = Axis(f"{K_name}_i", bk)

    sigma_outer = Sigma(
        {
            M_name: Var(M_b.name) * Literal(bm * fm, "int") + Var(M_t.name) * Literal(fm, "int") + Var(M_r.name),
            N_name: Var(N_b.name) * Literal(bn * fn, "int") + Var(N_t.name) * Literal(fn, "int") + Var(N_r.name),
        }
    )
    if splitk > 1:
        sigma_k = Sigma({K_name: Var(K_s.name) * Literal(K_o_ext * bk, "int") + Var(K_o.name) * Literal(bk, "int") + Var(K_i.name)})
    else:
        sigma_k = Sigma({K_name: Var(K_o.name) * Literal(bk, "int") + Var(K_i.name)})

    inner_stmts = tuple(outer_n.body)

    # Step 1: apply σ_outer over all inner stmts. K loop's body Loads
    # pick up M/N substitutions; K iter var is untouched.
    inner_after_outer = tuple(s.rewrite(_identity_rename, sigma_outer) for s in inner_stmts)

    # Step 2: replace the matmul K Loop with K_o → K_i (σ-substituting
    # K inside the K body).
    new_inner, replaced = _replace_matmul_k(inner_after_outer, K_o, K_i, sigma_k)
    if not replaced:
        raise _BuildSkipped("matmul K reduce not found in body")

    # Step 3: when SPLITK > 1, rewrite the epilogue to atomic Write +
    # Cond(K_s == 0, residual).
    if splitk > 1:
        new_inner = _rewrite_epilogue_for_splitk(new_inner, K_s.name)

    # Step 4: wrap with REGISTER → THREAD → BLOCK loops (inside-out).
    current: tuple[Stmt, ...] = new_inner
    current = (Loop(axis=N_r, role=Role.REGISTER, body=current),)
    current = (Loop(axis=M_r, role=Role.REGISTER, body=current),)
    current = (Loop(axis=N_t, role=Role.THREAD, body=current),)
    current = (Loop(axis=M_t, role=Role.THREAD, body=current),)
    current = (Loop(axis=N_b, role=Role.BLOCK, body=current),)
    current = (Loop(axis=M_b, role=Role.BLOCK, body=current),)
    if splitk > 1:
        current = (Loop(axis=K_s, role=Role.SPLITK_BLOCK, body=current),)
    # Extra outer chain axes (e.g. head_idx in multi-head SDPA) become
    # BLOCK directly — they were already iteration axes in the original
    # body; we just re-stamp them so launch_geometry binds BIND_BLOCK.
    for outer_lp in reversed(extra_outer):
        current = (Loop(axis=outer_lp.axis, role=Role.BLOCK, body=current),)
    return current


def _replace_matmul_k(stmts: tuple[Stmt, ...], K_o: Axis, K_i: Axis, sigma_k: Sigma) -> tuple[tuple[Stmt, ...], bool]:
    """Walk ``stmts`` and replace the first matmul-shaped reduce Loop
    with ``Loop(K_o, SERIAL_OUTER, Loop(K_i, reduce, STAGE_INNER, σ_k(body)))``.
    Returns ``(new_stmts, replaced_flag)``."""
    out: list[Stmt] = []
    replaced = False
    for s in stmts:
        if replaced:
            out.append(s)
            continue
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            new_body = tuple(c.rewrite(_identity_rename, sigma_k) for c in s.body)
            k_i_loop = Loop(axis=K_i, role=Role.STAGE_INNER, body=new_body)
            k_o_loop = Loop(axis=K_o, role=Role.SERIAL_OUTER, body=(k_i_loop,))
            out.append(k_o_loop)
            replaced = True
            continue
        if isinstance(s, (Loop, StridedLoop)):
            inner, r = _replace_matmul_k(s.body, K_o, K_i, sigma_k)
            if r:
                out.append(replace(s, body=inner))
                replaced = True
                continue
        if isinstance(s, Cond):
            inner_b, rb = _replace_matmul_k(s.body, K_o, K_i, sigma_k)
            inner_e, re_ = _replace_matmul_k(s.else_body, K_o, K_i, sigma_k)
            if rb or re_:
                out.append(Cond(cond=s.cond, body=inner_b, else_body=inner_e))
                replaced = True
                continue
        out.append(s)
    return tuple(out), replaced


def _find_first_matmul_reduce(stmts) -> Loop | None:
    for s in stmts:
        if isinstance(s, Loop) and s.is_reduce and is_matmul_reduce(s):
            return s
        if isinstance(s, (Loop, StridedLoop)):
            found = _find_first_matmul_reduce(s.body)
            if found is not None:
                return found
        if isinstance(s, Cond):
            found = _find_first_matmul_reduce(s.body) or _find_first_matmul_reduce(s.else_body)
            if found is not None:
                return found
    return None


# --- split-K epilogue rewrite (planner-side) -------------------------


def _rewrite_epilogue_for_splitk(stmts: tuple[Stmt, ...], k_s_name: str) -> tuple[Stmt, ...]:
    """Rewrite the trailing ``Write`` so every CTA atomic-adds its
    partial. Two split-K-safe shapes are recognized:

    1. **Plain matmul**: trailing ``Write(out, idx, acc)`` directly →
       ``Write(out, idx, acc, reduce_op=add)``.
    2. **Linear-additive residual** (matmul_add): ``Assign(v, add, r, acc);
       Write(out, idx, v)`` where ``r`` is a K-independent ``Load`` (either
       in prelude or in epilogue position) → atomic ``Write(acc)`` +
       ``Cond(K_s == 0, atomic Write(r))``.
    3. **Linear-multiplicative chain**: ``Write.value`` is reachable from
       ``acc`` only through multiplies whose other operand is acc-
       independent → keep the chain, mark Write as atomic add.

    If the shape doesn't match a known pattern, raises ``_BuildSkipped``.
    """
    stmts_list = list(stmts)
    k_idx = next(
        (i for i, s in enumerate(stmts_list) if isinstance(s, Loop) and s.role is Role.SERIAL_OUTER),
        None,
    )
    if k_idx is None:
        raise _BuildSkipped("no SERIAL_OUTER K_o loop in inner stmts")
    write_idx = next((i for i, s in enumerate(stmts_list) if isinstance(s, Write)), None)
    if write_idx is None or write_idx <= k_idx:
        raise _BuildSkipped("no Write found after K_o")
    k_outer = stmts_list[k_idx]
    write = stmts_list[write_idx]
    prelude_stmts = stmts_list[:k_idx]
    epilogue_stmts = stmts_list[k_idx + 1 : write_idx]
    acc_name = _find_accum_name(k_outer)
    if acc_name is None:
        raise _BuildSkipped("could not find Accum name inside K_o body")

    if not epilogue_stmts and write.value == acc_name:
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        head = list(prelude_stmts) + [k_outer, new_write]
        return tuple(head + stmts_list[write_idx + 1 :])

    if _is_linear_multiplicative_chain(epilogue_stmts, acc_name, write.value):
        new_write = replace(write, reduce_op=ElementwiseImpl("add"))
        head = list(prelude_stmts) + [k_outer] + list(epilogue_stmts) + [new_write]
        return tuple(head + stmts_list[write_idx + 1 :])

    residual = _extract_simple_residual(prelude_stmts, epilogue_stmts, acc_name, write.value)
    if residual is None:
        raise _BuildSkipped("epilogue isn't a split-K-safe shape")
    residual_name, residual_in_epilogue = residual
    cond_write = Write(output=write.output, index=write.index, value=residual_name, reduce_op=ElementwiseImpl("add"))
    always_write = Write(output=write.output, index=write.index, value=acc_name, reduce_op=ElementwiseImpl("add"))
    if residual_in_epilogue:
        residual_load = next(s for s in epilogue_stmts[:-1] if hasattr(s, "name") and s.name == residual_name)
        cond_body = (residual_load, cond_write)
    else:
        cond_body = (cond_write,)
    cond = Cond(
        cond=BinaryExpr("==", Var(k_s_name), Literal(0, "int")),
        body=Body(cond_body),
        else_body=Body(()),
    )
    head = list(prelude_stmts) + [k_outer, always_write, cond]
    return tuple(head + stmts_list[write_idx + 1 :])


def _find_accum_name(k_o_loop: Loop) -> str | None:
    for s in k_o_loop.body.iter():
        if isinstance(s, Accum):
            return s.name
    return None


def _is_linear_multiplicative_chain(epilogue_stmts: list, acc_name: str, write_value: str) -> bool:
    """``Write.value`` is reachable from ``acc`` only through multiplies
    whose other operand is acc-independent (e.g. ``acc * silu(gate)``).
    Then ``sum_i (c * a_i) = c * sum_i a_i`` distributes — every CTA
    computes its own ``v = c * partial_acc`` and atomic-adds."""
    acc_dep = {acc_name}
    for s in epilogue_stmts:
        deps = set(s.deps())
        touches_acc = bool(deps & acc_dep)
        if not touches_acc:
            continue
        if not isinstance(s, Assign):
            return False
        if s.op.name != "multiply":
            return False
        if len(s.args) != 2:
            return False
        a, b = s.args
        a_dep = a in acc_dep
        b_dep = b in acc_dep
        if a_dep == b_dep:
            return False
        acc_dep.add(s.name)
    return write_value in acc_dep


def _extract_simple_residual(prelude_stmts, epilogue_stmts: list, acc_name: str, write_value: str):
    """Match ``Assign(v, add, r, acc); Write(v)`` where ``r`` is a Load
    (in prelude or epilogue position). Returns ``(residual_name, in_epilogue)``
    or ``None`` on any deviation."""
    if not epilogue_stmts:
        return None
    asn = epilogue_stmts[-1]
    if not isinstance(asn, Assign):
        return None
    if asn.name != write_value:
        return None
    if asn.op.name != "add":
        return None
    if acc_name not in asn.args or len(asn.args) != 2:
        return None
    other = next(a for a in asn.args if a != acc_name)
    earlier = epilogue_stmts[:-1]
    if any(not isinstance(s, Load) for s in earlier):
        return None
    if any(s.name == other for s in earlier if isinstance(s, Load)):
        return (other, True)
    if any(isinstance(s, Load) and s.name == other for s in prelude_stmts):
        return (other, False)
    return None


# --- pointwise / non-matmul reduce ----------------------------------


def _split_pointwise_fully(loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
    """Stamp BLOCK/THREAD on the output axes of a non-matmul kernel.

    For each chain axis (innermost-first), apply the per-axis target from
    :func:`tuning.thread_tile_shape`:

    - ``ext ≤ target`` (and within the THREAD slot): stamp ``Role.THREAD``.
    - ``ext > target`` and ``ext % target == 0``: σ-split into
      ``A_b`` (``Role.BLOCK``) over ``A_t`` (``Role.THREAD``), σ:
      ``A → A_b * target + A_t``.
    - Outer chain axes beyond the THREAD slot: stamp ``Role.BLOCK`` whole.

    Returns ``None`` on no-op (every axis already fits the target with no
    σ-split) or on non-divisible axes (the kernel falls through unhandled
    — current pointwise tests don't hit this)."""
    chain = _outer_free_loop_chain(loop_op.body)
    if not chain:
        return None

    leading, _ = _split_leading_non_loops(loop_op.body)

    # Heuristic target (innermost-first).
    extents_desc = tuple(sorted((int(lp.axis.extent) for lp in chain), reverse=True))
    target_tuple = thread_tile_shape(extents_desc, body_info)

    # Walk innermost-first; build (axis, role) entries. Every chain
    # axis is tagged (THREAD/BLOCK) so launch_geometry's chain walker
    # reads the role directly — no untagged fallback needed for kernels
    # the planner accepts.
    chain_inner_first = list(reversed(chain))
    new_levels_inner_first: list[tuple[Axis, Role]] = []
    sigma_map: dict[str, object] = {}
    for i, lp in enumerate(chain_inner_first):
        ext = int(lp.axis.extent)
        if i >= len(target_tuple):
            # Outer beyond target — BLOCK whole.
            new_levels_inner_first.append((lp.axis, Role.BLOCK))
            continue
        target = int(target_tuple[i])
        if ext <= target:
            new_levels_inner_first.append((lp.axis, Role.THREAD))
            continue
        if ext % target != 0:
            return None  # divisibility failure — can't σ-split cleanly
        inner_ax = Axis(f"{lp.axis.name}_t", target)
        outer_ax = Axis(f"{lp.axis.name}_b", ext // target)
        new_levels_inner_first.append((inner_ax, Role.THREAD))
        new_levels_inner_first.append((outer_ax, Role.BLOCK))
        sigma_map[lp.axis.name] = Var(outer_ax.name) * Literal(target, "int") + Var(inner_ax.name)

    inner_stmts = tuple(chain[-1].body)
    if sigma_map:
        sigma = Sigma(sigma_map)
        inner_stmts = tuple(s.rewrite(_identity_rename, sigma) for s in inner_stmts)

    # Build the wrapped body (outermost-first by reversing inner-first list).
    current: tuple[Stmt, ...] = inner_stmts
    for axis, role in new_levels_inner_first:
        current = (Loop(axis=axis, role=role, body=current),)

    new_body = leading + current
    bn = int(target_tuple[0]) if target_tuple else 256
    return [LoopOp(body=new_body, knobs={**loop_op.knobs, BN.name: bn})]
