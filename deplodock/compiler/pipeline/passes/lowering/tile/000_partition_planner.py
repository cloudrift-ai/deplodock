"""Partition planner — decide axis-structure splits up front.

Runs first in the Tile-IR lowering chain, **before** ``001_tileify``. The
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

Pruning order: divisibility → thread budget (≤ 1024) → register budget
(FM·FN ≤ ``MAX_CELLS_PER_THREAD``) → dedup after clamp. Heuristic combo
(from ``thread_tile_shape`` / ``register_tile_shape`` / ``forced_bk`` /
``auto_splitk`` on the logical extents) is emitted as variant 0 so
greedy callers without an autotune DB pick the heuristic.

Non-matmul branches (pointwise + chunk-reduce) are not handled here yet
— they fall through to ``004_launch_geometry`` (pointwise) and stay
chunk-less. M16 brings them into the planner.
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
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import (
    MAX_CELLS_PER_THREAD,
    TUNE_F_CHOICES,
    is_matmul_reduce,
)
from deplodock.compiler.tuning import (
    BodyInfo,
    auto_splitk,
    forced_bk,
    register_tile_shape,
    thread_tile_shape,
)

PATTERN = [Pattern("root", LoopOp)]

_PLANNER_KNOB = "PLANNER"

_BK_CANDIDATES = (64, 32, 16, 8, 4, 2)
_TUNE_AXIS_CHOICES: tuple[int, ...] = (16, 32, 64, 128, 256)
_SPLITK_CANDIDATES = (1, 2, 4, 8, 16, 32)

BK = Knob("BK", KnobType.INT, hints=_BK_CANDIDATES, help="Per-stage K-chunk size (intra-CTA K-loop trip count = K / BK)")


def rewrite(ctx: Context, root: Node) -> Graph | None | LoopOp | list[LoopOp]:
    loop_op: LoopOp = root.op
    if loop_op.knobs.get(_PLANNER_KNOB):
        raise RuleSkipped("already planned")

    body_info = BodyInfo.of(loop_op.body)

    variants = _split_matmul_fully(ctx, loop_op, body_info)
    if variants is None:
        raise RuleSkipped("non-matmul kernel — planner only handles matmul currently")

    if len(variants) == 1:
        return _stamp_planned(variants[0])
    return [_stamp_planned(v) for v in variants]


def _stamp_planned(op: LoopOp) -> LoopOp:
    knobs = dict(op.knobs)
    knobs[_PLANNER_KNOB] = True
    return LoopOp(body=op.body, knobs=knobs)


# --- chain helpers ----------------------------------------------------


def _split_leading_non_loops(body) -> tuple[tuple[Stmt, ...], tuple[Stmt, ...]]:
    """Mirror ``001_tileify``: strip leading non-Loop stmts off the body
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


# --- matmul: joint enumeration ---------------------------------------


def _split_matmul_fully(ctx: Context, loop_op: LoopOp, body_info: BodyInfo) -> list[LoopOp] | None:
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

    output_extents: tuple[int, ...] = tuple(sorted((E_M, E_N), reverse=True))

    # Heuristic combo (variant 0).
    h_thread = thread_tile_shape(output_extents, body_info)
    h_bn = int(h_thread[0]) if len(h_thread) >= 1 else _TUNE_AXIS_CHOICES[2]
    h_bm = int(h_thread[1]) if len(h_thread) >= 2 else _TUNE_AXIS_CHOICES[2]
    h_bn_c = min(h_bn, E_N)
    h_bm_c = min(h_bm, E_M)
    h_thread_extents = (h_bm_c, h_bn_c)
    h_fm, h_fn = (int(x) for x in register_tile_shape(output_extents, h_thread_extents, body_info))
    h_bk_pick = forced_bk(output_extents, body_info, ctx.static_smem_cap)
    h_bk = (
        int(h_bk_pick)
        if h_bk_pick is not None and E_K % h_bk_pick == 0 and E_K > h_bk_pick
        else next((c for c in _BK_CANDIDATES if E_K % c == 0 and E_K > c), 0)
    )
    h_k_o = E_K // h_bk if h_bk > 0 else 0
    h_splitk = int(auto_splitk(output_extents, body_info, h_k_o, h_thread_extents)) if h_k_o > 0 else 1
    if h_splitk < 1:
        h_splitk = 1

    seen: set[tuple[int, int, int, int, int, int]] = set()
    ordered: list[tuple[int, int, int, int, int, int]] = []

    def _add(bn: int, bm: int, fm: int, fn: int, bk: int, splitk: int) -> None:
        # Clamp BN/BM to output extents.
        bn_c = min(bn, E_N)
        bm_c = min(bm, E_M)
        if bn_c < 1 or bm_c < 1 or fm < 1 or fn < 1 or bk < 1 or splitk < 1:
            return
        if E_M % (bm_c * fm) != 0:
            return
        if E_N % (bn_c * fn) != 0:
            return
        # K split: BK divides K (intra-CTA chunking); SPLITK divides K_o_total.
        if E_K % bk != 0 or E_K <= bk:
            return
        if (E_K // bk) % splitk != 0:
            return
        if bn_c * bm_c > 1024:
            return
        if fm * fn > MAX_CELLS_PER_THREAD:
            return
        key = (bn_c, bm_c, fm, fn, bk, splitk)
        if key in seen:
            return
        seen.add(key)
        ordered.append(key)

    _add(h_bn, h_bm, h_fm, h_fn, h_bk, h_splitk)
    # Fast-pruned cartesian. Each valid combo is appended; we sort the
    # result by priority below so greedy gets a sensible variant 0 even
    # when the literal heuristic combo fails divisibility.
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
            for fm in TUNE_F_CHOICES:
                if E_M % (bm_c * fm) != 0:
                    continue
                for fn in TUNE_F_CHOICES:
                    if E_N % (bn_c * fn) != 0:
                        continue
                    if fm * fn > MAX_CELLS_PER_THREAD:
                        continue
                    for bk in _BK_CANDIDATES:
                        if E_K % bk != 0 or E_K <= bk:
                            continue
                        k_o_total = E_K // bk
                        for splitk in _SPLITK_CANDIDATES:
                            if k_o_total % splitk != 0:
                                continue
                            _add(bn_c, bm_c, fm, fn, bk, splitk)

    # Re-order so the highest-priority combo (closest to the heuristic
    # intent — high cells/thread, threads close to 256, larger BK) comes
    # first. Greedy compiles pick variant 0 = `ordered[0]`. Heuristic
    # combo (if it survived `_add`) is already at index 0 and stays
    # there via stable sort + matching priority key.
    heuristic_key = (h_bn, h_bm, h_fm, h_fn, h_bk, h_splitk)

    def _priority(combo: tuple[int, int, int, int, int, int]) -> tuple[int, ...]:
        bn, bm, fm, fn, bk, splitk = combo
        threads = bn * bm
        cells = fm * fn
        # Heuristic exact match sorts first.
        is_heuristic = 1 if combo == heuristic_key else 0
        # Then high cells/thread (capped to 32 — beyond hurts NVRTC).
        cell_score = min(cells, 32)
        # Then threads close to 256.
        thread_score = -abs(256 - threads)
        # Then bigger BK (fewer K iters).
        bk_score = bk
        # Then smaller SPLITK (less atomic contention).
        splitk_score = -splitk
        return (is_heuristic, cell_score, thread_score, bk_score, splitk_score)

    ordered.sort(key=_priority, reverse=True)

    leading, _ = _split_leading_non_loops(loop_op.body)
    variants: list[LoopOp] = []
    for bn, bm, fm, fn, bk, splitk in ordered:
        try:
            chain_body = _build_matmul_body(extra_outer, outer_m, outer_n, M_name, N_name, K_name, k_loop, bn, bm, fm, fn, bk, splitk)
        except _BuildSkipped:
            continue
        new_body = leading + chain_body
        knobs = {**loop_op.knobs, "BN": bn, "BM": bm, "FM": fm, "FN": fn, "BK": bk, "SPLITK": splitk}
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
    # body; we just re-stamp them so tileify binds BIND_BLOCK.
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
    """See ``003_split_matmul_k._is_linear_multiplicative_chain``."""
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
    """See ``003_split_matmul_k._extract_simple_residual``."""
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
