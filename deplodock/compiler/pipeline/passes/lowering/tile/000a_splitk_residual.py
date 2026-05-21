"""Cond-wrap the residual contribution for split-K matmul_add.

Runs after ``000_partition_planner`` (which has σ-split K into K_s /
K_o / K_i and wrapped the body with BLOCK / THREAD / REGISTER /
SPLITK_BLOCK loops) and before ``001_launch_geometry`` (which lifts
the tagged chain into ``Tile.axes``).

The planner handles the plain matmul case (``Write(value=acc)``) and
the linear-multiplicative chain case (``v = c·acc``) inline — both
just mark the trailing ``Write`` as ``reduce_op=add``. This rule
handles the third split-K-safe shape, which needs a body-level rewrite
the planner intentionally defers:

    Assign(v, add, r, acc); Write(out, idx, v)
              →
    Write(out, idx, acc, reduce_op=add)             # every CTA atomic-adds
    Cond(K_s == 0,
         Write(out, idx, r, reduce_op=add))         # only one CTA adds the residual

The residual ``r`` is the value of a Load that either sits in the
prelude (``stmts[:k_idx]``, K-independent and hoisted by the lifter)
or in the epilogue (``stmts[k_idx+1:write_idx]``, just before the
``Assign``). In the epilogue case we move the Load into the Cond body
so only one CTA pays for it.

Fires only when:

1. The body has a ``Loop`` tagged ``Role.SPLITK_BLOCK`` (planner stamped K_s).
2. The kernel level contains a ``SERIAL_OUTER`` (K_o) loop followed by
   the additive-residual pattern.
3. The trailing ``Write.reduce_op`` is still ``None`` (idempotence —
   the planner's plain / mult-chain path would have set it).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Role
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Cond, Load, Loop, Stmt, Write
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", LoopOp)]


def rewrite(root: Node) -> Graph | None:
    loop_op: LoopOp = root.op
    k_s = _find_splitk_block(loop_op.body)
    if k_s is None:
        raise RuleSkipped("no SPLITK_BLOCK in body")

    wrappers, kernel_stmts = _descend_to_kernel_level(loop_op.body)
    if kernel_stmts is None:
        raise RuleSkipped("could not locate kernel level (no SERIAL_OUTER inside wrapper chain)")

    new_kernel_stmts = _try_rewrite_residual(kernel_stmts, k_s.axis.name)
    if new_kernel_stmts is None:
        raise RuleSkipped("no additive-residual pattern at kernel level")

    new_body = _recompose(wrappers, new_kernel_stmts)
    return LoopOp(body=new_body, knobs=dict(loop_op.knobs))


# --- structural descent ---------------------------------------------


def _find_splitk_block(stmts) -> Loop | None:
    """Locate the outermost ``Role.SPLITK_BLOCK`` loop in ``stmts``.
    The planner always emits it as the outer wrapper (skipping leading
    non-Loop stmts), but we scan defensively in case other passes
    rearrange leading stmts."""
    cur = stmts
    while True:
        # Skip leading non-Loops (Stages, hoisted Loads).
        loop_idx = next((i for i, s in enumerate(cur) if isinstance(s, Loop)), None)
        if loop_idx is None:
            return None
        first = cur[loop_idx]
        if first.role is Role.SPLITK_BLOCK:
            return first
        # Descend through a single non-SPLITK_BLOCK wrapper if present.
        if len(cur) - loop_idx == 1 and first.role in (Role.BLOCK, Role.THREAD, Role.REGISTER):
            cur = tuple(first.body)
            continue
        return None


def _descend_to_kernel_level(stmts) -> tuple[list[Loop], tuple[Stmt, ...] | None]:
    """Descend through the planner's outer chain wrappers until we
    reach a level containing a SERIAL_OUTER (K_o) loop — the level
    where the Write and any epilogue stmts live. Returns a list of
    Loop wrappers (outermost-first) and the kernel-level stmts."""
    wrappers: list[Loop] = []
    cur: tuple[Stmt, ...] = tuple(stmts)
    while True:
        if any(isinstance(s, Loop) and s.role is Role.SERIAL_OUTER for s in cur):
            return wrappers, cur
        # Must be a single Loop wrapper at this point; otherwise we
        # can't descend further deterministically.
        if len(cur) != 1 or not isinstance(cur[0], Loop):
            return wrappers, None
        wrappers.append(cur[0])
        cur = tuple(cur[0].body)


def _recompose(wrappers: list[Loop], kernel_stmts: tuple[Stmt, ...]) -> tuple[Stmt, ...]:
    """Rebuild the body by wrapping ``kernel_stmts`` in the saved
    Loop wrappers, innermost first."""
    current: tuple[Stmt, ...] = kernel_stmts
    for w in reversed(wrappers):
        current = (dc_replace(w, body=current),)
    return current


# --- residual pattern ------------------------------------------------


def _try_rewrite_residual(stmts: tuple[Stmt, ...], k_s_name: str) -> tuple[Stmt, ...] | None:
    """Detect ``Assign(v, add, r, acc); Write(v)`` at the kernel level
    and rewrite to atomic ``Write(acc)`` + ``Cond(K_s == 0, atomic Write(r))``.
    Returns the rewritten stmts or ``None`` if the pattern doesn't match."""
    stmts_list = list(stmts)
    k_idx = next(
        (i for i, s in enumerate(stmts_list) if isinstance(s, Loop) and s.role is Role.SERIAL_OUTER),
        None,
    )
    if k_idx is None:
        return None
    write_idx = next((i for i, s in enumerate(stmts_list) if isinstance(s, Write)), None)
    if write_idx is None or write_idx <= k_idx:
        return None
    write = stmts_list[write_idx]
    # Idempotence: if the Write is already atomic, the planner (plain /
    # mult-chain path) or a prior firing of this rule handled it.
    if write.reduce_op is not None:
        return None
    k_outer = stmts_list[k_idx]
    prelude_stmts = stmts_list[:k_idx]
    epilogue_stmts = stmts_list[k_idx + 1 : write_idx]
    acc_name = _find_accum_name(k_outer)
    if acc_name is None:
        return None

    residual = _extract_simple_residual(prelude_stmts, epilogue_stmts, acc_name, write.value)
    if residual is None:
        return None
    residual_name, in_epilogue = residual
    cond_write = Write(output=write.output, index=write.index, value=residual_name, reduce_op=ElementwiseImpl("add"))
    always_write = Write(output=write.output, index=write.index, value=acc_name, reduce_op=ElementwiseImpl("add"))
    if in_epilogue:
        residual_load = next(s for s in epilogue_stmts[:-1] if hasattr(s, "name") and s.name == residual_name)
        cond_body: tuple[Stmt, ...] = (residual_load, cond_write)
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


def _extract_simple_residual(prelude_stmts, epilogue_stmts: list, acc_name: str, write_value: str) -> tuple[str, bool] | None:
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
