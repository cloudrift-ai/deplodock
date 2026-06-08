"""Gate matmul_add linear residual epilogue on ``K_s == 0``.

Post-planner pass that handles SPLITK > 1's residual atomicity. When the
partition planner emits a matmul-shape kernel with SPLITK > 1, the K-reduce
splits across ``K_s`` CTAs that each atomic-add their partial Accum into
the output. For a ``matmul_add`` linear post-reduce epilogue
``Load(r) + Assign(v=acc+r) + Write(v)`` this would atomic-add the
residual ``K_s`` times instead of once. We rewrite the epilogue under
``Cond(K_s == 0, [Load, Assign, Write_v], else=[Write_acc])`` so K_s == 0
contributes the residual and every other K_s atomic-adds the bare Accum —
``sum_i acc_i + r = c · sum_k a_k + r`` (the residual added exactly once).

Non-linear epilogues (gated_mlp's ``silu(acc_g) * acc_u``, SDPA's softmax)
and fused prologues (SDPA P@V) clamp SPLITK = 1 at the planner via
``has_nonlinear_post_reduce_epilogue`` — by the time this pass runs they
have no K_s axis, the K_s lookup returns None, and this rule trivially
skips. The rewrite logic itself lives in
``_splitk_residual.gate_linear_epilogue_on_k_s_zero`` so both the planner
(for the enumeration-time gating predicate) and this pass (for the post-
emit rewrite) share one source of truth.

Idempotent: after firing, the epilogue lives inside a ``Cond`` at the
gating scope; re-running the rule finds no linear epilogue siblings at
any scope and returns the body unchanged → ``RuleSkipped``.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import is_warp
from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import (
    find_split_k_axis_name,
    gate_linear_epilogue_on_k_s_zero,
)

PATTERN = [Pattern("root", TileOp)]


def _walk_apply_gate(stmts: tuple[Stmt, ...], k_s_name: str) -> tuple[Stmt, ...]:
    """Recurse into every nested ``Body`` and apply the gate at each scope.

    The gate's structural precondition (last Accum-bearing stmt followed
    by a linear-add Write epilogue) only matches one specific level — the
    innermost ``RegisterTile`` body holding
    ``[Init, K-tower (SerialTile chain), Load(r), Assign(v=acc+r),
    Write(v)]``. At every other scope (outer tile wrappers, K-tower
    internals, prologue inserts) the gate returns its input unchanged,
    so a blanket walk is safe and avoids hand-coding the descent through
    GridTile → ThreadTile → RegisterTile(M_r) → RegisterTile(N_r).

    Children first, then this level — so the gate operates on the post-
    recursion form (immaterial today since the gate only fires at one
    scope, but keeps the algorithm robust to future nested-K shapes).
    """
    new_stmts: list[Stmt] = []
    for s in stmts:
        bodies = s.nested()
        if bodies:
            new_bodies = tuple(Body(_walk_apply_gate(tuple(b), k_s_name)) for b in bodies)
            if new_bodies != bodies:
                s = s.with_bodies(new_bodies)
        new_stmts.append(s)
    return gate_linear_epilogue_on_k_s_zero(tuple(new_stmts), k_s_name)


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    op: TileOp = root.op
    if is_warp(op.knobs):
        # MMA path: accumulation flows through the C fragment (one
        # per cell), not through scalar Init/Accum stmts; the
        # linear-residual gate has no surface to hoist here. SplitK on
        # MMA still relies on the codegen-derived atomic-Write rewrite
        # (see plans/mma-fragment-factorization.md M5 / Failure modes).
        raise RuleSkipped("MMA TileOp — split-K residual gate doesn't apply to fragment-accum bodies")
    k_s_name = find_split_k_axis_name(op)
    if k_s_name is None:
        raise RuleSkipped("no split-K axis (SPLITK = 1)")
    new_body = _walk_apply_gate(tuple(op.body), k_s_name)
    if new_body == tuple(op.body):
        raise RuleSkipped("no linear residual epilogue to gate")
    return TileOp(body=Body(new_body), name=op.name, knobs=op.knobs)
