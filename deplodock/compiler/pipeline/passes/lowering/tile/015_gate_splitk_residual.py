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
from deplodock.compiler.pipeline.passes.lowering.tile._splitk_residual import gate_linear_epilogue_on_k_s_zero

PATTERN = [Pattern("root", TileOp)]


def _find_split_k_axis_name(op: TileOp) -> str | None:
    """Return the split-K block axis name from the TileOp's coordination.

    Uses the same structural derivation the materializer + Kernel-IR
    render rely on for ``atomicAdd`` emission: ``Body.coordination``
    (``deplodock/compiler/ir/stmt/body.py:603``) classifies a Write's
    ``atomic_axes`` as the block axes NOT in the Write's index — which
    is the mathematical definition of split-K (block coord doesn't pick
    an output cell, so the CTAs at that coord all race on the same
    cell and need an atomic add). Same signal `010_lower_kernelop` uses
    to pick atomicAdd vs plain store; reading it here keeps the gate
    pass and codegen in lockstep without a redundant axis-tag or a
    name-suffix convention.

    Returns ``None`` when no Write has an atomic axis (the SPLITK = 1
    case — every block axis appears in every Write.index). For matmul
    with SPLITK > 1 there's exactly one split-K axis (``K_s``); we
    return its name as ``k_s_name`` for the Cond predicate.
    """
    coord = op.body.coordination
    for write in coord.writes:
        atomic = coord.atomic_axes(write)
        if atomic:
            # Single split-K axis per kernel by planner construction; if
            # multiple ever appear (e.g. nested split-K), pick any — the
            # gate would need a per-axis pass anyway, and that shape isn't
            # emitted today.
            return next(iter(atomic))
    return None


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
    k_s_name = _find_split_k_axis_name(op)
    if k_s_name is None:
        raise RuleSkipped("no split-K axis (SPLITK = 1)")
    new_body = _walk_apply_gate(tuple(op.body), k_s_name)
    if new_body == tuple(op.body):
        raise RuleSkipped("no linear residual epilogue to gate")
    return TileOp(body=Body(new_body), name=op.name, knobs=op.knobs)
