"""peel — deterministic post-``assemble`` software-pipelining of TMA K loops (R5).

``plans/tile-ir-block-dag.md``: the ``peel`` post-pass expands a ring-staged
``serial_outer`` K loop into prologue / main / epilogue, deterministically, from the
already-chosen ``ring_depth`` (``StageBundle.buffer_count``) — **no fork** (the transport
decision was the fork upstream; the peel has exactly one output for a given ring depth).
Runs after ``010_assemble`` materializes the basic tower, before the kernel passes lower
the cell, so ``kernel/005_lower_atom_tile`` sees the peeled shape D (prologue ``StageBundle``
+ ``K_o-1`` main loop with issue-next + ``AsyncWait`` + reduce + epilogue drains).

Ported from the deleted legacy ``080_pipeline_stages`` (its TMA / cp.async peel), now
choice-free: a ``serial_outer`` holding a single ring bundle (``policy ∈ {ASYNC, TMA}``,
``pipeline_depth == 1``, static extent ≥ ``buffer_count``) is always peeled. R5 emits only
TMA bundles; the cp.async branch is carried for when that transport lands.
"""

from __future__ import annotations

from dataclasses import replace as _replace
from typing import Any

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import AsyncWait, SerialTile, StageBundle, StagePolicy, TileOp
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering._predicates import reduce_body_has_coupled_accum

PATTERN = [Pattern("root", TileOp)]


def rewrite(ctx: Context, root: Node) -> TileOp:  # noqa: ARG001 — ctx required by rule dispatch signature
    new_body, changed = _walk(root.op.body)
    if not changed:
        raise RuleSkipped("no ring-staged serial_outer K loop to software-pipeline")
    return TileOp(body=new_body, name=root.op.name, knobs=dict(root.op.knobs))


def _walk(body: Body) -> tuple[Body, bool]:
    out: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            expanded = _try_pipeline(s)
            if expanded is not None:
                out.extend(expanded)
                changed = True
                continue
        nested = s.nested()
        if nested:
            new_bodies: list[Body] = []
            sub_changed = False
            for b in nested:
                nb, c = _walk(b)
                new_bodies.append(nb)
                sub_changed = sub_changed or c
            if sub_changed:
                s = s.with_bodies(tuple(new_bodies))
                changed = True
        out.append(s)
    return Body(tuple(out)), changed


def _try_pipeline(kouter: SerialTile) -> list[Stmt] | None:
    # Pipelining peels static iterations into prologue / epilogue (each a literal k).
    if not kouter.axis.extent.is_static or kouter.axis.extent.as_static() < 2:
        return None
    if len(kouter.body) != 1:
        return None
    bundle = kouter.body[0]
    if not isinstance(bundle, StageBundle):
        return None
    if bundle.policy not in (StagePolicy.ASYNC, StagePolicy.TMA):
        return None
    # Idempotence: already-pipelined bundles bear pipeline_depth > 1.
    if bundle.pipeline_depth != 1:
        return None
    reduces = [s for s in bundle.body.iter() if isinstance(s, SerialTile) and s.kind == "stage_inner" and s.is_reduce]
    if not reduces:
        return None
    # Reject inter-iter SSA dependencies in the K_i reduce — the σ peel assumes
    # each chunk is independent (online softmax / Welford compound fp32 drift).
    if any(reduce_body_has_coupled_accum(k_inner.body) for k_inner in reduces):
        return None

    n = kouter.axis.extent.as_static()
    k_var = kouter.axis.name
    buffer_count = bundle.buffer_count
    if n < buffer_count:
        return None
    is_tma = bundle.policy == StagePolicy.TMA
    pipeline_depth = buffer_count

    def _issue_only(sigma: Sigma) -> StageBundle:
        rewritten = bundle.rewrite(_id, sigma)
        return _replace(rewritten, body=Body(()), pipeline_depth=pipeline_depth)

    def _sigma_at(k_expr: Any) -> Sigma:  # noqa: ANN401 — Expr | int union locally
        return Sigma({k_var: k_expr if not isinstance(k_expr, int) else Literal(k_expr, "int")})

    # Prologue: issue chunks 0..pipeline_depth-2 (N-1 outstanding).
    prologue: list[Stmt] = [_issue_only(_sigma_at(i)) for i in range(pipeline_depth - 1)]

    sigma_consume = Sigma({k_var: Var(k_var)})
    sigma_issue_next = _sigma_at(Var(k_var) + Literal(pipeline_depth - 1, "int"))
    next_issue = _issue_only(sigma_issue_next)
    keep = pipeline_depth - 1
    if is_tma:
        body_slot = Var(k_var) % Literal(buffer_count, "int")
        body_phase = (Var(k_var) / Literal(buffer_count, "int")) % Literal(2, "int")
        main_body_stmts: tuple[Stmt, ...] = (
            AsyncWait(keep=keep, phase=body_phase, slot=body_slot),
            *(c.rewrite(_id, sigma_consume) for c in bundle.body),
            next_issue,
        )
    else:
        main_body_stmts = (
            next_issue,
            AsyncWait(keep=keep),
            *(c.rewrite(_id, sigma_consume) for c in bundle.body),
            AsyncWait(keep=keep),
        )

    main_extent = n - (pipeline_depth - 1)
    main_loop = SerialTile(axis=Axis(k_var, main_extent), body=Body(main_body_stmts), kind="serial_outer", unroll=kouter.unroll)

    # Epilogue: drain the remaining N-1 chunks (k = main_extent .. n-1).
    epilogue: list[Stmt] = []
    for offset in range(pipeline_depth - 1):
        k_idx = main_extent + offset
        sigma_k = _sigma_at(k_idx)
        remaining_keep = pipeline_depth - 2 - offset
        if is_tma:
            epi_slot = Literal(k_idx % buffer_count, "int")
            epi_phase = Literal((k_idx // buffer_count) % 2, "int")
            epilogue.append(AsyncWait(keep=remaining_keep, phase=epi_phase, slot=epi_slot))
        else:
            epilogue.append(AsyncWait(keep=remaining_keep))
        epilogue.extend(c.rewrite(_id, sigma_k) for c in bundle.body)

    return [*prologue, main_loop, *epilogue]


def _id(name: str) -> str:
    return name
