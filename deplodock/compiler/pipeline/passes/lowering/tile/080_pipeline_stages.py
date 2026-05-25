"""Software-pipeline an async-staged K-outer loop into prologue/main/epilogue.

Input shape (post 030 / 040 / 050):

    SerialTile(K_o, kind="serial_outer", body=[
        AsyncBufferedStage(sources=..., body=[
            SerialTile(K_i, kind="stage_inner", reduce, body=[<reduce stmts>])
        ])
    ])

Output shape (depth-2 pipeline):

    # Prologue — issue chunk 0
    AsyncBufferedStage(σ_first(sources), pipeline_depth=2, body=Body(()))

    # Steady state — issue chunk K_o+1 while waiting on chunk K_o
    SerialTile(K_o in 0..K-1, kind="serial_outer", body=[
        AsyncBufferedStage(σ_next(sources), pipeline_depth=2, body=Body(())),
        AsyncWait(keep=1),  # leave the just-issued chunk in flight
        SerialTile(K_i, kind="stage_inner", reduce, body=[<reduce stmts>]),
        AsyncWait(keep=1),  # CTA-wide barrier before next iter overwrites
                            # the slot this iter consumed (see cp.async
                            # branch comment in _try_pipeline).
    ])

    # Epilogue — drain final chunk, consume it
    AsyncWait(keep=0)
    SerialTile(K_i, kind="stage_inner", reduce, body=[<reduce stmts σ_last>])

Issue-only stages in the prologue and main-loop body carry
``pipeline_depth = 2``, which suppresses the implicit-wait at the wrap
boundary that ``_emit_stage`` would otherwise emit for
``pipeline_depth == 1``. Explicit ``AsyncWait`` Stmts carry the
schedule information (``keep`` for cp.async, ``phase`` + ``slot`` for
TMA) so the materializer's ``emit_async_wait`` dispatch lowers them to
the correct ``CpAsyncWait`` / ``MbarrierWait`` primitives.

Eligibility:

- Exactly one ``AsyncBufferedStage`` (or ``TmaBufferedStage``) directly
  inside the ``SerialTile(serial_outer)`` — wrap-body always emits a
  single multi-source Stage per K_o body, so the pre-refactor sibling-
  Stages branch is gone.
- The stage's body contains a ``SerialTile(stage_inner, reduce)``
  consumer.
- ``K_o.extent >= 2`` (need prologue + ≥ 1 steady-state iter).
- ``pipeline_depth == 1`` on the stage (idempotence: already-pipelined
  stages skip).
"""

from __future__ import annotations

from dataclasses import replace as _replace

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Stmt
from deplodock.compiler.ir.tile.ir import (
    AsyncWait,
    SerialTile,
    StageBundle,
    StagePolicy,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped

PATTERN = [Pattern("root", TileOp)]

_PIPELINE_DEPTH = 2


def rewrite(ctx: Context, root: Node) -> TileOp | None:
    body = root.op.body
    new_body, changed = _walk(body)
    if not changed:
        raise RuleSkipped("no eligible serial_outer with ASYNC/TMA StageBundle to pipeline")
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
    # Pipelining peels a static last iteration into a separate epilogue (sigma_last
    # references ``n - 1``). Symbolic K can't be peeled at compile time — defer.
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
    # Consumer body must hold the K_i reduce — otherwise pipelining has
    # nothing to peel into the epilogue.
    reduces = [s for s in bundle.body.iter() if isinstance(s, SerialTile) and s.kind == "stage_inner" and s.is_reduce]
    if not reduces:
        return None
    # Reject inter-iter SSA dependencies in the K_i reduce body — the
    # σ_next / σ_last rewrites assume each chunk is independent. Online
    # softmax (running max / sum read mid-iter) and similar shapes
    # compound fp32 drift under the prologue/main/epilogue peel and
    # produce wrong output.
    from deplodock.compiler.ir.stmt import Accum  # noqa: PLC0415

    for k_inner in reduces:
        for c in k_inner.body:
            if isinstance(c, Accum):
                continue
            if any(isinstance(d, Accum) for d in k_inner.body.deps_of(c) if d is not None):
                return None

    n = kouter.axis.extent.as_static()
    k_var = kouter.axis.name
    sigma_first = Sigma({k_var: Literal(0, "int")})
    sigma_next = Sigma({k_var: Var(k_var) + Literal(1, "int")})
    sigma_last = Sigma({k_var: Literal(n - 1, "int")})

    is_tma = bundle.policy == StagePolicy.TMA
    buffer_count = bundle.buffer_count

    def _issue_only(sigma: Sigma) -> StageBundle:
        rewritten = bundle.rewrite(_id, sigma)
        # Issue-only copy used in prologue / steady-state issue carries
        # empty body so the materializer skips the wrap-boundary wait
        # and emits just the producer scaffolding.
        return _replace(rewritten, body=Body(()), pipeline_depth=_PIPELINE_DEPTH)

    # Prologue: issue chunk 0.
    prologue: list[Stmt] = [_issue_only(sigma_first)]

    # Steady state: issue chunk K_o+1, wait on K_o, consume K_o.
    next_issue = _issue_only(sigma_next)
    if is_tma:
        # Per-slot mbarrier semantics: each ring slot has its own mbar
        # that alternates parity 0,1,0,1... over successive uses.
        body_slot = Var(k_var) % Literal(buffer_count, "int")
        body_phase = (Var(k_var) / Literal(buffer_count, "int")) % Literal(2, "int")
        main_body_stmts: tuple[Stmt, ...] = (
            AsyncWait(keep=1, phase=body_phase, slot=body_slot),
            *bundle.body,
            next_issue,
        )
    else:
        # cp.async: ISSUE → WAIT(keep=1) → CONSUME → WAIT(keep=1). The
        # trailing wait's paired __syncthreads prevents next-iter's
        # cp.async issue from overwriting the slot this iter just read.
        main_body_stmts = (
            next_issue,
            AsyncWait(keep=1),
            *bundle.body,
            AsyncWait(keep=1),
        )

    main_loop = SerialTile(
        axis=Axis(k_var, n - 1),
        body=Body(main_body_stmts),
        kind="serial_outer",
        unroll=kouter.unroll,
    )

    # Epilogue: drain final chunk, run last consumer.
    epilogue: list[Stmt] = []
    if is_tma:
        last_k = n - 1
        epi_slot = Literal(last_k % buffer_count, "int")
        epi_phase = Literal((last_k // buffer_count) % 2, "int")
        epilogue.append(AsyncWait(keep=0, phase=epi_phase, slot=epi_slot))
    else:
        epilogue.append(AsyncWait(keep=0))
    epilogue.extend(c.rewrite(_id, sigma_last) for c in bundle.body)

    return [*prologue, main_loop, *epilogue]


def _id(name: str) -> str:
    return name
