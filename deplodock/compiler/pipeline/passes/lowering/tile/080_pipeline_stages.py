"""Software-pipeline an async-staged K-outer loop into prologue/main/epilogue.

Input shape (post 030 / 040 / 050):

    SerialTile(K_o, kind="serial_outer", body=[
        AsyncBufferedStage(sources=..., body=[
            SerialTile(K_i, kind="stage_inner", reduce, body=[<reduce stmts>])
        ])
    ])

Output shape (depth-``N`` pipeline, where ``N = bundle.buffer_count``):

    # Prologue — issue chunks 0..N-2 (N-1 outstanding)
    AsyncBufferedStage(σ_{0}(sources),   pipeline_depth=N, body=Body(()))
    AsyncBufferedStage(σ_{1}(sources),   pipeline_depth=N, body=Body(()))
    ...
    AsyncBufferedStage(σ_{N-2}(sources), pipeline_depth=N, body=Body(()))

    # Steady state — for k in 0..K-N: issue chunk k+N-1, wait for ≤ N-1
    # in-flight (drains chunk k), consume chunk k.
    SerialTile(K_o in 0..K-N+1, kind="serial_outer", body=[
        AsyncBufferedStage(σ_{k+N-1}(sources), pipeline_depth=N, body=Body(())),
        AsyncWait(keep=N-1),                 # chunk k is done
        SerialTile(K_i, kind="stage_inner", reduce, body=[<reduce σ_k>]),
        AsyncWait(keep=N-1),                 # CTA-wide barrier so the next iter
                                             # doesn't overwrite the slot we just
                                             # consumed.
    ])

    # Epilogue — drain the N-1 chunks left in flight
    AsyncWait(keep=N-2); consume σ_{K-N+1}
    AsyncWait(keep=N-3); consume σ_{K-N+2}
    ...
    AsyncWait(keep=0);   consume σ_{K-1}

For ``N == 2`` this collapses to the classical double-buffer prologue/
main/epilogue. For ``N >= 3`` it is the CUTLASS-style multistage shape:
N-1 outstanding TMA / cp.async copies amortize the per-K-step memory
latency under compute.

Issue-only stages in the prologue and main-loop body carry
``pipeline_depth = N``, which suppresses the implicit-wait at the wrap
boundary that ``_emit_stage`` would otherwise emit for
``pipeline_depth == 1``. Explicit ``AsyncWait`` Stmts carry the
schedule information (``keep`` for cp.async, ``phase`` + ``slot`` for
TMA) so the materializer's ``emit_async_wait`` dispatch lowers them to
the correct ``CpAsyncWait`` / ``MbarrierWait`` primitives.

``buffer_count`` is set upstream by ``040_use_ring_buffers`` (its ``BUFFER_COUNT``
knob) — we just inherit it here. Coupling buffer_count and pipeline_depth
by construction matches the CUTLASS multistage model (one smem slot per
in-flight chunk).

Eligibility:

- Exactly one ``AsyncBufferedStage`` (or ``TmaBufferedStage``) directly
  inside the ``SerialTile(serial_outer)`` — wrap-body always emits a
  single multi-source Stage per K_o body, so the pre-refactor sibling-
  Stages branch is gone.
- The stage's body contains a ``SerialTile(stage_inner, reduce)``
  consumer.
- ``K_o.extent >= bundle.buffer_count`` (need ``N-1`` prologue chunks +
  ≥ 1 steady-state iter).
- ``pipeline_depth == 1`` on the stage (idempotence: already-pipelined
  stages skip).
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
from deplodock.compiler.ir.tile.ir import (
    AsyncWait,
    SerialTile,
    StageBundle,
    StagePolicy,
    TileOp,
)
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType

PATTERN = [Pattern("root", TileOp)]

# Default on: software-pipeline the async-staged K-outer loop into
# prologue/main/epilogue (issue chunk k+N-1 while consuming chunk k). Mirrors
# ``TMA`` / ``ASYNC_COPY`` so the pipeline can be controlled explicitly.
# ``DEPLODOCK_PIPELINE_STAGES=0`` keeps the simple depth-1 async-staged loop.
PIPELINE_STAGES = Knob(
    "PIPELINE_STAGES",
    KnobType.BOOL,
    hints=(True, False),
    help="Software-pipeline async-staged K-outer loops into prologue/main/epilogue. 0 = keep the depth-1 staged loop.",
    off=False,
)


def rewrite(ctx: Context, root: Node) -> TileOp | None:  # noqa: ARG001 — ctx required by rule dispatch signature
    # Idempotence: the decision is recorded as the PIPELINE_STAGES knob (every
    # path stamps it now), so a re-scan of the rebound op skips here.
    if PIPELINE_STAGES.name in root.op.knobs:
        raise RuleSkipped("PIPELINE_STAGES already decided (idempotence via knob)")

    def _off() -> TileOp:
        """Record the declined decision: PIPELINE_STAGES=False, body unchanged
        (keeps the depth-1 staged loop, or no staging at all)."""
        return TileOp(body=root.op.body, name=root.op.name, knobs={**root.op.knobs, PIPELINE_STAGES.name: False})

    if not PIPELINE_STAGES.narrow(PIPELINE_STAGES.hints)[0]:
        return _off()
    body = root.op.body
    new_body, changed = _walk(body)
    if not changed:
        # No eligible serial_outer with an ASYNC/TMA bundle to pipeline — record
        # the declined decision rather than leaving the knob unset.
        return _off()
    return TileOp(body=new_body, name=root.op.name, knobs={**root.op.knobs, PIPELINE_STAGES.name: True})


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
    # Pipelining peels static iterations into prologue / epilogue (each carries
    # a literal k index). Symbolic K can't be peeled at compile time — defer.
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
    buffer_count = bundle.buffer_count
    # The classical depth-2 schedule needs n >= 2 (1 prologue + ≥ 1 main).
    # Depth-N needs n >= N (N-1 prologue + ≥ 1 main + N-1 epilogue overlap).
    if n < buffer_count:
        return None
    is_tma = bundle.policy == StagePolicy.TMA
    pipeline_depth = buffer_count

    def _issue_only(sigma: Sigma) -> StageBundle:
        rewritten = bundle.rewrite(_id, sigma)
        # Issue-only copy used in prologue / steady-state issue carries
        # empty body so the materializer skips the wrap-boundary wait
        # and emits just the producer scaffolding.
        return _replace(rewritten, body=Body(()), pipeline_depth=pipeline_depth)

    def _sigma_at(k_expr: Any) -> Sigma:  # noqa: ANN401 — Expr | int union locally
        return Sigma({k_var: k_expr if not isinstance(k_expr, int) else Literal(k_expr, "int")})

    # Prologue: issue chunks 0..pipeline_depth-2 (N-1 outstanding).
    prologue: list[Stmt] = [_issue_only(_sigma_at(i)) for i in range(pipeline_depth - 1)]

    # Steady state: for k in 0..n-pipeline_depth, issue chunk k+N-1, then wait
    # for ≤ N-1 in-flight (which drains chunk k), then consume chunk k.
    sigma_consume = Sigma({k_var: Var(k_var)})
    sigma_issue_next = _sigma_at(Var(k_var) + Literal(pipeline_depth - 1, "int"))
    next_issue = _issue_only(sigma_issue_next)
    keep = pipeline_depth - 1
    if is_tma:
        # Per-slot mbarrier semantics: each ring slot has its own mbar
        # that alternates parity 0,1,0,1... over successive uses.
        body_slot = Var(k_var) % Literal(buffer_count, "int")
        body_phase = (Var(k_var) / Literal(buffer_count, "int")) % Literal(2, "int")
        main_body_stmts: tuple[Stmt, ...] = (
            AsyncWait(keep=keep, phase=body_phase, slot=body_slot),
            *(c.rewrite(_id, sigma_consume) for c in bundle.body),
            next_issue,
        )
    else:
        # cp.async: ISSUE → WAIT(keep=N-1) → CONSUME → WAIT(keep=N-1). The
        # trailing wait's paired __syncthreads prevents next-iter's cp.async
        # issue from overwriting the slot this iter just read.
        main_body_stmts = (
            next_issue,
            AsyncWait(keep=keep),
            *(c.rewrite(_id, sigma_consume) for c in bundle.body),
            AsyncWait(keep=keep),
        )

    main_extent = n - (pipeline_depth - 1)
    main_loop = SerialTile(
        axis=Axis(k_var, main_extent),
        body=Body(main_body_stmts),
        kind="serial_outer",
        unroll=kouter.unroll,
    )

    # Epilogue: drain the remaining N-1 chunks (k = main_extent .. n-1).
    epilogue: list[Stmt] = []
    for offset in range(pipeline_depth - 1):
        k_idx = main_extent + offset
        sigma_k = _sigma_at(k_idx)
        # After consuming k_idx, (pipeline_depth - 2 - offset) chunks remain
        # in flight; wait until only that many are still outstanding.
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
