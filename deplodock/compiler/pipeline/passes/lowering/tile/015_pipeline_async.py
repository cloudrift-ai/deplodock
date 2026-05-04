"""Software-pipeline async cp.async loads across K-outer iterations.

Transforms a double-buffered, async-staged K-outer loop from sync-style
(load-wait-compute per iter) to pipelined form: prologue loads chunk 0,
the steady-state body issues chunk N+1 *while* computing chunk N
(overlapping DRAM with FMA), epilogue computes the last chunk after a
final drain.

Input shape (post 013 + 014)::

    Loop(K_outer in 0..K, body=[
        AsyncBufferedStage(W, buffer_count=2, phase=K_outer%2),
        AsyncWait(keep=0),                       # synchronous-style; dropped here
        AsyncBufferedStage(X, buffer_count=2, phase=K_outer%2),
        AsyncWait(keep=0),                       # synchronous-style; dropped here
        reduce_loop reading slabs at phase=K_outer%2,
    ])

Output shape::

    # Prologue — issue chunk 0
    AsyncBufferedStage(W, K_outer→0, phase=0)
    AsyncBufferedStage(X, K_outer→0, phase=0)

    # Steady-state — issue chunk i+1 while computing chunk i
    Loop(K_outer in 0..K-1, body=[
        AsyncBufferedStage(W, K_outer→K_outer+1, phase=(K_outer+1)%2),
        AsyncBufferedStage(X, K_outer→K_outer+1, phase=(K_outer+1)%2),
        AsyncWait(keep=len(stages)),  # leave just-issued chunk in flight
        reduce_loop reading slabs at phase=K_outer%2,
    ])

    # Epilogue — drain final loads, compute last chunk
    AsyncWait(keep=0)
    reduce_loop with K_outer→K-1 substituted

Materialize lowers each ``AsyncWait`` directly to
``CpAsyncWait(group=keep)``; ``keep`` is set explicitly here so the
schedule is robust to structural rewrites (e.g. unrolling a 1-iter
steady-state loop, which would otherwise merge prologue + body commits
into the same lexical scope).

Trigger conditions:

- All staged loads in the K-outer body are ``AsyncBufferedStage``.
- Exactly one reduce Loop in the K-outer body (the K-inner reduce).
- ``K_outer.extent >= 2`` (need room for prologue + at least 1 main iter).
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Expr, Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import AsyncBufferedStage, AsyncWait, BufferedStage, TileOp, TmaBufferedStage
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import collect_invariant_names, is_matmul_k_outer, single_tile

PATTERN = [Pattern("root", TileOp)]


def rewrite(graph: Graph, root: Node) -> Graph | None:
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        return None
    root.op = TileOp(body=new_body, name=root.op.name)
    return None


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _process(tile.body)
    if new_tile_body is tile.body or new_tile_body == tile.body:
        raise RuleSkipped("no eligible K-outer Loop with AsyncBufferedStage loads to pipeline")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(body: Body) -> Body:
    new_body: list[Stmt] = []
    changed = False
    invariant_names: set[str] = set()
    for s in body:
        if isinstance(s, Loop) and not s.is_reduce and _eligible(s, invariant_names):
            replacement = _pipeline(s)
            if replacement is not None:
                new_body.extend(replacement)
                changed = True
                invariant_names.update(collect_invariant_names(s))
                continue
        new_body.append(s)
        invariant_names.update(collect_invariant_names(s))
    return tuple(new_body) if changed else body


def _eligible(loop: Loop, invariant_names: set[str]) -> bool:
    """K-outer loop with all body staged loads as ``AsyncBufferedStage``
    (and a *pure-matmul* reduce body — Load/Assign/Accum only AND no
    read of an SSA name not defined locally inside the reduce, except
    when that name is loop-invariant w.r.t. the K-outer).

    The cross-loop-deps gate originally rejected *all* reads of
    non-locally-defined SSA names: kernels like SDPA's per-output
    free loop have a reduce body that reads ``acc0`` / ``acc1`` from
    prior softmax reduces, and pipelining was suspected to compound
    fp32 drift in the rearranged commits.

    The relaxation: a cross-loop name is allowed when it lives in
    ``invariant_names`` — names defined by sibling stmts above the
    K-outer at the same scope. Those values are finalized before the
    K-outer starts, so the steady-state can read them from registers
    without phase-dependent reordering. The original Accum-running-
    value rejection (``Body.deps_of`` returning an in-body ``Accum``
    for a non-Accum stmt) still holds via 013's ``extra_gate`` and is
    the precise pattern the drift comment was about.

    ``013_double_buffer`` runs the same relaxed gate."""

    def gate(k_outer: Loop, k_inner: Loop) -> bool:
        if int(k_outer.axis.extent) < 2:
            return False
        stages = [s for s in k_outer.body if isinstance(s, (AsyncBufferedStage, TmaBufferedStage))]
        # Restrict to ≥ 2 Stages (the matmul-shape case). Single-stage
        # kernels — typically a Stage + a separate non-staged DRAM Load
        # in the reduce body (e.g. ``k_add_3`` with the SDPA-reduce
        # input) — produce noticeable accuracy drift when pipelined,
        # likely because the unstaged DRAM Load hits memory at different
        # times relative to the cp.async batch ordering. Empirically,
        # the gain on these is small and the drift compounds across
        # ~10 chained kernels in a transformer block.
        if len(stages) < 2:
            return False
        if len({s.buffer_count for s in stages}) != 1:
            return False
        # Cross-loop-deps gate, relaxed: reject only when an unresolved
        # dep is *not* in ``invariant_names``.
        for c in k_inner.body:
            for d, defstmt in zip(c.deps(), k_inner.body.deps_of(c), strict=False):
                if defstmt is None and d not in invariant_names:
                    return False
        return True

    return is_matmul_k_outer(loop, extra_gate=gate)


def _pipeline(loop: Loop) -> list[Stmt] | None:
    n_chunks = int(loop.axis.extent)
    k_var = loop.axis.name
    # Drop synchronous-style ``AsyncWait`` stmts inserted by 014; the
    # pipelined schedule re-emits its own waits at the correct positions.
    stages = [s for s in loop.body if isinstance(s, (AsyncBufferedStage, TmaBufferedStage))]
    others = [s for s in loop.body if not isinstance(s, (AsyncBufferedStage, TmaBufferedStage, AsyncWait))]

    sigma_first = Sigma({k_var: Literal(0, "int")})
    sigma_next = Sigma({k_var: Var(k_var) + Literal(1, "int")})
    sigma_last = Sigma({k_var: Literal(n_chunks - 1, "int")})

    def transform_stage(stage: BufferedStage, sigma: Sigma) -> BufferedStage:
        return stage.rewrite(_id, sigma)  # type: ignore[return-value]

    prologue: list[Stmt] = [transform_stage(s, sigma_first) for s in stages]

    body_stages = [transform_stage(s, sigma_next) for s in stages]
    # TMA waits need the consumer-side ring-buffer phase (the slot we're
    # about to read), distinct from the issuance-time phase the stage was
    # rewritten with by ``sigma_next``. cp.async stages don't use ``phase``
    # — their waits lower to ``CpAsyncWait(keep)+Sync``. Carry phase only
    # when any TMA stage is present so cp.async paths stay unchanged.
    has_tma = any(isinstance(s, TmaBufferedStage) for s in stages)
    buffer_count = stages[0].buffer_count if stages else 2
    body_phase: Expr | None = None
    epi_phase: Expr | None = None
    body_slot: Expr | None = None
    epi_slot: Expr | None = None
    if has_tma:
        # Per-slot mbarrier semantics: each ring slot has its own mbar that
        # alternates parity 0,1,0,1... over its successive uses. At iter k,
        # the consumer reads slot ``k % buffer_count`` and that slot has
        # been used ``k / buffer_count`` times before, so the parity to
        # test is ``(k / buffer_count) % 2``.
        body_slot = Var(k_var) % Literal(buffer_count, "int")
        body_phase = (Var(k_var) / Literal(buffer_count, "int")) % Literal(2, "int")
        last_k = n_chunks - 1
        epi_slot = Literal(last_k % buffer_count, "int")
        epi_phase = Literal((last_k // buffer_count) % 2, "int")
    # Each AsyncBufferedStage commits its own group. Per iter we issue
    # ``len(stages)`` commits. The steady-state wait must leave exactly
    # the just-issued chunk's groups in flight: compute on chunk N waits
    # for chunk N's loads (committed previously) while chunk N+1's loads
    # (just issued) stay outstanding. ``keep`` is carried explicitly so
    # the wait remains correct after structural rewrites such as
    # unrolling a 1-iter steady-state loop (which would otherwise merge
    # prologue + body commits into the same scope).
    if has_tma:
        # TMA pipelining requires WAIT-then-PREFETCH ordering: each iter
        # waits on the slot it's about to consume (filled in a prior
        # iteration / prologue), consumes, then prefetches the next slot
        # for a future iteration. This matches the standard NVIDIA
        # mbarrier-based ring buffer pattern. Issuing prefetch before
        # wait would arrive on a slot's mbarrier whose previous tx may
        # not yet be drained (the consumer hasn't waited), causing
        # non-deterministic data-race-style failures observed under
        # heavy K-loop pipelining.
        main_body = (
            AsyncWait(keep=len(stages), phase=body_phase, slot=body_slot),
            *others,
            *body_stages,
        )
    else:
        main_body = (*body_stages, AsyncWait(keep=len(stages)), *others)
    main_loop = Loop(axis=Axis(loop.axis.name, n_chunks - 1), body=main_body, unroll=loop.unroll)

    epilogue: list[Stmt] = [AsyncWait(keep=0, phase=epi_phase, slot=epi_slot)]
    for s in others:
        epilogue.append(s.rewrite(_id, sigma_last))

    return [*prologue, main_loop, *epilogue]


def _id(name: str) -> str:
    return name
