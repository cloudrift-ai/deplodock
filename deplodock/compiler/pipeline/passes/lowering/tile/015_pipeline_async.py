"""Software-pipeline async cp.async loads across K-outer iterations.

Transforms a double-buffered, async-staged K-outer loop from sync-style
(load-wait-compute per iter) to pipelined form: prologue loads chunk 0,
the steady-state body issues chunk N+1 *while* computing chunk N
(overlapping DRAM with FMA), epilogue computes the last chunk after a
final drain.

Input shape::

    Loop(K_outer in 0..K, body=[
        Stage(W, async, buffer_count=2, phase=K_outer%2),
        Stage(X, async, buffer_count=2, phase=K_outer%2),
        reduce_loop reading slabs at phase=K_outer%2,
    ])

Output shape::

    # Prologue — issue chunk 0
    Stage(W, ..., pipelined=True, K_outer→0, phase=0)
    Stage(X, ..., pipelined=True, K_outer→0, phase=0)

    # Steady-state — issue chunk i+1 while computing chunk i
    Loop(K_outer in 0..K-1, body=[
        Stage(W, ..., pipelined=True, K_outer→K_outer+1, phase=(K_outer+1)%2),
        Stage(X, ..., pipelined=True, K_outer→K_outer+1, phase=(K_outer+1)%2),
        AsyncWait(remaining=len(stages)),  # leave just-issued S groups in flight
        reduce_loop reading slabs at phase=K_outer%2,
    ])

    # Epilogue — drain final loads, compute last chunk
    AsyncWait(remaining=0)
    reduce_loop with K_outer→K-1 substituted

The ``pipelined=True`` flag tells materialize to emit *only* the
cooperative load + commit (no wait, no Sync), since the explicit
``AsyncWait`` Stmt placed by this pass synchronizes at exactly the
right point. Materialize emits ``CpAsyncWait(remaining)`` + ``Sync()``
when it sees an ``AsyncWait`` Stmt.

Trigger conditions:

- All Stages in the K-outer body have ``async_load=True`` and
  ``buffer_count > 1``.
- None already pipelined (idempotence).
- Exactly one reduce Loop in the K-outer body (the K-inner reduce).
- ``K_outer.extent >= 2`` (need room for prologue + at least 1 main iter).
"""

from __future__ import annotations

from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Body, Loop, Stmt, Tile
from deplodock.compiler.ir.tile.ir import AsyncWait, Stage, TileOp
from deplodock.compiler.pipeline.engine import Pattern, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import is_matmul_k_outer, single_tile

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
        raise RuleSkipped("no eligible K-outer Loop with async-loaded Stages to pipeline")
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _process(body: Body) -> Body:
    new_body: list[Stmt] = []
    changed = False
    for s in body:
        if isinstance(s, Loop) and not s.is_reduce and _eligible(s):
            replacement = _pipeline(s)
            if replacement is not None:
                new_body.extend(replacement)
                changed = True
                continue
        new_body.append(s)
    return tuple(new_body) if changed else body


def _eligible(loop: Loop) -> bool:
    """K-outer loop with all body Stages async + double-buffered + not yet
    pipelined, and a *pure-matmul* reduce body — Load/Assign/Accum only
    AND no read of an SSA name not defined locally inside the reduce.
    The cross-loop-deps gate is critical here: kernels like SDPA's
    per-output free loop have a reduce body that reads ``acc0`` /
    ``acc1`` from prior softmax reduces; pipelining keeps the math
    right but compounding fp32 drift in the rearranged commits
    manifests as too-large diff vs eager.

    ``013_double_buffer`` keeps its own variant of this gate for the
    same fp32-drift reason."""

    def gate(k_outer: Loop, k_inner: Loop) -> bool:
        if int(k_outer.axis.extent) < 2:
            return False
        stages = [s for s in k_outer.body if isinstance(s, Stage)]
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
        if not all(s.async_load and s.buffer_count > 1 and not s.pipelined for s in stages):
            return False
        if len({s.buffer_count for s in stages}) != 1:
            return False
        # Cross-loop-deps gate: no read of a non-locally-defined SSA name
        # (excludes online-softmax fusions). See _eligible's docstring.
        # ``Body.deps_of(c)`` returns one resolved Stmt (or None for
        # external reads) per dep; we reject any None.
        return all(s is not None for c in k_inner.body for s in k_inner.body.deps_of(c))

    return is_matmul_k_outer(loop, extra_gate=gate)


def _pipeline(loop: Loop) -> list[Stmt] | None:
    n_chunks = int(loop.axis.extent)
    k_var = loop.axis.name
    stages = [s for s in loop.body if isinstance(s, Stage)]
    others = [s for s in loop.body if not isinstance(s, Stage)]

    sigma_first = Sigma({k_var: Literal(0, "int")})
    sigma_next = Sigma({k_var: Var(k_var) + Literal(1, "int")})
    sigma_last = Sigma({k_var: Literal(n_chunks - 1, "int")})

    def transform_stage(stage: Stage, sigma: Sigma) -> Stage:
        new_origin = tuple(sigma.apply(e) for e in stage.origin)
        new_template = tuple(sigma.apply(e) for e in stage.source_index_template) if stage.source_index_template is not None else None
        new_phase = sigma.apply(stage.phase) if stage.phase is not None else None
        return dc_replace(stage, origin=new_origin, source_index_template=new_template, phase=new_phase, pipelined=True)

    prologue: list[Stmt] = [transform_stage(s, sigma_first) for s in stages]

    body_stages = [transform_stage(s, sigma_next) for s in stages]
    # Each Stage commits its own group. Per iter we issue ``len(stages)``
    # commits. Steady-state ``wait_group`` should leave exactly the just-
    # issued groups in flight: compute on chunk N waits for chunk N's
    # loads (committed previously), while chunk N+1's loads (just issued)
    # stay outstanding.
    wait_remaining = len(stages)
    main_body = (*body_stages, AsyncWait(remaining=wait_remaining), *others)
    main_loop = Loop(axis=Axis(loop.axis.name, n_chunks - 1), body=main_body, unroll=loop.unroll)

    epilogue: list[Stmt] = [AsyncWait(remaining=0)]
    for s in others:
        epilogue.append(s.rewrite(_id, sigma_last))

    return [*prologue, main_loop, *epilogue]


def _id(name: str) -> str:
    return name
