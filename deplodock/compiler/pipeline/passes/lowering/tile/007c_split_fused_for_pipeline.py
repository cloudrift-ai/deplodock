"""Split fused multi-source Stages into transport + compute stages so 015 can pipeline.

After ``007b_fuse_stage_epilogue`` runs, multi-source fused stages carry
both the gmem source Loads AND the producer compute (silu chain etc.) in
a single ``Stage.body``. ``015_pipeline_k_outer`` software-pipelines by
σ-shifting Stages and placing cloned-with-K_outer→K_outer+1 issues
ahead of the K_inner reduce — but it only operates on
``BufferedStage`` / ``TmaBufferedStage`` (single-source, transport-only)
because the σ-shift on a fused body would also shift the silu compute,
which has to stay at the original K_outer position.

This pass exposes the transport / compute boundary as separate Tile-IR
Stmts so the existing pipelining pass can do its job unchanged. For each
fused Stage in a free outer Loop's body::

    fused = Stage(fuse[A, B], axes=(M, K), body=[
        a_v = Load A[…]              # gmem Load
        b_v = Load B[…]              # gmem Load
        … silu / mul …               # Assigns
        Write fused[M,K] = result    # Write to fused smem
    ])

is rewritten to::

    A_xport = Stage(A, axes=(M, K))                       # transport
    B_xport = Stage(B, axes=(M, K))                       # transport
    fused   = Stage(fuse[A_xport, B_xport], axes=(M, K), body=[
        a_v = Load A_xport[M, K]      # smem Load (cache-local index)
        b_v = Load B_xport[M, K]
        … silu / mul …                # unchanged
        Write fused[M, K] = result
    ])

Now ``A_xport`` / ``B_xport`` are single-source Stages eligible for the
existing ``010 → 011 / 013 → 015`` chain (BufferedStage / TmaBufferedStage
promotion + K-outer pipelining). The compute Stage stays as a plain
multi-source ``Stage`` — its body reads from the transport stages' smem
buffers, runs the elementwise chain, writes to the original fused-smem
buffer. Because the compute Stage is multi-source it bypasses 010/011/013
(those guards already check ``len(s.source_loads) == 1``) and falls
naturally into 015's ``others`` list, where it's placed *between* the
wait and the K_inner reduce in the steady-state body — exactly the slot
the silu chain needs to land in.

Conditional firing: the split is only profitable when 015 will actually
pipeline, which requires the fused Stage to live inside a free outer
Loop with extent ≥ 2. Otherwise the split adds a smem round-trip
(scratch → fused) for no overlap benefit, so we leave the fused Stage
intact.
"""

from __future__ import annotations

import os
from dataclasses import replace as dc_replace

from deplodock.compiler.graph import Graph, Node
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Loop, Stmt, Tile, Write
from deplodock.compiler.ir.tile.ir import Stage, TileOp, trivial_stage_body
from deplodock.compiler.pipeline import Pattern, RuleSkipped
from deplodock.compiler.pipeline.knob import Knob, KnobType
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import single_tile

PATTERN = [Pattern("root", TileOp)]

FUSED_PIPELINE = Knob(
    "FUSED_PIPELINE",
    KnobType.BOOL,
    hints=(True, False),
    help=(
        "Split fused multi-source Stage into per-source transport + compute stages so "
        "015_pipeline_k_outer can overlap K+1 TMAs with K compute. Win on Hopper sm_90+ "
        "with TMA; on sm_120 (cp.async only) the extra smem scratch buffers eat occupancy "
        "and typically wash out the hide. Default off until per-recipe autotuning picks."
    ),
)


def _enabled() -> bool:
    """Opt-in via the ``FUSED_PIPELINE`` knob (env override ``DEPLODOCK_FUSED_PIPELINE``)."""
    return os.environ.get(FUSED_PIPELINE.env, "0") in ("1", "true", "True")


def rewrite(root: Node) -> Graph | None:
    if not _enabled():
        raise RuleSkipped(f"{FUSED_PIPELINE.env} not set")
    new_body = _maybe_rewrite(root.op.body)
    if new_body is None:
        raise RuleSkipped("no fused Stage to split for pipelining")
    return TileOp(body=new_body, name=root.op.name)


def _maybe_rewrite(body: Body) -> Body | None:
    idx, tile = single_tile(body)
    new_tile_body = _split_in_scope(tile.body)
    if new_tile_body is None:
        return None
    return body[:idx] + (Tile(axes=tile.axes, body=new_tile_body),) + body[idx + 1 :]


def _split_in_scope(scope_body: Body) -> Body | None:
    """Walk the scope; for each free outer Loop containing fused Stages
    immediately followed by a reduce Loop, split each fused Stage."""
    out: list[Stmt] = []
    changed = False
    for s in scope_body:
        if isinstance(s, Loop) and not s.is_reduce and int(s.axis.extent) >= 2:
            new_inner = _split_loop_body(s.body)
            if new_inner is not None:
                out.append(dc_replace(s, body=new_inner))
                changed = True
                continue
        out.append(s)
    return Body(out) if changed else None


def _split_loop_body(loop_body: Body) -> Body | None:
    """Inside a free K_outer Loop body, split each fused Stage into
    transport + compute. Leave non-fused stages and other stmts alone."""
    # Collect sibling stage names first so we can skip compute Stages
    # whose sources are already-split transport stages (their body Loads
    # read from sibling stage smem, not gmem — splitting again would
    # recurse on our own output).
    sibling_stage_names = {s.name for s in loop_body if isinstance(s, Stage)}

    out: list[Stmt] = []
    changed = False
    for s in loop_body:
        if isinstance(s, Stage) and len(s.source_loads) > 1 and _has_compute(s.body) and not _sources_are_siblings(s, sibling_stage_names):
            transports, compute = _split_fused_stage(s)
            out.extend(transports)
            out.append(compute)
            changed = True
        else:
            out.append(s)
    return Body(out) if changed else None


def _sources_are_siblings(stage: Stage, sibling_names: set[str]) -> bool:
    """True iff every source Load reads from a sibling Stage's smem
    (i.e. the stage is already a compute-side stage produced by a prior
    split). Such stages don't need further splitting — their inputs are
    not gmem latency to overlap."""
    return all(src.input in sibling_names for src in stage.source_loads)


def _has_compute(body: Body) -> bool:
    """A fused Stage with no Assigns is nominally multi-source but its
    body is just (Load A; Load B; Write fused = ?). Splitting wouldn't
    move any compute — skip."""
    return any(isinstance(stmt, Assign) for stmt in body)


def _split_fused_stage(fused: Stage) -> tuple[list[Stage], Stage]:
    """Decompose ``fused`` into per-source transport Stages + a compute
    Stage. The compute Stage keeps the original ``fused.name`` so
    consumers (the reduce body's Loads on ``fused.name``) don't need
    rewriting."""
    cache_axes = fused.axes
    cache_index = tuple(Var(ax.name) for ax in cache_axes)

    # Per-source transport Stages — one per body Load. Each is single-
    # source with a trivial Load→Write body, eligible for the standard
    # 010 / 011 / 013 promotion chain.
    transports: list[Stage] = []
    src_load_to_xport: dict[str, str] = {}  # body-Load SSA name → xport stage smem name
    for src_load in fused.source_loads:
        xport_name = f"{fused.name}__{src_load.input}_xport"
        # Reuse trivial_stage_body so the addressing classification works.
        # Origin/addressing are derived from the existing source Load's
        # index against the cache axes.
        transport = Stage(
            name=xport_name,
            axes=cache_axes,
            body=Body(
                (
                    Load(name=f"{xport_name}__src", input=src_load.input, index=src_load.index),
                    Write(output=xport_name, index=cache_index, value=f"{xport_name}__src"),
                )
            ),
            pad=fused.pad,
        )
        transports.append(transport)
        src_load_to_xport[src_load.name] = xport_name

    # Compute Stage. Its body re-reads from the per-source scratch smems
    # (cache-local index) and runs the original Assigns + final Write.
    compute_body_stmts: list[Stmt] = []
    # Replace each gmem source Load with a smem Load on the corresponding
    # transport stage's output; index becomes cache-local (the body's
    # later Assigns reference the same SSA name, so we keep the name).
    for s in fused.body:
        if isinstance(s, Load) and s.name in src_load_to_xport:
            xport_name = src_load_to_xport[s.name]
            compute_body_stmts.append(Load(name=s.name, input=xport_name, index=cache_index))
        elif isinstance(s, Load):
            # Body Load on something else (rare; keep verbatim).
            compute_body_stmts.append(s)
        else:
            # Assign / Write — copied verbatim, SSA names already resolved.
            compute_body_stmts.append(s)

    compute = Stage(
        name=fused.name,
        axes=cache_axes,
        body=Body(tuple(compute_body_stmts)),
        pad=fused.pad,
    )
    return transports, compute


# Keep the import alive (unused below but useful in tests / future
# helper additions); ``trivial_stage_body`` is the canonical builder
# the 007a fuser uses, so referencing it documents the relationship.
_ = trivial_stage_body
