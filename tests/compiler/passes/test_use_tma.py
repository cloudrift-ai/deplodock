"""Tests for ``050_use_tma`` — the BUFFERED/ASYNC → TMA promoter.

Pre-fix, the rule looked only at ``policy == ASYNC`` bundles, but the
pipeline rule order (050 < 060_use_async_copy) means ASYNC doesn't yet
exist when 050 runs and the cursor doesn't re-fire scans on Op rebinds.
Post-fix the rule also matches ``policy == BUFFERED`` (the actual
post-040_use_ring_buffers state) so it fires on the real input shape.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.ir import (
    CacheDim,
    SerialTile,
    Source,
    Stage,
    StageBundle,
    StagePolicy,
    SwizzleMode,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.tensor import Tensor

_use_tma = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.050_use_tma")


def _ctx() -> Context:
    return Context(compute_capability=(9, 0))


def _eligible_source(*, name: str, buf: str) -> Source:
    """A Source meeting every TMA-eligibility check: 2D affine
    addressing on dims (0, 1), 16 B-aligned box/source inner, source
    rank ≤ 5, source inner ≥ 2× box inner."""
    return Source(
        name=name,
        buf=buf,
        cache_dims=(
            CacheDim(axis=Axis(f"{name}_m", 16), source_dim=0),
            CacheDim(axis=Axis(f"{name}_k", 16), source_dim=1),
        ),
        origin=(Var("k_outer") * Literal(16, "int"), Literal(0, "int")),
    )


def _tile_op_buffered(*, policy: StagePolicy, buffer_count: int = 2) -> tuple[TileOp, Graph]:
    """Build a small TileOp with one BUFFERED (or ASYNC) StageBundle
    inside a SerialTile(serial_outer), plus a graph carrying the
    matching input shape so the eligibility check (which reads
    ``match.graph.nodes[buf].output.shape``) passes."""
    src = _eligible_source(name="a_smem", buf="a")
    # SYNC bundles use phase=None + buffer_count=1; buffered policies
    # carry a non-None phase + buffer_count >= 2.
    if policy == StagePolicy.SYNC:
        bundle = StageBundle(
            stages=(Stage(sources=(src,)),),
            body=Body(()),
            policy=policy,
            buffer_count=1,
            phase=None,
            pipeline_depth=1,
            swizzle=SwizzleMode.NONE,
        )
    else:
        bundle = StageBundle(
            stages=(Stage(sources=(src,)),),
            body=Body(()),
            policy=policy,
            buffer_count=buffer_count,
            phase=Var("k_outer") % Literal(buffer_count, "int"),
            pipeline_depth=1,
            swizzle=SwizzleMode.NONE,
        )
    k_outer = Axis("k_outer", 8)
    outer = SerialTile(axis=k_outer, body=Body((bundle,)), kind="serial_outer")
    body = Body((ThreadTile(axes=(Axis("t", 128),), body=Body((outer,))),))
    op = TileOp(body=body, name="k_tma_candidate")

    g = Graph()
    g.add_node(op=type("StubOp", (), {"__init__": lambda self: None})(), inputs=[], output=Tensor("a", (128, 128)), node_id="a")
    g.add_node(op=op, inputs=["a"], output=Tensor(op.name, ()), node_id="op")
    return op, g


def _run_rule(op: TileOp, g: Graph):
    from deplodock.compiler.pipeline.pipeline import Match  # noqa: PLC0415

    node = g.nodes["op"]
    # Build a minimal Match — the rule only reads ``match.graph`` for
    # the static-shape probe; no need to populate the full match tree.
    match = Match(graph=g, nodes={"root": "op"}, consumed=set(), root_node_id="op", pipeline=None, rule=None, is_last=True)  # type: ignore[arg-type]
    return _use_tma.rewrite(_ctx(), match, node)


# --- eligibility-positive cases --------------------------------------


def test_rule_fires_on_buffered_eligible():
    """Post-fix headline behavior: a BUFFERED StageBundle whose lone
    Stage passes ``_stage_eligible`` is promoted to TMA in place. This
    is the path that was dead pre-fix (rule only matched ASYNC,
    pre-promotion BUFFERED bundles fell through to RuleSkipped)."""
    op, g = _tile_op_buffered(policy=StagePolicy.BUFFERED)
    result = _run_rule(op, g)
    assert isinstance(result, TileOp), f"expected TileOp, got {type(result).__name__}"
    # Walk the body and confirm the bundle is now TMA.
    bundles = [s for s in result.body.iter() if isinstance(s, StageBundle)]
    assert len(bundles) == 1
    assert bundles[0].policy == StagePolicy.TMA, f"bundle policy is {bundles[0].policy}, expected TMA"
    # Buffering / phase / pipeline_depth survive the policy swap.
    assert bundles[0].buffer_count == 2
    assert bundles[0].pipeline_depth == 1
    assert bundles[0].swizzle == SwizzleMode.NONE


def test_rule_fires_on_async_eligible():
    """ASYNC StageBundles also promote — the rule needs to handle both
    so it's idempotent w.r.t. when in the pipeline order it runs."""
    op, g = _tile_op_buffered(policy=StagePolicy.ASYNC)
    result = _run_rule(op, g)
    assert isinstance(result, TileOp)
    bundles = [s for s in result.body.iter() if isinstance(s, StageBundle)]
    assert bundles[0].policy == StagePolicy.TMA


# --- eligibility-negative cases (RuleSkipped) ------------------------


def test_skipped_on_pre_hopper_target():
    op, g = _tile_op_buffered(policy=StagePolicy.BUFFERED)
    from deplodock.compiler.pipeline.pipeline import Match  # noqa: PLC0415

    match = Match(graph=g, nodes={"root": "op"}, consumed=set(), root_node_id="op", pipeline=None, rule=None, is_last=True)  # type: ignore[arg-type]
    with pytest.raises(RuleSkipped, match="TMA requires compute capability"):
        _use_tma.rewrite(Context(compute_capability=(8, 0)), match, g.nodes["op"])


def test_skipped_when_no_promotable_bundle():
    """SYNC bundles aren't in ``_PROMOTABLE`` — rule skips."""
    op, g = _tile_op_buffered(policy=StagePolicy.SYNC, buffer_count=1)
    with pytest.raises(RuleSkipped, match="no BUFFERED/ASYNC StageBundle"):
        _run_rule(op, g)


def test_idempotent_when_already_tma():
    """Re-running on a TileOp whose bundle is already TMA: nothing to
    promote, rule skips cleanly (no double-promote, no error)."""
    op, g = _tile_op_buffered(policy=StagePolicy.TMA)
    with pytest.raises(RuleSkipped, match="no BUFFERED/ASYNC StageBundle"):
        _run_rule(op, g)
