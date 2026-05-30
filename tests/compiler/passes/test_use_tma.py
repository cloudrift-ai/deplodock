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
    addressing on dims (0, 1), 128 B-aligned box/source inner (NVIDIA's
    recommended box-inner alignment — anything smaller would survive the
    hardware-minimum 16 B alignment but get padded to 128 B by
    ``100_materialize_tile``, and that pad mis-matches the contiguous
    box write at runtime), source rank ≤ 5, source inner ≥ 2× box
    inner."""
    return Source(
        name=name,
        buf=buf,
        cache_dims=(
            CacheDim(axis=Axis(f"{name}_m", 16), source_dim=0),
            CacheDim(axis=Axis(f"{name}_k", 32), source_dim=1),
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


# --- end-to-end staging-chain regression -----------------------------
#
# The unit tests above exercise ``050_use_tma`` in isolation against
# synthetic IR. They don't catch the regression class where the
# upstream rules (``020_stage_inputs`` produces a bundle,
# ``040_use_ring_buffers`` promotes SYNC → BUFFERED) silently stop
# firing on a real matmul shape — the bundle never reaches TMA, the
# rule SKIPS without an error, and ``060_use_async_copy`` quietly
# falls back to cp.async. The kernel still runs (and accuracy tests
# still pass), but the article-pin matmul we tuned to ~97% of cuBLAS
# slides back to ~78% silently. Pinning ``DEPLODOCK_USE_TMA=1``
# inverts that silent fall-back into a hard ``ValueError``, so the
# end-to-end "TMA fires" assertion becomes "the compile succeeds".


def _requires_cuda() -> bool:
    try:
        from ..conftest import requires_cuda  # noqa: PLC0415

        return requires_cuda.kwargs.get("condition", True)
    except Exception:  # noqa: BLE001
        return True


@pytest.mark.skipif(not _requires_cuda(), reason="needs CUDA backend")
def test_force_tma_succeeds_on_eligible_default_matmul(monkeypatch):
    """Positive end-to-end: a clean-divisor matmul with
    ``BK=32`` (128 B inner extent at fp32, meeting the
    ``050_use_tma`` alignment gate) compiles successfully with
    ``USE_TMA=1`` pinned. If ``020_stage_inputs`` stops producing a
    bundle for either operand, ``040_use_ring_buffers`` fails to
    promote SYNC → BUFFERED, or ``050_use_tma`` rejects the bundle
    as ineligible, the rule's pinned-mode raises ``ValueError`` and
    this test fails — making any silent regression in the staging
    chain visible at the exact link that broke."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from deplodock.compiler.ir.base import InputOp  # noqa: PLC0415
    from deplodock.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_USE_TMA", "1")
    monkeypatch.setenv("DEPLODOCK_BK", "32")
    M = K = N = 2048
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    # Compile must succeed — if any link of the staging chain skipped,
    # 050 would raise the pinned-mode ``ValueError``.
    CudaBackend().compile(g)


@pytest.mark.skipif(not _requires_cuda(), reason="needs CUDA backend")
def test_force_tma_errors_on_sub_aligned_inner_extent(monkeypatch):
    """Negative end-to-end: ``BK=16`` produces a 64 B inner extent at
    fp32, below the 128 B TMA-destination alignment gate
    (``100_materialize_tile`` rounds the slot inner up to 128 B for
    TMA, but the writer's natural contiguous box wouldn't fit, so
    ``050_use_tma`` rejects sub-aligned shapes at eligibility time).
    With ``USE_TMA=1`` pinned, the rule's hard-fail mode raises
    ``ValueError`` instead of silently falling back to cp.async —
    locking in that the eligibility gate is the right shape to
    differentiate the two transports."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from deplodock.compiler.ir.base import InputOp  # noqa: PLC0415
    from deplodock.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_USE_TMA", "1")
    monkeypatch.setenv("DEPLODOCK_BK", "16")
    M = K = N = 2048
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (M, N)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    with pytest.raises(ValueError, match="DEPLODOCK_USE_TMA=1 but TMA cannot fire.*not TMA-eligible"):
        CudaBackend().compile(g)


@pytest.mark.skipif(not _requires_cuda(), reason="needs CUDA backend")
def test_force_tma_errors_on_pointwise(monkeypatch):
    """Negative end-to-end: pointwise add has no K-loop, so
    ``020_stage_inputs`` doesn't produce a cooperative-load bundle.
    ``040_use_ring_buffers`` has nothing to promote and
    ``050_use_tma`` has no BUFFERED bundle to convert.
    ``USE_TMA=1`` then surfaces the "no eligible bundle" reason
    instead of silently lowering to a sync add. Pairs with the
    eligible-shape test above: together they guarantee the chain's
    on/off behaviour stays inverted from defaults exactly as the
    knob intends."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from deplodock.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from deplodock.compiler.ir.base import InputOp  # noqa: PLC0415
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp  # noqa: PLC0415

    monkeypatch.setenv("DEPLODOCK_USE_TMA", "1")
    N = 1024
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (N,)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (N,)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (N,)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    with pytest.raises(ValueError, match="DEPLODOCK_USE_TMA=1 but TMA cannot fire"):
        CudaBackend().compile(g)
