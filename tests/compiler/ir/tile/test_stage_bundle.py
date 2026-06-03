"""Tests for the ``StageBundle`` IR node.

The wrap-body ``Stage`` hierarchy (``Stage`` / ``BufferedStage`` /
``AsyncBufferedStage`` / ``TmaBufferedStage`` / ``ComputeStage``) has been
fully collapsed onto ``StageBundle``: the bundle holds its transport
``sources`` directly (a homogeneous group of gmem operands loaded behind one
barrier), the consumer ``body``, an optional hoisted-invariant ``compute``
phase body, and the transport policy (``StagePolicy`` + ``buffer_count`` /
``phase`` / ``pipeline_depth``). There is no longer any ``Stage`` class.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Body, Load
from deplodock.compiler.ir.tile.ir import (
    CacheDim,
    Source,
    StageBundle,
    StagePolicy,
)


def _source(name: str, buf: str) -> Source:
    cache_dims = (
        CacheDim(axis=Axis(f"{name}_c0", 16), source_dim=0),
        CacheDim(axis=Axis(f"{name}_c1", 8), source_dim=1),
    )
    origin = (Var("k_outer") * Literal(16, "int"), Literal(0, "int"))
    return Source(name=name, buf=buf, cache_dims=cache_dims, origin=origin)


def _consumer_body() -> Body:
    return Body((Load(name="v", input="w_smem", index=(Var("w_smem_c0"), Var("w_smem_c1"))),))


def _sources_two() -> tuple[Source, ...]:
    return (_source("w_smem", "w"), _source("y_smem", "y"))


# ---------------------------------------------------------------------------
# StageBundle — SYNC policy (default)
# ---------------------------------------------------------------------------


def test_bundle_sync_default_construction():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    assert bundle.policy == StagePolicy.SYNC
    assert bundle.buffer_count == 1
    assert bundle.phase is None
    assert bundle.pipeline_depth == 1
    assert bundle.compute is None


def test_bundle_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one Source"):
        StageBundle(sources=(), body=Body(()))


def test_bundle_body_coerced():
    bundle = StageBundle(sources=_sources_two(), body=())  # type: ignore[arg-type]
    assert isinstance(bundle.body, Body)


def test_bundle_sync_rejects_buffer_count_gt_1():
    with pytest.raises(ValueError, match="SYNC: buffer_count must be 1"):
        StageBundle(sources=_sources_two(), body=Body(()), policy=StagePolicy.SYNC, buffer_count=2)


def test_bundle_sync_rejects_phase():
    with pytest.raises(ValueError, match="SYNC: phase must be None"):
        StageBundle(sources=_sources_two(), body=Body(()), policy=StagePolicy.SYNC, phase=Var("k") % Literal(2, "int"))


def test_bundle_sync_rejects_pipeline_depth_gt_1():
    with pytest.raises(ValueError, match="SYNC: pipeline_depth must be 1"):
        StageBundle(sources=_sources_two(), body=Body(()), policy=StagePolicy.SYNC, pipeline_depth=2)


# ---------------------------------------------------------------------------
# StageBundle — BUFFERED policy
# ---------------------------------------------------------------------------


def test_bundle_buffered_requires_phase_when_buffer_count_ge_2():
    with pytest.raises(ValueError, match="phase required when buffer_count >= 2"):
        StageBundle(sources=_sources_two(), body=Body(()), policy=StagePolicy.BUFFERED, buffer_count=2)


def test_bundle_buffered_with_phase_ok():
    phase = Var("k") % Literal(2, "int")
    bundle = StageBundle(
        sources=_sources_two(),
        body=_consumer_body(),
        policy=StagePolicy.BUFFERED,
        buffer_count=2,
        phase=phase,
    )
    assert bundle.policy == StagePolicy.BUFFERED
    assert bundle.buffer_count == 2
    assert bundle.phase == phase


# ---------------------------------------------------------------------------
# StageBundle — ASYNC / TMA policies
# ---------------------------------------------------------------------------


def test_bundle_async_with_pipeline_depth_ok():
    bundle = StageBundle(
        sources=_sources_two(),
        body=_consumer_body(),
        policy=StagePolicy.ASYNC,
        buffer_count=2,
        phase=Var("k") % Literal(2, "int"),
        pipeline_depth=3,
    )
    assert bundle.policy == StagePolicy.ASYNC
    assert bundle.pipeline_depth == 3


def test_bundle_pipeline_depth_gt_1_requires_async_or_tma():
    with pytest.raises(ValueError, match="pipeline_depth > 1 requires ASYNC or TMA"):
        StageBundle(
            sources=_sources_two(),
            body=Body(()),
            policy=StagePolicy.BUFFERED,
            buffer_count=2,
            phase=Var("k") % Literal(2, "int"),
            pipeline_depth=2,
        )


def test_bundle_tma_rejects_padded_source():
    src_padded = Source(
        name="w_smem",
        buf="w",
        cache_dims=(CacheDim(axis=Axis("c0", 16), source_dim=0),),
        origin=(Literal(0, "int"),),
        pad=(4,),
    )
    with pytest.raises(ValueError, match="TMA: source"):
        StageBundle(
            sources=(src_padded,),
            body=Body(()),
            policy=StagePolicy.TMA,
            buffer_count=2,
            phase=Var("k") % Literal(2, "int"),
        )


# ---------------------------------------------------------------------------
# Iteration / recursion
# ---------------------------------------------------------------------------


def test_nested_exposes_compute_and_body():
    """nested() is (compute or Body(()), body) — sources carry no body."""
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    nested = bundle.nested()
    assert len(nested) == 2
    assert nested[0] == Body(())  # no compute phase
    assert nested[1] == bundle.body


def test_nested_exposes_compute_phase():
    compute = Body((Load(name="ca", input="w_smem", index=(Var("i"),)),))
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body(), compute=compute)
    nested = bundle.nested()
    assert len(nested) == 2
    assert nested[0] == compute
    assert nested[1] == bundle.body


def test_compute_body_coerced():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body(), compute=())  # type: ignore[arg-type]
    assert isinstance(bundle.compute, Body)


def test_with_bodies_round_trips_compute():
    compute = Body((Load(name="ca", input="w_smem", index=(Var("i"),)),))
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body(), compute=compute)
    new_compute = Body((Load(name="ca2", input="w_smem", index=(Var("j"),)),))
    new = bundle.with_bodies((new_compute, bundle.body))
    assert new.compute == new_compute
    assert new.sources == bundle.sources


def test_with_bodies_empty_compute_collapses_to_none():
    """An empty leading body (the ``Body(())`` placeholder for an absent
    compute phase) collapses back to ``None`` so the structure round-trips."""
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    new = bundle.with_bodies((Body(()), bundle.body))
    assert new.compute is None


def test_body_iter_yields_bundle_and_consumer():
    """Body.iter() yields the bundle then descends its compute + consumer
    bodies (sources are not Stmts, so they aren't re-yielded)."""
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    outer = Body((bundle,))
    iterated = list(outer.iter())
    assert bundle in iterated
    # The consumer Load is reached via the body descent.
    assert any(isinstance(s, Load) for s in iterated)


def test_with_bodies_round_trip():
    bundle = StageBundle(
        sources=_sources_two(),
        body=_consumer_body(),
        policy=StagePolicy.BUFFERED,
        buffer_count=2,
        phase=Var("k") % Literal(2, "int"),
    )
    new_body = Body((Load(name="v2", input="z_smem", index=(Var("c"),)),))
    new_bundle = bundle.with_bodies((Body(()), new_body))
    assert isinstance(new_bundle, StageBundle)
    assert new_bundle.body == new_body
    assert new_bundle.sources == bundle.sources
    # Policy fields preserved.
    assert new_bundle.policy == StagePolicy.BUFFERED
    assert new_bundle.buffer_count == 2


def test_with_bodies_wrong_count_rejected():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    with pytest.raises(ValueError, match="expected 2 bodies"):
        bundle.with_bodies((Body(()),))


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_external_reads_lists_source_bufs():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    assert bundle.external_reads() == ("w", "y")


def test_local_decls_lists_source_names():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    assert bundle.local_decls() == ("w_smem", "y_smem")


def test_local_decls_includes_compute_output_slab():
    """The compute phase's Write output is a kernel-local smem slab, so it
    rides on local_decls (keeping it off the kernel signature)."""
    from deplodock.compiler.ir.stmt import Write

    compute = Body(
        (
            Load(name="ca", input="w_smem", index=(Var("i"),)),
            Write(output="fused", index=(Var("i"),), value="ca"),
        )
    )
    bundle = StageBundle(sources=_sources_two(), body=Body(()), compute=compute)
    assert bundle.local_decls() == ("w_smem", "y_smem", "fused")


def test_smem_bytes_sync_no_buffer_factor():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    expected = sum(s.smem_bytes for s in bundle.sources)
    assert bundle.smem_bytes == expected


def test_smem_bytes_buffered_includes_buffer_count_factor():
    phase = Var("k") % Literal(3, "int")
    bundle = StageBundle(
        sources=_sources_two(),
        body=_consumer_body(),
        policy=StagePolicy.BUFFERED,
        buffer_count=3,
        phase=phase,
    )
    expected = sum(s.smem_bytes for s in bundle.sources) * 3
    assert bundle.smem_bytes == expected


def test_exprs_includes_source_origins_and_bundle_phase():
    """Used by σ-substitution / dependency walkers."""
    phase = Var("k") % Literal(2, "int")
    bundle = StageBundle(
        sources=_sources_two(),
        body=Body(()),
        policy=StagePolicy.BUFFERED,
        buffer_count=2,
        phase=phase,
    )
    all_exprs = bundle.exprs()
    # Two sources × 2 origin exprs each = 4, plus bundle.phase = 5.
    assert len(all_exprs) == 5
    assert phase in all_exprs


# ---------------------------------------------------------------------------
# Pretty
# ---------------------------------------------------------------------------


def test_pretty_sync_header():
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body())
    lines = bundle.pretty(indent="  ")
    assert lines[0] == "  bundle sync:"
    for line in lines[1:]:
        assert line.startswith("    "), f"expected indented bundle child, got: {line!r}"


def test_pretty_buffered_header_shows_policy_and_phase():
    phase = Var("k_outer") % Literal(2, "int")
    bundle = StageBundle(
        sources=_sources_two(),
        body=_consumer_body(),
        policy=StagePolicy.BUFFERED,
        buffer_count=2,
        phase=phase,
    )
    lines = bundle.pretty(indent="")
    assert lines[0].startswith("bundle buffered[2@")


def test_pretty_tma_header_shows_depth():
    phase = Var("k") % Literal(2, "int")
    bundle = StageBundle(
        sources=_sources_two(),
        body=_consumer_body(),
        policy=StagePolicy.TMA,
        buffer_count=2,
        phase=phase,
        pipeline_depth=3,
    )
    header = bundle.pretty()[0]
    assert "tma[2@" in header
    assert "depth=3" in header


def test_pretty_renders_compute_phase():
    from deplodock.compiler.ir.stmt import Write

    compute = Body(
        (
            Load(name="ca", input="w_smem", index=(Var("i"),)),
            Write(output="fused", index=(Var("i"),), value="ca"),
        )
    )
    bundle = StageBundle(sources=_sources_two(), body=_consumer_body(), compute=compute)
    lines = bundle.pretty()
    assert any("cooperative:" in line for line in lines)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_source_bundle_is_valid():
    bundle = StageBundle(sources=(_source("w_smem", "w"),), body=Body(()))
    assert len(bundle.sources) == 1
