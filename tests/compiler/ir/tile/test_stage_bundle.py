"""Tests for the ``StageBundle`` IR node + the single ``Stage`` type.

Subclass hierarchy (``BufferedStage`` / ``AsyncBufferedStage`` /
``TmaBufferedStage`` / ``ComputeStage``) has been collapsed:
``StageBundle`` carries the transport policy (``StagePolicy``) and
policy-specific fields (``buffer_count`` / ``phase`` /
``pipeline_depth``). ``Stage`` carries ``sources`` plus an optional
``compute`` template body.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Body, Load
from deplodock.compiler.ir.tile.ir import (
    CacheDim,
    Source,
    Stage,
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


def _stages_two() -> tuple[Stage, ...]:
    return (Stage(sources=(_source("w_smem", "w"),)), Stage(sources=(_source("y_smem", "y"),)))


# ---------------------------------------------------------------------------
# Stage (single-class, optional compute)
# ---------------------------------------------------------------------------


def test_stage_requires_at_least_one_source():
    with pytest.raises(ValueError, match="at least one Source"):
        Stage(sources=())


def test_stage_default_no_compute():
    s = Stage(sources=(_source("w_smem", "w"),))
    assert s.compute is None
    assert s.nested() == ()
    assert s.external_reads() == ("w",)
    assert s.local_decls() == ("w_smem",)


def test_stage_compute_present_is_compute_stage():
    """A Stage with compute != None is the old ComputeStage semantically:
    sibling-smem → own-smem cooperative producer. external_reads() returns
    empty because sibling-smem reads aren't external."""
    s = Stage(
        sources=(_source("a_smem", "a"),),
        compute=Body((Load(name="ca", input="x_smem", index=(Var("i"),)),)),
    )
    assert s.compute is not None
    assert s.nested() == (s.compute,)
    assert s.external_reads() == ()  # sibling smem


def test_stage_compute_body_coerced():
    s = Stage(sources=(_source("a_smem", "a"),), compute=())  # type: ignore[arg-type]
    assert isinstance(s.compute, Body)


def test_stage_with_bodies_round_trip_compute():
    s = Stage(
        sources=(_source("a_smem", "a"),),
        compute=Body((Load(name="ca", input="x_smem", index=(Var("i"),)),)),
    )
    new_compute = Body((Load(name="ca2", input="x_smem", index=(Var("j"),)),))
    s2 = s.with_bodies((new_compute,))
    assert isinstance(s2, Stage)
    assert s2.compute == new_compute


def test_stage_with_bodies_no_compute_expects_empty():
    s = Stage(sources=(_source("w_smem", "w"),))
    # No compute → with_bodies expects no body args.
    assert s.with_bodies(()) is s
    with pytest.raises(ValueError):
        s.with_bodies((Body(()),))


# ---------------------------------------------------------------------------
# StageBundle — SYNC policy (default)
# ---------------------------------------------------------------------------


def test_bundle_sync_default_construction():
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    assert bundle.policy == StagePolicy.SYNC
    assert bundle.buffer_count == 1
    assert bundle.phase is None
    assert bundle.pipeline_depth == 1


def test_bundle_requires_at_least_one_stage():
    with pytest.raises(ValueError, match="at least one Stage"):
        StageBundle(stages=(), body=Body(()))


def test_bundle_rejects_non_stage_members():
    with pytest.raises(TypeError, match="must be a Stage"):
        StageBundle(stages=("not a stage",), body=Body(()))  # type: ignore[arg-type]


def test_bundle_body_coerced():
    bundle = StageBundle(stages=_stages_two(), body=())  # type: ignore[arg-type]
    assert isinstance(bundle.body, Body)


def test_bundle_sync_rejects_buffer_count_gt_1():
    with pytest.raises(ValueError, match="SYNC: buffer_count must be 1"):
        StageBundle(stages=_stages_two(), body=Body(()), policy=StagePolicy.SYNC, buffer_count=2)


def test_bundle_sync_rejects_phase():
    with pytest.raises(ValueError, match="SYNC: phase must be None"):
        StageBundle(stages=_stages_two(), body=Body(()), policy=StagePolicy.SYNC, phase=Var("k") % Literal(2, "int"))


def test_bundle_sync_rejects_pipeline_depth_gt_1():
    with pytest.raises(ValueError, match="SYNC: pipeline_depth must be 1"):
        StageBundle(stages=_stages_two(), body=Body(()), policy=StagePolicy.SYNC, pipeline_depth=2)


# ---------------------------------------------------------------------------
# StageBundle — BUFFERED policy
# ---------------------------------------------------------------------------


def test_bundle_buffered_requires_phase_when_buffer_count_ge_2():
    with pytest.raises(ValueError, match="phase required when buffer_count >= 2"):
        StageBundle(stages=_stages_two(), body=Body(()), policy=StagePolicy.BUFFERED, buffer_count=2)


def test_bundle_buffered_with_phase_ok():
    phase = Var("k") % Literal(2, "int")
    bundle = StageBundle(
        stages=_stages_two(),
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
        stages=_stages_two(),
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
            stages=_stages_two(),
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
            stages=(Stage(sources=(src_padded,)),),
            body=Body(()),
            policy=StagePolicy.TMA,
            buffer_count=2,
            phase=Var("k") % Literal(2, "int"),
        )


# ---------------------------------------------------------------------------
# Iteration / recursion
# ---------------------------------------------------------------------------


def test_nested_exposes_stages_as_synthetic_body():
    """Body.iter traverses into stages naturally via the synthetic body."""
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    nested = bundle.nested()
    assert len(nested) == 2
    assert isinstance(nested[0], Body)
    assert tuple(nested[0]) == bundle.stages
    assert nested[1] == bundle.body


def test_body_iter_yields_stages_inside_bundle():
    """The whole point of synthetic-body nested: Body.iter() sees stages
    via the generic descent without special-casing StageBundle."""
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    outer = Body((bundle,))
    iterated = list(outer.iter())
    # First yielded: bundle itself; then Body(stages) descent yields each Stage;
    # then bundle.body descent yields the Load.
    assert bundle in iterated
    assert any(isinstance(s, Stage) for s in iterated)
    stages_yielded = [s for s in iterated if isinstance(s, Stage)]
    assert stages_yielded == list(bundle.stages)


def test_with_bodies_round_trip():
    bundle = StageBundle(
        stages=_stages_two(),
        body=_consumer_body(),
        policy=StagePolicy.BUFFERED,
        buffer_count=2,
        phase=Var("k") % Literal(2, "int"),
    )
    new_stage = Stage(sources=(_source("z_smem", "z"),))
    new_body = Body((Load(name="v2", input="z_smem", index=(Var("c"),)),))
    new_bundle = bundle.with_bodies((Body((new_stage,)), new_body))
    assert isinstance(new_bundle, StageBundle)
    assert new_bundle.stages == (new_stage,)
    assert new_bundle.body == new_body
    # Policy fields preserved.
    assert new_bundle.policy == StagePolicy.BUFFERED
    assert new_bundle.buffer_count == 2


def test_with_bodies_wrong_count_rejected():
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    with pytest.raises(ValueError, match="expected 2 bodies"):
        bundle.with_bodies((Body(()),))


# ---------------------------------------------------------------------------
# Aggregates
# ---------------------------------------------------------------------------


def test_external_reads_concatenates_member_reads():
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    assert bundle.external_reads() == ("w", "y")


def test_external_reads_skips_compute_stage_members():
    """Stage members with compute != None contribute no external reads
    (they read from sibling smem)."""
    s_compute = Stage(sources=(_source("a_smem", "a"),), compute=Body(()))
    s_plain = Stage(sources=(_source("w_smem", "w"),))
    bundle = StageBundle(stages=(s_compute, s_plain), body=Body(()))
    assert bundle.external_reads() == ("w",)


def test_local_decls_concatenates_member_decls():
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    assert bundle.local_decls() == ("w_smem", "y_smem")


def test_smem_bytes_sync_no_buffer_factor():
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    expected = sum(s.smem_bytes for s in bundle.stages)
    assert bundle.smem_bytes == expected


def test_smem_bytes_buffered_includes_buffer_count_factor():
    phase = Var("k") % Literal(3, "int")
    bundle = StageBundle(
        stages=_stages_two(),
        body=_consumer_body(),
        policy=StagePolicy.BUFFERED,
        buffer_count=3,
        phase=phase,
    )
    expected = sum(s.smem_bytes for s in bundle.stages) * 3
    assert bundle.smem_bytes == expected


def test_exprs_includes_member_origins_and_bundle_phase():
    """Used by σ-substitution / dependency walkers."""
    phase = Var("k") % Literal(2, "int")
    bundle = StageBundle(
        stages=_stages_two(),
        body=Body(()),
        policy=StagePolicy.BUFFERED,
        buffer_count=2,
        phase=phase,
    )
    all_exprs = bundle.exprs()
    # Two stages × 2 origin exprs each = 4, plus bundle.phase = 5.
    assert len(all_exprs) == 5
    assert phase in all_exprs


# ---------------------------------------------------------------------------
# Pretty
# ---------------------------------------------------------------------------


def test_pretty_sync_header():
    bundle = StageBundle(stages=_stages_two(), body=_consumer_body())
    lines = bundle.pretty(indent="  ")
    assert lines[0] == "  bundle sync:"
    for line in lines[1:]:
        assert line.startswith("    "), f"expected indented bundle child, got: {line!r}"


def test_pretty_buffered_header_shows_policy_and_phase():
    phase = Var("k_outer") % Literal(2, "int")
    bundle = StageBundle(
        stages=_stages_two(),
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
        stages=_stages_two(),
        body=_consumer_body(),
        policy=StagePolicy.TMA,
        buffer_count=2,
        phase=phase,
        pipeline_depth=3,
    )
    header = bundle.pretty()[0]
    assert "tma[2@" in header
    assert "depth=3" in header


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_single_stage_bundle_is_valid():
    bundle = StageBundle(stages=(Stage(sources=(_source("w_smem", "w"),)),), body=Body(()))
    assert len(bundle.stages) == 1
