"""Invariants for ``ComputeStage`` in the wrap-body shape.

A ``ComputeStage`` reads from sibling Stage smem (not gmem) via its
``compute`` body, and wraps a consumer body that reads its own staged
smem. ``external_reads`` returns ``()`` so 015 / tuning don't classify
compute stages' sibling-smem reads as gmem dependencies. Optional
``buffer_count`` + ``phase`` mirror ``BufferedStage`` for ring-buffering.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Write
from deplodock.compiler.ir.tile.ir import CacheDim, ComputeStage, Source, Stage


def _compute_stage(buffer_count: int = 1, phase=None) -> ComputeStage:
    """Representative compute stage: ``compute`` body reads two sibling
    Stage smems and writes into this stage's own smem; ``body`` (the
    consumer) is empty here."""
    name = "fused"
    cache_dims = (CacheDim(axis=Axis("m", 16), source_dim=0), CacheDim(axis=Axis("k", 8), source_dim=1))
    src = Source(
        name=name, buf="", cache_dims=cache_dims, origin=(Literal(0, "int"), Literal(0, "int"))
    )
    cache_index = (Var("m"), Var("k"))
    compute = Body(
        (
            Load(name="a_v", input="a_xport", index=cache_index),
            Load(name="b_v", input="b_xport", index=cache_index),
            Assign(name="prod", op="multiply", args=("a_v", "b_v")),
            Write(output=name, index=cache_index, value="prod"),
        )
    )
    return ComputeStage(sources=(src,), body=Body(()), compute=compute, buffer_count=buffer_count, phase=phase)


def test_compute_stage_is_a_stage_subclass():
    cs = _compute_stage()
    assert isinstance(cs, Stage)


def test_external_reads_is_empty():
    cs = _compute_stage()
    assert cs.external_reads() == ()


def test_nested_returns_compute_and_consumer_bodies():
    cs = _compute_stage()
    bodies = cs.nested()
    assert len(bodies) == 2
    assert bodies[0] is cs.compute
    assert bodies[1] is cs.body


def test_buffer_count_one_is_default():
    cs = _compute_stage()
    assert cs.buffer_count == 1
    assert cs.phase is None


def test_buffer_count_two_requires_phase():
    with pytest.raises(ValueError, match="phase required"):
        _compute_stage(buffer_count=2, phase=None)


def test_buffer_count_two_with_phase_doubles_smem():
    single = _compute_stage()
    buffered = _compute_stage(buffer_count=2, phase=Var("k_outer") % Literal(2, "int"))
    assert buffered.smem_bytes == single.smem_bytes * 2


def test_pretty_marks_as_compute():
    cs = _compute_stage()
    rendered = "\n".join(cs.pretty())
    assert "compute" in rendered


def test_pretty_marks_buffered_compute():
    cs = _compute_stage(buffer_count=2, phase=Var("k_outer") % Literal(2, "int"))
    rendered = "\n".join(cs.pretty())
    assert "compute buffers=2" in rendered
