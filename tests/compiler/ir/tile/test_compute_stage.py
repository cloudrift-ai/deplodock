"""Invariants for ``ComputeStage`` — the hoist-compute-side counterpart
to ``Stage`` (transport).

A ``ComputeStage`` reads from sibling Stage smem (not gmem), so the
gmem-slab-geometry properties (``buf`` / ``origin`` / ``addressing``)
are nonsensical and raise; ``external_reads`` returns ``()`` so 015 and
``tuning.py`` don't misclassify a compute stage's body Loads as gmem
dependencies. Optional ``buffer_count`` + ``phase`` mirror
``BufferedStage`` so 010 can ring-buffer the compute output without a
separate subclass.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Assign, Body, Load, Write
from deplodock.compiler.ir.tile.ir import ComputeStage, Stage


def _compute_stage(buffer_count: int = 1, phase=None) -> ComputeStage:
    """Build a representative compute stage: smem Loads on two sibling
    Stages (``a_xport`` / ``b_xport``), an elementwise multiply, then
    write into the compute stage's own smem buffer."""
    name = "fused"
    axes = (Axis("m", 16), Axis("k", 8))
    cache_index = (Var("m"), Var("k"))
    body = Body(
        (
            Load(name="a_v", input="a_xport", index=cache_index),
            Load(name="b_v", input="b_xport", index=cache_index),
            Assign(name="prod", op="multiply", args=("a_v", "b_v")),
            Write(output=name, index=cache_index, value="prod"),
        )
    )
    return ComputeStage(name=name, axes=axes, body=body, buffer_count=buffer_count, phase=phase)


def test_compute_stage_is_a_stage_subclass():
    cs = _compute_stage()
    assert isinstance(cs, Stage)


def test_external_reads_is_empty():
    # Critical invariant: sibling smem reads are NOT external. If this
    # returned ("a_xport", "b_xport"), 015's TMA/async eligibility checks
    # and tuning.py's gmem enumeration would misclassify them as gmem
    # dependencies and miscompile.
    cs = _compute_stage()
    assert cs.external_reads() == ()


def test_source_loads_still_returns_body_loads():
    # source_loads is unchanged — the body has two Loads (from sibling
    # smems). Distinct from external_reads, which filters to gmem only.
    cs = _compute_stage()
    assert len(cs.source_loads) == 2
    assert {ld.input for ld in cs.source_loads} == {"a_xport", "b_xport"}


def test_primary_load_raises():
    cs = _compute_stage()
    with pytest.raises(ValueError, match="primary_load undefined"):
        _ = cs.primary_load


def test_buf_origin_addressing_raise():
    cs = _compute_stage()
    with pytest.raises(ValueError, match="buf undefined"):
        _ = cs.buf
    with pytest.raises(ValueError, match="origin undefined"):
        _ = cs.origin
    with pytest.raises(ValueError, match="addressing undefined"):
        _ = cs.addressing


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
    # Multi-source rendering header from Stage.pretty still applies
    # (shows the sibling smem source names).
    assert "fuse[a_xport, b_xport]" in rendered


def test_pretty_marks_buffered_compute():
    cs = _compute_stage(buffer_count=2, phase=Var("k_outer") % Literal(2, "int"))
    rendered = "\n".join(cs.pretty())
    assert "compute buffers=2" in rendered
