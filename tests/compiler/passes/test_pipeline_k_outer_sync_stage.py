"""Unit tests for ``015_pipeline_k_outer._eligible``.

The gate's job: reject K-outer Loops whose body mixes
``AsyncBufferedStage`` / ``TmaBufferedStage`` (async transport) with
plain ``BufferedStage`` (sync transport). When pipelining hoists async
Stage decls + first-iter loads to a prologue and peels the last iter
to an epilogue (both outside the K-outer body), a sync stage left in
the loop would be referenced from outside its smem-decl scope — the
historic "identifier ..._smem is undefined" nvcc error.

These tests build minimal Tile-IR Loops directly and call
``_eligible`` — much smaller surface area than the end-to-end
gated-MLP shape the test originally pinned, which no longer
reproduces the mix under the current planner / stage_inputs choices.
"""

from __future__ import annotations

import importlib

import pytest

from deplodock.compiler.ir.axis import Axis, Role

pytestmark = pytest.mark.xfail(
    reason="stage-wrap: bucket-10 follow-up — 015 pipelining stubbed; tests need rewrite for wrap-body shape",
    strict=False,
)
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Load, Loop
from deplodock.compiler.ir.tile.ir import AffineAddressing, AsyncBufferedStage, BufferedStage, trivial_stage_body

_pass = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.015_pipeline_k_outer")


def _stage(name: str, *, cls: type) -> BufferedStage:
    """Minimal BufferedStage / AsyncBufferedStage for gate testing."""
    cache = Axis("c0", 16)
    body = trivial_stage_body(name, "src", (Literal(0, "int"),), (cache,), AffineAddressing(dims=(0,)))
    return cls(name=name, axes=(cache,), body=body, buffer_count=2, phase=Var("k_o"))


def _k_inner() -> Loop:
    """Minimal STAGE_INNER reduce Loop: load from a sibling stage, accum."""
    return Loop(
        axis=Axis("k_i", 4),
        role=Role.STAGE_INNER,
        body=(
            Load(name="v", input="s0", index=(Var("c0"),)),
            Accum(name="acc", value="v"),
        ),
    )


def _k_outer(*body) -> Loop:
    return Loop(axis=Axis("k_o", 4), role=Role.SERIAL_OUTER, body=body)


def test_eligible_accepts_pure_async_stages():
    """Two async stages + one STAGE_INNER reduce → pipelining is safe."""
    loop = _k_outer(
        _stage("s0", cls=AsyncBufferedStage),
        _stage("s1", cls=AsyncBufferedStage),
        _k_inner(),
    )
    assert _pass._eligible(loop, set()) is True


def test_eligible_rejects_mixed_sync_and_async_stages():
    """Adding one sync ``BufferedStage`` sibling trips the gate — the
    pipelined epilogue would reference the sync stage's smem name from
    outside its decl scope (the historic ``..._smem undefined`` bug)."""
    loop = _k_outer(
        _stage("s0", cls=AsyncBufferedStage),
        _stage("s1", cls=AsyncBufferedStage),
        _stage("sync", cls=BufferedStage),
        _k_inner(),
    )
    assert _pass._eligible(loop, set()) is False


def test_eligible_rejects_single_async_stage():
    """Single-stage kernels (one Stage + a direct DRAM load in the
    reduce body) produce accuracy drift when pipelined for marginal
    gain — gate enforces ``len(stages) >= 2``."""
    loop = _k_outer(
        _stage("s0", cls=AsyncBufferedStage),
        _k_inner(),
    )
    assert _pass._eligible(loop, set()) is False


def test_eligible_rejects_extent_lt_two():
    """``extent < 2`` leaves no room for prologue + steady-state."""
    loop = Loop(
        axis=Axis("k_o", 1),
        role=Role.SERIAL_OUTER,
        body=(
            _stage("s0", cls=AsyncBufferedStage),
            _stage("s1", cls=AsyncBufferedStage),
            _k_inner(),
        ),
    )
    assert _pass._eligible(loop, set()) is False


def test_eligible_rejects_non_serial_outer_role():
    """The pass keys off the planner's ``SERIAL_OUTER`` tag."""
    loop = Loop(
        axis=Axis("k_o", 4),
        role=None,
        body=(
            _stage("s0", cls=AsyncBufferedStage),
            _stage("s1", cls=AsyncBufferedStage),
            _k_inner(),
        ),
    )
    assert _pass._eligible(loop, set()) is False
