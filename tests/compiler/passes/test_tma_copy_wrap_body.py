"""Tests for ``040_use_tma`` (single-source AsyncBufferedStage → TmaBufferedStage).

The pass walks for ``AsyncBufferedStage`` inside ``SerialTile(serial_outer)``
and promotes to ``TmaBufferedStage(pipeline_depth=1, swizzle=NONE)`` when
exactly one ``Source``, ``AffineAddressing``, contiguous-suffix dims, and
inner alignment requirements are met. All-or-nothing per tile: any
ineligible stage leaves the whole tile on cp.async.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Accum, Body, Load
from deplodock.compiler.ir.tile.ir import (
    AsyncBufferedStage,
    CacheDim,
    GridTile,
    SerialTile,
    Source,
    ThreadTile,
    TileOp,
    TmaBufferedStage,
)
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    p = pathlib.Path(_helpers.__file__).parent / "040_use_tma.py"
    spec = importlib.util.spec_from_file_location("tma_pass", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_matmul(m: int = 128, k: int = 256, n: int = 128) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _hand_build_single_source_async() -> tuple[TileOp, Graph]:
    """Construct a TileOp with a single-source AsyncBufferedStage to exercise
    the TMA promotion. Mirrors the post-stage_inputs/use_ring_buffers/use_async_copy
    shape for a single-input reduction kernel."""
    M, K_i, K_o = 16, 4, 8
    m_ax = Axis("m", M)
    ki_ax = Axis("k_i", K_i)
    ko_ax = Axis("k_o", K_o)
    src = Source(
        name="x_smem",
        buf="x",
        cache_dims=(CacheDim(axis=m_ax, source_dim=0), CacheDim(axis=ki_ax, source_dim=1)),
        origin=(Literal(0, "int"), Literal(0, "int")),
    )
    phase = Var(ko_ax.name) % Literal(2, "int")
    reduce = SerialTile(
        axis=ki_ax,
        body=Body(
            (
                Load(name="lx", input="x_smem", index=(phase, Var(m_ax.name), Var(ki_ax.name))),
                Accum(name="acc", value="lx", op=ElementwiseImpl("add"), dtype=None),
            )
        ),
        kind="stage_inner",
    )
    async_stage = AsyncBufferedStage(sources=(src,), body=Body((reduce,)), buffer_count=2, phase=phase, pipeline_depth=1)
    kouter = SerialTile(axis=ko_ax, body=Body((async_stage,)), kind="serial_outer")
    thread = ThreadTile(axes=(Axis("t", 32),), body=Body((kouter,)))
    grid = GridTile(axes=(Axis("g", 4),), body=Body((thread,)))
    op = TileOp(body=Body((grid,)), name="t", knobs={})

    # Stub graph so the pass can read source shapes.
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (M, K_i * K_o * 8)), node_id="x")
    return op, g


class _FakeNode:
    def __init__(self, op):
        self.op = op
        self.inputs = []
        self.outputs = ["t"]


class _FakeMatch:
    """Minimal Match stand-in — 011 only reads ``match.graph.nodes`` to
    look up source shapes."""

    def __init__(self, graph):
        self.graph = graph


# --- firing tests --------------------------------------------------------


def test_matmul_skips_tma_multi_source(recording_dump):
    """Matmul has multi-source AsyncBufferedStage (A + B); 011 must not fire."""
    g = _build_matmul()
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "use_tma" not in recording_dump.fired_rules("lowering/tile")


def test_single_source_async_promotes_to_tma():
    op, graph = _hand_build_single_source_async()
    mod = _load_pass()
    ctx = Context.from_target((9, 0))
    match = _FakeMatch(graph)
    new_op = mod.rewrite(ctx, match, _FakeNode(op))
    tmas = [s for s in new_op.body.iter() if isinstance(s, TmaBufferedStage)]
    assert len(tmas) == 1, tmas
    assert tmas[0].pipeline_depth == 1
    assert tmas[0].swizzle.value == "NONE"


def test_arch_below_sm90_rejected():
    op, graph = _hand_build_single_source_async()
    mod = _load_pass()
    ctx = Context.from_target((8, 0))
    match = _FakeMatch(graph)
    try:
        mod.rewrite(ctx, match, _FakeNode(op))
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "TMA must reject < sm_90"


def test_promotion_is_idempotent():
    op, graph = _hand_build_single_source_async()
    mod = _load_pass()
    ctx = Context.from_target((9, 0))
    match = _FakeMatch(graph)
    promoted = mod.rewrite(ctx, match, _FakeNode(op))
    try:
        mod.rewrite(ctx, match, _FakeNode(promoted))
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "011 should self-skip when the stage is already TmaBufferedStage"
