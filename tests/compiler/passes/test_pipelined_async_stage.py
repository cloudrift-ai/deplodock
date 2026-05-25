"""Tests for ``080_pipeline_stages`` (depth-2 pipelining).

The pass takes a ``SerialTile(serial_outer)`` whose body is a single
``AsyncBufferedStage(pipeline_depth=1)`` wrapping a ``stage_inner`` reduce
and rewrites it into prologue / steady-state / epilogue form. The
issue-only stages in prologue + steady-state carry ``pipeline_depth=2``
to suppress the implicit-wait-at-wrap-boundary path; explicit
``AsyncWait`` Stmts carry the schedule for the materializer to lower.
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
    AsyncWait,
    BufferedStage,
    CacheDim,
    GridTile,
    SerialTile,
    Source,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    p = pathlib.Path(_helpers.__file__).parent / "080_pipeline_stages.py"
    spec = importlib.util.spec_from_file_location("pipe_pass", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hand_build_async_kouter(extent: int = 4) -> TileOp:
    """Construct a TileOp with the shape that 015 fires on:
    SerialTile(K_o, serial_outer) → AsyncBufferedStage(depth=1) → K_i reduce."""
    m_ax = Axis("m", 16)
    ki_ax = Axis("k_i", 4)
    ko_ax = Axis("k_o", extent)
    src = Source(
        name="x_smem",
        buf="x",
        cache_dims=(CacheDim(axis=m_ax, source_dim=0), CacheDim(axis=ki_ax, source_dim=1)),
        origin=(Literal(0, "int"), Var(ko_ax.name) * Literal(4, "int")),
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
    return TileOp(body=Body((grid,)), name="t", knobs={})


class _FakeNode:
    def __init__(self, op):
        self.op = op
        self.inputs = []
        self.outputs = ["t"]


# --- firing tests --------------------------------------------------------


def test_pipelining_emits_prologue_main_epilogue():
    mod = _load_pass()
    op = _hand_build_async_kouter(extent=4)
    new_op = mod.rewrite(Context.from_target((8, 0)), _FakeNode(op))

    n_issue = sum(1 for s in new_op.body.iter() if isinstance(s, AsyncBufferedStage) and s.pipeline_depth == 2 and len(s.body) == 0)
    n_waits = sum(1 for s in new_op.body.iter() if isinstance(s, AsyncWait))
    assert n_issue == 2, f"expected 2 issue-only async stages (prologue + main), got {n_issue}"
    # 3 = main steady-state leading wait + trailing cross-iter sync wait + epilogue drain.
    # The trailing wait in the main loop is the slot-aliasing barrier required for
    # buffer_count=2 (see 080_pipeline_stages docstring).
    assert n_waits == 3, f"expected 3 AsyncWait stmts (main leading + main trailing sync + epilogue drain), got {n_waits}"


def test_main_loop_extent_decrements_by_one():
    mod = _load_pass()
    op = _hand_build_async_kouter(extent=4)
    new_op = mod.rewrite(Context.from_target((8, 0)), _FakeNode(op))

    serial_outers = [s for s in new_op.body.iter() if isinstance(s, SerialTile) and s.kind == "serial_outer"]
    # Exactly one new serial_outer survives (the steady-state main loop).
    assert len(serial_outers) == 1
    assert serial_outers[0].axis.extent.as_static() == 3, serial_outers[0].axis.extent


def test_idempotent_on_already_pipelined():
    mod = _load_pass()
    op = _hand_build_async_kouter(extent=4)
    ctx = Context.from_target((8, 0))
    pipelined = mod.rewrite(ctx, _FakeNode(op))
    try:
        mod.rewrite(ctx, _FakeNode(pipelined))
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "015 must skip on already-pipelined input"


def test_kouter_extent_below_two_rejected():
    mod = _load_pass()
    op = _hand_build_async_kouter(extent=1)
    try:
        mod.rewrite(Context.from_target((8, 0)), _FakeNode(op))
        raised = False
    except RuleSkipped:
        raised = True
    assert raised


# --- end-to-end with matmul ----------------------------------------------


def test_matmul_pipelines_through_full_pipeline(recording_dump, monkeypatch):
    # Pin the planner knobs to the priority_fn legacy primary so the
    # staged K_o tower (which pipeline_stages acts on) actually gets
    # emitted. Score-driven primary (post 7c321867) picks SPLITK>1 /
    # tiny-cells configs that skip the staged tower entirely.
    for knob, value in {"BM": "16", "BN": "16", "FM": "4", "FN": "8", "BK": "64", "SPLITK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{knob}", value)
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (128, 256)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (256, 128)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (128, 128)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g, ctx=Context.from_target((8, 0)))
    assert "pipeline_stages" in recording_dump.fired_rules("lowering/tile")


def test_smem_validate_dedupes_pipelined_sources():
    """TileOp.validate must dedupe by source name so the prologue + main
    issue-only stages (same source name, same allocation) don't get
    double-charged on the smem budget."""
    op = _hand_build_async_kouter(extent=4)
    mod = _load_pass()
    ctx = Context.from_target((8, 0))
    pipelined = mod.rewrite(ctx, _FakeNode(op))
    # Pipelined has 2 issue-only AsyncBufferedStages writing the same x_smem
    # buffer. Naïve sum would count 2× the allocation. validate(ctx) must
    # accept the variant on a 1KB smem budget that the single-stage shape
    # also fits.
    from dataclasses import replace

    # Source is 16 (m) × 4 (k_i) × 4B × 2 buffer = 512B. Pipelined has 2 stages
    # writing into the same buffer; if double-counted it'd be 1024B.
    tight = replace(ctx, max_dynamic_smem=600)
    assert pipelined.validate(tight), "validate must dedupe by source name for pipelined variants"


# --- structural regression: BufferedStage but not AsyncBufferedStage skipped


def test_plain_buffered_stage_not_pipelined():
    """010 produces BufferedStage; pipelining only fires after 013 has
    promoted it to AsyncBufferedStage. A plain BufferedStage in the
    K_o body must skip."""
    mod = _load_pass()
    op = _hand_build_async_kouter(extent=4)
    # Replace the AsyncBufferedStage with a plain BufferedStage.
    kouter = [s for s in op.body.iter() if isinstance(s, SerialTile) and s.kind == "serial_outer"][0]
    async_stage = kouter.body[0]
    plain = BufferedStage(
        sources=async_stage.sources,
        body=async_stage.body,
        buffer_count=async_stage.buffer_count,
        phase=async_stage.phase,
    )
    new_kouter_body = Body((plain,))
    new_kouter = SerialTile(axis=kouter.axis, body=new_kouter_body, kind="serial_outer")
    # Rebuild the TileOp

    new_thread = ThreadTile(axes=(Axis("t", 32),), body=Body((new_kouter,)))
    new_grid = GridTile(axes=(Axis("g", 4),), body=Body((new_thread,)))
    new_op = TileOp(body=Body((new_grid,)), name="t", knobs={})

    try:
        mod.rewrite(Context.from_target((8, 0)), _FakeNode(new_op))
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "plain BufferedStage must not be pipelined"
