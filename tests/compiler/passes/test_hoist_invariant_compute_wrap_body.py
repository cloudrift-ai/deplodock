"""Tests for ``030_hoist_invariant_compute`` (cone detection + Stage split).

Cone detection finds a multi-source ``Stage`` whose K_i reduce body has a
chain of Assigns reading a subset of sources' smem (the cone) and feeding a
single boundary SSA consumed downstream. The True polarity splits the Stage:
outer transport for non-cone sources, inner transport for cone sources,
sibling ``ComputeStage`` cooperatively populating a fused smem slab.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.context import Context
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load
from deplodock.compiler.ir.tile.ir import (
    CacheDim,
    ComputeStage,
    GridTile,
    SerialTile,
    Source,
    Stage,
    ThreadTile,
    TileOp,
)
from deplodock.compiler.pipeline import RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


def _load_pass():
    p = pathlib.Path(_helpers.__file__).parent / "030_hoist_invariant_compute.py"
    spec = importlib.util.spec_from_file_location("hoist_pass", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_silu_gated_tile() -> TileOp:
    """Build a TileOp matching the silu·gate·matmul wrap-body shape:
    multi-source Stage(a, b, w) wrapping K_i reduce with silu(a)*b*w → Accum."""
    M, K_i, N, K_o = 16, 4, 16, 8
    m_ax = Axis("m", M)
    ki_ax = Axis("k_i", K_i)
    n_ax = Axis("n", N)
    ko_ax = Axis("k_o", K_o)
    a_src = Source(
        name="a_smem",
        buf="a",
        cache_dims=(CacheDim(axis=m_ax, source_dim=0), CacheDim(axis=ki_ax, source_dim=1)),
        origin=(Literal(0, "int"), Literal(0, "int")),
    )
    b_src = Source(
        name="b_smem",
        buf="b",
        cache_dims=(CacheDim(axis=m_ax, source_dim=0), CacheDim(axis=ki_ax, source_dim=1)),
        origin=(Literal(0, "int"), Literal(0, "int")),
    )
    w_src = Source(
        name="w_smem",
        buf="w",
        cache_dims=(CacheDim(axis=ki_ax, source_dim=0), CacheDim(axis=n_ax, source_dim=1)),
        origin=(Literal(0, "int"), Literal(0, "int")),
    )
    reduce_body = Body(
        (
            Load(name="la", input="a_smem", index=(Var("m"), Var("k_i"))),
            Load(name="lb", input="b_smem", index=(Var("m"), Var("k_i"))),
            Load(name="lw", input="w_smem", index=(Var("k_i"), Var("n"))),
            Assign(name="sg", op=ElementwiseImpl("sigmoid"), args=("la",)),
            Assign(name="silu", op=ElementwiseImpl("multiply"), args=("la", "sg")),
            Assign(name="gated", op=ElementwiseImpl("multiply"), args=("silu", "lb")),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("gated", "lw")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), dtype=None),
        )
    )
    reduce = SerialTile(axis=ki_ax, body=reduce_body, kind="stage_inner")
    stage = Stage(sources=(a_src, b_src, w_src), body=Body((reduce,)))
    kouter = SerialTile(axis=ko_ax, body=Body((stage,)), kind="serial_outer")
    thread = ThreadTile(axes=(Axis("t", 32),), body=Body((kouter,)))
    grid = GridTile(axes=(Axis("g", 4),), body=Body((thread,)))
    return TileOp(body=Body((grid,)), name="t", knobs={})


def _build_plain_matmul_tile() -> TileOp:
    """No cone: just multiply A*B + accum."""
    M, K_i, N, K_o = 16, 4, 16, 8
    m_ax = Axis("m", M)
    ki_ax = Axis("k_i", K_i)
    n_ax = Axis("n", N)
    ko_ax = Axis("k_o", K_o)
    a_src = Source(
        name="a_smem",
        buf="a",
        cache_dims=(CacheDim(axis=m_ax, source_dim=0), CacheDim(axis=ki_ax, source_dim=1)),
        origin=(Literal(0, "int"), Literal(0, "int")),
    )
    b_src = Source(
        name="b_smem",
        buf="b",
        cache_dims=(CacheDim(axis=ki_ax, source_dim=0), CacheDim(axis=n_ax, source_dim=1)),
        origin=(Literal(0, "int"), Literal(0, "int")),
    )
    reduce_body = Body(
        (
            Load(name="la", input="a_smem", index=(Var("m"), Var("k_i"))),
            Load(name="lb", input="b_smem", index=(Var("k_i"), Var("n"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("la", "lb")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add"), dtype=None),
        )
    )
    reduce = SerialTile(axis=ki_ax, body=reduce_body, kind="stage_inner")
    stage = Stage(sources=(a_src, b_src), body=Body((reduce,)))
    kouter = SerialTile(axis=ko_ax, body=Body((stage,)), kind="serial_outer")
    thread = ThreadTile(axes=(Axis("t", 32),), body=Body((kouter,)))
    grid = GridTile(axes=(Axis("g", 4),), body=Body((thread,)))
    return TileOp(body=Body((grid,)), name="t", knobs={})


class _FakeNode:
    def __init__(self, op):
        self.op = op
        self.inputs = []
        self.outputs = ["t"]


# --- detection -----------------------------------------------------------


def test_silu_cone_detected():
    mod = _load_pass()
    op = _build_silu_gated_tile()
    target = mod._find_first_cone_target(op.body)
    assert target is not None
    assert target.cone_sources == frozenset({"a_smem", "b_smem"})
    # The boundary is the cone's final Assign — there should be exactly
    # one boundary SSA and it should resolve to an Assign in the reduce
    # body (TileOp normalization renames SSAs, so check shape, not name).
    assert isinstance(target.boundary_name, str)
    assert len(target.cone_assigns) >= 2  # silu + gated multiply


def test_plain_matmul_has_no_cone():
    mod = _load_pass()
    op = _build_plain_matmul_tile()
    target = mod._find_first_cone_target(op.body)
    # The multiply chain has only one Assign and it depends on BOTH
    # sources — not a strict subset.
    assert target is None


# --- emission ------------------------------------------------------------


def test_fork_emits_both_polarities():
    mod = _load_pass()
    op = _build_silu_gated_tile()
    ctx = Context.from_target((9, 0))
    variants = mod.rewrite(ctx, _FakeNode(op))
    assert len(variants) == 2
    polarities = [v.knobs["FUSED_PIPELINE"] for v in variants]
    assert polarities == [False, True], polarities


def test_true_polarity_splits_into_compute_stage(monkeypatch):
    """Under PAD_SMEM=true the True variant emits a transport+ComputeStage
    nest. Verify ComputeStage.compute carries the cone Assigns and the
    K_i reduce reads the fused smem instead of the raw cone sources."""
    monkeypatch.setenv("DEPLODOCK_FUSED_PIPELINE", "true")
    mod = _load_pass()
    op = _build_silu_gated_tile()
    ctx = Context.from_target((9, 0))
    (variant,) = mod.rewrite(ctx, _FakeNode(op))
    assert variant.knobs["FUSED_PIPELINE"] is True
    compute_stages = [s for s in variant.body.iter() if isinstance(s, ComputeStage)]
    assert len(compute_stages) == 1
    cs = compute_stages[0]
    assert len(cs.sources) == 1
    fused_name = cs.sources[0].name
    assert fused_name.endswith("_fused")
    # The compute body must include the silu + multiply Assigns + final
    # Write to fused (3 Assigns, one Write — TileOp may have renamed SSAs).
    assigns = [s for s in cs.compute if isinstance(s, Assign)]
    assert len(assigns) >= 2, assigns
    ops = {a.op.name for a in assigns}
    assert "sigmoid" in ops
    assert "multiply" in ops
    # The K_i reduce inside ComputeStage.body must read the fused smem.
    fused_loads = [ld for ld in cs.body.iter() if isinstance(ld, Load) and ld.input == fused_name]
    assert len(fused_loads) >= 1
    # And the cone Loads (la/lb on a_smem/b_smem) must be gone from the reduce.
    leftover_cone_loads = [ld for ld in cs.body.iter() if isinstance(ld, Load) and ld.input in {"a_smem", "b_smem"}]
    assert not leftover_cone_loads


# --- idempotence ---------------------------------------------------------


def test_hoist_is_idempotent():
    mod = _load_pass()
    op = _build_silu_gated_tile()
    ctx = Context.from_target((9, 0))
    (greedy, _) = mod.rewrite(ctx, _FakeNode(op))
    try:
        mod.rewrite(ctx, _FakeNode(greedy))
        raised = False
    except RuleSkipped:
        raised = True
    assert raised
