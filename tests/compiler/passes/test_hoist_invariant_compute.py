"""Tests for ``007b_hoist_invariant_compute`` — hoists an invariant
compute cone out of the K-inner reduce body, emitting one of two shapes
per the ``FUSED_PIPELINE`` knob.

The pass walks each Tile body produced by ``007_stage_inputs``, finds
groups of Stages with identical cache axes whose epilogue cone in the
reduce loop body has free axes contained in those cache axes, then:

- ``FUSED_PIPELINE=False`` (inline-fuse, default): folds the cone +
  gmem Loads of the source Stages into a single multi-source ``Stage``.
  Source transports absorbed; 1 smem buffer.
- ``FUSED_PIPELINE=True`` (hoist-compute): keeps source transports
  intact, adds a ``ComputeStage`` that reads their smem + runs the
  cone Assigns + writes its own smem. 1 + N smem buffers; transports
  remain single-source so 010/011/013 can promote them.

Both shapes collapse the reduce-body silu chain to a single Load.
"""

from __future__ import annotations

import importlib

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Tile, Write
from deplodock.compiler.ir.tile.ir import AffineAddressing, ComputeStage, Stage, TileOp, trivial_stage_body

_pass = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.007b_hoist_invariant_compute")


def _silu_mul_matmul_tile() -> TileOp:
    """Hand-build the Tile-IR shape that ``007_stage_inputs`` would
    emit for ``F.silu(gate) * up @ W``: two same-axes Stages on
    ``gate`` / ``up``, a third single-source Stage on ``w``, then a
    reduce Loop containing the silu chain + matmul accumulate."""
    m = Axis("m", 16)
    n = Axis("n", 16)
    k = Axis("k", 8)

    gate_smem = Stage(
        name="gate_smem",
        axes=(m, k),
        body=trivial_stage_body("gate_smem", "gate", (Literal(0, "int"), Literal(0, "int")), (m, k), AffineAddressing(dims=(0, 1))),
    )
    up_smem = Stage(
        name="up_smem",
        axes=(m, k),
        body=trivial_stage_body("up_smem", "up", (Literal(0, "int"), Literal(0, "int")), (m, k), AffineAddressing(dims=(0, 1))),
    )
    w_smem = Stage(
        name="w_smem",
        axes=(k, n),
        body=trivial_stage_body("w_smem", "w", (Literal(0, "int"), Literal(0, "int")), (k, n), AffineAddressing(dims=(0, 1))),
    )

    reduce_body = Body(
        (
            Load(name="g", input="gate_smem", index=(Var("m"), Var("k"))),
            Assign(name="s0", op=ElementwiseImpl("negative"), args=("g",)),
            Assign(name="s1", op=ElementwiseImpl("exp"), args=("s0",)),
            Assign(name="s2", op=ElementwiseImpl("add"), args=("one", "s1")),
            Assign(name="s3", op=ElementwiseImpl("reciprocal"), args=("s2",)),
            Assign(name="sig", op=ElementwiseImpl("multiply"), args=("g", "s3")),
            Load(name="u", input="up_smem", index=(Var("m"), Var("k"))),
            Assign(name="silu_up", op=ElementwiseImpl("multiply"), args=("u", "sig")),
            Load(name="ww", input="w_smem", index=(Var("k"), Var("n"))),
            Assign(name="prod", op=ElementwiseImpl("multiply"), args=("ww", "silu_up")),
            Accum(name="acc", value="prod", op=ElementwiseImpl("add")),
        )
    )
    reduce_loop = Loop(axis=k, body=reduce_body)  # is_reduce=True implied by Accum in body

    tile = Tile(
        axes=(BoundAxis(axis=m, bind=BIND_THREAD), BoundAxis(axis=n, bind=BIND_THREAD), BoundAxis(axis=Axis("blk", 1), bind=BIND_BLOCK)),
        body=(
            Init(name="acc", op=ElementwiseImpl("add"), dtype=F32),
            Init(name="one", op=ElementwiseImpl("add"), dtype=F32),  # placeholder; tracer puts a real const here
            gate_smem,
            up_smem,
            w_smem,
            reduce_loop,
            Write(output="out", index=(Var("m"), Var("n")), value="acc"),
        ),
    )
    return TileOp(body=(tile,), name="silu_mul_matmul")


# ---------------------------------------------------------------------------
# Inline-fuse shape (FUSED_PIPELINE=False — default)
# ---------------------------------------------------------------------------


def test_inline_fuse_collapses_silu_cone_into_producer_stage():
    op = _silu_mul_matmul_tile()
    new_body = _pass._maybe_rewrite(op.body, fused_pipeline=False)
    assert new_body is not None, "fusion should fire on silu·gate·matmul shape"
    new_op = TileOp(body=new_body, name=op.name)

    tile = new_op.body[0]
    stages_after = [s for s in tile.body if isinstance(s, Stage)]
    assert len(stages_after) == 2, [s.name for s in stages_after]

    fused = next(s for s in stages_after if "fused" in s.name)
    standalone = next(s for s in stages_after if "fused" not in s.name)
    assert standalone.name == "w_smem"
    # Inline-fuse: fused is a plain Stage (multi-source), NOT a ComputeStage.
    assert not isinstance(fused, ComputeStage)
    assert isinstance(fused, Stage)

    # Fused body carries both gmem source loads + silu compute.
    fused_inputs = {ld.input for ld in fused.source_loads}
    assert fused_inputs == {"gate", "up"}, fused_inputs
    body_assigns = [s for s in fused.body if isinstance(s, Assign)]
    op_names = [a.op.name for a in body_assigns]
    assert "negative" in op_names and "exp" in op_names and "reciprocal" in op_names
    last = fused.body[-1]
    assert isinstance(last, Write) and last.output == fused.name

    # Reduce body: silu chain is gone; only Loads from fused + w_smem
    # remain plus the accum chain.
    reduce_loop = next(s for s in tile.body if isinstance(s, Loop) and s.is_reduce)
    smem_loads = [s for s in reduce_loop.body if isinstance(s, Load)]
    sources = {ld.input for ld in smem_loads}
    assert sources == {fused.name, "w_smem"}, sources
    assert not any(isinstance(s, Assign) and s.op.name in {"negative", "exp", "reciprocal"} for s in reduce_loop.body)


# ---------------------------------------------------------------------------
# Hoist-compute shape (FUSED_PIPELINE=True)
# ---------------------------------------------------------------------------


def test_hoist_compute_keeps_transports_and_adds_compute_stage():
    op = _silu_mul_matmul_tile()
    new_body = _pass._maybe_rewrite(op.body, fused_pipeline=True)
    assert new_body is not None

    tile = new_body[0]
    stages_after = [s for s in tile.body if isinstance(s, Stage)]
    # Hoist-compute keeps gate_smem + up_smem + w_smem AND adds a
    # ComputeStage on top.
    transports = [s for s in stages_after if not isinstance(s, ComputeStage)]
    computes = [s for s in stages_after if isinstance(s, ComputeStage)]
    assert len(transports) == 3, [s.name for s in transports]
    assert len(computes) == 1, [s.name for s in computes]
    compute = computes[0]

    # Original transport names survive (no rename to A_xport / B_xport
    # like the legacy 007b-then-split chain produced).
    transport_names = {s.name for s in transports}
    assert transport_names == {"gate_smem", "up_smem", "w_smem"}, transport_names

    # ComputeStage's body Loads read from sibling Stage smem (cache-local),
    # not gmem.
    compute_load_srcs = {ld.input for ld in compute.source_loads}
    assert compute_load_srcs == {"gate_smem", "up_smem"}, compute_load_srcs

    # external_reads is empty (sibling-smem reads aren't external) —
    # critical for 015/tuning.py classification.
    assert compute.external_reads() == ()

    # Compute body still carries the silu chain + final multiply.
    assert any(isinstance(s, Assign) and s.op.name == "negative" for s in compute.body)
    assert any(isinstance(s, Assign) and s.op.name == "reciprocal" for s in compute.body)

    # Reduce body: silu chain is gone; only Loads from compute + w_smem
    # remain plus the accum chain.
    reduce_loop = next(s for s in tile.body if isinstance(s, Loop) and s.is_reduce)
    smem_loads = [s for s in reduce_loop.body if isinstance(s, Load)]
    sources = {ld.input for ld in smem_loads}
    assert sources == {compute.name, "w_smem"}, sources


# ---------------------------------------------------------------------------
# Cone analysis + skip behaviour (shared between shapes)
# ---------------------------------------------------------------------------


def test_no_cone_returns_none():
    """A tile with no group of >=2 same-cache-axes Stages should bail."""
    m = Axis("m", 8)
    n = Axis("n", 8)
    k = Axis("k", 4)
    a_smem = Stage(
        name="a_smem",
        axes=(m, k),
        body=trivial_stage_body("a_smem", "a", (Literal(0, "int"), Literal(0, "int")), (m, k), AffineAddressing(dims=(0, 1))),
    )
    # Only one Stage with (m, k) — no group of >= 2.
    reduce_loop = Loop(
        axis=k,
        body=Body(
            (
                Load(name="a", input="a_smem", index=(Var("m"), Var("k"))),
                Accum(name="acc", value="a", op=ElementwiseImpl("add")),
            )
        ),
    )
    tile = Tile(
        axes=(BoundAxis(axis=m, bind=BIND_THREAD), BoundAxis(axis=n, bind=BIND_THREAD)),
        body=(
            Init(name="acc", op=ElementwiseImpl("add"), dtype=F32),
            a_smem,
            reduce_loop,
            Write(output="out", index=(Var("m"),), value="acc"),
        ),
    )
    op = TileOp(body=(tile,), name="no_cone")
    assert _pass._maybe_rewrite(op.body, fused_pipeline=False) is None
    assert _pass._maybe_rewrite(op.body, fused_pipeline=True) is None


# ---------------------------------------------------------------------------
# Forking rewrite() returns both polarities by default
# ---------------------------------------------------------------------------


class _Node:
    """Lightweight Node stand-in — the rule only reads ``root.op``."""

    def __init__(self, op):
        self.op = op


def test_rewrite_emits_both_variants_by_default(monkeypatch):
    monkeypatch.delenv("DEPLODOCK_FUSED_PIPELINE", raising=False)
    op = _silu_mul_matmul_tile()
    variants = _pass.rewrite(_Node(op))
    assert isinstance(variants, list) and len(variants) == 2

    # First variant = FUSED_PIPELINE=False (inline-fuse) per default-seed
    # ordering; second = True (hoist-compute).
    v0, v1 = variants
    assert v0.knobs["FUSED_PIPELINE"] is False
    assert v1.knobs["FUSED_PIPELINE"] is True

    # Stage type check: v0's fused stage is a plain Stage; v1's is
    # a ComputeStage.
    v0_tile = v0.body[0]
    v0_stages = [s for s in v0_tile.body if isinstance(s, Stage)]
    assert not any(isinstance(s, ComputeStage) for s in v0_stages)
    v1_tile = v1.body[0]
    v1_computes = [s for s in v1_tile.body if isinstance(s, ComputeStage)]
    assert len(v1_computes) == 1


def test_rewrite_env_pins_to_hoist(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_FUSED_PIPELINE", "1")
    op = _silu_mul_matmul_tile()
    variants = _pass.rewrite(_Node(op))
    assert len(variants) == 1
    assert variants[0].knobs["FUSED_PIPELINE"] is True


def test_rewrite_env_pins_to_inline(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_FUSED_PIPELINE", "0")
    op = _silu_mul_matmul_tile()
    variants = _pass.rewrite(_Node(op))
    assert len(variants) == 1
    assert variants[0].knobs["FUSED_PIPELINE"] is False
