"""Tests for ``007a_fuse_stage_epilogue`` — the silu·gate·matmul fuser.

The pass walks each Tile body produced by ``007_stage_inputs``, finds
groups of Stages with identical cache axes whose epilogue cone in the
reduce loop body has free axes contained in those cache axes, and
folds the cone into a single multi-source ``Stage`` with a non-trivial
body. The materializer's body-walk path then emits the fused compute
inside the cooperative load.
"""

from __future__ import annotations

import importlib

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.axis import BIND_BLOCK, BIND_THREAD, Axis, BoundAxis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Init, Load, Loop, Tile, Write
from deplodock.compiler.ir.tile.ir import AffineAddressing, Stage, TileOp, trivial_stage_body

_pass = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.007a_fuse_stage_epilogue")


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


def test_fusion_collapses_silu_cone_into_producer_stage():
    op = _silu_mul_matmul_tile()
    # Drive the pass via its internal helper so we don't have to wire a
    # full Graph + Node — the Graph wrapper only adds bookkeeping that
    # this test doesn't care about.
    new_body = _pass._maybe_rewrite(op.body)
    assert new_body is not None, "fusion should fire on silu·gate·matmul shape"
    new_op = TileOp(body=new_body, name=op.name)

    # Locate the (now sole) M·K cache-axis Stage — gate_smem and up_smem
    # should have collapsed into one fused stage; w_smem stays standalone.
    tile = new_op.body[0]
    stages_after = [s for s in tile.body if isinstance(s, Stage)]
    assert len(stages_after) == 2, [s.name for s in stages_after]

    fused = next(s for s in stages_after if "fused" in s.name)
    standalone = next(s for s in stages_after if "fused" not in s.name)
    assert standalone.name == "w_smem"

    # Fused stage carries both gmem source loads + the silu compute.
    fused_inputs = {ld.input for ld in fused.source_loads}
    assert fused_inputs == {"gate", "up"}, fused_inputs
    body_assigns = [s for s in fused.body if isinstance(s, Assign)]
    op_names = [a.op.name for a in body_assigns]
    # silu chain (negative, exp, add, reciprocal, multiply) + the
    # final multiply with up's value.
    assert "negative" in op_names and "exp" in op_names and "reciprocal" in op_names
    # Final stmt is a Write into the fused stage's smem buffer.
    last = fused.body[-1]
    assert isinstance(last, Write) and last.output == fused.name

    # Reduce body: silu chain is gone; only one Load from the fused
    # stage remains (plus the w_smem Load and the accum chain).
    reduce_loop = next(s for s in tile.body if isinstance(s, Loop) and s.is_reduce)
    smem_loads = [s for s in reduce_loop.body if isinstance(s, Load)]
    sources = {ld.input for ld in smem_loads}
    assert sources == {fused.name, "w_smem"}, sources
    assert not any(isinstance(s, Assign) and s.op.name in {"negative", "exp", "reciprocal"} for s in reduce_loop.body)
