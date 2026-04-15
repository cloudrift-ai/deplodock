"""Tests for LoopIR: dataclasses, pretty-printer, lowering, and round-trip codegen."""

import json

from deplodock.compiler.backend.cuda.schedule import GridSpec, Schedule
from deplodock.compiler.backend.ir.loop_ir import (
    Accum,
    Alloc,
    BinOp,
    Builtin,
    Guard,
    Let,
    Literal,
    Load,
    LoopNest,
    LoopProgram,
    OpCall,
    ParallelAxis,
    RawLoopOp,
    ShuffleReduce,
    Store,
    Var,
    pretty_print,
    to_dict,
)


def _mock_schedule(block_size=(256, 1, 1), tile_m=None, tile_n=None):
    """Build a minimal Schedule for tests that need block_size/tile info."""
    return Schedule(
        grid=GridSpec(block_size),
        tile_m=tile_m,
        tile_n=tile_n,
    )


# ---------------------------------------------------------------------------
# Unit tests: LoopIR construction and pretty-printing
# ---------------------------------------------------------------------------


def test_loop_program_construction():
    """Build a minimal LoopProgram and verify fields."""
    prog = LoopProgram(
        name="test_kernel",
        params=[("const float*", "X"), ("float*", "Y"), ("int", "n")],
        body=[
            ParallelAxis("i", "blockIdx.x", "n"),
            Guard(
                BinOp("<", Var("i"), Var("n")),
                [
                    Load("v", "X", [Var("i")], "global"),
                    Let("out", Var("v") * 2.0),
                    Store("Y", [Var("i")], Var("out"), "global"),
                ],
            ),
        ],
    )
    assert prog.name == "test_kernel"
    assert len(prog.params) == 3
    assert len(prog.body) == 2


def test_pretty_print_pointwise():
    """Pretty-print a pointwise program and check format."""
    prog = LoopProgram(
        name="relu_kernel",
        params=[("const float*", "X"), ("float*", "Y"), ("int", "n")],
        body=[
            ParallelAxis("i", "blockIdx.x", "n"),
            Guard(
                BinOp("<", Var("i"), Var("n")),
                [
                    Load("v", "X", [Var("i")], "global"),
                    Let("out", OpCall("relu", [Var("v")])),
                    Store("Y", [Var("i")], Var("out"), "global"),
                ],
            ),
        ],
    )
    sched = _mock_schedule()
    text = pretty_print(prog, schedule=sched)
    assert "loop_program relu_kernel" in text
    assert "block_size: (256, 1, 1)" in text
    assert "parallel i" in text
    assert "guard" in text
    assert "load v" in text
    assert "let float out = relu" in text
    assert "store Y" in text


def test_pretty_print_reduce():
    """Pretty-print a reduce program with ShuffleReduce."""
    prog = LoopProgram(
        name="sum_kernel",
        params=[("const float*", "X"), ("float*", "Y"), ("int", "rows"), ("int", "cols")],
        body=[
            ParallelAxis("row", "blockIdx.x", "rows"),
            Guard(
                BinOp("<", Var("row"), Var("rows")),
                [
                    Alloc("acc", "float", None, "reg", Literal(0.0)),
                    LoopNest(
                        "j",
                        Builtin("threadIdx.x"),
                        Var("cols"),
                        Builtin("blockDim.x"),
                        [
                            Load("v", "X", [Var("row"), Var("j")], "global"),
                            Accum("acc", "sum", Var("v")),
                        ],
                    ),
                    ShuffleReduce("acc", "sum"),
                ],
            ),
        ],
    )
    text = pretty_print(prog, schedule=_mock_schedule())
    assert "alloc reg float acc" in text
    assert "for j" in text
    assert "accum acc sum" in text
    assert "shuffle_reduce acc sum" in text


def test_pretty_print_contraction():
    """Pretty-print includes LoopNest for K-loop."""
    prog = LoopProgram(
        name="matmul",
        params=[("float*", "C")],
        body=[
            RawLoopOp("/* CTA swizzle grid setup */", "grid setup"),
            Alloc("c00", "float", None, "reg", Literal(0.0)),
            LoopNest(
                "k",
                Literal(0, "int"),
                Var("K"),
                None,
                [RawLoopOp("/* FMA body */", "K-loop body")],
            ),
        ],
    )
    sched = _mock_schedule(block_size=(32, 8, 1), tile_m=64, tile_n=128)
    text = pretty_print(prog, schedule=sched)
    assert "tile: (64, 128)" in text
    assert "for k" in text
    assert "raw" in text


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------


def test_loop_ir_json_roundtrip():
    """to_dict produces valid JSON."""
    prog = LoopProgram(
        name="test",
        params=[("float*", "X"), ("int", "n")],
        body=[
            ParallelAxis("i", "blockIdx.x", "n"),
            Guard(
                BinOp("<", Var("i"), Var("n")),
                [Store("X", [Var("i")], Literal(1.0), "global")],
            ),
        ],
    )
    d = to_dict(prog)
    # Must be JSON-serializable
    s = json.dumps(d)
    parsed = json.loads(s)
    assert parsed["name"] == "test"
    assert len(parsed["body"]) == 2
    assert parsed["body"][0]["type"] == "parallel_axis"
    assert parsed["body"][1]["type"] == "guard"
