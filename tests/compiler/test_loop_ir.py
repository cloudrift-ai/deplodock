"""Tests for LoopIR: dataclasses, pretty-printer, lowering, and round-trip codegen."""

import json

from deplodock.compiler.backend.cuda.generators import analyze, build_schedule, lower_generic, lower_to_loop_ir
from deplodock.compiler.backend.cuda.generators.loop_codegen import loop_ir_to_kernel
from deplodock.compiler.backend.cuda.schedule import GridSpec, Schedule
from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel
from deplodock.compiler.backend.ir.loop_ir import (
    Accum,
    AccumInit,
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
from deplodock.compiler.ops import ElementwiseOp, ReduceOp
from tests.compiler._kernel_builder import build_kernel


def _mock_schedule(block_size=(256, 1, 1), tile_m=None, tile_n=None):
    """Build a minimal Schedule for tests that need block_size/tile info."""
    return Schedule(
        grid=GridSpec("1d", block_size),
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


# ---------------------------------------------------------------------------
# Structure tests: verify LoopIR shape for each pattern
# ---------------------------------------------------------------------------


def _make_region_and_analysis(region_ops, input_names, output_names, shapes):
    """Build KernelOp + TileAnalysis."""
    region = build_kernel(
        region_ops=region_ops,
        input_names=input_names,
        output_names=output_names,
        shapes=shapes,
    )
    analysis = analyze(region, shapes)
    return region, analysis


def _find_ops(body, op_type, recursive=True):
    """Find all LoopOps of a given type in a body, optionally recursing."""
    found = []
    for op in body:
        if isinstance(op, op_type):
            found.append(op)
        if recursive:
            if isinstance(op, (LoopNest, Guard)):
                found.extend(_find_ops(op.body, op_type, recursive))
    return found


def test_pointwise_structure():
    """Pointwise LoopIR has: ParallelAxis + Guard with Compute + Store."""
    n = 256
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("neg", ElementwiseOp("neg"), ["x"]),
            ("exp", ElementwiseOp("exp"), ["neg"]),
        ],
        input_names=["x"],
        output_names=["exp"],
        shapes={"x": (n,), "neg": (n,), "exp": (n,)},
    )
    prog, sched = lower_to_loop_ir(region, "test_pw", {"x": (n,), "neg": (n,), "exp": (n,)}, analysis)

    assert sched.grid.block_size == (256, 1, 1)
    assert any(isinstance(op, ParallelAxis) for op in prog.body)
    guards = _find_ops(prog.body, Guard, recursive=False)
    assert len(guards) == 1
    lets = _find_ops(guards[0].body, Let)
    stores = _find_ops(guards[0].body, Store)
    assert len(lets) >= 1  # elementwise ops as Let(name, OpCall(...))
    assert len(stores) == 1


def test_single_reduce_structure():
    """Row reduce has: ParallelAxis + Alloc + LoopNest + ShuffleReduce."""
    rows, cols = 4, 8
    region, analysis = _make_region_and_analysis(
        region_ops=[("red", ReduceOp("sum", axis=1), ["x"])],
        input_names=["x"],
        output_names=["red"],
        shapes={"x": (rows, cols), "red": (rows,)},
    )
    prog, sched = lower_to_loop_ir(region, "test_red", {"x": (rows, cols), "red": (rows,)}, analysis)

    assert sched.grid.block_size == (256, 1, 1)
    accum_inits = _find_ops(prog.body, AccumInit)
    assert any(a.name.startswith("acc_") for a in accum_inits)
    loops = _find_ops(prog.body, LoopNest)
    assert len(loops) >= 1
    reduces = _find_ops(prog.body, ShuffleReduce)
    assert len(reduces) == 1
    assert reduces[0].op == "sum"


def test_multi_reduce_structure():
    """Multi-reduce (softmax) has multiple ShuffleReduce passes."""
    rows, cols = 4, 8
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("mx", ReduceOp("max", axis=1), ["x"]),
            ("sub", ElementwiseOp("sub"), ["x", "mx"]),
            ("exp", ElementwiseOp("exp"), ["sub"]),
            ("sm", ReduceOp("sum", axis=1), ["exp"]),
            ("div", ElementwiseOp("div"), ["exp", "sm"]),
        ],
        input_names=["x"],
        output_names=["div"],
        shapes={
            "x": (rows, cols),
            "mx": (rows, 1),
            "sub": (rows, cols),
            "exp": (rows, cols),
            "sm": (rows, 1),
            "div": (rows, cols),
        },
    )
    prog, _sched = lower_to_loop_ir(
        region,
        "test_softmax",
        {"x": (rows, cols), "mx": (rows, 1), "sub": (rows, cols), "exp": (rows, cols), "sm": (rows, 1), "div": (rows, cols)},
        analysis,
    )

    reduces = _find_ops(prog.body, ShuffleReduce)
    assert len(reduces) == 2  # max + sum
    assert reduces[0].op == "max"
    assert reduces[1].op == "sum"

    loops = _find_ops(prog.body, LoopNest)
    assert len(loops) >= 3  # max pass + sum pass + epilogue pass


def test_contraction_structure():
    """Contraction has register tile alloc and K-loop."""
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("ew", ElementwiseOp("mul"), ["A", "B"]),
            ("C", ReduceOp("sum", axis=1), ["ew"]),
        ],
        input_names=["A", "B"],
        output_names=["C"],
        shapes={"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)},
    )
    prog, sched = lower_to_loop_ir(
        region,
        "test_mm",
        {"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)},
        analysis,
        strategy="naive",
    )

    assert sched.grid.block_size == (32, 8, 1)
    # Grid setup uses Let ops, K-loop is LoopNest, write is Guard+Store.
    lets = _find_ops(prog.body, Let, recursive=False)
    assert any(v.name == "bm" for v in lets)  # CTA-swizzle output
    assert any(v.name == "tr" for v in lets)  # thread row offset
    allocs = _find_ops(prog.body, Alloc, recursive=False)
    assert any(a.name == "c" and a.shape == (8, 4) for a in allocs)  # register array
    loops = _find_ops(prog.body, LoopNest, recursive=False)
    assert any(ln.var == "k" for ln in loops)  # K-loop
    guards = _find_ops(prog.body, Guard, recursive=False)
    assert len(guards) >= 1  # write guards + early return
    stores = _find_ops(prog.body, Store)
    assert len(stores) >= 1


# ---------------------------------------------------------------------------
# Round-trip tests: lower_to_loop_ir → loop_ir_to_kernel → emit_kernel
# ---------------------------------------------------------------------------


def _roundtrip(region, name, shapes, analysis, **kwargs):
    """Lower to LoopIR, codegen to KernelDef, emit to CUDA source."""
    prog, sched = lower_to_loop_ir(region, name, shapes, analysis, **kwargs)
    kernel = loop_ir_to_kernel(prog, sched)
    source = emit_kernel(kernel)
    return prog, kernel, source, sched


def test_roundtrip_pointwise():
    """SiLU pointwise round-trips to valid CUDA."""
    n = 256
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("neg", ElementwiseOp("neg"), ["gate"]),
            ("exp", ElementwiseOp("exp"), ["neg"]),
            ("add", ElementwiseOp("add"), ["one", "exp"]),
            ("recip", ElementwiseOp("recip"), ["add"]),
            ("out", ElementwiseOp("mul"), ["gate", "recip"]),
        ],
        input_names=["gate", "one"],
        output_names=["out"],
        shapes={"gate": (n,), "one": (1,), "neg": (n,), "exp": (n,), "add": (n,), "recip": (n,), "out": (n,)},
    )
    shapes = {"gate": (n,), "one": (1,), "neg": (n,), "exp": (n,), "add": (n,), "recip": (n,), "out": (n,)}
    _, kernel, source, _ = _roundtrip(region, "silu", shapes, analysis)

    assert "__global__" in source
    assert "void silu" in source
    assert "expf" in source
    assert kernel.block_size == (256, 1, 1)


def test_roundtrip_row_reduce():
    """Row sum round-trips to valid CUDA with warp shuffle."""
    rows, cols = 8, 64
    region, analysis = _make_region_and_analysis(
        region_ops=[("red", ReduceOp("sum", axis=1), ["x"])],
        input_names=["x"],
        output_names=["red"],
        shapes={"x": (rows, cols), "red": (rows,)},
    )
    _, kernel, source, _ = _roundtrip(region, "row_sum", {"x": (rows, cols), "red": (rows,)}, analysis)

    assert "__global__" in source
    assert "void row_sum" in source
    assert "__shfl_down_sync" in source
    assert "+=" in source


def test_roundtrip_rmsnorm():
    """RMSNorm (reduce_broadcast) round-trips to valid CUDA."""
    rows, dim = 4, 8
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("sq", ElementwiseOp("mul"), ["x", "x"]),
            ("red", ReduceOp("sum", axis=1), ["sq"]),
            ("add_eps", ElementwiseOp("add"), ["red", "eps"]),
            ("rsq", ElementwiseOp("rsqrt"), ["add_eps"]),
            ("norm", ElementwiseOp("mul"), ["x", "rsq"]),
            ("out", ElementwiseOp("mul"), ["norm", "w"]),
        ],
        input_names=["x", "eps", "w"],
        output_names=["out"],
        shapes={
            "x": (rows, dim),
            "eps": (1,),
            "w": (dim,),
            "sq": (rows, dim),
            "red": (rows, 1),
            "add_eps": (rows, 1),
            "rsq": (rows, 1),
            "norm": (rows, dim),
            "out": (rows, dim),
        },
    )
    shapes = {
        "x": (rows, dim),
        "eps": (1,),
        "w": (dim,),
        "sq": (rows, dim),
        "red": (rows, 1),
        "add_eps": (rows, 1),
        "rsq": (rows, 1),
        "norm": (rows, dim),
        "out": (rows, dim),
    }
    _, _, source, _ = _roundtrip(region, "rmsnorm", shapes, analysis)

    assert "rsqrtf" in source
    assert "__shfl_down_sync" in source


def test_roundtrip_softmax():
    """Softmax (multi-reduce) round-trips to valid CUDA."""
    rows, cols = 4, 8
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("mx", ReduceOp("max", axis=1), ["x"]),
            ("sub", ElementwiseOp("sub"), ["x", "mx"]),
            ("exp", ElementwiseOp("exp"), ["sub"]),
            ("sm", ReduceOp("sum", axis=1), ["exp"]),
            ("div", ElementwiseOp("div"), ["exp", "sm"]),
        ],
        input_names=["x"],
        output_names=["div"],
        shapes={
            "x": (rows, cols),
            "mx": (rows, 1),
            "sub": (rows, cols),
            "exp": (rows, cols),
            "sm": (rows, 1),
            "div": (rows, cols),
        },
    )
    shapes = {
        "x": (rows, cols),
        "mx": (rows, 1),
        "sub": (rows, cols),
        "exp": (rows, cols),
        "sm": (rows, 1),
        "div": (rows, cols),
    }
    _, _, source, _ = _roundtrip(region, "softmax", shapes, analysis)

    assert "fmaxf" in source  # max reduce
    assert "expf" in source  # exp
    # Two warp reduces (max + sum) → two shuffle sequences
    assert source.count("__shfl_down_sync") >= 2


def test_roundtrip_matmul_naive():
    """Matmul naive round-trips to valid CUDA."""
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("ew", ElementwiseOp("mul"), ["A", "B"]),
            ("C", ReduceOp("sum", axis=1), ["ew"]),
        ],
        input_names=["A", "B"],
        output_names=["C"],
        shapes={"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)},
    )
    shapes = {"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)}
    prog, kernel, source, _ = _roundtrip(region, "matmul", shapes, analysis, strategy="naive")

    assert "__global__" in source
    assert "void matmul" in source
    assert kernel.block_size == (32, 8, 1)


def test_roundtrip_matmul_tma():
    """Matmul TMA round-trips to valid CUDA."""
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("ew", ElementwiseOp("mul"), ["A", "B"]),
            ("C", ReduceOp("sum", axis=1), ["ew"]),
        ],
        input_names=["A", "B"],
        output_names=["C"],
        shapes={"A": (256, 256), "B": (256, 256), "ew": (256, 256, 256), "C": (256, 256)},
    )
    shapes = {"A": (256, 256), "B": (256, 256), "ew": (256, 256, 256), "C": (256, 256)}
    prog, kernel, source, sched = _roundtrip(
        region, "matmul_tma", shapes, analysis, strategy="tma_db", hints={"block_k": 32, "thread_m": 8}
    )

    assert "__global__" in source
    assert "void matmul_tma" in source
    assert sched.tma_params is not None


# ---------------------------------------------------------------------------
# Schedule tests
# ---------------------------------------------------------------------------


def test_build_schedule_pointwise():
    """Pointwise schedule: 1D grid, no accumulators, no reductions."""
    _, analysis = _make_region_and_analysis(
        region_ops=[("neg", ElementwiseOp("neg"), ["x"])],
        input_names=["x"],
        output_names=["neg"],
        shapes={"x": (256,), "neg": (256,)},
    )
    sched = build_schedule(analysis)
    assert sched.grid.type == "1d"
    assert sched.grid.bound == "n"


def test_build_schedule_reduce():
    """Reduce schedule: 1D grid, scalar accum, warp reduce."""
    _, analysis = _make_region_and_analysis(
        region_ops=[("red", ReduceOp("sum", axis=1), ["x"])],
        input_names=["x"],
        output_names=["red"],
        shapes={"x": (4, 8), "red": (4,)},
    )
    sched = build_schedule(analysis)
    assert sched.grid.type == "1d"
    assert sched.grid.bound == "rows"


def test_build_schedule_contraction():
    """Contraction schedule: 2D swizzle grid, register tile accum."""
    _, analysis = _make_region_and_analysis(
        region_ops=[
            ("ew", ElementwiseOp("mul"), ["A", "B"]),
            ("C", ReduceOp("sum", axis=1), ["ew"]),
        ],
        input_names=["A", "B"],
        output_names=["C"],
        shapes={"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)},
    )
    sched = build_schedule(analysis, strategy="naive")
    assert sched.grid.type == "2d_swizzle"
    assert sched.thread_m == 8
    assert sched.thread_n == 4
    assert sched.tile_m == 64  # ty(8) * thread_m(8)
    assert sched.tile_n == 128  # tx(32) * thread_n(4)


def test_lower_generic_matches_lower_to_loop_ir():
    """lower_generic via build_schedule produces same structure as lower_to_loop_ir."""
    region, analysis = _make_region_and_analysis(
        region_ops=[("red", ReduceOp("sum", axis=1), ["x"])],
        input_names=["x"],
        output_names=["red"],
        shapes={"x": (4, 8), "red": (4,)},
    )
    shapes = {"x": (4, 8), "red": (4,)}

    # Via lower_to_loop_ir (routes through lower_generic internally)
    prog1, sched1 = lower_to_loop_ir(region, "test", shapes, analysis)
    # Via explicit build_schedule + lower_generic
    sched2 = build_schedule(analysis)
    prog2 = lower_generic(region, "test", shapes, sched2)

    # Both should produce the same LoopIR structure
    assert sched1.grid.block_size == sched2.grid.block_size
    assert len(prog1.body) == len(prog2.body)
    assert len(prog1.params) == len(prog2.params)
