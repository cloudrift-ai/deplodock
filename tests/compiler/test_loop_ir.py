"""Tests for LoopIR: dataclasses, pretty-printer, lowering, and round-trip codegen."""

import json

from deplodock.compiler.backend.codegen import emit_kernel
from deplodock.compiler.backend.cuda.generators import analyze, lower_to_loop_ir
from deplodock.compiler.backend.cuda.generators.loop_codegen import loop_ir_to_kernel
from deplodock.compiler.backend.loop_ir import (
    Accumulate,
    Alloc,
    Compute,
    Guard,
    Load,
    LoopBinOp,
    LoopBuiltin,
    LoopLiteral,
    LoopNest,
    LoopProgram,
    LoopVar,
    ParallelAxis,
    RawLoopOp,
    Store,
    WarpReduce,
    pretty_print,
    to_dict,
)
from deplodock.compiler.ops import ElementwiseOp, FusedRegionOp, ReduceOp

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
                LoopBinOp("<", LoopVar("i"), LoopVar("n")),
                [
                    Load("v", "X", LoopVar("i"), "global"),
                    Compute("out", "mul", [LoopVar("v"), LoopLiteral(2.0)]),
                    Store("Y", LoopVar("i"), LoopVar("out"), "global"),
                ],
            ),
        ],
        block_size=(256, 1, 1),
    )
    assert prog.name == "test_kernel"
    assert prog.block_size == (256, 1, 1)
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
                LoopBinOp("<", LoopVar("i"), LoopVar("n")),
                [
                    Load("v", "X", LoopVar("i"), "global"),
                    Compute("out", "relu", [LoopVar("v")]),
                    Store("Y", LoopVar("i"), LoopVar("out"), "global"),
                ],
            ),
        ],
        block_size=(256, 1, 1),
    )
    text = pretty_print(prog)
    assert "loop_program relu_kernel" in text
    assert "block_size: (256, 1, 1)" in text
    assert "parallel i" in text
    assert "guard" in text
    assert "load v" in text
    assert "compute out = relu" in text
    assert "store Y" in text


def test_pretty_print_reduce():
    """Pretty-print a reduce program with WarpReduce."""
    prog = LoopProgram(
        name="sum_kernel",
        params=[("const float*", "X"), ("float*", "Y"), ("int", "rows"), ("int", "cols")],
        body=[
            ParallelAxis("row", "blockIdx.x", "rows"),
            Guard(
                LoopBinOp("<", LoopVar("row"), LoopVar("rows")),
                [
                    Alloc("acc", "float", None, "reg", LoopLiteral(0.0)),
                    LoopNest(
                        "j",
                        LoopBuiltin("threadIdx.x"),
                        LoopVar("cols"),
                        LoopBuiltin("blockDim.x"),
                        [
                            Load("v", "X", LoopBinOp("+", LoopBinOp("*", LoopVar("row"), LoopVar("cols")), LoopVar("j")), "global"),
                            Accumulate("acc", "sum", LoopVar("v")),
                        ],
                    ),
                    WarpReduce("acc", "sum"),
                ],
            ),
        ],
        block_size=(256, 1, 1),
    )
    text = pretty_print(prog)
    assert "alloc reg float acc" in text
    assert "for j" in text
    assert "accumulate acc sum" in text
    assert "warp_reduce acc sum" in text


def test_pretty_print_contraction():
    """Pretty-print includes LoopNest for K-loop."""
    prog = LoopProgram(
        name="matmul",
        params=[("float*", "C")],
        body=[
            RawLoopOp("/* CTA swizzle grid setup */", "grid setup"),
            Alloc("c00", "float", None, "reg", LoopLiteral(0.0)),
            LoopNest(
                "k",
                LoopLiteral(0, "int"),
                LoopVar("K"),
                None,
                [RawLoopOp("/* FMA body */", "K-loop body")],
            ),
        ],
        block_size=(32, 8, 1),
        tile_m=64,
        tile_n=128,
    )
    text = pretty_print(prog)
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
                LoopBinOp("<", LoopVar("i"), LoopVar("n")),
                [Store("X", LoopVar("i"), LoopLiteral(1.0), "global")],
            ),
        ],
        block_size=(256, 1, 1),
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
    """Build FusedRegionOp + TileAnalysis."""
    region = FusedRegionOp(
        region_ops=region_ops,
        input_names=input_names,
        output_names=output_names,
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
    prog = lower_to_loop_ir(region, "test_pw", {**region.shapes, **{"x": (n,), "neg": (n,), "exp": (n,)}}, analysis)

    assert prog.block_size == (256, 1, 1)
    assert any(isinstance(op, ParallelAxis) for op in prog.body)
    guards = _find_ops(prog.body, Guard, recursive=False)
    assert len(guards) == 1
    computes = _find_ops(guards[0].body, Compute)
    stores = _find_ops(guards[0].body, Store)
    assert len(computes) >= 1
    assert len(stores) == 1


def test_single_reduce_structure():
    """Row reduce has: ParallelAxis + Alloc + LoopNest + WarpReduce."""
    rows, cols = 4, 8
    region, analysis = _make_region_and_analysis(
        region_ops=[("red", ReduceOp("sum", axis=1), ["x"])],
        input_names=["x"],
        output_names=["red"],
        shapes={"x": (rows, cols), "red": (rows,)},
    )
    prog = lower_to_loop_ir(region, "test_red", {"x": (rows, cols), "red": (rows,)}, analysis)

    assert prog.block_size == (256, 1, 1)
    allocs = _find_ops(prog.body, Alloc)
    assert any(a.name.startswith("acc_") for a in allocs)
    loops = _find_ops(prog.body, LoopNest)
    assert len(loops) >= 1
    reduces = _find_ops(prog.body, WarpReduce)
    assert len(reduces) == 1
    assert reduces[0].op == "sum"


def test_multi_reduce_structure():
    """Multi-reduce (softmax) has multiple WarpReduce passes."""
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
    prog = lower_to_loop_ir(
        region,
        "test_softmax",
        {"x": (rows, cols), "mx": (rows, 1), "sub": (rows, cols), "exp": (rows, cols), "sm": (rows, 1), "div": (rows, cols)},
        analysis,
    )

    reduces = _find_ops(prog.body, WarpReduce)
    assert len(reduces) == 2  # max + sum
    assert reduces[0].op == "max"
    assert reduces[1].op == "sum"

    loops = _find_ops(prog.body, LoopNest)
    assert len(loops) >= 3  # max pass + sum pass + epilogue pass


def test_contraction_structure():
    """Contraction has tile_m/tile_n set and uses RawLoopOp for legacy body."""
    region, analysis = _make_region_and_analysis(
        region_ops=[
            ("ew", ElementwiseOp("mul"), ["A", "B"]),
            ("C", ReduceOp("sum", axis=1), ["ew"]),
        ],
        input_names=["A", "B"],
        output_names=["C"],
        shapes={"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)},
    )
    prog = lower_to_loop_ir(
        region,
        "test_mm",
        {"A": (8, 4), "B": (4, 6), "ew": (8, 4, 6), "C": (8, 6)},
        analysis,
        strategy="naive",
    )

    # Contraction uses legacy wrapper — body is RawLoopOp
    raws = _find_ops(prog.body, RawLoopOp, recursive=False)
    assert len(raws) == 1
    assert prog.block_size == (32, 8, 1)


# ---------------------------------------------------------------------------
# Round-trip tests: lower_to_loop_ir → loop_ir_to_kernel → emit_kernel
# ---------------------------------------------------------------------------


def _roundtrip(region, name, shapes, analysis, **kwargs):
    """Lower to LoopIR, codegen to KernelDef, emit to CUDA source."""
    prog = lower_to_loop_ir(region, name, shapes, analysis, **kwargs)
    kernel = loop_ir_to_kernel(prog)
    source = emit_kernel(kernel)
    return prog, kernel, source


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
    _, kernel, source = _roundtrip(region, "silu", shapes, analysis)

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
    _, kernel, source = _roundtrip(region, "row_sum", {"x": (rows, cols), "red": (rows,)}, analysis)

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
    _, _, source = _roundtrip(region, "rmsnorm", shapes, analysis)

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
    _, _, source = _roundtrip(region, "softmax", shapes, analysis)

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
    prog, kernel, source = _roundtrip(region, "matmul", shapes, analysis, strategy="naive")

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
    prog, kernel, source = _roundtrip(region, "matmul_tma", shapes, analysis, strategy="tma_db", hints={"block_k": 32, "thread_m": 8})

    assert "__global__" in source
    assert "void matmul_tma" in source
    assert prog.tma_params is not None
