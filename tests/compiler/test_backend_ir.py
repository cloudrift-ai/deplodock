"""Unit tests for the backend IR AST nodes and codegen emission.

Tests each expression and statement type directly, verifying that
emit_kernel / _emit_expr / _emit_stmt produce correct C/CUDA source.
"""

from deplodock.compiler.backend.ir.kernel_codegen import emit_kernel
from deplodock.compiler.backend.ir.kernel_ir import (
    ArrayAccess,
    ArrayDecl,
    Assign,
    AugAssign,
    BinOp,
    Builtin,
    Cast,
    FieldAccess,
    ForLoop,
    FuncCall,
    IfStmt,
    KernelDef,
    KernelParam,
    Literal,
    PragmaUnroll,
    RawCode,
    SyncThreads,
    Ternary,
    Var,
    VarDecl,
    VectorLoad,
)

# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------


def _emit_expr(expr, parent_prec=0) -> str:
    """Helper: access the private _emit_expr from codegen."""
    from deplodock.compiler.backend.ir.kernel_codegen import _emit_expr

    return _emit_expr(expr, parent_prec)


def test_var():
    assert _emit_expr(Var("x")) == "x"


def test_literal_int():
    assert _emit_expr(Literal(42, "int")) == "42"


def test_literal_float():
    result = _emit_expr(Literal(3.14, "float"))
    assert result.endswith("f")
    assert "3.1" in result


def test_literal_int_as_float():
    """Integer value with dtype='float' emits as float literal."""
    result = _emit_expr(Literal(0, "float"))
    assert result == "0.0f"


def test_binop_add():
    expr = BinOp("+", Var("a"), Var("b"))
    assert _emit_expr(expr) == "a + b"


def test_binop_mul():
    expr = BinOp("*", Var("a"), Var("b"))
    assert _emit_expr(expr) == "a * b"


def test_binop_precedence_parenthesizes():
    """Lower-precedence op nested inside higher-precedence context gets parens."""
    # (a + b) * c — the add needs parens when parent expects * precedence.
    inner = BinOp("+", Var("a"), Var("b"))
    outer = BinOp("*", inner, Var("c"))
    result = _emit_expr(outer)
    assert "(" in result  # (a + b) * c
    assert result == "(a + b) * c"


def test_binop_same_precedence_right_assoc():
    """Right-nested same-precedence gets parens (codegen uses strict right prec)."""
    expr = BinOp("+", Var("a"), BinOp("+", Var("b"), Var("c")))
    result = _emit_expr(expr)
    assert result == "a + (b + c)"


def test_array_access():
    expr = ArrayAccess("arr", Var("i"))
    assert _emit_expr(expr) == "arr[i]"


def test_array_access_complex_index():
    expr = ArrayAccess("buf", BinOp("+", Var("row"), Var("col")))
    assert _emit_expr(expr) == "buf[row + col]"


def test_cuda_builtin():
    assert _emit_expr(Builtin("threadIdx.x")) == "threadIdx.x"
    assert _emit_expr(Builtin("blockIdx.y")) == "blockIdx.y"


def test_func_call_no_args():
    expr = FuncCall("__syncwarp", [])
    assert _emit_expr(expr) == "__syncwarp()"


def test_func_call_with_args():
    expr = FuncCall("fmaxf", [Var("a"), Var("b")])
    assert _emit_expr(expr) == "fmaxf(a, b)"


def test_func_call_nested():
    expr = FuncCall("expf", [FuncCall("__ldg", [ArrayAccess("buf", Var("i"))])])
    assert _emit_expr(expr) == "expf(__ldg(buf[i]))"


def test_cast():
    expr = Cast("float4", Var("ptr"))
    assert _emit_expr(expr) == "((float4)(ptr))"


def test_field_access():
    expr = FieldAccess(Var("v"), "x")
    assert _emit_expr(expr) == "v.x"


def test_field_access_chained():
    """Chained field access through a cast."""
    expr = FieldAccess(Cast("float4", Var("ptr")), "w")
    assert "w" in _emit_expr(expr)


def test_ternary():
    expr = Ternary(BinOp("<", Var("i"), Var("n")), Var("a"), Var("b"))
    result = _emit_expr(expr)
    assert "?" in result
    assert ":" in result


def test_vector_load_float4():
    expr = VectorLoad("buf", Var("i"), width=4)
    result = _emit_expr(expr)
    assert "float4" in result
    assert "reinterpret_cast" in result
    assert "buf" in result


def test_vector_load_float2():
    expr = VectorLoad("buf", Var("i"), width=2)
    result = _emit_expr(expr)
    assert "float2" in result


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------


def _emit_stmt(stmt, indent=0) -> str:
    """Helper: access the private _emit_stmt from codegen."""
    from deplodock.compiler.backend.ir.kernel_codegen import _emit_stmt

    return _emit_stmt(stmt, indent)


def test_var_decl():
    stmt = VarDecl("float", "x")
    assert _emit_stmt(stmt) == "float x;"


def test_var_decl_with_init():
    stmt = VarDecl("float", "x", init=Literal(0.0, "float"))
    result = _emit_stmt(stmt)
    assert "float x = 0.0f;" == result


def test_var_decl_indented():
    stmt = VarDecl("int", "i")
    result = _emit_stmt(stmt, indent=2)
    assert result.startswith("        ")  # 8 spaces for indent=2


def test_assign():
    stmt = Assign(target=ArrayAccess("C", Var("idx")), value=Var("acc"))
    result = _emit_stmt(stmt)
    assert result == "C[idx] = acc;"


def test_aug_assign():
    stmt = AugAssign(target="acc", op="+=", value=Var("val"))
    result = _emit_stmt(stmt)
    assert result == "acc += val;"


def test_for_loop_simple():
    stmt = ForLoop(
        var="i",
        start=Literal(0, "int"),
        end=Var("N"),
        body=[AugAssign("sum", "+=", ArrayAccess("arr", Var("i")))],
    )
    result = _emit_stmt(stmt)
    assert "for (int i = 0; i < N; i++)" in result
    assert "sum += arr[i];" in result


def test_for_loop_with_step():
    stmt = ForLoop(
        var="k",
        start=Literal(0, "int"),
        end=Var("K"),
        body=[],
        step=Literal(32, "int"),
    )
    result = _emit_stmt(stmt)
    assert "k += 32" in result


def test_if_stmt():
    stmt = IfStmt(
        cond=BinOp("<", Var("tid"), Var("N")),
        body=[Assign(ArrayAccess("out", Var("tid")), Var("val"))],
    )
    result = _emit_stmt(stmt)
    assert "if (tid < N)" in result
    assert "out[tid] = val;" in result


def test_if_stmt_with_else():
    stmt = IfStmt(
        cond=BinOp("<", Var("i"), Var("n")),
        body=[VarDecl("float", "a", Literal(1.0, "float"))],
        else_body=[VarDecl("float", "a", Literal(0.0, "float"))],
    )
    result = _emit_stmt(stmt)
    assert "if" in result
    assert "else" in result


def test_sync_threads():
    result = _emit_stmt(SyncThreads())
    assert result == "__syncthreads();"


def test_array_decl_1d():
    stmt = ArrayDecl("__shared__ float", "tile", [256])
    assert _emit_stmt(stmt) == "__shared__ float tile[256];"


def test_array_decl_2d():
    stmt = ArrayDecl("__shared__ float", "smem", [64, 64])
    assert _emit_stmt(stmt) == "__shared__ float smem[64][64];"


def test_pragma_unroll_full():
    stmt = PragmaUnroll()
    assert _emit_stmt(stmt) == "#pragma unroll"


def test_pragma_unroll_factor():
    stmt = PragmaUnroll(factor=4)
    assert _emit_stmt(stmt) == "#pragma unroll 4"


def test_raw_code():
    stmt = RawCode('asm volatile("bar.sync 0;");')
    result = _emit_stmt(stmt)
    assert "asm volatile" in result


def test_raw_code_multiline_indented():
    stmt = RawCode("line1;\nline2;")
    result = _emit_stmt(stmt, indent=1)
    lines = result.split("\n")
    assert len(lines) == 2
    assert all(line.startswith("    ") for line in lines)


# ---------------------------------------------------------------------------
# KernelDef emission
# ---------------------------------------------------------------------------


def test_kernel_def_minimal():
    """Minimal kernel with no body emits valid CUDA."""
    kd = KernelDef(
        name="empty_kernel",
        params=[],
        body=[],
        block_size=(32, 1, 1),
    )
    source = emit_kernel(kd)
    assert "__global__" in source
    assert "void empty_kernel()" in source
    assert "__launch_bounds__(32)" in source


def test_kernel_def_with_params():
    """Kernel with float* and int params."""
    kd = KernelDef(
        name="add_kernel",
        params=[
            KernelParam("float*", "A"),
            KernelParam("float*", "B"),
            KernelParam("float*", "C"),
            KernelParam("int", "N"),
        ],
        body=[
            VarDecl("int", "i", Builtin("threadIdx.x")),
            IfStmt(
                cond=BinOp("<", Var("i"), Var("N")),
                body=[Assign(ArrayAccess("C", Var("i")), BinOp("+", ArrayAccess("A", Var("i")), ArrayAccess("B", Var("i"))))],
            ),
        ],
        block_size=(256, 1, 1),
    )
    source = emit_kernel(kd)
    assert "float* A" in source
    assert "float* B" in source
    assert "int N" in source
    assert "C[i] = A[i] + B[i];" in source


def test_kernel_def_with_includes():
    """Kernel with extra includes."""
    kd = KernelDef(
        name="test",
        params=[],
        body=[],
        block_size=(32, 1, 1),
        includes=["cooperative_groups.h"],
    )
    source = emit_kernel(kd)
    assert "#include <cooperative_groups.h>" in source


def test_kernel_def_with_tma_params():
    """Kernel with TMA descriptor params (grid_constant)."""
    kd = KernelDef(
        name="matmul",
        params=[KernelParam("float*", "C")],
        body=[],
        block_size=(32, 8, 1),
        tma_params=["A_tma", "B_tma"],
    )
    source = emit_kernel(kd)
    assert "__grid_constant__ CUtensorMap A_tma" in source
    assert "__grid_constant__ CUtensorMap B_tma" in source
    # Regular param comes after TMA params.
    a_pos = source.index("A_tma")
    c_pos = source.index("float* C")
    assert a_pos < c_pos


def test_kernel_def_launch_bounds_2d():
    """2D block computes correct max threads."""
    kd = KernelDef(name="k", params=[], body=[], block_size=(32, 8, 1))
    source = emit_kernel(kd)
    assert "__launch_bounds__(256)" in source


def test_kernel_def_min_blocks_per_sm():
    """min_blocks_per_sm emits two-arg __launch_bounds__."""
    kd = KernelDef(name="k", params=[], body=[], block_size=(128, 1, 1), min_blocks_per_sm=2)
    source = emit_kernel(kd)
    assert "__launch_bounds__(128, 2)" in source
