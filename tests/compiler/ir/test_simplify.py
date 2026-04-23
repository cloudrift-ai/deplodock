"""Unit tests for ``compiler.ir.simplify`` — the generic Expr rewriter.

Covers:
- Constant folding for every BinOp op / Cast.
- Algebraic identities (``x+0``, ``x*0``, ``x-x``, etc.).
- Range-based comparison folding using Context intervals.
- Ternary collapse (literal cond, equal branches).
- Idempotence — simplifying twice equals simplifying once.
- Kernel IR walker — tid-range propagation from IfStmt, ForLoop bounds.
- Loop IR walker — axis extents seed ranges, no clamp survives on a
  provably-in-range index.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import BinOp, Builtin, Cast, FuncCall, Literal, Ternary, Var
from deplodock.compiler.ir.kernel import (
    ArrayAccess,
    AugAssign,
    ForLoop,
    GpuKernel,
    GpuKernelParam,
    IfStmt,
    VarDecl,
)
from deplodock.compiler.ir.kernel import normalize_kernel as simplify_kernel
from deplodock.compiler.ir.loop import Assign, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.loop.simplify import (
    Context,
    Interval,
    infer_range,
    simplify_expr,
    simplify_loop_op,
)
from deplodock.compiler.ir.tensor.ir import ElementwiseOp

# ---------------------------------------------------------------------------
# Core Expr folding
# ---------------------------------------------------------------------------


def _int(v: int) -> Literal:
    return Literal(int(v), "int")


def test_constant_fold_add():
    assert simplify_expr(BinOp("+", _int(2), _int(3)), Context.empty()) == _int(5)


def test_constant_fold_mul():
    assert simplify_expr(BinOp("*", _int(4), _int(5)), Context.empty()) == _int(20)


def test_constant_fold_mod():
    assert simplify_expr(BinOp("%", _int(10), _int(3)), Context.empty()) == _int(1)


def test_constant_fold_comparison():
    e = BinOp("<", _int(2), _int(3))
    assert simplify_expr(e, Context.empty()) == _int(1)


def test_x_plus_zero():
    x = Var("x")
    assert simplify_expr(BinOp("+", x, _int(0)), Context.empty()) == x
    assert simplify_expr(BinOp("+", _int(0), x), Context.empty()) == x


def test_x_times_zero():
    x = Var("x")
    assert simplify_expr(BinOp("*", x, _int(0)), Context.empty()) == _int(0)
    assert simplify_expr(BinOp("*", _int(0), x), Context.empty()) == _int(0)


def test_x_times_one():
    x = Var("x")
    assert simplify_expr(BinOp("*", x, _int(1)), Context.empty()) == x


def test_x_minus_x():
    x = Var("x")
    assert simplify_expr(BinOp("-", x, x), Context.empty()) == _int(0)


def test_x_mod_one():
    x = Var("x")
    assert simplify_expr(BinOp("%", x, _int(1)), Context.empty()) == _int(0)


def test_zero_div_x():
    x = Var("x")
    assert simplify_expr(BinOp("/", _int(0), x), Context.empty()) == _int(0)


# ---------------------------------------------------------------------------
# Ternary collapse
# ---------------------------------------------------------------------------


def test_ternary_literal_true_cond():
    a, b = Var("a"), Var("b")
    assert simplify_expr(Ternary(_int(1), a, b), Context.empty()) == a


def test_ternary_literal_false_cond():
    a, b = Var("a"), Var("b")
    assert simplify_expr(Ternary(_int(0), a, b), Context.empty()) == b


def test_ternary_equal_branches():
    a = Var("a")
    assert simplify_expr(Ternary(Var("c"), a, a), Context.empty()) == a


# ---------------------------------------------------------------------------
# Range-based folding — the clamp-elimination case
# ---------------------------------------------------------------------------


def test_range_decides_comparison_false():
    k = Var("k")
    ctx = Context({"k": Interval(0, 2047)})
    # k > 2047 is always false in [0, 2047]
    assert simplify_expr(BinOp(">", k, _int(2047)), ctx) == _int(0)


def test_range_decides_comparison_true():
    k = Var("k")
    ctx = Context({"k": Interval(0, 2047)})
    # k < 2048 is always true
    assert simplify_expr(BinOp("<", k, _int(2048)), ctx) == _int(1)


def test_chained_clamp_collapses_to_var():
    """(k0 > N-1 ? N-1 : k0) < 0 ? 0 : ... → k0, when k0 ∈ [0, N-1]."""
    k = Var("k0")
    upper = Ternary(BinOp(">", k, _int(2047)), _int(2047), k)
    full = Ternary(BinOp("<", upper, _int(0)), _int(0), upper)
    ctx = Context({"k0": Interval(0, 2047)})
    assert simplify_expr(full, ctx) == k


def test_mod_produces_known_range():
    k = Var("tid")
    ctx = Context({"tid": Interval(0, 65535)})
    # tid % 2048 has range [0, 2047]
    mod = BinOp("%", k, _int(2048))
    assert infer_range(mod, ctx) == Interval(0, 2047)
    # (tid % 2048) > 2047 is always false
    assert simplify_expr(BinOp(">", mod, _int(2047)), ctx) == _int(0)


def test_infer_range_plus():
    ctx = Context({"a": Interval(0, 10), "b": Interval(5, 20)})
    assert infer_range(BinOp("+", Var("a"), Var("b")), ctx) == Interval(5, 30)


def test_infer_range_div_by_const():
    ctx = Context({"tid": Interval(0, 65535)})
    assert infer_range(BinOp("/", Var("tid"), _int(2048)), ctx) == Interval(0, 31)


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_idempotent_on_clamp_chain():
    k = Var("k0")
    upper = Ternary(BinOp(">", k, _int(2047)), _int(2047), k)
    full = Ternary(BinOp("<", upper, _int(0)), _int(0), upper)
    ctx = Context({"k0": Interval(0, 2047)})
    once = simplify_expr(full, ctx)
    twice = simplify_expr(once, ctx)
    assert once == twice


def test_idempotent_on_var():
    x = Var("x")
    assert simplify_expr(simplify_expr(x, Context.empty()), Context.empty()) == x


# ---------------------------------------------------------------------------
# Cast / FuncCall / ArrayAccess
# ---------------------------------------------------------------------------


def test_cast_literal_to_int():
    assert simplify_expr(Cast("int", Literal(3.7, "float")), Context.empty()) == _int(3)


def test_funccall_args_simplified():
    e = FuncCall("expf", [BinOp("+", _int(1), _int(2))])
    assert simplify_expr(e, Context.empty()) == FuncCall("expf", [_int(3)])


def test_array_access_index_simplified():
    e = ArrayAccess("x", BinOp("+", Var("i"), _int(0)))
    assert simplify_expr(e, Context.empty()) == ArrayAccess("x", Var("i"))


# ---------------------------------------------------------------------------
# Kernel IR walker
# ---------------------------------------------------------------------------


def _rms_kernel() -> GpuKernel:
    """Minimal kernel mirroring the RMSNorm layout: tid = bx*bdx+tx, if tid<N."""
    tid_init = BinOp(
        "+",
        BinOp("*", Builtin("blockIdx.x"), Builtin("blockDim.x")),
        Builtin("threadIdx.x"),
    )
    body = [
        VarDecl("long long", "tid", tid_init),
        IfStmt(
            BinOp("<", Var("tid"), _int(65536)),
            [
                # target has a redundant clamp: (tid%2048 > 2047) ? 2047 : tid%2048
                VarDecl(
                    "float",
                    "t",
                    ArrayAccess(
                        "x",
                        Ternary(
                            BinOp(">", BinOp("%", Var("tid"), _int(2048)), _int(2047)),
                            _int(2047),
                            BinOp("%", Var("tid"), _int(2048)),
                        ),
                    ),
                ),
            ],
            None,
        ),
    ]
    return GpuKernel(
        name="rms",
        params=[GpuKernelParam("float*", "x")],
        body=body,
        block_size=(256, 1, 1),
    )


def test_kernel_walker_eliminates_clamp_via_if_tightening():
    kernel = _rms_kernel()
    simplified = simplify_kernel(kernel)
    # Inside the if body, the ternary-clamped index should collapse to tid % 2048.
    if_stmt = simplified.body[1]
    assert isinstance(if_stmt, IfStmt)
    inner_decl = if_stmt.body[0]
    assert isinstance(inner_decl, VarDecl)
    assert inner_decl.init == ArrayAccess("x", BinOp("%", Var("tid"), _int(2048)))


def test_kernel_walker_for_loop_bounds():
    """for (k0 = 0; k0 < 2048; ...) ⇒ k0 ∈ [0, 2047] inside body; k0 > 2047 folds."""
    inner = AugAssign(
        "acc",
        "+=",
        Ternary(BinOp(">", Var("k0"), _int(2047)), _int(0), Var("k0")),
    )
    kernel = GpuKernel(
        name="k",
        params=[GpuKernelParam("float*", "acc")],
        body=[
            VarDecl("float", "acc", _int(0)),
            ForLoop("k0", _int(0), _int(2048), [inner], None),
        ],
        block_size=(256, 1, 1),
    )
    simplified = simplify_kernel(kernel)
    for_stmt = simplified.body[1]
    assert isinstance(for_stmt, ForLoop)
    inner_simp = for_stmt.body[0]
    assert isinstance(inner_simp, AugAssign)
    # k0 > 2047 is always False in [0, 2047] → Ternary collapses to the else branch, Var("k0").
    assert inner_simp.value == Var("k0")


def test_kernel_walker_idempotent():
    kernel = _rms_kernel()
    once = simplify_kernel(kernel)
    twice = simplify_kernel(once)
    assert once == twice


# ---------------------------------------------------------------------------
# Loop IR walker
# ---------------------------------------------------------------------------


def test_loop_op_walker_simplifies_load_index():
    """Body Load index (a0 + 0) * 1 simplifies to a0 under axis range info."""
    load = Load(name="x", source=0, index=(BinOp("*", BinOp("+", Var("a0"), _int(0)), _int(1)),))
    assign = Assign("v", ElementwiseOp(fn="neg"), ("x",))
    write = Write(0, (Var("a0"),), "v")
    loop = Loop(Axis("a0", 8), (load, assign, write))
    op = LoopOp(body=(loop,))
    simplified = simplify_loop_op(op)
    simplified_load = simplified.loads[0]
    assert simplified_load.index == (Var("a0"),)


def test_loop_op_walker_idempotent():
    load = Load(name="x", source=0, index=(BinOp("+", Var("a0"), _int(0)),))
    assign = Assign("v", ElementwiseOp(fn="neg"), ("x",))
    write = Write(0, (Var("a0"),), "v")
    loop = Loop(Axis("a0", 8), (load, assign, write))
    op = LoopOp(body=(loop,))
    once = simplify_loop_op(op)
    twice = simplify_loop_op(once)
    assert once == twice
