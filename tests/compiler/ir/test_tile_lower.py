"""Loop IR → Tile IR lowering tests.

Step 3 of the Tile IR refactor: ``lower_naive`` translates a ``LoopOp``
into a single-thread serial ``Kernel``. These tests check both:

- Structure: the produced ``Kernel`` has the expected node tree
  (FreeLoop nesting, Reduce + Acc placement, prologue contents).
- Round-trip: lower then ``render_kernel`` and assert that recognisable
  CUDA snippets emerge end-to-end.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Var as ExprVar
from deplodock.compiler.ir.loop import (
    Accum,
    Assign,
    Axis,
    Load,
    Loop,
    LoopOp,
    Write,
)
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.ir.tile import (
    AccumFold,
    BinaryExpr,
    FreeLoop,
    Index,
    Kernel,
    Let,
    Param,
    Reduce,
    Store,
)
from deplodock.compiler.ir.tile.lower import lower_naive
from deplodock.compiler.ir.tile.render import render_kernel

# ---------------------------------------------------------------------------
# Pointwise add — minimal shape exercise
# ---------------------------------------------------------------------------


def _pointwise_add_loop_op() -> LoopOp:
    a0, a1 = Axis("a0", 4), Axis("a1", 8)
    return LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=a1,
                        body=(
                            Load("ax", input="src_0", index=(ExprVar("a0"), ExprVar("a1"))),
                            Load("bx", input="src_1", index=(ExprVar("a0"), ExprVar("a1"))),
                            Assign("y", ElementwiseOp("add"), ("ax", "bx")),
                            Write(output="out_0", index=(ExprVar("a0"), ExprVar("a1")), value="y"),
                        ),
                    ),
                ),
            ),
        )
    )


def _pointwise_add_params() -> tuple[tuple[Param, ...], Param]:
    return (
        (Param("src_0", "const float*", shape=(4, 8)), Param("src_1", "const float*", shape=(4, 8))),
        Param("out_0", "float*", shape=(4, 8)),
    )


def test_pointwise_add_structure():
    op = _pointwise_add_loop_op()
    inputs, output = _pointwise_add_params()
    k = lower_naive(op, "add", inputs, output)

    assert isinstance(k, Kernel)
    assert k.name == "add"
    assert k.thread_axes == ()  # naive: single-thread
    assert k.prologue == ()  # no scalar Loads
    assert len(k.body) == 1
    outer = k.body[0]
    assert isinstance(outer, FreeLoop) and outer.axis.name == "a0"
    inner = outer.body[0]
    assert isinstance(inner, FreeLoop) and inner.axis.name == "a1"
    # Inner body: 2 Lets (loads), 1 Let (add), 1 Store.
    assert isinstance(inner.body[0], Let) and isinstance(inner.body[0].init, Index)
    assert isinstance(inner.body[2], Let) and isinstance(inner.body[2].init, BinaryExpr)
    assert isinstance(inner.body[3], Store)
    assert inner.body[3].buf == "out_0"


def test_pointwise_add_renders():
    op = _pointwise_add_loop_op()
    inputs, output = _pointwise_add_params()
    src = render_kernel(lower_naive(op, "add", inputs, output))
    # ``LoopOp.__post_init__`` normalizes SSA names → in0/in1 for Loads, v0 for Assign.
    assert "for (int a0 = 0; a0 < 4; a0++) {" in src
    assert "for (int a1 = 0; a1 < 8; a1++) {" in src
    assert "float in0 = src_0[a0 * 8 + a1];" in src
    assert "float v0 = in0 + in1;" in src
    assert "out_0[a0 * 8 + a1] = v0;" in src


# ---------------------------------------------------------------------------
# Reduce — sum along k
# ---------------------------------------------------------------------------


def _row_sum_loop_op() -> LoopOp:
    a0 = Axis("a0", 4)
    k = Axis("a1", 32)
    return LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=k,
                        body=(
                            Load("x", input="src_0", index=(ExprVar("a0"), ExprVar("a1"))),
                            Accum(name="s", value="x", op="add"),
                        ),
                    ),
                    Write(output="out_0", index=(ExprVar("a0"),), value="s"),
                ),
            ),
        )
    )


def test_row_sum_structure():
    op = _row_sum_loop_op()
    inputs = (Param("src_0", "const float*", shape=(4, 32)),)
    output = Param("out_0", "float*", shape=(4,))
    k = lower_naive(op, "row_sum", inputs, output)

    outer = k.body[0]
    assert isinstance(outer, FreeLoop)
    reduce_node = outer.body[0]
    assert isinstance(reduce_node, Reduce)
    assert reduce_node.axis.name == "a1"
    assert len(reduce_node.accs) == 1
    # Loop IR normalizes Accum name → acc0; op is the same ElementwiseImpl
    # that came in on the source Accum (canonical "add" name from the fixture).
    assert reduce_node.accs[0].name == "acc0"
    assert reduce_node.accs[0].op.name == "add"
    assert isinstance(reduce_node.body[0], Let)
    assert isinstance(reduce_node.body[1], AccumFold)
    assert reduce_node.body[1].target == "acc0"
    assert isinstance(outer.body[1], Store)


def test_row_sum_renders():
    op = _row_sum_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "row_sum",
            (Param("src_0", "const float*", shape=(4, 32)),),
            Param("out_0", "float*", shape=(4,)),
        )
    )
    assert "float acc0 = 0.0f;" in src
    assert "for (int a1 = 0; a1 < 32; a1++) {" in src
    assert "acc0 += in0;" in src
    assert "out_0[a0] = acc0;" in src


# ---------------------------------------------------------------------------
# Two reductions sharing live axis (softmax-like)
# ---------------------------------------------------------------------------


def _softmax_like_loop_op() -> LoopOp:
    a0 = Axis("a0", 4)
    k = Axis("a1", 32)
    j = Axis("a2", 32)
    return LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=k,
                        body=(
                            Load("xk", input="src_0", index=(ExprVar("a0"), ExprVar("a1"))),
                            Accum(name="m", value="xk", op="maximum"),
                        ),
                    ),
                    Loop(
                        axis=k,
                        body=(
                            Load("xk2", input="src_0", index=(ExprVar("a0"), ExprVar("a1"))),
                            Assign("d", ElementwiseOp("subtract"), ("xk2", "m")),
                            Assign("e", ElementwiseOp("exp"), ("d",)),
                            Accum(name="s", value="e", op="add"),
                        ),
                    ),
                    Loop(
                        axis=j,
                        body=(
                            Load("xj", input="src_0", index=(ExprVar("a0"), ExprVar("a2"))),
                            Assign("dj", ElementwiseOp("subtract"), ("xj", "m")),
                            Assign("ej", ElementwiseOp("exp"), ("dj",)),
                            Assign("y", ElementwiseOp("divide"), ("ej", "s")),
                            Write(output="out_0", index=(ExprVar("a0"), ExprVar("a2")), value="y"),
                        ),
                    ),
                ),
            ),
        )
    )


def test_softmax_like_structure():
    op = _softmax_like_loop_op()
    k_kern = lower_naive(
        op,
        "softmax",
        (Param("src_0", "const float*", shape=(4, 32)),),
        Param("out_0", "float*", shape=(4, 32)),
    )
    outer = k_kern.body[0]
    assert isinstance(outer, FreeLoop) and outer.axis.name == "a0"
    r_max, r_sum, out_loop = outer.body
    assert isinstance(r_max, Reduce) and r_max.accs[0].op.name == "maximum"
    assert isinstance(r_sum, Reduce) and r_sum.accs[0].op.name == "add"
    assert isinstance(out_loop, FreeLoop) and out_loop.axis.name == "a2"


def test_softmax_like_renders():
    op = _softmax_like_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "softmax",
            (Param("src_0", "const float*", shape=(4, 32)),),
            Param("out_0", "float*", shape=(4, 32)),
        )
    )
    # Two reduces: acc0 = max, acc1 = sum.
    assert "acc0 = fmaxf(acc0, " in src
    # sum-reduce body has subtract + exp + accumfold; expf translation in.
    assert "expf(" in src
    assert "acc1 +=" in src
    # output loop final divide and store.
    assert " / acc1;" in src
    assert "out_0[a0 * 32 + a2] = " in src


# ---------------------------------------------------------------------------
# Matmul — direct Write at reduce-loop level (no output loop)
# ---------------------------------------------------------------------------


def _matmul_loop_op() -> LoopOp:
    m, n, k = Axis("a0", 8), Axis("a1", 8), Axis("a2", 16)
    return LoopOp(
        body=(
            Loop(
                axis=m,
                body=(
                    Loop(
                        axis=n,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load("a", input="src_0", index=(ExprVar("a0"), ExprVar("a2"))),
                                    Load("b", input="src_1", index=(ExprVar("a2"), ExprVar("a1"))),
                                    Assign("p", ElementwiseOp("multiply"), ("a", "b")),
                                    Accum(name="c", value="p", op="add"),
                                ),
                            ),
                            Write(output="out_0", index=(ExprVar("a0"), ExprVar("a1")), value="c"),
                        ),
                    ),
                ),
            ),
        )
    )


def test_matmul_structure():
    op = _matmul_loop_op()
    k = lower_naive(
        op,
        "matmul",
        (
            Param("src_0", "const float*", shape=(8, 16)),
            Param("src_1", "const float*", shape=(16, 8)),
        ),
        Param("out_0", "float*", shape=(8, 8)),
    )
    m_loop = k.body[0]
    assert isinstance(m_loop, FreeLoop) and m_loop.axis.name == "a0"
    n_loop = m_loop.body[0]
    assert isinstance(n_loop, FreeLoop) and n_loop.axis.name == "a1"
    reduce_node = n_loop.body[0]
    assert isinstance(reduce_node, Reduce) and reduce_node.axis.name == "a2"
    assert isinstance(n_loop.body[1], Store)


def test_matmul_renders():
    op = _matmul_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "matmul",
            (
                Param("src_0", "const float*", shape=(8, 16)),
                Param("src_1", "const float*", shape=(16, 8)),
            ),
            Param("out_0", "float*", shape=(8, 8)),
        )
    )
    assert "float acc0 = 0.0f;" in src
    assert "for (int a2 = 0; a2 < 16; a2++) {" in src
    # Loads renamed in0/in1; multiply assign renamed v0.
    assert "float v0 = in0 * in1;" in src
    assert "acc0 += v0;" in src
    assert "out_0[a0 * 8 + a1] = acc0;" in src


# ---------------------------------------------------------------------------
# Top-level scalar Loads → prologue
# ---------------------------------------------------------------------------


def _scalar_prologue_loop_op() -> LoopOp:
    """A LoopOp with a scalar Load above the outer FreeLoop — should land in prologue."""
    a0 = Axis("a0", 4)
    return LoopOp(
        body=(
            Load("eps", input="src_1", index=()),
            Loop(
                axis=a0,
                body=(
                    Load("x", input="src_0", index=(ExprVar("a0"),)),
                    Assign("y", ElementwiseOp("add"), ("x", "eps")),
                    Write(output="out_0", index=(ExprVar("a0"),), value="y"),
                ),
            ),
        )
    )


def test_scalar_prologue_lifts_to_kernel_prologue():
    op = _scalar_prologue_loop_op()
    k = lower_naive(
        op,
        "scalar_pro",
        (Param("src_0", "const float*", shape=(4,)), Param("src_1", "const float*", shape=(1,))),
        Param("out_0", "float*", shape=(4,)),
    )
    assert len(k.prologue) == 1
    assert isinstance(k.prologue[0], Let)
    # Loop IR canonicalizes Load names → in0 (the eps load is the first Load in pre-order).
    assert k.prologue[0].name == "in0"
    assert isinstance(k.body[0], FreeLoop)


def test_scalar_prologue_renders_above_body():
    op = _scalar_prologue_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "scalar_pro",
            (Param("src_0", "const float*", shape=(4,)), Param("src_1", "const float*", shape=(1,))),
            Param("out_0", "float*", shape=(4,)),
        )
    )
    # Scalar load: empty index → "src_1[0]" (renderer flattens () to "0").
    # Eps is the first Load in body pre-order, so it gets `in0`.
    assert "float in0 = src_1[0];" in src
    assert src.index("float in0 = src_1[0];") < src.index("for (int a0 = 0; a0 < 4; a0++)")
