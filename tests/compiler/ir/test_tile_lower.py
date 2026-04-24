"""Loop IR → Tile IR lowering tests.

After dropping ``Let`` / ``Store`` etc, ``lower_naive`` is a thin
mechanical pass: Loop IR's ``Loop`` becomes ``Loop`` (no Accum) or
``Reduce`` (Accums present); leaves pass through. These tests verify the
structural shape and round-trip render output.
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
    Kernel,
    Param,
    Reduce,
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
                            Load("ax", input="A", index=(ExprVar("a0"), ExprVar("a1"))),
                            Load("bx", input="B", index=(ExprVar("a0"), ExprVar("a1"))),
                            Assign("y", ElementwiseOp("add"), ("ax", "bx")),
                            Write(output="out", index=(ExprVar("a0"), ExprVar("a1")), value="y"),
                        ),
                    ),
                ),
            ),
        )
    )


def _pointwise_add_params() -> tuple[tuple[Param, ...], Param]:
    return (
        (Param("A", "const float*", shape=(4, 8)), Param("B", "const float*", shape=(4, 8))),
        Param("out", "float*", shape=(4, 8)),
    )


def test_pointwise_add_structure():
    op = _pointwise_add_loop_op()
    inputs, output = _pointwise_add_params()
    k = lower_naive(op, "add", inputs, output)

    assert isinstance(k, Kernel)
    assert k.name == "add"
    assert k.thread_axes == ()
    assert k.prologue == ()
    outer = k.body[0]
    assert isinstance(outer, Loop) and outer.axis.name == "a0"
    inner = outer.body[0]
    assert isinstance(inner, Loop) and inner.axis.name == "a1"
    # Loop IR leaves pass through unchanged.
    assert isinstance(inner.body[0], Load) and inner.body[0].input == "A"
    assert isinstance(inner.body[3], Write) and inner.body[3].output == "out"


def test_pointwise_add_renders():
    op = _pointwise_add_loop_op()
    inputs, output = _pointwise_add_params()
    src = render_kernel(lower_naive(op, "add", inputs, output))
    # ``LoopOp.__post_init__`` normalizes SSA names → in0/in1 for Loads, v0 for Assign.
    assert "for (int a0 = 0; a0 < 4; a0++) {" in src
    assert "for (int a1 = 0; a1 < 8; a1++) {" in src
    assert "float in0 = A[a0 * 8 + a1];" in src
    assert "float v0 = in0 + in1;" in src
    assert "out[a0 * 8 + a1] = v0;" in src


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
                            Load("x", input="X", index=(ExprVar("a0"), ExprVar("a1"))),
                            Accum(name="s", value="x", op="add"),
                        ),
                    ),
                    Write(output="out", index=(ExprVar("a0"),), value="s"),
                ),
            ),
        )
    )


def test_row_sum_structure():
    op = _row_sum_loop_op()
    inputs = (Param("X", "const float*", shape=(4, 32)),)
    output = Param("out", "float*", shape=(4,))
    k = lower_naive(op, "row_sum", inputs, output)

    outer = k.body[0]
    assert isinstance(outer, Loop)
    reduce_node = outer.body[0]
    assert isinstance(reduce_node, Reduce)
    assert reduce_node.axis.name == "a1"
    # Body has the Load + the Accum (Loop IR leaves pass through).
    assert isinstance(reduce_node.body[0], Load)
    assert isinstance(reduce_node.body[1], Accum)
    assert reduce_node.body[1].op.name == "add"
    assert isinstance(outer.body[1], Write)


def test_row_sum_renders():
    op = _row_sum_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "row_sum",
            (Param("X", "const float*", shape=(4, 32)),),
            Param("out", "float*", shape=(4,)),
        )
    )
    # Renderer collects accumulator decls from body Accums.
    assert "float acc0 = 0.0f;" in src
    assert "for (int a1 = 0; a1 < 32; a1++) {" in src
    assert "acc0 += in0;" in src
    assert "out[a0] = acc0;" in src


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
                            Load("xk", input="X", index=(ExprVar("a0"), ExprVar("a1"))),
                            Accum(name="m", value="xk", op="maximum"),
                        ),
                    ),
                    Loop(
                        axis=k,
                        body=(
                            Load("xk2", input="X", index=(ExprVar("a0"), ExprVar("a1"))),
                            Assign("d", ElementwiseOp("subtract"), ("xk2", "m")),
                            Assign("e", ElementwiseOp("exp"), ("d",)),
                            Accum(name="s", value="e", op="add"),
                        ),
                    ),
                    Loop(
                        axis=j,
                        body=(
                            Load("xj", input="X", index=(ExprVar("a0"), ExprVar("a2"))),
                            Assign("dj", ElementwiseOp("subtract"), ("xj", "m")),
                            Assign("ej", ElementwiseOp("exp"), ("dj",)),
                            Assign("y", ElementwiseOp("divide"), ("ej", "s")),
                            Write(output="out", index=(ExprVar("a0"), ExprVar("a2")), value="y"),
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
        (Param("X", "const float*", shape=(4, 32)),),
        Param("out", "float*", shape=(4, 32)),
    )
    outer = k_kern.body[0]
    assert isinstance(outer, Loop) and outer.axis.name == "a0"
    r_max, r_sum, out_loop = outer.body
    assert isinstance(r_max, Reduce)
    # Find the Accum in r_max's body.
    accums = [s for s in r_max.body if isinstance(s, Accum)]
    assert accums[0].op.name == "maximum"
    assert isinstance(r_sum, Reduce)
    assert isinstance(out_loop, Loop) and out_loop.axis.name == "a2"


def test_softmax_like_renders():
    op = _softmax_like_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "softmax",
            (Param("X", "const float*", shape=(4, 32)),),
            Param("out", "float*", shape=(4, 32)),
        )
    )
    assert "acc0 = fmaxf(acc0, " in src
    assert "expf(" in src
    assert "acc1 +=" in src
    assert " / acc1;" in src
    assert "out[a0 * 32 + a2] = " in src


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
                                    Load("a", input="A", index=(ExprVar("a0"), ExprVar("a2"))),
                                    Load("b", input="B", index=(ExprVar("a2"), ExprVar("a1"))),
                                    Assign("p", ElementwiseOp("multiply"), ("a", "b")),
                                    Accum(name="c", value="p", op="add"),
                                ),
                            ),
                            Write(output="out", index=(ExprVar("a0"), ExprVar("a1")), value="c"),
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
            Param("A", "const float*", shape=(8, 16)),
            Param("B", "const float*", shape=(16, 8)),
        ),
        Param("out", "float*", shape=(8, 8)),
    )
    m_loop = k.body[0]
    assert isinstance(m_loop, Loop) and m_loop.axis.name == "a0"
    n_loop = m_loop.body[0]
    assert isinstance(n_loop, Loop) and n_loop.axis.name == "a1"
    reduce_node = n_loop.body[0]
    assert isinstance(reduce_node, Reduce) and reduce_node.axis.name == "a2"
    assert isinstance(n_loop.body[1], Write)


def test_matmul_renders():
    op = _matmul_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "matmul",
            (
                Param("A", "const float*", shape=(8, 16)),
                Param("B", "const float*", shape=(16, 8)),
            ),
            Param("out", "float*", shape=(8, 8)),
        )
    )
    assert "float acc0 = 0.0f;" in src
    assert "for (int a2 = 0; a2 < 16; a2++) {" in src
    assert "float v0 = in0 * in1;" in src
    assert "acc0 += v0;" in src
    assert "out[a0 * 8 + a1] = acc0;" in src


# ---------------------------------------------------------------------------
# Top-level scalar Loads → prologue
# ---------------------------------------------------------------------------


def _scalar_prologue_loop_op() -> LoopOp:
    """A LoopOp with a scalar Load above the outer Loop — should land in prologue."""
    a0 = Axis("a0", 4)
    return LoopOp(
        body=(
            Load("eps", input="Eps", index=()),
            Loop(
                axis=a0,
                body=(
                    Load("x", input="X", index=(ExprVar("a0"),)),
                    Assign("y", ElementwiseOp("add"), ("x", "eps")),
                    Write(output="out", index=(ExprVar("a0"),), value="y"),
                ),
            ),
        )
    )


def test_scalar_prologue_lifts_to_kernel_prologue():
    op = _scalar_prologue_loop_op()
    k = lower_naive(
        op,
        "scalar_pro",
        (Param("X", "const float*", shape=(4,)), Param("Eps", "const float*", shape=(1,))),
        Param("out", "float*", shape=(4,)),
    )
    assert len(k.prologue) == 1
    # Loop IR leaf passes through; Load.name canonicalized to in0 (first Load in pre-order).
    assert isinstance(k.prologue[0], Load)
    assert k.prologue[0].name == "in0"
    assert isinstance(k.body[0], Loop)


def test_scalar_prologue_renders_above_body():
    op = _scalar_prologue_loop_op()
    src = render_kernel(
        lower_naive(
            op,
            "scalar_pro",
            (Param("X", "const float*", shape=(4,)), Param("Eps", "const float*", shape=(1,))),
            Param("out", "float*", shape=(4,)),
        )
    )
    # Empty Load.index → "Eps[0]".
    assert "float in0 = Eps[0];" in src
    assert src.index("float in0 = Eps[0];") < src.index("for (int a0 = 0; a0 < 4; a0++)")
