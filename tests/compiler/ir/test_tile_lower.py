"""Loop IR → Tile IR lowering tests.

``lower_naive`` is mechanical:
- Loop IR leaves (``Load`` / ``Assign`` / ``Select`` / ``Write`` /
  ``Accum`` / ``Cond``) pass through.
- Loop IR ``Loop`` becomes ``Reduce`` (Accums in body) or stays as a
  ``Loop``.
- The outer free-Loop chain is stripped into an ``Enclosure(thread_axes=...)``
  at the kernel root; leading non-Loop stmts (scalar Loads) sit above
  the Enclosure in ``TileOp.body``.

These tests cover the structural shape and the round-trip render. The
renderer takes a shapes dict (``{buf_name: shape}``) — production code
builds it from the surrounding graph; tests pass it as a literal dict.
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
    Enclosure,
    Reduce,
    TileOp,
)
from deplodock.compiler.ir.tile.lower import lower_naive
from deplodock.compiler.ir.tile.render import render_tileop

# ---------------------------------------------------------------------------
# Pointwise add — outer free chain (a0, a1) strips into Enclosure
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


_POINTWISE_SHAPES = {"A": (4, 8), "B": (4, 8), "out": (4, 8)}


def test_pointwise_add_structure():
    k = lower_naive(_pointwise_add_loop_op(), "add")

    assert isinstance(k, TileOp)
    assert k.name == "add"
    # Body inputs / outputs derived from Load.input / Write.output.
    assert k.inputs == ("A", "B")
    assert k.output_bufs == ("out",)
    # Outer free-Loop chain (a0, a1) → Enclosure.thread_axes.
    enc = k.body[0]
    assert isinstance(enc, Enclosure)
    assert tuple(a.name for a in enc.thread_axes) == ("a0", "a1")
    assert enc.block_axes == ()
    assert isinstance(enc.body[0], Load) and enc.body[0].input == "A"
    assert isinstance(enc.body[3], Write) and enc.body[3].output == "out"


def test_pointwise_add_renders():
    src = render_tileop(lower_naive(_pointwise_add_loop_op(), "add"), shapes=_POINTWISE_SHAPES)
    assert "void add(const float* A, const float* B, float* out)" in src
    assert "long long tid = blockIdx.x * blockDim.x + threadIdx.x;" in src
    assert "if (tid < 32) {" in src
    assert "int a1 = tid % 8;" in src
    assert "int a0 = tid / 8;" in src
    assert "float in0 = A[a0 * 8 + a1];" in src
    assert "float v0 = in0 + in1;" in src
    assert "out[a0 * 8 + a1] = v0;" in src
    assert "for (int a0" not in src
    assert "for (int a1" not in src


# ---------------------------------------------------------------------------
# Reduce — sum along k. Only a0 strips (a1 is the reduce axis).
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
    k = lower_naive(_row_sum_loop_op(), "row_sum")
    assert k.inputs == ("X",)
    assert k.output_bufs == ("out",)
    enc = k.body[0]
    assert isinstance(enc, Enclosure)
    assert tuple(a.name for a in enc.thread_axes) == ("a0",)
    reduce_node = enc.body[0]
    assert isinstance(reduce_node, Reduce) and reduce_node.axis.name == "a1"
    assert isinstance(reduce_node.body[0], Load)
    assert isinstance(reduce_node.body[1], Accum)
    assert reduce_node.body[1].op.name == "add"
    assert isinstance(enc.body[1], Write)


def test_row_sum_renders():
    src = render_tileop(
        lower_naive(_row_sum_loop_op(), "row_sum"),
        shapes={"X": (4, 32), "out": (4,)},
    )
    assert "void row_sum(const float* X, float* out)" in src
    assert "if (tid < 4) {" in src
    assert "int a0 = tid;" in src
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
    k = lower_naive(_softmax_like_loop_op(), "softmax")
    enc = k.body[0]
    assert isinstance(enc, Enclosure)
    # Only a0 strips — the level beneath has 3 sibling Loops, breaks the chain.
    assert tuple(a.name for a in enc.thread_axes) == ("a0",)
    r_max, r_sum, out_loop = enc.body
    assert isinstance(r_max, Reduce)
    accums = [s for s in r_max.body if isinstance(s, Accum)]
    assert accums[0].op.name == "maximum"
    assert isinstance(r_sum, Reduce)
    assert isinstance(out_loop, Loop) and out_loop.axis.name == "a2"


def test_softmax_like_renders():
    src = render_tileop(
        lower_naive(_softmax_like_loop_op(), "softmax"),
        shapes={"X": (4, 32), "out": (4, 32)},
    )
    assert "if (tid < 4) {" in src
    assert "acc0 = fmaxf(acc0, " in src
    assert "expf(" in src
    assert "acc1 +=" in src
    assert " / acc1;" in src
    assert "out[a0 * 32 + a2] = " in src


# ---------------------------------------------------------------------------
# Matmul — outer chain (a0, a1) strips, leaving Reduce(a2) + Write inside.
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
    k = lower_naive(_matmul_loop_op(), "matmul")
    assert k.inputs == ("A", "B")
    enc = k.body[0]
    assert isinstance(enc, Enclosure)
    assert tuple(a.name for a in enc.thread_axes) == ("a0", "a1")
    reduce_node = enc.body[0]
    assert isinstance(reduce_node, Reduce) and reduce_node.axis.name == "a2"
    assert isinstance(enc.body[1], Write)


def test_matmul_renders():
    src = render_tileop(
        lower_naive(_matmul_loop_op(), "matmul"),
        shapes={"A": (8, 16), "B": (16, 8), "out": (8, 8)},
    )
    assert "void matmul(const float* A, const float* B, float* out)" in src
    assert "if (tid < 64) {" in src
    assert "int a1 = tid % 8;" in src
    assert "int a0 = tid / 8;" in src
    assert "float acc0 = 0.0f;" in src
    assert "for (int a2 = 0; a2 < 16; a2++) {" in src
    assert "acc0 += v0;" in src
    assert "out[a0 * 8 + a1] = acc0;" in src


# ---------------------------------------------------------------------------
# Top-level scalar Loads sit at the start of TileOp.body, above any Enclosure
# ---------------------------------------------------------------------------


def _scalar_prologue_loop_op() -> LoopOp:
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


def test_scalar_prologue_at_body_start():
    k = lower_naive(_scalar_prologue_loop_op(), "scalar_pro")
    # Leading scalar Load passes through; outer Loop(a0) gets stripped into Enclosure.
    assert isinstance(k.body[0], Load)
    assert k.body[0].name == "in0"
    enc = k.body[1]
    assert isinstance(enc, Enclosure)
    assert tuple(a.name for a in enc.thread_axes) == ("a0",)


def test_scalar_prologue_renders_above_body():
    src = render_tileop(
        lower_naive(_scalar_prologue_loop_op(), "scalar_pro"),
        shapes={"X": (4,), "Eps": (1,), "out": (4,)},
    )
    # Scalar load above the tid guard (every thread sees the same value).
    assert "float in0 = Eps[0];" in src
    assert src.index("float in0 = Eps[0];") < src.index("long long tid =")
    # Per-thread work runs inside the tid guard, not in a for-loop.
    assert "for (int a0" not in src
    assert "if (tid < 4) {" in src
