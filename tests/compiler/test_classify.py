"""Tests for the unified-emit classifier (``_classify.classify``).

``LoopOp.__post_init__`` canonicalizes axis names to ``a0``/``a1``/... and SSA
names to ``in0``/``v0``/... so fixtures below use those post-normalization
spellings.
"""

from __future__ import annotations

from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Accum, Assign, Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline.passes.lowering.kernel._classify import classify

# ---------------------------------------------------------------------------
# Pointwise
# ---------------------------------------------------------------------------


def test_pointwise_add_2d():
    """C[a0, a1] = A[a0, a1] + B[a0, a1] — no reductions, |live|=0."""
    a0, a1 = Axis("a0", 4), Axis("a1", 8)
    op = LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=a1,
                        body=(
                            Load("a", source=0, index=(Var("a0"), Var("a1"))),
                            Load("b", source=1, index=(Var("a0"), Var("a1"))),
                            Assign("c", ElementwiseOp("add"), ("a", "b")),
                            Write(output=0, index=(Var("a0"), Var("a1")), value="c"),
                        ),
                    ),
                ),
            ),
        )
    )
    sig = classify(op)
    assert sig is not None
    assert sig.kind == "pointwise"
    assert [a.name for a in sig.free_axes] == ["a0", "a1"]
    assert sig.live_axes == ()
    assert sig.reduce_blocks == ()


def test_pointwise_1d():
    """Degenerate 1D pointwise still classifies."""
    a0 = Axis("a0", 32)
    op = LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Load("x", source=0, index=(Var("a0"),)),
                    Assign("y", ElementwiseOp("exp"), ("x",)),
                    Write(output=0, index=(Var("a0"),), value="y"),
                ),
            ),
        )
    )
    sig = classify(op)
    assert sig is not None
    assert sig.kind == "pointwise"
    assert [a.name for a in sig.free_axes] == ["a0"]


# ---------------------------------------------------------------------------
# Per-row reduction
# ---------------------------------------------------------------------------


def _rmsnorm_like_op() -> LoopOp:
    """y[a0, a2] = x[a0, a2] * rsqrt(sum_a1 x[a0,a1]^2 + eps).

    Matches what fusion produces: scalar prologue Load, outer free
    Loop(a0), one reduce Loop(a1), interlude, output free Loop(a2) + Write.
    """
    a0, a1, a2 = Axis("a0", 4), Axis("a1", 32), Axis("a2", 32)
    return LoopOp(
        body=(
            Load("eps", source=1, index=()),
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=a1,
                        body=(
                            Load("xk", source=0, index=(Var("a0"), Var("a1"))),
                            Assign("sq", ElementwiseOp("multiply"), ("xk", "xk")),
                            Accum(name="s", value="sq", op="add"),
                        ),
                    ),
                    Assign("se", ElementwiseOp("add"), ("s", "eps")),
                    Assign("r", ElementwiseOp("rsqrt"), ("se",)),
                    Loop(
                        axis=a2,
                        body=(
                            Load("xj", source=0, index=(Var("a0"), Var("a2"))),
                            Assign("y", ElementwiseOp("multiply"), ("xj", "r")),
                            Write(output=0, index=(Var("a0"), Var("a2")), value="y"),
                        ),
                    ),
                ),
            ),
        )
    )


def test_rmsnorm_like_classifies():
    sig = classify(_rmsnorm_like_op())
    assert sig is not None
    assert sig.kind == "per_row_reduce"
    assert [a.name for a in sig.free_axes] == ["a0", "a2"]
    assert [a.name for a in sig.live_axes] == ["a0"]
    assert len(sig.reduce_blocks) == 1
    assert sig.reduce_blocks[0].accum.op.name == "add"
    assert len(sig.reduce_blocks[0].staged_loads) == 1
    # Only one Load in the reduce body (multiplicand/multiplier SSA dedupe during normalize),
    # and it references the reduce axis a1 so it gets staged.
    assert sig.reduce_blocks[0].staged_loads[0].source == 0
    # interlude has two Assigns (se, r) — post-normalization SSA names may differ.
    assert len(sig.interlude) == 2
    assert sig.output_axis is not None and sig.output_axis.name == "a2"


def _softmax_like_op() -> LoopOp:
    """y[a0, a2] = exp(x[a0,a2] - m) / s, with m = max_k, s = sum exp."""
    a0, a1 = Axis("a0", 4), Axis("a1", 32)
    a2 = Axis("a2", 32)
    return LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=a1,
                        body=(
                            Load("xk", source=0, index=(Var("a0"), Var("a1"))),
                            Accum(name="m", value="xk", op="maximum"),
                        ),
                    ),
                    Loop(
                        axis=a1,
                        body=(
                            Load("xk2", source=0, index=(Var("a0"), Var("a1"))),
                            Assign("d", ElementwiseOp("subtract"), ("xk2", "m")),
                            Assign("e", ElementwiseOp("exp"), ("d",)),
                            Accum(name="s", value="e", op="add"),
                        ),
                    ),
                    Loop(
                        axis=a2,
                        body=(
                            Load("xj", source=0, index=(Var("a0"), Var("a2"))),
                            Assign("dj", ElementwiseOp("subtract"), ("xj", "m")),
                            Assign("ej", ElementwiseOp("exp"), ("dj",)),
                            Assign("y", ElementwiseOp("divide"), ("ej", "s")),
                            Write(output=0, index=(Var("a0"), Var("a2")), value="y"),
                        ),
                    ),
                ),
            ),
        )
    )


def test_softmax_like_classifies():
    sig = classify(_softmax_like_op())
    assert sig is not None
    assert sig.kind == "per_row_reduce"
    assert [a.name for a in sig.free_axes] == ["a0", "a2"]
    assert [a.name for a in sig.live_axes] == ["a0"]
    assert len(sig.reduce_blocks) == 2
    assert {b.accum.op.name for b in sig.reduce_blocks} == {"maximum", "add"}
    assert sig.output_axis is not None and sig.output_axis.name == "a2"


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------


def test_live_axes_disagree_rejected():
    """Two reduce blocks with different live-axis sets → classifier rejects."""
    a0, a1 = Axis("a0", 4), Axis("a1", 32)
    a2 = Axis("a2", 8)
    op = LoopOp(
        body=(
            Loop(
                axis=a0,
                body=(
                    Loop(
                        axis=a1,
                        body=(
                            Load("xk", source=0, index=(Var("a0"), Var("a1"))),
                            Accum(name="m", value="xk", op="add"),
                        ),
                    ),
                    Loop(
                        axis=a1,
                        body=(
                            Load("yk", source=1, index=(Var("a1"),)),
                            Accum(name="n", value="yk", op="add"),
                        ),
                    ),
                    Loop(
                        axis=a2,
                        body=(
                            Assign("z", ElementwiseOp("add"), ("m", "n")),
                            Write(output=0, index=(Var("a0"), Var("a2")), value="z"),
                        ),
                    ),
                ),
            ),
        )
    )
    assert classify(op) is None


def test_matmul_shape_classifies_as_per_row_reduce():
    """Matmul-like body (free, free, reduce + direct Write) classifies with no output Loop.

    Each thread owns one (m, n) output element and walks the reduce serially —
    correct but slow for matmul (the annotated tile template is still
    preferred for matmul-annotated nodes; this is the unannotated fallback).
    """
    m, n, k = Axis("a0", 4), Axis("a1", 4), Axis("a2", 8)
    op = LoopOp(
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
                                    Load("a", source=0, index=(Var("a0"), Var("a2"))),
                                    Load("b", source=1, index=(Var("a2"), Var("a1"))),
                                    Assign("p", ElementwiseOp("multiply"), ("a", "b")),
                                    Accum(name="c", value="p", op="add"),
                                ),
                            ),
                            Write(output=0, index=(Var("a0"), Var("a1")), value="c"),
                        ),
                    ),
                ),
            ),
        )
    )
    sig = classify(op)
    assert sig is not None
    assert sig.kind == "per_row_reduce"
    assert sig.output_axis is None
    assert [a.name for a in sig.live_axes] == ["a0", "a1"]
