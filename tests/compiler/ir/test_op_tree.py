"""High-level op tree (``ir/tile/ops.py``): Map / Reduce lower to loop IR generically.

The kernel *is* the lowered tree — no per-kernel builder. A contraction is
``Reduce(+) ∘ Map(×)``; a plain reduction is ``Reduce`` over a direct operand load.
Validated by running the lowered ``LoopOp`` (cppyy, CPU) against numpy — the carrier
generates the ``Init`` + fold, the nested ops the lift, the ``TensorRef``s the loads.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Body, Loop, Write
from deplodock.compiler.ir.tile.ops import Map, Reduce, TensorRef, lower
from deplodock.compiler.pipeline.passes.lowering.tile._flash import flash_combine

ADD = ElementwiseImpl("add")
MUL = ElementwiseImpl("multiply")
MAX = ElementwiseImpl("maximum")
DIV = ElementwiseImpl("divide")


def _wrap(free_axes, cell, out_buf, out_index) -> LoopOp:
    """Wrap a per-cell stmt list in the free-axis loop nest + the output Write."""
    body = tuple(cell) + (Write(output=out_buf, index=out_index, value="acc"),)
    for ax in reversed(free_axes):
        body = (Loop(axis=ax, body=Body(body)),)
    return LoopOp(body=Body(body))


def test_contraction_op_tree_matches_numpy() -> None:
    M, K, N = 3, 4, 5
    m, k, n = Axis("m", Dim(M)), Axis("k", Dim(K)), Axis("n", Dim(N))
    red = Reduce(
        out="acc",
        axis=k,
        carrier=Accum(name="acc", value="p", op="add").as_monoid(),
        partials=(Map(out="p", op=MUL, args=(TensorRef("A", (Var("m"), Var("k"))), TensorRef("B", (Var("k"), Var("n"))))),),
        init_ops=(ADD,),
    )
    op = _wrap((m, n), lower(red), "C", (Var("m"), Var("n")))
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    got = np.asarray(op.forward(a, b)).reshape(M, N)
    np.testing.assert_allclose(got, a @ b, rtol=1e-5, atol=1e-5)


def test_plain_reduce_op_tree_matches_numpy() -> None:
    R, K = 4, 8
    r, k = Axis("r", Dim(R)), Axis("k", Dim(K))
    red = Reduce(
        out="acc",
        axis=k,
        carrier=Accum(name="acc", value="v", op="add").as_monoid(),
        partials=(TensorRef("x", (Var("r"), Var("k"))),),  # the partial is a direct load
        init_ops=(ADD,),
    )
    op = _wrap((r,), lower(red), "y", (Var("r"),))
    x = np.random.rand(R, K).astype(np.float32)
    got = np.asarray(op.forward(x)).reshape(R)
    np.testing.assert_allclose(got, x.sum(-1), rtol=1e-5, atol=1e-5)


def test_flash_op_tree_matches_softmax_qkv() -> None:
    """Flash attention as a pure op tree — a 3-state twisted carrier (m,l,O) folding
    over kv, whose score partial is a NESTED contraction (Σ_k Q·K), with the O/l
    projection as the root Map. Lowers generically (no build_flash_*) and matches
    softmax(QK^T)·V."""
    S, D = 4, 3
    i, d = Axis("i", Dim(S)), Axis("d", Dim(D))  # free: query row, value dim
    j, k = Axis("j", Dim(S)), Axis("k", Dim(D))  # reduce: kv (outer), head-dim (inner)
    score = Reduce(
        out="s",
        axis=k,
        carrier=Accum(name="s", value="qk", op="add").as_monoid(),
        partials=(Map(out="qk", op=MUL, args=(TensorRef("Q", (Var("i"), Var("k"))), TensorRef("K", (Var("j"), Var("k"))))),),
        init_ops=(ADD,),
    )
    flash = Map(  # the projection O/l is the root, wrapping the (m,l,O) fold
        out="acc",
        op=DIV,
        args=(
            Reduce(
                out="O",
                axis=j,
                carrier=flash_combine("m", "l", "O", "s", "v"),
                partials=(score, TensorRef("V", (Var("j"), Var("d")))),
                init_ops=(MAX, ADD, ADD),
            ),
            "l",
        ),
    )
    op = _wrap((i, d), lower(flash), "attn", (Var("i"), Var("d")))
    rng = np.random.default_rng(0)
    q, ky, v = (rng.standard_normal((S, D)).astype(np.float32) for _ in range(3))
    got = np.asarray(op.forward(q, ky, v)).reshape(S, D)
    smat = q @ ky.T
    p = np.exp(smat - smat.max(-1, keepdims=True))
    ref = (p / p.sum(-1, keepdims=True)) @ v
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
