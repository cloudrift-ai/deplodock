"""High-level op tree (``ir/tile/ops.py``): Map / Reduce lower to loop IR generically.

The kernel *is* the lowered tree — no per-kernel builder. A contraction is
``Reduce(+)`` over a ``Map`` (the ``×`` lift, a stmt body); a plain reduction is
``Reduce`` over a direct ``TensorRef`` load. Validated by running the lowered ``LoopOp``
(cppyy, CPU) against numpy — the carrier generates the ``Init`` + fold, a ``Map`` is the
lift stmts, the ``TensorRef``s the loads.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from deplodock.compiler.ir.tile.ops import Map, Reduce, TensorRef, lower
from deplodock.compiler.pipeline.passes.lowering.tile._flash import flash_combine

ADD = ElementwiseImpl("add")
MAX = ElementwiseImpl("maximum")


def _wrap(free_axes, cell) -> LoopOp:
    """Wrap a per-cell stmt list (already ending in its output ``Write``) in the
    free-axis loop nest."""
    body = tuple(cell)
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
        partials=(
            Map(
                [
                    Load(name="a_e", input="A", index=(Var("m"), Var("k"))),
                    Load(name="b_e", input="B", index=(Var("k"), Var("n"))),
                    Assign(name="p", op="multiply", args=("a_e", "b_e")),  # the ⊗ lift
                ]
            ),
        ),
        init_ops=(ADD,),
    )
    cell = (*lower(red), Write(output="C", index=(Var("m"), Var("n")), value="acc"))
    op = _wrap((m, n), cell)
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
    cell = (*lower(red), Write(output="y", index=(Var("r"),), value="acc"))
    op = _wrap((r,), cell)
    x = np.random.rand(R, K).astype(np.float32)
    got = np.asarray(op.forward(x)).reshape(R)
    np.testing.assert_allclose(got, x.sum(-1), rtol=1e-5, atol=1e-5)


def test_flash_op_tree_matches_softmax_qkv() -> None:
    """Flash attention as a pure op tree — a 3-state twisted carrier (m,l,O) folding
    over kv, whose score partial is a NESTED contraction (Σ_k Q·K); the kernel root is
    the O/l projection (a Map whose last stmt is the Write). Lowers generically (no
    build_flash_*) and matches softmax(QK^T)·V."""
    S, D = 4, 3
    i, d = Axis("i", Dim(S)), Axis("d", Dim(D))  # free: query row, value dim
    j, k = Axis("j", Dim(S)), Axis("k", Dim(D))  # reduce: kv (outer), head-dim (inner)
    score = Reduce(
        out="s",
        axis=k,
        carrier=Accum(name="s", value="qk", op="add").as_monoid(),
        partials=(
            Map(
                [
                    Load(name="q_e", input="Q", index=(Var("i"), Var("k"))),
                    Load(name="k_e", input="K", index=(Var("j"), Var("k"))),
                    Assign(name="qk", op="multiply", args=("q_e", "k_e")),
                ]
            ),
        ),
        init_ops=(ADD,),
    )
    flash = Reduce(
        out="O",
        axis=j,
        carrier=flash_combine("m", "l", "O", "s", "v"),
        partials=(score, TensorRef("V", (Var("j"), Var("d")))),
        init_ops=(MAX, ADD, ADD),
    )
    cell = (*lower(flash), Assign(name="acc", op="divide", args=("O", "l")), Write(output="attn", index=(Var("i"), Var("d")), value="acc"))
    op = _wrap((i, d), cell)
    rng = np.random.default_rng(0)
    q, ky, v = (rng.standard_normal((S, D)).astype(np.float32) for _ in range(3))
    got = np.asarray(op.forward(q, ky, v)).reshape(S, D)
    smat = q @ ky.T
    p = np.exp(smat - smat.max(-1, keepdims=True))
    ref = (p / p.sum(-1, keepdims=True)) @ v
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
