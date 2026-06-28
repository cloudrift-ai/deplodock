"""High-level op tree (``ir/tile/ops.py``): Map / Monoid / Semiring lower to loop IR.

The kernel *is* the lowered tree — no per-kernel builder. A contraction is a
``Semiring`` (the ``×`` lift over its operands, folded by ``+``); a plain reduction is a
degenerate ``Monoid`` over a ``Map`` partial; flash is the ``(m, l, O)`` ``Monoid`` whose
score partial is a NESTED ``Semiring``. Validated by running the lowered ``LoopOp``
(cppyy, CPU) against numpy — each node generates its ``Init`` + fold, a ``Map`` is the
lift stmts, and partial / operand sources are ``Map`` / ``Monoid`` / ``Semiring`` nodes.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Semiring, Write
from deplodock.compiler.ir.tile.ops import Map, lower
from deplodock.compiler.pipeline.passes.lowering.tile._flash import flash_combine

MUL = ElementwiseImpl("multiply")


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
    # A matmul as a first-class Semiring node: ⊗ = multiply over two one-Load Map
    # operands, ⊕ = the additive fold over k.
    semi = Semiring(
        lift=MUL,
        fold=Accum(name="acc", value="p", op="add"),
        operands=(
            Map(body=[Load(name="a_e", input="A", index=(Var("m"), Var("k")))]),
            Map(body=[Load(name="b_e", input="B", index=(Var("k"), Var("n")))]),
        ),
        reduce_axis=k,
    )
    cell = (*lower(semi), Write(output="C", index=(Var("m"), Var("n")), value="acc"))
    op = _wrap((m, n), cell)
    a = np.random.rand(M, K).astype(np.float32)
    b = np.random.rand(K, N).astype(np.float32)
    got = np.asarray(op.forward(a, b)).reshape(M, N)
    np.testing.assert_allclose(got, a @ b, rtol=1e-5, atol=1e-5)


def test_plain_reduce_op_tree_matches_numpy() -> None:
    R, K = 4, 8
    r, k = Axis("r", Dim(R)), Axis("k", Dim(K))
    # A plain sum as a degenerate (self-contained) Monoid: its one partial source is a
    # one-Load Map (the direct operand read).
    red = replace(
        Accum(name="acc", value="v", op="add").as_monoid(),
        partial=(Map(body=[Load(name="v", input="x", index=(Var("r"), Var("k")))]),),
        axis=k,  # out ("acc") + seed (identity 0) derived from the carrier
    )
    cell = (*lower(red), Write(output="y", index=(Var("r"),), value="acc"))
    op = _wrap((r,), cell)
    x = np.random.rand(R, K).astype(np.float32)
    got = np.asarray(op.forward(x)).reshape(R)
    np.testing.assert_allclose(got, x.sum(-1), rtol=1e-5, atol=1e-5)


def test_flash_op_tree_matches_softmax_qkv() -> None:
    """Flash attention as a pure op tree — a 3-state twisted ``Monoid`` (m,l,O) folding
    over kv, whose score partial is a NESTED ``Semiring`` (Σ_k Q·K). The O/l projection
    is a ``Map`` *over* that Monoid (``project ∘ reduce``), so the kernel root is that Map
    + the Write. Lowers generically and matches softmax(QK^T)·V."""
    S, D = 4, 3
    i, d = Axis("i", Dim(S)), Axis("d", Dim(D))  # free: query row, value dim
    j, k = Axis("j", Dim(S)), Axis("k", Dim(D))  # reduce: kv (outer), head-dim (inner)
    score = Semiring(
        lift=MUL,
        fold=Accum(name="s", value="qk", op="add"),
        operands=(
            Map(body=[Load(name="q_e", input="Q", index=(Var("i"), Var("k")))]),
            Map(body=[Load(name="k_e", input="K", index=(Var("j"), Var("k")))]),
        ),
        reduce_axis=k,
    )
    flash_monoid = replace(
        flash_combine("m", "l", "O", "s", "v"),
        partial=(score, Map(body=[Load(name="v", input="V", index=(Var("j"), Var("d")))])),
        axis=j,
    )
    # The O/l projection is a Map *over* the reduce (project ∘ reduce).
    flash = Map(source=flash_monoid, body=[Assign(name="O__proj", op="divide", args=("O", "l"))])
    cell = (*lower(flash), Write(output="attn", index=(Var("i"), Var("d")), value="O__proj"))
    op = _wrap((i, d), cell)
    rng = np.random.default_rng(0)
    q, ky, v = (rng.standard_normal((S, D)).astype(np.float32) for _ in range(3))
    got = np.asarray(op.forward(q, ky, v)).reshape(S, D)
    smat = q @ ky.T
    p = np.exp(smat - smat.max(-1, keepdims=True))
    ref = (p / p.sum(-1, keepdims=True)) @ v
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-4)
