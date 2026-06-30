"""``build_skeleton`` over hand-built kernels — the five recognized kinds.

Mirrors the op-tree fixtures in ``tests/compiler/ir/test_op_tree.py``: build the algebra node,
wrap it in its ``*Kernel`` via ``kernel_for``, and assert the skeleton facts the scheduler reads.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Semiring
from deplodock.compiler.ir.tile import Placement, kernel_for
from deplodock.compiler.ir.tile.ops import Map
from deplodock.compiler.pipeline.passes.lowering.tile._flash import flash_combine
from deplodock.compiler.pipeline.passes.lowering.tile._skeleton import build_skeleton

MUL = ElementwiseImpl("multiply")


def _matmul_semiring(m, k, n) -> Semiring:
    return Semiring(
        lift=MUL,
        fold=Accum(name="acc", value="p", op="add"),
        operands=(
            Map(body=[Load(name="a_e", input="A", index=(Var(m.name), Var(k.name)))]),
            Map(body=[Load(name="b_e", input="B", index=(Var(k.name), Var(n.name)))]),
        ),
        reduce_axis=k,
    )


def test_pointwise_map_has_no_reduce():
    node = Map(body=[Load(name="v", input="x", index=(Var("m"),)), Assign(name="y", op="exp", args=("v",))])
    skel = build_skeleton(kernel_for(node, Placement(free=(Axis("m", 8),))), (Axis("m", 8),))
    assert skel.root.reduce is None
    assert skel.root.children == ()


def test_plain_reduce_is_degenerate_coop_eligible():
    r, k = Axis("r", Dim(4)), Axis("k", Dim(8))
    red = replace(
        Accum(name="acc", value="v", op="add").as_monoid(),
        partial=(Map(body=[Load(name="v", input="x", index=(Var("r"), Var("k")))]),),
        axis=k,
    )
    skel = build_skeleton(kernel_for(red, Placement(free=(r,))), (r,))
    red_axis = skel.root.reduce
    assert red_axis is not None
    assert red_axis.contraction is False
    assert red_axis.coop_eligible is True
    assert red_axis.carrier.twist.family == "id"
    assert red_axis.axis is k


def test_bindable_contraction_normalizes_k_and_binds():
    m, k, n = Axis("m", Dim(3)), Axis("k", Dim(4)), Axis("n", Dim(5))
    semi = _matmul_semiring(m, k, n)
    skel = build_skeleton(kernel_for(semi, Placement(free=(m, n))), (m, n))
    red = skel.root.reduce
    assert red.contraction is True
    assert red.coop_eligible is False
    assert red.axis is k
    assert red.carrier.twist.family == "id"  # Semiring.as_monoid() — the K-as-reduce normalization
    assert red.binding is not None
    assert red.binding.a.load.input == "A" and red.binding.a.role == "a"
    assert red.binding.b.load.input == "B" and red.binding.b.role == "b"
    assert red.binding.b_trans is False
    assert red.binding.acc == "acc"


def test_unbindable_contraction_stores_none_without_raising():
    m, k, n = Axis("m", Dim(3)), Axis("k", Dim(4)), Axis("n", Dim(5))
    semi = _matmul_semiring(m, k, n)
    # A 1-axis grid can't carry (m, n) — semiring_binding raises; the builder must swallow it.
    skel = build_skeleton(kernel_for(semi, Placement(free=(m,))), (m,))
    assert skel.root.reduce.contraction is True
    assert skel.root.reduce.binding is None


def test_flash_nests_an_inner_contraction_scope():
    s, d = Dim(4), Dim(3)
    i, dd = Axis("i", s), Axis("d", d)
    j, k = Axis("j", s), Axis("k", d)
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
    flash = Map(source=flash_monoid, body=[Assign(name="O__proj", op="divide", args=("O", "l"))])
    skel = build_skeleton(kernel_for(flash, Placement(free=(i, dd))), (i, dd))
    outer = skel.root.reduce
    assert outer.contraction is False
    assert outer.coop_eligible is True
    assert outer.carrier.twist.family == "exp"  # the streaming-softmax twist
    assert outer.axis is j
    # The score Semiring rides a nested scope; the value Map (no reduce) does not.
    assert len(skel.root.children) == 1
    inner = skel.root.children[0].reduce
    assert inner.contraction is True
    assert inner.axis is k
    assert inner.carrier.twist.family == "id"
