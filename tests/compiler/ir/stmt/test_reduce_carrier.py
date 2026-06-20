"""``ReduceCarrier`` protocol + ``ElementwiseImpl`` algebraic traits.

The trait-agnostic machinery keys off ``isinstance(s, ReduceCarrier)`` instead
of an ``isinstance(s, (Accum, Mma))`` ladder, and rules that care about the
combine query ``combine_op()``'s traits rather than matching op names. These
tests pin both halves of that contract.
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.stmt import Accum, Loop, Mma, ReduceCarrier, StridedLoop
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY

# --------------------------------------------------------------------------- #
# ElementwiseImpl traits
# --------------------------------------------------------------------------- #


def test_associative_and_identity_traits():
    for name in ("add", "sum", "multiply", "prod", "maximum", "minimum", "amax"):
        op = ElementwiseImpl(name)
        assert op.associative, name
        assert op.has_identity, name
        assert op.identity is not None, name


def test_non_reassociable_ops_lack_traits():
    for name in ("subtract", "divide"):
        op = ElementwiseImpl(name)
        assert not op.associative, name
        assert not op.has_identity, name
        assert op.identity is None, name


def test_max_is_associative_but_existing_commutative_unchanged():
    mx = ElementwiseImpl("maximum")
    assert mx.commutative and mx.associative
    assert ElementwiseImpl("maximum").identity == -1e30


# --------------------------------------------------------------------------- #
# ReduceCarrier protocol — Accum
# --------------------------------------------------------------------------- #


def test_accum_is_reduce_carrier():
    a = Accum(name="acc", value="v", op=ElementwiseImpl("maximum"))
    assert isinstance(a, ReduceCarrier)
    assert a.carried_names() == ("acc",)
    assert a.partial_deps() == ("v",)  # excludes the implicit carried read
    assert a.combine_op().name == "maximum"
    assert a.combine_op().associative


# --------------------------------------------------------------------------- #
# ReduceCarrier protocol — Mma
# --------------------------------------------------------------------------- #


def _mma() -> Mma:
    return Mma(a="a0", b="b0", c="acc", atom=ATOM_REGISTRY["mma_m16n8k16_f16"])


def test_mma_is_reduce_carrier():
    m = _mma()
    assert isinstance(m, ReduceCarrier)
    assert m.carried_names() == ("acc",)
    assert m.partial_deps() == ("a0", "b0")  # operands, not the carried c
    # The tensor-core accumulation is an additive fold — reported so split-K
    # reassociation gates see the same algebra as a scalar sum Accum.
    op = m.combine_op()
    assert op.name == "add"
    assert op.associative and op.commutative and op.has_identity


# --------------------------------------------------------------------------- #
# is_reduce keys off the protocol — Accum AND Mma make a loop a reduce loop
# --------------------------------------------------------------------------- #


def _kloop(stmt):
    return Loop(axis=Axis(name="k", extent=Dim(16)), body=(stmt,))


def test_loop_is_reduce_for_accum():
    assert _kloop(Accum(name="acc", value="v")).is_reduce


def test_loop_is_reduce_for_mma():
    # Before the protocol, Loop.is_reduce checked Accum only — an Mma-bearing
    # loop read as non-reduce. The protocol unifies the two.
    assert _kloop(_mma()).is_reduce


def test_strided_loop_is_reduce_for_mma():
    sl = StridedLoop(axis=Axis(name="k", extent=Dim(16)), start=Var("t"), step=Literal(32, "int"), body=(_mma(),))
    assert sl.is_reduce


def test_non_carrier_loop_is_not_reduce():
    from deplodock.compiler.ir.stmt import Load

    assert not _kloop(Load(name="x", input="buf", index=(Var("k"),))).is_reduce
