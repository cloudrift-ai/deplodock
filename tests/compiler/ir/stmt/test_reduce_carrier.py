"""``ReduceCarrier`` protocol + ``ElementwiseImpl`` algebraic traits.

The trait-agnostic machinery keys off ``isinstance(s, ReduceCarrier)`` instead
of an ``isinstance(s, (Accum, Mma))`` ladder, and rules that care about the
combine read the carrier's ``associative`` / ``commutative`` / ``has_identity``
traits rather than matching op names. These tests pin both halves of that
contract.
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Literal, Var
from deplodock.compiler.ir.sigma import Sigma
from deplodock.compiler.ir.stmt import Accum, FlashCombine, Loop, Mma, ReduceCarrier, StridedLoop
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
    # Traits forward to the scalar op — a `maximum` Accum is associative.
    assert a.associative and a.commutative and a.has_identity


def test_accum_traits_forward_to_op():
    # A non-reassociable op flips the carrier's traits — they are not hardcoded.
    a = Accum(name="acc", value="v", op=ElementwiseImpl("subtract"))
    assert not a.associative and not a.has_identity


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
    # The tensor-core accumulation is an additive fold — its traits match a
    # scalar sum Accum so split-K reassociation gates see the same algebra.
    assert m.associative and m.commutative and m.has_identity


# --------------------------------------------------------------------------- #
# ReduceCarrier protocol — FlashCombine (the tuple-monoid LSE combine)
# --------------------------------------------------------------------------- #


def _flash() -> FlashCombine:
    return FlashCombine(
        state=("m_i", "l_i", "O_i"),
        partial=("tile_max", "tile_sum", "tile_pv"),
        axes=("kv",),
    )


def test_flash_combine_is_reduce_carrier():
    f = _flash()
    assert isinstance(f, ReduceCarrier)
    # The carried triple is the def surface; the carried read is implicit.
    assert f.carried_names() == ("m_i", "l_i", "O_i")
    assert f.defines() == ("m_i", "l_i", "O_i")
    # Only this tile's partial contribution is a same-scope read.
    assert f.deps() == ("tile_max", "tile_sum", "tile_pv")
    assert f.partial_deps() == ("tile_max", "tile_sum", "tile_pv")
    # The LSE monoid: associative AND commutative with identity — the property
    # that makes split-KV / cooperative-combine legal.
    assert f.associative and f.commutative and f.has_identity


def test_flash_combine_rewrite_renames_state_and_partial():
    f = _flash()
    renamed = f.rewrite(lambda n: f"{n}__r")
    assert isinstance(renamed, FlashCombine)
    assert renamed.state == ("m_i__r", "l_i__r", "O_i__r")
    assert renamed.partial == ("tile_max__r", "tile_sum__r", "tile_pv__r")
    assert renamed.axes == ("kv",)  # identity Sigma leaves the axis untouched


def test_flash_combine_rewrite_threads_axis_split():
    # A σ that splits the KV axis (kv → kv_o*BK + kv_i) re-targets the carrier's
    # reduction axes onto the sub-axes, exactly like Accum/Mma.
    from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var

    sigma = Sigma({"kv": BinaryExpr("+", BinaryExpr("*", Var("kv_o"), Literal(8, "int")), Var("kv_i"))})
    renamed = _flash().rewrite(lambda n: n, sigma)
    assert set(renamed.axes) == {"kv_o", "kv_i"}


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


def test_loop_is_reduce_for_flash_combine():
    # A FlashCombine-bearing loop is a reduce loop via the same protocol hook —
    # no isinstance ladder over (Accum, Mma, FlashCombine) needed.
    assert _kloop(_flash()).is_reduce


def test_non_carrier_loop_is_not_reduce():
    from deplodock.compiler.ir.stmt import Load

    assert not _kloop(Load(name="x", input="buf", index=(Var("k"),))).is_reduce
