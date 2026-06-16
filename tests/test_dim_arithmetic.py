"""``Dim`` arithmetic — eager-fold via ``Expr.simplify``.

Covers:
- static-static folds to a static Dim (matches today's int math byte-for-byte)
- static-symbolic / symbolic-symbolic compose into BinaryExpr-backed Dims
- algebraic identities (``*1``, ``+0``) collapse
- ``Dim(32) == 32`` and ``Dim('s') == 's'`` ergonomics preserved
- composite Dims compare structurally, never to bare int/str
- Dim is hashable for use as dict key / set member / Axis(extent=...)
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var


def test_static_static_arithmetic_folds():
    assert Dim(32) * Dim(64) == Dim(2048)
    assert (Dim(32) * Dim(64)).expr == Literal(2048, "int")
    assert Dim(7) + Dim(5) == Dim(12)
    assert Dim(20) - Dim(3) == Dim(17)
    assert Dim(64) // Dim(2) == Dim(32)
    assert Dim(10) % Dim(3) == Dim(1)


def test_symbolic_compose_keeps_binary_expr():
    assert (Dim("s") * Dim(2)).expr == BinaryExpr("*", Var("s"), Literal(2, "int"))
    assert (Dim("s") + Dim("s")).expr == BinaryExpr("+", Var("s"), Var("s"))


def test_algebraic_identities_collapse():
    assert (Dim("s") * Dim(1)).expr == Var("s")
    assert (Dim("s") + Dim(0)).expr == Var("s")
    assert (Dim("s") * Dim(0)).expr == Literal(0, "int")
    assert (Dim(1) * Dim("s")).expr == Var("s")


def test_int_dim_arithmetic_ergonomic():
    assert Dim("s") * 2 == Dim("s") * Dim(2)
    assert 2 * Dim("s") == Dim(2) * Dim("s")
    assert Dim("s") + 0 == Dim("s")


def test_ceil_div_static_folds_to_int_ceil():
    # Matches the ``-(-E // b)`` idiom (positive extents) the masked-tile sites used.
    for ext, div in [(65, 16), (64, 16), (151669, 128), (1, 1), (512, 128), (100, 32)]:
        got = Dim(ext).ceil_div(div)
        assert got.is_static and got.as_static() == -(-ext // div), (ext, div)


def test_ceil_div_symbolic_builds_composite_expr():
    # ``(seq + (b - 1)) // b`` — the launch resolver evals it from sym_values.
    assert Dim("seq").ceil_div(16).expr == BinaryExpr("//", BinaryExpr("+", Var("seq"), Literal(15, "int")), Literal(16, "int"))
    # Degenerate divisor 1 collapses to the bare axis (matches the symbolic-serial passthrough).
    assert Dim("seq").ceil_div(1).expr == Var("seq")


def test_equality_with_bare_int_and_str():
    assert Dim(32) == 32
    assert Dim("s") == "s"
    assert Dim(32) != 33
    assert Dim("s") != "t"
    assert Dim(32) != "32"  # int Dim never matches string
    assert Dim("s") != 5  # symbolic Dim never matches int


def test_composite_does_not_compare_to_bare():
    composite = Dim("s") * Dim(2)
    assert composite != 32
    assert composite != "s*2"
    # structural eq still works against another composite
    assert composite == Dim("s") * Dim(2)


def test_dim_hashable_and_dict_key():
    d = {Dim(32): "a", Dim("s"): "b", Dim("s") * Dim(2): "c"}
    assert d[Dim(32)] == "a"
    assert d[Dim("s")] == "b"
    assert d[Dim("s") * Dim(2)] == "c"


def test_value_back_compat_atomic_only():
    assert Dim(32).value == 32
    assert Dim("s").value == "s"
    import pytest

    with pytest.raises(TypeError, match="composite"):
        _ = (Dim("s") * Dim(2)).value


def test_as_static_raises_on_symbolic():
    import pytest

    assert Dim(32).as_static() == 32
    with pytest.raises(TypeError):
        Dim("s").as_static()
    with pytest.raises(TypeError):
        (Dim("s") * Dim(2)).as_static()


def test_str_renders_transparently():
    # Pretty IR (``for i in 0..32``) and structural keys depend on this.
    assert str(Dim(32)) == "32"
    assert str(Dim("s")) == "s"


def test_dim_from_expr():
    assert Dim(Var("s")) == Dim("s")
    assert Dim(Literal(32, "int")) == Dim(32)


def test_dim_idempotent():
    assert Dim(Dim(32)) == Dim(32)
    assert Dim(Dim("s")) == Dim("s")
