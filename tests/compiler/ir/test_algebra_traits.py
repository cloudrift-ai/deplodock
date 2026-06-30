"""Behavior-preservation tests for the op algebra traits.

These pin the trait queries (``selecting`` / ``associative`` / ``commutative``
/ ``has_identity`` / semiring role / reduce render-spelling) to the values the
pre-refactor hardcoded sets gave, so the "stop switching on op names" refactor
is provably byte-identical. The hardcoded sets they replaced, verbatim:

- ``020_place_inits._SELECTING_OPS`` = {maximum, amax, minimum, max, min}
- ``010_split_register_axes._ASSOCIATIVE_REDUCE_OPS`` =
  {add, sum, maximum, amax, minimum, amin, multiply, prod}  (amin / min / max
  are numpy-reduce aliases that never appear as an ``Accum`` combine)
- ``recognize_flash._SUM`` = (add, sum), ``_MAX`` = (maximum, amax), plus the
  literal ``"multiply"`` product check
- the per-op CUDA / numpy reduce spellings (Accum.render, _binary_combine_expr,
  ReduceOp.forward / ScanOp.forward)
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.ir.elementwise import (
    _REDUCE_SPELLING,
    ElementwiseImpl,
    reduce_canon,
    reduce_spelling,
)

# Every op name the trait queries are asked about in the pipeline.
_REDUCE_OPS = ["add", "sum", "multiply", "prod", "maximum", "amax", "minimum"]

# --- selecting (was _SELECTING_OPS) -----------------------------------------

_OLD_SELECTING = frozenset({"maximum", "amax", "minimum", "max", "min"})


@pytest.mark.parametrize("name", sorted(_OLD_SELECTING | set(_REDUCE_OPS)))
def test_selecting_matches_old_set(name: str) -> None:
    assert ElementwiseImpl(name).selecting == (name in _OLD_SELECTING)


# --- associative / commutative (was duplicated _ASSOCIATIVE_REDUCE_OPS) ------


@pytest.mark.parametrize("name", _REDUCE_OPS)
def test_reduce_ops_are_associative_and_commutative(name: str) -> None:
    # Every op that the FK strip-mine / cross-thread combine reorders was in the
    # old _ASSOCIATIVE_REDUCE_OPS set — and stays reorderable via op.associative.
    op = ElementwiseImpl(name)
    assert op.associative
    assert op.commutative


@pytest.mark.parametrize("name", ["subtract", "divide"])
def test_non_associative_ops(name: str) -> None:
    assert not ElementwiseImpl(name).associative


# --- has_identity ------------------------------------------------------------


@pytest.mark.parametrize("name", _REDUCE_OPS)
def test_reduce_ops_have_identity(name: str) -> None:
    assert ElementwiseImpl(name).has_identity


# --- semiring role (was the literal "multiply" / "add" checks) ---------------


def test_semiring_product_is_multiply_only() -> None:
    assert ElementwiseImpl("multiply").semiring_product
    for name in ["add", "sum", "maximum", "minimum", "subtract"]:
        assert not ElementwiseImpl(name).semiring_product


def test_distributes_over_plus_times_only() -> None:
    mul, add = ElementwiseImpl("multiply"), ElementwiseImpl("add")
    assert mul.distributes_over(add)
    assert mul.distributes_over("sum")  # alias of add; reduce arg accepts a name
    assert not mul.distributes_over("maximum")
    assert not add.distributes_over(add)


# --- reduce canonicalization (was _COMBINE / the alias tuples) ---------------


@pytest.mark.parametrize(
    ("name", "canon"),
    [
        ("add", "add"),
        ("sum", "add"),
        ("multiply", "multiply"),
        ("prod", "multiply"),
        ("maximum", "maximum"),
        ("amax", "maximum"),
        ("fmax", "maximum"),
        ("minimum", "minimum"),
        ("fmin", "minimum"),
        ("copy", "copy"),  # aliasless → itself
    ],
)
def test_reduce_canon(name: str, canon: str) -> None:
    assert reduce_canon(name) == canon


# --- render spellings (was four duplicated dispatch tables) ------------------


@pytest.mark.parametrize(
    ("name", "compound", "intrinsic"),
    [
        ("add", "+=", None),
        ("sum", "+=", None),
        ("multiply", "*=", None),
        ("prod", "*=", None),
        ("maximum", None, "fmax"),
        ("amax", None, "fmax"),
        ("minimum", None, "fmin"),
    ],
)
def test_reduce_spelling_render(name: str, compound: str | None, intrinsic: str | None) -> None:
    sp = reduce_spelling(name)
    assert sp.compound == compound
    assert sp.intrinsic == intrinsic


def test_reduce_spelling_defaults_to_add() -> None:
    # Accum.render's legacy fallback: an unknown / non-reduce op renders as +=.
    assert reduce_spelling("copy").compound == "+="


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("add", np.array([6.0])),
        ("sum", np.array([6.0])),
        ("multiply", np.array([6.0])),
        ("prod", np.array([6.0])),
        ("maximum", np.array([3.0])),
        ("amax", np.array([3.0])),
        ("minimum", np.array([1.0])),
    ],
)
def test_np_reduce_matches(name: str, expected) -> None:
    a = np.array([1.0, 2.0, 3.0])
    out = _REDUCE_SPELLING[reduce_canon(name)].np_reduce(a, axis=0, keepdims=True)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("add", np.array([1.0, 3.0, 6.0])),
        ("multiply", np.array([1.0, 2.0, 6.0])),
        ("maximum", np.array([1.0, 2.0, 3.0])),
    ],
)
def test_np_scan_matches(name: str, expected) -> None:
    a = np.array([1.0, 2.0, 3.0])
    np_scan = _REDUCE_SPELLING[reduce_canon(name)].np_scan
    np.testing.assert_allclose(np_scan(a, axis=0), expected)


def test_semiring_as_monoid_lowers_identically() -> None:
    """A SEMIRING contraction is a MONOID with a ⊗ lift: lowering ``Semiring.as_monoid()`` (the
    degenerate ``id``-twist carrier whose partial is the operand product) reproduces the direct
    ``_lower_semiring`` byte-for-byte — the carrier bridge that lets a contraction flow through the
    shared carrier-generic reduce machinery rather than a contraction special case."""
    from deplodock.compiler.ir.axis import Axis
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.expr import Var
    from deplodock.compiler.ir.stmt import Accum, Load
    from deplodock.compiler.ir.stmt.algebra import Map, Semiring
    from deplodock.compiler.ir.stmt.base import pretty_body
    from deplodock.compiler.ir.tile.ops import lower

    dd = Axis("dd", 64)
    semi = Semiring(
        lift=ElementwiseImpl("multiply"),
        fold=Accum(name="sacc", value="qk", op=ElementwiseImpl("add")),
        operands=(
            Map(body=[Load(name="q_e", input="q", index=(Var("m"), Var("dd")))]),
            Map(body=[Load(name="k_e", input="k", index=(Var("kv"), Var("dd")))]),
        ),
        reduce_axis=dd,
    )
    direct = "\n".join(pretty_body(lower(semi)))
    via_monoid = "\n".join(pretty_body(lower(semi.as_monoid())))
    assert direct == via_monoid, f"as_monoid diverges:\n--- direct ---\n{direct}\n--- monoid ---\n{via_monoid}"
