"""Regression tests for ``%`` factor-cancellation in ``BinaryExpr.simplify``.

``_cancel_common_factors`` cancels a positive factor ``c`` common to both sides
of ``/`` // ``%``. For division ``(c·x)/(c·y) == x/y`` — the factor cancels
outright. For **modulo** the identity is ``(c·x) % (c·y) == c·(x % y)`` — the
factor scales the remainder, it does NOT vanish. The old code returned ``x % y``
(e.g. ``(f*8) % 64 → f % 8``), silently dropping the ``×c`` — which corrupted
e.g. vectorized cp.async chunk offsets (``8*(flat%8)`` rendered as ``flat%8`` →
overlapping 16-byte copies). These tests pin the identity numerically.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.expr import BinaryExpr, Literal, SimplifyCtx, Var


def _simplify(e):
    return e.simplify(SimplifyCtx(ranges={}))


@pytest.mark.parametrize(
    "coeff,modulus",
    [(8, 64), (2, 4), (3, 6), (4, 16), (16, 128), (2, 8), (5, 10)],
)
def test_mod_factor_cancellation_preserves_value(coeff: int, modulus: int):
    """``(f * coeff) % modulus`` must stay numerically equal after simplify,
    for every ``f`` — the modulo factor-cancel keeps the ``× gcd`` scale."""
    f = Var("f")
    e = BinaryExpr("%", BinaryExpr("*", f, Literal(coeff, "int")), Literal(modulus, "int"))
    s = _simplify(e)
    for v in range(0, 256):
        assert e.eval({"f": v}) == s.eval({"f": v}), f"mismatch at f={v}: {e.pretty()} vs {s.pretty()}"


def test_mod_cancel_specific_shape():
    """The exact shape that broke vectorized cp.async: ``(f*8) % 64`` ⇒
    ``8 * (f % 8)``, not ``f % 8``."""
    f = Var("f")
    e = BinaryExpr("%", BinaryExpr("*", f, Literal(8, "int")), Literal(64, "int"))
    s = _simplify(e)
    # f=1 ⇒ 8, not 1 (the old bug returned 1).
    assert s.eval({"f": 1}) == 8
    assert s.eval({"f": 9}) == 8  # (72 % 64)


def test_div_factor_cancellation_unchanged():
    """Division still cancels the factor outright — ``(f*8) / 64 == f / 8``."""
    f = Var("f")
    e = BinaryExpr("/", BinaryExpr("*", f, Literal(8, "int")), Literal(64, "int"))
    s = _simplify(e)
    for v in range(0, 256):
        assert e.eval({"f": v}) == s.eval({"f": v})
