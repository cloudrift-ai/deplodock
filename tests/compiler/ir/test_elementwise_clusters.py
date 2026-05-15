"""Tests for op clustering — :func:`cluster_representative` collapses
ops that share a GPU compute unit so :attr:`Body.structural_key` can
treat (e.g.) ``add`` and ``subtract`` as the same kernel shape.

End-to-end clustering behaviour at the Body level is covered by
``tests/compiler/ir/stmt/test_structural_key.py``. These tests pin
down the mapping itself so a future contributor reshuffling the
``_OP_CLUSTERS`` table can't silently break the semantics autotune
relies on.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.elementwise import ElementwiseImpl, cluster_representative


@pytest.mark.parametrize(
    ("op_name", "rep_name"),
    [
        # FMA / cheap ALU
        ("add", "add"),
        ("sum", "add"),
        ("subtract", "add"),
        ("multiply", "add"),
        ("negative", "add"),
        ("abs", "add"),
        # Compare / predicate ALU
        ("maximum", "maximum"),
        ("minimum", "maximum"),
        ("amax", "maximum"),
        ("sign", "maximum"),
        # SFU integer-divide path
        ("divide", "divide"),
        ("true_divide", "divide"),
        ("floor_divide", "divide"),
        ("remainder", "divide"),
        ("mod", "divide"),
        ("reciprocal", "divide"),
        # SFU MUFU transcendental path
        ("sqrt", "exp"),
        ("rsqrt", "exp"),
        ("exp", "exp"),
        ("log", "exp"),
        ("sin", "exp"),
        ("cos", "exp"),
        ("tanh", "exp"),
        ("sigmoid", "exp"),
        ("silu", "exp"),
        ("gelu", "exp"),
        ("gelu_tanh", "exp"),
        ("pow", "exp"),
        ("relu", "exp"),
        # Passthrough
        ("copy", "copy"),
    ],
)
def test_cluster_representative_maps_known_op(op_name: str, rep_name: str) -> None:
    assert cluster_representative(ElementwiseImpl(op_name)) == ElementwiseImpl(rep_name)


def test_unknown_op_passes_through_unchanged() -> None:
    """Ops not in ``_OP_CLUSTERS`` are returned as-is — the table is
    best-effort and a missing entry must not crash."""
    # ``isnan`` is a numpy ufunc that's not in the cluster map. The
    # ElementwiseImpl construction succeeds (it's a valid numpy name),
    # and the representative is just the op itself.
    op = ElementwiseImpl("isnan")
    assert cluster_representative(op) == op
