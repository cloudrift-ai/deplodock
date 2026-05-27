"""Unit tests for IndexMapOp + coord_expr helpers."""

import pytest
import torch

from deplodock.compiler.ir.expr import (
    PLACEHOLDER_PREFIX,
    BinaryExpr,
    Literal,
    TernaryExpr,
    Var,
    is_placeholder,
    placeholder,
)
from deplodock.compiler.ir.tensor.ir import IndexMapOp, IndexSource

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------- placeholder helpers ----------


def test_placeholder_uses_prefix():
    assert placeholder(0).name == f"{PLACEHOLDER_PREFIX}0"
    assert placeholder(3).name == f"{PLACEHOLDER_PREFIX}3"


def test_is_placeholder_unbound():
    assert is_placeholder(placeholder(0))
    assert is_placeholder(placeholder(7))
    assert not is_placeholder(Var("row"))


def test_is_placeholder_specific_axis():
    assert is_placeholder(placeholder(2), d=2)
    assert not is_placeholder(placeholder(2), d=3)


# ---------- substitute ----------


def test_substitute_replaces_var():
    expr = placeholder(0)
    result = expr.substitute({placeholder(0).name: Var("row")})
    assert isinstance(result, Var) and result.name == "row"


def test_substitute_passes_unmapped_vars():
    expr = placeholder(1)
    result = expr.substitute({placeholder(0).name: Var("row")})
    assert isinstance(result, Var) and result.name == placeholder(1).name


def test_substitute_leaves_literal_alone():
    expr = Literal(5, "int")
    assert expr.substitute({}) is expr


def test_substitute_rewrites_binop():
    expr = placeholder(0) + Literal(7, "int")
    result = expr.substitute({placeholder(0).name: Var("row")})
    assert isinstance(result, BinaryExpr) and result.op == "+"
    assert isinstance(result.left, Var) and result.left.name == "row"
    assert isinstance(result.right, Literal) and result.right.value == 7


def test_substitute_rewrites_ternary():
    cond = placeholder(0).lt(Literal(64, "int"))
    expr = TernaryExpr(cond, placeholder(0), placeholder(0) - Literal(64, "int"))
    result = expr.substitute({placeholder(0).name: Var("col")})
    assert isinstance(result, TernaryExpr)
    assert isinstance(result.if_true, Var) and result.if_true.name == "col"
    assert isinstance(result.if_false, BinaryExpr) and result.if_false.op == "-"


# ---------- IndexMapOp.is_identity ----------


def test_identity_detects_pure_passthrough():
    op = IndexMapOp(
        out_shape=(8, 128),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1))),),
    )
    assert op.is_identity((8, 128))


def test_identity_rejects_shape_change():
    op = IndexMapOp(
        out_shape=(1, 8, 128),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1), placeholder(2))),),
    )
    assert not op.is_identity((8, 128))


def test_identity_rejects_offset():
    op = IndexMapOp(
        out_shape=(8,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(1, "int"),)),),
    )
    assert not op.is_identity((8,))


def test_identity_rejects_select():
    op = IndexMapOp(
        out_shape=(8,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0),), select=placeholder(0).lt(Literal(4, "int"))),),
    )
    assert not op.is_identity((8,))


def test_identity_rejects_multisource():
    op = IndexMapOp(
        out_shape=(8,),
        sources=(
            IndexSource(input_idx=0, coord_map=(placeholder(0),)),
            IndexSource(input_idx=1, coord_map=(placeholder(0),)),
        ),
    )
    assert not op.is_identity((8,))


def test_indexmap_infer_output_shape_returns_out_shape():
    op = IndexMapOp(
        out_shape=(4, 5, 6),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1), placeholder(2))),),
    )
    assert op.infer_output_shape([(99, 99, 99)]) == (4, 5, 6)


def test_indexmap_sources_round_trip_through_json():
    """``IndexMapOp.sources`` (IndexSource dataclasses holding Exprs) survive
    ``Graph.to_dict`` → ``from_dict``. Regression: they used to be stringified
    by ``json.dumps(default=str)`` and reloaded as ``str``, crashing
    ``030_lift_indexmap`` (``'str' has no attribute 'input_idx'``) on
    ``run --ir`` recompiles."""
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    src0 = IndexSource(
        input_idx=0,
        coord_map=(placeholder(0), placeholder(1) + Literal(2, "int")),
        select=placeholder(1).lt(Literal(8, "int")),
    )
    src1 = IndexSource(input_idx=1, coord_map=(placeholder(0), placeholder(1)))
    g.add_node(IndexMapOp(out_shape=(4, 16), sources=(src0, src1)), ["a", "b"], Tensor("c", (4, 16)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]

    sources = Graph.from_dict(g.to_dict()).nodes["c"].op.sources
    assert all(isinstance(s, IndexSource) for s in sources)
    assert [s.input_idx for s in sources] == [0, 1]
    assert sources[0].coord_map == src0.coord_map and sources[1].coord_map == src1.coord_map
    assert sources[0].select == src0.select and sources[1].select is None
