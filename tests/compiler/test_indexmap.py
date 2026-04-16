"""Unit tests for IndexMapOp + coord_expr helpers."""

import pytest
import torch

from deplodock.compiler.backend.ir.expr import BinOp, Literal, Ternary, Var
from deplodock.compiler.coord_expr import (
    PLACEHOLDER_PREFIX,
    compose_index_maps,
    is_placeholder,
    placeholder,
    substitute,
)
from deplodock.compiler.ops import IndexMapOp, IndexSource

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
    result = substitute(expr, {placeholder(0).name: Var("row")})
    assert isinstance(result, Var) and result.name == "row"


def test_substitute_passes_unmapped_vars():
    expr = placeholder(1)
    result = substitute(expr, {placeholder(0).name: Var("row")})
    assert isinstance(result, Var) and result.name == placeholder(1).name


def test_substitute_leaves_literal_alone():
    expr = Literal(5, "int")
    assert substitute(expr, {}) is expr


def test_substitute_rewrites_binop():
    expr = placeholder(0) + Literal(7, "int")
    result = substitute(expr, {placeholder(0).name: Var("row")})
    assert isinstance(result, BinOp) and result.op == "+"
    assert isinstance(result.left, Var) and result.left.name == "row"
    assert isinstance(result.right, Literal) and result.right.value == 7


def test_substitute_rewrites_ternary():
    cond = placeholder(0).lt(Literal(64, "int"))
    expr = Ternary(cond, placeholder(0), placeholder(0) - Literal(64, "int"))
    result = substitute(expr, {placeholder(0).name: Var("col")})
    assert isinstance(result, Ternary)
    assert isinstance(result.if_true, Var) and result.if_true.name == "col"
    assert isinstance(result.if_false, BinOp) and result.if_false.op == "-"


# ---------- compose_index_maps ----------


def test_compose_identity_with_offset():
    # outer: identity (out_coord_0)
    # inner: x[out_coord_0 + 5]  (slice with start=5)
    outer = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0),)),),
    )
    inner = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(5, "int"),)),),
    )
    merged = compose_index_maps(outer, inner)
    assert merged.out_shape == (10,)
    assert len(merged.sources) == 1
    cm = merged.sources[0].coord_map
    assert len(cm) == 1
    # Composed coord should be (out_coord_0 + 5)
    assert isinstance(cm[0], BinOp) and cm[0].op == "+"
    assert isinstance(cm[0].left, Var) and cm[0].left.name == placeholder(0).name
    assert isinstance(cm[0].right, Literal) and cm[0].right.value == 5


def test_compose_two_offsets():
    # outer: x[out_coord_0 + 3]
    # inner: x[out_coord_0 + 5]
    # merged: x[(out_coord_0 + 3) + 5]
    outer = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(3, "int"),)),),
    )
    inner = IndexMapOp(
        out_shape=(10,),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0) + Literal(5, "int"),)),),
    )
    merged = compose_index_maps(outer, inner)
    cm = merged.sources[0].coord_map
    # Should be (out_coord_0 + 3) + 5 — outer's expr substituted into inner's placeholder
    assert isinstance(cm[0], BinOp) and cm[0].op == "+"
    assert isinstance(cm[0].right, Literal) and cm[0].right.value == 5
    inner_add = cm[0].left
    assert isinstance(inner_add, BinOp) and inner_add.op == "+"
    assert isinstance(inner_add.right, Literal) and inner_add.right.value == 3


def test_compose_transpose_with_slice():
    # outer (slice, dim=1, start=64): x[out_coord_0, out_coord_1 + 64]
    # inner (transpose, swap 0/1): x[out_coord_1, out_coord_0]
    # merged: composing reads inner's coord_map under outer's placeholder mapping
    outer = IndexMapOp(
        out_shape=(8, 64),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(0), placeholder(1) + Literal(64, "int"))),),
    )
    inner = IndexMapOp(
        out_shape=(8, 128),
        sources=(IndexSource(input_idx=0, coord_map=(placeholder(1), placeholder(0))),),
    )
    merged = compose_index_maps(outer, inner)
    cm = merged.sources[0].coord_map
    # inner read coord_map is (out_coord_1, out_coord_0); we substitute outer's
    # placeholder(d) := outer.coord_map[d]:
    #   placeholder(0) → outer.coord_map[0] = out_coord_0
    #   placeholder(1) → outer.coord_map[1] = out_coord_1 + 64
    # So merged coord_map = (out_coord_1 + 64, out_coord_0)
    assert isinstance(cm[0], BinOp) and cm[0].op == "+"
    assert isinstance(cm[0].left, Var) and cm[0].left.name == placeholder(1).name
    assert isinstance(cm[0].right, Literal) and cm[0].right.value == 64
    assert isinstance(cm[1], Var) and cm[1].name == placeholder(0).name


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
