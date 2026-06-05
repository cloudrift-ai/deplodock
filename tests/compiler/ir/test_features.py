"""Unit tests for structural feature extraction (``ir/features.py``).

Hand-built ``Body`` fixtures (same style as ``tests/compiler/ir/stmt/
test_structural_key.py``) exercise the skeleton histogram, the extent-free
invariant, the ``S_ext_*`` extent block, and the ``S_dtype_*`` multiset.
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.features import STRUCT_PREFIX, structure_features
from deplodock.compiler.ir.stmt.blocks import Loop
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from deplodock.compiler.tensor import Tensor


def _rms_body(ext_i: int = 8, ext_k: int = 64) -> Body:
    """Free ``i`` over reduce ``k``: sum of squares of ``a`` → ``o``. One
    reduce (RMSNorm-like)."""
    return Body(
        (
            Loop(
                axis=Axis("i", ext_i),
                body=(
                    Loop(
                        axis=Axis("k", ext_k),
                        body=(
                            Load(name="x", input="a", index=(Var("i"), Var("k"))),
                            Assign(name="sq", op="multiply", args=("x", "x")),
                            Accum(name="s", value="sq", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("i"),), value="s"),
                ),
            ),
        )
    )


def _softmax_body() -> Body:
    """Free ``i`` with two reduce loops (max then sum) — two reduces (softmax-like)."""
    return Body(
        (
            Loop(
                axis=Axis("i", 8),
                body=(
                    Loop(
                        axis=Axis("k", 64),
                        body=(
                            Load(name="x", input="a", index=(Var("i"), Var("k"))),
                            Accum(name="m", value="x", op=ElementwiseImpl("max")),
                        ),
                    ),
                    Loop(
                        axis=Axis("k2", 64),
                        body=(
                            Load(name="x2", input="a", index=(Var("i"), Var("k2"))),
                            Accum(name="s", value="x2", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("i"),), value="s"),
                ),
            ),
        )
    )


def test_all_keys_struct_prefixed():
    feats = structure_features(_rms_body())
    assert feats and all(k.startswith(STRUCT_PREFIX) for k in feats)


def test_skeleton_histogram():
    feats = structure_features(_rms_body())
    assert feats["S_n_load"] == 1.0
    assert feats["S_n_distinct_input"] == 1.0
    assert feats["S_n_write"] == 1.0
    assert feats["S_n_accum"] == 1.0
    assert feats["S_n_assign"] == 1.0
    assert feats["S_pw_multiply"] == 1.0
    assert feats["S_reduce_add"] == 1.0
    assert feats["S_n_loop"] == 2.0
    assert feats["S_n_reduce_loop"] == 1.0
    assert feats["S_n_free_loop"] == 1.0
    assert feats["S_loop_depth"] == 2.0


def test_reduce_multiset_distinguishes_one_vs_two_reduce():
    one = structure_features(_rms_body())
    two = structure_features(_softmax_body())
    assert one["S_n_reduce_loop"] == 1.0
    assert two["S_n_reduce_loop"] == 2.0
    assert two["S_reduce_max"] == 1.0 and two["S_reduce_add"] == 1.0
    assert "S_reduce_max" not in one
    assert one != two


def test_skeleton_is_extent_free():
    """Two bodies differing only in axis extents share every non-``S_ext_`` key;
    only the ``S_ext_*`` block differs."""
    small = structure_features(_rms_body(ext_i=8, ext_k=64))
    big = structure_features(_rms_body(ext_i=16, ext_k=128))
    skel_small = {k: v for k, v in small.items() if not k.startswith("S_ext_")}
    skel_big = {k: v for k, v in big.items() if not k.startswith("S_ext_")}
    assert skel_small == skel_big
    assert small["S_ext_free_prod"] != big["S_ext_free_prod"]
    assert small["S_ext_reduce_prod"] != big["S_ext_reduce_prod"]


def test_extents_split_free_vs_reduce():
    feats = structure_features(_rms_body(ext_i=8, ext_k=64))
    assert feats["S_ext_free_prod"] == 8.0
    assert feats["S_ext_free_max"] == 8.0
    assert feats["S_ext_n_free_axis"] == 1.0
    assert feats["S_ext_reduce_prod"] == 64.0
    assert feats["S_ext_reduce_max"] == 64.0
    assert feats["S_ext_n_reduce_axis"] == 1.0
    assert feats["S_ext_n_symbolic_axis"] == 0.0


def test_symbolic_axis_counted_and_excluded_from_prod():
    body = Body(
        (
            Loop(
                axis=Axis("s", Dim("seq_len")),
                body=(
                    Loop(
                        axis=Axis("k", 64),
                        body=(
                            Load(name="x", input="a", index=(Var("s"), Var("k"))),
                            Accum(name="acc", value="x", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("s"),), value="acc"),
                ),
            ),
        )
    )
    feats = structure_features(body)
    assert feats["S_ext_n_symbolic_axis"] == 1.0
    # The symbolic free axis is excluded from the product → empty free → 1.0.
    assert feats["S_ext_free_prod"] == 1.0
    assert feats["S_ext_n_free_axis"] == 0.0
    assert feats["S_ext_reduce_prod"] == 64.0


def test_dtype_multiset_needs_graph():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (8, 64), "f16"), node_id="a")
    feats = structure_features(_rms_body(), g)
    assert feats["S_dtype_f16"] == 1.0
    # Without a graph there are no dtype features.
    assert not any(k.startswith("S_dtype_") for k in structure_features(_rms_body()))
