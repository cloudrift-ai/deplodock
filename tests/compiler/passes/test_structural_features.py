"""Unit tests for structural feature extraction
(``loop/fusion/992_stamp_structural_features.structure_features``).

Hand-built ``Body`` fixtures (same style as ``tests/compiler/ir/stmt/
test_structural_key.py``) exercise the skeleton histogram, the extent-free
invariant, the ``S_ext_*`` extent block, and the ``S_dtype_*`` multiset; a
second group compiles real frontend graphs (triple-matmul, matmul + epilogue,
attention-like) through the loop passes and checks the stamped features.
"""

from __future__ import annotations

import importlib

from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt.blocks import Loop
from deplodock.compiler.ir.stmt.body import Body
from deplodock.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from deplodock.compiler.pipeline.knob import STRUCT_PREFIX
from deplodock.compiler.tensor import Tensor

# ``structure_features`` lives in the stamp pass (loaded under a bare stem).
_stamp = importlib.import_module("deplodock.compiler.pipeline.passes.loop.fusion.992_stamp_structural_features")
structure_features = _stamp.structure_features


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


# --- complex, compiled-through-the-loop-passes graphs ----------------------


def _fused_loops(graph: Graph):
    """Run the loop dialect (incl. the structural-feature stamp) and return
    ``(fused_graph, [LoopOp, ...])``."""
    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415
    from deplodock.compiler.pipeline.search.db import SearchDB  # noqa: PLC0415

    fused = Pipeline.build(LOOP_PASSES).run(graph, ctx=Context(compute_capability="sm_80"), db=SearchDB())
    return fused, [n.op for n in fused.nodes.values() if isinstance(n.op, LoopOp)]


def _matmul_chain(shapes: list[tuple[str, tuple[int, int]]], mms: list[tuple[str, str, str, tuple[int, int]]]) -> Graph:
    """Build a frontend matmul graph: ``shapes`` are (id, shape) inputs; ``mms``
    are (out_id, lhs_id, rhs_id, out_shape) MatmulOps applied in order."""
    from deplodock.compiler.ir.frontend.ir import MatmulOp  # noqa: PLC0415

    g = Graph()
    for nid, shape in shapes:
        g.add_node(InputOp(), [], Tensor(nid, shape), node_id=nid)
    for out, lhs, rhs, shape in mms:
        g.add_node(MatmulOp(), [lhs, rhs], Tensor(out, shape), node_id=out)
    g.inputs = [nid for nid, _ in shapes]
    g.outputs = [mms[-1][0]]
    return g


def test_triple_matmul_features_consistent_and_per_kernel_reduce():
    """A chained triple-matmul ``((a@b)@d)`` fuses to ≥2 matmul LoopOps; each
    carries stamped ``S_*`` features equal to :func:`structure_features`, has a
    K-reduce loop, and the distinct per-matmul K extents both show up."""
    g = _matmul_chain(
        [("a", (64, 128)), ("b", (128, 48)), ("d", (48, 80))],
        [("c", "a", "b", (64, 48)), ("e", "c", "d", (64, 80))],
    )
    fused, loops = _fused_loops(g)
    assert len(loops) >= 2, "two chained matmuls should not fuse into one kernel"
    reduce_maxes = set()
    for op in loops:
        struct = {k: v for k, v in op.knobs.items() if k.startswith(STRUCT_PREFIX)}
        assert struct == structure_features(op.body, fused), "stamped S_* must match structure_features"
        assert struct["S_n_reduce_loop"] >= 1.0, "each matmul kernel has a K reduce"
        reduce_maxes.add(struct["S_ext_reduce_max"])
    assert {128.0, 48.0} <= reduce_maxes, f"both matmul K extents should appear, got {reduce_maxes}"


def test_uncommon_shape_extents_land_in_features():
    """A non-power-of-2 matmul (48×80×96): the free/reduce extents land in the
    ``S_ext_*`` block (max free = N=80, reduce = K=96)."""
    g = _matmul_chain([("a", (48, 96)), ("b", (96, 80))], [("c", "a", "b", (48, 80))])
    fused, loops = _fused_loops(g)
    assert len(loops) == 1
    struct = {k: v for k, v in loops[0].knobs.items() if k.startswith(STRUCT_PREFIX)}
    assert struct == structure_features(loops[0].body, fused)
    assert struct["S_ext_reduce_max"] == 96.0
    assert struct["S_ext_free_max"] == 80.0


def test_matmul_features_differ_from_reduction():
    """A matmul kernel's structural skeleton differs from a pure reduction's —
    the multiply-accumulate inner vs the single reduce — so the prior can tell
    them apart from features alone."""
    g = _matmul_chain([("a", (48, 96)), ("b", (96, 80))], [("c", "a", "b", (48, 80))])
    _, loops = _fused_loops(g)
    matmul_skel = {k: v for k, v in structure_features(loops[0].body).items() if not k.startswith("S_ext_")}
    rms_skel = {k: v for k, v in structure_features(_rms_body()).items() if not k.startswith("S_ext_")}
    assert matmul_skel != rms_skel
