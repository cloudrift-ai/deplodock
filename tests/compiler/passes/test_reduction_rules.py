"""Tests that reduction-related compiler rules fire on representative
graphs, plus scoped unit tests for the cooperative-reduce rule's
multi-Accum independence check.

Uses the ``recording_dump`` fixture (see ``conftest.py``) to collect
rule names of every rewrite, with numeric ordering prefix stripped, so
reordering rule files doesn't break these tests.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.kernel.ir import TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Accum, Assign, Load
from deplodock.compiler.ir.tensor.ir import ReduceOp
from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.tile._helpers import accums_independent as _accums_independent


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- cooperative_reduce firing on frontend graphs --------------------
# Triggers on single-buffer reductions whose first reduce-axis extent is
# ≥ WARP_SIZE (32) — the threshold dropped from BLOCK_SIZE so softmax /
# rmsnorm rows in the 32–128 range get a parallel reduce instead of
# every thread redundantly walking the row. chunk_matmul_k skips
# single-buffer reduces, so they reach cooperative_reduce unchanged.


_M, _K, _N = 32, 32, 32

_COOP_XFAIL = pytest.mark.xfail(reason="cooperative-reduce removed; planner-driven replacement pending", strict=False)


def test_long_axis_sum_fires_cooperative_reduce(recording_dump):
    """``sum(x, axis=-1)`` with K=256 → cooperative_reduce fires; matmul-
    shape rule (chunk_matmul_k) does not (single-buffer reduce)."""
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" in fired
    assert "chunk_matmul_k" not in fired


def test_short_axis_sum_does_not_fire_cooperative_reduce(recording_dump):
    """K=16 < WARP_SIZE → cooperative_reduce does not fire (too small
    to stage a meaningful cross-thread tree-halve)."""
    g = Graph()
    _input(g, "x", (4, 16))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" not in fired


def test_warp_sized_axis_fires_cooperative_reduce(recording_dump):
    """K=32 ≥ WARP_SIZE → cooperative_reduce fires with a 32-thread
    cooperative block (the gate was lowered from BLOCK_SIZE to
    WARP_SIZE so K∈[32, BLOCK_SIZE) gets a parallel reduce instead of
    every thread redundantly walking the row sequentially)."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" in fired


def test_warp_cooperative_skips_stage_inputs(recording_dump):
    """K=32 → cooperative tile has 32 threads (one warp); stage_inputs
    must skip so the kernel stays smem-free (the WarpShuffle combine
    in materialize_tile is register-only and L1 absorbs repeat loads)."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" in fired
    assert "stage_inputs" not in fired


def _kernel_body_stmts(g: Graph):
    out: list = []
    for node in g.nodes.values():
        body = getattr(node.op, "body", None)
        if body is None:
            continue
        for s in body.iter():
            out.append(s)
    return out


def test_warp_cooperative_emits_warpshuffle(recording_dump):
    """K=32 cooperative tile → ``materialize_tile._emit_combine`` picks
    the warp path: ``WarpShuffle`` Stmt appears, no ``TreeHalve``."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES, dump=recording_dump).run(g)
    stmts = _kernel_body_stmts(out)
    assert any(isinstance(s, WarpShuffle) for s in stmts)
    assert not any(isinstance(s, TreeHalve) for s in stmts)


def test_block_cooperative_emits_hierarchical_reduce(recording_dump):
    """K=256 cooperative tile → ``materialize_tile._emit_combine`` picks
    the hierarchical path: ``WarpShuffle`` reduces lanes within each
    warp, then a tiny ``TreeHalve(length=n_warps)`` collapses across
    warps. Both Stmts are present; the TreeHalve's length is far
    smaller than the legacy ``length=n_threads`` form."""
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES, dump=recording_dump).run(g)
    stmts = _kernel_body_stmts(out)
    warp_shuffles = [s for s in stmts if isinstance(s, WarpShuffle)]
    tree_halves = [s for s in stmts if isinstance(s, TreeHalve)]
    assert warp_shuffles, "expected WarpShuffle for the per-warp combine"
    assert tree_halves, "expected TreeHalve for the cross-warp combine"
    # Cross-warp TreeHalve runs over n_warps partials, not n_threads.
    assert all(t.length < 256 for t in tree_halves), [t.length for t in tree_halves]


@pytest.mark.xfail(reason="v1 cooperative path skips smem staging (no reuse with sole K_c THREAD axis)", strict=False)
def test_block_cooperative_still_uses_stage_inputs(recording_dump):
    """K=256 → cooperative tile has BLOCK_SIZE threads; stage_inputs
    still fires (the smem stage avoids redundant DRAM reads when the
    row is too wide to keep register-resident across the warp)."""
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" in fired
    assert "stage_inputs" in fired


def test_matmul_does_not_fire_cooperative_reduce(recording_dump):
    """Matmul-shape reduce → chunk_matmul_k handles K splitting; the
    cooperative-reduce strategy is for single-buffer reductions and
    must not fire here."""
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" not in fired


# --- _accums_independent: scoped unit tests --------------------------
# The cooperative-reduce rule permits multiple Accums in one reduce
# loop iff none of them transitively reads a prior Accum's running
# value. Below we exercise the helper directly (constructing a full
# multi-Accum reduce graph at the frontend level requires a fusion pass
# that doesn't exist yet — the helper is the contract that rule will
# use once such graphs reach it).


def _load(name: str, src: str) -> Load:
    return Load(name=name, input=src, index=(Var("k"),))


def test_accums_independent_single():
    body = (_load("v", "x"), Accum(name="acc", value="v", op="add"))
    assert _accums_independent(body) is True


def test_accums_independent_two_independent():
    """sum + sum_of_squares: both read inputs (Loads), neither reads the
    other's running value → independent."""
    body = (
        _load("v", "x"),
        Assign(name="vv", op="multiply", args=("v", "v")),
        Accum(name="s", value="v", op="add"),
        Accum(name="s2", value="vv", op="add"),
    )
    assert _accums_independent(body) is True


def test_accums_dependent_via_direct_read():
    """Second Accum reads the first Accum's running value directly."""
    body = (
        _load("v", "x"),
        Accum(name="acc_max", value="v", op="max"),
        Accum(name="acc_sum", value="acc_max", op="add"),
    )
    assert _accums_independent(body) is False


def test_accums_dependent_via_assign_chain():
    """Online softmax pattern: ``e = exp(v - acc_max); acc_sum += e``.
    The Assign chain transitively taints ``e``, so ``acc_sum`` is
    rejected as dependent."""
    body = (
        _load("v", "x"),
        Accum(name="acc_max", value="v", op="max"),
        Assign(name="d", op="subtract", args=("v", "acc_max")),
        Assign(name="e", op="exp", args=("d",)),
        Accum(name="acc_sum", value="e", op="add"),
    )
    assert _accums_independent(body) is False
