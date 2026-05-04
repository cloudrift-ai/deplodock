"""Tests that reduction-related compiler rules fire on representative
graphs, plus scoped unit tests for the cooperative-reduce rule's
multi-Accum independence check.

Uses the ``recording_dump`` fixture (see ``conftest.py``) to collect
rule names of every rewrite, with numeric ordering prefix stripped, so
reordering rule files doesn't break these tests.
"""

from __future__ import annotations

import importlib

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Accum, Assign, Load
from deplodock.compiler.ir.tensor.ir import ReduceOp
from deplodock.compiler.pipeline import TILE_PASSES, run_pipeline

_accums_independent = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.004_cooperative_reduce")._accums_independent


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- cooperative_reduce firing on frontend graphs --------------------
# Triggers on single-buffer reductions whose first reduce-axis extent is
# ≥ BLOCK_SIZE (256). split_matmul_k skips these (single Load), so they
# reach cooperative_reduce unchanged.


_M, _K, _N = 32, 32, 32


def test_long_axis_sum_fires_cooperative_reduce(recording_dump):
    """``sum(x, axis=-1)`` with K=256 → cooperative_reduce fires; matmul-
    shape rule (split_matmul_k) does not (single-buffer reduce)."""
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    run_pipeline(g, TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" in fired
    assert "split_matmul_k" not in fired


def test_short_axis_sum_does_not_fire_cooperative_reduce(recording_dump):
    """K=32 < BLOCK_SIZE → cooperative_reduce does not fire."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    run_pipeline(g, TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "cooperative_reduce" not in fired


def test_matmul_does_not_fire_cooperative_reduce(recording_dump):
    """Matmul-shape reduce → split_matmul_k handles K splitting; the
    cooperative-reduce strategy is for single-buffer reductions and
    must not fire here."""
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    run_pipeline(g, TILE_PASSES, dump=recording_dump)
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
