"""Tests that matmul-related compiler rules fire on representative graphs.

Uses the ``recording_dump`` fixture (see ``conftest.py``) to collect
rule names of every rewrite, with numeric ordering prefix stripped, so
reordering rule files doesn't break these tests.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from tests.compiler.passes.conftest import strip_rule_prefix


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# M and N must each exceed the per-axis matmul tile widths from
# ``tuning.thread_tile_shape`` so launch_geometry actually splits each
# axis into BLOCK + THREAD. Defaults are (BN=128, BM=64); pick 256²
# to leave headroom on both. K > BK so chunk_matmul_k fires.
_M, _K, _N = 256, 64, 256


def _make_plain_matmul() -> Graph:
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _make_elwise_lhs_matmul() -> Graph:
    """``relu(A) @ B`` — one elementwise consumer feeds the matmul LHS."""
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=ElementwiseOp("relu"), inputs=["a"], output=Tensor("a1", (_M, _K)), node_id="a1")
    g.add_node(op=MatmulOp(), inputs=["a1", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _make_two_elwise_lhs_matmul() -> Graph:
    """``relu(exp(A)) @ B`` — chained elementwise on the LHS."""
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=ElementwiseOp("exp"), inputs=["a"], output=Tensor("a1", (_M, _K)), node_id="a1")
    g.add_node(op=ElementwiseOp("relu"), inputs=["a1"], output=Tensor("a2", (_M, _K)), node_id="a2")
    g.add_node(op=MatmulOp(), inputs=["a2", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _make_matmul_then_elwise() -> Graph:
    """``relu(A @ B)`` — elementwise consumes the matmul output."""
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("m", (_M, _N)), node_id="m")
    g.add_node(op=ElementwiseOp("relu"), inputs=["m"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def test_plain_matmul_fires_split_k_and_blockify(recording_dump):
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(_make_plain_matmul())
    fired = recording_dump.fired_rules("lowering/tile")
    # M14: planner owns matmul partition; ``launch_geometry`` (004) skips matmul.
    assert "partition_planner" in fired


def test_elwise_lhs_matmul_fires_split_k_and_blockify(recording_dump):
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(_make_elwise_lhs_matmul())
    fired = recording_dump.fired_rules("lowering/tile")
    # M14: planner owns matmul partition; ``launch_geometry`` (004) skips matmul.
    assert "partition_planner" in fired


def test_two_elwise_lhs_matmul_fires_split_k_and_blockify(recording_dump):
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(_make_two_elwise_lhs_matmul())
    fired = recording_dump.fired_rules("lowering/tile")
    # M14: planner owns matmul partition; ``launch_geometry`` (004) skips matmul.
    assert "partition_planner" in fired


def test_matmul_then_elwise_fires_split_k_and_blockify(recording_dump):
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(_make_matmul_then_elwise())
    fired = recording_dump.fired_rules("lowering/tile")
    # M14: planner owns matmul partition; ``launch_geometry`` (004) skips matmul.
    assert "partition_planner" in fired


def test_pure_elementwise_does_not_fire_split_k(recording_dump):
    """No matmul-shaped reduce → planner still partitions output axes
    (BLOCK/THREAD split), but no chunk_matmul_k / split-K fires."""
    g = Graph()
    _input(g, "x", (_M, _N))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "chunk_matmul_k" not in fired
    # M16: planner owns partition for both matmul and pointwise.
    assert "partition_planner" in fired


# --- engine: RuleSkipped exception path ------------------------------


def test_rule_skipped_logs_reason_and_continues(capsys):
    """A rule that raises ``RuleSkipped`` is treated as no-op; the
    engine emits the reason on stdout (so users can ``grep`` ``-vv``
    output without ``2>&1``) and proceeds. We verify both: pipeline
    still completes, and the reason text is on stdout.
    """
    import logging

    g = Graph()
    _input(g, "x", (_M, _N))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    # Pure elementwise → ``006a_register_tile_planned`` raises RuleSkipped
    # (no REG axes in body). The engine's ``debug_on`` gate keys off the
    # engine logger's level — bump that to DEBUG so ``emit`` fires.
    logging.getLogger("deplodock.compiler.pipeline").setLevel(logging.DEBUG)
    try:
        Pipeline.build(TILE_PASSES).run(g)
    finally:
        logging.getLogger("deplodock.compiler.pipeline").setLevel(logging.NOTSET)

    out = capsys.readouterr().out
    skip_messages = [ln for ln in out.splitlines() if ln.startswith("--- ") and "skipped" in ln]
    assert skip_messages, "expected at least one tile rule to log a skip reason"


def test_strip_prefix_handles_letter_suffix():
    assert strip_rule_prefix("005_cooperative_reduce") == "cooperative_reduce"
    assert strip_rule_prefix("005b_cooperative_reduce") == "cooperative_reduce"
    assert strip_rule_prefix("001_tileify") == "tileify"
    assert strip_rule_prefix("chunk_matmul_k") == "chunk_matmul_k"
