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
from deplodock.compiler.pipeline import TILE_PASSES, run_pipeline
from tests.compiler.passes.conftest import strip_rule_prefix


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# M and N must each exceed the per-axis matmul tile widths from
# ``tuning.thread_tile_shape`` so blockify_launch actually splits each
# axis into BLOCK + THREAD. Defaults are (BN=128, BM=64); pick 256²
# to leave headroom on both. K > BK so tile_matmul_k fires.
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
    run_pipeline(_make_plain_matmul(), TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "tile_matmul_k" in fired
    assert "blockify_launch" in fired


def test_elwise_lhs_matmul_fires_split_k_and_blockify(recording_dump):
    run_pipeline(_make_elwise_lhs_matmul(), TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "tile_matmul_k" in fired
    assert "blockify_launch" in fired


def test_two_elwise_lhs_matmul_fires_split_k_and_blockify(recording_dump):
    run_pipeline(_make_two_elwise_lhs_matmul(), TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "tile_matmul_k" in fired
    assert "blockify_launch" in fired


def test_matmul_then_elwise_fires_split_k_and_blockify(recording_dump):
    run_pipeline(_make_matmul_then_elwise(), TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "tile_matmul_k" in fired
    assert "blockify_launch" in fired


def test_pure_elementwise_does_not_fire_split_k(recording_dump):
    """No matmul-shaped reduce → ``tile_matmul_k`` must not fire, but
    ``blockify_launch`` still does (every TileOp gets launch geometry)."""
    g = Graph()
    _input(g, "x", (_M, _N))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    run_pipeline(g, TILE_PASSES, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "tile_matmul_k" not in fired
    assert "blockify_launch" in fired


# --- engine: RuleSkipped exception path ------------------------------


def test_rule_skipped_logs_reason_and_continues(caplog):
    """A rule that raises ``RuleSkipped`` is treated as no-op; the
    engine logs the reason at DEBUG and proceeds. We verify both:
    pipeline still completes, and the reason text is in the debug log.
    """
    import logging

    g = Graph()
    _input(g, "x", (_M, _N))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    # Pure elementwise → tile_matmul_k raises RuleSkipped (no matmul-shaped
    # reduce) and cooperative_reduce raises (no reduce Loop).
    with caplog.at_level(logging.DEBUG, logger="deplodock.compiler.pipeline.engine"):
        run_pipeline(g, TILE_PASSES)

    skip_messages = [r.message for r in caplog.records if "skipped" in r.message]
    assert any("tile_matmul_k" in m for m in skip_messages), skip_messages
    assert any("cooperative_reduce" in m for m in skip_messages), skip_messages


def test_strip_prefix_handles_letter_suffix():
    assert strip_rule_prefix("004_cooperative_reduce") == "cooperative_reduce"
    assert strip_rule_prefix("004b_cooperative_reduce") == "cooperative_reduce"
    assert strip_rule_prefix("001_tileify") == "tileify"
    assert strip_rule_prefix("tile_matmul_k") == "tile_matmul_k"
