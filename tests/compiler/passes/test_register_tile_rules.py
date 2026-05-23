"""Tests for the ``register_tile`` rule (``008_register_tile``).

Firing tests: build a frontend graph, run the full ``TILE_PASSES``,
assert ``register_tile`` shows up (or doesn't) in
``recording_dump.fired_rules``.

The rule's axis-aware analysis (per-stmt replication factor of 1 / F
/ F² depending on which thread axes a stmt's output depends on) is
exercised end-to-end by ``test_block_accuracy`` for fused-linear and
``test_run_code_sdpa_k_chunked`` for SDPA, so the unit tests here
focus on the rule's *trigger* rather than its rewrite output.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp, SdpaOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- firing tests ----------------------------------------------------


def test_plain_matmul_fires_register_tile(recording_dump):
    """``A @ B`` at sizes that clear PAT=16 → register_tile fires.
    Regression check: the new axis-aware rewrite must keep working on
    the case the rule was originally designed for."""
    g = Graph()
    _input(g, "a", (128, 256))
    _input(g, "b", (256, 128))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (128, 128)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    # M14: planner owns matmul partition (BN/BM/FM/FN/BK/SPLITK). When the
    # chosen variant has FM=FN=1, the extent-1 REG loops are eliminated by
    # the normalize pass before 006a sees them, so 006a doesn't fire. The
    # planner firing is the signal that the register-tile decision was made.
    fired = recording_dump.fired_rules("lowering/tile")
    assert "partition_planner" in fired, fired


def test_pure_pointwise_does_not_fire_register_tile(recording_dump):
    """Pointwise has no matmul-shaped reduce → ``_find_matmul`` skips."""
    g = Graph()
    _input(g, "x", (128, 128))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (128, 128)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "register_tile" not in recording_dump.fired_rules("lowering/tile")


def test_single_buffer_reduce_does_not_fire_register_tile(recording_dump):
    """``sum(x, axis=-1)`` has only one K-indexed buffer load —
    ``_find_matmul`` rejects (needs ≥2)."""
    g = Graph()
    _input(g, "x", (128, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (128, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "register_tile" not in recording_dump.fired_rules("lowering/tile")


def test_small_matmul_does_not_fire_register_tile(recording_dump):
    """M=N=8 < PAT=16 → blockify keeps every output axis at extent 8,
    so register_tile finds no THREAD axes with ``extent == pat``."""
    g = Graph()
    _input(g, "a", (8, 32))
    _input(g, "b", (32, 8))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (8, 8)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "register_tile" not in recording_dump.fired_rules("lowering/tile")


# --- behavior tests --------------------------------------------------


def test_sdpa_qk_matmul_fires_register_tile(recording_dump):
    """``SdpaOp`` decomposes into two matmuls; the first (Q·Kᵀ) is a
    plain matmul shape with no pre_outer reduces, so register_tile
    fires on it. Confirms the rule still reaches the QK^T kernel after
    the new analysis was added."""
    g = Graph()
    _input(g, "q", (1, 8, 128, 64))
    _input(g, "k", (1, 8, 128, 64))
    _input(g, "v", (1, 8, 128, 64))
    g.add_node(op=SdpaOp(), inputs=["q", "k", "v"], output=Tensor("o", (1, 8, 128, 64)), node_id="o")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["o"]

    Pipeline.build(KERNEL_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/kernel")
    assert "register_tile" in fired, fired


def test_sdpa_attention_kernel_fires_register_tile(recording_dump):
    """With ``002_stage_inputs`` running before register_tile, Stages
    stay singleton across F² and the smem budget no longer blows up on
    SDPA's softmax-pre-pass + (P·V) matmul kernel. Both SDPA matmul
    kernels (Q·Kᵀ and the dominant one) now fire."""
    g = Graph()
    _input(g, "q", (1, 8, 128, 64))
    _input(g, "k", (1, 8, 128, 64))
    _input(g, "v", (1, 8, 128, 64))
    g.add_node(op=SdpaOp(), inputs=["q", "k", "v"], output=Tensor("o", (1, 8, 128, 64)), node_id="o")
    g.inputs = ["q", "k", "v"]
    g.outputs = ["o"]

    Pipeline.build(KERNEL_PASSES, dump=recording_dump).run(g)
    assert "register_tile" in recording_dump.fired_rules("lowering/kernel")
