"""Regression tests for the WS=1 stranded-TMA deadlock (Qwen3 ``k_linear_mean_reduce``).

The fused RMSNorm + gate/up-linear + SwiGLU kernel of a Qwen3-class decoder MLP wraps its whole body in a
``SerialTile(kind='plain')`` per-thread M-fragment loop. ``085_warp_specialize._split_by_role`` only recurses through
``serial_outer`` / ``RegisterTile`` / ``AtomTile``, so on this shape the WS=1 rewrite used to strand every TMA
``StageBundle`` in the **consumer** branch: the producer branch came out empty, the TMA issues sat behind a
``threadIdx.x == 0`` guard that no consumer-branch thread can satisfy (thread 0 is a producer-warp thread), and every
consumer ``mbarrier.wait`` spun forever — a deterministic device hang at any nvcc opt level. ``_eligible`` had used a
*deep* body walk to find the bundle, so it declared the shape WS-eligible even though the *shallow* split could never
reach it; the learned prior then deployed WS=1 for this op (generalizing from warp-tier MMA rows where WS wins) and
every ``run`` / ``compile`` of Qwen3-Embedding-0.6B layer 0 hung. See
``plans/qwen3-embedding-tune-hung-kernel.md``.

The fix makes ``_eligible`` run the same producer/consumer split the transform uses and reject when no TMA depth-2
bundle lands producer-side. The compile-only tests below lock that in GPU-less CI (``set_target`` pins the TMA path);
the CUDA test confirms the shape's pinned knob family lowers to a kernel that actually completes under the watchdog.

The knob pins mirror the deployed greedy pick for the hanging op (BM=8 BN=32 BK=64 RING=2 STAGE=100 on the
(1, 32, 1024) → (1, 32, 3072) fp16 slice) so the partition planner reproduces the exact tile structure.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler import target as target_mod
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, RmsNormOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp

from ..conftest import requires_cuda

# The hanging op's tile family (the greedy pick deployed for Qwen3-Embedding-0.6B layer 0's
# k_linear_mean_reduce), plus the transport pins that force the TMA depth-2 pipelined path
# the WS rule matches on.
_KNOBS = {
    "BM": 8,
    "BN": 32,
    "BK": 64,
    "BR": 1,
    "FM": 1,
    "FN": 1,
    "FK": 1,
    "SPLITK": 1,
    "RING": 2,
    "STAGE": 100,
    "GROUP_M": 8,
    "TMA": 1,
    "PIPELINE_STAGES": 1,
}

_S, _H, _I = 32, 1024, 3072  # seq, hidden, intermediate — the Qwen3-Embedding-0.6B MLP slice


def _pin_knobs(monkeypatch) -> None:
    for key, value in _KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))


def _build_mlp_slice() -> Graph:
    """RMSNorm → gate/up linears → SwiGLU, fp16 — the ``k_linear_mean_reduce`` fusion."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, _S, _H), f16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("nw", (_H,), f16), node_id="nw")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wg", (_I, _H), f16), node_id="wg")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("wu", (_I, _H), f16), node_id="wu")
    g.add_node(op=RmsNormOp(eps=1e-6), inputs=["x", "nw"], output=Tensor("xn", (1, _S, _H), f16), node_id="xn")
    g.add_node(op=LinearOp(), inputs=["xn", "wg"], output=Tensor("mg", (1, _S, _I), f16), node_id="mg")
    g.add_node(op=LinearOp(), inputs=["xn", "wu"], output=Tensor("mu", (1, _S, _I), f16), node_id="mu")
    g.add_node(op=ElementwiseOp("silu"), inputs=["mg"], output=Tensor("sg", (1, _S, _I), f16), node_id="sg")
    g.add_node(op=ElementwiseOp("multiply"), inputs=["sg", "mu"], output=Tensor("o", (1, _S, _I), f16), node_id="o")
    g.inputs = ["x", "nw", "wg", "wu"]
    g.outputs = ["o"]
    return g


@pytest.fixture
def _sm120_target():
    """Pin the sm_120 codegen path so the TMA-gated passes fire without a live device."""
    target_mod.set_target((12, 0))
    try:
        yield
    finally:
        target_mod.set_target(None)


def test_pinned_ws1_on_mlp_slice_raises(monkeypatch, _sm120_target):
    """Pinned WS=1 on the fused linear+mean shape must raise the loud unhonorable-pin
    error, not lower into the empty-producer kernel (the pre-fix behavior, which
    deadlocked the device on first launch)."""
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    _pin_knobs(monkeypatch)
    monkeypatch.setenv("DEPLODOCK_WARP_SPECIALIZE", "1")
    with pytest.raises(ValueError, match="not reachable by the producer split"):
        Pipeline.build(TILE_PASSES).run(_build_mlp_slice())


def test_mlp_slice_never_offers_ws1(monkeypatch, _sm120_target):
    """Without a WS pin the shape lowers with WARPSPEC stamped False — WS=1 is not in
    the enumeration, so neither the tuner nor the learned prior can ever deploy it."""
    from deplodock.compiler.ir.tile.ir import TileOp, WarpSpecialize
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    _pin_knobs(monkeypatch)
    g2 = Pipeline.build(TILE_PASSES).run(_build_mlp_slice())
    op = g2.nodes["o"].op
    assert isinstance(op, TileOp)
    assert op.knobs.get("WARPSPEC") is False
    assert not any(isinstance(s, WarpSpecialize) for s in op.body.iter())


@requires_cuda
def test_mlp_slice_completes_and_matches(monkeypatch):
    """The pinned knob family compiles and runs to completion (a regression would
    trip the per-launch watchdog's HungKernelError) and matches the numpy reference."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend

    _pin_knobs(monkeypatch)
    g = _build_mlp_slice()
    rng = np.random.default_rng(0)
    f16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((1, _S, _H), dtype=np.float32).astype(f16),
        "nw": rng.standard_normal((_H,), dtype=np.float32).astype(f16),
        "wg": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(f16),
        "wu": (rng.standard_normal((_I, _H), dtype=np.float32) * 0.05).astype(f16),
    }

    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_build_mlp_slice()), input_data=inputs)[0].outputs["o"]

    be = CudaBackend()
    out = be.run(be.compile(g), input_data=inputs)[0].outputs["o"]

    assert out.shape == ref.shape
    assert np.all(np.isfinite(out.astype(np.float32))), "output has non-finite values"
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(1e-2, 0.05 * peak)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.05)
