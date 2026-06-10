"""Regression tests for two TMA eligibility gates in ``050_use_tma`` (Qwen3-Embedding layer 0 tune fallout).

Both bugs ship a kernel that nvcc compiles cleanly and that only fails on the device — the worst failure mode for
the tuner (a ``bench_fail`` per variant) and for greedy ``run`` / ``compile`` (a crash or hang in deployment). See
``plans/qwen3-embedding-layer0-tune-findings.md``.

1. **boxDim > 256 launch failure.** ``cuTensorMapEncodeTiled`` enforces ``boxDim[i] <= 256`` per dim, and the
   descriptor is encoded at *launch* (it embeds the device pointer), so an oversized collapsed box — e.g. the
   scalar register-tile matmul's M box ``BM·FM``, tuned up to 768 on the down_proj — used to pass eligibility,
   compile, and then die with the opaque ``CUresult=1``. 16 launch-time bench_fails per layer-0 tune, every one
   with ``BM·FM > 256``; every TMA-ok variant sat at ``BM·FM <= 256``. The fix mirrors the materializer's
   ``box_per_dim`` collapse in ``_source_eligible`` and declines any source whose collapsed extent exceeds 256.

2. **Re-entered ring-pipeline deadlock.** The materializer emits ``MbarrierInit`` once at kernel entry, and the
   pipeline's parity schedule assumes every slot starts at phase 0. The fused RMSNorm + gate/up kernel nests its
   ``serial_outer`` K loop *inside* the per-thread FM cell loop (the norm reduction is per output row), so at
   ``FM=2`` the pipeline runs twice over the same mbarriers: with 32 K-tiles on a RING=3 schedule, iteration 0
   leaves the slots at mixed parities (11/11/10 completed rounds) and iteration 1's waits desync — a
   deterministic device hang (6 ``HungKernelError`` per layer-0 tune, all ``FM=2``; every ``FM=1`` TMA variant
   ran fine). The fix declines TMA when a promotable bundle's ``serial_outer`` is nested inside a serial loop
   with trip count > 1; cp.async (commit/wait_prior — no cross-iteration phase state) handles re-entry correctly.

The knob pins mirror the failing DB rows so the partition planner reproduces the exact tile structures.
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

_S, _H, _I = 32, 1024, 3072  # seq, hidden, intermediate — the Qwen3-Embedding-0.6B slices

# Scalar register-tile family shared by both repros (mirrors the failing tune rows).
_BASE_KNOBS = {
    "BR": 1,
    "FK": 1,
    "SPLITK": 1,
    "GROUP_M": 8,
    "PIPELINE_STAGES": 1,
    "MMA": 0,
    "WARP_SPECIALIZE": 0,
}

# down_proj (the boxDim repro): collapsed M box = BM·FM = 512 > 256 — one of the 16
# launch-failing rows. The control swaps FM=64 → FM=16 (box 128).
_BOXDIM_KNOBS = {**_BASE_KNOBS, "BM": 8, "BN": 64, "BK": 32, "FM": 64, "FN": 2, "RING": 2, "STAGE": 11}

# Fused norm+matmul (the hang repro): FM=2 nests the K pipeline inside the cell loop —
# one of the 6 genuinely hung rows. The control swaps FM=2 → FM=1 (pipeline runs once).
_HANG_KNOBS = {**_BASE_KNOBS, "BM": 8, "BN": 64, "BK": 32, "FM": 2, "FN": 2, "RING": 3, "STAGE": 110}


def _pin_knobs(monkeypatch, knobs: dict) -> None:
    for key, value in knobs.items():
        monkeypatch.setenv(f"DEPLODOCK_{key}", str(value))


def _build_down_proj() -> Graph:
    """Plain fp16 linear at the Qwen3-Embedding down_proj shape (K=3072)."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, _S, _I), f16), node_id="x")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("w", (_H, _I), f16), node_id="w")
    g.add_node(op=LinearOp(), inputs=["x", "w"], output=Tensor("o", (1, _S, _H), f16), node_id="o")
    g.inputs = ["x", "w"]
    g.outputs = ["o"]
    return g


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


def _tile_op(graph: Graph):
    from deplodock.compiler.ir.tile.ir import TileOp
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g2 = Pipeline.build(TILE_PASSES).run(graph)
    op = g2.nodes["o"].op
    assert isinstance(op, TileOp)
    return op


def _has_tma_bundle(op) -> bool:
    from deplodock.compiler.ir.tile.ir import StageBundle, StagePolicy

    return any(isinstance(s, StageBundle) and s.policy == StagePolicy.TMA for s in op.body.iter())


# --- boxDim > 256 ----------------------------------------------------------------


def test_oversized_box_declines_tma(monkeypatch, _sm120_target):
    """BM·FM = 512 > 256: the collapsed M box exceeds ``cuTensorMapEncodeTiled``'s
    per-dim limit, so the tile must decline to cp.async instead of compiling a
    kernel that fails at launch with CUresult=1 (the pre-fix behavior)."""
    _pin_knobs(monkeypatch, _BOXDIM_KNOBS)
    op = _tile_op(_build_down_proj())
    assert op.knobs.get("TMA") is False
    assert not _has_tma_bundle(op)


def test_oversized_box_pinned_tma_raises(monkeypatch, _sm120_target):
    """Pinned TMA=1 on the oversized box must raise the loud unhonorable-pin error."""
    _pin_knobs(monkeypatch, {**_BOXDIM_KNOBS, "TMA": 1})
    with pytest.raises(ValueError, match="not TMA-eligible"):
        _tile_op(_build_down_proj())


def test_box_at_limit_keeps_tma(monkeypatch, _sm120_target):
    """Control: BM·FM = 128 <= 256 on the same shape still promotes to TMA."""
    _pin_knobs(monkeypatch, {**_BOXDIM_KNOBS, "FM": 16})
    op = _tile_op(_build_down_proj())
    assert op.knobs.get("TMA") is True
    assert _has_tma_bundle(op)


def test_encode_tiled_rejects_oversized_box():
    """The runtime encoder names the offending dim instead of surfacing the
    driver's opaque CUresult=1 (defensive twin of the eligibility gate)."""
    from deplodock.compiler.backend.cuda._tma import encode_tiled

    with pytest.raises(ValueError, match="box dim 0 extent 512"):
        encode_tiled(global_address=0, src_shape=(1024, 1024), box_extents=(512, 64), elem_size=2)


# --- re-entered ring pipeline ----------------------------------------------------


def test_reentered_pipeline_declines_tma(monkeypatch, _sm120_target):
    """FM=2 nests the K pipeline inside the cell loop: the once-initialized
    mbarrier ring would be re-entered at stale phase parity (the pre-fix
    deterministic hang), so the tile must decline to cp.async."""
    _pin_knobs(monkeypatch, _HANG_KNOBS)
    op = _tile_op(_build_mlp_slice())
    assert op.knobs.get("TMA") is False
    assert not _has_tma_bundle(op)


def test_reentered_pipeline_pinned_tma_raises(monkeypatch, _sm120_target):
    """Pinned TMA=1 on the re-entered pipeline must raise the loud unhonorable-pin error."""
    _pin_knobs(monkeypatch, {**_HANG_KNOBS, "TMA": 1})
    with pytest.raises(ValueError, match="nested inside a serial loop"):
        _tile_op(_build_mlp_slice())


def test_single_trip_cell_loop_keeps_tma(monkeypatch, _sm120_target):
    """Control: FM=1 runs the pipeline exactly once — TMA stays eligible."""
    _pin_knobs(monkeypatch, {**_HANG_KNOBS, "FM": 1, "FN": 1, "BN": 32})
    op = _tile_op(_build_mlp_slice())
    assert op.knobs.get("TMA") is True
    assert _has_tma_bundle(op)


# --- device ----------------------------------------------------------------------


@requires_cuda
def test_hang_knob_family_completes_and_matches(monkeypatch):
    """The FM=2 family compiles and runs to completion on the cp.async fallback (a
    regression would trip the per-launch watchdog's HungKernelError) and matches
    the numpy reference."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend

    _pin_knobs(monkeypatch, _HANG_KNOBS)
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


@requires_cuda
def test_boxdim_knob_family_completes_and_matches(monkeypatch):
    """The BM·FM=512 family compiles and runs on the cp.async fallback (a regression
    would fail at launch with cuTensorMapEncodeTiled CUresult=1) and matches numpy."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.numpy import NumpyBackend

    _pin_knobs(monkeypatch, _BOXDIM_KNOBS)
    g = _build_down_proj()
    rng = np.random.default_rng(0)
    f16 = np.dtype(np.float16)
    inputs = {
        "x": rng.standard_normal((1, _S, _I), dtype=np.float32).astype(f16),
        "w": (rng.standard_normal((_H, _I), dtype=np.float32) * 0.05).astype(f16),
    }

    ref_be = NumpyBackend()
    ref = ref_be.run(ref_be.compile(_build_down_proj()), input_data=inputs)[0].outputs["o"]

    be = CudaBackend()
    out = be.run(be.compile(g), input_data=inputs)[0].outputs["o"]

    assert out.shape == ref.shape
    assert np.all(np.isfinite(out.astype(np.float32))), "output has non-finite values"
    peak = float(np.max(np.abs(ref.astype(np.float32))))
    atol = max(1e-2, 0.05 * peak)
    np.testing.assert_allclose(out.astype(np.float32), ref.astype(np.float32), atol=atol, rtol=0.05)
