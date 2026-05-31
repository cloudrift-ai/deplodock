"""Adaptive Stream-K — end-to-end matmul correctness (Phase B3b).

Runs matmuls through the CUDA backend with adaptive Stream-K pinned and checks
the result against a numpy reference. Each CTA walks a contiguous slice of
``M_blocks·N_blocks·K_blocks`` MAC units, runs a runtime-bounded partial K-loop
over its ``[k_lo, k_hi)`` sub-range, writes full tiles directly and combines
boundary partials via ``atomicAdd`` into the pre-zeroed output — the real
wave-tail mechanism (mid-tile K-splitting). Gated to SYNC / BUFFERED-ring staging
until B5 (the async/TMA/pipelined prologue assumes a contiguous-from-0 K-loop);
the structural test pins that gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda import CudaOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.pipeline import CUDA_PASSES, Pipeline

try:
    import cupy as cp

    _HAS_CUDA = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="needs a CUDA device")


def _matmul_graph(m: int, k: int, n: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


# Adaptive Stream-K is gated to SYNC / BUFFERED-ring staging (no TMA / async /
# pipeline) until B5 — the runtime-bounded K-loop has no contiguous-from-0
# prefetch state to break. Pinned as individual knobs (not DEPLODOCK_KNOBS,
# which only the CLI splat expands) so the raw Pipeline path sees them too.
_ADAPTIVE_KNOBS = {
    "BM": "8",
    "BN": "16",
    "FM": "1",
    "FN": "1",
    "BK": "32",
    "STREAMK": "1",
    "TMA": "0",
    "ASYNC_COPY": "0",
    "PIPELINE_STAGES": "0",
}


def _pin_adaptive(monkeypatch):
    for k, v in _ADAPTIVE_KNOBS.items():
        monkeypatch.setenv(f"DEPLODOCK_{k}", v)


@requires_cuda
@pytest.mark.parametrize(
    "m,k,n",
    [
        (256, 256, 256),  # 512 tiles, K=256/BK=32 → 8 K-chunks; under-occupied
        (512, 256, 128),  # K=256 → 8 K-chunks, many mid-tile boundaries
        (320, 192, 256),  # non-square
        (1024, 512, 1024),  # multi-wave: many MAC units per CTA
    ],
)
def test_adaptive_streamk_matches_numpy(m, k, n, monkeypatch):
    """Real Phase-B adaptive Stream-K: CTAs walk MAC units, run partial K-loops
    over [k_lo, k_hi), and combine boundary partials via atomicAdd into the
    pre-zeroed output. Mid-tile splitting — the actual wave-tail mechanism."""
    _pin_adaptive(monkeypatch)
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    rng = np.random.default_rng(2)
    a = rng.standard_normal((m, k), dtype=np.float32)
    b = rng.standard_normal((k, n), dtype=np.float32)
    be = CudaBackend()
    out = be.run(be.compile(_matmul_graph(m, k, n)), input_data={"a": a, "b": b})[0].outputs["o"]
    np.testing.assert_allclose(out.reshape(m, n), a @ b, rtol=3e-3, atol=3e-3)


@requires_cuda
def test_adaptive_streamk_lowers_mac_walk_and_atomic_boundary(monkeypatch):
    """The adaptive lowering emits the MAC-segment walk, a runtime-bounded K-loop,
    a full-vs-partial write branch with an atomicAdd boundary, and zeroes the
    output (so the atomic partials accumulate from zero)."""
    _pin_adaptive(monkeypatch)
    lowered = Pipeline.build(CUDA_PASSES).run(_matmul_graph(256, 256, 256))
    ops = [n.op for n in lowered.nodes.values() if isinstance(n.op, CudaOp) and n.op.streamk_work_arrays]
    assert ops, "no adaptive Stream-K CudaOp produced"
    op = ops[0]
    src = op.kernel_source
    assert "while (__mac < __wend)" in src
    assert "streamk_k_lo" in src and "streamk_k_hi" in src
    assert "atomicAdd(" in src, "boundary partials must atomicAdd"
    assert op.zero_outputs, "output must be pre-zeroed for the atomic combine"
    assert op.grid[0] == ("__num_sms__",)
    # work units = output tiles × K_blocks; K=256 / BK=32 → 8 K-chunks per tile.
    assert op.streamk_total_units > 0 and op.streamk_total_units % 8 == 0


@requires_cuda
def test_pinned_streamk_on_pipelined_staging_fails_loudly(monkeypatch):
    """Pinning STREAMK=1 where it can't apply (default async/TMA/pipelined
    staging, B5 pending) is a user error — raise, don't silently compile a
    non-Stream-K kernel (fail-loud convention)."""
    monkeypatch.setenv("DEPLODOCK_STREAMK", "1")  # default staging → pipelined
    with pytest.raises(ValueError, match="SYNC/BUFFERED staging"):
        Pipeline.build(CUDA_PASSES).run(_matmul_graph(512, 512, 512))


@requires_cuda
def test_streamk_is_mutually_exclusive_with_splitk(monkeypatch):
    """Stream-K supplies its own K-split, so it can't compose with fixed split-K.
    Pinning both is a user error — fail loudly rather than silently dropping one."""
    _pin_adaptive(monkeypatch)
    monkeypatch.setenv("DEPLODOCK_SPLITK", "4")
    with pytest.raises(ValueError, match="mutually exclusive"):
        Pipeline.build(CUDA_PASSES).run(_matmul_graph(512, 512, 512))
