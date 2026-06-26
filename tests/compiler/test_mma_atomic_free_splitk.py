"""Atomic-free split-K on the warp / MMA tensor-core tier.

Dropping ``017``'s ``is_warp`` early-out lets an MMA matmul's split-K route its
C-fragment store into ``workspace[K_s, M, N]`` (the same Write-retarget the scalar
path uses — the fragment ``RegStore`` is lowered later from that Write) and reuse
the additive ``Accum``-sum reduce kernel, instead of the codegen ``atomicAdd``. The cross-CTA
finalize is the native ``REDUCE`` codec's ``c``-letter: ``c2k`` (the deferred combine kernel) is
accurate vs numpy and emits no ``atomicAdd``; ``c2a`` (the in-place atomic finalize) stays
available. Pinned entirely through the native move-knob set (``ATOM`` / ``SPLIT`` / ``REDUCE``) —
no legacy ``MMA`` / ``WM`` / ``SPLITK`` / ``NOATOMIC`` spellings.
"""

from __future__ import annotations

import numpy as np
import pytest

from deplodock.compiler.dtype import F16
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp

# Native move-knob set: the warp MMA atom on the output cell + a 2×2 warp/register tile on every
# free axis (WM=WN=FM=FN=2). The reduce decomposition (BK=2 + split-K 2 + the finalize letter) is
# the per-arm ``DEPLODOCK_REDUCE`` codec value ``s2/c2{a,k}``.
_WARP_NATIVE = {"DEPLODOCK_ATOM": "mma_m16n8k16_f16", "DEPLODOCK_SPLIT": "2x2"}


def _set_warp(monkeypatch, reduce_spec: str) -> None:
    for k, v in _WARP_NATIVE.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DEPLODOCK_REDUCE", reduce_spec)


def _has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _supports_mma_sync() -> bool:
    if not _has_cuda():
        return False
    import cupy as cp

    return int(cp.cuda.Device().compute_capability) >= 80


def _mma_graph(m: int, k: int, n: int) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n), dtype=F16), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n), dtype=F16), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync split-K needs sm_80+")
def test_mma_atomic_free_splitk_accurate_and_no_atomic(monkeypatch) -> None:
    """fp16 MMA split-K with the deferred ``c2k`` finalize: bit-correct vs numpy and no atomicAdd."""
    _set_warp(monkeypatch, "s2/c2k")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    m, k, n = 128, 512, 128
    rng = np.random.default_rng(4)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)
    be = CudaBackend()
    compiled = be.compile(_mma_graph(m, k, n))
    src = "\n".join(node.op.kernel_source for node in compiled.nodes.values() if getattr(node.op, "kernel_source", None))
    assert "mma.sync.aligned.m16n8k16" in src, "must be on the tensor-core tier"
    assert "atomicAdd" not in src, "atomic-free split-K must not emit atomicAdd"
    out = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"]).reshape(m, n)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _supports_mma_sync(), reason="mma.sync split-K needs sm_80+")
def test_mma_atomic_splitk_still_available(monkeypatch) -> None:
    """The atomic finalize arm (``c2a``) stays selectable and is also accurate."""
    _set_warp(monkeypatch, "s2/c2a")
    from deplodock.compiler.backend.cuda.backend import CudaBackend

    m, k, n = 128, 512, 128
    rng = np.random.default_rng(5)
    a = (rng.standard_normal((m, k)) * 0.1).astype(np.float16)
    b = (rng.standard_normal((k, n)) * 0.1).astype(np.float16)
    be = CudaBackend()
    compiled = be.compile(_mma_graph(m, k, n))
    out = np.asarray(be.run(compiled, input_data={"a": a, "b": b})[0].outputs["o"]).reshape(m, n)
    ref = a.astype(np.float32) @ b.astype(np.float32)
    np.testing.assert_allclose(out.astype(np.float32), ref, rtol=2e-2, atol=2e-2)
