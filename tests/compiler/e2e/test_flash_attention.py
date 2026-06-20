"""Flash-attention decomposition of ``SdpaOp`` — GPU parity vs torch SDPA.

The ``FLASH`` knob (loop-lifting pass ``loop/lifting/015_lift_sdpa_flash``, with
``frontend/decomposition/010_sdpa`` deferring the intact ``SdpaOp`` to it) fuses
scaled-dot-product attention into a single streaming online-softmax kernel (the
``FlashCombine`` carrier) instead of the score-materializing ``010_sdpa`` path.
These tests pin: (1) with ``FLASH`` on, one fused kernel is emitted and it
matches torch SDPA on the GPU; (2) with ``FLASH`` off, the fork falls through to
the multi-kernel decomposition (default unchanged). Covers static + dynamic
(symbolic ``seq_len``) and causal / non-causal — the scalar-tier flash from
``plans/online-softmax-flash-attention.md`` (Steps 4–6; the tensor-core P@V tier
is future work).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..conftest import requires_cuda


class _Sdpa(torch.nn.Module):
    def __init__(self, causal: bool = False):
        super().__init__()
        self.causal = causal

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=self.causal)


def _compile_sdpa(q, k, v, *, causal: bool = False):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    graph = trace_module(_Sdpa(causal).cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, kernels


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 1, 8, 8), (1, 2, 16, 8), (2, 3, 32, 16)])
def test_flash_sdpa_matches_torch(monkeypatch, B, H, S, D, causal):
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))

    backend, compiled, kernels = _compile_sdpa(q, k, v, causal=causal)
    # One fused kernel — no [S, S] score matrix materialized.
    assert len(kernels) == 1, f"flash should emit one fused kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "fmaxf" in src and "expf" in src, "fused kernel should carry the streaming softmax (max + exp)"

    cuda_q, cuda_k, cuda_v = q.cuda(), k.cuda(), v.cuda()

    def eager_pre_run() -> np.ndarray:
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(cuda_q, cuda_k, cuda_v, is_causal=causal).cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager_pre_run)
    got = list(run_result.outputs.values())[0].flatten()
    assert got.shape == eager.shape
    assert not np.any(np.isnan(got))
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 1e-4, f"flash vs torch SDPA (causal={causal}) max_diff={max_diff:.6e}"


@requires_cuda
@pytest.mark.parametrize("causal", [False, True])
def test_flash_sdpa_dynamic_matches_torch(monkeypatch, causal):
    """Symbolic ``seq_len`` (Q/K/V dim -2): ONE cached kernel carrying ``int
    seq_len`` serves every runtime size — flash's single dynamic axis lands on
    BOTH the query (masked-row M) and KV (reduce) positions, so this exercises
    the masked-row and symbolic-reduce paths together."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    B, H, D = 1, 2, 8
    seq = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _Sdpa(causal).cpu(),
        (torch.randn(B, H, 16, D), torch.randn(B, H, 16, D), torch.randn(B, H, 16, D)),
        dynamic_shapes={"q": {2: seq}, "k": {2: seq}, "v": {2: seq}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic flash should emit one fused kernel, got {len(kernels)}"
    assert "int seq_len" in compiled.nodes[kernels[0]].op.kernel_source, "dynamic kernel must carry the runtime seq_len arg"

    for s in (8, 16, 37):
        q, k, v = (torch.randn(B, H, s, D) for _ in range(3))

        def eager_pre_run(q=q, k=k, v=v) -> np.ndarray:
            with torch.no_grad():
                return (
                    torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=causal).cpu().flatten().numpy()
                )

        run_result, eager = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager_pre_run)
        got = list(run_result.outputs.values())[0].flatten()
        max_diff = float(np.max(np.abs(got - eager)))
        assert max_diff < 1e-4, f"dynamic flash (causal={causal}) seq={s} max_diff={max_diff:.6e}"


@requires_cuda
def test_flash_fork_falls_through_when_off(monkeypatch):
    """FLASH off → the generic 010_sdpa decomposition (multiple kernels: QK^T,
    softmax, P@V), proving the knob gates the fork and the default is unchanged."""
    monkeypatch.delenv("DEPLODOCK_FLASH", raising=False)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    _backend, _compiled, kernels = _compile_sdpa(q, k, v)
    assert len(kernels) > 1, "FLASH off should keep the score-materializing multi-kernel path"
