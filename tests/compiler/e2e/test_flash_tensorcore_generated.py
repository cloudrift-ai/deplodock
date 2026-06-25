"""The compiler GENERATES the fused tensor-core flash (Phase 2.3 + Phase 3 of
``plans/tensor-core-streaming-flash-mma.md``).

End-to-end: a fp16 SDPA traced + compiled with ``DEPLODOCK_CHAIN=1`` lowers — via the
``split/005_warp_chain`` pass — to a single ``mma.sync`` kernel (the warp-chain build
``assembly/_warp_chain``, the validated FA-2 kernel generalized over ``(B,H,S,D)``, reusing
the ``FragmentRowReduce`` op for the fragment softmax), and matches torch SDPA. The
default path (no ``CHAIN`` pin) is unchanged — the scalar streaming flash / materialized
path still deploys, so this only fires under the explicit opt-in.

v1 scope: fp16, non-causal, equal-head, ``D % 16 == 0``, ``S % 16 == 0`` — the reference
kernel's scope (``test_flash_tensorcore_reference.py``). Out of scope falls back cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..conftest import requires_cuda


class _Sdpa(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _compile(q, k, v):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    graph = trace_module(_Sdpa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, graph, kernels


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (2, 3, 64, 32), (1, 4, 128, 64), (1, 1, 16, 16)])
def test_generated_tensorcore_flash_matches_torch(monkeypatch, B, H, S, D):
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    torch.manual_seed(S + D)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    backend, compiled, graph, kernels = _compile(q, k, v)
    assert len(kernels) == 1, f"fused TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "mma.sync" in src and "dpl_wc_mma" in src, "the generated kernel must use the tensor-core warp-chain"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"generated TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
def test_default_path_is_not_the_warp_chain(monkeypatch):
    """Without the ``CHAIN`` pin, a fp16 SDPA does NOT take the warp-chain — the deployed
    default (scalar streaming flash / materialized) is unchanged."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.delenv("DEPLODOCK_CHAIN", raising=False)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 32, 16, dtype=torch.float16) for _ in range(3))
    _backend, compiled, _graph, kernels = _compile(q, k, v)
    assert not any("dpl_wc_mma" in compiled.nodes[k].op.kernel_source for k in kernels), "default must not be the warp-chain"
