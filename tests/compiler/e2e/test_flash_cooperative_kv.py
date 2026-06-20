"""Flash attention with a COOPERATIVE-K (KV split across threads) reduce — Step 4
of ``plans/atomic-free-monoid-combine.md``.

The deployed scalar flash kernel kept its KV (online-softmax ``Monoid``) reduce
serial — one thread per output element, the KV axis unparallelized. Step 4 routes
the flash-pattern body to the cooperative-reduce path so the KV axis splits across
the CTA's threads and the per-thread partial ``(m, l, O)`` states merge via the
monoid combine (Step 2's ``__shfl_xor_sync`` / smem tree over ``combine_states``).
These tests pin a cooperative ``BR`` and assert (1) the kernel actually carries the
cross-thread monoid combine, and (2) it still matches torch SDPA — the KV
parallelization is accuracy-preserving (the LSE monoid is associative +
commutative).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..conftest import requires_cuda


class _Sdpa(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _compile(q, k, v):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    graph = trace_module(_Sdpa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, kernels


@requires_cuda
@pytest.mark.parametrize("br", ["32", "64"])
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 64, 16), (1, 4, 128, 32)])
def test_cooperative_flash_matches_torch(monkeypatch, br, B, H, S, D):
    """A cooperative-KV flash (BR>1) fuses to one kernel carrying the monoid
    cross-thread combine and matches torch SDPA."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_BR", br)
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    backend, compiled, kernels = _compile(q, k, v)
    assert len(kernels) == 1, f"flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    # The cross-thread monoid combine: __shfl_xor_sync (warp path, BR<=32) or a
    # per-component smem tree (MonoidTreeHalve, BR>32). Scalar flash has no other
    # smem, so either marker confirms the KV reduce went cooperative.
    assert "__shfl_xor_sync" in src or "_smem" in src, "cooperative-KV flash must carry the cross-thread monoid combine"

    def eager(q=q, k=k, v=v) -> np.ndarray:
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().numpy()

    run_result, ref = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager)
    got = list(run_result.outputs.values())[0].flatten()
    assert float(np.max(np.abs(got - ref))) < 1e-4
