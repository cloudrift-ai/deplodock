"""The move composer covers the fused flash-attention nest.

With `DEPLODOCK_FLASH=1` the recognizer rewrites SDPA into one streaming
online-softmax `LoopOp` (the `FlashCombine` carrier); with the composer also on,
`walk.walk_nest` routes that `TWISTED_MONOID` nest to `build_flash_tile` — the
free output axes tile, the streaming KV reduce + nested QK^T reduce serialize,
and the carrier renders its own rescale. Verifies GPU parity vs torch SDPA so
the composer's flash path is exercised in CI (the scalar tier; the tensor-core
P@V tier is future work).
"""

from __future__ import annotations

import numpy as np
import torch

from ..conftest import requires_cuda


class _Sdpa(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


@requires_cuda
def test_composer_flash_sdpa_matches_eager(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_MOVE_COMPOSER", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 4, 64, 32) for _ in range(3))
    graph = trace_module(_Sdpa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    # One fused flash kernel — no `xn` score materialization.
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"flash should fuse to one kernel, got {kernels}"

    def eager() -> np.ndarray:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v).numpy().flatten()

    run_result, ref = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager)
    got = next(iter(run_result.outputs.values())).flatten()
    assert not np.any(np.isnan(got))
    np.testing.assert_allclose(got, ref, atol=2e-3, rtol=2e-3)
