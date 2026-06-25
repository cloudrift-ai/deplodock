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
    # The kernel is built through kernel-IR: the TC primitives are the project's SHARED
    # codegen helpers (the same the matmul emits), + the C->A smem slab (flash_pv_smem).
    assert "dpl_mma_m16n8k16_f16" in src and "dpl_ldmatrix_x4" in src, "the generated kernel must use the shared tensor-core ops"
    assert "flash_pv_smem" in src, "the generated kernel must be the fused warp-chain (C->A smem handoff)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"generated TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (1, 4, 128, 64)])
def test_generated_tensorcore_flash_bf16_matches_torch(monkeypatch, B, H, S, D):
    """Phase 4 — bf16 in, f32 accumulate. Same fused warp-chain as fp16 (the 16-bit
    operand dtype only swaps the mma atom / PTX dtype field); validated vs torch SDPA."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    torch.manual_seed(S + D + 1)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.bfloat16) for _ in range(3))
    backend, compiled, graph, kernels = _compile(q, k, v)
    assert len(kernels) == 1, f"fused TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_mma_m16n8k16_bf16" in src, "the bf16 flash must use the bf16 mma atom"
    assert "flash_pv_smem" in src, "the generated kernel must be the fused warp-chain (C->A smem handoff)"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    # bf16 has no native numpy dtype — the backend maps it to uint16 (the raw bit
    # pattern), so feed the bf16 bits as uint16 and reinterpret the uint16 output back.
    data = {n: t.view(torch.uint16).numpy() for n, t in zip(graph.inputs, (q, k, v), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got_bits = list(run_result.outputs.values())[0].flatten().astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy()
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-2, f"generated bf16 TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


def test_warp_chain_cell_layout_falls_out_of_atomize():
    """The two cells' operand layout (``b_trans``, the atom) is DERIVED by the ``atomize``
    move (``_classify_cell`` → ``atomize_cell`` → ``classify_matmul_operands``), not
    hard-coded: QK^T is a transposed-B Q@K^T, P@V a canonical-B P@V. CPU-only."""
    import importlib

    from deplodock.compiler.ir.expr import Var

    _classify_cell = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.split.005_warp_chain")._classify_cell

    qk = _classify_cell("Q", (Var("m"), Var("dd")), "K", (Var("kv"), Var("dd")), "dd", (Var("m"), Var("kv")))
    pv = _classify_cell("flash_pv_smem", (Var("m"), Var("kv")), "V", (Var("kv"), Var("d")), "kv", None)
    assert qk.b_trans is True, "Q@K^T must be transposed-B"
    assert pv.b_trans is False, "P@V must be canonical-B"
    assert (qk.a, qk.b) == ("ca", "cb") and (pv.a, pv.b) == ("ca", "cb")
    assert qk.atom.shape == (16, 8, 16)


@requires_cuda
def test_default_path_is_not_the_warp_chain(monkeypatch):
    """Without the ``CHAIN`` pin, a fp16 SDPA does NOT take the warp-chain — the deployed
    default (scalar streaming flash / materialized) is unchanged."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.delenv("DEPLODOCK_CHAIN", raising=False)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 32, 16, dtype=torch.float16) for _ in range(3))
    _backend, compiled, _graph, kernels = _compile(q, k, v)
    # ``flash_pv_smem`` (the C->A smem slab) is unique to the warp-chain kernel.
    assert not any("flash_pv_smem" in compiled.nodes[k].op.kernel_source for k in kernels), "default must not be the warp-chain"
