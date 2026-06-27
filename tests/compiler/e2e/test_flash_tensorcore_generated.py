"""The compiler GENERATES the fused tensor-core flash.

End-to-end: a fp16 SDPA traced + compiled with ``DEPLODOCK_CHAIN=1`` lowers — via the
``070_coop_reduce`` warp-flash fork — to a single ``mma.sync`` kernel (``_build.warp_chain_build``
σ-tiles + atomizes the two contractions, ``_assemble.carry_scope_from_graph`` realizes the
fragment softmax via ``FragmentRowReduce``, generalized over ``(B,H,S,D)``), and matches torch SDPA. The
default path (no ``CHAIN`` pin) is unchanged — the scalar streaming flash / materialized
path still deploys, so this only fires under the explicit opt-in.

v1 scope: fp16 / bf16, causal or non-causal, equal-head, ``D % 16 == 0``, ``S % 16 == 0``.
The softmax is generated from the carrier by the fragment realizer (``_frag_softmax``), so
dtype (f32 algebra) and causal (a score-partial mask) are orthogonal — this file covers
their cross-product. Out of scope falls back cleanly.
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


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (1, 4, 128, 64)])
def test_generated_tensorcore_flash_causal_bf16_matches_torch(monkeypatch, B, H, S, D):
    """The cross-product: bf16 operands AND the fragment causal mask, together. The
    softmax realizer is dtype-agnostic (f32 algebra) and causal is a score-partial mask,
    so the two compose with no special-casing — validated vs torch's bf16 is_causal SDPA."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    torch.manual_seed(S + D + 2)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.bfloat16) for _ in range(3))
    graph = _CausalSdpa().cpu()
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    g = trace_module(graph, (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(g)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"fused causal bf16 TC flash should be one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "dpl_mma_m16n8k16_bf16" in src and "flash_pv_smem" in src

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t.view(torch.uint16).numpy() for n, t in zip(g.inputs, (q, k, v), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got_bits = list(run_result.outputs.values())[0].flatten().astype(np.uint16)
    got = torch.from_numpy(got_bits).view(torch.bfloat16).float().numpy()
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-2, f"generated causal bf16 TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


class _CausalSdpa(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 2, 32, 16), (2, 3, 64, 32), (1, 4, 128, 64), (1, 1, 16, 16)])
def test_generated_tensorcore_flash_causal_matches_torch(monkeypatch, B, H, S, D):
    """Phase 5 — causal masking at the fragment tier. The fused warp-chain inserts a
    per-element ``FragmentMask`` (causal) on the score fragment (strict upper triangle →
    ``-1e30`` before the rowmax), matching torch's ``is_causal=True`` SDPA."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(S + D)
    q, k, v = (torch.randn(B, H, S, D, dtype=torch.float16) for _ in range(3))
    graph = trace_module(_CausalSdpa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"fused causal TC flash should be one kernel, got {len(kernels)}"
    assert "flash_pv_smem" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"generated causal TC flash {(B, H, S, D)} max_diff={max_diff:.2e}"


# NOTE: the former ``test_warp_chain_cell_layout_falls_out_of_atomize`` (which imported the
# split shim's ``_classify_cell`` to assert the operand ``b_trans`` was derived by ``atomize``)
# was removed when the flash realization moved to ``_flash.realize_flash``, which sets the v1
# m16n8k16 layout (transposed-B Q@K^T, canonical-B P@V) as a structural constant. That layout is
# now accuracy-checked end-to-end by the dynamic / static tests in this file — a wrong ``b_trans``
# corrupts the attention and fails them.


@requires_cuda
@pytest.mark.parametrize("seq", [8, 16, 37, 64])
def test_warp_chain_dynamic_matches_torch(monkeypatch, seq):
    """Phase 1 — symbolic ``seq_len`` warp-chain flash. ONE cached fused-TC kernel
    carrying ``int seq_len`` serves every runtime size: the partial final KV / query
    tile (``seq=37`` straddles both) is masked at the score fragment (``kv_col >=
    seq_len`` → ``-1e30`` before the rowmax), its K/V gmem loads clamped, and its
    output store guarded. Non-causal, equal-head, ``D % 16 == 0``. Matches torch SDPA
    at seq ∈ {8, 16, 37, 64} (37 is the partial-tile oracle that caught the
    materialized-path −inf bug)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    B, H, D = 1, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _Sdpa().cpu(),
        (
            torch.randn(B, H, 16, D, dtype=torch.float16),
            torch.randn(B, H, 16, D, dtype=torch.float16),
            torch.randn(B, H, 16, D, dtype=torch.float16),
        ),
        dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic warp-chain flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "flash_pv_smem" in src, "the symbolic flash must be the fused warp-chain (C->A smem handoff)"
    assert "int seq_len" in src, "the symbolic warp-chain must carry the runtime seq_len arg"

    torch.manual_seed(seq)
    q, k, v = (torch.randn(B, H, seq, D, dtype=torch.float16) for _ in range(3))

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().float().numpy()

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), f"symbolic warp-chain flash seq={seq} produced NaN"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"symbolic warp-chain flash seq={seq} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("seq", [8, 16, 37, 64])
def test_warp_chain_causal_dynamic_matches_torch(monkeypatch, seq):
    """Phase 4 — symbolic ``seq_len`` warp-chain flash with **causal** masking (equal-head).
    The causal score-fragment mask (``kv_col > q_row`` → ``-1e30``) composes with the
    symbolic boundary mask (``kv_col >= seq_len`` → ``-1e30``): both write the soft −inf
    before the rowmax, so emitting them in sequence is the AND of the keep predicates.
    Matches torch ``is_causal=True`` SDPA at seq ∈ {8, 16, 37, 64}."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    B, H, D = 1, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _CausalSdpa().cpu(),
        (
            torch.randn(B, H, 16, D, dtype=torch.float16),
            torch.randn(B, H, 16, D, dtype=torch.float16),
            torch.randn(B, H, 16, D, dtype=torch.float16),
        ),
        dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic causal warp-chain flash should fuse to one kernel, got {len(kernels)}"
    assert "flash_pv_smem" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

    torch.manual_seed(seq)
    q, k, v = (torch.randn(B, H, seq, D, dtype=torch.float16) for _ in range(3))

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), f"causal symbolic warp-chain flash seq={seq} produced NaN"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"causal symbolic warp-chain flash seq={seq} max_diff={max_diff:.2e}"


class _GqaSdpa(torch.nn.Module):
    """GQA SDPA. ``enable_gqa=True`` is grabbed by the tracer's is_causal scan (the default
    ``is_causal=False`` is dropped by dynamo), so this traces as GQA **and** causal — the
    Qwen3-Embedding layer-0 shape, and the only public-API GQA form here."""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)


@requires_cuda
@pytest.mark.parametrize(("Hq", "Hkv", "S", "D"), [(4, 2, 32, 16), (16, 8, 32, 32)])
def test_warp_chain_gqa_static_matches_torch(monkeypatch, Hq, Hkv, S, D):
    """Phase 2 — STATIC ``S`` warp-chain flash with GQA (``head // group`` K/V indexing).
    ``_GqaSdpa`` traces as GQA+causal; the fused warp-chain reads K/V at the kv-head with no
    materialized broadcast and matches torch."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    torch.manual_seed(S + D)
    q = torch.randn(1, Hq, S, D, dtype=torch.float16)
    k, v = (torch.randn(1, Hkv, S, D, dtype=torch.float16) for _ in range(2))
    graph = trace_module(_GqaSdpa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"static GQA warp-chain flash should be one kernel, got {len(kernels)}"
    assert "flash_pv_smem" in compiled.nodes[kernels[0]].op.kernel_source, "must be the fused warp-chain"

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"static GQA warp-chain flash {(Hq, Hkv, S, D)} max_diff={max_diff:.2e}"


@requires_cuda
@pytest.mark.parametrize("seq", [8, 16, 37, 64])
def test_warp_chain_gqa_dynamic_matches_torch(monkeypatch, seq):
    """Phase 2 — symbolic ``seq_len`` warp-chain flash with **GQA** (``Hq=4 / Hkv=2``, group
    2): K/V are read at ``head // group`` directly (no materialized broadcast). ``_GqaSdpa``
    traces as GQA+causal, so this also exercises the causal mask composed with the symbolic
    boundary mask (both write ``-1e30`` before the rowmax). ONE cached kernel carrying ``int
    seq_len``; matches torch GQA+causal SDPA at seq ∈ {8, 16, 37, 64}."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    B, Hq, Hkv, D = 1, 4, 2, 32
    sd = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _GqaSdpa().cpu(),
        (
            torch.randn(B, Hq, 16, D, dtype=torch.float16),
            torch.randn(B, Hkv, 16, D, dtype=torch.float16),
            torch.randn(B, Hkv, 16, D, dtype=torch.float16),
        ),
        dynamic_shapes={"q": {2: sd}, "k": {2: sd}, "v": {2: sd}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic GQA warp-chain flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "flash_pv_smem" in src and "int seq_len" in src, "must be the symbolic fused warp-chain"

    torch.manual_seed(seq)
    q = torch.randn(B, Hq, seq, D, dtype=torch.float16)
    k, v = (torch.randn(B, Hkv, seq, D, dtype=torch.float16) for _ in range(2))

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .float()
                .numpy()
            )

    data = {n: t for n, t in zip(graph.inputs, (q.numpy(), k.numpy(), v.numpy()), strict=True)}
    run_result, eager = backend.run(compiled, input_data=data, pre_run=ref)
    got = list(run_result.outputs.values())[0].flatten().astype(np.float32)
    assert not np.any(np.isnan(got)), f"GQA symbolic warp-chain flash seq={seq} produced NaN"
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 5e-3, f"GQA symbolic warp-chain flash seq={seq} max_diff={max_diff:.2e}"


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
