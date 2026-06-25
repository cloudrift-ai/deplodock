"""Flash-attention recognition at Loop IR — GPU parity vs torch SDPA.

The ``FLASH`` knob drives a **Loop-IR** pass (``loop/fusion/025_recognize_flash``)
that runs AFTER the generic fuser and pattern-matches the consolidated softmax-
attention kernel (post-fusion a non-causal SDPA is two ``LoopOp``s: scaled scores
+ a softmax-then-P@V kernel), rewriting it into a single streaming online-softmax
kernel (the ``Monoid`` carrier) — with NO modification to the decomposition
stage. These tests pin: (1) with ``FLASH`` on,
non-causal SDPA fuses to one kernel matching torch on the GPU (static + dynamic
symbolic ``seq_len``); (2) with ``FLASH`` off, the score-materializing
decomposition is untouched (default unchanged); (3) causal, explicit additive
mask, and GQA (``head // group``) are recognized structurally from the fused body
and fuse to one masked nest, static and dynamic — the scalar-tier masked/GQA flash
from ``plans/masked-gqa-mma-flash-attention.md``. The score producer must be a
clean scaled-QK (RoPE-fused producers — real decoder layers — are a follow-up; see
the plan). The MMA tensor-core tier remains future work
(``plans/online-softmax-flash-attention.md``).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ..conftest import requires_cuda


class _Sdpa(torch.nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v)


def _compile_sdpa(q, k, v):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    graph = trace_module(_Sdpa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, kernels


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 1, 8, 8), (1, 2, 16, 8), (2, 3, 32, 16)])
def test_flash_sdpa_matches_torch(monkeypatch, B, H, S, D):
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))

    backend, compiled, kernels = _compile_sdpa(q, k, v)
    # One fused kernel — no [S, S] score matrix materialized.
    assert len(kernels) == 1, f"flash should fuse to one kernel, got {len(kernels)}"
    src = compiled.nodes[kernels[0]].op.kernel_source
    assert "fmaxf" in src and "expf" in src, "fused kernel should carry the streaming softmax (max + exp)"

    cuda_q, cuda_k, cuda_v = q.cuda(), k.cuda(), v.cuda()

    def eager_pre_run() -> np.ndarray:
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(cuda_q, cuda_k, cuda_v).cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager_pre_run)
    got = list(run_result.outputs.values())[0].flatten()
    assert got.shape == eager.shape
    assert not np.any(np.isnan(got))
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 1e-4, f"flash vs torch SDPA max_diff={max_diff:.6e}"


@requires_cuda
@pytest.mark.parametrize("bk", [2, 4])
def test_flash_sdpa_kv_tile_matches_torch(monkeypatch, bk):
    """Phase 1 KV tiling (``plans/tensor-core-streaming-flash-mma.md``): a ``DEPLODOCK_BK``
    pin re-brackets the streaming reduce ``S_k → S_k/BK · BK`` (serial within the tile). The
    fused flash kernel must still match torch — BN=1 is degenerate-identical, BN>1 the
    re-bracketed serial fold. ``S=32`` / ``D=16`` are divisible by both 2 and 4, so the pin
    is honored (``_streaming_bk`` requires every reduce extent divisible)."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_BK", str(bk))
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 3, 32, 16) for _ in range(3))

    backend, compiled, kernels = _compile_sdpa(q, k, v)
    assert len(kernels) == 1, f"flash should still fuse to one kernel under BK={bk}, got {len(kernels)}"

    def eager_pre_run() -> np.ndarray:
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager_pre_run)
    got = list(run_result.outputs.values())[0].flatten()
    assert not np.any(np.isnan(got))
    max_diff = float(np.max(np.abs(got - eager)))
    assert max_diff < 1e-4, f"BK={bk} KV-tiled flash vs torch SDPA max_diff={max_diff:.6e}"


@requires_cuda
@pytest.mark.parametrize(("B", "H", "S", "D"), [(1, 1, 8, 8), (1, 2, 16, 8), (2, 3, 32, 16)])
def test_flash_chain_matches_torch(monkeypatch, B, H, S, D):
    """Phase 1c (``plans/tensor-core-streaming-flash-mma.md``): ``DEPLODOCK_CHAIN=1``
    restructures the streaming flash into the FA-2 **shared-score** form — the P@V output
    ``d`` rides a register vector ``O[BM, D]``, the QK^T score is computed once per KV step
    and shared across ``d`` (the INLINE score edge), the twisted carrier splits into a
    scalar stats cell + a register-tiled accumulation cell. Still one kernel, still matches
    torch (scalar FMA P@V — the first accuracy check of the crux)."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    torch.manual_seed(0)
    q, k, v = (torch.randn(B, H, S, D) for _ in range(3))
    backend, compiled, kernels = _compile_sdpa(q, k, v)
    assert len(kernels) == 1, f"chain flash should fuse to one kernel, got {len(kernels)}"
    # The shared-score form carries the register accumulator vector O_i_0 (not a grid-d scalar).
    assert "O_i_0" in compiled.nodes[kernels[0]].op.kernel_source, "chain form must carry the O[d] register vector"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().numpy()

    md = _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
    assert md < 1e-4, f"chain flash max_diff={md:.6e}"


@requires_cuda
def test_flash_chain_causal_and_gqa_match_torch(monkeypatch):
    """The chain form keeps the causal / GQA masks in the ``d``-invariant score prefix, so
    masked + grouped-head flash also shares the score and matches torch."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.setenv("DEPLODOCK_CHAIN", "1")
    torch.manual_seed(0)

    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    backend, compiled, kernels = _compile_causal(q, k, v)
    assert len(kernels) == 1

    def rc():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True).cpu().flatten().numpy()

    assert _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, rc) < 1e-4

    qg = torch.randn(1, 4, 16, 8)
    kg, vg = (torch.randn(1, 2, 16, 8) for _ in range(2))
    graph = trace_module(_Gqa().cpu(), (qg, kg, vg))
    compiled = CudaBackend().compile(graph)

    def rg():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(qg.cuda(), kg.cuda(), vg.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .numpy()
            )

    assert _run_flash(CudaBackend(), compiled, {"q": qg.numpy(), "k": kg.numpy(), "v": vg.numpy()}, rg) < 1e-4


@requires_cuda
def test_flash_chain_default_off_keeps_scalar_stream(monkeypatch):
    """Greedy default (no ``CHAIN`` pin) keeps the scalar streaming nest — the
    restructuring is opt-in until the search-fork integration (Phase 6), so the deployed
    flash kernel is unchanged."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    monkeypatch.delenv("DEPLODOCK_CHAIN", raising=False)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    _backend, compiled, kernels = _compile_sdpa(q, k, v)
    assert "O_i_0" not in compiled.nodes[kernels[0]].op.kernel_source, "default flash must stay the scalar streaming nest"


@requires_cuda
def test_flash_sdpa_dynamic_matches_torch(monkeypatch):
    """Symbolic ``seq_len`` (Q/K/V dim -2): ONE cached kernel carrying ``int
    seq_len`` serves every runtime size — flash's single dynamic axis lands on
    BOTH the query (masked-row M) and KV (reduce) positions."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    B, H, D = 1, 2, 8
    seq = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _Sdpa().cpu(),
        (torch.randn(B, H, 16, D), torch.randn(B, H, 16, D), torch.randn(B, H, 16, D)),
        dynamic_shapes={"q": {2: seq}, "k": {2: seq}, "v": {2: seq}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic flash should fuse to one kernel, got {len(kernels)}"
    assert "int seq_len" in compiled.nodes[kernels[0]].op.kernel_source, "dynamic kernel must carry the runtime seq_len arg"

    for s in (8, 16, 37):
        q, k, v = (torch.randn(B, H, s, D) for _ in range(3))

        def eager_pre_run(q=q, k=k, v=v) -> np.ndarray:
            with torch.no_grad():
                return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda()).cpu().flatten().numpy()

        run_result, eager = backend.run(compiled, input_data={"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, pre_run=eager_pre_run)
        got = list(run_result.outputs.values())[0].flatten()
        max_diff = float(np.max(np.abs(got - eager)))
        assert max_diff < 1e-4, f"dynamic flash seq={s} max_diff={max_diff:.6e}"


@requires_cuda
def test_flash_off_keeps_decomposition(monkeypatch):
    """FLASH off — the score-materializing 010_sdpa decomposition (multiple
    kernels: QK^T, softmax, P@V), proving the knob gates the Loop-IR recognition
    and the default is unchanged. The move composer implies flash only for
    SYMBOLIC-seq SDPA, so this static case decomposes under the composer too."""
    monkeypatch.delenv("DEPLODOCK_FLASH", raising=False)
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    _backend, _compiled, kernels = _compile_sdpa(q, k, v)
    assert len(kernels) > 1, "FLASH off should keep the score-materializing multi-kernel path"


def _run_flash(backend, compiled, inputs: dict, ref_fn) -> float:
    """Compile-then-run helper: returns max|deplodock − torch| for a fused flash
    kernel. ``inputs`` keys must match the traced graph's input names."""
    run_result, eager = backend.run(compiled, input_data=inputs, pre_run=ref_fn)
    got = list(run_result.outputs.values())[0].flatten()
    assert got.shape == eager.shape
    assert not np.any(np.isnan(got))
    return float(np.max(np.abs(got - eager)))


@requires_cuda
def test_flash_causal_matches_torch(monkeypatch):
    """Causal SDPA: the recognizer must DETECT the per-element ``kv ≤ m`` mask
    (the lifted causal IndexMapOp ``Select``) and build the masked nest — building
    an unmasked nest was a silent-correctness trap (wrong output, not a fall-
    through)."""
    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    backend, compiled, kernels = _compile_causal(q, k, v)
    assert len(kernels) == 1, f"causal flash should fuse to one kernel, got {len(kernels)}"

    def ref():
        with torch.no_grad():
            return torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True).cpu().flatten().numpy()

    md = _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
    assert md < 1e-4, f"causal flash max_diff={md:.6e}"


class _Causal(torch.nn.Module):
    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)


def _compile_causal(q, k, v):
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    graph = trace_module(_Causal().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    return backend, compiled, kernels


class _Gqa(torch.nn.Module):
    """GQA SDPA. NOTE: ``enable_gqa=True`` is a bool kwarg, which the tracer's
    is_causal scan grabs (the default ``is_causal=False`` is dropped by dynamo),
    so this traces as GQA **and** causal — the only GQA form reachable through the
    public torch API here, and exactly the Qwen3-Embedding layer-0 shape."""

    def forward(self, q, k, v):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)


@requires_cuda
@pytest.mark.parametrize(("Hq", "Hkv", "S", "D"), [(4, 2, 16, 8), (16, 8, 32, 16)])
def test_flash_gqa_matches_torch(monkeypatch, Hq, Hkv, S, D):
    """GQA flash: K/V read at ``head // group`` directly (no materialized
    broadcast). Traces as GQA+causal (see ``_Gqa``)."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    q = torch.randn(1, Hq, S, D)
    k, v = (torch.randn(1, Hkv, S, D) for _ in range(2))
    graph = trace_module(_Gqa().cpu(), (q, k, v))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"GQA flash should fuse to one kernel, got {len(kernels)}"

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True, enable_gqa=True)
                .cpu()
                .flatten()
                .numpy()
            )

    md = _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
    assert md < 1e-4, f"GQA flash max_diff={md:.6e}"


class _Masked(torch.nn.Module):
    def forward(self, q, k, v, mask):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@requires_cuda
def test_flash_additive_mask_matches_torch(monkeypatch):
    """Explicit additive ``(1,1,S,S)`` bias (the HF whole-model mask path): the
    recognizer detects the ``add(score, Load(mask))`` and loads the bias per
    ``(m, kv)`` in the nest. ``-inf`` entries zero out via ``exp``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    S, D = 16, 8
    q, k, v = (torch.randn(1, 2, S, D) for _ in range(3))
    mask = torch.zeros(1, 1, S, S)
    mask[0, 0, :, S // 2 :] = float("-inf")  # mask out the second half of the keys
    graph = trace_module(_Masked().cpu(), (q, k, v, mask))
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"masked flash should fuse to one kernel, got {len(kernels)}"

    def ref():
        with torch.no_grad():
            return (
                torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=mask.cuda())
                .cpu()
                .flatten()
                .numpy()
            )

    md = _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()}, ref)
    assert md < 1e-4, f"masked flash max_diff={md:.6e}"


@requires_cuda
def test_flash_gqa_dynamic_matches_torch(monkeypatch):
    """GQA+causal over symbolic ``seq_len`` — one cached kernel, the dynamic axis
    on the masked-row M, the symbolic reduce, AND the causal guard at once."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    Hq, Hkv, D = 4, 2, 8
    seq = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _Gqa().cpu(),
        (torch.randn(1, Hq, 16, D), torch.randn(1, Hkv, 16, D), torch.randn(1, Hkv, 16, D)),
        dynamic_shapes={"q": {2: seq}, "k": {2: seq}, "v": {2: seq}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic GQA flash should fuse to one kernel, got {len(kernels)}"

    for s in (8, 16, 37):
        q = torch.randn(1, Hq, s, D)
        k, v = (torch.randn(1, Hkv, s, D) for _ in range(2))

        def ref(q=q, k=k, v=v):
            with torch.no_grad():
                return (
                    torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), is_causal=True, enable_gqa=True)
                    .cpu()
                    .flatten()
                    .numpy()
                )

        md = _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy()}, ref)
        assert md < 1e-4, f"dynamic GQA flash seq={s} max_diff={md:.6e}"


@requires_cuda
def test_flash_additive_mask_dynamic_matches_torch(monkeypatch):
    """Symbolic-seq additive mask ``(1,1,seq,seq)``: the masked final KV tile
    zero-fills past the runtime extent, consistent with the mask's own ``-inf``."""
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    monkeypatch.setenv("DEPLODOCK_FLASH", "1")
    torch.manual_seed(0)
    D = 8
    seq = torch.export.Dim("seq_len", min=4, max=4096)
    graph = trace_module(
        _Masked().cpu(),
        (torch.randn(1, 2, 16, D), torch.randn(1, 2, 16, D), torch.randn(1, 2, 16, D), torch.zeros(1, 1, 16, 16)),
        dynamic_shapes={"q": {2: seq}, "k": {2: seq}, "v": {2: seq}, "mask": {2: seq, 3: seq}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    kernels = [nid for nid in compiled.nodes if getattr(compiled.nodes[nid].op, "kernel_source", None)]
    assert len(kernels) == 1, f"dynamic masked flash should fuse to one kernel, got {len(kernels)}"

    for s in (8, 16, 37):
        q, k, v = (torch.randn(1, 2, s, D) for _ in range(3))
        mask = torch.zeros(1, 1, s, s)
        mask[0, 0, :, s // 2 :] = float("-inf")

        def ref(q=q, k=k, v=v, mask=mask):
            with torch.no_grad():
                return (
                    torch.nn.functional.scaled_dot_product_attention(q.cuda(), k.cuda(), v.cuda(), attn_mask=mask.cuda())
                    .cpu()
                    .flatten()
                    .numpy()
                )

        md = _run_flash(backend, compiled, {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()}, ref)
        assert md < 1e-4, f"dynamic masked flash seq={s} max_diff={md:.6e}"
