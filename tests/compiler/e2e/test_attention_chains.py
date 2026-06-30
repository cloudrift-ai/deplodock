"""Scoped regression tests for the masked-attention path under the
larger-PAT tile shape.

When ``test_block_accuracy::test_tinyllama_block_accuracy[cuda]`` fails,
these tests bisect *where* in the attention sub-block the divergence
appears. Three increasingly-comprehensive chains:

1. :func:`test_two_linears_tinyllama_shape` — chained Linears at
   TinyLlama hidden=2048. Should always be exact (drift only). If this
   fails, every matmul is broken.
2. :func:`test_qkv_attn_no_rope` — Q/K/V projections + masked SDPA + O
   projection, but **no RoPE**. Isolates whether the bug is in matmul
   chaining + masked SDPA composition.
3. :func:`test_full_self_attn_tinyllama` — the real ``LlamaAttention``
   sub-module from the TinyLlama config: Q/K/V Linears + RoPE + masked
   SDPA + O Linear. Smallest scope that reproduces the full-block
   accuracy regression. If (1) and (2) pass but this fails, the bug is
   in the RoPE elementwise kernel composed with the surrounding
   attention numerics.

Random fp32 weights are used so we hit the same magnitude regime that
``test_block_accuracy``'s composite test exercises. The thresholds are
tight (1e-4) — these chains have so few kernels that benign drift
shouldn't exceed it; anything larger is a real miscompute.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ..conftest import from_pretrained_or_skip, requires_cuda


@pytest.fixture(autouse=True)
def _seed(monkeypatch):
    torch.manual_seed(42)
    # Pin a small, budget-safe scalar tile. These chains compile the real attention path
    # UNPINNED, which relied on the learned/analytic prior to pick an in-smem-budget tile;
    # with the prior retired (algebra-knob-schema "break it"), the cold emission-order pick
    # can choose an over-budget tile (deep BK staging slab / wide MONOID-combine smem) and
    # hard-fail. The tile is irrelevant to these accuracy checks, so pin a fitting one
    # (legacy env pins route through the ingest mapper).
    for k, v in (("BN", "16"), ("BM", "8"), ("FN", "2"), ("FM", "2"), ("BK", "8"), ("BR", "4")):
        monkeypatch.setenv(f"EMMY_{k}", v)


def _run_module_with_eager(module: torch.nn.Module, args: tuple, inputs_by_name: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Trace + compile ``module``, then run the emmy kernels and
    the torch eager reference under one ``backend.run`` GPU-lock window
    via the ``pre_run`` callback. Returns ``(emmy_flat, eager_flat)``.

    Keeping both halves under the same lock means peer xdist workers
    can't interleave CUDA work between the eager forward and the
    emmy comparison — which used to surface as small numerical
    divergence (max_diff ~0.03 on max_eager ~1.2) that flaked the
    accuracy assertion."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(module.cpu(), args)
    backend = CudaBackend()
    compiled = backend.compile(graph)

    input_set = set(compiled.inputs)
    feed: dict[str, np.ndarray] = {}
    for nid in compiled.nodes:
        node = compiled.nodes[nid]
        if nid in input_set and nid in inputs_by_name:
            feed[nid] = inputs_by_name[nid]
        elif isinstance(node.op, ConstantOp):
            from emmy.compiler.loader.binder import apply_load_ops

            n = 1
            for d in node.output.shape:
                n *= d.as_static()
            for key, p in module.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                # Match by the ConstantOp's stored name (which carries
                # the placeholder identity through ``004a`` const-fold,
                # even when the graph node id changes).
                if safe_key.endswith(node.op.name[2:]) and p.numel() == n:
                    arr = p.detach().cpu().numpy()
                    feed[nid] = apply_load_ops(arr, node.op.load_ops)
                    break
            if nid not in feed and node.op.value is not None:
                feed[nid] = np.array([node.op.value], dtype=np.float32)

    cuda_module = module.cuda()
    cuda_args = tuple(a.cuda() for a in args)

    def eager_pre_run() -> np.ndarray:
        with torch.no_grad():
            out = cuda_module(*cuda_args)
        if isinstance(out, tuple):
            out = out[0]
        return out.cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data=feed, pre_run=eager_pre_run)
    dpd = list(run_result.outputs.values())[0].flatten()
    return dpd, eager


def _assert_close(emmy: np.ndarray, eager: np.ndarray, threshold: float = 1e-4) -> None:
    assert emmy.shape == eager.shape, f"shape: {emmy.shape} vs {eager.shape}"
    assert not np.any(np.isnan(emmy)), "emmy output has NaN"
    max_diff = float(np.max(np.abs(emmy - eager)))
    mean_diff = float(np.mean(np.abs(emmy - eager)))
    max_eager = float(np.max(np.abs(eager)))
    assert max_diff < threshold, f"max_diff={max_diff:.6f} >= {threshold} (mean={mean_diff:.6f}, max_eager={max_eager:.3f})"


# --- Layer 1: chained Linears (no SDPA, no RoPE) -----------------------------


class _StackedLinears(torch.nn.Module):
    def __init__(self, hidden: int = 2048):
        super().__init__()
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.o = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o(self.q(x))


@requires_cuda
def test_two_linears_tinyllama_shape():
    """Two chained 2048×2048 Linears at TinyLlama hidden size and seq=32.
    Confirms basic matmul-chain accuracy."""
    m = _StackedLinears().eval()
    x = torch.randn(1, 32, 2048)
    dpd, eager = _run_module_with_eager(m, (x,), {"x": x.numpy()})
    _assert_close(dpd, eager)


# --- Layer 2: full attention pipeline minus RoPE -----------------------------


class _QKVAttnNoRope(torch.nn.Module):
    def __init__(self, hidden: int = 2048, n_heads: int = 32, head_dim: int = 64):
        super().__init__()
        self.h, self.d = n_heads, head_dim
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.k = torch.nn.Linear(hidden, hidden, bias=False)
        self.v = torch.nn.Linear(hidden, hidden, bias=False)
        self.o = torch.nn.Linear(hidden, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q(x).view(B, S, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(B, S, self.h, self.d).transpose(1, 2)
        v = self.v(x).view(B, S, self.h, self.d).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o(out)


@requires_cuda
def test_qkv_attn_no_rope():
    """Q/K/V Linears + causal SDPA + O Linear, no RoPE. Confirms that
    the matmul-chain + masked-SDPA composition is numerically sound on
    its own."""
    m = _QKVAttnNoRope().eval()
    x = torch.randn(1, 32, 2048)
    dpd, eager = _run_module_with_eager(m, (x,), {"x": x.numpy()})
    _assert_close(dpd, eager)


class _SdpaExplicitMask(torch.nn.Module):
    """SDPA fed an explicit additive float ``attn_mask`` (the way HF passes
    its precomputed causal mask) rather than ``is_causal=True``."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@requires_cuda
@pytest.mark.parametrize("n_heads,seq_len", [(1, 32), (16, 32)])
def test_sdpa_explicit_additive_mask(n_heads: int, seq_len: int):
    """SDPA with an explicit additive float mask must apply the mask, not
    silently drop it. Regression for the tracer capturing only ``Q/K/V`` and
    discarding ``attn_mask`` — which turned the whole-model causal attention
    into full bidirectional attention. Invisible to uniform input (a softmax-
    weighted sum of identical V is V regardless of the weights), so this uses
    varying random Q/K/V and a tight threshold to actually exercise masking.
    The mask is a ``(1, 1, S, S)`` additive bias broadcast over the head axis."""
    head_dim = 128
    m = _SdpaExplicitMask().eval()
    q = torch.randn(1, n_heads, seq_len, head_dim)
    k = torch.randn(1, n_heads, seq_len, head_dim)
    v = torch.randn(1, n_heads, seq_len, head_dim)
    mask = torch.zeros((seq_len, seq_len))
    mask.masked_fill_(torch.triu(torch.ones_like(mask, dtype=torch.bool), diagonal=1), float("-inf"))
    mask = mask[None, None]
    dpd, eager = _run_module_with_eager(m, (q, k, v, mask), {"q": q.numpy(), "k": k.numpy(), "v": v.numpy(), "mask": mask.numpy()})
    _assert_close(dpd, eager)


# --- Layer 3: real LlamaAttention (= block.self_attn) ------------------------


def _run_self_attn_tinyllama(seq_len: int, threshold: float = 1e-4) -> None:
    """Run TinyLlama's ``LlamaAttention`` sub-module at the given seq_len
    and verify emmy matches eager within ``threshold``."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = from_pretrained_or_skip(AutoConfig.from_pretrained, "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    config.num_hidden_layers = 1
    block = AutoModelForCausalLM.from_config(config).float().model.layers[0].eval()
    attn = block.self_attn

    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads

    x = torch.randn(1, seq_len, hidden)
    cos = torch.randn(1, 1, seq_len, head_dim)
    sin = torch.randn(1, 1, seq_len, head_dim)

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.loader.binder import apply_load_ops
    from emmy.compiler.trace.torch import trace_module

    attn_cpu = attn.cpu()
    graph = trace_module(attn_cpu, (x,), kwargs={"position_embeddings": (cos, sin)})
    backend = CudaBackend()
    compiled = backend.compile(graph)

    input_set = set(compiled.inputs)
    feed: dict[str, np.ndarray] = {}
    for nid in compiled.nodes:
        node = compiled.nodes[nid]
        if nid in input_set:
            if nid == "hidden_states":
                feed[nid] = x.numpy()
            elif nid == "position_embeddings_0":
                feed[nid] = cos.numpy()
            elif nid == "position_embeddings_1":
                feed[nid] = sin.numpy()
        elif isinstance(node.op, ConstantOp):
            n = 1
            for d in node.output.shape:
                n *= d.as_static()
            for key, p in attn_cpu.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(node.op.name[2:]) and p.numel() == n:
                    arr = p.detach().cpu().numpy()
                    feed[nid] = apply_load_ops(arr, node.op.load_ops)
                    break
            if nid not in feed and node.op.value is not None:
                feed[nid] = np.array([node.op.value], dtype=np.float32)

    attn_cuda = attn.cuda()
    x_cuda, cos_cuda, sin_cuda = x.cuda(), cos.cuda(), sin.cuda()

    def eager_pre_run() -> np.ndarray:
        # Force the math (naive) SDPA backend so eager and emmy
        # compare the same algorithm — flash-attention re-orders FMAs
        # and would otherwise drift O(0.5 × max_eager) from naive at
        # seq ≥ 512. Runs inside ``backend.run``'s GPU lock so the
        # eager forward and the emmy launches share one
        # uninterrupted GPU window.
        with torch.no_grad(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            out = attn_cuda(x_cuda, position_embeddings=(cos_cuda, sin_cuda))[0]
        return out.cpu().flatten().numpy()

    run_result, eager = backend.run(compiled, input_data=feed, pre_run=eager_pre_run)
    dpd = list(run_result.outputs.values())[0].flatten()
    _assert_close(dpd, eager, threshold=threshold)


@requires_cuda
def test_full_self_attn_tinyllama():
    """The real ``LlamaAttention`` from a TinyLlama config — the smallest
    scope that includes Q/K/V Linears, RoPE, masked SDPA, and O Linear
    with the actual decomposition graph the block test exercises. If
    this fails while (1) and (2) pass, the regression is in the RoPE
    elementwise kernel or its interaction with the surrounding
    attention numerics."""
    _run_self_attn_tinyllama(seq_len=32, threshold=1e-4)


@requires_cuda
def test_full_self_attn_tinyllama_seq512():
    """Same as ``test_full_self_attn_tinyllama`` but at seq_len=512 — the
    shape that makes the SDPA P@V kernel
    (``k_scaled_dot_product_attention_reduce_reduce``) the dominant
    cost. At this size:

    - The masked-attention matrix is 32 × 512 × 512 = 32 MB, materialized
      to HBM and re-read by the P@V pass (no flash-attention fusion in
      the current pipeline).
    - The P@V pass currently emits one CTA per output element
      (heads × q_pos × head_dim = 1 M CTAs) with redundant softmax
      recomputation across the 64 CTAs sharing a (head, q_pos) row.

    This test pins correctness at this shape so future fusion /
    cooperative-output-tiling work doesn't regress accuracy. Threshold
    is loose (2.0 ≈ 90% of max_eager) because at seq=512 with random
    fp32 weights the naive-vs-naive comparison drifts substantially —
    the softmax over 512 random scores plus 2048-K projections are at
    the edge of fp32 precision, and TMA (default on sm_90+) further
    reorders FMAs vs cp.async (verified via EMMY_FP64_ACC=1).
    The intent is to catch order-of-magnitude regressions (NaN, sign
    flip, off-by-N, structural miscompute), not bit-equivalence; a
    future flash-style fused-attention recipe should let us tighten
    this dramatically.
    """
    _run_self_attn_tinyllama(seq_len=512, threshold=2.0)
