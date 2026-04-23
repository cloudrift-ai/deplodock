"""End-to-end accuracy test: compile a real transformer block and compare
against PyTorch eager.

Both models use random weights for a single block (num_hidden_layers=1),
which keeps the test fast and avoids weight downloads. This catches bugs
that small synthetic graphs miss — deep IndexMapOp chains, 5D+ tensor
shapes, multi-kernel pipelines with composed coordinate mappings.

Runs under two backends:

- ``loop`` — ``LoopBackend`` (numpy interpreter), eager on CPU. Always on.
  Slow (~40s per TinyLlama run) but catches compile/fusion bugs without
  needing a GPU. Qwen is excluded from the CPU lane because the matmuls
  scale cubically and CPU numpy takes several minutes.
- ``cuda`` — ``CudaBackend``, eager on CUDA. Gated by ``@requires_cuda``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deplodock.compiler.backend.cuda.runtime import has_cuda_gpu

requires_cuda = pytest.mark.skipif(
    not has_cuda_gpu(),
    reason="CUDA not available (need cupy + GPU)",
)


def _compile_and_run_block(model_id: str, seq_len: int = 32, backend_kind: str = "cuda"):
    """Compile a single transformer block (random weights) and compare against eager.

    ``backend_kind`` selects ``"cuda"`` (CudaBackend + GPU eager) or
    ``"loop"`` (LoopBackend + CPU eager).
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(42)
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config).float()
    block = model.model.layers[0].eval()

    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads

    x = torch.randn(1, seq_len, hidden)
    cos = torch.randn(1, 1, seq_len, head_dim)
    sin = torch.randn(1, 1, seq_len, head_dim)

    if backend_kind == "cuda":
        from deplodock.compiler.backend.cuda.backend import CudaBackend

        block_eager = block.cuda()
        x_eager, cos_eager, sin_eager = x.cuda(), cos.cuda(), sin.cuda()
        with torch.no_grad():
            eager_out = block_eager(x_eager, position_embeddings=(cos_eager, sin_eager))[0]
        eager_flat = eager_out.cpu().flatten().tolist()
        backend = CudaBackend()
    elif backend_kind == "loop":
        from deplodock.compiler.backend.loop.backend import LoopBackend

        with torch.no_grad():
            eager_out = block(x, position_embeddings=(cos, sin))[0]
        eager_flat = eager_out.flatten().tolist()
        backend = LoopBackend()
    else:
        raise ValueError(f"Unknown backend_kind: {backend_kind!r}")

    graph = trace_module(block.cpu(), (x,), kwargs={"position_embeddings": (cos, sin)})
    compiled = backend.compile(graph)

    from deplodock.compiler.ir.base import ConstantOp

    input_set = set(compiled.inputs)

    def _numel(shape):
        n = 1
        for d in shape:
            n *= int(d)
        return n

    input_data: dict[str, np.ndarray] = {}
    for nid in compiled.nodes:
        node = compiled.nodes[nid]
        if nid in input_set:
            if nid == "hidden_states":
                input_data[nid] = x.numpy()
            elif nid == "position_embeddings_0":
                input_data[nid] = cos.numpy()
            elif nid == "position_embeddings_1":
                input_data[nid] = sin.numpy()
        elif isinstance(node.op, ConstantOp):
            size = _numel(node.output.shape)
            for key, param in block.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(nid[2:]) and param.numel() == size:
                    input_data[nid] = param.detach().cpu().numpy()
                    break
            if nid not in input_data and node.op.value is not None:
                input_data[nid] = np.array([node.op.value], dtype=np.float32)

    run_result = backend.run(compiled, input_data=input_data)
    deplodock_flat = list(run_result.outputs.values())[0].flatten().tolist()

    return deplodock_flat, eager_flat


def _assert_accuracy(deplodock, eager, max_threshold=1e-3, mean_threshold=1e-4):
    """Tight fp32-precision check: measured max_diff is ~1e-6 (TinyLlama) and
    ~7e-6 (Qwen) so 1e-3/1e-4 give two-to-three orders of slack over observed
    drift, tight enough to catch any real miscompute."""
    assert len(deplodock) == len(eager), f"output length mismatch: {len(deplodock)} vs {len(eager)}"
    assert not any(v != v for v in deplodock), "deplodock output contains NaN"
    assert sum(1 for v in deplodock if abs(v) > 1e-12) > len(deplodock) // 2, "deplodock output is mostly zeros"
    # Output magnitude sanity: eager should span a real range (not all tiny).
    max_eager = max(abs(e) for e in eager)
    assert max_eager > 0.1, f"eager output is suspiciously small (max_abs={max_eager}); threshold would be trivial"
    max_diff = max(abs(a - e) for a, e in zip(deplodock, eager, strict=True))
    mean_diff = sum(abs(a - e) for a, e in zip(deplodock, eager, strict=True)) / len(deplodock)
    assert max_diff < max_threshold, (
        f"max_diff={max_diff:.6f} >= {max_threshold:.6f} (mean_diff={mean_diff:.6f}, max_eager={max_eager:.3f})"
    )
    assert mean_diff < mean_threshold, f"mean_diff={mean_diff:.6f} >= {mean_threshold:.6f} (max_diff={max_diff:.6f})"


@pytest.mark.parametrize(
    "backend_kind,seq_len",
    [
        # CPU lane: ``LoopBackend`` + CPU eager. Always on. ``seq_len=8`` keeps
        # the numpy interpreter under ~5s while still exercising the full block
        # (QKV projections, rotary, SDPA + reduce sweeps, softmax, o_proj,
        # RMSNorm, SwiGLU MLP, residual add).
        pytest.param("loop", 8, id="cpu"),
        # CUDA lane: ``CudaBackend`` + GPU eager. Uses the full ``seq_len=32``.
        pytest.param("cuda", 32, id="cuda", marks=requires_cuda),
    ],
)
def test_tinyllama_block_accuracy(backend_kind, seq_len):
    """TinyLlama block: deplodock output matches PyTorch eager within tolerance."""
    deplodock, eager = _compile_and_run_block("TinyLlama/TinyLlama-1.1B-Chat-v1.0", seq_len=seq_len, backend_kind=backend_kind)
    _assert_accuracy(deplodock, eager)


@requires_cuda
def test_qwen_block_accuracy():
    """Qwen 7B block on CUDA: deplodock output matches PyTorch eager within tolerance."""
    deplodock, eager = _compile_and_run_block("Qwen/Qwen2.5-7B", seq_len=32, backend_kind="cuda")
    _assert_accuracy(deplodock, eager)
