"""End-to-end accuracy test: compile a real transformer block and compare
against PyTorch eager on GPU.

Both models use random weights for a single block (num_hidden_layers=1),
which keeps the test fast and avoids weight downloads. This catches bugs
that small synthetic graphs miss — deep IndexMapOp chains, 5D+ tensor
shapes, multi-kernel pipelines with composed coordinate mappings.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _compile_and_run_block(model_id: str, seq_len: int = 32):
    """Compile a single transformer block (random weights) and compare against eager."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.torch_trace import trace_module

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

    # Eager reference.
    block_gpu = block.cuda()
    with torch.no_grad():
        eager_out = block_gpu(x.cuda(), position_embeddings=(cos.cuda(), sin.cuda()))[0]
    eager_flat = eager_out.cpu().flatten().tolist()

    # Compile.
    graph = trace_module(block.cpu(), (x,), kwargs={"position_embeddings": (cos, sin)})

    backend = CudaBackend()
    compiled = backend.compile(graph)

    input_data: dict[str, np.ndarray] = {}
    for buf in compiled.buffers:
        if buf.role == "input":
            if buf.name == "hidden_states":
                input_data[buf.name] = x.numpy()
            elif buf.name == "position_embeddings_0":
                input_data[buf.name] = cos.numpy()
            elif buf.name == "position_embeddings_1":
                input_data[buf.name] = sin.numpy()
        elif buf.role == "constant":
            for key, param in block.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(buf.name[2:]) and param.numel() == buf.size:
                    input_data[buf.name] = param.detach().cpu().numpy()
                    break
            if buf.name not in input_data and buf.name in compiled.constant_values:
                input_data[buf.name] = np.array([compiled.constant_values[buf.name]], dtype=np.float32)

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
    assert max_diff < max_threshold, f"max_diff={max_diff:.6f} >= {max_threshold:.6f} (mean_diff={mean_diff:.6f}, max_eager={max_eager:.3f})"
    assert mean_diff < mean_threshold, f"mean_diff={mean_diff:.6f} >= {mean_threshold:.6f} (max_diff={max_diff:.6f})"


@requires_cuda
def test_tinyllama_block_accuracy():
    """TinyLlama block: deplodock output matches PyTorch eager within tolerance."""
    deplodock, eager = _compile_and_run_block("TinyLlama/TinyLlama-1.1B-Chat-v1.0", seq_len=32)
    _assert_accuracy(deplodock, eager)


@requires_cuda
def test_qwen_block_accuracy():
    """Qwen 7B block: deplodock output matches PyTorch eager within tolerance."""
    deplodock, eager = _compile_and_run_block("Qwen/Qwen2.5-7B", seq_len=32)
    _assert_accuracy(deplodock, eager)
