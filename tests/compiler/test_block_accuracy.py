"""End-to-end accuracy test: compile a real transformer block and compare
against PyTorch eager on GPU.

Uses TinyLlama (ungated, small) to keep CI fast. This catches bugs that
small synthetic graphs miss — deep IndexMapOp chains, 5D+ tensor shapes,
multi-kernel pipelines with composed coordinate mappings.
"""

from __future__ import annotations

import pytest
import torch

from deplodock.compiler.backend.cuda.runner import has_cuda_gpu, has_nvcc

requires_cuda = pytest.mark.skipif(
    not has_nvcc() or not has_cuda_gpu(),
    reason="CUDA not available (need nvcc + GPU)",
)


def _compile_and_run_block(model_id: str, seq_len: int = 32):
    """Compile a transformer block and compare against eager."""
    from transformers import AutoModelForCausalLM

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.torch_trace import trace_module

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    config = model.config
    block = model.model.layers[0].eval()

    hidden = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden // num_heads

    torch.manual_seed(42)
    x = torch.randn(1, seq_len, hidden)
    cos = torch.randn(1, 1, seq_len, head_dim)
    sin = torch.randn(1, 1, seq_len, head_dim)

    # Eager reference.
    block_gpu = block.cuda()
    with torch.no_grad():
        eager_out = block_gpu(x.cuda(), position_embeddings=(cos.cuda(), sin.cuda()))[0]
    eager_flat = eager_out.cpu().flatten().tolist()

    # Compile.
    graph = trace_module(block.cpu(), (x.cpu(),), kwargs={"position_embeddings": (cos.cpu(), sin.cpu())})

    backend = CudaBackend()
    compiled = backend.compile(graph)

    # Build input data.
    input_data: dict[str, list[float]] = {}
    for buf in compiled.buffers:
        if buf.role == "input":
            if buf.name == "hidden_states":
                input_data[buf.name] = x.cpu().flatten().tolist()
            elif buf.name == "position_embeddings_0":
                input_data[buf.name] = cos.cpu().flatten().tolist()
            elif buf.name == "position_embeddings_1":
                input_data[buf.name] = sin.cpu().flatten().tolist()
        elif buf.role == "constant":
            for key, param in block.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(buf.name[2:]) and param.numel() == buf.size:
                    input_data[buf.name] = param.detach().cpu().flatten().tolist()
                    break
            if buf.name not in input_data and buf.name in compiled.constant_values:
                input_data[buf.name] = [compiled.constant_values[buf.name]]

    # Run.
    run_result = backend.run(compiled, input_data=input_data)
    deplodock_flat = list(run_result.outputs.values())[0].flatten().tolist()

    return deplodock_flat, eager_flat


def _assert_accuracy(deplodock, eager, max_threshold=1.0):
    assert len(deplodock) == len(eager), f"output length mismatch: {len(deplodock)} vs {len(eager)}"
    assert not any(v != v for v in deplodock), "deplodock output contains NaN"
    assert sum(1 for v in deplodock if abs(v) > 1e-12) > len(deplodock) // 2, "deplodock output is mostly zeros"
    max_diff = max(abs(a - e) for a, e in zip(deplodock, eager, strict=True))
    mean_diff = sum(abs(a - e) for a, e in zip(deplodock, eager, strict=True)) / len(deplodock)
    assert max_diff < max_threshold, f"max_diff={max_diff:.4f}, mean_diff={mean_diff:.4f}"


@requires_cuda
def test_tinyllama_block_accuracy():
    """TinyLlama block: deplodock output matches PyTorch eager within tolerance."""
    deplodock, eager = _compile_and_run_block("TinyLlama/TinyLlama-1.1B-Chat-v1.0", seq_len=32)
    _assert_accuracy(deplodock, eager)


@requires_cuda
@pytest.mark.xfail(reason="Fusion drops interior IndexMapOps; port out_shape < kernel output")
def test_qwen_block_accuracy():
    """Qwen 7B block (random weights): deplodock output matches PyTorch eager."""
    from transformers import AutoConfig, AutoModelForCausalLM

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.torch_trace import trace_module

    torch.set_default_dtype(torch.float32)
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-7B")
    model = AutoModelForCausalLM.from_config(config).float()
    block = model.model.layers[0].eval().float()
    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads

    torch.manual_seed(42)
    x = torch.randn(1, 32, hidden)
    cos = torch.randn(1, 1, 32, head_dim)
    sin = torch.randn(1, 1, 32, head_dim)

    block_gpu = block.cuda()
    with torch.no_grad():
        eager_out = block_gpu(x.cuda(), position_embeddings=(cos.cuda(), sin.cuda()))[0]
    eager_flat = eager_out.cpu().flatten().tolist()

    graph = trace_module(block.cpu(), (x,), kwargs={"position_embeddings": (cos, sin)})

    backend = CudaBackend()
    compiled = backend.compile(graph)

    input_data: dict[str, list[float]] = {}
    for buf in compiled.buffers:
        if buf.role == "input":
            if buf.name == "hidden_states":
                input_data[buf.name] = x.flatten().tolist()
            elif buf.name == "position_embeddings_0":
                input_data[buf.name] = cos.flatten().tolist()
            elif buf.name == "position_embeddings_1":
                input_data[buf.name] = sin.flatten().tolist()
        elif buf.role == "constant":
            for key, param in block.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(buf.name[2:]) and param.numel() == buf.size:
                    input_data[buf.name] = param.detach().cpu().float().flatten().tolist()
                    break
            if buf.name not in input_data and buf.name in compiled.constant_values:
                input_data[buf.name] = [compiled.constant_values[buf.name]]

    run_result = backend.run(compiled, input_data=input_data)
    deplodock_flat = list(run_result.outputs.values())[0].flatten().tolist()
    _assert_accuracy(deplodock_flat, eager_flat)
