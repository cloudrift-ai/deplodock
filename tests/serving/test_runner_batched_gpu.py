"""Static batched serving path (``DEPLODOCK_SERVING_STATIC=1``).

``perf``-marked (deselected by default — run with ``pytest -m perf``): needs CUDA
+ cupy + the Qwen3-Embedding config. Builds a 1-layer static ``(batch, S)`` trunk,
wraps a ``DeplodockForwardRunner`` around it, and checks that
``forward_hidden_states_batched`` runs several different-length sequences in ONE
padded batched forward and matches eager per row — the causal-independence claim
(a row's real prefix is unaffected by right-padding, dummy rows below the batch cap
are ignored). The whole-model accuracy gate lives in ``test_vllm_plugin_gpu.py``.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def test_runner_batched_matches_eager():
    pytest.importorskip("cupy")
    import torch
    from transformers import AutoConfig, AutoModel

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.loader.binder import bind_constants
    from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from deplodock.compiler.trace.torch import trace_module
    from deplodock.serving.runner import DeplodockForwardRunner

    torch.manual_seed(0)
    cfg = AutoConfig.from_pretrained(MODEL)
    cfg.num_hidden_layers = 1
    model = AutoModel.from_config(cfg).float().eval()

    B, S, dtype = 4, 32, torch.float32
    wrapper = build_full_model_wrapper(model, S, dtype, dynamic=True)
    example = (
        torch.zeros((B, S), dtype=torch.long),
        build_causal_mask(S, dtype),
        torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
    )
    compiled = CudaBackend().compile(trace_module(wrapper, example))  # fully static

    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)
    ids_name, mask_name, pos_name = compiled.inputs
    feed = {
        ids_name: np.zeros((B, S), dtype=np.int64),
        mask_name: build_causal_mask(S, dtype).numpy(),
        pos_name: np.tile(np.arange(S, dtype=np.int64), (B, 1)),
    }
    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed})
        runner = DeplodockForwardRunner(
            program=program,
            input_names=(ids_name, mask_name, pos_name),
            output_name=compiled.outputs[0],
            np_dtype=np.dtype("float32"),
            max_seq_len=S,
            batch_cap=B,
        )
        # Three different-length sequences (full, mid, short) + an implicit dummy
        # 4th row (only 3 supplied < batch_cap=4). Each must match eager on its own
        # length despite right-padding to S.
        lens = [S, 20, 7]
        seqs = [torch.from_numpy(((np.arange(n, dtype=np.int64) * 97) % 100) + 1).cuda() for n in lens]
        outs = runner.forward_hidden_states_batched(seqs)
        for n, out, ids_t in zip(lens, outs, seqs, strict=True):
            with torch.no_grad():
                ref = wrapper(ids_t.reshape(1, n).cpu(), build_causal_mask(n, dtype), torch.arange(n).reshape(1, n)).numpy()[0]
            got = out.cpu().numpy()
            assert got.shape == (n, 1024)
            np.testing.assert_allclose(got, ref, rtol=1e-3, atol=1e-3)
