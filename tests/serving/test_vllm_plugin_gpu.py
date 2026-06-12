"""In-process vLLM engine test of the deplodock embedding plugin.

``perf``-marked (deselected by default — run with ``pytest -m perf``): needs
CUDA, vllm, and the Qwen3-Embedding-0.6B checkpoint, and spends ~2 min
compiling the whole model at startup. The fast everywhere-tests for this
package live in ``test_packed.py``.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

MODEL = "Qwen/Qwen3-Embedding-0.6B"


def test_vllm_plugin_embed_matches_hf_eager(monkeypatch):
    pytest.importorskip("cupy")
    vllm = pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # The test process has CUDA initialized (conftest seeds it); vLLM's forked
    # EngineCore would die on re-init. Run the engine in-process instead.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    texts = [
        "What is the capital of France?",
        "Paris is the capital and largest city of France.",
        "The mitochondria is the powerhouse of the cell.",
    ]

    llm = vllm.LLM(
        model=MODEL,
        runner="pooling",
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        hf_overrides={"architectures": ["DeplodockEmbedModel"]},
    )
    outs = llm.embed(texts)
    got = np.array([o.outputs.embedding for o in outs], dtype=np.float64)

    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    ref_model = AutoModel.from_pretrained(MODEL, dtype=torch.float32).eval()
    refs = []
    for t in texts:
        ids = torch.tensor(tok(t)["input_ids"])[None, :]
        with torch.no_grad():
            h = ref_model(input_ids=ids).last_hidden_state[0, -1].numpy()
        refs.append(h / np.linalg.norm(h))
    ref = np.array(refs)

    got /= np.linalg.norm(got, axis=1, keepdims=True)
    cos = (got * ref).sum(axis=1)
    assert cos.min() > 0.99, f"cosine vs HF eager too low: {cos}"
