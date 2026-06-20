"""In-process vLLM engine test of the deplodock **generative** plugin (Phase 3).

``perf``-marked (deselected by default): needs CUDA + cupy + vllm. Saves a TINY random
Llama (vocab matches a cached Llama tokenizer, 2 layers — no network), serves it through
``DeplodockGenModel`` in an in-process vLLM engine (real paged ``Attention`` + KV cache +
``lm_head`` + ``get_rope``), and greedily generates — checking it runs end-to-end and that
the generated tokens agree with HF eager greedy on the same weights. This is the Phase-3
integration proof: vLLM accepts the model, allocates the KV-cache spec from the per-layer
``Attention`` prefixes, and the deplodock↔attention forward interleave produces correct logits.
"""

import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]

TOKENIZER = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # cached; vocab 32000


def _save_tiny_llama(path):
    import torch
    import transformers

    config = transformers.LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval().to(torch.float16)
    model.save_pretrained(path)
    return config


def test_vllm_gen_plugin_matches_hf_eager(tmp_path, monkeypatch):
    pytest.importorskip("cupy")
    vllm = pytest.importorskip("vllm")
    import torch
    import transformers

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    import deplodock.serving

    # The test process has CUDA initialized (conftest seeds it); vLLM's forked
    # EngineCore would die on re-init. Run the engine in-process instead.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    model_dir = tmp_path / "tiny_llama"
    _save_tiny_llama(str(model_dir))
    deplodock.serving.register()  # ModelRegistry.register_model("DeplodockGenModel", ...)

    tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER)
    prompt_ids = tok("The quick brown fox", add_special_tokens=True)["input_ids"]
    max_new = 8

    llm = vllm.LLM(
        model=str(model_dir),
        tokenizer=TOKENIZER,
        runner="generate",
        hf_overrides={"architectures": ["DeplodockGenModel"]},
        enforce_eager=True,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.4,
    )
    out = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), SamplingParams(temperature=0.0, max_tokens=max_new))
    gen = list(out[0].outputs[0].token_ids)

    # end-to-end gate: ran, produced max_new valid in-vocab tokens
    assert len(gen) == max_new
    assert all(0 <= t < 32000 for t in gen)

    # correctness gate: greedy agrees with HF eager on the same weights
    ref = transformers.LlamaForCausalLM.from_pretrained(str(model_dir), dtype=torch.float16).to("cuda").eval()
    with torch.no_grad():
        ref_out = ref.generate(torch.tensor([prompt_ids], device="cuda"), do_sample=False, max_new_tokens=max_new, use_cache=True)
    ref_gen = ref_out[0, len(prompt_ids) :].tolist()
    assert gen[0] == ref_gen[0], f"first token mismatch: deplodock {gen[0]} vs HF {ref_gen[0]}"
