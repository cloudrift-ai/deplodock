"""GPU spike for the Phase-0 generation oracle.

``perf``-marked (deselected by default): needs CUDA + cupy. Builds a TINY random-weight
Llama CausalLM (no network), compiles the whole-model fp16 dynamic path through
``_CompiledLM`` (full logits, last row sliced on the host), and checks the compiled
next-token logits against an eager fp16 reference across a few growing prefixes — the
plan's "compile-and-run spike" that de-risks whole-model lowering (int64 embedding-gather,
lm_head matmul) before the generate loop is trusted. The ``slice_last_logits`` xfail below
tracks the cold-path M=1 lm_head lowering gap that spike surfaced.
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]


def _tiny_llama():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval().to(torch.float16)
    return model


def test_generate_oracle_matches_eager_fp16():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.commands.generate import _CompiledLM

    # IMPORTANT: build_full_model_wrapper mutates the model in place (replaces rotary_emb
    # with _SlicedRotary, patches the causal-mask builder). Compare against an INDEPENDENT,
    # untouched model (same seed → identical weights, but HF's real rotary) so a wrapper/RoPE
    # bug can't hide by corrupting both sides.
    lm = _CompiledLM.from_model(_tiny_llama(), seq_len=8)
    ref = _tiny_llama().to("cuda").eval()

    prefixes = [[1, 2, 3], [1, 2, 3, 4], [5, 6, 7, 8, 9, 10]]
    for prefix in prefixes:
        dep = lm.logits(prefix)  # [vocab] fp32
        ids = torch.tensor([prefix], dtype=torch.long, device="cuda")
        with torch.no_grad():
            eager = ref(ids).logits[0, -1, :].float().cpu().numpy()
        # fp16 path vs fp16 eager: same dtype, so the only gap is kernel numerics.
        assert dep.shape == eager.shape
        np.testing.assert_allclose(dep, eager, rtol=2e-2, atol=2e-2)
        assert int(np.argmax(dep)) == int(np.argmax(eager))  # greedy token agrees


@pytest.mark.xfail(reason="M=1 demoted lm_head (slice_last_logits) not lowered on the cold path; tracked", strict=False)
def test_slice_last_logits_lowers_cold():
    """Tripwire for the in-graph last-token slice optimization: compiling the
    ``slice_last_logits=True`` whole-model graph cold (no prior) currently leaves the M=1
    demoted lm_head as an unlowered ``LoopOp`` (``CompiledProgram.build`` rejects it). When
    the cold lowering covers that shape this xfail flips to xpass — re-enable the slice in
    ``_CompiledLM``."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from emmy.compiler.trace.torch import trace_module

    model = _tiny_llama()
    s = 8
    wrapper = build_full_model_wrapper(model, s, torch.float16, dynamic=True, slice_last_logits=True)
    specs = parse_position_specs(["seq_len@input_ids:1", "seq_len@attention_mask:2", "seq_len@attention_mask:3", "seq_len@position_ids:1"])
    example = (torch.zeros((1, s), dtype=torch.long), build_causal_mask(s, torch.float16), torch.arange(s).unsqueeze(0))
    graph = trace_module(wrapper, example, dynamic_shapes=build_torch_dynamic_shapes(specs))
    compiled = CudaBackend(tune_db=None).compile(graph)
    with gpu_lock():
        CompiledProgram.build(compiled)  # raises TypeError on the leftover LoopOp today


def test_generate_loop_runs_end_to_end():
    """The full host loop over the compiled program produces a fixed-length output."""
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.commands.generate import _CompiledLM, generate
    from emmy.serving.sampling import Sampler

    lm = _CompiledLM.from_model(_tiny_llama(), seq_len=8)
    out = generate(lm.logits, [1, 2, 3], max_new_tokens=5, eos_ids=None, sampler=Sampler())
    assert len(out) == 5
    assert all(0 <= t < 64 for t in out)
