"""GPU dynamic-compile test for the Phase-1 attention-split wrappers (the compiler enabler).

``perf``-marked: needs CUDA + cupy. Traces the ``pre`` and ``post`` wrappers over the
flattened ``[num_tokens, H]`` layout with ``num_tokens`` **symbolic**, compiles each, and
runs at two different token counts — matching the eager wrapper output. This proves the
carved subgraphs actually lower + run dynamically (the core of Phase 2's gen_runner), and
that the ``post`` wrapper's two inputs share one ``Dim`` (without the second spec the
``residual`` axis would stay trace-sized). fp32 (carve correctness is dtype-independent;
the fp16 path is covered by the Phase-0 oracle).
"""

import numpy as np
import pytest

pytestmark = [pytest.mark.perf, pytest.mark.xdist_group("cuda")]


def _qwen3_block():
    import torch
    import transformers

    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )
    torch.manual_seed(0)
    return config, transformers.Qwen3ForCausalLM(config).eval().model.layers[0]


def _compile_wrapper(wrapper, example_args, argnames):
    """Trace ``wrapper`` with axis 0 of every arg bound to a shared ``num_tokens`` Dim,
    compile on the CUDA backend, bind fp32 constants. Returns (program, input_names, output_names)."""

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.loader.binder import bind_constants
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    specs = [f"num_tokens@{name}:0" for name in argnames]  # shared NAME ties all axes
    graph = trace_module(wrapper, tuple(example_args), dynamic_shapes=build_torch_dynamic_shapes(parse_position_specs(specs)))
    compiled = CudaBackend(tune_db="auto").compile(graph)

    sources = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)

    feed = {n: a.detach().cpu().numpy().astype(np.float32) for n, a in zip(compiled.inputs, example_args, strict=True)}
    with gpu_lock():
        program = CompiledProgram.build(compiled, {**const_feed, **feed})
    return program, list(compiled.inputs), list(compiled.outputs)


def _run(program, input_names, output_names, arrays):
    from emmy.compiler.backend.gpu_lock import gpu_lock

    t = arrays[0].shape[0]
    feed = {n: a.detach().cpu().numpy().astype(np.float32) for n, a in zip(input_names, arrays, strict=True)}
    with gpu_lock():
        program.rebind(feed)  # resolves num_tokens from the input shapes
        program.run_once()
        out = program.outputs({"num_tokens": t})
    return [out[n] for n in output_names]


def test_pre_wrapper_compiles_and_runs_dynamic():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    config, block = _qwen3_block()
    pre, _ = build_attention_split_wrapper(block)
    h = config.hidden_size

    program, in_names, out_names = _compile_wrapper(pre, [torch.randn(8, h)], ["hidden"])
    for t in (4, 7):  # different token counts → replay at new num_tokens, not recapture
        hidden = torch.randn(t, h)
        got = _run(program, in_names, out_names, [hidden])
        with torch.no_grad():
            want = pre(hidden)  # (q, k, v)
        assert len(got) == 3
        for g, w in zip(got, want, strict=True):
            np.testing.assert_allclose(g, w.numpy(), rtol=1e-3, atol=1e-3)


def test_post_wrapper_compiles_with_shared_dim():
    pytest.importorskip("cupy")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    config, block = _qwen3_block()
    _, post = build_attention_split_wrapper(block)
    h = config.hidden_size
    attn_width = config.num_attention_heads * (config.head_dim or h // config.num_attention_heads)

    # post(attn_out[T, Hq*D], residual[T, H]) — BOTH axis-0 share the num_tokens Dim.
    program, in_names, out_names = _compile_wrapper(post, [torch.randn(8, attn_width), torch.randn(8, h)], ["attn_out", "residual"])
    for t in (4, 7):
        attn_out, residual = torch.randn(t, attn_width), torch.randn(t, h)
        (got,) = _run(program, in_names, out_names, [attn_out, residual])
        with torch.no_grad():
            want = post(attn_out, residual)
        np.testing.assert_allclose(got, want.numpy(), rtol=1e-3, atol=1e-3)
