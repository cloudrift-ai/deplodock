"""Symbolic-extent ``Dim`` round-trips through trace → lift → ``LoopOp.forward``.

Tests cover the position-based dynamic
shape flow: ``torch.export(..., dynamic_shapes={input: {axis: Dim(name)}})``
threads ``Dim(name)`` through every downstream FX node via SymInt, the
compile pipeline emits one CudaOp whose signature carries
``int <name>`` runtime arg, and the launch resolves the value from the
runtime input shape.
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler import dtype as dt
from deplodock.compiler.dim import Dim
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.loop.ir import LoopOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import Pipeline

from ..conftest import from_pretrained_or_skip


def _seq_len_dim(*, min: int = 5, max: int = 4096):
    """``torch.export.Dim('seq_len')`` instance for tests below.

    ``min=5`` by default skirts torch.export's specialisation guards on
    small concrete dims that SDPA / matmul reshape paths introduce
    (e.g. ``num_heads=4`` would otherwise specialise the dim at
    ``seq_len=4``)."""
    import torch

    return torch.export.Dim("seq_len", min=min, max=max)


def _symbolic_elementwise_graph() -> Graph:
    """``y = exp(x)`` where ``x`` has shape ``(1, S, 2048)`` — S symbolic."""
    g = Graph()
    sym_shape = (Dim(1), Dim("seq_len"), Dim(2048))
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", sym_shape, dt.F32), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("y", sym_shape, dt.F32), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def _symbolic_reduce_graph() -> Graph:
    """``y = sum(x, axis=-1)`` where ``x`` has shape ``(1, S, 2048)`` — S symbolic, reduce axis static."""
    g = Graph()
    in_shape = (Dim(1), Dim("seq_len"), Dim(2048))
    out_shape = (Dim(1), Dim("seq_len"), Dim(1))  # keepdim
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", in_shape, dt.F32), node_id="x")
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("y", out_shape, dt.F32), node_id="y")
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_lift_elementwise_preserves_symbolic_free_axes():
    graph = _symbolic_elementwise_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_nodes = [n for n in lifted.nodes.values() if isinstance(n.op, LoopOp)]
    assert loop_nodes, "expected at least one LoopOp after lifting"
    extents = {ax.name: ax.extent for n in loop_nodes for ax in n.op.axes}
    assert Dim("seq_len") in extents.values(), f"symbolic seq_len axis lost: {extents}"


def test_lift_reduce_allows_symbolic_free_axis_static_reduce():
    graph = _symbolic_reduce_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_nodes = [n for n in lifted.nodes.values() if isinstance(n.op, LoopOp)]
    assert loop_nodes, "expected at least one LoopOp after lifting"
    extents = {ax.name: ax.extent for n in loop_nodes for ax in n.op.axes}
    assert Dim("seq_len") in extents.values(), f"free symbolic axis lost: {extents}"
    assert Dim(2048) in extents.values(), f"static reduce axis missing: {extents}"


def test_loop_forward_binds_symbolic_axis_from_input_shape():
    graph = _symbolic_elementwise_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_node = next(n for n in lifted.nodes.values() if isinstance(n.op, LoopOp))
    x = np.random.RandomState(0).standard_normal((1, 7, 2048)).astype(np.float32)
    out = loop_node.op.forward(x)
    assert out.shape == (1, 7, 2048)
    np.testing.assert_allclose(out, np.exp(x), rtol=1e-5, atol=1e-6)


def test_loop_forward_same_kernel_different_seq_lens():
    """Same symbolic LoopOp, two distinct runtime ``seq_len`` values both run cleanly."""
    graph = _symbolic_elementwise_graph()
    lifted = Pipeline.build(["loop/lifting"]).run(graph)
    loop_op = next(n.op for n in lifted.nodes.values() if isinstance(n.op, LoopOp))
    for s in (3, 11):
        x = np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)
        out = loop_op.forward(x)
        assert out.shape == (1, s, 2048)
        np.testing.assert_allclose(out, np.exp(x), rtol=1e-5, atol=1e-6)


def test_cuda_symbolic_elementwise_one_kernel_multiple_seq_lens():
    """M2 validation slice: same symbolic graph compiles to ONE CudaOp
    whose kernel signature carries ``int seq_len``; running it at two
    different ``seq_len`` values resolves the launch geometry from the
    actual input shape without recompiling."""
    pytest = __import__("pytest")
    cupy = pytest.importorskip("cupy")
    del cupy
    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.cuda import CudaOp

    graph = _symbolic_elementwise_graph()
    backend = CudaBackend()
    compiled = backend.compile(graph)
    cuda_ops = [n.op for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
    assert len(cuda_ops) == 1, f"expected one CudaOp, got {len(cuda_ops)}"
    op = cuda_ops[0]
    assert "seq_len" in op.runtime_args, f"runtime_args missing 'seq_len': {op.runtime_args}"
    assert "int seq_len" in op.kernel_source, f"kernel signature missing int seq_len param:\n{op.kernel_source}"

    cached_source = op.kernel_source
    for s in (5, 13):
        x = np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={"x": x})
        y = result.outputs["y"]
        assert y.shape == (1, s, 2048)
        np.testing.assert_allclose(y, np.exp(x), rtol=1e-5, atol=1e-5)
    # Sanity-check: nothing in the lowering path swapped the kernel source between runs.
    assert op.kernel_source == cached_source


def test_symbolic_sdpa_traces_and_decomposes():
    """A single-layer causal SDPA + reshape traced with
    ``dynamic_shapes={"q","k","v" → {2: Dim("seq_len")}}`` decomposes + lifts
    cleanly. Stops at the loop-lifted graph and asserts the surviving op
    shapes carry symbolic dims end-to-end."""
    import torch

    from deplodock.compiler.trace.torch import trace_module

    class AttnBlock(torch.nn.Module):
        def forward(self, q, k, v):
            attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
            B, H, S, D = attn.shape
            return attn.reshape(B, S, H * D)

    b, h, s, d = 1, 4, 16, 64
    seq_len = _seq_len_dim()
    graph = trace_module(
        AttnBlock(),
        (torch.randn(b, h, s, d), torch.randn(b, h, s, d), torch.randn(b, h, s, d)),
        dynamic_shapes={"q": {2: seq_len}, "k": {2: seq_len}, "v": {2: seq_len}},
    )
    decomp = Pipeline.build(["frontend/decomposition", "frontend/optimization"]).run(graph)
    assert any(Dim("seq_len") in n.output.shape for n in decomp.nodes.values()), (
        "expected at least one node to carry symbolic seq_len after decomposition"
    )
    lifted = Pipeline.build(["frontend/decomposition", "frontend/optimization", "loop/lifting"]).run(graph)
    loop_op_count = sum(1 for n in lifted.nodes.values() if isinstance(n.op, LoopOp))
    assert loop_op_count >= 1, "expected at least one LoopOp after lifting"


def test_cuda_softmax_over_symbolic_seq_len():
    """Softmax reducing over a symbolic ``seq_len`` axis compiles to a
    single kernel whose serial reduce loop's bound is the runtime
    ``int seq_len`` arg."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    class Softmax(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.softmax(x, dim=-1)

    graph = trace_module(Softmax(), (torch.randn(4, 16),), dynamic_shapes={"x": {1: _seq_len_dim()}})
    backend = CudaBackend()
    compiled = backend.compile(graph)
    for s in (5, 16, 32):
        x = np.random.RandomState(s).standard_normal((4, s)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={"x": x})
        y = next(iter(result.outputs.values()))
        ref = torch.nn.functional.softmax(torch.from_numpy(x), dim=-1).numpy()
        assert y.shape == (4, s)
        np.testing.assert_allclose(y, ref, rtol=1e-5, atol=1e-5)


def test_cuda_sdpa_over_symbolic_seq_len():
    """Full causal SDPA with symbolic seq_len compiles + runs
    end-to-end. Stresses symbolic on free axes (Q/K/V leading seq dim)
    AND on the matmul K axis (attn @ V)."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    class Attn(torch.nn.Module):
        def forward(self, q, k, v):
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

    b, h, s_trace, d = 1, 4, 16, 64
    seq_len = _seq_len_dim()
    graph = trace_module(
        Attn(),
        (torch.randn(b, h, s_trace, d), torch.randn(b, h, s_trace, d), torch.randn(b, h, s_trace, d)),
        dynamic_shapes={"q": {2: seq_len}, "k": {2: seq_len}, "v": {2: seq_len}},
    )
    backend = CudaBackend()
    compiled = backend.compile(graph)
    for s in (8, 16):
        q = np.random.RandomState(s).standard_normal((b, h, s, d)).astype(np.float32)
        k = np.random.RandomState(s + 1).standard_normal((b, h, s, d)).astype(np.float32)
        v = np.random.RandomState(s + 2).standard_normal((b, h, s, d)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={"q": q, "k": k, "v": v})
        y = next(iter(result.outputs.values()))
        with torch.no_grad():
            ref = torch.nn.functional.scaled_dot_product_attention(
                torch.from_numpy(q), torch.from_numpy(k), torch.from_numpy(v), is_causal=True
            ).numpy()
        assert y.shape == (b, h, s, d)
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)


def test_cuda_symbolic_rmsnorm_traced_and_run():
    """End-to-end on a real ``torch.nn.RMSNorm`` traced with
    ``dynamic_shapes={"x": {1: Dim("seq_len")}}`` — compile once, run at
    two distinct seq_len values, compare to torch eager."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.trace.torch import trace_module

    m = torch.nn.RMSNorm(2048)
    graph = trace_module(m, (torch.randn(1, 32, 2048),), dynamic_shapes={"x": {1: _seq_len_dim()}})
    backend = CudaBackend()
    compiled = backend.compile(graph)

    weight = m.weight.detach().numpy().astype(np.float32)
    for s in (8, 32):
        x = np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={"x": x, "p_weight": weight})
        y = next(iter(result.outputs.values()))
        with torch.no_grad():
            y_ref = torch.nn.functional.rms_norm(torch.from_numpy(x), (2048,), m.weight, eps=m.eps).numpy()
        assert y.shape == (1, s, 2048)
        np.testing.assert_allclose(y, y_ref, rtol=1e-4, atol=1e-4)


def test_reshape_negative_one_infers_through_symbolic_dim():
    """``ReshapeOp(shape=(..., -1))`` flowing through a symbolic input
    dim infers the ``-1`` by cancelling the matching symbolic factor on
    each side — the typical ``x.reshape(B, S, H, -1)`` pattern."""
    import torch

    from deplodock.compiler.pipeline import Pipeline
    from deplodock.compiler.trace.torch import trace_module

    class ReshapeWithMinusOne(torch.nn.Module):
        def forward(self, x):
            b, s, d = x.shape
            return x.reshape(b, s, 4, -1)

    graph = trace_module(ReshapeWithMinusOne(), (torch.randn(1, 16, 128),), dynamic_shapes={"x": {1: _seq_len_dim()}})
    out = Pipeline.build(["frontend/decomposition", "frontend/optimization", "loop/lifting"]).run(graph)
    loops = [n for n in out.nodes.values() if isinstance(n.op, LoopOp)]
    assert loops, "expected at least one LoopOp after lifting"
    # The reshape's inferred -1 is the static 32 (128 / 4); seq_len threads through unchanged.
    extents = {(ax.name, ax.extent) for n in loops for ax in n.op.axes}
    assert any(e == Dim("seq_len") for _, e in extents), f"seq_len axis dropped: {extents}"
    assert any(e == Dim(32) for _, e in extents), f"reshape -1 didn't infer to static 32: {extents}"


def test_cuda_symbolic_linear_traced_and_run():
    """End-to-end on a real ``torch.nn.Linear`` traced with
    ``dynamic_shapes={"x": {1: Dim("seq_len")}}`` — covers the
    matmul-on-symbolic-M code path."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.ir.base import ConstantOp
    from deplodock.compiler.loader.binder import apply_load_ops
    from deplodock.compiler.trace.torch import trace_module_with_constants

    m = torch.nn.Linear(128, 256, bias=False)
    # ``Linear.forward`` declares the activation arg as ``input`` — torch.export
    # keys ``dynamic_shapes`` by the real forward-signature name.
    graph, _ = trace_module_with_constants(m, (torch.randn(1, 16, 128),), dynamic_shapes={"input": {1: _seq_len_dim()}})
    backend = CudaBackend()
    compiled = backend.compile(graph)

    # Constants take their post-fold shape — bind via ``apply_load_ops``
    # against the original parameter so any tracer-side transpose flows in.
    const_feed: dict[str, np.ndarray] = {}
    for nid, node in compiled.nodes.items():
        if not isinstance(node.op, ConstantOp):
            continue
        numel = 1
        for d in node.output.shape:
            numel *= d.as_static()
        for key, p in m.named_parameters():
            safe = "p_" + key.replace(".", "_")
            if safe.endswith(node.op.name[2:]) and p.numel() == numel:
                const_feed[nid] = apply_load_ops(p.detach().numpy().astype(np.float32), node.op.load_ops)
                break

    in_name = graph.inputs[0]
    for s in (4, 16, 32):
        x = np.random.RandomState(s).standard_normal((1, s, 128)).astype(np.float32)
        result, _ = backend.run(compiled, input_data={in_name: x, **const_feed})
        y = next(iter(result.outputs.values()))
        with torch.no_grad():
            ref = m(torch.from_numpy(x)).numpy()
        assert y.shape == (1, s, 256)
        np.testing.assert_allclose(y, ref, rtol=1e-4, atol=1e-4)


def test_qwen_whole_model_dynamic_compiles_and_matches_eager():
    """1-layer random-weight Qwen3 trunk, dynamic seq_len: compile once, run at
    the trace size AND two other seq_lens (via ``CompiledProgram.rebind``),
    compare against torch eager — with NON-ZERO token ids.

    Non-zero ids are the load-bearing part: with ``input_ids = zeros`` every
    value row is identical, so the attention output is independent of the
    attention weights — an accuracy check at zero ids is blind to a degenerate
    RoPE (the in-graph rotary used to constant-fold to ``cos=1, sin=0`` under
    ``torch.export``; the wrapper now precomputes + slices instead) and to
    wrong attention scores."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch
    from transformers import AutoConfig, AutoModel

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.loader.binder import bind_constants
    from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(0)
    config = from_pretrained_or_skip(AutoConfig.from_pretrained, "Qwen/Qwen3-Embedding-0.6B")
    config.num_hidden_layers = 1
    model = AutoModel.from_config(config).float().eval()

    hint, dtype = 32, torch.float32
    wrapper = build_full_model_wrapper(model, hint, dtype, dynamic=True)
    seq_dim = _seq_len_dim(min=1)
    graph = trace_module(
        wrapper,
        (torch.zeros((1, hint), dtype=torch.long), build_causal_mask(hint, dtype), torch.arange(hint).unsqueeze(0)),
        dynamic_shapes={"input_ids": {1: seq_dim}, "attention_mask": {2: seq_dim, 3: seq_dim}, "position_ids": {1: seq_dim}},
    )
    compiled = CudaBackend().compile(graph)

    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)
    ids_name, mask_name, pos_name = compiled.inputs
    out_name = compiled.outputs[0]

    def feed(s: int) -> dict[str, np.ndarray]:
        ids = (np.arange(s, dtype=np.int64).reshape(1, s) * 97) % config.vocab_size
        return {
            ids_name: ids,
            mask_name: build_causal_mask(s, dtype).numpy(),
            pos_name: np.arange(s, dtype=np.int64).reshape(1, s),
        }

    with gpu_lock():
        prog = None
        for s in (hint, 17, 64):
            fd = feed(s)
            if prog is None:
                prog = CompiledProgram.build(compiled, {**const_feed, **fd})
            else:
                prog.rebind(fd)
            prog.run_once()
            out = prog.outputs()[out_name]
            with torch.no_grad():
                ref = wrapper(torch.from_numpy(fd[ids_name]), build_causal_mask(s, dtype), torch.arange(s).unsqueeze(0)).numpy()
            assert out.shape == ref.shape
            np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_qwen_layer_dynamic_compiles_and_matches_eager():
    """Single decoder layer (random-weight Qwen3 trunk) traced through
    ``build_layer_wrapper`` with dynamic seq_len — the per-layer
    ``--dynamic seq_len@x:1`` CLI path in test form. Compile once, run at the
    trace size AND two other seq_lens (via ``CompiledProgram.rebind``),
    compare against torch eager with non-trivial activations.

    The wrapper is load-bearing: tracing the bare block with concrete
    ``(cos, sin)`` kwargs specialises rotary to the trace seq_len, so the
    in-graph sliced-rotary buffers are what make per-layer dynamic work."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch
    from transformers import AutoConfig, AutoModel

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.loader.binder import bind_constants
    from deplodock.compiler.trace.huggingface import build_layer_wrapper
    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(0)
    config = from_pretrained_or_skip(AutoConfig.from_pretrained, "Qwen/Qwen3-Embedding-0.6B")
    config.num_hidden_layers = 1
    model = AutoModel.from_config(config).float().eval()

    hint, dtype = 32, torch.float32
    wrapper = build_layer_wrapper(model.layers[0], model.rotary_emb, config.hidden_size, dtype)
    graph = trace_module(wrapper, (torch.randn(1, hint, config.hidden_size, dtype=dtype),), dynamic_shapes={"x": {1: _seq_len_dim()}})
    compiled = CudaBackend().compile(graph)

    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)
    in_name = compiled.inputs[0]
    out_name = compiled.outputs[0]

    with gpu_lock():
        prog = None
        for s in (hint, 17, 64):
            x = np.random.RandomState(s).standard_normal((1, s, config.hidden_size)).astype(np.float32)
            fd = {in_name: x}
            if prog is None:
                prog = CompiledProgram.build(compiled, {**const_feed, **fd})
            else:
                prog.rebind(fd)
            prog.run_once()
            out = prog.outputs()[out_name]
            with torch.no_grad():
                ref = wrapper(torch.from_numpy(x))
                ref = (ref[0] if isinstance(ref, tuple) else ref).numpy()
            assert out.shape == ref.shape
            np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)


def test_qwen_whole_model_dynamic_traces():
    """End-to-end whole-model dynamic trace on Qwen3-Embedding-0.6B (1 layer,
    random weights so no checkpoint download). Exercises the CLI's
    canonical whole-model invocation in test form: wrapper switches to
    ``dynamic=True``, every seq_len-carrying axis (``input_ids:1``,
    ``attention_mask:2``, ``attention_mask:3``, ``position_ids:1``) is
    marked with the SAME ``Dim('seq_len')`` instance, and the trace
    threads the symbolic dim through every downstream FX tensor.

    Asserts the trace yields a graph whose internal tensors carry
    ``Dim('seq_len')`` — full CUDA lowering of the whole model is the
    next-stretch (int64 index-math kernels for embedding lookup are
    still uncovered)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(0)
    config = from_pretrained_or_skip(AutoConfig.from_pretrained, "Qwen/Qwen3-Embedding-0.6B")
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_config(config).float().eval()

    seq_len_int = 32
    dtype = torch.float32
    wrapper = build_full_model_wrapper(model, seq_len_int, dtype, dynamic=True)
    input_ids = torch.zeros((1, seq_len_int), dtype=torch.long)
    attention_mask = build_causal_mask(seq_len_int, dtype)
    position_ids = torch.arange(seq_len_int).unsqueeze(0)

    seq_dim = _seq_len_dim()
    dynamic_shapes = {
        "input_ids": {1: seq_dim},
        "attention_mask": {2: seq_dim, 3: seq_dim},
        "position_ids": {1: seq_dim},
    }
    graph = trace_module(wrapper, (input_ids, attention_mask, position_ids), dynamic_shapes=dynamic_shapes)

    symbolic_nodes = [nid for nid, n in graph.nodes.items() if Dim("seq_len") in n.output.shape]
    assert len(symbolic_nodes) > 10, (
        f"expected the seq_len symbol to propagate to most downstream tensors, got only {len(symbolic_nodes)} of {len(graph.nodes)}"
    )
    # The three runtime inputs all carry the symbolic dim at their declared positions.
    inputs_by_name = {nid: graph.nodes[nid] for nid in graph.inputs}
    for name in ("input_ids", "attention_mask", "position_ids"):
        node = inputs_by_name.get(name)
        assert node is not None, f"missing graph input {name!r}; got {list(inputs_by_name)}"
        assert Dim("seq_len") in node.output.shape, f"{name} shape {node.output.shape} missing Dim('seq_len')"


# ---------------------------------------------------------------------------
# Captured-CUDA-graph serving path: one captured whole-program graph per
# seq_len over a shared capacity-sized buffer set (CompiledProgram
# set_sym_values + upload_prefix + capture_program_graph[cached] +
# replay_program_graph + outputs(sym_values)). Each graph is captured at its
# EXACT seq_len, so every kernel runs at its exact grid — no oversized-grid
# guard story needed.
# ---------------------------------------------------------------------------


def test_capture_replay_cache_rmsnorm_over_capacity_buffers():
    """RMSNorm built once at capacity 64; serve S ∈ {5,12,33,64,12} through the
    per-seq_len graph cache — capture lazily, replay at each S, slice the output
    to the real shape, match torch eager. Repeats hit the cache (no re-capture)."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.trace.torch import trace_module

    cap = 64
    m = torch.nn.RMSNorm(2048)
    compiled = CudaBackend().compile(trace_module(m, (torch.randn(1, 32, 2048),), dynamic_shapes={"x": {1: _seq_len_dim()}}))
    weight = m.weight.detach().numpy().astype(np.float32)
    out_name = compiled.outputs[0]

    def x_at(s: int) -> np.ndarray:
        return np.random.RandomState(s).standard_normal((1, s, 2048)).astype(np.float32)

    with gpu_lock():
        prog = CompiledProgram.build(compiled, {"x": x_at(cap), "p_weight": weight})
        for s in (5, 12, 33, 64, 12):
            x = x_at(s)
            prog.set_sym_values({"seq_len": s})
            prog.upload_prefix({"x": x})
            prog.capture_program_graph()
            prog.replay_program_graph()
            out = prog.outputs({"seq_len": s})[out_name]
            with torch.no_grad():
                ref = torch.nn.functional.rms_norm(torch.from_numpy(x), (2048,), m.weight, eps=m.eps).numpy()
            assert out.shape == (1, s, 2048)
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)
        assert set(k[0][1] for k in prog._graph_cache) == {5, 12, 33, 64}, "expected one cached graph per distinct seq_len"


def test_capture_replay_device_io_matches_eager():
    """Serving zero-copy device I/O: feed cupy inputs through ``upload_prefix_device``
    and read the output buffer's prefix back as a torch tensor via ``output_prefix_device``
    + ``torch.from_dlpack`` — NO host round-trip — and confirm it matches torch eager
    across seq_lens. The dlpack bridge (cupy ↔ torch) is what lets the runner accept
    torch tensors straight from vLLM. Repeats hit the per-S graph cache."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import cupy as cp
    import torch

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.trace.torch import trace_module

    cap = 48
    m = torch.nn.RMSNorm(1024)
    compiled = CudaBackend().compile(trace_module(m, (torch.randn(1, 32, 1024),), dynamic_shapes={"x": {1: _seq_len_dim()}}))
    weight = m.weight.detach().numpy().astype(np.float32)
    out_name = compiled.outputs[0]

    with gpu_lock():
        prog = CompiledProgram.build(compiled, {"x": np.zeros((1, cap, 1024), np.float32), "p_weight": weight})
        for s in (7, 32, 48, 7):
            x = np.random.RandomState(s).standard_normal((1, s, 1024)).astype(np.float32)
            prog.set_sym_values({"seq_len": s})
            prog.upload_prefix_device({"x": cp.asarray(x)})  # cupy in — no host upload
            prog.capture_program_graph()
            prog.replay_program_graph()
            out_view = prog.output_prefix_device({"seq_len": s})[out_name]  # cupy view, no .get()
            cp.cuda.runtime.deviceSynchronize()
            out = torch.from_dlpack(out_view).clone().cpu().numpy()  # torch view of cupy mem
            with torch.no_grad():
                ref = torch.nn.functional.rms_norm(torch.from_numpy(x), (1024,), m.weight, eps=m.eps).numpy()
            assert out.shape == (1, s, 1024)
            np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_qwen_whole_model_capture_replay_cache_matches_eager():
    """1-layer random-weight Qwen3 trunk through the captured-graph serving path:
    build once at capacity, serve several seq_lens via the per-S graph cache
    (capture at exact S, replay, slice), compare against torch eager with NON-ZERO
    ids. End-to-end gate for the attention / mask / shared-capacity-buffer story.
    Run under compute-sanitizer in dev to confirm zero illegal accesses."""
    pytest = __import__("pytest")
    pytest.importorskip("cupy")
    import torch
    from transformers import AutoConfig, AutoModel

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.backend.cuda.program import CompiledProgram
    from deplodock.compiler.backend.gpu_lock import gpu_lock
    from deplodock.compiler.loader.binder import bind_constants
    from deplodock.compiler.trace.huggingface import build_causal_mask, build_full_model_wrapper
    from deplodock.compiler.trace.torch import trace_module

    torch.manual_seed(0)
    config = from_pretrained_or_skip(AutoConfig.from_pretrained, "Qwen/Qwen3-Embedding-0.6B")
    config.num_hidden_layers = 1
    model = AutoModel.from_config(config).float().eval()

    hint, cap, dtype = 32, 64, torch.float32
    wrapper = build_full_model_wrapper(model, hint, dtype, dynamic=True)
    seq_dim = _seq_len_dim(min=1)
    graph = trace_module(
        wrapper,
        (torch.zeros((1, hint), dtype=torch.long), build_causal_mask(hint, dtype), torch.arange(hint).unsqueeze(0)),
        dynamic_shapes={"input_ids": {1: seq_dim}, "attention_mask": {2: seq_dim, 3: seq_dim}, "position_ids": {1: seq_dim}},
    )
    compiled = CudaBackend().compile(graph)

    sources: dict[str, np.ndarray] = {}
    for path, t in wrapper.named_parameters(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    for path, t in wrapper.named_buffers(remove_duplicate=False):
        sources[path] = t.detach().cpu().numpy().astype(np.float32, copy=False)
    const_feed = bind_constants(compiled, sources)
    ids_name, mask_name, pos_name = compiled.inputs
    out_name = compiled.outputs[0]

    def feed(s: int) -> dict[str, np.ndarray]:
        ids = (np.arange(s, dtype=np.int64).reshape(1, s) * 97) % config.vocab_size
        return {
            ids_name: ids,
            mask_name: build_causal_mask(s, dtype).numpy(),
            pos_name: np.arange(s, dtype=np.int64).reshape(1, s),
        }

    with gpu_lock():
        prog = CompiledProgram.build(compiled, {**const_feed, **feed(cap)})
        for s in (5, 17, 64, 17):
            fd = feed(s)
            prog.set_sym_values({"seq_len": s})
            prog.upload_prefix(fd)
            prog.capture_program_graph()
            prog.replay_program_graph()
            out = prog.outputs({"seq_len": s})[out_name]
            with torch.no_grad():
                ref = wrapper(torch.from_numpy(fd[ids_name]), build_causal_mask(s, dtype), torch.arange(s).unsqueeze(0)).numpy()
            assert out.shape == ref.shape
            np.testing.assert_allclose(out, ref, rtol=1e-3, atol=1e-3)
