"""Tests for decomposition rules: structural checks and numerical correctness.

Every rule is tested both structurally (ops present after rewrite) and for
correctness (numpy backend output before rewrite == after rewrite).
"""

from pathlib import Path

import numpy as np

from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.frontend_ir import (
    CatOp,
    LinearOp,
    MatmulOp,
    MeanOp,
    ReshapeOp,
    SdpaOp,
    SliceOp,
    TransposeOp,
    UnsqueezeOp,
)
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor_ir import ElementwiseOp, ReduceOp
from deplodock.compiler.rewriter import Pass, Rule

RULES_DIR = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "passes" / "decomposition"

rng = np.random.default_rng(42)
_backend = NumpyBackend()


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs).outputs


def _assert_close(before: dict, after: dict, *, rtol=1e-5, atol=1e-6):
    bvals = list(before.values())
    avals = list(after.values())
    assert len(bvals) == len(avals), f"output count mismatch: {len(bvals)} vs {len(avals)}"
    for i, (b, a) in enumerate(zip(bvals, avals, strict=True)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output[{i}]")


def _load(name: str) -> Rule:
    return Rule.from_file(RULES_DIR / name)


def _apply(graph: Graph, rule: Rule) -> Graph:
    return Pass(name="decomp", rules=[rule]).apply(graph)


# ===================================================================
# Pow decomposition: pow(x, 2) → mul(x, x)
# ===================================================================


def _make_pow_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 128)), node_id="x")
    g.add_node(op=ConstantOp(name="exp", value=2.0), inputs=[], output=Tensor("exp", (1,)), node_id="exp")
    g.add_node(op=ElementwiseOp(fn="pow"), inputs=["x", "exp"], output=Tensor("out", (4, 128)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_pow_decomposes_to_self_mul():
    result = _apply(_make_pow_graph(), _load("003_decompose_pow.py"))
    fns = [n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)]
    assert "pow" not in fns
    assert "mul" in fns


def test_pow_output_uses_self_multiply():
    result = _apply(_make_pow_graph(), _load("003_decompose_pow.py"))
    out_node = result.nodes[result.outputs[0]]
    assert out_node.op.fn == "mul"
    assert out_node.inputs[0] == out_node.inputs[1]


def test_pow_preserves_shape_and_dtype():
    g = _make_pow_graph()
    orig = g.nodes[g.outputs[0]].output
    result = _apply(g, _load("003_decompose_pow.py"))
    assert result.nodes[result.outputs[0]].output.shape == orig.shape
    assert result.nodes[result.outputs[0]].output.dtype == orig.dtype


def test_pow_idempotent():
    p = Pass(name="decomp", rules=[_load("003_decompose_pow.py")])
    once = p.apply(_make_pow_graph())
    twice = p.apply(once)
    assert len(twice.nodes) == len(once.nodes)


def test_pow_correctness():
    g = _make_pow_graph()
    x = rng.standard_normal((4, 128)).astype(np.float32)
    inputs = {"x": x}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("003_decompose_pow.py")), inputs)
    _assert_close(before, after)


# ===================================================================
# SiLU decomposition: silu(x) → x * recip(1 + exp(-x))
# ===================================================================


def _make_silu_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 32)), node_id="x")
    g.add_node(op=ElementwiseOp(fn="silu"), inputs=["x"], output=Tensor("out", (4, 32)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_silu_decomposes():
    result = _apply(_make_silu_graph(), _load("002_decompose_silu.py"))
    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "silu" not in fns
    assert "exp" in fns


def test_silu_correctness():
    g = _make_silu_graph()
    x = rng.standard_normal((4, 32)).astype(np.float32)
    inputs = {"x": x}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("002_decompose_silu.py")), inputs)
    _assert_close(before, after)


# ===================================================================
# Mean decomposition: mean(x, axis) → sum(x, axis) / dim_size
# ===================================================================


def _make_mean_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=MeanOp(axis=-1), inputs=["x"], output=Tensor("out", (4,)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_mean_decomposes():
    result = _apply(_make_mean_graph(), _load("007_decompose_mean.py"))
    has_mean = any(isinstance(n.op, MeanOp) for n in result.nodes.values())
    assert not has_mean
    has_sum = any(isinstance(n.op, ReduceOp) and n.op.fn == "sum" for n in result.nodes.values())
    has_div = any(isinstance(n.op, ElementwiseOp) and n.op.fn == "div" for n in result.nodes.values())
    assert has_sum and has_div


def test_mean_correctness():
    g = _make_mean_graph()
    x = rng.standard_normal((4, 8)).astype(np.float32)
    inputs = {"x": x}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("007_decompose_mean.py")), inputs)
    _assert_close(before, after)


# ===================================================================
# RMSNorm decomposition: rms_norm(x, w) → x * rsqrt(mean(x^2) + eps) * w
# ===================================================================


def _make_rms_norm_graph(dim=64, seq_len=8, eps_input=False):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, seq_len, dim)), node_id="x")
    g.add_node(op=ConstantOp(name="w"), inputs=[], output=Tensor("w", (dim,)), node_id="w")
    ins = ["x", "w"]
    if eps_input:
        g.add_node(op=ConstantOp(name="eps", value=1e-5), inputs=[], output=Tensor("eps", (1,)), node_id="eps")
        ins.append("eps")
    g.add_node(op=ElementwiseOp(fn="rms_norm"), inputs=ins, output=Tensor("out", (1, seq_len, dim)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_rms_norm_decomposes():
    result = _apply(_make_rms_norm_graph(), _load("006_decompose_rms_norm.py"))
    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "rms_norm" not in fns
    assert {"mul", "add", "rsqrt"} <= fns
    assert any(isinstance(n.op, MeanOp) for n in result.nodes.values())


def test_rms_norm_with_eps_input_decomposes():
    """Three-input form (x, w, eps) from explicit torch.nn.RMSNorm(dim, eps=...)."""
    result = _apply(_make_rms_norm_graph(eps_input=True), _load("006_decompose_rms_norm.py"))
    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "rms_norm" not in fns


def test_rms_norm_preserves_io_shape():
    g = _make_rms_norm_graph(dim=64, seq_len=8)
    result = _apply(g, _load("006_decompose_rms_norm.py"))
    assert result.nodes[result.outputs[0]].output.shape == (1, 8, 64)


def test_rms_norm_trace_to_tensor_ir_primitives_only():
    """End-to-end protection: torch.nn.RMSNorm → trace → compile through decomposition
    must yield a graph of primitives only (no rms_norm fused op).
    """
    import torch

    from deplodock.compiler.rewriter import Rewriter
    from deplodock.compiler.torch_trace import trace_module

    rules_dir = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "passes"
    graph = trace_module(torch.nn.RMSNorm(64), (torch.randn(1, 8, 64),))
    pre = Rewriter.from_directory(rules_dir, pass_order=["decomposition", "optimization"])
    decomposed = pre.apply(graph)
    for n in decomposed.nodes.values():
        if isinstance(n.op, ElementwiseOp):
            assert n.op.fn != "rms_norm", f"rms_norm survived decomposition at {n.output.name}"


# ===================================================================
# Softmax decomposition: softmax(x, dim) → exp(x-max) / sum(exp(x-max))
# ===================================================================


def _make_softmax_graph(dim_extent=8, seq_len=4, axis=-1):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (1, seq_len, dim_extent)), node_id="x")
    g.add_node(op=ConstantOp(name="axis", value=float(axis)), inputs=[], output=Tensor("axis", (1,)), node_id="axis")
    g.add_node(
        op=ElementwiseOp(fn="softmax"),
        inputs=["x", "axis"],
        output=Tensor("out", (1, seq_len, dim_extent)),
        node_id="out",
    )
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_softmax_decomposes():
    result = _apply(_make_softmax_graph(), _load("008_decompose_softmax.py"))
    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    assert "softmax" not in fns
    assert {"sub", "exp", "div"} <= fns
    reduce_fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ReduceOp)}
    assert {"max", "sum"} <= reduce_fns


def test_softmax_preserves_io_shape():
    g = _make_softmax_graph(dim_extent=8, seq_len=4)
    result = _apply(g, _load("008_decompose_softmax.py"))
    assert result.nodes[result.outputs[0]].output.shape == (1, 4, 8)


def test_softmax_correctness():
    g = _make_softmax_graph(dim_extent=8, seq_len=4)
    # Numpy backend doesn't know how to execute the fused softmax op, so we
    # compare the decomposed graph against a numpy reference directly.
    x = rng.standard_normal((1, 4, 8)).astype(np.float32)
    shifted = x - x.max(axis=-1, keepdims=True)
    expected = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    after = _run(_apply(g, _load("008_decompose_softmax.py")), {"x": x})
    np.testing.assert_allclose(list(after.values())[0], expected, rtol=1e-5, atol=1e-6)


def test_softmax_trace_to_tensor_ir_primitives_only():
    """End-to-end: torch.nn.Softmax → trace → decomposition yields primitives only."""
    import torch

    from deplodock.compiler.rewriter import Rewriter
    from deplodock.compiler.torch_trace import trace_module

    rules_dir = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "passes"
    graph = trace_module(torch.nn.Softmax(dim=-1), (torch.randn(1, 4, 8),))
    pre = Rewriter.from_directory(rules_dir, pass_order=["decomposition", "optimization"])
    decomposed = pre.apply(graph)
    for n in decomposed.nodes.values():
        if isinstance(n.op, ElementwiseOp):
            assert n.op.fn != "softmax", f"softmax survived decomposition at {n.output.name}"


# ===================================================================
# SDPA decomposition: sdpa(Q, K, V) → QK^T → scale → softmax → @V
# ===================================================================


def _make_sdpa_graph(seq_len=4, head_dim=8, num_heads=1, is_causal=False):
    g = Graph()
    s = (num_heads, seq_len, head_dim)
    g.add_node(op=InputOp(), inputs=[], output=Tensor("Q", s), node_id="Q")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("K", s), node_id="K")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("V", s), node_id="V")
    g.add_node(op=SdpaOp(is_causal=is_causal), inputs=["Q", "K", "V"], output=Tensor("out", s), node_id="sdpa_out")
    g.inputs, g.outputs = ["Q", "K", "V"], ["sdpa_out"]
    return g


def test_sdpa_decomposes():
    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    assert not any(isinstance(n.op, SdpaOp) for n in result.nodes.values())


def test_sdpa_produces_transpose():
    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    transposes = [n for n in result.nodes.values() if isinstance(n.op, TransposeOp)]
    assert len(transposes) >= 1
    assert transposes[0].op.axes == (-2, -1)


def test_sdpa_produces_two_matmuls():
    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    muls = [n for n in result.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.fn == "mul"]
    sums = [n for n in result.nodes.values() if isinstance(n.op, ReduceOp) and n.op.fn == "sum"]
    assert len(muls) >= 3
    assert len(sums) >= 2


def test_sdpa_produces_softmax_pattern():
    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ElementwiseOp)}
    reduce_fns = {n.op.fn for n in result.nodes.values() if isinstance(n.op, ReduceOp)}
    assert {"sub", "exp", "div"} <= fns
    assert "max" in reduce_fns


def test_sdpa_produces_scale_constant():
    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    scale_constants = [n for n in result.nodes.values() if isinstance(n.op, ConstantOp) and "scale" in n.op.name]
    assert len(scale_constants) >= 1


def test_sdpa_preserves_io_count():
    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    assert len(result.inputs) == 3
    assert len(result.outputs) == 1


def test_sdpa_idempotent():
    p = Pass(name="decomp", rules=[_load("001_decompose_sdpa.py")])
    once = p.apply(_make_sdpa_graph())
    twice = p.apply(once)
    assert len(twice.nodes) == len(once.nodes)


def test_sdpa_output_is_valid():
    """SDPA ends with a squeeze IndexMapOp (after the keepdim reduce)."""
    from deplodock.compiler.ir.tensor_ir import IndexMapOp

    result = _apply(_make_sdpa_graph(), _load("001_decompose_sdpa.py"))
    assert isinstance(result.nodes[result.outputs[0]].op, (ElementwiseOp, ReduceOp, IndexMapOp))


def test_sdpa_with_extra_args_decomposes():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("Q", (1, 4, 8)), node_id="Q")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("K", (1, 4, 8)), node_id="K")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("V", (1, 4, 8)), node_id="V")
    g.add_node(op=ConstantOp(name="dropout_p"), inputs=[], output=Tensor("dp", (1,)), node_id="dp")
    g.add_node(op=SdpaOp(), inputs=["Q", "K", "V", "dp"], output=Tensor("out", (1, 4, 8)), node_id="sdpa_out")
    g.inputs, g.outputs = ["Q", "K", "V"], ["sdpa_out"]
    result = _apply(g, _load("001_decompose_sdpa.py"))
    assert not any(isinstance(n.op, SdpaOp) for n in result.nodes.values())


def test_sdpa_correctness():
    g = _make_sdpa_graph(seq_len=4, head_dim=8, num_heads=1)
    q = rng.standard_normal((1, 4, 8)).astype(np.float32)
    k = rng.standard_normal((1, 4, 8)).astype(np.float32)
    v = rng.standard_normal((1, 4, 8)).astype(np.float32)
    inputs = {"Q": q, "K": k, "V": v}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("001_decompose_sdpa.py")), inputs)
    _assert_close(before, after, rtol=1e-4, atol=1e-5)


def test_sdpa_causal_decomposes():
    """Causal SDPA decomposes into primitives including a causal mask IndexMapOp."""
    g = _make_sdpa_graph(seq_len=4, head_dim=8, num_heads=1, is_causal=True)
    result = _apply(g, _load("001_decompose_sdpa.py"))
    assert not any(isinstance(n.op, SdpaOp) for n in result.nodes.values())


def test_sdpa_causal_correctness():
    """Causal SDPA decomposition matches the original causal SdpaOp numerically."""
    g = _make_sdpa_graph(seq_len=8, head_dim=16, num_heads=2, is_causal=True)
    q = rng.standard_normal((2, 8, 16)).astype(np.float32)
    k = rng.standard_normal((2, 8, 16)).astype(np.float32)
    v = rng.standard_normal((2, 8, 16)).astype(np.float32)
    inputs = {"Q": q, "K": k, "V": v}
    before = _run(g, inputs)
    after = _run(_apply(g, _load("001_decompose_sdpa.py")), inputs)
    _assert_close(before, after, rtol=1e-4, atol=1e-5)


# ===================================================================
# Linear decomposition: linear(x, W, b) → x @ W.T [+ b]
# ===================================================================


def _make_linear_graph(has_bias=False):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (2, 8)), node_id="x")
    g.add_node(op=ConstantOp(name="w"), inputs=[], output=Tensor("w", (4, 8)), node_id="w")
    ins = ["x", "w"]
    g.inputs = ["x"]
    if has_bias:
        g.add_node(op=ConstantOp(name="b"), inputs=[], output=Tensor("b", (4,)), node_id="b")
        ins.append("b")
    g.add_node(op=LinearOp(has_bias=has_bias), inputs=ins, output=Tensor("out", (2, 4)), node_id="out")
    g.outputs = ["out"]
    return g


def test_linear_decomposes():
    result = _apply(_make_linear_graph(), _load("004_decompose_linear.py"))
    assert not any(isinstance(n.op, LinearOp) for n in result.nodes.values())


def test_linear_correctness():
    g = _make_linear_graph()
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    before = _run(g, {"x": x, "w": w})
    after = _run(_apply(g, _load("004_decompose_linear.py")), {"x": x, "w": w})
    _assert_close(before, after, rtol=1e-4, atol=1e-5)


def test_linear_with_bias_correctness():
    g = _make_linear_graph(has_bias=True)
    x = rng.standard_normal((2, 8)).astype(np.float32)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    before = _run(g, {"x": x, "w": w, "b": b})
    after = _run(_apply(g, _load("004_decompose_linear.py")), {"x": x, "w": w, "b": b})
    _assert_close(before, after, rtol=1e-4, atol=1e-5)


# ===================================================================
# Matmul decomposition: A @ B [+ bias]
# ===================================================================


def _make_matmul_graph(has_bias=False):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 3)), node_id="b")
    ins = ["a", "b"]
    g.inputs = ["a", "b"]
    if has_bias:
        g.add_node(op=InputOp(), inputs=[], output=Tensor("bias", (3,)), node_id="bias")
        ins.append("bias")
        g.inputs.append("bias")
    g.add_node(op=MatmulOp(has_bias=has_bias), inputs=ins, output=Tensor("out", (4, 3)), node_id="out")
    g.outputs = ["out"]
    return g


def test_matmul_decomposes():
    result = _apply(_make_matmul_graph(), _load("005_decompose_matmul.py"))
    assert not any(isinstance(n.op, MatmulOp) for n in result.nodes.values())


def test_matmul_correctness():
    g = _make_matmul_graph()
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((8, 3)).astype(np.float32)
    before = _run(g, {"a": a, "b": b})
    after = _run(_apply(g, _load("005_decompose_matmul.py")), {"a": a, "b": b})
    _assert_close(before, after, rtol=1e-4, atol=1e-5)


def test_matmul_with_bias_correctness():
    g = _make_matmul_graph(has_bias=True)
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((8, 3)).astype(np.float32)
    bias = rng.standard_normal(3).astype(np.float32)
    before = _run(g, {"a": a, "b": b, "bias": bias})
    after = _run(_apply(g, _load("005_decompose_matmul.py")), {"a": a, "b": b, "bias": bias})
    _assert_close(before, after, rtol=1e-4, atol=1e-5)


# ===================================================================
# Unsqueeze → IndexMapOp
# ===================================================================


def _make_unsqueeze_graph(dim=0):
    g = Graph()
    in_shape = (4, 8)
    out_shape = list(in_shape)
    d = dim if dim >= 0 else len(in_shape) + 1 + dim
    out_shape.insert(d, 1)
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", in_shape), node_id="x")
    g.add_node(op=UnsqueezeOp(dim=dim), inputs=["x"], output=Tensor("out", tuple(out_shape)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_unsqueeze_to_indexmap_correctness():
    g = _make_unsqueeze_graph(dim=0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    before = _run(g, {"x": x})
    after = _run(_apply(g, _load("010_unsqueeze_to_indexmap.py")), {"x": x})
    _assert_close(before, after)


def test_unsqueeze_last_dim_correctness():
    g = _make_unsqueeze_graph(dim=-1)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    before = _run(g, {"x": x})
    after = _run(_apply(g, _load("010_unsqueeze_to_indexmap.py")), {"x": x})
    _assert_close(before, after)


# ===================================================================
# Transpose → IndexMapOp
# ===================================================================


def _make_transpose_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (3, 4)), node_id="x")
    g.add_node(op=TransposeOp(axes=(0, 1)), inputs=["x"], output=Tensor("out", (4, 3)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_transpose_to_indexmap_correctness():
    g = _make_transpose_graph()
    x = rng.standard_normal((3, 4)).astype(np.float32)
    before = _run(g, {"x": x})
    after = _run(_apply(g, _load("011_transpose_to_indexmap.py")), {"x": x})
    _assert_close(before, after)


# ===================================================================
# Reshape → IndexMapOp
# ===================================================================


def _make_reshape_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (3, 4)), node_id="x")
    g.add_node(op=ReshapeOp(shape=(2, 6)), inputs=["x"], output=Tensor("out", (2, 6)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_reshape_to_indexmap_correctness():
    g = _make_reshape_graph()
    x = rng.standard_normal((3, 4)).astype(np.float32)
    before = _run(g, {"x": x})
    after = _run(_apply(g, _load("012_reshape_to_indexmap.py")), {"x": x})
    _assert_close(before, after)


# ===================================================================
# Slice → IndexMapOp
# ===================================================================


def _make_slice_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4, 8)), node_id="x")
    g.add_node(op=ConstantOp(name="dim", value=1.0), inputs=[], output=Tensor("dim", (1,)), node_id="dim")
    g.add_node(op=ConstantOp(name="start", value=2.0), inputs=[], output=Tensor("start", (1,)), node_id="start")
    g.add_node(op=ConstantOp(name="end", value=6.0), inputs=[], output=Tensor("end", (1,)), node_id="end")
    g.add_node(op=SliceOp(shape=(4, 4)), inputs=["x", "dim", "start", "end"], output=Tensor("out", (4, 4)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_slice_to_indexmap_correctness():
    g = _make_slice_graph()
    x = rng.standard_normal((4, 8)).astype(np.float32)
    before = _run(g, {"x": x})
    after = _run(_apply(g, _load("013_slice_to_indexmap.py")), {"x": x})
    _assert_close(before, after)


# ===================================================================
# Cat → IndexMapOp
# ===================================================================


def _make_cat_graph():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 3)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (4, 5)), node_id="b")
    g.add_node(op=ConstantOp(name="dim", value=1.0), inputs=[], output=Tensor("dim", (1,)), node_id="dim")
    g.add_node(op=CatOp(), inputs=["a", "b", "dim"], output=Tensor("out", (4, 8)), node_id="out")
    g.inputs, g.outputs = ["a", "b"], ["out"]
    return g


def test_cat_to_indexmap_correctness():
    g = _make_cat_graph()
    a = rng.standard_normal((4, 3)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    before = _run(g, {"a": a, "b": b})
    after = _run(_apply(g, _load("014_cat_to_indexmap.py")), {"a": a, "b": b})
    _assert_close(before, after)
