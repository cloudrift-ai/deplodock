"""End-to-end accuracy tests: compare deplodock output against PyTorch eager.

Uses actual PyTorch tensor data (not pseudo-random) so the numerical
comparison is meaningful.  Requires a GPU and transformers.
"""

import math

import pytest
import torch

from deplodock.compiler.fusion import auto_fuse
from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ConstantOp, ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.plan import plan_graph

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)


def _compile_and_run_with_data(graph: Graph, input_data: dict[str, list[float]]) -> dict[str, list[float]]:
    """Full pipeline with actual input data — runs the rewriter so view ops
    (TransposeOp/SliceOp/CatOp/UnsqueezeOp) get lowered to IndexMapOp before
    fusion + compile."""
    from pathlib import Path

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.rewriter import Rewriter

    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    graph = Rewriter.from_directory(rules_dir).apply(graph)
    fused = auto_fuse(graph)
    plan = plan_graph(fused)
    backend = CudaBackend()
    program = backend.compile(plan)
    result = backend.run(program, input_data=input_data)
    return result.outputs


@requires_cuda
def test_e2e_rmsnorm_matches_torch():
    """RMSNorm: deplodock matches PyTorch RMSNorm with same inputs."""
    torch.manual_seed(42)
    rows, dim = 8, 64
    eps = 1e-6

    x_t = torch.randn(rows, dim).cuda()
    w_t = torch.randn(dim).cuda()

    # PyTorch reference: rsqrt(sum(x^2) + eps) * x * w
    # Note: LlamaRMSNorm uses sum (not mean) in the traced decomposition.
    sq_sum = x_t.pow(2).sum(-1, keepdim=True)
    ref = x_t * torch.rsqrt(sq_sum + eps) * w_t
    expected = ref.cpu().flatten().tolist()

    # Deplodock graph — same decomposition as torch.export trace
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (rows, dim)), node_id="X")
    eps_n = g.add_node(ConstantOp(name="eps"), [], Tensor("eps", (1,)), node_id="eps")
    w = g.add_node(InputOp(), [], Tensor("w", (dim,)), node_id="w")
    g.inputs = [x, w]

    sq = g.add_node(ElementwiseOp("mul"), [x, x], Tensor("sq", (rows, dim)), node_id="sq")
    red = g.add_node(ReduceOp("sum", axis=-1), [sq], Tensor("red", (rows, 1)), node_id="red")
    ae = g.add_node(ElementwiseOp("add"), [red, eps_n], Tensor("ae", (rows, 1)), node_id="ae")
    rsq = g.add_node(ElementwiseOp("rsqrt"), [ae], Tensor("rsq", (rows, 1)), node_id="rsq")
    norm = g.add_node(ElementwiseOp("mul"), [x, rsq], Tensor("norm", (rows, dim)), node_id="norm")
    out = g.add_node(ElementwiseOp("mul"), [norm, w], Tensor("out", (rows, dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "X": x_t.cpu().flatten().tolist(),
        "w": w_t.cpu().flatten().tolist(),
        "eps": [eps],
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    assert len(actual) == len(expected)
    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.01, f"RMSNorm max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_matmul_matches_torch():
    """Matmul: deplodock matches torch.mm with same inputs."""
    torch.manual_seed(42)
    m, k, n = 16, 32, 64

    a_t = torch.randn(m, k).cuda()
    b_t = torch.randn(k, n).cuda()

    ref = torch.mm(a_t, b_t)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    g.inputs = [a, b]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (m, k, n)), node_id="ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("C", (m, n)), node_id="C")
    g.outputs = [out]

    input_data = {
        "A": a_t.cpu().flatten().tolist(),
        "B": b_t.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.01, f"Matmul max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_softmax_matches_torch():
    """Softmax: deplodock matches torch.softmax with same inputs."""
    torch.manual_seed(42)
    rows, cols = 8, 32

    x_t = torch.randn(rows, cols).cuda()
    ref = torch.softmax(x_t, dim=-1)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (rows, cols)), node_id="X")
    g.inputs = [x]

    mx = g.add_node(ReduceOp("max", axis=-1), [x], Tensor("mx", (rows, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [x, mx], Tensor("sub", (rows, cols)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (rows, cols)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (rows, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (rows, cols)), node_id="out")
    g.outputs = [out]

    input_data = {"X": x_t.cpu().flatten().tolist()}
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-4, f"Softmax max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_matmul_softmax_matches_torch():
    """Matmul + softmax: deplodock matches torch with same inputs."""
    torch.manual_seed(42)
    m, k, n = 8, 16, 32

    a_t = torch.randn(m, k).cuda()
    b_t = torch.randn(k, n).cuda()
    scale = 1.0 / math.sqrt(k)

    scores = torch.mm(a_t, b_t) * scale
    ref = torch.softmax(scores, dim=-1)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (m, k)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (k, n)), node_id="B")
    sc = g.add_node(ConstantOp(name="scale"), [], Tensor("scale", (1,)), node_id="scale")
    g.inputs = [a, b]

    ew = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("ew", (m, k, n)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (m, n)), node_id="mm")
    scaled = g.add_node(ElementwiseOp("mul"), [mm, sc], Tensor("sc", (m, n)), node_id="sc")
    mx = g.add_node(ReduceOp("max", axis=-1), [scaled], Tensor("mx", (m, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [scaled, mx], Tensor("sub", (m, n)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (m, n)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (m, 1)), node_id="sm")
    out = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("out", (m, n)), node_id="out")
    g.outputs = [out]

    input_data = {
        "A": a_t.cpu().flatten().tolist(),
        "B": b_t.cpu().flatten().tolist(),
        "scale": [scale],
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 1e-3, f"Matmul+softmax max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_gqa_attn_v_matches_torch():
    """GQA attn@V: batched matmul with broadcast batch dims matches torch.

    attn_weights(q_heads, seq, seq) @ V(kv_heads, seq, dim) where
    q_heads > kv_heads (Grouped Query Attention).
    """
    torch.manual_seed(42)
    q_heads, kv_heads = 4, 2  # group_size = 2
    seq, dim = 8, 16

    attn_t = torch.randn(q_heads, seq, seq).cuda()
    v_t = torch.randn(kv_heads, seq, dim).cuda()

    # PyTorch reference: each Q head group shares one KV head
    group_size = q_heads // kv_heads
    v_expanded = v_t.repeat_interleave(group_size, dim=0)  # (q_heads, seq, dim)
    ref = torch.bmm(attn_t, v_expanded)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("attn", (q_heads, seq, seq)), node_id="attn")
    v = g.add_node(InputOp(), [], Tensor("V", (kv_heads, seq, dim)), node_id="V")
    g.inputs = [a, v]

    ew = g.add_node(ElementwiseOp("mul"), [a, v], Tensor("ew", (q_heads, seq, seq, dim)), node_id="ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("out", (q_heads, seq, dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "attn": attn_t.cpu().flatten().tolist(),
        "V": v_t.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.05, f"GQA attn@V max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_gqa_attn_v_qwen_shapes():
    """GQA attn@V with Qwen-like shapes: 28 Q heads, 4 KV heads."""
    torch.manual_seed(42)
    q_heads, kv_heads = 28, 4
    seq, dim = 8, 16  # small for test speed

    attn_t = torch.randn(q_heads, seq, seq).cuda()
    v_t = torch.randn(kv_heads, seq, dim).cuda()

    group_size = q_heads // kv_heads
    v_expanded = v_t.repeat_interleave(group_size, dim=0)
    ref = torch.bmm(attn_t, v_expanded)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("attn", (q_heads, seq, seq)), node_id="attn")
    v = g.add_node(InputOp(), [], Tensor("V", (kv_heads, seq, dim)), node_id="V")
    g.inputs = [a, v]

    ew = g.add_node(ElementwiseOp("mul"), [a, v], Tensor("ew", (q_heads, seq, seq, dim)), node_id="ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("out", (q_heads, seq, dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "attn": attn_t.cpu().flatten().tolist(),
        "V": v_t.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.1, f"GQA attn@V (Qwen) max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_5d_broadcast_mul():
    """5D broadcast mul: (1,28,32,128) * (1,1,1,32,128) → (1,1,28,32,128).

    This is the rotary embedding cos/sin multiply pattern from Qwen.
    """
    torch.manual_seed(42)
    a = torch.randn(1, 28, 32, 128).cuda()
    b = torch.randn(1, 1, 1, 32, 128).cuda()
    ref = (a * b).cpu().flatten().tolist()

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (1, 28, 32, 128)), node_id="X")
    c = g.add_node(InputOp(), [], Tensor("cos", (1, 1, 1, 32, 128)), node_id="cos")
    g.inputs = [x, c]
    out = g.add_node(ElementwiseOp("mul"), [x, c], Tensor("out", (1, 1, 28, 32, 128)), node_id="out")
    g.outputs = [out]

    input_data = {"X": a.cpu().flatten().tolist(), "cos": b.cpu().flatten().tolist()}
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, ref, strict=True))
    assert max_diff < 1e-5, f"5D broadcast mul max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_rotary_embedding():
    """Full rotary embedding: Q*cos + rotate_half(Q)*sin with 4D+5D shapes.

    rotate_half(x) = cat([-x[..., dim//2:], x[..., :dim//2]], dim=-1)
    """
    torch.manual_seed(42)
    heads, seq, dim = 4, 8, 16

    q = torch.randn(1, heads, seq, dim).cuda()
    cos_t = torch.randn(1, 1, seq, dim).cuda()
    sin_t = torch.randn(1, 1, seq, dim).cuda()

    # PyTorch reference
    half = dim // 2
    q_rot = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    ref = (q * cos_t + q_rot * sin_t).cpu().flatten().tolist()

    # Deplodock graph: same ops as the traced rotary
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("Q", (1, heads, seq, dim)), node_id="Q")
    cos_n = g.add_node(InputOp(), [], Tensor("cos", (1, 1, seq, dim)), node_id="cos")
    sin_n = g.add_node(InputOp(), [], Tensor("sin", (1, 1, seq, dim)), node_id="sin")
    g.inputs = [x, cos_n, sin_n]

    # Q * cos (broadcasts cos from (1,1,seq,dim) to (1,heads,seq,dim))
    mul_cos = g.add_node(ElementwiseOp("mul"), [x, cos_n], Tensor("qcos", (1, heads, seq, dim)), node_id="qcos")

    # rotate_half: cat(-x[half:], x[:half])  — simplified as neg + cat
    # For the test, just compute q_rot directly and multiply
    # (testing the broadcast, not the slice/cat decomposition)
    # Pass q_rot as an input since slice/cat decomposition is complex
    q_rot_n = g.add_node(InputOp(), [], Tensor("Qrot", (1, heads, seq, dim)), node_id="Qrot")
    g.inputs.append(q_rot_n)

    mul_sin = g.add_node(ElementwiseOp("mul"), [q_rot_n, sin_n], Tensor("qsin", (1, heads, seq, dim)), node_id="qsin")
    out = g.add_node(ElementwiseOp("add"), [mul_cos, mul_sin], Tensor("out", (1, heads, seq, dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "Q": q.cpu().flatten().tolist(),
        "cos": cos_t.cpu().flatten().tolist(),
        "sin": sin_t.cpu().flatten().tolist(),
        "Qrot": q_rot.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, ref, strict=True))
    assert max_diff < 1e-4, f"Rotary embedding max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_mid_dim_broadcast():
    """Mid-dimension broadcast: (2,3,4,5) * (1,3,1,5) — fails with flat modulo."""
    torch.manual_seed(42)
    a = torch.randn(2, 3, 4, 5).cuda()
    b = torch.randn(1, 3, 1, 5).cuda()
    ref = (a * b).cpu().flatten().tolist()

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("X", (2, 3, 4, 5)), node_id="X")
    y = g.add_node(InputOp(), [], Tensor("Y", (1, 3, 1, 5)), node_id="Y")
    g.inputs = [x, y]
    out = g.add_node(ElementwiseOp("mul"), [x, y], Tensor("out", (2, 3, 4, 5)), node_id="out")
    g.outputs = [out]

    input_data = {"X": a.cpu().flatten().tolist(), "Y": b.cpu().flatten().tolist()}
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, ref, strict=True))
    assert max_diff < 1e-5, f"Mid-dim broadcast max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_gqa_qk_softmax_v():
    """Full GQA attention: QK^T + scale + softmax + @V with mismatched heads."""
    torch.manual_seed(42)
    q_heads, kv_heads = 4, 2
    seq, head_dim = 4, 8
    scale = 1.0 / (head_dim**0.5)

    q = torch.randn(q_heads, seq, head_dim).cuda()
    k = torch.randn(kv_heads, seq, head_dim).cuda()
    v = torch.randn(kv_heads, seq, head_dim).cuda()

    # PyTorch reference
    group = q_heads // kv_heads
    k_exp = k.repeat_interleave(group, dim=0)
    v_exp = v.repeat_interleave(group, dim=0)
    scores = torch.bmm(q, k_exp.transpose(-2, -1)) * scale
    attn = torch.softmax(scores, dim=-1)
    ref = torch.bmm(attn, v_exp).cpu().flatten().tolist()

    # Build the full graph: QK^T → scale → softmax → @V
    g = Graph()
    q_n = g.add_node(InputOp(), [], Tensor("Q", (q_heads, seq, head_dim)), node_id="Q")
    k_n = g.add_node(InputOp(), [], Tensor("K", (kv_heads, seq, head_dim)), node_id="K")
    v_n = g.add_node(InputOp(), [], Tensor("V", (kv_heads, seq, head_dim)), node_id="V")
    sc_n = g.add_node(ConstantOp(name="scale", value=scale), [], Tensor("scale", (1,)), node_id="scale")
    g.inputs = [q_n, k_n, v_n]

    # QK^T = mul(Q, K^T) → reduce_sum
    from deplodock.compiler.ops import TransposeOp

    kt_n = g.add_node(TransposeOp(axes=(-2, -1)), [k_n], Tensor("KT", (kv_heads, head_dim, seq)), node_id="KT")
    qk_ew = g.add_node(ElementwiseOp("mul"), [q_n, kt_n], Tensor("qk_ew", (q_heads, seq, head_dim, seq)), node_id="qk_ew")
    qk = g.add_node(ReduceOp("sum", axis=-1), [qk_ew], Tensor("qk", (q_heads, seq, seq)), node_id="qk")
    scaled = g.add_node(ElementwiseOp("mul"), [qk, sc_n], Tensor("scaled", (q_heads, seq, seq)), node_id="scaled")

    # Softmax
    mx = g.add_node(ReduceOp("max", axis=-1), [scaled], Tensor("mx", (q_heads, seq, 1)), node_id="mx")
    sub = g.add_node(ElementwiseOp("sub"), [scaled, mx], Tensor("sub", (q_heads, seq, seq)), node_id="sub")
    exp = g.add_node(ElementwiseOp("exp"), [sub], Tensor("exp", (q_heads, seq, seq)), node_id="exp")
    sm = g.add_node(ReduceOp("sum", axis=-1), [exp], Tensor("sm", (q_heads, seq, 1)), node_id="sm")
    attn_w = g.add_node(ElementwiseOp("div"), [exp, sm], Tensor("attn_w", (q_heads, seq, seq)), node_id="attn_w")

    # Attn @ V
    sv_ew = g.add_node(ElementwiseOp("mul"), [attn_w, v_n], Tensor("sv_ew", (q_heads, seq, seq, head_dim)), node_id="sv_ew")
    out = g.add_node(ReduceOp("sum", axis=-1), [sv_ew], Tensor("out", (q_heads, seq, head_dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "Q": q.cpu().flatten().tolist(),
        "K": k.cpu().flatten().tolist(),
        "V": v.cpu().flatten().tolist(),
        "scale": [scale],
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, ref, strict=True))
    assert max_diff < 0.05, f"GQA QK+softmax+V max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_transpose_4d():
    """4D transpose: (1,32,28,128) → (1,28,32,128) — swap dims 1 and 2."""
    torch.manual_seed(42)
    a = torch.randn(1, 32, 28, 128).cuda()
    ref = a.transpose(1, 2).contiguous().cpu().flatten().tolist()

    g = Graph()
    from deplodock.compiler.ops import TransposeOp

    x = g.add_node(InputOp(), [], Tensor("X", (1, 32, 28, 128)), node_id="X")
    g.inputs = [x]
    out = g.add_node(TransposeOp(axes=(1, 2)), [x], Tensor("out", (1, 28, 32, 128)), node_id="out")
    g.outputs = [out]

    input_data = {"X": a.cpu().flatten().tolist()}
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, ref, strict=True))
    assert max_diff < 1e-6, f"4D transpose max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_linear_reshape_transpose():
    """matmul + bias + reshape + transpose — Q projection chain."""
    torch.manual_seed(42)
    seq, hidden, heads, dim = 8, 64, 4, 16
    assert heads * dim == hidden

    x = torch.randn(1, seq, hidden).cuda()
    w = torch.randn(hidden, hidden).cuda()
    bias = torch.randn(hidden).cuda()

    # PyTorch reference: our graph does X @ W (not X @ W.T)
    out_ref = (x @ w + bias).view(1, seq, heads, dim).transpose(1, 2)
    expected = out_ref.contiguous().cpu().flatten().tolist()

    g = Graph()
    from deplodock.compiler.ops import TransposeOp

    x_n = g.add_node(InputOp(), [], Tensor("X", (1, seq, hidden)), node_id="X")
    w_n = g.add_node(InputOp(), [], Tensor("W", (hidden, hidden)), node_id="W")
    b_n = g.add_node(InputOp(), [], Tensor("bias", (hidden,)), node_id="bias")
    g.inputs = [x_n, w_n, b_n]

    # matmul
    ew = g.add_node(ElementwiseOp("mul"), [x_n, w_n], Tensor("ew", (1, seq, hidden, hidden)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (1, seq, hidden)), node_id="mm")
    # + bias
    add = g.add_node(ElementwiseOp("add"), [mm, b_n], Tensor("add", (1, seq, hidden)), node_id="add")
    # reshape (1, seq, hidden) → (1, seq, heads, dim)
    from deplodock.compiler.ops import ReshapeOp

    view = g.add_node(ReshapeOp(shape=(1, seq, heads, dim)), [add], Tensor("view", (1, seq, heads, dim)), node_id="view")
    # transpose (1, seq, heads, dim) → (1, heads, seq, dim)
    out = g.add_node(TransposeOp(axes=(1, 2)), [view], Tensor("out", (1, heads, seq, dim)), node_id="out")
    g.outputs = [out]

    input_data = {
        "X": x.cpu().flatten().tolist(),
        "W": w.cpu().flatten().tolist(),
        "bias": bias.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.01, f"Linear+reshape+transpose max diff = {max_diff:.6f}"


@requires_cuda
def test_e2e_linear_nonsquare():
    """nn.Linear with non-square weight: x(1,4,8) @ w.T(8,6) + b(6)."""
    torch.manual_seed(42)

    x = torch.randn(1, 4, 8).cuda()
    w = torch.randn(6, 8).cuda()  # (out=6, in=8) — PyTorch convention
    b = torch.randn(6).cuda()

    ref = torch.nn.functional.linear(x, w, b)
    expected = ref.cpu().flatten().tolist()

    g = Graph()
    from deplodock.compiler.ops import TransposeOp

    x_n = g.add_node(InputOp(), [], Tensor("X", (1, 4, 8)), node_id="X")
    w_n = g.add_node(InputOp(), [], Tensor("W", (6, 8)), node_id="W")
    b_n = g.add_node(InputOp(), [], Tensor("bias", (6,)), node_id="bias")
    g.inputs = [x_n, w_n, b_n]

    # Transpose weight: (6,8) → (8,6)
    wt = g.add_node(TransposeOp(axes=(-2, -1)), [w_n], Tensor("WT", (8, 6)), node_id="WT")
    # matmul: x(1,4,8) @ wt(8,6)
    ew = g.add_node(ElementwiseOp("mul"), [x_n, wt], Tensor("ew", (1, 4, 8, 6)), node_id="ew")
    mm = g.add_node(ReduceOp("sum", axis=-1), [ew], Tensor("mm", (1, 4, 6)), node_id="mm")
    out = g.add_node(ElementwiseOp("add"), [mm, b_n], Tensor("out", (1, 4, 6)), node_id="out")
    g.outputs = [out]

    input_data = {
        "X": x.cpu().flatten().tolist(),
        "W": w.cpu().flatten().tolist(),
        "bias": b.cpu().flatten().tolist(),
    }
    outputs = _compile_and_run_with_data(g, input_data)
    actual = list(outputs.values())[0]

    max_diff = max(abs(a - e) for a, e in zip(actual, expected, strict=True))
    assert max_diff < 0.01, f"Linear non-square max diff = {max_diff:.6f}"


# ===========================================================================
# Full transformer block e2e tests
# ===========================================================================


def _run_block_e2e(model_name: str, seq_len: int = 8):
    """Trace a transformer block, compile with deplodock, compare against eager.

    Returns (max_diff, mean_diff, num_elements).
    """
    from pathlib import Path

    from transformers import AutoModelForCausalLM

    from deplodock.compiler.backend.cuda.backend import CudaBackend
    from deplodock.compiler.rewriter import Rewriter
    from deplodock.compiler.torch_trace import trace_module

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    config = model.config
    block = model.model.layers[0].cuda()
    hidden = config.hidden_size
    head_dim = hidden // config.num_attention_heads

    torch.manual_seed(42)
    x = torch.randn(1, seq_len, hidden, device="cuda")
    cos = torch.randn(1, 1, seq_len, head_dim, device="cuda")
    sin = torch.randn(1, 1, seq_len, head_dim, device="cuda")

    # Eager reference
    with torch.no_grad():
        eager_out = block(x, position_embeddings=(cos, sin))[0]
    eager_flat = eager_out.cpu().flatten()

    # Trace + compile
    block.cpu()
    graph = trace_module(block, (x.cpu(),), kwargs={"position_embeddings": (cos.cpu(), sin.cpu())})
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    compiled = Rewriter.from_directory(rules_dir).apply(graph)
    fused = auto_fuse(compiled)
    plan = plan_graph(fused)
    backend = CudaBackend()
    program = backend.compile(plan)

    # Build input_data from actual tensors and weights
    block.cuda()
    input_data: dict[str, list[float]] = {
        "hidden_states": x.cpu().flatten().tolist(),
        "position_embeddings_0": cos.cpu().flatten().tolist(),
        "position_embeddings_1": sin.cpu().flatten().tolist(),
    }
    for buf in program.buffers:
        if buf.role == "constant":
            for key, param in block.named_parameters():
                safe_key = "p_" + key.replace(".", "_")
                if safe_key.endswith(buf.name[2:]) and param.numel() == buf.size:
                    input_data[buf.name] = param.detach().cpu().flatten().tolist()
                    break
            if buf.name not in input_data and buf.size == 1:
                for src_graph in (compiled, graph):
                    for nid, node in src_graph.nodes.items():
                        if isinstance(node.op, ConstantOp) and node.op.value is not None:
                            if buf.name == nid or buf.name.endswith(nid):
                                input_data[buf.name] = [node.op.value]
                                break
                    if buf.name in input_data:
                        break

    result = backend.run(program, input_data=input_data)
    deplodock_flat = torch.tensor(list(result.outputs.values())[0])

    diff = (deplodock_flat - eager_flat).abs()
    return diff.max().item(), diff.mean().item(), diff.numel()


@requires_cuda
def test_e2e_tinyllama_block():
    """TinyLlama transformer block matches eager PyTorch."""
    max_diff, mean_diff, n = _run_block_e2e("TinyLlama/TinyLlama-1.1B-Chat-v1.0", seq_len=8)
    assert max_diff < 0.5, f"TinyLlama max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f} ({n} elements)"


@requires_cuda
def test_e2e_qwen_block():
    """Qwen 2.5-7B transformer block: track accuracy vs eager."""
    max_diff, mean_diff, n = _run_block_e2e("Qwen/Qwen2.5-7B", seq_len=8)
    # Qwen's larger hidden_size (3584 vs TinyLlama's 2048) means more fp32
    # accumulation noise, so tolerance is larger than TinyLlama's 0.5.
    assert max_diff < 1.0, f"Qwen max_diff={max_diff:.4f}, mean_diff={mean_diff:.6f} ({n} elements)"
