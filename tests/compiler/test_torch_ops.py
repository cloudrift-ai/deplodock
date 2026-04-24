"""Tests for graph ops across every backend.

Parametrized over backends (numpy / loop / cuda) via the ``run_graph``
fixture in ``conftest.py``. For each aten operation the tracer captures,
each backend's output is compared against PyTorch eager as the ground
truth. Requires PyTorch.
"""

import numpy as np
import pytest
import torch

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.frontend.ir import (
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
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, ReduceOp

rng = np.random.default_rng(42)


def _torch_to_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _run(run_graph, graph: Graph, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Execute through ``run_graph`` fixture and return the single output array."""
    return list(run_graph(graph, inputs).values())[0]


# ---------------------------------------------------------------------------
# Elementwise unary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,torch_fn",
    [
        ("neg", lambda x: torch.neg(x)),
        ("exp", lambda x: torch.exp(x)),
        ("rsqrt", lambda x: torch.rsqrt(x)),
        ("reciprocal", lambda x: torch.reciprocal(x)),
        ("relu", lambda x: torch.relu(x)),
        ("tanh", lambda x: torch.tanh(x)),
        ("sigmoid", lambda x: torch.sigmoid(x)),
        ("abs", lambda x: torch.abs(x)),
        ("silu", lambda x: torch.nn.functional.silu(x)),
    ],
)
def test_unary(fn, torch_fn, run_graph):
    x_np = rng.uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ElementwiseOp(fn), ["x"], Tensor("y", (4, 8)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch_fn(torch.from_numpy(x_np)))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=2e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Elementwise binary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn,torch_fn",
    [
        ("add", lambda x, y: x + y),
        ("sub", lambda x, y: x - y),
        ("mul", lambda x, y: x * y),
        ("div", lambda x, y: x / y),
    ],
)
def test_binary(fn, torch_fn, run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    y_np = rng.uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 8)), node_id="y")
    g.add_node(ElementwiseOp(fn), ["x", "y"], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    expected = _torch_to_np(torch_fn(torch.from_numpy(x_np), torch.from_numpy(y_np)))
    rtol = 1e-3 if fn == "div" else 1e-5
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np, "y": y_np}), expected, rtol=rtol, atol=1e-5)


def test_pow(run_graph):
    x_np = rng.uniform(0.1, 5.0, size=(4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ConstantOp(name="p", value=2.0), [], Tensor("p", (1,)), node_id="p")
    g.add_node(ElementwiseOp("pow"), ["x", "p"], Tensor("y", (4, 8)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).pow(2.0))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-5, atol=1e-5)


def test_add_broadcast(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    y_np = rng.standard_normal(8).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (8,)), node_id="y")
    g.add_node(ElementwiseOp("add"), ["x", "y"], Tensor("z", (4, 8)), node_id="z")
    g.inputs, g.outputs = ["x", "y"], ["z"]
    expected = _torch_to_np(torch.from_numpy(x_np) + torch.from_numpy(y_np))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np, "y": y_np}), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_reduce_sum(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).sum(dim=-1, keepdim=True))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-5, atol=1e-5)


def test_reduce_max(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("max", -1), ["x"], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).amax(dim=-1, keepdim=True))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-5, atol=1e-5)


def test_reduce_sum_keepdim(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).sum(dim=-1, keepdim=True))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Mean
# ---------------------------------------------------------------------------


def test_mean(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(MeanOp(axis=-1), ["x"], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).mean(dim=-1, keepdim=True))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# Layout ops
# ---------------------------------------------------------------------------


def test_transpose(run_graph):
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (3, 4)), node_id="x")
    g.add_node(TransposeOp(axes=(1, 0)), ["x"], Tensor("y", (4, 3)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).transpose(0, 1))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-6, atol=1e-6)


def test_transpose_perm(run_graph):
    x_np = rng.standard_normal((2, 3, 4)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 3, 4)), node_id="x")
    g.add_node(TransposeOp(axes=(0, 2, 1)), ["x"], Tensor("y", (2, 4, 3)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).permute(0, 2, 1))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-6, atol=1e-6)


def test_reshape(run_graph):
    x_np = rng.standard_normal((3, 4)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (3, 4)), node_id="x")
    g.add_node(ReshapeOp(shape=(2, 6)), ["x"], Tensor("y", (2, 6)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).reshape(2, 6))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-6, atol=1e-6)


def test_unsqueeze(run_graph):
    x_np = rng.standard_normal(4).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4,)), node_id="x")
    g.add_node(UnsqueezeOp(dim=0), ["x"], Tensor("y", (1, 4)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).unsqueeze(0))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Slice / Cat / Gather
# ---------------------------------------------------------------------------


def test_slice(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ConstantOp(name="dim", value=1.0), [], Tensor("dim", (1,)), node_id="dim")
    g.add_node(ConstantOp(name="start", value=2.0), [], Tensor("start", (1,)), node_id="start")
    g.add_node(ConstantOp(name="end", value=6.0), [], Tensor("end", (1,)), node_id="end")
    g.add_node(SliceOp(shape=(4, 4)), ["x", "dim", "start", "end"], Tensor("y", (4, 4)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np)[:, 2:6])
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=1e-6, atol=1e-6)


def test_cat(run_graph):
    a_np = rng.standard_normal((4, 3)).astype(np.float32)
    b_np = rng.standard_normal((4, 5)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 3)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 5)), node_id="b")
    g.add_node(ConstantOp(name="dim", value=1.0), [], Tensor("dim", (1,)), node_id="dim")
    g.add_node(CatOp(), ["a", "b", "dim"], Tensor("y", (4, 8)), node_id="y")
    g.inputs, g.outputs = ["a", "b"], ["y"]
    expected = _torch_to_np(torch.cat([torch.from_numpy(a_np), torch.from_numpy(b_np)], dim=1))
    np.testing.assert_allclose(_run(run_graph, g, {"a": a_np, "b": b_np}), expected, rtol=1e-6, atol=1e-6)


def test_gather(run_graph):
    x_np = rng.standard_normal((4, 8)).astype(np.float32)
    idx = rng.integers(0, 8, size=(4, 3))
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("idx", (4, 3)), node_id="idx")
    g.add_node(GatherOp(axis=1), ["x", "idx"], Tensor("y", (4, 3)), node_id="y")
    g.inputs, g.outputs = ["x", "idx"], ["y"]
    expected = _torch_to_np(torch.from_numpy(x_np).gather(1, torch.from_numpy(idx).long()))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np, "idx": idx.astype(np.float32)}), expected, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Matmul / Linear
# ---------------------------------------------------------------------------


def test_matmul(run_graph):
    a_np = rng.standard_normal((4, 8)).astype(np.float32)
    b_np = rng.standard_normal((8, 3)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 3)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (4, 3)), node_id="c")
    g.inputs, g.outputs = ["a", "b"], ["c"]
    expected = _torch_to_np(torch.from_numpy(a_np) @ torch.from_numpy(b_np))
    np.testing.assert_allclose(_run(run_graph, g, {"a": a_np, "b": b_np}), expected, rtol=5e-5, atol=2e-6)


def test_matmul_with_bias(run_graph):
    a_np = rng.standard_normal((4, 8)).astype(np.float32)
    b_np = rng.standard_normal((8, 3)).astype(np.float32)
    bias_np = rng.standard_normal(3).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 3)), node_id="b")
    g.add_node(InputOp(), [], Tensor("bias", (3,)), node_id="bias")
    g.add_node(MatmulOp(has_bias=True), ["a", "b", "bias"], Tensor("c", (4, 3)), node_id="c")
    g.inputs, g.outputs = ["a", "b", "bias"], ["c"]
    expected = _torch_to_np(torch.addmm(torch.from_numpy(bias_np), torch.from_numpy(a_np), torch.from_numpy(b_np)))
    np.testing.assert_allclose(_run(run_graph, g, {"a": a_np, "b": b_np, "bias": bias_np}), expected, rtol=5e-5, atol=2e-6)


def test_linear(run_graph):
    x_np = rng.standard_normal((2, 8)).astype(np.float32)
    w_np = rng.standard_normal((4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 8)), node_id="x")
    g.add_node(ConstantOp(name="w"), [], Tensor("w", (4, 8)), node_id="w")
    g.add_node(LinearOp(has_bias=False), ["x", "w"], Tensor("y", (2, 4)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.nn.functional.linear(torch.from_numpy(x_np), torch.from_numpy(w_np)))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np, "w": w_np}), expected, rtol=1e-4, atol=2e-6)


def test_linear_with_bias(run_graph):
    x_np = rng.standard_normal((2, 8)).astype(np.float32)
    w_np = rng.standard_normal((4, 8)).astype(np.float32)
    b_np = rng.standard_normal(4).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (2, 8)), node_id="x")
    g.add_node(ConstantOp(name="w"), [], Tensor("w", (4, 8)), node_id="w")
    g.add_node(ConstantOp(name="b"), [], Tensor("b", (4,)), node_id="b")
    g.add_node(LinearOp(has_bias=True), ["x", "w", "b"], Tensor("y", (2, 4)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    expected = _torch_to_np(torch.nn.functional.linear(torch.from_numpy(x_np), torch.from_numpy(w_np), torch.from_numpy(b_np)))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np, "w": w_np, "b": b_np}), expected, rtol=1e-4, atol=2e-6)


# ---------------------------------------------------------------------------
# SDPA
# ---------------------------------------------------------------------------


def test_sdpa(run_graph):
    q_np = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    k_np = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    v_np = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("q", (1, 2, 4, 8)), node_id="q")
    g.add_node(InputOp(), [], Tensor("k", (1, 2, 4, 8)), node_id="k")
    g.add_node(InputOp(), [], Tensor("v", (1, 2, 4, 8)), node_id="v")
    g.add_node(SdpaOp(), ["q", "k", "v"], Tensor("out", (1, 2, 4, 8)), node_id="out")
    g.inputs, g.outputs = ["q", "k", "v"], ["out"]
    expected = _torch_to_np(
        torch.nn.functional.scaled_dot_product_attention(torch.from_numpy(q_np), torch.from_numpy(k_np), torch.from_numpy(v_np))
    )
    np.testing.assert_allclose(_run(run_graph, g, {"q": q_np, "k": k_np, "v": v_np}), expected, rtol=5e-5, atol=2e-6)


def test_sdpa_causal(run_graph):
    """SDPA with causal masking: future positions should be masked out."""
    q_np = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
    k_np = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
    v_np = rng.standard_normal((1, 2, 8, 16)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("q", (1, 2, 8, 16)), node_id="q")
    g.add_node(InputOp(), [], Tensor("k", (1, 2, 8, 16)), node_id="k")
    g.add_node(InputOp(), [], Tensor("v", (1, 2, 8, 16)), node_id="v")
    g.add_node(SdpaOp(is_causal=True), ["q", "k", "v"], Tensor("out", (1, 2, 8, 16)), node_id="out")
    g.inputs, g.outputs = ["q", "k", "v"], ["out"]
    expected = _torch_to_np(
        torch.nn.functional.scaled_dot_product_attention(
            torch.from_numpy(q_np), torch.from_numpy(k_np), torch.from_numpy(v_np), is_causal=True
        )
    )
    np.testing.assert_allclose(_run(run_graph, g, {"q": q_np, "k": k_np, "v": v_np}), expected, rtol=1e-4, atol=1e-5)


def test_sdpa_gqa(run_graph):
    """GQA: Q has more heads than K/V (28 Q heads, 4 KV heads)."""
    B, Hq, Hkv, S, D = 1, 28, 4, 8, 16
    q_np = rng.standard_normal((B, Hq, S, D)).astype(np.float32)
    k_np = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
    v_np = rng.standard_normal((B, Hkv, S, D)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("q", (B, Hq, S, D)), node_id="q")
    g.add_node(InputOp(), [], Tensor("k", (B, Hkv, S, D)), node_id="k")
    g.add_node(InputOp(), [], Tensor("v", (B, Hkv, S, D)), node_id="v")
    g.add_node(SdpaOp(), ["q", "k", "v"], Tensor("out", (B, Hq, S, D)), node_id="out")
    g.inputs, g.outputs = ["q", "k", "v"], ["out"]
    # Reference: expand K/V heads to match Q, then standard SDPA.
    group = Hq // Hkv
    k_exp = torch.from_numpy(k_np).repeat_interleave(group, dim=1)
    v_exp = torch.from_numpy(v_np).repeat_interleave(group, dim=1)
    expected = _torch_to_np(torch.nn.functional.scaled_dot_product_attention(torch.from_numpy(q_np), k_exp, v_exp))
    np.testing.assert_allclose(_run(run_graph, g, {"q": q_np, "k": k_np, "v": v_np}), expected, rtol=5e-3, atol=1e-4)


# ---------------------------------------------------------------------------
# Compound graphs
# ---------------------------------------------------------------------------


def test_softmax_graph(run_graph):
    rows, cols = 4, 8
    x_np = rng.standard_normal((rows, cols)).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (rows, cols)), node_id="x")
    g.add_node(ReduceOp("max", -1), ["x"], Tensor("mx", (rows, 1)), node_id="mx")
    g.add_node(ElementwiseOp("sub"), ["x", "mx"], Tensor("sub", (rows, cols)), node_id="sub")
    g.add_node(ElementwiseOp("exp"), ["sub"], Tensor("exp", (rows, cols)), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (rows, 1)), node_id="sm")
    g.add_node(ElementwiseOp("div"), ["exp", "sm"], Tensor("out", (rows, cols)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    expected = _torch_to_np(torch.softmax(torch.from_numpy(x_np), dim=-1))
    np.testing.assert_allclose(_run(run_graph, g, {"x": x_np}), expected, rtol=2e-4, atol=1e-5)


def test_rmsnorm_graph(run_graph):
    rows, dim = 8, 64
    eps = 1e-6
    X_np = rng.standard_normal((rows, dim)).astype(np.float32)
    w_np = rng.standard_normal(dim).astype(np.float32)
    g = Graph()
    g.add_node(InputOp(), [], Tensor("X", (rows, dim)), node_id="X")
    g.add_node(ConstantOp(name="eps", value=eps), [], Tensor("eps", (1,)), node_id="eps")
    g.add_node(InputOp(), [], Tensor("w", (dim,)), node_id="w")
    g.add_node(ElementwiseOp("mul"), ["X", "X"], Tensor("sq", (rows, dim)), node_id="sq")
    g.add_node(ReduceOp("sum", axis=-1), ["sq"], Tensor("red", (rows, 1)), node_id="red")
    g.add_node(ElementwiseOp("add"), ["red", "eps"], Tensor("ae", (rows, 1)), node_id="ae")
    g.add_node(ElementwiseOp("rsqrt"), ["ae"], Tensor("rsq", (rows, 1)), node_id="rsq")
    g.add_node(ElementwiseOp("mul"), ["X", "rsq"], Tensor("norm", (rows, dim)), node_id="norm")
    g.add_node(ElementwiseOp("mul"), ["norm", "w"], Tensor("out", (rows, dim)), node_id="out")
    g.inputs, g.outputs = ["X", "w"], ["out"]

    X_t = torch.from_numpy(X_np)
    w_t = torch.from_numpy(w_np)
    sq_sum = X_t.pow(2).sum(-1, keepdim=True)
    expected = _torch_to_np(X_t * torch.rsqrt(sq_sum + eps) * w_t)
    np.testing.assert_allclose(_run(run_graph, g, {"X": X_np, "w": w_np}), expected, rtol=1e-4, atol=1e-5)
