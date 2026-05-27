"""torch_ref: the Graph→torch evaluator used as the eager reference for
``deplodock run --ir``. Validated against each op's numpy ``forward()`` on
CPU (no GPU needed)."""

import numpy as np
import pytest

from deplodock.compiler.backend import torch_ref
from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp, SoftmaxOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, GatherOp, ReduceOp

# torch is only needed to build reference tensors / call the evaluator; the
# deplodock imports above are torch-free, so gate after them.
torch = pytest.importorskip("torch")


def _assert_matches_numpy(g: Graph, arrays: dict[str, np.ndarray]):
    """torch_ref output == numpy forward() output for graph ``g``."""
    be = NumpyBackend()
    npy = be.run(be.compile(g), input_data=arrays)[0].outputs[g.outputs[0]]

    tin = {k: torch.from_numpy(v.astype(np.float32)) for k, v in arrays.items()}
    fn, inputs = torch_ref.build_callable(g, tin)
    with torch.no_grad():
        tout = fn(*inputs).cpu().numpy()

    np.testing.assert_allclose(tout, npy, rtol=1e-4, atol=1e-4)


def _rng():
    return np.random.default_rng(0)


def test_rms_norm():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("o", (1, 4, 8)), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]
    r = _rng()
    _assert_matches_numpy(g, {"x": r.standard_normal((1, 4, 8)), "w": r.standard_normal((8,))})


def test_linear_and_elementwise():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (16, 8)), node_id="w")
    g.add_node(LinearOp(), ["x", "w"], Tensor("h", (4, 16)), node_id="h")
    g.add_node(ElementwiseOp(op="silu"), ["h"], Tensor("o", (4, 16)), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]
    r = _rng()
    _assert_matches_numpy(g, {"x": r.standard_normal((4, 8)), "w": r.standard_normal((16, 8))})


def test_matmul_softmax():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (8, 4)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("s", (4, 4)), node_id="s")
    g.add_node(SoftmaxOp(axis=-1), ["s"], Tensor("o", (4, 4)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    r = _rng()
    _assert_matches_numpy(g, {"a": r.standard_normal((4, 8)), "b": r.standard_normal((8, 4))})


def test_reduce_sum_keepdim():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp(op="sum", axis=-1), ["x"], Tensor("o", (4, 1)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    _assert_matches_numpy(g, {"x": _rng().standard_normal((4, 8))})


def test_is_runnable_rejects_gather():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("idx", (4, 8)), node_id="idx")
    g.add_node(GatherOp(axis=0), ["x", "idx"], Tensor("o", (4, 8)), node_id="o")
    g.inputs, g.outputs = ["x", "idx"], ["o"]
    assert not torch_ref.is_runnable(g)


def test_is_runnable_accepts_frontend():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("o", (1, 4, 8)), node_id="o")
    g.inputs, g.outputs = ["x", "w"], ["o"]
    assert torch_ref.is_runnable(g)
