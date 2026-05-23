"""End-to-end fp16 elementwise chain on the numpy backend.

Pure-Python paths only; the CUDA fp16 path is exercised in
``tests/compiler/test_dtype_cuda.py`` (step 4).
"""

from __future__ import annotations

import numpy as np

from deplodock.compiler import dtype as dt
from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.dtype import DataType
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp


def test_datatype_resolution_aliases():
    assert dt.get("float16") is dt.F16
    assert dt.get("half") is dt.F16
    assert dt.get("f16") is dt.F16
    assert dt.get(dt.F16) is dt.F16
    assert dt.F16.nbytes == 2
    assert dt.F32.nbytes == 4
    assert str(dt.F16) == "f16"


def test_tensor_dtype_coerces_string():
    t = Tensor("a", (4,), "float16")
    assert isinstance(t.dtype, DataType)
    assert t.dtype is dt.F16


def test_numpy_backend_elementwise_chain_fp16():
    """Build a tiny exp -> negate -> add chain on fp16 inputs; compare to numpy eager."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (8,), dt.F16), node_id="x")
    g.inputs.append(x)

    e = g.add_node(
        op=ElementwiseOp(op=np.exp),
        inputs=[x],
        output=Tensor("e", (8,), dt.F16),
        node_id="e",
    )
    n = g.add_node(
        op=ElementwiseOp(op=np.negative),
        inputs=[e],
        output=Tensor("n", (8,), dt.F16),
        node_id="n",
    )
    g.outputs.append(n)

    rng = np.random.default_rng(0)
    x_data = rng.standard_normal(8).astype(np.float16)

    be = NumpyBackend()
    out = be.run(be.compile(g), input_data={"x": x_data})[0].outputs["n"]

    assert out.dtype == np.float16, f"expected float16 output, got {out.dtype}"
    expected = (-np.exp(x_data.astype(np.float32))).astype(np.float16)
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)
