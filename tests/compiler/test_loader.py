"""Tests for the constant binder + safetensors loader."""

from __future__ import annotations

import json

import numpy as np
import pytest

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.frontend.ir import ReshapeOp, TransposeOp
from deplodock.compiler.loader.binder import apply_load_ops, bind_constants


def test_apply_load_ops_empty_returns_source():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = apply_load_ops(src, ())
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out, src)


def test_apply_load_ops_transpose():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = apply_load_ops(src, (TransposeOp(axes=(1, 0)),))
    assert out.shape == (4, 3)
    np.testing.assert_array_equal(out, src.T)


def test_apply_load_ops_chain_transpose_then_reshape():
    src = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    chain = (TransposeOp(axes=(0, 2, 1)), ReshapeOp(shape=(2, 12)))
    out = apply_load_ops(src, chain)
    assert out.shape == (2, 12)
    np.testing.assert_array_equal(out, np.transpose(src, (0, 2, 1)).reshape(2, 12))


def test_bind_constants_resolves_by_source_path():
    g = Graph()
    g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(2, 3), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_w", (2, 3), "f32"),
        node_id="p_w",
    )
    sources = {"layer.weight": np.ones((2, 3), dtype=np.float32) * 7.0}
    out = bind_constants(g, sources)
    assert "p_w" in out
    np.testing.assert_array_equal(out["p_w"], sources["layer.weight"])


def test_bind_constants_runs_load_ops_chain():
    g = Graph()
    g.add_node(
        op=ConstantOp(
            name="p_w",
            load_ops=(TransposeOp(axes=(1, 0)),),
            source_path="layer.weight",
            source_shape=(2, 3),
            source_dtype="f32",
        ),
        inputs=[],
        output=Tensor("p_w", (3, 2), "f32"),
        node_id="p_w",
    )
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = bind_constants(g, {"layer.weight": src})
    assert out["p_w"].shape == (3, 2)
    np.testing.assert_array_equal(out["p_w"], src.T)


def test_bind_constants_skips_scalars_and_unknowns():
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (3,), "f32"), node_id="x")
    g.add_node(op=ConstantOp(name="eps", value=1e-6), inputs=[], output=Tensor("eps", (1,), "f32"), node_id="eps")
    g.add_node(
        op=ConstantOp(name="p_unknown", source_path="missing.weight", source_shape=(2,), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_unknown", (2,), "f32"),
        node_id="p_unknown",
    )
    out = bind_constants(g, sources={})
    assert out == {}


def test_safetensors_loader_roundtrip(tmp_path):
    """Write a tiny single-shard safetensors and load it through our loader."""
    safetensors = pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    weights = {
        "layer.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "layer.bias": np.arange(4, dtype=np.float32),
    }
    save_file(weights, str(tmp_path / "model.safetensors"))

    g = Graph()
    g.add_node(
        op=ConstantOp(
            name="p_w",
            load_ops=(TransposeOp(axes=(1, 0)),),
            source_path="layer.weight",
            source_shape=(3, 4),
            source_dtype="f32",
        ),
        inputs=[],
        output=Tensor("p_w", (4, 3), "f32"),
        node_id="p_w",
    )
    g.add_node(
        op=ConstantOp(
            name="p_b",
            source_path="layer.bias",
            source_shape=(4,),
            source_dtype="f32",
        ),
        inputs=[],
        output=Tensor("p_b", (4,), "f32"),
        node_id="p_b",
    )

    from deplodock.compiler.loader.safetensors import load_constants_from_safetensors

    out = load_constants_from_safetensors(g, str(tmp_path))
    np.testing.assert_array_equal(out["p_w"], weights["layer.weight"].T)
    np.testing.assert_array_equal(out["p_b"], weights["layer.bias"])
    _ = safetensors  # silence unused import


def test_safetensors_loader_sharded_index(tmp_path):
    """Two-shard layout via ``model.safetensors.index.json``."""
    pytest.importorskip("safetensors")
    from safetensors.numpy import save_file

    save_file({"layer.weight": np.full((2, 2), 1.0, dtype=np.float32)}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"layer.bias": np.full((2,), 2.0, dtype=np.float32)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.weight": "model-00001-of-00002.safetensors",
                    "layer.bias": "model-00002-of-00002.safetensors",
                }
            }
        )
    )

    g = Graph()
    g.add_node(
        op=ConstantOp(name="p_w", source_path="layer.weight", source_shape=(2, 2), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_w", (2, 2), "f32"),
        node_id="p_w",
    )
    g.add_node(
        op=ConstantOp(name="p_b", source_path="layer.bias", source_shape=(2,), source_dtype="f32"),
        inputs=[],
        output=Tensor("p_b", (2,), "f32"),
        node_id="p_b",
    )

    from deplodock.compiler.loader.safetensors import load_constants_from_safetensors

    out = load_constants_from_safetensors(g, str(tmp_path))
    np.testing.assert_array_equal(out["p_w"], np.full((2, 2), 1.0, dtype=np.float32))
    np.testing.assert_array_equal(out["p_b"], np.full((2,), 2.0, dtype=np.float32))


def test_load_ops_serialize_roundtrip():
    g = Graph()
    g.add_node(
        op=ConstantOp(
            name="p_w",
            load_ops=(TransposeOp(axes=(1, 0)), ReshapeOp(shape=(2, 6))),
            source_path="layer.weight",
            source_shape=(3, 4),
            source_dtype="f32",
        ),
        inputs=[],
        output=Tensor("p_w", (2, 6), "f32"),
        node_id="p_w",
    )
    data = g.to_dict()
    g2 = Graph.from_dict(data)
    op = g2.nodes["p_w"].op
    assert isinstance(op, ConstantOp)
    assert op.source_path == "layer.weight"
    assert op.source_shape == (3, 4)
    assert len(op.load_ops) == 2
    assert isinstance(op.load_ops[0], TransposeOp)
    assert op.load_ops[0].axes == (1, 0)
    assert isinstance(op.load_ops[1], ReshapeOp)
    assert op.load_ops[1].shape == (2, 6)
