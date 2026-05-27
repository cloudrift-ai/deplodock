"""Provenance-driven kernel naming + name-invariant CudaOp cache key (M3)."""

from deplodock.compiler import provenance as prov
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.cuda.ir import CudaOp
from deplodock.compiler.ir.frontend.ir import RmsNormOp
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.tile.ir import TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.search.keys import op_cache_key


def _pointwise_loop() -> LoopOp:
    i, j = Axis("i", 4), Axis("j", 8)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(Loop(axis=j, body=(Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                                              Write(output="o", index=(Var("i"), Var("j")), value="x_v"))),),
            ),
        )
    )


def _tile_names(g: Graph) -> list[str]:
    out = Pipeline.build(TILE_PASSES).run(g)
    return [n.op.name for n in out.nodes.values() if isinstance(n.op, TileOp)]


def test_full_single_op_gets_bare_name():
    """A kernel that fully realizes one meaningful op is named after it."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(_pointwise_loop(), ["x"], Tensor("o", (4, 8)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    prov.put(g.nodes["o"], {"rms_norm_0": {"kind": "RmsNormOp", "pieces": ["o"]}})

    assert "k_rms_norm" in _tile_names(g)


def test_glue_only_kernel_falls_back_to_node_id():
    """A kernel covering only generic glue ops keeps the node-id name."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(_pointwise_loop(), ["x"], Tensor("o", (4, 8)), node_id="o")
    g.inputs, g.outputs = ["x"], ["o"]
    prov.put(g.nodes["o"], {"mul_0": {"kind": "ElementwiseOp", "pieces": ["o"]}})

    assert "k_o_pointwise" in _tile_names(g)


def test_real_rms_norm_kernels_named_by_op():
    """A real (decomposed + fused) rms_norm names every kernel that carries
    it after the op — fully covered -> ``k_rms_norm``, split halves ->
    ``k_rms_norm_reduce`` / ``k_rms_norm_pointwise``."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("rms_norm_0", (1, 4, 8)), node_id="rms_norm_0")
    g.inputs, g.outputs = ["x", "w"], ["rms_norm_0"]

    names = _tile_names(g)
    assert names, "rms_norm should lower to at least one kernel"
    assert all("rms_norm" in nm for nm in names), names


def test_cuda_op_cache_key_is_name_invariant():
    """Renaming a kernel must not change its CudaOp cache key (so renames
    don't bust the tune DB and isolated-kernel tuning transfers)."""
    src = 'extern "C" __global__ void {name}(float* x, float* o) {{ o[0] = x[0]; }}'
    a = CudaOp(kernel_source=src.format(name="k_rms_norm"), kernel_name="k_rms_norm", arg_order=("x", "o"))
    b = CudaOp(kernel_source=src.format(name="k_merged_lift_n5"), kernel_name="k_merged_lift_n5", arg_order=("x", "o"))
    assert op_cache_key(a) == op_cache_key(b)

    # A genuine structural difference still changes the key.
    c = CudaOp(kernel_source=src.format(name="k_rms_norm"), kernel_name="k_rms_norm", arg_order=("x", "o"), smem_bytes=128)
    assert op_cache_key(a) != op_cache_key(c)
