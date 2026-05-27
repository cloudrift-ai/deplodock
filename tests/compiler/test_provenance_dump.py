"""Per-kernel Torch reproducer dump (M4).

The dump slices the pristine pre-decomposition graph by each kernel's prov
origins → ``<kname>.torch.json`` (a standalone, runnable sub-graph of whole
Torch ops) + ``<kname>.torch.txt`` (an ``i/N`` coverage summary).
"""

import json

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import RmsNormOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.dump import CompilerDump


def _rms_graph() -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, 4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("w", (8,)), node_id="w")
    g.add_node(RmsNormOp(), ["x", "w"], Tensor("rms_norm_0", (1, 4, 8)), node_id="rms_norm_0")
    g.inputs, g.outputs = ["x", "w"], ["rms_norm_0"]
    return g


def test_torch_repro_is_whole_op_and_loadable(tmp_path):
    g = _rms_graph()
    dump = CompilerDump(dir=tmp_path)
    dump.dump_input_graph(g)
    Pipeline.build(TILE_PASSES, dump=dump).run(g)

    repros = sorted(tmp_path.glob("*.kernels/*.torch.json"))
    assert repros, "expected a per-kernel .torch.json reproducer"

    # Sliced from the pristine graph → contains the whole original RmsNormOp,
    # not its decomposed primitives.
    sub = Graph.from_dict(json.loads(repros[-1].read_text()))
    kinds = {type(n.op).__name__ for n in sub.nodes.values()}
    assert "RmsNormOp" in kinds
    assert sub.outputs == ["rms_norm_0"]
    assert set(sub.inputs) == {"x", "w"}  # the rms_norm's own inputs become boundaries


def test_torch_repro_coverage_header(tmp_path):
    g = _rms_graph()
    dump = CompilerDump(dir=tmp_path)
    dump.dump_input_graph(g)
    Pipeline.build(TILE_PASSES, dump=dump).run(g)

    txts = sorted(tmp_path.glob("*.kernels/*.torch.txt"))
    assert txts
    body = txts[-1].read_text()
    assert "rms_norm_0 (RmsNormOp):" in body
    # rms_norm fully fuses at this shape → full coverage.
    assert "— full" in body


def test_no_repro_without_input_graph(tmp_path):
    """A dump that never captured the input graph writes no reproducers."""
    g = _rms_graph()
    dump = CompilerDump(dir=tmp_path)  # no dump_input_graph call
    Pipeline.build(TILE_PASSES, dump=dump).run(g)
    assert not list(tmp_path.glob("*.kernels/*.torch.json"))
