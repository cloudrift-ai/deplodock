"""Tests for fusion rules (assemble_kernels).

The fusion rule produces LoopOp nodes with SSA body statements
(``Assign``, ``Update``, ``Write``). Tests verify structural properties:
kernel count, graph composition (only LoopOp/InputOp/ConstantOp remain),
and that the SSA body contains the expected ops.
"""

from pathlib import Path

from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Assign, LoopOp, Port, Update, Write
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.rewriter import Pass, Rule

RULES_DIR = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion"

_RULE = "001_assemble_kernels.py"


def _load() -> Rule:
    return Rule.from_file(RULES_DIR / _RULE)


def _fuse(graph: Graph) -> Graph:
    return Pass(name="fusion", rules=[_load()]).apply(graph)


def _kernel_nodes(graph: Graph) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, LoopOp)]


def _assign_fns(body) -> list[str]:
    return [s.op.fn for s in body if isinstance(s, Assign)]


def _has_update(body) -> bool:
    return any(isinstance(s, Update) for s in body)


def _local_combine_fns(locals_) -> set[str]:
    return {lb.combine.fn for lb in locals_ if lb.combine is not None}


# ===================================================================
# Pointwise chain: neg → exp → single kernel
# ===================================================================


def _make_pointwise_chain():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("neg"), ["x"], Tensor("n", (8,)), node_id="n")
    g.add_node(ElementwiseOp("exp"), ["n"], Tensor("y", (8,)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    return g


def test_pointwise_chain_fuses_to_one_kernel():
    result = _fuse(_make_pointwise_chain())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_pointwise_chain_only_kernel_input_constant():
    result = _fuse(_make_pointwise_chain())
    for n in result.nodes.values():
        assert isinstance(n.op, (LoopOp, InputOp, ConstantOp))


def test_pointwise_chain_body_ops():
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    body_ops = _assign_fns(kernel.op.body)
    assert "neg" in body_ops
    assert "exp" in body_ops


def test_pointwise_chain_inputs_are_ports():
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert all(isinstance(inp, Port) for inp in kernel.op.inputs)


def test_pointwise_chain_has_write():
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert any(isinstance(s, Write) for s in kernel.op.body)


# ===================================================================
# Reduce chain: mul → reduce_sum (contraction)
# ===================================================================


def _make_contraction():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    g.add_node(ElementwiseOp("mul"), ["a", "b"], Tensor("m", (4, 8)), node_id="m")
    g.add_node(ReduceOp("sum", -1), ["m"], Tensor("y", (4,)), node_id="y")
    g.inputs, g.outputs = ["a", "b"], ["y"]
    return g


def test_contraction_fuses_to_one_kernel():
    result = _fuse(_make_contraction())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_contraction_body_has_mul_and_sum():
    result = _fuse(_make_contraction())
    kernel = _kernel_nodes(result)[0]
    assert "mul" in _assign_fns(kernel.op.body)
    assert _has_update(kernel.op.body)
    assert "add" in _local_combine_fns(kernel.op.locals)


# ===================================================================
# Contraction + epilogue: mul → sum → add(bias)
# ===================================================================


def _make_contraction_with_epilogue():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    g.add_node(InputOp(), [], Tensor("bias", (4,)), node_id="bias")
    g.add_node(ElementwiseOp("mul"), ["a", "b"], Tensor("m", (4, 8)), node_id="m")
    g.add_node(ReduceOp("sum", -1), ["m"], Tensor("s", (4,)), node_id="s")
    g.add_node(ElementwiseOp("add"), ["s", "bias"], Tensor("y", (4,)), node_id="y")
    g.inputs, g.outputs = ["a", "b", "bias"], ["y"]
    return g


def test_contraction_epilogue_fuses_to_one_kernel():
    result = _fuse(_make_contraction_with_epilogue())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_contraction_epilogue_body_has_add():
    result = _fuse(_make_contraction_with_epilogue())
    kernel = _kernel_nodes(result)[0]
    assert "add" in _assign_fns(kernel.op.body)


# ===================================================================
# Softmax: reduce_max → sub → exp → reduce_sum → div
# ===================================================================


def _make_softmax():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("max", -1), ["x"], Tensor("mx", (4, 1)), node_id="mx")
    g.add_node(ElementwiseOp("sub"), ["x", "mx"], Tensor("sub", (4, 8)), node_id="sub")
    g.add_node(ElementwiseOp("exp"), ["sub"], Tensor("exp", (4, 8)), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (4, 1)), node_id="sm")
    g.add_node(ElementwiseOp("div"), ["exp", "sm"], Tensor("out", (4, 8)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_softmax_fuses():
    result = _fuse(_make_softmax())
    kernels = _kernel_nodes(result)
    assert len(kernels) >= 1


def test_softmax_only_kernel_input_constant():
    result = _fuse(_make_softmax())
    for n in result.nodes.values():
        assert isinstance(n.op, (LoopOp, InputOp, ConstantOp))


def test_softmax_body_covers_all_ops():
    result = _fuse(_make_softmax())
    all_fns = set()
    for k in _kernel_nodes(result):
        all_fns |= set(_assign_fns(k.op.body))
        all_fns |= _local_combine_fns(k.op.locals)
    # Expect elementwise sub/exp/div and reduce combine add/max from the
    # max and sum accumulators.
    assert {"sub", "exp", "div"} <= all_fns
    assert {"add", "max"} <= all_fns


# ===================================================================
# Single elementwise: identity case
# ===================================================================


def test_single_elementwise_fuses():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("neg"), ["x"], Tensor("y", (8,)), node_id="y")
    g.inputs, g.outputs = ["x"], ["y"]
    result = _fuse(g)
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


# ===================================================================
# SSA invariants: unique names, defined-before-use
# ===================================================================


def test_ssa_invariants_hold():
    """LoopOp.__post_init__ validates SSA; this just confirms no crash."""
    result = _fuse(_make_softmax())
    for k in _kernel_nodes(result):
        # Re-validate explicitly
        defined = set()
        port_idx = 0
        for inp in k.op.inputs:
            if isinstance(inp, Port):
                defined.add(f"${port_idx}")
                port_idx += 1
        for lb in k.op.locals:
            defined.add(lb.name)
        for s in k.op.body:
            if isinstance(s, Assign):
                for arg in s.args:
                    assert arg in defined, f"arg {arg!r} not defined before use in {s.name}"
                defined.add(s.name)
            elif isinstance(s, Update):
                assert s.target in defined
                assert s.value in defined
            elif isinstance(s, Write):
                assert s.value in defined
