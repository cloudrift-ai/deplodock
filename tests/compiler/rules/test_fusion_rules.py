"""Tests for the fusion pass (lift-then-merge).

The fusion pass lifts each tensor op into a trivial ``LoopOp`` and then
merges adjacent ``LoopOp`` pairs via the σ-based merge rule. Tests verify
post-fixpoint structural properties (kernel count, graph composition,
expected ops in SSA bodies) *and* numeric correctness — each fixture is
executed via ``NumpyBackend`` both pre- and post-fusion, and the outputs
must match. ``LoopOp.forward`` makes the post-fusion run possible without
a GPU.
"""

from pathlib import Path

import numpy as np

from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.loop import Assign, LoopOp, Port, Update, Write
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.rewriter import Pass

RULES_DIR = Path(__file__).parent.parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion"

rng = np.random.default_rng(0)
_backend = NumpyBackend()


def _fuse(graph: Graph) -> Graph:
    return Pass.from_directory(RULES_DIR).apply(graph)


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs).outputs


def _assert_close(before: dict, after: dict, *, rtol=1e-5, atol=1e-5):
    bvals = list(before.values())
    avals = list(after.values())
    assert len(bvals) == len(avals), f"output count mismatch: {len(bvals)} vs {len(avals)}"
    for i, (b, a) in enumerate(zip(bvals, avals, strict=True)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=f"output[{i}]")


def _assert_correctness(make_graph, inputs):
    """Run the pre- and post-fusion graph through NumpyBackend; assert outputs match.

    Exercises fusion rules for *semantic* equivalence on top of the
    structural checks in this file — ``LoopOp.forward`` executes the
    lifted+merged kernels numerically against the original tensor-IR
    evaluation.
    """
    g_before = make_graph()
    g_after = _fuse(make_graph())
    before = _run(g_before, inputs)
    after = _run(g_after, inputs)
    _assert_close(before, after)


def _kernel_nodes(graph: Graph) -> list:
    return [n for n in graph.nodes.values() if isinstance(n.op, LoopOp)]


def _assign_fns(body) -> list[str]:
    from deplodock.compiler.ir.loop import flatten_body

    return [s.op.fn for s in flatten_body(body) if isinstance(s, Assign)]


def _count_copies(body) -> int:
    """Count identity ``Assign(op=copy)`` statements in a LoopOp body."""
    from deplodock.compiler.ir.loop import flatten_body

    return sum(1 for s in flatten_body(body) if isinstance(s, Assign) and s.op.fn == "copy")


def _has_update(body) -> bool:
    from deplodock.compiler.ir.loop import flatten_body

    return any(isinstance(s, Update) for s in flatten_body(body))


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
    from deplodock.compiler.ir.loop import flatten_body

    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert any(isinstance(s, Write) for s in flatten_body(kernel.op.body))


def test_pointwise_chain_no_residual_copies():
    """After fusion + copy-elimination, no identity ``copy`` Assigns remain."""
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert _count_copies(kernel.op.body) == 0


# ===================================================================
# Copy elimination: transitive alias chain + port-ref rewriting
# ===================================================================


def _make_rms_norm_like():
    """A 4-op fused graph (mul → reduce_sum → div → mul-by-weight) that
    produces many bridge copies during merge — the realistic target of the
    copy-elimination pass."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ConstantOp(name="w"), [], Tensor("w", (4, 8)), node_id="w")
    g.add_node(ElementwiseOp("mul"), ["x", "x"], Tensor("sq", (4, 8)), node_id="sq")
    g.add_node(ReduceOp("sum", -1), ["sq"], Tensor("red", (4,)), node_id="red")
    g.add_node(ElementwiseOp("mul"), ["red", "red"], Tensor("sqr", (4,)), node_id="sqr")
    g.inputs, g.outputs = ["x", "w"], ["sqr"]
    return g


def test_rms_norm_like_no_residual_copies():
    result = _fuse(_make_rms_norm_like())
    kernel = _kernel_nodes(result)[0]
    assert _count_copies(kernel.op.body) == 0, "copy-elimination must clear all bridge copies"


def test_rms_norm_like_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    w = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_rms_norm_like, {"x": x, "w": w})


def test_rms_norm_like_ssa_names_are_canonical():
    """After rename pass, every SSA name in the body is v0, v1, v2, ... in order."""
    from deplodock.compiler.ir.loop import Select, flatten_body

    result = _fuse(_make_rms_norm_like())
    kernel = _kernel_nodes(result)[0]
    ssa_names = [s.name for s in flatten_body(kernel.op.body) if isinstance(s, (Assign, Select))]
    assert ssa_names == [f"v{i}" for i in range(len(ssa_names))], f"unexpected SSA names: {ssa_names}"


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
    from deplodock.compiler.ir.loop import flatten_body

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
        for s in flatten_body(k.op.body):
            if isinstance(s, Assign):
                for arg in s.args:
                    assert arg in defined, f"arg {arg!r} not defined before use in {s.name}"
                defined.add(s.name)
            elif isinstance(s, Update):
                assert s.target in defined
                assert s.value in defined
            elif isinstance(s, Write):
                assert s.value in defined


# ===================================================================
# Correctness: pre-fusion vs post-fusion numeric equivalence via NumpyBackend.
# ===================================================================


def test_pointwise_chain_correctness():
    x = rng.standard_normal(8).astype(np.float32)
    _assert_correctness(_make_pointwise_chain, {"x": x})


def test_contraction_correctness():
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_contraction, {"a": a, "b": b})


def test_contraction_epilogue_correctness():
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    bias = rng.standard_normal(4).astype(np.float32)
    _assert_correctness(_make_contraction_with_epilogue, {"a": a, "b": b, "bias": bias})


def test_softmax_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_softmax, {"x": x})


def test_single_elementwise_correctness():
    def _make():
        g = Graph()
        g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
        g.add_node(ElementwiseOp("neg"), ["x"], Tensor("y", (8,)), node_id="y")
        g.inputs, g.outputs = ["x"], ["y"]
        return g

    x = rng.standard_normal(8).astype(np.float32)
    _assert_correctness(_make, {"x": x})


# ===================================================================
# Sibling reductions: axis aliasing collapses them into one kernel.
# ===================================================================


def _make_sibling_reductions():
    """``s = sum(x, -1); m = max(x, -1); out = s + m`` — two reduces over x
    feeding one elementwise. With reduce-axis aliasing, fuses into one kernel
    with two accumulators sharing one reduce axis."""
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("s", (4,)), node_id="s")
    g.add_node(ReduceOp("max", -1), ["x"], Tensor("m", (4,)), node_id="m")
    g.add_node(ElementwiseOp("add"), ["s", "m"], Tensor("out", (4,)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_sibling_reductions_fuse_to_one_kernel():
    result = _fuse(_make_sibling_reductions())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1, f"expected 1 kernel, got {len(kernels)}"


def test_sibling_reductions_share_reduce_axis():
    result = _fuse(_make_sibling_reductions())
    kernel = _kernel_nodes(result)[0]
    reduce_axes = [a for a in kernel.op.axes if a.kind == "reduce"]
    assert len(reduce_axes) == 1, f"expected 1 reduce axis, got {[a.name for a in reduce_axes]}"


def test_sibling_reductions_have_both_accumulators():
    result = _fuse(_make_sibling_reductions())
    kernel = _kernel_nodes(result)[0]
    combine_fns = _local_combine_fns(kernel.op.locals)
    assert {"add", "max"} <= combine_fns


def test_sibling_reductions_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_sibling_reductions, {"x": x})


# ===================================================================
# Softmax: multi-port producer consumption + reduce-axis aliasing.
# ===================================================================


def test_softmax_fuses_to_one_kernel():
    """Softmax's two-reduce pattern (max sweep → sub → exp → sum sweep → div)
    fuses into a single kernel with two accumulators sharing one reduce axis."""
    result = _fuse(_make_softmax())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1, f"expected 1 kernel, got {len(kernels)}"


def test_softmax_single_reduce_axis():
    result = _fuse(_make_softmax())
    kernel = _kernel_nodes(result)[0]
    reduce_axes = [a for a in kernel.op.axes if a.kind == "reduce"]
    assert len(reduce_axes) == 1


def test_softmax_has_both_accumulators():
    result = _fuse(_make_softmax())
    kernel = _kernel_nodes(result)[0]
    combine_fns = _local_combine_fns(kernel.op.locals)
    assert {"add", "max"} <= combine_fns, f"missing accumulators: {combine_fns}"


def test_softmax_single_kernel_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_softmax, {"x": x})
