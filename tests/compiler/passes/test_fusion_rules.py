"""Tests for the fusion pass (lift-then-splice).

The fusion pass lifts each tensor op into a trivial ``LoopOp`` and then
splices adjacent ``LoopOp`` pairs via the tree-splicer in
``passes/fusion/_splice.py``. Tests verify post-fixpoint structural
properties (kernel count, graph composition, expected ops in SSA bodies)
*and* numeric correctness — each fixture is executed via ``NumpyBackend``
both pre- and post-fusion, and the outputs must match. ``LoopOp.forward``
makes the post-fusion run possible without a GPU.
"""

import numpy as np

from deplodock.compiler.backend.numpy import NumpyBackend
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import ConstantOp, InputOp
from deplodock.compiler.ir.loop import Accum, Assign, LoopOp, Write
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp
from deplodock.compiler.pipeline import Pipeline

rng = np.random.default_rng(0)
_backend = NumpyBackend()


def _fuse(graph: Graph) -> Graph:
    return Pipeline.build(["loop/lifting", "loop/fusion"]).run(graph)


def _run(graph: Graph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return _backend.run(_backend.compile(graph), input_data=inputs)[0].outputs


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
    return [s.op.name for s in body.iter() if isinstance(s, Assign)]


def _count_copies(body) -> int:
    """Count identity ``Assign(op=copy)`` statements in a LoopOp body."""
    return sum(1 for s in body.iter() if isinstance(s, Assign) and s.op.name == "copy")


def _has_update(body) -> bool:
    return any(isinstance(s, Accum) for s in body.iter())


def _local_combine_fns(locals_) -> set[str]:
    return {lb.op.name for lb in locals_ if lb.op is not None}


# ===================================================================
# Pointwise chain: neg → exp → single kernel
# ===================================================================


def _make_pointwise_chain():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("negative"), ["x"], Tensor("n", (8,)), node_id="n")
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
    assert "negative" in body_ops
    assert "exp" in body_ops


def test_pointwise_chain_inputs_are_loads():
    from deplodock.compiler.ir.loop import Load

    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    loads = kernel.op.body.loads
    assert len(loads) >= 1
    assert all(isinstance(ld, Load) for ld in loads)


def test_pointwise_chain_has_write():
    result = _fuse(_make_pointwise_chain())
    kernel = _kernel_nodes(result)[0]
    assert any(isinstance(s, Write) for s in kernel.op)


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
    g.add_node(ElementwiseOp("multiply"), ["x", "x"], Tensor("sq", (4, 8)), node_id="sq")
    g.add_node(ReduceOp("sum", -1), ["sq"], Tensor("red", (4, 1)), node_id="red")
    g.add_node(ElementwiseOp("multiply"), ["red", "red"], Tensor("sqr", (4, 1)), node_id="sqr")
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
    from deplodock.compiler.ir.loop import Select

    result = _fuse(_make_rms_norm_like())
    kernel = _kernel_nodes(result)[0]
    ssa_names = [s.name for s in kernel.op if isinstance(s, (Assign, Select))]
    assert ssa_names == [f"v{i}" for i in range(len(ssa_names))], f"unexpected SSA names: {ssa_names}"


# ===================================================================
# Reduce chain: mul → reduce_sum (contraction)
# ===================================================================


def _make_contraction():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    g.add_node(ElementwiseOp("multiply"), ["a", "b"], Tensor("m", (4, 8)), node_id="m")
    g.add_node(ReduceOp("sum", -1), ["m"], Tensor("y", (4, 1)), node_id="y")
    g.inputs, g.outputs = ["a", "b"], ["y"]
    return g


def test_contraction_fuses_to_one_kernel():
    result = _fuse(_make_contraction())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1


def test_contraction_body_has_mul_and_sum():
    result = _fuse(_make_contraction())
    kernel = _kernel_nodes(result)[0]
    assert "multiply" in _assign_fns(kernel.op.body)
    assert _has_update(kernel.op.body)
    assert "add" in _local_combine_fns(kernel.op.body.accums)


# ===================================================================
# Contraction + epilogue: mul → sum → add(bias)
# ===================================================================


def _make_contraction_with_epilogue():
    from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (4, 8)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (4, 8)), node_id="b")
    g.add_node(InputOp(), [], Tensor("bias", (4, 1)), node_id="bias")
    g.add_node(ElementwiseOp("multiply"), ["a", "b"], Tensor("m", (4, 8)), node_id="m")
    g.add_node(ReduceOp("sum", -1), ["m"], Tensor("s", (4, 1)), node_id="s")
    g.add_node(ElementwiseOp("add"), ["s", broadcast_to(g, "bias", (4, 1))], Tensor("y", (4, 1)), node_id="y")
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
    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("mx", (4, 1)), node_id="mx")
    g.add_node(ElementwiseOp("subtract"), ["x", "mx"], Tensor("subtract", (4, 8)), node_id="subtract")
    g.add_node(ElementwiseOp("exp"), ["subtract"], Tensor("exp", (4, 8)), node_id="exp")
    g.add_node(ReduceOp("sum", -1), ["exp"], Tensor("sm", (4, 1)), node_id="sm")
    g.add_node(ElementwiseOp("divide"), ["exp", "sm"], Tensor("out", (4, 8)), node_id="out")
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
        all_fns |= _local_combine_fns(k.op.body.accums)
    # Expect elementwise sub/exp and reduce combine add/max from the
    # max and sum accumulators. ``divide(x, acc_sum)`` is split by
    # ``split_invariant_divides`` (in ``ir/stmt/normalize.py``) into
    # ``reciprocal(acc_sum) + multiply(x, recip)`` so the rcp can hoist
    # out of the inner reduce — divide no longer appears as a body op.
    assert {"subtract", "exp", "reciprocal", "multiply"} <= all_fns
    assert {"add", "maximum"} <= all_fns


# ===================================================================
# Single elementwise: identity case
# ===================================================================


def test_single_elementwise_fuses():
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
    g.add_node(ElementwiseOp("negative"), ["x"], Tensor("y", (8,)), node_id="y")
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
        for decl in k.op.body.accums:
            defined.add(decl.name)
        from deplodock.compiler.ir.loop import Accum, Load

        for s in k.op:
            if isinstance(s, Assign):
                for arg in s.args:
                    assert arg in defined, f"arg {arg!r} not defined before use in {s.name}"
                defined.add(s.name)
            elif isinstance(s, (Load, Accum)):
                defined.add(s.name)
            elif isinstance(s, Accum):
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
    bias = rng.standard_normal((4, 1)).astype(np.float32)
    _assert_correctness(_make_contraction_with_epilogue, {"a": a, "b": b, "bias": bias})


def test_softmax_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_softmax, {"x": x})


def test_single_elementwise_correctness():
    def _make():
        g = Graph()
        g.add_node(InputOp(), [], Tensor("x", (8,)), node_id="x")
        g.add_node(ElementwiseOp("negative"), ["x"], Tensor("y", (8,)), node_id="y")
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
    g.add_node(ReduceOp("sum", -1), ["x"], Tensor("s", (4, 1)), node_id="s")
    g.add_node(ReduceOp("maximum", -1), ["x"], Tensor("m", (4, 1)), node_id="m")
    g.add_node(ElementwiseOp("add"), ["s", "m"], Tensor("out", (4, 1)), node_id="out")
    g.inputs, g.outputs = ["x"], ["out"]
    return g


def test_sibling_reductions_fuse_to_one_kernel():
    result = _fuse(_make_sibling_reductions())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 1, f"expected 1 kernel, got {len(kernels)}"


def test_sibling_reductions_share_reduce_axis():
    result = _fuse(_make_sibling_reductions())
    kernel = _kernel_nodes(result)[0]
    reduce_axes = [a for a in kernel.op.axes if a.name in kernel.op.reduce_axis_names]
    assert len(reduce_axes) == 1, f"expected 1 reduce axis, got {[a.name for a in reduce_axes]}"


def test_sibling_reductions_have_both_accumulators():
    result = _fuse(_make_sibling_reductions())
    kernel = _kernel_nodes(result)[0]
    combine_fns = _local_combine_fns(kernel.op.body.accums)
    assert {"add", "maximum"} <= combine_fns


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
    reduce_axes = [a for a in kernel.op.axes if a.name in kernel.op.reduce_axis_names]
    assert len(reduce_axes) == 1


def test_softmax_has_both_accumulators():
    result = _fuse(_make_softmax())
    kernel = _kernel_nodes(result)[0]
    combine_fns = _local_combine_fns(kernel.op.body.accums)
    assert {"add", "maximum"} <= combine_fns, f"missing accumulators: {combine_fns}"


def test_softmax_single_kernel_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_softmax, {"x": x})


# ===================================================================
# Split shared (multi-consumer) index-map fan-out (005) so merge inlines it.
#
# A pure-indexmap LoopOp (broadcast / transpose) that fans out to ≥2
# consumers is never absorbed by ``merge_loop_ops`` (the match-walker only
# extends single-consumer edges). ``005_split_shared_indexmap`` peels each
# consumer onto a private copy so every edge becomes single-consumer and
# merge folds them — leaving no pure-indexmap copy kernel behind.
# ===================================================================


def _is_pure_indexmap(op: LoopOp) -> bool:
    return not any(isinstance(s, (Assign, Accum)) for s in op.body.iter())


def _pure_indexmap_kernels(graph: Graph) -> list:
    return [n for n in _kernel_nodes(graph) if _is_pure_indexmap(n.op)]


def _loads_from(op: LoopOp) -> set[str]:
    return {ld.input for ld in op.body.loads}


def _split_only(graph: Graph) -> Graph:
    """Run lifting + the split rule alone (no merge) — the in-isolation view."""
    return Pipeline.build(
        ["loop/lifting", "loop/fusion"],
        select={"lift_elementwise", "lift_reduce", "lift_indexmap", "lift_gather", "split_shared_indexmap"},
    ).run(graph)


def _make_shared_const_broadcast():
    """A scalar ``const_bc(1.0)`` broadcast feeding two elementwise consumers —
    the headline case: torch.export folds attention-mask / RoPE scaffolding to
    scalar broadcasts that fan out (Qwen3 GQA query + key paths)."""
    from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    g.add_node(InputOp(), [], Tensor("y", (4, 8)), node_id="y")
    bc = const_bc(g, name="one", value=1.0, target_shape=(4, 8), dtype="f32")
    g.add_node(ElementwiseOp("multiply"), ["x", bc.id], Tensor("c1", (4, 8)), node_id="c1")
    g.add_node(ElementwiseOp("add"), ["y", bc.id], Tensor("c2", (4, 8)), node_id="c2")
    g.inputs, g.outputs = ["x", "y"], ["c1", "c2"]
    return g


def test_shared_const_broadcast_no_pure_indexmap_remains():
    """After fusion the shared constant broadcast is gone — no pure-indexmap copy kernel."""
    result = _fuse(_make_shared_const_broadcast())
    assert _pure_indexmap_kernels(result) == []


def test_shared_const_broadcast_consumers_load_constant_directly():
    """Both consumers Load the ``ConstantOp`` (``one``) directly — the broadcast
    folded in, so the cuda literal-inline path can stamp ``float x = 1.0f;``."""
    result = _fuse(_make_shared_const_broadcast())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2, f"expected 2 consumer kernels, got {len(kernels)}"
    for k in kernels:
        assert "one" in _loads_from(k.op), f"{k.id} does not load the constant directly: {_loads_from(k.op)}"


def test_shared_const_broadcast_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    y = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_shared_const_broadcast, {"x": x, "y": y})


def test_shared_const_broadcast_split_in_isolation():
    """Split rule alone (no merge): the producer is fused into *all* consumers in
    one shot — no broadcast kernel survives and each consumer reads the constant
    directly, without ``merge_loop_ops`` running at all."""
    result = _split_only(_make_shared_const_broadcast())
    assert _pure_indexmap_kernels(result) == [], "the shared broadcast should be fully dissolved by the split rule alone"
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2, f"expected 2 fused consumer kernels, got {len(kernels)}"
    for k in kernels:
        assert "one" in _loads_from(k.op), f"{k.id} does not load the constant directly: {_loads_from(k.op)}"


def _make_shared_transpose():
    """A transpose (general layout op) feeding two elementwise consumers —
    covers the non-constant pure-indexmap path."""
    from deplodock.compiler.ir.expr import placeholder
    from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import single_indexmap

    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (4, 8)), node_id="x")
    # out[i, j] = x[j, i] — transpose (4, 8) -> (8, 4).
    t = single_indexmap(g, "x", out_shape=(8, 4), coord_map=(placeholder(1), placeholder(0)), name="xt")
    g.add_node(ElementwiseOp("exp"), [t.id], Tensor("c1", (8, 4)), node_id="c1")
    g.add_node(ElementwiseOp("negative"), [t.id], Tensor("c2", (8, 4)), node_id="c2")
    g.inputs, g.outputs = ["x"], ["c1", "c2"]
    return g


def test_shared_transpose_no_pure_indexmap_remains():
    result = _fuse(_make_shared_transpose())
    assert _pure_indexmap_kernels(result) == []


def test_shared_transpose_consumers_index_source_directly():
    """Both consumers read the transpose's source (``x``) directly — the layout
    op folded into each as lazy per-consumer indexing."""
    result = _fuse(_make_shared_transpose())
    kernels = _kernel_nodes(result)
    assert len(kernels) == 2, f"expected 2 consumer kernels, got {len(kernels)}"
    for k in kernels:
        assert "x" in _loads_from(k.op), f"{k.id} does not load the source directly: {_loads_from(k.op)}"


def test_shared_transpose_correctness():
    x = rng.standard_normal((4, 8)).astype(np.float32)
    _assert_correctness(_make_shared_transpose, {"x": x})


def _make_shared_broadcast_chain():
    """A scalar broadcast (shared) feeding two *further* broadcasts that each feed
    a downstream multiply — mirrors the full-model RoPE/mask shape:
    ``scalar -> [1,1,4,8] (shared) -> {[1,3,4,8], [1,2,4,8]} -> multiply``.

    The intermediate broadcasts are the regression trigger: lifting gives each a
    node id (``lift_<id>``) that differs from its output Tensor name, so when the
    split rule peels one onto a private copy it MUST rename the new node's
    ``Write.output`` to its new id. Forgetting that leaves the node writing the
    old buf — ``splice_graph`` (which assumes Write.output == node id) then can't
    fold it into the downstream multiply, and a pure-indexmap copy survives."""
    from deplodock.compiler.pipeline.passes.frontend.decomposition._broadcast import broadcast_to
    from deplodock.compiler.pipeline.passes.frontend.decomposition._helpers import const_bc

    g = Graph()
    g.add_node(InputOp(), [], Tensor("mq", (1, 3, 4, 8)), node_id="mq")
    g.add_node(InputOp(), [], Tensor("mk", (1, 2, 4, 8)), node_id="mk")
    shared = const_bc(g, name="zero", value=0.0, target_shape=(1, 1, 4, 8), dtype="f32")
    bq = broadcast_to(g, shared.id, (1, 3, 4, 8))
    bk = broadcast_to(g, shared.id, (1, 2, 4, 8))
    g.add_node(ElementwiseOp("multiply"), ["mq", bq.id], Tensor("fq", (1, 3, 4, 8)), node_id="fq")
    g.add_node(ElementwiseOp("multiply"), ["mk", bk.id], Tensor("fk", (1, 2, 4, 8)), node_id="fk")
    g.inputs, g.outputs = ["mq", "mk"], ["fq", "fk"]
    return g


def test_shared_broadcast_chain_no_pure_indexmap_remains():
    """Regression: the intermediate broadcasts (node id != output name) must fully
    fold — no pure-indexmap copy left writing a stale buf."""
    result = _fuse(_make_shared_broadcast_chain())
    assert _pure_indexmap_kernels(result) == []


def test_shared_broadcast_chain_correctness():
    mq = rng.standard_normal((1, 3, 4, 8)).astype(np.float32)
    mk = rng.standard_normal((1, 2, 4, 8)).astype(np.float32)
    _assert_correctness(_make_shared_broadcast_chain, {"mq": mq, "mk": mk})
