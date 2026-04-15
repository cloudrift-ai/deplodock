"""Tests for the 055_merge_kernels rule.

Positive cases:
  - Two adjacent pointwise KernelOps merge into one.
  - Pointwise + reduce-kernel → reduce-kernel with prologue.
  - Reduce-kernel + pointwise → reduce-kernel with epilogue.
  - Contraction + pointwise → contraction with epilogue.

Negative cases:
  - Pointwise + contraction must NOT merge (defer to 080_absorb_a_chain).
  - Two contractions never merge.
  - Reduce kernels with incompatible axes don't merge.
  - Shape mismatch (matmul-output-sized add into a non-broadcast-compatible
    pointwise kernel) is rejected.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deplodock.compiler.ir import Graph, Node, Tensor
from deplodock.compiler.matcher import match_pattern
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    InputOp,
    KernelOp,
    Port,
    ReduceOp,
    ReduceStage,
)
from deplodock.compiler.pattern import parse_pattern

RULES_DIR = Path(__file__).parent.parent.parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion"


def _load_rule(stem: str):
    path = RULES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_rule(graph: Graph, rule_stem: str) -> Graph:
    rule = _load_rule(rule_stem)
    pattern = parse_pattern(rule.PATTERN)
    changed = True
    while changed:
        changed = False
        for match in match_pattern(graph, pattern):
            new_graph = rule.rewrite(graph, match)
            if new_graph is not graph:
                graph = new_graph
                changed = True
                break
    return graph


def _kernels(g: Graph) -> list[KernelOp]:
    return [n.op for n in g.nodes.values() if isinstance(n.op, KernelOp)]


# ---------------------------------------------------------------------------
# Positive cases
# ---------------------------------------------------------------------------


def test_pointwise_plus_pointwise_merges():
    """Two adjacent pointwise kernels become one."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 16)), node_id="x")
    y = g.add_node(InputOp(), [], Tensor("y", (4, 16)), node_id="y")
    g.inputs = [x, y]

    add_node = Node(id="add", op=ElementwiseOp("add"), inputs=["x", "y"], output=Tensor("add", (4, 16)))
    k_a = KernelOp(
        inputs=[Port("x"), Port("y")],
        outputs=[Port("add")],
        prologue=(add_node,),
        core=None,
        external_shapes={"x": (4, 16), "y": (4, 16)},
    )
    k_a_id = g.add_node(op=k_a, inputs=[x, y], output=Tensor("a_out", (4, 16)), node_id="a_out")

    relu_node = Node(id="relu", op=ElementwiseOp("relu"), inputs=["a_out"], output=Tensor("relu", (4, 16)))
    k_b = KernelOp(
        inputs=[Port("a_out")],
        outputs=[Port("relu")],
        prologue=(relu_node,),
        core=None,
        external_shapes={"a_out": (4, 16)},
    )
    k_b_id = g.add_node(op=k_b, inputs=[k_a_id], output=Tensor("b_out", (4, 16)), node_id="b_out")
    g.outputs = [k_b_id]

    result = _apply_rule(g, "055_merge_kernels")

    ks = _kernels(result)
    assert len(ks) == 1
    k = ks[0]
    assert k.core is None
    assert len(k.prologue) == 2
    fns = [n.op.fn for n in k.prologue]
    assert fns == ["add", "relu"]
    # Relu's input must reference the actual add Node id, not the old k_a outer id.
    assert k.prologue[1].inputs == ["add"]


def test_pointwise_plus_reduce_merges():
    """Pointwise → reduce kernel: pointwise body becomes pre_ops of first stage."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 16)), node_id="x")
    g.inputs = [x]

    sq = Node(id="sq", op=ElementwiseOp("mul"), inputs=["x", "x"], output=Tensor("sq", (4, 16)))
    k_a = KernelOp(
        inputs=[Port("x")],
        outputs=[Port("sq")],
        prologue=(sq,),
        core=None,
        external_shapes={"x": (4, 16)},
    )
    k_a_id = g.add_node(op=k_a, inputs=[x], output=Tensor("a_out", (4, 16)), node_id="a_out")

    red = Node(id="red", op=ReduceOp("sum", -1), inputs=["a_out"], output=Tensor("red", (4,)))
    k_b = KernelOp(
        inputs=[Port("a_out")],
        outputs=[Port("red")],
        core=(ReduceStage(pre_ops=(), reduce=red),),
        external_shapes={"a_out": (4, 16)},
    )
    k_b_id = g.add_node(op=k_b, inputs=[k_a_id], output=Tensor("b_out", (4,)), node_id="b_out")
    g.outputs = [k_b_id]

    result = _apply_rule(g, "055_merge_kernels")

    ks = _kernels(result)
    assert len(ks) == 1
    k = ks[0]
    assert isinstance(k.core, tuple) and len(k.core) == 1
    stage = k.core[0]
    # sq moved into the stage's pre_ops; reduce's input is the sq id.
    assert any(n.id == "sq" for n in stage.pre_ops)
    assert stage.reduce.inputs == ["sq"]


def test_reduce_plus_pointwise_merges():
    """Reduce → pointwise: pointwise body becomes epilogue."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 16)), node_id="x")
    g.inputs = [x]

    red = Node(id="red", op=ReduceOp("sum", -1), inputs=["x"], output=Tensor("red", (4,)))
    k_a = KernelOp(
        inputs=[Port("x")],
        outputs=[Port("red")],
        core=(ReduceStage(pre_ops=(), reduce=red),),
        external_shapes={"x": (4, 16)},
    )
    k_a_id = g.add_node(op=k_a, inputs=[x], output=Tensor("a_out", (4,)), node_id="a_out")

    rsqrt = Node(id="rsqrt", op=ElementwiseOp("rsqrt"), inputs=["a_out"], output=Tensor("rsqrt", (4,)))
    k_b = KernelOp(
        inputs=[Port("a_out")],
        outputs=[Port("rsqrt")],
        prologue=(rsqrt,),
        core=None,
        external_shapes={"a_out": (4,)},
    )
    k_b_id = g.add_node(op=k_b, inputs=[k_a_id], output=Tensor("b_out", (4,)), node_id="b_out")
    g.outputs = [k_b_id]

    result = _apply_rule(g, "055_merge_kernels")

    ks = _kernels(result)
    assert len(ks) == 1
    k = ks[0]
    assert isinstance(k.core, tuple) and len(k.core) == 1
    assert any(n.id == "rsqrt" for n in k.epilogue)
    # rsqrt's input rewired to reduce node's id.
    rsqrt_in_kernel = next(n for n in k.epilogue if n.id == "rsqrt")
    assert rsqrt_in_kernel.inputs == ["red"]


def test_contraction_plus_pointwise_merges():
    """Contraction → pointwise (bias add) becomes epilogue."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 16)), node_id="B")
    bias = g.add_node(InputOp(), [], Tensor("bias", (16,)), node_id="bias")
    g.inputs = [a, b, bias]

    mul = Node(id="mul", op=ElementwiseOp("mul"), inputs=["A", "B"], output=Tensor("mul", (4, 8, 16)))
    red = Node(id="red", op=ReduceOp("sum", 1), inputs=["mul"], output=Tensor("red", (4, 16)))
    core = ContractionCore(a=Port("A"), b=Port("B"), k_axis=1, mul=mul, reduce=red)
    k_a = KernelOp(
        inputs=[Port("A"), Port("B")],
        outputs=[Port("red")],
        core=core,
        external_shapes={"A": (4, 8), "B": (8, 16)},
    )
    k_a_id = g.add_node(op=k_a, inputs=[a, b], output=Tensor("a_out", (4, 16)), node_id="a_out")

    add = Node(id="add", op=ElementwiseOp("add"), inputs=["a_out", "bias"], output=Tensor("add", (4, 16)))
    k_b = KernelOp(
        inputs=[Port("a_out"), Port("bias")],
        outputs=[Port("add")],
        prologue=(add,),
        core=None,
        external_shapes={"a_out": (4, 16), "bias": (16,)},
    )
    k_b_id = g.add_node(op=k_b, inputs=[k_a_id, bias], output=Tensor("b_out", (4, 16)), node_id="b_out")
    g.outputs = [k_b_id]

    result = _apply_rule(g, "055_merge_kernels")

    ks = _kernels(result)
    assert len(ks) == 1
    k = ks[0]
    assert isinstance(k.core, ContractionCore)
    assert any(n.id == "add" for n in k.epilogue)
    add_n = next(n for n in k.epilogue if n.id == "add")
    # a_out → red rewire; bias unchanged.
    assert add_n.inputs == ["red", "bias"]
    # bias is now part of merged kernel's external inputs.
    assert any(p.buffer_id == "bias" for p in k.inputs)


def test_rmsnorm_chain_merges_to_one_kernel():
    """5 chained pointwise kernels collapse to one (RMSNorm body shape)."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 16)), node_id="x")
    g.inputs = [x]

    last = x
    last_shape = (4, 16)
    for i in range(5):
        nid = f"e{i}"
        ne = Node(id=nid, op=ElementwiseOp("relu"), inputs=[last], output=Tensor(nid, last_shape))
        kop = KernelOp(
            inputs=[Port(last)],
            outputs=[Port(nid)],
            prologue=(ne,),
            core=None,
            external_shapes={last: last_shape},
        )
        kid = g.add_node(op=kop, inputs=[last], output=Tensor(f"k{i}", last_shape), node_id=f"k{i}")
        last = kid
    g.outputs = [last]

    result = _apply_rule(g, "055_merge_kernels")

    ks = _kernels(result)
    assert len(ks) == 1
    k = ks[0]
    assert k.core is None
    assert len(k.prologue) == 5


# ---------------------------------------------------------------------------
# Negative cases
# ---------------------------------------------------------------------------


def test_two_contractions_not_merged():
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 16)), node_id="B")
    c = g.add_node(InputOp(), [], Tensor("C", (16, 32)), node_id="C")
    g.inputs = [a, b, c]

    # Build first contraction kernel.
    mul1 = Node(id="mul1", op=ElementwiseOp("mul"), inputs=["A", "B"], output=Tensor("mul1", (4, 8, 16)))
    red1 = Node(id="red1", op=ReduceOp("sum", 1), inputs=["mul1"], output=Tensor("red1", (4, 16)))
    k1 = KernelOp(
        inputs=[Port("A"), Port("B")],
        outputs=[Port("red1")],
        core=ContractionCore(a=Port("A"), b=Port("B"), k_axis=1, mul=mul1, reduce=red1),
        external_shapes={"A": (4, 8), "B": (8, 16)},
    )
    k1_id = g.add_node(op=k1, inputs=[a, b], output=Tensor("k1_out", (4, 16)), node_id="k1_out")

    # Build second contraction kernel reading k1_out × C.
    mul2 = Node(id="mul2", op=ElementwiseOp("mul"), inputs=["k1_out", "C"], output=Tensor("mul2", (4, 16, 32)))
    red2 = Node(id="red2", op=ReduceOp("sum", 1), inputs=["mul2"], output=Tensor("red2", (4, 32)))
    k2 = KernelOp(
        inputs=[Port("k1_out"), Port("C")],
        outputs=[Port("red2")],
        core=ContractionCore(a=Port("k1_out"), b=Port("C"), k_axis=1, mul=mul2, reduce=red2),
        external_shapes={"k1_out": (4, 16), "C": (16, 32)},
    )
    k2_id = g.add_node(op=k2, inputs=[k1_id, c], output=Tensor("k2_out", (4, 32)), node_id="k2_out")
    g.outputs = [k2_id]

    result = _apply_rule(g, "055_merge_kernels")

    # Both kernels survive — the rule must NOT merge contraction+contraction.
    assert len(_kernels(result)) == 2


def test_pointwise_plus_contraction_not_merged():
    """pointwise+contraction is reserved for 080_absorb_a_chain (today: deferred)."""
    g = Graph()
    x_raw = g.add_node(InputOp(), [], Tensor("X", (4, 8)), node_id="X")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 16)), node_id="B")
    g.inputs = [x_raw, b]

    # Pointwise kernel that just exposes X.
    relu = Node(id="relu", op=ElementwiseOp("relu"), inputs=["X"], output=Tensor("relu", (4, 8)))
    kp = KernelOp(
        inputs=[Port("X")],
        outputs=[Port("relu")],
        prologue=(relu,),
        core=None,
        external_shapes={"X": (4, 8)},
    )
    kp_id = g.add_node(op=kp, inputs=[x_raw], output=Tensor("kp_out", (4, 8)), node_id="kp_out")

    # Contraction kernel consuming kp_out.
    mul = Node(id="mul", op=ElementwiseOp("mul"), inputs=["kp_out", "B"], output=Tensor("mul", (4, 8, 16)))
    red = Node(id="red", op=ReduceOp("sum", 1), inputs=["mul"], output=Tensor("red", (4, 16)))
    kc = KernelOp(
        inputs=[Port("kp_out"), Port("B")],
        outputs=[Port("red")],
        core=ContractionCore(a=Port("kp_out"), b=Port("B"), k_axis=1, mul=mul, reduce=red),
        external_shapes={"kp_out": (4, 8), "B": (8, 16)},
    )
    kc_id = g.add_node(op=kc, inputs=[kp_id, b], output=Tensor("kc_out", (4, 16)), node_id="kc_out")
    g.outputs = [kc_id]

    result = _apply_rule(g, "055_merge_kernels")

    # Both kernels survive.
    assert len(_kernels(result)) == 2


def test_shape_mismatch_blocks_merge():
    """Two pointwise kernels with non-broadcast-compatible inputs don't merge."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4, 16)), node_id="x")
    y = g.add_node(InputOp(), [], Tensor("y", (8, 16)), node_id="y")  # different rows
    g.inputs = [x, y]

    relu_x = Node(id="rx", op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("rx", (4, 16)))
    k_a = KernelOp(
        inputs=[Port("x")],
        outputs=[Port("rx")],
        prologue=(relu_x,),
        core=None,
        external_shapes={"x": (4, 16)},
    )
    k_a_id = g.add_node(op=k_a, inputs=[x], output=Tensor("a_out", (4, 16)), node_id="a_out")

    # k_b consumes both a_out (4,16) and y (8,16) — incompatible.
    add_y = Node(id="add", op=ElementwiseOp("add"), inputs=["a_out", "y"], output=Tensor("add", (4, 16)))
    k_b = KernelOp(
        inputs=[Port("a_out"), Port("y")],
        outputs=[Port("add")],
        prologue=(add_y,),
        core=None,
        external_shapes={"a_out": (4, 16), "y": (8, 16)},
    )
    k_b_id = g.add_node(op=k_b, inputs=[k_a_id, y], output=Tensor("b_out", (4, 16)), node_id="b_out")
    g.outputs = [k_b_id]

    result = _apply_rule(g, "055_merge_kernels")

    # Merge would create a kernel with inputs (4,16) + (8,16) — rejected.
    assert len(_kernels(result)) == 2
