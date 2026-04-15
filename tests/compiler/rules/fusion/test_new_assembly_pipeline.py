"""Validate the new rule-based KernelOp assembly path end-to-end.

The seed + absorb rules are staged with underscore-prefixed filenames
(e.g. `020_seed_contraction.py`) so they don't run through the default
Rewriter pipeline while the backend is still consuming the old
flat-prologue format. These tests load the new rules directly and run
them against hand-crafted raw-op graphs, asserting structural output.

Once the backend migration (Commit 2) lands, these rules get renamed to
their active `020-080` prefixes and the default pipeline uses them.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.matcher import match_pattern
from deplodock.compiler.ops import (
    ContractionCore,
    ElementwiseOp,
    InputOp,
    KernelOp,
    ReduceOp,
    ReduceStage,
)
from deplodock.compiler.pattern import parse_pattern

RULES_DIR = Path(__file__).parent.parent.parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion"


def _load_rule(stem: str):
    """Load a disabled (underscore-prefixed) rule file by its stem."""
    path = RULES_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _apply_rule(graph: Graph, rule_stem: str) -> Graph:
    """Run one rule to fixed point."""
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


# ---- Seed rules ----


def test_020_seed_contraction_emits_contraction_core():
    """sum(mul(A, B)) → KernelOp(ContractionCore(a, b, mul, reduce))."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 16)), node_id="B")
    g.inputs = [a, b]
    mul = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("mul", (4, 8, 16)), node_id="mul")
    red = g.add_node(ReduceOp("sum", axis=1), [mul], Tensor("C", (4, 16)), node_id="C")
    g.outputs = [red]

    result = _apply_rule(g, "020_seed_contraction")

    # Expect one KernelOp with ContractionCore.
    kernels = [n for n in result.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    k = kernels[0].op
    assert isinstance(k.core, ContractionCore)
    assert k.core.a.buffer_id == "A"
    assert k.core.b.buffer_id == "B"
    assert k.core.k_axis == 1
    # Mul and reduce moved into core.
    assert k.core.mul is not None and isinstance(k.core.mul.op, ElementwiseOp) and k.core.mul.op.fn == "mul"
    assert k.core.reduce is not None and isinstance(k.core.reduce.op, ReduceOp) and k.core.reduce.op.fn == "sum"
    # Original mul and reduce removed from outer graph.
    assert "mul" not in result.nodes
    assert "C" not in result.nodes


def test_021_seed_reduce_emits_reduce_stage():
    """sum(x) → KernelOp(core=(ReduceStage,))."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (8, 16)), node_id="x")
    g.inputs = [x]
    red = g.add_node(ReduceOp("sum", axis=-1), [x], Tensor("y", (8,)), node_id="y")
    g.outputs = [red]

    result = _apply_rule(g, "021_seed_reduce")

    kernels = [n for n in result.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    k = kernels[0].op
    assert isinstance(k.core, tuple) and len(k.core) == 1
    stage = k.core[0]
    assert isinstance(stage, ReduceStage)
    assert isinstance(stage.reduce.op, ReduceOp) and stage.reduce.op.fn == "sum"


def test_040_seed_pointwise_wraps_standalone_elementwise():
    """A standalone add → KernelOp(prologue=(add,), core=None)."""
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4,)), node_id="x")
    y = g.add_node(InputOp(), [], Tensor("y", (4,)), node_id="y")
    g.inputs = [x, y]
    add = g.add_node(ElementwiseOp("add"), [x, y], Tensor("z", (4,)), node_id="z")
    g.outputs = [add]

    result = _apply_rule(g, "040_seed_pointwise")

    kernels = [n for n in result.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    k = kernels[0].op
    assert k.core is None
    assert len(k.prologue) == 1
    assert isinstance(k.prologue[0].op, ElementwiseOp) and k.prologue[0].op.fn == "add"


# ---- Absorption rules ----


def test_050_absorb_prologue_grows_pointwise_kernel():
    """A pointwise KernelOp whose input is a standalone Elementwise with fan-out=1
    should absorb the Elementwise into its prologue."""
    # Build: inner = mul(x, y); outer_kernel = KernelOp(prologue=(add(inner, z),))
    # After rule: kernel.prologue contains both mul AND add.
    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4,)), node_id="x")
    y = g.add_node(InputOp(), [], Tensor("y", (4,)), node_id="y")
    z = g.add_node(InputOp(), [], Tensor("z", (4,)), node_id="z")
    g.inputs = [x, y, z]
    # Produce the Elementwise that we want absorbed.
    mul_node = g.add_node(ElementwiseOp("mul"), [x, y], Tensor("m", (4,)), node_id="m")
    # Build a pointwise KernelOp by hand that consumes m.
    from deplodock.compiler.ir import Node
    from deplodock.compiler.ops import Port

    inner = Node(
        id="inner",
        op=ElementwiseOp("add"),
        inputs=["m", "z"],
        output=Tensor(name="inner", shape=(4,), dtype="f32"),
    )
    kop = KernelOp(
        inputs=[Port("m"), Port("z")],
        outputs=[Port("out")],
        prologue=(inner,),
        core=None,
        epilogue=(),
        external_shapes={"m": (4,), "z": (4,)},
    )
    kid = g.add_node(op=kop, inputs=[mul_node, z], output=Tensor("out", (4,), "f32"), node_id="out")
    g.outputs = [kid]

    result = _apply_rule(g, "050_absorb_prologue")

    kernels = [n for n in result.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    k = kernels[0].op
    # Prologue should have absorbed the mul (2 nodes: mul + add).
    assert len(k.prologue) == 2
    fns = {n.op.fn for n in k.prologue}
    assert fns == {"mul", "add"}
    # Original standalone mul removed from outer graph.
    assert "m" not in result.nodes


def test_060_absorb_epilogue_appends_downstream_elementwise():
    """Elementwise(Kernel(...)) with fan-out=1 on the kernel → append to epilogue."""
    from deplodock.compiler.ir import Node
    from deplodock.compiler.ops import Port

    g = Graph()
    x = g.add_node(InputOp(), [], Tensor("x", (4,)), node_id="x")
    y = g.add_node(InputOp(), [], Tensor("y", (4,)), node_id="y")
    g.inputs = [x, y]
    # Pointwise kernel producing intermediate.
    inner = Node(
        id="inner",
        op=ElementwiseOp("mul"),
        inputs=["x", "y"],
        output=Tensor(name="inner", shape=(4,), dtype="f32"),
    )
    kop = KernelOp(
        inputs=[Port("x"), Port("y")],
        outputs=[Port("mid")],
        prologue=(inner,),
        core=None,
        epilogue=(),
        external_shapes={"x": (4,), "y": (4,)},
    )
    kid = g.add_node(op=kop, inputs=[x, y], output=Tensor("mid", (4,), "f32"), node_id="mid")
    # Downstream elementwise.
    nid = g.add_node(ElementwiseOp("relu"), [kid], Tensor("out", (4,)), node_id="out")
    g.outputs = [nid]

    result = _apply_rule(g, "060_absorb_epilogue")

    kernels = [n for n in result.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    k = kernels[0].op
    assert len(k.epilogue) == 1
    assert k.epilogue[0].op.fn == "relu"
    assert "out" not in result.nodes or isinstance(result.nodes.get("out"), type(None))


def test_assembly_pipeline_runs_for_matmul_plus_bias():
    """Composed: sum(mul(A, B)) + bias → run 020 then 060."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8)), node_id="A")
    b = g.add_node(InputOp(), [], Tensor("B", (8, 16)), node_id="B")
    bias = g.add_node(InputOp(), [], Tensor("bias", (16,)), node_id="bias")
    g.inputs = [a, b, bias]
    mul = g.add_node(ElementwiseOp("mul"), [a, b], Tensor("mul", (4, 8, 16)), node_id="mul")
    red = g.add_node(ReduceOp("sum", axis=1), [mul], Tensor("C", (4, 16)), node_id="C")
    add = g.add_node(ElementwiseOp("add"), [red, bias], Tensor("y", (4, 16)), node_id="y")
    g.outputs = [add]

    g = _apply_rule(g, "020_seed_contraction")
    g = _apply_rule(g, "060_absorb_epilogue")

    kernels = [n for n in g.nodes.values() if isinstance(n.op, KernelOp)]
    assert len(kernels) == 1
    k = kernels[0].op
    assert isinstance(k.core, ContractionCore)
    # Bias add absorbed into epilogue.
    assert len(k.epilogue) == 1
    assert k.epilogue[0].op.fn == "add"
