"""Tests for structured compiler trace and pipeline."""

import json
from pathlib import Path

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp, ReduceOp
from deplodock.compiler.rewriter import Pass, Rewriter, Rule
from deplodock.compiler.trace import CompilerTrace, ExecutionResult, PassTrace, RuleApplication

# ---- helpers ----


def _make_matmul_graph():
    g = Graph()
    a = g.add_node(op=InputOp(), inputs=[], output=Tensor("A", (4, 3)), node_id="A")
    b = g.add_node(op=InputOp(), inputs=[], output=Tensor("B", (3, 2)), node_id="B")
    g.inputs = [a, b]
    ew = g.add_node(
        op=ElementwiseOp(fn="mul"),
        inputs=[a, b],
        output=Tensor("AB", (4, 3, 2)),
        node_id="ew",
    )
    red = g.add_node(
        op=ReduceOp(fn="sum", axis=1),
        inputs=[ew],
        output=Tensor("C", (4, 2)),
        node_id="red",
    )
    g.outputs = [red]
    return g


def _load_fusion_rule():
    rule_path = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "fusion" / "001_fuse_reduce_elementwise.py"
    return Rule.from_file(rule_path)


# ---- graph serialization ----


def test_graph_to_dict():
    g = _make_matmul_graph()
    d = g.to_dict()
    assert d["inputs"] == ["A", "B"]
    assert d["outputs"] == ["red"]
    assert "A" in d["nodes"]
    assert d["nodes"]["ew"]["op"] == "ElementwiseOp"
    assert d["nodes"]["ew"]["op_fields"]["fn"] == "mul"
    assert d["nodes"]["red"]["op"] == "ReduceOp"


def test_graph_to_dict_is_json_serializable():
    g = _make_matmul_graph()
    d = g.to_dict()
    s = json.dumps(d)
    assert isinstance(s, str)
    assert json.loads(s) == d


# ---- trace structure ----


def test_trace_round_trip():
    trace = CompilerTrace(
        input_graph={"nodes": {}, "inputs": [], "outputs": []},
        passes=[
            PassTrace(
                name="fusion",
                rules_applied=[
                    RuleApplication(
                        rule_name="001_fuse",
                        matched_at="red",
                        bindings={"A": "A", "B": "B"},
                        captured_constraints={"f": "sum", "ax": 1},
                    )
                ],
                graph_before={"nodes": {}},
                graph_after={"nodes": {}},
            )
        ],
        generated_code="__global__ void test() {}",
        execution=ExecutionResult(
            output=[1.0, 2.0],
            expected=[1.0, 2.0],
            correct=True,
            max_error=0.0,
            kernel_time_ms=0.042,
            dimensions={"M": 2, "N": 1, "K": 3},
        ),
    )
    j = trace.to_json()
    parsed = json.loads(j)

    assert parsed["passes"][0]["pass"] == "fusion"
    assert parsed["passes"][0]["rules_applied"][0]["rule"] == "001_fuse"
    assert parsed["generated_code"] == "__global__ void test() {}"
    assert parsed["execution"]["correct"] is True
    assert parsed["execution"]["kernel_time_ms"] == 0.042


def test_trace_with_error():
    trace = CompilerTrace(error="Lowering failed: unsupported op")
    parsed = json.loads(trace.to_json())
    assert parsed["error"] == "Lowering failed: unsupported op"
    assert "execution" not in parsed


# ---- rewriter with trace ----


def test_rewriter_populates_trace():
    g = _make_matmul_graph()
    rule = _load_fusion_rule()
    rewriter = Rewriter(passes=[Pass(name="fusion", rules=[rule])])

    pass_traces: list[PassTrace] = []
    rewriter.apply(g, pass_traces=pass_traces)

    assert len(pass_traces) == 1
    pt = pass_traces[0]
    assert pt.name == "fusion"
    assert pt.graph_before is not None
    assert pt.graph_after is not None
    assert len(pt.rules_applied) == 1
    assert pt.rules_applied[0].rule_name == "001_fuse_reduce_elementwise"
    assert pt.rules_applied[0].matched_at == "red"

    # graph_after should have the fused op, not the original.
    op_types = {n["op"] for n in pt.graph_after["nodes"].values()}
    assert "MatmulOp" in op_types
    assert "ReduceOp" not in op_types


def test_rewriter_trace_no_match():
    """When no rules match, trace should still record before/after with empty rules."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    g.inputs = [x]
    g.outputs = [x]

    rule = _load_fusion_rule()
    rewriter = Rewriter(passes=[Pass(name="fusion", rules=[rule])])

    pass_traces: list[PassTrace] = []
    rewriter.apply(g, pass_traces=pass_traces)

    assert len(pass_traces) == 1
    assert pass_traces[0].rules_applied == []
    assert pass_traces[0].graph_before == pass_traces[0].graph_after


def test_full_trace_json_is_parseable():
    """Full trace from rewriter produces valid JSON with expected structure."""
    g = _make_matmul_graph()
    rule = _load_fusion_rule()
    rewriter = Rewriter(passes=[Pass(name="fusion", rules=[rule])])

    trace = CompilerTrace()
    trace.input_graph = g.to_dict()
    g = rewriter.apply(g, pass_traces=trace.passes)

    j = trace.to_json()
    parsed = json.loads(j)

    assert "input_graph" in parsed
    assert len(parsed["passes"]) == 1
    assert parsed["passes"][0]["rules_applied"][0]["rule"] == "001_fuse_reduce_elementwise"
