"""Tests for structured compiler trace and pipeline."""

import json
from pathlib import Path

from deplodock.compiler.ir import Graph, Tensor
from deplodock.compiler.ops import ElementwiseOp, InputOp
from deplodock.compiler.rewriter import Pass, Rewriter, Rule
from deplodock.compiler.trace import CompilerTrace, ExecutionResult, PassTrace, RuleApplication

# ---- helpers ----


def _make_pow_graph():
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.inputs = [x]
    pw = g.add_node(op=ElementwiseOp(fn="pow"), inputs=[x], output=Tensor("sq", (4,)), node_id="sq")
    g.outputs = [pw]
    return g


def _load_decomp_rule():
    rule_path = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules" / "decomposition" / "003_decompose_pow.py"
    return Rule.from_file(rule_path)


# ---- graph serialization ----


def test_graph_to_dict():
    g = _make_pow_graph()
    d = g.to_dict()
    assert d["inputs"] == ["x"]
    assert d["outputs"] == ["sq"]
    assert "x" in d["nodes"]
    assert d["nodes"]["sq"]["op"] == "ElementwiseOp"
    assert d["nodes"]["sq"]["op_fields"]["fn"] == "pow"


def test_graph_to_dict_is_json_serializable():
    g = _make_pow_graph()
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
                name="decomposition",
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

    assert parsed["passes"][0]["pass"] == "decomposition"
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
    g = _make_pow_graph()
    rule = _load_decomp_rule()
    rewriter = Rewriter(passes=[Pass(name="decomposition", rules=[rule])])

    pass_traces: list[PassTrace] = []
    rewriter.apply(g, pass_traces=pass_traces)

    assert len(pass_traces) == 1
    pt = pass_traces[0]
    assert pt.name == "decomposition"
    assert pt.graph_before is not None
    assert pt.graph_after is not None
    assert len(pt.rules_applied) == 1
    assert pt.rules_applied[0].rule_name == "003_decompose_pow"

    # graph_after should have the fused op, not the original.


def test_rewriter_trace_no_match():
    """When no rules match, trace should still record before/after with empty rules."""
    g = Graph()
    x = g.add_node(op=InputOp(), inputs=[], output=Tensor("X", (4,)), node_id="x")
    g.inputs = [x]
    g.outputs = [x]

    rule = _load_decomp_rule()
    rewriter = Rewriter(passes=[Pass(name="decomposition", rules=[rule])])

    pass_traces: list[PassTrace] = []
    rewriter.apply(g, pass_traces=pass_traces)

    assert len(pass_traces) == 1
    assert pass_traces[0].rules_applied == []
    assert pass_traces[0].graph_before == pass_traces[0].graph_after


def test_full_trace_json_is_parseable():
    """Full trace from rewriter produces valid JSON with expected structure."""
    g = _make_pow_graph()
    rule = _load_decomp_rule()
    rewriter = Rewriter(passes=[Pass(name="decomposition", rules=[rule])])

    trace = CompilerTrace()
    trace.input_graph = g.to_dict()
    g = rewriter.apply(g, pass_traces=trace.passes)

    j = trace.to_json()
    parsed = json.loads(j)

    assert "input_graph" in parsed
    assert len(parsed["passes"]) == 1
    assert parsed["passes"][0]["rules_applied"][0]["rule"] == "003_decompose_pow"
