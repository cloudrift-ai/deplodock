"""Unit tests for the chain-grammar parser (matcher.parse_chain)."""

from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.graph import Graph, Tensor
from deplodock.compiler.ir.tensor import ElementwiseOp, ReduceOp
from deplodock.compiler.matcher import Group, Production, match_grammar, parse_chain


def _build_chain(ops: list[tuple[str, object]]) -> Graph:
    """Build a linear fan-out-1 chain: input -> op[0] -> op[1] -> ..."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.inputs = ["x"]
    prev = "x"
    for nid, op in ops:
        g.add_node(op=op, inputs=[prev], output=Tensor(nid, (4,)), node_id=nid)
        prev = nid
    g.outputs = [prev]
    return g


# ---------------------------------------------------------------------------
# Single production with various quantifiers
# ---------------------------------------------------------------------------


def test_production_quantifier_1_matches_exactly_one():
    g = _build_chain([("e", ElementwiseOp("exp"))])
    m = parse_chain(g, "e", [Production("op", ElementwiseOp, "1")])
    assert m is not None
    assert m.get("op") == ["e"]


def test_production_quantifier_1_fails_on_wrong_type():
    g = _build_chain([("r", ReduceOp("sum", -1))])
    m = parse_chain(g, "r", [Production("op", ElementwiseOp, "1")])
    assert m is None


def test_production_quantifier_optional_matches_zero():
    g = _build_chain([("r", ReduceOp("sum", -1))])
    # Optional elementwise at the start; cursor is a ReduceOp, so it's skipped.
    m = parse_chain(
        g,
        "r",
        [
            Production("ew", ElementwiseOp, "?"),
            Production("red", ReduceOp, "1"),
        ],
    )
    assert m is not None
    assert m.get("ew") == []
    assert m.get("red") == ["r"]


def test_production_quantifier_star_matches_many():
    g = _build_chain(
        [
            ("e1", ElementwiseOp("exp")),
            ("e2", ElementwiseOp("neg")),
            ("e3", ElementwiseOp("abs")),
        ]
    )
    m = parse_chain(g, "e1", [Production("ops", ElementwiseOp, "*")])
    assert m is not None
    assert m.get("ops") == ["e1", "e2", "e3"]


def test_production_quantifier_plus_fails_on_zero():
    g = _build_chain([("r", ReduceOp("sum", -1))])
    m = parse_chain(g, "r", [Production("ew", ElementwiseOp, "+")])
    assert m is None


def test_production_quantifier_plus_matches_one_or_more():
    g = _build_chain([("e", ElementwiseOp("exp")), ("r", ReduceOp("sum", -1))])
    m = parse_chain(
        g,
        "e",
        [
            Production("ew", ElementwiseOp, "+"),
            Production("red", ReduceOp, "?"),
        ],
    )
    assert m is not None
    assert m.get("ew") == ["e"]
    assert m.get("red") == ["r"]


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------


def test_constraint_matches_field():
    g = _build_chain([("m", ElementwiseOp("mul")), ("a", ElementwiseOp("add"))])
    m = parse_chain(
        g,
        "m",
        [
            Production("mul", ElementwiseOp, "1", {"fn": "mul"}),
            Production("add", ElementwiseOp, "1", {"fn": "add"}),
        ],
    )
    assert m is not None
    assert m.get("mul") == ["m"]
    assert m.get("add") == ["a"]


def test_constraint_rejects_mismatch():
    g = _build_chain([("a", ElementwiseOp("add"))])
    m = parse_chain(g, "a", [Production("mul", ElementwiseOp, "1", {"fn": "mul"})])
    assert m is None


# ---------------------------------------------------------------------------
# Fan-out stops the chain
# ---------------------------------------------------------------------------


def test_fan_out_stops_chain():
    """A node with 2 consumers is a chain boundary."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("e", (4,)), node_id="e")
    g.add_node(op=ElementwiseOp("neg"), inputs=["e"], output=Tensor("n1", (4,)), node_id="n1")
    g.add_node(op=ElementwiseOp("abs"), inputs=["e"], output=Tensor("n2", (4,)), node_id="n2")
    g.inputs = ["x"]
    g.outputs = ["n1", "n2"]

    m = parse_chain(g, "e", [Production("ops", ElementwiseOp, "*")])
    assert m is not None
    assert m.get("ops") == ["e"]  # only e; n1/n2 not consumed (fan-out > 1)


# ---------------------------------------------------------------------------
# Groups with backtracking
# ---------------------------------------------------------------------------


def test_group_optional_backtracks_on_partial_match():
    """Group("?") with (mul + sum): if mul matches but sum doesn't, backtrack."""
    g = _build_chain(
        [
            ("m", ElementwiseOp("mul")),
            ("a", ElementwiseOp("add")),  # not a ReduceOp(sum)
        ]
    )
    m = parse_chain(
        g,
        "m",
        [
            Group(
                "contraction",
                [
                    Production("mul", ElementwiseOp, "1", {"fn": "mul"}),
                    Production("reduce", ReduceOp, "1", {"fn": "sum"}),
                ],
                "?",
            ),
            Production("epilogue", ElementwiseOp, "*"),
        ],
    )
    assert m is not None
    # Contraction group backtracked; both nodes consumed as epilogue.
    assert m.get("mul") == []
    assert m.get("epilogue") == ["m", "a"]


def test_group_optional_matches_when_both_present():
    g = _build_chain(
        [
            ("m", ElementwiseOp("mul")),
            ("s", ReduceOp("sum", -1)),
        ]
    )
    m = parse_chain(
        g,
        "m",
        [
            Group(
                "contraction",
                [
                    Production("mul", ElementwiseOp, "1", {"fn": "mul"}),
                    Production("reduce", ReduceOp, "1", {"fn": "sum"}),
                ],
                "?",
            ),
            Production("epilogue", ElementwiseOp, "*"),
        ],
    )
    assert m is not None
    assert m.get("mul") == ["m"]
    assert m.get("reduce") == ["s"]
    assert m.get("epilogue") == []


def test_group_star_repeats():
    """Group("*") repeats: ew* + reduce matches multiple stages."""
    g = _build_chain(
        [
            ("r1", ReduceOp("max", -1)),
            ("sub", ElementwiseOp("sub")),
            ("exp", ElementwiseOp("exp")),
            ("r2", ReduceOp("sum", -1)),
        ]
    )
    m = parse_chain(
        g,
        "r1",
        [
            Group(
                "stage",
                [
                    Production("pre_ops", ElementwiseOp, "*"),
                    Production("reduce", ReduceOp, "1"),
                ],
                "*",
            ),
            Production("epilogue", ElementwiseOp, "*"),
        ],
    )
    assert m is not None
    groups = m.get_groups("stage")
    assert len(groups) == 2
    # Stage 0: no pre_ops (empty, not emitted), reduce=max
    assert groups[0][0].name == "stage[0].reduce"
    assert groups[0][0].node_ids == ["r1"]
    # Stage 1: pre_ops=[sub, exp], reduce=sum
    assert len(groups[1]) == 2
    assert groups[1][0].name == "stage[1].pre_ops"
    assert groups[1][0].node_ids == ["sub", "exp"]
    assert groups[1][1].name == "stage[1].reduce"
    assert groups[1][1].node_ids == ["r2"]


def test_group_plus_fails_on_zero():
    g = _build_chain([("e", ElementwiseOp("exp"))])
    m = parse_chain(
        g,
        "e",
        [
            Group(
                "stage",
                [
                    Production("pre_ops", ElementwiseOp, "*"),
                    Production("reduce", ReduceOp, "1"),
                ],
                "+",
            ),
        ],
    )
    assert m is None  # no reduce found, group "+" requires at least 1


# ---------------------------------------------------------------------------
# Full kernel grammar shape
# ---------------------------------------------------------------------------


def test_kernel_grammar_softmax():
    """Softmax chain: max → sub → exp → sum → div."""
    g = _build_chain(
        [
            ("rmax", ReduceOp("max", -1)),
            ("sub", ElementwiseOp("sub")),
            ("exp", ElementwiseOp("exp")),
            ("rsum", ReduceOp("sum", -1)),
            ("div", ElementwiseOp("div")),
        ]
    )
    grammar = [
        Group(
            "contraction",
            [
                Production("mul", ElementwiseOp, "1", {"fn": "mul"}),
                Production("reduce", ReduceOp, "1", {"fn": "sum"}),
            ],
            "?",
        ),
        Group(
            "stage",
            [
                Production("pre_ops", ElementwiseOp, "*"),
                Production("reduce", ReduceOp, "1"),
            ],
            "*",
        ),
        Production("epilogue", ElementwiseOp, "*"),
    ]
    m = parse_chain(g, "rmax", grammar)
    assert m is not None
    assert m.get("mul") == []  # no contraction
    stages = m.get_groups("stage")
    assert len(stages) == 2
    assert m.get("epilogue") == ["div"]


def test_kernel_grammar_matmul():
    """Matmul: mul → sum."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (4, 8)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (8, 4)), node_id="b")
    g.add_node(op=ElementwiseOp("mul"), inputs=["a", "b"], output=Tensor("m", (4, 8, 4)), node_id="m")
    g.add_node(op=ReduceOp("sum", 1), inputs=["m"], output=Tensor("o", (4, 4)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    grammar = [
        Group(
            "contraction",
            [
                Production("mul", ElementwiseOp, "1", {"fn": "mul"}),
                Production("reduce", ReduceOp, "1", {"fn": "sum"}),
            ],
            "?",
        ),
        Group(
            "stage",
            [
                Production("pre_ops", ElementwiseOp, "*"),
                Production("reduce", ReduceOp, "1"),
            ],
            "*",
        ),
        Production("epilogue", ElementwiseOp, "*"),
    ]
    m = parse_chain(g, "m", grammar)
    assert m is not None
    assert m.get("mul") == ["m"]
    assert m.get("reduce") == ["o"]


# ---------------------------------------------------------------------------
# match_grammar (top-level)
# ---------------------------------------------------------------------------


def test_match_grammar_finds_all_singletons():
    g = _build_chain(
        [
            ("e1", ElementwiseOp("exp")),
            ("e2", ElementwiseOp("neg")),
        ]
    )
    # Fan-out = 1 on e1, so with a "*" production both get consumed in one match.
    matches = match_grammar(g, [Production("op", ElementwiseOp, "+")])
    assert len(matches) == 1
    assert matches[0].get("op") == ["e1", "e2"]


def test_match_grammar_non_overlapping():
    """Two parallel chains from the same input should produce separate matches."""
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (4,)), node_id="x")
    g.add_node(op=ElementwiseOp("exp"), inputs=["x"], output=Tensor("a", (4,)), node_id="a")
    g.add_node(op=ElementwiseOp("neg"), inputs=["x"], output=Tensor("b", (4,)), node_id="b")
    g.inputs = ["x"]
    g.outputs = ["a", "b"]

    matches = match_grammar(g, [Production("op", ElementwiseOp, "1")])
    assert len(matches) == 2
    ids = {m.root_node_id for m in matches}
    assert ids == {"a", "b"}


def test_already_consumed_skipped():
    g = _build_chain([("e", ElementwiseOp("exp"))])
    m = parse_chain(g, "e", [Production("op", ElementwiseOp, "1")], already_consumed={"e"})
    assert m is None
