"""Tests for op provenance (deplodock.compiler.provenance).

M1 covers the data model + seeding; the splice-propagation behavior
(mint / aggregate through decomposition + fusion) is exercised in
``test_provenance_splice.py``.
"""

from deplodock.compiler import provenance as prov
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.tensor.ir import ElementwiseOp, ReduceOp


def _graph() -> tuple[Graph, str, str, str, str]:
    """C = Reduce{sum}(Elementwise{mul}(A, B)) — two compute nodes, two inputs."""
    g = Graph()
    a = g.add_node(InputOp(), [], Tensor("A", (4, 8, 4)))
    b = g.add_node(InputOp(), [], Tensor("B", (4, 8, 4)))
    g.inputs = [a, b]
    ew = g.add_node(ElementwiseOp(op="multiply"), [a, b], Tensor("AB", (4, 8, 4)))
    c = g.add_node(ReduceOp(op="sum", axis=1), [ew], Tensor("C", (4, 1, 4)))
    g.outputs = [c]
    return g, a, b, ew, c


# --- seed ---


def test_seed_stamps_compute_skips_boundary():
    g, a, b, ew, c = _graph()
    prov.seed(g)
    assert prov.get(g.nodes[ew]) == {ew: {"kind": "ElementwiseOp", "pieces": [ew]}}
    assert prov.get(g.nodes[c]) == {c: {"kind": "ReduceOp", "pieces": [c]}}
    # InputOp sentinels compute nothing — never seeded.
    assert prov.get(g.nodes[a]) == {}
    assert prov.get(g.nodes[b]) == {}


def test_seed_idempotent():
    g, _, _, ew, _ = _graph()
    prov.seed(g)
    prov.put(g.nodes[ew], {"custom": {"kind": "X", "pieces": ["p"]}})
    prov.seed(g)  # must not overwrite an already-seeded node
    assert prov.get(g.nodes[ew]) == {"custom": {"kind": "X", "pieces": ["p"]}}


# --- helpers ---


def test_union_merges_pieces_keeps_kind():
    a = {"o": {"kind": "RmsNormOp", "pieces": ["p1", "p2"]}}
    b = {"o": {"kind": "RmsNormOp", "pieces": ["p2", "p3"]}, "q": {"kind": "LinearOp", "pieces": ["x"]}}
    u = prov.union(a, b)
    assert u["o"] == {"kind": "RmsNormOp", "pieces": ["p1", "p2", "p3"]}
    assert u["q"] == {"kind": "LinearOp", "pieces": ["x"]}


def test_mint_fresh_piece_per_origin():
    m = prov.mint({"o": "RmsNormOp", "q": "LinearOp"}, "newnode")
    assert m == {
        "o": {"kind": "RmsNormOp", "pieces": ["newnode"]},
        "q": {"kind": "LinearOp", "pieces": ["newnode"]},
    }


def test_origins():
    p = {"o": {"kind": "RmsNormOp", "pieces": ["p1"]}, "q": {"kind": "LinearOp", "pieces": ["x"]}}
    assert prov.origins(p) == {"o": "RmsNormOp", "q": "LinearOp"}


def test_totals_and_coverage():
    g, _, _, ew, c = _graph()
    prov.put(g.nodes[ew], {"o": {"kind": "RmsNormOp", "pieces": ["p1", "p2"]}})
    prov.put(g.nodes[c], {"o": {"kind": "RmsNormOp", "pieces": ["p3"]}})
    t = prov.totals(g)
    assert t["o"] == {"p1", "p2", "p3"}
    assert prov.coverage(prov.get(g.nodes[ew]), t)["o"] == (2, 3, False)

    # A node holding every piece is "full".
    prov.put(g.nodes[c], {"o": {"kind": "RmsNormOp", "pieces": ["p1", "p2", "p3"]}})
    t = prov.totals(g)
    assert prov.coverage(prov.get(g.nodes[c]), t)["o"] == (3, 3, True)


# --- serialization ---


def test_prov_roundtrips_through_serialization():
    g, _, _, ew, _ = _graph()
    prov.put(g.nodes[ew], {"rms_0": {"kind": "RmsNormOp", "pieces": ["sq", "mean"]}})
    g2 = Graph.from_dict(g.to_dict())
    got = prov.get(g2.nodes[ew])
    assert got == {"rms_0": {"kind": "RmsNormOp", "pieces": ["sq", "mean"]}}
    assert isinstance(got["rms_0"]["pieces"], list)  # stays a list, not stringified
