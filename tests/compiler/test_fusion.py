"""Tests for automatic fusion region discovery."""

import json
from pathlib import Path

from deplodock.compiler.fusion import UnionFind, auto_fuse
from deplodock.compiler.ir import Graph
from deplodock.compiler.ops import FusedRegionOp
from deplodock.compiler.rewriter import Rewriter

FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_and_decompose() -> Graph:
    """Load TinyLlama fixture and run decomposition (no fusion)."""
    with open(FIXTURE_DIR / "tinyllama_layer0.json") as f:
        g = Graph.from_dict(json.load(f))
    rules_dir = Path(__file__).parent.parent.parent / "deplodock" / "compiler" / "rules"
    rewriter = Rewriter.from_directory(rules_dir)
    # Run only decomposition pass.
    decomp_pass = [p for p in rewriter.passes if p.name == "decomposition"][0]
    return decomp_pass.apply(g)


# ---- UnionFind tests ----


def test_union_find_basic():
    uf = UnionFind()
    uf.add("a")
    uf.add("b")
    uf.add("c")
    assert uf.find("a") != uf.find("b")
    uf.merge("a", "b")
    assert uf.find("a") == uf.find("b")
    assert uf.members("a") == {"a", "b"}


def test_union_find_all_groups():
    uf = UnionFind()
    for x in "abcde":
        uf.add(x)
    uf.merge("a", "b")
    uf.merge("c", "d")
    groups = uf.all_groups()
    assert len(groups) == 3  # {a,b}, {c,d}, {e}


# ---- auto_fuse tests on decomposed TinyLlama ----


def test_auto_fuse_produces_fused_regions():
    """auto_fuse groups primitive ops into FusedRegionOp nodes."""
    decomposed = _load_and_decompose()
    fused = auto_fuse(decomposed)

    fused_ops = [n for n in fused.nodes.values() if isinstance(n.op, FusedRegionOp)]
    assert len(fused_ops) > 0, "Expected at least one FusedRegionOp"


def test_auto_fuse_reduces_node_count():
    """Fusion should reduce the total node count."""
    decomposed = _load_and_decompose()
    fused = auto_fuse(decomposed)

    assert len(fused.nodes) < len(decomposed.nodes), f"Expected fewer nodes after fusion: {len(fused.nodes)} >= {len(decomposed.nodes)}"


def test_auto_fuse_preserves_io():
    """Fusion preserves graph inputs and outputs."""
    decomposed = _load_and_decompose()
    fused = auto_fuse(decomposed)

    assert len(fused.inputs) == len(decomposed.inputs)
    assert len(fused.outputs) == len(decomposed.outputs)


def test_auto_fuse_region_has_ops():
    """Each FusedRegionOp contains primitive ops."""
    decomposed = _load_and_decompose()
    fused = auto_fuse(decomposed)

    for n in fused.nodes.values():
        if isinstance(n.op, FusedRegionOp):
            assert len(n.op.region_ops) >= 2, f"Region too small: {len(n.op.region_ops)} ops"
            assert len(n.op.input_names) > 0, "Region has no inputs"
            assert len(n.op.output_names) > 0, "Region has no outputs"
