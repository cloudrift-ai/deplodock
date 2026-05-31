"""M1 of ``plans/inline-fma-cluster.md``: the two FFMA-cluster IR nodes.

``FmaClusterTile`` (tile-IR, pre-threading) and ``FmaCluster`` (kernel-IR,
post-threading) are inert in M1 — no pass constructs them, and ``render()``
raises because they're lowered/emitted in later milestones. These tests pin
the contract M2/M3 build on: frozen-dataclass hashability + equality, the
``repr`` → eval round-trip the JSON serializer relies on, a full
``Graph.to_dict`` → ``from_dict`` body round-trip, and the render guard.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.dim import Dim
from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor, _stmt_eval_scope
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.kernel.ir import FmaCluster, KernelOp
from deplodock.compiler.ir.stmt import Body
from deplodock.compiler.ir.tile.ir import FmaClusterTile, TileOp


def _axis(name: str, extent: int) -> Axis:
    return Axis(name=name, extent=Dim(extent))


def _cluster_tile() -> FmaClusterTile:
    return FmaClusterTile(
        fm=26,
        fn=4,
        bk=32,
        a_axis=_axis("m", 26),
        b_axis=_axis("n", 4),
        k_axis=_axis("k", 32),
        a_smem="a_smem",
        b_smem="b_smem",
        a_index=(Var("m"), Var("k")),
        b_index=(Var("k"), Var("n")),
        acc_base="acc",
    )


def _cluster() -> FmaCluster:
    return FmaCluster(
        a_names=("a0", "a1", "a2"),
        b_names=("b0", "b1"),
        acc_names=tuple(f"acc{i}" for i in range(6)),
        a_addr=Var("a_smem_addr"),
        b_addr=Var("b_smem_addr"),
        a_vec=4,
        b_vec=2,
    )


def test_nodes_are_frozen_hashable_and_equal():
    """Both nodes are ``@dataclass(frozen=True)`` per ``feedback_stmt_hashable``:
    hashable, structurally equal, and immutable."""
    for node in (_cluster_tile(), _cluster()):
        assert hash(node) == hash(node)
        # A distinct-but-equal rebuild compares equal and hashes equal.
        rebuilt = type(node)(**{f: getattr(node, f) for f in node.__dataclass_fields__})
        assert rebuilt == node
        assert hash(rebuilt) == hash(node)
        with pytest.raises((AttributeError, TypeError)):
            node.fm = 1  # type: ignore[misc]  # frozen


def test_defaults_on_fma_cluster():
    c = _cluster()
    assert c.dtype is F32
    assert c.policy == "B_INNER"


def test_repr_eval_round_trip_in_stmt_scope():
    """Op bodies serialize via ``repr`` and reload via ``eval`` in
    ``_stmt_eval_scope``. Both new nodes must round-trip there — the scope was
    extended to carry them (regression guard against forgetting that wiring)."""
    scope = dict(_stmt_eval_scope())
    for node in (_cluster_tile(), _cluster()):
        back = eval(repr(node), scope)  # noqa: S307 — trusted IR repr, sandboxed scope
        assert back == node
        assert hash(back) == hash(node)


def test_graph_json_body_round_trip():
    """Full ``Graph.to_dict`` → ``from_dict`` round-trip with each node living
    inside an op body — the path ``deplodock run --ir <dump>`` exercises."""
    g = Graph()
    g.add_node(TileOp(body=Body((_cluster_tile(),)), name="k_tile"), [], Tensor("t", (26, 4)), node_id="t")
    g.add_node(KernelOp(body=Body((_cluster(),)), name="k_kernel"), [], Tensor("k", (3, 2)), node_id="k")

    loaded = Graph.from_dict(g.to_dict())
    (tile_stmt,) = loaded.nodes["t"].op.body
    (kernel_stmt,) = loaded.nodes["k"].op.body
    assert tile_stmt == _cluster_tile()
    assert kernel_stmt == _cluster()


def test_render_raises_until_lowered():
    """Both nodes are lowered/emitted before render; a stray unlowered node
    must fail loudly rather than silently emit nothing."""
    for node in (_cluster_tile(), _cluster()):
        with pytest.raises(NotImplementedError):
            node.render(None)  # type: ignore[arg-type]
