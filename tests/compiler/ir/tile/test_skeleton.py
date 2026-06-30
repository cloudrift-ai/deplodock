"""Construction / field tests for the tile :mod:`~deplodock.compiler.ir.tile.skeleton` types."""

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.tile import AxisRole, ReduceAxis, Scope, Skeleton


def test_axis_role_members():
    assert AxisRole.PARALLEL.value == "parallel"
    assert AxisRole.REDUCE.value == "reduce"


def test_scope_defaults_are_a_pointwise_leaf():
    node = object()  # a stand-in AlgebraNode (the types are structural; no algebra needed here)
    scope = Scope(node=node, parallel=(Axis("m", 32),))
    assert scope.node is node
    assert scope.reduce is None
    assert scope.children == ()


def test_reduce_axis_carries_recognized_facts():
    k = Axis("k", 64)
    carrier = object()  # a stand-in Monoid carrier
    red = ReduceAxis(axis=k, carrier=carrier, contraction=True, coop_eligible=False, binding=None)
    assert red.axis is k
    assert red.carrier is carrier
    assert red.contraction is True
    assert red.coop_eligible is False
    assert red.binding is None


def test_skeleton_nests_scopes():
    inner = Scope(node=object(), reduce=ReduceAxis(axis=Axis("dd", 16), carrier=object(), contraction=True, coop_eligible=False))
    outer = Scope(
        node=object(),
        parallel=(Axis("m", 8), Axis("d", 16)),
        reduce=ReduceAxis(axis=Axis("kv", 128), carrier=object(), contraction=False, coop_eligible=True),
        children=(inner,),
    )
    skel = Skeleton(root=outer)
    assert skel.root is outer
    assert skel.root.children[0] is inner
    assert skel.root.reduce.coop_eligible is True
    assert skel.root.children[0].reduce.contraction is True


def test_frozen():
    red = ReduceAxis(axis=Axis("k", 4), carrier=object(), contraction=False, coop_eligible=True)
    try:
        red.contraction = True  # type: ignore[misc]
    except AttributeError:
        return
    raise AssertionError("ReduceAxis should be frozen")
