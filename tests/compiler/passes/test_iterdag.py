"""The iteration-DAG derived view (``partition/iterdag.py``).

``iter_dag`` is the one structure the partition consumes; the regime skeletons
are projections of it. These tests pin the node tagging (PARALLEL chain +
carrier-tagged REDUCE axes) and prove the DAG carries the fields the skeletons
read off it. See ``plans/algebra-licensed-decomposition-moves.md`` (phase 2).
"""

from __future__ import annotations

from deplodock.compiler.dim import Dim
from deplodock.compiler.ir.algebra import AlgebraKind
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import AxisRole, iter_dag


def _matmul(m: int, n: int, k) -> LoopOp:
    return LoopOp(
        body=(
            Loop(
                axis=Axis("m", m),
                body=(
                    Loop(
                        axis=Axis("n", n),
                        body=(
                            Loop(
                                axis=Axis("k", k),
                                body=(
                                    Load(name="a", input="a", index=(Var("m"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("n"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a", "b")),
                                    Accum(name="acc", value="p", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Write(output="o", index=(Var("m"), Var("n")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )


def _pointwise(n: int) -> LoopOp:
    return LoopOp(
        body=(
            Loop(
                axis=Axis("n", n),
                body=(
                    Load(name="x", input="x", index=(Var("n"),)),
                    Assign(name="y", op=ElementwiseImpl("relu"), args=("x",)),
                    Write(output="o", index=(Var("n"),), value="y"),
                ),
            ),
        ),
    )


def _coop_reduce(rows: int, k: int) -> LoopOp:
    """A per-row sum reduce over a static K (the MONOID regime)."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("r", rows),
                body=(
                    Loop(
                        axis=Axis("k", k),
                        body=(
                            Load(name="x", input="x", index=(Var("r"), Var("k"))),
                            Accum(name="acc", value="x", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("r"),), value="acc"),
                ),
            ),
        ),
    )


def test_matmul_dag_tags_parallel_chain_and_reduce_carrier():
    # LoopOp canonicalizes axis names (a0, a1, …), so assert on structure.
    dag = iter_dag(_matmul(128, 256, 64))
    # Two PARALLEL free axes (m=128 outer, n=256 inner), one REDUCE axis (k=64).
    assert [n.extent for n in dag.parallel] == [128, 256]
    assert all(n.role is AxisRole.PARALLEL and n.carrier is None for n in dag.parallel)
    assert [n.extent for n in dag.reduce] == [64]
    (k,) = dag.reduce
    assert k.role is AxisRole.REDUCE
    assert isinstance(k.carrier, Accum) and k.carrier.op.name == "add"
    assert k.algebra is AlgebraKind.SEMIRING
    # The reduce node nests under the innermost free axis.
    assert k.parent is dag.parallel[-1]
    assert dag.algebras == {AlgebraKind.SEMIRING}


def test_pointwise_dag_has_no_reduce():
    dag = iter_dag(_pointwise(512))
    assert [n.extent for n in dag.parallel] == [512]
    assert dag.reduce == ()
    assert dag.algebras == set()


def test_coop_reduce_dag_tags_monoid():
    dag = iter_dag(_coop_reduce(32, 256))
    assert [n.extent for n in dag.parallel] == [32]
    assert [n.extent for n in dag.reduce] == [256]
    assert dag.algebras == {AlgebraKind.MONOID}


def test_symbolic_axis_flagged_and_hint_extent():
    dag = iter_dag(_matmul(Dim("seq_len", hint=512), 256, 64))
    m = dag.parallel[0]
    assert m.symbolic and m.extent == 512  # tiles at the Dim hint


def test_dag_accessors_feed_the_partition():
    # The partition consumes the DAG directly (no skeleton): the free-axis + K-info
    # accessors give the tiled axes and the contraction extent/bound.
    lo = _matmul(128, 256, 64)
    dag = iter_dag(lo)
    assert dag.inner_n is dag.parallel[-1] and dag.inner_n.extent == 256
    assert dag.outer_m is dag.parallel[-2] and dag.outer_m.extent == 128
    assert dag.extra_outer == ()
    assert dag.k_node is dag.reduce[0]
    assert dag.k_extent == 64 and dag.k_bound is None  # static K
