"""Byte-identical equivalence: assemble(block-DAG) == legacy composer tower.

The plan's safety contract (``plans/tile-ir-block-dag.md`` step 1): the composer
emits a ``TileGraph`` (algorithm + reference Schedule) that ``assemble`` lowers to
the SAME ``TileOp`` the legacy ``materialize.build_pointwise_tile`` produced. This
test pins that equivalence for the pointwise regime.
"""

from __future__ import annotations

import pytest

from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import LoopOp
from deplodock.compiler.ir.stmt import Accum, Assign, Load, Loop, Write
from deplodock.compiler.pipeline.passes.lowering.tile.partition.assemble import assemble_block, assemble_pointwise
from deplodock.compiler.pipeline.passes.lowering.tile.partition.build_dag import build_matmul_dag, build_pointwise_dag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.iterdag import iter_dag
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_matmul_tile, build_pointwise_tile


def _pointwise_loop_op(*, M: int, N: int) -> LoopOp:
    i, j = Axis("i", M), Axis("j", N)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                            Assign(name="y_v", op=ElementwiseImpl("relu"), args=("x_v",)),
                            Write(output="y", index=(Var("i"), Var("j")), value="y_v"),
                        ),
                    ),
                ),
            ),
        ),
        name="k_relu",
    )


def _knobs_2d() -> dict:
    return {"BN": 32, "FN": 2, "BM": 8, "FM": 2}


@pytest.mark.parametrize("M,N", [(64, 128), (128, 256)])
def test_assemble_matches_legacy_pointwise(M, N):
    loop_op = _pointwise_loop_op(M=M, N=N)
    dag = iter_dag(loop_op)
    knobs = _knobs_2d()
    base = dict(loop_op.knobs)

    legacy = build_pointwise_tile(knobs, kernel_name="k_relu", base_knobs=base, dag=dag)

    tg = build_pointwise_dag(dag, knobs, kernel_name="k_relu")
    assembled = assemble_pointwise(tg, knobs=knobs, base_knobs=base, kernel_name="k_relu", leading=dag.leading)

    assert assembled.body == legacy.body, f"\nASSEMBLED:\n{assembled.pretty_body()}\n\nLEGACY:\n{legacy.pretty_body()}"
    assert assembled.knobs == legacy.knobs


def test_assemble_1d_pointwise():
    # 1-D pointwise (single free axis, no M) — exercises the outer_m=None path.
    j = Axis("j", 256)
    loop_op = LoopOp(
        body=(
            Loop(
                axis=j,
                body=(
                    Load(name="x_v", input="x", index=(Var("j"),)),
                    Assign(name="y_v", op=ElementwiseImpl("relu"), args=("x_v",)),
                    Write(output="y", index=(Var("j"),), value="y_v"),
                ),
            ),
        ),
        name="k_relu1d",
    )
    dag = iter_dag(loop_op)
    knobs = {"BN": 64, "FN": 4, "BM": 1, "FM": 1}
    legacy = build_pointwise_tile(knobs, kernel_name="k_relu1d", base_knobs={}, dag=dag)
    tg = build_pointwise_dag(dag, knobs, kernel_name="k_relu1d")
    assembled = assemble_pointwise(tg, knobs=knobs, base_knobs={}, kernel_name="k_relu1d", leading=dag.leading)
    assert assembled.body == legacy.body, f"\nASSEMBLED:\n{assembled.pretty_body()}\n\nLEGACY:\n{legacy.pretty_body()}"


def _matmul_loop_op(*, M: int, N: int, K: int) -> LoopOp:
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Loop(
                                axis=k,
                                body=(
                                    Load(name="a_v", input="a", index=(Var("i"), Var("k"))),
                                    Load(name="b_v", input="b", index=(Var("k"), Var("j"))),
                                    Assign(name="p", op=ElementwiseImpl("multiply"), args=("a_v", "b_v")),
                                    Accum(name="acc", value="p"),
                                ),
                            ),
                            Write(output="c", index=(Var("i"), Var("j")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
        name="k_matmul",
    )


@pytest.mark.parametrize(
    "bk,fk,splitk",
    [
        (16, 1, 1),  # plain
        (8, 2, 1),  # FK strip-mine
        (16, 1, 2),  # split-K
    ],
)
def test_assemble_matches_legacy_matmul_scalar(bk, fk, splitk):
    loop_op = _matmul_loop_op(M=64, N=64, K=64)
    dag = iter_dag(loop_op)
    knobs = {"BN": 16, "FN": 2, "BM": 16, "FM": 2, "BK": bk, "FK": fk, "SPLITK": splitk}
    base = dict(loop_op.knobs)
    targets = frozenset({"k"})

    legacy = build_matmul_tile(knobs, kernel_name="k_matmul", base_knobs=base, dag=dag, target_names=targets)

    tg = build_matmul_dag(dag, knobs, kernel_name="k_matmul", target_names=targets)
    stamped = {"MMA": "0", "WM": 0, "WN": 0, "BR": 1, **knobs}
    assembled = assemble_block(tg, knobs=stamped, base_knobs=base, kernel_name="k_matmul", leading=dag.leading)

    assert assembled.body == legacy.body, f"\nASSEMBLED:\n{assembled.pretty_body()}\n\nLEGACY:\n{legacy.pretty_body()}"
    assert assembled.knobs == legacy.knobs
