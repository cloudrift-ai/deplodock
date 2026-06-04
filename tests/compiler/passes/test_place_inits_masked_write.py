"""Regression test for ``020_place_inits`` on a masked register-tile write.

A register-M (FM>1) tile that writes its output through a boundary mask
emits ``SerialTile(a4=FM) > [ reduce > Accum, Cond(coord < N, [Write acc]) ]``
(``010_split_register_axes`` unrolls the register cells before this pass, so
the masked ``Write`` sits directly in the FM loop's body, wrapped in a Cond).

``_is_reduce_recursive`` must treat a ``Cond`` wrapping a ``Write`` as the same
per-iteration output escape as a bare ``Write`` — so the FM loop is judged
*non-crossable* and the accumulator ``Init`` lands **inside** it (reset per
row). The bug this guards against: the mask hid the Write, the FM loop was
judged crossable, the ``Init`` was hoisted **above** it, and accumulators
leaked across register-M rows (every row past the first got the previous rows'
sums — the whole-model fp32 lm_head ``max_diff ~68`` logits failure).
"""

from __future__ import annotations

import importlib

from deplodock.compiler.graph import Graph
from deplodock.compiler.ir.axis import Axis
from deplodock.compiler.ir.expr import BinaryExpr, Literal, Var
from deplodock.compiler.ir.stmt import Accum, Body, Cond, Init, Write
from deplodock.compiler.ir.tile.ir import SerialTile, TileOp
from deplodock.compiler.pipeline.pipeline import Match
from deplodock.compiler.tensor import Tensor

_place_inits = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.020_place_inits")


def _masked_regm_tile_op() -> TileOp:
    """FM register-M loop holding a K-reduce ``Accum`` and a masked ``Write``."""
    a4 = Axis("a4", 2)  # register-M (FM), source axis M
    k = Axis("k", 16)
    reduce_loop = SerialTile(axis=k, body=Body((Accum(name="acc", value="v"),)))
    masked_write = Cond(
        cond=BinaryExpr(op="<", left=Var("a1"), right=Literal(1025, "int")),
        body=Body((Write(output="out", index=(Var("a4"), Var("a1")), values=("acc",)),)),
        else_body=(),
    )
    fm = SerialTile(axis=a4, body=Body((reduce_loop, masked_write)), kind="plain")
    return TileOp(body=Body((fm,)), name="k_masked_regm")


def _run(op: TileOp) -> TileOp:
    g = Graph()
    g.add_node(op=op, inputs=[], output=Tensor(op.name, ()), node_id="op")
    match = Match(
        graph=g,
        nodes={"root": "op"},
        consumed=set(),
        root_node_id="op",
        pipeline=None,
        rule=None,
        is_last=True,  # type: ignore[arg-type]
    )
    result = _place_inits.rewrite(match, g.nodes["op"])
    assert isinstance(result, TileOp)
    return result


def _fm_loop(op: TileOp) -> SerialTile:
    for s in op.body:
        if isinstance(s, SerialTile):
            return s
    raise AssertionError("no FM SerialTile in result")


def test_init_inside_masked_register_m_loop():
    """The accumulator Init lands inside the FM loop (reset per row)."""
    result = _run(_masked_regm_tile_op())
    fm = _fm_loop(result)
    assert isinstance(fm.body[0], Init), f"first FM-body stmt is {type(fm.body[0]).__name__}, expected Init"
    accum = next(s for s in fm.body.iter() if isinstance(s, Accum))
    assert fm.body[0].name == accum.name


def test_no_init_hoisted_above_masked_fm_loop():
    """The Init must NOT appear at the TileOp scope above the FM loop — that is
    the leak mode (accumulator persists across register-M rows)."""
    result = _run(_masked_regm_tile_op())
    top_level_inits = [s for s in result.body if isinstance(s, Init)]
    assert not top_level_inits, f"Init leaked above the FM loop: {top_level_inits}"
