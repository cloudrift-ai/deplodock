"""Regression test for ``020_place_inits`` on a warp-specialized body.

A ``WarpSpecialize`` is wrapped inside a ``WarpTile(role)``; the consumer half owns the
reduce loop and its ``Accum``. ``020_place_inits`` must descend into ``WarpSpecialize`` and
place the accumulator ``Init`` at the **consumer_body head** (the per-consumer-thread scope
just above the K loop).

The bug this guards against: before the fix, ``_open_scope`` recursed into the ``WarpTile``
but ``_place_inits_in_scope`` / ``_accums_under_reduces_only`` didn't understand
``WarpSpecialize``, so **no explicit Init was placed at all**. The renderer then fell back to
its default per-reduce-loop init, which lands *inside* the consumer K loop — resetting the
accumulators every K chunk and corrupting the result (the WS=1 ``max_diff ~228`` accuracy
failure).

No live pass constructs a ``WarpSpecialize`` today (the ``tile/085_warp_specialize.py``
producer pass was removed), but the consumer-head Init placement in ``020_place_inits`` is
still live; we build the node directly and run the pass over it.
"""

from __future__ import annotations

import importlib

from emmy.compiler.dim import Dim
from emmy.compiler.graph import Graph
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.stmt import Accum, Body, Init
from emmy.compiler.ir.tile.ir import (
    SerialTile,
    TileOp,
    WarpSpecialize,
    WarpTile,
)
from emmy.compiler.pipeline.pipeline import Match
from emmy.compiler.tensor import Tensor

_place_inits = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.020_place_inits")


def _tile_op_with_ws_reduce() -> TileOp:
    """``WarpTile(role)`` holding a ``WarpSpecialize`` whose consumer half
    is the matmul K-chunk shape: ``SerialTile(k_o, serial_outer) >
    SerialTile(k_i, reduce) > Accum``. The producer half is the bare K_o
    loop (no accumulator). Total threads = 32 consumer + 32 producer = 64
    → role axis extent 2."""
    k_o = Axis("k_outer", 8)
    k_i = Axis("k_inner", 4)
    reduce_loop = SerialTile(axis=k_i, body=Body((Accum(name="acc", value="v"),)))
    consumer_body = Body((SerialTile(axis=k_o, body=Body((reduce_loop,)), kind="serial_outer"),))
    producer_body = Body((SerialTile(axis=k_o, body=Body(()), kind="serial_outer"),))
    ws = WarpSpecialize(
        producer_body=producer_body,
        consumer_body=consumer_body,
        ring_depth=2,
        n_producer_threads=32,
        consumer_thread_axes=(Axis("c_i", 32),),
    )
    body = Body((WarpTile(axes=(Axis("ws_role", Dim(2)),), body=Body((ws,))),))
    return TileOp(body=body, name="k_ws_place_inits")


def _run_place_inits(op: TileOp) -> TileOp:
    g = Graph()
    g.add_node(op=op, inputs=[], output=Tensor(op.name, ()), node_id="op")
    node = g.nodes["op"]
    match = Match(
        graph=g,
        nodes={"root": "op"},
        consumed=set(),
        root_node_id="op",
        rule=None,  # type: ignore[arg-type]
        is_last=True,
    )
    result = _place_inits.rewrite(match, node)
    assert isinstance(result, TileOp), f"expected TileOp, got {type(result).__name__}"
    return result


def _find_ws(op: TileOp) -> WarpSpecialize:
    for s in op.body.iter():
        if isinstance(s, WarpSpecialize):
            return s
    raise AssertionError("no WarpSpecialize in result body")


def test_init_placed_at_consumer_body_head():
    """The consumer half gains an ``Init`` as its first stmt, directly
    above the serial_outer K loop."""
    result = _run_place_inits(_tile_op_with_ws_reduce())
    ws = _find_ws(result)
    cons = tuple(ws.consumer_body)
    assert isinstance(cons[0], Init), f"first consumer stmt is {type(cons[0]).__name__}, expected Init"
    # The Init targets the accumulator nested in the consumer K loop.
    # (``normalize_body`` canonicalises the ``acc`` name to ``acc0``.)
    accum = next(s for s in ws.consumer_body.iter() if isinstance(s, Accum))
    assert cons[0].name == accum.name
    # The loop still follows the Init at the same (consumer-head) scope.
    assert isinstance(cons[1], SerialTile) and cons[1].kind == "serial_outer"


def test_no_init_inside_consumer_k_loop():
    """The accumulator Init must NOT appear inside the serial_outer K loop
    (that is the corruption mode — reset every chunk)."""
    result = _run_place_inits(_tile_op_with_ws_reduce())
    ws = _find_ws(result)
    k_o = next(s for s in ws.consumer_body if isinstance(s, SerialTile) and s.kind == "serial_outer")
    inits_in_loop = [s for s in k_o.body.iter() if isinstance(s, Init)]
    assert not inits_in_loop, f"Init leaked inside the K loop: {inits_in_loop}"


def test_producer_body_gets_no_init():
    """The producer half issues TMA only — no accumulator, no Init."""
    result = _run_place_inits(_tile_op_with_ws_reduce())
    ws = _find_ws(result)
    inits_in_producer = [s for s in ws.producer_body.iter() if isinstance(s, Init)]
    assert not inits_in_producer, f"unexpected Init in producer body: {inits_in_producer}"
