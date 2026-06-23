"""General multi-block / multi-launch ``assemble`` (``plans/dag-edge-placement-split-as-enumeration.md`` → step 1).

The single-block restriction in ``assemble_block`` is the coexistence stopgap the
edge-placement plan lifts first: a ``GMEM`` cut yields a multi-block ``TileGraph``
(``Schedule.launch`` partitions blocks into kernels), and ``assemble`` must partition
by launch group into a ``Graph`` of ``TileOp`` kernels with cross-group edges
materialized as intermediate tensors. These tests pin that partition against a
hand-built two-block matmul chain (``o1 = a @ b``; ``o2 = o1 @ b2``) — built by
renaming the I/O buffers of two oracle-tiled matmul blocks so each block is a real,
fully-tiled, assemble-able algorithm — and assert it is deterministic.
"""

from __future__ import annotations

from dataclasses import replace

from deplodock.compiler.dtype import F32
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Literal
from deplodock.compiler.ir.stmt import Body, Load, Write
from deplodock.compiler.ir.tile.ir import Block, Buffer, Space, TileGraph, TileOp
from deplodock.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block
from tests.compiler.passes.test_tile_ir_invariants import _MM_KNOBS, _matmul_graph, _oracle_tilegraph


def _rename_bufs(block: Block, mapping: dict[str, str], *, name: str) -> Block:
    """Rebuild ``block`` with every ``Load.input`` / ``Write.output`` buffer name
    remapped (so two copies of one oracle matmul block form a producer→consumer
    chain), under a fresh block ``name``."""

    def walk(stmts):
        out = []
        for s in stmts:
            bodies = s.nested()
            if bodies:
                s = s.with_bodies(tuple(Body(walk(tuple(b))) for b in bodies))
            if isinstance(s, Load) and s.input in mapping:
                s = replace(s, input=mapping[s.input])
            elif isinstance(s, Write) and s.output in mapping:
                s = replace(s, output=mapping[s.output])
            out.append(s)
        return out

    return Block(name=name, domain=block.domain, compute=Body(walk(tuple(block.compute))))


def _two_block_chain() -> TileGraph:
    """``o1 = a @ b`` (block ``mm1``) → ``o2 = o1 @ b2`` (block ``mm2``): two
    oracle-tiled 64³ matmul blocks wired into one ``TileGraph``, every buffer
    declared (inputs + the ``o1`` intermediate + the ``o2`` output)."""
    base_tg = _oracle_tilegraph(_matmul_graph(), _MM_KNOBS)
    base = base_tg.blocks[0]
    mm1 = _rename_bufs(base, {"o": "o1"}, name="mm1")
    mm2 = _rename_bufs(base, {"a": "o1", "b": "b2", "o": "o2"}, name="mm2")
    sq = (Literal(64, "int"), Literal(64, "int"))
    buffers = {
        "a": Buffer("a", sq, F32),
        "b": Buffer("b", sq, F32),
        "b2": Buffer("b2", sq, F32),
        "o1": Buffer("o1", sq, F32, space=Space.GMEM),
        "o2": Buffer("o2", sq, F32),
    }
    return TileGraph(name="chain", buffers=buffers, blocks=(mm1, mm2), schedule=base_tg.schedule)


def test_multi_block_assembles_to_kernel_graph() -> None:
    """A two-block DAG assembles to a ``Graph`` of two ``TileOp`` kernels plus the
    external ``InputOp``s, with the ``o1`` intermediate wiring producer→consumer."""
    frag = assemble_block(_two_block_chain(), knobs=_MM_KNOBS, base_knobs={}, kernel_name="k_chain")

    tile_nodes = {nid: n for nid, n in frag.nodes.items() if isinstance(n.op, TileOp)}
    input_nodes = {nid for nid, n in frag.nodes.items() if isinstance(n.op, InputOp)}
    assert set(tile_nodes) == {"o1", "o2"}  # node id = the block's output buffer
    assert input_nodes == {"a", "b", "b2"}  # every external read became an InputOp
    assert frag.outputs == ["o2"]  # the one buffer written but never read internally
    # the o1 intermediate edge wires the producer kernel to the consumer kernel
    assert set(frag.nodes["o2"].inputs) == {"o1", "b2"}
    assert set(frag.nodes["o1"].inputs) == {"a", "b"}
    # producer carries a derived kernel name, the terminal keeps the op name
    assert frag.nodes["o2"].op.name == "k_chain"
    assert frag.nodes["o1"].op.name == "mm1"


def test_multi_block_assemble_is_deterministic() -> None:
    """Same ``TileGraph`` → byte-identical kernel set (node ids, ordering, and each
    kernel's body) — the RF determinism guard the multi-launch path must hold."""
    tg = _two_block_chain()
    a = assemble_block(tg, knobs=_MM_KNOBS, base_knobs={}, kernel_name="k_chain")
    b = assemble_block(tg, knobs=_MM_KNOBS, base_knobs={}, kernel_name="k_chain")
    assert list(a.nodes) == list(b.nodes)
    assert a.outputs == b.outputs
    for nid in a.nodes:
        assert a.nodes[nid].inputs == b.nodes[nid].inputs
        if isinstance(a.nodes[nid].op, TileOp):
            assert a.nodes[nid].op.body.structural_key() == b.nodes[nid].op.body.structural_key()
