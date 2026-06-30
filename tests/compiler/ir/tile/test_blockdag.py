"""Derived-projection tests for the block-DAG Tile IR (``ir/tile/ir.py``).

The whole point of the new IR is that ``reads`` / ``writes`` / ``carrier`` /
``atom`` / ``edges`` are *projections* of the body — never stored. These tests
pin the projections against hand-built blocks so they cannot silently drift.
"""

from __future__ import annotations

from emmy.compiler.dtype import F16, F32
from emmy.compiler.ir.algebra import AlgebraKind
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt.blocks import Loop
from emmy.compiler.ir.stmt.leaves import Accum, Assign, Load, Write
from emmy.compiler.ir.tile.ir import (
    AddrKind,
    Binding,
    Block,
    Buffer,
    Edge,
    Placement,
    Schedule,
    Space,
    TileGraph,
    Transport,
    classify_access,
)


def _mul(a, b):
    return BinaryExpr("*", a, b)


def _add(a, b):
    return BinaryExpr("+", a, b)


# --- classify_access ---------------------------------------------------------


def test_classify_identity_affine():
    acc = classify_access((Var("i"), Var("k")), frozenset({"i", "k"}))
    assert acc.kind is AddrKind.AFFINE
    assert acc.axes == ("i", "k")
    assert acc.dims == (0, 1)
    assert acc.block == (1, 1)
    assert acc.free_axes() == frozenset({"i", "k"})


def test_classify_blocked_stride():
    # i*64 + i_t  along dim 0 (a tiled row coordinate)
    idx = (_add(_mul(Var("i_b"), Literal(64, "int")), Var("i_t")), Var("k"))
    acc = classify_access(idx, frozenset({"i_b", "i_t", "k"}))
    assert acc.kind is AddrKind.AFFINE
    # two axes land on dim 0 with their stride multipliers
    by_axis = dict(zip(acc.axes, zip(acc.dims, acc.block, strict=True), strict=True))
    assert by_axis["i_b"] == (0, 64)
    assert by_axis["i_t"] == (0, 1)
    assert by_axis["k"] == (1, 1)


def test_classify_cta_uniform_offset_folds():
    # head*stride is CTA-uniform (head ∉ domain) — it folds into the dim offset.
    idx = (_add(_mul(Var("head"), Literal(128, "int")), Var("n")),)
    acc = classify_access(idx, frozenset({"n"}))
    assert acc.kind is AddrKind.AFFINE
    assert acc.axes == ("n",)
    assert acc.dims == (0,)
    assert acc.offset[0].free_vars() == frozenset({"head"})


def test_classify_template_fallback_on_modulo():
    # collapsed-reshape view: (x // W, x % W) is not affine → TEMPLATE
    idx = (BinaryExpr("//", Var("x"), Literal(8, "int")), BinaryExpr("%", Var("x"), Literal(8, "int")))
    acc = classify_access(idx, frozenset({"x"}))
    assert acc.kind is AddrKind.TEMPLATE
    assert acc.template == idx
    assert acc.free_axes() == frozenset({"x"})


# --- Block projections -------------------------------------------------------


def _matmul_block(M=64, N=64, K=64) -> Block:
    i, j, k = Axis("i", M), Axis("j", N), Axis("k", K)
    compute = (
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
    )
    return Block(name="mm", domain=(i, j, k), compute=compute)


def test_block_reads_writes():
    b = _matmul_block()
    read_bufs = {p.buffer for p in b.reads}
    assert read_bufs == {"a", "b"}
    assert [p.buffer for p in b.writes] == ["c"]
    # every read is affine over the domain
    assert all(p.access.kind is AddrKind.AFFINE for p in b.reads)


def test_block_carrier_semiring():
    b = _matmul_block()
    c = b.carrier
    assert c is not None
    assert c.kind is AlgebraKind.SEMIRING
    assert c.associative and c.commutative
    assert c.mask is None  # static K


def test_block_atom_none_for_scalar():
    assert _matmul_block().atom is None


def test_pointwise_block_no_carrier():
    n = Axis("n", 128)
    compute = (
        Load(name="x_v", input="x", index=(Var("n"),)),
        Assign(name="y", op=ElementwiseImpl("relu"), args=("x_v",)),
        Write(output="y", index=(Var("n"),), value="y"),
    )
    b = Block(name="pw", domain=(n,), compute=compute)
    assert b.carrier is None
    assert {p.buffer for p in b.reads} == {"x"}


# --- TileGraph edge topology -------------------------------------------------


def test_edges_input_source_and_intermediate():
    n = Axis("n", 128)
    # producer: xn[n] = norm(x[n]); consumer: y[n] = xn[n] + b[n]
    producer = Block(
        name="prod",
        domain=(n,),
        compute=(
            Load(name="x_v", input="x", index=(Var("n"),)),
            Assign(name="xn", op=ElementwiseImpl("relu"), args=("x_v",)),
            Write(output="xn", index=(Var("n"),), value="xn"),
        ),
    )
    consumer = Block(
        name="cons",
        domain=(n,),
        compute=(
            Load(name="xn_v", input="xn", index=(Var("n"),)),
            Load(name="b_v", input="b", index=(Var("n"),)),
            Assign(name="o", op=ElementwiseImpl("add"), args=("xn_v", "b_v")),
            Write(output="y", index=(Var("n"),), value="o"),
        ),
    )
    g = TileGraph(
        name="g",
        buffers={
            "x": Buffer("x", (Literal(128, "int"),), F32),
            "b": Buffer("b", (Literal(128, "int"),), F32),
            "xn": Buffer("xn", (Literal(128, "int"),), F32, space=Space.GMEM),
            "y": Buffer("y", (Literal(128, "int"),), F32),
        },
        blocks=(producer, consumer),
        schedule=Schedule(),
    )
    edges = set(g.edges)
    assert Edge(src="x", dst="prod", buffer="x") in edges  # input-source edge
    assert Edge(src="prod", dst="cons", buffer="xn") in edges  # intermediate def-use
    assert Edge(src="b", dst="cons", buffer="b") in edges  # input-source edge
    # the producer→consumer intermediate edge names the writer block, not the buffer
    assert {e for e in edges if e.buffer == "xn"} == {Edge(src="prod", dst="cons", buffer="xn")}


def test_block_self_read_is_not_edge():
    # an accumulator (block reads what it writes) must not produce a self-edge
    b = _matmul_block()
    g = TileGraph(
        name="g",
        buffers={
            "a": Buffer("a", (Literal(64, "int"), Literal(64, "int")), F16),
            "b": Buffer("b", (Literal(64, "int"), Literal(64, "int")), F16),
            "c": Buffer("c", (Literal(64, "int"), Literal(64, "int")), F32),
        },
        blocks=(b,),
        schedule=Schedule(),
    )
    assert all(e.dst != e.src for e in g.edges)
    assert {e.buffer for e in g.edges} == {"a", "b"}


# --- pretty rendering --------------------------------------------------------


def test_tilegraph_pretty_readable():
    # A matmul block with a bound domain + a staged read should render as a
    # readable multi-line listing — not the nested-dataclass repr.
    b = _matmul_block()
    g = TileGraph(
        name="mm_g",
        buffers={
            "a": Buffer("a", (Literal(64, "int"), Literal(64, "int")), F16),
            "b": Buffer("b", (Literal(64, "int"), Literal(64, "int")), F16),
            "c": Buffer("c", (Literal(64, "int"), Literal(64, "int")), F32),
        },
        blocks=(b,),
        schedule=Schedule(
            binding={"i": Binding.GRID, "j": Binding.THREAD},
            staged={Edge(src="a", dst="mm", buffer="a"): Transport.SYNC},
        ),
    )
    text = "\n".join(g.pretty())
    # sections present
    assert "buffers:" in text
    assert "a: f16[64, 64]" in text
    assert "block mm [" in text
    # bindings fold into the block domain header
    assert "i:64=grid" in text and "j:64=thread" in text
    # compute body rendered via the shared stmt pretty-printer
    assert "for k in 0..64" in text
    # non-empty schedule field surfaces; edge keys are buffer:src->dst
    assert "schedule:" in text
    assert "staged: a:a->mm=sync" in text
    # derived edges listed
    assert "a: a -> mm" in text


def test_schedule_pretty_empty_is_blank():
    assert Schedule().pretty() == []


# --- derived edge placement --------------------------------------------------


def _prod_cons_graph(schedule: Schedule) -> TileGraph:
    """``xn = relu(x)`` (block ``prod``) → ``y = xn + b`` (block ``cons``), with the
    given ``schedule`` — the chain the placement query reads off."""
    n = Axis("n", 128)
    producer = Block(
        name="prod",
        domain=(n,),
        compute=(
            Load(name="x_v", input="x", index=(Var("n"),)),
            Assign(name="xn", op=ElementwiseImpl("relu"), args=("x_v",)),
            Write(output="xn", index=(Var("n"),), value="xn"),
        ),
    )
    consumer = Block(
        name="cons",
        domain=(n,),
        compute=(
            Load(name="xn_v", input="xn", index=(Var("n"),)),
            Load(name="b_v", input="b", index=(Var("n"),)),
            Assign(name="o", op=ElementwiseImpl("add"), args=("xn_v", "b_v")),
            Write(output="y", index=(Var("n"),), value="o"),
        ),
    )
    one = (Literal(128, "int"),)
    return TileGraph(
        name="g",
        buffers={k: Buffer(k, one, F32) for k in ("x", "b", "xn", "y")},
        blocks=(producer, consumer),
        schedule=schedule,
    )


def test_placement_inline_for_unstaged_input_read():
    # an input-source read with no staging / cut is INLINE (gmem-direct in body)
    g = _prod_cons_graph(Schedule())
    assert g.placement(Edge(src="x", dst="prod", buffer="x")) is Placement.INLINE
    assert g.placement(Edge(src="b", dst="cons", buffer="b")) is Placement.INLINE


def test_placement_smem_when_staged():
    e = Edge(src="b", dst="cons", buffer="b")
    g = _prod_cons_graph(Schedule(staged={e: Transport.SYNC}))
    assert g.placement(e) is Placement.SMEM
    # an un-staged sibling input read stays INLINE
    assert g.placement(Edge(src="x", dst="prod", buffer="x")) is Placement.INLINE


def test_placement_gmem_for_cross_block_edge():
    # a block->block edge is materialized in gmem — by default each block is its own
    # launch group (v1 two-launch cut), so the xn intermediate defaults to GMEM...
    assert _prod_cons_graph(Schedule()).placement(Edge(src="prod", dst="cons", buffer="xn")) is Placement.GMEM
    # ...and an explicit differing launch group is GMEM too
    g = _prod_cons_graph(Schedule(launch={"prod": 0, "cons": 1}))
    assert g.placement(Edge(src="prod", dst="cons", buffer="xn")) is Placement.GMEM
    assert g.placement(Edge(src="b", dst="cons", buffer="b")) is Placement.INLINE


def test_placement_gmem_dominates_smem_on_cut_edge():
    # a cross-group edge whose consumer ALSO stages its read is still GMEM (the
    # buffer lives in gmem; the smem stage is the consumer's read transport)
    e = Edge(src="prod", dst="cons", buffer="xn")
    g = _prod_cons_graph(Schedule(launch={"prod": 0, "cons": 1}, staged={e: Transport.SYNC}))
    assert g.placement(e) is Placement.GMEM


def test_place_edge_round_trips_with_placement():
    # place_edge writes the Schedule fields a placement implies; placement reads them
    # back — the two are inverse over the block→block edge.
    g = _prod_cons_graph(Schedule())
    xn = Edge(src="prod", dst="cons", buffer="xn")
    for p in (Placement.GMEM, Placement.SMEM, Placement.INLINE):
        assert g.place_edge(xn, p).placement(xn) is p


def test_place_edge_smem_stages_and_co_locates():
    # SMEM fuses: one launch group + the edge staged (the producer fills an smem slab
    # inside one kernel, the consumer reads it)
    g = _prod_cons_graph(Schedule())
    xn = Edge(src="prod", dst="cons", buffer="xn")
    placed = g.place_edge(xn, Placement.SMEM)
    assert placed.schedule.launch["prod"] == placed.schedule.launch["cons"]
    assert placed.schedule.staged[xn] is Transport.SYNC


def test_place_edge_gmem_splits_launch_groups():
    # GMEM cuts: producer and consumer in different launch groups, no staging
    g = _prod_cons_graph(Schedule())
    xn = Edge(src="prod", dst="cons", buffer="xn")
    placed = g.place_edge(xn, Placement.GMEM)
    assert placed.schedule.launch["prod"] != placed.schedule.launch["cons"]
    assert xn not in placed.schedule.staged
