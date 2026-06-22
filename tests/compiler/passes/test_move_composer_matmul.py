"""Tests for the Phase-2a scalar-matmul move composer.

Covers ``skeleton.lift_matmul`` (SEMIRING regime detection),
``tree.build_matmul_tree`` (generative reduce → thread → register Fork tree),
and ``materialize.build_matmul_tile`` (the K_o/K_i serial tower + masked-axis
guards). A final test drives the whole tile pipeline with the composer enabled.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.dtype import F16, F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Assign
from deplodock.compiler.ir.tile.ir import ATOM_REGISTRY, AtomTile, GridTile, SerialTile, TileOp, WarpTile
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.fork import flatten_leaves
from deplodock.compiler.pipeline.passes.lowering.tile.partition.materialize import build_matmul_tile, build_warp_matmul_tile
from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import eligible_atoms
from deplodock.compiler.pipeline.passes.lowering.tile.partition.skeleton import MatmulSkeleton, PointwiseSkeleton
from deplodock.compiler.pipeline.passes.lowering.tile.partition.tree import build_matmul_tree
from deplodock.compiler.pipeline.passes.lowering.tile.partition.walk import walk_nest

_MM_KEYS = {"MAP_N_THREAD", "MAP_N_REG", "MAP_M_THREAD", "MAP_M_REG", "RED_BK", "RED_FK"}


def _matmul(m: int, n: int, k: int) -> LoopOp:
    """Canonical decomposed matmul ``o[m,n] = sum_k a[m,k]·b[k,n]``."""
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


def _sum_reduce() -> LoopOp:
    """A plain (non-matmul) reduce — single K-indexed load → MONOID, not SEMIRING."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("i", 8),
                body=(
                    Loop(
                        axis=Axis("k", 16),
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("k"))),
                            Accum(name="acc", value="x_v", op=ElementwiseImpl("add")),
                        ),
                    ),
                    Write(output="o", index=(Var("i"),), value="acc"),
                ),
            ),
        ),
    )


def test_lift_matmul_names_axes():
    skel = walk_nest(_matmul(64, 96, 128))
    assert skel is not None
    assert skel.inner_n.extent == 96
    assert skel.outer_m.extent == 64
    # axis names are canonicalized (a0/a1/a2…) by LoopOp normalization
    assert skel.k_extent == 128
    assert skel.k_name == skel.k_loop.axis.name


def test_lift_matmul_rejects_non_matmul_reduce():
    assert not isinstance(walk_nest(_sum_reduce()), MatmulSkeleton)


def test_lift_pointwise_rejects_matmul():
    # the matmul must NOT be claimed by the pointwise regime (it has a reduce)
    assert not isinstance(walk_nest(_matmul(64, 96, 128)), PointwiseSkeleton)


def _matmul_scaled(m: int, n: int, k: int) -> LoopOp:
    """Matmul with a `* c` MAP epilogue (the QK^T scale shape)."""
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
                            Assign(name="s", op=ElementwiseImpl("multiply"), args=("acc", "acc")),
                            Write(output="o", index=(Var("m"), Var("n")), value="s"),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_matmul_with_scale_epilogue_composes_and_forces_splitk1():
    from deplodock.compiler.pipeline.passes.lowering.tile.partition.moves import matmul_reduce_offers  # noqa: PLC0415

    skel = walk_nest(_matmul_scaled(64, 96, 256))
    assert isinstance(skel, MatmulSkeleton), "matmul + MAP epilogue should compose (epilogue rides the output tile)"
    # A MAP epilogue forces SPLITK=1 (cross-CTA atomic-add over a partial would be wrong).
    assert {sk for _, _, sk in matmul_reduce_offers(skel)} == {1}


def _gated_matmul(m: int, n: int, k: int) -> LoopOp:
    """Two same-K matmuls (gate, up) + a fused combine — the gated-MLP shape."""
    return LoopOp(
        body=(
            Loop(
                axis=Axis("m", m),
                body=(
                    Loop(
                        axis=Axis("n", n),
                        body=(
                            Loop(
                                axis=Axis("kg", k),
                                body=(
                                    Load(name="a", input="a", index=(Var("m"), Var("kg"))),
                                    Load(name="g", input="g", index=(Var("kg"), Var("n"))),
                                    Assign(name="pg", op=ElementwiseImpl("multiply"), args=("a", "g")),
                                    Accum(name="acc_g", value="pg", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Loop(
                                axis=Axis("ku", k),
                                body=(
                                    Load(name="a2", input="a", index=(Var("m"), Var("ku"))),
                                    Load(name="u", input="u", index=(Var("ku"), Var("n"))),
                                    Assign(name="pu", op=ElementwiseImpl("multiply"), args=("a2", "u")),
                                    Accum(name="acc_u", value="pu", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Assign(name="o_v", op=ElementwiseImpl("multiply"), args=("acc_g", "acc_u")),
                            Write(output="o", index=(Var("m"), Var("n")), value="o_v"),
                        ),
                    ),
                ),
            ),
        ),
    )


def test_multi_accumulator_matmul_composes():
    # Two same-K matmuls (gate, up) sharing the `a` operand + a fused combine.
    # unify_sibling_reduce_axes collapses the two K axes to one name and fusion
    # merges them into a multi-accumulator reduce, which classifies MONOID and is
    # scheduled on the cooperative path (each accumulator gets its own combine).
    # Either classification is a valid schedule — the point is the composer
    # RECOGNIZES it (not None → legacy fallthrough) rather than dropping it.
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (64, 256), dtype=F16), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("g", (256, 96), dtype=F16), node_id="g")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("u", (256, 96), dtype=F16), node_id="u")
    g.add_node(op=_gated_matmul(64, 96, 256), inputs=["a", "g", "u"], output=Tensor("o", (64, 96), dtype=F16), node_id="o")
    g.inputs = ["a", "g", "u"]
    g.outputs = ["o"]
    skel = walk_nest(g.nodes["o"].op)
    assert skel is not None, "gated MLP (multi-accumulator) should compose, not fall through to legacy"
    # Fusion merges the two same-K contractions into one loop with two Accums.
    from deplodock.compiler.ir.stmt import Body  # noqa: PLC0415

    assert len(list(Body(skel.inner_body).iter_of_type(Accum))) == 2, "both gate + up accumulators present"


def _graph(m: int, n: int, k: int, dtype):
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (m, k), dtype=dtype), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (k, n), dtype=dtype), node_id="b")
    g.add_node(op=_matmul(m, n, k), inputs=["a", "b"], output=Tensor("o", (m, n), dtype=dtype), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


_CTX = Context(compute_capability=(8, 0))


def test_tree_leaves_complete_and_within_budget():
    g = _graph(128, 128, 128, F32)  # fp32 → scalar subtree only
    skel = walk_nest(g.nodes["o"].op)
    tree = build_matmul_tree(skel, loop_op=g.nodes["o"].op, context=_CTX, graph=g, base_knobs={}, kernel_name="k")
    leaves = flatten_leaves([tree])
    assert len(leaves) > 1
    for leaf in leaves:
        kn = leaf.knobs
        assert _MM_KEYS <= set(kn), f"leaf missing matmul knobs: {kn}"
        assert kn["MAP_N_THREAD"] * kn["MAP_M_THREAD"] <= 1024, kn
        assert kn["RED_FK"] * kn["MAP_N_REG"] * kn["MAP_M_REG"] <= 128, kn
        assert 128 % (kn["RED_SPLITK"] * kn["RED_BK"] * kn["RED_FK"]) == 0, f"splitk·bk·fk must divide K: {kn}"


def test_materialize_emits_k_serial_tower():
    skel = walk_nest(_matmul(128, 128, 128))
    knobs = {"MAP_N_THREAD": 8, "MAP_N_REG": 1, "MAP_M_THREAD": 16, "MAP_M_REG": 4, "RED_BK": 32, "RED_FK": 1, "RED_SPLITK": 1}
    tile = build_matmul_tile(skel, knobs, kernel_name="k", base_knobs={})
    assert isinstance(tile, TileOp)
    serials = list(tile.body.iter_of_type(SerialTile))
    kinds = {s.kind for s in serials}
    assert "serial_outer" in kinds and "stage_inner" in kinds, f"K tower missing: {kinds}"
    # K_o = K / BK = 128 / 32 = 4 ; K_i = BK = 32
    outer = next(s for s in serials if s.kind == "serial_outer")
    inner = next(s for s in serials if s.kind == "stage_inner")
    assert outer.axis.extent.as_static() == 4
    assert inner.axis.extent.as_static() == 32


def test_pipeline_uses_composer_for_matmul(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_MOVE_COMPOSER", "1")
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (64, 128)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (128, 96)), node_id="b")
    g.add_node(op=_matmul(64, 96, 128), inputs=["a", "b"], output=Tensor("o", (64, 96)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    out = Pipeline.build(TILE_PASSES).run(g)
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    assert tile_op.knobs.get("RED_BK", 0) >= 1, "composer should stamp the reduce-tile vocabulary"
    assert tile_op.knobs.get("BN", 0) == 0, "legacy planner must not have run for a composer-covered matmul"
    assert isinstance(tile_op.body[0], GridTile)


# --- Tensorize (warp-tier MMA) ---------------------------------------------


def test_eligible_atoms_fp16_yes_fp32_no():
    g16 = _graph(64, 128, 64, F16)
    assert eligible_atoms(g16.nodes["o"].op, _CTX, g16), "fp16 matmul should admit a tensor-core atom"
    g32 = _graph(64, 128, 64, F32)
    assert not eligible_atoms(g32.nodes["o"].op, _CTX, g32), "fp32 matmul admits no atom"


def test_matmul_tree_offers_warp_for_fp16():
    g = _graph(64, 128, 64, F16)
    skel = walk_nest(g.nodes["o"].op)
    tree = build_matmul_tree(skel, loop_op=g.nodes["o"].op, context=_CTX, graph=g, base_knobs={}, kernel_name="k")
    leaves = flatten_leaves([tree])
    warp_leaves = [leaf for leaf in leaves if leaf.knobs.get("TC_ATOM")]
    assert warp_leaves, "fp16 matmul tree should offer warp (tensorize) leaves"
    # The materialized TileOp carries the legacy MMA knob (the downstream tier
    # discriminator); the Fork-leaf identity stays greenfield TC_*.
    tile = warp_leaves[0].expand()[0]
    assert tile.knobs["MMA"] == warp_leaves[0].knobs["TC_ATOM"]


def test_build_warp_tile_emits_warp_atom_and_mma_knob():
    g = _graph(64, 128, 64, F16)
    skel = walk_nest(g.nodes["o"].op)
    atom = ATOM_REGISTRY["mma_m16n8k16_f16"]
    # 64x128x64 / atom(16,8,16): cells_m=4, cells_n=16, kc=4 → wm=2,wn=2 ; pm=2,pn=8
    knobs = {"TC_ATOM": atom.name, "WARP_M": 2, "WARP_N": 2, "TC_REG_M": 2, "TC_REG_N": 4, "TC_BK": 4}
    tile = build_warp_matmul_tile(skel, atom, knobs, kernel_name="k", base_knobs={})
    assert isinstance(tile, TileOp)
    assert tile.knobs["MMA"] == atom.name, "warp tile must carry the MMA knob (020/005/is_warp tier discriminator)"
    assert list(tile.body.iter_of_type(WarpTile)), "warp tower must nest a WarpTile"
    atoms = list(tile.body.iter_of_type(AtomTile))
    assert atoms and atoms[0].atom is atom, "warp tower must nest an AtomTile carrying the atom spec"


# --- Split-K -----------------------------------------------------------------


def test_build_matmul_tile_splitk_adds_grid_axis():
    # M=64 (tile 16·4=64 → M_b=1), N=64 (tile 8 → N_b=8); only the K_s axis has
    # extent 4. (Axis names canonicalize to a0/a1/… so match on extent, not name.)
    skel = walk_nest(_matmul(64, 64, 256))
    base = {"MAP_N_THREAD": 8, "MAP_N_REG": 1, "MAP_M_THREAD": 16, "MAP_M_REG": 4, "RED_BK": 1, "RED_FK": 1}
    g1 = list(build_matmul_tile(skel, {**base, "RED_SPLITK": 1}, kernel_name="k", base_knobs={}).body.iter_of_type(GridTile))
    g4 = list(build_matmul_tile(skel, {**base, "RED_SPLITK": 4}, kernel_name="k", base_knobs={}).body.iter_of_type(GridTile))
    n1 = sum(len(g.axes) for g in g1)
    extents4 = [ax.extent.as_static() for g in g4 for ax in g.axes]
    assert sum(len(g.axes) for g in g4) == n1 + 1, "split-K must add exactly one grid axis"
    assert 4 in extents4, "the added grid axis has extent SPLITK=4"
