"""Tests for the partition planner pass.

After the planner-emits-tiles refactor the planner constructs typed tile
flavors directly; the ``Role`` enum and ``Loop.role`` / ``StridedLoop.role``
fields it used to communicate decisions to downstream passes were
deleted. These tests cover the surviving surface: planner fires on
pointwise, and ``006a_register_tile_planned`` replicates a synthetic
``RegisterTile`` body. Earlier tests that built ``Loop(role=…)`` graphs
were deleted in the same refactor.
"""

from __future__ import annotations

from deplodock.compiler.dtype import F32
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum, Init
from deplodock.compiler.ir.tile.ir import RegisterTile, ThreadTile, TileOp
from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


# --- Planner stub ----------------------------------------------------


def test_planner_fires_on_pointwise(recording_dump, monkeypatch):
    """M16: the planner partitions pointwise kernels too (BLOCK/THREAD
    on output axes). The old ``DEPLODOCK_PLANNER`` env gate is gone."""
    monkeypatch.delenv("DEPLODOCK_PLANNER", raising=False)
    g = Graph()
    _input(g, "x", (4, 8))
    g.add_node(op=loop_op_pointwise(), inputs=["x"], output=Tensor("o", (4, 8)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "partition_loops" in recording_dump.fired_rules("lowering/tile")


def loop_op_pointwise() -> LoopOp:
    i = Axis("i", 4)
    j = Axis("j", 8)
    return LoopOp(
        body=(
            Loop(
                axis=i,
                body=(
                    Loop(
                        axis=j,
                        body=(
                            Load(name="x_v", input="x", index=(Var("i"), Var("j"))),
                            Write(output="o", index=(Var("i"), Var("j")), value="x_v"),
                        ),
                    ),
                ),
            ),
        ),
    )


# --- 006a register_tile_planned (M2 plumbing → M4 active) -----------


def test_006a_replicates_register_tagged_body_loop():
    """When a TileOp body contains a ``RegisterTile``,
    ``006a_register_tile_planned`` replicates the body by the tile's
    axis extent (σ: axis → literal(i)), unwraps the wrapper, and stamps
    ``FM`` / ``FN`` so the legacy ``008_register_tile`` falls through
    via its ``FN-in-knobs`` idempotence check.

    Tests the simplest synthetic case — no Stages, just one
    ``RegisterTile`` wrapping a Load+Write body."""
    a_outer = Axis("a_outer", 8)
    a_reg = Axis("a_reg", 4)
    tile = ThreadTile(
        axes=(a_outer,),
        body=(
            RegisterTile(
                axes=(a_reg,),
                body=(
                    Load(name="x_v", input="x", index=(Var("a_outer"), Var("a_reg"))),
                    Write(output="o", index=(Var("a_outer"), Var("a_reg")), value="x_v"),
                ),
            ),
        ),
    )
    tile_op = TileOp(body=(tile,), name="t")

    g = Graph()
    _input(g, "x", (8, 4))
    g.add_node(op=tile_op, inputs=["x"], output=Tensor("o", (8, 4)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES, select={"split_register_axes"}).run(g)
    new_tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))

    # The RegisterTile wrapper is gone; its body got replicated 4 times.
    new_tile = next(s for s in new_tile_op.body if isinstance(s, ThreadTile))
    register_tiles = [s for s in new_tile.body if isinstance(s, RegisterTile)]
    assert register_tiles == [], "RegisterTile wrapper should have been unwrapped"
    # 4 Loads + 4 Writes after replication.
    body_loads = list(new_tile.body.iter_of_type(Load))
    body_writes = [s for s in new_tile.body if isinstance(s, Write)]
    assert len(body_loads) == 4
    assert len(body_writes) == 4
    # The σ should have substituted ``a_reg`` Var with the literal index
    # in each replica — no surviving ``a_reg`` references.
    for w in body_writes:
        vars_in_index = {v for e in w.index for v in e.free_vars()}
        assert "a_reg" not in vars_in_index, f"unreplicated a_reg in Write: {w.index}"
    # FM gets stamped from the outermost (and only) RegisterTile axis extent.
    assert new_tile_op.knobs.get("FM") == 4


def test_006a_sibling_register_tile_towers_share_keep():
    """The blocked-GEMM nest emits three sibling ``RegisterTile(N_r)``
    towers (Init / K-reduce-Accum / Write) at the same ThreadTile level.
    Each tower's local keep-analysis is incomplete — the Init tower
    doesn't itself reference N_r, so a per-body fold would keep ``acc``
    one name there, while the Accum tower would produce ``acc_0..F-1``.
    Body-global keep aligns the three towers: ``acc`` is keep=True
    everywhere because some sibling's Accum reads N_r → all three
    towers replicate consistently to ``acc_0..F-1``.
    """
    n_outer = Axis("n_outer", 8)
    n_reg = Axis("n_reg", 4)
    k = Axis("k", 16)

    init_tower = RegisterTile(axes=(n_reg,), body=(Init(name="acc", op=ElementwiseImpl("add"), dtype=F32),))
    reduce_tower = Loop(
        axis=k,
        body=(
            Load(name="w", input="w", index=(Var("k"), Var("n_outer"), Var("n_reg"))),
            RegisterTile(axes=(n_reg,), body=(Accum(name="acc", value="w", op=ElementwiseImpl("add")),)),
        ),
    )
    write_tower = RegisterTile(
        axes=(n_reg,),
        body=(Write(output="o", index=(Var("n_outer"), Var("n_reg")), value="acc"),),
    )
    tile = ThreadTile(
        axes=(n_outer,),
        body=(init_tower, reduce_tower, write_tower),
    )
    tile_op = TileOp(body=(tile,), name="t")

    g = Graph()
    _input(g, "w", (16, 8, 4))
    g.add_node(op=tile_op, inputs=["w"], output=Tensor("o", (8, 4)), node_id="o")
    g.inputs = ["w"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES, select={"split_register_axes"}).run(g)
    new_tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))
    new_thread = next(s for s in new_tile_op.body if isinstance(s, ThreadTile))

    assert not any(isinstance(s, RegisterTile) for s in new_thread.body.iter()), "RegisterTile should be fully unwrapped"

    # All four init / write / accum cells should reference a per-cell ``acc<i>``
    # — body-global keep saw the Accum's N_r dep and propagated. (Post-replicate
    # ``rename_ssa_sequential`` canonicalizes ``acc_0..3`` → ``acc0..3``.)
    inits = [s for s in new_thread.body if isinstance(s, Init)]
    init_names = {s.name for s in inits}
    assert len(init_names) == 4 and all(n.startswith("acc") for n in init_names), init_names
    writes = [s for s in new_thread.body if isinstance(s, Write)]
    write_values = {w.value for w in writes}
    assert len(write_values) == 4 and all(n.startswith("acc") for n in write_values), write_values
    # The Init / Accum / Write towers must align on the SAME per-cell name
    # — that's the whole point of body-global keep. If the Init tower kept
    # ``acc`` as one name (local fold), the rendered kernel would have one
    # init with four uses → name collision.
    accums = [s for s in new_thread.body.iter() if isinstance(s, Accum)]
    accum_names = {a.name for a in accums}
    assert init_names == write_values == accum_names, (init_names, write_values, accum_names)


def test_replicator_keeps_n_invariant_loads_once():
    """For a plain matmul with FN > 1, the kernel-IR replicator
    (``010_split_register_axes``) emits ONE ``Load a[m, k]`` (N-invariant
    — kept once) and ``FN`` ``Load b[k, n_r=i]`` (N-dependent — replicated)
    inside the K_i loop body. The replicator's per-stmt dependency
    analysis (``_needs_replication`` reading the ``axis in deps[id(s)]``
    set + the ``keep[name]`` SSA propagation) does the same N-invariant
    cone / N-dependent tail split that the deleted blocked-GEMM builder
    used to hand-code structurally — see
    ``plans/obsolete-blocked-gemm-builder.md`` for the motivation and
    the deletion's Phase 5 commit.

    Bypasses greedy variant selection (other knobs may dominate the
    lazy-score yardstick depending on shape) by calling ``_materialize``
    directly with a hand-picked FN > 1 ``TileParams`` from the planner's
    enumeration, then runs the replicator pass on the result."""
    import importlib  # noqa: PLC0415

    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.graph import Graph  # noqa: PLC0415
    from deplodock.compiler.ir.stmt import Load as _Load  # noqa: PLC0415
    from deplodock.compiler.tensor import Tensor  # noqa: PLC0415

    planner = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")
    replicator = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.kernel.010_split_register_axes")

    m_ax, n_ax, k_ax = Axis("m", 64), Axis("n", 128), Axis("k", 32)
    loop_op = LoopOp(
        body=(
            Loop(
                axis=m_ax,
                body=(
                    Loop(
                        axis=n_ax,
                        body=(
                            Loop(
                                axis=k_ax,
                                body=(
                                    Load(name="a", input="a", index=(Var("m"), Var("k"))),
                                    Load(name="b", input="b", index=(Var("k"), Var("n"))),
                                    Accum(name="acc", value="b", op=ElementwiseImpl("add")),
                                ),
                            ),
                            Write(output="o", index=(Var("m"), Var("n")), value="acc"),
                        ),
                    ),
                ),
            ),
        ),
    )

    plan = planner._plan_kernel(loop_op, Context(compute_capability="sm_80"), kernel_name="k_blk")
    assert plan is not None
    candidates = [p for p in plan.params if p.fn > 1 and p.splitk == 1 and p.br == 1]
    assert candidates, "no FN > 1 variants enumerated for matmul"
    chosen = candidates[0]
    tile_op = planner._materialize(plan, chosen)

    # Run the replicator (and its prerequisite engine plumbing) by
    # constructing a single-node Graph and calling the rule directly.
    g = Graph()
    g.add_node(op=tile_op, inputs=[], output=Tensor(tile_op.name, ()), node_id="op")
    after = replicator.rewrite(g.nodes["op"])
    assert after is not None, "replicator should fire on a FN > 1 matmul TileOp"

    # Count Loads per gmem buffer. ``a`` (M-dep, N-invariant) replicates
    # along the M_r axis (FM cells) but NOT along N_r — the replicator's
    # per-stmt dep analysis sees ``axis a_n_r not in deps[Load a]`` and
    # keeps the M_r-replicated copies without further N_r multiplication.
    # ``b`` (M-invariant, N-dep) replicates along N_r only. So the totals
    # are FM and FN respectively, NOT FM·FN.
    loads_by_buf: dict[str, int] = {}
    for s in after.body.iter():
        if isinstance(s, _Load):
            loads_by_buf[s.input] = loads_by_buf.get(s.input, 0) + 1
    assert loads_by_buf.get("a") == chosen.fm, f"expected {chosen.fm} Load a (FM-replicated, N-invariant), got {loads_by_buf}"
    assert loads_by_buf.get("b") == chosen.fn, f"expected {chosen.fn} Load b (FN-replicated, M-invariant), got {loads_by_buf}"
    # The blocked-layout invariant: total Loads (FM + FN) is much less than
    # the naive product (FM · FN) — what a non-smart replicator would emit.
    assert loads_by_buf["a"] + loads_by_buf["b"] < chosen.fm * chosen.fn or chosen.fm * chosen.fn <= 2
