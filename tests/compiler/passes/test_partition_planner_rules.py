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


def test_planner_emits_register_blocked_structure():
    """The blocked-GEMM materialization path produces sibling
    RegisterTile(N_r) wrappers around the Write and inside the K-tower's
    K_i body for a plain matmul. The K_i body has the M-axis Load BEFORE
    the inner RegisterTile(N_r) wrapper (the N-invariant cone, shared
    across F_N cells); the Write sits in its own RegisterTile(N_r).

    Bypasses greedy variant selection (other knobs may dominate the
    lazy-score yardstick depending on shape) by calling ``_materialize``
    directly with a hand-picked ``TileParams(reg_block=True)`` from the
    planner's enumeration. The planner's blocked emit is what we want
    to lock down here — Fork-tree priority is the M3 concern.
    """
    import importlib  # noqa: PLC0415

    from deplodock.compiler.context import Context  # noqa: PLC0415
    from deplodock.compiler.ir.tile.ir import SerialTile  # noqa: PLC0415

    planner = importlib.import_module("deplodock.compiler.pipeline.passes.lowering.tile.010_partition_loops")

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
    blocked_params = [p for p in plan.params if p.reg_block and p.fn > 1]
    assert blocked_params, "no reg_block=True variants enumerated for matmul"
    chosen = blocked_params[0]
    tile_op = planner._materialize(plan, chosen)

    assert tile_op.knobs.get("REG_BLOCK") is True

    thread = next(iter(tile_op.body.iter_of_type(ThreadTile)))
    layer_body: tuple = tuple(thread.body)
    # Descend through any outer M_r RegisterTile (FM > 1 with M_r wrapping
    # the N_r sibling towers).
    if len(layer_body) == 1 and isinstance(layer_body[0], RegisterTile):
        layer_body = tuple(layer_body[0].body)

    # Init isn't in the source LoopOp body (Accum's identity is implicit) so
    # the planner doesn't emit a separate Init tower; only the Write tower
    # appears at this stage. 020_place_inits adds explicit Inits at the
    # ThreadTile scope later, after 010_split_register_axes unwraps the
    # per-cell RegisterTile in the K_i body.
    top_register_tiles = [s for s in layer_body if isinstance(s, RegisterTile)]
    assert len(top_register_tiles) == 1, [type(s).__name__ for s in layer_body]
    write_tower = top_register_tiles[0]
    assert any(isinstance(s, Write) for s in write_tower.body)

    # K-tower sibling has its N-dep tail wrapped in its own RegisterTile(N_r).
    k_towers = [s for s in layer_body if isinstance(s, SerialTile)]
    assert k_towers, "expected K-tower SerialTile sibling of the Write tower"
    nested = [s for s in k_towers[0].body.iter() if isinstance(s, RegisterTile)]
    assert nested, "K-tower body should hold a RegisterTile(N_r) for the N-dep tail"
