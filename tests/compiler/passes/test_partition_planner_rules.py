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

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.elementwise import ElementwiseImpl
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.loop import Axis, Load, Loop, LoopOp, Write
from deplodock.compiler.ir.stmt import Accum
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

    Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
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
    # Need BOTH FM > 1 and FN > 1 to exercise the blocked-layout invariant
    # (``a + b < a·b``); a single-axis register tile (FM=1 or FN=1) trivially
    # satisfies the equality, and (FM=2, FN=2) is the degenerate ``a·b == a+b``
    # boundary. Pin to (FM=4, FN=4) so the strict ``<`` assertion fires and
    # the test stays decoupled from sibling-ordering changes in the prior.
    candidates = [p for p in plan.params if p["FM"] == 4 and p["FN"] == 4 and p["SPLITK"] == 1 and p["BR"] == 1]
    assert candidates, "no FM=4, FN=4 variant enumerated for matmul"
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
    assert loads_by_buf.get("a") == chosen["FM"], f"expected {chosen['FM']} Load a (FM-replicated, N-invariant), got {loads_by_buf}"
    assert loads_by_buf.get("b") == chosen["FN"], f"expected {chosen['FN']} Load b (FN-replicated, M-invariant), got {loads_by_buf}"
    # The blocked-layout invariant: total Loads (FM + FN) is much less than
    # the naive product (FM · FN) — what a non-smart replicator would emit.
    assert loads_by_buf["a"] + loads_by_buf["b"] < chosen["FM"] * chosen["FN"] or chosen["FM"] * chosen["FN"] <= 2
