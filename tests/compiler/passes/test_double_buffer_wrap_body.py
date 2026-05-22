"""Tests for ``010_double_buffer`` (wrap-body promotion).

The pass walks a Tile body for ``SerialTile(kind="serial_outer")`` whose body
contains a wrap-body ``Stage`` carrying a ``stage_inner`` reduce; it swaps
the ``Stage`` for a ``BufferedStage`` with ``buffer_count=2`` and
``phase = Var(K_o.name) % 2``, and prepends the phase to every Load inside
the consumer body that reads from staged smem.

Tests are end-to-end: build a frontend graph, run ``TILE_PASSES``, walk the
resulting TileOp body and assert the structural rewrite landed.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.ir.tile.ir import BufferedStage, SerialTile, Stage, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def _build_matmul(m: int = 128, k: int = 256, n: int = 128) -> Graph:
    g = Graph()
    _input(g, "a", (m, k))
    _input(g, "b", (k, n))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (m, n)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _find_kouter(op: TileOp) -> SerialTile | None:
    for s in op.body.iter():
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            return s
    return None


# --- firing tests --------------------------------------------------------


def test_matmul_fires_double_buffer(recording_dump):
    """Plain matmul → 010_double_buffer fires after 002_stage_inputs."""
    g = _build_matmul()
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "double_buffer" in fired, fired


def test_double_buffer_emits_buffered_stage():
    """The K_o SerialTile body has a BufferedStage with buffer_count=2 and
    phase=Var(K_o.name) % 2 after the pass runs."""
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    kouter = _find_kouter(op)
    assert kouter is not None, "no SerialTile(serial_outer) survived the pipeline"
    buffered = [s for s in kouter.body if isinstance(s, BufferedStage)]
    assert buffered, f"K_o body has no BufferedStage: {[type(s).__name__ for s in kouter.body]}"
    for bs in buffered:
        assert bs.buffer_count == 2, bs.buffer_count
        # phase is "Var(K_o.name) % 2"
        assert bs.phase.pretty() == f"({kouter.axis.name} % 2)", bs.phase.pretty()


def test_double_buffer_phase_prepended_to_body_loads():
    """Loads inside the BufferedStage's wrapped body that read from staged
    smem must have phase as the leading index dim."""
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    kouter = _find_kouter(op)
    assert kouter is not None
    for bs in [s for s in kouter.body if isinstance(s, BufferedStage)]:
        staged = {src.name for src in bs.sources}
        phase_str = bs.phase.pretty()
        loads_against_smem = [s for s in bs.body.iter() if isinstance(s, Load) and s.input in staged]
        assert loads_against_smem, "no Loads in body read from staged smem — pass found nothing to rewrite"
        for ld in loads_against_smem:
            assert ld.index, f"empty index on staged Load: {ld}"
            assert ld.index[0].pretty() == phase_str, (
                f"Load {ld.name!r} from {ld.input!r} missing phase prefix: leading={ld.index[0].pretty()!r}, expected={phase_str!r}"
            )


# --- idempotence ---------------------------------------------------------


def test_double_buffer_is_idempotent():
    """Running the rewrite a second time on its own output is a no-op:
    BufferedStage is already present, so the rule skips."""
    import importlib.util
    import pathlib

    from deplodock.compiler.context import Context
    from deplodock.compiler.pipeline import RuleSkipped
    from deplodock.compiler.pipeline.passes.lowering.tile import _helpers

    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    kouter = _find_kouter(op)
    assert kouter is not None
    n_buffered = sum(1 for s in kouter.body if isinstance(s, BufferedStage))
    assert n_buffered > 0

    pass_path = pathlib.Path(_helpers.__file__).parent / "010_double_buffer.py"
    spec = importlib.util.spec_from_file_location("dbl_pass", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = Context.from_target((8, 0))

    try:
        mod.rewrite(ctx, g2.nodes["o"])
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "010_double_buffer should be idempotent (BufferedStage already present)"


# --- eligibility regression ----------------------------------------------


def test_small_kouter_not_promoted():
    """K_o extent == 1 should not be promoted (need ≥ 2 ring slots)."""
    # Tiny K so K_o ends up at extent 1 (single chunk).
    g = _build_matmul(m=32, k=32, n=32)
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    kouter = _find_kouter(op)
    if kouter is not None and int(kouter.axis.extent) >= 2:
        # Skip: the planner gave us a K_o with enough room — not the case
        # we're testing.
        return
    # Either no K_o at all (planner inlined K) or K_o.extent == 1.
    # In either case, no BufferedStage should appear anywhere.
    assert not any(isinstance(s, BufferedStage) for s in op.body.iter())


def test_no_stage_means_no_promotion(recording_dump):
    """Pointwise kernels have no Stage; 010 must not fire."""
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()
    _input(g, "x", (128, 128))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (128, 128)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "double_buffer" not in recording_dump.fired_rules("lowering/tile")


def test_buffered_stages_in_tile_validate_under_smem_budget():
    """The promoted TileOp must still validate: smem_bytes × 2 must fit
    in ctx.max_dynamic_smem. The pass enforces this; verify the budget
    on the resulting BufferedStage."""
    from deplodock.compiler.context import Context

    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    ctx = Context.from_target((8, 0))
    staged = sum(s.smem_bytes for s in op.body.iter() if isinstance(s, Stage))
    assert staged <= ctx.max_dynamic_smem, f"smem {staged}B exceeds budget {ctx.max_dynamic_smem}B"
