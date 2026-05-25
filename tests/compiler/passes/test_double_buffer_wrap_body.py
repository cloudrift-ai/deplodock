"""Tests for ``030_use_ring_buffers`` (wrap-body promotion).

The pass walks a Tile body for ``SerialTile(kind="serial_outer")`` whose body
contains a wrap-body ``Stage`` carrying a ``stage_inner`` reduce; it swaps
the ``Stage`` for a ``BufferedStage`` with ``buffer_count=2`` and
``phase = Var(K_o.name) % 2``, and prepends the phase to every Load inside
the consumer body that reads from staged smem.

Tests are end-to-end: build a frontend graph, run ``TILE_PASSES``, walk the
resulting TileOp body and assert the structural rewrite landed.
"""

from __future__ import annotations

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.stmt import Load
from deplodock.compiler.ir.tile.ir import BufferedStage, SerialTile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

# Pin the compile context to sm_80 so cp.async / ring-buffer / pipelining passes
# fire on CI runners (no GPU → ``Context.probe()`` returns cc=(0,0), which gates
# off every wrap-body promotion this file is meant to assert).
_TEST_CTX = Context.from_target((8, 0))


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


def _pin_legacy_matmul_primary(monkeypatch) -> None:
    """Pin planner knobs to the priority_fn legacy primary so downstream
    passes (use_ring_buffers / async_copy / pad_smem) see the staged
    matmul shape they're designed to act on. The score-driven primary
    in 7c321867 picks SPLITK>1 / tiny-FM-FN configs without a useful
    K_o tower for these passes."""
    for knob, value in {"BM": "16", "BN": "16", "FM": "4", "FN": "8", "BK": "64", "SPLITK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{knob}", value)


def _find_kouter(op: TileOp) -> SerialTile | None:
    for s in op.body.iter():
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            return s
    return None


# --- firing tests --------------------------------------------------------


def test_matmul_fires_double_buffer(recording_dump, monkeypatch):
    """Plain matmul → 030_use_ring_buffers fires after 010_stage_inputs."""
    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g, ctx=_TEST_CTX)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "use_ring_buffers" in fired, fired


def test_double_buffer_emits_buffered_stage(monkeypatch):
    """At least one BufferedStage (or subclass) with buffer_count=2 lands
    in the lowered TileOp. Post-015 pipelining the K_o body's stage is an
    AsyncBufferedStage; the phase may be σ-substituted (``(K_o+1) % 2``
    in the main-loop issue, ``Literal(0)`` in the prologue) so the check
    is structural: any buffer_count=2 stage anywhere in the body."""
    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
    op = g2.nodes["o"].op
    buffered = [s for s in op.body.iter() if isinstance(s, BufferedStage)]
    assert buffered, f"no BufferedStage anywhere in the lowered body: {[type(s).__name__ for s in op.body.iter()]}"
    assert all(bs.buffer_count == 2 for bs in buffered), [bs.buffer_count for bs in buffered]


def test_double_buffer_phase_prepended_to_body_loads(monkeypatch):
    """Loads against staged smem buffers carry a leading phase index dim
    (set by 030_use_ring_buffers). Post-015 the consumer body lives as
    siblings of the issue-only stage, so we scan the whole TileOp body
    for staged-smem Loads."""
    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
    op = g2.nodes["o"].op
    staged_names: set[str] = set()
    for s in op.body.iter():
        if isinstance(s, BufferedStage):
            staged_names.update(src.name for src in s.sources)
    assert staged_names, "no staged-smem buffer names found"
    loads_against_smem = [s for s in op.body.iter() if isinstance(s, Load) and s.input in staged_names]
    assert loads_against_smem, "no Loads in body read from staged smem"
    # Every staged-smem Load's leading index dim is a phase expression —
    # either ``K_o % 2`` (the original 010-stamped phase) or a literal
    # ring slot (post-015 σ_last on the epilogue consumer).
    for ld in loads_against_smem:
        assert ld.index, f"empty index on staged Load: {ld}"
        leading = ld.index[0].pretty()
        is_phase = "% 2" in leading or leading.strip("()").isdigit()
        assert is_phase, f"Load {ld.name!r} leading index {leading!r} doesn't look like a ring-slot phase"


# --- idempotence ---------------------------------------------------------


def test_double_buffer_is_idempotent(monkeypatch):
    """Running the rewrite a second time on its own output is a no-op:
    BufferedStage is already present, so the rule skips."""
    import importlib.util
    import pathlib

    from deplodock.compiler.pipeline import RuleSkipped
    from deplodock.compiler.pipeline.passes.lowering.tile import _helpers

    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
    op = g2.nodes["o"].op
    kouter = _find_kouter(op)
    assert kouter is not None
    n_buffered = sum(1 for s in kouter.body if isinstance(s, BufferedStage))
    assert n_buffered > 0

    pass_path = pathlib.Path(_helpers.__file__).parent / "030_use_ring_buffers.py"
    spec = importlib.util.spec_from_file_location("dbl_pass", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.rewrite(_TEST_CTX, g2.nodes["o"])
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "030_use_ring_buffers should be idempotent (BufferedStage already present)"


# --- eligibility regression ----------------------------------------------


def test_small_kouter_not_promoted():
    """K_o extent == 1 should not be promoted (need ≥ 2 ring slots)."""
    # Tiny K so K_o ends up at extent 1 (single chunk).
    g = _build_matmul(m=32, k=32, n=32)
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
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
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g, ctx=_TEST_CTX)
    assert "use_ring_buffers" not in recording_dump.fired_rules("lowering/tile")


def test_buffered_stages_in_tile_validate_under_smem_budget():
    """The promoted TileOp must validate: total per-Source smem
    (deduped by name, accounting for buffer_count) fits in
    ``ctx.max_dynamic_smem``. Pipelined variants emit multiple Stages
    against the same source name — ``TileOp.validate`` dedupes."""
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
    op = g2.nodes["o"].op
    assert op.validate(_TEST_CTX), "lowered TileOp must validate under sm_80 smem budget"
