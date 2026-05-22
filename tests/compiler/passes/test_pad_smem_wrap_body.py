"""Tests for ``014_pad_smem`` (per-source bank-conflict padding).

The pass emits a BOOL ``PAD_SMEM`` autotune fork. Under ``PAD_SMEM=True``,
each ``Source`` inside a ``BufferedStage`` / ``AsyncBufferedStage`` gets a
``+1`` pad on the cache dim that drives the worst-case ``max_way`` from
``lane_bank_distribution`` down to 1 (when such a pad exists). Under
``PAD_SMEM=False``, every source stays unpadded.

``TmaBufferedStage`` is skipped — TMA box copies + hardware swizzle don't
tolerate ``+1`` pad and the IR class itself asserts pad-empty.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import BufferedStage, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers


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


def _load_pass():
    pass_path = pathlib.Path(_helpers.__file__).parent / "014_pad_smem.py"
    spec = importlib.util.spec_from_file_location("pad_pass", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _padded_sources(op: TileOp) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for s in op.body.iter():
        if isinstance(s, BufferedStage):
            for src in s.sources:
                out[src.name] = src.pad
    return out


# --- firing tests --------------------------------------------------------


def test_pad_smem_fires_on_matmul(recording_dump):
    g = _build_matmul()
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g)
    assert "pad_smem" in recording_dump.fired_rules("lowering/tile")


def test_greedy_default_picks_pad_on(monkeypatch):
    """With no env pin, the greedy run picks variant 0 — PAD_SMEM=True."""
    monkeypatch.delenv("DEPLODOCK_PAD_SMEM", raising=False)
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    assert op.knobs.get("PAD_SMEM") is True
    pads = _padded_sources(op)
    # At least one source carried a pad fix (the actual choice depends on
    # which knob tuple the planner picked, but for the default matmul
    # configuration at least one stage should benefit).
    assert any(p and any(p) for p in pads.values()), pads


def test_env_pin_false_drops_all_pads(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_PAD_SMEM", "false")
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    assert op.knobs.get("PAD_SMEM") is False
    pads = _padded_sources(op)
    assert all(not p or not any(p) for p in pads.values()), pads


def test_env_pin_true_applies_pad(monkeypatch):
    monkeypatch.setenv("DEPLODOCK_PAD_SMEM", "true")
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    assert op.knobs.get("PAD_SMEM") is True
    pads = _padded_sources(op)
    assert any(p and any(p) for p in pads.values()), pads


# --- idempotence + eligibility -------------------------------------------


def test_pad_smem_is_idempotent():
    """Already-knobbed TileOps skip — the variant fork would otherwise
    re-emit on every pass."""
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    mod = _load_pass()
    ctx = Context.from_target((8, 0))
    try:
        mod.rewrite(ctx, g2.nodes["o"])
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "014_pad_smem must self-skip when PAD_SMEM is already stamped"


def test_no_bank_conflict_means_no_variants():
    """A pointwise kernel has no Stage → no pad-eligible source → skip."""
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()
    _input(g, "x", (128, 128))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (128, 128)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]
    rec: list[tuple[str, str]] = []

    class R:
        def on_rule(self, p, r, rec_, t):
            rec.append((p.name, r.name))

        def on_pass(self, *a):
            pass

    Pipeline.build(TILE_PASSES, dump=R()).run(g)
    assert not any("pad_smem" in r for _, r in rec)


def test_pad_drives_max_way_to_one():
    """The picked pad must zero out the bank conflict it was applied for."""
    from deplodock.compiler.diagnostics.bank_conflicts import lane_bank_distribution
    from deplodock.compiler.ir.stmt import Load
    from deplodock.compiler.ir.tile.ir import ThreadTile

    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g)
    op = g2.nodes["o"].op
    if op.knobs.get("PAD_SMEM") is not True:
        return  # variant order skipped pad; the env-pin test covers this case
    # Locate the ThreadTile + each padded source, recompute bank conflicts
    # post-pad and assert they're all 1.
    thread_axes: tuple = ()
    for s in op.body.iter():
        if isinstance(s, ThreadTile):
            thread_axes = s.axes
            break
    assert thread_axes
    for s in op.body.iter():
        if not isinstance(s, BufferedStage):
            continue
        for src in s.sources:
            if not src.pad or not any(src.pad):
                continue
            extents = src.alloc_extents
            loads = [ld for ld in s.body.iter() if isinstance(ld, Load) and ld.input == src.name]
            for ld in loads:
                cache_idx = ld.index[1:] if len(ld.index) == len(extents) + 1 else ld.index
                dist = lane_bank_distribution(tuple(cache_idx), extents, thread_axes)
                if dist is None:
                    continue
                assert dist.max_way == 1, f"Source {src.name} Load {ld.name} still has max_way={dist.max_way}"
