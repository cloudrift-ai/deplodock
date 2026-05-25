"""Tests for ``060_use_async_copy`` (wrap-body BufferedStage → AsyncBufferedStage promotion).

The pass walks for ``BufferedStage`` inside ``SerialTile(serial_outer)`` and
promotes to ``AsyncBufferedStage(pipeline_depth=1)`` when the target supports
cp.async (sm_80+). Materialization (in ``100_materialize_tile._emit_stage``)
emits ``CpAsyncCopy`` per Source + ``CpAsyncCommit + CpAsyncWait(0) + Sync``
at the wrap boundary.
"""

from __future__ import annotations

import importlib.util
import pathlib

from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.tile.ir import AsyncBufferedStage, BufferedStage, SerialTile, TileOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline, RuleSkipped
from deplodock.compiler.pipeline.passes.lowering.tile import _helpers

# Pin the compile context to sm_80 so cp.async fires on CI runners (no GPU →
# ``Context.probe()`` returns cc=(0,0), which gates off the cp.async promotion
# this file is meant to assert).
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


def _load_pass():
    pass_path = pathlib.Path(_helpers.__file__).parent / "060_use_async_copy.py"
    spec = importlib.util.spec_from_file_location("async_pass", pass_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _find_kouter(op: TileOp) -> SerialTile | None:
    for s in op.body.iter():
        if isinstance(s, SerialTile) and s.kind == "serial_outer":
            return s
    return None


def _pin_legacy_matmul_primary(monkeypatch) -> None:
    """Pin planner knobs to the priority_fn legacy primary so async_copy /
    pipelined_async_stage downstream passes see the staged matmul shape
    they're designed to act on. Score-driven primary picks SPLITK>1 /
    tiny-cells configs that skip the staged K_o tower."""
    for knob, value in {"BM": "16", "BN": "16", "FM": "4", "FN": "8", "BK": "64", "SPLITK": "1"}.items():
        monkeypatch.setenv(f"DEPLODOCK_{knob}", value)


# --- firing tests --------------------------------------------------------


def test_matmul_fires_async_copy(recording_dump, monkeypatch):
    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    Pipeline.build(TILE_PASSES, dump=recording_dump).run(g, ctx=_TEST_CTX)
    fired = recording_dump.fired_rules("lowering/tile")
    assert "use_async_copy" in fired, fired


def test_async_copy_emits_async_buffered_stage(monkeypatch):
    """At least one AsyncBufferedStage with buffer_count=2 lands in the
    lowered TileOp. Post-015 pipelining wraps two issue-only stages
    (prologue + steady-state issue, pipeline_depth=2) around the original
    stage; the structural assertion is "any async stage present" rather
    than the specific pre-015 depth=1 shape."""
    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
    op = g2.nodes["o"].op
    async_stages = [s for s in op.body.iter() if isinstance(s, AsyncBufferedStage)]
    assert async_stages, "no AsyncBufferedStage anywhere in the body"
    assert all(st.buffer_count == 2 for st in async_stages), [st.buffer_count for st in async_stages]


def test_async_copy_preserves_buffered_fields():
    """The subclass swap is the only change — buffer_count / phase /
    sources / body pass through unchanged."""
    from deplodock.compiler.ir.axis import Axis
    from deplodock.compiler.ir.elementwise import ElementwiseImpl
    from deplodock.compiler.ir.expr import Literal, Var
    from deplodock.compiler.ir.stmt import Accum, Body, Load
    from deplodock.compiler.ir.tile.ir import (
        CacheDim,
        GridTile,
        SerialTile,
        Source,
        ThreadTile,
    )

    # Hand-build a minimal TileOp with a BufferedStage inside a serial_outer.
    a_src = Source(
        name="a_smem",
        buf="a",
        cache_dims=(CacheDim(axis=Axis("ci", 4), source_dim=0),),
        origin=(Literal(0, "int"),),
    )
    K_i = Axis("K_i", 4)
    K_o = Axis("K_o", 8)
    phase = Var(K_o.name) % Literal(2, "int")
    reduce = SerialTile(
        axis=K_i,
        body=Body(
            (
                Load(name="x", input="a_smem", index=(phase, Var(K_i.name))),
                Accum(name="acc", value="x", op=ElementwiseImpl("add"), dtype=None),
            )
        ),
        kind="stage_inner",
    )
    buffered = BufferedStage(sources=(a_src,), body=Body((reduce,)), buffer_count=2, phase=phase)
    kouter = SerialTile(axis=K_o, body=Body((buffered,)), kind="serial_outer")
    thread = ThreadTile(axes=(Axis("t", 32),), body=Body((kouter,)))
    grid = GridTile(axes=(Axis("g", 1),), body=Body((thread,)))
    op = TileOp(body=Body((grid,)), name="t", knobs={})

    class FakeNode:
        def __init__(self, op):
            self.op = op
            self.inputs = []
            self.outputs = ["t"]

    mod = _load_pass()
    new_op = mod.rewrite(_TEST_CTX, FakeNode(op))
    # Locate the lone BufferedStage in the input (post-normalize) and its
    # AsyncBufferedStage counterpart in the output to compare structurally.
    pre_kouter = _find_kouter(op)
    pre_buffered = [s for s in pre_kouter.body if isinstance(s, BufferedStage)]
    assert len(pre_buffered) == 1
    new_kouter = _find_kouter(new_op)
    new_async = [s for s in new_kouter.body if isinstance(s, AsyncBufferedStage)]
    assert len(new_async) == 1
    promoted = new_async[0]
    pre = pre_buffered[0]
    assert promoted.buffer_count == pre.buffer_count
    assert promoted.phase.pretty() == pre.phase.pretty()
    assert tuple(s.name for s in promoted.sources) == tuple(s.name for s in pre.sources)
    assert tuple(s.buf for s in promoted.sources) == tuple(s.buf for s in pre.sources)
    assert promoted.pipeline_depth == 1


# --- materializer end-to-end --------------------------------------------


def test_kernel_source_contains_cp_async_commit_wait_sync(monkeypatch):
    """The materializer's wrap-boundary contract: CpAsyncCopy + Commit +
    Wait(0) + Sync per AsyncBufferedStage with pipeline_depth=1."""
    from deplodock.compiler.pipeline import CUDA_PASSES

    _pin_legacy_matmul_primary(monkeypatch)
    g = _build_matmul()
    g2 = Pipeline.build(CUDA_PASSES).run(g, ctx=_TEST_CTX)
    src = g2.nodes["o"].op.kernel_source
    assert "cp.async.ca.shared.global" in src
    assert "cp.async.commit_group" in src
    assert "cp.async.wait_group 0" in src
    # The Sync after the wait — every wrap-body async stage emits this.
    # (Existence is enough; materializer's drop_redundant_syncs may collapse
    # back-to-back syncs but at least one sync must follow.)
    after_wait = src.split("cp.async.wait_group 0", 1)[1]
    assert "__syncthreads" in after_wait


# --- idempotence + eligibility regression --------------------------------


def test_async_copy_is_idempotent():
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)
    mod = _load_pass()
    try:
        mod.rewrite(_TEST_CTX, g2.nodes["o"])
        raised = False
    except RuleSkipped:
        raised = True
    assert raised, "060_use_async_copy should be idempotent — AsyncBufferedStage already present"


def test_arch_below_sm80_rejected():
    """sm_75 (Turing) and earlier don't have cp.async — pass must skip."""
    g = _build_matmul()
    g2 = Pipeline.build(TILE_PASSES).run(g, ctx=_TEST_CTX)  # ends in AsyncBufferedStage on sm_80+
    # Re-run the rule on a context with cc < (8, 0) — must skip.
    mod = _load_pass()
    ctx = Context.from_target((7, 5))
    try:
        mod.rewrite(ctx, g2.nodes["o"])
        raised = False
        msg = ""
    except RuleSkipped as e:
        raised = True
        msg = str(e)
    assert raised, "060_use_async_copy should reject sm_75"
    assert "compute capability" in msg.lower(), msg


def test_no_buffered_stage_means_no_promotion():
    """Pointwise has no Stage; 010 doesn't run; 013 has nothing to promote."""
    from deplodock.compiler.ir.tensor.ir import ElementwiseOp

    g = Graph()
    _input(g, "x", (128, 128))
    g.add_node(op=ElementwiseOp("relu"), inputs=["x"], output=Tensor("o", (128, 128)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    class R:
        def __init__(self):
            self.fired = []

        def on_rule(self, p, r, rec, t):
            self.fired.append((p.name, r.name))

        def on_pass(self, *a):
            pass

    r = R()
    Pipeline.build(TILE_PASSES, dump=r).run(g, ctx=_TEST_CTX)
    fired = {name for _, name in r.fired}
    assert not any("use_async_copy" in name for name in fired), fired
