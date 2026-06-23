"""Tests that reduction-related compiler rules fire on representative
graphs, plus scoped unit tests for the cooperative-reduce rule's
multi-Accum independence check.

Uses the ``recording_dump`` fixture (see ``conftest.py``) to collect
rule names of every rewrite, with numeric ordering prefix stripped, so
reordering rule files doesn't break these tests.
"""

from __future__ import annotations

from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.expr import Var
from deplodock.compiler.ir.frontend.ir import MatmulOp
from deplodock.compiler.ir.kernel.ir import TreeHalve, WarpShuffle
from deplodock.compiler.ir.stmt import Accum, Assign, Load
from deplodock.compiler.ir.tensor.ir import ReduceOp
from deplodock.compiler.ir.tile.ir import StageBundle, ThreadTile
from deplodock.compiler.pipeline import KERNEL_PASSES, TILE_PASSES, Pipeline
from deplodock.compiler.pipeline.passes.lowering.kernel._helpers import accums_independent as _accums_independent
from deplodock.compiler.pipeline.passes.lowering.kernel._helpers import reduce_body_has_coupled_accum


def _input(g: Graph, name: str, shape: tuple) -> str:
    return g.add_node(op=InputOp(), inputs=[], output=Tensor(name, shape), node_id=name)


def _tile_has_combine(g: Graph) -> bool:
    """True iff some Accum reduces over an enclosing ThreadTile axis —
    the structural signal that cross-thread reduction will be emitted
    by ``100_materialize_tile``. (Pre-refactor variants: "Tile body
    contains a ``Monoid`` stmt", then "ThreadTile.cooperative_axes
    set"; both are gone — cooperativity now lives on ``Accum.axes``
    and the materializer / escape-analysis helper recovers it via
    ``Accum.axes ∩ ThreadTile.axes``.)"""
    for node in g.nodes.values():
        body = getattr(node.op, "body", None)
        if body is None:
            continue
        for tt in body.iter():
            if not isinstance(tt, ThreadTile):
                continue
            tt_axis_names = frozenset(ax.name for ax in tt.axes)
            for s in tt.body.iter():
                if isinstance(s, Accum) and tt_axis_names & frozenset(s.axes):
                    return True
    return False


# --- cooperative-reduce firing on frontend graphs -------------------
# Triggers on single-buffer reductions whose first reduce-axis extent is
# ≥ WARP_SIZE (32) — the threshold dropped from BLOCK_SIZE so softmax /
# rmsnorm rows in the 32–128 range get a parallel reduce instead of
# every thread redundantly walking the row.
#
# Cooperative coordination today lives inside ``001_launch_geometry``
# (folded from the deleted ``002_cooperative_reduce``). The structural
# signal is "Tile body contains Monoid"; assertions check via
# :func:`_tile_has_combine` rather than rule-name firing.


_M, _K, _N = 32, 32, 32


def test_long_axis_sum_fires_cooperative_reduce(recording_dump):
    """``sum(x, axis=-1)`` with K=256 → cooperative_reduce fires; matmul-
    shape rule (chunk_matmul_k) does not (single-buffer reduce)."""
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    fired = recording_dump.fired_rules("lowering/tile/enumeration")
    assert _tile_has_combine(out)
    assert "chunk_matmul_k" not in fired


def test_short_axis_sum_does_not_fire_cooperative_reduce(recording_dump):
    """K=16 < WARP_SIZE → cooperative-reduce does not fire (too small
    to stage a meaningful cross-thread tree-halve)."""
    g = Graph()
    _input(g, "x", (4, 16))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert not _tile_has_combine(out)


def test_warp_sized_axis_fires_cooperative_reduce(recording_dump):
    """K=32 ≥ WARP_SIZE → cooperative-reduce fires with a 32-thread
    cooperative block (the gate was lowered from BLOCK_SIZE to
    WARP_SIZE so K∈[32, BLOCK_SIZE) gets a parallel reduce instead of
    every thread redundantly walking the row sequentially)."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert _tile_has_combine(out)


def test_warp_cooperative_skips_stage_inputs(recording_dump):
    """K=32 → cooperative tile has 32 threads (one warp); stage_inputs
    must skip so the kernel stays smem-free (the WarpShuffle combine
    in materialize_tile is register-only and L1 absorbs repeat loads)."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert _tile_has_combine(out)
    # stage_inputs now records its no-stage decision (empty STAGE) on every
    # kernel for a uniform knob set, so it "fires" — but the kernel must still be
    # smem-free: no StageBundle is emitted.
    assert not _has_stage_bundle(out)


def _has_stage_bundle(g: Graph) -> bool:
    """True iff any kernel body carries a ``StageBundle`` — the smem-staging
    structure. The no-stage ``STAGE`` decision adds only a knob, no bundle."""
    for node in g.nodes.values():
        body = getattr(node.op, "body", None)
        if body is not None and any(isinstance(s, StageBundle) for s in body.iter()):
            return True
    return False


def _kernel_body_stmts(g: Graph):
    out: list = []
    for node in g.nodes.values():
        body = getattr(node.op, "body", None)
        if body is None:
            continue
        for s in body.iter():
            out.append(s)
    return out


def test_warp_cooperative_emits_warpshuffle(recording_dump):
    """K=32 cooperative tile → ``materialize_tile._emit_combine`` picks
    the warp path: ``WarpShuffle`` Stmt appears, no ``TreeHalve``."""
    g = Graph()
    _input(g, "x", (4, 32))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES).run(g, dump=recording_dump)
    stmts = _kernel_body_stmts(out)
    assert any(isinstance(s, WarpShuffle) for s in stmts)
    assert not any(isinstance(s, TreeHalve) for s in stmts)


def test_block_cooperative_emits_hierarchical_reduce(recording_dump, monkeypatch):
    """A 2-warp (BR=64) cooperative K=256 tile → ``materialize_tile._emit_combine``
    picks the hierarchical path: ``WarpShuffle`` reduces lanes within each warp,
    then a tiny ``TreeHalve(length=n_warps)`` collapses across warps. Both Stmts
    are present; the TreeHalve's length is far smaller than the legacy
    ``length=n_threads`` form. The tile is PINNED (``BR=64`` forces 2 cooperative
    warps): the cold default is now ranked by the ``AnalyticPrior`` (GPU/shape
    dependent — it picks a sub-warp BR for this tiny reduce), so the multi-warp
    code path must be pinned to be exercised deterministically."""
    monkeypatch.setenv("DEPLODOCK_BN", "1")
    monkeypatch.setenv("DEPLODOCK_BM", "4")
    monkeypatch.setenv("DEPLODOCK_BR", "64")
    monkeypatch.setenv("DEPLODOCK_BK", "4")
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(KERNEL_PASSES).run(g, dump=recording_dump)
    stmts = _kernel_body_stmts(out)
    warp_shuffles = [s for s in stmts if isinstance(s, WarpShuffle)]
    tree_halves = [s for s in stmts if isinstance(s, TreeHalve)]
    assert warp_shuffles, "expected WarpShuffle for the per-warp combine"
    assert tree_halves, "expected TreeHalve for the cross-warp combine"
    # Cross-warp TreeHalve runs over n_warps partials, not n_threads.
    assert all(t.length < 256 for t in tree_halves), [t.length for t in tree_halves]


def test_block_cooperative_skips_stage_inputs(recording_dump):
    """K=256 cooperative reduce: the v1 cooperative path uses a sole
    ``K_c`` THREAD axis (BR>1 ⇒ BN=BM=1), so each thread reads its
    own K_c-strided slice of the row with no cross-thread reuse —
    ``stage_inputs`` correctly skips. The kernel still gets a Monoid
    (cross-thread reduce after the per-thread partial sums) and lowers
    via the planner's cooperative branch."""
    g = Graph()
    _input(g, "x", (4, 256))
    g.add_node(op=ReduceOp(op="sum", axis=-1), inputs=["x"], output=Tensor("o", (4, 1)), node_id="o")
    g.inputs = ["x"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert _tile_has_combine(out)
    # stage_inputs records its no-stage decision (knob only) but emits no
    # StageBundle — the cooperative reduce stays smem-free.
    assert not _has_stage_bundle(out)


def test_matmul_does_not_fire_cooperative_reduce(recording_dump):
    """Matmul-shape reduce → chunk_matmul_k handles K splitting; the
    cooperative-reduce strategy is for single-buffer reductions and
    must not fire here.

    The signal is the absence of ``Monoid`` in the Tile body
    (cooperative coordination would have emitted one)."""
    g = Graph()
    _input(g, "a", (_M, _K))
    _input(g, "b", (_K, _N))
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (_M, _N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]

    out = Pipeline.build(TILE_PASSES).run(g, dump=recording_dump)
    assert not _tile_has_combine(out)


# --- _accums_independent: scoped unit tests --------------------------
# The cooperative-reduce rule permits multiple Accums in one reduce
# loop iff none of them transitively reads a prior Accum's running
# value. Below we exercise the helper directly (constructing a full
# multi-Accum reduce graph at the frontend level requires a fusion pass
# that doesn't exist yet — the helper is the contract that rule will
# use once such graphs reach it).


def _load(name: str, src: str) -> Load:
    return Load(name=name, input=src, index=(Var("k"),))


def test_accums_independent_single():
    body = (_load("v", "x"), Accum(name="acc", value="v", op="add"))
    assert _accums_independent(body) is True


def test_accums_independent_two_independent():
    """sum + sum_of_squares: both read inputs (Loads), neither reads the
    other's running value → independent."""
    body = (
        _load("v", "x"),
        Assign(name="vv", op="multiply", args=("v", "v")),
        Accum(name="s", value="v", op="add"),
        Accum(name="s2", value="vv", op="add"),
    )
    assert _accums_independent(body) is True


def test_accums_dependent_via_direct_read():
    """Second Accum reads the first Accum's running value directly."""
    body = (
        _load("v", "x"),
        Accum(name="acc_max", value="v", op="max"),
        Accum(name="acc_sum", value="acc_max", op="add"),
    )
    assert _accums_independent(body) is False


def test_accums_dependent_via_assign_chain():
    """Online softmax pattern: ``e = exp(v - acc_max); acc_sum += e``.
    The Assign chain transitively taints ``e``, so ``acc_sum`` is
    rejected as dependent."""
    body = (
        _load("v", "x"),
        Accum(name="acc_max", value="v", op="max"),
        Assign(name="d", op="subtract", args=("v", "acc_max")),
        Assign(name="e", op="exp", args=("d",)),
        Accum(name="acc_sum", value="e", op="add"),
    )
    assert _accums_independent(body) is False


# --- reduce_body_has_coupled_accum: the per-reduce-scope counterpart shared by
#     040_use_ring_buffers / 080_pipeline_stages (a NON-Accum stmt reads a
#     sibling Accum's running value).


def test_coupled_accum_false_when_independent():
    body = (
        _load("v", "x"),
        Assign(name="vv", op="multiply", args=("v", "v")),
        Accum(name="s", value="v", op="add"),
        Accum(name="s2", value="vv", op="add"),
    )
    assert reduce_body_has_coupled_accum(body) is False


def test_coupled_accum_true_for_online_softmax():
    # The rescale ``d = subtract(v, acc_max)`` is a non-Accum stmt reading the
    # running max — the coupled-reduction shape the ring/pipeline peels reject.
    body = (
        _load("v", "x"),
        Accum(name="acc_max", value="v", op="max"),
        Assign(name="d", op="subtract", args=("v", "acc_max")),
        Assign(name="e", op="exp", args=("d",)),
        Accum(name="acc_sum", value="e", op="add"),
    )
    assert reduce_body_has_coupled_accum(body) is True


def test_coupled_accum_only_inspects_non_accum_stmts():
    # An Accum reading another Accum directly is the *accums_independent*
    # concern; this predicate inspects only non-Accum stmts, so the bare
    # Accum→Accum read (no intermediate non-Accum) is not coupled here.
    body = (
        _load("v", "x"),
        Accum(name="acc_max", value="v", op="max"),
        Accum(name="acc_sum", value="acc_max", op="add"),
    )
    assert reduce_body_has_coupled_accum(body) is False
    assert _accums_independent(body) is False  # the other predicate does catch it
