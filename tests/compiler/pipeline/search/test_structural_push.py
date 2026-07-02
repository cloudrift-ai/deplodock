"""Fork classification by effect.

The engine classifies every multi-option fork at the spawn site in ``Run.drive``
— where the raw option list is concrete — and threads the result through
``Search.push(structural=)``: any ``Graph``-splicing option (a kernel-set
change) marks the fork structural; pure ``Op`` rebinds and the body-move tiling
forks are op-variant. These tests pin the predicate itself and the engine-level
flag for the two structural emitters (``tile/010_split_demoted`` — R7,
``tile/enumeration/150_cross_cta_finalize``) vs the op-variant rules (the
``060_reduce_tile`` / ``090_thread_tile`` / ``100_register_tile`` tiling forks,
``120_stage`` rebinds). No GPU: terminals stay in the tile dialect
(``TILE_PASSES``) and nothing is benched.
"""

from __future__ import annotations

import pytest

from deplodock.compiler import dtype as _dt
from deplodock.compiler import target as target_mod
from deplodock.compiler.context import Context
from deplodock.compiler.graph import Graph, Tensor
from deplodock.compiler.ir.base import InputOp
from deplodock.compiler.ir.frontend.ir import LinearOp, MatmulOp, RmsNormOp
from deplodock.compiler.pipeline import TILE_PASSES, Pipeline, TuningSearch
from deplodock.compiler.pipeline.fork import Fork, OptionFork
from deplodock.compiler.pipeline.pipeline import _is_structural_option
from deplodock.compiler.pipeline.search.db import SearchDB
from tests.compiler.conftest import drain_tune


class _BranchFork(Fork):
    """Minimal non-leaf ``Fork`` — untypable without ``expand()``, so never structural."""

    knobs: dict = {}

    def expand(self):
        return []


_S, _H, _I = 32, 1024, 3072


@pytest.fixture(autouse=True)
def _isolated_prior(monkeypatch, tmp_path):
    """Untrained prior so descents are deterministic regardless of the host's
    checkpoint; target reset after each test."""
    monkeypatch.setenv("DEPLODOCK_PRIOR_FILE", str(tmp_path / "prior.json"))
    yield
    target_mod.set_target(None)


class _RecordingSearch(TuningSearch):
    """Records ``(rule_name, structural)`` per push — the engine-side view of
    the spawn-site classification."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.pushes: list[tuple[str | None, bool]] = []

    def push(self, *cands, parent=None, structural=False):  # noqa: ANN002
        rule = cands[0].pending[0].rule.name if cands and cands[0].pending is not None else None
        self.pushes.append((rule, structural))
        super().push(*cands, parent=parent, structural=structural)


def _norm_linear_graph() -> Graph:
    """RMSNorm → Linear (f16): fusion yields the prologue-demoted matmul that
    005 offers the structural split on."""
    f16 = _dt.get("f16")
    g = Graph()
    g.add_node(InputOp(), [], Tensor("x", (1, _S, _H), f16), node_id="x")
    g.add_node(InputOp(), [], Tensor("nw", (_H,), f16), node_id="nw")
    g.add_node(InputOp(), [], Tensor("wg", (_I, _H), f16), node_id="wg")
    g.add_node(RmsNormOp(eps=1e-6), ["x", "nw"], Tensor("xn", (1, _S, _H), f16), node_id="xn")
    g.add_node(LinearOp(), ["xn", "wg"], Tensor("o", (1, _S, _I), f16), node_id="o")
    g.inputs = ["x", "nw", "wg"]
    g.outputs = ["o"]
    return g


def _f32_matmul_graph(M: int = 128, K: int = 128, N: int = 128) -> Graph:
    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (M, K)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (K, N)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("o", (M, N)), node_id="o")
    g.inputs = ["a", "b"]
    g.outputs = ["o"]
    return g


def _drive_one_terminal(graph: Graph, cc: tuple[int, int]) -> _RecordingSearch:
    """Drive ``TILE_PASSES`` to the first terminal with the recording search."""
    search = _RecordingSearch(patience=10**6)
    drain_tune(Pipeline.build(TILE_PASSES), graph, search=search, ctx=Context.from_target(cc), db=SearchDB(), on=lambda c: True)
    return search


def test_is_structural_option_predicate() -> None:
    """The Op/Graph return-type split IS the classification: a raw ``Graph`` or
    a leaf ``OptionFork`` wrapping one is structural; an ``Op`` rebind, an
    Op-wrapping leaf, and a branch ``Fork`` (untypable without ``expand()`` —
    today always the partition planner's op-variant tree) are not."""
    assert _is_structural_option(Graph())
    assert _is_structural_option(OptionFork(option=Graph()))
    assert not _is_structural_option(InputOp())
    assert not _is_structural_option(OptionFork(option=InputOp()))
    assert not _is_structural_option(_BranchFork())
