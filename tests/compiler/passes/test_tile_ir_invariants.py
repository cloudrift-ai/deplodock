"""RF hardening guards for the block-DAG tile IR.

These lock the load-bearing invariants the refactor rests on — the ones a future
change could silently break and that the perf DB / learned prior can't migrate around:

- **derived-view discipline** — ``Block`` stores *only* ``name`` / ``domain`` /
  ``compute``; every projection (``reads`` / ``writes`` / ``carrier`` / ``atom``) is
  computed, so it can't drift and can't enter ``op_cache_key``.
- **assemble determinism** — ``assemble`` is a total function of (algorithm +
  Schedule); the same ``TileGraph`` assembles to the byte-identical ``TileOp`` every
  time, so its ``op_cache_key`` is stable.
- **op_cache_key canonicality** — two *independent* builds of the same shape + knobs
  (fresh ``Axis`` objects, same structure) produce the **same** key, and a different
  tile knob produces a **different** key. This is what lets a tuned perf row match a
  later recompile of the same variant.
- **build_dag is the byte-identity oracle, not a dead duplicate** — the
  ``TILE_PASSES`` pipeline (which applies the moves incrementally, pass by pass) and
  the ``build_dag`` composition (``seed_graph → reduce_decomp → free_tile`` in one
  call) assemble to the **same** ``op_cache_key`` for the same knobs.
- **assembly ⟂ enumeration** — the materialization side imports nothing from the
  search side's offer / knob / move vocabulary (the directory boundary is the
  "is this allowed to fork?" guardrail).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.frontend.ir import MatmulOp
from emmy.compiler.ir.tile.ir import Block, Buffer, Space, TileGraphOp, TileOp
from emmy.compiler.pipeline import LOOP_PASSES, TILE_PASSES, Pipeline
from emmy.compiler.pipeline.passes.lowering.tile.assembly._assemble import assemble_block
from emmy.compiler.pipeline.passes.lowering.tile.enumeration import _families as fam
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._build import build_dag
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._classify import classify
from emmy.compiler.pipeline.passes.lowering.tile.enumeration._iterdag import iter_dag
from emmy.compiler.pipeline.search.keys import op_cache_key

# Native ``MOVE@element`` knobs for the matmul oracle: the free axes are ``a0`` (M, outer)
# and ``a1`` (N, inner), the contraction axis ``a2``. ``SPLIT@<axis>`` packs par×reg
# (BN/FN on the inner N axis, BM/FM on the outer M); ``REDUCE@a2`` the contraction tower.
_REDUCE_AXIS = "a2"
_MM_KNOBS = {
    fam.split_key("a1"): fam.enc_split(8, 2),
    fam.split_key("a0"): fam.enc_split(8, 2),
    fam.reduce_key(_REDUCE_AXIS): fam.enc_reduce(serial=16, fold=1, cta=1),
}


def _matmul_graph(M: int = 64, N: int = 64, K: int = 64) -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("a", (M, K)), node_id="a")
    g.add_node(op=InputOp(), inputs=[], output=Tensor("b", (K, N)), node_id="b")
    g.add_node(op=MatmulOp(), inputs=["a", "b"], output=Tensor("o", (M, N)), node_id="o")
    g.inputs, g.outputs = ["a", "b"], ["o"]
    return g


def _loop_dag_buffers(graph: Graph):
    """The fused ``LoopOp`` + its ``iter_dag`` / regime / logical buffers — the seed the
    move composer tiles. Mirrors ``010_build`` so the oracle path matches the pipeline."""
    loop = next(n.op for n in Pipeline.build(LOOP_PASSES).run(graph).nodes.values() if type(n.op).__name__ == "LoopOp")
    dag = iter_dag(loop)
    buffers = {name: Buffer(name=name, shape=tuple(t.shape), dtype=t.dtype, space=Space.GMEM) for name, t in loop.inputs.items()}
    return loop, dag, classify(dag), buffers


def _oracle_tilegraph(graph: Graph, knobs: dict):
    """``build_dag`` (the composition oracle) for ``graph`` at ``knobs``."""
    _loop, dag, regime, buffers = _loop_dag_buffers(graph)
    return build_dag(dag, knobs, kernel_name="k_matmul", target_names=regime.target_names, buffers=buffers)


def test_block_stores_only_name_domain_compute() -> None:
    """Derived-view discipline: ``Block``'s stored fields are exactly the algorithm
    (``name`` / ``domain`` / ``compute``). Everything else is a ``@property``, so a
    projection can never drift from the body nor leak into the cache key."""
    assert set(Block.__dataclass_fields__) == {"name", "domain", "compute"}


def test_assemble_is_deterministic() -> None:
    """``assemble`` is total + choice-free: the same ``TileGraph`` lowers to the
    byte-identical ``TileOp`` (body + knobs + cache key) on every call."""
    tg = _oracle_tilegraph(_matmul_graph(), _MM_KNOBS)
    a = assemble_block(tg, knobs=_MM_KNOBS, base_knobs={}, kernel_name="k_matmul")
    b = assemble_block(tg, knobs=_MM_KNOBS, base_knobs={}, kernel_name="k_matmul")
    assert a.body.structural_key() == b.body.structural_key()
    assert a.knobs == b.knobs
    assert op_cache_key(a) == op_cache_key(b)


def test_op_cache_key_is_canonical_and_knob_sensitive() -> None:
    """Two independent builds of the same shape + knobs (fresh ``Axis`` objects, equal
    structure) key the same — the round-trip a perf row depends on — while a changed
    reduce knob keys differently."""
    g = _matmul_graph()
    op_a = TileGraphOp(name="k_matmul", tilegraph=_oracle_tilegraph(g, _MM_KNOBS), knobs=dict(_MM_KNOBS))
    op_b = TileGraphOp(name="k_matmul", tilegraph=_oracle_tilegraph(g, _MM_KNOBS), knobs=dict(_MM_KNOBS))
    assert op_cache_key(op_a) == op_cache_key(op_b)

    other = {**_MM_KNOBS, fam.reduce_key(_REDUCE_AXIS): fam.enc_reduce(serial=8, fold=1, cta=1)}
    op_c = TileGraphOp(name="k_matmul", tilegraph=_oracle_tilegraph(g, other), knobs=other)
    assert op_cache_key(op_c) != op_cache_key(op_a)


def test_pipeline_incremental_build_matches_build_dag_oracle(monkeypatch) -> None:
    """The pipeline applies the F3-b moves incrementally (one per pass); ``build_dag``
    composes the same moves in one call. Assemble both at the pipeline's own picked
    knobs and assert the same ``op_cache_key`` — so ``build_dag`` is the live oracle for
    the distribution, not a stale duplicate that can drift."""
    monkeypatch.setenv("EMMY_STAGE", "none")  # pin staging off so the oracle needn't replay the STAGE fork
    g = _matmul_graph()
    out = Pipeline.build(TILE_PASSES).run(g, ctx=Context.from_target((8, 0)))
    tile_op = next(n.op for n in out.nodes.values() if isinstance(n.op, TileOp))

    _loop, dag, regime, buffers = _loop_dag_buffers(g)
    oracle_tg = build_dag(dag, tile_op.knobs, kernel_name=tile_op.name, target_names=regime.target_names, buffers=buffers)
    oracle = assemble_block(oracle_tg, knobs=tile_op.knobs, base_knobs={}, kernel_name=tile_op.name, leading=tuple(dag.leading))
    assert op_cache_key(oracle) == op_cache_key(tile_op)


def test_assembly_package_imports_no_enumeration_module() -> None:
    """The materialization side (``assembly/``) must not import the search side's
    offer / knob / move vocabulary (``enumeration/``): the dir boundary is the
    "allowed to fork?" guardrail, and a leak would couple lowering to the search."""
    asm_dir = Path(importlib.import_module("emmy.compiler.pipeline.passes.lowering.tile.assembly").__file__).parent
    pattern = re.compile(r"lowering\.tile\.enumeration|from\s+\S*enumeration\s+import|import\s+\S*enumeration")
    offenders = [f.name for f in sorted(asm_dir.glob("*.py")) if pattern.search(f.read_text())]
    assert offenders == [], f"assembly/ imports enumeration: {offenders}"
