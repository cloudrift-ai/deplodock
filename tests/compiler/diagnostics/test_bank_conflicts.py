"""STAGE-WRAP-BODY REFACTOR: xfailed pending bucket-11 (materializer rewrite).

The test exercises Tile→Kernel lowering; the materializer's Stage handling
still reads the old API (source_loads, .name etc.) and needs to be
rewritten to consume per-Source smem decls. Phase C.5 sweeps the xfail.

Cross-validation: Tile-IR oracle vs Kernel-IR static analyzer.

For each ``(Stage, body Load)`` binding produced by the Tile-IR pipeline,
asks ``lane_bank_distribution`` (the oracle that ``007a_permute_register_tile``
and ``014_pad_smem`` score against) what each lane's smem address should be at
``k_iter=0``. Then runs ``simulate_graph`` (which lowers the graph through
``KERNEL_PASSES`` and analyzes ``Smem`` Loads at the Kernel-IR level) and
checks both compute the same addresses. Pure static — no GPU required.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.xfail(reason="stage-wrap: bucket-11 follow-up — materializer rewrite", strict=False)


def test_oracle_matches_kernel_analyzer_on_small_matmul():
    from deplodock.compiler.diagnostics.bank_conflicts import (
        find_all_bindings,
        lane_bank_distribution,
        simulate_graph,
    )
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline import TILE_PASSES, Pipeline

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (256, 64)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (64, 256)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (256, 256)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    tile_graph = Pipeline.build(TILE_PASSES).run(g)

    oracle_by_key: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    for binding in find_all_bindings(tile_graph):
        src, load, tile = binding.source, binding.load, binding.tile
        cache_axes = src.cache_axes
        if not cache_axes or len(load.index) < len(cache_axes):
            continue
        cache_idx = tuple(load.index[-len(cache_axes):])
        extra_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
        for ax in binding.enclosing_loop_axes:
            extra_env.setdefault(ax.name, 0)
        dist = lane_bank_distribution(cache_idx, src.alloc_extents, tile.thread_axes, extra_env=extra_env)
        if dist is None:
            continue
        oracle_by_key[(binding.tile_op_name, src.name, load.name)] = (dist.lane_addrs, dist.lane_banks)

    analyzer_results = simulate_graph(tile_graph)
    assert analyzer_results, "simulate_graph returned no results"
    assert oracle_by_key, "oracle produced no predictions"

    analyzer_by_key = {(r.tile_op_name, r.stage_name, r.load_name): r for r in analyzer_results}
    common = set(oracle_by_key) & set(analyzer_by_key)
    assert common, (
        f"no overlapping (kernel, stage, load) keys between oracle and analyzer\n"
        f"  oracle:   {sorted(oracle_by_key)}\n  analyzer: {sorted(analyzer_by_key)}"
    )

    mismatches: list[str] = []
    for key in sorted(common):
        oracle_addrs, oracle_banks = oracle_by_key[key]
        d = analyzer_by_key[key]
        if oracle_addrs != d.lane_addrs:
            mismatches.append(f"{key}: lane_addrs differ\n  oracle:   {oracle_addrs}\n  analyzer: {d.lane_addrs}")
        if oracle_banks != d.lane_banks:
            mismatches.append(f"{key}: lane_banks differ\n  oracle:   {oracle_banks}\n  analyzer: {d.lane_banks}")

    assert not mismatches, "oracle/analyzer disagreement:\n" + "\n".join(mismatches)
