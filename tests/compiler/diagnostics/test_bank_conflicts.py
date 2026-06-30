"""Cross-validation: Tile-IR oracle vs Kernel-IR static analyzer.

For each ``(Source, body Load)`` binding produced by the Tile-IR pipeline,
asks ``lane_bank_distribution`` (the pure Tile-IR oracle) what each lane's smem
address should be at ``k_iter=0``. Then
runs ``simulate_graph`` (which lowers the graph through ``KERNEL_PASSES`` and
analyzes ``Smem`` Loads at the Kernel-IR level) and checks both compute the
same addresses. Pure static — no GPU required.
"""

from __future__ import annotations

from tests.compiler.conftest import matmul_graph


def test_oracle_matches_kernel_analyzer_on_small_matmul():
    from emmy.compiler.diagnostics.bank_conflicts import (
        find_all_bindings,
        lane_bank_distribution,
        simulate_graph,
    )
    from emmy.compiler.pipeline import TILE_PASSES, Pipeline

    tile_graph = Pipeline.build(TILE_PASSES).run(matmul_graph(256, 64, 256))

    oracle_by_key: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    for binding in find_all_bindings(tile_graph):
        src, load = binding.source, binding.load
        cache_axes = src.cache_axes
        if not cache_axes or len(load.index) < len(cache_axes):
            continue
        cache_idx = tuple(load.index[-len(cache_axes) :])
        extra_env: dict[str, int] = {ax.name: 0 for ax in binding.block_axes}
        for ax in binding.enclosing_loop_axes:
            extra_env.setdefault(ax.name, 0)
        dist = lane_bank_distribution(cache_idx, src.alloc_extents, binding.tile.axes, extra_env=extra_env)
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
