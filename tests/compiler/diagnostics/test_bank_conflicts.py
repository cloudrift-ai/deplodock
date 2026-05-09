"""Cross-validation: GPU bank-conflict trace vs the static oracle.

For each ``(Stage, body Load)`` binding produced by the Tile-IR pipeline,
asks ``lane_bank_distribution`` (the same oracle that ``009_permute_register_tile``
and ``014_pad_smem`` score against) what each lane's smem address should be at
``k_iter=0``, and checks ``simulate_graph`` records the same addresses.
CUDA-only — skipped when no GPU is available.
"""

from __future__ import annotations

from tests.compiler.conftest import requires_cuda


@requires_cuda
def test_oracle_matches_trace_on_small_matmul():
    from deplodock.compiler.diagnostics.bank_conflicts import (
        find_all_bindings,
        lane_bank_distribution,
        simulate_graph,
    )
    from deplodock.compiler.graph import Graph, Tensor
    from deplodock.compiler.ir.base import InputOp
    from deplodock.compiler.ir.frontend.ir import MatmulOp
    from deplodock.compiler.pipeline import TILE_PASSES, run_pipeline

    g = Graph()
    g.add_node(InputOp(), [], Tensor("a", (256, 64)), node_id="a")
    g.add_node(InputOp(), [], Tensor("b", (64, 256)), node_id="b")
    g.add_node(MatmulOp(), ["a", "b"], Tensor("c", (256, 256)), node_id="c")
    g.inputs = ["a", "b"]
    g.outputs = ["c"]

    tile_graph = run_pipeline(g, TILE_PASSES)

    oracle_by_key: dict[tuple[str, str, str], tuple[list[int], list[int]]] = {}
    for binding in find_all_bindings(tile_graph):
        stage, load, tile = binding.stage, binding.load, binding.tile
        if not stage.axes or len(load.index) < len(stage.axes):
            continue
        cache_idx = tuple(load.index[-len(stage.axes) :])
        extra_env: dict[str, int] = {ax.name: 0 for ax in tile.block_axes}
        for ax in binding.enclosing_loop_axes:
            extra_env.setdefault(ax.name, 0)
        dist = lane_bank_distribution(cache_idx, stage.alloc_extents, tile.thread_axes, extra_env=extra_env)
        if dist is None:
            continue
        oracle_by_key[(binding.tile_op_name, stage.name, load.name)] = (dist.lane_addrs, dist.lane_banks)

    trace_results = simulate_graph(tile_graph)
    assert trace_results, "simulate_graph returned no results"
    assert oracle_by_key, "oracle produced no predictions"

    trace_by_key = {(r.tile_op_name, r.stage_name, r.load_name): r for r in trace_results}
    common = set(oracle_by_key) & set(trace_by_key)
    assert common, (
        f"no overlapping (kernel, stage, load) keys between oracle and trace\n"
        f"  oracle: {sorted(oracle_by_key)}\n  trace:  {sorted(trace_by_key)}"
    )

    mismatches: list[str] = []
    for key in sorted(common):
        oracle_addrs, oracle_banks = oracle_by_key[key]
        d = trace_by_key[key]
        if oracle_addrs != d.lane_addrs:
            mismatches.append(f"{key}: lane_addrs differ\n  oracle: {oracle_addrs}\n  trace:  {d.lane_addrs}")
        if oracle_banks != d.lane_banks:
            mismatches.append(f"{key}: lane_banks differ\n  oracle: {oracle_banks}\n  trace:  {d.lane_banks}")

    assert not mismatches, "oracle/trace disagreement:\n" + "\n".join(mismatches)
