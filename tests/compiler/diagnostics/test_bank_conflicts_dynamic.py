"""Cross-validation: dynamic bank-conflict simulator vs static.

Runs both simulators on the same Tile-IR graph and asserts per-(stage,
load, lane) agreement on the recorded smem addresses. CUDA-only —
skipped when no GPU is available.
"""

from __future__ import annotations

from tests.compiler.conftest import requires_cuda


@requires_cuda
def test_dynamic_matches_static_on_small_matmul():
    from deplodock.compiler.diagnostics.bank_conflicts import simulate_graph
    from deplodock.compiler.diagnostics.bank_conflicts_dynamic import simulate_graph_dynamic
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

    static_results = simulate_graph(tile_graph)
    dynamic_results = simulate_graph_dynamic(tile_graph)

    assert dynamic_results, "dynamic simulator returned no results"
    assert static_results, "static simulator returned no results"

    static_by_key = {(r.tile_op_name, r.stage_name, r.load_name): r for r in static_results}
    dynamic_by_key = {(r.tile_op_name, r.stage_name, r.load_name): r for r in dynamic_results}

    common = set(static_by_key) & set(dynamic_by_key)
    assert common, (
        f"no overlapping (kernel, stage, load) keys between static and dynamic\n"
        f"  static: {sorted(static_by_key)}\n  dynamic: {sorted(dynamic_by_key)}"
    )

    mismatches: list[str] = []
    for key in sorted(common):
        s = static_by_key[key]
        d = dynamic_by_key[key]
        if s.lane_addrs != d.lane_addrs:
            mismatches.append(f"{key}: lane_addrs differ\n  static:  {s.lane_addrs}\n  dynamic: {d.lane_addrs}")
        if s.lane_banks != d.lane_banks:
            mismatches.append(f"{key}: lane_banks differ\n  static:  {s.lane_banks}\n  dynamic: {d.lane_banks}")

    assert not mismatches, "static/dynamic disagreement:\n" + "\n".join(mismatches)
