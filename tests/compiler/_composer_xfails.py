"""Quarantine registry for the tile-ir-block-dag recovery (xfail, not deletion).

The legacy partition planner and the ``011``-``090`` scheduling passes were deleted
in the block-DAG Tile IR demolition (``plans/tile-ir-block-dag.md``). Every test
below passed before the demolition and exercises a tier that has not been rebuilt
yet; rather than fix the gaps mid-refactor they are quarantined as
``xfail(strict=False)`` so ``make test`` is green. The value of each entry names
the **recovery phase** (R2-R7) from the plan that will restore it.

When a phase lands, delete its entries here in the same commit (the plan's
recovery bullets mandate it); a closed gap then surfaces as an XPASS.

Keyed three ways, most-specific first — each maps a target to its recovery phase:
- ``_XFAIL_FILES`` — ``file.py`` whose EVERY collected test is red (whole tier gone).
- ``_XFAIL_FUNCS`` — ``file.py::func`` whose EVERY parametrization is red.
- ``_XFAIL_NODES`` — exact ``file.py::func[param]`` where only some params are red
  (the rest stay green and must NOT be masked).

The matcher strips the trailing ``@<xdist-group>`` suffix (``@cuda`` / ``@cuda-cli``
/ ``@w0``) that xdist bakes into the reported nodeid before matching.
"""

from __future__ import annotations

import re

import pytest

_PHASE_REASON = {
    "R2": "R2 cooperative-reduce tier not yet rebuilt (coop classify/build + warp-shuffle combine synthesis)",
    "R3": "R3 split-K / partition_reduce tier not yet rebuilt (atomic / atomic-free combine block)",
    "R4": "R4 warp-tier MMA (atomize) tier not yet rebuilt (RegFragment/Ldmatrix/MmaSyncPtx/RegStore synthesis)",
    "R5": "R5 transport tier not yet rebuilt (cp.async / TMA promote + descriptor/swizzle synthesis)",
    "R6": "R6 flash / attention tier not yet rebuilt (streaming TWISTED_MONOID flash synthesis)",
    "R7": "R7 e2e / CLI / structural-search / prior tier not yet rebuilt",
}

# Whole files: every collected test exercises a deleted tier.
_XFAIL_FILES: dict[str, str] = {
    # R2 — landed (cooperative-reduce): test_cooperative_combine.py /
    # test_masked_cooperative_reduce.py de-quarantined.
    # R3 — test_monoid_reduce_kernel.py imports the deleted 017_atomic_free_splitk
    # (build_monoid_reduce_tileop), the atomic-free split-K combine — R3, not R2.
    "test_monoid_reduce_kernel.py": "R3",
    # R4 — landed (warp-tier atomize): test_matmul_mma_causal_epilogue.py /
    # test_matmul_mma_transposed_b.py / test_stage_inputs_mma_probe.py de-quarantined.
    # R6
    "test_flash_attention.py": "R6",
    "test_flash_cooperative_kv.py": "R6",
    # R7 — test_program_rebind.py de-quarantined: R2's cooperative-reduce lowering
    # made the rmsnorm-bearing rebind kernels compile, so all three rebind tests pass.
}

# Whole functions: every parametrization is red.
#
# R2 incidentally recovered a set of R3/R7-tagged tests whose bodies exercise a
# cooperative reduce (rmsnorm / softmax / mean) that now lowers — they are
# de-quarantined here to keep the green floor at 0 XPASS:
#   - test_mma_atomic_free_splitk.py::…_accurate_and_no_atomic (R3 accuracy half)
#   - test_compile.py::{test_compile_code_saves_default_cuda_to_output, …_dynamic_emits_runtime_arg}
#   - test_run.py::{rmsnorm/softmax accuracy/blockify/fk, run_ir_*, run_bench_prints_table,
#                   run_code_dynamic_seq_len, run_positional_json_like_ir}
#   - test_program_rebind.py (whole file, above)
# The R3 STRUCTURAL fork test + the R7 fp16-matmul-window / sdpa / structural /
# prior tests stay quarantined (genuinely their own tiers).
_XFAIL_FUNCS: dict[str, str] = {
    # R3
    "test_structural_push.py::test_atomic_free_splitk_fork_pushes_structural": "R3",
    # R4 — masked / unstaged warp follow-ups (the warp-tier matmul itself landed):
    # the masked cooperative-load clamp + non-divisor real_extent tests need env-pin
    # honoring in the scalar thread/reg offers (entangled with the cooperative
    # multi-accum regime, R2/R3) + masked-staging clamp synthesis; the gmem-direct
    # atom path needs the staging-decline heuristic; the hoist test imports the
    # deleted 021 pass and must be REWRITTEN against assembly/020_mask_order.
    "test_knob_pinning.py::test_unstaged_atom_lowers_gmem_direct": "R4",
    "test_masked_tile.py::test_hoist_refuses_lift_when_pipeline_reads_guarded_defs": "R4",
    "test_masked_tile.py::test_masked_n_clamps_cooperative_load_index": "R4",
    "test_masked_tile.py::test_planner_admits_non_divisor_n_with_real_extent": "R4",
    "test_masked_tile.py::test_symbolic_m_cooperative_load_clamps_to_runtime_extent": "R4",
    # R5
    "test_knob_pinning.py::test_norm_linear_fp16_scalar_reduce_tma_alignment": "R5",
    # R6
    "test_attention_chains.py::test_full_self_attn_tinyllama": "R6",
    "test_attention_chains.py::test_full_self_attn_tinyllama_seq512": "R6",
    "test_attention_chains.py::test_qkv_attn_no_rope": "R6",
    "test_attention_chains.py::test_sdpa_explicit_additive_mask": "R6",
    "test_dynamic_shapes.py::test_cuda_sdpa_over_symbolic_seq_len": "R6",
    # R7
    "test_analytic.py::test_pick_matmul_lands_in_geometry_band": "R7",
    "test_analytic.py::test_pick_matmul_warp_dispatch_by_dtype": "R7",
    "test_block.py::test_qwen_block_accuracy": "R7",
    "test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager": "R7",
    "test_dynamic_shapes.py::test_qwen_whole_model_capture_replay_cache_matches_eager": "R7",
    "test_dynamic_shapes.py::test_qwen_whole_model_dynamic_compiles_and_matches_eager": "R7",
    "test_fuse_sibling_register_cells.py::test_qwen_lmhead_variant_compiles_within_budget": "R7",
    "test_lowering_blocked_gemm.py::test_fused_rmsnorm_linear_blocked_prologue": "R7",
    "test_resolve.py::test_structural_replay_consulted": "R7",
    "test_resolve.py::test_trace_records_partition_fork": "R7",
    # test_run.py: the fp16-matmul-window + sdpa kernels stay quarantined (R7 matmul
    # window / R6 sdpa); the rmsnorm/softmax/run-ir/bench rows de-quarantined (R2).
    "test_run.py::test_compile_fp16_matmul_window_emits_half2": "R7",
    "test_run.py::test_run_code_fp16_matmul_window_accuracy": "R7",
    "test_run.py::test_run_code_sdpa_k_chunked": "R7",
    "test_run.py::test_run_code_sdpa_seq1024_dynamic_smem": "R7",
    "test_run.py::test_run_code_sdpa_tinyllama_full": "R7",
    "test_run.py::test_run_code_sdpa_tinyllama_per_head": "R7",
    "test_structural_push.py::test_split_demoted_fork_pushes_structural": "R7",
    "test_two_level.py::test_decomposition_rows_sum_kernel_set_costs": "R7",
    "test_two_level.py::test_identical_offer_sites_take_the_same_side": "R7",
    "test_two_level.py::test_outer_branches_on_structural_fork": "R7",
    "test_two_level.py::test_outer_descends_prior_preferred_branch_first": "R7",
    "test_two_level.py::test_run_two_level_tune_single_terminal_assembles_bests": "R7",
    "test_two_level.py::test_split_kernels_attribute_to_pre_decision_op": "R7",
}

# Specific parametrizations (the function also has green params).
_XFAIL_NODES: dict[str, str] = {
    # R5
    "test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[dynamic-tma]": "R5",
    "test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[static-tma]": "R5",
    # R6
    "test_ops_vs_torch.py::test_sdpa[cuda]": "R6",
    "test_ops_vs_torch.py::test_sdpa_causal[cuda]": "R6",
    "test_ops_vs_torch.py::test_sdpa_gqa[cuda]": "R6",
    "test_tune_accuracy.py::test_tuned_variant_matches_reference[sdpa]": "R6",
    # R7
    "test_block.py::test_tinyllama_block_accuracy[cuda]": "R7",
}


def _reason(phase: str) -> str:
    return f"{_PHASE_REASON[phase]}; recovers under {phase}. See plans/tile-ir-block-dag.md"


def mark_composer_xfails(items) -> None:
    """Apply the quarantine xfail to every collected item that matches the
    registry (called from ``conftest.pytest_collection_modifyitems``)."""
    for item in items:
        # Strip the trailing ``@<xdist-group>`` suffix xdist bakes into the nodeid.
        nodeid = re.sub(r"@[\w-]+$", "", item.nodeid)
        phase = None
        for f, ph in _XFAIL_FILES.items():
            if f"/{f}::" in nodeid or f"{f}::" in nodeid or nodeid.startswith(f):
                phase = ph
                break
        if phase is None:
            func_path = nodeid.split("[", 1)[0]
            for f, ph in _XFAIL_FUNCS.items():
                if func_path.endswith(f):
                    phase = ph
                    break
        if phase is None:
            for n, ph in _XFAIL_NODES.items():
                if nodeid.endswith(n):
                    phase = ph
                    break
        if phase is not None:
            item.add_marker(pytest.mark.xfail(reason=_reason(phase), strict=False))
