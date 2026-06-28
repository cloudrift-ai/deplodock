"""Single source of truth for tests expected to fail during the tile-IR rebuild.

The tile IR (``deplodock/compiler/ir/tile/`` + ``.../pipeline/passes/lowering/tile/`` +
``.../pipeline/passes/lowering/kernel/``) was demolished to a clean slate and is being rebuilt
from scratch. Until each capability is restored, the integration / accuracy tests that exercise
it cannot pass. Rather than scatter ``@pytest.mark.xfail`` decorators across the suite, every
expected failure is registered HERE and applied centrally by the
``pytest_collection_modifyitems`` hook in ``tests/conftest.py``.

How it works
------------
- Each entry maps a **node-id substring** to a one-line reason. Any collected test whose ``nodeid``
  contains the substring is marked ``xfail(strict=False)``.
- ``strict=False`` means a test that starts passing again shows up as **XPASS** (not a failure) —
  that is the recovery signal. When the rebuild restores a capability, delete its entry here and
  the test reverts to a hard requirement.
- A substring matches broadly: ``"test_matmul_mma.py"`` xfails the whole file (used where every
  collected test in the file currently fails); a full nodeid like
  ``"test_accuracy.py::test_e2e_accuracy[rmsnorm]"`` xfails one case (used where a file still has
  passing tests).

An **empty registry means the rebuild is fully recovered.**

Note on collection-time import errors
-------------------------------------
A file whose *module-level* import of a tile symbol breaks raises at COLLECTION, before any item
exists to mark — pytest reports it as an error, which xfail cannot catch. The demolition handled
this two ways: pure tile-IR unit-test files (they only inspected tile Python objects) were deleted;
integration/accuracy files whose imports were load-bearing had
those imports **guarded** (``try/except ModuleNotFoundError``) so the module still collects and its
tests become markable items registered below. ``TILE_ENTANGLED_FILES`` is therefore empty now.
"""

from __future__ import annotations

_R = "tile IR demolished — rebuild in progress"

# nodeid-substring -> reason. Populated by the demolition; emptied as the rebuild restores each
# capability (delete an entry when its test flips to XPASS).
XFAIL: dict[str, str] = {
    # --- whole files: every collected test currently fails ---
    "test_attention_split_gpu.py": _R,
    "test_bank_conflicts.py": _R,
    "test_flash_cooperative_kv.py": _R,
    "test_fused_edge.py": _R,
    "test_gen_runner_gpu.py": _R,
    "test_launch_geometry_rules.py": _R,
    "test_matmul_mma.py": _R,
    # test_matmul_mma_masked.py: the symbolic-K masked matmul recovered with the symbolic
    # reduce tier (the reduce axis strides to the runtime extent, scalar fallback accurate).
    # The symbolic-M/N (free-axis) + mma-tier residuals still need those tiers — registered
    # per function so the recovered ``test_symbolic_k_masked_mma_accuracy`` is a hard
    # requirement again.
    "test_matmul_mma_masked.py::test_batched_symbolic_mk_masked_mma_accuracy": _R,
    "test_matmul_mma_masked.py::test_batched_symbolic_mk_reaches_warp": _R,
    "test_matmul_mma_masked.py::test_demoted_masked_k_pv_tma_accuracy": _R,
    "test_matmul_mma_masked.py::test_demoted_symbolic_n_accuracy": _R,
    "test_matmul_mma_masked.py::test_symbolic_m_masked_mma_accuracy": _R,
    "test_matmul_mma_masked.py::test_symbolic_m_masked_mma_kernel_structure": _R,
    "test_matmul_mma_masked.py::test_symbolic_m_masked_mma_residual_epilogue_accuracy": _R,
    "test_matmul_mma_masked.py::test_symbolic_m_masked_mma_tma_accuracy": _R,
    "test_matmul_mma_masked.py::test_symbolic_m_masked_mma_tma_structure": _R,
    "test_matmul_mma_masked.py::test_symbolic_mn_masked_mma_accuracy": _R,
    "test_matmul_mma_transposed_b.py": _R,
    "test_monoid_reduce_kernel.py": _R,
    "test_program_rebind.py": _R,
    "test_runner_batched_gpu.py": _R,
    "test_vllm_plugin_gen_gpu.py": _R,
    "test_vllm_plugin_gpu.py": _R,
    # --- individual cases: the file still has passing tests ---
    # scalar flash landed — test_flash_attention residuals are the dynamic/symbolic variants
    # (need the dynamic-shape tier) and the now-obsolete flash-knob-off test.
    "tests/compiler/e2e/test_flash_attention.py::test_flash_additive_mask_dynamic_matches_torch": _R,
    "tests/compiler/e2e/test_flash_attention.py::test_flash_chain_matches_torch[1-1-8-8]": _R,
    "tests/compiler/e2e/test_flash_attention.py::test_flash_chain_matches_torch[1-2-16-8]": _R,
    "tests/compiler/e2e/test_flash_attention.py::test_flash_chain_matches_torch[2-3-32-16]": _R,
    "tests/compiler/e2e/test_flash_attention.py::test_flash_gqa_dynamic_matches_torch": _R,
    "tests/compiler/e2e/test_flash_attention.py::test_flash_off_keeps_decomposition": _R,
    "tests/compiler/e2e/test_flash_attention.py::test_flash_sdpa_dynamic_matches_torch": _R,
    # matmul enabled at the scalar tier — these files partially recovered; residuals still need
    # the mma / staging / split-K / dynamic / attention tiers (scalar fallback gives correct
    # accuracy for the rest, which is un-xfailed). The whole-block (TinyLlama / Qwen) and RoPE
    # self-attention cases recovered once the op-tree lift covered the un-fused RoPE-attention
    # fallback (multi-reduce kernels lower as a flat ``Map``), so they are no longer registered.
    "tests/compiler/e2e/test_knob_pinning.py::test_article_tma_sgemm_reproduction": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_gated_mlp_single_cta_f_replicated[dynamic-BN16_BM32_FM1_FN16]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_gated_mlp_single_cta_f_replicated[dynamic-BN32_BM16_FM2_FN8]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_gated_mlp_single_cta_f_replicated[dynamic-BN32_BM32_FM1_FN8]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_gated_mlp_single_cta_f_replicated[dynamic-BN64_BM16_FM2_FN4]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_matmul_single_cta_f_replicated[dynamic-BN16_BM32_FM1_FN4]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_matmul_single_cta_f_replicated[dynamic-BN32_BM16_FM2_FN2]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_matmul_single_cta_f_replicated[dynamic-BN32_BM32_FM1_FN2]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_matmul_single_cta_f_replicated[dynamic-BN64_BM16_FM2_FN1]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_norm_linear_fp16_scalar_reduce_tma_alignment[dynamic]": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_sgemm_inner_reduce_is_unrolled": _R,
    "tests/compiler/e2e/test_knob_pinning.py::test_unstaged_atom_lowers_gmem_direct": _R,
    "tests/compiler/e2e/test_lowering_blocked_gemm.py::test_fused_rmsnorm_linear_blocked_prologue": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[dynamic-cp.async]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[dynamic-tma]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[static-cp.async]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_pinned_transport_and_shape_fire[static-tma]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_static_dynamic_mma_parity[dynamic-cp.async-256]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_static_dynamic_mma_parity[dynamic-cp.async-512]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_static_dynamic_mma_parity[dynamic-tma-256]": _R,
    "tests/compiler/e2e/test_matmul_mma_parity.py::test_static_dynamic_mma_parity[dynamic-tma-512]": _R,
    "tests/compiler/e2e/test_mma_atomic_free_splitk.py::test_mma_atomic_free_splitk_accurate_and_no_atomic": _R,
    "tests/compiler/e2e/test_stage_scalar.py::test_scalar_matmul_stages_through_pipeline": _R,
    # test_reduction_combine_coverage.py / test_tune_accuracy.py: scalar-tier reduction
    # recovered the serial + cooperative reduce kernels; these residuals need flash /
    # cross-CTA split-reduce / matmul tiers.
    "tests/compiler/e2e/test_reduction_combine_coverage.py::test_cross_cta_finalize_accuracy_and_structure[flash-kernel]": _R,
    "tests/compiler/e2e/test_reduction_combine_coverage.py::test_cross_cta_finalize_accuracy_and_structure[matmul-atomic]": _R,
    "tests/compiler/e2e/test_reduction_combine_coverage.py::test_cross_cta_finalize_accuracy_and_structure[matmul-kernel]": _R,
    "tests/compiler/e2e/test_reduction_combine_coverage.py::test_cross_cta_finalize_accuracy_and_structure[sum-atomic]": _R,
    "tests/compiler/e2e/test_reduction_combine_coverage.py::test_cross_cta_finalize_accuracy_and_structure[sum-kernel]": _R,
    # test_lowering_error_guardrail.py: the guardrail-engine tests recovered once TileOp
    # exists again; these still need un-rebuilt tile internals (Source / StageBundle / real
    # TileGraph lowering).
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_compute_phase_info_raises_on_collapsed_index": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_greedy_run_falls_back_to_option0_when_prior_overflows": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_greedy_run_raises_lowering_error": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_greedy_run_still_raises_when_no_in_budget_option": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_raise_on_unlowered_fires_for_stuck_tileop": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_run_leaves_no_state_on_pipeline": _R,
    "tests/compiler/backend/test_emit.py::test_softmax_emits_multiple_k_loops": _R,
    "tests/compiler/cli/test_compile.py::test_compile_dynamic_emits_runtime_arg": _R,
    "tests/compiler/cli/test_run.py::test_compile_fp16_matmul_window_emits_half2": _R,
    "tests/compiler/cli/test_run.py::test_run_code_dynamic_seq_len": _R,
    "tests/compiler/cli/test_run.py::test_run_ir_bench": _R,
    "tests/compiler/cli/test_run.py::test_run_ir_kernel_stage": _R,
    "tests/compiler/cli/test_run.py::test_run_ir_seed_reproducible": _R,
    "tests/compiler/cli/test_run.py::test_run_ir_tile_stage": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_bf16_matches_torch[1-2-32-16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_bf16_matches_torch[1-4-128-64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_causal_bf16_matches_torch[1-2-32-16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_causal_bf16_matches_torch[1-4-128-64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_causal_matches_torch[1-1-16-16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_causal_matches_torch[1-2-32-16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_causal_matches_torch[1-4-128-64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_causal_matches_torch[2-3-64-32]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_matches_torch[1-1-16-16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_matches_torch[1-2-32-16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_matches_torch[1-4-128-64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_generated_tensorcore_flash_matches_torch[2-3-64-32]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_causal_dynamic_matches_torch[16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_causal_dynamic_matches_torch[37]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_causal_dynamic_matches_torch[64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_causal_dynamic_matches_torch[8]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_dynamic_matches_torch[16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_dynamic_matches_torch[37]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_dynamic_matches_torch[64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_dynamic_matches_torch[8]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_gqa_dynamic_matches_torch[16]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_gqa_dynamic_matches_torch[37]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_gqa_dynamic_matches_torch[64]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_gqa_dynamic_matches_torch[8]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_gqa_static_matches_torch[16-8-32-32]": _R,
    "tests/compiler/e2e/test_flash_tensorcore_generated.py::test_warp_chain_gqa_static_matches_torch[4-2-32-16]": _R,
    "tests/compiler/e2e/test_matmul_mma_causal_epilogue.py::test_causal_mask_epilogue_mma[dynamic-128-out_dtype0]": _R,
    "tests/compiler/e2e/test_matmul_mma_causal_epilogue.py::test_causal_mask_epilogue_mma[dynamic-128-out_dtype1]": _R,
    "tests/compiler/e2e/test_matmul_mma_causal_epilogue.py::test_causal_mask_epilogue_mma[dynamic-130-out_dtype0]": _R,
    "tests/compiler/e2e/test_matmul_mma_causal_epilogue.py::test_causal_mask_epilogue_mma[dynamic-130-out_dtype1]": _R,
    "tests/compiler/e2e/test_matmul_mma_causal_epilogue.py::test_causal_mask_epilogue_mma[static-128-out_dtype0]": _R,
    "tests/compiler/e2e/test_matmul_mma_causal_epilogue.py::test_causal_mask_epilogue_mma[static-128-out_dtype1]": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_chain_epilogue_mma_matches_reference[1]": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_chain_epilogue_mma_matches_reference[4]": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_epilogue_warp_rows_stay_splitk_one": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_multiply_epilogue_admits_warp_tier": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_pointwise_chain_with_broadcast_admits_warp_tier": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_residual_epilogue_admits_warp_tier": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_residual_mma_matches_reference[128-256-128-4-out_dtype1]": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_residual_mma_matches_reference[128-256-128-4-out_dtype2]": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_residual_mma_matches_reference[32-1024-3072-1-out_dtype0]": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_transposed_residual_admits_warp_tier": _R,
    "tests/compiler/e2e/test_matmul_mma_residual.py::test_transposed_residual_mma_matches_reference": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_capture_replay_cache_rmsnorm_over_capacity_buffers": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_capture_replay_device_io_matches_eager": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_cuda_sdpa_over_symbolic_seq_len": _R,
    # test_cuda_softmax_over_symbolic_seq_len recovered with the symbolic cooperative reduce tier.
    "tests/compiler/ir/test_dynamic_shapes.py::test_cuda_symbolic_elementwise_one_kernel_multiple_seq_lens": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_cuda_symbolic_linear_traced_and_run": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_cuda_symbolic_rmsnorm_traced_and_run": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_layer_dynamic_compiles_and_matches_eager": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_capture_replay_cache_matches_eager": _R,
    "tests/compiler/ir/test_dynamic_shapes.py::test_qwen_whole_model_dynamic_compiles_and_matches_eager": _R,
    "tests/compiler/passes/test_matmul_rules.py::test_elwise_lhs_matmul_fires_split_k_and_blockify": _R,
    "tests/compiler/passes/test_matmul_rules.py::test_matmul_then_elwise_fires_split_k_and_blockify": _R,
    "tests/compiler/passes/test_matmul_rules.py::test_plain_matmul_fires_split_k_and_blockify": _R,
    "tests/compiler/passes/test_matmul_rules.py::test_pure_elementwise_does_not_fire_split_k": _R,
    "tests/compiler/passes/test_matmul_rules.py::test_two_elwise_lhs_matmul_fires_split_k_and_blockify": _R,
    "tests/compiler/passes/test_register_tile_rules.py::test_plain_matmul_fires_register_tile": _R,
    "tests/compiler/passes/test_register_tile_rules.py::test_sdpa_qk_matmul_fires_register_tile": _R,
    "tests/compiler/pipeline/search/test_diagnostics.py::test_golden_prior_eval_joins_fp16_goldens": _R,
    "tests/compiler/pipeline/search/test_structural_push.py::test_atomic_free_splitk_fork_pushes_structural": _R,
    "tests/compiler/pipeline/search/test_structural_push.py::test_split_demoted_fork_pushes_structural": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_decomposition_rows_sum_kernel_set_costs": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_identical_offer_sites_take_the_same_side": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_inner_reward_deeper_patience_benches_new_variants": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_inner_reward_is_separable_not_a_product": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_outer_branches_on_structural_fork": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_outer_descends_prior_preferred_branch_first": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_split_kernels_attribute_to_pre_decision_op": _R,
    "tests/compiler/pipeline/test_knob.py::test_knob_features_mma_expansion": _R,
    "tests/compiler/pipeline/test_resolve.py::test_decide_score_lands_on_trace": _R,
    "tests/compiler/pipeline/test_resolve.py::test_resolve_applies_in_place": _R,
    "tests/compiler/pipeline/test_resolve.py::test_structural_replay_consulted": _R,
    "tests/compiler/pipeline/test_resolve.py::test_trace_records_partition_fork": _R,
    "tests/serving/test_generate_gpu.py::test_generate_loop_runs_end_to_end": _R,
    "tests/serving/test_generate_gpu.py::test_generate_oracle_matches_eager_fp16": _R,
}

# Files that error at COLLECTION on a tile import (xfail can't catch those). Empty now:
# the entangled integration files had their tile imports guarded so they collect and
# are registered in XFAIL above; pure unit-test files were deleted.
TILE_ENTANGLED_FILES: tuple[str, ...] = ()
