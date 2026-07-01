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
    "test_bank_conflicts.py": _R,
    "test_attention_coverage.py::test_cooperative_flash_matches_torch": _R,
    "test_fused_edge.py": _R,
    # test_matmul_mma.py / _transposed_b.py / _residual.py / _causal_epilogue.py deleted — those
    # legacy-API (DEPLODOCK_MMA / WM / WN / BK pin) per-capability tests are superseded by the
    # warp-tier matrix in test_matmul_coverage (the WARP codec): plain + transposed-B + the
    # bias/relu/residual/causal epilogues, static AND dynamic, all recovered by the gmem-direct
    # mma.sync _warp materializer. test_matmul_rules.py / test_register_tile_rules.py deleted too
    # — unit tests on the demolished split-K / register-tile rule passes.
    # test_matmul_coverage.py (masked symbolic warp tier): the masked symbolic-M/N matmul
    # **accuracy** recovered with the dynamic-grid tier; the batched-M+K structure render
    # (test_batched_symbolic_mk_reaches_warp) reaches the warp tier again, so its entry is gone.
    "test_vllm_plugin_gen_gpu.py": _R,
    "test_vllm_plugin_gpu.py": _R,
    # --- individual cases: the file still has passing tests ---
    # scalar flash landed; the dynamic single-flash variants (sdpa / gqa / additive-mask)
    # recovered with the dynamic-grid tier. The residual flash-chain cases (a chained flash
    # whose producer still needs a tier) and the obsolete flash-knob-off test remain.
    "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_matches_torch[1-1-8-8]": _R,
    "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_matches_torch[1-2-16-8]": _R,
    "tests/compiler/e2e/test_attention_coverage.py::test_flash_chain_matches_torch[2-3-32-16]": _R,
    "tests/compiler/e2e/test_attention_coverage.py::test_flash_off_keeps_decomposition": _R,
    # matmul enabled at the scalar tier — these files partially recovered; residuals still need
    # the mma / staging / split-K / dynamic / attention tiers (scalar fallback gives correct
    # accuracy for the rest, which is un-xfailed). The whole-block (TinyLlama / Qwen) and RoPE
    # self-attention cases recovered once the op-tree lift covered the un-fused RoPE-attention
    # fallback (multi-reduce kernels lower as a flat ``Map``), so they are no longer registered.
    "tests/compiler/e2e/test_knob_pinning.py::test_article_tma_sgemm_reproduction": _R,
    # test_matmul_single_cta_f_replicated / test_gated_mlp_single_cta_f_replicated deleted —
    # the register-tile (``TILE`` codec) capability they exercised is now covered, static AND
    # dynamic, by test_matmul_coverage.
    "tests/compiler/e2e/test_knob_pinning.py::test_sgemm_inner_reduce_is_unrolled": _R,
    # mma operand staging (cp.async / TMA / gmem→smem ring / smem→register double-buffer) landed —
    # the six warp-tier STAGE structure / bit-identity tests are recovered. Scalar-tier staging is
    # NOT restored: ``test_article_tma_sgemm_reproduction`` (fp32 SGEMM via the demolished
    # ``StageBundle`` API) stays below, and ``test_bank_conflicts.py`` (already xfailed above) needs
    # the demolished ``find_all_bindings`` staging-diagnostics oracle rebuilt (a separate follow-up).
    # test_lowering_error_guardrail.py: the guardrail-engine tests recovered once TileOp
    # exists again; these still need un-rebuilt tile internals (Source / StageBundle / real
    # TileGraph lowering).
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_compute_phase_info_raises_on_collapsed_index": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_greedy_run_falls_back_to_option0_when_prior_overflows": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_greedy_run_raises_lowering_error": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_greedy_run_still_raises_when_no_in_budget_option": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_raise_on_unlowered_fires_for_stuck_tileop": _R,
    "tests/compiler/pipeline/test_lowering_error_guardrail.py::test_run_leaves_no_state_on_pipeline": _R,
    # test_compile_dynamic_emits_runtime_arg / test_run_code_dynamic_seq_len recovered with the
    # dynamic-grid tier (a symbolic free axis lowers to a symbolic launch + runtime ``int`` arg).
    # test_run_ir_{tile,kernel}_stage / _bench / _seed_reproducible recovered once the tile-IR +
    # kernel-IR JSON round-trip returned (TileOp / KernelOp reconstruct via graph.py's repr-eval).
    # The FK half2-window codegen (packed __half2 accumulate) is not yet rebuilt at the scalar tier:
    "tests/compiler/cli/test_run.py::test_compile_fp16_matmul_window_emits_half2": _R,
    "tests/compiler/pipeline/search/test_diagnostics.py::test_golden_prior_eval_joins_fp16_goldens": _R,
    "tests/compiler/pipeline/search/test_structural_push.py::test_atomic_free_splitk_fork_pushes_structural": _R,
    "tests/compiler/pipeline/search/test_structural_push.py::test_split_demoted_fork_pushes_structural": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_decomposition_rows_sum_kernel_set_costs": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_identical_offer_sites_take_the_same_side": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_outer_branches_on_structural_fork": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_outer_descends_prior_preferred_branch_first": _R,
    "tests/compiler/pipeline/search/test_two_level.py::test_split_kernels_attribute_to_pre_decision_op": _R,
    "tests/compiler/pipeline/test_resolve.py::test_structural_replay_consulted": _R,
    "tests/compiler/pipeline/test_resolve.py::test_trace_records_partition_fork": _R,
}

# Files that error at COLLECTION on a tile import (xfail can't catch those). Empty now:
# the entangled integration files had their tile imports guarded so they collect and
# are registered in XFAIL above; pure unit-test files were deleted.
TILE_ENTANGLED_FILES: tuple[str, ...] = ()
