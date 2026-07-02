"""Single source of truth for tests expected to fail during the tile-IR rebuild.

The tile IR (``emmy/compiler/ir/tile/`` + ``.../pipeline/passes/lowering/tile/`` +
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
    # test_bank_conflicts.py deleted — a unit cross-validation of the demolished
    # ``diagnostics.bank_conflicts`` oracle (``find_all_bindings`` is a NotImplementedError stub;
    # the oracle rebuild stays a tile-ir-rebuild follow-up, with the visualizer scripts as its
    # consumers).
    # test_cooperative_flash_matches_torch RECOVERED — the carrier-generic coop tier handles the
    # twisted flash carrier; the test's dead legacy ``EMMY_BR`` pin was modernized to the
    # live ``REDUCE=b<n>`` codec (shuffle combine at b32, the hierarchical smem tree at b64).
    # test_fused_edge.py was rewritten black-box off the demolished enumeration/assembly API. The
    # scalar cells AND the pure-MAP warp cells (relu / sigmoid / multiply) are green — the demoted
    # cone nodifies to a computed-A Contraction under a warp TILE pin and the producer compute-fills
    # the A slab (the mma tier's sync transport). Two residuals: the BROADCAST producer recognizes
    # as a flat un-annotated Map (no Reduction node for the option to nodify — a recognition gap),
    # and the MONOID (rmsnorm) producer's cone carries a reduce (not compute-fillable per cell —
    # needs the cooperative-prologue warp fusion).
    "test_fused_edge.py::test_fused_map_matmul[warp-broadcast": _R,
    "test_fused_edge.py::test_fused_rmsnorm_linear[warp": _R,
    # test_matmul_mma.py / _transposed_b.py / _residual.py / _causal_epilogue.py deleted — those
    # legacy-API (EMMY_MMA / WM / WN / BK pin) per-capability tests are superseded by the
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
    # recovered with the dynamic-grid tier. The flash CHAIN (the FA-2 shared-score register-vector
    # form) RECOVERED as the deterministic scalar default (`_schedule._twisted_chain_option` +
    # `_factor._realize_chain`); the `test_flash_default_is_scalar_stream` gap snapshot (asserting
    # the chain's ABSENCE) is deleted with it.
    # Tensor-core flash RECOVERED through the one emitter: `_schedule._twisted_warp_option` stamps
    # the mma TilePlans on the Q@K / P@V Contractions (a schedule decision, no recognizer stamp),
    # `_bind`'s reduce arm realizes the TWISTED carrier at fragment residence (`_twist`), and the
    # C→A handoff rides the shared kernel-IR nodes — all 26 `test_generated_tensorcore_flash_*` /
    # `test_warp_chain_*` cases are green again (their entries deleted).
    # test_flash_off_keeps_decomposition deleted — it pinned the OLD pipeline's score-
    # materializing multi-kernel decomposition; the rebuilt pipeline fuses SDPA to one correct
    # kernel with FLASH off too (accuracy covered by test_ops_vs_torch::test_sdpa).
    # matmul enabled at the scalar tier — these files partially recovered; residuals still need
    # the mma / staging / split-K / dynamic / attention tiers (scalar fallback gives correct
    # accuracy for the rest, which is un-xfailed). The whole-block (TinyLlama / Qwen) and RoPE
    # self-attention cases recovered once the op-tree lift covered the un-fused RoPE-attention
    # fallback (multi-reduce kernels lower as a flat ``Map``), so they are no longer registered.
    # test_matmul_single_cta_f_replicated / test_gated_mlp_single_cta_f_replicated deleted —
    # the register-tile (``TILE`` codec) capability they exercised is now covered, static AND
    # dynamic, by test_matmul_coverage. test_sgemm_inner_reduce_is_unrolled recovered with the
    # scalar-tier operand staging (the STAGE-pinned scalar contraction stages via a smem slab +
    # #pragma-unrolled inner drain).
    # mma operand staging (cp.async / TMA / gmem→smem ring / smem→register double-buffer) landed —
    # the six warp-tier STAGE structure / bit-identity tests are recovered. Scalar-tier operand
    # staging landed too (the STAGE-pinned scalar contraction stages its
    # operands through an smem slab via tma / cp.async). ``test_bank_conflicts.py`` (already xfailed
    # above) still needs the demolished ``find_all_bindings`` staging-diagnostics oracle rebuilt (a
    # separate follow-up).
    # test_lowering_error_guardrail.py: the guardrail-engine tests recovered once
    # _raise_on_unlowered detected a stuck TileOp (not only a LoopOp); the greedy option-0
    # fallback recovered once _unlowered_tiles blocklisted a stuck TileOp too.
    # test_compute_phase_info_raises_on_collapsed_index deleted — a unit test of the demolished
    # kernel/_stage_expand module against old-IR Source objects.
    # test_compile_dynamic_emits_runtime_arg / test_run_code_dynamic_seq_len recovered with the
    # dynamic-grid tier (a symbolic free axis lowers to a symbolic launch + runtime ``int`` arg).
    # test_run_ir_{tile,kernel}_stage / _bench / _seed_reproducible recovered once the tile-IR +
    # kernel-IR JSON round-trip returned (TileOp / KernelOp reconstruct via graph.py's repr-eval).
    # test_compile_fp16_matmul_window_emits_half2 deleted — it pinned the demolished
    # 015_pack_fk_window pass through the dead legacy FK/BN/BM knob set; the packed-__half2
    # scalar-tier accumulate is a perf codegen follow-on (no TILE-codec spelling yet), tracked
    # in tile-ir-rebuild.md.
    # test_golden_prior_eval_joins_fp16_goldens recovered — analytic._enumerate is rebuilt on
    # the restored _schedule enumeration (it resolves the shape through TILE_PASSES and captures
    # the contraction fork's leaf rows).
    # The split-K structural-fork search tests (test_structural_push's two drive tests +
    # test_two_level's five outer-branching tests + test_resolve's structural replay) pinned the
    # demolished 010_split_demoted / 150_cross_cta_finalize structural fork emitters — DELETED:
    # the rebuilt tree has no multi-option Graph-splicing fork (PLACE@cone is pin-only; split-K is
    # an op-variant row of the restored schedule enumeration, consumed deterministically by
    # 030_split). The engine-side machinery (structural classification, replay, decomposition
    # rows) stays, tested at the unit level; its tests return with a structural fork producer
    # (the PLACE auto knobification follow-up). test_resolve's partition-fork trace test was
    # rewritten against the ONE hierarchical schedule fork.
}

# Files that error at COLLECTION on a tile import (xfail can't catch those). Empty now:
# the entangled integration files had their tile imports guarded so they collect and
# are registered in XFAIL above; pure unit-test files were deleted.
TILE_ENTANGLED_FILES: tuple[str, ...] = ()
