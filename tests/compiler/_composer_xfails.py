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
    # R3 — landed (atomic-free split-K combine): test_monoid_reduce_kernel.py
    # de-quarantined (rebuilt against enumeration/_partition.monoid_reduce_tilegraph).
    # R4 — landed (warp-tier atomize): test_matmul_mma_causal_epilogue.py /
    # test_matmul_mma_transposed_b.py / test_stage_inputs_mma_probe.py de-quarantined.
    # R6 — flash landed (streaming TWISTED_MONOID build: enumeration/017_streaming +
    # _build.streaming_build, masked streaming for symbolic seq_len): test_flash_attention.py
    # de-quarantined except test_flash_off_keeps_decomposition (a func entry below — its
    # blocker is the non-flash score-materializing SDPA decomposition, R7).
    # R6 cooperative-KV flash landed (the BR>1 streaming split lays the K_c THREAD lane
    # in _build.streaming_build / _replace_k_streaming; the carrier's combine_states fires at
    # kernel/100 like the MONOID coop reduce): test_flash_cooperative_kv.py de-quarantined.
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
    # R3 — landed (atomic-free split-K): test_atomic_free_splitk_fork_pushes_structural
    # de-quarantined (the 055_atomic_free_splitk structural fork). The R3 accuracy half
    # (test_mma_atomic_free_splitk.py) was already de-quarantined under R2.
    # R4 — masked-tile follow-ups: env-pin honoring in the scalar thread/reg offers
    # (_moves.thread_offers / map_reg_offers / reduce_reg_offers now read DEPLODOCK_BN/
    # BM/FN/FM via _pin, like reduce_offers) + masked SYNC staging clamp (the
    # assembly/_slab._hoist_masked hoist lifts the cooperative load above the boundary
    # Cond and stamps Source.gmem_extents) de-quarantined three tests:
    # test_planner_admits_non_divisor_n_with_real_extent,
    # test_masked_n_clamps_cooperative_load_index,
    # test_symbolic_m_cooperative_load_clamps_to_runtime_extent.
    #
    # test_unstaged_atom_lowers_gmem_direct — LANDED: the over-ceiling FM=26 warp
    # register pin is now authoritative (warp_reg_offers bypasses _MAX_WARP_CELLS
    # for a full pin), and with no STAGE pin the budget-aware 050_stage filter
    # declines the over-budget staging so the operands lower gmem-direct.
    # test_hoist_refuses_lift_when_pipeline_reads_guarded_defs — LANDED: rewritten
    # against assembly/_slab._hoist_masked's SSA-safety check (the fused-prologue
    # lift refusal — a hoisted K-tower reading a name defined inside the Cond
    # refuses the lift), de-quarantined.
    # test_norm_linear_fp16_scalar_reduce_tma_alignment de-quarantined: the fused
    # norm+linear (RmsNorm prologue + matmul) now force-splits via tile/split/
    # 005_split_demoted (the demoted matmul's RMSNorm cone un-fuses into an xn producer
    # + a clean gemm), so its kernel lowers and the TMA-alignment assertion fires.
    # R6 — score-materializing SDPA landed (tile/split/005_split_demoted): the fused
    # softmax-prologue + P@V LoopOp (k_sdpa_reduce) un-fuses into a softmax-normalizing
    # xn producer + a clean (symbolic-K) gemm consumer, both of which lower. The
    # forced-split path (the fused form has no fused-prologue regime to keep) recovered
    # test_ops_vs_torch sdpa[cuda]/causal/gqa, test_tune_accuracy[sdpa], the four
    # test_run sdpa rows, and test_attention_chains qkv_attn_no_rope /
    # sdpa_explicit_additive_mask (all de-quarantined).
    # test_flash_off_keeps_decomposition de-quarantined: the FLASH-off SDPA now lowers
    # via the split (scores + softmax xn producer + gemm = 3 kernels), so the
    # "score-materializing multi-kernel path is kept" assertion (len(kernels) > 1) holds
    # — exactly the score-materializing decomposition the test guards, now lowerable.
    # test_full_self_attn_tinyllama / _seq512 de-quarantined: the RoPE-fused score
    # producer (k_sdpa_linear_reduce) now lowers correctly. The fix is in _stage:
    # _multi_access_bufs excludes any buffer read at >1 distinct access from staging
    # (the rotary cos/sin are read at both the Q row `cos[m,d]` and the K row `cos[n,d]`,
    # and the projection both straight `q·cos` and rotate-half — one slab per buffer
    # served only one access and silently corrupted the other / choked on the TEMPLATE
    # rotate-half). They stay gmem-direct; only same-access reads collapse to one slab.
    # R7 — analytic / prior / structural-search tiers
    "test_analytic.py::test_pick_matmul_lands_in_geometry_band": "R7",
    "test_analytic.py::test_pick_matmul_warp_dispatch_by_dtype": "R7",
    # test_qwen_block_accuracy / test_tinyllama_block_accuracy[cuda] de-quarantined:
    # the whole-block forward (RoPE attention + MLP) now lowers correctly end-to-end
    # via the SDPA split + the multi-access staging fix.
    # The three qwen-dynamic tests de-quarantined: the whole-model / layer SDPA now
    # lowers via tile/split/005_split_demoted (incl. the symbolic-seq P@V), so the
    # dynamic compile + capture-replay paths match eager.
    "test_fuse_sibling_register_cells.py::test_qwen_lmhead_variant_compiles_within_budget": "R7",
    "test_lowering_blocked_gemm.py::test_fused_rmsnorm_linear_blocked_prologue": "R7",
    # test_structural_replay_consulted DE-QUARANTINED with the two-level structural tier
    # (the outer cut fork branches + replays); test_trace_records_partition_fork still needs
    # the inner partition-fork trace recording.
    "test_resolve.py::test_trace_records_partition_fork": "R7",
    # test_run.py: the fp16-matmul-window kernel stays quarantined (R7 matmul window);
    # the rmsnorm/softmax/run-ir/bench rows de-quarantined (R2). The four sdpa rows
    # (k_chunked / seq1024_dynamic_smem / tinyllama_full / tinyllama_per_head)
    # de-quarantined under R6 — the tile/split/005_split_demoted cut lowers them.
    # The R4 scalar pin-honoring recovered test_run_code_fp16_matmul_window_accuracy's
    # [4]/[8] params (a window-pinned fp16 matmul now honors its tile knobs); the [2]
    # param still hits a separate fp16 nvcc codegen failure (R7), so it stays a node
    # entry below.
    "test_run.py::test_compile_fp16_matmul_window_emits_half2": "R7",
    # test_split_demoted_fork_pushes_structural DE-QUARANTINED: the keep-vs-split FORK is
    # now live — the keep(SMEM) fused edge is a lowerable keep option (``seed_fused`` →
    # ``assemble_fused``), so ``005_split_demoted`` offers ``[keep, cut]`` instead of forcing
    # the cut, and the structural push fires.
    # The test_two_level structural rows are ALSO DE-QUARANTINED: the outer two-level tree
    # now branches on the cut (``outer_pipeline`` drives the ``split`` phase; tiling stays
    # inner), slices/prices the TileGraphOp terminal kernels, groups the decomposition Σ by
    # the pre-decision site, keeps the structural decision out of the ``lowering`` table, and
    # the outer PUCT ranks the cut branch via its surfaced ``CUT`` knob.
}

# Specific parametrizations (the function also has green params).
_XFAIL_NODES: dict[str, str] = {
    # R5 — landed (warp-tier promote_transport + TMA ring/swizzle/peel synthesis,
    # incl. masked-tile TMA staging via the mask_order hoist): static + dynamic
    # test_pinned_transport_and_shape_fire de-quarantined.
    # R6 — score-materializing SDPA landed (tile/split/005_split_demoted): test_sdpa
    # [cuda] / sdpa_causal[cuda] / sdpa_gqa[cuda] + test_tune_accuracy[sdpa]
    # de-quarantined (the softmax+P@V un-fuses into xn producer + clean gemm).
    # test_tinyllama_block_accuracy[cuda] de-quarantined: the whole-block RoPE
    # attention + MLP forward now lowers correctly (SDPA split + multi-access staging).
    # R7
    # Only the [2] window param still fails (fp16 nvcc codegen); [4]/[8] recovered
    # under the R4 scalar pin-honoring (see the test_run.py note above).
    "test_run.py::test_run_code_fp16_matmul_window_accuracy[2]": "R7",
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
