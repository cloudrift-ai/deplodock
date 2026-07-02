# Tile IR: cleanup, consolidation, xfail fixes, debt — REMAINDER

Successor to the executed `tile-ir-rebuild.md` (deleted). Most of this plan has landed (2026-07-02):
the fused-edge xfails (broadcast `_peel` sink + the cooperative-prologue warp fusion), the recognize-time
contraction nodify, the scalar gmem→smem ring, the staged split-K threading, the bank-conflict oracle rebuild,
and the analytic-weight fitter rebuild + refit. The rebuild's mandate stays in force for what remains — one
hierarchical emitter, zero divergent codegen paths, zero back-compat shims.

The recovery contract is unchanged: `tests/compiler/e2e/` is black-box and MUST stay green.

## Remaining

- **vLLM plugin e2e validation** (`test_vllm_plugin_gpu.py` / `test_vllm_plugin_gen_gpu.py`, the last two xfail
  registry entries). The serving path is functionally correct (validated offline at S=512/4096); the blocker is
  cold-prior kernel latency (~0.36 s `k_linear_mean_reduce` launches × hundreds of kernels × vLLM's warmup
  forwards ≈ tens of minutes of 100% GPU). Re-run after the model tune trains the prior/DB the serving backend
  reads (`tune_db="auto"`); fix what still falls out.
- **Endgame purge (when the registry hits empty).** Delete `tests/xfail_registry.py`, the
  `pytest_collection_modifyitems` hook in `tests/conftest.py`, `TILE_ENTANGLED_FILES`, and unwrap every guarded
  `try/except ModuleNotFoundError` tile import. The recovery apparatus is itself the last shim.

## Verification

Per item: `make test` + `make lint`; staging changes assert bit-identity vs gmem-direct; `tile_signature`
invariance for any featurization-adjacent change.
