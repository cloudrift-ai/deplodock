"""Quarantine list for the structural-compiler refactor.

Tests in ``collect_ignore`` are skipped during collection. Each commit
in the refactor removes entries here as the corresponding pipeline stage
comes back online:

  c1 (this commit): every test that touches lowering, codegen, runtime,
                    or the legacy KernelOp shape is quarantined.
                    Only the L0 base suite (test_ir, test_shape_inference,
                    test_backend_ir) stays green.
  c2: test_kernel_op (rewritten against the new structural IR).
  c5: tracer tests (test_torch_trace*, test_real_trace, test_hints).
  c6: codegen tests (test_cuda*, test_kernel_gen, test_tuning,
                     test_indexmap once auto_fuse callsites are ported).
  c7: E2E numerical (test_e2e_accuracy, test_llama_block, test_pipeline,
                     test_program).
"""

collect_ignore = [
    "test_indexmap.py",
    "test_kernel_gen.py",
    "test_cuda.py",
    "test_cuda_backend.py",
    "test_tuning.py",
    "test_e2e_accuracy.py",
    "test_llama_block.py",
    "test_pipeline.py",
    "test_program.py",
]
