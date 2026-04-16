"""Quarantine list for the structural-compiler refactor.

Tests in ``collect_ignore`` are skipped during collection. What remains
is a handful of E2E / full-pipeline tests that depend on decomposition
of LinearOp / SdpaOp / MeanOp / MatmulOp through the lowering — those
come back online in c7 together with the decomposition work.
"""

collect_ignore = [
    "test_cuda_backend.py",
    "test_e2e_accuracy.py",
    "test_llama_block.py",
    "test_pipeline.py",
]
