"""CUDA-specific TMA (Tensor Memory Accelerator) configuration.

The pipeline schedule (``SmemPipelineKLoop``) is backend-agnostic and
lives in ``backend/ir/loop_ir.py``.  This module holds only the
hardware-specific TMA descriptor references needed by the CUDA codegen
to emit ``cp.async.bulk`` + ``mbarrier`` inline PTX.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TMALoadConfig:
    """TMA descriptor references for async bulk copy.

    Passed via ``LoopProgram.tma_config`` so the CUDA codegen can emit
    the correct ``cp.async.bulk.tensor.2d`` instructions with the right
    descriptor addresses.
    """

    a_tma_ref: str  # "&A_tma" or "&A_tma[batch]"
    b_tma_ref: str  # "&B_tma" or "&B_tma[batch]"
