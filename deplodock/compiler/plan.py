"""Backend-agnostic execution plan.

An ExecutionPlan describes WHAT to compute (operations, buffers, data flow)
without specifying HOW (no kernel source, no grid/block, no GPU API calls).
A Backend converts an ExecutionPlan into a runnable Program.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BufferSpec:
    """Buffer description: name, shape, dtype, and role."""

    name: str
    shape: tuple[int, ...]
    dtype: str = "f32"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"


@dataclass
class OpKernel:
    """One operation in the execution plan.

    The ``op`` field is a string tag that the backend looks up in its
    kernel registry (e.g., "rmsnorm" → rmsnorm.cu for CUDA, rmsnorm.hip
    for ROCm).  ``params`` carries op-specific configuration (dimensions,
    epsilon, scale, etc.) that the backend uses to compute grid/block and
    format kernel arguments.
    """

    op: str
    inputs: list[str]
    outputs: list[str]
    params: dict[str, int | float | str] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """Ordered sequence of operations on named buffers.

    Backend-agnostic: no kernel source, no grid/block, no GPU specifics.
    """

    name: str
    buffers: list[BufferSpec]
    ops: list[OpKernel]
