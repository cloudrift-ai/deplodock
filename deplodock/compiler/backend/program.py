"""GPU program abstraction: buffers, launches, and execution order.

Backend-agnostic: describes a complete GPU computation as a sequence of
kernel launches over named buffers. Backend-specific extensions (e.g.,
TMA descriptors for CUDA) subclass Launch to add extra fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Buffer:
    """GPU buffer specification."""

    name: str
    size: int  # total number of elements
    dtype: str = "float"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"


@dataclass
class Launch:
    """One kernel invocation."""

    kernel_source: str  # complete __global__ function
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[str]  # buffer names and scalar literals in param order
    smem_bytes: int = 0
    zero_outputs: list[str] = field(default_factory=list)  # buffers to zero before launch


@dataclass
class Program:
    """A complete GPU program: buffers + kernels + launch order."""

    name: str
    buffers: list[Buffer]
    launches: list[Launch]
    defines: dict[str, str] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)
    # Buffer aliases: {alias_name: target_name}. The alias shares the
    # target's device pointer (no separate allocation). Used for
    # reshape/transpose which are metadata-only ops.
    aliases: dict[str, str] = field(default_factory=dict)
