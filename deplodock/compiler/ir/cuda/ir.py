"""CudaOp — graph-level wrapper around a rendered CUDA kernel.

Produced by ``passes/lowering/cuda`` by rendering each ``KernelOp`` body
to a ``__global__`` source string. The final graph before codegen is
``Graph[CudaOp + InputOp + ConstantOp]``; the CUDA backend walks it in
topological order, emits one ``kernel_name<<<grid, block>>>(args)``
launch per node, and wires buffer pointers by node id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import Op
from deplodock.compiler.ir.expr import Expr


@dataclass(frozen=True)
class TmaDescMeta:
    """Metadata the CUDA backend needs to encode a TMA descriptor at launch.

    ``name`` matches the kernel signature parameter (added to
    ``arg_order`` after the buffer args). ``src_buf`` names the graph
    buffer whose device pointer + shape feed
    ``cuTensorMapEncodeTiled``. ``box_extents`` and ``swizzle`` are the
    descriptor's per-dim box and swizzle mode."""

    name: str
    src_buf: str
    box_extents: tuple[int, ...]
    swizzle: str = "NONE"


# int = static factor; str = symbolic axis name (resolved at launch from
# sym_values); Expr = a composite extent (e.g. ceil-div ``(seq_len+15)//16`` for
# a hint-driven masked block axis) evaluated against sym_values at launch.
_GridFactor = int | str | Expr
GridDimSpec = tuple[_GridFactor, ...]  # product of factors → one grid dim's extent

# Reserved sym-value name for the device SM count. A Stream-K kernel's grid is
# ``((STREAMK_NUM_SMS,), (1,), (1,))``; the launch injects the live device's
# ``MultiProcessorCount`` under this key into ``sym_values`` so ``resolve_dim``
# resolves the grid to the actual SM count at launch (not baked at compile).
STREAMK_NUM_SMS = "__num_sms__"


@dataclass
class CudaOp(Op):
    """One CUDA kernel invocation as a graph-op.

    ``grid`` and ``block`` each carry three per-dim ``GridDimSpec`` tuples;
    every entry in a spec is multiplied together at launch time. Pure-int
    specs (e.g. ``((128,), (1,), (1,))``) describe static launch geometry;
    specs containing strings (e.g. ``(("seq_len",), (1,), (1,))``)
    reference symbolic dims that the launch resolver looks up in the
    runtime ``sym_values`` env. ``runtime_args`` lists those symbolic
    names in the order they appear in the kernel signature (one ``int``
    parameter per name, slotted after the buffer args and before any
    TMA descriptor params).
    """

    kernel_source: str = ""  # complete __global__ function
    kernel_name: str = ""
    arg_order: tuple[str, ...] = ()  # kernel-param names in positional order
    grid: tuple[GridDimSpec, GridDimSpec, GridDimSpec] = ((1,), (1,), (1,))
    block: tuple[GridDimSpec, GridDimSpec, GridDimSpec] = ((1,), (1,), (1,))
    smem_bytes: int = 0
    zero_outputs: tuple[str, ...] = ()
    comment: str = ""
    tma_descriptors: tuple[TmaDescMeta, ...] = field(default_factory=tuple)
    runtime_args: tuple[str, ...] = ()
    # Stream-K persistent-CTA launch. When set, ``(work_start, work_end)`` are
    # two ``const int*`` kernel params (in ``arg_order`` after the buffers /
    # descriptors); the launch allocates them as ``int32[num_sms]`` arrays and
    # fills each CTA's contiguous [start, end) slice of ``streamk_total_units``
    # tile-work units. The grid is ``num_sms`` (resolved via the reserved
    # ``STREAMK_NUM_SMS`` sym name, queried live at launch — not baked, so the
    # cached kernel is portable across SM counts). Empty = ordinary launch.
    streamk_work_arrays: tuple[str, str] | tuple[()] = ()
    streamk_total_units: int = 0

    def pretty_body(self) -> str:
        return self.kernel_source


def resolve_dim(spec, sym_values: dict[str, int]) -> int:
    """Multiply a ``GridDimSpec``'s factors, resolving ``str`` factors
    from ``sym_values`` and ``Expr`` factors via ``Expr.eval`` (e.g. a
    ceil-div block extent ``(seq_len+15)//16``). Accepts a bare ``int``
    (legacy static grid) as shorthand for a single-int spec — keeps
    pre-symbolic CudaOps working until every producer has been migrated.
    Raises ``KeyError`` on an unknown symbolic name."""
    if isinstance(spec, int):
        return spec
    total = 1
    for factor in spec:
        if isinstance(factor, int):
            total *= factor
        elif isinstance(factor, str):
            total *= sym_values[factor]
        else:
            total *= factor.eval(sym_values)
    return total
