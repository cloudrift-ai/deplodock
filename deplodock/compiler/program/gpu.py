"""GPU program form: buffers, launches, and execution order.

Backend-agnostic (within GPU land): describes a complete GPU computation
as a sequence of kernel launches over named buffers. Backend-specific
extensions (e.g. TMA descriptors for CUDA) subclass ``GpuLaunch`` to add
extra fields (see ``backend/cuda/program.py``'s ``CudaLaunch``).

``GpuProgram`` is the program-form pair of ``ir/gpu.py``'s ``GpuKernel``:
the latter describes one ``__global__`` function, the former describes
many of them wired together into a runnable program. This mirrors how
``LoopProgram`` (``program/loop.py``) pairs with ``ir/loop.py``'s
``LoopOp``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GpuBuffer:
    """GPU buffer specification.

    ``shape`` carries the buffer's declared shape; ``size`` is the derived
    element count.
    """

    name: str
    shape: tuple[int | str, ...]
    dtype: str = "float"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"

    @property
    def size(self) -> int:
        """Element count. Requires concrete int dims."""
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass
class GpuLaunch:
    """One GPU kernel invocation."""

    kernel_source: str  # complete __global__ function
    kernel_name: str
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    args: list[str]  # buffer names and scalar literals in param order
    smem_bytes: int = 0
    zero_outputs: list[str] = field(default_factory=list)  # buffers to zero before launch
    # Free-form text metadata captured at codegen time (e.g. pretty-printed
    # LoopOp the kernel was lowered from). Informational only — emitted as a
    # comment above the kernel source when dumping; never parsed.
    comment: str = ""


@dataclass
class GpuProgram:
    """A complete GPU program: buffers + kernels + launch order."""

    name: str
    buffers: list[GpuBuffer]
    launches: list[GpuLaunch]
    defines: dict[str, str] = field(default_factory=dict)
    includes: list[str] = field(default_factory=list)
    # Buffer aliases: {alias_name: target_name}. The alias shares the
    # target's device pointer (no separate allocation). Used for
    # reshape/transpose which are metadata-only ops.
    aliases: dict[str, str] = field(default_factory=dict)
    # Scalar constant values captured at trace time (graph-input role
    # buffers with a compile-time scalar). Callers (e.g. fixtures) may
    # auto-inject these at run time so tests don't have to supply them.
    constant_values: dict[str, float] = field(default_factory=dict)
    # Free-form text metadata (e.g. pretty-printed LoopProgram). Written
    # by codegen, consumed by dump tooling.
    comment: str = ""

    def shape(self, name: str) -> tuple:
        """Return the declared shape of the named buffer."""
        for b in self.buffers:
            if b.name == name:
                return tuple(b.shape)
        raise KeyError(f"Buffer {name!r} not in GpuProgram")

    def pretty_print(self) -> str:
        """Human-readable program listing: buffers, aliases, launch schedule."""
        lines: list[str] = []
        buf_names = {b.name for b in self.buffers}
        aliased = set(self.aliases.keys())

        # Header.
        n_real = sum(1 for b in self.buffers if b.name not in aliased)
        lines.append(f"# Program: {self.name}")
        lines.append(f"# {n_real} buffers, {len(self.aliases)} aliases, {len(self.launches)} launches")
        lines.append("")

        # Buffers grouped by role.
        for role in ("input", "constant", "output", "scratch"):
            bufs = [b for b in self.buffers if b.role == role and b.name not in aliased]
            if not bufs:
                continue
            for b in bufs:
                lines.append(f"{b.name} = buffer({b.size}, {b.dtype}, {b.role})")
            lines.append("")

        # Aliases.
        if self.aliases:
            for alias, target in self.aliases.items():
                lines.append(f"{alias} = alias({target})")
            lines.append("")

        # Launches.
        input_bufs_set = {b.name for b in self.buffers if b.role in ("input", "constant")} | aliased

        for launch in self.launches:
            # Extract kernel parameter names from source signature.
            import re

            sig_match = re.search(r"void \w+\(([^)]*)\)", launch.kernel_source)
            if sig_match:
                param_names = [p.strip().split()[-1].lstrip("*") for p in sig_match.group(1).split(",") if p.strip()]
            else:
                param_names = []

            # Build ordered (param_name, value) pairs: TMA descriptors first, then regular args.
            tma_descs = getattr(launch, "tma_descriptors", [])
            pairs: list[tuple[str, str]] = []
            for desc in tma_descs:
                pairs.append((desc.param_name, f"tma {desc.buffer}"))
            pi = len(tma_descs)  # param index (TMA params already consumed)
            for arg in launch.args:
                pname = param_names[pi] if pi < len(param_names) else arg
                pairs.append((pname, arg))
                pi += 1

            # Separate outputs from inputs.
            outs: list[str] = []
            rhs_pairs: list[tuple[str, str]] = []
            for pname, val in pairs:
                bare = val.removeprefix("tma ")
                is_buf = bare in buf_names or bare in aliased
                is_tma = val.startswith("tma ")
                if is_buf and not is_tma and bare not in input_bufs_set:
                    outs.append(bare)
                else:
                    rhs_pairs.append((pname, val))

            # Format: param=value, eliding param name when it matches the value.
            rhs_parts = []
            for pname, val in rhs_pairs:
                bare = val.removeprefix("tma ")
                if bare == pname or pname == val:
                    rhs_parts.append(val)
                else:
                    rhs_parts.append(f"{pname}={val}")
            rhs = ", ".join(rhs_parts)

            # Grid/block.
            gx, gy, gz = launch.grid
            bx, by, bz = launch.block
            grid_str = f"{gx}" if gy == 1 and gz == 1 else f"{gx}x{gy}x{gz}"
            block_str = f"{bx}" if by == 1 and bz == 1 else f"{bx}x{by}"

            # Annotations.
            notes: list[str] = []
            if launch.smem_bytes:
                notes.append(f"smem={launch.smem_bytes}")
            if launch.zero_outputs:
                notes.append(f"zero={','.join(launch.zero_outputs)}")
            suffix = f"  # {', '.join(notes)}" if notes else ""

            if outs:
                lhs = ", ".join(outs)
                lines.append(f"{lhs} = {launch.kernel_name}({rhs})  <<<{grid_str}, {block_str}>>>{suffix}")
            else:
                lines.append(f"{launch.kernel_name}({rhs})  <<<{grid_str}, {block_str}>>>{suffix}")

        return "\n".join(lines)
