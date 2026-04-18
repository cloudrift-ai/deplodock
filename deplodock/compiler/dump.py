"""Debug/diagnostics dump infrastructure.

When a dump directory is set, captures all intermediate compilation
artifacts to disk for debugging and performance analysis.

Activation:
    - DEPLODOCK_DUMP_DIR env var
    - --dump-dir CLI argument
    - dump_dir pytest fixture (writes to _test_data/<test_name>/)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deplodock.compiler.backend.base import BenchmarkResult, ProgramResult
    from deplodock.compiler.ir.gpu import GpuKernel
    from deplodock.compiler.ir.graph import Graph
    from deplodock.compiler.plan import ExecutionPlan
    from deplodock.compiler.program.gpu import GpuProgram
    from deplodock.compiler.program.loop import LoopProgram
    from deplodock.compiler.rewriter import PassTrace

logger = logging.getLogger(__name__)

ENV_VAR = "DEPLODOCK_DUMP_DIR"


@dataclass
class CompilerDump:
    """Artifact collector that writes intermediate compilation results to disk.

    Each dump method writes one or more files with a numbered prefix so that
    alphabetical listing matches pipeline order.
    """

    dir: Path

    def __post_init__(self) -> None:
        self.dir = Path(self.dir)
        # Clear any previous artifacts to avoid stale overlap.
        if self.dir.exists():
            import shutil

            shutil.rmtree(self.dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> CompilerDump | None:
        """Create from DEPLODOCK_DUMP_DIR env var, or return None."""
        dump_dir = os.environ.get(ENV_VAR)
        if dump_dir:
            return cls(dir=Path(dump_dir).expanduser())
        return None

    @classmethod
    def resolve(cls, cli_dir: str | Path | None = None) -> CompilerDump | None:
        """Resolve dump dir from CLI arg (precedence) or env var. None if neither."""
        if cli_dir is not None:
            return cls(dir=Path(cli_dir))
        return cls.from_env()

    # --- Dump methods ---

    def dump_input_graph(self, graph: Graph) -> None:
        self._write_json("00_input_graph.json", graph.to_dict())

    def dump_tensor_ir(self, graph: Graph) -> None:
        """Graph after decomposition + optimization passes, before fusion.

        This is the article's "Tensor IR" — only primitive ops (add/mul/sub/div/rsqrt/exp,
        sum/max reductions, reshape/transpose/slice, gather). High-level ops like
        rms_norm, linear, sdpa have been lowered to this canonical op set.

        Dumped twice: ``10_tensor_ir.json`` (round-trippable via Graph.from_dict)
        and ``10_tensor_ir.txt`` (human-readable, consumed by ``compile --ir tensor``).
        """
        self._write_json("10_tensor_ir.json", graph.to_dict())
        self._write_text("10_tensor_ir.txt", graph.pretty_print())

    def dump_pass(self, index: int, pt: PassTrace) -> None:
        prefix = f"{index + 1:02d}_pass_{pt.name}"
        data = pt.to_dict()
        if data.get("graph_before"):
            self._write_json(f"{prefix}_before.json", data["graph_before"])
        if data.get("graph_after"):
            self._write_json(f"{prefix}_after.json", data["graph_after"])
        if data.get("rules_applied"):
            self._write_json(f"{prefix}_rules.json", data["rules_applied"])

    def dump_passes(self, pass_traces: list[PassTrace]) -> None:
        for i, pt in enumerate(pass_traces):
            self.dump_pass(i, pt)

    def dump_fused_graph(self, graph: Graph) -> None:
        self._write_json("20_fused_graph.json", graph.to_dict())

    def dump_plan(self, plan: ExecutionPlan) -> None:
        summary = {
            "name": plan.name,
            "buffers": [{"name": b.name, "shape": list(b.shape), "dtype": b.dtype, "role": b.role} for b in plan.buffers],
            "ops": [
                {
                    "op": op.op,
                    "inputs": op.inputs,
                    "outputs": op.outputs,
                    "params": _safe_params(op.params),
                }
                for op in plan.ops
            ],
        }
        self._write_json("30_execution_plan.json", summary)

    def dump_loop_program(self, program: LoopProgram) -> None:
        """LoopProgram pretty-print (post-fusion).

        Writes both a kernel-only view (``37_loop_kernels.txt``, one LoopOp
        per launch with ``=== launch N: ... ===`` separators — analogous to
        ``39_kernel_ir.txt`` and ``40_kernels.cu``) and the full program view
        (``38_loop_program.txt``, buffers + launch schedule + bodies).
        """
        blocks: list[str] = []
        for i, launch in enumerate(program.launches):
            blocks.append(f"=== launch {i}: {launch.output_name} ===")
            blocks.append(program.pretty_print_launch(i))
            blocks.append("")
        self._write_text("37_loop_kernels.txt", "\n".join(blocks))
        self._write_text("38_loop_program.txt", program.pretty_print())

    def dump_kernel_ir(self, kernels: list[GpuKernel]) -> None:
        """Pretty-printed KernelIR AST for each kernel (pre-source-emission).

        This is the article's "Kernel IR" stage — the C-like AST with named axes
        still visible (``for j in [threadIdx.x, cols), step=blockDim.x``) before
        the tree-walk codegen renders it to CUDA source.
        """
        from deplodock.compiler.ir.gpu import pretty_print

        blocks: list[str] = []
        for i, kernel in enumerate(kernels):
            blocks.append(f"=== launch {i}: {kernel.name} ===")
            blocks.append(pretty_print(kernel))
            blocks.append("")
        self._write_text("39_kernel_ir.txt", "\n".join(blocks))

    def dump_program(self, program: GpuProgram) -> None:
        from deplodock.compiler.backend.cuda.program import generate_source

        summary = {
            "name": program.name,
            "buffers": [{"name": b.name, "size": b.size, "dtype": b.dtype, "role": b.role} for b in program.buffers],
            "aliases": program.aliases,
            "launches": [
                {
                    "kernel_name": launch.kernel_name,
                    "grid": list(launch.grid),
                    "block": list(launch.block),
                    "args": launch.args,
                    "smem_bytes": launch.smem_bytes,
                }
                for launch in program.launches
            ],
        }
        self._write_json("40_program_summary.json", summary)
        self._write_text("40_program.txt", program.pretty_print())

        # Program-level LoopIR metadata (populated by codegen).
        if program.comment:
            self._write_text("38_loop_program.txt", program.comment)

        # Kernel sources concatenated (deduplicated by name, same order as
        # generate_source), each preceded by its LoopIR metadata as a C++
        # block comment so reviewers can see the lowering source-of-truth.
        seen: set[str] = set()
        blocks: list[str] = []
        for launch in program.launches:
            if launch.kernel_name in seen:
                continue
            seen.add(launch.kernel_name)
            if launch.comment:
                banner = "\n".join(f" * {line}" if line else " *" for line in launch.comment.split("\n"))
                blocks.append(f"/*\n{banner}\n */\n{launch.kernel_source}")
            else:
                blocks.append(launch.kernel_source)
        self._write_text("40_kernels.cu", "\n\n".join(blocks))

        # Full nvcc input (kernels + host main) — reproduces what nvcc compiles.
        self._write_text("40_full_program.cu", generate_source(program, mode="benchmark"))

    def dump_source(self, source: str) -> None:
        self._write_text("50_full_program.cu", source)

    def dump_result(self, result: ProgramResult) -> None:
        # outputs are ndarrays; serialize as nested lists for JSON.
        data: dict = {"outputs": {n: arr.tolist() for n, arr in result.outputs.items()}}
        if result.time_ms is not None:
            data["time_ms"] = result.time_ms
        self._write_json("60_result.json", data)

    def dump_per_launch_values(self, per_launch: dict) -> None:
        """Dump per-kernel tensor snapshots from a debug run.

        ``per_launch`` maps launch_idx → {buf_name: ndarray}. Writes one
        JSON per launch so large runs don't produce a single giant file.
        """
        for li, bufs in per_launch.items():
            data = {name: arr.tolist() for name, arr in bufs.items()}
            self._write_json(f"70_launch_{li:03d}.json", data)

    def dump_benchmark(self, result: BenchmarkResult) -> None:
        data = {
            "time_ms": result.time_ms,
            "min_ms": result.min_ms,
            "max_ms": result.max_ms,
            "num_launches": result.num_launches,
        }
        self._write_json("60_benchmark.json", data)

    # --- Internal helpers ---

    def _write_json(self, filename: str, data: dict | list) -> None:
        path = self.dir / filename
        path.write_text(json.dumps(data, indent=2, default=str))
        logger.debug("Dumped %s", path)

    def _write_text(self, filename: str, text: str) -> None:
        path = self.dir / filename
        path.write_text(text)
        logger.debug("Dumped %s", path)


def _safe_params(params: dict) -> dict:
    """Serialize OpKernel params, skipping internal fields and handling non-JSON types."""
    safe: dict = {}
    for k, v in params.items():
        if k.startswith("_"):
            continue
        try:
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    return safe
