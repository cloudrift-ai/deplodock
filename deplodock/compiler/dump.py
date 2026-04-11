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
    from deplodock.compiler.backend.cuda.program import Program
    from deplodock.compiler.ir import Graph
    from deplodock.compiler.plan import ExecutionPlan
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
            return cls(dir=Path(dump_dir))
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

    def dump_program(self, program: Program) -> None:
        summary = {
            "name": program.name,
            "buffers": [{"name": b.name, "size": b.size, "dtype": b.dtype, "role": b.role} for b in program.buffers],
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
        for i, launch in enumerate(program.launches):
            self._write_text(f"40_kernel_{i:02d}_{launch.kernel_name}.cu", launch.kernel_source)

    def dump_source(self, source: str) -> None:
        self._write_text("50_full_program.cu", source)

    def dump_result(self, result: ProgramResult) -> None:
        data: dict = {"outputs": result.outputs}
        if result.time_ms is not None:
            data["time_ms"] = result.time_ms
        self._write_json("60_result.json", data)

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
