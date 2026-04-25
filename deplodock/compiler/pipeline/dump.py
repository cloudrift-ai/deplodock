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
    from deplodock.compiler.backend.base import BenchmarkResult, RunResult
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.plan import ExecutionPlan

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

    def on_pass(self, idx: int, pass_name: str, graph: Graph) -> None:
        """Dump the graph after a pass as json / txt / dot (+ kernels.txt if any).

        Every pass is treated uniformly: no per-pass special cases, so
        adding a new pass automatically gets dumped. File names are
        ``{idx:02d}_{pass_name}.{ext}`` with slashes in ``pass_name``
        flattened to underscores.
        """
        self._dump_graph(f"{idx:02d}_{pass_name.replace('/', '_')}", graph)

    def dump_input_graph(self, graph: Graph) -> None:
        """Graph as captured by the tracer, before any rewriter pass runs."""
        self._dump_graph("00_input", graph)

    def _dump_graph(self, prefix: str, graph: Graph) -> None:
        self._write_json(f"{prefix}.json", graph.to_dict())
        self._write_text(f"{prefix}.txt", graph.pretty_print())
        self._write_text(f"{prefix}.dot", _graph_to_dot(graph))
        kernels = format_kernels(graph)
        if kernels:
            self._write_text(f"{prefix}.kernels.txt", kernels)

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

    def dump_source(self, source: str) -> None:
        self._write_text("50_full_program.cu", source)

    def dump_result(self, result: RunResult) -> None:
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
        data: dict = {
            "time_ms": result.time_ms,
            "min_ms": result.min_ms,
            "max_ms": result.max_ms,
            "num_launches": result.num_launches,
        }
        if result.per_launch:
            data["per_launch"] = [{"idx": lt.idx, "kernel_name": lt.kernel_name, "time_ms": lt.time_ms} for lt in result.per_launch]
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


def format_kernels(graph: Graph) -> str:
    """Render post-lowering kernel bodies for each compute node in ``graph``.

    Dispatches on op type: ``LoopOp`` → loop body, ``TileOp`` → CUDA source
    (rendered on the fly), ``CudaOp`` → CUDA source. Non-kernel ops
    (Input/Constant/primitive tensor ops) are skipped. The surrounding
    syntax itself identifies the IR level, so the block headers stay
    minimal: ``=== N: <name> ===``.
    """
    from deplodock.compiler.ir.cuda import CudaOp
    from deplodock.compiler.ir.kernel import KernelOp
    from deplodock.compiler.ir.kernel.pretty import pretty_print as pp_kernel
    from deplodock.compiler.ir.loop import LoopOp
    from deplodock.compiler.ir.loop import pretty_print as pp_loop
    from deplodock.compiler.ir.tile import TileOp
    from deplodock.compiler.ir.tile.pretty import pretty_print as pp_tile

    seen_cuda: set[str] = set()
    blocks: list[str] = []
    i = 0
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        op = node.op
        if isinstance(op, LoopOp):
            port_buffers = [graph.nodes[src].output.name for src in node.inputs] + [node.output.name]
            name = f"{nid} -> {node.output.name}"
            body = pp_loop(op, port_buffers=port_buffers)
        elif isinstance(op, TileOp):
            name = op.name
            body = pp_tile(op)
        elif isinstance(op, KernelOp):
            name = op.name
            body = pp_kernel(op)
        elif isinstance(op, CudaOp):
            if op.kernel_name in seen_cuda:
                continue
            seen_cuda.add(op.kernel_name)
            name = op.kernel_name
            body = op.kernel_source
        else:
            continue
        blocks.append(f"=== {i}: {name} ===")
        blocks.append(body)
        blocks.append("")
        i += 1
    return "\n".join(blocks)


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


# ---------------------------------------------------------------------------
# Graphviz DOT emitter
# ---------------------------------------------------------------------------
#
# Renders a ``Graph`` as Graphviz DOT source. Plain text — no graphviz binary
# required at dump time; users render with ``dot -Tsvg foo.dot -o foo.svg``.


def _op_label(op: object) -> str:
    """Short op label for a DOT node."""
    cls = type(op).__name__
    if cls in ("ElementwiseOp", "ReduceOp", "ScanOp"):
        return getattr(op, "fn", cls.lower())
    if cls == "InputOp":
        return "input"
    if cls == "ConstantOp":
        name = getattr(op, "name", None)
        return f"const {name}" if name else "const"
    if cls == "GatherOp":
        return "gather"
    if cls == "ScatterOp":
        return "scatter"
    if cls == "IndexMapOp":
        return "index_map"
    return cls.removesuffix("Op").lower() or cls


def _node_style(op: object, is_output: bool) -> tuple[str, str]:
    """(shape, fillcolor) for a Graph Node based on its op."""
    cls = type(op).__name__
    if cls == "InputOp":
        return "ellipse", "#d0f0c0"
    if cls == "ConstantOp":
        return "ellipse", "#e0e0e0"
    if cls == "IndexMapOp":
        return "parallelogram", "#c0d8f0" if is_output else "white"
    return "box", "#c0d8f0" if is_output else "white"


def _fmt_shape(shape: tuple) -> str:
    if not shape:
        return "scalar"
    return "(" + ",".join(str(d) for d in shape) + ")"


def _graph_to_dot(graph: Graph) -> str:
    """Render a ``Graph`` as Graphviz DOT source.

    Nodes are walked in topological order for deterministic output. Each node
    is labeled with its id, op label, and output shape + dtype. Boundary
    sentinels get coloured ellipses; layout-only ``IndexMapOp`` gets a
    parallelogram (to reinforce that no math happens there). Output nodes get
    a light-blue fill regardless of shape.

    Consumed via ``dot -Tsvg FILE.dot -o FILE.svg`` (or -Tpng / -Tpdf).
    """
    outputs = set(graph.outputs)
    lines: list[str] = [
        "digraph G {",
        "  rankdir=TB;",
        '  node [style="rounded,filled", fontname="Helvetica", fontsize=10];',
        '  edge [arrowsize=0.7, color="#555555"];',
        "",
    ]
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        op_text = _op_label(node.op)
        shape = _fmt_shape(tuple(node.output.shape))
        dtype = node.output.dtype
        label = f"{nid}\\n{op_text}\\n{shape} {dtype}"
        dot_shape, fill = _node_style(node.op, nid in outputs)
        lines.append(f'  "{nid}" [label="{label}", shape={dot_shape}, fillcolor="{fill}"];')

    lines.append("")
    for nid in graph.topological_order():
        for src in graph.nodes[nid].inputs:
            lines.append(f'  "{src}" -> "{nid}";')

    lines.append("}")
    lines.append("")  # trailing newline
    return "\n".join(lines)
