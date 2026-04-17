"""Loop program form: authoritative buffer shapes + LoopOp launch order.

Parallels ``program/gpu.py``'s ``GpuProgram`` one level up in the
compilation stack. Built by ``LoopProgram.from_graph`` after the rewriter
has finished fusion. Everything downstream (codegen) reads shapes from
here; nothing recomputes them.

The pairing:

    ir/loop.py   : LoopOp, Port, Mux, Combine, Assign           (structural IR)
    program/loop.py: LoopBuffer, LoopLaunch, LoopProgram          (program form)

    ir/gpu.py    : GpuKernel, GpuKernelParam, Stmts               (structural IR)
    program/gpu.py: GpuBuffer, GpuLaunch, GpuProgram              (program form)

``backend/cuda/emit.py::compile_kernels`` is the program-to-program
lowering: ``LoopProgram → GpuProgram``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from deplodock.compiler.ir.base import ConstantOp, InputOp, Op
from deplodock.compiler.ir.graph import Graph
from deplodock.compiler.ir.loop import Combine, Port


@dataclass
class LoopBuffer:
    """Buffer at loop-IR level: authoritative shape + role.

    The ``size`` (element count) is derivable from ``shape``; computing it
    requires concrete int dims (no symbolic dim names).
    """

    name: str
    shape: tuple[int | str, ...]
    dtype: str = "f32"
    role: str = "scratch"  # "input" | "output" | "constant" | "scratch"

    @property
    def size(self) -> int:
        """Element count — requires concrete dims (raises on symbolic)."""
        n = 1
        for d in self.shape:
            n *= int(d)
        return n


@dataclass
class LoopLaunch:
    """One op invocation wired to named buffers.

    ``loop`` is typically a ``LoopOp`` (for fused compute); LoopBackend
    also accepts any graph-level ``Op`` as a "direct-call" launch (evaluated
    via ``Op.forward``) to support ops that fusion didn't wrap (e.g.
    non-2-axis ``TransposeOp``, ``GatherOp``). CudaBackend only handles
    ``LoopOp`` launches and raises otherwise.

    ``input_names`` is in buffer / Port order — the i-th Port reads
    ``input_names[i]``. ``output_name`` identifies the buffer written.
    """

    loop: Op
    input_names: list[str]
    output_name: str


@dataclass
class LoopProgram:
    """Post-fusion program: ``LoopOp`` launches over named buffers.

    Single authoritative source for:
      - buffer shapes (``LoopBuffer.shape``),
      - roles (input / output / constant / scratch),
      - launch order (topological),
      - constant scalar values.

    Downstream consumers (codegen) read shapes via the helper methods;
    no one recomputes shapes from ``LoopOp.infer_shapes``.
    """

    name: str
    buffers: list[LoopBuffer]
    launches: list[LoopLaunch]
    graph_inputs: list[str] = field(default_factory=list)
    graph_outputs: list[str] = field(default_factory=list)
    graph_constants: list[str] = field(default_factory=list)
    constant_values: dict[str, float] = field(default_factory=dict)

    # ── shape queries ────────────────────────────────────────────────

    def shape(self, name: str) -> tuple:
        """Return the shape of the named buffer."""
        for b in self.buffers:
            if b.name == name:
                return tuple(b.shape)
        raise KeyError(f"Buffer {name!r} not in LoopProgram")

    def input_shapes(self, launch: LoopLaunch) -> tuple[tuple, ...]:
        """Shapes of the external buffers read by ``launch`` (Port order)."""
        return tuple(self.shape(n) for n in launch.input_names)

    def output_shape(self, launch: LoopLaunch) -> tuple:
        """Shape of the buffer written by ``launch``."""
        return self.shape(launch.output_name)

    def dollar_shapes(self, launch: LoopLaunch) -> dict[str, tuple]:
        """Map ``$N`` → shape for ``launch``'s Ports.

        A Port with an ``indexmap`` contributes the indexmap's ``out_shape``
        (the effective shape the body sees); a Port without one reads
        the external buffer shape from the program's buffer table.
        """
        out: dict[str, tuple] = {}
        port_idx = 0

        def record(port: Port) -> None:
            nonlocal port_idx
            key = f"${port_idx}"
            if port.indexmap is not None:
                out[key] = tuple(port.indexmap.out_shape)
            else:
                out[key] = self.shape(launch.input_names[port_idx])
            port_idx += 1

        for inp in launch.loop.inputs:
            if isinstance(inp, Port):
                record(inp)
            elif isinstance(inp, Combine):
                for src in inp.sources:
                    if isinstance(src, Port):
                        record(src)
        return out

    # ── construction ─────────────────────────────────────────────────

    @classmethod
    def from_graph(cls, graph: Graph, name: str = "prog") -> LoopProgram:
        """Build a ``LoopProgram`` from a fully-fused graph.

        After fusion the graph contains only ``LoopOp`` compute nodes plus
        ``InputOp`` / ``ConstantOp`` boundary sentinels. Every node
        contributes one buffer (keyed by node id); every ``LoopOp`` node
        contributes one launch.
        """
        graph_inputs = list(graph.inputs)
        graph_outputs = list(graph.outputs)
        graph_input_set = set(graph_inputs)
        graph_output_set = set(graph_outputs)

        graph_constants: list[str] = []
        constant_values: dict[str, float] = {}
        for nid, node in graph.nodes.items():
            if isinstance(node.op, ConstantOp):
                graph_constants.append(nid)
                if node.op.value is not None:
                    constant_values[nid] = node.op.value
        graph_constant_set = set(graph_constants)

        def _role(nid: str) -> str:
            if nid in graph_input_set:
                return "input"
            if nid in graph_constant_set:
                return "constant"
            if nid in graph_output_set:
                return "output"
            return "scratch"

        # One LoopBuffer per graph node, keyed by node id.
        buffers = [
            LoopBuffer(
                name=nid,
                shape=tuple(node.output.shape),
                dtype=node.output.dtype,
                role=_role(nid),
            )
            for nid, node in graph.nodes.items()
        ]

        # One LoopLaunch per compute node, in topological order. Any graph-level
        # op that isn't Input/Constant becomes a launch; fused nodes contain a
        # LoopOp (typical), while unfused primitives carry their original op
        # and LoopBackend evaluates them via Op.forward directly.
        launches: list[LoopLaunch] = []
        for nid in graph.topological_order():
            node = graph.nodes[nid]
            if isinstance(node.op, (InputOp, ConstantOp)):
                continue
            launches.append(
                LoopLaunch(
                    loop=node.op,
                    input_names=list(node.inputs),
                    output_name=nid,
                )
            )

        return cls(
            name=name,
            buffers=buffers,
            launches=launches,
            graph_inputs=graph_inputs,
            graph_outputs=graph_outputs,
            graph_constants=graph_constants,
            constant_values=constant_values,
        )
