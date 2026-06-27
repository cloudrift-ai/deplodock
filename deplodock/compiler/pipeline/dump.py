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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from deplodock import config

if TYPE_CHECKING:
    from deplodock.compiler.backend.base import BenchmarkResult
    from deplodock.compiler.graph import Graph
    from deplodock.compiler.pipeline.pipeline import Pass, Rule

logger = logging.getLogger(__name__)

ENV_VAR = config.DUMP_DIR


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
        # Per-rule accumulator. Keyed by (pass_idx, pass_name, rule_name);
        # flushed to disk in ``on_pass`` so we write each rule's text/json
        # files once instead of N times for N rule applications.
        self._rule_records: dict[tuple[int, str, str], list[dict]] = {}
        self._rule_texts: dict[tuple[int, str, str], list[str]] = {}
        # Pristine, pre-decomposition Torch-dialect snapshot — the source for
        # the per-kernel ``.torch.json`` reproducers. Stashed by
        # ``dump_input_graph`` before any pass mutates the graph in place.
        self._input_graph: Graph | None = None

    @classmethod
    def from_env(cls) -> CompilerDump | None:
        """Create from DEPLODOCK_DUMP_DIR env var, or return None."""
        d = config.dump_dir()
        return cls(dir=d) if d else None

    @classmethod
    def resolve(cls, cli_dir: str | Path | None = None) -> CompilerDump | None:
        """Resolve dump dir from CLI arg (precedence) or env var. None if neither."""
        if cli_dir is not None:
            return cls(dir=Path(cli_dir))
        return cls.from_env()

    # --- Dump methods ---

    def on_rule(self, pass_: Pass, rule: Rule, record: dict, text: str) -> None:
        """Buffer one rule-application record for later flush in ``on_pass``.

        The text snapshot comes from ``_format_rule_application`` (or the
        in-place variant); the structured ``record`` mirrors it for
        post-hoc analysis. Both are written to
        ``{idx+1:02d}_{pass}__{rule}.rules.{txt,json}`` when the pass
        ends — ``pass_.index`` is the engine's 0-based pass index; the
        dump adds ``+1`` for filename display so the on-disk layout
        matches the human-readable pass ordering.
        """
        key = (pass_.index, pass_.name, rule.name)
        self._rule_records.setdefault(key, []).append(record)
        self._rule_texts.setdefault(key, []).append(text)

    def on_pass(self, pass_: Pass, graph: Graph) -> None:
        """Dump the graph after a pass as json / txt / dot (+ kernels.txt if any).

        Every pass is treated uniformly: no per-pass special cases, so
        adding a new pass automatically gets dumped. File names are
        ``{idx+1:02d}_{pass_name}.{ext}`` with slashes in ``pass_name``
        flattened to underscores — ``pass_.index`` is the engine's
        0-based pass index; the dump bumps it to 1-based for display.
        Also flushes any per-rule application snapshots accumulated
        during this pass to ``.rules.{txt,json}`` files alongside the
        post-pass graph dump.
        """
        idx, pass_name = pass_.index, pass_.name
        self._dump_graph(f"{idx + 1:02d}_{pass_name.replace('/', '_')}", graph)
        self._flush_rule_dumps(idx, pass_name)

    def _flush_rule_dumps(self, idx: int, pass_name: str) -> None:
        flat_pass = pass_name.replace("/", "_")
        for (pass_idx, pname, rule_name), records in list(self._rule_records.items()):
            if pass_idx != idx or pname != pass_name:
                continue
            prefix = f"{idx + 1:02d}_{flat_pass}__{rule_name}.rules"
            self._write_text(f"{prefix}.txt", "\n\n".join(self._rule_texts.pop((pass_idx, pname, rule_name))) + "\n")
            self._write_json(f"{prefix}.json", records)
            del self._rule_records[(pass_idx, pname, rule_name)]

    def dump_input_graph(self, graph: Graph) -> None:
        """Graph as captured by the tracer, before any rewriter pass runs.

        Stashes a *copy* — the pipeline mutates the passed graph in place, and
        the per-kernel reproducers (``_dump_torch_repro``) need the pristine
        frontend ops, keyed by the trace-time node ids that prov uses as
        origins."""
        self._input_graph = graph.copy()
        self._dump_graph("00_input", graph)

    def _dump_graph(self, prefix: str, graph: Graph) -> None:
        self._write_json(f"{prefix}.json", graph.to_dict())
        self._write_text(f"{prefix}.txt", graph.pretty_print())
        self._write_text(f"{prefix}.dot", _graph_to_dot(graph))
        kernels = format_kernels(graph)
        if kernels:
            self._write_text(f"{prefix}.kernels.txt", kernels)
            self._dump_per_kernel(prefix, graph)

    def _dump_per_kernel(self, prefix: str, graph: Graph) -> None:
        """Split each compute-kernel node (LoopOp / KernelOp /
        CudaOp) into its own minimal sub-graph and write to
        ``<prefix>.kernels/<kname>.json``. Each sub-graph contains the
        kernel node plus its transitive ``InputOp`` / ``ConstantOp``
        producers, so it can be loaded standalone via
        ``deplodock run --ir <subgraph>.json --bench`` for per-kernel
        diagnosis. Also writes ``<prefix>.kernels/<kname>.txt`` with
        the op's pretty-printed body (CUDA source for ``CudaOp``,
        loop / kernel IR for the lower-level dialects) so the readable
        version is co-located with the sub-graph reproducer.

        Multiple ``CudaOp`` nodes that share a ``kernel_name`` (a single
        kernel reused at several launch sites) are deduped on the
        ``.txt`` side — the body is identical so writing it once is
        enough — but each launch still gets its own ``.json``
        sub-graph since the input bindings differ per node."""
        from deplodock.compiler.ir.cuda.ir import CudaOp
        from deplodock.compiler.ir.kernel.ir import KernelOp
        from deplodock.compiler.ir.loop import LoopOp
        from deplodock.compiler.pipeline.search.slice import single_node_graph

        compute_types = (LoopOp, KernelOp, CudaOp)
        compute_nodes = [(nid, n) for nid, n in graph.nodes.items() if isinstance(n.op, compute_types)]
        if not compute_nodes:
            return

        out_dir = self.dir / f"{prefix}.kernels"
        out_dir.mkdir(parents=True, exist_ok=True)

        seen_cuda: set[str] = set()
        for nid, node in compute_nodes:
            kname = getattr(node.op, "kernel_name", None) or getattr(node.op, "name", None) or nid
            sub = single_node_graph(graph, nid)
            safe = self._safe_filename(kname)
            self._write_json(f"{prefix}.kernels/{safe}.json", sub.to_dict())
            self._dump_torch_repro(prefix, safe, node, graph)
            # Co-locate the pretty body. Skip duplicate CUDA kernel
            # names (same body across reuse sites).
            if isinstance(node.op, CudaOp):
                if kname in seen_cuda:
                    continue
                seen_cuda.add(kname)
            body = node.op.pretty_body()
            if body is not None:
                self._write_text(f"{prefix}.kernels/{safe}.txt", body)

    def _dump_torch_repro(self, prefix: str, safe: str, node, graph: Graph) -> None:
        """Write the original Torch ops a kernel implements as a standalone
        sub-graph (``<safe>.torch.json``) + a readable summary with per-origin
        ``i/N`` coverage (``<safe>.torch.txt``).

        Sliced from the pristine ``_input_graph`` by the kernel's prov origins,
        so it is always whole Torch ops — runnable via
        ``deplodock run --ir <f>.torch.json --bench`` to reproduce accuracy /
        latency vs torch for exactly those ops."""
        from deplodock.compiler import provenance  # noqa: PLC0415

        if self._input_graph is None:
            return
        node_prov = provenance.get(node)
        origins = {oid for oid in node_prov if oid in self._input_graph.nodes}
        if not origins:
            return
        sub = self._torch_repro_subgraph(origins)
        self._write_json(f"{prefix}.kernels/{safe}.torch.json", sub.to_dict())
        cov = provenance.coverage(node_prov, provenance.totals(graph))
        header = [f"# matching torch ops for kernel '{safe}'"]
        for oid in sorted(origins):
            have, total, full = cov[oid]
            header.append(f"#   {oid} ({node_prov[oid]['kind']}): {have}/{total} — {'full' if full else 'partial'}")
        self._write_text(f"{prefix}.kernels/{safe}.torch.txt", "\n".join(header) + "\n\n" + sub.pretty_print())

    def _torch_repro_subgraph(self, origins: set[str]) -> Graph:
        """Slice ``_input_graph`` to ``origins`` + their input closure.

        A frontend op feeding an origin but not itself an origin becomes a
        synthetic ``InputOp`` boundary (so the slice is standalone); constants
        and real graph inputs are kept. A feed that is a pure function of
        constants (e.g. the broadcast of an RMSNorm's pow-exponent scalar) keeps
        its whole constant chain instead — demoting it to an input would feed it
        random bench data, which is out of domain (``pow(x, random)`` → NaN) and
        breaks the reproducer's accuracy check. Outputs are the sink origins —
        those no other kept node consumes."""
        from deplodock.compiler.graph import Graph as _Graph  # noqa: PLC0415
        from deplodock.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415
        from deplodock.compiler.pipeline.search.slice import topo_order  # noqa: PLC0415

        src = self._input_graph
        keep: set[str] = set(origins)
        synthetic: set[str] = set()
        stack = [inp for oid in origins for inp in src.nodes[oid].inputs]
        while stack:
            cur = stack.pop()
            if cur in keep or cur not in src.nodes:
                continue
            if isinstance(src.nodes[cur].op, (ConstantOp, InputOp)):
                keep.add(cur)
                stack.extend(src.nodes[cur].inputs)
                continue
            const_chain = self._constant_closure(src, cur)
            if const_chain is not None:
                keep |= const_chain
            else:
                keep.add(cur)
                synthetic.add(cur)  # frontend op feeding an origin → boundary

        consumed = {inp for nid in keep for inp in src.nodes[nid].inputs if inp in keep}
        sub = _Graph()
        for kid in topo_order(src, keep):
            s = src.nodes[kid]
            if kid in synthetic:
                sub.add_node(InputOp(), [], s.output, node_id=s.id)
                sub.inputs.append(kid)
            else:
                sub.add_node(s.op, list(s.inputs), s.output, node_id=s.id)
                if isinstance(s.op, InputOp):
                    sub.inputs.append(kid)
        sub.outputs = [oid for oid in origins if oid not in consumed]
        return sub

    @staticmethod
    def _constant_closure(src: Graph, root: str) -> set[str] | None:
        """The transitive-input closure of ``root`` iff it is constant-derived —
        every leaf a ``ConstantOp`` — else ``None``. An ``InputOp`` anywhere in
        the closure means ``root`` carries runtime data and must stay a synthetic
        boundary."""
        from deplodock.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415

        closure: set[str] = set()
        stack = [root]
        while stack:
            cur = stack.pop()
            if cur in closure:
                continue
            node = src.nodes.get(cur)
            if node is None or isinstance(node.op, InputOp):
                return None
            closure.add(cur)
            if not isinstance(node.op, ConstantOp):
                stack.extend(node.inputs)
        return closure

    def _collect_subgraph(self, graph: Graph, root_id: str) -> set[str]:
        """Transitive-input closure for a compute node: itself + every
        ``ConstantOp`` / ``InputOp`` reachable via ``node.inputs``."""
        from deplodock.compiler.ir.base import ConstantOp, InputOp

        keep: set[str] = set()
        stack = [root_id]
        while stack:
            cur = stack.pop()
            if cur in keep:
                continue
            keep.add(cur)
            node = graph.nodes.get(cur)
            if node is None:
                continue
            if cur == root_id or isinstance(node.op, (ConstantOp, InputOp)):
                stack.extend(node.inputs)
        return keep

    @staticmethod
    def _safe_filename(name: str) -> str:
        return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)

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
        if result.e2e_ms is not None:
            data["e2e_ms"] = result.e2e_ms
            data["e2e_min_ms"] = result.e2e_min_ms
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


def _canonical_node_id(nid: str) -> str:
    """Collapse repeated ``merged_`` prefixes and a leading ``lift_`` for display.

    Fusion prefixes one ``merged_`` per merge step into the same consumer
    (``merged_merged_merged_lift_n0``); lifting prefixes ``lift_`` once.
    Both are implementation artifacts — readers want ``merged_n0``.
    """
    name = nid
    while name.startswith("merged_"):
        name = name[len("merged_") :]
    if name.startswith("lift_"):
        name = name[len("lift_") :]
    return f"merged_{name}" if nid.startswith("merged_") else name


def format_kernels(graph: Graph) -> str:
    """Render post-lowering kernel bodies for each compute node in ``graph``.

    Each op's own ``pretty_body()`` returns its rendered body (or None to
    skip — boundary sentinels and primitive tensor ops). ``CudaOp`` nodes
    that share a ``kernel_name`` are emitted only once. The surrounding
    syntax identifies the IR level, so block headers stay minimal:
    ``=== N: <name> ===``.

    Scalar ``ConstantOp`` inputs are inlined as literals: header
    ``in_k = load <buf>[0]`` lines for scalar constants are dropped, and
    the literal value is substituted at every use site in the body.
    Behaves like ``RenderCtx.literal_constants`` does for the cuda
    renderer — same idea, applied to the human-readable IR dump.
    """
    from deplodock.compiler.ir.base import ConstantOp
    from deplodock.compiler.ir.cuda import CudaOp

    rename_map = {nid: _canonical_node_id(nid) for nid in graph.nodes if _canonical_node_id(nid) != nid}

    seen_cuda: set[str] = set()
    blocks: list[str] = []
    i = 0
    for nid in graph.topological_order():
        node = graph.nodes[nid]
        op = node.op
        body = op.pretty_body()
        if body is None:
            continue
        if isinstance(op, CudaOp):
            if op.kernel_name in seen_cuda:
                continue
            seen_cuda.add(op.kernel_name)
            name = op.kernel_name
        elif hasattr(op, "name") and op.name:
            name = op.name
        else:
            name = f"{rename_map.get(nid, nid)} -> {node.output.name}"
        # Inline scalar constants before applying the canonicalization
        # rename: Load.input fields carry the producer's pre-rename id
        # (e.g. ``mul_1_c1``), which is also the graph node id we look
        # up here.
        scalar_inputs = _scalar_constant_inputs(graph, node, ConstantOp)
        if scalar_inputs:
            body = _inline_scalar_loads(body, scalar_inputs)
        # Apply the same canonicalization to the rendered body so Load/Write
        # references to peer nodes use clean names too. Replace longest names
        # first to avoid prefix-eating ``merged_lift_n0`` before ``merged_merged_lift_n0``.
        for old in sorted(rename_map, key=len, reverse=True):
            body = body.replace(old, rename_map[old])
        blocks.append(f"=== {i}: {name} ===")
        blocks.append(body)
        blocks.append("")
        i += 1
    return "\n".join(blocks)


def _scalar_constant_inputs(graph, node, constant_op_type) -> dict[str, float]:
    """``{producer_id: value}`` for every direct input of ``node`` whose
    producer is a 0-D ``ConstantOp`` with a captured scalar value."""
    out: dict[str, float] = {}
    for inp in node.inputs:
        if inp not in graph.nodes:
            continue
        op = graph.nodes[inp].op
        if isinstance(op, constant_op_type) and op.value is not None:
            out[inp] = float(op.value)
    return out


def _inline_scalar_loads(body: str, scalar_inputs: dict[str, float]) -> str:
    """Drop ``in_k = load <buf>[0]`` lines whose buf is a scalar constant
    and substitute the literal value at every use site in the body."""
    import re

    pat = re.compile(r"^(\s*)(\w+)\s*=\s*load\s+(\S+)\[0\]\s*$")
    name_to_lit: dict[str, str] = {}
    out_lines: list[str] = []
    for line in body.splitlines():
        m = pat.match(line)
        if m and m.group(3) in scalar_inputs:
            name_to_lit[m.group(2)] = format(scalar_inputs[m.group(3)], "g")
            continue
        out_lines.append(line)
    if not name_to_lit:
        return body
    name_pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in name_to_lit) + r")\b")
    return "\n".join(name_pat.sub(lambda m: name_to_lit[m.group(1)], ln) for ln in out_lines)


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
