"""Display graph summary and structure."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_inspect_command(subparsers):
    parser = subparsers.add_parser("inspect", help="Display graph IR summary")
    parser.add_argument("ir_file", help="Path to a .json Graph IR file")
    parser.set_defaults(func=handle_inspect)


def handle_inspect(args):
    from deplodock.compiler.ir import Graph

    path = Path(args.ir_file)
    if not path.exists():
        logger.error("File not found: %s", path)
        return

    with open(path) as f:
        data = json.load(f)

    graph = Graph.from_dict(data)

    logger.info("Graph: %d nodes, %d inputs, %d outputs", len(graph.nodes), len(graph.inputs), len(graph.outputs))

    # Count ops by type, with sub-counts for parameterized ops.
    ops_count: dict[str, int] = {}
    ops_detail: dict[str, dict[str, int]] = {}

    for n in graph.nodes.values():
        op_name = type(n.op).__name__
        ops_count[op_name] = ops_count.get(op_name, 0) + 1

        # Collect sub-details for parameterized ops.
        detail_key = None
        if hasattr(n.op, "fn"):
            detail_key = n.op.fn
        elif hasattr(n.op, "name") and op_name == "ConstantOp":
            detail_key = n.op.name

        if detail_key:
            if op_name not in ops_detail:
                ops_detail[op_name] = {}
            ops_detail[op_name][detail_key] = ops_detail[op_name].get(detail_key, 0) + 1

    logger.info("Op breakdown:")
    for op_name, count in sorted(ops_count.items()):
        detail = ops_detail.get(op_name)
        if detail:
            detail_str = ", ".join(f"{k}:{v}" for k, v in sorted(detail.items()))
            logger.info("  %-24s %d  (%s)", op_name, count, detail_str)
        else:
            logger.info("  %-24s %d", op_name, count)

    # Show input/output names.
    if graph.inputs:
        input_names = [graph.nodes[nid].output.name for nid in graph.inputs if nid in graph.nodes]
        logger.info("Inputs: %s", ", ".join(input_names))
    if graph.outputs:
        output_names = [graph.nodes[nid].output.name for nid in graph.outputs if nid in graph.nodes]
        logger.info("Outputs: %s", ", ".join(output_names))
