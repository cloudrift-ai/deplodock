"""Report command CLI handler."""

from deplodock.report import (
    # Re-export for backward compatibility
    parse_benchmark_result,
    get_gpu_price,
    generate_report,
    collect_tasks_from_manifests as _collect_tasks_from_manifests,
)
from deplodock.report.collector import load_config


def handle_report(args):
    """Handle the report command."""
    config = load_config(args.config)
    generate_report(config, args.results_dir, args.output)


def register_report_command(subparsers):
    """Register the report subcommand."""
    parser = subparsers.add_parser(
        "report",
        help="Generate Excel report from benchmark results",
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)',
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory containing benchmark results (default: results)',
    )
    parser.add_argument(
        '--output',
        default='results/benchmark_report.xlsx',
        help='Output Excel file path (default: results/benchmark_report.xlsx)',
    )
    parser.set_defaults(func=handle_report)
