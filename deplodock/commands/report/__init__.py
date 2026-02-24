"""Report command: generate Excel reports from vLLM benchmark results."""

import json
import re
import sys
from pathlib import Path
from typing import Dict, Tuple

import yaml


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def parse_benchmark_result(result_file: Path) -> Tuple[float, Dict]:
    """Parse vLLM benchmark result file and extract total token throughput."""
    with open(result_file, 'r') as f:
        content = f.read()

    total_match = re.search(r'Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)', content)
    if not total_match:
        return None, {}

    total_throughput = float(total_match.group(1))

    metrics = {}

    req_match = re.search(r'Request throughput \(req/s\):\s+([\d.]+)', content)
    if req_match:
        metrics['request_throughput'] = float(req_match.group(1))

    output_match = re.search(r'Output token throughput \(tok/s\):\s+([\d.]+)', content)
    if output_match:
        metrics['output_throughput'] = float(output_match.group(1))

    ttft_match = re.search(r'Mean TTFT \(ms\):\s+([\d.]+)', content)
    if ttft_match:
        metrics['mean_ttft_ms'] = float(ttft_match.group(1))

    tpot_match = re.search(r'Mean TPOT \(ms\):\s+([\d.]+)', content)
    if tpot_match:
        metrics['mean_tpot_ms'] = float(tpot_match.group(1))

    return total_throughput, metrics


def get_gpu_price(config: dict, gpu_type: str, gpu_count: int) -> float:
    """Get GPU price from config."""
    if 'pricing' not in config:
        return 0.0

    pricing = config['pricing']
    gpu_type_normalized = gpu_type.lower()

    if gpu_type_normalized in pricing:
        price_per_gpu = pricing[gpu_type_normalized]
        return price_per_gpu * gpu_count

    variations = {
        'rtx4090': ['4090', 'rtx_4090'],
        'rtx5090': ['5090', 'rtx_5090'],
        'pro6000': ['6000', 'rtx_6000', 'rtx6000', 'quadro_rtx_6000']
    }

    for base_name, alternatives in variations.items():
        if gpu_type_normalized in alternatives or base_name == gpu_type_normalized:
            if base_name in pricing:
                price_per_gpu = pricing[base_name]
                return price_per_gpu * gpu_count

    return 0.0


def _collect_tasks_from_manifests(results_path: Path):
    """Scan results_dir for run subdirectories containing manifest.json.

    Yields (task_meta, result_file_path) tuples for each completed task.
    """
    for run_dir in sorted(results_path.iterdir()):
        manifest_path = run_dir / "manifest.json"
        if not run_dir.is_dir() or not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = json.load(f)
        for task in manifest.get("tasks", []):
            if task.get("status") != "completed":
                continue
            result_file = run_dir / task["result_file"]
            if result_file.exists():
                yield task, result_file


def generate_report(config: dict, results_dir: str, output_file: str):
    """Generate Excel report from benchmark results."""
    import pandas as pd

    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    data_by_model = {}
    all_data = []

    for task_meta, result_file in _collect_tasks_from_manifests(results_path):
        gpu_type = task_meta["gpu_short"]
        gpu_count = task_meta["gpu_count"]
        model_name = task_meta["model_name"]

        total_throughput, metrics = parse_benchmark_result(result_file)

        if total_throughput is None:
            print(f"Warning: Could not extract throughput from {result_file}")
            continue

        price = get_gpu_price(config, gpu_type, gpu_count)

        gpu_display = gpu_type.upper().replace('RTX', '').replace('PRO', '')
        machine_name = f"{gpu_count}x{gpu_display}"

        price_per_mtok = (price / (total_throughput * 3600)) * 1_000_000 if total_throughput > 0 else 0

        row_data = {
            'Machine': machine_name,
            'Throughput (tok/s)': total_throughput,
            'GPU Price ($/hour)': price,
            'Token Price ($/mtok)': price_per_mtok,
        }

        all_data.append(row_data)

        if model_name not in data_by_model:
            data_by_model[model_name] = []
        data_by_model[model_name].append(row_data)

    if not all_data:
        print("Error: No valid benchmark data found")
        return

    print(f"Found {len(all_data)} benchmark results across run directories")

    df = pd.DataFrame(all_data)
    df = df.sort_values('Machine')

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Benchmark Results', index=False)
        worksheet = writer.sheets['Benchmark Results']

        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"Combined report generated: {output_path}")
    print(f"   - {len(df)} benchmark results")

    output_dir = output_path.parent
    for model_name, model_data in data_by_model.items():
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        model_output_path = output_dir / f"benchmark_report_{safe_model_name}.xlsx"

        df_model = pd.DataFrame(model_data)
        df_model = df_model.sort_values('Machine')

        with pd.ExcelWriter(model_output_path, engine='openpyxl') as writer:
            df_model.to_excel(writer, sheet_name='Benchmark Results', index=False)
            worksheet = writer.sheets['Benchmark Results']

            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"Model report generated: {model_output_path}")
        print(f"   - {len(df_model)} results for {model_name}")


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
