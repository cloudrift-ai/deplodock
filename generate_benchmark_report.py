#!/usr/bin/env python3
"""
Generate Excel report from vLLM benchmark results with pricing information.
"""

import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd


def load_config(config_file: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def extract_gpu_type(server_name: str) -> str:
    """Extract GPU type from server name.

    Examples:
        rtx4090_x_1 -> rtx4090
        rtx5090_x_4 -> rtx5090
        pro6000_x_1 -> pro6000
    """
    match = re.match(r'([a-z0-9]+)_x_\d+', server_name.lower())
    if match:
        return match.group(1)
    return server_name


def extract_gpu_count(server_name: str) -> int:
    """Extract GPU count from server name.

    Examples:
        rtx4090_x_1 -> 1
        rtx5090_x_4 -> 4
    """
    match = re.search(r'_x_(\d+)', server_name)
    if match:
        return int(match.group(1))
    return 1


def parse_benchmark_result(result_file: Path) -> Tuple[float, Dict]:
    """Parse vLLM benchmark result file and extract total token throughput.

    Returns:
        Tuple of (total_throughput, all_metrics_dict)
    """
    with open(result_file, 'r') as f:
        content = f.read()

    # Extract total token throughput (tok/s)
    # Look for pattern like: "Total Token throughput (tok/s):       1332.32"
    total_match = re.search(r'Total [Tt]oken throughput \(tok/s\):\s+([\d.]+)', content)
    if not total_match:
        return None, {}

    total_throughput = float(total_match.group(1))

    # Extract other metrics for reference
    metrics = {}

    # Request throughput
    req_match = re.search(r'Request throughput \(req/s\):\s+([\d.]+)', content)
    if req_match:
        metrics['request_throughput'] = float(req_match.group(1))

    # Output token throughput
    output_match = re.search(r'Output token throughput \(tok/s\):\s+([\d.]+)', content)
    if output_match:
        metrics['output_throughput'] = float(output_match.group(1))

    # Mean TTFT (Time to First Token)
    ttft_match = re.search(r'Mean TTFT \(ms\):\s+([\d.]+)', content)
    if ttft_match:
        metrics['mean_ttft_ms'] = float(ttft_match.group(1))

    # Mean TPOT (Time Per Output Token)
    tpot_match = re.search(r'Mean TPOT \(ms\):\s+([\d.]+)', content)
    if tpot_match:
        metrics['mean_tpot_ms'] = float(tpot_match.group(1))

    return total_throughput, metrics


def get_gpu_price(config: dict, gpu_type: str, gpu_count: int) -> float:
    """Get GPU price from config.

    Args:
        config: Configuration dict
        gpu_type: GPU type (rtx4090, rtx5090, pro6000)
        gpu_count: Number of GPUs

    Returns:
        Price per hour
    """
    if 'pricing' not in config:
        return 0.0

    pricing = config['pricing']

    # Normalize gpu_type
    gpu_type_normalized = gpu_type.lower()

    # Try exact match first
    if gpu_type_normalized in pricing:
        price_per_gpu = pricing[gpu_type_normalized]
        return price_per_gpu * gpu_count

    # Try variations
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


def generate_report(config: dict, results_dir: str, output_file: str):
    """Generate Excel report from benchmark results."""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Find all vllm_benchmark.txt files
    benchmark_files = list(results_path.glob("*_vllm_benchmark.txt"))

    if not benchmark_files:
        print(f"Error: No benchmark result files found in {results_dir}")
        return

    print(f"Found {len(benchmark_files)} benchmark result files")

    # Parse results - organize by model
    data_by_model = {}
    all_data = []

    for result_file in benchmark_files:
        # Parse filename: {server_name}_{model_name}_vllm_benchmark.txt
        filename = result_file.stem  # Remove .txt
        parts = filename.rsplit('_vllm_benchmark', 1)[0]  # Remove _vllm_benchmark suffix

        # Split server and model
        # Format: server_name_model_name
        # Supports: h100_x_8_model or gcloud_RTXPro6000_model
        match = re.match(r'([^_]+_x_\d+)_(.+)', parts)
        if match:
            server_name = match.group(1)
            model_name = match.group(2).replace('_', '/')
        else:
            # Try alternative format: gcloud_GPUType_model
            match = re.match(r'(gcloud_[^_]+)_(.+)', parts)
            if match:
                server_name = match.group(1)
                model_name = match.group(2).replace('_', '/')
            else:
                print(f"Warning: Could not parse filename: {result_file.name}")
                continue

        # Parse benchmark results
        total_throughput, metrics = parse_benchmark_result(result_file)

        if total_throughput is None:
            print(f"Warning: Could not extract throughput from {result_file.name}")
            continue

        # Extract GPU info
        gpu_type = extract_gpu_type(server_name)
        gpu_count = extract_gpu_count(server_name)

        # Get price
        price = get_gpu_price(config, gpu_type, gpu_count)

        # Format machine name: 1x4090, 2x5090, etc
        # Simplify GPU type names
        gpu_short_name = gpu_type.upper().replace('RTX', '').replace('PRO', '')
        machine_name = f"{gpu_count}x{gpu_short_name}"

        # Calculate price per million tokens ($/mtok)
        # price_per_hour / (tokens_per_second * 3600 seconds) * 1,000,000 tokens
        price_per_mtok = (price / (total_throughput * 3600)) * 1_000_000 if total_throughput > 0 else 0

        row_data = {
            'Machine': machine_name,
            'Throughput (tok/s)': total_throughput,
            'GPU Price ($/hour)': price,
            'Token Price ($/mtok)': price_per_mtok,
        }

        # Add to all data
        all_data.append(row_data)

        # Add to model-specific data
        if model_name not in data_by_model:
            data_by_model[model_name] = []
        data_by_model[model_name].append(row_data)

    if not all_data:
        print("Error: No valid benchmark data found")
        return

    # Generate combined report
    df = pd.DataFrame(all_data)
    df = df.sort_values('Machine')

    output_path = Path(output_file)
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Benchmark Results', index=False)
        worksheet = writer.sheets['Benchmark Results']

        # Auto-adjust column widths
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

    print(f"✅ Combined report generated: {output_path}")
    print(f"   - {len(df)} benchmark results")

    # Generate per-model reports
    output_dir = output_path.parent
    for model_name, model_data in data_by_model.items():
        # Create safe filename from model name
        safe_model_name = model_name.replace('/', '_').replace(' ', '_')
        model_output_path = output_dir / f"benchmark_report_{safe_model_name}.xlsx"

        df_model = pd.DataFrame(model_data)
        df_model = df_model.sort_values('Machine')

        with pd.ExcelWriter(model_output_path, engine='openpyxl') as writer:
            df_model.to_excel(writer, sheet_name='Benchmark Results', index=False)
            worksheet = writer.sheets['Benchmark Results']

            # Auto-adjust column widths
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

        print(f"✅ Model report generated: {model_output_path}")
        print(f"   - {len(df_model)} results for {model_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Excel report from vLLM benchmark results'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--results-dir',
        default='results',
        help='Directory containing benchmark results (default: results)'
    )
    parser.add_argument(
        '--output',
        default='results/benchmark_report.xlsx',
        help='Output Excel file path (default: benchmark_report.xlsx)'
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Generate report
    generate_report(config, args.results_dir, args.output)


if __name__ == '__main__':
    main()
