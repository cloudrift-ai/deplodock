"""Report generation: Excel reports from benchmark results."""

from pathlib import Path

from deplodock.report.collector import collect_tasks_from_manifests
from deplodock.report.parser import parse_benchmark_result
from deplodock.report.pricing import get_gpu_price


def generate_report(config: dict, results_dir: str, output_file: str):
    """Generate Excel report from benchmark results."""
    import pandas as pd

    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    data_by_model = {}
    all_data = []

    for task_meta, result_file in collect_tasks_from_manifests(results_path):
        gpu_type = task_meta["gpu_short"]
        gpu_count = task_meta["gpu_count"]
        model_name = task_meta["model_name"]

        total_throughput, metrics = parse_benchmark_result(result_file)

        if total_throughput is None:
            print(f"Warning: Could not extract throughput from {result_file}")
            continue

        price = get_gpu_price(config, gpu_type, gpu_count)

        gpu_display = gpu_type.upper().replace("RTX", "").replace("PRO", "")
        machine_name = f"{gpu_count}x{gpu_display}"

        price_per_mtok = (price / (total_throughput * 3600)) * 1_000_000 if total_throughput > 0 else 0

        row_data = {
            "Machine": machine_name,
            "Throughput (tok/s)": total_throughput,
            "GPU Price ($/hour)": price,
            "Token Price ($/mtok)": price_per_mtok,
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
    df = df.sort_values("Machine")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Benchmark Results", index=False)
        worksheet = writer.sheets["Benchmark Results"]

        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except Exception:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width

    print(f"Combined report generated: {output_path}")
    print(f"   - {len(df)} benchmark results")

    output_dir = output_path.parent
    for model_name, model_data in data_by_model.items():
        safe_model_name = model_name.replace("/", "_").replace(" ", "_")
        model_output_path = output_dir / f"benchmark_report_{safe_model_name}.xlsx"

        df_model = pd.DataFrame(model_data)
        df_model = df_model.sort_values("Machine")

        with pd.ExcelWriter(model_output_path, engine="openpyxl") as writer:
            df_model.to_excel(writer, sheet_name="Benchmark Results", index=False)
            worksheet = writer.sheets["Benchmark Results"]

            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width

        print(f"Model report generated: {model_output_path}")
        print(f"   - {len(df_model)} results for {model_name}")
