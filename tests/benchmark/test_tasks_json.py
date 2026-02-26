"""Tests for tasks.json write/read and parse_task_from_result."""

from deplodock.benchmark import parse_task_from_result, read_tasks_json, write_tasks_json


def test_tasks_json_round_trip(tmp_path):
    tasks = [
        {
            "variant": "rtx5090_c8_mcr8",
            "result_file": "rtx5090_c8_mcr8_vllm_benchmark.txt",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_short": "rtx5090",
            "gpu_count": 1,
            "model_name": "QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ",
        },
    ]
    write_tasks_json(tmp_path, tasks)

    result = read_tasks_json(tmp_path)
    assert len(result) == 1
    assert result[0]["variant"] == "rtx5090_c8_mcr8"
    assert result[0]["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert "status" not in result[0]


def test_tasks_json_multiple(tmp_path):
    tasks = [
        {
            "variant": "V1",
            "result_file": "V1_vllm_benchmark.txt",
            "gpu_name": "G",
            "gpu_short": "g",
            "gpu_count": 1,
            "model_name": "m",
        },
        {
            "variant": "V2",
            "result_file": "V2_vllm_benchmark.txt",
            "gpu_name": "G",
            "gpu_short": "g",
            "gpu_count": 2,
            "model_name": "m",
        },
    ]
    write_tasks_json(tmp_path, tasks)

    result = read_tasks_json(tmp_path)
    assert len(result) == 2
    assert result[0]["variant"] == "V1"
    assert result[1]["gpu_count"] == 2


def test_parse_task_from_result(tmp_path):
    content = """\
============ Benchmark Task ============
recipe_dir: experiments/MyModel/experiment1
variant: rtx5090
gpu_name: NVIDIA GeForce RTX 5090
gpu_count: 1
recipe:
  model:
    huggingface: org/MyModel
  engine:
    llm:
      context_length: 8192
  benchmark:
    max_concurrency: 16
  deploy:
    gpu: NVIDIA GeForce RTX 5090
    gpu_count: 1
==================================================

============ Serving Benchmark Result ============
Total token throughput (tok/s):                 5000.00
"""
    result_file = tmp_path / "rtx5090_vllm_benchmark.txt"
    result_file.write_text(content)

    meta = parse_task_from_result(result_file)
    assert meta is not None
    assert meta["variant"] == "rtx5090"
    assert meta["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert meta["gpu_count"] == 1
    assert meta["model_name"] == "org/MyModel"


def test_parse_task_from_result_missing_section(tmp_path):
    result_file = tmp_path / "plain_benchmark.txt"
    result_file.write_text("Total token throughput (tok/s): 5000.00\n")

    meta = parse_task_from_result(result_file)
    assert meta is None
