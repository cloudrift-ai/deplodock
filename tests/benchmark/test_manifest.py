"""Tests for write_manifest / read_manifest round-trip."""

from deplodock.benchmark import read_manifest, write_manifest


def test_manifest_round_trip(tmp_path):
    tasks = [
        {
            "recipe": "MyModel",
            "variant": "RTX5090",
            "gpu_name": "NVIDIA GeForce RTX 5090",
            "gpu_short": "rtx5090",
            "gpu_count": 1,
            "model_name": "org/MyModel",
            "result_file": "MyModel/RTX5090_vllm_benchmark.txt",
            "status": "completed",
        },
    ]
    write_manifest(tmp_path, "2026-02-23T14:30:00", "abc123", ["MyModel"], tasks)

    result = read_manifest(tmp_path)
    assert result["timestamp"] == "2026-02-23T14:30:00"
    assert result["code_hash"] == "abc123"
    assert result["recipes"] == ["MyModel"]
    assert len(result["tasks"]) == 1
    assert result["tasks"][0]["variant"] == "RTX5090"
    assert result["tasks"][0]["status"] == "completed"


def test_manifest_multiple_tasks(tmp_path):
    tasks = [
        {
            "recipe": "A",
            "variant": "V1",
            "status": "completed",
            "gpu_name": "G",
            "gpu_short": "g",
            "gpu_count": 1,
            "model_name": "m",
            "result_file": "A/V1_vllm_benchmark.txt",
        },
        {
            "recipe": "B",
            "variant": "V2",
            "status": "failed",
            "gpu_name": "G",
            "gpu_short": "g",
            "gpu_count": 2,
            "model_name": "m",
            "result_file": "B/V2_vllm_benchmark.txt",
        },
    ]
    write_manifest(tmp_path, "2026-01-01T00:00:00", "def456", ["A", "B"], tasks)

    result = read_manifest(tmp_path)
    assert len(result["tasks"]) == 2
    assert result["tasks"][1]["status"] == "failed"
