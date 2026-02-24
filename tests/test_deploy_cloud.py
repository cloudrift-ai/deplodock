"""Unit tests for deploy/cloud module: resolve_vm_spec, provision_cloud_vm, delete_cloud_vm."""

import os

import pytest
import yaml

from deplodock.deploy.recipe import load_recipe
from deplodock.provisioning.cloud import resolve_vm_spec, delete_cloud_vm
from deplodock.provisioning.types import VMConnectionInfo


def _load_entries(entries):
    """Pre-load recipe configs from raw entries for resolve_vm_spec."""
    loaded = []
    for entry in entries:
        config = load_recipe(entry['recipe'], variant=entry.get('variant'))
        loaded.append((entry, config))
    return loaded


# ── resolve_vm_spec ──────────────────────────────────────────────


def test_resolve_vm_spec_single_recipe(recipes_dir):
    entries = [{"recipe": os.path.join(recipes_dir, "Qwen3-Coder-30B-A3B-Instruct-AWQ"), "variant": "RTX5090"}]
    gpu_name, gpu_count = resolve_vm_spec(_load_entries(entries))
    assert gpu_name == "NVIDIA GeForce RTX 5090"
    assert gpu_count == 1


def test_resolve_vm_spec_max_gpu_count(tmp_path):
    """Uses max gpu_count across all entries."""
    recipe = {
        "model": {"name": "test/model"},
        "backend": {"vllm": {"image": "vllm/vllm-openai:latest", "tensor_parallel_size": 1,
                              "pipeline_parallel_size": 1, "gpu_memory_utilization": 0.9, "extra_args": ""}},
        "variants": {
            "1x": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
            "2x": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 2},
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    entries = [
        {"recipe": str(tmp_path), "variant": "1x"},
        {"recipe": str(tmp_path), "variant": "2x"},
    ]
    gpu_name, gpu_count = resolve_vm_spec(_load_entries(entries))
    assert gpu_name == "NVIDIA GeForce RTX 5090"
    assert gpu_count == 2


def test_resolve_vm_spec_mixed_gpus_raises(tmp_path):
    """Raises ValueError when recipes target different GPUs."""
    recipe = {
        "model": {"name": "test/model"},
        "backend": {"vllm": {"image": "vllm/vllm-openai:latest", "tensor_parallel_size": 1,
                              "pipeline_parallel_size": 1, "gpu_memory_utilization": 0.9, "extra_args": ""}},
        "variants": {
            "rtx": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
            "h100": {"gpu": "NVIDIA H100 80GB", "gpu_count": 1},
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    entries = [
        {"recipe": str(tmp_path), "variant": "rtx"},
        {"recipe": str(tmp_path), "variant": "h100"},
    ]
    with pytest.raises(ValueError, match="mixed GPUs"):
        resolve_vm_spec(_load_entries(entries), server_name="test")


def test_resolve_vm_spec_missing_gpu_raises(tmp_path):
    """Raises ValueError when recipe has no gpu field."""
    recipe = {
        "model": {"name": "test/model"},
        "backend": {"vllm": {"image": "vllm/vllm-openai:latest", "tensor_parallel_size": 1,
                              "pipeline_parallel_size": 1, "gpu_memory_utilization": 0.9, "extra_args": ""}},
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    entries = [{"recipe": str(tmp_path)}]
    with pytest.raises(ValueError, match="missing 'gpu' field"):
        resolve_vm_spec(_load_entries(entries))


# ── delete_cloud_vm ──────────────────────────────────────────────


def test_delete_cloud_vm_cloudrift_dry_run(capsys):
    delete_cloud_vm(("cloudrift", "test-instance-id"), dry_run=True)
    output = capsys.readouterr().out
    assert "[dry-run]" in output
    assert "test-instance-id" in output


def test_delete_cloud_vm_gcp_dry_run(capsys):
    delete_cloud_vm(("gcp", "bench-test", "us-central1-b"), dry_run=True)
    output = capsys.readouterr().out
    assert "[dry-run]" in output
    assert "bench-test" in output


# ── VMConnectionInfo ─────────────────────────────────────────────


def test_vm_connection_info_address():
    conn = VMConnectionInfo(host="1.2.3.4", username="user", ssh_port=22222)
    assert conn.address == "user@1.2.3.4"


def test_vm_connection_info_address_no_username():
    conn = VMConnectionInfo(host="1.2.3.4", username="", ssh_port=22)
    assert conn.address == "1.2.3.4"


def test_vm_connection_info_defaults():
    conn = VMConnectionInfo(host="1.2.3.4", username="user")
    assert conn.ssh_port == 22
    assert conn.port_mappings == []
    assert conn.delete_info == ()
