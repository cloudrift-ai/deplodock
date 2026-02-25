"""Unit tests for deploy/cloud module: resolve_vm_spec, provision_cloud_vm, delete_cloud_vm."""

import pytest
import yaml

from deplodock.provisioning.cloud import delete_cloud_vm, resolve_vm_spec
from deplodock.provisioning.types import VMConnectionInfo
from deplodock.recipe import load_recipe


def _load_entries(entries):
    """Pre-load recipe configs from raw entries for resolve_vm_spec."""
    loaded = []
    for entry in entries:
        config = load_recipe(entry["recipe"])
        loaded.append((entry, config))
    return loaded


# ── resolve_vm_spec ──────────────────────────────────────────────


def test_resolve_vm_spec_single_recipe(tmp_path):
    recipe = {
        "model": {"huggingface": "test/model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:latest"},
            }
        },
        "deploy": {
            "gpu": "NVIDIA GeForce RTX 5090",
            "gpu_count": 1,
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    entries = [{"recipe": str(tmp_path)}]
    gpu_name, gpu_count = resolve_vm_spec(_load_entries(entries))
    assert gpu_name == "NVIDIA GeForce RTX 5090"
    assert gpu_count == 1


def test_resolve_vm_spec_max_gpu_count(tmp_path):
    """Uses max gpu_count across all entries."""
    r1 = tmp_path / "r1"
    r1.mkdir()
    recipe1 = {
        "model": {"huggingface": "test/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:latest"}}},
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
    }
    with open(r1 / "recipe.yaml", "w") as f:
        yaml.dump(recipe1, f)

    r2 = tmp_path / "r2"
    r2.mkdir()
    recipe2 = {
        "model": {"huggingface": "test/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:latest"}}},
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 2},
    }
    with open(r2 / "recipe.yaml", "w") as f:
        yaml.dump(recipe2, f)

    entries = [
        {"recipe": str(r1)},
        {"recipe": str(r2)},
    ]
    gpu_name, gpu_count = resolve_vm_spec(_load_entries(entries))
    assert gpu_name == "NVIDIA GeForce RTX 5090"
    assert gpu_count == 2


def test_resolve_vm_spec_mixed_gpus_raises(tmp_path):
    """Raises ValueError when recipes target different GPUs."""
    r1 = tmp_path / "r1"
    r1.mkdir()
    recipe1 = {
        "model": {"huggingface": "test/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:latest"}}},
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
    }
    with open(r1 / "recipe.yaml", "w") as f:
        yaml.dump(recipe1, f)

    r2 = tmp_path / "r2"
    r2.mkdir()
    recipe2 = {
        "model": {"huggingface": "test/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:latest"}}},
        "deploy": {"gpu": "NVIDIA H100 80GB", "gpu_count": 1},
    }
    with open(r2 / "recipe.yaml", "w") as f:
        yaml.dump(recipe2, f)

    entries = [
        {"recipe": str(r1)},
        {"recipe": str(r2)},
    ]
    with pytest.raises(ValueError, match="mixed GPUs"):
        resolve_vm_spec(_load_entries(entries), server_name="test")


def test_resolve_vm_spec_missing_gpu_raises(tmp_path):
    """Raises ValueError when recipe has no deploy.gpu field."""
    recipe = {
        "model": {"huggingface": "test/model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "vllm": {"image": "vllm/vllm-openai:latest"},
            }
        },
    }
    with open(tmp_path / "recipe.yaml", "w") as f:
        yaml.dump(recipe, f)

    entries = [{"recipe": str(tmp_path)}]
    with pytest.raises(ValueError, match="missing 'deploy.gpu' field"):
        resolve_vm_spec(_load_entries(entries))


# ── delete_cloud_vm ──────────────────────────────────────────────


async def test_delete_cloud_vm_cloudrift_dry_run(caplog):
    with caplog.at_level("INFO"):
        await delete_cloud_vm(("cloudrift", "test-instance-id"), dry_run=True)
    assert "[dry-run]" in caplog.text
    assert "test-instance-id" in caplog.text


async def test_delete_cloud_vm_gcp_dry_run(caplog):
    with caplog.at_level("INFO"):
        await delete_cloud_vm(("gcp", "bench-test", "us-central1-b"), dry_run=True)
    assert "[dry-run]" in caplog.text
    assert "bench-test" in caplog.text


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
