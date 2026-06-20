"""Unit tests for deploy/cloud module: resolve_vm_spec, provision_cloud_vm, delete_cloud_vm."""

import logging
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from deplodock.provisioning.candidates import VmCandidate
from deplodock.provisioning.cloud import (
    _provision_cloudrift,
    _provision_gcp,
    _ssh_keys_metadata_value,
    delete_cloud_vm,
    read_public_key_files,
    resolve_vm_spec,
)
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
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
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
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
    }
    with open(r1 / "recipe.yaml", "w") as f:
        yaml.dump(recipe1, f)

    r2 = tmp_path / "r2"
    r2.mkdir()
    recipe2 = {
        "model": {"huggingface": "test/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
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
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
        "deploy": {"gpu": "NVIDIA GeForce RTX 5090", "gpu_count": 1},
    }
    with open(r1 / "recipe.yaml", "w") as f:
        yaml.dump(recipe1, f)

    r2 = tmp_path / "r2"
    r2.mkdir()
    recipe2 = {
        "model": {"huggingface": "test/model"},
        "engine": {"llm": {"tensor_parallel_size": 1, "vllm": {"image": "vllm/vllm-openai:v0.17.0"}}},
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
                "vllm": {"image": "vllm/vllm-openai:v0.17.0"},
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


# ── _provision_cloudrift ────────────────────────────────────────


def _cr_cand():
    return VmCandidate(provider="cloudrift", base_type="rtx49-7c-kn", instance_type="rtx49-7c-kn.1", zone=None)


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("deplodock.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_provision_cloudrift_billing_exempt(mock_create, tmp_path):
    """billing_exempt in providers_config is forwarded to create_instance."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="user", ssh_port=22222)

    providers_config = {"cloudrift": {"billing_exempt": True}}

    conn = await _provision_cloudrift(_cr_cand(), str(key_file), providers_config, False, logging.getLogger())

    assert conn is not None
    assert mock_create.call_args.kwargs["billing_exempt"] is True


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("deplodock.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_provision_cloudrift_no_billing_exempt(mock_create, tmp_path):
    """billing_exempt defaults to False when not in providers_config."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="user", ssh_port=22222)

    conn = await _provision_cloudrift(_cr_cand(), str(key_file), {}, False, logging.getLogger())

    assert conn is not None
    assert mock_create.call_args.kwargs["billing_exempt"] is False


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("deplodock.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_provision_cloudrift_network(mock_create, tmp_path):
    """network in providers_config is forwarded to create_instance."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="user", ssh_port=22222)

    providers_config = {"cloudrift": {"network": "public"}}

    conn = await _provision_cloudrift(_cr_cand(), str(key_file), providers_config, False, logging.getLogger())

    assert conn is not None
    assert mock_create.call_args.kwargs["network"] == "public"


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("deplodock.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_provision_cloudrift_no_network(mock_create, tmp_path):
    """network defaults to None when not in providers_config."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="user", ssh_port=22222)

    conn = await _provision_cloudrift(_cr_cand(), str(key_file), {}, False, logging.getLogger())

    assert conn is not None
    assert mock_create.call_args.kwargs["network"] is None


# ── read_public_key_files (extra authorized keys) ────────────────


def test_read_public_key_files_reads_all(tmp_path):
    a = tmp_path / "alice.pub"
    a.write_text("ssh-ed25519 AAAA alice@host\n")
    b = tmp_path / "bob.pub"
    b.write_text("ssh-ed25519 BBBB bob@host\n")

    keys = read_public_key_files([str(a), str(b)])

    assert keys == ["ssh-ed25519 AAAA alice@host", "ssh-ed25519 BBBB bob@host"]


def test_read_public_key_files_none_returns_empty():
    assert read_public_key_files(None) == []


def test_read_public_key_files_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing.pub"):
        read_public_key_files([str(tmp_path / "missing.pub")])


def test_read_public_key_files_empty_raises(tmp_path):
    empty = tmp_path / "empty.pub"
    empty.write_text("   \n")
    with pytest.raises(ValueError, match="empty"):
        read_public_key_files([str(empty)])


# ── _ssh_keys_metadata_value (GCP) ───────────────────────────────


def test_ssh_keys_metadata_value_single():
    assert _ssh_keys_metadata_value("deploy", "ssh-ed25519 AAAA own@host", None) == "deploy:ssh-ed25519 AAAA own@host"


def test_ssh_keys_metadata_value_multiple():
    value = _ssh_keys_metadata_value("deploy", "ssh-ed25519 AAAA own@host", ["ssh-ed25519 BBBB bob@host"])
    assert value == "deploy:ssh-ed25519 AAAA own@host\ndeploy:ssh-ed25519 BBBB bob@host"


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("deplodock.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_provision_cloudrift_forwards_extra_keys(mock_create, tmp_path):
    """extra_authorized_keys is forwarded to create_instance as extra_public_keys."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")
    pub_file = tmp_path / "id_ed25519.pub"
    pub_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="user", ssh_port=22222)

    extra = ["ssh-ed25519 BBBB bob@host"]
    conn = await _provision_cloudrift(_cr_cand(), str(key_file), {}, False, logging.getLogger(), extra)

    assert conn is not None
    assert mock_create.call_args.kwargs["extra_public_keys"] == extra


@patch("deplodock.provisioning.cloud.gcp_provider.create_instance", new_callable=AsyncMock)
async def test_provision_gcp_disables_oslogin_with_ssh_keys(mock_create, tmp_path):
    """GCP metadata pins enable-oslogin=FALSE alongside the per-VM ssh-keys (one --metadata flag),
    so the instance key is honored on a project that enables OS Login via metadata."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")
    (tmp_path / "id_ed25519.pub").write_text("ssh-ed25519 AAAA own@host\n")

    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="", ssh_port=22)

    cand = VmCandidate(provider="gcp", base_type="a3-highgpu-8g", instance_type="a3-highgpu-8g", zone="us-central1-a")
    conn = await _provision_gcp(cand, "NVIDIA B200", str(key_file), {"gcp": {"ssh_user": "deploy"}}, "srv", False, logging.getLogger())

    assert conn is not None
    extra = mock_create.call_args.kwargs["extra_gcloud_args"]
    assert "--metadata=enable-oslogin=FALSE,ssh-keys=deploy:ssh-ed25519 AAAA own@host" in extra


@patch("deplodock.provisioning.cloud.gcp_provider.create_instance", new_callable=AsyncMock)
async def test_provision_gcp_warns_when_pubkey_missing(mock_create, tmp_path, caplog):
    """A missing .pub no longer silently omits the key: it warns and adds no ssh-keys metadata."""
    key_file = tmp_path / "id_ed25519"
    key_file.write_text("private-key")  # no matching .pub
    mock_create.return_value = VMConnectionInfo(host="1.2.3.4", username="", ssh_port=22)

    cand = VmCandidate(provider="gcp", base_type="a3-highgpu-8g", instance_type="a3-highgpu-8g", zone="us-central1-a")
    with caplog.at_level(logging.WARNING):
        await _provision_gcp(cand, "NVIDIA B200", str(key_file), {}, "srv", False, logging.getLogger())

    assert any("No SSH public key" in r.message for r in caplog.records)
    extra = mock_create.call_args.kwargs["extra_gcloud_args"] or ""
    assert "ssh-keys" not in extra
