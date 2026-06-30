"""End-to-end tests for the orchestrator's cross-candidate fallback behavior."""

import logging
from unittest.mock import AsyncMock, patch

import pytest

from emmy.hardware import GPU_GCP_ZONES
from emmy.provisioning.cloud import provision_cloud_vm
from emmy.provisioning.errors import CapacityExhausted, TerminalProvisionError
from emmy.provisioning.types import VMConnectionInfo


@pytest.fixture
def ssh_key(tmp_path):
    """Write a fake SSH keypair and return the private-key path."""
    priv = tmp_path / "id_ed25519"
    priv.write_text("private")
    pub = tmp_path / "id_ed25519.pub"
    pub.write_text("ssh-ed25519 AAAA test@host\n")
    return str(priv)


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("emmy.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_fallback_to_next_cloudrift_type_on_capacity_exhausted(mock_create, ssh_key):
    """RTX 4090 has 4 CloudRift candidates; first 503s, second succeeds."""
    success = VMConnectionInfo(host="1.2.3.4", username="riftuser", ssh_port=22222)
    mock_create.side_effect = [
        CapacityExhausted("rtx49-10c-kn full"),
        success,
    ]

    conn = await provision_cloud_vm(
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        ssh_key=ssh_key,
        provider="cloudrift",
    )

    assert conn is success
    # Verify the orchestrator advanced to the second candidate, not retried the first.
    assert mock_create.await_count == 2
    types_tried = [call.kwargs["instance_type"] for call in mock_create.await_args_list]
    assert types_tried == ["rtx49-10c-kn.1", "rtx49-7-50-500-nr.1"]


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("emmy.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_orchestrator_returns_none_when_all_candidates_capacity_exhausted(mock_create, ssh_key):
    """If every candidate raises CapacityExhausted, the orchestrator returns None."""
    mock_create.side_effect = CapacityExhausted("no capacity")

    conn = await provision_cloud_vm(
        gpu_name="NVIDIA GeForce RTX 4090",
        gpu_count=1,
        ssh_key=ssh_key,
        provider="cloudrift",
    )

    assert conn is None
    # Four CloudRift entries listed for RTX 4090 in the hardware table.
    assert mock_create.await_count == 4


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("emmy.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_terminal_error_aborts_immediately(mock_create, ssh_key):
    """TerminalProvisionError must propagate and stop trying further candidates."""
    mock_create.side_effect = TerminalProvisionError("bad credentials")

    with pytest.raises(TerminalProvisionError):
        await provision_cloud_vm(
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_count=1,
            ssh_key=ssh_key,
            provider="cloudrift",
        )

    # First candidate only — no fallback on terminal errors.
    assert mock_create.await_count == 1


@patch("emmy.provisioning.cloud.gcp_provider.create_instance", new_callable=AsyncMock)
async def test_gcp_iterates_zones_before_giving_up(mock_create, ssh_key):
    """B200 lists multiple GCP zones; the orchestrator must try the second after the first fails."""
    success = VMConnectionInfo(host="10.0.0.1", username="", ssh_port=22)
    mock_create.side_effect = [
        CapacityExhausted("zone 1 exhausted"),
        success,
    ]

    conn = await provision_cloud_vm(
        gpu_name="NVIDIA B200",
        gpu_count=8,
        ssh_key=ssh_key,
        provider="gcp",
    )

    assert conn is not None
    zones_tried = [call.kwargs["zone"] for call in mock_create.await_args_list]
    assert zones_tried == GPU_GCP_ZONES["NVIDIA B200"][:2]


@patch("emmy.provisioning.cloud.gcp_provider.create_instance", new_callable=AsyncMock)
async def test_provisioning_model_override_flows_to_gcp(mock_create, ssh_key):
    """An explicit --provisioning-model override reaches gcp_provider.create_instance."""
    mock_create.return_value = VMConnectionInfo(host="10.0.0.1", username="", ssh_port=22)
    await provision_cloud_vm(
        gpu_name="NVIDIA B200",
        gpu_count=8,
        ssh_key=ssh_key,
        provider="gcp",
        provisioning_model="STANDARD",
    )
    assert mock_create.await_args_list[0].kwargs["provisioning_model"] == "STANDARD"


@patch("emmy.provisioning.cloud.gcp_provider.create_instance", new_callable=AsyncMock)
async def test_provisioning_model_defaults_to_hardware_table(mock_create, ssh_key):
    """Without an override, B200 falls back to the hardware-table default (FLEX_START)."""
    mock_create.return_value = VMConnectionInfo(host="10.0.0.1", username="", ssh_port=22)
    await provision_cloud_vm(gpu_name="NVIDIA B200", gpu_count=8, ssh_key=ssh_key, provider="gcp")
    assert mock_create.await_args_list[0].kwargs["provisioning_model"] == "FLEX_START"


@patch.dict("os.environ", {"CLOUDRIFT_API_KEY": "test-key"})
@patch("emmy.provisioning.cloud.asyncio.sleep", new_callable=AsyncMock)
@patch("emmy.provisioning.cloud.cr_provider.create_instance", new_callable=AsyncMock)
async def test_transient_error_retries_same_candidate_then_advances(mock_create, mock_sleep, ssh_key, caplog):
    """A generic exception is treated as transient: retry same candidate, then advance."""
    success = VMConnectionInfo(host="1.2.3.4", username="riftuser", ssh_port=22222)
    # First candidate: two transient failures (exhausts same-candidate retries), then advance.
    # Second candidate: succeeds.
    mock_create.side_effect = [
        RuntimeError("transient 1"),
        RuntimeError("transient 2"),
        success,
    ]

    with caplog.at_level(logging.WARNING):
        conn = await provision_cloud_vm(
            gpu_name="NVIDIA GeForce RTX 4090",
            gpu_count=1,
            ssh_key=ssh_key,
            provider="cloudrift",
        )

    assert conn is success
    # Two attempts on first candidate + one on second.
    assert mock_create.await_count == 3
    types_tried = [call.kwargs["instance_type"] for call in mock_create.await_args_list]
    assert types_tried == ["rtx49-10c-kn.1", "rtx49-10c-kn.1", "rtx49-7-50-500-nr.1"]
    assert "advancing to next candidate" in caplog.text
