"""GCP provider orphan-cleanup and capacity-error classification tests."""

from unittest.mock import AsyncMock, patch

import pytest

from emmy.provisioning.errors import CapacityExhausted, TerminalProvisionError
from emmy.provisioning.gcp import _classify_create_failure, create_instance


def test_classify_zone_exhausted_is_capacity():
    err = _classify_create_failure("ZONE_RESOURCE_POOL_EXHAUSTED: No resources available in this zone.")
    assert isinstance(err, CapacityExhausted)


def test_classify_quota_exceeded_is_capacity():
    err = _classify_create_failure("Quota 'GPUS_ALL_REGIONS' exceeded. QUOTA_EXCEEDED")
    assert isinstance(err, CapacityExhausted)


def test_classify_stockout_is_capacity():
    err = _classify_create_failure("Instance creation failed: STOCKOUT for machine type a3-highgpu-8g.")
    assert isinstance(err, CapacityExhausted)


def test_classify_unrelated_error_is_terminal():
    err = _classify_create_failure("ERROR: bad credentials or unauthorized")
    assert isinstance(err, TerminalProvisionError)


@patch("emmy.provisioning.gcp.wait_for_ssh", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.wait_for_status", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.run_shell_cmd", new_callable=AsyncMock)
async def test_create_instance_capacity_error_no_orphan(mock_shell, mock_wait, mock_ssh):
    """When the create command itself fails with a capacity error, no orphan exists yet."""
    mock_shell.return_value = (1, "", "ZONE_RESOURCE_POOL_EXHAUSTED at us-central1-b")

    with pytest.raises(CapacityExhausted):
        await create_instance("my-vm", "us-central1-b", "a3-highgpu-8g")

    # Only one shell call: the create itself. No follow-up delete because nothing was provisioned.
    assert mock_shell.await_count == 1
    mock_wait.assert_not_awaited()


@patch("emmy.provisioning.gcp.wait_for_ssh", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.wait_for_status", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.run_shell_cmd", new_callable=AsyncMock)
async def test_create_instance_orphan_cleanup_on_wait_timeout(mock_shell, mock_wait, mock_ssh):
    """When create succeeds but wait_for_status times out, the VM must be deleted before raising."""
    mock_shell.return_value = (0, "Created", "")
    mock_wait.return_value = False  # status never reached RUNNING

    with pytest.raises(CapacityExhausted):
        await create_instance("orphan-vm", "us-central1-b", "a3-highgpu-8g")

    # Two shell calls: create, then delete (orphan cleanup).
    assert mock_shell.await_count == 2
    delete_cmd = mock_shell.await_args_list[1].args[0]
    assert "delete" in delete_cmd
    assert "orphan-vm" in delete_cmd


@patch("emmy.provisioning.gcp.wait_for_ssh", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.wait_for_status", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.run_shell_cmd", new_callable=AsyncMock)
async def test_create_instance_orphan_cleanup_on_exception_during_wait(mock_shell, mock_wait, mock_ssh):
    """An unexpected exception during wait_for_status must still trigger orphan cleanup."""
    mock_shell.return_value = (0, "Created", "")
    mock_wait.side_effect = RuntimeError("gcloud crashed")

    with pytest.raises(RuntimeError, match="gcloud crashed"):
        await create_instance("orphan-vm", "us-central1-b", "a3-highgpu-8g")

    assert mock_shell.await_count == 2
    delete_cmd = mock_shell.await_args_list[1].args[0]
    assert "delete" in delete_cmd
    assert "orphan-vm" in delete_cmd


@patch("emmy.provisioning.gcp.wait_for_ssh", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.wait_for_status", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.run_shell_cmd", new_callable=AsyncMock)
async def test_create_instance_orphan_cleanup_on_ssh_failure(mock_shell, mock_wait, mock_ssh):
    """SSH never coming up should still terminate the orphan."""
    # 3 shell calls in success path: create, external-IP, plus delete in cleanup
    mock_shell.side_effect = [
        (0, "Created", ""),
        (0, "10.0.0.1", ""),
        (0, "Deleted", ""),
    ]
    mock_wait.return_value = True
    mock_ssh.return_value = False  # SSH never up

    with pytest.raises(RuntimeError, match="SSH never came up"):
        await create_instance("orphan-vm", "us-central1-b", "a3-highgpu-8g", wait_ssh=True)

    delete_cmd = mock_shell.await_args_list[-1].args[0]
    assert "delete" in delete_cmd
    assert "orphan-vm" in delete_cmd


@patch("emmy.provisioning.gcp.wait_for_ssh", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.wait_for_status", new_callable=AsyncMock)
@patch("emmy.provisioning.gcp.run_shell_cmd", new_callable=AsyncMock)
async def test_create_instance_swallows_cleanup_failure(mock_shell, mock_wait, mock_ssh, caplog):
    """A failed orphan-delete must not mask the original exception."""
    mock_shell.side_effect = [
        (0, "Created", ""),
        RuntimeError("delete also failed"),
    ]
    mock_wait.return_value = False  # triggers cleanup path

    with caplog.at_level("ERROR", logger="emmy.provisioning.gcp"):
        with pytest.raises(CapacityExhausted):
            await create_instance("orphan-vm", "us-central1-b", "a3-highgpu-8g")

    assert "Failed to delete orphan GCP instance" in caplog.text
