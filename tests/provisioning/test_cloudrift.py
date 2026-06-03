"""Unit tests for CloudRift API helper functions.

Response fixtures are captured from real CloudRift API calls.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from deplodock.provisioning.cloudrift import (
    API_VERSION,
    DEFAULT_CLOUDINIT_URL,
    DEFAULT_IMAGE_URL_AMD,
    DEFAULT_IMAGE_URL_NVIDIA,
    DEFAULT_IMAGE_URL_NVIDIA_PROPRIETARY,
    _add_ssh_key,
    _api_request,
    _ensure_ssh_key,
    _get_instance_info,
    _instance_fully_ready,
    _list_ssh_keys,
    _log_connection_info,
    _rent_instance,
    _terminate_instance,
    create_instance,
    select_image_url,
    wait_for_status,
)
from deplodock.provisioning.errors import CapacityExhausted, TerminalProvisionError

API_KEY = "test-api-key"
API_URL = "https://api.test.cloudrift.ai"


# ── Real API response fixtures ────────────────────────────────────

SSH_KEYS_LIST_RESPONSE = {
    "keys": [
        {
            "id": "3742bd6e-0bf3-11f0-9066-6f5a35a68ee9",
            "name": "Berserk",
            "public_key": "ssh-rsa AAAAB3NzaC1yc2EAAAA... user@desktop",
        },
        {
            "id": "5eb89e22-2a98-11f0-9c02-771633991431",
            "name": "MacBook",
            "public_key": "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID37 user@example.com",
        },
    ]
}

RENT_RESPONSE = {"instance_ids": ["c4bf5e16-1063-11f1-9096-5f6ae8f8983f"]}

INSTANCE_ACTIVE_RESPONSE = {
    "instances": [
        {
            "id": "c4bf5e16-1063-11f1-9096-5f6ae8f8983f",
            "status": "Active",
            "node_id": "2cb28df8-86fe-11f0-b32f-3b150b0bc471",
            "node_mode": "VirtualMachine",
            "host_address": "211.21.50.85",
            "resource_info": {
                "provider_name": "KonstTech",
                "instance_type": "rtx49-7c-kn.1",
                "cost_per_hour": 38.75,
            },
            "virtual_machines": [
                {
                    "vmid": 12,
                    "name": "noble-server-cloudimg-amd64-1771815611",
                    "login_info": {
                        "UsernameAndPassword": {
                            "username": "riftuser",
                            "password": "9W93nSnhWvPqUPwx",
                        }
                    },
                    "ready": True,
                    "state": "Running",
                }
            ],
            "port_mappings": [
                [22, 57011],
                [80, 57010],
                [443, 57008],
                [8080, 57001],
                [8443, 57000],
            ],
        }
    ]
}

TERMINATE_RESPONSE = {
    "terminated": [
        {
            "id": "c4bf5e16-1063-11f1-9096-5f6ae8f8983f",
            "status": "Active",
            "host_address": "211.21.50.85",
        }
    ]
}


# ── _api_request ──────────────────────────────────────────────────


async def test_api_request_sends_correct_payload():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"version": "2025-02-10", "data": {"ok": True}}
    mock_resp.raise_for_status = MagicMock()

    mock_client_instance = AsyncMock()
    mock_client_instance.request.return_value = mock_resp
    mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
    mock_client_instance.__aexit__ = AsyncMock(return_value=False)

    with patch("deplodock.provisioning.cloudrift.httpx.AsyncClient", return_value=mock_client_instance):
        result = await _api_request("POST", "/api/v1/test", {"foo": "bar"}, API_KEY, API_URL)

    mock_client_instance.request.assert_called_once_with(
        "POST",
        f"{API_URL}/api/v1/test",
        json={"version": API_VERSION, "data": {"foo": "bar"}},
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
        timeout=60,
    )
    assert result == {"ok": True}


async def test_api_request_dry_run(caplog):
    with caplog.at_level("INFO", logger="deplodock.provisioning.cloudrift"):
        result = await _api_request("POST", "/api/v1/test", {"foo": "bar"}, API_KEY, API_URL, dry_run=True)

    assert result is None
    assert "[dry-run] POST" in caplog.text
    assert "/api/v1/test" in caplog.text
    assert '"foo": "bar"' in caplog.text


# ── _rent_instance ────────────────────────────────────────────────


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_rent_instance_payload(mock_api):
    mock_api.return_value = RENT_RESPONSE

    result = await _rent_instance(
        API_KEY,
        "rtx49-7c-kn.1",
        ["ssh-ed25519 AAAA user@host"],
        DEFAULT_IMAGE_URL_NVIDIA,
        ports=[22, 8000],
        api_url=API_URL,
    )

    call_data = mock_api.call_args[0][2]
    assert call_data["selector"] == {"ByInstanceTypeAndLocation": {"instance_type": "rtx49-7c-kn.1"}}
    assert call_data["config"] == {
        "VirtualMachine": {
            "ssh_key": {"PublicKeys": ["ssh-ed25519 AAAA user@host"]},
            "image_url": DEFAULT_IMAGE_URL_NVIDIA,
            "cloudinit_url": DEFAULT_CLOUDINIT_URL,
            "ports": ["22", "8000"],
        }
    }
    assert call_data["with_public_ip"] is True
    assert result["instance_ids"] == ["c4bf5e16-1063-11f1-9096-5f6ae8f8983f"]


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_rent_instance_no_ports(mock_api):
    mock_api.return_value = RENT_RESPONSE

    await _rent_instance(API_KEY, "rtx49-7c-kn.1", ["ssh-ed25519 AAAA user@host"], DEFAULT_IMAGE_URL_NVIDIA, api_url=API_URL)

    call_data = mock_api.call_args[0][2]
    assert "ports" not in call_data
    assert call_data["with_public_ip"] is True


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_rent_instance_billing_exempt(mock_api):
    mock_api.return_value = RENT_RESPONSE

    await _rent_instance(
        API_KEY,
        "rtx49-7c-kn.1",
        ["ssh-ed25519 AAAA user@host"],
        DEFAULT_IMAGE_URL_NVIDIA,
        api_url=API_URL,
        billing_exempt=True,
    )

    call_data = mock_api.call_args[0][2]
    assert call_data["billing_exempt"] is True


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_rent_instance_no_billing_exempt_by_default(mock_api):
    mock_api.return_value = RENT_RESPONSE

    await _rent_instance(API_KEY, "rtx49-7c-kn.1", ["ssh-ed25519 AAAA user@host"], DEFAULT_IMAGE_URL_NVIDIA, api_url=API_URL)

    call_data = mock_api.call_args[0][2]
    assert "billing_exempt" not in call_data


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_rent_instance_network(mock_api):
    mock_api.return_value = RENT_RESPONSE

    await _rent_instance(
        API_KEY,
        "rtx49-7c-kn.1",
        ["ssh-ed25519 AAAA user@host"],
        DEFAULT_IMAGE_URL_NVIDIA,
        api_url=API_URL,
        network="public",
    )

    call_data = mock_api.call_args[0][2]
    assert call_data["network"] == "public"


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_rent_instance_no_network_by_default(mock_api):
    mock_api.return_value = RENT_RESPONSE

    await _rent_instance(API_KEY, "rtx49-7c-kn.1", ["ssh-ed25519 AAAA user@host"], DEFAULT_IMAGE_URL_NVIDIA, api_url=API_URL)

    call_data = mock_api.call_args[0][2]
    assert "network" not in call_data


# ── _terminate_instance ───────────────────────────────────────────


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_terminate_instance_payload(mock_api):
    mock_api.return_value = TERMINATE_RESPONSE

    await _terminate_instance(API_KEY, "inst-123", api_url=API_URL)

    mock_api.assert_called_once_with(
        "POST",
        "/api/v1/instances/terminate",
        {"selector": {"ById": ["inst-123"]}},
        API_KEY,
        API_URL,
        False,
    )


# ── _get_instance_info ────────────────────────────────────────────


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_get_instance_info_found(mock_api):
    mock_api.return_value = INSTANCE_ACTIVE_RESPONSE

    info = await _get_instance_info(API_KEY, "c4bf5e16-1063-11f1-9096-5f6ae8f8983f", api_url=API_URL)

    assert info["id"] == "c4bf5e16-1063-11f1-9096-5f6ae8f8983f"
    assert info["status"] == "Active"
    call_data = mock_api.call_args[0][2]
    assert call_data == {
        "selector": {"ById": ["c4bf5e16-1063-11f1-9096-5f6ae8f8983f"]},
        "mask": {"with_connection_info": True},
    }


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_get_instance_info_not_found(mock_api):
    mock_api.return_value = {"instances": []}

    info = await _get_instance_info(API_KEY, "inst-999", api_url=API_URL)
    assert info is None


# ── _list_ssh_keys / _add_ssh_key ─────────────────────────────────


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_list_ssh_keys(mock_api):
    mock_api.return_value = SSH_KEYS_LIST_RESPONSE

    result = await _list_ssh_keys(API_KEY, api_url=API_URL)

    mock_api.assert_called_once_with("POST", "/api/v1/ssh-keys/list", {}, API_KEY, API_URL, False)
    assert len(result["keys"]) == 2


@patch("deplodock.provisioning.cloudrift._api_request", new_callable=AsyncMock)
async def test_add_ssh_key(mock_api):
    mock_api.return_value = {"ssh_key": {"id": "key-new"}}

    result = await _add_ssh_key(API_KEY, "my-key", "ssh-ed25519 AAAA", api_url=API_URL)

    mock_api.assert_called_once_with(
        "POST",
        "/api/v1/ssh-keys/add",
        {"name": "my-key", "public_key": "ssh-ed25519 AAAA"},
        API_KEY,
        API_URL,
        False,
    )
    assert result["ssh_key"]["id"] == "key-new"


# ── _ensure_ssh_key ───────────────────────────────────────────────


@patch("deplodock.provisioning.cloudrift._add_ssh_key", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._list_ssh_keys", new_callable=AsyncMock)
async def test_ensure_ssh_key_already_registered(mock_list, mock_add, tmp_path):
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID37 user@example.com\n")

    mock_list.return_value = SSH_KEYS_LIST_RESPONSE

    key_id = await _ensure_ssh_key(API_KEY, str(key_file), api_url=API_URL)

    assert key_id == "5eb89e22-2a98-11f0-9c02-771633991431"
    mock_add.assert_not_called()


@patch("deplodock.provisioning.cloudrift._add_ssh_key", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._list_ssh_keys", new_callable=AsyncMock)
async def test_ensure_ssh_key_registers_new(mock_list, mock_add, tmp_path):
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 BBBB test@host\n")

    mock_list.return_value = SSH_KEYS_LIST_RESPONSE
    mock_add.return_value = {"ssh_key": {"id": "key-new"}}

    key_id = await _ensure_ssh_key(API_KEY, str(key_file), api_url=API_URL)

    assert key_id == "key-new"
    mock_add.assert_called_once()


# ── _log_connection_info ──────────────────────────────────────────


def test_log_connection_info_with_port_mappings(caplog):
    """Test with real response: shared IP + port mappings + VM credentials."""
    instance = INSTANCE_ACTIVE_RESPONSE["instances"][0]
    with caplog.at_level("INFO", logger="deplodock.provisioning.cloudrift"):
        _log_connection_info(instance)
    assert "211.21.50.85" in caplog.text
    assert "riftuser" in caplog.text
    assert "9W93nSnhWvPqUPwx" not in caplog.text  # password must not be logged
    assert "ssh -p 57011 riftuser@211.21.50.85" in caplog.text
    assert "Port 22 -> 211.21.50.85:57011" in caplog.text
    assert "Port 8080 -> 211.21.50.85:57001" in caplog.text


def test_log_connection_info_no_port_mappings(caplog):
    """Test with direct host address, no port mappings."""
    instance = {
        "host_address": "1.2.3.4",
        "port_mappings": [],
        "virtual_machines": [
            {
                "login_info": {
                    "UsernameAndPassword": {
                        "username": "riftuser",
                        "password": "abc123",
                    }
                }
            }
        ],
    }
    with caplog.at_level("INFO", logger="deplodock.provisioning.cloudrift"):
        _log_connection_info(instance)
    assert "ssh riftuser@1.2.3.4" in caplog.text
    assert "abc123" not in caplog.text  # password must not be logged


# ── DEFAULT_API_URL env var ──────────────────────────────────────


def test_default_api_url_env_var_override(monkeypatch):
    """DEFAULT_API_URL honors CLOUDRIFT_API_URL env var at import time."""
    import importlib

    from deplodock.provisioning import cloudrift

    monkeypatch.setenv("CLOUDRIFT_API_URL", "https://api.staging.cloudrift.ai")
    try:
        importlib.reload(cloudrift)
        assert cloudrift.DEFAULT_API_URL == "https://api.staging.cloudrift.ai"
    finally:
        monkeypatch.delenv("CLOUDRIFT_API_URL", raising=False)
        importlib.reload(cloudrift)


# ── select_image_url ─────────────────────────────────────────────


def test_select_image_url_amd_mi350x():
    assert select_image_url("mi350x-15-250-1000-gv.1") == DEFAULT_IMAGE_URL_AMD


def test_select_image_url_amd_mi300x():
    assert select_image_url("mi300x-8-100-500-foo.4") == DEFAULT_IMAGE_URL_AMD


def test_select_image_url_nvidia_rtx5090():
    assert select_image_url("rtx59-7-50-400-ec.1") == DEFAULT_IMAGE_URL_NVIDIA


def test_select_image_url_nvidia_h200():
    assert select_image_url("h200-24-200-1000-generic.8") == DEFAULT_IMAGE_URL_NVIDIA


def test_select_image_url_nvidia_v100_uses_proprietary_driver():
    assert select_image_url("v100-5-85-800-generic.16") == DEFAULT_IMAGE_URL_NVIDIA_PROPRIETARY


def test_select_image_url_nvidia_p100_uses_proprietary_driver():
    assert select_image_url("p100-4-60-500-foo.1") == DEFAULT_IMAGE_URL_NVIDIA_PROPRIETARY


# ── _instance_fully_ready ────────────────────────────────────────


def test_instance_fully_ready_happy_path():
    info = INSTANCE_ACTIVE_RESPONSE["instances"][0]
    assert _instance_fully_ready(info) is True


def test_instance_fully_ready_missing_host_address():
    info = {"status": "Active", "host_address": None, "port_mappings": [[22, 22222]]}
    assert _instance_fully_ready(info) is False


def test_instance_fully_ready_null_port_mappings():
    info = {"status": "Active", "host_address": "1.2.3.4", "port_mappings": None}
    assert _instance_fully_ready(info) is False


def test_instance_fully_ready_vm_not_ready_yet():
    info = {
        "status": "Active",
        "host_address": "1.2.3.4",
        "port_mappings": [[22, 22222]],
        "virtual_machines": [{"ready": False}],
    }
    assert _instance_fully_ready(info) is False


def test_instance_fully_ready_bare_metal_no_vms():
    """Bare-metal instances have no virtual_machines list; networking alone suffices."""
    info = {"status": "Active", "host_address": "1.2.3.4", "port_mappings": [[22, 22222]]}
    assert _instance_fully_ready(info) is True


def test_instance_fully_ready_direct_host_empty_port_mappings():
    """Some providers expose ports directly on the host IP (no NAT). port_mappings=[] is ready."""
    info = {
        "status": "Active",
        "host_address": "37.206.67.141",
        "port_mappings": [],
        "virtual_machines": [{"ready": True, "state": "Running"}],
    }
    assert _instance_fully_ready(info) is True


def test_default_api_url_fallback(monkeypatch):
    """DEFAULT_API_URL falls back to https://api.cloudrift.ai when env var unset."""
    import importlib

    from deplodock.provisioning import cloudrift

    monkeypatch.delenv("CLOUDRIFT_API_URL", raising=False)
    importlib.reload(cloudrift)
    assert cloudrift.DEFAULT_API_URL == "https://api.cloudrift.ai"


# ── wait_for_status ───────────────────────────────────────────────


def _active_response(ready=True, host="1.2.3.4", ports=None):
    """Build a minimal instance dict for wait_for_status tests."""
    if ports is None:
        ports = [[22, 22222]]
    return {
        "id": "inst-123",
        "status": "Active",
        "host_address": host,
        "port_mappings": ports,
        "virtual_machines": [{"ready": ready}],
    }


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_polls_until_ready(mock_get, mock_sleep):
    """wait_for_status must keep polling while status is Active but networking is missing."""
    mock_get.side_effect = [
        {"id": "inst-123", "status": "Active", "host_address": None, "port_mappings": None},
        _active_response(ready=False),
        _active_response(ready=True),
    ]
    info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=120, interval=10)
    assert info is not None
    assert info["host_address"] == "1.2.3.4"
    assert mock_get.await_count == 3


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_timeout_logs_readiness_components(mock_get, mock_sleep, caplog):
    """At timeout, the log message must identify the readiness components that blocked us."""
    mock_get.return_value = {
        "id": "inst-123",
        "status": "Active",
        "host_address": None,
        "port_mappings": None,
        "virtual_machines": [{"ready": False}],
    }
    with caplog.at_level("ERROR", logger="deplodock.provisioning.cloudrift"):
        info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=20, interval=10)
    assert info is None
    assert "never became ready" in caplog.text
    assert "host_address=None" in caplog.text
    assert "port_mappings=None" in caplog.text
    assert "vm_ready=False" in caplog.text


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_returns_none_on_fail_status(mock_get, mock_sleep):
    """fail_statuses must short-circuit polling."""
    mock_get.return_value = {"id": "inst-123", "status": "Inactive"}
    info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=120, fail_statuses={"Inactive"})
    assert info is None


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_returns_none_on_failed_status(mock_get, mock_sleep, caplog):
    """The v059 'Failed' status short-circuits without the caller listing it, and logs the reason."""
    mock_get.return_value = {
        "id": "inst-123",
        "status": "Failed",
        "failure": {"cause": "DockerImagePullFailed", "user_message": "image pull failed: unauthorized"},
    }
    with caplog.at_level("ERROR"):
        info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=120)
    assert info is None
    assert "image pull failed: unauthorized" in caplog.text


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_target_wins_over_fail_for_same_string(mock_get, mock_sleep):
    """If a status appears in both target_status and fail_statuses, success takes precedence."""
    mock_get.return_value = _active_response(ready=True)
    info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=120, fail_statuses={"Active"})
    assert info is not None
    assert info["host_address"] == "1.2.3.4"


# ── create_instance orphan termination ───────────────────────────


@patch("deplodock.provisioning.cloudrift._terminate_instance", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift.wait_for_status", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_terminates_orphan_on_timeout(mock_rent, mock_wait, mock_terminate, tmp_path):
    """When wait_for_status fails, the rented instance must be terminated and CapacityExhausted raised."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.return_value = {"instance_ids": ["inst-orphan"]}
    mock_wait.return_value = None

    with pytest.raises(CapacityExhausted):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)

    mock_terminate.assert_awaited_once()
    args = mock_terminate.await_args
    assert args.args[1] == "inst-orphan"


@patch("deplodock.provisioning.cloudrift._terminate_instance", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift.wait_for_status", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_swallows_termination_errors(mock_rent, mock_wait, mock_terminate, tmp_path, caplog):
    """A failed terminate during orphan cleanup must not mask the original CapacityExhausted."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.return_value = {"instance_ids": ["inst-orphan"]}
    mock_wait.return_value = None
    mock_terminate.side_effect = RuntimeError("network down")

    with caplog.at_level("ERROR", logger="deplodock.provisioning.cloudrift"):
        with pytest.raises(CapacityExhausted):
            await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)

    assert "Failed to terminate orphaned instance inst-orphan" in caplog.text


@patch("deplodock.provisioning.cloudrift._terminate_instance", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift.wait_for_status", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_terminates_orphan_on_exception(mock_rent, mock_wait, mock_terminate, tmp_path):
    """When wait_for_status raises, the rented instance must be terminated and the exception re-raised."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.return_value = {"instance_ids": ["inst-orphan"]}
    mock_wait.side_effect = RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)
    mock_terminate.assert_awaited_once()
    assert mock_terminate.await_args.args[1] == "inst-orphan"


# ── wait_for_status transient-error handling ─────────────────────


def _http_status_error(code, body=""):
    """Construct an httpx.HTTPStatusError with a given status code and optional body."""
    request = httpx.Request("POST", "https://api.test/instances/list")
    response = httpx.Response(code, request=request, text=body)
    return httpx.HTTPStatusError(f"HTTP {code}", request=request, response=response)


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_retries_transient_5xx(mock_get, mock_sleep):
    """A transient 5xx during polling must be retried, not propagated."""
    mock_get.side_effect = [
        _http_status_error(502),
        _http_status_error(503),
        _active_response(ready=True),
    ]
    info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=120, interval=10)
    assert info is not None
    assert mock_get.await_count == 3


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_propagates_4xx(mock_get, mock_sleep):
    """A 4xx (auth, bad request) is terminal and must propagate."""
    import pytest

    mock_get.side_effect = _http_status_error(401)
    with pytest.raises(httpx.HTTPStatusError):
        await wait_for_status(API_KEY, "inst-123", "Active", timeout=120, interval=10)


@patch("deplodock.provisioning.cloudrift.asyncio.sleep", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._get_instance_info", new_callable=AsyncMock)
async def test_wait_for_status_retries_network_error(mock_get, mock_sleep):
    """httpx network errors (timeout, connection refused, etc.) must be retried."""
    mock_get.side_effect = [
        httpx.ConnectError("connection refused"),
        _active_response(ready=True),
    ]
    info = await wait_for_status(API_KEY, "inst-123", "Active", timeout=120, interval=10)
    assert info is not None


@patch("deplodock.provisioning.cloudrift._terminate_instance", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift.wait_for_status", new_callable=AsyncMock)
@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_appends_extra_public_keys(mock_rent, mock_wait, mock_terminate, tmp_path):
    """extra_public_keys are installed alongside the key from ssh_key_path."""
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA own@host\n")

    mock_rent.return_value = {"instance_ids": ["inst-1"]}
    mock_wait.return_value = {"instance_id": "inst-1"}

    await create_instance(
        API_KEY,
        "rtx49-7c-kn.1",
        str(key_file),
        api_url=API_URL,
        extra_public_keys=["ssh-ed25519 BBBB bob@host"],
    )

    # _rent_instance(api_key, instance_type, ssh_public_keys, ...)
    assert mock_rent.await_args.args[2] == ["ssh-ed25519 AAAA own@host", "ssh-ed25519 BBBB bob@host"]


# ── create_instance HTTP-code classification ────────────────────


@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_503_raises_capacity_exhausted(mock_rent, tmp_path):
    """HTTP 503 on rent must be classified as CapacityExhausted for orchestrator fallback."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.side_effect = _http_status_error(503)

    with pytest.raises(CapacityExhausted):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)


@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_429_raises_capacity_exhausted(mock_rent, tmp_path):
    """HTTP 429 (rate limit) is also capacity-class."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.side_effect = _http_status_error(429)

    with pytest.raises(CapacityExhausted):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)


@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_400_instance_not_found_raises_capacity(mock_rent, tmp_path):
    """400 'Instance X not found' is a per-datacenter availability signal; advance candidates."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.side_effect = _http_status_error(400, body="Instance h200-8-generic.1 not found")

    with pytest.raises(CapacityExhausted):
        await create_instance(API_KEY, "h200-8-generic.1", str(key_file), api_url=API_URL)


@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_400_other_raises_terminal(mock_rent, tmp_path):
    """A 400 whose body is not a not-found signal must stay terminal (e.g. malformed body)."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.side_effect = _http_status_error(400, body="malformed request: missing field 'selector'")

    with pytest.raises(TerminalProvisionError):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)


@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_401_raises_terminal(mock_rent, tmp_path):
    """HTTP 401/403 must surface as TerminalProvisionError so the orchestrator aborts."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.side_effect = _http_status_error(401)

    with pytest.raises(TerminalProvisionError):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)


@patch("deplodock.provisioning.cloudrift._rent_instance", new_callable=AsyncMock)
async def test_create_instance_empty_instance_ids_raises_capacity(mock_rent, tmp_path):
    """Rent succeeding HTTP-wise but returning no instance is still no-capacity."""
    import pytest

    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    mock_rent.return_value = {"instance_ids": []}

    with pytest.raises(CapacityExhausted):
        await create_instance(API_KEY, "rtx49-7c-kn.1", str(key_file), api_url=API_URL)
