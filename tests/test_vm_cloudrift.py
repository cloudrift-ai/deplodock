"""Unit tests for CloudRift API helper functions.

Response fixtures are captured from real CloudRift API calls.
"""

import json
from unittest.mock import patch, MagicMock

from deplodock.commands.vm.cloudrift import (
    _api_request,
    _rent_instance,
    _terminate_instance,
    _get_instance_info,
    _list_ssh_keys,
    _add_ssh_key,
    _ensure_ssh_key,
    _print_connection_info,
    API_VERSION,
    DEFAULT_IMAGE_URL,
    DEFAULT_CLOUDINIT_URL,
)


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

RENT_RESPONSE = {
    "instance_ids": ["c4bf5e16-1063-11f1-9096-5f6ae8f8983f"]
}

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


@patch("deplodock.commands.vm.cloudrift.requests.request")
def test_api_request_sends_correct_payload(mock_req):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"version": "2025-02-10", "data": {"ok": True}}
    mock_resp.raise_for_status = MagicMock()
    mock_req.return_value = mock_resp

    result = _api_request("POST", "/api/v1/test", {"foo": "bar"}, API_KEY, API_URL)

    mock_req.assert_called_once_with(
        "POST",
        f"{API_URL}/api/v1/test",
        json={"version": API_VERSION, "data": {"foo": "bar"}},
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    )
    assert result == {"ok": True}


def test_api_request_dry_run(capsys):
    result = _api_request("POST", "/api/v1/test", {"foo": "bar"}, API_KEY, API_URL, dry_run=True)

    assert result is None
    output = capsys.readouterr().out
    assert "[dry-run] POST" in output
    assert "/api/v1/test" in output
    assert '"foo": "bar"' in output


# ── _rent_instance ────────────────────────────────────────────────


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_rent_instance_payload(mock_api):
    mock_api.return_value = RENT_RESPONSE

    result = _rent_instance(API_KEY, "rtx49-7c-kn.1", ["ssh-ed25519 AAAA user@host"],
                            ports=[22, 8000], api_url=API_URL)

    call_data = mock_api.call_args[0][2]
    assert call_data["selector"] == {"ByInstanceTypeAndLocation": {"instance_type": "rtx49-7c-kn.1"}}
    assert call_data["config"] == {
        "VirtualMachine": {
            "ssh_key": {"PublicKeys": ["ssh-ed25519 AAAA user@host"]},
            "image_url": DEFAULT_IMAGE_URL,
            "cloudinit_url": DEFAULT_CLOUDINIT_URL,
        }
    }
    assert call_data["with_public_ip"] is True
    assert call_data["ports"] == ["22", "8000"]
    assert result["instance_ids"] == ["c4bf5e16-1063-11f1-9096-5f6ae8f8983f"]


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_rent_instance_no_ports(mock_api):
    mock_api.return_value = RENT_RESPONSE

    _rent_instance(API_KEY, "rtx49-7c-kn.1", ["ssh-ed25519 AAAA user@host"], api_url=API_URL)

    call_data = mock_api.call_args[0][2]
    assert "ports" not in call_data
    assert call_data["with_public_ip"] is True


# ── _terminate_instance ───────────────────────────────────────────


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_terminate_instance_payload(mock_api):
    mock_api.return_value = TERMINATE_RESPONSE

    _terminate_instance(API_KEY, "inst-123", api_url=API_URL)

    mock_api.assert_called_once_with(
        "POST", "/api/v1/instances/terminate",
        {"selector": {"ById": ["inst-123"]}},
        API_KEY, API_URL, False,
    )


# ── _get_instance_info ────────────────────────────────────────────


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_get_instance_info_found(mock_api):
    mock_api.return_value = INSTANCE_ACTIVE_RESPONSE

    info = _get_instance_info(API_KEY, "c4bf5e16-1063-11f1-9096-5f6ae8f8983f", api_url=API_URL)

    assert info["id"] == "c4bf5e16-1063-11f1-9096-5f6ae8f8983f"
    assert info["status"] == "Active"
    call_data = mock_api.call_args[0][2]
    assert call_data == {"selector": {"ById": ["c4bf5e16-1063-11f1-9096-5f6ae8f8983f"]}}


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_get_instance_info_not_found(mock_api):
    mock_api.return_value = {"instances": []}

    info = _get_instance_info(API_KEY, "inst-999", api_url=API_URL)
    assert info is None


# ── _list_ssh_keys / _add_ssh_key ─────────────────────────────────


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_list_ssh_keys(mock_api):
    mock_api.return_value = SSH_KEYS_LIST_RESPONSE

    result = _list_ssh_keys(API_KEY, api_url=API_URL)

    mock_api.assert_called_once_with("POST", "/api/v1/ssh-keys/list", {}, API_KEY, API_URL, False)
    assert len(result["keys"]) == 2


@patch("deplodock.commands.vm.cloudrift._api_request")
def test_add_ssh_key(mock_api):
    mock_api.return_value = {"ssh_key": {"id": "key-new"}}

    result = _add_ssh_key(API_KEY, "my-key", "ssh-ed25519 AAAA", api_url=API_URL)

    mock_api.assert_called_once_with(
        "POST", "/api/v1/ssh-keys/add",
        {"name": "my-key", "public_key": "ssh-ed25519 AAAA"},
        API_KEY, API_URL, False,
    )
    assert result["ssh_key"]["id"] == "key-new"


# ── _ensure_ssh_key ───────────────────────────────────────────────


@patch("deplodock.commands.vm.cloudrift._add_ssh_key")
@patch("deplodock.commands.vm.cloudrift._list_ssh_keys")
def test_ensure_ssh_key_already_registered(mock_list, mock_add, tmp_path):
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAID37 user@example.com\n")

    mock_list.return_value = SSH_KEYS_LIST_RESPONSE

    key_id = _ensure_ssh_key(API_KEY, str(key_file), api_url=API_URL)

    assert key_id == "5eb89e22-2a98-11f0-9c02-771633991431"
    mock_add.assert_not_called()


@patch("deplodock.commands.vm.cloudrift._add_ssh_key")
@patch("deplodock.commands.vm.cloudrift._list_ssh_keys")
def test_ensure_ssh_key_registers_new(mock_list, mock_add, tmp_path):
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 BBBB test@host\n")

    mock_list.return_value = SSH_KEYS_LIST_RESPONSE
    mock_add.return_value = {"ssh_key": {"id": "key-new"}}

    key_id = _ensure_ssh_key(API_KEY, str(key_file), api_url=API_URL)

    assert key_id == "key-new"
    mock_add.assert_called_once()


# ── _print_connection_info ────────────────────────────────────────


def test_print_connection_info_with_port_mappings(capsys):
    """Test with real response: shared IP + port mappings + VM credentials."""
    instance = INSTANCE_ACTIVE_RESPONSE["instances"][0]
    _print_connection_info(instance)
    output = capsys.readouterr().out
    assert "211.21.50.85" in output
    assert "riftuser" in output
    assert "9W93nSnhWvPqUPwx" in output
    assert "ssh -p 57011 riftuser@211.21.50.85" in output
    assert "Port 22 -> 211.21.50.85:57011" in output
    assert "Port 8080 -> 211.21.50.85:57001" in output


def test_print_connection_info_no_port_mappings(capsys):
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
    _print_connection_info(instance)
    output = capsys.readouterr().out
    assert "ssh riftuser@1.2.3.4" in output
    assert "abc123" in output
