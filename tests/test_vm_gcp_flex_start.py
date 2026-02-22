"""Unit tests for GCP flex-start command builders."""

from deplodock.commands.vm.gcp_flex_start import (
    _gcloud_create_cmd,
    _gcloud_delete_cmd,
    _gcloud_status_cmd,
    _gcloud_external_ip_cmd,
    _gcloud_ssh_check_cmd,
)


# ── Command builder tests ─────────────────────────────────────────


def test_gcloud_create_cmd():
    cmd = _gcloud_create_cmd("my-vm", "us-central1-a", "a2-highgpu-1g")
    assert cmd == [
        "gcloud", "compute", "instances", "create", "my-vm",
        "--zone", "us-central1-a",
        "--machine-type", "a2-highgpu-1g",
        "--provisioning-model=FLEX_START",
        "--maintenance-policy=TERMINATE",
        "--reservation-affinity=none",
        "--max-run-duration", "7d",
        "--instance-termination-action=DELETE",
        "--image-family", "debian-12",
        "--image-project", "debian-cloud",
        "--request-valid-for-duration", "2h",
    ]


def test_gcloud_create_cmd_with_gcloud_args():
    cmd = _gcloud_create_cmd(
        "my-vm", "us-central1-a", "e2-micro",
        extra_gcloud_args="--no-service-account --no-scopes",
    )
    assert cmd[-2:] == ["--no-service-account", "--no-scopes"]


def test_gcloud_create_cmd_custom_options():
    cmd = _gcloud_create_cmd(
        "my-vm", "us-central1-a", "e2-micro",
        max_run_duration="1h",
        termination_action="STOP",
        image_family="ubuntu-2204-lts",
        image_project="ubuntu-os-cloud",
    )
    assert "--max-run-duration" in cmd
    idx = cmd.index("--max-run-duration")
    assert cmd[idx + 1] == "1h"
    assert "--instance-termination-action=STOP" in cmd
    idx = cmd.index("--image-family")
    assert cmd[idx + 1] == "ubuntu-2204-lts"
    idx = cmd.index("--image-project")
    assert cmd[idx + 1] == "ubuntu-os-cloud"


def test_gcloud_delete_cmd():
    cmd = _gcloud_delete_cmd("my-vm", "us-central1-a")
    assert cmd == ["gcloud", "compute", "instances", "delete", "my-vm", "--zone", "us-central1-a", "--quiet"]


def test_gcloud_status_cmd():
    cmd = _gcloud_status_cmd("my-vm", "us-central1-a")
    assert cmd == [
        "gcloud", "compute", "instances", "describe", "my-vm",
        "--zone", "us-central1-a", "--format", "value(status)",
    ]


def test_gcloud_external_ip_cmd():
    cmd = _gcloud_external_ip_cmd("my-vm", "us-central1-a")
    assert cmd == [
        "gcloud", "compute", "instances", "describe", "my-vm",
        "--zone", "us-central1-a",
        "--format", "value(networkInterfaces[0].accessConfigs[0].natIP)",
    ]


def test_gcloud_ssh_check_cmd():
    cmd = _gcloud_ssh_check_cmd("my-vm", "us-central1-a")
    assert cmd == [
        "gcloud", "compute", "ssh", "my-vm",
        "--zone", "us-central1-a", "--command", "true",
        "--ssh-flag=-o", "--ssh-flag=ConnectTimeout=5",
        "--ssh-flag=-o", "--ssh-flag=StrictHostKeyChecking=no",
    ]


def test_gcloud_ssh_check_cmd_with_gateway():
    cmd = _gcloud_ssh_check_cmd("my-vm", "us-central1-a", ssh_gateway="gcp-ssh-gateway")
    assert cmd == [
        "gcloud", "compute", "ssh", "my-vm",
        "--zone", "us-central1-a", "--command", "true",
        "--ssh-flag=-o", "--ssh-flag=ConnectTimeout=5",
        "--ssh-flag=-o", "--ssh-flag=StrictHostKeyChecking=no",
        "--ssh-flag=-o", "--ssh-flag=ProxyJump=gcp-ssh-gateway",
    ]
