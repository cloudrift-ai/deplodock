"""Unit tests for GCP flex-start command builders."""

from deplodock.commands.vm.gcp_flex_start import (
    _gcloud_start_cmd,
    _gcloud_stop_cmd,
    _gcloud_status_cmd,
    _gcloud_external_ip_cmd,
    _gcloud_ssh_check_cmd,
)


# ── Command builder tests ─────────────────────────────────────────


def test_gcloud_start_cmd():
    cmd = _gcloud_start_cmd("my-vm", "us-central1-a")
    assert cmd == ["gcloud", "compute", "instances", "start", "my-vm", "--zone", "us-central1-a"]


def test_gcloud_stop_cmd():
    cmd = _gcloud_stop_cmd("my-vm", "us-central1-a")
    assert cmd == ["gcloud", "compute", "instances", "stop", "my-vm", "--zone", "us-central1-a"]


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
