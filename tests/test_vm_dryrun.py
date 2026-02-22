"""Dry-run end-to-end tests for the vm command."""


# ── Dry-run start/stop ─────────────────────────────────────────────


def test_vm_start_dry_run(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "start", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "gcloud compute instances start" in stdout
    assert "my-gpu-vm" in stdout
    assert "us-central1-a" in stdout


def test_vm_start_dry_run_with_wait_ssh(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "start", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--wait-ssh",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "gcloud compute instances start" in stdout
    assert "gcloud compute ssh" in stdout
    assert "Waiting for SSH connectivity" in stdout


def test_vm_stop_dry_run(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "stop", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "gcloud compute instances stop" in stdout
    assert "my-gpu-vm" in stdout


# ── Argparse validation ────────────────────────────────────────────


def test_vm_start_missing_instance(run_cli):
    rc, _, stderr = run_cli(
        "vm", "start", "gcp-flex-start",
        "--zone", "us-central1-a",
    )
    assert rc != 0
    assert "--instance" in stderr


def test_vm_start_missing_zone(run_cli):
    rc, _, stderr = run_cli(
        "vm", "start", "gcp-flex-start",
        "--instance", "my-gpu-vm",
    )
    assert rc != 0
    assert "--zone" in stderr


# ── CLI help ───────────────────────────────────────────────────────


def test_vm_help(run_cli):
    rc, stdout, _ = run_cli("vm", "--help")
    assert rc == 0
    assert "start" in stdout
    assert "stop" in stdout


def test_vm_start_help(run_cli):
    rc, stdout, _ = run_cli("vm", "start", "--help")
    assert rc == 0
    assert "gcp-flex-start" in stdout


def test_vm_start_gcp_flex_start_help(run_cli):
    rc, stdout, _ = run_cli("vm", "start", "gcp-flex-start", "--help")
    assert rc == 0
    assert "--instance" in stdout
    assert "--zone" in stdout
    assert "--timeout" in stdout
    assert "--dry-run" in stdout
    assert "--wait-ssh" in stdout
    assert "--wait-ssh-timeout" in stdout


def test_vm_stop_gcp_flex_start_help(run_cli):
    rc, stdout, _ = run_cli("vm", "stop", "gcp-flex-start", "--help")
    assert rc == 0
    assert "--instance" in stdout
    assert "--zone" in stdout
    assert "--timeout" in stdout
    assert "--dry-run" in stdout


def test_top_level_help_includes_vm(run_cli):
    rc, stdout, _ = run_cli("--help")
    assert rc == 0
    assert "vm" in stdout
