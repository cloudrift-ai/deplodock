"""Dry-run end-to-end tests for the vm command."""


# ── Dry-run create/delete ─────────────────────────────────────────


def test_vm_create_dry_run(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "create", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--machine-type", "a2-highgpu-1g",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "gcloud compute instances create" in stdout
    assert "my-gpu-vm" in stdout
    assert "us-central1-a" in stdout
    assert "--provisioning-model=FLEX_START" in stdout
    assert "--machine-type" in stdout


def test_vm_create_dry_run_with_wait_ssh(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "create", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--machine-type", "e2-micro",
        "--wait-ssh",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "gcloud compute instances create" in stdout
    assert "gcloud compute ssh" in stdout
    assert "Waiting for SSH connectivity" in stdout


def test_vm_create_dry_run_with_gcloud_args(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "create", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--machine-type", "e2-micro",
        "--gcloud-args", "--no-service-account --no-scopes",
        "--dry-run",
    )
    assert rc == 0
    assert "--no-service-account" in stdout
    assert "--no-scopes" in stdout


def test_vm_create_dry_run_with_ssh_gateway(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "create", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--machine-type", "e2-micro",
        "--wait-ssh",
        "--ssh-gateway", "gcp-ssh-gateway",
        "--dry-run",
    )
    assert rc == 0
    assert "ProxyJump=gcp-ssh-gateway" in stdout


def test_vm_delete_dry_run(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "delete", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "gcloud compute instances delete" in stdout
    assert "--quiet" in stdout
    assert "my-gpu-vm" in stdout


# ── Argparse validation ────────────────────────────────────────────


def test_vm_create_missing_instance(run_cli):
    rc, _, stderr = run_cli(
        "vm", "create", "gcp-flex-start",
        "--zone", "us-central1-a",
        "--machine-type", "e2-micro",
    )
    assert rc != 0
    assert "--instance" in stderr


def test_vm_create_missing_zone(run_cli):
    rc, _, stderr = run_cli(
        "vm", "create", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--machine-type", "e2-micro",
    )
    assert rc != 0
    assert "--zone" in stderr


def test_vm_create_missing_machine_type(run_cli):
    rc, _, stderr = run_cli(
        "vm", "create", "gcp-flex-start",
        "--instance", "my-gpu-vm",
        "--zone", "us-central1-a",
    )
    assert rc != 0
    assert "--machine-type" in stderr


# ── CLI help ───────────────────────────────────────────────────────


def test_vm_help(run_cli):
    rc, stdout, _ = run_cli("vm", "--help")
    assert rc == 0
    assert "create" in stdout
    assert "delete" in stdout


def test_vm_create_help(run_cli):
    rc, stdout, _ = run_cli("vm", "create", "--help")
    assert rc == 0
    assert "gcp-flex-start" in stdout


def test_vm_create_gcp_flex_start_help(run_cli):
    rc, stdout, _ = run_cli("vm", "create", "gcp-flex-start", "--help")
    assert rc == 0
    assert "--instance" in stdout
    assert "--zone" in stdout
    assert "--machine-type" in stdout
    assert "--timeout" in stdout
    assert "--dry-run" in stdout
    assert "--wait-ssh" in stdout
    assert "--wait-ssh-timeout" in stdout
    assert "--max-run-duration" in stdout
    assert "--gcloud-args" in stdout
    assert "--ssh-gateway" in stdout


def test_vm_delete_gcp_flex_start_help(run_cli):
    rc, stdout, _ = run_cli("vm", "delete", "gcp-flex-start", "--help")
    assert rc == 0
    assert "--instance" in stdout
    assert "--zone" in stdout
    assert "--dry-run" in stdout


def test_top_level_help_includes_vm(run_cli):
    rc, stdout, _ = run_cli("--help")
    assert rc == 0
    assert "vm" in stdout


# ── CloudRift dry-run create/delete ───────────────────────────────


def test_vm_create_cloudrift_dry_run(run_cli, tmp_path):
    key_file = tmp_path / "id_ed25519.pub"
    key_file.write_text("ssh-ed25519 AAAA test@host\n")

    rc, stdout, _ = run_cli(
        "vm", "create", "cloudrift",
        "--instance-type", "rtx4090.1",
        "--ssh-key", str(key_file),
        "--api-key", "test-key",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "POST" in stdout
    assert "PublicKeys" in stdout
    assert "instances/rent" in stdout


def test_vm_delete_cloudrift_dry_run(run_cli):
    rc, stdout, _ = run_cli(
        "vm", "delete", "cloudrift",
        "--instance-id", "inst-123",
        "--api-key", "test-key",
        "--dry-run",
    )
    assert rc == 0
    assert "[dry-run]" in stdout
    assert "POST" in stdout
    assert "instances/terminate" in stdout
    assert "inst-123" in stdout


# ── CloudRift CLI help ────────────────────────────────────────────


def test_vm_create_cloudrift_help(run_cli):
    rc, stdout, _ = run_cli("vm", "create", "cloudrift", "--help")
    assert rc == 0
    assert "--instance-type" in stdout
    assert "--ssh-key" in stdout
    assert "--api-key" in stdout
    assert "--image-url" in stdout
    assert "--ports" in stdout
    assert "--timeout" in stdout
    assert "--dry-run" in stdout
    assert "--api-url" in stdout


def test_vm_delete_cloudrift_help(run_cli):
    rc, stdout, _ = run_cli("vm", "delete", "cloudrift", "--help")
    assert rc == 0
    assert "--instance-id" in stdout
    assert "--api-key" in stdout
    assert "--dry-run" in stdout
    assert "--api-url" in stdout


def test_vm_create_help_includes_cloudrift(run_cli):
    rc, stdout, _ = run_cli("vm", "create", "--help")
    assert rc == 0
    assert "cloudrift" in stdout
