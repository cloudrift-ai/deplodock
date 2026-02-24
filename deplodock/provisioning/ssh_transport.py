"""SSH transport: run commands and write files on remote servers via SSH/SCP."""

import os
import subprocess
import sys
import tempfile

REMOTE_DEPLOY_DIR = "~/deploy"


def ssh_base_args(server, ssh_key, ssh_port):
    """Build base SSH arguments."""
    args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes"]
    if ssh_key:
        args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        args += ["-p", str(ssh_port)]
    args.append(server)
    return args


def make_run_cmd(server, ssh_key, ssh_port, dry_run=False):
    """Create a run_cmd callable for SSH execution."""

    def run_cmd(command, stream=True):
        # Use sg to run docker commands under the docker group
        if command.strip().startswith("docker"):
            escaped = command.replace('"', '\\"')
            full_cmd = f'sg docker -c "cd {REMOTE_DEPLOY_DIR} && {escaped}"'
        else:
            full_cmd = f"cd {REMOTE_DEPLOY_DIR} && {command}"
        if dry_run:
            print(f"[dry-run] ssh {server}: {full_cmd}")
            return 0, "", ""

        ssh_args = ssh_base_args(server, ssh_key, ssh_port)
        ssh_args.append(full_cmd)

        try:
            result = subprocess.run(
                ssh_args,
                text=True,
                stdout=None if stream else subprocess.PIPE,
                stderr=None if stream else subprocess.PIPE,
            )
            stdout = "" if stream else (result.stdout or "")
            stderr = "" if stream else (result.stderr or "")
            return result.returncode, stdout, stderr
        except Exception as e:
            print(f"Error running SSH command: {e}", file=sys.stderr)
            return 1, "", ""

    return run_cmd


def scp_file(local_path, server, ssh_key, ssh_port, remote_path):
    """Copy a file to the remote server via SCP."""
    scp_args = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes"]
    if ssh_key:
        scp_args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        scp_args += ["-P", str(ssh_port)]
    scp_args += [local_path, f"{server}:{remote_path}"]

    result = subprocess.run(scp_args, capture_output=True, text=True)
    return result.returncode


def make_write_file(server, ssh_key, ssh_port, dry_run=False):
    """Create a write_file callable that SCPs files to the remote server."""

    def write_file(path, content):
        remote_path = f"{REMOTE_DEPLOY_DIR}/{path}"
        if dry_run:
            print(f"[dry-run] scp {path} -> {server}:{remote_path}")
            return

        # Write to a temp file locally, then SCP
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{path}", delete=False) as f:
            f.write(content)
            tmp_path = f.name

        try:
            rc = scp_file(tmp_path, server, ssh_key, ssh_port, remote_path)
            if rc != 0:
                print(f"Failed to SCP {path} to {server}:{remote_path}", file=sys.stderr)
        finally:
            os.unlink(tmp_path)

    return write_file
