"""SSH deploy target: runs commands via SSH + SCP."""

import os
import subprocess
import sys
import tempfile

from deplodock.commands.deploy import load_recipe, run_deploy, run_teardown


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
            return 0, ""

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
            return result.returncode, stdout
        except Exception as e:
            print(f"Error running SSH command: {e}", file=sys.stderr)
            return 1, ""

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


def provision_remote(server, ssh_key, ssh_port, dry_run=False):
    """Ensure the remote server is ready for deployment.

    Steps (each checks before installing):
    1. Create ~/deploy directory
    2. Install Docker if not found
    3. Install NVIDIA Container Toolkit if not found
    4. Add user to docker group
    """
    def _run_ssh(command, capture=False):
        args = ssh_base_args(server, ssh_key, ssh_port)
        args.append(command)
        result = subprocess.run(args, capture_output=True, text=True)
        if capture:
            return result.returncode, result.stdout.strip()
        return result.returncode, ""

    # 1. Create deploy directory
    if dry_run:
        print(f"[dry-run] ssh {server}: mkdir -p {REMOTE_DEPLOY_DIR}")
    else:
        _run_ssh(f"mkdir -p {REMOTE_DEPLOY_DIR}")

    # 2. Install Docker if not found
    if dry_run:
        print(f"[dry-run] ssh {server}: install docker (if not present)")
    else:
        rc, _ = _run_ssh("command -v docker", capture=True)
        if rc != 0:
            print("Installing Docker...")
            _run_ssh("curl -fsSL https://get.docker.com | sudo sh")

    # 3. Install NVIDIA Container Toolkit if not found
    if dry_run:
        print(f"[dry-run] ssh {server}: install nvidia-container-toolkit (if not present)")
    else:
        rc, _ = _run_ssh("command -v nvidia-ctk", capture=True)
        if rc != 0:
            print("Installing NVIDIA Container Toolkit...")
            _run_ssh(
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey"
                " | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
                " && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
                " | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g'"
                " | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
                " && sudo apt-get update"
                " && sudo apt-get install -y nvidia-container-toolkit"
                " && sudo nvidia-ctk runtime configure --runtime=docker"
                " && sudo systemctl restart docker"
            )

    # 4. Add user to docker group
    if dry_run:
        print(f"[dry-run] ssh {server}: add user to docker group")
    else:
        _run_ssh("groups | grep -q docker || sudo usermod -aG docker $(whoami)")


def handle_ssh(args):
    """Handle the SSH deploy target."""
    recipe_dir = args.recipe
    variant = args.variant
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    model_dir = args.model_dir
    dry_run = args.dry_run
    teardown = args.teardown
    server = args.server
    ssh_key = args.ssh_key
    ssh_port = args.ssh_port

    run_cmd = make_run_cmd(server, ssh_key, ssh_port, dry_run=dry_run)
    write_file = make_write_file(server, ssh_key, ssh_port, dry_run=dry_run)

    provision_remote(server, ssh_key, ssh_port, dry_run=dry_run)

    if teardown:
        return run_teardown(run_cmd)

    config = load_recipe(recipe_dir, variant=variant)

    # Extract host from server address (user@host -> host)
    host = server.split("@")[-1] if "@" in server else server

    success = run_deploy(
        run_cmd=run_cmd,
        write_file=write_file,
        config=config,
        model_dir=model_dir,
        hf_token=hf_token,
        host=host,
        dry_run=dry_run,
    )

    if not success:
        sys.exit(1)


def register_ssh_target(subparsers):
    """Register the SSH deploy target."""
    parser = subparsers.add_parser("ssh", help="Deploy via SSH to a remote server")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--variant", default=None, help="Hardware variant (e.g. 8xH200)")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/mnt/models", help="Model cache directory")
    parser.add_argument("--teardown", action="store_true", help="Stop containers instead of deploying")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--server", required=True, help="SSH address (user@host)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH port")
    parser.set_defaults(func=handle_ssh)
