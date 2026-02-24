"""Remote server provisioning: install Docker, NVIDIA toolkit, etc."""

import subprocess


def ssh_base_args(server, ssh_key, ssh_port):
    """Build base SSH arguments."""
    args = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "BatchMode=yes"]
    if ssh_key:
        args += ["-i", ssh_key]
    if ssh_port and ssh_port != 22:
        args += ["-p", str(ssh_port)]
    args.append(server)
    return args


def provision_remote(server, ssh_key, ssh_port, dry_run=False):
    """Ensure the remote server is ready for deployment.

    Steps (each checks before installing):
    1. Create ~/deploy directory
    2. Install Docker if not found
    3. Install NVIDIA Container Toolkit if not found
    4. Add user to docker group
    """
    REMOTE_DEPLOY_DIR = "~/deploy"

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
