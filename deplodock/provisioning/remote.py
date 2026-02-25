"""Remote server provisioning: install Docker, NVIDIA toolkit, etc."""

import asyncio
import logging

from deplodock.provisioning.ssh_transport import ssh_base_args

logger = logging.getLogger(__name__)


async def provision_remote(server, ssh_key, ssh_port, dry_run=False):
    """Ensure the remote server is ready for deployment.

    Steps (each checks before installing):
    1. Create ~/deploy directory
    2. Install Docker if not found
    3. Install NVIDIA Container Toolkit if not found
    4. Add user to docker group
    """
    REMOTE_DEPLOY_DIR = "~/deploy"

    async def _run_ssh(command, capture=False, timeout=600):
        args = ssh_base_args(server, ssh_key, ssh_port)
        args.append(command)
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode != 0 and stderr_bytes:
                logger.error(f"SSH error ({server}): {stderr_bytes.decode().strip()}")
            if capture:
                return proc.returncode, stdout_bytes.decode().strip() if stdout_bytes else ""
            return proc.returncode, ""
        except TimeoutError:
            logger.error(f"SSH command timed out after {timeout}s: {command}")
            proc.kill()
            await proc.wait()
            return 1, ""

    # 1. Create deploy directory
    if dry_run:
        logger.info(f"[dry-run] ssh {server}: mkdir -p {REMOTE_DEPLOY_DIR}")
    else:
        await _run_ssh(f"mkdir -p {REMOTE_DEPLOY_DIR}")

    # 2. Install Docker if not found
    if dry_run:
        logger.info(f"[dry-run] ssh {server}: install docker (if not present)")
    else:
        rc, _ = await _run_ssh("command -v docker", capture=True)
        if rc != 0:
            logger.info("Installing Docker...")
            await _run_ssh("curl -fsSL https://get.docker.com | sudo sh")

    # 3. Install NVIDIA Container Toolkit if not found
    if dry_run:
        logger.info(f"[dry-run] ssh {server}: install nvidia-container-toolkit (if not present)")
    else:
        rc, _ = await _run_ssh("command -v nvidia-ctk", capture=True)
        if rc != 0:
            logger.info("Installing NVIDIA Container Toolkit...")
            await _run_ssh(
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
        logger.info(f"[dry-run] ssh {server}: add user to docker group")
    else:
        await _run_ssh("groups | grep -q docker || sudo usermod -aG docker $(whoami)")
