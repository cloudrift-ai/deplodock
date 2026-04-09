"""Remote server provisioning: install Docker, NVIDIA driver/toolkit, etc."""

import asyncio
import logging

from deplodock.provisioning.host import Host

logger = logging.getLogger(__name__)

REMOTE_DEPLOY_DIR = "~/deploy"


async def provision_remote(
    host: Host,
    *,
    skip_nvidia: bool = False,
    driver_version: str | None = None,
    cuda_version: str | None = None,
):
    """Ensure ``host`` is ready for deployment.

    Steps (each checks before installing):
    1. Create ~/deploy directory
    2. Install Docker if not found
    3. Install NVIDIA driver / CUDA toolkit if requested versions don't match
       (reboots and waits for the host to come back if anything was installed)
    4. Install NVIDIA Container Toolkit if not found
    5. Add user to docker group
    """
    dry_run = bool(getattr(host, "dry_run", False))

    # 1. Create deploy directory
    await host.run(f"mkdir -p {REMOTE_DEPLOY_DIR}")

    # 2. Install Docker if not found
    if dry_run:
        logger.info(f"{host.name}: [dry-run] would install docker (if not present)")
    else:
        rc, _ = await host.run("command -v docker", capture=True)
        if rc != 0:
            logger.info(f"{host.name}: installing Docker...")
            await host.run("curl -fsSL https://get.docker.com | sh", sudo=True)

    # 3. NVIDIA driver / CUDA (skip for AMD/ROCm)
    if not skip_nvidia and (driver_version or cuda_version):
        installed_anything = await _ensure_nvidia_versions(host, driver_version=driver_version, cuda_version=cuda_version)
        if installed_anything:
            await reboot_and_wait(host)

    # 4. Install NVIDIA Container Toolkit if not found
    if not skip_nvidia and dry_run:
        logger.info(f"{host.name}: [dry-run] would install nvidia-container-toolkit (if not present)")
    elif not skip_nvidia:
        rc, _ = await host.run("command -v nvidia-ctk", capture=True)
        if rc != 0:
            logger.info(f"{host.name}: installing NVIDIA Container Toolkit...")
            await host.run(
                "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey"
                " | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
                " && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
                " | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g'"
                " > /etc/apt/sources.list.d/nvidia-container-toolkit.list"
                " && apt-get update"
                " && apt-get install -y nvidia-container-toolkit"
                " && nvidia-ctk runtime configure --runtime=docker"
                " && systemctl restart docker",
                sudo=True,
            )

    # 5. Add user to docker group
    await host.run("groups | grep -q docker || sudo usermod -aG docker $(whoami)")


async def _ensure_nvidia_versions(
    host: Host,
    *,
    driver_version: str | None,
    cuda_version: str | None,
) -> bool:
    """Install requested driver / CUDA toolkit if they don't already match.

    Returns True if anything was installed (caller should reboot).
    """
    installed = False

    need_driver = driver_version and not _matches(await _current_driver_version(host), driver_version)
    need_cuda = cuda_version and not await _cuda_installed(host, cuda_version)

    if driver_version and not need_driver:
        logger.info(f"{host.name}: NVIDIA driver already matches {driver_version}, skipping install")
    if cuda_version and not need_cuda:
        logger.info(f"{host.name}: CUDA toolkit {cuda_version} already installed, skipping")

    if need_driver or need_cuda:
        await _setup_nvidia_cuda_repo(host)

    if need_driver:
        # cuda-drivers-<major> is the canonical NVIDIA-published driver metapackage.
        # Strip non-numeric suffix and take the first dot component.
        major = driver_version.split(".")[0]
        logger.info(f"{host.name}: installing cuda-drivers-{major} (requested {driver_version})")
        await host.run(
            f"DEBIAN_FRONTEND=noninteractive apt-get install -y cuda-drivers-{major}",
            sudo=True,
            timeout=1800,
        )
        installed = True

    if need_cuda:
        pkg = "cuda-toolkit-" + cuda_version.replace(".", "-")
        logger.info(f"{host.name}: installing {pkg}")
        await host.run(
            f"DEBIAN_FRONTEND=noninteractive apt-get install -y {pkg}",
            sudo=True,
            timeout=1800,
        )
        installed = True

    return installed


async def _setup_nvidia_cuda_repo(host: Host) -> None:
    """Add NVIDIA's official CUDA apt repo (idempotent).

    NVIDIA publishes a per-Ubuntu-version repo at
    ``developer.download.nvidia.com/compute/cuda/repos/ubuntu<VER>/x86_64/``
    that carries the latest drivers and CUDA toolkit metapackages
    (``cuda-drivers-<branch>``, ``cuda-toolkit-X-Y``).
    """
    logger.info(f"{host.name}: setting up NVIDIA CUDA apt repo")
    # Detect Ubuntu version → e.g. "24.04" → "ubuntu2404"
    script = (
        "set -e; "
        ". /etc/os-release; "
        'distro="ubuntu$(echo "$VERSION_ID" | tr -d .)"; '
        'arch="$(dpkg --print-architecture)"; '
        'case "$arch" in amd64) cuda_arch=x86_64;; arm64) cuda_arch=sbsa;; *) echo "unsupported arch $arch" >&2; exit 1;; esac; '
        'base="https://developer.download.nvidia.com/compute/cuda/repos/${distro}/${cuda_arch}"; '
        "if [ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]; then "
        '  tmp="$(mktemp)"; '
        '  curl -fsSL "${base}/cuda-keyring_1.1-1_all.deb" -o "$tmp"; '
        '  dpkg -i "$tmp"; rm -f "$tmp"; '
        "fi; "
        "apt-get update"
    )
    await host.run(script, sudo=True, timeout=600)


async def _current_driver_version(host: Host) -> str | None:
    rc, out = await host.run(
        "nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1",
        capture=True,
    )
    if rc != 0 or not out:
        return None
    return out.strip()


async def _cuda_installed(host: Host, requested: str) -> bool:
    """Return True if CUDA toolkit ``requested`` (e.g. ``"13.2"``) is installed.

    The NVIDIA apt packages drop side-by-side trees at ``/usr/local/cuda-X.Y``,
    so a directory check is the most reliable signal — independent of whether
    ``/usr/local/cuda`` happens to point at this version or what's on PATH.
    """
    rc, _ = await host.run(f"test -d /usr/local/cuda-{requested}", capture=True)
    return rc == 0


def _matches(current: str | None, requested: str) -> bool:
    """Prefix-match on dot components: '550' matches '550.127.05', '12.4' matches '12.4.1'."""
    if not current:
        return False
    cur_parts = current.split(".")
    req_parts = requested.split(".")
    return cur_parts[: len(req_parts)] == req_parts


async def reboot_and_wait(host: Host, timeout: int = 600):
    """Reboot ``host`` and wait for it to come back."""
    logger.info(f"Rebooting {host.name} ...")
    # nohup so the SSH channel can close cleanly; ignore non-zero exit
    await host.run(
        "nohup shutdown -r now >/dev/null 2>&1 &",
        sudo=True,
        timeout=30,
    )
    if getattr(host, "dry_run", False):
        logger.info(f"[dry-run] would wait for {host.name} to come back")
        return
    await asyncio.sleep(10)  # let sshd actually go down
    await wait_for_host(host, timeout=timeout)


async def wait_for_host(host: Host, timeout: int = 600, interval: int = 5):
    """Poll until ``host`` answers a trivial command, or raise."""
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        try:
            kwargs = {"capture": True, "timeout": 10}
            # RemoteHost supports connect_timeout; LocalHost does not.
            if hasattr(host, "_build_args"):
                kwargs["connect_timeout"] = 5  # type: ignore[assignment]
            rc, _ = await host.run("true", **kwargs)
            if rc == 0:
                logger.info(f"{host.name}: back up")
                return
        except Exception as e:
            logger.debug(f"{host.name}: probe failed: {e}")
        await asyncio.sleep(interval)
    raise RuntimeError(f"{host.name}: did not come back within {timeout}s after reboot")
