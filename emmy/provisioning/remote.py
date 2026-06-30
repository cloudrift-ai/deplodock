"""Remote server provisioning: install Docker, NVIDIA driver/toolkit, etc."""

import asyncio
import logging

from deplodock.provisioning.host import Host
from deplodock.provisioning.ssh_transport import REMOTE_DEPLOY_DIR

logger = logging.getLogger(__name__)


async def provision_remote(
    host: Host,
    *,
    skip_nvidia: bool = False,
    driver_version: str | None = None,
    cuda_version: str | None = None,
):
    """Ensure ``host`` is ready for deployment.

    Steps (each checks before installing):
    1. Create the deplodock workspace directory
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
            # Post-reboot verification: catch silent install failures (e.g.
            # CloudRift base images that ship pinned libnvidia-* packages
            # which conflict with the cuda repo's libnvidia-compute version,
            # leaving cuda-drivers in `iU` state with no `nvidia-smi` binary).
            await _verify_nvidia_install(host, driver_version=driver_version, cuda_version=cuda_version)

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

    Returns True if anything was installed (caller should reboot + verify).

    Each `apt-get install` exit code is checked and surfaced as a hard error.
    A previous version of this function silently ignored install failures,
    which let the harness march on past broken `iU` (unpacked-but-unconfigured)
    package states and produce empty bench results without flagging the cause.
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
        # Heal any pre-existing partial dpkg state from the base image so that
        # the new install isn't blocked by leftover unconfigured packages.
        await _heal_dpkg(host)

    if need_driver:
        # CloudRift / GCP / generic cloud images often ship a pre-installed
        # `nvidia-driver-510` (or similar) from Ubuntu's archive. The cuda repo's
        # `nvidia-open-595` carries `libnvidia-* (= 595.58.03-1ubuntu1)` which
        # cannot be unpacked over the older versions — apt's dpkg unpack step
        # fails on libnvidia-gl, libnvidia-cfg1, libnvidia-compute. We purge the
        # old nvidia packages first so the new ones can land cleanly.
        await _purge_existing_nvidia(host)
        # `nvidia-open-<major>` (NOT `cuda-drivers-<major>`) is the open-kernel-
        # module variant published by NVIDIA. Blackwell GPUs (sm_120: RTX 5090,
        # RTX PRO 6000) REQUIRE the open kernel modules — the proprietary
        # `nvidia.ko` from `cuda-drivers-XYZ` loads but its RmInitAdapter fails
        # with NVRM error `(0x22:0x56:1017)` on every Blackwell card, leaving
        # `nvidia-smi` to report "No devices were found". The open driver
        # supports every GPU from Turing (sm_75) onward, so always preferring
        # it is the safe default. Strip non-numeric suffix and take the first
        # dot component.
        major = driver_version.split(".")[0]
        logger.info(f"{host.name}: installing nvidia-open-{major} (requested {driver_version})")
        rc, out = await host.run(
            f"DEBIAN_FRONTEND=noninteractive apt-get install -y nvidia-open-{major}",
            sudo=True,
            timeout=1800,
            capture=True,
        )
        if rc != 0:
            tail = "\n".join(out.splitlines()[-20:]) if out else ""
            raise RuntimeError(f"{host.name}: apt-get install nvidia-open-{major} failed (rc={rc}). Last lines:\n{tail}")
        installed = True

    if need_cuda:
        pkg = "cuda-toolkit-" + cuda_version.replace(".", "-")
        logger.info(f"{host.name}: installing {pkg}")
        rc, out = await host.run(
            f"DEBIAN_FRONTEND=noninteractive apt-get install -y {pkg}",
            sudo=True,
            timeout=1800,
            capture=True,
        )
        if rc != 0:
            tail = "\n".join(out.splitlines()[-20:]) if out else ""
            raise RuntimeError(f"{host.name}: apt-get install {pkg} failed (rc={rc}). Last lines:\n{tail}")
        # Sanity-check immediately: did the package actually create the
        # /usr/local/cuda-X.Y tree? apt occasionally exits 0 even when nothing
        # got installed (e.g., kept-back packages with --no-install-recommends).
        if not await _cuda_installed(host, cuda_version):
            raise RuntimeError(
                f"{host.name}: apt-get install {pkg} reported success but /usr/local/cuda-{cuda_version} does not exist on the host"
            )
        installed = True

    return installed


async def _purge_existing_nvidia(host: Host) -> None:
    """Purge any pre-installed nvidia / libnvidia packages so the cuda repo's
    versioned libs can unpack without dpkg conflicts.

    CloudRift's stock Ubuntu image ships nvidia-driver-510 + libnvidia-* from
    Ubuntu's archive. The NVIDIA cuda repo's cuda-drivers-595 brings strict
    version dependencies (libnvidia-compute = 595.58.03-1ubuntu1) that cannot
    be unpacked over the existing 510 files. Purging first is heavy-handed
    but reliable; the install step right after will pull the requested
    versions back in.
    """
    logger.info(f"{host.name}: purging any existing nvidia/libnvidia packages")
    rc, out = await host.run(
        "DEBIAN_FRONTEND=noninteractive apt-get purge -y "
        "'nvidia-*' 'libnvidia-*' 'cuda-drivers*' 2>&1 || true; "
        "DEBIAN_FRONTEND=noninteractive apt-get autoremove -y || true",
        sudo=True,
        timeout=600,
        capture=True,
    )
    if rc != 0:
        tail = "\n".join(out.splitlines()[-10:]) if out else ""
        logger.warning(f"{host.name}: nvidia purge step exited rc={rc} (continuing). Tail:\n{tail}")


async def _heal_dpkg(host: Host) -> None:
    """Run dpkg --configure -a + apt --fix-broken install before a fresh apt op.

    CloudRift / GCP / generic Ubuntu base images sometimes ship with leftover
    partially-configured packages (e.g. nvidia-driver in `iU` state from a
    failed earlier provisioning attempt). Trying to install on top of that
    state produces 'Unmet dependencies' errors that are very hard to debug
    after the fact. Healing first is cheap and idempotent.
    """
    logger.info(f"{host.name}: healing dpkg state (configure -a + fix-broken install)")
    rc, out = await host.run(
        "DEBIAN_FRONTEND=noninteractive dpkg --configure -a && DEBIAN_FRONTEND=noninteractive apt-get install -y --fix-broken",
        sudo=True,
        timeout=600,
        capture=True,
    )
    if rc != 0:
        tail = "\n".join(out.splitlines()[-15:]) if out else ""
        logger.warning(f"{host.name}: dpkg heal step exited rc={rc} (continuing anyway). Tail:\n{tail}")


async def _verify_nvidia_install(
    host: Host,
    *,
    driver_version: str | None,
    cuda_version: str | None,
) -> None:
    """After install + reboot, confirm the box is actually usable.

    Checks (raises RuntimeError on any failure):
    - `nvidia-smi` exists and exits 0 (with retry — the kernel module takes
      a few seconds to finish loading after sshd comes back, and probing
      nvidia-smi too early returns rc=6 / NVML-not-found)
    - reported driver version matches `driver_version` if requested
    - `/usr/local/cuda-{cuda_version}` exists if requested
    """
    smi_out: str | None = None
    last_rc = -1
    for _attempt in range(12):  # ~60s of retries
        rc, out = await host.run(
            "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader",
            capture=True,
        )
        if rc == 0 and out:
            smi_out = out
            break
        last_rc = rc
        await asyncio.sleep(5)
    if smi_out is None:
        raise RuntimeError(
            f"{host.name}: post-install verification failed — nvidia-smi did not "
            f"return success after 60s of retries (last rc={last_rc}). The driver "
            f"install likely produced an unconfigured (iU) package state, or the "
            f"kernel module failed to load. Check `dpkg -l | grep -i nvidia` and "
            f"`dmesg | grep -i nvidia` on the host."
        )
    logger.info(f"{host.name}: post-install nvidia-smi: {smi_out.strip()}")

    if driver_version:
        current = await _current_driver_version(host)
        if not _matches(current, driver_version):
            raise RuntimeError(f"{host.name}: post-install driver version mismatch — requested {driver_version}, got {current}")

    if cuda_version and not await _cuda_installed(host, cuda_version):
        raise RuntimeError(
            f"{host.name}: post-install CUDA toolkit /usr/local/cuda-{cuda_version} "
            f"is missing despite the install command reporting success"
        )


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
