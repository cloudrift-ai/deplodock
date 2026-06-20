"""Cloud VM provisioning: resolve specs, create/delete VMs.

Bridge between the deploy layer and VM providers. All VM-related logic for
``deploy cloud``, ``bench``, and ``vm create --gpu`` goes through this
module so the three paths share retry, fallback, and orphan-cleanup
semantics.

Allocation model — see :mod:`deplodock.provisioning.candidates`:

* The orchestrator enumerates *candidates* (provider + instance type +
  optional zone) in preference order from the hardware table.
* For each candidate, it makes up to :data:`SAME_CANDIDATE_RETRIES`
  attempts on transient errors.
* On :class:`~deplodock.provisioning.errors.CapacityExhausted`, it
  advances to the next candidate immediately (no same-candidate retry).
* On :class:`~deplodock.provisioning.errors.TerminalProvisionError`, it
  aborts and propagates to the caller.

Provider orphans (VMs that were created but never reached a usable
state) are terminated inside each provider's ``create_instance`` before
it raises — see :mod:`deplodock.provisioning.cloudrift` and
:mod:`deplodock.provisioning.gcp`.
"""

import asyncio
import logging
import os
import secrets
import shlex
from datetime import datetime

from deplodock.hardware import (
    DEFAULT_GCP_PROVISIONING_MODEL,
    GPU_GCP_PROVISIONING_MODEL,
)
from deplodock.provisioning import cloudrift as cr_provider
from deplodock.provisioning import gcp as gcp_provider
from deplodock.provisioning.candidates import VmCandidate, iter_candidates
from deplodock.provisioning.errors import CapacityExhausted, TerminalProvisionError
from deplodock.redact import register_secret

# How many times to retry the *same* candidate on a transient (non-capacity,
# non-terminal) error before moving on to the next candidate.
SAME_CANDIDATE_RETRIES = 2
PROVISION_RETRY_DELAY = 10  # seconds

logger = logging.getLogger(__name__)


def resolve_vm_spec(loaded_configs, server_name=None):
    """Resolve GPU type and count from pre-loaded recipe configs for VM provisioning.

    All recipes in a server entry must target the same GPU. Uses the max
    gpu_count across all entries.

    Args:
        loaded_configs: list of (entry, recipe) tuples where each entry
            is a dict with 'recipe'/'variant' keys and recipe is the
            already-loaded Recipe object.
        server_name: optional server name for error messages.

    Returns:
        (gpu_name, gpu_count) tuple.
    """
    gpu_name = None
    max_gpu_count = 0

    for entry, recipe in loaded_configs:
        recipe_path = entry["recipe"]
        variant = entry.get("variant", "")

        entry_gpu = recipe.deploy.gpu
        entry_gpu_count = recipe.deploy.gpu_count

        if entry_gpu is None:
            raise ValueError(f"Recipe '{recipe_path}' variant '{variant}' is missing 'deploy.gpu' field")

        if gpu_name is None:
            gpu_name = entry_gpu
        elif entry_gpu != gpu_name:
            raise ValueError(
                f"Server '{server_name}': mixed GPUs ({gpu_name} vs {entry_gpu}). All recipes in a server entry must target the same GPU."
            )

        max_gpu_count = max(max_gpu_count, entry_gpu_count)

    return gpu_name, max_gpu_count


def read_public_key_files(paths):
    """Read SSH public key files into a list of key strings (fail-fast).

    Each path must point to a single readable, non-empty public key file.
    Raises so callers abort *before* a VM is provisioned rather than silently
    omitting access.

    Raises:
        FileNotFoundError: a path does not exist.
        ValueError: a file is empty.
    """
    keys = []
    for path in paths or []:
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"Authorized key file not found: {path}")
        with open(expanded) as f:
            content = f.read().strip()
        if not content:
            raise ValueError(f"Authorized key file is empty: {path}")
        keys.append(content)
    return keys


def _ssh_keys_metadata_value(ssh_user, pub_key, extra_keys):
    """Build GCP's newline-separated ``ssh-keys`` metadata value (``user:key`` per line)."""
    entries = [f"{ssh_user}:{pub_key}"]
    entries += [f"{ssh_user}:{key}" for key in (extra_keys or [])]
    return "\n".join(entries)


async def provision_cloud_vm(
    gpu_name,
    gpu_count,
    ssh_key,
    providers_config=None,
    server_name=None,
    dry_run=False,
    logger=None,
    provider=None,
    extra_authorized_keys=None,
    provisioning_model=None,
):
    """Provision a cloud VM for the given GPU requirements.

    Iterates candidates from :func:`iter_candidates` in preference order.
    For each candidate, makes up to :data:`SAME_CANDIDATE_RETRIES` attempts
    on transient failures, then advances to the next candidate on capacity
    failures or after exhausting transient retries.

    Args:
        provider: optional provider filter. When set, candidates are
            restricted to that provider — fallback never silently crosses
            to another provider behind the caller's back.
        extra_authorized_keys: optional list of SSH public key strings to
            install in the VM's authorized_keys alongside ``ssh_key``'s own
            ``.pub`` (use :func:`read_public_key_files` to resolve paths first).
        provisioning_model: GCP-only override for the hardware-table
            provisioning model (FLEX_START / SPOT / STANDARD). ``None`` keeps
            the per-GPU default; ignored for CloudRift candidates.

    Returns:
        VMConnectionInfo on success, None when every candidate is exhausted.

    Raises:
        TerminalProvisionError: on non-retryable provider errors (auth,
            malformed request). Surfaces immediately without trying further
            candidates.
    """
    logger = logger or logging.getLogger(__name__)

    candidates = iter_candidates(gpu_name, gpu_count, provider)
    logger.info(f"GPU: {gpu_name} x{gpu_count} -> {len(candidates)} candidate(s): " + ", ".join(c.describe() for c in candidates))

    last_err: Exception | None = None
    for cand in candidates:
        logger.info(f"Trying candidate: {cand.describe()}")
        for attempt in range(1, SAME_CANDIDATE_RETRIES + 1):
            try:
                conn = await _provision_candidate(
                    cand,
                    gpu_name,
                    gpu_count,
                    ssh_key,
                    providers_config,
                    server_name,
                    dry_run,
                    logger,
                    extra_authorized_keys,
                    provisioning_model,
                )
            except CapacityExhausted as exc:
                logger.warning(f"{cand.describe()}: capacity exhausted ({exc}); advancing to next candidate.")
                last_err = exc
                break
            except TerminalProvisionError:
                # Auth / bad request / etc. — won't change across candidates.
                raise
            except Exception as exc:
                last_err = exc
                if attempt < SAME_CANDIDATE_RETRIES:
                    logger.warning(
                        f"{cand.describe()}: transient failure on attempt {attempt}/{SAME_CANDIDATE_RETRIES} ({exc}); "
                        f"retrying same candidate in {PROVISION_RETRY_DELAY}s."
                    )
                    await asyncio.sleep(PROVISION_RETRY_DELAY)
                    continue
                logger.warning(
                    f"{cand.describe()}: exhausted {SAME_CANDIDATE_RETRIES} same-candidate retries ({exc}); advancing to next candidate."
                )
                break
            else:
                if conn is not None or dry_run:
                    return conn
                # Soft None — treat like capacity exhaustion, advance immediately.
                logger.warning(f"{cand.describe()}: provider returned None; advancing to next candidate.")
                break

    logger.error(f"All {len(candidates)} candidate(s) exhausted for {gpu_name} x{gpu_count}. Last error: {last_err}")
    return None


async def _provision_candidate(
    cand: VmCandidate,
    gpu_name: str,
    gpu_count: int,
    ssh_key: str,
    providers_config,
    server_name,
    dry_run: bool,
    logger,
    extra_authorized_keys=None,
    provisioning_model=None,
):
    """Single provisioning attempt for one resolved candidate.

    Returns a ``VMConnectionInfo`` on success or ``None`` on a soft failure
    the provider couldn't classify (extremely rare; treated like
    ``CapacityExhausted`` by the orchestrator). Capacity-class and terminal
    failures are raised, not returned.
    """
    if cand.provider == "cloudrift":
        return await _provision_cloudrift(cand, ssh_key, providers_config, dry_run, logger, extra_authorized_keys)
    if cand.provider == "gcp":
        return await _provision_gcp(
            cand, gpu_name, ssh_key, providers_config, server_name, dry_run, logger, extra_authorized_keys, provisioning_model
        )
    raise ValueError(f"Unknown provider: {cand.provider}")


async def _provision_cloudrift(cand: VmCandidate, ssh_key, providers_config, dry_run, logger, extra_authorized_keys=None):
    api_key = os.environ.get("CLOUDRIFT_API_KEY")
    if not api_key and not dry_run:
        raise TerminalProvisionError("CLOUDRIFT_API_KEY env var required for CloudRift provisioning")
    register_secret(api_key or "")

    pub_key_path = f"{ssh_key}.pub"

    if dry_run:
        logger.info(f"[dry-run] create instance type={cand.instance_type} ssh_key={pub_key_path}")

    cr_config = (providers_config or {}).get("cloudrift") or {}
    image_url = cr_config.get("image_url")
    billing_exempt = cr_config.get("billing_exempt", False)
    network = cr_config.get("network")

    return await cr_provider.create_instance(
        api_key=api_key or "",
        instance_type=cand.instance_type,
        ssh_key_path=pub_key_path,
        image_url=image_url,
        ports=[22, 8000, 8080],
        timeout=1800,
        dry_run=dry_run,
        fail_statuses={"Inactive"},
        wait_ssh=True,
        ssh_private_key_path=ssh_key,
        billing_exempt=billing_exempt,
        network=network,
        extra_public_keys=extra_authorized_keys,
    )


async def _provision_gcp(
    cand: VmCandidate,
    gpu_name,
    ssh_key,
    providers_config,
    server_name,
    dry_run,
    logger,
    extra_authorized_keys=None,
    provisioning_model=None,
):
    gcp_config = (providers_config or {}).get("gcp", {})
    provisioning_model = provisioning_model or GPU_GCP_PROVISIONING_MODEL.get(gpu_name, DEFAULT_GCP_PROVISIONING_MODEL)
    ssh_user = gcp_config.get("ssh_user", os.environ.get("USER", "deploy"))
    ts = datetime.now().strftime("%m%d-%H%M")
    suffix = secrets.token_hex(2)
    raw_name = f"bench-{server_name}-{ts}-{suffix}" if server_name else f"bench-vm-{ts}-{suffix}"
    instance_name = raw_name.lower().replace("_", "-")

    if dry_run:
        logger.info(f"[dry-run] create instance={instance_name} zone={cand.zone} type={cand.instance_type}")

    image_family = gcp_config.get("image_family", "debian-12")
    image_project = gcp_config.get("image_project", "debian-cloud")

    extra_parts = []

    service_account = gcp_config.get("service_account", os.environ.get("GCP_SERVICE_ACCOUNT", ""))
    if service_account:
        register_secret(service_account)
        extra_parts.append(f"--service-account={service_account}")
        extra_parts.append("--scopes=https://www.googleapis.com/auth/cloud-platform")

    boot_disk_size = gcp_config.get("boot_disk_size")
    if boot_disk_size:
        extra_parts.append(f"--boot-disk-size={boot_disk_size}")

    tags = gcp_config.get("tags")
    if tags:
        extra_parts.append(f"--tags={tags}")

    pub_key_path = f"{ssh_key}.pub"
    if not os.path.exists(pub_key_path):
        logger.warning(
            f"No SSH public key at {pub_key_path}; the VM will get no ssh-keys metadata and SSH may rely "
            f"on project/OS-Login keys. Pass --ssh-key pointing at a private key whose .pub exists."
        )
    elif not dry_run:
        with open(pub_key_path) as f:
            pub_key = f.read().strip()
        metadata_value = _ssh_keys_metadata_value(ssh_user, pub_key, extra_authorized_keys)
        # Pin OS Login off for this instance so the per-VM ssh-keys below is honored: a project whose
        # metadata sets enable-oslogin=TRUE otherwise ignores instance ssh-keys entirely. The instance
        # value overrides the project one (this does NOT work if an org policy *enforces*
        # compute.requireOsLogin — then keys must go through OS Login). Both pairs must ride a single
        # --metadata flag (a second --metadata overwrites the first); enable-oslogin goes first so the
        # multi-line ssh-keys value stays last.
        extra_parts.append(shlex.quote(f"--metadata=enable-oslogin=FALSE,ssh-keys={metadata_value}"))

    raw_extra = gcp_config.get("extra_gcloud_args", "")
    if raw_extra:
        extra_parts.append(raw_extra)

    extra_gcloud_args = " ".join(extra_parts) if extra_parts else None

    if provisioning_model == "FLEX_START":
        create_timeout = gcp_config.get("create_timeout_flex_start", 14400)
    else:
        create_timeout = gcp_config.get("create_timeout_spot", 600)

    conn = await gcp_provider.create_instance(
        instance=instance_name,
        zone=cand.zone,
        machine_type=cand.instance_type,
        provisioning_model=provisioning_model,
        image_family=image_family,
        image_project=image_project,
        extra_gcloud_args=extra_gcloud_args,
        timeout=create_timeout,
        wait_ssh=True,
        dry_run=dry_run,
    )
    if conn and conn.username == "":
        conn.username = ssh_user
    return conn


async def delete_cloud_vm(delete_info, dry_run=False):
    """Delete a cloud VM using the info from VMConnectionInfo.delete_info."""
    provider = delete_info[0]

    if provider == "cloudrift":
        instance_id = delete_info[1]
        if dry_run:
            logger.info(f"[dry-run] cloudrift: terminate instance {instance_id}")
            return
        api_key = os.environ.get("CLOUDRIFT_API_KEY", "")
        register_secret(api_key)
        await cr_provider.delete_instance(api_key, instance_id)

    elif provider == "gcp":
        instance_name = delete_info[1]
        zone = delete_info[2]
        await gcp_provider.delete_instance(instance_name, zone, dry_run=dry_run)
