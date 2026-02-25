"""Cloud VM provisioning: resolve specs, create/delete VMs.

Bridge between the deploy layer and VM providers. All VM-related logic for
bench and deploy CLI goes through this module.
"""

import logging
import os
import shlex

from deplodock.hardware import GPU_INSTANCE_TYPES, resolve_instance_type
from deplodock.provisioning import cloudrift as cr_provider
from deplodock.provisioning import gcp as gcp_provider

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


async def provision_cloud_vm(gpu_name, gpu_count, ssh_key, providers_config=None, server_name=None, dry_run=False, logger=None):
    """Provision a cloud VM for the given GPU requirements.

    Looks up the provider from the hardware table and dispatches to the
    provider's create_instance().

    Returns:
        VMConnectionInfo on success, None on failure.
    """
    logger = logger or logging.getLogger(__name__)

    gpu_entries = GPU_INSTANCE_TYPES.get(gpu_name)
    if not gpu_entries:
        logger.error(f"Unknown GPU '{gpu_name}' — not in hardware table")
        return None

    provider, base_type = gpu_entries[0]
    instance_type = resolve_instance_type(provider, base_type, gpu_count)
    logger.info(f"GPU: {gpu_name} x{gpu_count} -> {provider} {instance_type}")

    if provider == "cloudrift":
        api_key = os.environ.get("CLOUDRIFT_API_KEY")
        if not api_key and not dry_run:
            raise RuntimeError("CLOUDRIFT_API_KEY env var required for CloudRift provisioning")

        pub_key_path = f"{ssh_key}.pub"

        if dry_run:
            logger.info(f"[dry-run] create instance type={instance_type} ssh_key={pub_key_path}")

        conn = await cr_provider.create_instance(
            api_key=api_key or "",
            instance_type=instance_type,
            ssh_key_path=pub_key_path,
            ports=[22, 8000, 8080],
            timeout=600,
            dry_run=dry_run,
            fail_statuses={"Inactive"},
            wait_ssh=True,
            ssh_private_key_path=ssh_key,
        )
        return conn

    elif provider == "gcp":
        gcp_config = (providers_config or {}).get("gcp", {})
        zone = gcp_config.get("zone", "us-central1-b")
        provisioning_model = gcp_config.get("provisioning_model", "FLEX_START")
        ssh_user = gcp_config.get("ssh_user", os.environ.get("USER", "deploy"))
        raw_name = f"bench-{server_name}" if server_name else "bench-vm"
        instance_name = raw_name.lower().replace("_", "-")

        if dry_run:
            logger.info(f"[dry-run] create instance={instance_name} zone={zone} type={instance_type}")

        image_family = gcp_config.get("image_family", "debian-12")
        image_project = gcp_config.get("image_project", "debian-cloud")

        # Build extra gcloud args from config properties and env vars
        extra_parts = []

        service_account = gcp_config.get(
            "service_account",
            os.environ.get("GCP_SERVICE_ACCOUNT", ""),
        )
        if service_account:
            extra_parts.append(f"--service-account={service_account}")
            extra_parts.append("--scopes=https://www.googleapis.com/auth/cloud-platform")

        boot_disk_size = gcp_config.get("boot_disk_size")
        if boot_disk_size:
            extra_parts.append(f"--boot-disk-size={boot_disk_size}")

        tags = gcp_config.get("tags")
        if tags:
            extra_parts.append(f"--tags={tags}")

        # Inject SSH public key into VM metadata for direct SSH access
        pub_key_path = f"{ssh_key}.pub"
        if os.path.exists(pub_key_path) and not dry_run:
            with open(pub_key_path) as f:
                pub_key = f.read().strip()
            metadata_arg = f"--metadata=ssh-keys={ssh_user}:{pub_key}"
            extra_parts.append(shlex.quote(metadata_arg))

        # Append any raw extra_gcloud_args from config
        raw_extra = gcp_config.get("extra_gcloud_args", "")
        if raw_extra:
            extra_parts.append(raw_extra)

        extra_gcloud_args = " ".join(extra_parts) if extra_parts else None

        conn = await gcp_provider.create_instance(
            instance=instance_name,
            zone=zone,
            machine_type=instance_type,
            provisioning_model=provisioning_model,
            image_family=image_family,
            image_project=image_project,
            extra_gcloud_args=extra_gcloud_args,
            wait_ssh=True,
            dry_run=dry_run,
        )
        if conn and conn.username == "":
            conn.username = ssh_user
        return conn

    else:
        raise ValueError(f"Unknown provider: {provider}")


async def delete_cloud_vm(delete_info, dry_run=False):
    """Delete a cloud VM using the info from VMConnectionInfo.delete_info."""
    provider = delete_info[0]

    if provider == "cloudrift":
        instance_id = delete_info[1]
        if dry_run:
            logger.info(f"[dry-run] cloudrift: terminate instance {instance_id}")
            return
        api_key = os.environ.get("CLOUDRIFT_API_KEY", "")
        await cr_provider.delete_instance(api_key, instance_id)

    elif provider == "gcp":
        instance_name = delete_info[1]
        zone = delete_info[2]
        await gcp_provider.delete_instance(instance_name, zone, dry_run=dry_run)
