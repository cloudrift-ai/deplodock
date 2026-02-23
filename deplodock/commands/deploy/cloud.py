"""Cloud deploy target: provision VM, deploy recipe, teardown.

Bridge between the deploy layer and vm providers. All VM-related logic for
bench and deploy CLI goes through this module.
"""

import logging
import os
import sys

from deplodock.commands.deploy import (
    DeployParams,
    deploy as deploy_entry,
    load_recipe,
)
from deplodock.commands.deploy.ssh import provision_remote
from deplodock.hardware import GPU_INSTANCE_TYPES, resolve_instance_type


def resolve_vm_spec(recipe_entries, server_name=None):
    """Resolve GPU type and count from recipe entries for VM provisioning.

    All recipes in a server entry must target the same GPU. Uses the max
    gpu_count across all entries.

    Returns:
        (gpu_name, gpu_count, loaded_configs) where loaded_configs is a list
        of (entry, recipe_config) tuples.
    """
    gpu_name = None
    max_gpu_count = 0
    loaded = []

    for entry in recipe_entries:
        recipe_path = entry['recipe']
        variant = entry.get('variant')
        recipe_config = load_recipe(recipe_path, variant=variant)

        entry_gpu = recipe_config.get('gpu')
        entry_gpu_count = recipe_config.get('gpu_count', 1)

        if entry_gpu is None:
            raise ValueError(
                f"Recipe '{recipe_path}' variant '{variant}' is missing 'gpu' field"
            )

        if gpu_name is None:
            gpu_name = entry_gpu
        elif entry_gpu != gpu_name:
            raise ValueError(
                f"Server '{server_name}': mixed GPUs ({gpu_name} vs {entry_gpu}). "
                "All recipes in a server entry must target the same GPU."
            )

        max_gpu_count = max(max_gpu_count, entry_gpu_count)
        loaded.append((entry, recipe_config))

    return gpu_name, max_gpu_count, loaded


def provision_cloud_vm(gpu_name, gpu_count, ssh_key, providers_config=None,
                       server_name=None, dry_run=False, logger=None):
    """Provision a cloud VM for the given GPU requirements.

    Looks up the provider from the hardware table and dispatches to the
    provider's create_instance().

    Returns:
        VMConnectionInfo on success, None on failure.
    """
    from deplodock.commands.vm import cloudrift as cr_provider
    from deplodock.commands.vm import gcp_flex_start as gcp_provider

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

        conn = cr_provider.create_instance(
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
        instance_name = f"bench-{server_name}" if server_name else "bench-vm"

        if dry_run:
            logger.info(f"[dry-run] create instance={instance_name} zone={zone} type={instance_type}")

        conn = gcp_provider.create_instance(
            instance=instance_name,
            zone=zone,
            machine_type=instance_type,
            wait_ssh=True,
            dry_run=dry_run,
        )
        return conn

    else:
        raise ValueError(f"Unknown provider: {provider}")


def delete_cloud_vm(delete_info, dry_run=False):
    """Delete a cloud VM using the info from VMConnectionInfo.delete_info."""
    from deplodock.commands.vm import cloudrift as cr_provider
    from deplodock.commands.vm import gcp_flex_start as gcp_provider

    provider = delete_info[0]

    if provider == "cloudrift":
        instance_id = delete_info[1]
        if dry_run:
            print(f"[dry-run] cloudrift: terminate instance {instance_id}")
            return
        api_key = os.environ.get("CLOUDRIFT_API_KEY", "")
        cr_provider.delete_instance(api_key, instance_id)

    elif provider == "gcp":
        instance_name = delete_info[1]
        zone = delete_info[2]
        gcp_provider.delete_instance(instance_name, zone, dry_run=dry_run)


# ── CLI handler ────────────────────────────────────────────────────


def handle_cloud(args):
    """CLI handler for 'deploy cloud'."""
    config = load_recipe(args.recipe, variant=args.variant)

    gpu_name = config.get('gpu')
    gpu_count = config.get('gpu_count', 1)
    if not gpu_name:
        print("Error: recipe must have a 'gpu' field (use a variant with GPU info).", file=sys.stderr)
        sys.exit(1)

    ssh_key = os.path.expanduser(args.ssh_key)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")

    conn = provision_cloud_vm(
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        ssh_key=ssh_key,
        server_name=args.name,
        dry_run=args.dry_run,
    )
    if conn is None:
        print("Error: VM provisioning failed.", file=sys.stderr)
        sys.exit(1)

    params = DeployParams(
        server=conn.address,
        ssh_key=ssh_key,
        ssh_port=conn.ssh_port,
        recipe_config=config,
        model_dir=args.model_dir,
        hf_token=hf_token,
        dry_run=args.dry_run,
    )
    provision_remote(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)

    if not deploy_entry(params):
        sys.exit(1)


def register_cloud_target(subparsers):
    """Register the cloud deploy target."""
    parser = subparsers.add_parser("cloud", help="Provision a cloud VM and deploy via SSH")
    parser.add_argument("--recipe", required=True, help="Path to recipe directory")
    parser.add_argument("--variant", default=None, help="Hardware variant (e.g. RTX5090)")
    parser.add_argument("--name", default="cloud-deploy", help="VM name prefix (default: cloud-deploy)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/hf_models", help="Model cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.set_defaults(func=handle_cloud)
