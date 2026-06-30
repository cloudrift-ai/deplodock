"""Deploy command layer — registers deploy subcommands."""

from deplodock.commands.deploy.cloud import register_cloud_target
from deplodock.commands.deploy.local import register_local_target
from deplodock.commands.deploy.ssh import register_ssh_target

__all__ = [
    "register_cloud_target",
    "register_local_target",
    "register_ssh_target",
]
