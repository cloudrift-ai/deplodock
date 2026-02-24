"""VM and cloud provisioning: types, SSH polling, shell helpers, providers."""

from deplodock.provisioning.types import VMConnectionInfo
from deplodock.provisioning.ssh import wait_for_ssh
from deplodock.provisioning.shell import run_shell_cmd
from deplodock.provisioning.remote import provision_remote
from deplodock.provisioning.ssh_transport import (
    ssh_base_args,
    make_run_cmd,
    make_write_file,
    scp_file,
    REMOTE_DEPLOY_DIR,
)
from deplodock.provisioning.cloud import (
    resolve_vm_spec,
    provision_cloud_vm,
    delete_cloud_vm,
)

__all__ = [
    "VMConnectionInfo",
    "wait_for_ssh",
    "run_shell_cmd",
    "provision_remote",
    "ssh_base_args",
    "make_run_cmd",
    "make_write_file",
    "scp_file",
    "REMOTE_DEPLOY_DIR",
    "resolve_vm_spec",
    "provision_cloud_vm",
    "delete_cloud_vm",
]
