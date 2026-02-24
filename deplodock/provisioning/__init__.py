"""VM and cloud provisioning: types, SSH polling, shell helpers, providers."""

from deplodock.provisioning.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
    resolve_vm_spec,
)
from deplodock.provisioning.remote import provision_remote
from deplodock.provisioning.shell import run_shell_cmd
from deplodock.provisioning.ssh import wait_for_ssh
from deplodock.provisioning.ssh_transport import (
    REMOTE_DEPLOY_DIR,
    make_run_cmd,
    make_write_file,
    scp_file,
    ssh_base_args,
)
from deplodock.provisioning.types import VMConnectionInfo

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
