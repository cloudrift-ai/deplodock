"""VM and cloud provisioning: types, SSH polling, shell helpers, providers."""

from emmy.provisioning.cloud import (
    delete_cloud_vm,
    provision_cloud_vm,
    resolve_vm_spec,
)
from emmy.provisioning.remote import provision_remote
from emmy.provisioning.shell import run_shell_cmd
from emmy.provisioning.ssh import wait_for_ssh
from emmy.provisioning.ssh_transport import (
    REMOTE_DEPLOY_DIR,
    make_run_cmd,
    make_write_file,
    scp_file,
    scp_from_remote,
    ssh_base_args,
)
from emmy.provisioning.staging import (
    build_stage_tar,
    enumerate_staged_files,
    stage_to_remote,
)
from emmy.provisioning.types import VMConnectionInfo

__all__ = [
    "VMConnectionInfo",
    "wait_for_ssh",
    "run_shell_cmd",
    "provision_remote",
    "ssh_base_args",
    "make_run_cmd",
    "make_write_file",
    "scp_file",
    "scp_from_remote",
    "stage_to_remote",
    "enumerate_staged_files",
    "build_stage_tar",
    "REMOTE_DEPLOY_DIR",
    "resolve_vm_spec",
    "provision_cloud_vm",
    "delete_cloud_vm",
]
