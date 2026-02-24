"""VM lifecycle management: create/delete cloud GPU instances."""


def register_vm_command(subparsers):
    """Register the 'vm' command with create/delete action subparsers."""
    from deplodock.commands.vm.cloudrift import (
        register_create_target as register_cloudrift_create,
    )
    from deplodock.commands.vm.cloudrift import (
        register_delete_target as register_cloudrift_delete,
    )
    from deplodock.commands.vm.gcp_flex_start import register_create_target, register_delete_target

    vm_parser = subparsers.add_parser("vm", help="Manage cloud VM instances")

    action_subparsers = vm_parser.add_subparsers(dest="action", required=True)

    # create action
    create_parser = action_subparsers.add_parser("create", help="Create a VM instance")
    create_subparsers = create_parser.add_subparsers(dest="provider", required=True)
    register_create_target(create_subparsers)
    register_cloudrift_create(create_subparsers)

    # delete action
    delete_parser = action_subparsers.add_parser("delete", help="Delete a VM instance")
    delete_subparsers = delete_parser.add_subparsers(dest="provider", required=True)
    register_delete_target(delete_subparsers)
    register_cloudrift_delete(delete_subparsers)
