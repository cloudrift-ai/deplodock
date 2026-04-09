"""Shared parser for `USER@HOST[:PORT]` SSH targets.

Used by `deplodock deploy ssh` and `deplodock bench --ssh` so both commands
accept the exact same syntax.
"""


def parse_ssh_target(target: str) -> tuple[str, str, int]:
    """Parse `user@host[:port]` into (user, host, port).

    Raises ValueError on malformed input. Default port is 22.
    """
    if "@" not in target:
        raise ValueError(f"ssh target must be in user@host[:port] form: {target!r}")
    user, _, rest = target.partition("@")
    if not user or not rest:
        raise ValueError(f"ssh target must be in user@host[:port] form: {target!r}")
    if ":" in rest:
        host, _, port_str = rest.partition(":")
        try:
            port = int(port_str)
        except ValueError as e:
            raise ValueError(f"Invalid port in ssh target {target!r}: {port_str}") from e
    else:
        host, port = rest, 22
    if not host:
        raise ValueError(f"ssh target missing host: {target!r}")
    return user, host, port
