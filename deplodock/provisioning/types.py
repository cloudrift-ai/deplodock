"""Shared data types for VM providers."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class VMConnectionInfo:
    """Structured return from VM providers with all connection details."""

    host: str
    username: str
    ssh_port: int = 22
    port_mappings: List[Tuple[int, int]] = field(default_factory=list)
    delete_info: tuple = ()

    @property
    def address(self) -> str:
        """SSH address string (user@host)."""
        return f"{self.username}@{self.host}" if self.username else self.host
