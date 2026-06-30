"""Build the ordered list of VM allocation candidates for a GPU spec.

A *candidate* is one concrete attempt: a provider, a resolved instance
type, and (for GCP) a specific zone. The orchestrator in
:mod:`emmy.provisioning.cloud` iterates these in priority order,
falling back to the next candidate when one reports
:class:`~emmy.provisioning.errors.CapacityExhausted`.

Order:

1. Filter ``GPU_INSTANCE_TYPES[gpu_name]`` by ``provider`` if set; else
   use the whole list (preference order from the hardware table).
2. For each ``(provider, base_type)`` entry:

   * CloudRift → emit one candidate.
   * GCP → emit one candidate per zone in
     ``GPU_GCP_ZONES.get(gpu_name, [DEFAULT_GCP_ZONE])``, **all zones for
     this base_type before moving to the next entry**.

Fallback never crosses the provider boundary of the initially-selected
entry: if the caller passes ``--provider gcp`` (or implicitly via the
first hardware-table entry), candidates stay within that provider.
"""

from dataclasses import dataclass

from emmy.hardware import (
    DEFAULT_GCP_ZONE,
    GPU_GCP_ZONES,
    GPU_INSTANCE_TYPES,
    resolve_instance_type,
)


@dataclass(frozen=True)
class VmCandidate:
    """One concrete provisioning attempt.

    Attributes:
        provider: ``"cloudrift"`` or ``"gcp"``.
        base_type: hardware-table base name (e.g. ``"a3-highgpu"``).
        instance_type: resolved full name (e.g. ``"a3-highgpu-8g"``).
        zone: GCP zone; ``None`` for CloudRift.
    """

    provider: str
    base_type: str
    instance_type: str
    zone: str | None

    def describe(self) -> str:
        if self.zone:
            return f"{self.provider} {self.instance_type} zone={self.zone}"
        return f"{self.provider} {self.instance_type}"


def iter_candidates(gpu_name: str, gpu_count: int, provider: str | None) -> list[VmCandidate]:
    """Return the ordered list of candidates for a (GPU, count) pair.

    Args:
        gpu_name: full GPU name from the hardware table (e.g. ``"NVIDIA H200 141GB"``).
        gpu_count: requested GPU count for the instance.
        provider: optional provider filter (``"cloudrift"`` or ``"gcp"``).
            When set, only entries matching the provider are emitted, and
            an explicit ``ValueError`` is raised if no entry matches.

    Returns:
        A list of :class:`VmCandidate` in preference order. The orchestrator
        is expected to try them in that order until one succeeds.

    Raises:
        ValueError: if ``gpu_name`` is not in the hardware table or if
            ``provider`` is set and no matching entry exists.
    """
    entries = GPU_INSTANCE_TYPES.get(gpu_name)
    if not entries:
        raise ValueError(f"Unknown GPU '{gpu_name}' — not in hardware table")

    if provider is not None:
        filtered = [e for e in entries if e[0] == provider]
        if not filtered:
            available = sorted({p for p, _ in entries})
            raise ValueError(f"GPU '{gpu_name}' not available on provider '{provider}'. Available: {available}")
        entries = filtered

    candidates: list[VmCandidate] = []
    for entry_provider, base_type in entries:
        instance_type = resolve_instance_type(entry_provider, base_type, gpu_count)
        if entry_provider == "gcp":
            zones = GPU_GCP_ZONES.get(gpu_name) or [DEFAULT_GCP_ZONE]
            for zone in zones:
                candidates.append(VmCandidate(entry_provider, base_type, instance_type, zone))
        else:
            candidates.append(VmCandidate(entry_provider, base_type, instance_type, None))
    return candidates
