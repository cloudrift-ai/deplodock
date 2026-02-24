"""Hardware lookup tables: GPU brand -> provider instance types."""

# GPU brand -> list of (provider, base_instance_type) in preference order.
#
# Instance type naming:
#   CloudRift: "{base}.{gpu_count}"       e.g. rtx49-10c-kn.4
#   GCP:       "{base}-{gpu_count}g"      e.g. a3-highgpu-8g
#   GCP g4:    "g4-standard-{gpu_count * 48}"  e.g. g4-standard-192
GPU_INSTANCE_TYPES = {
    "NVIDIA GeForce RTX 4090": [
        ("cloudrift", "rtx49-10c-kn"),
        ("cloudrift", "rtx49-7-50-500-nr"),
        ("cloudrift", "rtx49-15-80-400-ec"),
        ("cloudrift", "rtx49-7c-kn"),
    ],
    "NVIDIA GeForce RTX 5090": [
        ("cloudrift", "rtx59-7-50-400-ec"),
        ("cloudrift", "rtx59-15-80-400-ec"),
        ("cloudrift", "rtx59-16c-nr"),
        ("cloudrift", "rtx59-11-56-850-1lg"),
    ],
    "NVIDIA RTX PRO 6000 Workstation Edition": [
        ("cloudrift", "rtxpro6000-12-100-1500-nr"),
        ("cloudrift", "rtxpro6000-4-100-1000-ti"),
        ("cloudrift", "rtxpro6000-11-50-500-1l"),
    ],
    "NVIDIA RTX PRO 6000 Server Edition": [
        ("gcp", "g4-standard"),
    ],
    "NVIDIA L40S": [
        ("cloudrift", "l40s-24c-kn"),
    ],
    "NVIDIA H100 80GB": [
        ("gcp", "a3-highgpu"),
    ],
    "NVIDIA H200 141GB": [
        ("gcp", "a3-ultragpu"),
    ],
    "NVIDIA B200": [
        ("gcp", "a4-highgpu"),
    ],
    "NVIDIA A100 40GB": [
        ("gcp", "a2-highgpu"),
    ],
    "NVIDIA A100 80GB": [
        ("gcp", "a2-ultragpu"),
    ],
    "AMD Instinct MI350X": [
        ("cloudrift", "mi350x-15-250-1000-gv"),
    ],
}


# Full GPU name -> short name for result filenames.
GPU_SHORT_NAMES = {
    "NVIDIA GeForce RTX 4090": "rtx4090",
    "NVIDIA GeForce RTX 5090": "rtx5090",
    "NVIDIA RTX PRO 6000 Workstation Edition": "pro6000",
    "NVIDIA RTX PRO 6000 Server Edition": "pro6000",
    "NVIDIA L40S": "l40s",
    "NVIDIA H100 80GB": "h100",
    "NVIDIA H200 141GB": "h200",
    "NVIDIA B200": "b200",
    "NVIDIA A100 40GB": "a100",
    "NVIDIA A100 80GB": "a100",
    "AMD Instinct MI350X": "mi350x",
}


def gpu_short_name(full_name):
    """Map a full GPU name to a short name for filenames.

    Falls back to lowercased alphanumeric if not in the lookup table.
    """
    if full_name in GPU_SHORT_NAMES:
        return GPU_SHORT_NAMES[full_name]
    import re

    return re.sub(r"[^a-z0-9]", "", full_name.lower())


def resolve_instance_type(provider, base, gpu_count):
    """Derive full instance type name from base name and GPU count."""
    if provider == "cloudrift":
        return f"{base}.{gpu_count}"
    if base == "g4-standard":
        return f"g4-standard-{gpu_count * 48}"
    return f"{base}-{gpu_count}g"
