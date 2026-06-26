"""Common GPU registry — the project-wide single source of truth for *physical*
GPU facts: PCI device IDs, the canonical name, and hardware specs (compute
capability, SM count, shared-memory sizes, register file, VRAM).

By convention the **PCIe product name** (what ``cudaDeviceProp.name`` /
``nvidia-smi --query-gpu=name`` report, e.g. ``"NVIDIA GeForce RTX 5090"``) is
*the* GPU name used everywhere in the project — golden files, the hardware /
provisioning tables, detection, results filenames. Look a card up by that name
(:func:`by_name`) or by its PCI device id (:func:`by_pci_device_id`).

When a live CUDA probe is available (cupy present), :func:`probe_live_features`
returns the device's real properties; otherwise it falls back to the **memorized
specs** recorded here for :data:`DEFAULT_GPU` (or a name passed in). This lets
GPU-less hosts (CI, offline golden ranking, cross-target compiles) get faithful
per-SKU regime features instead of a single hardcoded constant.

Cloud-provisioning facts (instance types, zones, provisioning models) stay in
:mod:`deplodock.hardware` — those are provider/account specifics, not physical
properties of the silicon; this module owns only the hardware itself.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

_MIB = 1024 * 1024

# Per-block static-smem cap — hard hardware limit since Maxwell (sm_50); anything
# declared ``__shared__`` at compile time must fit. Universal across every arch.
STATIC_SMEM_CAP = 48 * 1024

# Common register-file-per-block ceiling (modern NVIDIA: 64K 32-bit regs/CTA).
_REGS_PER_BLOCK = 64 * 1024

# Per-block dynamic-smem opt-in cap by compute capability
# (``cudaDevAttrMaxSharedMemoryPerBlockOptin``). Keyed on cc, not "arch
# generation": NVIDIA assigns different sm_XX to datacenter vs consumer SKUs in
# one arch family (sm_80 A100 vs sm_86 RTX 30xx; sm_90 H100 vs sm_120 RTX 50xx).
MAX_DYNAMIC_SMEM_BY_CC: dict[tuple[int, int], int] = {
    (7, 0): 96 * 1024,
    (7, 5): 64 * 1024,
    (8, 0): 163 * 1024,
    (8, 6): 99 * 1024,
    (8, 9): 99 * 1024,
    (9, 0): 227 * 1024,
    (10, 0): 227 * 1024,
    (12, 0): 99 * 1024,
}

# Tensor-core generation by compute capability —
# Volta(1)/Turing(2)/Ampere+Ada(3)/Hopper(4)/Blackwell(5). A coarse arch-capability
# axis for the learned prior's regime features; unknown ccs fall back to the major.
TENSOR_CORE_GEN: dict[tuple[int, int], int] = {
    (7, 0): 1,
    (7, 5): 2,
    (8, 0): 3,
    (8, 6): 3,
    (8, 9): 3,
    (9, 0): 4,
    (10, 0): 5,
    (12, 0): 5,
}


@dataclass(frozen=True, kw_only=True)
class GpuSpec:
    """Physical facts for one GPU SKU. ``name`` is the canonical PCIe product
    name used as the GPU name across the whole project.

    Per-SKU fields are stored directly; cc-derived facts (``smem_optin``,
    ``tensor_core_gen``) are looked up from the cc-keyed tables so they can't
    drift. ``sm_count`` / ``smem_per_sm`` / ``vram_mib`` recorded here are the
    **memorized fallback** used when no live probe is available; ``None`` means
    "unknown" (the consumer degrades as it would on a GPU-less host)."""

    name: str
    pci_device_ids: tuple[str, ...] = ()  # hex device ids (lowercase, no 0x prefix)
    vendor: str = "NVIDIA"
    short_name: str | None = None  # for results filenames (see hardware.gpu_short_name)
    compute_capability: tuple[int, int] | None = None
    sm_count: int | None = None
    smem_per_sm: int | None = None
    smem_per_block: int = STATIC_SMEM_CAP
    regs_per_block: int = _REGS_PER_BLOCK
    warp_size: int = 32
    vram_mib: int | None = None
    aliases: tuple[str, ...] = field(default=())  # alternate reported names

    @property
    def vram_bytes(self) -> int | None:
        return self.vram_mib * _MIB if self.vram_mib is not None else None

    @property
    def smem_optin(self) -> int:
        """Per-block dynamic-smem opt-in cap (cc-derived; static cap if unknown)."""
        return MAX_DYNAMIC_SMEM_BY_CC.get(self.compute_capability, STATIC_SMEM_CAP)

    @property
    def tensor_core_gen(self) -> int | None:
        if self.compute_capability is None:
            return None
        return TENSOR_CORE_GEN.get(self.compute_capability, self.compute_capability[0])

    def device_features(self) -> dict[str, float]:
        """The ``sm_count`` / ``smem_*`` / ``regs`` / ``warp`` / ``total_mem`` subset,
        matching the keys :func:`probe_live_features` reads from a live device — the
        memorized fallback. ``total_mem`` (VRAM bytes) is the one feature that
        distinguishes same-die SKUs the prior otherwise can't tell apart (H100 80GB vs
        H200 141GB share cc + SM count). Empty when ``sm_count`` is unknown (degrades
        like a GPU-less host); ``total_mem`` is omitted when VRAM is unknown."""
        if self.sm_count is None or self.smem_per_sm is None:
            return {}
        feats = {
            "sm_count": float(self.sm_count),
            "smem_per_sm": float(self.smem_per_sm),
            "smem_per_block": float(self.smem_per_block),
            "regs_per_block": float(self.regs_per_block),
            "warp_size": float(self.warp_size),
        }
        if self.vram_bytes is not None:
            feats["total_mem"] = float(self.vram_bytes)
        return feats


# Specs marked "measured" are exact values probed off the live card (recorded in
# the golden-sweep priors). The rest carry the canonical name / PCI ids / cc /
# VRAM (public nominal SKU specs) so detection + name resolution work; their
# ``sm_count`` is the documented nominal count where known.
KNOWN_GPUS: tuple[GpuSpec, ...] = (
    # --- measured (golden-sweep cards) ------------------------------------
    GpuSpec(
        name="NVIDIA GeForce RTX 4090",
        pci_device_ids=("2684",),
        short_name="rtx4090",
        compute_capability=(8, 9),
        sm_count=128,
        smem_per_sm=102400,
        vram_mib=24564,
    ),  # measured
    GpuSpec(
        name="NVIDIA GeForce RTX 5090",
        pci_device_ids=("2b85",),
        short_name="rtx5090",
        compute_capability=(12, 0),
        sm_count=170,
        smem_per_sm=102400,
        vram_mib=32607,
    ),  # measured
    GpuSpec(
        name="NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        pci_device_ids=("2ba2", "2bb4"),
        short_name="pro6000",
        compute_capability=(12, 0),
        sm_count=188,
        smem_per_sm=102400,
        vram_mib=97887,
    ),  # measured
    # --- other PRO 6000 Blackwell variants (same GB202 die, cc 12.0) ------
    GpuSpec(
        name="NVIDIA RTX PRO 6000 Blackwell Workstation Edition",
        pci_device_ids=("2ba0",),
        short_name="pro6000",
        compute_capability=(12, 0),
        sm_count=188,
        smem_per_sm=102400,
        vram_mib=98304,
    ),
    GpuSpec(
        name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        pci_device_ids=("2ba4",),
        short_name="pro6000",
        compute_capability=(12, 0),
        sm_count=188,
        smem_per_sm=102400,
        vram_mib=98304,
    ),
    # --- datacenter / workstation (nominal public specs) -----------------
    GpuSpec(
        name="NVIDIA L40S",
        pci_device_ids=("26b9",),
        short_name="l40s",
        compute_capability=(8, 9),
        sm_count=142,
        smem_per_sm=102400,
        vram_mib=46068,
    ),
    GpuSpec(
        name="NVIDIA H100 80GB",
        pci_device_ids=("2330", "2331"),
        short_name="h100",
        compute_capability=(9, 0),
        sm_count=132,
        smem_per_sm=233472,
        vram_mib=81559,
        # ``cudaDeviceProp.name`` reports the SXM5 form, not the capacity-suffixed
        # registry name — so a live card canonicalizes via the alias (see ``live_name``).
        aliases=("NVIDIA H100 80GB HBM3",),
    ),
    GpuSpec(
        name="NVIDIA H200 141GB",
        pci_device_ids=("2335",),
        short_name="h200",
        compute_capability=(9, 0),
        sm_count=132,
        smem_per_sm=233472,
        vram_mib=143771,
        aliases=("NVIDIA H200",),  # bare reported name (SXM); registry name carries the capacity
    ),
    GpuSpec(
        name="NVIDIA B200",
        pci_device_ids=("2900", "2901"),
        short_name="b200",
        compute_capability=(10, 0),
        sm_count=148,
        smem_per_sm=233472,
        vram_mib=183359,
    ),
    GpuSpec(
        name="NVIDIA A100 40GB",
        pci_device_ids=("20f1",),
        short_name="a100",
        compute_capability=(8, 0),
        sm_count=108,
        smem_per_sm=167936,
        vram_mib=40960,
        # SXM4 / PCIe report distinct ``cudaDeviceProp.name`` forms; both 40GB A100s
        # share this spec's SM count + VRAM, so canonicalize them to one identity.
        aliases=("NVIDIA A100-SXM4-40GB", "NVIDIA A100-PCIE-40GB"),
    ),
    GpuSpec(
        name="NVIDIA A100 80GB",
        pci_device_ids=("20b2", "20b5"),
        short_name="a100",
        compute_capability=(8, 0),
        sm_count=108,
        smem_per_sm=167936,
        vram_mib=81920,
        aliases=("NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"),
    ),
    GpuSpec(
        name="NVIDIA Tesla V100 SXM3 32GB",
        pci_device_ids=("1db8",),
        short_name="v100",
        compute_capability=(7, 0),
        sm_count=80,
        smem_per_sm=98304,
        vram_mib=32768,
    ),
    # --- AMD (no CUDA compute capability) --------------------------------
    GpuSpec(name="AMD Instinct MI350X", pci_device_ids=("75b0",), vendor="AMD", short_name="mi350x", vram_mib=294912),
)

# The default card when no live probe is available — the GPU the golden configs
# were measured on, so offline golden ranking matches deploy. (RTX 5090, 170 SMs.)
DEFAULT_GPU: GpuSpec = next(g for g in KNOWN_GPUS if g.name == "NVIDIA GeForce RTX 5090")

_BY_NAME: dict[str, GpuSpec] = {}
for _g in KNOWN_GPUS:
    _BY_NAME[_g.name] = _g
    for _a in _g.aliases:
        _BY_NAME.setdefault(_a, _g)
_BY_PCI: dict[str, GpuSpec] = {pid: g for g in KNOWN_GPUS for pid in g.pci_device_ids}


def by_name(name: str) -> GpuSpec | None:
    """The :class:`GpuSpec` for a PCIe product name (or a recorded alias), or ``None``."""
    return _BY_NAME.get(name)


def by_pci_device_id(device_id: str) -> GpuSpec | None:
    """The :class:`GpuSpec` for a PCI device id (hex, lowercase, no ``0x``), or ``None``."""
    return _BY_PCI.get(device_id.lower().removeprefix("0x"))


def pci_device_id_to_name() -> dict[str, str]:
    """``{pci_device_id: canonical_name}`` over the registry — the back-compat map
    :mod:`deplodock.detect` exposes as ``GPU_PCI_DEVICE_IDS``."""
    return {pid: g.name for g in KNOWN_GPUS for pid in g.pci_device_ids}


def short_names() -> dict[str, str]:
    """``{canonical_name: short_name}`` over the registry (back-compat for
    :data:`deplodock.hardware.GPU_SHORT_NAMES`)."""
    return {g.name: g.short_name for g in KNOWN_GPUS if g.short_name}


def probe_live_features(fallback_name: str | None = None) -> dict[str, float]:
    """Physical device features (``sm_count`` / ``smem_per_sm`` / ``smem_per_block``
    / ``regs_per_block`` / ``warp_size``) of the live CUDA device via cupy. When no
    device is visible (or cupy is missing), fall back to the **memorized** specs of
    ``fallback_name`` (a PCIe name) — or :data:`DEFAULT_GPU` — so GPU-less hosts
    still get per-SKU features. Empty only when even the fallback spec is unknown."""
    try:
        import cupy as cp  # noqa: PLC0415

        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        return {
            "sm_count": float(props["multiProcessorCount"]),
            "smem_per_sm": float(props["sharedMemPerMultiprocessor"]),
            "smem_per_block": float(props["sharedMemPerBlock"]),
            "regs_per_block": float(props["regsPerBlock"]),
            "warp_size": float(props["warpSize"]),
            "total_mem": float(props["totalGlobalMem"]),
        }
    except Exception:  # noqa: BLE001 — no live device ⇒ memorized fallback
        spec = (by_name(fallback_name) if fallback_name else None) or DEFAULT_GPU
        return spec.device_features()


@functools.cache
def live_name() -> str | None:
    """The live CUDA device's PCIe product name (``cudaDeviceProp.name``), canonicalized
    to the registry name via :func:`by_name` (datacenter cards report a bare name —
    ``"NVIDIA H200"`` — that the registry carries as an alias of the capacity-suffixed
    ``"NVIDIA H200 141GB"``); the raw reported name when unrecognized, ``None`` when no
    device is visible. The hardware identity that distinguishes same-die SKUs (H100 vs
    H200) — ``compute_capability`` + SM features alone can't. Canonicalizing matters so a
    live node-store row and a golden reconstructed from the canonical name share one
    ``gpu`` string. Cached: physical, target-independent."""
    try:
        import cupy as cp  # noqa: PLC0415

        raw = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    except Exception:  # noqa: BLE001 — no live device
        return None
    name = raw.decode() if isinstance(raw, (bytes, bytearray)) else str(raw)
    spec = by_name(name)
    return spec.name if spec else name
