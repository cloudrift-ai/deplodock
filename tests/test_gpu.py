"""Unit tests for the common GPU registry (:mod:`deplodock.gpu`)."""

from deplodock import gpu


def test_lookup_by_name_and_pci():
    g = gpu.by_name("NVIDIA GeForce RTX 5090")
    assert g is not None and g.compute_capability == (12, 0) and g.sm_count == 170
    assert gpu.by_pci_device_id("2b85") is g
    assert gpu.by_pci_device_id("0x2B85") is g  # case / 0x-prefix tolerant
    assert gpu.by_name("No Such GPU") is None
    assert gpu.by_pci_device_id("ffff") is None


def test_same_cap_cards_are_distinct_by_specs():
    # The case that motivated the registry: 5090 and PRO 6000 share compute_cap
    # (12, 0) but differ in SM count / VRAM, so they ARE distinguishable here.
    a = gpu.by_name("NVIDIA GeForce RTX 5090")
    b = gpu.by_name("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition")
    assert a.compute_capability == b.compute_capability == (12, 0)
    assert a.sm_count != b.sm_count and (a.sm_count, b.sm_count) == (170, 188)
    assert a.vram_mib != b.vram_mib


def test_derived_cc_facts():
    g = gpu.by_name("NVIDIA GeForce RTX 4090")  # sm_89
    assert g.smem_optin == gpu.MAX_DYNAMIC_SMEM_BY_CC[(8, 9)]
    assert g.tensor_core_gen == gpu.TENSOR_CORE_GEN[(8, 9)] == 3
    assert g.vram_bytes == g.vram_mib * 1024 * 1024


def test_device_features_shape():
    feats = gpu.by_name("NVIDIA GeForce RTX 5090").device_features()
    assert set(feats) == {"sm_count", "smem_per_sm", "smem_per_block", "regs_per_block", "warp_size"}
    assert feats["sm_count"] == 170.0
    # AMD card has no CUDA specs → empty feature dict (degrades like a GPU-less host).
    assert gpu.by_name("AMD Instinct MI350X").device_features() == {}


def test_probe_falls_back_to_memorized(monkeypatch):
    # Force the cupy probe to fail → memorized fallback of the named card / default.
    import builtins

    real_import = builtins.__import__

    def no_cupy(name, *a, **k):
        if name == "cupy":
            raise ImportError("no cupy")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_cupy)
    assert gpu.probe_live_features() == gpu.DEFAULT_GPU.device_features()
    assert gpu.probe_live_features("NVIDIA GeForce RTX 4090")["sm_count"] == 128.0


def test_back_compat_maps_match_registry():
    assert gpu.pci_device_id_to_name()["2684"] == "NVIDIA GeForce RTX 4090"
    assert gpu.short_names()["NVIDIA GeForce RTX 5090"] == "rtx5090"
