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
    assert set(feats) == {"sm_count", "smem_per_sm", "smem_per_block", "regs_per_block", "warp_size", "total_mem"}
    assert feats["sm_count"] == 170.0
    assert feats["total_mem"] == gpu.by_name("NVIDIA GeForce RTX 5090").vram_bytes  # distinguishes same-die SKUs
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
    # total_mem (the same-die SKU discriminator) rides the fallback too.
    assert gpu.probe_live_features("NVIDIA GeForce RTX 4090")["total_mem"] == gpu.by_name("NVIDIA GeForce RTX 4090").vram_bytes


def test_by_name_canonicalizes_reported_aliases():
    """Live datacenter cards report bare ``cudaDeviceProp.name`` strings that differ from
    the capacity-suffixed registry names; the registry aliases canonicalize them (so a
    live node-store row and a golden built from the canonical name share one gpu string).
    Without these, ``live_name`` canonicalization was a no-op for the exact cards it
    targets."""
    assert gpu.by_name("NVIDIA H200").name == "NVIDIA H200 141GB"
    assert gpu.by_name("NVIDIA H100 80GB HBM3").name == "NVIDIA H100 80GB"
    assert gpu.by_name("NVIDIA A100-SXM4-80GB").name == "NVIDIA A100 80GB"
    assert gpu.by_name("NVIDIA H200 141GB").name == "NVIDIA H200 141GB"  # canonical resolves to itself
    assert gpu.by_name("NVIDIA Totally Made Up") is None  # unknown → None (live_name keeps the raw string)


def test_registry_names_and_aliases_are_unique():
    """No string maps to two specs — every canonical name / alias is unique across the
    registry, so a typo'd alias is caught here instead of being silently shadowed by the
    ``setdefault`` in the ``_BY_NAME`` build (or mis-canonicalizing a card)."""
    seen: dict[str, str] = {}
    for g in gpu.KNOWN_GPUS:
        for label in (g.name, *g.aliases):
            assert label not in seen, f"{label!r} claimed by both {seen[label]!r} and {g.name!r}"
            seen[label] = g.name


def test_hardware_id_distinguishes_same_die_skus(monkeypatch):
    """``Context.hardware_id`` keys on the product name when known — so H100 and H200
    (identical cc + SM features) get distinct identities; an unnamed context falls back
    to a device-regime digest. And ``features()`` carries ``H_total_mem``."""
    from deplodock.compiler.context import Context

    h100 = Context.from_target((9, 0), gpu_name="NVIDIA H100 80GB")
    h200 = Context.from_target((9, 0), gpu_name="NVIDIA H200 141GB")
    assert h100.hardware_id() != h200.hardware_id()  # same cc, distinct product → distinct id
    assert h100.features()["H_total_mem"] != h200.features()["H_total_mem"]  # the distinguishing feature
    # No product name → a stable digest, not a crash.
    anon = Context(compute_capability=(9, 0))
    assert isinstance(anon.hardware_id(), str) and anon.hardware_id()


def test_back_compat_maps_match_registry():
    assert gpu.pci_device_id_to_name()["2684"] == "NVIDIA GeForce RTX 4090"
    assert gpu.short_names()["NVIDIA GeForce RTX 5090"] == "rtx5090"
