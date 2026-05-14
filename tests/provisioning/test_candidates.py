"""Unit tests for VM candidate iteration."""

import pytest

from deplodock.provisioning.candidates import VmCandidate, iter_candidates


def test_iter_candidates_cloudrift_rtx4090_lists_all_alternates():
    """RTX 4090 has four CloudRift entries — all must appear in order."""
    cands = iter_candidates("NVIDIA GeForce RTX 4090", 1, provider=None)
    assert [c.base_type for c in cands] == [
        "rtx49-10c-kn",
        "rtx49-7-50-500-nr",
        "rtx49-15-80-400-ec",
        "rtx49-7c-kn",
    ]
    assert all(c.provider == "cloudrift" for c in cands)
    assert all(c.zone is None for c in cands)
    assert cands[0].instance_type == "rtx49-10c-kn.1"


def test_iter_candidates_cloudrift_uses_gpu_count_in_instance_type():
    cands = iter_candidates("NVIDIA GeForce RTX 4090", 4, provider=None)
    assert cands[0].instance_type == "rtx49-10c-kn.4"


def test_iter_candidates_h200_default_picks_cloudrift_first():
    """H200 has CloudRift first, GCP second in the hardware table."""
    cands = iter_candidates("NVIDIA H200 141GB", 8, provider=None)
    # Cloudrift entry, then GCP entries (one per zone)
    assert cands[0].provider == "cloudrift"
    assert cands[0].instance_type == "h200-8-generic.8"
    gcp_cands = [c for c in cands if c.provider == "gcp"]
    assert len(gcp_cands) == 2
    assert {c.zone for c in gcp_cands} == {"europe-west1-b", "europe-west4-a"}


def test_iter_candidates_provider_filter_restricts_to_gcp():
    """--provider gcp must drop CloudRift entries even when listed first."""
    cands = iter_candidates("NVIDIA H200 141GB", 8, provider="gcp")
    assert {c.provider for c in cands} == {"gcp"}
    assert len(cands) == 2  # one per zone
    assert [c.zone for c in cands] == ["europe-west1-b", "europe-west4-a"]


def test_iter_candidates_provider_filter_unavailable_raises():
    """Asking for CloudRift on an H100 (GCP-only) should explain the available providers."""
    with pytest.raises(ValueError, match="not available on provider 'cloudrift'"):
        iter_candidates("NVIDIA H100 80GB", 1, provider="cloudrift")


def test_iter_candidates_gcp_b200_iterates_zones_before_advancing():
    """GCP B200 has multiple zones for the same machine type; zones come first in iteration."""
    cands = iter_candidates("NVIDIA B200", 8, provider=None)
    assert {c.provider for c in cands} == {"gcp"}
    assert [c.zone for c in cands] == ["asia-southeast1-b", "asia-northeast1-b"]
    assert all(c.instance_type == "a4-highgpu-8g" for c in cands)


def test_iter_candidates_unknown_gpu_raises():
    with pytest.raises(ValueError, match="Unknown GPU"):
        iter_candidates("NVIDIA Made Up 9999", 1, provider=None)


def test_iter_candidates_gcp_fallback_zone_when_not_in_table():
    """GPUs without an explicit GPU_GCP_ZONES entry fall back to DEFAULT_GCP_ZONE."""
    # A100 80GB is GCP-only but not listed in GPU_GCP_ZONES, so it uses DEFAULT_GCP_ZONE
    cands = iter_candidates("NVIDIA A100 80GB", 1, provider=None)
    assert len(cands) == 1
    assert cands[0].provider == "gcp"
    assert cands[0].zone == "us-central1-b"


def test_vm_candidate_describe_cloudrift():
    cand = VmCandidate(provider="cloudrift", base_type="rtx49-7c-kn", instance_type="rtx49-7c-kn.1", zone=None)
    assert cand.describe() == "cloudrift rtx49-7c-kn.1"


def test_vm_candidate_describe_gcp_includes_zone():
    cand = VmCandidate(provider="gcp", base_type="a3-highgpu", instance_type="a3-highgpu-8g", zone="us-central1-b")
    assert "zone=us-central1-b" in cand.describe()
