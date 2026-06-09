"""Tests for scripts/new_models.py — the pure normalization / filtering logic (no network)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import new_models  # noqa: E402

# ── _base_key: org prefix dropped, lowercased, quant suffix peeled ──────────────────


def test_base_key_strips_org_and_lowercases():
    assert new_models._base_key("deepseek-ai/DeepSeek-V4-Flash") == "deepseek-v4-flash"


def test_base_key_collapses_quant_variants():
    # Every quant variant of one model must map to the same base key.
    variants = [
        "Qwen/Qwen3.6-35B-A3B",
        "Qwen/Qwen3.6-35B-A3B-FP8",
        "nvidia/Qwen3.6-35B-A3B-NVFP4",
        "amd/Qwen3.6-35B-A3B-MXFP4",
    ]
    assert {new_models._base_key(v) for v in variants} == {"qwen3.6-35b-a3b"}


def test_base_key_peels_compound_suffix():
    # "-awq-int4" must peel as a unit (longest-first), not leave "qwen3.6-27b-awq".
    assert new_models._base_key("cyankiwi/Qwen3.6-27B-AWQ-INT4") == "qwen3.6-27b"
    assert new_models._base_key("RedHatAI/gemma-4-31B-it-FP8-Dynamic") == "gemma-4-31b-it"


# ── supported-set membership: the exclude-supported decision ─────────────────────────


def test_supported_match_excludes_known_family_keeps_new():
    supported = {
        new_models._base_key(x) for x in ["Qwen/Qwen3.6-35B-A3B-FP8", "google/gemma-4-31B-it", "amd/Kimi-K2.5-MXFP4", "zai-org/GLM-5.1-FP8"]
    }
    # A new quant of a supported family is recognized as supported (excluded).
    assert new_models._base_key("nvidia/Qwen3.6-35B-A3B-NVFP4") in supported
    assert new_models._base_key("nvidia/Kimi-K2.5-NVFP4") in supported
    # A genuinely new model is not (kept).
    assert new_models._base_key("deepseek-ai/DeepSeek-V4-Flash") not in supported
    assert new_models._base_key("MiniMaxAI/MiniMax-M3") not in supported


def test_supported_base_keys_reads_recipes():
    keys = new_models.supported_base_keys(ROOT / "recipes")
    assert keys, "expected at least one supported model from recipes/"
    assert all("/" not in k and k == k.lower() for k in keys), "keys must be normalized base names"


# ── open_source_candidates: keep open-weight, dedup by HF repo ───────────────────────


def test_open_source_candidates_filters_and_dedups():
    catalog = [
        {
            "id": "org/a:free",
            "canonical_slug": "org/a",
            "hugging_face_id": "org/A",
            "architecture": {"modality": "text->text", "input_modalities": ["text"]},
            "context_length": 4096,
        },
        {  # duplicate endpoint of the same HF repo -> collapsed
            "id": "org/a",
            "canonical_slug": "org/a",
            "hugging_face_id": "org/A",
            "architecture": {"modality": "text->text", "input_modalities": ["text"]},
        },
        {  # closed model (no hugging_face_id) -> dropped
            "id": "anthropic/claude",
            "canonical_slug": "anthropic/claude",
            "hugging_face_id": None,
            "architecture": {"modality": "text->text", "input_modalities": ["text"]},
        },
        {
            "id": "org/b",
            "canonical_slug": "org/b",
            "hugging_face_id": "org/B",
            "architecture": {"modality": "text+image->text", "input_modalities": ["text", "image"]},
        },
    ]
    out = new_models.open_source_candidates(catalog)
    assert [c["hf_id"] for c in out] == ["org/A", "org/B"]  # closed dropped, dup collapsed, order kept
    assert out[0]["openrouter_id"] == "org/a"
    assert out[1]["input_modalities"] == ["text", "image"]


# ── arena: fuzzy name key, index, open-license, candidate linking ────────────────────


def test_arena_key_bridges_hf_and_arena_naming():
    # An HF instruct repo and the bare arena slug must collapse to the same key.
    assert new_models._arena_key("google/gemma-4-31B-it") == new_models._arena_key("gemma-4-31b")
    # …while distinct arena entries (e.g. a thinking variant) stay separate.
    assert new_models._arena_key("deepseek-v4-pro") != new_models._arena_key("deepseek-v4-pro-thinking")


def test_is_open_license():
    assert new_models._is_open_license("MIT")
    assert new_models._is_open_license("Apache 2.0")
    assert not new_models._is_open_license("Proprietary")


def test_build_arena_index_keeps_lowest_rank_on_collision():
    rows = [
        {"model_name": "glm-5.1", "rating": 1474.4, "rank": 16, "license": "MIT"},
        {"model_name": "glm-5.1-it", "rating": 1470.0, "rank": 20, "license": "MIT"},  # same key, worse rank
        {"model_name": "deepseek-v4-pro", "rating": 1454.1, "rank": 38, "license": "MIT"},
    ]
    index = new_models.build_arena_index(rows)
    assert index[new_models._arena_key("glm-5.1")]["arena_rank"] == 16  # best kept
    assert index[new_models._arena_key("glm-5.1")]["elo"] == 1474  # rounded
    # A candidate links via _arena_key(_base_key(hf_id)).
    cand_key = new_models._arena_key(new_models._base_key("zai-org/GLM-5.1-FP8"))
    assert cand_key in index
