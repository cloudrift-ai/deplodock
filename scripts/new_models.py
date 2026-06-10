#!/usr/bin/env python3
"""Discover new open-source models worth benchmarking.

Pulls the open-weight models OpenRouter currently hosts (catalog entries that carry a
`hugging_face_id` — closed models like Claude/GPT/Gemini do not), verifies each one exists on
HuggingFace, and reports its HF popularity (downloads / likes / trendingScore) and release date.
By default it drops models deplodock already supports (anything whose base model already appears
in `recipes/`) and anything older than ~3 months, leaving a ranked shortlist of fresh candidates.

Both APIs are queried keyless and read-only: OpenRouter `GET /api/v1/models` (public) and the HF
`GET /api/models/{id}` metadata endpoint (200 = on HF + popularity, 404 = stale OpenRouter mapping).

Usage:
    python scripts/new_models.py
    python scripts/new_models.py --since 2026-03-01
    python scripts/new_models.py --since 2026-03-01 --text-only --min-downloads 1000
    python scripts/new_models.py --arena                      # add LMArena Elo + unmatched-open tail
    python scripts/new_models.py --include-supported          # keep families we already have
    python scripts/new_models.py --json > candidates.json
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import re
import sys
from pathlib import Path

import httpx

# Add project root to path (for deplodock.recipe reuse).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from deplodock.recipe.recipe import _load_raw_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
logging.getLogger("httpx").setLevel(logging.WARNING)  # silence per-request INFO chatter
logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
HF_MODEL_URL = "https://huggingface.co/api/models/{}"

# LMArena Elo, published as an HF dataset (keyless). The `text`/`latest` split's `overall`
# category is one row per ranked model (~360), contiguous at the front of the split.
ARENA_ROWS_URL = "https://datasets-server.huggingface.co/rows"
ARENA_DATASET = "lmarena-ai/leaderboard-dataset"

# Quant / format suffixes stripped to collapse a repo id onto its base model. Applied repeatedly
# (longest first) so e.g. "-awq-int4" peels in one pass. Lowercased before matching.
_QUANT_SUFFIXES = (
    "-fp8-dynamic",
    "-awq-4bit",
    "-awq-int4",
    "-nvfp4",
    "-mxfp4",
    "-w4a16",
    "-fp8",
    "-awq",
    "-int4",
    "-4bit",
    "-bf16",
    "-gguf",
    "-dynamic",
)


def _base_key(repo_id: str) -> str:
    """Normalize an HF repo id to a base-model key: drop org prefix, lowercase, strip quant suffix.

    `nvidia/Qwen3.6-35B-A3B-NVFP4` -> `qwen3.6-35b-a3b`; `deepseek-ai/DeepSeek-V4-Flash` ->
    `deepseek-v4-flash`. Heuristic — collapses quant variants of one model, not a true equivalence.
    """
    name = repo_id.split("/")[-1].lower()
    changed = True
    while changed:
        changed = False
        for suffix in _QUANT_SUFFIXES:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
    return name


def supported_base_keys(recipe_root: Path) -> set[str]:
    """Base keys of every `model.huggingface` referenced across `recipes/*/recipe.yaml`."""
    keys: set[str] = set()
    for recipe_dir in sorted(recipe_root.glob("*")):
        if not (recipe_dir / "recipe.yaml").is_file():
            continue
        cfg = _load_raw_config(str(recipe_dir))
        hf_id = (cfg.get("model") or {}).get("huggingface")
        if hf_id:
            keys.add(_base_key(hf_id))
    return keys


def open_source_candidates(catalog: list[dict]) -> list[dict]:
    """Open-weight rows (non-null `hugging_face_id`), deduped by HF repo, first occurrence wins."""
    seen: set[str] = set()
    out: list[dict] = []
    for m in catalog:
        hf_id = m.get("hugging_face_id")
        if not hf_id or hf_id in seen:
            continue
        seen.add(hf_id)
        arch = m.get("architecture") or {}
        out.append(
            {
                "hf_id": hf_id,
                "openrouter_id": m.get("canonical_slug") or m.get("id", ""),
                "modality": arch.get("modality", ""),
                "input_modalities": arch.get("input_modalities") or [],
                "context_length": m.get("context_length"),
            }
        )
    return out


async def fetch_openrouter_catalog(client: httpx.AsyncClient) -> list[dict]:
    """The full OpenRouter model catalog (public, no key)."""
    resp = await client.get(OPENROUTER_MODELS_URL)
    resp.raise_for_status()
    return resp.json()["data"]


async def fetch_hf_info(client: httpx.AsyncClient, hf_id: str, attempts: int = 4) -> dict:
    """One HF metadata GET: existence + popularity, with backoff on rate-limit / 5xx.

    Returns ``status`` ∈ {ok, missing, error, unverified}: 200 -> ok (+ popularity); 404 -> missing
    (stale/private OpenRouter mapping); 429 / 5xx retried (honoring ``Retry-After``) then -> unverified
    if still failing; any other status -> error. ``unverified`` means we couldn't check, NOT absent.
    """
    params = [
        ("expand[]", "downloads"),
        ("expand[]", "likes"),
        ("expand[]", "trendingScore"),
        ("expand[]", "createdAt"),
    ]
    last = ""
    for i in range(attempts):
        retry_after = ""
        try:
            resp = await client.get(HF_MODEL_URL.format(hf_id), params=params)
        except httpx.HTTPError as e:
            last = str(e)
        else:
            if resp.status_code == 200:
                d = resp.json()
                return {
                    "on_hf": True,
                    "status": "ok",
                    "downloads": d.get("downloads", 0) or 0,
                    "likes": d.get("likes", 0) or 0,
                    "trending": d.get("trendingScore", 0) or 0,
                    "created_at": d.get("createdAt", ""),
                }
            if resp.status_code == 404:
                return {"on_hf": False, "status": "missing", "error": "HTTP 404 (not on HF)"}
            if resp.status_code != 429 and resp.status_code < 500:
                return {"on_hf": False, "status": "error", "error": f"HTTP {resp.status_code}"}
            last = f"HTTP {resp.status_code}"
            retry_after = resp.headers.get("Retry-After", "")
        if i < attempts - 1:
            delay = float(retry_after) if retry_after.isdigit() else 1.5 * (i + 1)
            await asyncio.sleep(min(delay, 10.0))
    return {"on_hf": False, "status": "unverified", "error": f"{last} after {attempts} attempts"}


def _created_date(created_at: str) -> dt.date | None:
    if not created_at:
        return None
    try:
        return dt.datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
    except ValueError:
        return None


async def build_rows(client: httpx.AsyncClient, candidates: list[dict], workers: int) -> list[dict]:
    """Enrich each candidate with HF info, with at most `workers` lookups in flight at once."""
    sem = asyncio.Semaphore(workers)

    async def _one(c: dict) -> dict:
        async with sem:
            return {**c, **await fetch_hf_info(client, c["hf_id"])}

    return await asyncio.gather(*(_one(c) for c in candidates))


def _modality_short(modality: str) -> str:
    if modality == "text->text":
        return "text"
    if "image" in modality:
        return "txt+img"
    if "audio" in modality:
        return "txt+aud"
    return modality[:7] or "?"


# Suffixes that are HF naming conventions but absent from arena slugs — dropped so a candidate
# like `gemma-4-31b-it` meets arena's `gemma-4-31b`. NOT including `-thinking`/`-instruct-variant`
# style tags that arena tracks as distinct entries.
_ARENA_DROP_SUFFIXES = ("-it", "-instruct", "-chat", "-base")


def _arena_key(name: str) -> str:
    """Loose match key shared by arena `model_name` and candidate base names: lowercase, drop a
    trailing HF suffix, strip all non-alphanumerics. `gemma-4-31b-it` and `gemma-4-31b` -> `gemma431b`.
    """
    n = name.split("/")[-1].lower()
    for suffix in _ARENA_DROP_SUFFIXES:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
            break
    return re.sub(r"[^a-z0-9]", "", n)


def _is_open_license(license_str: str) -> bool:
    """Arena `license` is open unless explicitly proprietary (Llama/Gemma community licenses count)."""
    return "proprietary" not in (license_str or "").lower()


async def fetch_arena_overall(client: httpx.AsyncClient, page: int = 100) -> list[dict]:
    """The LMArena `overall` text ranking (~360 rows) — the contiguous front block of text/latest."""
    out: list[dict] = []
    offset = 0
    while True:
        resp = await client.get(
            ARENA_ROWS_URL,
            params={"dataset": ARENA_DATASET, "config": "text", "split": "latest", "offset": offset, "length": page},
        )
        resp.raise_for_status()
        rows = [r["row"] for r in resp.json().get("rows", [])]
        if not rows:
            break
        overall = [r for r in rows if r.get("category") == "overall"]
        out.extend(overall)
        if len(overall) < len(rows):  # hit the boundary into the next category
            break
        offset += page
    return out


def build_arena_index(arena_rows: list[dict]) -> dict[str, dict]:
    """Map `_arena_key` -> best (lowest-rank) arena entry."""
    index: dict[str, dict] = {}
    for r in arena_rows:
        key = _arena_key(r.get("model_name", ""))
        entry = {
            "key": key,
            "arena_name": r.get("model_name", ""),
            "elo": round(r.get("rating") or 0),
            "arena_rank": int(r.get("rank") or 0),
            "license": r.get("license", ""),
        }
        if key not in index or entry["arena_rank"] < index[key]["arena_rank"]:
            index[key] = entry
    return index


def print_table(
    rows: list[dict],
    show_supported_col: bool,
    arena: bool = False,
    misses: list[dict] | None = None,
    other_open_count: int = 0,
) -> None:
    found = [r for r in rows if r.get("on_hf")]
    missing = [r for r in rows if r.get("status") in ("missing", "error")]
    unverified = [r for r in rows if r.get("status") == "unverified"]

    sup = f" {'sup':<3}" if show_supported_col else ""
    are = f" {'elo':>4} {'rnk':>4}" if arena else ""
    header = f"{'created':<10} {'dl':>10} {'likes':>6} {'trend':>6} {'mod':<7}{sup}{are} {'model'}"
    print(header)
    print("-" * len(header))
    for r in found:
        date = (r.get("created_at") or "")[:10] or "?"
        s = f" {'yes' if r.get('supported') else '—':<3}" if show_supported_col else ""
        a = f" {r.get('elo', '—'):>4} {r.get('arena_rank', '—'):>4}" if arena else ""
        print(
            f"{date:<10} {r.get('downloads', 0):>10} {r.get('likes', 0):>6} "
            f"{r.get('trending', 0):>6} {_modality_short(r.get('modality', '')):<7}{s}{a} {r['hf_id']}"
        )
    print(f"\n{len(found)} model(s) shown.")

    if missing:
        print(f"\n=== NOT ON HF ({len(missing)}) — stale/private/gated OpenRouter mapping ===")
        for r in missing:
            print(f"  {r['hf_id']:<55} (openrouter: {r['openrouter_id']}; {r.get('error', '')})")
    if unverified:
        print(f"\n=== COULD NOT VERIFY ({len(unverified)}) — transient (rate-limit/5xx); re-run ===")
        for r in unverified:
            print(f"  {r['hf_id']:<55} ({r.get('error', '')})")
    if arena:
        if misses:
            print(f"\n=== LIKELY FUZZY-MATCH MISSES ({len(misses)}) — open arena entry resembling an unlinked candidate ===")
            for e in misses:
                print(f"  rank {e['arena_rank']:>3}  elo {e['elo']:>4}  {e['license']:<14} {e['arena_name']}")
        else:
            print("\nNo likely fuzzy-match misses among shown candidates.")
        if other_open_count:
            print(f"({other_open_count} more open arena models outside the OpenRouter/date window — use --json for the full board.)")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_since = (dt.datetime.now(dt.UTC) - dt.timedelta(days=90)).date().isoformat()
    parser.add_argument(
        "--since",
        default=default_since,
        help=f"Keep models with HF createdAt >= this date (YYYY-MM-DD). Use 'any' to disable. Default: {default_since} (~3 months ago).",
    )
    parser.add_argument("--include-supported", action="store_true", help="Keep models deplodock already supports.")
    parser.add_argument("--text-only", action="store_true", help="Drop multimodal models (keep text->text only).")
    parser.add_argument("--min-downloads", type=int, default=0, help="Drop models below this HF download count.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent HF lookups (default 8).")
    parser.add_argument(
        "--arena",
        action="store_true",
        help="Annotate with LMArena Elo/rank (HF dataset, keyless) and list open arena models we couldn't link.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON rows instead of a table.")
    args = parser.parse_args()

    since = None if args.since.lower() == "any" else _created_date(args.since)
    if args.since.lower() != "any" and since is None:
        parser.error(f"--since must be YYYY-MM-DD or 'any', got {args.since!r}")

    supported = supported_base_keys(ROOT / "recipes")
    logger.info("Supported base models in recipes/: %d", len(supported))

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        catalog = await fetch_openrouter_catalog(client)
        candidates = open_source_candidates(catalog)
        logger.info("OpenRouter: %d models, %d open-source (unique HF repos)", len(catalog), len(candidates))

        # Cheap OpenRouter-only filters before spending HF requests.
        for c in candidates:
            c["supported"] = _base_key(c["hf_id"]) in supported
        if not args.include_supported:
            candidates = [c for c in candidates if not c["supported"]]
        if args.text_only:
            candidates = [c for c in candidates if c.get("input_modalities") == ["text"]]
        logger.info("After supported/modality filters: %d candidates; fetching HF metadata...", len(candidates))

        rows = await build_rows(client, candidates, args.workers)

        arena_index = None
        if args.arena:
            arena_index = build_arena_index(await fetch_arena_overall(client))
            logger.info("LMArena: %d ranked models in 'overall'", len(arena_index))

    # HF-dependent filters.
    if since is not None:
        rows = [r for r in rows if not r.get("on_hf") or (_created_date(r.get("created_at", "")) or dt.date.min) >= since]
    if args.min_downloads:
        rows = [r for r in rows if not r.get("on_hf") or r.get("downloads", 0) >= args.min_downloads]

    rows.sort(key=lambda r: (r.get("on_hf", False), r.get("trending", 0), r.get("downloads", 0)), reverse=True)

    # Arena annotation: link each shown candidate to its Elo. The tail focuses on *likely misses* —
    # open arena entries that resemble a shown candidate we FAILED to link (the fuzzy-match gap the
    # tail exists to catch) — and reports a count for the rest of the open board (full set in --json).
    unmatched_open = misses = None
    other_open_count = 0
    if arena_index is not None:
        matched_keys = set()
        for r in rows:
            entry = arena_index.get(_arena_key(_base_key(r["hf_id"])))
            if entry:
                r["elo"], r["arena_rank"], r["arena_name"] = entry["elo"], entry["arena_rank"], entry["arena_name"]
                matched_keys.add(entry["key"])
        unmatched_open = sorted(
            (e for k, e in arena_index.items() if k not in matched_keys and _is_open_license(e["license"])),
            key=lambda e: e["arena_rank"],
        )
        blank = [_arena_key(_base_key(r["hf_id"])) for r in rows if r.get("on_hf") and "elo" not in r]
        blank = [k for k in blank if len(k) >= 6]
        misses = [e for e in unmatched_open if any(k in e["key"] or e["key"] in k for k in blank)]
        other_open_count = len(unmatched_open) - len(misses)

    if args.json:
        payload = {"models": rows, "unmatched_open_arena": unmatched_open} if args.arena else rows
        print(json.dumps(payload, indent=2))
    else:
        print_table(rows, args.include_supported, arena=args.arena, misses=misses, other_open_count=other_open_count)


if __name__ == "__main__":
    asyncio.run(main())
