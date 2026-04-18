"""Shared pytest fixtures for all test modules."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RECIPES_DIR = os.path.join(PROJECT_ROOT, "recipes")

# ── LPT static bucketing for pytest-xdist ───────────────────────────
# Record per-test call durations in the pytest cache; next run partitions
# items across N worker buckets via LPT (longest-processing-time-first)
# greedy — each item goes to the currently-lightest bucket. Buckets are
# tagged via @pytest.mark.xdist_group so `--dist=loadgroup` routes every
# item in a bucket to the same worker. Theoretical makespan is the load
# of the heaviest bucket (lower bound = longest single test).

_DURATIONS_KEY = "test_durations/call"
_CALL_DURATIONS: dict[str, float] = {}


def pytest_runtest_logreport(report):
    if report.when == "call":
        _CALL_DURATIONS[report.nodeid] = report.duration


def pytest_sessionfinish(session):
    cache = getattr(session.config, "cache", None)
    if cache is None or not _CALL_DURATIONS:
        return
    existing = cache.get(_DURATIONS_KEY, {}) or {}
    existing.update(_CALL_DURATIONS)
    cache.set(_DURATIONS_KEY, existing)


def _num_workers(config) -> int | None:
    """Mirror xdist's -n resolution: int, 'auto', 'logical', or None."""
    try:
        n = config.getoption("numprocesses", None)
    except ValueError:
        return None
    if n in (None, 0):
        return None
    if isinstance(n, int):
        return n if n >= 1 else None
    if n in ("auto", "logical"):
        return os.cpu_count() or 1
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def pytest_collection_modifyitems(config, items):
    import heapq

    cache = getattr(config, "cache", None)
    if cache is None:
        return
    durations = cache.get(_DURATIONS_KEY, {}) or {}
    if not durations:
        return
    nworkers = _num_workers(config)
    if nworkers is None or nworkers < 2:
        return

    known = sorted(durations[it.nodeid] for it in items if it.nodeid in durations)
    fallback = known[len(known) // 2] if known else 0.0

    def dur(item) -> float:
        return durations.get(item.nodeid, fallback)

    sorted_items = sorted(items, key=dur, reverse=True)

    # LPT: pop the lightest bucket, add this item, push back.
    buckets: list[tuple[float, int, list]] = [(0.0, w, []) for w in range(nworkers)]
    heapq.heapify(buckets)
    for it in sorted_items:
        load, wid, bucket = heapq.heappop(buckets)
        bucket.append(it)
        heapq.heappush(buckets, (load + dur(it), wid, bucket))

    # Tag items with their bucket's xdist_group so loadgroup routes them together.
    # Reorder items so same-bucket tests are contiguous and heaviest bucket leads
    # (helps xdist dispatch long work first).
    buckets_sorted = sorted(buckets, key=lambda b: -b[0])
    reordered: list = []
    for _load, wid, bucket in buckets_sorted:
        group = f"w{wid}"
        for it in bucket:
            it.add_marker(pytest.mark.xdist_group(group))
            reordered.append(it)
    items[:] = reordered


@pytest.fixture(scope="session")
def project_root():
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def recipes_dir():
    """Absolute path to the recipes/ directory."""
    return RECIPES_DIR


@pytest.fixture(scope="session")
def run_cli(project_root):
    """Return a callable that invokes the deplodock CLI as a subprocess."""

    def _run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "deplodock.deplodock", *args],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        return result.returncode, result.stdout, result.stderr

    return _run


@pytest.fixture
def make_bench_config(recipes_dir):
    """Return a factory that writes a temporary bench config.yaml."""

    def _make(tmp_dir):
        config = {
            "benchmark": {
                "local_results_dir": os.path.join(str(tmp_dir), "results"),
                "model_dir": "/hf_models",
            },
        }
        config_path = os.path.join(str(tmp_dir), "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    return _make


# ── Compiler dump fixture ──────────────────────────────────────────


@pytest.fixture
def dump_dir(request):
    """Dump compilation artifacts to _test_data/<test_name>/ for manual inspection."""
    safe_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    dump_path = Path(PROJECT_ROOT) / "_test_data" / safe_name

    from deplodock.compiler.dump import CompilerDump

    return CompilerDump(dir=dump_path)


# ── Unit-test fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_recipe_dir(tmp_path):
    """Create a temp directory with a sample recipe.yaml using matrices format."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
            {
                "deploy.gpu": "NVIDIA H200 141GB",
                "deploy.gpu_count": 8,
                "engine.llm.tensor_parallel_size": 8,
                "engine.llm.context_length": 16384,
                "engine.llm.vllm.extra_args": "--kv-cache-dtype fp8",
                "benchmark.random_input_len": 8000,
                "benchmark.random_output_len": 8000,
            },
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 4,
                "engine.llm.tensor_parallel_size": 4,
                "engine.llm.vllm.extra_args": "--kv-cache-dtype fp8",
            },
        ],
    }

    recipe_path = tmp_path / "recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    return str(tmp_path)


@pytest.fixture
def sample_config():
    """Return a resolved config dict for testing compose generation."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
    }


@pytest.fixture
def sample_config_sglang():
    """Return a resolved config dict for SGLang compose generation."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "sglang": {
                    "image": "lmsysorg/sglang:v0.5.9",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
    }


@pytest.fixture
def sample_config_multi():
    """Return a resolved config dict for multi-instance testing."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 16384,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "_num_instances": 2,
    }
