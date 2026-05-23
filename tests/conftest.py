"""Shared pytest fixtures for all test modules."""

import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture(autouse=True)
def _seed_rng():
    """Pin RNGs for every test so numerical-tolerance assertions
    (e.g. ``test_torch_ops.test_unary``) don't flake on inputs that
    happen to land in tight regions. Determinism > tolerance — a real
    precision regression should still trip these tests.

    Also reseeds module-level ``rng = np.random.default_rng(...)``
    Generators in test modules. They're instantiated once at import,
    so successive ``rng.uniform`` calls inside parametrized tests
    drift across the session and produce order-dependent flakes
    (sigmoid/tanh/rsqrt at near-zero inputs etc.). Re-binding ``rng``
    to a fresh ``default_rng`` with the original seed restores
    intra-test determinism without changing any test's input
    distribution."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith("tests."):
            continue
        rng = getattr(mod, "rng", None)
        if isinstance(rng, np.random.Generator):
            seed = getattr(mod, "_RNG_SEED", 0)
            mod.rng = np.random.default_rng(seed)


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


def _is_cuda_item(item) -> bool:
    """True iff this test item issues CUDA work.

    Detected via either (a) a ``skipif`` marker whose reason starts with
    ``"CUDA not available"`` (the ``requires_cuda`` decorator used across
    ``tests/compiler/``), or (b) a ``[cuda...]`` callspec id (the
    ``run_graph`` fixture's third variant + every ``test_e2e_accuracy``
    parametrization). One of those signals is true for every test that
    actually touches the device today; new CUDA-using tests inherit
    routing for free as long as they reuse the conventions."""
    for mark in item.iter_markers(name="skipif"):
        reason = mark.kwargs.get("reason", "")
        if isinstance(reason, str) and reason.startswith("CUDA not available"):
            return True
    nid = item.nodeid
    return "[cuda" in nid or "-cuda-" in nid or nid.endswith("-cuda]")


# Single xdist_group used for every CUDA-touching test. The host only has
# one GPU; running CUDA tests across multiple xdist workers concurrently
# would mean two processes pushing kernels onto the same device. With
# multi-kernel attention schedules in the suite, that cross-worker GPU
# contention surfaces as small but real numerical divergence in tests
# that compare against PyTorch eager (the kernel-launch scheduling +
# fp32 reduction ordering differ from the isolated-run baseline). Pinning
# all CUDA tests to one group makes them run sequentially on one worker;
# non-CUDA tests still parallelize via the LPT buckets below.
_CUDA_GROUP = "cuda"


def pytest_collection_modifyitems(config, items):
    import heapq

    # Step 1: pin every CUDA-touching item to a single xdist_group so
    # they all land on one worker and run sequentially. Skip the LPT
    # bucketing for those items entirely — they're already grouped.
    cuda_items: list = []
    other_items: list = []
    for it in items:
        if _is_cuda_item(it):
            it.add_marker(pytest.mark.xdist_group(_CUDA_GROUP))
            cuda_items.append(it)
        else:
            other_items.append(it)

    cache = getattr(config, "cache", None)
    if cache is None:
        items[:] = cuda_items + other_items
        return
    durations = cache.get(_DURATIONS_KEY, {}) or {}
    nworkers = _num_workers(config)
    if not durations or nworkers is None or nworkers < 2:
        items[:] = cuda_items + other_items
        return

    known = sorted(durations[it.nodeid] for it in other_items if it.nodeid in durations)
    fallback = known[len(known) // 2] if known else 0.0

    def dur(item) -> float:
        return durations.get(item.nodeid, fallback)

    sorted_others = sorted(other_items, key=dur, reverse=True)

    # Reserve one worker for the CUDA group; LPT-bucket the rest across
    # the remaining workers. With nworkers == 1 we fall back to a single
    # bucket (no-op grouping). Sum CUDA-item durations into the CUDA
    # bucket so it competes for ordering with the other heavy buckets.
    cuda_load = sum(dur(it) for it in cuda_items)
    other_workers = max(1, nworkers - 1)

    # LPT: pop the lightest bucket, add this item, push back.
    buckets: list[tuple[float, int, list]] = [(0.0, w, []) for w in range(other_workers)]
    heapq.heapify(buckets)
    for it in sorted_others:
        load, wid, bucket = heapq.heappop(buckets)
        bucket.append(it)
        heapq.heappush(buckets, (load + dur(it), wid, bucket))

    # Tag non-CUDA items with their bucket's xdist_group so loadgroup
    # routes them together. CUDA items keep their pre-applied ``cuda``
    # group from step 1.
    buckets_sorted = sorted(buckets, key=lambda b: -b[0])
    reordered: list = []
    for _load, wid, bucket in buckets_sorted:
        group = f"w{wid}"
        for it in bucket:
            it.add_marker(pytest.mark.xdist_group(group))
            reordered.append(it)
    # Put CUDA bucket first when it dominates load, otherwise interleave
    # with the largest non-CUDA bucket. Heaviest-first dispatch lets xdist
    # start the longest serial chain immediately.
    if cuda_load >= buckets_sorted[0][0]:
        items[:] = cuda_items + reordered
    else:
        items[:] = reordered + cuda_items


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

    from deplodock.compiler.pipeline.dump import CompilerDump

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
