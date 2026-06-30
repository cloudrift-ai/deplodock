"""nvcc compile path: env-driven flags + cache/context partitioning by opt level.

These are hermetic (no GPU, no actual nvcc invocation) — they exercise the flag
resolution, the cubin cache key, the Context perf-cache key, and the CLI
override precedence, all of which gate whether -O1-tuned and -O3-run results
stay separate in the DB.
"""

from __future__ import annotations

import argparse

from emmy.commands.compile import apply_nvcc_flags
from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.context import Context


def test_effective_flags_reads_env(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    assert nvcc.effective_flags() == ["--use_fast_math"]
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")
    assert nvcc.effective_flags() == ["--use_fast_math", "-Xcicc", "-O1"]


def test_cubin_cache_key_partitions_by_flags(monkeypatch) -> None:
    """Same source compiled at different opt levels must key to different
    cubins — otherwise an -O1 sweep would serve its cubin to an -O3 run."""
    monkeypatch.setattr(nvcc, "_toolkit_tag", lambda: "tag")  # avoid the nvcc --version subprocess
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    k_o3 = nvcc._cache_key("src", "k", "sm_80")
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")
    k_o1 = nvcc._cache_key("src", "k", "sm_80")
    assert k_o3 != k_o1


def test_context_key_partitions_by_flags(monkeypatch) -> None:
    """The perf cache key (Context.structural_key) must differ by opt level so
    -O1-measured latencies never clobber -O3 ones."""
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "")
    k_o3 = Context.from_target((8, 0)).structural_key()
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-Xcicc -O1")
    k_o1 = Context.from_target((8, 0)).structural_key()
    assert k_o3 != k_o1
    # Same flags ⇒ same key (cache hit across runs).
    assert Context.from_target((8, 0)).structural_key() == k_o1


def test_apply_nvcc_flags_precedence(monkeypatch) -> None:
    # default applies when neither CLI flag nor env is set
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags=None), default="-Xcicc -O1") == "-Xcicc -O1"

    # CLI override wins over the default
    monkeypatch.delenv("EMMY_NVCC_FLAGS", raising=False)
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags="-Xcicc -O3"), default="-Xcicc -O1") == "-Xcicc -O3"

    # a pre-set env var is respected over the command default (but CLI still wins)
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "preset")
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags=None), default="-Xcicc -O1") == "preset"
    assert apply_nvcc_flags(argparse.Namespace(nvcc_flags="cliwins"), default="-Xcicc -O1") == "cliwins"
