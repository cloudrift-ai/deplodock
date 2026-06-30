"""Per-primitive perf comparisons (matmul / rmsnorm / softmax / silu+mul).

Each test compiles a tiny single-op ``Graph`` and times it against the
PyTorch eager equivalent at the same shape. The ``bench_pair`` fixture
records the result; ``pytest_terminal_summary`` prints the table.

These tests do not assert on ratios — the perf suite tracks performance,
it doesn't gate on it. Failures here mean the compile or run path
errored, not that emmy is slow.
"""

from __future__ import annotations

import pytest

from tests.compiler.conftest import requires_cuda
from tests.perf.cases import PRIMITIVE_CASES, Case

pytestmark = [pytest.mark.perf, requires_cuda]


@pytest.mark.parametrize("case", PRIMITIVE_CASES, ids=lambda c: c.name)
def test_primitive_perf(case: Case, bench_pair):
    bench_pair(case)
