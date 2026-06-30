"""Fused-kernel perf comparisons (currently SDPA).

The Emmy graph contains the high-level frontend op (``SdpaOp``);
the compiler decomposes + fuses it however it can. The ``launches``
column in the summary table records how many kernels Emmy emitted —
launches=1 means full fusion into a single kernel, launches>1 means
the chain was emitted as multiple launches. The PyTorch reference uses
``F.scaled_dot_product_attention``.
"""

from __future__ import annotations

import pytest

from tests.compiler.conftest import requires_cuda
from tests.perf.cases import FUSED_CASES, Case

pytestmark = [pytest.mark.perf, requires_cuda]


@pytest.mark.parametrize("case", FUSED_CASES, ids=lambda c: c.name)
def test_fused_perf(case: Case, bench_pair):
    bench_pair(case)
