"""Fast CPU tests for ``gen_runner`` helpers (no GPU/model). The decode-bucket compile +
correctness are covered on GPU by ``test_gen_runner_gpu.py`` / ``test_vllm_plugin_gen_gpu.py``."""

import numpy as np

from emmy.serving.gen_runner import _pad_rows


def test_pad_rows_pads_with_zeros_and_preserves_real_rows():
    a = np.arange(6, dtype=np.float16).reshape(3, 2)
    out = _pad_rows(a, 5)
    assert out.shape == (5, 2)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out[:3], a)  # real rows intact
    assert (out[3:] == 0).all()  # padding is zeros (computed then sliced away)


def test_pad_rows_is_passthrough_when_already_at_bucket():
    a = np.ones((4, 8), dtype=np.float16)
    assert _pad_rows(a, 4) is a  # no copy when t == bucket
