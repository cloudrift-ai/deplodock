"""``serving.packed.split_spans`` — pure logic, no GPU/torch/vllm."""

from emmy.serving.packed import split_spans


def test_single_sequence():
    assert split_spans([0, 1, 2, 3], max_seq_len=16) == [(0, 4)]


def test_multiple_sequences_split_at_position_zero():
    # Three packed requests of lengths 3, 1, 4.
    pos = [0, 1, 2, 0, 0, 1, 2, 3]
    assert split_spans(pos, max_seq_len=16) == [(0, 3), (3, 4), (4, 8)]


def test_empty_batch():
    assert split_spans([], max_seq_len=16) == []


def test_dummy_run_nonzero_start_still_spans_from_zero():
    # Garbage profiling batch: positions never hit 0 — index 0 still opens a span.
    assert split_spans([5, 6, 7], max_seq_len=16) == [(0, 3)]


def test_overlong_span_chopped_to_max_seq_len():
    # Garbage profiling batch longer than max_seq_len is chunked, not crashed.
    pos = list(range(10))
    assert split_spans(pos, max_seq_len=4) == [(0, 4), (4, 8), (8, 10)]


def test_overlong_middle_sequence():
    pos = [0, 1, 0, 1, 2, 3, 4, 0]
    assert split_spans(pos, max_seq_len=3) == [(0, 2), (2, 5), (5, 7), (7, 8)]


def test_numpy_positions():
    import numpy as np

    pos = np.array([0, 1, 0, 1, 2])
    assert split_spans(pos, max_seq_len=16) == [(0, 2), (2, 5)]
