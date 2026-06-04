"""Tests for the PhaseTimer collector and timing renderer."""

import asyncio

import pytest

from deplodock.timing import (
    PHASE_CUDA_GRAPH,
    PHASE_IMAGE_PULL,
    PHASE_MODEL_LOAD_AND_WARMUP,
    PHASE_TORCH_COMPILE,
    PHASE_VM_PROVISION,
    PHASE_WEIGHTS_LOAD,
    PhaseTimer,
    format_timing,
)


def test_record_accumulates():
    t = PhaseTimer()
    t.record(PHASE_VM_PROVISION, 10.0, log=False)
    t.record(PHASE_VM_PROVISION, 5.0, log=False)
    assert t.phases[PHASE_VM_PROVISION] == 15.0


def test_measure_records_elapsed():
    t = PhaseTimer()
    with t.measure(PHASE_IMAGE_PULL):
        pass
    assert PHASE_IMAGE_PULL in t.phases
    assert t.phases[PHASE_IMAGE_PULL] >= 0.0


def test_ameasure_records_elapsed():
    t = PhaseTimer()

    async def go():
        async with t.ameasure(PHASE_IMAGE_PULL):
            await asyncio.sleep(0)

    asyncio.run(go())
    assert PHASE_IMAGE_PULL in t.phases


def test_measure_records_on_exception():
    t = PhaseTimer()
    with pytest.raises(ValueError):
        with t.measure(PHASE_IMAGE_PULL):
            raise ValueError("boom")
    # The phase is still recorded despite the exception.
    assert PHASE_IMAGE_PULL in t.phases


def test_ameasure_records_on_exception():
    t = PhaseTimer()

    async def go():
        async with t.ameasure(PHASE_IMAGE_PULL):
            raise ValueError("boom")

    with pytest.raises(ValueError):
        asyncio.run(go())
    assert PHASE_IMAGE_PULL in t.phases


def test_total_excludes_subphases():
    t = PhaseTimer()
    t.record(PHASE_MODEL_LOAD_AND_WARMUP, 70.0, log=False)
    t.record(PHASE_WEIGHTS_LOAD, 11.0, log=False)
    t.record(PHASE_TORCH_COMPILE, 20.0, log=False)
    t.record(PHASE_CUDA_GRAPH, 18.0, log=False)
    t.record(PHASE_IMAGE_PULL, 30.0, log=False)
    # weights_load + torch_compile + cuda_graph_capture are a breakdown of
    # model_load_and_warmup, so they must not inflate the total.
    assert t.total() == 100.0


def test_as_dict_rounds_and_appends_total():
    t = PhaseTimer()
    t.record(PHASE_IMAGE_PULL, 1.239, log=False)
    t.record(PHASE_VM_PROVISION, 2.5, log=False)
    d = t.as_dict()
    assert d[PHASE_IMAGE_PULL] == 1.24
    assert d["total"] == 3.74


def test_format_timing_indents_subphases_and_ends_with_total():
    timing = {
        PHASE_VM_PROVISION: 142.5,
        PHASE_MODEL_LOAD_AND_WARMUP: 73.1,
        PHASE_WEIGHTS_LOAD: 11.4,
        PHASE_CUDA_GRAPH: 18.9,
        "total": 234.5,
    }
    out = format_timing(timing)
    lines = out.splitlines()
    # Sub-phases are indented under model_load_and_warmup.
    assert any(line.startswith("  weights_load") for line in lines)
    assert any(line.startswith("  cuda_graph_capture") for line in lines)
    # total is rendered last.
    assert lines[-1].startswith("total")


def test_format_timing_empty():
    assert format_timing({}) == ""
