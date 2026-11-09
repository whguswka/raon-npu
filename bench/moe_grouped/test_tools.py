# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""NPU-free lifecycle checks for the grouped replay command-line tools."""

from __future__ import annotations

import json
from types import SimpleNamespace
import sys

import numpy as np

from . import benchmark as benchmark_module
from . import stress as stress_module
from .benchmark import mismatch_count, stress_result_passes
from .stress import _checked_random_call
from .runtime import ReplayResult, Timing


class FakeEngine:
    instances: list["FakeEngine"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.backend = object()
        self.unloaded = False
        self.instances.append(self)

    def unload(self) -> None:
        self.unloaded = True


def _result() -> ReplayResult:
    values = np.zeros((1, 1), dtype=np.uint16)
    return ReplayResult(
        values, values, values, values, Timing(npu_s=1.0), 1, 1, 1.0,
    )


def test_mismatch_count_rejects_equal_values_with_different_dtype() -> None:
    a = np.zeros((2, 3), dtype=np.uint16)
    b = np.zeros((2, 3), dtype=np.int32)
    assert mismatch_count(a, b) > 0


def test_stability_requires_all_success_fields() -> None:
    valid = {
        "pass": True, "other_errors": 0, "timeouts": 0, "mismatches": 0,
        "calls": 10000, "random_checks": 100,
    }
    assert stress_result_passes(valid)
    for changed in ({"other_errors": 1}, {"pass": False}):
        assert not stress_result_passes({**valid, **changed})
    for missing in valid:
        assert not stress_result_passes({key: value for key, value in valid.items()
                                         if key != missing})


def test_random_call_compares_the_actual_slot_and_empty_rows() -> None:
    expected = np.zeros((8, 8, 3), dtype=np.uint16)
    corrupted = expected.copy()
    corrupted[4, 7, 1] = 1

    class FakeRuntime:
        def _run_grouped_call(self, *_args):
            return corrupted, None

        def _run_decode_reference(self, *_args):
            return expected

    inputs = np.zeros((8, 8, 2), dtype=np.uint16)
    count = _checked_random_call(
        FakeRuntime(), object(), 0, (0,) * 8, inputs, (1,) * 8
    )
    assert count > 0


def test_benchmark_unloads_before_writing_json(monkeypatch, tmp_path) -> None:
    class FakeRuntime:
        def __init__(self, _backend) -> None:
            pass

        def baseline(self, _batch, warmup=True):
            return _result()

        def grouped(self, _batch, warmup=True):
            return _result()

        def transition_cost_samples(self, _batch, _b1):
            return {
                "baseline_to_b1": 0.0,
                "b1_to_b2": 0.0,
                "b2_to_baseline": 0.0,
            }

    record = SimpleNamespace(
        layer=0,
        token=0,
        output=np.zeros((1,), dtype=np.uint16),
        shared_down=np.zeros((1,), dtype=np.uint16),
    )
    output = tmp_path / "benchmark.json"
    FakeEngine.instances.clear()
    monkeypatch.setattr(benchmark_module, "Engine", FakeEngine)
    monkeypatch.setattr(benchmark_module, "ReplayRuntime", FakeRuntime)
    monkeypatch.setattr(benchmark_module, "load_records", lambda *_args: [record])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark", "--weights-dir", str(tmp_path), "--records", str(tmp_path),
            "--batches", "64", "--out", str(output),
        ],
    )
    assert benchmark_module.main() == 1
    assert FakeEngine.instances[0].unloaded is True
    assert json.loads(output.read_text(encoding="utf-8"))["record_count"] == 1


def test_stress_unloads_before_writing_json(monkeypatch, tmp_path) -> None:
    class FakeRuntime:
        def __init__(self, _backend) -> None:
            pass

        def grouped(self, _sample):
            return SimpleNamespace(b1=np.zeros((1, 1), dtype=np.uint16))

        def _run_grouped_call(self, *_args) -> None:
            pass

    call = SimpleNamespace(expert_slots=(0,) * 8)
    plan = SimpleNamespace(calls=[call])
    record = SimpleNamespace(
        layer=0,
        token=0,
        top_ids=np.zeros((1,), dtype=np.int64),
        x=np.zeros((1,), dtype=np.uint16),
        shared_gate_up=np.zeros((1,), dtype=np.uint16),
        gate_beta=4.0,
        value_beta=25.0,
    )
    output = tmp_path / "stress.json"
    FakeEngine.instances.clear()
    monkeypatch.setattr(stress_module, "Engine", FakeEngine)
    monkeypatch.setattr(stress_module, "ReplayRuntime", FakeRuntime)
    monkeypatch.setattr(stress_module, "load_records", lambda *_args: [record])
    monkeypatch.setattr(stress_module, "build_plan", lambda *_args, **_kwargs: plan)
    monkeypatch.setattr(
        stress_module,
        "gather_inputs",
        lambda *_args: [np.zeros((8, 8, 1), dtype=np.uint16)],
    )
    monkeypatch.setattr(
        stress_module,
        "make_b2_inputs",
        lambda *_args, **_kwargs: (
            np.zeros((1, 1), dtype=np.uint16),
            np.zeros((1, 1), dtype=np.uint16),
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stress", "--weights-dir", str(tmp_path), "--records", str(tmp_path),
            "--calls", "1", "--out", str(output),
        ],
    )
    assert stress_module.main() == 0
    assert FakeEngine.instances[0].unloaded is True
    assert json.loads(output.read_text(encoding="utf-8"))["calls"] == 1
