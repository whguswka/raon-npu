# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""NPU-free timing-boundary regression tests."""

from __future__ import annotations

import numpy as np

from kernels.gemv_unified.shapes import ALL_SHAPES as UNIFIED_SHAPES, M_PER_CHUNK
from kernels.moe.pack import (
    B1_M_PER_EXPERT,
    B2_M_PER_EXPERT,
    NUM_ACTIVE_ROUTED,
    NUM_ACTIVE_TOTAL,
)
from kernels.moe_grouped.layout import B1, B2, SLOTS, TMAX
from npu import backend as backend_module
from .data import Record
from . import runtime as runtime_module
from .runtime import GroupedCallTiming, ReplayRuntime


def _record() -> Record:
    return Record(
        0, 0, np.zeros(2560, dtype=np.uint16),
        np.arange(6, dtype=np.int64), np.ones(6, dtype=np.float32),
        np.zeros(2176, dtype=np.uint16), np.zeros(2560, dtype=np.uint16),
        np.zeros(2560, dtype=np.uint16),
    )


def test_grouped_call_separates_write_run_and_read_time(monkeypatch) -> None:
    clock = [0.0]

    class Backend:
        def _get_prefill_weight_tensor(self, _name):
            return object()

        def _overwrite_buffer(self, _target, _values):
            clock[0] += 2.0

        def _record_xclbin_use(self, _name):
            pass

    class Output:
        def numpy(self):
            clock[0] += 5.0
            return np.zeros(SLOTS * TMAX * B1.m, dtype=np.uint16)

    def run(*_args, **_kwargs):
        clock[0] += 3.0

    runtime = ReplayRuntime.__new__(ReplayRuntime)
    runtime.backend = Backend()
    runtime._grouped = {B1.name: (run, object(), Output())}
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: clock[0])
    output, timing = runtime._run_grouped_call(
        B1, 0, (0,) * SLOTS,
        np.zeros((SLOTS, TMAX, B1.k), dtype=np.uint16),
    )
    assert output.shape == (SLOTS, TMAX, B1.m)
    assert timing == GroupedCallTiming(2.0, 3.0, 5.0)


def test_grouped_call_writes_x_in_device_tiled_order(monkeypatch) -> None:
    captured: list[np.ndarray] = []

    class Backend:
        def _get_prefill_weight_tensor(self, _name):
            return object()

        def _overwrite_buffer(self, _target, values):
            captured.append(np.asarray(values).view(np.uint16).copy())

        def _record_xclbin_use(self, _name):
            pass

    class Output:
        def numpy(self):
            return np.zeros(SLOTS * TMAX * B1.m, dtype=np.uint16)

    def run(*_args, **_kwargs):
        pass

    values = np.arange(8 * 8 * B1.k, dtype=np.uint16).reshape(SLOTS, TMAX, B1.k)
    runtime = ReplayRuntime.__new__(ReplayRuntime)
    runtime.backend = Backend()
    runtime._grouped = {B1.name: (run, object(), Output())}
    runtime._run_grouped_call(B1, 0, (0,) * SLOTS, values)

    assert len(captured) == 1
    expected = backend_module._grouped_x_tiled(values, B1).reshape(-1)
    np.testing.assert_array_equal(captured[0], expected)


def test_grouped_total_includes_both_scatters_and_call_boundaries(monkeypatch) -> None:
    clock = [0.0]
    original_scatter_b1 = runtime_module.scatter_b1
    original_scatter_b2 = runtime_module.scatter_b2

    def scatter_b1(*args, **kwargs):
        clock[0] += 7.0
        return original_scatter_b1(*args, **kwargs)

    def scatter_b2(*args, **kwargs):
        clock[0] += 11.0
        return original_scatter_b2(*args, **kwargs)

    runtime = ReplayRuntime.__new__(ReplayRuntime)

    def grouped_call(shape, _layer, _experts, _values, rows=None):
        if shape is B1:
            assert rows is not None and len(rows) == SLOTS
        else:
            assert rows is None
        output = np.zeros((SLOTS, TMAX, shape.m), dtype=np.uint16)
        return output, GroupedCallTiming(2.0, 3.0, 5.0)

    runtime._run_grouped_call = grouped_call
    monkeypatch.setattr(runtime_module, "scatter_b1", scatter_b1)
    monkeypatch.setattr(runtime_module, "scatter_b2", scatter_b2)
    monkeypatch.setattr(runtime_module.time, "perf_counter", lambda: clock[0])
    result = runtime.grouped([_record()])
    assert result.calls == 2
    assert result.timing.input_write_s == 4.0
    assert result.timing.npu_s == 6.0
    assert result.timing.output_read_s == 10.0
    assert result.timing.scatter_s == 18.0
    assert result.timing.total_s == 38.0


class _FakeTensor:
    """Prepared-buffer stand-in holding the uint16 bits of bfloat16 values."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.uint16)

    def numpy(self):
        return self._buf

    def overwrite(self, values) -> None:
        self._buf = np.asarray(values).view(np.uint16).reshape(-1).copy()


class _FakeDecodeBackend:
    """Records weight lookups and xclbin uses; every buffer starts empty."""

    def __init__(self) -> None:
        self._buffers = {
            name: _FakeTensor()
            for name in ("moe_x_gu", "moe_y_gu", "moe_x_dn", "moe_y_dn")
        }
        self.weight_names: list[str] = []
        self.uses: list[str] = []

    def _prepared_buffer(self, name: str) -> _FakeTensor:
        return self._buffers[name]

    def _get_weight_tensor(self, name: str):
        self.weight_names.append(name)
        return object()

    def _overwrite_buffer(self, target: _FakeTensor, values) -> None:
        target.overwrite(values)

    def _record_xclbin_use(self, name: str) -> None:
        self.uses.append(name)


def _fake_decode_op(shape, op_name: str, calls: list):
    """Stand-in for one unified decode GEMV with that operator's real geometry.

    Output expert row j replays template input row j; B1's single input row
    feeds every routed-expert row because the reference passes one expert in
    each scalar slot. Each recorded ``(name, rows, scalars, stride)`` tuple
    lets tests pin the template against the NUM_ACTIVE_* constants.
    """
    unified = next(s for s in UNIFIED_SHAPES if s.name == op_name)
    rows = unified.M // shape.m

    def run(y, _weight, x, **scalars):
        packed = np.asarray(x.numpy()).view(np.uint16)
        stride = packed.size // unified.x_slots
        x_rows = packed.reshape(unified.x_slots, stride)
        width = min(M_PER_CHUNK, unified.K)
        out = np.zeros((rows, shape.m), dtype=np.uint16)
        out[:, :width] = x_rows[:, :width]
        calls.append((unified.name, out.shape[0], dict(scalars), stride))
        y.overwrite(out)

    return run


def _reference_case(shape, ids: list[int]):
    """Seed each grouped slot with one id; slot s row r holds distinct bits.

    Only B2 maps its shared id (``shape.max_expert``) to scalar 0, so the
    B1 cases below deliberately avoid an id equal to B1.max_expert.
    """
    assert len(ids) == SLOTS
    pattern = (
        np.arange(SLOTS, dtype=np.uint16)[:, None, None] * 7
        + np.arange(TMAX, dtype=np.uint16)[None, :, None] * 3
        + np.arange(5, dtype=np.uint16)[None, None, :]
        + 1
    )
    counts = [int(slot % 4) for slot in range(SLOTS)]
    values = np.zeros((SLOTS, TMAX, shape.k), dtype=np.uint16)
    expected = np.zeros((SLOTS, TMAX, shape.m), dtype=np.uint16)
    for slot, count in enumerate(counts):
        values[slot, :count, :5] = pattern[slot, :count]
        expected[slot, :count, :5] = pattern[slot, :count]
    expected_calls = [
        {
            f"e{i}": 0 if eid == shape.max_expert else int(eid)
            for i in range(NUM_ACTIVE_ROUTED)
        }
        for eid, count in zip(ids, counts)
        for _ in range(count)
    ]
    return (
        tuple(int(eid) for eid in ids), values, tuple(counts),
        expected, expected_calls,
    )


def test_decode_reference_b1_replays_routed_template_rows() -> None:
    """The B1 reference replays each valid row through the 6-row template."""
    calls: list = []
    backend = _FakeDecodeBackend()
    runtime = ReplayRuntime.__new__(ReplayRuntime)
    runtime.backend = backend
    runtime.run_gu = _fake_decode_op(B1, "moe_b1_6", calls)
    experts, values, valid_rows, expected, expected_calls = _reference_case(
        B1, [3, 0, 1, 2, 4, 5, 6, 7])
    output = runtime._run_decode_reference(B1, 0, experts, values, valid_rows)
    assert output.shape == (SLOTS, TMAX, B1.m)
    assert output.dtype == np.uint16
    assert np.array_equal(output, expected)
    assert len(calls) == len(expected_calls) == 12
    for name, rows, scalars, stride in calls:
        assert name == "moe_b1_6"
        assert rows == NUM_ACTIVE_ROUTED
        assert stride > B1.k
    assert [scalars for _, _, scalars, _ in calls] == expected_calls
    assert backend.weight_names == ["L00/moe_b1"]
    assert backend.uses == ["gemv_unified"] * len(calls)


def test_decode_reference_b2_selects_routed_and_shared_template_rows() -> None:
    """Expert 128 replays from B2's last template row, routed experts from 0."""
    calls: list = []
    backend = _FakeDecodeBackend()
    runtime = ReplayRuntime.__new__(ReplayRuntime)
    runtime.backend = backend
    runtime.run_dn = _fake_decode_op(B2, "moe_b2_9", calls)
    experts, values, valid_rows, expected, expected_calls = _reference_case(
        B2, [126, 127, 128, 3, 0, 44, 128, 2])
    output = runtime._run_decode_reference(B2, 12, experts, values, valid_rows)
    assert output.shape == (SLOTS, TMAX, B2.m)
    assert np.array_equal(output, expected)
    assert len(calls) == len(expected_calls) == sum(valid_rows) == 12
    for name, rows, scalars, stride in calls:
        assert name == "moe_b2_9"
        assert rows == NUM_ACTIVE_TOTAL
        assert stride > B2.k
    assert [scalars for _, _, scalars, _ in calls] == expected_calls
    shared_calls = [
        scalars for _, _, scalars, _ in calls
        if all(value == 0 for value in scalars.values())
    ]
    assert len(shared_calls) == 4  # slots 2 and 6 each hold two valid rows
    assert backend.weight_names == ["L12/moe_b2"]
    assert backend.uses == ["gemv_unified"] * len(calls)


def test_decode_template_geometry_is_independent_of_group_slots() -> None:
    """Decode rows, input slots, and scalar counts come from NUM_ACTIVE_*."""
    assert SLOTS != NUM_ACTIVE_TOTAL
    assert NUM_ACTIVE_TOTAL == NUM_ACTIVE_ROUTED + 1
    unified = {shape.name: shape for shape in UNIFIED_SHAPES}
    assert unified["moe_b1_6"].M == NUM_ACTIVE_ROUTED * B1_M_PER_EXPERT
    assert unified["moe_b2_9"].M == NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT
    assert unified["moe_b2_9"].x_slots == NUM_ACTIVE_TOTAL


def test_grouped_call_passes_row_scalars_when_rows_provided() -> None:
    seen: list[dict[str, int]] = []

    class Backend:
        def _get_prefill_weight_tensor(self, _name):
            return object()

        def _overwrite_buffer(self, _target, _values):
            pass

        def _record_xclbin_use(self, _name):
            pass

    class Output:
        def numpy(self):
            return np.zeros(SLOTS * TMAX * B1.m, dtype=np.uint16)

    def run(*_args, **scalars):
        seen.append(dict(scalars))

    runtime = ReplayRuntime.__new__(ReplayRuntime)
    runtime.backend = Backend()
    runtime._grouped = {B1.name: (run, object(), Output())}
    experts = tuple(range(SLOTS))
    rows = (8, 4, 2, 1, 0, 3, 7, 5)
    values = np.zeros((SLOTS, TMAX, B1.k), dtype=np.uint16)

    runtime._run_grouped_call(B1, 0, experts, values, rows=rows)
    assert seen[-1] == {
        **{f"e{i}": i for i in range(SLOTS)},
        **{f"r{i}": r for i, r in enumerate(rows)},
    }

    runtime._run_grouped_call(B1, 0, experts, values)
    assert seen[-1] == {f"e{i}": i for i in range(SLOTS)}
