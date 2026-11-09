# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""NPU-free contract tests for grouped prefill MoE overlap execution."""

from __future__ import annotations

from contextlib import contextmanager
import sys
from types import MethodType, SimpleNamespace
import time

import numpy as np
import pytest

from host.weights import f32_to_bf16_rtne
from kernels.moe_grouped.planner import build_plan
from npu import backend as backend_module
from npu.backend import NpuBackend, _GroupedDirectRunner
from npu.test_prefill_grouped_backend import LAYER, _inputs, lightweight_backend


@pytest.mark.parametrize("tokens", [1, 9, 17, 63])
def test_mock_overlap_matches_disabled_path_byte_for_byte(
    lightweight_backend, tokens
) -> None:
    backend = lightweight_backend
    x, ids, weights, shared_gu = _inputs(tokens)
    backend.prefill_moe_overlap = False
    expected = backend.moe_experts_grouped(LAYER, x, ids, weights, shared_gu)
    disabled_counts = dict(backend.call_counts)
    assert disabled_counts.get("moe_pipe_wait", 0) == 0

    backend.reset_timings()
    backend.prefill_moe_overlap = True
    actual = backend.moe_experts_grouped(LAYER, x, ids, weights, shared_gu)
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    assert LAYER not in backend._cached_shared
    assert backend.call_counts["moe_pipe_wait"] == (
        len(build_plan(ids).calls) + len(build_plan(ids, include_shared=True).calls)
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_shared_mismatch_fails_before_first_grouped_call(
    lightweight_backend, monkeypatch, enabled
) -> None:
    backend = lightweight_backend
    backend.prefill_moe_overlap = enabled
    x, ids, weights, shared_gu = _inputs(1)
    shared_gu[0, 0] ^= np.uint16(1)
    called = []
    monkeypatch.setattr(
        backend,
        "_run_grouped_moe_numpy_call",
        lambda *_args, **_kwargs: called.append(True),
    )
    with pytest.raises(ValueError, match="shared_gu mismatch"):
        backend.moe_experts_grouped(LAYER, x, ids, weights, shared_gu)
    assert called == []


@pytest.mark.parametrize("enabled", [False, True])
def test_backend_duplicate_experts_fails_with_exact_message_before_first_call(
    lightweight_backend, monkeypatch, enabled
) -> None:
    backend = lightweight_backend
    backend.prefill_moe_overlap = enabled
    x, ids, weights, shared_gu = _inputs(2)
    ids[0, -1] = ids[0, 0]

    called: list[str] = []
    monkeypatch.setattr(
        backend_module._GroupedNumpyRunner,
        "prepare",
        lambda self, io, call, values: called.append(f"runner.prepare({io})"),
    )
    monkeypatch.setattr(
        backend_module._GroupedNumpyRunner,
        "start",
        lambda self, io, layer: called.append(f"runner.start({io})"),
    )
    monkeypatch.setattr(
        backend,
        "_run_grouped_moe_numpy_call",
        lambda *args, **kwargs: called.append("numpy_call"),
    )
    monkeypatch.setattr(
        backend,
        "_run_grouped_moe_hardware_call",
        lambda *args, **kwargs: called.append("hardware_call"),
    )

    with pytest.raises(
        ValueError, match=r"^each token must route to 6 distinct experts$"
    ):
        backend.moe_experts_grouped(LAYER, x, ids, weights, shared_gu)

    assert called == []


@pytest.mark.parametrize(
    "value,expected",
    [(None, True), ("1", True), ("yes", True), ("0", False),
     ("false", False), ("False", False), ("no", False)],
)
def test_overlap_environment_switch(monkeypatch, value, expected) -> None:
    if value is None:
        monkeypatch.delenv("RAON_PREFILL_MOE_OVERLAP", raising=False)
    else:
        monkeypatch.setenv("RAON_PREFILL_MOE_OVERLAP", value)
    backend = NpuBackend(mock=True)
    assert backend.prefill_moe_overlap is expected
    backend.prefill_moe_overlap = not expected
    assert backend.prefill_moe_overlap is (not expected)
    backend.prefill_moe_overlap = None
    assert backend.prefill_moe_overlap is expected


def test_overlap_runner_is_created_on_first_overlap_call(lightweight_backend) -> None:
    backend = lightweight_backend
    assert "grouped_moe_overlap" not in backend._compiled_ops
    x, ids, weights, shared_gu = _inputs(1)
    backend.moe_experts_grouped(LAYER, x, ids, weights, shared_gu)
    assert set(backend._compiled_ops["grouped_moe_overlap"]) == {"b1", "b2"}


class _FakeTensor:
    def __init__(
        self,
        value,
        *,
        flags=None,
        group_id=None,
        xrt_device=None,
        label=None,
        events=None,
    ) -> None:
        del flags, xrt_device
        self.data = np.array(value, copy=True)
        self.group_id = group_id
        self.label = label or "inst"
        self.events = events if events is not None else []
        self.active = False
        self.overwrites = 0
        self.uploaded_overwrite = -1

    @contextmanager
    def overwrite(self):
        if self.active:
            raise AssertionError(f"overwrite of active {self.label}")
        self.events.append(("overwrite", self.label))
        self.overwrites += 1
        yield self.data

    def to(self, device):
        self.events.append(("to", self.label, device))
        if self.label.startswith("inst") or ".inst" in self.label:
            self.uploaded_overwrite = self.overwrites
        return self

    def buffer_object(self):
        return self

    def numpy(self):
        if self.active:
            raise AssertionError(f"read of active {self.label}")
        self.events.append(("numpy", self.label))
        return self.data


class _FakeRun:
    def __init__(self, state, tensors, events, label) -> None:
        self.state = state
        self.tensors = tensors
        self.events = events
        self.label = label
        self.wait_count = 0
        for tensor in tensors:
            tensor.active = True

    def wait(self):
        self.wait_count += 1
        if self.wait_count != 1:
            raise AssertionError(f"run {self.label} waited more than once")
        self.events.append(("xrt_wait", self.label))
        for tensor in self.tensors:
            tensor.active = False
        return self.state


class _FakeSubmitKernel:
    def __init__(self, state, events, group_id) -> None:
        self.state = state
        self.events = events
        self._group_id = group_id
        self.runs = []
        self.submissions = []

    def group_id(self, index):
        assert index == 1
        return self._group_id

    def __call__(self, opcode, inst, nbytes, *bos):
        assert opcode == 3
        assert nbytes == inst.data.nbytes
        assert inst.uploaded_overwrite == inst.overwrites > 0
        assert len(bos) == 3
        label = bos[0].label
        self.events.append(("xrt_start", label))
        self.submissions.append(
            {"opcode": opcode, "inst": inst, "nbytes": nbytes, "bos": bos}
        )
        run = _FakeRun(self.state, (inst, *bos), self.events, label)
        self.runs.append(run)
        return run


class _FakeXclbinKernel:
    def __init__(self, name="MLIR_AIE", num_args=6) -> None:
        self.name = name
        self.num_args = num_args

    def get_name(self):
        return self.name

    def get_num_args(self):
        return self.num_args


class _FakeXclbin:
    def __init__(self, num_args=6) -> None:
        self.kernels = [_FakeXclbinKernel(num_args=num_args)]

    def get_kernels(self):
        return self.kernels


class _FakeHandle:
    def __init__(self, state, events, generation, num_args=6) -> None:
        self.name = "MLIR_AIE"
        self.xclbin = _FakeXclbin(num_args)
        self.kernel = _FakeSubmitKernel(state, events, 100 + generation)
        self._is_valid = True


class _FakeRuntime:
    _tensor_class = _FakeTensor

    def __init__(self, state="completed", num_args=6) -> None:
        self.state = state
        self.num_args = num_args
        self.events = []
        self._device = object()
        self.handles = []
        self.consistency_checks = 0

    def load(self, _kernel):
        handle = _FakeHandle(
            self.state, self.events, len(self.handles), self.num_args
        )
        self.handles.append(handle)
        return handle

    def check_device_consistency(self):
        self.consistency_checks += 1
        self.events.append(("consistency",))


class _FakePatchedKernel:
    expert_params = tuple(f"e{i}" for i in range(8))
    row_params = tuple(f"r{i}" for i in range(8))
    dispatch_params = expert_params

    def __init__(self, words=16, *, with_rows=False) -> None:
        self.words = words
        self.length_delta = 0
        self.calls = []
        self.row_calls = []
        self.scalar_calls = []
        if with_rows:
            self.dispatch_params = self.expert_params + self.row_params

    def instructions(self, **scalars):
        expected_experts = set(self.expert_params)
        expected_full = expected_experts | set(self.row_params)
        given = set(scalars)
        if given not in (expected_experts, expected_full):
            raise AssertionError(f"unexpected dispatch scalars: {sorted(given)}")
        experts = tuple(int(scalars[name]) for name in self.expert_params)
        rows = (
            tuple(int(scalars[name]) for name in self.row_params)
            if given == expected_full
            else None
        )
        self.calls.append(experts)
        self.row_calls.append(rows)
        self.scalar_calls.append(dict(scalars))
        result = np.zeros(self.words + self.length_delta, dtype=np.uint32)
        result[: min(8, result.size)] = experts[: min(8, result.size)]
        if rows is not None and result.size >= 16:
            result[8:16] = rows
        return result


def _tensor(size, label, events, dtype=np.uint16):
    return _FakeTensor(np.zeros(size, dtype=dtype), label=label, events=events)


def _make_runner(backend, shape, *, state="completed", num_args=6, x_dtype=None):
    runtime = _FakeRuntime(state=state, num_args=num_args)
    kernel = _FakePatchedKernel(with_rows=shape is backend_module.GROUPED_B1)
    actual_x_dtype = backend_module.bfloat16 if x_dtype is None else x_dtype
    x = tuple(
        _tensor(
            8 * 8 * shape.k,
            f"{shape.name}.x{io}",
            runtime.events,
            dtype=actual_x_dtype,
        )
        for io in range(2)
    )
    y = tuple(
        _tensor(8 * 8 * shape.m, f"{shape.name}.y{io}", runtime.events)
        for io in range(2)
    )
    runner = _GroupedDirectRunner(
        backend, shape, kernel, runtime, "completed", "cacheable", x, y
    )
    for io, inst in enumerate(runner.insts):
        inst.label = f"{shape.name}.inst{io}"
        inst.events = runtime.events
    return runner, runtime, kernel


def _call_for(shape, tokens=1):
    ids = np.tile(np.array([0, 1, 2, 3, 4, 127], dtype=np.int32), (tokens, 1))
    return build_plan(ids, include_shared=shape is backend_module.GROUPED_B2).calls[0]


def test_direct_runner_writes_uploads_and_submits_all_buffers(monkeypatch) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runner, runtime, kernel = _make_runner(backend, shape)
    weight = _tensor(4, "weight", runtime.events)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: weight)
    call = _call_for(shape)
    values = np.zeros((8, 8, shape.k), dtype=np.uint16)
    runner.prepare(0, call, values)
    run = runner.start(0, LAYER)
    runner.wait(run)
    out = runner.read(0)
    assert out.shape == (8, 8, shape.m)
    assert kernel.calls[-1] == call.expert_slots
    assert ("overwrite", "b1.inst0") in runtime.events
    assert ("to", "b1.inst0", "npu") in runtime.events
    for label in ("b1.y0", "weight", "b1.x0"):
        assert ("to", label, "npu") in runtime.events
    assert runtime.consistency_checks == 1


@pytest.mark.parametrize("shape", [backend_module.GROUPED_B1, backend_module.GROUPED_B2])
def test_direct_runner_overwrites_x_buffer_and_submits_expected_bo(
    monkeypatch, shape
) -> None:
    backend = NpuBackend(mock=True)
    runner, runtime, _kernel = _make_runner(backend, shape)
    weight = _tensor(4, f"{shape.name}.weight", runtime.events, dtype=backend_module.bfloat16)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: weight)

    call = _call_for(shape)
    sequence = [0, 1, 0, 1]
    values_list = [
        (
            np.arange(8 * 8 * shape.k, dtype=np.uint16).reshape(8, 8, shape.k)
            + np.uint16(step * 1000 + 1)
        )
        for step in range(len(sequence))
    ]

    for io, values in zip(sequence, values_list):
        other_io = 1 - io
        prev_other_data = runner.x[other_io].data.copy()
        prev_other_overwrites = runner.x[other_io].overwrites
        prev_io_overwrites = runner.x[io].overwrites

        runner.prepare(io, call, values)

        # 1. runner.x[io]에 _grouped_x_tiled(values, shape)의 uint16 바이트가 그대로 쓰였는지 확인
        expected_u16 = backend_module._grouped_x_tiled(values, shape)
        actual_u16 = runner.x[io].data.view(np.uint16).reshape(expected_u16.shape)
        np.testing.assert_array_equal(actual_u16, expected_u16)

        # 2. 그 쓰기가 overwrite()로 기록됐는지 확인
        assert runner.x[io].overwrites == prev_io_overwrites + 1
        assert ("overwrite", runner.x[io].label) in runtime.events

        # 3. 다른 io의 X는 그대로여야 한다
        np.testing.assert_array_equal(runner.x[other_io].data, prev_other_data)
        assert runner.x[other_io].overwrites == prev_other_overwrites

        # 4. start(io, layer)가 커널에 넘긴 X BO가 runner.x[io]의 것인지 확인
        run = runner.start(io, LAYER)
        assert len(runner.handle.kernel.submissions) > 0
        sub = runner.handle.kernel.submissions[-1]
        assert sub["bos"][2] is runner.x[io].buffer_object()
        assert run.tensors[3] is runner.x[io].buffer_object()

        runner.wait(run)


def test_direct_runner_keeps_per_io_instruction_words(monkeypatch) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runner, runtime, _kernel = _make_runner(backend, shape)
    monkeypatch.setattr(
        backend,
        "_get_prefill_weight_tensor",
        lambda _name: _tensor(4, "weight", runtime.events),
    )
    ids = np.tile(np.array([0, 1, 2, 3, 4, 127], dtype=np.int32), (9, 1))
    plan = build_plan(ids)
    values = np.zeros((8, 8, shape.k), dtype=np.uint16)
    for io, call in enumerate(plan.calls):
        runner.prepare(io, call, values)
        np.testing.assert_array_equal(
            runner.insts[io].data[:8], np.asarray(call.expert_slots, dtype=np.uint32)
        )
        run = runner.start(io, LAYER)
        runner.wait(run)


def test_direct_runner_reloads_invalid_handle_and_instruction_buffers() -> None:
    backend = NpuBackend(mock=True)
    runner, runtime, _kernel = _make_runner(backend, backend_module.GROUPED_B1)
    old_insts = runner.insts
    old_groups = tuple(inst.group_id for inst in old_insts)
    runner.handle._is_valid = False
    runner.prepare(
        0,
        _call_for(backend_module.GROUPED_B1),
        np.zeros((8, 8, backend_module.GROUPED_B1.k), dtype=np.uint16),
    )
    assert len(runtime.handles) == 2
    assert runner.insts[0] is not old_insts[0]
    assert tuple(inst.group_id for inst in runner.insts) != old_groups


def test_direct_runner_rejects_invalid_start_length_state_abi_and_argument(
    monkeypatch,
) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    with pytest.raises(RuntimeError, match="at least 3"):
        _make_runner(backend, shape, num_args=5)

    runner, runtime, kernel = _make_runner(backend, shape)
    call = _call_for(shape)
    values = np.zeros((8, 8, shape.k), dtype=np.uint16)
    kernel.length_delta = 1
    with pytest.raises(RuntimeError, match="instruction length changed"):
        runner.prepare(0, call, values)
    kernel.length_delta = 0
    runner.prepare(0, call, values)
    runner.handle._is_valid = False
    with pytest.raises(RuntimeError, match="became invalid"):
        runner.start(0, LAYER)

    runner, runtime, _kernel = _make_runner(backend, shape, state="failed")
    weight = _tensor(4, "weight", runtime.events)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: weight)
    runner.prepare(0, call, values)
    run = runner.start(0, LAYER)
    with pytest.raises(RuntimeError, match="Kernel execution failed"):
        runner.wait(run)
    assert run.wait_count == 1
    assert not any(tensor.active for tensor in run.tensors)

    runner, runtime, _kernel = _make_runner(backend, shape)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: object())
    runner.prepare(0, call, values)
    with pytest.raises(RuntimeError, match="arguments must use"):
        runner.start(0, LAYER)


@pytest.mark.parametrize("state", ["completed", "failed"])
def test_direct_runner_wait_releases_kept_run_for_every_returned_state(
    monkeypatch, state
) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runner, runtime, _kernel = _make_runner(backend, shape, state=state)
    monkeypatch.setattr(
        backend,
        "_get_prefill_weight_tensor",
        lambda _name: _tensor(4, "weight", runtime.events),
    )
    runner.prepare(
        0, _call_for(shape), np.zeros((8, 8, shape.k), dtype=np.uint16)
    )
    run = runner.start(0, LAYER)
    assert runner._kept[0][-1] is run

    if state == "completed":
        runner.wait(run)
    else:
        with pytest.raises(RuntimeError, match="Kernel execution failed"):
            runner.wait(run)

    assert runner._kept == [None, None]


def test_direct_runner_close_releases_xrt_references_and_blocks_reuse() -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runner, _runtime, _kernel = _make_runner(backend, shape)
    call = _call_for(shape)
    values = np.zeros((8, 8, shape.k), dtype=np.uint16)

    runner.close()
    runner.close()

    assert runner._kept == [None, None]
    assert runner.insts == (None, None)
    assert runner.x == (None, None)
    assert runner.y == (None, None)
    assert runner.handle is None
    assert runner.kernel is None
    with pytest.raises(RuntimeError, match="closed"):
        runner.prepare(0, call, values)
    with pytest.raises(RuntimeError, match="closed"):
        runner.start(0, LAYER)


def test_grouped_runner_factory_registers_one_cleanup_and_closes_all(
    monkeypatch,
) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runtime = _FakeRuntime()
    tensor_index = 0

    def make_tensor(value):
        nonlocal tensor_index
        tensor = _FakeTensor(value, label=f"factory.tensor{tensor_index}")
        tensor_index += 1
        return tensor

    fake_aie_utils = SimpleNamespace(
        DefaultNPURuntime=runtime,
        tensor=make_tensor,
        full=lambda shape, fill, dtype: make_tensor(
            np.full(shape, fill, dtype=dtype)
        ),
    )
    fake_pyxrt = SimpleNamespace(
        ert_cmd_state=SimpleNamespace(ERT_CMD_STATE_COMPLETED="completed"),
        bo=SimpleNamespace(cacheable="cacheable"),
    )
    registered = []
    monkeypatch.setattr(backend_module, "aie_utils", fake_aie_utils)
    monkeypatch.setitem(sys.modules, "pyxrt", fake_pyxrt)
    monkeypatch.setattr(
        backend_module, "_LIVE_GROUPED_RUNNERS", backend_module.weakref.WeakSet()
    )
    monkeypatch.setattr(backend_module, "_GROUPED_RUNNER_ATEXIT_REGISTERED", False)
    monkeypatch.setattr(backend_module.atexit, "register", registered.append)

    runners = [
        backend_module._make_grouped_direct_runner(
            backend,
            shape,
            _FakePatchedKernel(),
            make_tensor(np.zeros(8 * 8 * shape.k, dtype=backend_module.bfloat16)),
            make_tensor(np.zeros(8 * 8 * shape.m, dtype=backend_module.bfloat16)),
        )
        for _ in range(2)
    ]

    assert registered == [backend_module._close_live_grouped_runners]
    assert len(backend_module._LIVE_GROUPED_RUNNERS) == 2
    registered[0]()
    assert all(runner._closed for runner in runners)
    assert all(runner.handle is None for runner in runners)


def test_direct_runner_prepare_uploads_x_before_start(monkeypatch) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runner, runtime, _kernel = _make_runner(backend, shape)
    monkeypatch.setattr(
        backend,
        "_get_prefill_weight_tensor",
        lambda _name: _tensor(4, "weight", runtime.events),
    )

    runner.prepare(
        0, _call_for(shape), np.zeros((8, 8, shape.k), dtype=np.uint16)
    )
    upload = ("to", "b1.x0", "npu")
    assert upload in runtime.events
    runner.start(0, LAYER)
    assert runtime.events.index(upload) < runtime.events.index(("xrt_start", "b1.y0"))


def _instrument_runner(runner, high_events):
    original_prepare = runner.prepare
    original_start = runner.start
    original_wait = runner.wait
    original_read = runner.read

    def prepare(self, io, call, values):
        high_events.append(("prepare", self.shape.name, io))
        return original_prepare(io, call, values)

    def start(self, io, layer):
        result = original_start(io, layer)
        high_events.append(("start", self.shape.name, io, result))
        return result

    def wait(self, run):
        high_events.append(("wait", self.shape.name, run))
        return original_wait(run)

    def read(self, io):
        high_events.append(("read", self.shape.name, io))
        return original_read(io)

    runner.prepare = MethodType(prepare, runner)
    runner.start = MethodType(start, runner)
    runner.wait = MethodType(wait, runner)
    runner.read = MethodType(read, runner)


@pytest.mark.parametrize("tokens,call_count", [(1, 1), (9, 2), (17, 3)])
def test_pipeline_order_and_io_ownership_with_direct_runner(
    monkeypatch, tokens, call_count
) -> None:
    backend = NpuBackend(mock=True)
    runners = {}
    high_events = []
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        runner, runtime, _kernel = _make_runner(backend, shape)
        weight = _tensor(4, f"{shape.name}.weight", runtime.events)
        # Keep the per-shape weight reachable from the runner for the backend hook.
        runner.test_weight = weight
        _instrument_runner(runner, high_events)
        runners[shape.name] = runner
    monkeypatch.setattr(
        backend, "_get_grouped_overlap_runner", lambda shape, _dw: runners[shape.name]
    )
    monkeypatch.setattr(
        backend,
        "_get_prefill_weight_tensor",
        lambda name: runners[name.rsplit("_", 1)[-1]].test_weight,
    )
    ids = np.tile(np.array([0, 1, 2, 3, 4, 127], dtype=np.int32), (tokens, 1))
    weights = np.ones((tokens, 6), dtype=np.float32)
    x = np.zeros((tokens, 2560), dtype=np.uint16)
    shared = np.zeros((tokens, 2176), dtype=np.uint16)
    backend._moe_grouped_overlapped(
        LAYER, x, ids, weights, shared, 4.0, 25.0, None, time.perf_counter()
    )

    for shape_name in ("b1", "b2"):
        events = [event for event in high_events if event[1] == shape_name]
        starts = [event for event in events if event[0] == "start"]
        waits = [event for event in events if event[0] == "wait"]
        reads = [event for event in events if event[0] == "read"]
        assert len(starts) == len(waits) == len(reads) == call_count
        for index in range(call_count):
            start_pos = events.index(starts[index])
            wait_pos = events.index(waits[index])
            assert start_pos < wait_pos
            if index > 0:
                read_prev = ("read", shape_name, (index - 1) % 2)
                assert start_pos < events.index(read_prev) < wait_pos
                assert events.index(waits[index - 1]) < start_pos
            if index + 1 < call_count:
                prepare_next = ("prepare", shape_name, (index + 1) % 2)
                prepare_pos = next(
                    pos
                    for pos in range(start_pos + 1, len(events))
                    if events[pos] == prepare_next
                )
                assert prepare_pos < wait_pos
                if index > 0:
                    assert events.index(read_prev) < prepare_pos
        final_read_pos = max(
            pos for pos, event in enumerate(events) if event == reads[-1]
        )
        assert final_read_pos > events.index(waits[-1])


@pytest.mark.parametrize("failure", ["situ", "accumulate", "prepare", "start", "wait"])
def test_pipeline_failure_preserves_original_and_waits_each_run_once(
    monkeypatch, failure
) -> None:
    backend = NpuBackend(mock=True)
    runners = {}
    all_runs = []
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        runner, runtime, _kernel = _make_runner(backend, shape)
        runner.test_weight = _tensor(4, f"{shape.name}.weight", runtime.events)
        original_start = runner.start

        def start_and_collect(self, io, layer, _original=original_start):
            run = _original(io, layer)
            all_runs.append(run)
            return run

        runner.start = MethodType(start_and_collect, runner)
        runners[shape.name] = runner
    monkeypatch.setattr(
        backend, "_get_grouped_overlap_runner", lambda shape, _dw: runners[shape.name]
    )
    monkeypatch.setattr(
        backend,
        "_get_prefill_weight_tensor",
        lambda name: runners[name.rsplit("_", 1)[-1]].test_weight,
    )

    def injected(*_args, **_kwargs):
        raise LookupError(f"injected {failure}")

    if failure == "situ":
        monkeypatch.setattr(backend_module, "situ_call", injected)
    elif failure == "accumulate":
        monkeypatch.setattr(backend_module, "accumulate_call", injected)
    else:
        runner = runners["b1"]
        original = getattr(runner, failure)
        calls = 0

        def fail_method(self, *args, **kwargs):
            nonlocal calls
            calls += 1
            if failure == "wait":
                result = original(*args, **kwargs)
                if calls == 1:
                    raise LookupError("injected wait")
                return result
            if calls == 2:
                raise LookupError(f"injected {failure}")
            return original(*args, **kwargs)

        setattr(runner, failure, MethodType(fail_method, runner))

    tokens = 9
    ids = np.tile(np.array([0, 1, 2, 3, 4, 127], dtype=np.int32), (tokens, 1))
    with pytest.raises(LookupError, match=f"injected {failure}"):
        backend._moe_grouped_overlapped(
            LAYER,
            np.zeros((tokens, 2560), dtype=np.uint16),
            ids,
            np.ones((tokens, 6), dtype=np.float32),
            np.zeros((tokens, 2176), dtype=np.uint16),
            4.0,
            25.0,
            None,
            time.perf_counter(),
        )
    assert all(run.wait_count == 1 for run in all_runs)
    assert all(not tensor.active for run in all_runs for tensor in run.tensors)
    # Failed waits and pre-wait failures must not create a success record for
    # the still-pending call.
    assert sum(backend.xclbin_calls.values()) <= len(all_runs)
    if failure in ("prepare", "wait"):
        assert sum(backend.xclbin_calls.values()) == 0


def test_direct_and_callable_runners_pass_rows_only_for_b1(monkeypatch) -> None:
    backend = NpuBackend(mock=True)
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        runner, runtime, kernel = _make_runner(backend, shape)
        monkeypatch.setattr(
            backend,
            "_get_prefill_weight_tensor",
            lambda _name, _rt=runtime: _tensor(4, "weight", _rt.events),
        )
        call = _call_for(shape, tokens=3)
        values = np.zeros((8, 8, shape.k), dtype=np.uint16)
        runner.prepare(0, call, values)
        if shape is backend_module.GROUPED_B1:
            assert kernel.row_calls[-1] == call.rows
            assert kernel.scalar_calls[-1] == {
                **{f"e{i}": int(e) for i, e in enumerate(call.expert_slots)},
                **{f"r{i}": int(r) for i, r in enumerate(call.rows)},
            }
        else:
            assert kernel.row_calls[-1] is None
            assert kernel.scalar_calls[-1] == {
                f"e{i}": int(e) for i, e in enumerate(call.expert_slots)
            }

        recorded = []
        x_buf = _tensor(8 * 8 * shape.k, "x", runtime.events, dtype=backend_module.bfloat16)
        y_buf = _tensor(8 * 8 * shape.m, "y", runtime.events)
        callable_runner = backend_module._GroupedCallableRunner(
            backend,
            shape,
            (lambda _y, _w, _x, **scalars: recorded.append(scalars), x_buf, y_buf),
        )
        callable_runner.prepare(0, call, values)
        callable_runner.start(0, LAYER)
        if shape is backend_module.GROUPED_B1:
            assert recorded[-1] == {
                **{f"e{i}": int(e) for i, e in enumerate(call.expert_slots)},
                **{f"r{i}": int(r) for i, r in enumerate(call.rows)},
            }
        else:
            assert recorded[-1] == {
                f"e{i}": int(e) for i, e in enumerate(call.expert_slots)
            }


def test_non_overlapped_b1_passes_call_rows_and_requires_rows_for_b1(
    lightweight_backend, monkeypatch
) -> None:
    backend = lightweight_backend
    backend.prefill_moe_overlap = False
    backend.mock = False
    monkeypatch.setattr(backend_module, "HAVE_NPU", True)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: object())

    overwritten = []

    def overwrite(tensor, value):
        overwritten.append(tensor)
        tensor.data[:] = np.asarray(value).view(np.uint16).reshape(-1)

    monkeypatch.setattr(backend, "_overwrite_buffer", overwrite)

    class _Buf:
        def __init__(self, size: int) -> None:
            self.data = np.zeros(size, dtype=np.uint16)

        def numpy(self) -> np.ndarray:
            return self.data

    seen_scalars = []
    grouped = {}
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        x_bo = _Buf(8 * 8 * shape.k)
        y_bo = _Buf(8 * 8 * shape.m)

        def run(_y, _w, _x, _name=shape.name, **scalars):
            seen_scalars.append((_name, dict(scalars)))

        grouped[shape.name] = (run, x_bo, y_bo)

    # B1 without rows must fail before touching any buffer or running the kernel
    with pytest.raises(ValueError, match="requires rows"):
        backend._run_grouped_moe_hardware_call(
            backend_module.GROUPED_B1,
            LAYER,
            (0,) * 8,
            np.zeros((8, 8, backend_module.GROUPED_B1.k), dtype=np.uint16),
            grouped,
        )
    assert overwritten == []
    assert seen_scalars == []

    # Synchronous moe_experts_grouped path passes rows=call.rows to B1 and rows=None to B2
    hw_calls = []
    orig_hw = backend._run_grouped_moe_hardware_call

    def spy_hw(shape, layer, expert_slots, values, grp, rows=None):
        hw_calls.append((shape.name, expert_slots, rows))
        return orig_hw(shape, layer, expert_slots, values, grp, rows=rows)

    monkeypatch.setattr(backend, "_get_or_compile_grouped_moe", lambda: grouped)
    monkeypatch.setattr(backend, "_run_grouped_moe_hardware_call", spy_hw)
    x, ids, weights, shared_gu = _inputs(9)
    b1_plan = build_plan(ids)
    b2_plan = build_plan(ids, include_shared=True)
    backend.moe_experts_grouped(LAYER, x, ids, weights, shared_gu)

    expected_hw = [
        ("b1", call.expert_slots, call.rows) for call in b1_plan.calls
    ] + [
        ("b2", call.expert_slots, None) for call in b2_plan.calls
    ]
    assert hw_calls == expected_hw
    for (_name, _experts, rows), (seen_name, scalars) in zip(hw_calls, seen_scalars):
        assert _name == seen_name
        if seen_name == "b1":
            assert tuple(scalars[f"r{i}"] for i in range(8)) == rows
        else:
            assert not any(key.startswith("r") for key in scalars)


def test_direct_runner_five_calls_alternate_instruction_buffers_with_distinct_rows(
    monkeypatch,
) -> None:
    backend = NpuBackend(mock=True)
    shape = backend_module.GROUPED_B1
    runner, runtime, _kernel = _make_runner(backend, shape)
    monkeypatch.setattr(
        backend,
        "_get_prefill_weight_tensor",
        lambda _name: _tensor(4, "weight", runtime.events),
    )

    # Build 5 calls with distinct row tuples
    distinct_calls = []
    for t in (1, 2, 3, 5, 7):
        ids = np.tile(np.array([0, 1, 2, 3, 4, 127], dtype=np.int32), (t, 1))
        distinct_calls.append(build_plan(ids).calls[0])
    assert len({call.rows for call in distinct_calls}) == 5

    values = np.zeros((8, 8, shape.k), dtype=np.uint16)
    submitted_words = []
    for index, call in enumerate(distinct_calls):
        io = index % 2
        runner.prepare(io, call, values)
        run = runner.start(io, LAYER)
        sub = runner.handle.kernel.submissions[-1]
        assert sub["inst"] is runner.insts[io].buffer_object()
        words = sub["inst"].data.copy()
        submitted_words.append(words)
        assert tuple(int(v) for v in words[:8]) == call.expert_slots
        assert tuple(int(v) for v in words[8:16]) == call.rows
        runner.wait(run)

    for call, words in zip(distinct_calls, submitted_words):
        assert tuple(int(v) for v in words[8:16]) == call.rows
