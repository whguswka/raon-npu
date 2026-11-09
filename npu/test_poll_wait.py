# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
import inspect
import sys
import types

import pytest

import npu.backend as backend_module


NEW = 10
QUEUED = 11
RUNNING = 12
SUBMITTED = 13
COMPLETED = 14
ERROR = 15
PENDING = (NEW, QUEUED, RUNNING, SUBMITTED)


class FakeRun:
    def __init__(
        self,
        states,
        wait_state=COMPLETED,
        limit=1_000,
        state_error=None,
        wait_error=None,
    ):
        self._states = iter(states)
        self._last_state = None
        self.wait_state = wait_state
        self.limit = limit
        self.state_error = state_error
        self.wait_error = wait_error
        self.state_calls = 0
        self.wait_calls = 0
        self.events = []

    def state(self):
        self.state_calls += 1
        if self.state_calls > self.limit:
            raise AssertionError("poll did not stop")
        if self.state_error is not None:
            raise self.state_error
        try:
            self._last_state = next(self._states)
        except StopIteration:
            pass
        self.events.append(("state", self._last_state))
        return self._last_state

    def wait(self):
        self.wait_calls += 1
        if self.wait_error is not None:
            raise self.wait_error
        self.events.append(("wait", self.wait_state))
        return self.wait_state


def fake_pyxrt(include_running=True):
    values = {
        "ERT_CMD_STATE_NEW": NEW,
        "ERT_CMD_STATE_QUEUED": QUEUED,
        "ERT_CMD_STATE_SUBMITTED": SUBMITTED,
        "ERT_CMD_STATE_COMPLETED": COMPLETED,
        "ERT_CMD_STATE_ERROR": ERROR,
    }
    if include_running:
        values["ERT_CMD_STATE_RUNNING"] = RUNNING
    return types.SimpleNamespace(ert_cmd_state=types.SimpleNamespace(**values))


def install_fake_clock(monkeypatch, values):
    ticks = iter(values)
    sleeps = []
    monkeypatch.setattr(backend_module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setattr(backend_module.time, "sleep", lambda value: sleeps.append(value))
    return sleeps


def test_poll_run_walks_pending_states_without_wait(monkeypatch):
    run = FakeRun([NEW, QUEUED, SUBMITTED, COMPLETED])
    sleeps = install_fake_clock(monkeypatch, [0.0] * 10)

    backend_module._poll_run(run, PENDING)

    assert run.state_calls == 4
    assert sleeps == [0, 0, 0]
    assert run.wait_calls == 0


def test_poll_run_completed_immediately(monkeypatch):
    run = FakeRun([COMPLETED])
    sleeps = install_fake_clock(monkeypatch, [0.0] * 4)

    backend_module._poll_run(run, PENDING)

    assert run.state_calls == 1
    assert sleeps == []
    assert run.wait_calls == 0


def test_poll_run_stops_on_noncompleted_terminal_state(monkeypatch):
    run = FakeRun([ERROR])
    sleeps = install_fake_clock(monkeypatch, [0.0] * 4)

    backend_module._poll_run(run, PENDING)

    assert run.state_calls == 1
    assert sleeps == []
    assert run.wait_calls == 0


def test_poll_run_obeys_one_second_upper_bound(monkeypatch):
    run = FakeRun([RUNNING], limit=10)
    sleeps = install_fake_clock(monkeypatch, (index * 0.3 for index in range(20)))

    backend_module._poll_run(run, PENDING)

    assert run.state_calls <= 10
    assert run.wait_calls == 0
    assert all(value == 0 for value in sleeps)


def test_poll_run_state_failure_returns_without_wait_or_sleep(monkeypatch):
    run = FakeRun([], state_error=RuntimeError("state failed"))
    sleeps = install_fake_clock(monkeypatch, [0.0])

    backend_module._poll_run(run, PENDING)

    assert run.state_calls == 1
    assert run.wait_calls == 0
    assert sleeps == []


@pytest.mark.parametrize(
    ("include_running", "expected"),
    [(False, (NEW, QUEUED, SUBMITTED)), (True, PENDING)],
)
def test_poll_wait_states_selects_only_available_pending_states(
    monkeypatch, include_running, expected
):
    monkeypatch.delenv("RAON_NPU_POLL_WAIT", raising=False)

    assert backend_module._poll_wait_states(fake_pyxrt(include_running)) == expected


@pytest.mark.parametrize("value", [None, "1", "yes"])
def test_poll_wait_states_enabled_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("RAON_NPU_POLL_WAIT", raising=False)
    else:
        monkeypatch.setenv("RAON_NPU_POLL_WAIT", value)

    assert backend_module._poll_wait_states(fake_pyxrt()) == PENDING


@pytest.mark.parametrize("value", ["0", "false", "False", "no"])
def test_poll_wait_states_disabled_values(monkeypatch, value):
    monkeypatch.setenv("RAON_NPU_POLL_WAIT", value)

    assert backend_module._poll_wait_states(fake_pyxrt()) == ()


def make_xrt_runner(monkeypatch, pending_states=()):
    monkeypatch.setattr(backend_module, "_prepare_xrt_handle", lambda runtime, kernel: object())
    return backend_module._XrtDirectRunner(
        object(), object(), completed_state=COMPLETED, pending_states=pending_states
    )


def test_xrt_runner_default_skips_polling(monkeypatch):
    assert inspect.signature(backend_module._XrtDirectRunner).parameters[
        "pending_states"
    ].default == ()
    runner = make_xrt_runner(monkeypatch)
    run = FakeRun([COMPLETED])

    runner.wait(run)

    assert run.state_calls == 0
    assert run.wait_calls == 1
    assert run.events == [("wait", COMPLETED)]


def test_xrt_runner_polls_before_one_wait(monkeypatch):
    runner = make_xrt_runner(monkeypatch, PENDING)
    run = FakeRun([NEW, QUEUED, COMPLETED])
    sleeps = install_fake_clock(monkeypatch, [0.0] * 10)

    runner.wait(run)

    assert run.events == [
        ("state", NEW),
        ("state", QUEUED),
        ("state", COMPLETED),
        ("wait", COMPLETED),
    ]
    assert sleeps == [0, 0]
    assert run.wait_calls == 1


def test_xrt_runner_waits_after_state_failure(monkeypatch):
    runner = make_xrt_runner(monkeypatch, PENDING)
    run = FakeRun([], state_error=RuntimeError("state failed"))
    install_fake_clock(monkeypatch, [0.0])

    runner.wait(run)

    assert run.state_calls == 1
    assert run.wait_calls == 1


def test_xrt_runner_reports_wait_state_after_state_failure(monkeypatch):
    runner = make_xrt_runner(monkeypatch, PENDING)
    run = FakeRun(
        [], wait_state=ERROR, state_error=RuntimeError("state failed")
    )
    install_fake_clock(monkeypatch, [0.0])

    with pytest.raises(
        RuntimeError,
        match=r"^Kernel execution failed with state 15 \(expected 14\)$",
    ):
        runner.wait(run)

    assert run.state_calls == 1
    assert run.wait_calls == 1


@pytest.mark.parametrize("pending_states", [(), PENDING])
def test_xrt_runner_rejects_failed_wait_state(monkeypatch, pending_states):
    runner = make_xrt_runner(monkeypatch, pending_states)
    run = FakeRun([COMPLETED], wait_state=ERROR)
    install_fake_clock(monkeypatch, [0.0] * 5)

    with pytest.raises(
        RuntimeError,
        match=r"^Kernel execution failed with state 15 \(expected 14\)$",
    ):
        runner.wait(run)

    assert run.wait_calls == 1


def bare_grouped_runner(pending_states=()):
    runner = object.__new__(backend_module._GroupedDirectRunner)
    runner.completed_state = COMPLETED
    runner.pending_states = pending_states
    runner._kept = [None, None]
    return runner


def test_grouped_runner_default_and_poll_order(monkeypatch):
    assert inspect.signature(backend_module._GroupedDirectRunner).parameters[
        "pending_states"
    ].default == ()
    run = FakeRun([NEW, COMPLETED])
    runner = bare_grouped_runner(PENDING)
    runner._kept = [("first", object()), ("second", run)]
    sleeps = install_fake_clock(monkeypatch, [0.0] * 8)

    runner.wait(run)

    assert run.events == [("state", NEW), ("state", COMPLETED), ("wait", COMPLETED)]
    assert sleeps == [0]
    assert run.wait_calls == 1
    assert runner._kept[0][0] == "first"
    assert runner._kept[1] is None


def test_grouped_runner_empty_pending_skips_polling(monkeypatch):
    runner = bare_grouped_runner()
    run = FakeRun([COMPLETED])
    runner._kept = [("first", run), None]

    runner.wait(run)

    assert run.state_calls == 0
    assert run.wait_calls == 1
    assert runner._kept == [None, None]


def test_grouped_runner_cleans_kept_before_failure(monkeypatch):
    runner = bare_grouped_runner(PENDING)
    run = FakeRun([COMPLETED], wait_state=ERROR)
    runner._kept = [("first", object()), ("second", run)]
    install_fake_clock(monkeypatch, [0.0] * 5)

    with pytest.raises(
        RuntimeError,
        match=r"^Kernel execution failed with state 15 \(expected 14\)$",
    ):
        runner.wait(run)

    assert runner._kept[0][0] == "first"
    assert runner._kept[1] is None
    assert run.wait_calls == 1


def test_grouped_runner_waits_and_cleans_after_state_failure(monkeypatch):
    runner = bare_grouped_runner(PENDING)
    run = FakeRun([], state_error=RuntimeError("state failed"))
    other_run = object()
    runner._kept = [("first", other_run), ("second", run)]
    install_fake_clock(monkeypatch, [0.0])

    runner.wait(run)

    assert run.state_calls == 1
    assert run.wait_calls == 1
    assert runner._kept == [("first", other_run), None]


def test_grouped_runner_reports_wait_state_and_cleans_after_state_failure(
    monkeypatch,
):
    runner = bare_grouped_runner(PENDING)
    run = FakeRun(
        [], wait_state=ERROR, state_error=RuntimeError("state failed")
    )
    other_run = object()
    runner._kept = [("first", other_run), ("second", run)]
    install_fake_clock(monkeypatch, [0.0])

    with pytest.raises(
        RuntimeError,
        match=r"^Kernel execution failed with state 15 \(expected 14\)$",
    ):
        runner.wait(run)

    assert run.state_calls == 1
    assert run.wait_calls == 1
    assert runner._kept == [("first", other_run), None]


@pytest.mark.parametrize("wait_fails", [False, True])
def test_rda_abort_all_waits_after_state_failure_and_cleans(
    monkeypatch, wait_fails
):
    runner = make_xrt_runner(monkeypatch, PENDING)
    run = FakeRun(
        [],
        state_error=RuntimeError("state failed"),
        wait_error=RuntimeError("wait failed") if wait_fails else None,
    )
    install_fake_clock(monkeypatch, [0.0])
    ticket = backend_module._RdaTicket(
        layer=0,
        io=0,
        phase="launched",
        xclbin_id="test-xclbin",
        run=run,
        runner=runner,
    )
    npu_backend = object.__new__(backend_module.NpuBackend)
    npu_backend._rda_live = [ticket]
    npu_backend._rda_head_busy = [True, False]

    npu_backend.rda_abort_all()

    assert run.state_calls == 1
    assert run.wait_calls == 1
    assert ticket.phase == "waited"
    assert ticket.run is None
    assert ticket.runner is None
    assert npu_backend._rda_live == []
    assert npu_backend._rda_head_busy == [False, False]


class FakeNPUKernel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.mark.parametrize(("env_value", "expected"), [(None, PENDING), ("0", ())])
def test_xrt_factory_passes_poll_states(monkeypatch, env_value, expected):
    pyxrt = fake_pyxrt()
    aie = types.ModuleType("aie")
    utils = types.ModuleType("aie.utils")
    npukernel = types.ModuleType("aie.utils.npukernel")
    npukernel.NPUKernel = FakeNPUKernel
    aie.utils = utils
    utils.npukernel = npukernel
    monkeypatch.setitem(sys.modules, "pyxrt", pyxrt)
    monkeypatch.setitem(sys.modules, "aie", aie)
    monkeypatch.setitem(sys.modules, "aie.utils", utils)
    monkeypatch.setitem(sys.modules, "aie.utils.npukernel", npukernel)
    monkeypatch.setattr(
        backend_module, "aie_utils", types.SimpleNamespace(DefaultNPURuntime=object())
    )
    monkeypatch.setattr(backend_module, "_prepare_xrt_handle", lambda runtime, kernel: object())
    if env_value is None:
        monkeypatch.delenv("RAON_NPU_POLL_WAIT", raising=False)
    else:
        monkeypatch.setenv("RAON_NPU_POLL_WAIT", env_value)
    op = types.SimpleNamespace(
        xclbin_artifact=types.SimpleNamespace(filename="kernel.xclbin", kernel_name="kernel"),
        insts_artifact=types.SimpleNamespace(filename="kernel.bin"),
    )

    runner = backend_module._make_xrt_direct_runner(op)

    assert runner.pending_states == expected
    assert runner.npu_kernel.kwargs == {
        "xclbin_path": "kernel.xclbin",
        "kernel_name": "kernel",
        "insts_path": "kernel.bin",
    }


@pytest.mark.parametrize(("env_value", "expected"), [(None, PENDING), ("0", ())])
def test_grouped_factory_passes_poll_states(monkeypatch, env_value, expected):
    calls = []

    class RecordingGroupedRunner:
        def __init__(self, *args):
            calls.append(args)

        def close(self):
            pass

    pyxrt = fake_pyxrt()
    pyxrt.bo = types.SimpleNamespace(cacheable="cacheable")
    monkeypatch.setitem(sys.modules, "pyxrt", pyxrt)
    monkeypatch.setattr(backend_module, "_GroupedDirectRunner", RecordingGroupedRunner)
    monkeypatch.setattr(
        backend_module,
        "aie_utils",
        types.SimpleNamespace(
            tensor=lambda value: ("tensor", value.shape),
            full=lambda shape, value, dtype: ("full", shape, value, dtype),
            DefaultNPURuntime="runtime",
        ),
    )
    monkeypatch.setattr(backend_module, "_GROUPED_RUNNER_ATEXIT_REGISTERED", True)
    if env_value is None:
        monkeypatch.delenv("RAON_NPU_POLL_WAIT", raising=False)
    else:
        monkeypatch.setenv("RAON_NPU_POLL_WAIT", env_value)
    shape = types.SimpleNamespace(k=8, m=16)

    runner = backend_module._make_grouped_direct_runner(
        "backend", shape, "kernel", "x0", "y0"
    )
    try:
        assert len(calls) == 1
        assert calls[0][-1] == expected
        assert calls[0][-4] == "cacheable"
    finally:
        backend_module._LIVE_GROUPED_RUNNERS.discard(runner)

    assert runner not in backend_module._LIVE_GROUPED_RUNNERS
