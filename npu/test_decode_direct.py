# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only fault-injection tests for the decode direct runner."""

from __future__ import annotations

import gc
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import weakref

import numpy as np
import pytest

import npu.backend as nb


class FakeBO:
    # Global settle duration used by all fake BOs in this process.  Each test
    # resets this via the fake_world fixture; it is read when a BO is created
    # and again on every access to decide whether reads/writes are settled.
    _settle_duration: float = 0.0

    def __init__(self, host: np.ndarray, events: list[str], name: str) -> None:
        self.host = host
        self._device = np.array(host, copy=True)
        self.events = events
        self.name = name
        self._alloc_time = time.monotonic()

    @property
    def device(self) -> np.ndarray:
        if time.monotonic() - self._alloc_time >= FakeBO._settle_duration:
            return self._device
        return np.zeros_like(self._device)

    @device.setter
    def device(self, value: np.ndarray) -> None:
        if time.monotonic() - self._alloc_time >= FakeBO._settle_duration:
            self._device[...] = value

    def sync(self, direction: int) -> None:
        if direction == 1:
            self.events.append(f"sync-to:{self.name}")
            self.device[...] = self.host
        elif direction == 2:
            self.events.append(f"sync-from:{self.name}")
            self.host[...] = self.device
        else:  # pragma: no cover - catches an invalid fake contract
            raise AssertionError(direction)


class FakeTensor:
    serial = 0

    def __init__(self, data: np.ndarray, events: list[str], name: str | None = None) -> None:
        FakeTensor.serial += 1
        self.data = np.array(data, copy=True)
        self.name = name or f"tensor-{FakeTensor.serial}"
        self.bo = FakeBO(self.data, events, self.name)
        self.events = events
        self.to_calls: list[str] = []

    def buffer_object(self) -> FakeBO:
        return self.bo

    def to(self, device: str) -> "FakeTensor":
        self.events.append(f"to:{self.name}:{device}")
        self.to_calls.append(device)
        self.bo.device[...] = self.data
        return self


class FakeKernel:
    _serial = 0

    def __init__(self, name: str, events: list[str]) -> None:
        FakeKernel._serial += 1
        self.kernel_name = name
        self.events = events
        self._group_id = 17 + FakeKernel._serial

    def group_id(self, index: int) -> int:
        return self._group_id * 10 + index

    @staticmethod
    def _is_settled(bo: FakeBO) -> bool:
        return time.monotonic() - bo._alloc_time >= FakeBO._settle_duration

    def _validate(self, args: dict[int, object]) -> None:
        # MoE dynamic instruction words must be written to the device before any
        # start; this is checked even on the first (warmup) start that does not
        # yet touch the output BO.  The check is skipped when the instruction BO
        # has not yet settled, because the fake defect device discards the whole
        # execution in that case.
        if 6 not in args:
            insts = args.get(1)
            if (
                isinstance(insts, FakeBO)
                and insts.name == "insts"
                and self._is_settled(insts)
                and np.all(insts.device == 0)
            ):
                raise AssertionError("MoE instructions were zero at start")

    def execute(self, args: dict[int, object]) -> None:
        self._validate(args)
        insts = args.get(1)
        if isinstance(insts, FakeBO) and not self._is_settled(insts):
            # The command-stream BO was allocated after the last settle: drop
            # this execution exactly like an unsettled data BO.
            if 6 in args and isinstance(args[6], FakeBO):
                args[6].device.fill(0)
            elif 3 in args and isinstance(args[3], FakeBO):
                args[3].device.fill(0)
            return
        if 6 in args:  # RDA has seven arguments.
            head = args[5]
            out = args[6]
            assert isinstance(head, FakeBO) and isinstance(out, FakeBO)
            out.device.fill(float(np.sum(head.device, dtype=np.float64)))
            state_out = args[4]
            assert isinstance(state_out, FakeBO)
            state_out.device.fill(9.0)
            return
        y, w, x = args[3], args[4], args[5]
        assert isinstance(y, FakeBO) and isinstance(w, FakeBO) and isinstance(x, FakeBO)
        value = (int(np.sum(x.device.view(np.uint16), dtype=np.uint64)) +
                 int(w.device.reshape(-1)[0])) & 0xFFFF
        y.device.view(np.uint16).fill(value)


class FakeRun:
    def __init__(self, kernel: FakeKernel, events: list[str], completed: int) -> None:
        self.kernel = kernel
        self.events = events
        self.completed = completed
        self.wait_result = completed
        self.args: dict[int, object] = {}
        self.set_calls: list[tuple[int, object]] = []
        self.starts = 0
        self.waits = 0
        self.states = 0
        self.start_error: Exception | None = None

    def set_arg(self, index: int, arg: object) -> None:
        self.events.append(f"set:{index}")
        self.set_calls.append((index, arg))
        self.args[index] = arg

    def start(self) -> None:
        self.events.append("start")
        if self.start_error is not None:
            raise self.start_error
        self.starts += 1
        self.kernel.execute(self.args)

    def state(self) -> int:
        self.events.append("state")
        self.states += 1
        if self.states == 1:
            return 2
        return self.completed

    def wait(self) -> int:
        self.events.append("wait")
        self.waits += 1
        if isinstance(self.wait_result, Exception):
            raise self.wait_result
        return self.wait_result


class FakeAbiKernel:
    def __init__(self, name: str, args: int = 7) -> None:
        self.name = name
        self.args = args

    def get_name(self) -> str:
        return self.name

    def get_num_args(self) -> int:
        return self.args


class FakeHandle:
    def __init__(self, name: str, events: list[str]) -> None:
        self._is_valid = True
        self.kernel = FakeKernel(name, events)
        # The command-stream BO is also a FakeBO so that the fake defect device
        # can drop an execution started before the BO has settled.
        self.insts = np.zeros(4, dtype=np.uint32)
        self.insts_bo = FakeBO(self.insts, events, f"insts-{name}")
        self.xclbin = SimpleNamespace(get_kernels=lambda: [FakeAbiKernel(name)])


class FakeNPUKernel:
    def __init__(self, *, xclbin_path: str, kernel_name: str, insts_path: str) -> None:
        self.xclbin_path = xclbin_path
        self.kernel_name = kernel_name
        self.insts_path = insts_path


class FakeTemplateKernel:
    dispatch_params = tuple(f"e{i}" for i in range(6))

    def __init__(self, name: str, xclbin_path: Path) -> None:
        self.kernel_name = name
        self.xclbin_path = str(xclbin_path)

    def instructions(self, **values: int) -> np.ndarray:
        return np.array([values.get(f"e{i}", 0) for i in range(6)], dtype=np.uint32)


class FakeRuntime:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.loads: list[object] = []
        self.checks = 0
        self._device = object()
        self._tensor_class = self.tensor_class

    def check_device_consistency(self) -> None:
        self.checks += 1

    def load(self, kernel: object) -> FakeHandle:
        self.loads.append(kernel)
        name = getattr(kernel, "kernel_name", "kernel")
        self.events.append(f"load:{name}")
        return FakeHandle(name, self.events)

    def tensor_class(self, data: np.ndarray, **kwargs: object) -> FakeTensor:
        self.events.append("alloc-tensor:insts")
        t = FakeTensor(data, self.events, "insts")
        t.kwargs = kwargs
        t.group_id = kwargs.get("group_id")
        return t


@pytest.fixture
def fake_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events: list[str] = []
    runs: list[FakeRun] = []
    completed = 7
    pyxrt = SimpleNamespace(
        ert_cmd_state=SimpleNamespace(
            ERT_CMD_STATE_COMPLETED=completed,
            ERT_CMD_STATE_RUNNING=2,
        ),
        xclBOSyncDirection=SimpleNamespace(
            XCL_BO_SYNC_BO_TO_DEVICE=1,
            XCL_BO_SYNC_BO_FROM_DEVICE=2,
        ),
        bo=SimpleNamespace(cacheable=99),
    )

    def make_run(kernel: FakeKernel) -> FakeRun:
        run = FakeRun(kernel, events, completed)
        runs.append(run)
        return run

    pyxrt.run = make_run
    runtime = FakeRuntime(events)
    tensor = lambda data: FakeTensor(np.asarray(data), events)
    fake_aie = SimpleNamespace(
        tensor=tensor,
        full=lambda shape, value, dtype: FakeTensor(
            np.full(shape, value, dtype=dtype), events
        ),
        DefaultNPURuntime=runtime,
    )
    fake_iron = SimpleNamespace(
        zeros=lambda shape, dtype, device: FakeTensor(
            np.zeros(shape, dtype=dtype), events
        )
    )
    monkeypatch.setattr(nb, "aie_utils", fake_aie)
    monkeypatch.setattr(nb, "iron", fake_iron)
    pkg = ModuleType("aie")
    utils = ModuleType("aie.utils")
    npukernel = ModuleType("aie.utils.npukernel")
    npukernel.NPUKernel = FakeNPUKernel
    monkeypatch.setitem(sys.modules, "aie", pkg)
    monkeypatch.setitem(sys.modules, "aie.utils", utils)
    monkeypatch.setitem(sys.modules, "aie.utils.npukernel", npukernel)
    monkeypatch.setitem(sys.modules, "pyxrt", pyxrt)

    fake_time = SimpleNamespace(now=0.0)
    def fake_sleep(seconds: float) -> None:
        fake_time.now += seconds
        events.append(f"sleep:{seconds}")

    monkeypatch.setattr(time, "monotonic", lambda: fake_time.now)
    monkeypatch.setattr(time, "sleep", fake_sleep)
    FakeBO._settle_duration = 0.0

    xclbin = tmp_path / "unified.xclbin"
    xclbin.touch()
    runner = nb._DecodeDirectRunner(runtime, pyxrt, xclbin)

    def op(name: str = "gemv") -> object:
        return SimpleNamespace(
            xclbin_artifact=SimpleNamespace(filename=str(xclbin), kernel_name=name),
            insts_artifact=SimpleNamespace(filename=str(tmp_path / f"{name}.bin")),
        )

    return SimpleNamespace(
        events=events, runs=runs, runtime=runtime, pyxrt=pyxrt,
        runner=runner, op=op, xclbin=xclbin, tensor=tensor,
        fake_time=fake_time,
    )


def test_01_02_gemv_sync_copy_and_reuse(fake_world) -> None:
    """성질 1·2: 장치 사본 동기화, 0 패딩, 복사 출력, run 재사용, 첫 호출 start 1회."""
    f = fake_world
    weight = f.tensor(np.array([3], dtype=np.uint16))
    weight.to("npu")
    entry = f.runner.gemv_entry("bundle", f.op(), weight, 4, 3)
    # No warmup; the run is created but not started until the first real call.
    assert (entry.run.starts, entry.run.waits) == (0, 0)
    outputs = [f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) for _ in range(3)]
    assert np.all(outputs[0] == 13)
    assert np.all(outputs[-1] == 13)
    assert not np.shares_memory(outputs[-1], entry.y_host)
    assert np.all(entry.x.bo.device.view(np.uint16)[4:] == 0)
    assert len(f.runs) == 1
    assert len(f.runs[0].set_calls) == 6
    assert (f.runs[0].starts, f.runs[0].waits) == (3, 3)
    assert f.events.index("state") < f.events.index("wait")
    # First real call: x sync, then the actual start, then wait.
    first_sync = f.events.index(f"sync-to:{entry.x.name}")
    first_start = f.events.index("start")
    assert first_sync < first_start


def test_03_noncompleted_state_raises(fake_world) -> None:
    """성질 3: 완료가 아닌 XRT 상태는 정해진 RuntimeError다."""
    f = fake_world
    entry = f.runner.gemv_entry("b", f.op(), f.tensor(np.array([0], np.uint16)), 1, 1)
    f.runs[0].wait_result = 55
    with pytest.raises(RuntimeError, match="Kernel execution failed with state 55"):
        f.runner.gemv(entry, np.array([1], np.uint16))


def test_04_invalidation_discards_every_family(fake_world) -> None:
    """성질 4: 한 핸들 무효화는 GEMV·MoE·RDA 전 항목을 재생성한다."""
    f = fake_world
    weight = f.tensor(np.array([1], dtype=np.uint16))
    old = f.runner.gemv_entry("b", f.op(), weight, 2, 1)
    kernel = FakeTemplateKernel("moe", f.xclbin)
    f.runner.moe_start("b1", 0, kernel, weight, kernel.instructions(e0=1),
                       np.array([1, 2], np.uint16), nb.MAX_K, 2)

    s0 = f.tensor(np.zeros((1,), np.float32))
    s1 = f.tensor(np.zeros((1,), np.float32))
    f.runner.rda(f.op("rda"), s0, s1, np.zeros((40, 640), np.float32))
    old.handle._is_valid = False
    replacement = f.runner.gemv_entry("b", f.op(), weight, 2, 1)
    assert replacement is not old
    assert not f.runner._moe and f.runner._rda is None
    # Without warmup the discarded run was never started.
    assert old.run.starts == 0


def test_05_rda_rebinds_states_and_orders_sync(fake_world) -> None:
    """성질 5: RDA 상태 BO를 매번 다시 묶고 TO/start/wait/FROM 순서를 지킨다."""
    f = fake_world
    pairs = []
    for tag in ("a", "b"):
        s_in = FakeTensor(np.zeros((2,), np.float32), f.events, f"in-{tag}")
        s_out = FakeTensor(np.zeros((2,), np.float32), f.events, f"out-{tag}")
        pairs.append((s_in, s_out))
        out = f.runner.rda(f.op("rda"), s_in, s_out,
                           np.full((40, 640), 1.0, np.float32))
        assert np.all(out == 25600.0)
    run = f.runs[0]
    args3 = [arg for index, arg in run.set_calls if index == 3]
    args4 = [arg for index, arg in run.set_calls if index == 4]
    # States are rebound before each of the two real calls.
    assert args3 == [p[0].bo for p in pairs]
    assert args4 == [p[1].bo for p in pairs]
    first_start = f.events.index("start")
    assert f.events.index("to:in-a:npu") < first_start
    assert f.events.index("to:out-a:npu") < first_start
    assert any(event.startswith("sync-to:") for event in f.events[:first_start])
    assert f.events.index("wait") < f.events.index(next(e for e in f.events if e.startswith("sync-from:")))


def test_05_backend_rda_changes_slot_only_after_success(monkeypatch, tmp_path) -> None:
    """성질 5·8: RDA 실패 시 활성 칸/장부 불변, 성공 시 전환 및 직접 셈 1회."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    monkeypatch.setattr(backend, "_use_prefill_rda", lambda: False)
    rda_module = ModuleType("bench.rda_state.rda_state")
    rda_module.pack_head_bundle = lambda *args: np.zeros((40, 640), np.float32)
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", rda_module)
    state0 = SimpleNamespace()
    state1 = SimpleNamespace()
    backend.rda_states[2] = [state0, state1]
    backend.rda_active_slot[2] = 0
    op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="rda"
        ),
        insts_artifact=SimpleNamespace(filename="rda.bin"),
    )
    backend._compiled_ops["rda_step"] = lambda *args: None
    backend._compiled_ops["rda_step_op"] = op

    class Direct:
        fail = True

        def rda(self, *_args):
            if self.fail:
                raise RuntimeError("wait failed")
            return np.zeros((40, 128), dtype=np.float32)

    direct = Direct()
    monkeypatch.setattr(backend, "_decode_runner", lambda: direct)
    args = (
        np.zeros((40, 128), np.float32),
        np.zeros((40, 128), np.float32),
        np.zeros((40, 128), np.float32),
        np.zeros((40,), np.float32),
        np.zeros((40,), np.float32),
    )
    with pytest.raises(RuntimeError, match="wait failed"):
        backend.rda_update(2, *args)
    assert backend.rda_active_slot[2] == 0
    assert backend.xclbin_calls == {}

    direct.fail = False
    out = backend.rda_update(2, *args)
    assert out.shape == (40, 128)
    assert backend.rda_active_slot[2] == 1
    assert backend.call_counts["decode_direct_rda"] == 1
    assert backend.xclbin_calls == {"gemv_unified": 1}


def test_06_moe_instruction_length_and_layer_run_reuse(fake_world) -> None:
    """성질 6: MoE 명령열은 매번 TO 동기화하며 길이와 층별 run 재사용을 지킨다."""
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    words = kernel.instructions(e0=1)
    for _ in range(2):
        run = f.runner.moe_start("b1", 3, kernel, weight, words,
                                 np.array([4, 5], np.uint16), nb.MAX_K, 2)
        f.runner.moe_finish("b1", run)
    assert len(f.runs) == 1 and f.runs[0].starts == 2
    assert f.events.count("sync-to:insts") == 2
    starts = f.runs[0].starts
    with pytest.raises(RuntimeError, match="instruction length changed"):
        f.runner.moe_start("b1", 3, kernel, weight, np.arange(7, dtype=np.uint32),
                           np.array([1], np.uint16), nb.MAX_K, 2)
    assert f.runs[0].starts == starts


@pytest.mark.parametrize("failure", [None, "shared-start", "b1-wait"])
def test_06_backend_moe_overlap_and_drain(monkeypatch, tmp_path, failure) -> None:
    """성질 6·8: B1/공유 동시 시작, 실패 drain, 호출별 직접 셈을 검증한다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend.decode_moe_overlap = False
    from kernels.moe.pack import B1_M_PER_EXPERT, B2_M_PER_EXPERT

    op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="shared"
        ),
        insts_artifact=SimpleNamespace(filename="shared.bin"),
    )
    shared_fn = SimpleNamespace(op=op)
    gu = FakeTemplateKernel("gu", backend._unified_xclbin_path)
    dn = FakeTemplateKernel("dn", backend._unified_xclbin_path)
    backend._compiled_ops["unified_moe_kernels"] = (gu, dn)
    monkeypatch.setattr(
        backend, "_get_or_compile_moe_ops", lambda: (object(), shared_fn, object())
    )
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda name: object())
    monkeypatch.setattr(backend, "_decode_moe_eligible", lambda *args: (True, ""))
    events = []

    class Direct:
        def moe_start(self, key, *_args):
            events.append(f"start-{key}")
            return key

        def gemv_entry(self, *_args):
            return "shared-entry"

        def gemv_start(self, *_args):
            events.append("start-shared")
            if failure == "shared-start":
                raise RuntimeError("shared start failed")
            return "shared"

        def moe_finish(self, key, _run):
            events.append(f"finish-{key}")
            if key == "b1" and failure == "b1-wait":
                raise RuntimeError("b1 wait failed")
            size = 6 * B1_M_PER_EXPERT if key == "b1" else 7 * B2_M_PER_EXPERT
            return np.zeros(size, dtype=np.uint16)

        def gemv_finish(self, *_args):
            events.append("finish-shared")
            return np.zeros(B1_M_PER_EXPERT, dtype=np.uint16)

        def drain(self, runs):
            events.append(f"drain:{','.join(runs)}")

    monkeypatch.setattr(backend, "_decode_runner", lambda: Direct())
    args = (
        0, np.zeros(2560, np.uint16), np.arange(6, dtype=np.int32),
        np.full(6, 1 / 6, np.float32),
    )
    if failure is not None:
        with pytest.raises(RuntimeError):
            backend.moe_experts(*args)
        expected = "drain:b1" if failure == "shared-start" else "drain:shared"
        assert expected in events
        return

    out = backend.moe_experts(*args)
    assert out.shape == (B2_M_PER_EXPERT,)
    assert events.index("start-b1") < events.index("start-shared")
    assert events.index("start-shared") < events.index("finish-b1")
    assert backend.call_counts["decode_direct_moe"] == 3
    for key in ("moe_b1_npu", "moe_sh_npu", "moe_dn_npu", "moe"):
        assert backend.call_counts[key] == 1


def _minimal_backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> nb.NpuBackend:
    backend = nb.NpuBackend(mock=True, gemv="unified")
    backend.mock = False
    backend._unified_xclbin_path = tmp_path / "u.xclbin"
    backend._unified_xclbin_path.touch()
    monkeypatch.setattr(nb, "HAVE_NPU", True)
    return backend


def _meta(k: int, m: int) -> nb.BundleMeta:
    return nb.BundleMeta(
        name="test", rel_file="w.bin", fmt="q4nx_gemv",
        orig_shape=[m, k], padded_shape=[m, k], k=k, m=m, components=[],
    )


def test_07_switch_and_ineligible_fallback(monkeypatch, tmp_path, capsys) -> None:
    """성질 7: 스위치와 op 자격이 직접/IRON 경로 및 경고·셈을 결정한다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    meta = _meta(2, 1)
    monkeypatch.setattr(nb, "resolve_unified_shape_name", lambda *args: "fake")
    monkeypatch.setattr(nb, "pad_x", lambda values, shape: values)
    fn_calls = []
    fn = lambda *args: fn_calls.append(args)
    monkeypatch.setattr(backend, "_get_or_compile_gemv", lambda *args: fn)
    weight = SimpleNamespace()
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: weight)
    x = FakeTensor(np.zeros(nb.MAX_K, dtype=nb.bfloat16), [], "x")
    y = FakeTensor(np.zeros(1, dtype=nb.bfloat16), [], "y")
    y.numpy = lambda: y.data
    monkeypatch.setattr(backend, "_prepared_buffer", lambda name: x if "x_" in name else y)
    monkeypatch.setattr(backend, "_overwrite_buffer", lambda tensor, value: tensor.data.__setitem__(slice(None, len(value)), value))
    made = []
    fake_runner = SimpleNamespace(
        gemv_cached_entry=lambda _bundle: None,
        prepare=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        nb, "_make_decode_direct_runner", lambda path: made.append(path) or fake_runner
    )

    backend.decode_direct = False
    backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    assert fn_calls and not made and "decode_direct_fallback_gemv" not in backend.call_counts

    backend.decode_direct = True
    backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    assert backend.call_counts["decode_direct_fallback_gemv"] == 2
    assert capsys.readouterr().err.count("[decode-direct] gemv") == 1

    fn.op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(filename=str(tmp_path / "other"), kernel_name="k"),
        insts_artifact=SimpleNamespace(filename="i"),
    )
    eligible, _ = backend._decode_op_eligible(fn.op)
    assert not eligible


def test_07_environment_switch_and_mock_never_create_runner(monkeypatch, tmp_path) -> None:
    """성질 7: 환경 스위치 0과 mock은 직접 실행기를 만들지 않는다."""
    monkeypatch.setenv("RAON_DECODE_DIRECT", "0")
    backend = nb.NpuBackend(mock=True, gemv="unified")
    backend._unified_xclbin_path = tmp_path / "u.xclbin"
    monkeypatch.setattr(nb, "HAVE_NPU", True)
    assert backend.decode_direct is False
    assert backend._decode_runner() is None
    backend.decode_direct = True
    assert backend._decode_runner() is None


def test_08_direct_gemv_counts_and_bytes(monkeypatch, tmp_path) -> None:
    """성질 8: 직접 GEMV는 출력 바이트·기존 장부를 유지하고 직접 호출 셈을 더한다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    meta = _meta(2, 2)
    monkeypatch.setattr(nb, "resolve_unified_shape_name", lambda *args: "fake")
    op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(filename=str(backend._unified_xclbin_path), kernel_name="k"),
        insts_artifact=SimpleNamespace(filename="i"),
    )
    fn = SimpleNamespace(op=op)
    monkeypatch.setattr(backend, "_get_or_compile_gemv", lambda *args: fn)
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda name: object())
    direct = SimpleNamespace(
        gemv_cached_entry=lambda _bundle: None,
        gemv_entry=lambda *args, **kwargs: object(),
        gemv=lambda entry, x: np.array([0x1234, 0x5678], np.uint16),
    )
    monkeypatch.setattr(backend, "_decode_runner", lambda: direct)
    out = backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    assert out.tobytes() == np.array([0x1234, 0x5678], np.uint16).tobytes()
    assert backend.call_counts["decode_direct_gemv"] == 1
    assert backend.call_counts["gemv:test"] == 1
    assert backend.call_counts["linear_npu"] == 1
    assert backend.xclbin_calls == {"gemv_unified": 1}


def test_09_close_factory_atexit_and_no_backend_reference(fake_world, monkeypatch) -> None:
    """성질 9: close 멱등성, 단일 atexit 등록, 백엔드 비참조 수명을 보장한다."""
    runner = fake_world.runner
    runner.close()
    runner.close()
    with pytest.raises(RuntimeError, match="decode direct runner is closed"):
        runner.gemv_entry("b", fake_world.op(), fake_world.tensor(np.zeros(1)), 1, 1)

    registrations = []
    monkeypatch.setattr(nb.atexit, "register", lambda fn: registrations.append(fn))
    monkeypatch.setattr(nb, "_DECODE_RUNNER_ATEXIT_REGISTERED", False)
    r1 = nb._make_decode_direct_runner(fake_world.xclbin)
    r2 = nb._make_decode_direct_runner(fake_world.xclbin)
    assert len(registrations) == 1

    backend = SimpleNamespace(_compiled_ops={"decode_direct": r1})
    ref = weakref.ref(r1)
    del r1
    backend._compiled_ops.clear()
    assert ref() is None
    r2.close()


def test_10_gemv_fast_path_skips_all_slow_lookups(
    fake_world, monkeypatch, tmp_path
) -> None:
    """2절 2.1: 두 번째 GEMV는 컴파일·가중치·경로·모양 조회를 건너뛴다."""
    f = fake_world
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend._unified_xclbin_path = f.xclbin
    meta = _meta(2, 2)
    op = f.op()
    weight = f.tensor(np.array([3], dtype=np.uint16))
    calls = {"shape": 0, "compile": 0, "weight": 0, "resolve": 0}

    def shape(*_args):
        calls["shape"] += 1
        return "fake-shape"

    def compile_op(*_args):
        calls["compile"] += 1
        return SimpleNamespace(op=op)

    def get_weight(_name):
        calls["weight"] += 1
        return weight

    monkeypatch.setattr(nb, "resolve_unified_shape_name", shape)
    monkeypatch.setattr(backend, "_get_or_compile_gemv", compile_op)
    monkeypatch.setattr(backend, "_get_weight_tensor", get_weight)
    monkeypatch.setattr(backend, "_decode_runner", lambda: f.runner)
    first = backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    first_counts = dict(backend.call_counts)
    first_loads = len(f.runtime.loads)

    real_resolve = Path.resolve

    def counted_resolve(path, *args, **kwargs):
        calls["resolve"] += 1
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    calls.update(shape=0, compile=0, weight=0, resolve=0)
    second = backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))

    assert first.tobytes() == second.tobytes()
    assert calls == {"shape": 0, "compile": 0, "weight": 0, "resolve": 0}
    assert len(f.runtime.loads) == first_loads
    for key in ("gemv:test", "linear_npu", "decode_direct_gemv"):
        assert backend.call_counts[key] == first_counts[key] + 1
    assert backend.call_counts["linear_host"] == 0
    assert backend.call_counts["host_prep"] == 0


def test_11_static_handles_are_shared_by_instruction_stream(fake_world) -> None:
    """2절 2.2: 같은 명령열은 핸들 하나, 다른 명령열은 별도 핸들을 쓴다."""
    f = fake_world
    weight = f.tensor(np.array([1], dtype=np.uint16))
    first = f.runner.gemv_entry("L00/a", f.op("gemv"), weight, 2, 1)
    second = f.runner.gemv_entry("L01/b", f.op("gemv"), weight, 2, 1)
    assert first.handle is second.handle
    assert len(f.runtime.loads) == 1

    third = f.runner.gemv_entry("L00/c", f.op("other"), weight, 2, 1)
    assert third.handle is not first.handle
    assert len(f.runtime.loads) == 2


def test_12_invalidation_is_checked_only_at_next_top_level_call(fake_world) -> None:
    """2절 2.3: 실행 중 무효화는 finish를 깨지 않고 다음 작업이 전부 재생성한다."""
    f = fake_world
    weight = f.tensor(np.array([3], dtype=np.uint16))
    entry = f.runner.gemv_entry("bundle", f.op(), weight, 2, 1)
    run = f.runner.gemv_start(entry, np.array([1, 2], np.uint16))
    entry.handle._is_valid = False
    out = f.runner.gemv_finish(entry, run)
    assert out.tolist() == [6]
    assert f.runner._gemv["bundle"] is entry

    replacement = f.runner.gemv_entry("bundle", f.op(), weight, 2, 1)
    assert replacement is not entry
    assert replacement.handle is not entry.handle


def test_13_drain_waits_after_runner_is_closed(fake_world) -> None:
    """2절 2.4: 닫힌 실행기의 drain도 원래 예외를 가리지 않고 run을 기다린다."""
    f = fake_world
    run = FakeRun(FakeKernel("drain", f.events), f.events, 7)
    f.runner.close()
    f.runner.drain([run])
    assert run.waits == 1


def test_14_moe_rda_eligibility_cache_and_fallback_accounting(
    monkeypatch, tmp_path, capsys
) -> None:
    """2절 2.5: MoE·RDA 자격을 기억하고 fallback은 호출별 셈/계열별 경고를 지킨다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    kernels = (
        FakeTemplateKernel("gu", backend._unified_xclbin_path),
        FakeTemplateKernel("dn", backend._unified_xclbin_path),
    )
    backend._compiled_ops["unified_moe_kernels"] = kernels
    shared_op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="shared"
        ),
        insts_artifact=SimpleNamespace(filename="shared.bin"),
    )
    shared_fn = SimpleNamespace(op=shared_op)
    rda_op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="rda"
        ),
        insts_artifact=SimpleNamespace(filename="rda.bin"),
    )
    real_resolve = Path.resolve
    resolves = []

    def counted_resolve(path, *args, **kwargs):
        resolves.append(str(path))
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", counted_resolve)
    assert backend._decode_moe_eligible(True, shared_fn) == (True, "")
    after_moe = len(resolves)
    assert backend._decode_moe_eligible(True, shared_fn) == (True, "")
    assert len(resolves) == after_moe
    assert backend._decode_rda_eligible(rda_op) == (True, "")
    after_rda = len(resolves)
    assert backend._decode_rda_eligible(rda_op) == (True, "")
    assert len(resolves) == after_rda

    for family in ("moe", "rda"):
        backend._decode_fallback(family, "ineligible")
        backend._decode_fallback(family, "ineligible")
        assert backend.call_counts[f"decode_direct_fallback_{family}"] == 2
    stderr = capsys.readouterr().err
    assert stderr.count("[decode-direct] moe") == 1
    assert stderr.count("[decode-direct] rda") == 1


def _counts_without_direct(backend: nb.NpuBackend) -> dict[str, int]:
    return {
        key: value
        for key, value in backend.call_counts.items()
        if not key.startswith("decode_direct_")
    }


def test_15_gemv_direct_and_iron_paths_have_same_observable_state(
    monkeypatch, tmp_path
) -> None:
    """2절 2.6: GEMV 직접/끔 경로의 바이트와 기존 장부가 같다."""
    expected = np.array([0x1234, 0x5678], dtype=np.uint16)

    def configured(direct: bool) -> nb.NpuBackend:
        backend = _minimal_backend(monkeypatch, tmp_path)
        backend.decode_direct = direct
        meta_op = SimpleNamespace(
            xclbin_artifact=SimpleNamespace(
                filename=str(backend._unified_xclbin_path), kernel_name="gemv"
            ),
            insts_artifact=SimpleNamespace(filename="gemv.bin"),
        )

        def iron_call(y, _w, _x):
            y.data[...] = expected.view(nb.bfloat16)

        iron_call.op = meta_op
        monkeypatch.setattr(backend, "_get_or_compile_gemv", lambda *_args: iron_call)
        monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: object())
        monkeypatch.setattr(
            nb, "resolve_unified_shape_name", lambda *_args, **_kwargs: "fake"
        )
        monkeypatch.setattr(nb, "pad_x", lambda values, _shape: values)
        x_buf = FakeTensor(np.zeros(nb.MAX_K, dtype=nb.bfloat16), [], "x")
        y_buf = FakeTensor(np.zeros(2, dtype=nb.bfloat16), [], "y")
        y_buf.numpy = lambda: y_buf.data
        monkeypatch.setattr(
            backend, "_prepared_buffer", lambda name: x_buf if "x_" in name else y_buf
        )
        monkeypatch.setattr(
            backend, "_overwrite_buffer",
            lambda tensor, value: tensor.data.__setitem__(slice(None, len(value)), value),
        )
        if direct:
            entry = SimpleNamespace(timing_key="gemv:test")
            runner = SimpleNamespace(
                gemv_cached_entry=lambda _bundle: None,
                gemv_entry=lambda *_args, **_kwargs: entry,
                gemv=lambda _entry, _x: expected.copy(),
            )
            monkeypatch.setattr(backend, "_decode_runner", lambda: runner)
        return backend

    direct = configured(True)
    iron_backend = configured(False)
    meta = _meta(2, 2)
    direct_out = direct._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    iron_out = iron_backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    assert direct_out.tobytes() == iron_out.tobytes()
    assert _counts_without_direct(direct) == _counts_without_direct(iron_backend)
    assert direct.xclbin_calls == iron_backend.xclbin_calls
    assert direct._last_active_xclbin == iron_backend._last_active_xclbin
    assert direct._bundle_cache == iron_backend._bundle_cache


def test_16_rda_direct_and_iron_paths_have_same_observable_state(
    monkeypatch, tmp_path
) -> None:
    """2절 2.6: RDA 직접/끔 경로의 바이트·장부·활성 슬롯이 같다."""
    rda_module = ModuleType("bench.rda_state.rda_state")
    rda_module.pack_head_bundle = lambda *_args: np.zeros((40, 640), np.float32)
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", rda_module)
    expected = np.full((40, 128), 1.25, dtype=np.float32)

    def configured(direct: bool) -> nb.NpuBackend:
        backend = _minimal_backend(monkeypatch, tmp_path)
        backend.decode_direct = direct
        backend.rda_states[0] = [SimpleNamespace(), SimpleNamespace()]
        backend.rda_active_slot[0] = 0
        op = SimpleNamespace(
            xclbin_artifact=SimpleNamespace(
                filename=str(backend._unified_xclbin_path), kernel_name="rda"
            ),
            insts_artifact=SimpleNamespace(filename="rda.bin"),
        )
        out_buf = FakeTensor(np.zeros((40, 128), np.float32), [], "rda-out")
        out_buf.numpy = lambda: out_buf.data
        head_buf = FakeTensor(np.zeros((40, 640), np.float32), [], "rda-head")

        def iron_call(_s_in, _s_out, _head, out):
            out.data[...] = expected

        backend._compiled_ops.update(rda_step=iron_call, rda_step_op=op)
        monkeypatch.setattr(
            backend, "_prepared_buffer", lambda name: head_buf if name == "rda_head_in" else out_buf
        )
        monkeypatch.setattr(
            backend, "_overwrite_buffer", lambda tensor, value: tensor.data.__setitem__(Ellipsis, value)
        )
        if direct:
            runner = SimpleNamespace(rda=lambda *_args: expected.copy())
            monkeypatch.setattr(backend, "_decode_runner", lambda: runner)
        return backend

    direct = configured(True)
    iron_backend = configured(False)
    args = (
        np.zeros((40, 128), np.float32), np.zeros((40, 128), np.float32),
        np.zeros((40, 128), np.float32), np.zeros(40, np.float32),
        np.zeros(40, np.float32),
    )
    direct_out = direct.rda_update(0, *args)
    iron_out = iron_backend.rda_update(0, *args)
    assert direct_out.tobytes() == iron_out.tobytes()
    assert _counts_without_direct(direct) == _counts_without_direct(iron_backend)
    assert direct.xclbin_calls == iron_backend.xclbin_calls
    assert direct._last_active_xclbin == iron_backend._last_active_xclbin
    assert direct.rda_active_slot == iron_backend.rda_active_slot


def test_17_moe_direct_and_iron_paths_have_same_observable_state(
    monkeypatch, tmp_path
) -> None:
    """2절 2.6: 공유 포함 MoE 직접/끔 경로의 바이트·캐시·장부가 같다."""
    from kernels.moe.pack import B1_M_PER_EXPERT, B2_M_PER_EXPERT, NUM_ACTIVE_TOTAL

    def configured(direct: bool) -> nb.NpuBackend:
        backend = _minimal_backend(monkeypatch, tmp_path)
        backend.decode_direct = direct
        backend.decode_moe_overlap = False
        op = SimpleNamespace(
            xclbin_artifact=SimpleNamespace(
                filename=str(backend._unified_xclbin_path), kernel_name="shared"
            ),
            insts_artifact=SimpleNamespace(filename="shared.bin"),
        )
        kernels = (
            FakeTemplateKernel("gu", backend._unified_xclbin_path),
            FakeTemplateKernel("dn", backend._unified_xclbin_path),
        )
        backend._compiled_ops["unified_moe_kernels"] = kernels

        def fill_zero(y, *_args, **_kwargs):
            y.data.fill(0)

        fill_zero.op = op
        monkeypatch.setattr(
            backend, "_get_or_compile_moe_ops", lambda: (fill_zero, fill_zero, fill_zero)
        )
        monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: object())
        buffers = {
            "moe_y_gu": FakeTensor(
                np.zeros(6 * B1_M_PER_EXPERT, dtype=nb.bfloat16), [], "y-gu"
            ),
            "moe_y_sh": FakeTensor(
                np.zeros(B1_M_PER_EXPERT, dtype=nb.bfloat16), [], "y-sh"
            ),
            "moe_y_dn": FakeTensor(
                np.zeros(NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT, dtype=nb.bfloat16),
                [], "y-dn",
            ),
            "moe_x_gu": FakeTensor(
                np.zeros(nb.MAX_K, dtype=nb.bfloat16), [], "x-gu"
            ),
            "moe_x_dn": FakeTensor(
                np.zeros(NUM_ACTIVE_TOTAL * nb.MAX_K, dtype=nb.bfloat16), [], "x-dn"
            ),
        }
        for tensor in buffers.values():
            tensor.numpy = lambda tensor=tensor: tensor.data
        monkeypatch.setattr(backend, "_prepared_buffer", buffers.__getitem__)

        def overwrite(tensor, value):
            flat = tensor.data.reshape(-1)
            source = np.asarray(value).reshape(-1)
            flat[: source.size] = source

        monkeypatch.setattr(backend, "_overwrite_buffer", overwrite)
        if direct:
            class Direct:
                def moe_start(self, key, *_args):
                    return key

                def moe_finish(self, key, _run):
                    size = (
                        6 * B1_M_PER_EXPERT
                        if key == "b1"
                        else NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT
                    )
                    return np.zeros(size, dtype=np.uint16)

                def gemv_entry(self, *_args):
                    return object()

                def gemv_start(self, *_args):
                    return "shared"

                def gemv_finish(self, *_args):
                    return np.zeros(B1_M_PER_EXPERT, dtype=np.uint16)

                def drain(self, _runs):
                    return None

            monkeypatch.setattr(backend, "_decode_runner", lambda: Direct())
        return backend

    direct = configured(True)
    iron_backend = configured(False)
    x = np.zeros(2560, dtype=np.uint16)
    ids = np.arange(6, dtype=np.int32)
    weights = np.full(6, 1 / 6, dtype=np.float32)
    direct_out = direct.moe_experts(0, x, ids, weights)
    iron_out = iron_backend.moe_experts(0, x, ids, weights)
    assert direct_out.tobytes() == iron_out.tobytes()
    assert _counts_without_direct(direct) == _counts_without_direct(iron_backend)
    assert direct.xclbin_calls == iron_backend.xclbin_calls
    assert direct._last_active_xclbin == iron_backend._last_active_xclbin
    assert direct._bundle_cache.keys() == iron_backend._bundle_cache.keys()
    for key in direct._bundle_cache:
        d_step, d_input, d_value = direct._bundle_cache[key]
        i_step, i_input, i_value = iron_backend._bundle_cache[key]
        assert (d_step, d_input) == (i_step, i_input)
        np.testing.assert_array_equal(d_value, i_value)
    assert direct._cached_shared.keys() == iron_backend._cached_shared.keys()
    for layer in direct._cached_shared:
        assert direct._cached_shared[layer].keys() == iron_backend._cached_shared[layer].keys()
        for key, d_value in direct._cached_shared[layer].items():
            i_value = iron_backend._cached_shared[layer][key]
            if isinstance(d_value, np.ndarray):
                np.testing.assert_array_equal(d_value, i_value)
            else:
                assert d_value == i_value


def test_r17_r18_moe_syncs_x_to_device_and_y_from_device(fake_world) -> None:
    """성질 R17·R18: B1/B2 MoE 는 x 를 장치로 동기화하고 y 를 호스트로 동기화한다."""
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))

    def run_b1(rows):
        run = f.runner.moe_start(
            "b1", 0, kernel, weight, kernel.instructions(e0=1),
            np.asarray(rows, dtype=np.uint16), nb.MAX_K, 2,
        )
        return f.runner.moe_finish("b1", run)

    assert run_b1([1, 2]).tolist() == [5, 5]
    assert run_b1([3, 4]).tolist() == [9, 9]

    def run_b2(rows):
        rows7 = np.tile(rows, (7, 1)).astype(np.uint16)
        run = f.runner.moe_start(
            "b2", 0, kernel, weight, kernel.instructions(e0=1),
            rows7, 7 * nb.MAX_K, 14,
        )
        return f.runner.moe_finish("b2", run)

    assert run_b2([1, 2]).tolist() == [23] * 14
    assert run_b2([3, 4]).tolist() == [51] * 14


def test_r19_drain_swallows_run_wait_exceptions(fake_world) -> None:
    """성질 R19: drain 은 run.wait() 예외를 삼키고 모든 run 을 한 번씩 기다린다."""
    f = fake_world
    run1 = FakeRun(FakeKernel("a", f.events), f.events, 7)
    run1.wait_result = RuntimeError("a failed")
    run2 = FakeRun(FakeKernel("b", f.events), f.events, 7)
    run2.wait_result = RuntimeError("b failed")
    f.runner.drain([run1, run2])
    assert run1.waits == 1
    assert run2.waits == 1


def test_r19_drain_swallows_exceptions_after_close(fake_world) -> None:
    """성질 R19: 닫힌 실행기의 drain 도 예외를 삼키고 run 을 기다린다."""
    f = fake_world
    run = FakeRun(FakeKernel("a", f.events), f.events, 7)
    run.wait_result = RuntimeError("a failed")
    f.runner.close()
    f.runner.drain([run])
    assert run.waits == 1


def test_r22_decode_runner_not_kept_by_backend(monkeypatch, tmp_path) -> None:
    """성질 R22: 백엔드는 _decode_runner() 실행기를 순환 참조로 붙잡지 않는다."""

    class FakeRunner:
        def prepare(self, *_args, **_kwargs):
            pass

    def make_runner() -> tuple[object, nb.NpuBackend]:
        backend = _minimal_backend(monkeypatch, tmp_path)
        backend._unified_xclbin_path = tmp_path / "u.xclbin"
        monkeypatch.setattr(nb, "_make_decode_direct_runner", lambda _path: FakeRunner())
        monkeypatch.setattr(nb.NpuBackend, "_get_or_compile_gemv", lambda self, *_args: SimpleNamespace(op=None))
        monkeypatch.setattr(nb.NpuBackend, "_get_or_compile_moe_ops", lambda self: (object(), object(), object()))
        monkeypatch.setattr(nb.NpuBackend, "_decode_moe_eligible", lambda self, *_args: (False, ""))
        monkeypatch.setattr(nb.NpuBackend, "_decode_rda_eligible", lambda self, _op: (False, ""))
        runner = backend._decode_runner()
        assert isinstance(runner, FakeRunner)
        return runner, backend

    def backend_in_referents(root: object, depth: int = 2) -> bool:
        seen: set[int] = set()

        def walk(obj: object, d: int) -> bool:
            if d == 0:
                return False
            i = id(obj)
            if i in seen:
                return False
            seen.add(i)
            if isinstance(obj, nb.NpuBackend):
                return True
            return any(walk(ref, d - 1) for ref in gc.get_referents(obj))

        return walk(root, depth)

    runner, backend = make_runner()
    monkeypatch.undo()
    # (가) No backend is reachable from the runner within two referent hops.
    assert not backend_in_referents(runner, depth=2)
    ref = weakref.ref(runner)
    # (나) Without a manual gc.collect the runner is still collected.
    del runner, backend
    assert ref() is None

    runner, backend = make_runner()
    runner._owner = backend  # injected cycle
    assert backend_in_referents(runner, depth=2)
    ref = weakref.ref(runner)
    del runner, backend
    assert ref() is not None


def test_r25_moe_start_checks_handle_and_reloads(fake_world) -> None:
    """성질 R25: 핸들 무효화 후 첫 MoE 호출은 핸들을 다시 로드하고 새 run 을 만든다."""
    f = fake_world
    weight = f.tensor(np.array([1], dtype=np.uint16))
    f.runner.gemv_entry("b", f.op(), weight, 2, 1)
    kernel = FakeTemplateKernel("moe", f.xclbin)
    old_run = f.runner.moe_start(
        "b1", 0, kernel, weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    old_starts = old_run.starts
    s0 = f.tensor(np.zeros((1,), np.float32))
    s1 = f.tensor(np.zeros((1,), np.float32))
    f.runner.rda(f.op("rda"), s0, s1, np.zeros((40, 640), np.float32))

    loads_before = len(f.runtime.loads)
    for handle in f.runner._handles.values():
        handle._is_valid = False

    new_run = f.runner.moe_start(
        "b1", 0, kernel, weight, kernel.instructions(e0=1),
        np.array([3, 4], np.uint16), nb.MAX_K, 2,
    )
    assert len(f.runtime.loads) > loads_before
    assert new_run is not old_run
    assert old_run.starts == old_starts
    assert new_run.starts == 1  # actual start only inside moe_start


def test_r26_rda_checks_handle_and_reloads(fake_world) -> None:
    """성질 R26: 핸들 무효화 후 첫 RDA 호출은 핸들을 다시 로드하고 새 run 을 만든다."""
    f = fake_world
    weight = f.tensor(np.array([1], dtype=np.uint16))
    f.runner.gemv_entry("b", f.op(), weight, 2, 1)
    kernel = FakeTemplateKernel("moe", f.xclbin)
    f.runner.moe_start(
        "b1", 0, kernel, weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    s0 = f.tensor(np.zeros((1,), np.float32))
    s1 = f.tensor(np.zeros((1,), np.float32))
    f.runner.rda(f.op("rda"), s0, s1, np.zeros((40, 640), np.float32))
    old_run = f.runner._rda.run
    old_starts = old_run.starts

    loads_before = len(f.runtime.loads)
    for handle in f.runner._handles.values():
        handle._is_valid = False

    s2 = f.tensor(np.zeros((1,), np.float32))
    s3 = f.tensor(np.zeros((1,), np.float32))
    f.runner.rda(f.op("rda"), s2, s3, np.zeros((40, 640), np.float32))
    new_run = f.runner._rda.run
    assert len(f.runtime.loads) > loads_before
    assert new_run is not old_run
    assert old_run.starts == old_starts
    assert new_run.starts == 1  # actual start only inside rda()


def test_b06_gemv_fast_path_records_xclbin(fake_world, monkeypatch, tmp_path) -> None:
    """성질 B06: GEMV 빠른 경로도 xclbin 사용을 기록하고 마지막 xclbin 을 복원한다."""
    f = fake_world
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend._unified_xclbin_path = f.xclbin
    meta = _meta(2, 2)
    op = f.op()
    weight = f.tensor(np.array([3], dtype=np.uint16))

    monkeypatch.setattr(nb, "resolve_unified_shape_name", lambda *_args, **_kwargs: "fake")
    monkeypatch.setattr(backend, "_get_or_compile_gemv", lambda *_args: SimpleNamespace(op=op))
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: weight)
    monkeypatch.setattr(backend, "_decode_runner", lambda: f.runner)

    backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    assert backend.xclbin_calls == {"gemv_unified": 2}
    assert backend._last_active_xclbin == "gemv_unified"

    backend._record_xclbin_use("prefill_gemm")
    backend._execute_gemv("L00/test", meta, np.array([1, 2], np.uint16))
    assert backend.xclbin_calls == {"gemv_unified": 3, "prefill_gemm": 1}
    assert backend.xclbin_switches == 2
    assert backend._last_active_xclbin == "gemv_unified"


def test_b17_moe_kernels_stored_in_compiled_ops(monkeypatch, tmp_path) -> None:
    """성질 B17: _get_or_compile_moe_ops() 는 B1·B2 callable 을 _compiled_ops 에 보관한다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend._unified_xclbin_path = tmp_path / "u.xclbin"
    backend._unified_xclbin_path.touch()

    gu_callable = object()
    sh_callable = object()
    dn_callable = object()

    class MockUnifiedOp:
        dispatch_params = tuple(f"e{i}" for i in range(6))
        xclbin_path = str(backend._unified_xclbin_path)

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compile(self):
            pass

        def instructions(self, **values):
            return np.array([values.get(f"e{i}", 0) for i in range(6)], dtype=np.uint32)

    gu_op = MockUnifiedOp()
    sh_op = MockUnifiedOp()
    dn_op = MockUnifiedOp()
    gu_op.get_callable = lambda: gu_callable
    sh_op.get_callable = lambda: sh_callable
    dn_op.get_callable = lambda: dn_callable

    class FakeUnifiedGEMV:
        @classmethod
        def for_shape(cls, _shape_name, **_kwargs):
            return sh_op

    fake_op = ModuleType("kernels.gemv_unified.op")
    fake_op.UnifiedGEMV = FakeUnifiedGEMV
    monkeypatch.setitem(sys.modules, "kernels.gemv_unified.op", fake_op)

    fake_stage2 = ModuleType("kernels.gemv_unified.stage2")
    fake_stage2.UnifiedMoEGateUp = lambda **kwargs: gu_op
    fake_stage2.UnifiedMoEDown9 = lambda **kwargs: dn_op
    monkeypatch.setitem(sys.modules, "kernels.gemv_unified.stage2", fake_stage2)

    monkeypatch.setattr(
        nb, "_require_single_unified_xclbin",
        lambda _ops, expected=None: backend._unified_xclbin_path,
    )

    prepared = []

    def fake_prepare_dynamic_kernel(kernel):
        prepared.append(kernel)
        def run(*_args, **_kwargs):
            pass
        return run

    monkeypatch.setattr(backend, "_prepare_dynamic_kernel", fake_prepare_dynamic_kernel)

    fns = backend._get_or_compile_moe_ops()
    assert fns is backend._compiled_ops["unified_moe_ops"]
    assert backend._compiled_ops["unified_moe_kernels"] == (gu_callable, dn_callable)
    assert prepared == [gu_callable, dn_callable]

    fns2 = backend._get_or_compile_moe_ops()
    assert fns2 is fns
    assert len(prepared) == 2


def test_b18_rda_op_stored_in_compiled_ops(monkeypatch, tmp_path) -> None:
    """성질 B18: _prepare_unified_rda_kernel() 는 op 객체를 _compiled_ops 에 보관한다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend._unified_xclbin_path = tmp_path / "u.xclbin"
    backend._unified_xclbin_path.touch()

    rda_callable = object()

    class MockRDA:
        xclbin_path = str(backend._unified_xclbin_path)

        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def compile(self):
            pass

        def get_callable(self):
            return rda_callable

    fake_rda = ModuleType("kernels.gemv_unified.rda")
    fake_rda.UnifiedRDA = lambda **kwargs: MockRDA()
    monkeypatch.setitem(sys.modules, "kernels.gemv_unified.rda", fake_rda)

    monkeypatch.setattr(
        nb, "_require_single_unified_xclbin",
        lambda _ops, expected=None: backend._unified_xclbin_path,
    )

    fn = backend._prepare_unified_rda_kernel()
    assert fn is rda_callable
    op = backend._compiled_ops["rda_step_op"]
    assert isinstance(op, MockRDA)
    created = [v for v in backend._compiled_ops.values() if isinstance(v, MockRDA)]
    assert len(created) == 1


def test_s1a_shared_entry_failure_does_not_start_b1(monkeypatch, tmp_path) -> None:
    """1절 고침: 공유 항목 얻기가 실패하면 B1 은 시작하지 않는다."""
    backend = _minimal_backend(monkeypatch, tmp_path)

    op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="shared"
        ),
        insts_artifact=SimpleNamespace(filename="shared.bin"),
    )
    shared_fn = SimpleNamespace(op=op)
    gu = FakeTemplateKernel("gu", backend._unified_xclbin_path)
    dn = FakeTemplateKernel("dn", backend._unified_xclbin_path)
    backend._compiled_ops["unified_moe_kernels"] = (gu, dn)
    monkeypatch.setattr(backend, "_get_or_compile_moe_ops", lambda: (object(), shared_fn, object()))
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: object())
    monkeypatch.setattr(backend, "_decode_moe_eligible", lambda *_args: (True, ""))
    events = []

    class Direct:
        def gemv_entry(self, *_args):
            raise RuntimeError("shared entry failed")

        def moe_start(self, *_args):
            events.append("moe_start")
            return "b1"

        def moe_finish(self, *_args):
            return np.zeros(6 * 2176, dtype=np.uint16)

        def gemv_start(self, *_args):
            events.append("gemv_start")
            return "shared"

        def drain(self, _runs):
            events.append("drain")

    monkeypatch.setattr(backend, "_decode_runner", lambda: Direct())
    args = (
        0, np.zeros(2560, np.uint16), np.arange(6, dtype=np.int32),
        np.full(6, 1 / 6, np.float32),
    )
    with pytest.raises(RuntimeError, match="shared entry failed"):
        backend.moe_experts(*args)
    assert "moe_start" not in events
    assert "drain" not in events


def test_s1b_moe_finish_waits_before_key_error(fake_world) -> None:
    """1절 고침(나): _moe 가 비어 있어도 moe_finish 는 먼저 run.wait() 를 부른다."""
    f = fake_world
    run = FakeRun(FakeKernel("b1", f.events), f.events, 7)
    f.runner._moe.clear()
    with pytest.raises(KeyError):
        f.runner.moe_finish("b1", run)
    assert run.waits == 1


# -----------------------------------------------------------------------------
# 6라운드: BO 를 미리 할당하고 한 번 쉬기
# -----------------------------------------------------------------------------

def test_r35_prepare_settles_once_and_first_calls_correct(fake_world) -> None:
    """6라운드: prepare 는 한 번만 쉬고 첫 GEMV·MoE·RDA 출력이 맞는다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    kernel = FakeTemplateKernel("moe", f.xclbin)
    gemv_weight = f.tensor(np.array([3], dtype=np.uint16))
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    dn_weight = f.tensor(np.array([4], dtype=np.uint16))
    gemv_specs = {"gemv": (f.op(), 4, 3)}
    moe_specs = {
        "b1": (kernel, nb.MAX_K, 2),
        "b2": (kernel, 7 * nb.MAX_K, 14),
    }
    loads_before_prepare = len(f.runtime.loads)
    f.runner.prepare(gemv_specs, moe_specs, f.op("rda"), 0.5)
    assert [e for e in f.events if e.startswith("sleep:")] == ["sleep:0.5"]
    # One GEMV command stream + B1/B2 MoE + RDA = 4 handle loads before the sleep.
    assert len(f.runtime.loads) - loads_before_prepare == 4

    loads_after_prepare = len(f.runtime.loads)
    entry = f.runner.gemv_entry("gemv", f.op(), gemv_weight, 4, 3)
    assert len(f.runtime.loads) == loads_after_prepare
    gemv_out = f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16))
    assert np.all(gemv_out == 13)

    b1_run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    b1_out = f.runner.moe_finish("b1", b1_run)
    assert b1_out.tolist() == [5, 5]

    rows7 = np.tile(np.array([1, 2], np.uint16), (7, 1))
    b2_run = f.runner.moe_start(
        "b2", 0, kernel, dn_weight, kernel.instructions(e0=1),
        rows7, 7 * nb.MAX_K, 14,
    )
    b2_out = f.runner.moe_finish("b2", b2_run)
    assert b2_out.tolist() == [25] * 14

    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    rda_out = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
    assert np.all(rda_out == 25600.0)


def test_r35b_prepare_settle_is_required(fake_world) -> None:
    """6라운드: prepare 의 쉬기를 빼면 가짜 결함 장치에서 첫 출력이 틀린다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    weight = f.tensor(np.array([3], dtype=np.uint16))
    gemv_specs = {"gemv": (f.op(), 4, 3)}
    # settle_s = 0 -> no sleep, BOs remain unsettled
    f.runner.prepare(gemv_specs, {}, None, 0.0)
    assert not any(e.startswith("sleep:") for e in f.events)

    entry = f.runner.gemv_entry("gemv", f.op(), weight, 4, 3)
    out = f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16))
    assert np.all(out == 0)


def test_r36_settle_time_env_var(monkeypatch, tmp_path, capsys) -> None:
    """6라운드: RAON_DECODE_DIRECT_SETTLE_MS 가 prepare 에 전달되는 settle 시간을 바꾼다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    monkeypatch.setenv("RAON_DECODE_DIRECT_SETTLE_MS", "250")
    assert backend._decode_settle_s() == 0.25

    monkeypatch.setenv("RAON_DECODE_DIRECT_SETTLE_MS", "not_a_number")
    assert backend._decode_settle_s() == 0.5
    assert "invalid RAON_DECODE_DIRECT_SETTLE_MS" in capsys.readouterr().err

    monkeypatch.setenv("RAON_DECODE_DIRECT_SETTLE_MS", "-10")
    assert backend._decode_settle_s() == 0.5


def test_r37_invalidation_reloads_all_and_settles_once(fake_world) -> None:
    """7라운드: 핸들 무효화 후에는 기억한 명세의 모든 핸들을 다시 로드하고 한 번 쉰다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([3], dtype=np.uint16))
    f.runner.prepare(
        {"b": (f.op(), 4, 3)},
        {"b1": (kernel, nb.MAX_K, 2)},
        None,
        0.5,
    )
    old = f.runner.gemv_entry("b", f.op(), weight, 4, 3)
    assert np.all(f.runner.gemv(old, np.array([1, 2, 3, 4], np.uint16)) == 13)

    sleeps_before = f.events.count("sleep:0.5")
    loads_before = len(f.runtime.loads)
    old.handle._is_valid = False
    new = f.runner.gemv_entry("b", f.op(), weight, 4, 3)
    assert new is not old
    assert new.x is old.x and new.y is old.y
    # The spec contains one GEMV stream and one MoE stream, so two handle loads
    # plus exactly one post-invalidation settle happen before the call resumes.
    assert len(f.runtime.loads) - loads_before == 2
    assert f.events.count("sleep:0.5") == sleeps_before + 1
    assert np.all(f.runner.gemv(new, np.array([1, 2, 3, 4], np.uint16)) == 13)


def test_r38_unprepared_bundle_allocates_settles_and_outputs(fake_world) -> None:
    """6라운드: 준비에 없던 묶음은 처음 부를 때 할당하고 settle 한 뒤 출력이 맞다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    weight = f.tensor(np.array([3], dtype=np.uint16))
    f.runner.prepare({}, {}, None, 0.5)
    assert f.events.count("sleep:0.5") == 1

    events_len_before = len(f.events)
    entry = f.runner.gemv_entry("unprepared", f.op(), weight, 4, 3)
    new_sleeps = [e for e in f.events[events_len_before:] if e.startswith("sleep:")]
    assert new_sleeps == ["sleep:0.5"]
    out = f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16))
    assert np.all(out == 13)


def test_r39_backend_prepares_once_and_preload_triggers(monkeypatch, tmp_path) -> None:
    """6라운드: 백엔드는 실행기를 만들 때 한 번 prepare 하고 preload 끝에서 트리거한다."""
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend._unified_xclbin_path = tmp_path / "u.xclbin"
    backend._unified_xclbin_path.touch()
    from kernels.moe.pack import B1_M_PER_EXPERT, B2_M_PER_EXPERT, NUM_ACTIVE_TOTAL

    prepared = []

    class FakeRunner:
        def prepare(self, gemv_specs, moe_specs, rda_op, settle_s):
            prepared.append((gemv_specs, moe_specs, rda_op, settle_s))

    monkeypatch.setattr(nb, "_make_decode_direct_runner", lambda _path: FakeRunner())
    monkeypatch.setattr(backend, "_decode_settle_s", lambda: 0.123)

    op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="gemv"
        ),
        insts_artifact=SimpleNamespace(filename="gemv.bin"),
    )
    fn = SimpleNamespace(op=op)
    monkeypatch.setattr(backend, "_get_or_compile_gemv", lambda *_args: fn)
    monkeypatch.setattr(backend, "_decode_op_eligible", lambda _op: (True, ""))
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: object())

    gu = FakeTemplateKernel("gu", backend._unified_xclbin_path)
    dn = FakeTemplateKernel("dn", backend._unified_xclbin_path)
    backend._compiled_ops["unified_moe_kernels"] = (gu, dn)
    monkeypatch.setattr(backend, "_decode_moe_eligible", lambda *_args: (True, ""))

    rda_op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="rda"
        ),
        insts_artifact=SimpleNamespace(filename="rda.bin"),
    )
    backend._compiled_ops["rda_step_op"] = rda_op
    monkeypatch.setattr(backend, "_decode_rda_eligible", lambda _op: (True, ""))

    # bundle_map entry to be included in the GEMV spec
    backend.bundle_map.clear()
    backend.bundle_map["L00/test"] = nb.BundleMeta(
        name="L00/test", rel_file="w.bin", fmt="q4nx_gemv",
        orig_shape=[3, nb.MAX_K], padded_shape=[3, nb.MAX_K],
        k=nb.MAX_K, m=3, components=[],
    )

    result = backend.preload()
    assert len(prepared) == 1
    gemv_specs, moe_specs, rda_op_out, settle_s = prepared[0]
    assert gemv_specs == {"L00/test": (op, nb.MAX_K, 3)}
    assert moe_specs == {
        "b1": (gu, nb.MAX_K, 6 * B1_M_PER_EXPERT),
        "b2": (dn, 7 * nb.MAX_K, NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT),
    }
    assert rda_op_out is rda_op
    assert settle_s == 0.123


def test_r40_no_warmup_first_call_starts_once(fake_world) -> None:
    """6라운드: 예열 실행이 없어 첫 호출의 run start 는 1번이다."""
    f = fake_world
    weight = f.tensor(np.array([3], dtype=np.uint16))
    entry = f.runner.gemv_entry("b", f.op(), weight, 4, 3)
    assert entry.run.starts == 0
    f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16))
    assert entry.run.starts == 1

    kernel = FakeTemplateKernel("moe", f.xclbin)
    run = f.runner.moe_start(
        "b1", 0, kernel, weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert run.starts == 1
    assert f.runner.moe_finish("b1", run).tolist() == [6, 6]


def test_r34_moe_new_layer_warmup_before_shared_gemv(monkeypatch, tmp_path) -> None:
    """2절 2.2.2: 공유 GEMV 항목 얻기 -> B1 moe_start -> 공유 gemv_start -> B1 moe_finish 순서를 지킨다.

    백엔드 수준: shared_gu 를 넘기면 공유 GEMV 항목 얻기/시작이 없다.
    """
    backend = _minimal_backend(monkeypatch, tmp_path)
    backend.decode_moe_overlap = False
    from kernels.moe.pack import B1_M_PER_EXPERT, B2_M_PER_EXPERT

    op = SimpleNamespace(
        xclbin_artifact=SimpleNamespace(
            filename=str(backend._unified_xclbin_path), kernel_name="shared"
        ),
        insts_artifact=SimpleNamespace(filename="shared.bin"),
    )
    shared_fn = SimpleNamespace(op=op)
    gu = FakeTemplateKernel("gu", backend._unified_xclbin_path)
    dn = FakeTemplateKernel("dn", backend._unified_xclbin_path)
    backend._compiled_ops["unified_moe_kernels"] = (gu, dn)
    monkeypatch.setattr(backend, "_get_or_compile_moe_ops", lambda: (object(), shared_fn, object()))
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: object())
    monkeypatch.setattr(backend, "_decode_moe_eligible", lambda *_args: (True, ""))
    events = []

    from kernels.moe.pack import NUM_ACTIVE_TOTAL

    class Direct:
        def gemv_entry(self, *_args):
            events.append("gemv_entry")
            return "shared-entry"

        def moe_start(self, key, *_args):
            events.append("moe_start")
            return f"{key}-run"

        def gemv_start(self, *_args):
            events.append("gemv_start")
            return "shared-run"

        def moe_finish(self, key, _run):
            events.append("moe_finish")
            size = (
                6 * B1_M_PER_EXPERT
                if key == "b1"
                else NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT
            )
            return np.zeros(size, dtype=np.uint16)

        def gemv_finish(self, *_args):
            events.append("gemv_finish")
            return np.zeros(B1_M_PER_EXPERT, dtype=np.uint16)

        def drain(self, _runs):
            events.append("drain")

    monkeypatch.setattr(backend, "_decode_runner", lambda: Direct())
    x = np.zeros(2560, dtype=np.uint16)
    ids = np.arange(6, dtype=np.int32)
    weights = np.full(6, 1 / 6, dtype=np.float32)

    out = backend.moe_experts(0, x, ids, weights)
    assert out.shape == (B2_M_PER_EXPERT,)
    assert events.index("gemv_entry") < events.index("moe_start")
    assert events.index("moe_start") < events.index("gemv_start")
    assert events.index("gemv_start") < events.index("moe_finish")

    events.clear()
    shared_gu = np.zeros(2176, dtype=np.uint16)
    out2 = backend.moe_experts(0, x, ids, weights, shared_gu=shared_gu)
    assert out2.shape == (B2_M_PER_EXPERT,)
    assert "gemv_entry" not in events
    assert "gemv_start" not in events
    assert "moe_start" in events


# -----------------------------------------------------------------------------
# 7라운드: 핸들도 미리 load, 무효화 뒤 한 번 쉬기, 시험 빈틈
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("next_call", ["gemv", "moe", "rda"])
def test_r41_invalidation_reloads_all_spec_handles_and_settles_once(
    fake_world, next_call
) -> None:
    """7라운드: 무효화 뒤 다음 최상위 호출은 기억한 명세 전체를 다시 로드하고 한 번 쉰다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    kernel = FakeTemplateKernel("moe", f.xclbin)
    gemv_weight = f.tensor(np.array([3], dtype=np.uint16))
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    gemv_specs = {"g": (f.op(), 4, 3)}
    moe_specs = {"b1": (kernel, nb.MAX_K, 2)}
    f.runner.prepare(gemv_specs, moe_specs, f.op("rda"), 0.5)

    entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
    assert np.all(f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) == 13)
    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]
    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    rda_out = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
    assert np.all(rda_out == 25600.0)

    loads_before = len(f.runtime.loads)
    sleeps_before = f.events.count("sleep:0.5")
    for handle in f.runner._handles.values():
        handle._is_valid = False

    if next_call == "gemv":
        new = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
        out = f.runner.gemv(new, np.array([1, 2, 3, 4], np.uint16))
        assert np.all(out == 13)
    elif next_call == "moe":
        run = f.runner.moe_start(
            "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
            np.array([1, 2], np.uint16), nb.MAX_K, 2,
        )
        out = f.runner.moe_finish("b1", run)
        assert out.tolist() == [5, 5]
    else:
        s_in = f.tensor(np.zeros((2,), np.float32))
        s_out = f.tensor(np.zeros((2,), np.float32))
        out = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
        assert np.all(out == 25600.0)

    assert len(f.runtime.loads) - loads_before == 3  # GEMV + MoE + RDA
    assert f.events.count("sleep:0.5") == sleeps_before + 1


def test_r42_post_invalidation_settle_is_required(fake_world) -> None:
    """7라운드: 무효화 뒤 쉬기를 빼면 새로 로드한 핸들의 명령열 BO 가 unsettled 여서 실패한다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([3], dtype=np.uint16))
    f.runner.prepare(
        {"g": (f.op(), 4, 3)}, {"b1": (kernel, nb.MAX_K, 2)}, f.op("rda"), 0.5
    )
    entry = f.runner.gemv_entry("g", f.op(), weight, 4, 3)
    assert np.all(f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) == 13)

    for handle in f.runner._handles.values():
        handle._is_valid = False
    # Disable the post-invalidation settle.  The reloaded handle is created but
    # not given time to settle, so the next execution must be dropped.
    f.runner._settle_s = 0.0
    new = f.runner.gemv_entry("g", f.op(), weight, 4, 3)
    bad = f.runner.gemv(new, np.array([1, 2, 3, 4], np.uint16))
    assert np.all(bad == 0)

    # Advancing time lets the already-reloaded handle settle without a new load.
    time.sleep(0.5)
    good = f.runner.gemv(new, np.array([1, 2, 3, 4], np.uint16))
    assert np.all(good == 13)


def test_s7_unprepared_moe_settles_once_and_outputs(fake_world) -> None:
    """7라운드: 준비에 없던 MoE 모양은 처음 부를 때 BO 를 할당하고 settle 한 뒤 맞는다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    f.runner.prepare({}, {}, None, 0.5)

    run = f.runner.moe_start(
        "b1", 0, kernel, weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]

    # A fresh runner asked to settle for 0 s keeps the new MoE BOs unsettled.
    runner2 = nb._DecodeDirectRunner(f.runtime, f.pyxrt, f.xclbin)
    runner2.prepare({}, {}, None, 0.0)
    run2 = runner2.moe_start(
        "b1", 0, kernel, weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    out2 = runner2.moe_finish("b1", run2)
    assert not np.array_equal(out2, np.array([5, 5], dtype=np.uint16))
    assert np.all(out2 == 0)


def test_s8_unprepared_rda_settles_once_and_outputs(fake_world) -> None:
    """7라운드: 준비에 없던 RDA 는 처음 부를 때 BO 를 할당하고 settle 한 뒤 맞는다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    weight = f.tensor(np.array([1], dtype=np.uint16))
    f.runner.prepare({}, {}, None, 0.5)

    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    out = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
    assert np.all(out == 25600.0)

    runner2 = nb._DecodeDirectRunner(f.runtime, f.pyxrt, f.xclbin)
    runner2.prepare({}, {}, None, 0.0)
    s_in2 = f.tensor(np.zeros((2,), np.float32))
    s_out2 = f.tensor(np.zeros((2,), np.float32))
    out2 = runner2.rda(f.op("rda"), s_in2, s_out2, np.full((40, 640), 1.0, np.float32))
    assert not np.array_equal(out2, np.full((40, 128), 25600.0, dtype=np.float32))
    assert np.all(out2 == 0)


def test_r43_moe_insts_bo_reallocated_to_context_kernel_group(fake_world) -> None:
    """8라운드: 무효화 뒤 MoE 명령열 BO 는 새 핸들의 kernel.group_id(1) 로 다시 할당되고 옛 BO 는 어떤 run 에도 묶이지 않는다."""
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    f.runner.prepare({}, {"b1": (kernel, nb.MAX_K, 2)}, None, 0.5)

    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]

    old_buf = f.runner._moe_bufs["b1"]
    old_handle = old_buf.handle
    old_insts = old_buf.insts
    old_insts_bo = old_insts.buffer_object()
    old_shape = f.runner._moe["b1"]
    old_run = old_shape.runs[0]
    assert old_run.args[1] is old_insts_bo

    # Invalidate handle
    old_handle._is_valid = False

    # Next MoE call reloads handle and reallocates the command-stream BO
    run2 = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run2).tolist() == [5, 5]

    new_buf = f.runner._moe_bufs["b1"]
    new_handle = new_buf.handle
    new_insts = new_buf.insts
    new_insts_bo = new_insts.buffer_object()
    new_shape = f.runner._moe["b1"]
    new_run = new_shape.runs[0]

    # (1) Handle is newly loaded
    assert new_handle is not old_handle
    # (2) Instruction BO is a brand-new BO
    assert new_insts is not old_insts
    assert new_insts_bo is not old_insts_bo
    # (3) Allocated with the new handle's kernel.group_id(1)
    assert new_insts.group_id == new_handle.kernel.group_id(1)
    assert new_insts.group_id != old_handle.kernel.group_id(1)
    # (4) New run binds the new instruction BO
    assert new_run is not old_run
    assert new_run.args[1] is new_insts_bo
    # (5) Old instruction BO is NOT bound to any run in the runner
    assert not any(old_insts_bo is r.args.get(1) for r in new_shape.runs.values())


def test_r44_xy_data_buffers_preserved_across_invalidation(fake_world) -> None:
    """8라운드: x·y 데이터 BO 는 무효화 뒤에도 같은 객체다 (GEMV·MoE·RDA)."""
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    gemv_weight = f.tensor(np.array([3], dtype=np.uint16))
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    gemv_specs = {"g": (f.op(), 4, 3)}
    moe_specs = {"b1": (kernel, nb.MAX_K, 2)}
    f.runner.prepare(gemv_specs, moe_specs, f.op("rda"), 0.5)

    entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
    assert np.all(f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) == 13)
    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]
    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    assert np.all(f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32)) == 25600.0)

    # Save original data BO references
    gemv_x = f.runner._gemv_bufs["g"].x
    gemv_y = f.runner._gemv_bufs["g"].y
    moe_x = f.runner._moe_bufs["b1"].x
    moe_y = f.runner._moe_bufs["b1"].y
    rda_head = f.runner._rda_buf.head
    rda_out = f.runner._rda_buf.out

    # Invalidate all handles
    for handle in f.runner._handles.values():
        handle._is_valid = False

    # Execute calls again
    new_entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
    assert np.all(f.runner.gemv(new_entry, np.array([1, 2, 3, 4], np.uint16)) == 13)
    new_run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", new_run).tolist() == [5, 5]
    s_in2 = f.tensor(np.zeros((2,), np.float32))
    s_out2 = f.tensor(np.zeros((2,), np.float32))
    assert np.all(f.runner.rda(f.op("rda"), s_in2, s_out2, np.full((40, 640), 1.0, np.float32)) == 25600.0)

    # Data BOs must remain the exact same objects
    assert f.runner._gemv_bufs["g"].x is gemv_x
    assert f.runner._gemv_bufs["g"].y is gemv_y
    assert f.runner._moe_bufs["b1"].x is moe_x
    assert f.runner._moe_bufs["b1"].y is moe_y
    assert f.runner._rda_buf.head is rda_head
    assert f.runner._rda_buf.out is rda_out


def test_r45_reallocated_moe_insts_bo_precedes_post_invalidation_settle(fake_world) -> None:
    """8라운드: 새 명령열 BO 할당이 무효화 뒤 쉬기보다 먼저다."""
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    f.runner.prepare({}, {"b1": (kernel, nb.MAX_K, 2)}, None, 0.5)

    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]

    for handle in f.runner._handles.values():
        handle._is_valid = False

    events_before = len(f.events)
    run2 = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run2).tolist() == [5, 5]

    new_events = f.events[events_before:]
    assert "alloc-tensor:insts" in new_events
    assert "sleep:0.5" in new_events
    alloc_idx = new_events.index("alloc-tensor:insts")
    sleep_idx = new_events.index("sleep:0.5")
    assert alloc_idx < sleep_idx, f"allocation index {alloc_idx} must precede sleep {sleep_idx}"


def test_r46_reallocated_moe_insts_settles_and_produces_correct_output(fake_world) -> None:
    """8라운드: 가짜 결함 장치에서 무효화 뒤 첫 MoE 출력이 맞고, 쉬기가 빠지면 출력 0으로 실패한다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    kernel = FakeTemplateKernel("moe", f.xclbin)
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    f.runner.prepare({}, {"b1": (kernel, nb.MAX_K, 2)}, None, 0.5)

    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]

    for handle in f.runner._handles.values():
        handle._is_valid = False

    # With normal settle (0.5s), reallocated insts BO is settled before execution
    run2 = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    out2 = f.runner.moe_finish("b1", run2)
    assert out2.tolist() == [5, 5]

    # Control / fault injection: if post-invalidation settle is omitted, newly allocated insts BO is unsettled
    for handle in f.runner._handles.values():
        handle._is_valid = False
    f.runner._settle_s = 0.0
    run_bad = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    bad_out = f.runner.moe_finish("b1", run_bad)
    assert np.all(bad_out == 0)


def test_r47_close_releases_all_runs_bos_handles_without_gc(fake_world) -> None:
    """8라운드: close() 뒤 실행기가 BO·핸들·run 을 하나도 붙들지 않는다 (weakref 로 gc.collect() 없이)."""
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    gemv_weight = f.tensor(np.array([3], dtype=np.uint16))
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))
    gemv_specs = {"g": (f.op(), 4, 3)}
    moe_specs = {"b1": (kernel, nb.MAX_K, 2)}
    f.runner.prepare(gemv_specs, moe_specs, f.op("rda"), 0.5)

    entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
    assert np.all(f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) == 13)
    moe_run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", moe_run).tolist() == [5, 5]
    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    assert np.all(f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32)) == 25600.0)

    gemv_run = entry.run
    rda_run = f.runner._rda.run
    gemv_x = entry.x
    gemv_y = entry.y
    moe_insts = f.runner._moe_bufs["b1"].insts
    moe_x = f.runner._moe_bufs["b1"].x
    moe_y = f.runner._moe_bufs["b1"].y
    rda_head = f.runner._rda_buf.head
    rda_out = f.runner._rda_buf.out
    gemv_handle = entry.handle
    moe_handle = f.runner._moe_bufs["b1"].handle
    rda_handle = f.runner._rda.handle

    refs = [
        weakref.ref(gemv_run),
        weakref.ref(moe_run),
        weakref.ref(rda_run),
        weakref.ref(gemv_x),
        weakref.ref(gemv_y),
        weakref.ref(moe_insts),
        weakref.ref(moe_x),
        weakref.ref(moe_y),
        weakref.ref(rda_head),
        weakref.ref(rda_out),
        weakref.ref(gemv_handle),
        weakref.ref(moe_handle),
        weakref.ref(rda_handle),
    ]

    # Delete local references and fixture tracker references
    del entry, moe_run, gemv_run, rda_run
    del gemv_x, gemv_y, moe_insts, moe_x, moe_y, rda_head, rda_out
    del gemv_handle, moe_handle, rda_handle
    f.runs.clear()

    # close() can be called twice safely
    f.runner.close()
    f.runner.close()

    # All objects must be released immediately without gc.collect()
    dead = [r() is None for r in refs]
    assert all(dead), f"Some objects remained alive: {[r() for r in refs if r() is not None]}"
    assert not f.runner._gemv
    assert not f.runner._moe
    assert f.runner._rda is None
    assert not f.runner._gemv_bufs
    assert not f.runner._moe_bufs
    assert f.runner._rda_buf is None
    assert not f.runner._handles
    assert not f.runner._pending_settle


# -----------------------------------------------------------------------------
# 9라운드: 명세 없는 무효화 경로도 새 핸들·BO 할당 뒤에 한 번 쉬기
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["gemv", "moe", "rda"])
@pytest.mark.parametrize("runner_mode", ["prepare_empty", "settle_s_direct"])
def test_r48_unprepared_invalidation_settles_after_alloc_and_outputs(
    fake_world, family: str, runner_mode: str
) -> None:
    """9라운드: 명세 없는 실행기에서 핸들 무효화 뒤 새 핸들·명령열 BO 할당이 쉬기보다 앞서고 정확히 한 번 쉰다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    if runner_mode == "prepare_empty":
        f.runner.prepare({}, {}, None, 0.5)
    else:
        f.runner._settle_s = 0.5

    kernel = FakeTemplateKernel("moe", f.xclbin)
    gemv_weight = f.tensor(np.array([3], dtype=np.uint16))
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))

    # GEMV·MoE·RDA 항목 최초 생성 및 1회 실행
    entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
    assert np.all(f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) == 13)

    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]

    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    assert np.all(
        f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32)) == 25600.0
    )

    # 정상 상태 호출 (무효화 없음): 쉬기가 0번이다
    sleeps_steady_before = f.events.count("sleep:0.5")
    if family == "gemv":
        out_steady = f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16))
        assert np.all(out_steady == 13)
    elif family == "moe":
        run_steady = f.runner.moe_start(
            "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
            np.array([1, 2], np.uint16), nb.MAX_K, 2,
        )
        assert f.runner.moe_finish("b1", run_steady).tolist() == [5, 5]
    else:
        out_steady = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
        assert np.all(out_steady == 25600.0)
    assert f.events.count("sleep:0.5") == sleeps_steady_before

    # 핸들을 무효로 만든다
    for handle in f.runner._handles.values():
        handle._is_valid = False

    events_before = len(f.events)
    sleeps_before = f.events.count("sleep:0.5")

    # 무효화 후 대상 계열의 다음 호출
    if family == "gemv":
        new_entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
        out = f.runner.gemv(new_entry, np.array([1, 2, 3, 4], np.uint16))
        assert np.all(out == 13)
    elif family == "moe":
        run = f.runner.moe_start(
            "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
            np.array([1, 2], np.uint16), nb.MAX_K, 2,
        )
        out = f.runner.moe_finish("b1", run)
        assert out.tolist() == [5, 5]
    else:
        out = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
        assert np.all(out == 25600.0)

    new_events = f.events[events_before:]

    # 1. 새 핸들 load (MoE 는 새 명령열 BO 할당까지)가 쉬기보다 먼저다 (사건 순서)
    assert f"load:{family}" in new_events
    assert "sleep:0.5" in new_events
    assert "start" in new_events
    assert new_events.index(f"load:{family}") < new_events.index("sleep:0.5")
    if family == "moe":
        assert "alloc-tensor:insts" in new_events
        assert new_events.index("alloc-tensor:insts") < new_events.index("sleep:0.5")
    assert new_events.index("sleep:0.5") < new_events.index("start")

    # 2. 쉬기는 정확히 한 번이다
    assert f.events.count("sleep:0.5") == sleeps_before + 1

    # 3. 가짜 결함 장치에서 첫 출력이 맞다 (위에서 이미 assert 통과)


@pytest.mark.parametrize("family", ["gemv", "moe", "rda"])
def test_r49_unprepared_invalidation_settle_is_required(fake_world, family: str) -> None:
    """9라운드: 무효화 뒤 쉬기를 빼면 가짜 결함 장치에서 첫 출력이 0으로 실패한다."""
    f = fake_world
    FakeBO._settle_duration = 0.5
    f.runner.prepare({}, {}, None, 0.5)

    kernel = FakeTemplateKernel("moe", f.xclbin)
    gemv_weight = f.tensor(np.array([3], dtype=np.uint16))
    moe_weight = f.tensor(np.array([2], dtype=np.uint16))

    entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
    assert np.all(f.runner.gemv(entry, np.array([1, 2, 3, 4], np.uint16)) == 13)

    run = f.runner.moe_start(
        "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
        np.array([1, 2], np.uint16), nb.MAX_K, 2,
    )
    assert f.runner.moe_finish("b1", run).tolist() == [5, 5]

    s_in = f.tensor(np.zeros((2,), np.float32))
    s_out = f.tensor(np.zeros((2,), np.float32))
    assert np.all(
        f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32)) == 25600.0
    )

    for handle in f.runner._handles.values():
        handle._is_valid = False

    # settle_s = 0.0 으로 쉬기를 생략하면 새로 할당된 핸들/BO 가 settle 되지 않아 출력이 0이 된다
    f.runner._settle_s = 0.0

    if family == "gemv":
        new_entry = f.runner.gemv_entry("g", f.op(), gemv_weight, 4, 3)
        bad_out = f.runner.gemv(new_entry, np.array([1, 2, 3, 4], np.uint16))
        assert np.all(bad_out == 0)
    elif family == "moe":
        bad_run = f.runner.moe_start(
            "b1", 0, kernel, moe_weight, kernel.instructions(e0=1),
            np.array([1, 2], np.uint16), nb.MAX_K, 2,
        )
        bad_out = f.runner.moe_finish("b1", bad_run)
        assert np.all(bad_out == 0)
    else:
        bad_out = f.runner.rda(f.op("rda"), s_in, s_out, np.full((40, 640), 1.0, np.float32))
        assert np.all(bad_out == 0)
