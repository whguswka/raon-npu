# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for decode MoE shared-GEMV and B2-command overlap."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from host.backend import NumpyBackend
from host.raon import RaonModel
import npu.backend as nb
from npu.test_decode_direct import FakeTemplateKernel, fake_world


class StopNow(BaseException):
    pass


class Run:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Direct:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.entry = SimpleNamespace(run=Run("shared-entry-run"))
        self.fail: str | None = None
        self.started: list[Run] = []
        self.completed: list[Run] = []
        self.drained: list[Run] = []
        self.b2_words: list[np.ndarray] = []
        self.staged_words: np.ndarray | None = None
        self.ready = True

    def overlap_ready(self, _entry):
        return self.ready

    def gemv_entry(self, *_args):
        self.events.append("gemv_entry")
        return self.entry

    def gemv_start(self, _entry, _x):
        self.events.append("gemv_start(shared)")
        if self.fail == "gemv_start":
            raise StopNow("gemv start")
        run = Run("shared")
        self.started.append(run)
        return run

    def gemv_finish(self, _entry, run):
        self.events.append("gemv_finish(shared)")
        self.completed.append(run)
        from kernels.moe.pack import B1_M_PER_EXPERT

        return np.arange(B1_M_PER_EXPERT, dtype=np.uint16)

    def moe_start(self, key, _layer, _kernel, _weight, words, *_args):
        self.events.append(f'moe_start("{key}")')
        if key == "b1" and self.fail == "b1_start":
            raise StopNow("b1 start")
        run = Run(key)
        self.started.append(run)
        if key == "b2":
            self.b2_words.append(words)
        return run

    def moe_stage(self, key, _layer, _kernel, _weight, words, *_args):
        assert key == "b2"
        self.events.append('moe_stage("b2")')
        if self.fail == "moe_stage":
            raise StopNow("moe stage")
        self.staged_words = words

    def moe_finish(self, key, run):
        self.events.append(f'moe_finish("{key}")')
        self.completed.append(run)
        if key == "b1" and self.fail == "b1_finish":
            raise StopNow("b1 finish")
        from kernels.moe.pack import B1_M_PER_EXPERT, B2_M_PER_EXPERT

        size = 6 * B1_M_PER_EXPERT if key == "b1" else 7 * B2_M_PER_EXPERT
        return np.arange(size, dtype=np.uint16)

    def drain(self, runs):
        for run in runs:
            self.events.append(f"drain({run.name})")
            self.drained.append(run)

    def moe_clear_staged(self):
        self.events.append("moe_clear_staged")
        self.staged_words = None


def _backend(monkeypatch: pytest.MonkeyPatch, tmp_path, direct: Direct | None = None):
    backend = nb.NpuBackend(mock=True, gemv="unified")
    backend.mock = False
    backend._unified_xclbin_path = tmp_path / "unified.xclbin"
    backend._unified_xclbin_path.parent.mkdir(parents=True, exist_ok=True)
    backend._unified_xclbin_path.touch()
    monkeypatch.setattr(nb, "HAVE_NPU", True)
    direct = direct or Direct()
    backend._compiled_ops["decode_direct"] = direct
    gu = FakeTemplateKernel("gu", backend._unified_xclbin_path)
    dn = FakeTemplateKernel("dn", backend._unified_xclbin_path)
    backend._compiled_ops["unified_moe_kernels"] = (gu, dn)
    shared = SimpleNamespace(op=object())
    monkeypatch.setattr(
        backend, "_get_or_compile_moe_ops", lambda: (object(), shared, object())
    )
    monkeypatch.setattr(backend, "_get_weight_tensor", lambda _name: object())
    monkeypatch.setattr(backend, "_decode_moe_eligible", lambda *_args: (True, ""))
    monkeypatch.setattr(backend, "_decode_runner", lambda: direct)
    return backend, direct


def _moe_args(layer: int = 3, x: np.ndarray | None = None):
    if x is None:
        x = np.arange(2560, dtype=np.uint16)
    return (
        layer,
        x,
        np.arange(6, dtype=np.int32),
        np.full(6, 1 / 6, dtype=np.float32),
    )


def _assert_once_for_every_started(direct: Direct) -> None:
    done = Counter(run.name for run in direct.completed + direct.drained)
    assert done == Counter(run.name for run in direct.started)


def test_enabled_prestart_order_reuses_shared_and_staged_words(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    args = _moe_args()
    backend.moe_prestart_shared(args[0], args[1])
    out = backend.moe_experts(*args)

    order = [
        "gemv_start(shared)",
        'moe_start("b1")',
        'moe_stage("b2")',
        'moe_finish("b1")',
        "gemv_finish(shared)",
        'moe_start("b2")',
    ]
    positions = [direct.events.index(event) for event in order]
    assert positions == sorted(positions)
    assert direct.events.count("gemv_start(shared)") == 1
    assert direct.b2_words == [direct.staged_words]
    assert backend._moe_shared_pre is None
    assert out.dtype == np.uint16


def test_runner_stage_skips_one_sync_only_for_same_object(fake_world) -> None:
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    words = kernel.instructions(e0=1)
    rows = np.zeros((7, 4), dtype=np.uint16)

    run = f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    f.runner.moe_finish("b2", run)
    base = f.events.count("sync-to:insts")
    f.runner.moe_stage("b2", 0, kernel, weight, words, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base + 1
    f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base + 1
    assert f.runner._moe["b2"].staged is None

    f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base + 2
    other = words.copy()
    f.runner.moe_stage("b2", 0, kernel, weight, words, 7 * nb.MAX_K, 7 * 2560)
    f.runner.moe_start("b2", 0, kernel, weight, other, rows, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base + 4


def test_runner_rebuild_and_clear_force_instruction_sync(fake_world) -> None:
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    words = kernel.instructions(e0=1)
    rows = np.zeros((7, 4), dtype=np.uint16)

    run = f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    f.runner.moe_finish("b2", run)
    base = f.events.count("sync-to:insts")
    f.runner.moe_stage("b2", 0, kernel, weight, words, 7 * nb.MAX_K, 7 * 2560)
    f.runner._moe["b2"].handle._is_valid = False
    f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base + 2

    f.runner.moe_stage("b2", 0, kernel, weight, words, 7 * nb.MAX_K, 7 * 2560)
    f.runner.moe_clear_staged()
    f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base + 4


@pytest.mark.parametrize("mismatch", ["layer", "entry", "input"])
def test_prestart_mismatch_drains_then_starts_shared(monkeypatch, tmp_path, mismatch) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    x = np.arange(2560, dtype=np.uint16)
    backend.moe_prestart_shared(3, x)
    old = direct.started[-1]
    layer, current = 3, x
    if mismatch == "layer":
        layer = 4
    elif mismatch == "entry":
        direct.entry = SimpleNamespace(run=Run("different-entry-run"))
    else:
        current = x.copy()
        current[0] ^= np.uint16(1)

    backend.moe_experts(*_moe_args(layer, current))
    assert direct.drained.count(old) == 1
    assert direct.events.count("gemv_start(shared)") == 2
    assert backend._moe_shared_pre is None


def test_next_prestart_drains_previous_before_second_start(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    x = np.arange(2560, dtype=np.uint16)
    backend.moe_prestart_shared(2, x)
    first = direct.started[-1]
    backend.moe_prestart_shared(2, x)
    assert direct.events.index("drain(shared)") < len(direct.events) - 1
    assert direct.events[-1] == "gemv_start(shared)"
    assert direct.drained == [first]


@pytest.mark.parametrize("failure", ["b1_start", "moe_stage", "b1_finish"])
def test_baseexception_drains_every_started_run(monkeypatch, tmp_path, failure) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    args = _moe_args()
    backend.moe_prestart_shared(args[0], args[1])
    direct.fail = failure
    with pytest.raises(StopNow, match=failure.replace("_", " ")):
        backend.moe_experts(*args)
    _assert_once_for_every_started(direct)
    assert backend._moe_shared_pre is None


def test_prestart_gemv_start_baseexception_drains_entry_run(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    direct.fail = "gemv_start"
    with pytest.raises(StopNow, match="gemv start"):
        backend.moe_prestart_shared(0, np.zeros(2560, dtype=np.uint16))
    assert direct.drained == [direct.entry.run]
    assert backend._moe_shared_pre is None


def test_disabled_preserves_old_order_and_bytes(monkeypatch, tmp_path) -> None:
    enabled, direct_on = _backend(monkeypatch, tmp_path / "on")
    args = _moe_args()
    enabled.moe_prestart_shared(args[0], args[1])
    out_on = enabled.moe_experts(*args)

    disabled, direct_off = _backend(monkeypatch, tmp_path / "off")
    disabled.decode_moe_overlap = False
    disabled.moe_prestart_shared(args[0], args[1])
    out_off = disabled.moe_experts(*args)
    assert direct_off.events.count("gemv_start(shared)") == 1
    assert 'moe_stage("b2")' not in direct_off.events
    assert direct_off.events.index('moe_start("b1")') < direct_off.events.index("gemv_start(shared)")
    assert direct_off.events.index("gemv_start(shared)") < direct_off.events.index('moe_finish("b1")')
    assert out_on.tobytes() == out_off.tobytes()


@pytest.mark.parametrize("value", ["0", "false", "False", "no"])
def test_overlap_environment_switch_and_setter(monkeypatch, value) -> None:
    monkeypatch.setenv("RAON_DECODE_MOE_OVERLAP", value)
    backend = nb.NpuBackend(mock=True)
    assert backend.decode_moe_overlap is False
    backend.decode_moe_overlap = True
    assert backend.decode_moe_overlap is True
    backend.decode_moe_overlap = None
    assert backend.decode_moe_overlap is False


def test_overlap_default_enabled(monkeypatch) -> None:
    monkeypatch.delenv("RAON_DECODE_MOE_OVERLAP", raising=False)
    assert nb.NpuBackend(mock=True).decode_moe_overlap is True


def test_model_prestarts_before_router_and_skips_when_shared() -> None:
    events: list[object] = []

    class ModelHooks:
        def moe_prestart_shared(self, layer, hidden):
            events.append(("pre", layer, hidden))

        def moe_abort_prestart_shared(self):
            events.append("abort")

        def moe_experts(self, layer, hidden, *_args, **_kwargs):
            events.append(("moe", layer, hidden))
            return np.array([7], dtype=np.uint16)

    model = object.__new__(RaonModel)
    model.backend = ModelHooks()
    model.config = SimpleNamespace(
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0
    )
    hidden = np.array([1], dtype=np.uint16)
    model._route_token = lambda layer, token: (
        events.append(("route", layer, token))
        or (np.arange(6), np.ones(6, np.float32), np.zeros(8, np.float32))
    )

    model._step_moe_router(5, hidden)
    assert events[0] == ("pre", 5, hidden)
    assert events[1] == ("route", 5, hidden)
    assert events[0][2] is hidden and events[1][2] is hidden
    events.clear()
    model._step_moe_router(5, hidden, shared_gu=np.zeros(2176, np.uint16))
    assert not any(isinstance(event, tuple) and event[0] == "pre" for event in events)


@pytest.mark.parametrize("site", ["prestart", "route", "moe"])
@pytest.mark.parametrize("error", [RuntimeError("failed"), StopNow("stopped")])
def test_model_aborts_once_and_reraises_same_exception(site, error) -> None:
    class ErrorHooks:
        aborts = 0

        def moe_prestart_shared(self, *_args):
            if site == "prestart":
                raise error

        def moe_abort_prestart_shared(self):
            self.aborts += 1

        def moe_experts(self, *_args, **_kwargs):
            if site == "moe":
                raise error
            return np.zeros(1, np.uint16)

    model = object.__new__(RaonModel)
    model.backend = ErrorHooks()
    model.config = SimpleNamespace(
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0
    )
    model._route_token = lambda *_args: (
        (_ for _ in ()).throw(error)
        if site == "route"
        else (np.arange(6), np.ones(6), np.zeros(8))
    )
    with pytest.raises(BaseException) as caught:
        model._step_moe_router(0, np.zeros(1, np.uint16))
    assert caught.value is error
    assert model.backend.aborts == 1


def test_numpy_backend_default_methods_are_noops() -> None:
    backend = object.__new__(NumpyBackend)
    before = dict(vars(backend))
    assert backend.moe_prestart_shared(0, np.zeros(1, np.uint16)) is None
    assert backend.moe_abort_prestart_shared() is None
    assert vars(backend) == before


def test_mock_noop_and_iron_fallback_drains_leftover(monkeypatch, tmp_path) -> None:
    mock = nb.NpuBackend(mock=True, gemv="unified")
    mock.moe_prestart_shared(0, np.zeros(2560, np.uint16))
    assert mock._moe_shared_pre is None

    backend, direct = _backend(monkeypatch, tmp_path)
    leftover = Run("leftover")
    backend._moe_shared_pre = (0, direct.entry, leftover, np.zeros(2560, np.uint16))
    monkeypatch.setattr(backend, "_decode_moe_eligible", lambda *_args: (False, "iron"))
    monkeypatch.setattr(backend, "_decode_fallback", lambda *_args: None)

    class Prepared(Exception):
        pass

    monkeypatch.setattr(backend, "_prepared_buffer", lambda _name: (_ for _ in ()).throw(Prepared()))
    with pytest.raises(Prepared):
        backend.moe_experts(*_moe_args())
    assert direct.drained == [leftover]
    assert backend._moe_shared_pre is None


def test_abort_reset_and_set_step_cleanup(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    direct.staged_words = np.array([1], np.uint32)
    pre = Run("pre-abort")
    backend._moe_shared_pre = (0, direct.entry, pre, np.zeros(2560, np.uint16))
    backend.moe_abort_prestart_shared()
    assert direct.drained == [pre]
    assert direct.staged_words is None

    pre_reset = Run("pre-reset")
    direct.staged_words = np.array([2], np.uint32)
    backend._moe_shared_pre = (0, direct.entry, pre_reset, np.zeros(2560, np.uint16))
    backend.reset()
    assert direct.drained[-1] is pre_reset
    assert direct.staged_words is None

    pre_step = Run("pre-step")
    backend._moe_shared_pre = (0, direct.entry, pre_step, np.zeros(2560, np.uint16))
    backend.set_step(9)
    assert direct.drained[-1] is pre_step
    assert backend._moe_shared_pre is None


@pytest.mark.parametrize(
    "failure_site",
    ["_get_or_compile_moe_ops", "_get_weight_tensor", "_decode_runner", "_decode_moe_eligible"],
)
def test_model_abort_drains_prestart_when_moe_setup_fails(
    monkeypatch, tmp_path, failure_site
) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    hidden = np.arange(2560, dtype=np.uint16)
    backend.moe_prestart_shared(0, hidden)
    pre = direct.started[-1]
    original = getattr(backend, failure_site)

    def fail(*_args, **_kwargs):
        raise RuntimeError(failure_site)

    monkeypatch.setattr(backend, failure_site, fail)
    model = object.__new__(RaonModel)
    model.backend = backend
    model.config = SimpleNamespace(
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0
    )
    model._route_token = lambda *_args: (
        np.arange(6), np.ones(6, np.float32), np.zeros(8, np.float32)
    )
    # Keep the already-started run: this invocation models entry after the router.
    monkeypatch.setattr(backend, "moe_prestart_shared", lambda *_args: None)
    with pytest.raises(RuntimeError, match=failure_site):
        model._step_moe_router(0, hidden)
    assert direct.drained.count(pre) == 1
    assert backend._moe_shared_pre is None
    monkeypatch.setattr(backend, failure_site, original)


def test_same_shared_run_is_drained_before_next_start(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    x = np.arange(2560, dtype=np.uint16)
    backend.moe_prestart_shared(0, x)
    backend.moe_prestart_shared(0, x)
    first_start = direct.events.index("gemv_start(shared)")
    drain = direct.events.index("drain(shared)")
    second_start = direct.events.index("gemv_start(shared)", first_start + 1)
    assert first_start < drain < second_start


def test_runner_stage_never_reloads_while_b1_in_flight(fake_world) -> None:
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    words = kernel.instructions(e0=1)
    rows = np.zeros((7, 4), dtype=np.uint16)

    b1_run = f.runner.moe_start("b1", 0, kernel, weight, words, rows, nb.MAX_K, 6 * 2560)
    f.runner.moe_finish("b1", b1_run)
    b2_run = f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    f.runner.moe_finish("b2", b2_run)

    loads_before = len(f.runtime.loads)
    b1_shape = f.runner._moe["b1"]

    b1_run = f.runner.moe_start("b1", 0, kernel, weight, words, rows, nb.MAX_K, 6 * 2560)
    syncs_before = f.events.count("sync-to:insts")
    f.runner._moe["b2"].handle._is_valid = False
    f.runner.moe_stage("b2", 0, kernel, weight, words, 7 * nb.MAX_K, 7 * 2560)

    assert len(f.runtime.loads) == loads_before
    assert f.runner._moe["b1"] is b1_shape
    assert f.events.count("sync-to:insts") == syncs_before
    assert f.runner._moe["b2"].staged is None

    f.runner.moe_finish("b1", b1_run)
    f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    assert len(f.runtime.loads) == loads_before + 1
    assert f.events.count("sync-to:insts") == syncs_before + 1


def test_runner_stage_skips_missing_shape_and_other_length(fake_world) -> None:
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    words = kernel.instructions(e0=1)
    rows = np.zeros((7, 4), dtype=np.uint16)

    loads_before = len(f.runtime.loads)
    f.runner.moe_stage("b2", 0, kernel, weight, words, 7 * nb.MAX_K, 7 * 2560)
    assert "b2" not in f.runner._moe
    assert len(f.runtime.loads) == loads_before
    assert f.events.count("sync-to:insts") == 0

    run = f.runner.moe_start("b2", 0, kernel, weight, words, rows, 7 * nb.MAX_K, 7 * 2560)
    f.runner.moe_finish("b2", run)
    base = f.events.count("sync-to:insts")
    other_words = np.concatenate([words, words])
    f.runner.moe_stage("b2", 0, kernel, weight, other_words, 7 * nb.MAX_K, 7 * 2560)
    assert f.events.count("sync-to:insts") == base
    assert f.runner._moe["b2"].staged is None


def test_runner_overlap_ready_is_side_effect_free(fake_world) -> None:
    f = fake_world
    kernel = FakeTemplateKernel("moe", f.xclbin)
    weight = f.tensor(np.array([2], dtype=np.uint16))
    words = kernel.instructions(e0=1)
    rows = np.zeros((7, 4), dtype=np.uint16)

    entry = f.runner.gemv_entry("bundle", f.op(), weight, 4, 3)
    assert f.runner.overlap_ready(entry) is False

    b1_run = f.runner.moe_start("b1", 0, kernel, weight, words, rows, nb.MAX_K, 6 * 2560)
    f.runner.moe_finish("b1", b1_run)
    assert f.runner.overlap_ready(entry) is True

    original_gemv = f.runner._gemv["bundle"]
    other_entry = f.runner.gemv_entry("other", f.op(), weight, 4, 3)
    f.runner._gemv["bundle"] = other_entry
    assert f.runner.overlap_ready(entry) is False
    f.runner._gemv["bundle"] = original_gemv
    assert f.runner.overlap_ready(entry) is True

    entry.handle._is_valid = False
    assert f.runner.overlap_ready(entry) is False
    entry.handle._is_valid = True
    assert f.runner.overlap_ready(entry) is True

    f.runner._moe["b1"].handle._is_valid = False
    assert f.runner.overlap_ready(entry) is False
    f.runner._moe["b1"].handle._is_valid = True
    assert f.runner.overlap_ready(entry) is True

    keys_before = set(f.runner._gemv) | set(f.runner._moe)
    loads_before = len(f.runtime.loads)
    f.runner._moe["b1"].handle._is_valid = False
    assert f.runner.overlap_ready(entry) is False
    assert set(f.runner._gemv) | set(f.runner._moe) == keys_before
    assert len(f.runtime.loads) == loads_before
    f.runner._moe["b1"].handle._is_valid = True

    f.runner.close()
    assert f.runner.overlap_ready(entry) is False


def test_prestart_not_ready_is_drained_before_any_handle_check(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    args = _moe_args()
    backend.moe_prestart_shared(args[0], args[1])
    direct.ready = False
    backend.moe_experts(*args)

    drain_idx = direct.events.index("drain(shared)")
    second_gemv_entry = direct.events.index("gemv_entry", drain_idx + 1)
    b1_start = direct.events.index('moe_start("b1")')
    assert drain_idx < second_gemv_entry < b1_start
    assert direct.events.count("gemv_start(shared)") == 2
    shared_starts = [i for i, e in enumerate(direct.events) if e == "gemv_start(shared)"]
    assert b1_start < shared_starts[-1]
    assert backend._moe_shared_pre is None
    _assert_once_for_every_started(direct)


def test_prestart_record_interrupt_drains_entry_run(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)

    def raising_setattr(obj, name, value):
        if name == "_moe_shared_pre" and value is not None:
            raise StopNow("record interrupt")
        object.__setattr__(obj, name, value)

    monkeypatch.setattr(nb.NpuBackend, "__setattr__", raising_setattr)
    with pytest.raises(StopNow, match="record interrupt"):
        backend.moe_prestart_shared(0, np.zeros(2560, dtype=np.uint16))
    assert direct.drained == [direct.entry.run]
    assert backend._moe_shared_pre is None


def test_model_abort_drains_prestart_when_shared_lookup_fails(monkeypatch, tmp_path) -> None:
    backend, direct = _backend(monkeypatch, tmp_path)
    hidden = np.zeros(2560, dtype=np.uint16)
    calls = [0]

    original_gemv_entry = direct.gemv_entry

    def failing_gemv_entry(*args):
        calls[0] += 1
        if calls[0] == 2:
            raise StopNow("shared lookup")
        return original_gemv_entry(*args)

    direct.gemv_entry = failing_gemv_entry

    model = object.__new__(RaonModel)
    model.backend = backend
    model.config = SimpleNamespace(
        activation_situ_beta=4.0, activation_situ_linear_beta=25.0
    )
    model._route_token = lambda *_args: (
        np.arange(6), np.ones(6, np.float32), np.zeros(8, np.float32)
    )

    with pytest.raises(StopNow, match="shared lookup"):
        model._step_moe_router(0, hidden)
    assert len(direct.started) == 1
    assert direct.drained == [direct.started[0]]
    assert backend._moe_shared_pre is None
