# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for decode RDA f_b GEMV linear_start overlap (DESIGN-prefill.md section 48)."""

from __future__ import annotations

import types
from typing import Any

import numpy as np
import pytest

from host.backend import NumpyBackend
from host.raon import RaonConfig, RaonModel
from host.weights import Weights, f32_to_bf16_rtne
import npu.backend as nb
from npu.test_decode_mla_group import FakeDirect, _backend, _mla_names


from npu.test_decode_mla_group import StopNow


def _rda_names(layer: int = 0):
    pfx = f"model.layers.{layer}.self_attn"
    return {
        "q": f"{pfx}.q_proj.weight",
        "k": f"{pfx}.k_proj.weight",
        "v": f"{pfx}.v_proj.weight",
        "g": f"{pfx}.g_proj.weight",
        "b": f"{pfx}.b_proj.weight",
        "rda_w": f"{pfx}.rda_w_proj.weight",
        "f_a": f"{pfx}.f_a_proj.weight",
        "f_b": f"{pfx}.f_b_proj.weight",
        "o": f"{pfx}.o_proj.weight",
    }


def _ls_backend(monkeypatch, tmp_path, runner=None):
    """_backend 를 감싸 linear_start/finish 테스트에 필요한 compiled_ops 와 mock 배열을 채운다."""
    backend, direct = _backend(monkeypatch, tmp_path, runner)
    backend.mock = True
    backend._compiled_ops["decode_direct"] = direct
    rng = np.random.default_rng(7)
    backend.mock_weight_arrays["L00/f_b"] = f32_to_bf16_rtne(rng.normal(size=(5120, 256)))
    return backend, direct


def _make_rda_weights(layer: int = 0) -> Weights:
    """Minimal in-memory weights for a single RDA layer."""
    rng = np.random.default_rng(42)
    w = object.__new__(Weights)
    w.tensors = {}
    w._f32_cache = {}
    pfx = f"model.layers.{layer}"
    spfx = f"{pfx}.self_attn"
    hidden_size = 2560
    num_heads = 40
    head_dim = 128
    decay_rank = 128
    proj_size = num_heads * head_dim

    w.tensors[f"{spfx}.q_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, hidden_size)))
    w.tensors[f"{spfx}.k_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, hidden_size)))
    w.tensors[f"{spfx}.v_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, hidden_size)))
    w.tensors[f"{spfx}.g_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, hidden_size)))
    w.tensors[f"{spfx}.f_a_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(decay_rank, hidden_size)))
    w.tensors[f"{spfx}.b_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(num_heads, hidden_size)))
    w.tensors[f"{spfx}.rda_w_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(num_heads, hidden_size)))
    w.tensors[f"{spfx}.f_b_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, decay_rank)))
    w.tensors[f"{spfx}.o_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(hidden_size, proj_size)))
    w.tensors[f"{spfx}.q_conv1d.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, 4)))
    w.tensors[f"{spfx}.k_conv1d.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, 4)))
    w.tensors[f"{spfx}.v_conv1d.weight"] = f32_to_bf16_rtne(rng.normal(size=(proj_size, 4)))
    w.tensors[f"{spfx}.dt_bias"] = f32_to_bf16_rtne(rng.normal(size=(num_heads, head_dim)))
    w.tensors[f"{spfx}.A_log"] = f32_to_bf16_rtne(rng.normal(size=(num_heads,)))
    w.tensors[f"{spfx}.o_norm.weight"] = f32_to_bf16_rtne(rng.normal(size=(head_dim,)))
    w.tensors[f"{pfx}.input_layernorm.weight"] = f32_to_bf16_rtne(rng.normal(size=(hidden_size,)))
    w.tensors[f"{pfx}.post_attention_layernorm.weight"] = f32_to_bf16_rtne(rng.normal(size=(hidden_size,)))
    return w


def test_start_then_finish_matches_linear(monkeypatch, tmp_path) -> None:
    """linear_start -> linear_finish 의 출력이 하나씩 linear 와 바이트 같고, 셈이 cache_hits +1 을 제외하고 같다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    f_a = np.arange(128, dtype=np.uint16)

    pending = backend.linear_start(name, f_a)
    assert direct.events == ["entry:L00/f_b", "start:L00/f_b"]
    out = backend.linear_finish(pending)

    ref_backend, _ = _ls_backend(monkeypatch, tmp_path / "ref")
    ref = ref_backend.linear(name, f_a)
    assert out.tobytes() == ref.tobytes()

    # finish 안의 cache-hit linear() 때문에 하나씩 경로보다 cache_hits 가 1 많다.
    assert backend.cache_hits == ref_backend.cache_hits + 1
    assert backend.cache_misses == ref_backend.cache_misses
    assert backend.call_counts["decode_direct_gemv"] == ref_backend.call_counts["decode_direct_gemv"]
    assert backend.call_counts["gemv:f_b"] == ref_backend.call_counts["gemv:f_b"]
    assert backend.call_counts["linear_npu"] == ref_backend.call_counts["linear_npu"]
    assert sum(backend.xclbin_calls.values()) == sum(ref_backend.xclbin_calls.values())


def test_start_input_padding_matches_linear(monkeypatch, tmp_path) -> None:
    """f_b 의 128->256 패딩이 시작 경로와 하나씩 linear 경로에서 바이트 같다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    backend.linear_start(name, x)
    group_input = direct.started[-1][1]

    backend.set_step(1)  # 캐시 비우고 linear 를 새로 계산
    backend.linear(name, x)
    single_input = direct.started[-1][1]

    assert group_input.shape == (256,)
    assert single_input.shape == (256,)
    assert group_input.tobytes() == single_input.tobytes()
    assert np.array_equal(group_input[128:], np.zeros(128, dtype=np.uint16))


@pytest.mark.parametrize(
    "setup",
    [
        "switch_off",
        "runner_none",
        "ndim2",
        "missing_name",
        "mla_kv_b",
        "shared_down",
        "entry_none",
        "cached",
    ],
)
def test_deferred_cases(monkeypatch, tmp_path, setup) -> None:
    """linear_start 가 GEMV 를 시작하지 않는 조건들: 토큰의 run 은 None, finish 는 linear 와 같다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    if setup == "switch_off":
        backend.decode_linear_start = False
    elif setup == "runner_none":
        monkeypatch.setattr(backend, "_decode_runner", lambda: None)
    elif setup == "ndim2":
        x = x.reshape(1, -1)
    elif setup == "missing_name":
        name = "does.not.exist.weight"
    elif setup == "mla_kv_b":
        mla = _mla_names(3)
        name = mla["kv_b"]
        x = np.arange(512, dtype=np.uint16)
    elif setup == "shared_down":
        name = "model.layers.1.moe.shared_experts.down_proj.weight"
        x = np.arange(2560, dtype=np.uint16)
    elif setup == "entry_none":
        direct._missing = "L00/f_b"
    elif setup == "cached":
        backend.linear(name, x)  # 캐시 채우기

    pending = backend.linear_start(name, x)
    assert pending.run is None

    # finish 는 linear(name, x) 를 부른다. 같은 백엔드에서 새 step 으로 다시 계산해 비교한다.
    backend.set_step(99)
    try:
        expected = backend.linear(name, x)
    except BaseException as exc:
        with pytest.raises(type(exc)):
            backend.linear_finish(pending)
        return

    out = backend.linear_finish(pending)
    assert out.tobytes() == expected.tobytes()


@pytest.mark.parametrize("site", ["start", "finish"])
def test_start_failure_drains_and_reraises(monkeypatch, tmp_path, site) -> None:
    """gemv_start / gemv_finish 예외 시 run 을 drain 하고 예외를 다시 올린다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    if site == "start":
        direct._fail_start = 0
        with pytest.raises(StopNow, match="f_b"):
            backend.linear_start(name, x)
        entry = direct.entries["L00/f_b"]
        assert direct.drained == [entry.run]
    else:
        pending = backend.linear_start(name, x)
        direct._fail_finish = 0
        with pytest.raises(StopNow, match="f_b"):
            backend.linear_finish(pending)
        assert direct.drained == [pending.entry.run]

    assert backend._linear_pending == []


def test_abort_drains_once_and_finish_after_drain_recomputes(monkeypatch, tmp_path) -> None:
    """abort 는 run 을 한 번 기다리고 제거; drain 된 토큰의 finish 는 linear 로 다시 계산한다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    pending = backend.linear_start(name, x)
    backend.linear_abort(pending)
    assert direct.drained == [pending.entry.run]
    backend.linear_abort(pending)  # 두 번째는 아무것도 안 함
    assert direct.drained == [pending.entry.run]

    backend.set_step(1)  # 다른 경로로 비우기
    out = backend.linear_finish(pending)

    ref_backend, _ = _ls_backend(monkeypatch, tmp_path / "ref")
    ref = ref_backend.linear(name, x)
    assert out.tobytes() == ref.tobytes()


@pytest.mark.parametrize(
    "fn_name",
    [
        "linear",
        "prefetch_linears",
        "moe_prestart_shared",
        "moe_experts",
        "rda_update",
        "linear_batch",
        "moe_experts_grouped",
        "rda_stage",
        "rda_launch",
        "set_step",
        "reset",
    ],
)
def test_entry_points_drain_pending_first(monkeypatch, tmp_path, fn_name) -> None:
    """각 진입점이 본문 작업보다 먼저 떠 있는 linear_start run 을 기다린다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)
    backend.linear_start(name, x)

    drained: list[bool] = []
    orig = backend._drain_linear_pending

    def wrap() -> None:
        drained.append(True)
        orig()

    monkeypatch.setattr(backend, "_drain_linear_pending", wrap)

    try:
        if fn_name == "linear":
            backend.linear("model.layers.3.self_attn.q_b_proj.weight", np.arange(576, dtype=np.uint16))
        elif fn_name == "prefetch_linears":
            backend.prefetch_linears(())
        elif fn_name == "moe_prestart_shared":
            backend.moe_prestart_shared(1, np.arange(2560, dtype=np.uint16))
        elif fn_name == "moe_experts":
            backend.moe_experts(
                1,
                np.arange(2560, dtype=np.uint16),
                np.array([0, 1, 2, 3, 4, 5]),
                np.ones(6, dtype=np.float32),
            )
        elif fn_name == "rda_update":
            backend.rda_update(
                0,
                np.zeros((40, 128), dtype=np.uint16),
                np.zeros((40, 128), dtype=np.uint16),
                np.zeros((40, 128), dtype=np.uint16),
                np.zeros((40, 128), dtype=np.float32),
                np.zeros(40, dtype=np.uint16),
            )
        elif fn_name == "linear_batch":
            backend.linear_batch(name, np.zeros((1, 128), dtype=np.uint16))
        elif fn_name == "moe_experts_grouped":
            backend.moe_experts_grouped(
                1,
                np.zeros((1, 2560), dtype=np.uint16),
                np.array([[0, 1, 2, 3, 4, 5]]),
                np.ones((1, 6), dtype=np.float32),
                np.zeros((1, 2176), dtype=np.uint16),
            )
        elif fn_name == "rda_stage":
            backend.rda_stage(
                0,
                [
                    (
                        np.zeros((40, 128), dtype=np.uint16),
                        np.zeros((40, 128), dtype=np.uint16),
                        np.zeros((40, 128), dtype=np.uint16),
                        np.zeros((40, 128), dtype=np.float32),
                        np.zeros(40, dtype=np.uint16),
                    )
                ],
            )
        elif fn_name == "rda_launch":
            fake_ticket = types.SimpleNamespace(
                phase="staged",
                io=0,
                layer=0,
                mock_inputs=[
                    (
                        np.zeros((40, 128), dtype=np.float32),
                        np.zeros((40, 128), dtype=np.float32),
                        np.zeros((40, 128), dtype=np.float32),
                        np.zeros((40, 128), dtype=np.float32),
                        np.zeros(40, dtype=np.float32),
                    )
                ],
            )
            backend.rda_launch(fake_ticket)
        elif fn_name == "set_step":
            backend.set_step(7)
        elif fn_name == "reset":
            backend.reset()
    except BaseException:
        pass

    assert drained


class _NoOverlapFakeDirect(FakeDirect):
    """이미 떠 있는 run 이 있으면 시작을 거부하는 가짜 실행기."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight: list[Any] = []

    def gemv_start(self, entry, x):
        if self.in_flight:
            raise AssertionError("previous run still in flight")
        run = super().gemv_start(entry, x)
        self.in_flight.append(run)
        return run

    def gemv_finish(self, entry, run):
        self.in_flight.remove(run)
        return super().gemv_finish(entry, run)

    def drain(self, runs):
        for run in runs:
            if run in self.in_flight:
                self.in_flight.remove(run)
        return super().drain(runs)


def test_second_start_drains_previous(monkeypatch, tmp_path) -> None:
    """끝내지 않은 시작이 있는데 다시 시작하면 앞의 run 을 먼저 기다린다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path, runner=_NoOverlapFakeDirect())
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    pending1 = backend.linear_start(name, x)
    pending2 = backend.linear_start(name, x)
    assert direct.drained == [pending1.entry.run]

    backend.set_step(1)
    out1 = backend.linear_finish(pending1)
    ref_backend, _ = _ls_backend(monkeypatch, tmp_path / "ref")
    ref = ref_backend.linear(name, x)
    assert out1.tobytes() == ref.tobytes()

    out2 = backend.linear_finish(pending2)
    assert out2.tobytes() == ref.tobytes()
    assert backend._linear_pending == []


@pytest.mark.parametrize("path", ["finish", "start_fail", "finish_fail", "abort", "set_step"])
def test_paths_leave_nothing_and_next_start_works(monkeypatch, tmp_path, path) -> None:
    """정상 끝/시작 실패/끝 실패/취소/set_step 비운 뒤에는 기록이 비고 이어서 시작·끝이 된다."""
    backend, direct = _ls_backend(monkeypatch, tmp_path)
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    pending = backend.linear_start(name, x)
    if path == "finish":
        backend.linear_finish(pending)
    elif path == "start_fail":
        direct._fail_start = 1
        with pytest.raises(StopNow):
            backend.linear_start(name, x)
    elif path == "finish_fail":
        direct._fail_finish = 0
        with pytest.raises(StopNow):
            backend.linear_finish(pending)
    elif path == "abort":
        backend.linear_abort(pending)
    elif path == "set_step":
        backend.set_step(1)

    assert backend._linear_pending == []

    backend.set_step(2)
    pending2 = backend.linear_start(name, x)
    out = backend.linear_finish(pending2)
    ref_backend, _ = _ls_backend(monkeypatch, tmp_path / "ref")
    ref = ref_backend.linear(name, x)
    assert out.tobytes() == ref.tobytes()
    assert backend._linear_pending == []


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0


class _FakeClockDirect(FakeDirect):
    def __init__(self, clock: _FakeClock) -> None:
        super().__init__()
        self.clock = clock

    def gemv_cached_entry(self, bundle: str) -> Entry | None:
        self.clock.now += 0.001
        return super().gemv_cached_entry(bundle)

    def gemv_start(self, entry: Entry, x: np.ndarray) -> Run:
        self.clock.now += 0.002
        return super().gemv_start(entry, x)

    def gemv_finish(self, entry: Entry, run: Run) -> np.ndarray:
        self.clock.now += 0.005
        return super().gemv_finish(entry, run)


def test_timing_with_fake_clock(monkeypatch, tmp_path) -> None:
    """가짜 시계로 시작-끝 사이 호스트 일은 timing 에 잡히지 않음을 확인한다."""
    clock = _FakeClock()
    monkeypatch.setattr(nb.time, "perf_counter", lambda: clock.now)
    backend, direct = _ls_backend(monkeypatch, tmp_path, runner=_FakeClockDirect(clock))
    name = _rda_names(0)["f_b"]
    x = np.arange(128, dtype=np.uint16)

    baseline_timings = dict(backend.timings)
    baseline_counts = dict(backend.call_counts)

    pending = backend.linear_start(name, x)
    clock.now += 0.100  # 호스트가 conv 등 다른 일을 하는 시간
    backend.linear_finish(pending)

    assert backend.call_counts["gemv:f_b"] == baseline_counts.get("gemv:f_b", 0) + 1
    assert backend.timings["gemv:f_b"] - baseline_timings.get("gemv:f_b", 0.0) == pytest.approx(0.007)
    assert backend.call_counts["linear_npu"] == baseline_counts.get("linear_npu", 0) + 1
    assert backend.timings["linear_npu"] - baseline_timings.get("linear_npu", 0.0) == pytest.approx(0.007)
    # 입력 준비 시간은 0.001 이며, 시작/끝 사이의 0.100 호스트 일은 timing 에 잡히지 않는다.
    assert backend.timings["linear_host"] - baseline_timings.get("linear_host", 0.0) == pytest.approx(0.001)
    assert backend.timings["host_prep"] - baseline_timings.get("host_prep", 0.0) == pytest.approx(0.001)


@pytest.mark.parametrize("value", ["0", "false", "False", "no"])
def test_switch_env_and_setter(monkeypatch, tmp_path, value) -> None:
    """RAON_DECODE_LINEAR_START 환경 변수·setter 우선순위."""
    monkeypatch.setenv("RAON_DECODE_LINEAR_START", value)
    backend = nb.NpuBackend(mock=True, gemv="unified")
    assert backend.decode_linear_start is False
    backend.decode_linear_start = True
    assert backend.decode_linear_start is True
    backend.decode_linear_start = None
    assert backend.decode_linear_start is False


def test_switch_default_enabled(monkeypatch) -> None:
    monkeypatch.delenv("RAON_DECODE_LINEAR_START", raising=False)
    assert nb.NpuBackend(mock=True, gemv="unified").decode_linear_start is True


def test_backend_defaults_are_linear() -> None:
    """NumpyBackend 의 linear_start/finish/abort 가 linear 와 같고 상태를 바꾸지 않는다."""
    w = object.__new__(Weights)
    w.tensors = {}
    w._f32_cache = {}
    backend = NumpyBackend(w)
    x = np.zeros(1, dtype=np.uint16)
    w.tensors["x.weight"] = np.zeros((1, 1), dtype=np.uint16)

    pending = backend.linear_start("x.weight", x)
    out = backend.linear_finish(pending)
    assert out.tobytes() == backend.linear("x.weight", x).tobytes()

    before = {k: v for k, v in vars(backend).items() if not callable(v)}
    backend.linear_abort(pending)
    after = {k: v for k, v in vars(backend).items() if not callable(v)}
    assert after == before


class _RecordingNumpy(NumpyBackend):
    """NumpyBackend that records linear/linear_start/finish/abort/rda_update calls."""

    def __init__(self, weights: Weights) -> None:
        super().__init__(weights)
        self.event_log: list[tuple[str, ...]] = []

    def linear_start(self, name: str, x: np.ndarray) -> Any:
        self.event_log.append(("linear_start", name, id(x)))
        return super().linear_start(name, x)

    def linear_finish(self, pending: Any) -> np.ndarray:
        name = pending[0] if isinstance(pending, tuple) else pending.name
        self.event_log.append(("linear_finish", name))
        return super().linear_finish(pending)

    def linear_abort(self, pending: Any) -> None:
        name = pending[0] if isinstance(pending, tuple) else pending.name
        self.event_log.append(("linear_abort", name))
        return super().linear_abort(pending)

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        out = super().linear(name, x)
        self.event_log.append(("linear", name, id(x), id(out)))
        return out

    def rda_update(self, layer, q, k, v, log_alpha, beta):
        self.event_log.append(("rda_update", layer))
        return super().rda_update(layer, q, k, v, log_alpha, beta)


def _old_forward_rda(model: RaonModel, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
    """바꾸기 전 _forward_rda 본문 (linear_start/finish 없음)."""
    pfx = f"model.layers.{layer_idx}.self_attn"
    q = model.backend.linear(f"{pfx}.q_proj.weight", hidden)
    k = model.backend.linear(f"{pfx}.k_proj.weight", hidden)
    v = model.backend.linear(f"{pfx}.v_proj.weight", hidden)
    gate = model.backend.linear(f"{pfx}.g_proj.weight", hidden)
    beta_raw = model.backend.linear(f"{pfx}.b_proj.weight", hidden)
    rda_delta = model.backend.linear(f"{pfx}.rda_w_proj.weight", hidden)
    f_a = model.backend.linear(f"{pfx}.f_a_proj.weight", hidden)
    decay_raw = model.backend.linear(f"{pfx}.f_b_proj.weight", f_a)
    gated = model._step_rda_core(layer_idx, q, k, v, gate, beta_raw, rda_delta, decay_raw)
    return model.backend.linear(f"{pfx}.o_proj.weight", gated)


def test_model_forward_rda_order_and_bytes(monkeypatch, tmp_path) -> None:
    """_forward_rda 의 호출 순서와 객체 동일성, 그리고 옛 본문과의 바이트 동일성."""
    cfg = RaonConfig.from_dict({"num_hidden_layers": 1, "rda_config": {"rda_layers": [1]}})
    weights = _make_rda_weights(0)
    rec = _RecordingNumpy(weights)
    model = RaonModel(cfg, weights, backend=rec)
    hidden = np.arange(2560, dtype=np.uint16)

    def wrap_conv(layer_idx, q, k, v, weight):
        rec.event_log.append(("conv", layer_idx))
        return RaonModel._conv_step_qkv(model, layer_idx, q, k, v, weight)

    monkeypatch.setattr(model, "_conv_step_qkv", wrap_conv)
    out = model._forward_rda(0, hidden)

    pfx = "model.layers.0.self_attn"
    expected = [
        ("linear", f"{pfx}.q_proj.weight"),
        ("linear", f"{pfx}.k_proj.weight"),
        ("linear", f"{pfx}.v_proj.weight"),
        ("linear", f"{pfx}.g_proj.weight"),
        ("linear", f"{pfx}.b_proj.weight"),
        ("linear", f"{pfx}.rda_w_proj.weight"),
        ("linear", f"{pfx}.f_a_proj.weight"),
        ("linear_start", f"{pfx}.f_b_proj.weight"),
        ("conv", 0),
        ("linear_finish", f"{pfx}.f_b_proj.weight"),
        ("linear", f"{pfx}.f_b_proj.weight"),
        ("rda_update", 0),
        ("linear", f"{pfx}.o_proj.weight"),
    ]
    assert [(e[0], e[1]) for e in rec.event_log] == expected
    # linear_start 에 전달된 x 는 f_a linear 의 출력 객체여야 한다.
    assert rec.event_log[7][2] == rec.event_log[6][3]

    old_model = RaonModel(cfg, weights, backend=NumpyBackend(weights))
    old_out = _old_forward_rda(old_model, 0, hidden)
    assert out.tobytes() == old_out.tobytes()


@pytest.mark.parametrize(
    "exc_class,exc_match",
    [
        (RuntimeError, "conv boom"),
        (StopNow, "conv stop"),
    ],
)
def test_model_forward_rda_aborts_on_conv_error(monkeypatch, tmp_path, exc_class, exc_match) -> None:
    """conv 가 예외를 내면 linear_abort 를 한 번 부르고 같은 예외가 올라온다."""
    cfg = RaonConfig.from_dict({"num_hidden_layers": 1, "rda_config": {"rda_layers": [1]}})
    weights = _make_rda_weights(0)
    rec = _RecordingNumpy(weights)
    model = RaonModel(cfg, weights, backend=rec)
    hidden = np.arange(2560, dtype=np.uint16)

    def boom(*args, **kwargs):
        raise exc_class(exc_match)

    monkeypatch.setattr(model, "_conv_step_qkv", boom)
    with pytest.raises(exc_class, match=exc_match):
        model._forward_rda(0, hidden)

    aborts = [e for e in rec.event_log if e[0] == "linear_abort"]
    assert len(aborts) == 1
    assert aborts[0][1] == "model.layers.0.self_attn.f_b_proj.weight"
    assert not [e for e in rec.event_log if e[0] == "linear_finish"]


def test_rda_inputs_split_is_byte_identical() -> None:
    """_rda_inputs(conv 포함) 와 conv + _rda_inputs_after_conv 의 출력·conv 상태가 바이트 같다."""
    cfg = RaonConfig.from_dict({"num_hidden_layers": 1, "rda_config": {"rda_layers": [1]}})
    weights = _make_rda_weights(0)
    model = RaonModel(cfg, weights, backend=NumpyBackend(weights))
    hidden = np.arange(2560, dtype=np.uint16)
    names = _rda_names(0)

    q = model.backend.linear(names["q"], hidden)
    k = model.backend.linear(names["k"], hidden)
    v = model.backend.linear(names["v"], hidden)
    beta_raw = model.backend.linear(names["b"], hidden)
    rda_delta = model.backend.linear(names["rda_w"], hidden)
    f_a = model.backend.linear(names["f_a"], hidden)
    decay_raw = model.backend.linear(names["f_b"], f_a)

    model.reset_state()
    full = model._rda_inputs(0, q, k, v, beta_raw, rda_delta, decay_raw)
    state_after_full = {kind: arr.copy() for kind, arr in model.conv_states[0].items()}

    model.reset_state()
    constants = model._layer_constants(0)
    q2, k2, v2 = model._conv_step_qkv(0, q, k, v, constants["conv_weight"])
    split = model._rda_inputs_after_conv(0, q2, k2, v2, beta_raw, rda_delta, decay_raw)
    state_after_split = {kind: arr.copy() for kind, arr in model.conv_states[0].items()}

    for a, b in zip(full, split):
        assert a.tobytes() == b.tobytes()

    assert set(state_after_full) == set(state_after_split)
    for kind in state_after_full:
        assert state_after_full[kind].tobytes() == state_after_split[kind].tobytes()
