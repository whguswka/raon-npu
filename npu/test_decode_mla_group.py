# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for decode MLA GEMV grouping (DESIGN-prefill.md section 47)."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from host.backend import Backend, NumpyBackend
from host.raon import RaonConfig, RaonModel, rms_norm
from host.weights import Weights, f32_to_bf16_rtne
import npu.backend as nb


class StopNow(BaseException):
    pass


class Run:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


_BUNDLE_DIMS: dict[str, int] = {
    "mla_in": 1152,
    "q_b": 7680,
    "k_b": 5120,
    "v_b": 5120,
    "g": 5120,
    "f_b": 5120,
    "o_proj": 2560,
    "dense_down": 2560,
    "lm_head": 260096,
    "shared_gu": 2176,
}


def _bundle_m(bundle: str) -> int:
    kind = bundle.split("/")[-1]
    return _BUNDLE_DIMS.get(kind, 1)


def _fake_gemv_output(bundle: str, x: np.ndarray, m: int) -> np.ndarray:
    """Deterministic output: a function of bundle name and input bytes only."""
    digest = hashlib.sha256(bundle.encode() + x.tobytes()).digest()
    seed = int.from_bytes(digest[:8], "little")
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2**16, size=m, dtype=np.uint16)


class Entry:
    def __init__(self, bundle: str) -> None:
        self.bundle = bundle
        self.k = 1  # not used by the fake; real shape is irrelevant for these tests
        self.m = _bundle_m(bundle)
        self.timing_key = f"gemv:{bundle.split('/')[-1]}"
        self.run = Run(bundle)


class FakeDirect:
    """Minimal decode direct runner for byte-identical grouping tests."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.entries: dict[str, Entry] = {}
        self.started: list[tuple[Entry, np.ndarray]] = []
        self.finished: list[Run] = []
        self.drained: list[Run] = []
        self._missing: str | None = None
        self._fail_start: int | None = None
        self._fail_finish: int | None = None

    def gemv_cached_entry(self, bundle: str) -> Entry | None:
        self.events.append(f"entry:{bundle}")
        if self._missing == bundle:
            return None
        if bundle not in self.entries:
            self.entries[bundle] = Entry(bundle)
        return self.entries[bundle]

    def gemv_start(self, entry: Entry, x: np.ndarray) -> Run:
        self.events.append(f"start:{entry.bundle}")
        self.started.append((entry, np.array(x, copy=True)))
        if self._fail_start == len(self.started) - 1:
            raise StopNow(f"start {entry.bundle}")
        return entry.run

    def gemv_finish(self, entry: Entry, run: Run) -> np.ndarray:
        self.events.append(f"finish:{entry.bundle}")
        self.finished.append(run)
        idx = len(self.finished) - 1
        if self._fail_finish == idx:
            raise StopNow(f"finish {entry.bundle}")
        _, x = self.started[idx]
        return _fake_gemv_output(entry.bundle, x, entry.m)

    def gemv(self, entry: Entry, x: np.ndarray) -> np.ndarray:
        run = self.gemv_start(entry, x)
        return self.gemv_finish(entry, run)

    def drain(self, runs: list[Any]) -> None:
        for run in runs:
            self.events.append(f"drain:{run}")
            self.drained.append(run)


def _backend(monkeypatch: pytest.MonkeyPatch, tmp_path, runner: FakeDirect | None = None):
    backend = nb.NpuBackend(mock=True, gemv="unified")
    backend.mock = False
    backend._unified_xclbin_path = tmp_path / "unified.xclbin"
    backend._unified_xclbin_path.parent.mkdir(parents=True, exist_ok=True)
    backend._unified_xclbin_path.touch()
    monkeypatch.setattr(nb, "HAVE_NPU", True)
    runner = runner or FakeDirect()
    monkeypatch.setattr(backend, "_decode_runner", lambda: runner)
    return backend, runner


def _mla_names(layer: int = 3):
    pfx = f"model.layers.{layer}.self_attn"
    return {
        "q_a": f"{pfx}.q_a_proj.weight",
        "kv_a": f"{pfx}.kv_a_proj_with_mqa.weight",
        "q_b": f"{pfx}.q_b_proj.weight",
        "kv_b": f"{pfx}.kv_b_proj.weight",
        "g": f"{pfx}.g_proj.weight",
        "o": f"{pfx}.o_proj.weight",
    }


def test_group_fetches_all_then_starts_all_then_finishes_in_order(monkeypatch, tmp_path) -> None:
    """(q_b, q_lat)·(kv_b, kv_lat) 묶음 → 항목 3 → 시작 3 → 끝 3, 사이에 적재 없음."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    q_lat = np.arange(576, dtype=np.uint16)
    kv_lat = np.arange(512, dtype=np.uint16)

    backend.prefetch_linears(((names["q_b"], q_lat), (names["kv_b"], kv_lat)))

    entries = [i for i, e in enumerate(direct.events) if e.startswith("entry:")]
    starts = [i for i, e in enumerate(direct.events) if e.startswith("start:")]
    finishes = [i for i, e in enumerate(direct.events) if e.startswith("finish:")]

    assert len(entries) == 3
    assert len(starts) == 3
    assert len(finishes) == 3
    assert max(entries) < min(starts)
    assert max(starts) < min(finishes)
    assert [direct.events[i].split("/", 1)[1] for i in finishes] == [
        "q_b", "k_b", "v_b",
    ]
    assert backend.call_counts["decode_mla_group"] == 1
    assert backend.call_counts["decode_direct_gemv"] == 3


def test_following_linears_hit_cache_with_same_bytes(monkeypatch, tmp_path) -> None:
    """묶음 경로와 하나씩 경로의 출력이 바이트 같고, 묶음 뒤 linear 는 GEMV 를 더 부르지 않는다."""
    hidden = np.arange(2560, dtype=np.uint16)
    q_lat = np.arange(576, dtype=np.uint16)
    kv_lat = np.arange(512, dtype=np.uint16)
    names = _mla_names(3)

    backend_a, direct_a = _backend(monkeypatch, tmp_path / "a")
    first = [(names["q_a"], hidden)]
    first.append((names["g"], hidden))
    backend_a.prefetch_linears(tuple(first))
    backend_a.prefetch_linears(((names["q_b"], q_lat), (names["kv_b"], kv_lat)))
    out_a = {
        "q_a": backend_a.linear(names["q_a"], hidden),
        "kv_a": backend_a.linear(names["kv_a"], hidden),
        "g": backend_a.linear(names["g"], hidden),
        "q_b": backend_a.linear(names["q_b"], q_lat),
        "kv_b": backend_a.linear(names["kv_b"], kv_lat),
    }

    backend_b, direct_b = _backend(monkeypatch, tmp_path / "b")
    out_b = {
        "q_a": backend_b.linear(names["q_a"], hidden),
        "kv_a": backend_b.linear(names["kv_a"], hidden),
        "g": backend_b.linear(names["g"], hidden),
        "q_b": backend_b.linear(names["q_b"], q_lat),
        "kv_b": backend_b.linear(names["kv_b"], kv_lat),
    }

    for key in out_a:
        assert out_a[key].tobytes() == out_b[key].tobytes()

    assert backend_a.cache_hits == backend_b.cache_hits + 5
    assert backend_a.cache_misses == backend_b.cache_misses + 2

    def gemv_sum(be):
        return sum(v for k, v in be.call_counts.items() if k.startswith("gemv:"))

    assert gemv_sum(backend_a) == gemv_sum(backend_b)
    assert backend_a.call_counts["decode_direct_gemv"] == backend_b.call_counts["decode_direct_gemv"]
    assert backend_a.call_counts["linear_npu"] == backend_b.call_counts["linear_npu"]
    assert sum(backend_a.xclbin_calls.values()) == sum(backend_b.xclbin_calls.values())
    assert direct_a.finished and direct_b.finished
    assert direct_a.finished == [direct_a.entries[b].run for b in direct_a.entries]


def test_kv_b_uses_prefetched_k_b_v_b_only_for_same_object_and_step(monkeypatch, tmp_path) -> None:
    """같은 바이트의 다른 객체나 set_step 뒤에는 kv_b 가 k_b·v_b GEMV 를 새로 실행한다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    kv_lat = np.arange(512, dtype=np.uint16)

    backend.prefetch_linears(((names["kv_b"], kv_lat),))
    base = len(direct.finished)

    kv_lat_copy = kv_lat.copy()
    backend.linear(names["kv_b"], kv_lat_copy)
    assert len(direct.finished) == base + 2

    backend.set_step(7)
    backend.linear(names["kv_b"], kv_lat)
    assert len(direct.finished) == base + 4


def test_one_start_per_bundle_and_cached_bundles_skipped(monkeypatch, tmp_path) -> None:
    """같은 mla_in 묶음은 한 번만 시작하고, 캐시·객체 규칙에 따라 건너뛴다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    hidden1 = np.arange(2560, dtype=np.uint16)
    hidden2 = hidden1.copy()

    backend.prefetch_linears(((names["q_a"], hidden1), (names["kv_a"], hidden1)))
    assert direct.events.count("start:L03/mla_in") == 1
    backend.linear(names["q_a"], hidden1)
    backend.linear(names["kv_a"], hidden1)
    assert backend.cache_hits == 2

    backend.prefetch_linears(((names["q_a"], hidden1), (names["kv_a"], hidden2)))
    assert direct.events.count("start:L03/mla_in") == 2
    backend.linear(names["kv_a"], hidden2)
    assert backend.cache_hits == 3

    backend.set_step(2)
    backend.prefetch_linears(((names["q_a"], hidden1), (names["kv_a"], hidden1)))
    assert direct.events.count("start:L03/mla_in") == 3
    backend.prefetch_linears(((names["q_a"], hidden1), (names["kv_a"], hidden1)))
    assert direct.events.count("start:L03/mla_in") == 3
    assert backend.call_counts["decode_mla_group"] == 3


def test_same_bundle_different_objects_in_one_call_start_once(monkeypatch, tmp_path) -> None:
    """같은 묶음 안에서 입력 객체가 다르면 첫 객체만 묶음으로 시작하고, 둘째는 캐시 규칙에 맡긴다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    hidden1 = np.arange(2560, dtype=np.uint16)
    hidden2 = hidden1.copy()

    backend.prefetch_linears(((names["q_a"], hidden1), (names["kv_a"], hidden2)))
    assert direct.events.count("start:L03/mla_in") == 1
    assert direct.started[-1][1].tobytes() == hidden1.tobytes()

    out_q = backend.linear(names["q_a"], hidden1)
    assert direct.events.count("start:L03/mla_in") == 1  # cache hit

    misses_before = backend.cache_misses
    out_kv = backend.linear(names["kv_a"], hidden2)
    assert backend.cache_misses == misses_before + 1
    assert direct.events.count("start:L03/mla_in") == 2  # one-by-one path

    ref_backend, _ = _backend(monkeypatch, tmp_path / "ref")
    ref_q = ref_backend.linear(names["q_a"], hidden1)
    ref_kv = ref_backend.linear(names["kv_a"], hidden2)

    assert out_q.tobytes() == ref_q.tobytes()
    assert out_kv.tobytes() == ref_kv.tobytes()


@pytest.mark.parametrize(
    "setup",
    [
        "entry_none",
        "ndim2",
        "missing_name",
        "shared_down",
        "switch_off",
        "runner_none",
    ],
)
def test_noop_cases(monkeypatch, tmp_path, setup) -> None:
    """조건 하나라도 깨지면 prefetch_linears 는 아무 것도 시작하지 않는다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    hidden = np.arange(2560, dtype=np.uint16)
    kv_lat = np.arange(512, dtype=np.uint16)
    cache_before = dict(backend._bundle_cache)
    counts_before = dict(backend.call_counts)

    items = [(names["g"], hidden)]
    if setup == "entry_none":
        direct._missing = "L03/k_b"
        items.append((names["kv_b"], kv_lat))
    elif setup == "ndim2":
        items.append((names["g"], hidden.reshape(1, -1)))
    elif setup == "missing_name":
        items.append(("does.not.exist.weight", hidden))
    elif setup == "shared_down":
        items.append(("model.layers.1.moe.shared_experts.down_proj.weight", hidden))
    elif setup == "switch_off":
        backend.decode_mla_group = False
    elif setup == "runner_none":
        monkeypatch.setattr(backend, "_decode_runner", lambda: None)

    backend.prefetch_linears(tuple(items))

    assert not any(e.startswith("start:") for e in direct.events)
    assert backend._bundle_cache == cache_before
    assert dict(backend.call_counts) == counts_before


@pytest.mark.parametrize("site", ["start", "finish"])
def test_start_failure_drains_started_and_failing_and_finish_failure_drains_rest(
    monkeypatch, tmp_path, site
) -> None:
    """(q_b, kv_b) 묶음에서 둘째 GEMV 의 시작/끝 예외가 올라오고 남은 run 을 drain 한다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    q_lat = np.arange(576, dtype=np.uint16)
    kv_lat = np.arange(512, dtype=np.uint16)

    if site == "start":
        direct._fail_start = 1
    else:
        direct._fail_finish = 1

    with pytest.raises(StopNow, match="k_b"):
        backend.prefetch_linears(((names["q_b"], q_lat), (names["kv_b"], kv_lat)))

    entries = [direct.entries[b] for b in ("L03/q_b", "L03/k_b", "L03/v_b")]
    if site == "start":
        assert direct.drained == [entries[0].run, entries[1].run]
    else:
        assert direct.drained == [entries[1].run, entries[2].run]


@pytest.mark.parametrize("value", ["0", "false", "False", "no"])
def test_switch_env_and_setter(monkeypatch, tmp_path, value) -> None:
    """RAON_DECODE_MLA_GROUP 환경 변수·setter 우선순위."""
    monkeypatch.setenv("RAON_DECODE_MLA_GROUP", value)
    backend = nb.NpuBackend(mock=True, gemv="unified")
    assert backend.decode_mla_group is False
    backend.decode_mla_group = True
    assert backend.decode_mla_group is True
    backend.decode_mla_group = None
    assert backend.decode_mla_group is False


def test_switch_default_enabled(monkeypatch) -> None:
    monkeypatch.delenv("RAON_DECODE_MLA_GROUP", raising=False)
    assert nb.NpuBackend(mock=True, gemv="unified").decode_mla_group is True


def test_numpy_backend_prefetch_is_noop() -> None:
    """NumpyBackend 의 prefetch_linears 는 상태를 바꾸지 않는다."""
    w = object.__new__(Weights)
    w.tensors = {}
    w._f32_cache = {}
    backend = NumpyBackend(w)
    before = {k: v for k, v in vars(backend).items() if not callable(v)}
    assert backend.prefetch_linears((("x", np.zeros(1, dtype=np.uint16)),)) is None
    after = {k: v for k, v in vars(backend).items() if not callable(v)}
    assert after == before


class _RecordingNumpy(NumpyBackend):
    """NumpyBackend that records prefetch/linear calls in one ordered log."""

    def __init__(self, weights: Weights) -> None:
        super().__init__(weights)
        self.event_log: list[tuple[str, ...]] = []

    def prefetch_linears(self, items: tuple[tuple[str, np.ndarray], ...]) -> None:
        self.event_log.append(("prefetch", tuple((n, id(x)) for n, x in items)))

    def linear(self, name: str, x: np.ndarray) -> np.ndarray:
        self.event_log.append(("linear", name, id(x)))
        return super().linear(name, x)


def _make_mla_weights(layer: int = 0) -> Weights:
    """Minimal in-memory weights for a single MLA layer."""
    rng = np.random.default_rng(42)
    w = object.__new__(Weights)
    w.tensors = {}
    w._f32_cache = {}
    pfx = f"model.layers.{layer}.self_attn"
    c = {
        "hidden_size": 2560,
        "q_lora_rank": 576,
        "kv_lora_rank": 512,
        "num_attention_heads": 40,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
    }
    w.tensors[f"{pfx}.q_a_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["q_lora_rank"], c["hidden_size"])))
    w.tensors[f"{pfx}.q_b_proj.weight"] = f32_to_bf16_rtne(
        rng.normal(size=(c["num_attention_heads"] * (c["qk_nope_head_dim"] + c["qk_rope_head_dim"]), c["q_lora_rank"]))
    )
    w.tensors[f"{pfx}.kv_a_proj_with_mqa.weight"] = f32_to_bf16_rtne(
        rng.normal(size=(c["kv_lora_rank"] + c["qk_rope_head_dim"], c["hidden_size"]))
    )
    w.tensors[f"{pfx}.kv_b_proj.weight"] = f32_to_bf16_rtne(
        rng.normal(size=(c["num_attention_heads"] * (c["qk_nope_head_dim"] + c["v_head_dim"]), c["kv_lora_rank"]))
    )
    w.tensors[f"{pfx}.g_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["num_attention_heads"] * c["v_head_dim"], c["hidden_size"])))
    w.tensors[f"{pfx}.o_proj.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["hidden_size"], c["num_attention_heads"] * c["v_head_dim"])))
    w.tensors[f"{pfx}.q_a_layernorm.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["q_lora_rank"],)))
    w.tensors[f"{pfx}.kv_a_layernorm.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["kv_lora_rank"],)))
    w.tensors[f"model.layers.{layer}.input_layernorm.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["hidden_size"],)))
    w.tensors[f"model.layers.{layer}.post_attention_layernorm.weight"] = f32_to_bf16_rtne(rng.normal(size=(c["hidden_size"],)))
    return w


def _old_forward_mla(model: RaonModel, layer_idx: int, hidden: np.ndarray) -> np.ndarray:
    """바꾸기 전 _forward_mla 본문 (prefetch_linears 호출 없음)."""
    pfx = f"model.layers.{layer_idx}.self_attn"
    cfg = model.config
    heads = cfg.num_attention_heads
    qk_nope = cfg.qk_nope_head_dim
    qk_pe = cfg.qk_rope_head_dim
    v_dim = cfg.v_head_dim
    q_head_dim = qk_nope + qk_pe

    q_lat_raw = model.backend.linear(f"{pfx}.q_a_proj.weight", hidden)
    constants = model._layer_constants(layer_idx)
    q_lat = rms_norm(q_lat_raw, constants["q_a_norm"], eps=cfg.mla_latent_norm_eps)
    q = model.backend.linear(f"{pfx}.q_b_proj.weight", q_lat).reshape(heads, q_head_dim)

    kv_full = model.backend.linear(f"{pfx}.kv_a_proj_with_mqa.weight", hidden)
    kv_lat_raw = kv_full[: cfg.kv_lora_rank]
    k_pe = kv_full[cfg.kv_lora_rank :]
    kv_lat = rms_norm(kv_lat_raw, constants["kv_a_norm"], eps=cfg.mla_latent_norm_eps)
    kv = model.backend.linear(f"{pfx}.kv_b_proj.weight", kv_lat).reshape(heads, qk_nope + v_dim)
    k_nope = kv[:, :qk_nope]
    value = kv[:, qk_nope:]

    gate = (
        model.backend.linear(f"{pfx}.g_proj.weight", hidden)
        if cfg.mla_use_output_gate
        else None
    )
    context = model._step_mla_attend(layer_idx, q, k_nope, k_pe, value, gate)
    return model.backend.linear(f"{pfx}.o_proj.weight", context)


def test_model_forward_mla_calls_prefetch_in_place_and_keeps_bytes() -> None:
    """_forward_mla 의 prefetch/linear 순서와 객체 동일성, 그리고 바이트 불변성."""
    cfg = RaonConfig.from_dict({"num_hidden_layers": 1, "rda_config": {"rda_layers": []}})
    weights = _make_mla_weights(0)
    rec = _RecordingNumpy(weights)
    model = RaonModel(cfg, weights, backend=rec)
    hidden = np.arange(2560, dtype=np.uint16)

    out = model._forward_mla(0, hidden)

    pfx = "model.layers.0.self_attn"
    q_a = f"{pfx}.q_a_proj.weight"
    kv_a = f"{pfx}.kv_a_proj_with_mqa.weight"
    q_b = f"{pfx}.q_b_proj.weight"
    kv_b = f"{pfx}.kv_b_proj.weight"
    g = f"{pfx}.g_proj.weight"
    o = f"{pfx}.o_proj.weight"

    def _event_names(log):
        return [
            ("prefetch", tuple(n for n, _ in e[1])) if e[0] == "prefetch" else e[1]
            for e in log
        ]

    assert _event_names(rec.event_log) == [
        ("prefetch", (q_a, g)),
        q_a,
        kv_a,
        ("prefetch", (q_b, kv_b)),
        q_b,
        kv_b,
        g,
        o,
    ]

    hidden_id = id(hidden)
    q_lat_id = rec.event_log[3][1][0][1]
    kv_lat_id = rec.event_log[3][1][1][1]
    linear_ids = {e[1]: e[2] for e in rec.event_log if e[0] == "linear"}
    assert rec.event_log[0][1][0][1] == hidden_id
    assert rec.event_log[0][1][1][1] == hidden_id
    assert linear_ids[q_a] == hidden_id
    assert linear_ids[kv_a] == hidden_id
    assert linear_ids[q_b] == q_lat_id
    assert linear_ids[kv_b] == kv_lat_id
    assert linear_ids[g] == hidden_id

    np_model = RaonModel(cfg, weights, backend=NumpyBackend(weights))
    out_old = _old_forward_mla(np_model, 0, hidden)
    assert out.tobytes() == out_old.tobytes()


def test_model_forward_mla_gate_off_prefetch_and_bytes() -> None:
    """mla_use_output_gate=False 일 때 prefetch 첫 묶음에 g_proj 를 끼지 않고 바이트를 유지한다."""
    cfg = RaonConfig.from_dict(
        {
            "num_hidden_layers": 1,
            "rda_config": {"rda_layers": []},
            "mla_use_output_gate": False,
        }
    )
    weights = _make_mla_weights(0)
    del weights.tensors["model.layers.0.self_attn.g_proj.weight"]
    rec = _RecordingNumpy(weights)
    model = RaonModel(cfg, weights, backend=rec)
    hidden = np.arange(2560, dtype=np.uint16)

    out = model._forward_mla(0, hidden)

    pfx = "model.layers.0.self_attn"
    q_a = f"{pfx}.q_a_proj.weight"
    kv_a = f"{pfx}.kv_a_proj_with_mqa.weight"
    q_b = f"{pfx}.q_b_proj.weight"
    kv_b = f"{pfx}.kv_b_proj.weight"
    o = f"{pfx}.o_proj.weight"

    def _event_names(log):
        return [
            ("prefetch", tuple(n for n, _ in e[1])) if e[0] == "prefetch" else e[1]
            for e in log
        ]

    assert _event_names(rec.event_log) == [
        ("prefetch", (q_a,)),
        q_a,
        kv_a,
        ("prefetch", (q_b, kv_b)),
        q_b,
        kv_b,
        o,
    ]

    hidden_id = id(hidden)
    q_lat_id = rec.event_log[3][1][0][1]
    kv_lat_id = rec.event_log[3][1][1][1]
    linear_ids = {e[1]: e[2] for e in rec.event_log if e[0] == "linear"}
    assert rec.event_log[0][1][0][1] == hidden_id
    assert linear_ids[q_a] == hidden_id
    assert linear_ids[kv_a] == hidden_id
    assert linear_ids[q_b] == q_lat_id
    assert linear_ids[kv_b] == kv_lat_id

    np_model = RaonModel(cfg, weights, backend=NumpyBackend(weights))
    out_old = _old_forward_mla(np_model, 0, hidden)
    assert out.tobytes() == out_old.tobytes()


def test_padded_component_input_matches_linear(monkeypatch, tmp_path) -> None:
    """RDA f_b_proj 의 128→256 패딩이 묶음 경로와 하나씩 경로에서 바이트 같다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    name = "model.layers.0.self_attn.f_b_proj.weight"
    x = np.arange(128, dtype=np.uint16)

    backend.prefetch_linears(((name, x),))
    group_input = direct.started[-1][1]

    backend.set_step(1)  # clear cache so linear computes anew
    backend.linear(name, x)
    single_input = direct.started[-1][1]

    assert group_input.shape == (256,)
    assert single_input.shape == (256,)
    assert group_input.tobytes() == single_input.tobytes()
    assert np.array_equal(group_input[128:], np.zeros(128, dtype=np.uint16))


def test_group_timing_counts_per_gemv(monkeypatch, tmp_path) -> None:
    """묶음으로 실행한 GEMV 마다 timing_key·linear_npu 카운트가 1씩 는다."""
    backend, direct = _backend(monkeypatch, tmp_path)
    names = _mla_names(3)
    q_lat = np.arange(576, dtype=np.uint16)
    kv_lat = np.arange(512, dtype=np.uint16)

    backend.prefetch_linears(((names["q_b"], q_lat), (names["kv_b"], kv_lat)))

    for bundle in ("L03/q_b", "L03/k_b", "L03/v_b"):
        key = direct.entries[bundle].timing_key
        assert backend.call_counts[key] == 1
        assert backend.timings[key] >= 0.0
    assert backend.call_counts["linear_npu"] == 3
    assert backend.call_counts["decode_mla_group"] == 1


class _FakeClock:
    """Manual clock for deterministic prefetch_linears timing assertions."""

    def __init__(self) -> None:
        self.now = 0.0


class _FakeClockDirect(FakeDirect):
    """FakeDirect whose gemv_start/gemv_finish advance the clock by fixed amounts."""

    START_DELAYS = [0.001, 0.002, 0.003]
    FINISH_DELAYS = [0.010, 0.004, 0.005]

    def __init__(self, clock: _FakeClock) -> None:
        super().__init__()
        self.clock = clock

    def gemv_start(self, entry, x):
        self.clock.now += self.START_DELAYS[len(self.started)]
        return super().gemv_start(entry, x)

    def gemv_finish(self, entry, run):
        self.clock.now += self.FINISH_DELAYS[len(self.finished)]
        return super().gemv_finish(entry, run)


def test_group_timing_formula_with_fake_clock(monkeypatch, tmp_path) -> None:
    """가짜 시계로 GEMV 시작·끝 시간 배분 산식을 정확히 검증한다."""
    clock = _FakeClock()
    monkeypatch.setattr(nb.time, "perf_counter", lambda: clock.now)
    backend, direct = _backend(monkeypatch, tmp_path, runner=_FakeClockDirect(clock))
    names = _mla_names(3)
    q_lat = np.arange(576, dtype=np.uint16)
    kv_lat = np.arange(512, dtype=np.uint16)

    baseline_timings = dict(backend.timings)
    baseline_counts = dict(backend.call_counts)

    backend.prefetch_linears(((names["q_b"], q_lat), (names["kv_b"], kv_lat)))

    expected_dts = [0.011, 0.006, 0.008]
    for i, bundle in enumerate(("L03/q_b", "L03/k_b", "L03/v_b")):
        key = direct.entries[bundle].timing_key
        assert backend.call_counts[key] == baseline_counts.get(key, 0) + 1
        assert backend.timings[key] - baseline_timings.get(key, 0.0) == pytest.approx(
            expected_dts[i]
        )

    assert backend.call_counts["linear_npu"] == baseline_counts.get("linear_npu", 0) + 3
    assert backend.timings["linear_npu"] - baseline_timings.get("linear_npu", 0.0) == pytest.approx(
        0.025
    )
