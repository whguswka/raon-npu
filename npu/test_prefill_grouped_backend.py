# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for grouped prefill MoE without an NPU or full weight allocation."""

from __future__ import annotations

from types import MethodType

import numpy as np
import pytest

from host.weights import bf16_to_f32, f32_to_bf16_rtne
from kernels.moe_grouped.planner import build_plan
from npu.backend import NpuBackend
from npu import backend as backend_module


LAYER = 1


def _shared_gu_for_token_x(x_row_u16: np.ndarray) -> np.ndarray:
    token_value = float(bf16_to_f32(np.asarray(x_row_u16, dtype=np.uint16)[:1])[0])
    values = np.zeros(2176, dtype=np.float32)
    values[:1088] = token_value / 8.0 + 0.125
    values[1088:] = token_value / 4.0 + 0.25
    return f32_to_bf16_rtne(values)


def _inputs(token_count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_f32 = np.zeros((token_count, 2560), dtype=np.float32)
    x_f32[:, 0] = np.arange(1, token_count + 1, dtype=np.float32) / 32.0
    x = f32_to_bf16_rtne(x_f32)
    pool = np.array([0, 1, 2, 3, 4, 5, 7, 11, 63, 95, 126, 127], dtype=np.int32)
    top = np.stack([np.roll(pool, -token)[:6] for token in range(token_count)])
    top[:, 0] = 0
    for token in range(token_count):
        if len(set(map(int, top[token]))) != 6:
            top[token] = np.array([0, 1, 2, 3, 4, 127], dtype=np.int32)
    raw = np.arange(1, 7, dtype=np.float32)[None, :] + np.arange(
        token_count, dtype=np.float32
    )[:, None] / 17.0
    weights = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)
    shared_gu = np.stack(
        [_shared_gu_for_token_x(x[token]) for token in range(token_count)]
    )
    return x, top, weights, shared_gu


@pytest.fixture
def lightweight_backend(monkeypatch) -> NpuBackend:
    backend = NpuBackend(mock=True)
    # Membership is all the production mock path needs before the row helpers run.
    backend.mock_weight_arrays[f"L{LAYER:02d}/moe_b1"] = object()
    backend.mock_weight_arrays[f"L{LAYER:02d}/moe_b2"] = object()
    backend.mock_weight_arrays[f"L{LAYER:02d}/shared_gu"] = object()

    def b1_row(self, layer, expert, x_u16, dw):
        del self, layer, dw
        token_value = float(bf16_to_f32(x_u16[:1])[0])
        values = np.zeros(2176, dtype=np.float32)
        values[:1088] = (expert + 1) / 512.0 + token_value / 16.0
        values[1088:] = (expert + 2) / 384.0 - token_value / 32.0
        return f32_to_bf16_rtne(values)

    def b2_row(self, layer, expert, x_u16, dw):
        del self, layer, dw
        active = bf16_to_f32(np.asarray(x_u16, dtype=np.uint16)[:4]).sum(dtype=np.float32)
        values = np.full(2560, (expert + 1) / 1024.0 + active / 64.0, dtype=np.float32)
        values[::97] += np.float32(expert / 4096.0)
        return f32_to_bf16_rtne(values)

    def shared_reference(self, layer, x_u16, dw):
        del self, layer, dw
        return _shared_gu_for_token_x(x_u16)

    monkeypatch.setattr(backend, "_moe_b1_row_numpy", MethodType(b1_row, backend))
    monkeypatch.setattr(backend, "_moe_b2_row_numpy", MethodType(b2_row, backend))
    monkeypatch.setattr(
        backend, "_moe_shared_gu_reference_numpy", MethodType(shared_reference, backend)
    )
    return backend


def _token_reference(backend, x, top, weights, shared_gu):
    routed = []
    shared = []
    down_name = f"model.layers.{LAYER}.moe.shared_experts.down_proj.weight"
    for token in range(x.shape[0]):
        routed.append(
            backend.moe_experts(
                LAYER,
                x[token],
                top[token],
                weights[token],
                shared_gu=shared_gu[token],
            )
        )
        sh_act = backend._cached_shared[LAYER]["sh_act_u16"]
        shared.append(backend.linear(down_name, sh_act))
    return np.stack(routed), np.stack(shared)


@pytest.mark.parametrize("token_count", [1, 7, 8, 9, 15, 16, 61, 63, 64])
def test_grouped_mock_matches_token_path(lightweight_backend, token_count):
    backend = lightweight_backend
    x, top, weights, shared_gu = _inputs(token_count)
    expected_routed, expected_shared = _token_reference(
        backend, x, top, weights, shared_gu
    )
    backend.reset_timings()

    routed, shared = backend.moe_experts_grouped(
        LAYER, x, top, weights, shared_gu
    )

    assert np.array_equal(routed, expected_routed)
    assert np.array_equal(shared, expected_shared)
    assert backend.xclbin_calls["moe_grouped_b1"] == len(build_plan(top).calls)
    assert backend.xclbin_calls["moe_grouped_b2"] == len(
        build_plan(top, include_shared=True).calls
    )
    assert not any(name.startswith("gemv_unified") for name in backend.xclbin_calls)
    assert LAYER not in backend._cached_shared
    with pytest.raises(ValueError, match="without a valid"):
        backend.linear(
            f"model.layers.{LAYER}.moe.shared_experts.down_proj.weight",
            np.zeros(1088, dtype=np.uint16),
        )


def test_grouped_api_validation_happens_before_execution(lightweight_backend):
    backend = lightweight_backend
    x, top, weights, shared_gu = _inputs(2)
    invalid = [
        (x.astype(np.float32), top, weights, shared_gu),
        (x[:, :-1], top, weights, shared_gu),
        (x, top.astype(np.float32), weights, shared_gu),
        (x, top[:, :-1], weights, shared_gu),
        (x, top[:1], weights, shared_gu),
        (x, top, weights.astype(np.float64), shared_gu),
        (x, top, weights[:1], shared_gu),
        (x, top, weights, shared_gu[:, :-1]),
        (x, top, weights, shared_gu[:1]),
    ]
    for args in invalid:
        with pytest.raises(ValueError):
            backend.moe_experts_grouped(LAYER, *args)
    for bad_id in (-1, 128):
        bad = top.copy()
        bad[0, 0] = bad_id
        with pytest.raises(ValueError):
            backend.moe_experts_grouped(LAYER, x, bad, weights, shared_gu)
    duplicate = top.copy()
    duplicate[0, 1] = duplicate[0, 0]
    with pytest.raises(ValueError):
        backend.moe_experts_grouped(LAYER, x, duplicate, weights, shared_gu)
    assert backend.xclbin_calls == {}


def test_grouped_shared_gu_bit_mismatch_is_rejected(lightweight_backend):
    x, top, weights, shared_gu = _inputs(1)
    shared_gu[0, 0] ^= np.uint16(1)
    with pytest.raises(ValueError, match="shared_gu mismatch"):
        lightweight_backend.moe_experts_grouped(
            LAYER, x, top, weights, shared_gu
        )


def test_grouped_switch_contract(monkeypatch):
    monkeypatch.delenv("RAON_PREFILL_GROUPED_MOE", raising=False)
    backend = NpuBackend(mock=True)
    assert backend.prefill_grouped_moe is True
    backend.prefill_grouped_moe = False
    assert backend.prefill_grouped_moe is False
    backend.prefill_grouped_moe = True
    backend.prefill_shared_precompute = False
    assert backend.prefill_grouped_moe is False
    backend.prefill_shared_precompute = True
    backend.gemv = "legacy"
    assert backend.prefill_grouped_moe is False


class _FakeBO:
    def __init__(self, size: int):
        self.data = np.zeros(size, dtype=np.uint16)

    def numpy(self) -> np.ndarray:
        return self.data


def _fake_grouped_ops(backend, seen):
    grouped = {}
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        x_bo = _FakeBO(8 * 8 * shape.k)
        y_bo = _FakeBO(8 * 8 * shape.m)

        def run(y_tensor, _weight, x_tensor, _shape=shape, **scalars):
            experts = tuple(int(scalars[f"e{i}"]) for i in range(8))
            seen.append((_shape.name, experts))
            k = _shape.k
            # The backend writes X pre-tiled as [slot][k_block][32][TMAX][8]; undo that layout to read it.
            values = (
                x_tensor.data.reshape(8, k // 256, 32, 8, 8)
                .transpose(0, 3, 1, 2, 4)
                .reshape(8, 8, k)
            )
            output = y_tensor.data.reshape(8, 8, _shape.m)
            for slot, expert in enumerate(experts):
                for row in range(8):
                    if _shape is backend_module.GROUPED_B1:
                        output[slot, row] = backend._moe_b1_row_numpy(
                            LAYER, expert, values[slot, row], None
                        )
                    else:
                        output[slot, row] = backend._moe_b2_row_numpy(
                            LAYER, expert, values[slot, row], None
                        )

        grouped[shape.name] = (run, x_bo, y_bo)
    return grouped


@pytest.mark.parametrize("token_count", [1, 7, 8, 9, 15, 16, 61, 63, 64])
def test_grouped_hardware_branch_with_fake_kernel(
    lightweight_backend, monkeypatch, token_count
):
    backend = lightweight_backend
    x, top, weights, shared_gu = _inputs(token_count)
    expected = _token_reference(backend, x, top, weights, shared_gu)
    backend.reset_timings()
    backend.mock = False
    monkeypatch.setattr(backend_module, "HAVE_NPU", True)
    seen = []
    grouped = _fake_grouped_ops(backend, seen)
    monkeypatch.setattr(backend, "_get_or_compile_grouped_moe", lambda: grouped)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: object())

    def overwrite(tensor, value):
        tensor.data[:] = np.asarray(value).view(np.uint16).reshape(-1)

    monkeypatch.setattr(backend, "_overwrite_buffer", overwrite)
    actual = backend.moe_experts_grouped(LAYER, x, top, weights, shared_gu)

    assert np.array_equal(actual[0], expected[0])
    assert np.array_equal(actual[1], expected[1])
    expected_seen = [
        ("b1", call.expert_slots) for call in build_plan(top).calls
    ] + [
        ("b2", call.expert_slots)
        for call in build_plan(top, include_shared=True).calls
    ]
    assert seen == expected_seen
    assert backend.call_counts["moe_grouped_b1_npu"] == len(build_plan(top).calls)
    assert backend.call_counts["moe_grouped_b2_npu"] == len(
        build_plan(top, include_shared=True).calls
    )
    assert backend.call_counts["moe_grouped_host"] == 1
    assert backend.call_counts["moe"] == token_count
    assert backend.timings["moe"] == (
        backend.timings["moe_grouped_b1_npu"]
        + backend.timings["moe_grouped_b2_npu"]
    )


def test_grouped_failed_hardware_run_does_not_record_xclbin(
    lightweight_backend, monkeypatch
):
    backend = lightweight_backend
    x, top, weights, shared_gu = _inputs(1)
    backend.mock = False
    monkeypatch.setattr(backend_module, "HAVE_NPU", True)
    grouped = _fake_grouped_ops(backend, [])

    def fail(*_args, **_kwargs):
        raise RuntimeError("injected grouped failure")

    run, x_bo, y_bo = grouped["b1"]
    del run
    grouped["b1"] = (fail, x_bo, y_bo)
    monkeypatch.setattr(backend, "_get_or_compile_grouped_moe", lambda: grouped)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: object())
    monkeypatch.setattr(
        backend,
        "_overwrite_buffer",
        lambda tensor, value: tensor.data.__setitem__(
            slice(None), np.asarray(value).view(np.uint16).reshape(-1)
        ),
    )
    with pytest.raises(RuntimeError, match="injected grouped failure"):
        backend.moe_experts_grouped(LAYER, x, top, weights, shared_gu)
    assert backend.xclbin_calls.get("moe_grouped_b1", 0) == 0
    assert backend.xclbin_calls.get("moe_grouped_b2", 0) == 0


def test_grouped_hardware_call_writes_pre_tiled_x_bytes_for_b1_and_b2(
    lightweight_backend, monkeypatch
):
    """Synchronous _run_grouped_moe_hardware_call must write the grouped X layout, not logical batch bytes."""
    backend = lightweight_backend
    backend.mock = False
    monkeypatch.setattr(backend_module, "HAVE_NPU", True)
    monkeypatch.setattr(backend, "_get_prefill_weight_tensor", lambda _name: object())

    captured: dict[str, object] = {}

    def overwrite_and_capture(tensor, value):
        captured["tensor"] = tensor
        captured["value"] = value
        tensor.data[:] = np.asarray(value).view(np.uint16).reshape(-1)

    monkeypatch.setattr(backend, "_overwrite_buffer", overwrite_and_capture)

    grouped = {}
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        x_bo = _FakeBO(8 * 8 * shape.k)
        y_bo = _FakeBO(8 * 8 * shape.m)

        def run(_y_tensor, _w_tensor, _x_tensor, **_scalars):
            pass

        grouped[shape.name] = (run, x_bo, y_bo)

    rng = np.random.RandomState(6000)
    for shape in (backend_module.GROUPED_B1, backend_module.GROUPED_B2):
        K = shape.k
        values = rng.randint(0, 65536, size=(8, 8, K), dtype=np.uint16)
        experts = tuple(range(8))
        rows = tuple(range(8)) if shape is backend_module.GROUPED_B1 else None

        backend._run_grouped_moe_hardware_call(
            shape, LAYER, experts, values, grouped, rows=rows
        )

        assert captured["tensor"] is grouped[shape.name][1]
        actual_u16 = np.asarray(captured["value"]).view(np.uint16)
        expected_u16 = backend_module._grouped_x_tiled(values, shape).reshape(-1)
        np.testing.assert_array_equal(actual_u16, expected_u16)


class _FakeWeights:
    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = tensors

    def __contains__(self, key: str) -> bool:
        return key in self.tensors

    def get(self, key: str, to_float: bool = True) -> np.ndarray:
        if key not in self.tensors:
            raise KeyError(key)
        arr = self.tensors[key]
        return arr.astype(np.float32) if to_float else arr


def test_moe_b2_row_weight_source_selection():
    backend = NpuBackend(mock=True)
    layer = 1
    shared_name = f"model.layers.{layer}.moe.shared_experts.down_proj.weight"
    routed_name = f"model.layers.{layer}.moe.experts.0.down_proj.weight"

    val_a = np.full((2560, 1088), 1.0, dtype=np.float32)
    val_b = np.full((2560, 1088), 2.0, dtype=np.float32)

    host_w = _FakeWeights({shared_name: val_a, routed_name: val_a})
    dw = _FakeWeights({shared_name: val_b, routed_name: val_b})

    backend.host_weights = host_w
    backend.mock_weight_arrays.pop(f"L{layer:02d}/moe_b2", None)

    x_u16 = f32_to_bf16_rtne(np.ones(1088, dtype=np.float32))

    # Shared down (expert 128) must use dw (val_b)
    out_shared = backend._moe_b2_row_numpy(layer, 128, x_u16, dw)
    expected_shared = f32_to_bf16_rtne(np.dot(bf16_to_f32(x_u16), val_b.T))
    assert np.array_equal(out_shared, expected_shared)

    # Routed down (expert 0) must use host_weights (val_a)
    out_routed = backend._moe_b2_row_numpy(layer, 0, x_u16, dw)
    expected_routed = f32_to_bf16_rtne(np.dot(bf16_to_f32(x_u16), val_a.T))
    assert np.array_equal(out_routed, expected_routed)

    # Missing dw for expert 128 raises ValueError
    with pytest.raises(ValueError, match="missing DWeights"):
        backend._moe_b2_row_numpy(layer, 128, x_u16, None)

    # B2 padded input width (1280) trimmed to weight width (1088)
    x_padded = np.zeros(1280, dtype=np.uint16)
    x_padded[:1088] = x_u16
    out_padded = backend._moe_b2_row_numpy(layer, 128, x_padded, dw)
    assert np.array_equal(out_padded, expected_shared)
    assert np.array_equal(backend._moe_b2_row_numpy(layer, 0, x_padded, dw), expected_routed)


class _LazyExpertBundle:
    def __init__(self, shape: tuple[int, int], base_seed: int):
        self.shape = shape
        self.base_seed = base_seed
        self.cache: dict[int, np.ndarray] = {}

    def __getitem__(self, expert_id: int) -> np.ndarray:
        expert_id = int(expert_id)
        if expert_id not in self.cache:
            rng = np.random.RandomState(self.base_seed + expert_id * 37)
            f32 = rng.uniform(-0.05, 0.05, size=self.shape).astype(np.float32)
            self.cache[expert_id] = f32_to_bf16_rtne(f32)
        return self.cache[expert_id]


@pytest.fixture(scope="module")
def realistic_grouped_backend() -> NpuBackend:
    backend = NpuBackend(mock=True)
    layer = LAYER
    backend.mock_weight_arrays[f"L{layer:02d}/moe_b1"] = _LazyExpertBundle(
        (2176, 2560), 1000
    )
    backend.mock_weight_arrays[f"L{layer:02d}/moe_b2"] = _LazyExpertBundle(
        (2560, 1280), 2000
    )
    rng_sh = np.random.RandomState(3000)
    sh_f32 = rng_sh.uniform(-0.05, 0.05, size=(2176, 2560)).astype(np.float32)
    backend.mock_weight_arrays[f"L{layer:02d}/shared_gu"] = f32_to_bf16_rtne(sh_f32)
    return backend


@pytest.mark.parametrize("token_count", [1, 9, 64])
def test_grouped_mock_real_helpers_parity(realistic_grouped_backend, token_count):
    backend = realistic_grouped_backend
    rng = np.random.RandomState(5000 + token_count)
    x_f32 = rng.uniform(-0.1, 0.1, size=(token_count, 2560)).astype(np.float32)
    x = f32_to_bf16_rtne(x_f32)

    pool = [0, 1, 2, 3, 4, 5, 7, 11, 63, 95, 126, 127]
    other_pool = [e for e in pool if e != 0]
    top = np.zeros((token_count, 6), dtype=np.int32)
    for t in range(token_count):
        chosen = [0] + list(rng.choice(other_pool, size=5, replace=False))
        rng.shuffle(chosen)
        top[t] = chosen

    raw = rng.uniform(0.1, 1.0, size=(token_count, 6)).astype(np.float32)
    weights = (raw / raw.sum(axis=1, keepdims=True)).astype(np.float32)

    shared_gu = np.stack(
        [
            backend._moe_shared_gu_reference_numpy(LAYER, x[t], None)
            for t in range(token_count)
        ]
    )

    down_name = f"model.layers.{LAYER}.moe.shared_experts.down_proj.weight"
    expected_routed = []
    expected_shared = []
    for t in range(token_count):
        r = backend.moe_experts(
            LAYER, x[t], top[t], weights[t], shared_gu=shared_gu[t]
        )
        sh_act = backend._cached_shared[LAYER]["sh_act_u16"]
        s = backend.linear(down_name, sh_act)
        expected_routed.append(r)
        expected_shared.append(s)

    expected_routed = np.stack(expected_routed)
    expected_shared = np.stack(expected_shared)

    actual_routed, actual_shared = backend.moe_experts_grouped(
        LAYER, x, top, weights, shared_gu
    )

    assert np.array_equal(actual_routed, expected_routed)
    assert np.array_equal(actual_shared, expected_shared)
