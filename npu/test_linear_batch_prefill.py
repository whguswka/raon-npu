# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import types

import numpy as np
import pytest

from npu.backend import NpuBackend, PREFILL_GEMM_MAX_T, PREFILL_GEMM_XCLBIN_ID


CASES = (
    ("rda_in", "model.layers.1.self_attn.q_proj.weight"),
    ("f_b", "model.layers.1.self_attn.f_b_proj.weight"),
    ("o_proj", "model.layers.1.self_attn.o_proj.weight"),
    ("mla_in", "model.layers.3.self_attn.q_a_proj.weight"),
    ("q_b", "model.layers.3.self_attn.q_b_proj.weight"),
    ("k_b_v_b", "model.layers.3.self_attn.kv_b_proj.weight"),
    ("g", "model.layers.3.self_attn.g_proj.weight"),
    ("dense_down", "model.layers.0.mlp.down_proj.weight"),
)


def _full_output(rows: np.ndarray, m: int) -> np.ndarray:
    base = rows.astype(np.uint64).sum(axis=1)[:, None]
    return ((base + np.arange(m, dtype=np.uint64)) & 0xFFFF).astype(np.uint16)


def _backend_with_synthetic_executors(monkeypatch):
    backend = NpuBackend(mock=True)
    calls = []

    def batch(self, bundle_name, bmeta, shape_name, x_u16):
        calls.append((bundle_name, shape_name, x_u16.copy()))
        self._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
        return _full_output(x_u16[:, : bmeta.k], bmeta.m)

    def gemv(self, bundle_name, bmeta, x_u16):
        return _full_output(x_u16[None, : bmeta.k], bmeta.m)[0]

    monkeypatch.setattr(backend, "_execute_gemm_batch", types.MethodType(batch, backend))
    monkeypatch.setattr(backend, "_execute_gemv", types.MethodType(gemv, backend))
    return backend, calls


@pytest.mark.parametrize("shape_name,name", CASES)
@pytest.mark.parametrize("t", (1, 2, 3, 23, 24))
def test_all_shapes_and_t_equal_rowwise_linear(monkeypatch, shape_name, name, t):
    backend, calls = _backend_with_synthetic_executors(monkeypatch)
    comp = backend.comp_map[name]
    x = np.arange(t * comp.orig_k, dtype=np.uint16).reshape(t, comp.orig_k)
    expected = np.stack([backend.linear(name, row) for row in x])
    backend.set_step(1)
    actual = backend.linear_batch(name, x)
    np.testing.assert_array_equal(actual, expected)
    assert calls[-1][1] == shape_name


def test_bundle_cache_uses_step_and_exact_object_identity(monkeypatch):
    backend, calls = _backend_with_synthetic_executors(monkeypatch)
    q = "model.layers.1.self_attn.q_proj.weight"
    k = "model.layers.1.self_attn.k_proj.weight"
    x = np.zeros((3, 2560), np.uint16)
    backend.linear_batch(q, x)
    backend.linear_batch(k, x)
    assert len(calls) == 1
    backend.linear_batch(q, x.copy())
    assert len(calls) == 2
    backend.set_step(backend._step_id + 1)
    backend.linear_batch(q, x)
    assert len(calls) == 3
    backend.reset()
    backend.linear_batch(q, x)
    assert len(calls) == 4


@pytest.mark.parametrize("t", (1, 24, 25))
def test_mla_kv_b_gemm_matches_rowwise_and_reuses_cache(monkeypatch, t):
    backend, calls = _backend_with_synthetic_executors(monkeypatch)
    name = "model.layers.3.self_attn.kv_b_proj.weight"
    x = np.arange(t * 512, dtype=np.uint16).reshape(t, 512)
    expected = np.stack([backend.linear(name, row) for row in x])

    backend.set_step(7)
    actual = backend.linear_batch(name, x)
    np.testing.assert_array_equal(actual, expected)
    assert [call[1] for call in calls] == ["k_b_v_b"] * (2 * ((t + 23) // 24))

    before = len(calls)
    np.testing.assert_array_equal(backend.linear_batch(name, x), expected)
    assert len(calls) == before

    backend.linear_batch(name, x.copy())
    assert len(calls) > before


def test_padding_and_chunking(monkeypatch):
    backend, calls = _backend_with_synthetic_executors(monkeypatch)
    name = "model.layers.1.self_attn.f_b_proj.weight"
    x = np.ones((25, 128), np.uint16)
    out = backend.linear_batch(name, x)
    assert out.shape == (25, 5120)
    assert [call[2].shape for call in calls] == [(24, 256), (1, 256)]
    assert not np.any(calls[0][2][:, 128:])
    assert not np.any(calls[1][2][:, 128:])


@pytest.mark.parametrize("t", (1, 23, 24, 25, 47, 48, 49, 256))
def test_chunk_row_counts_and_rowwise_byte_parity(monkeypatch, t):
    backend, calls = _backend_with_synthetic_executors(monkeypatch)
    name = "model.layers.1.self_attn.q_proj.weight"
    comp = backend.comp_map[name]
    x = np.arange(t * comp.orig_k, dtype=np.uint16).reshape(t, comp.orig_k)
    expected = np.stack([backend.linear(name, row) for row in x])

    actual = backend.linear_batch(name, x)

    q, r = divmod(t, PREFILL_GEMM_MAX_T)
    expected_rows = [PREFILL_GEMM_MAX_T] * q + ([r] if r else [])
    assert [call[2].shape[0] for call in calls] == expected_rows
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "name,k",
    (
        ("model.layers.0.mlp.gate_proj.weight", 2560),
        ("model.layers.1.moe.shared_experts.down_proj.weight", 1088),
    ),
)
def test_excluded_or_special_bundle_falls_back(monkeypatch, name, k):
    backend = NpuBackend(mock=True)
    marker = np.full((2, 7), 0x1234, np.uint16)
    seen = []

    def fallback(called_name, x):
        seen.append((called_name, x))
        return marker

    monkeypatch.setattr(backend, "linear", fallback)
    x = np.zeros((2, k), np.uint16)
    assert backend.linear_batch(name, x) is marker
    assert seen == [(name, x)]


def test_switch_off_never_calls_gemm(monkeypatch):
    monkeypatch.setenv("RAON_PREFILL_GEMM", "0")
    backend = NpuBackend(mock=True)
    marker = np.zeros((2, 5), np.uint16)
    monkeypatch.setattr(backend, "linear", lambda name, x: marker)
    monkeypatch.setattr(
        backend, "_execute_gemm_batch",
        lambda *args, **kwargs: pytest.fail("prefill GEMM must not run"),
    )
    x = np.zeros((2, 2560), np.uint16)
    assert backend.linear_batch("model.layers.1.self_attn.q_proj.weight", x) is marker


@pytest.mark.parametrize("bad_k", (2559, 2561))
def test_switch_off_still_rejects_known_component_wrong_k(monkeypatch, bad_k):
    monkeypatch.setenv("RAON_PREFILL_GEMM", "0")
    backend = NpuBackend(mock=True)
    monkeypatch.setattr(
        backend, "linear", lambda *args: pytest.fail("invalid K must not reach fallback")
    )
    x = np.zeros((2, bad_k), np.uint16)
    with pytest.raises(ValueError, match=rf"input K={bad_k} .* K=2560"):
        backend.linear_batch("model.layers.1.self_attn.q_proj.weight", x)


def test_input_contract_rejects_wrong_rank_dtype_and_empty():
    backend = NpuBackend(mock=True)
    name = "model.layers.1.self_attn.q_proj.weight"
    for bad in (
        np.zeros(2560, np.uint16),
        np.zeros((1, 2560), np.float32),
        np.zeros((0, 2560), np.uint16),
    ):
        with pytest.raises(ValueError):
            backend.linear_batch(name, bad)
