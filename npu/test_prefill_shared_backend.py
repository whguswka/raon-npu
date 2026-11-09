# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for stage 2b-1 prefill shared_gu GEMM integration and NpuBackend moe_experts shared_gu path."""

from __future__ import annotations

import numpy as np
import pytest

from host.weights import f32_to_bf16_rtne, bf16_to_f32
from kernels.gemm_prefill.budget import resource_budget
from kernels.gemm_prefill.shapes import PREFILL_SHAPE_NAMES, get_shape
from npu.backend import (
    NpuBackend,
    PREFILL_GEMM_SHAPES,
    resolve_unified_shape_name,
)


def _setup_mock_backend(seed: int = 42) -> tuple[NpuBackend, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    backend = NpuBackend(mock=True)
    rng = np.random.default_rng(seed)

    w_b1 = f32_to_bf16_rtne(rng.standard_normal((128, 2176, 2560)).astype(np.float32) * 0.02)
    w_sh = f32_to_bf16_rtne(rng.standard_normal((2176, 2560)).astype(np.float32) * 0.02)
    w_b2 = f32_to_bf16_rtne(rng.standard_normal((129, 2560, 1280)).astype(np.float32) * 0.02)

    backend.mock_weight_arrays["L01/moe_b1"] = w_b1
    backend.mock_weight_arrays["L01/shared_gu"] = w_sh
    backend.mock_weight_arrays["L01/moe_b2"] = w_b2

    x_u16 = f32_to_bf16_rtne(rng.standard_normal(2560).astype(np.float32))
    top_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    top_weights = np.ones(6, dtype=np.float32) / 6.0

    return backend, x_u16, top_ids, top_weights, w_sh


def test_shared_gu_shapes_and_budget():
    """Verify shared_gu is properly registered in prefill GEMM shapes and fits hardware resource budget."""
    assert "shared_gu" in PREFILL_SHAPE_NAMES
    assert "shared_gu" in PREFILL_GEMM_SHAPES
    assert len(PREFILL_SHAPE_NAMES) == 9
    assert len(PREFILL_GEMM_SHAPES) == 9

    shape = get_shape("shared_gu")
    assert shape.K == 2560
    assert shape.M == 2176
    assert shape.format == 0  # 9-bit bfp16

    budget = resource_budget("shared_gu", 24, cols=7)
    assert budget.limits_ok
    assert budget.core_fits
    assert budget.memtile_fits
    assert budget.core_total_bytes == 62472
    assert budget.memtile_total_bytes == 86016


def test_mock_moe_experts_shared_gu_exact_byte_parity():
    """(가) Verify moe_experts with supplied shared_gu yields bit-for-bit identical output, caches, and skips run_sh."""
    b_ref, x_u16, top_ids, top_weights, w_sh = _setup_mock_backend(seed=100)
    b_test, _, _, _, _ = _setup_mock_backend(seed=100)

    # 1. Reference call without shared_gu
    b_ref.set_step(1)
    out_ref = b_ref.moe_experts(1, x_u16, top_ids, top_weights)
    cached_shared_ref = b_ref._cached_shared[1]
    bundle_cache_ref = b_ref._bundle_cache["L01/shared_gu"]

    # Compute expected shared_gu uint16 bits
    w_sh_f32 = bf16_to_f32(w_sh)
    x_f32 = bf16_to_f32(x_u16)
    expected_shared_gu = f32_to_bf16_rtne(np.dot(x_f32, w_sh_f32.T))

    # Verify reference sh_gate and sh_up match expected_shared_gu
    assert np.array_equal(cached_shared_ref["sh_gate"], expected_shared_gu[:1088])
    assert np.array_equal(cached_shared_ref["sh_up"], expected_shared_gu[1088:])

    # 2. Test call with precomputed shared_gu
    b_test.set_step(1)
    b_test.reset_timings()
    out_test = b_test.moe_experts(1, x_u16, top_ids, top_weights, shared_gu=expected_shared_gu)
    cached_shared_test = b_test._cached_shared[1]
    bundle_cache_test = b_test._bundle_cache["L01/shared_gu"]

    # Assert bit-for-bit parity
    assert np.array_equal(out_ref, out_test)
    assert np.array_equal(bundle_cache_ref[2], bundle_cache_test[2])
    for key in ("sh_gate", "sh_up", "sh_act_u16", "sh_down"):
        assert np.array_equal(cached_shared_ref[key], cached_shared_test[key])

    # Assert run_sh was skipped (moe_sh_npu timing not recorded)
    assert "moe_sh_npu" not in b_test.timings or b_test.timings["moe_sh_npu"] == 0.0


def test_mock_moe_experts_shared_gu_invalid_input_and_bit_flip_fails():
    """(가) Verify invalid dtype, shape, or flipped bit in shared_gu raises ValueError."""
    backend, x_u16, top_ids, top_weights, w_sh = _setup_mock_backend(seed=200)

    w_sh_f32 = bf16_to_f32(w_sh)
    x_f32 = bf16_to_f32(x_u16)
    valid_shared_gu = f32_to_bf16_rtne(np.dot(x_f32, w_sh_f32.T))

    # Wrong shape
    with pytest.raises(ValueError, match="shared_gu must be a \\[2176\\] uint16 ndarray"):
        backend.moe_experts(1, x_u16, top_ids, top_weights, shared_gu=valid_shared_gu[:100])

    # Wrong dtype
    with pytest.raises(ValueError, match="shared_gu must be a \\[2176\\] uint16 ndarray"):
        backend.moe_experts(1, x_u16, top_ids, top_weights, shared_gu=valid_shared_gu.astype(np.float32))

    # Bit flip must raise ValueError in mock mode
    corrupted = valid_shared_gu.copy()
    corrupted[0] ^= 1
    with pytest.raises(ValueError, match="shared_gu mismatch in mock moe_experts"):
        backend.moe_experts(1, x_u16, top_ids, top_weights, shared_gu=corrupted)

    corrupted_up = valid_shared_gu.copy()
    corrupted_up[1500] ^= 1
    with pytest.raises(ValueError, match="shared_gu mismatch in mock moe_experts"):
        backend.moe_experts(1, x_u16, top_ids, top_weights, shared_gu=corrupted_up)


def test_linear_batch_shared_gu_shape_resolution_and_bundle_caching():
    """(다) Verify linear_batch resolves shared gate/up to shared_gu, uses GEMM path, and bundle-caches."""
    backend, _, _, _, w_sh = _setup_mock_backend(seed=300)
    rng = np.random.default_rng(301)

    gate_name = "model.layers.1.moe.shared_experts.gate_proj.weight"
    up_name = "model.layers.1.moe.shared_experts.up_proj.weight"

    # Shape resolution check
    comp_gate = backend.comp_map[gate_name]
    bmeta = backend.bundle_map[comp_gate.bundle_name]
    assert bmeta.k == 2560
    assert bmeta.m == 2176
    assert bmeta.fmt == "bfp16_gemv"
    shape_name = resolve_unified_shape_name(bmeta.k, bmeta.m, bmeta.fmt, comp_gate.bundle_name)
    assert shape_name == "shared_gu"
    assert shape_name in PREFILL_GEMM_SHAPES

    # Batch execution check
    T = 24
    x_batch = f32_to_bf16_rtne(rng.standard_normal((T, 2560), dtype=np.float32))

    backend.set_step(1)
    backend.cache_hits = 0
    backend.cache_misses = 0
    gemm_calls_before = backend.call_counts.get("gemm:shared_gu", 0)

    # 1. First call (gate_proj): executes batch GEMM
    sh_gate_batch = backend.linear_batch(gate_name, x_batch)
    gemm_calls_after_gate = backend.call_counts.get("gemm:shared_gu", 0)
    assert gemm_calls_after_gate == gemm_calls_before + 1
    assert backend.cache_misses == 1
    assert backend.cache_hits == 0

    # 2. Second call with same x_batch (up_proj): must hit batch bundle cache
    sh_up_batch = backend.linear_batch(up_name, x_batch)
    gemm_calls_after_up = backend.call_counts.get("gemm:shared_gu", 0)
    assert gemm_calls_after_up == gemm_calls_after_gate  # 0 additional GEMM calls
    assert backend.cache_hits == 1

    # 3. Verify gate and up concat order matches run_sh output
    w_sh_f32 = bf16_to_f32(w_sh)
    for t in range(T):
        x_row_f32 = bf16_to_f32(x_batch[t])
        expected_gu_row = f32_to_bf16_rtne(np.dot(x_row_f32, w_sh_f32.T))
        combined_gu_row = np.concatenate([sh_gate_batch[t], sh_up_batch[t]])
        assert np.array_equal(combined_gu_row, expected_gu_row)
        assert np.array_equal(sh_gate_batch[t], expected_gu_row[:1088])
        assert np.array_equal(sh_up_batch[t], expected_gu_row[1088:])


def test_prefill_shared_precompute_property(monkeypatch):
    """Verify backend prefill_shared_precompute switch behavior."""
    monkeypatch.delenv("RAON_PREFILL_SHARED_GEMM", raising=False)
    monkeypatch.delenv("RAON_PREFILL_GEMM", raising=False)
    backend = NpuBackend(mock=True)

    # By default, enabled when prefill_gemm_enabled is True and env is default ("1")
    assert backend.prefill_shared_precompute is True

    # Manual override
    backend.prefill_shared_precompute = False
    assert backend.prefill_shared_precompute is False

    backend.prefill_shared_precompute = True
    assert backend.prefill_shared_precompute is True

    # When prefill_gemm_enabled is False, manual override to True must still evaluate to False
    backend.prefill_gemm_enabled = False
    backend.prefill_shared_precompute = True
    assert backend.prefill_shared_precompute is False


def test_benchmark_passed_requires_all_shapes_gemm_used(monkeypatch, tmp_path):
    """Verify benchmark evaluation logic and main() exit code with fake backend."""
    from collections import defaultdict
    import json
    from types import SimpleNamespace
    from bench.gemm_prefill import benchmark

    # 1. Direct evaluation of benchmark.evaluate_passed
    records_ok = [
        {"shape": "rda_in", "byte_equal": True, "used_gemm": True},
        {"shape": "shared_gu", "byte_equal": True, "used_gemm": True},
    ]
    records_mismatch = [
        {"shape": "rda_in", "byte_equal": False, "used_gemm": True},
        {"shape": "shared_gu", "byte_equal": True, "used_gemm": True},
    ]
    shape_gemm_all = {"rda_in": True, "shared_gu": True}
    shape_gemm_missing = {"rda_in": True, "shared_gu": False}

    # Byte equal holds, but one shape did not use GEMM -> False
    assert benchmark.evaluate_passed(records_ok, shape_gemm_missing) is False

    # Byte mismatch in one record -> False
    assert benchmark.evaluate_passed(records_mismatch, shape_gemm_all) is False

    # Both conditions satisfied -> True
    assert benchmark.evaluate_passed(records_ok, shape_gemm_all) is True

    # 2. End-to-end execution of benchmark.run() and benchmark.main() using fake backend
    class FakeNpuBackend:
        def __init__(self, weights_dir=None, mock=False, missing_gemm_shape=None):
            self.missing_gemm_shape = missing_gemm_shape
            self.call_counts = defaultdict(int)
            self.xclbin_switches = 0
            self.timings = {}
            self.xclbin_calls = {}
            self.comp_map = defaultdict(lambda: SimpleNamespace(orig_k=16))
            cases = benchmark._cases(1, 3, 1)
            self.shape_by_name = {name: sname for sname, name in cases}

        def set_step(self, step):
            pass

        def preload(self):
            pass

        def reset_timings(self):
            pass

        def linear(self, name, row):
            return np.zeros(16, dtype=np.uint16)

        def linear_batch(self, name, x):
            shape_name = self.shape_by_name.get(name, "unknown")
            if shape_name != self.missing_gemm_shape:
                self.call_counts[f"gemm:{shape_name}"] += 1
            return np.zeros((x.shape[0], 16), dtype=np.uint16)

    # When shared_gu does not use GEMM, run() returns passed=False and main() exits with 1
    monkeypatch.setattr(
        benchmark,
        "NpuBackend",
        lambda **kwargs: FakeNpuBackend(missing_gemm_shape="shared_gu"),
    )
    out_fail = tmp_path / "bench_fail.json"
    code_fail = benchmark.main([str(tmp_path), "--iterations", "1", "--switch-iterations", "1", "--output", str(out_fail)])
    assert code_fail == 1
    data_fail = json.loads(out_fail.read_text(encoding="utf-8"))
    assert data_fail["passed"] is False
    assert data_fail["shape_gemm_used"]["shared_gu"] is False
    assert all(r["byte_equal"] for r in data_fail["records"])

    # When all shapes use GEMM, run() returns passed=True and main() exits with 0
    monkeypatch.setattr(
        benchmark,
        "NpuBackend",
        lambda **kwargs: FakeNpuBackend(missing_gemm_shape=None),
    )
    out_ok = tmp_path / "bench_ok.json"
    code_ok = benchmark.main([str(tmp_path), "--iterations", "1", "--switch-iterations", "1", "--output", str(out_ok)])
    assert code_ok == 0
    data_ok = json.loads(out_ok.read_text(encoding="utf-8"))
    assert data_ok["passed"] is True
    assert all(data_ok["shape_gemm_used"].values())
    assert all(r["byte_equal"] for r in data_ok["records"])
