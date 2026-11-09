# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import types

import numpy as np
import pytest

from bench.gemm_prefill.rda_context import _rda_inputs, run as run_rda_context
from bench.rda_state.rda_state_1col import pack_head_bundle
from host.weights import bf16_to_f32
import npu.backend as backend_mod
from npu.backend import NpuBackend, PREFILL_GEMM_XCLBIN_ID


DECODE_XCLBIN_ID = "gemv_unified"


def _inputs(seed: int = 20261102):
    rng = np.random.default_rng(seed)
    return _rda_inputs(rng, 1)[0]


def _cheap_numpy_update(self, layer, q, k, v, log_alpha, beta):
    return np.zeros((40, 128), dtype=np.uint16)


def test_rda_inputs_shape_contract_and_pack_head_bundle():
    rng = np.random.default_rng(20261102)
    updates = _rda_inputs(rng, 8)
    assert len(updates) == 8

    for q, k, v, log_alpha, beta in updates:
        assert q.shape == (40, 128)
        assert q.dtype == np.uint16
        assert k.shape == (40, 128)
        assert k.dtype == np.uint16
        assert v.shape == (40, 128)
        assert v.dtype == np.uint16
        assert log_alpha.shape == (40, 128)
        assert log_alpha.dtype == np.float32
        assert beta.shape == (40,)
        assert beta.dtype == np.uint16

        # Value ranges: negative decay and normalized q/k
        assert np.all(log_alpha <= 0.0)
        k_norms = np.linalg.norm(bf16_to_f32(k), axis=-1)
        np.testing.assert_allclose(k_norms, 1.0, atol=1e-2)
        q_norms = np.linalg.norm(bf16_to_f32(q), axis=-1)
        np.testing.assert_allclose(q_norms, 128.0 ** -0.5, atol=1e-2)

        # Hardware pack_head_bundle contract
        q_f32 = bf16_to_f32(q)
        k_f32 = bf16_to_f32(k)
        v_f32 = bf16_to_f32(v)
        log_alpha_f32 = log_alpha.astype(np.float32, copy=False)
        beta_f32 = bf16_to_f32(beta)
        a_f32 = np.exp(log_alpha_f32)

        bundle = pack_head_bundle(q_f32, k_f32, a_f32, v_f32, beta_f32)
        assert bundle.shape == (40, 640)
        assert bundle.dtype == np.float32
        np.testing.assert_array_equal(bundle[:, 0:128], q_f32)
        np.testing.assert_array_equal(bundle[:, 128:256], k_f32)
        np.testing.assert_array_equal(bundle[:, 256:384], a_f32)
        np.testing.assert_array_equal(bundle[:, 384:512], v_f32)
        np.testing.assert_array_equal(bundle[:, 512], beta_f32)
        np.testing.assert_array_equal(bundle[:, 513:640], 0.0)


def test_rda_inputs_mock_backend_64_steps():
    backend = NpuBackend(mock=True)
    rng = np.random.default_rng(20261102)
    updates = _rda_inputs(rng, 64)

    initial_state = np.zeros((40, 128, 128), dtype=np.float32)
    backend.set_rda_state(1, initial_state)

    outputs_decode = []
    for values in updates:
        out = backend.rda_update(1, *values)
        assert out.shape == (40, 128)
        assert out.dtype == np.uint16
        assert np.all(np.isfinite(bf16_to_f32(out)))
        outputs_decode.append(out)

    state_decode = backend.get_rda_state(1)
    assert state_decode.shape == (1, 40, 128, 128)
    assert state_decode.dtype == np.float32
    assert np.all(np.isfinite(state_decode))

    # Test prefill_rda path matches byte-for-byte
    backend.set_rda_state(1, initial_state)
    backend.prefill_rda_enabled = True
    backend._prefill_xclbin_path = Path("prefill-s2a.xclbin")
    backend._last_active_xclbin = PREFILL_GEMM_XCLBIN_ID

    outputs_prefill = []
    for values in updates:
        out = backend.rda_update(1, *values)
        outputs_prefill.append(out)

    state_prefill = backend.get_rda_state(1)
    np.testing.assert_array_equal(np.stack(outputs_decode), np.stack(outputs_prefill))
    np.testing.assert_array_equal(state_decode, state_prefill)


def test_rda_context_runner_breakdown_mock(monkeypatch):
    def fake_execute_gemm_batch(self, bundle_name, bmeta, shape_name, x_u16):
        self._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
        return np.zeros((x_u16.shape[0], bmeta.m), dtype=np.uint16)

    monkeypatch.setattr(NpuBackend, "_execute_gemm_batch", fake_execute_gemm_batch)

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        args = types.SimpleNamespace(
            weights_dir=None,
            rda_layer=1,
            seed=20261102,
            mock=True,
            output=tmp.name,
        )
        result = run_rda_context(args)

    assert result["passed"] is True
    assert result["calls"] == 64
    assert result["byte_parity"]["all_outputs_equal"] is True
    assert result["byte_parity"]["final_state_equal"] is True
    assert result["decode_call_counts"].get("rda_npu_prefill", 0) == 0
    assert result["prefill_call_counts"].get("rda_npu_prefill", 0) == 64
    assert result["prefill_xclbin_calls"].get(PREFILL_GEMM_XCLBIN_ID, 0) == 68

    ms_per_call = result["ms_per_call"]
    for key in (
        "decode_rda_continuous",
        "prefill_rda_continuous",
        "prefill_first_after_gemm",
        "decode_first_after_prefill",
        "decode_first_return",
    ):
        assert key in ms_per_call, f"Missing key {key} in ms_per_call"
        stats = ms_per_call[key]
        assert "calls" in stats
        assert "mean" in stats
        assert "p50" in stats
        assert "p90" in stats

    assert ms_per_call["decode_rda_continuous"]["calls"] == 64
    assert ms_per_call["prefill_rda_continuous"]["calls"] == 60
    assert ms_per_call["prefill_first_after_gemm"]["calls"] == 4
    assert ms_per_call["decode_first_after_prefill"]["calls"] == 4
    assert ms_per_call["decode_first_return"]["calls"] == 4


@pytest.mark.parametrize(
    "last,compiled,enabled,expected",
    (
        (PREFILL_GEMM_XCLBIN_ID, True, True, PREFILL_GEMM_XCLBIN_ID),
        (DECODE_XCLBIN_ID, True, True, DECODE_XCLBIN_ID),
        (PREFILL_GEMM_XCLBIN_ID, True, False, DECODE_XCLBIN_ID),
        (PREFILL_GEMM_XCLBIN_ID, False, True, DECODE_XCLBIN_ID),
    ),
)
def test_rda_context_friendly_selection(last, compiled, enabled, expected):
    backend = NpuBackend(mock=True)
    backend.prefill_rda_enabled = enabled
    backend._prefill_xclbin_path = Path("prefill-s2a.xclbin") if compiled else None
    backend._last_active_xclbin = last
    backend._rda_update_numpy = types.MethodType(_cheap_numpy_update, backend)

    backend.rda_update(1, *_inputs())

    assert backend._last_active_xclbin == expected
    assert backend.xclbin_calls == {expected: 1}
    assert backend.call_counts["rda_npu_prefill"] == (
        1 if expected == PREFILL_GEMM_XCLBIN_ID else 0
    )


def test_rda_switch_environment_zero_always_selects_decode(monkeypatch):
    monkeypatch.setenv("RAON_PREFILL_RDA", "0")
    backend = NpuBackend(mock=True)
    backend._prefill_xclbin_path = Path("prefill-s2a.xclbin")
    backend._last_active_xclbin = PREFILL_GEMM_XCLBIN_ID
    backend._rda_update_numpy = types.MethodType(_cheap_numpy_update, backend)

    backend.rda_update(1, *_inputs())

    assert backend._last_active_xclbin == DECODE_XCLBIN_ID
    assert backend.call_counts["rda_npu_prefill"] == 0


class _FakeTensor:
    def __init__(self, value):
        self.value = np.array(value, copy=True)

    def numpy(self):
        return self.value

    def __setitem__(self, key, value):
        self.value[key] = value


def _fake_hardware_backend(monkeypatch):
    backend = NpuBackend(mock=True)
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend.mock = False
    backend._prefill_xclbin_path = Path("prefill-s2a.xclbin")
    backend.rda_states[1] = [
        _FakeTensor(np.zeros((40, 128, 128), np.float32)),
        _FakeTensor(np.zeros((40, 128, 128), np.float32)),
    ]
    backend.rda_active_slot[1] = 0
    backend._work_buffers["rda_head_in"] = _FakeTensor(
        np.zeros((40, 640), np.float32)
    )
    backend._work_buffers["rda_out"] = _FakeTensor(np.zeros((40, 128), np.float32))

    monkeypatch.setitem(
        sys.modules,
        "bench.rda_state.rda_state",
        types.SimpleNamespace(pack_head_bundle=pack_head_bundle),
    )
    calls = []

    def run(label):
        def execute(s_in, s_out, head_in, out):
            calls.append(
                (label, id(s_in), id(s_out), id(head_in), id(out))
            )
            delta = head_in.value[:, 0][:, None, None]
            s_out.value[...] = s_in.value + delta
            out.value[...] = s_out.value[:, 0, :]

        return execute

    backend._compiled_ops["rda_step"] = run("decode")
    backend._compiled_ops["prefill_rda"] = run("prefill")
    return backend, calls


def test_mixed_rda_paths_share_bos_slots_and_match_one_path(monkeypatch):
    decode, decode_calls = _fake_hardware_backend(monkeypatch)
    mixed, mixed_calls = _fake_hardware_backend(monkeypatch)
    rng = np.random.default_rng(20261102)
    updates = _rda_inputs(rng, 6)

    decode_outputs = []
    mixed_outputs = []
    for index, values in enumerate(updates):
        decode._last_active_xclbin = DECODE_XCLBIN_ID
        decode_outputs.append(decode.rda_update(1, *values))
        mixed._last_active_xclbin = (
            PREFILL_GEMM_XCLBIN_ID if index % 2 == 0 else DECODE_XCLBIN_ID
        )
        mixed_outputs.append(mixed.rda_update(1, *values))

    np.testing.assert_array_equal(np.stack(mixed_outputs), np.stack(decode_outputs))
    np.testing.assert_array_equal(decode.get_rda_state(1), mixed.get_rda_state(1))
    assert mixed.rda_active_slot[1] == decode.rda_active_slot[1] == 0
    assert {call[0] for call in mixed_calls} == {"decode", "prefill"}
    assert {call[3] for call in mixed_calls} == {id(mixed._work_buffers["rda_head_in"])}
    assert {call[4] for call in mixed_calls} == {id(mixed._work_buffers["rda_out"])}
    assert {call[1] for call in mixed_calls} == {
        id(mixed.rda_states[1][0]), id(mixed.rda_states[1][1])
    }
    assert len(decode_calls) == len(mixed_calls) == len(updates)


def test_context_friendly_rda_reduces_scripted_switches():
    def run(enabled):
        backend = NpuBackend(mock=True)
        backend.prefill_rda_enabled = enabled
        backend._prefill_xclbin_path = Path("prefill-s2a.xclbin")
        backend._rda_update_numpy = types.MethodType(_cheap_numpy_update, backend)
        for _ in range(4):
            backend._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
            backend.rda_update(1, *_inputs())
        return backend.xclbin_switches

    assert run(True) < run(False)


def test_failed_operations_do_not_update_active_xclbin_or_switches():
    backend = NpuBackend(mock=True)
    backend.prefill_rda_enabled = True
    backend._prefill_xclbin_path = Path("prefill-s2a.xclbin")

    # Initial state
    assert backend._last_active_xclbin is None
    assert backend.xclbin_switches == 0
    assert backend._use_prefill_rda() is False

    # 1. Failed prefill GEMM (missing weights raises FileNotFoundError)
    with pytest.raises(FileNotFoundError):
        backend.linear_batch(
            "model.layers.1.self_attn.q_proj.weight",
            np.zeros((1, 2560), dtype=np.uint16),
        )

    assert backend._last_active_xclbin is None
    assert backend.xclbin_switches == 0
    assert backend.xclbin_calls == {}
    assert backend._use_prefill_rda() is False

    # 2. Failed decode GEMV
    backend._last_active_xclbin = "sentinel_context"
    backend.xclbin_switches = 1
    with pytest.raises(FileNotFoundError):
        backend.linear(
            "model.layers.1.self_attn.q_proj.weight",
            np.zeros(2560, dtype=np.uint16),
        )

    assert backend._last_active_xclbin == "sentinel_context"
    assert backend.xclbin_switches == 1
    assert "gemv_unified" not in backend.xclbin_calls

    # 3. Failed RDA update
    def raise_rda_error(self, layer, q, k, v, log_alpha, beta):
        raise RuntimeError("simulated RDA update failure")

    backend._rda_update_numpy = types.MethodType(raise_rda_error, backend)
    with pytest.raises(RuntimeError):
        backend.rda_update(1, *_inputs())

    assert backend._last_active_xclbin == "sentinel_context"
    assert backend.xclbin_switches == 1
    assert "rda_step" not in backend.xclbin_calls
    assert "gemv_unified" not in backend.xclbin_calls


def test_rda_context_runner_fault_injection_fails_when_prefill_rda_disabled(monkeypatch):
    def fake_execute_gemm_batch(self, bundle_name, bmeta, shape_name, x_u16):
        self._record_xclbin_use(PREFILL_GEMM_XCLBIN_ID)
        return np.zeros((x_u16.shape[0], bmeta.m), dtype=np.uint16)

    monkeypatch.setattr(NpuBackend, "_execute_gemm_batch", fake_execute_gemm_batch)
    # Fault injection: always disable prefill RDA selection
    monkeypatch.setattr(NpuBackend, "_use_prefill_rda", lambda self: False)

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        args = types.SimpleNamespace(
            weights_dir=None,
            rda_layer=1,
            seed=20261102,
            mock=True,
            output=tmp.name,
        )
        result = run_rda_context(args)

    assert result["passed"] is False
    assert result["byte_parity"]["all_outputs_equal"] is True
    assert result["byte_parity"]["final_state_equal"] is True
    assert result["prefill_call_counts"].get("rda_npu_prefill", 0) == 0
