# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Verification tests for stage 2b-1 prefill shared_gu GEMM execution and parity."""

from __future__ import annotations

import numpy as np
import pytest

from bench.prefill_replay.harness import run_prefill_replay
from host.raon import RaonModel
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.test_prefill_parity import NpuCoupledMockBackend

_ = tiny_raon_model


@pytest.mark.parametrize("n_tokens", [1, 16, 17, 64])
def test_prefill_parity_and_gemm_calls(tiny_raon_model, n_tokens: int):
    """(나) Verify bit-for-bit parity and GEMM call count assertions for prefill with shared_gu precompute."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    backend.prefill_shared_precompute = True
    model = RaonModel(host_cfg, weights, backend=backend)

    rng = np.random.default_rng(20260000 + n_tokens)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=n_tokens).tolist()

    # Pre-calculate expected counts
    num_moe_layers = sum(1 for i in range(host_cfg.num_hidden_layers) if host_cfg.is_moe_layer(i))
    chunk_size = 16
    num_chunks = (n_tokens + chunk_size - 1) // chunk_size

    # Reset counters and run harness
    backend.reset()
    res = run_prefill_replay(model, token_ids, chunk=chunk_size)
    assert res["match"] is True, f"Parity mismatch on {n_tokens} tokens: {res['first_mismatch']}"

    # Verify calls during prefill:
    # 1. run_prefill_replay runs advance first (which does not use linear_batch for shared experts),
    #    then runs prefill (which uses linear_batch).
    # 2. Advance makes 0 shared_gu_batch_calls and n_tokens * num_moe_layers single shared calls in moe_experts.
    # 3. Prefill makes 0 shared_gemv_calls and num_chunks * num_moe_layers shared_gu_batch_calls!
    # Let's isolate the prefill phase directly to assert exact counts.
    model.reset_state()
    backend.shared_gemv_calls = 0
    backend.shared_gu_batch_calls = 0
    backend.shared_gu_obj_mismatches = 0

    _ = model.prefill(token_ids, chunk=chunk_size, logits_positions=[n_tokens - 1])

    assert backend.shared_gemv_calls == 0, (
        f"Prefill with shared precompute must make 0 shared GEMV calls, got {backend.shared_gemv_calls}"
    )
    assert backend.shared_gu_batch_calls == num_moe_layers * num_chunks, (
        f"Prefill must make exactly 1 shared_gu batch call per MoE layer per chunk, "
        f"expected {num_moe_layers * num_chunks}, got {backend.shared_gu_batch_calls}"
    )
    assert backend.shared_gu_obj_mismatches == 0, (
        f"Object identity mismatch count should be 0, got {backend.shared_gu_obj_mismatches}"
    )


def test_prefill_switch_disabled_uses_gemv(tiny_raon_model):
    """(나) When prefill_shared_precompute is False, shared GEMV is called for every token and output matches."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    backend.prefill_shared_precompute = False
    model = RaonModel(host_cfg, weights, backend=backend)

    n_tokens = 17
    rng = np.random.default_rng(9999)
    token_ids = rng.integers(0, host_cfg.vocab_size, size=n_tokens).tolist()

    num_moe_layers = sum(1 for i in range(host_cfg.num_hidden_layers) if host_cfg.is_moe_layer(i))
    chunk_size = 16

    res = run_prefill_replay(model, token_ids, chunk=chunk_size)
    assert res["match"] is True, f"Mismatch when switch disabled: {res['first_mismatch']}"

    # Isolate prefill execution
    model.reset_state()
    backend.shared_gemv_calls = 0
    backend.shared_gu_batch_calls = 0

    _ = model.prefill(token_ids, chunk=chunk_size, logits_positions=[n_tokens - 1])

    assert backend.shared_gemv_calls == num_moe_layers * n_tokens, (
        f"Expected {num_moe_layers * n_tokens} shared GEMV calls, got {backend.shared_gemv_calls}"
    )
    assert backend.shared_gu_batch_calls == 0, (
        f"Expected 0 shared_gu batch calls when switch is disabled, got {backend.shared_gu_batch_calls}"
    )


def test_intentional_mutation_t_plus_one_fails(tiny_raon_model, monkeypatch: pytest.MonkeyPatch):
    """(라) Deliberately providing row t+1 instead of row t must be caught and fail."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    backend.prefill_shared_precompute = True
    model = RaonModel(host_cfg, weights, backend=backend)

    orig_linear_batch = backend.linear_batch

    def shifted_linear_batch(name: str, x: np.ndarray) -> np.ndarray:
        out = orig_linear_batch(name, x)
        if "moe.shared_experts." in name:
            # Shift rows so token t receives row t+1 (shifted by 1)
            return np.roll(out, -1, axis=0)
        return out

    monkeypatch.setattr(backend, "linear_batch", shifted_linear_batch)

    token_ids = [42, 161, 88]
    # Supplying row t+1 instead of row t causes moe_experts contract check to fail
    with pytest.raises(ValueError, match="shared_gu mismatch in moe_experts"):
        model.prefill(token_ids, chunk=16)


def test_advance_path_never_supplies_shared_gu(tiny_raon_model):
    """(마) Advance path must always call moe_experts with legacy signature (positional args only, no shared_gu kwarg)."""
    _, host_cfg, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    backend.prefill_shared_precompute = True
    model = RaonModel(host_cfg, weights, backend=backend)

    advance_calls = []
    orig_moe_experts = backend.moe_experts

    def tracking_moe_experts(*args, **kwargs):
        advance_calls.append((len(args), dict(kwargs)))
        return orig_moe_experts(*args, **kwargs)

    backend.moe_experts = tracking_moe_experts

    model.reset_state()
    for tok in [42, 161, 88]:
        _ = model.advance(tok, need_logits=True)

    assert len(advance_calls) > 0
    for nargs, kwargs in advance_calls:
        # Legacy advance call passes exactly 4 positional args (layer_idx, hidden, top, router_weights)
        # and keyword args (gate_beta, value_beta). 'shared_gu' key must NOT be present in kwargs.
        assert nargs == 4, f"Advance path expected 4 positional args to moe_experts, got {nargs}"
        assert "shared_gu" not in kwargs, (
            f"Advance path must not pass shared_gu key in kwargs, found: {list(kwargs.keys())}"
        )

    # In contrast, prefill path with precompute enabled MUST pass shared_gu in kwargs
    prefill_calls = []

    def tracking_prefill_moe_experts(*args, **kwargs):
        prefill_calls.append((len(args), dict(kwargs)))
        return orig_moe_experts(*args, **kwargs)

    backend.moe_experts = tracking_prefill_moe_experts
    model.reset_state()
    _ = model.prefill([42, 161, 88], chunk=16)

    assert len(prefill_calls) > 0
    for nargs, kwargs in prefill_calls:
        assert nargs == 4
        assert "shared_gu" in kwargs
        assert kwargs["shared_gu"] is not None
