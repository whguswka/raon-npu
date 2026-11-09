# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Host integration parity for grouped prefill MoE."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bench.prefill_replay.harness import run_prefill_replay
from host.raon import RaonModel
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.test_prefill_parity import NpuCoupledMockBackend

_ = tiny_raon_model


def _model_with_grouped(tiny_raon_model, enabled: bool):
    _, config, weights, _ = tiny_raon_model
    backend = NpuCoupledMockBackend(weights)
    backend.prefill_shared_precompute = True
    backend.prefill_grouped_moe = enabled
    return RaonModel(config, weights, backend=backend), backend, config


@pytest.mark.parametrize(
    ("token_count", "chunk"),
    [
        (1, 16),
        (7, 16),
        (8, 16),
        (9, 16),
        (16, 16),
        (17, 16),
        (64, 16),
        (17, 64),
        (64, 64),
    ],
)
def test_grouped_prefill_matches_token_serial(tiny_raon_model, token_count, chunk):
    model, backend, config = _model_with_grouped(tiny_raon_model, True)
    rng = np.random.default_rng(3100 + token_count + chunk)
    tokens = rng.integers(0, config.vocab_size, size=token_count).tolist()

    result = run_prefill_replay(model, tokens, chunk=chunk)

    assert result["match"] is True, result["first_mismatch"]
    moe_layers = sum(config.is_moe_layer(i) for i in range(config.num_hidden_layers))
    assert backend.grouped_moe_calls == moe_layers * math.ceil(token_count / chunk)
    assert backend.token_moe_calls == 0
    assert backend.shared_down_calls == 0
    assert backend.shared_gemv_calls == 0
    assert backend.shared_gu_batch_calls == backend.grouped_moe_calls


def test_grouped_off_preserves_fallback_and_router_records(tiny_raon_model):
    _, config, _, _ = tiny_raon_model
    rng = np.random.default_rng(4199)
    tokens = rng.integers(0, config.vocab_size, size=17).tolist()
    grouped_model, grouped_backend, _ = _model_with_grouped(tiny_raon_model, True)
    fallback_model, fallback_backend, _ = _model_with_grouped(tiny_raon_model, False)

    grouped_model.reset_state()
    grouped_backend.reset()
    grouped_out = grouped_model.prefill(
        tokens, logits_positions={16}, chunk=16, record_intermediates=True
    )
    grouped_intermediates = {
        key: value.copy()
        for key, value in grouped_model.last_intermediates.items()
        if key.endswith(".router_top") or key.endswith(".router_weights")
    }

    fallback_model.reset_state()
    fallback_backend.reset()
    fallback_out = fallback_model.prefill(
        tokens, logits_positions={16}, chunk=16, record_intermediates=True
    )
    fallback_intermediates = {
        key: value.copy()
        for key, value in fallback_model.last_intermediates.items()
        if key.endswith(".router_top") or key.endswith(".router_weights")
    }

    assert grouped_out.keys() == fallback_out.keys()
    for position in grouped_out:
        assert np.array_equal(grouped_out[position], fallback_out[position])
    assert grouped_intermediates.keys() == fallback_intermediates.keys()
    for key in grouped_intermediates:
        assert np.array_equal(grouped_intermediates[key], fallback_intermediates[key])
    assert grouped_backend.grouped_moe_calls > 0
    assert grouped_backend.token_moe_calls == 0
    assert fallback_backend.grouped_moe_calls == 0
    assert fallback_backend.token_moe_calls > 0


def test_decode_moe_keyword_set_is_unchanged(tiny_raon_model, monkeypatch):
    model, backend, _ = _model_with_grouped(tiny_raon_model, True)
    original = backend.moe_experts
    seen = []

    def record(*args, **kwargs):
        seen.append(set(kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(backend, "moe_experts", record)
    model.reset_state()
    backend.reset()
    model.advance(7, need_logits=False)

    assert seen
    assert all(keys == {"gate_beta", "value_beta"} for keys in seen)
    assert backend.grouped_moe_calls == 0
