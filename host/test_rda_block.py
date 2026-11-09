# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Contracts for multi-token prefill RDA block planning and recovery."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from bench.rda_state.rda_state_multi import RDA_TOKEN_CHOICES
from host.raon import RDA_BLOCK_SIZES, RaonModel, rda_block_plan
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.test_prefill_rda_overlap import OverlapMockBackend


def _new_model(host_cfg, weights, *, block: int | None = None, overlap: bool = True):
    backend = OverlapMockBackend(weights)
    backend.prefill_rda_overlap = overlap
    model = RaonModel(host_cfg, weights, backend=backend)
    if block is not None:
        model.prefill_rda_block = block
    return model, backend


def _snapshot(model: RaonModel, backend: OverlapMockBackend, logits):
    rda = {
        layer: backend.get_rda_state(layer).copy()
        for layer in range(model.config.num_hidden_layers)
        if model.config.is_rda_layer(layer)
    }
    conv = {
        layer: {kind: np.array(value, copy=True) for kind, value in histories.items()}
        for layer, histories in model.conv_states.items()
    }
    return logits, rda, conv


def _assert_snapshot_equal(actual, expected):
    actual_logits, actual_rda, actual_conv = actual
    expected_logits, expected_rda, expected_conv = expected
    assert actual_logits.keys() == expected_logits.keys()
    for pos in actual_logits:
        np.testing.assert_array_equal(actual_logits[pos], expected_logits[pos])
    assert actual_rda.keys() == expected_rda.keys()
    for layer in actual_rda:
        np.testing.assert_array_equal(actual_rda[layer], expected_rda[layer])
    assert actual_conv.keys() == expected_conv.keys()
    for layer in actual_conv:
        assert actual_conv[layer].keys() == expected_conv[layer].keys()
        for kind in actual_conv[layer]:
            np.testing.assert_array_equal(actual_conv[layer][kind], expected_conv[layer][kind])


@pytest.mark.parametrize("block", (1, 2, 4, 8))
def test_rda_block_plan_invariants(block: int):
    for n in range(1, 141):
        plan = rda_block_plan(n, block)
        sizes = [size for _, size in plan]
        assert sum(sizes) == n
        assert set(sizes) <= set(RDA_BLOCK_SIZES)
        assert all(size <= block for size in sizes)
        assert sizes == sorted(sizes, reverse=True)
        assert [start for start, _ in plan] == list(np.cumsum([0, *sizes[:-1]]))
    assert rda_block_plan(0, block) == []


def test_rda_block_plan_examples():
    assert [size for _, size in rda_block_plan(61, 8)] == [8] * 7 + [4, 1]
    assert [size for _, size in rda_block_plan(5, 1)] == [1] * 5
    assert [size for _, size in rda_block_plan(6, 4)] == [4, 2]


def test_prefill_rda_block_switch(monkeypatch: pytest.MonkeyPatch):
    model = object.__new__(RaonModel)
    model._prefill_rda_block = None
    monkeypatch.delenv("RAON_PREFILL_RDA_BLOCK", raising=False)
    assert model.prefill_rda_block == 8
    for value in (1, 2, 4, 8):
        monkeypatch.setenv("RAON_PREFILL_RDA_BLOCK", str(value))
        assert model.prefill_rda_block == value
    for value in ("0", "3", "16", "-8", "x"):
        monkeypatch.setenv("RAON_PREFILL_RDA_BLOCK", value)
        with pytest.raises(ValueError):
            _ = model.prefill_rda_block

    monkeypatch.setenv("RAON_PREFILL_RDA_BLOCK", "3")
    model.prefill_rda_block = 2
    assert model.prefill_rda_block == 2
    model.prefill_rda_block = None
    with pytest.raises(ValueError):
        _ = model.prefill_rda_block
    assert RDA_BLOCK_SIZES == RDA_TOKEN_CHOICES


@pytest.mark.parametrize("block", (1, 2, 4, 8))
def test_block_sizes_match_nonoverlapped_bytes_and_launch_count(
    tiny_raon_model, block: int
):
    _, host_cfg, weights, _ = tiny_raon_model
    rda_layers = sum(
        host_cfg.is_rda_layer(layer) for layer in range(host_cfg.num_hidden_layers)
    )
    lengths = [*range(1, 18), 31, 32, 33, *range(60, 66)]
    for n in lengths:
        tokens = list(range(n))
        reference, reference_backend = _new_model(host_cfg, weights, overlap=False)
        expected = _snapshot(
            reference,
            reference_backend,
            reference.prefill(tokens, chunk=64, logits_positions=range(n)),
        )
        candidate, backend = _new_model(host_cfg, weights, block=block)
        actual = _snapshot(
            candidate,
            backend,
            candidate.prefill(tokens, chunk=64, logits_positions=range(n)),
        )
        _assert_snapshot_equal(actual, expected)
        chunk_sizes = [min(64, n - start) for start in range(0, n, 64)]
        expected_launches = rda_layers * sum(
            len(rda_block_plan(size, block)) for size in chunk_sizes
        )
        assert backend.launch_count == expected_launches


@pytest.mark.parametrize("cut", (5, 13, 61))
def test_split_prefill_matches_single_call(tiny_raon_model, cut: int):
    _, host_cfg, weights, _ = tiny_raon_model
    tokens = list(range(70))
    whole, whole_backend = _new_model(host_cfg, weights, block=8)
    expected = _snapshot(
        whole,
        whole_backend,
        whole.prefill(tokens, chunk=64, logits_positions=[69]),
    )

    split, split_backend = _new_model(host_cfg, weights, block=8)
    split.prefill(tokens[:cut], chunk=64)
    logits = split.prefill(tokens[cut:], chunk=64, logits_positions=[69 - cut])
    actual = _snapshot(split, split_backend, {69: logits[69 - cut]})
    _assert_snapshot_equal(actual, expected)


@pytest.mark.parametrize(
    ("stage", "message"),
    (
        ("rda_stage", "stage failed"),
        ("rda_launch", "launch failed"),
        ("rda_wait", "wait failed"),
        ("rda_read", "read failed"),
        ("_rda_inputs", "inputs failed"),
    ),
)
def test_second_block_failure_aborts_and_reset_recovers(
    tiny_raon_model, monkeypatch: pytest.MonkeyPatch, stage: str, message: str
):
    _, host_cfg, weights, _ = tiny_raon_model
    tokens = list(range(20))
    clean, clean_backend = _new_model(host_cfg, weights, block=8)
    expected = _snapshot(
        clean,
        clean_backend,
        clean.prefill(tokens, chunk=64, logits_positions=range(20)),
    )

    model, backend = _new_model(host_cfg, weights, block=8)
    owner = model if stage == "_rda_inputs" else backend
    original: Callable = getattr(owner, stage)
    calls = 0
    fail_at = 9 if stage == "_rda_inputs" else 2

    def failing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise RuntimeError(message)
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, stage, failing)
    with pytest.raises(RuntimeError, match=message):
        model.prefill(tokens, chunk=64, logits_positions=range(20))
    assert backend.aborted is True
    assert backend._live_tickets == []
    assert model._in_rda_overlap is False

    monkeypatch.setattr(owner, stage, original)
    model.reset_state()
    actual = _snapshot(
        model,
        backend,
        model.prefill(tokens, chunk=64, logits_positions=range(20)),
    )
    _assert_snapshot_equal(actual, expected)
