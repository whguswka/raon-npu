# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the byte-identical fast RDA convolution path."""

from __future__ import annotations

import types

import numpy as np
import pytest

from host.raon import RaonModel
from host.weights import bf16_to_f32, f32_to_bf16_rtne


KERNEL = 4
HEADS = 40
HEAD_DIM = 128
PROJ = HEADS * HEAD_DIM
SPECIAL = np.array(
    [0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC0, 0x7FA0, 0xFFC1, 0x0000],
    dtype=np.uint16,
)


def _model(proj: int = PROJ, *, fast: bool | None = None) -> RaonModel:
    model = object.__new__(RaonModel)
    model.config = types.SimpleNamespace(rda_conv_kernel_size=KERNEL)
    model.conv_states = {}
    model._conv_fast = {}
    model._conv_wt = {}
    model._rda_conv_fast = fast
    model._rda_conv_fast_loop = None
    model._prefill_rda_block = None
    model._in_rda_overlap = False
    return model


def _inputs(step: int, layer: int, proj: int = PROJ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1000 * layer + step)
    values = f32_to_bf16_rtne(
        (rng.standard_normal(3 * proj) * np.exp2(rng.integers(-6, 6, 3 * proj))).astype(np.float32)
    )
    if step % 7 == 3:
        indices = rng.choice(3 * proj, 64, replace=False)
        values[indices] = SPECIAL[rng.integers(0, len(SPECIAL), 64)]
    if step % 11 == 5:
        values[: proj // 2] = 0x8000
    return values[:proj], values[proj : 2 * proj], values[2 * proj :]


def _weights(layer: int, proj: int = PROJ) -> np.ndarray:
    rng = np.random.default_rng(77 + layer)
    weight = (rng.standard_normal((3 * proj, KERNEL)) * 0.3).astype(np.float32)
    weight[:256] = 0.0
    weight[256:512] = -0.0
    weight[512:520, 1] = np.inf
    return weight


def _copy_histories(model: RaonModel) -> dict[int, dict[str, np.ndarray]]:
    return {
        layer: {kind: np.array(history, copy=True) for kind, history in histories.items()}
        for layer, histories in model.conv_states.items()
    }


def _assert_same(reference: RaonModel, candidate: RaonModel, layer: int, ref_out, candidate_out) -> None:
    for expected, actual in zip(ref_out, candidate_out):
        np.testing.assert_array_equal(expected.view(np.uint16), actual.view(np.uint16))
    for kind in ("q", "k", "v"):
        expected = np.asarray(reference.conv_states[layer][kind], dtype=np.float32).view(np.uint32)
        actual = np.asarray(candidate.conv_states[layer][kind], dtype=np.float32).view(np.uint32)
        np.testing.assert_array_equal(expected, actual)


def test_numpy_sum_matches_sequential_float32_addition() -> None:
    rng = np.random.default_rng(1)
    rows = (
        rng.standard_normal((100_000, KERNEL)) * np.exp2(rng.integers(-20, 20, (100_000, KERNEL)))
    ).astype(np.float32)
    sequential = np.float32(0.0) + rows[:, 0]
    for column in range(1, KERNEL):
        sequential = sequential + rows[:, column]
    np.testing.assert_array_equal(np.sum(rows, axis=-1).view(np.uint32), sequential.view(np.uint32))

    negative_zero = np.float32(-0.0)
    zero_rows = np.array(
        [
            [negative_zero] * KERNEL,
            [0.0] + [negative_zero] * (KERNEL - 1),
            [negative_zero] * (KERNEL - 1) + [0.0],
        ],
        dtype=np.float32,
    )
    assert not np.any(np.signbit(np.sum(zero_rows, axis=-1)))


def test_reference_fast_and_mixed_paths_are_byte_identical() -> None:
    reference = _model(fast=False)
    candidate = _model(fast=True)
    mixed = _model(fast=True)
    weights = {layer: _weights(layer) for layer in (0, 1)}

    with np.errstate(all="ignore"):
        for step in range(180):
            if step == 60:
                for model in (reference, candidate, mixed):
                    snapshot = _copy_histories(model)
                    model.conv_states.clear()
                    model.conv_states.update(snapshot)
            if step == 120:
                for model in (reference, candidate, mixed):
                    model.conv_states.clear()
                    model._conv_fast.clear()
                    rng = np.random.default_rng(5)
                    model.conv_states[1] = {
                        kind: f32_to_bf16_rtne(rng.standard_normal((PROJ, KERNEL - 1)).astype(np.float32))
                        for kind in ("q", "k", "v")
                    }
            for layer in (0, 1):
                q, k, v = _inputs(step, layer)
                ref_out = reference._conv_step_qkv(layer, q, k, v, weights[layer])
                candidate_out = candidate._conv_step_qkv(layer, q, k, v, weights[layer])
                mixed.rda_conv_fast = (step // 13) % 2 == 0
                mixed_out = mixed._conv_step_qkv(layer, q, k, v, weights[layer])
                _assert_same(reference, candidate, layer, ref_out, candidate_out)
                _assert_same(reference, mixed, layer, ref_out, mixed_out)


def test_fast_path_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAON_RDA_CONV_FAST", raising=False)
    monkeypatch.delenv("RAON_RDA_CONV_FAST_LOOP", raising=False)
    model = _model(proj=8)
    calls: list[int] = []

    def record_fast(layer, q, k, v, weight):
        calls.append(layer)
        return q, k, v

    model._conv_step_qkv_fast = record_fast
    q = k = v = np.zeros(8, dtype=np.uint16)
    weight = np.zeros((24, KERNEL), dtype=np.float32)

    model._conv_step_qkv(0, q, k, v, weight)
    assert calls == [0]
    model._in_rda_overlap = True
    model._conv_step_qkv(1, q, k, v, weight)
    assert calls == [0, 1]

    for off in ("0", "false", "False", "no"):
        monkeypatch.setenv("RAON_RDA_CONV_FAST_LOOP", off)
        model._conv_step_qkv(2, q, k, v, weight)
        assert calls == [0, 1]

    monkeypatch.setenv("RAON_RDA_CONV_FAST_LOOP", "1")
    model._conv_step_qkv(2, q, k, v, weight)
    assert calls == [0, 1, 2]

    monkeypatch.setenv("RAON_RDA_CONV_FAST", "0")
    model._in_rda_overlap = False
    model._conv_step_qkv(3, q, k, v, weight)
    assert calls == [0, 1, 2]
    model._in_rda_overlap = True
    model._conv_step_qkv(3, q, k, v, weight)
    assert calls == [0, 1, 2]

    model.rda_conv_fast = True
    model.rda_conv_fast_loop = False
    model._in_rda_overlap = False
    model._conv_step_qkv(4, q, k, v, weight)
    assert calls == [0, 1, 2, 4]
    model._in_rda_overlap = True
    model._conv_step_qkv(5, q, k, v, weight)
    assert calls == [0, 1, 2, 4]
    model.rda_conv_fast_loop = True
    model._conv_step_qkv(6, q, k, v, weight)
    assert calls == [0, 1, 2, 4, 6]


def test_reset_state_clears_fast_cache() -> None:
    class ResetBackend:
        reset_called = False

        def reset(self) -> None:
            self.reset_called = True

    model = _model(proj=8)
    model.backend = ResetBackend()
    model.last_intermediates = {"present": np.array([1])}
    model.conv_states[0] = {"q": np.ones((8, 3), dtype=np.float32)}
    model._conv_fast[0] = (np.empty((4, 24)), ())

    model.reset_state()

    assert model.conv_states == {}
    assert model._conv_fast == {}
    assert model.last_intermediates == {}
    assert model.backend.reset_called


class _OverlapBackend:
    def __init__(self, *, fail: bool = False):
        self.fail = fail
        self.aborts = 0
        self.stages = 0
        self.tokens: dict[int, int] = {}

    def rda_stage(self, layer, inputs):
        assert inputs
        self.stages += 1
        if self.fail:
            raise RuntimeError("stage failed")
        self.tokens[self.stages] = len(inputs)
        return self.stages

    def rda_launch(self, ticket) -> None:
        pass

    def rda_wait(self, ticket) -> None:
        pass

    def rda_read(self, ticket) -> np.ndarray:
        return np.full((self.tokens[ticket], 1), ticket, dtype=np.uint16)

    def rda_abort_all(self) -> None:
        self.aborts += 1


def _overlap_model(backend: _OverlapBackend, observations: list[bool]) -> RaonModel:
    model = _model(proj=1)
    model.backend = backend

    def rda_inputs(*args):
        observations.append(model._in_rda_overlap)
        value = np.zeros(1, dtype=np.uint16)
        return value, value, value, value.astype(np.float32), value

    model._rda_inputs = rda_inputs
    model._rda_output = lambda layer, output, gate: output
    return model


def test_overlap_flag_is_true_during_loop_and_cleared_after_success() -> None:
    observations: list[bool] = []
    model = _overlap_model(_OverlapBackend(), observations)
    batch = np.zeros((2, 1), dtype=np.uint16)

    result = model._rda_tokens_overlapped(0, batch, batch, batch, batch, batch, batch, batch)

    assert result.shape == (2, 1)
    assert observations and all(observations)
    assert model._in_rda_overlap is False


def test_overlap_flag_is_cleared_after_exception() -> None:
    observations: list[bool] = []
    backend = _OverlapBackend(fail=True)
    model = _overlap_model(backend, observations)
    batch = np.zeros((1, 1), dtype=np.uint16)

    with pytest.raises(RuntimeError, match="stage failed"):
        model._rda_tokens_overlapped(0, batch, batch, batch, batch, batch, batch, batch)

    assert observations == [True]
    assert model._in_rda_overlap is False
    assert backend.aborts == 1


def test_replaced_checkpoint_histories_rebuild_fast_window() -> None:
    proj = 64
    fast = _model(proj=proj, fast=True)
    weight = _weights(0, proj=proj)
    with np.errstate(all="ignore"):
        for step in range(5):
            fast._conv_step_qkv(0, *_inputs(step, 0, proj=proj), weight)

        snapshot = _copy_histories(fast)
        fast.conv_states.clear()
        fast.conv_states.update(snapshot)

        reference = _model(proj=proj, fast=False)
        reference.conv_states = {
            layer: {kind: np.array(history, copy=True) for kind, history in histories.items()}
            for layer, histories in snapshot.items()
        }
        q, k, v = _inputs(5, 0, proj=proj)
        fast_out = fast._conv_step_qkv(0, q, k, v, weight)
        ref_out = reference._conv_step_qkv(0, q, k, v, weight)

    _assert_same(reference, fast, 0, ref_out, fast_out)
