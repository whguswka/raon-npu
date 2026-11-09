# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Byte parity, scheduling, failure, and integration tests for threaded prefill mHC."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import host.raon as raon_module
from host.backend import NumpyBackend
from host.raon import RaonConfig, RaonModel, _mhc_run_rows, _mhc_split, rms_norm
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.weights import bf16_to_f32, f32_to_bf16_rtne


def _assert_same_bytes(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes()


def _make_inputs(
    rng: np.random.Generator, tokens: int, streams: int, hidden_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stream_f32 = rng.standard_normal((tokens, streams, hidden_size)).astype(np.float32)
    flat = stream_f32.reshape(-1)
    flat[rng.choice(flat.size, size=max(1, flat.size // 1000), replace=False)] *= 30.0
    stream = f32_to_bf16_rtne(stream_f32)
    branch_out = f32_to_bf16_rtne(
        (rng.standard_normal((tokens, hidden_size)) * 0.3).astype(np.float32)
    )
    return stream, bf16_to_f32(stream), branch_out


def _add_special_rows(
    stream_f32: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    special = stream_f32.copy()
    values = (np.nan, np.inf, -np.inf)
    for row, value in enumerate(values):
        special[row % special.shape[0], row % special.shape[1], row % special.shape[2]] = value
    stream = f32_to_bf16_rtne(special)
    return stream, bf16_to_f32(stream)


def _install_call_counters(monkeypatch, model: RaonModel) -> dict[str, list[tuple[int, int]]]:
    calls: dict[str, list[tuple[int, int]]] = {"route": [], "mix": []}
    lock = threading.Lock()
    for attribute, key in (("_route_mhc_batch", "route"), ("_mhc_mix_batch", "mix")):
        original = getattr(model, attribute)

        def counted(*args, _original=original, _key=key, **kwargs):
            with lock:
                calls[_key].append((args[0].shape[0], threading.get_ident()))
            return _original(*args, **kwargs)

        monkeypatch.setattr(model, attribute, counted)
    return calls


@pytest.fixture(scope="module")
def operating_case() -> tuple[RaonModel, np.ndarray]:
    config = RaonConfig(hidden_size=2560, raon_mhc_streams=4)
    model = RaonModel(config, None, backend=object())
    rng = np.random.default_rng(20261108)
    n = config.raon_mhc_streams
    c = config.hidden_size
    z_size = 2 * n + (n - 1) ** 2
    model._mhc_site_cache["synthetic"] = (
        (rng.standard_normal((z_size, n * c)) * 0.02).astype(np.float32),
        0.8,
        1.1,
        0.7,
        rng.standard_normal(n).astype(np.float32),
        rng.standard_normal(n).astype(np.float32),
        rng.standard_normal((n - 1) ** 2).astype(np.float32),
        -0.2,
    )
    return model, rng.standard_normal(c).astype(np.float32)


def _old_front(
    model: RaonModel, stream_f32: np.ndarray, norm_w: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h_pre, h_post, h_res = model._route_mhc_batch(stream_f32, "synthetic")
    branch_in = rms_norm(
        model._mhc_aggregate_batch(stream_f32, h_pre),
        norm_w,
        eps=model.config.rms_norm_eps,
    )
    return h_pre, h_post, h_res, branch_in


def _old_back(
    model: RaonModel,
    stream_f32: np.ndarray,
    branch_out: np.ndarray,
    h_post: np.ndarray,
    h_res: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    stream = model._mhc_mix_batch(stream_f32, branch_out, h_post, h_res)
    return stream, bf16_to_f32(stream)


@pytest.mark.parametrize(
    ("rows", "threads", "expected"),
    [
        (1, 4, 1),
        (15, 4, 1),
        (16, 4, 1),
        (31, 4, 1),
        (32, 4, 2),
        (47, 4, 2),
        (48, 4, 3),
        (63, 4, 3),
        (64, 4, 4),
        (256, 4, 4),
        (256, 8, 8),
        (256, 1, 1),
        (64, 2, 2),
        (100, 3, 3),
        (40, 16, 2),
    ],
)
def test_mhc_split_table(rows, threads, expected):
    """성질 1: 최소 행 문턱과 스레드 상한 표를 정확히 지킨다."""
    assert _mhc_split(rows, threads) == expected


@pytest.mark.parametrize("rows", [32, 33, 47, 61, 64, 189, 256])
@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_mhc_run_rows_exact_ranges_and_threads(rows, k):
    """성질 2: 공식 경계의 모든 범위를 한 번씩 실행하고 범위 0만 호출 스레드에서 돈다."""
    caller = threading.get_ident()
    calls: list[tuple[int, int, int]] = []
    lock = threading.Lock()

    def run(lo: int, hi: int) -> None:
        with lock:
            calls.append((lo, hi, threading.get_ident()))

    _mhc_run_rows(run, rows, k)
    expected = [(i * rows // k, (i + 1) * rows // k) for i in range(k)]
    assert sorted((lo, hi) for lo, hi, _ in calls) == expected
    by_range = {(lo, hi): ident for lo, hi, ident in calls}
    assert by_range[expected[0]] == caller
    assert all(by_range[item] != caller for item in expected[1:])


def test_mhc_run_rows_waits_when_calling_range_fails():
    """성질 3: 호출 범위 예외를 우선하며 느린 작업 범위가 끝난 뒤 올린다."""
    finished = threading.Event()

    def run(lo: int, hi: int) -> None:
        if lo == 0:
            raise RuntimeError("caller")
        time.sleep(0.2)
        finished.set()

    with pytest.raises(RuntimeError, match="caller"):
        _mhc_run_rows(run, 32, 2)
    assert finished.is_set()


def test_mhc_run_rows_waits_when_worker_fails():
    """성질 3: 빠른 작업 예외가 있어도 다른 느린 작업까지 끝내고 올린다."""
    finished = threading.Event()

    def run(lo: int, hi: int) -> None:
        if lo == 10:
            raise LookupError("worker")
        if lo == 20:
            time.sleep(0.2)
            finished.set()

    with pytest.raises(LookupError, match="worker"):
        _mhc_run_rows(run, 30, 3)
    assert finished.is_set()


def test_mhc_run_rows_raises_first_worker_by_range_order():
    """성질 3: 여러 작업 범위가 실패하면 범위 순서상 첫 예외를 올린다."""
    def run(lo: int, hi: int) -> None:
        if lo == 10:
            time.sleep(0.05)
            raise LookupError("first range")
        if lo == 20:
            raise RuntimeError("second range")

    with pytest.raises(LookupError, match="first range"):
        _mhc_run_rows(run, 30, 3)


def test_mhc_run_rows_waits_for_base_exception():
    """성질 3: 호출 범위의 BaseException도 느린 작업을 기다린 뒤 그대로 올린다."""
    class DeliberateBaseException(BaseException):
        pass

    finished = threading.Event()

    def run(lo: int, hi: int) -> None:
        if lo == 0:
            raise DeliberateBaseException("base")
        time.sleep(0.2)
        finished.set()

    with pytest.raises(DeliberateBaseException, match="base"):
        _mhc_run_rows(run, 32, 2)
    assert finished.is_set()


def test_mhc_run_rows_waits_after_submit_failure(monkeypatch):
    """성질 3: 두 번째 submit 실패도 이미 넣은 첫 작업이 끝난 뒤 올린다."""
    finished = threading.Event()
    real_pool = raon_module._MLA_POOL

    class FailingPool:
        def __init__(self) -> None:
            self.calls = 0

        def submit(self, fn, *args):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("submit two")
            return real_pool.submit(fn, *args)

    def run(lo: int, hi: int) -> None:
        time.sleep(0.2)
        finished.set()

    monkeypatch.setattr(raon_module, "_MLA_POOL", FailingPool())
    with pytest.raises(RuntimeError, match="submit two"):
        _mhc_run_rows(run, 64, 4)
    assert finished.is_set()


@pytest.mark.parametrize("tokens", [1, 15, 16, 31, 32, 33, 61, 64, 189, 256])
def test_operating_size_front_back_byte_parity(operating_case, tokens):
    """성질 4: 모든 T·스레드와 특수값에서 앞뒤 결과 및 제자리 버퍼가 옛 두 줄과 같다."""
    model, norm_w = operating_case
    rng = np.random.default_rng(20261200 + tokens)
    stream, stream_f32, branch_out = _make_inputs(
        rng, tokens, model.config.raon_mhc_streams, model.config.hidden_size
    )
    variants = [(stream, stream_f32), _add_special_rows(stream_f32)]

    with np.errstate(all="ignore"):
        for input_stream, input_f32 in variants:
            old_front = _old_front(model, input_f32, norm_w)
            expected_stream, expected_f32 = _old_back(
                model, input_f32, branch_out, old_front[1], old_front[2]
            )
            one_front = model._mhc_front_batch(input_f32, "synthetic", norm_w, 1)
            for actual, expected in zip(one_front, old_front):
                _assert_same_bytes(actual, expected)

            for threads in (1, 2, 3, 4, 8):
                actual_front = model._mhc_front_batch(input_f32, "synthetic", norm_w, threads)
                for actual, expected in zip(actual_front, one_front):
                    _assert_same_bytes(actual, expected)

                actual_stream = input_stream.copy()
                actual_f32 = input_f32.copy()
                stream_object = actual_stream
                f32_object = actual_f32
                stream_buffer = actual_stream.view()
                f32_buffer = actual_f32.view()
                model._mhc_back_batch(
                    actual_stream,
                    actual_f32,
                    branch_out,
                    actual_front[1],
                    actual_front[2],
                    threads,
                )
                assert actual_stream is stream_object
                assert actual_f32 is f32_object
                assert np.shares_memory(actual_stream, stream_buffer)
                assert np.shares_memory(actual_f32, f32_buffer)
                _assert_same_bytes(actual_stream, expected_stream)
                _assert_same_bytes(actual_f32, expected_f32)


@pytest.mark.parametrize(
    ("tokens", "threads"), [(31, 8), (32, 2), (33, 4), (61, 4), (64, 8)]
)
def test_front_back_call_shapes_and_worker_use(operating_case, monkeypatch, tokens, threads):
    """성질 5: 앞은 공식 범위별, 뒤는 최대 8행씩 호출되며 단일 경로는 한 번만 호출된다."""
    model, norm_w = operating_case
    rng = np.random.default_rng(20261300 + tokens + threads)
    stream, stream_f32, branch_out = _make_inputs(
        rng, tokens, model.config.raon_mhc_streams, model.config.hidden_size
    )
    calls = _install_call_counters(monkeypatch, model)
    caller = threading.get_ident()
    h_pre, h_post, h_res, _ = model._mhc_front_batch(
        stream_f32, "synthetic", norm_w, threads
    )
    assert h_pre.shape[0] == tokens
    model._mhc_back_batch(stream, stream_f32, branch_out, h_post, h_res, threads)
    k = _mhc_split(tokens, threads)
    if k == 1:
        assert calls["route"] == [(tokens, caller)]
        assert calls["mix"] == [(tokens, caller)]
    else:
        expected_sizes = [
            (i + 1) * tokens // k - i * tokens // k for i in range(k)
        ]
        assert sorted(rows for rows, _ in calls["route"]) == sorted(expected_sizes)
        assert any(ident != caller for _, ident in calls["route"])
        assert sum(rows for rows, _ in calls["mix"]) == tokens
        assert all(rows <= 8 for rows, _ in calls["mix"])
        assert any(ident != caller for _, ident in calls["mix"])


def test_threaded_front_back_is_repeatable(operating_case):
    """성질 6: T=256·4스레드 앞뒤를 10회 반복해 경합에 따른 바이트 변화를 탐지한다."""
    model, norm_w = operating_case
    rng = np.random.default_rng(20261400)
    source_stream, source_f32, branch_out = _make_inputs(
        rng, 256, model.config.raon_mhc_streams, model.config.hidden_size
    )
    expected = None
    for _ in range(10):
        front = model._mhc_front_batch(source_f32, "synthetic", norm_w, 4)
        stream = source_stream.copy()
        stream_f32 = source_f32.copy()
        model._mhc_back_batch(stream, stream_f32, branch_out, front[1], front[2], 4)
        current = (*front, stream, stream_f32)
        if expected is None:
            expected = tuple(value.copy() for value in current)
        else:
            for actual, reference in zip(current, expected):
                _assert_same_bytes(actual, reference)


def _randomize_norms(models: tuple[RaonModel, ...]) -> None:
    for layer_idx in range(models[0].config.num_hidden_layers):
        for key, seed in (("input_norm", 20261500), ("post_norm", 20261600)):
            shape = models[0]._layer_constants(layer_idx)[key].shape
            value = (
                1.0
                + 0.5
                * np.random.default_rng(seed + layer_idx).standard_normal(shape)
            ).astype(np.float32)
            for model in models:
                model._layer_constants(layer_idx)[key] = value.copy()


class _CountingPool:
    def __init__(self, pool) -> None:
        self.pool = pool
        self.count = 0
        self.lock = threading.Lock()

    def submit(self, fn, *args, **kwargs):
        with self.lock:
            self.count += 1
        return self.pool.submit(fn, *args, **kwargs)


def _last_stream(model: RaonModel) -> np.ndarray:
    layer = model.config.num_hidden_layers - 1
    return model.last_intermediates[f"layer_{layer:03d}.streams_out"]


def test_tiny_prefill_threads_four_matches_one(tiny_raon_model, monkeypatch):
    """성질 7: 100토큰 chunk 64에서 4·2분할을 쓰며 logits와 마지막 흐름이 1스레드와 같다."""
    _, config, weights, _ = tiny_raon_model
    threaded = RaonModel(config, weights, backend=NumpyBackend(weights))
    single = RaonModel(config, weights, backend=NumpyBackend(weights))
    threaded.prefill_mhc_threads = 4
    single.prefill_mhc_threads = 1
    _randomize_norms((threaded, single))
    counter = _CountingPool(raon_module._MLA_POOL)
    monkeypatch.setattr(raon_module, "_MLA_POOL", counter)
    tokens = [(index * 37 + 11) % config.vocab_size for index in range(100)]

    logits_threaded = threaded.prefill(tokens, logits_positions=[99], chunk=64, record_intermediates=True)[99]
    threaded_submits = counter.count
    logits_single = single.prefill(tokens, logits_positions=[99], chunk=64, record_intermediates=True)[99]

    _assert_same_bytes(logits_threaded, logits_single)
    _assert_same_bytes(_last_stream(threaded), _last_stream(single))
    assert threaded_submits > 0
    assert counter.count == threaded_submits


def test_tiny_prefill_chunk_sixteen_never_submits(tiny_raon_model, monkeypatch):
    """성질 7: 40토큰 chunk 16의 16·16·8 묶음은 4스레드 설정이어도 풀을 쓰지 않는다."""
    _, config, weights, _ = tiny_raon_model
    threaded = RaonModel(config, weights, backend=NumpyBackend(weights))
    single = RaonModel(config, weights, backend=NumpyBackend(weights))
    threaded.prefill_mhc_threads = 4
    single.prefill_mhc_threads = 1
    _randomize_norms((threaded, single))
    counter = _CountingPool(raon_module._MLA_POOL)
    monkeypatch.setattr(raon_module, "_MLA_POOL", counter)
    tokens = [(index * 29 + 7) % config.vocab_size for index in range(40)]

    logits_threaded = threaded.prefill(tokens, logits_positions=[39], chunk=16)[39]
    logits_single = single.prefill(tokens, logits_positions=[39], chunk=16)[39]
    _assert_same_bytes(logits_threaded, logits_single)
    assert counter.count == 0


@pytest.mark.parametrize("threads", [1, 8])
def test_token_path_does_not_call_batch_helpers(tiny_raon_model, monkeypatch, threads):
    """성질 7: prefill_mhc_batch=False이면 스레드 값과 무관하게 새 앞뒤 도우미를 호출하지 않는다."""
    _, config, weights, _ = tiny_raon_model
    model = RaonModel(config, weights, backend=NumpyBackend(weights))
    model.prefill_mhc_batch = False
    model.prefill_mhc_threads = threads

    def forbidden(*args, **kwargs):
        raise AssertionError("batch helper called from token path")

    monkeypatch.setattr(model, "_mhc_front_batch", forbidden)
    monkeypatch.setattr(model, "_mhc_back_batch", forbidden)
    tokens = [(index * 17 + 3) % config.vocab_size for index in range(8)]
    assert 7 in model.prefill(tokens, logits_positions=[7], chunk=8)


@pytest.mark.parametrize(("raw", "expected"), [(None, 4), ("1", 1), ("8", 8), (" 2 ", 2)])
def test_prefill_mhc_threads_valid_environment(monkeypatch, raw, expected):
    """성질 8: 기본값과 허용 환경 변수 값을 정수 스레드 수로 읽는다."""
    model = RaonModel(RaonConfig(), None, backend=object())
    if raw is None:
        monkeypatch.delenv("RAON_PREFILL_MHC_THREADS", raising=False)
    else:
        monkeypatch.setenv("RAON_PREFILL_MHC_THREADS", raw)
    assert model.prefill_mhc_threads == expected


@pytest.mark.parametrize("raw", ["0", "-1", "abc", "", "1.5"])
def test_prefill_mhc_threads_invalid_environment(monkeypatch, raw):
    """성질 8: 1 이상 정수가 아닌 환경 변수는 ValueError를 낸다."""
    model = RaonModel(RaonConfig(), None, backend=object())
    monkeypatch.setenv("RAON_PREFILL_MHC_THREADS", raw)
    with pytest.raises(ValueError, match="RAON_PREFILL_MHC_THREADS"):
        _ = model.prefill_mhc_threads


def test_prefill_mhc_threads_override_and_reset(monkeypatch):
    """성질 8: setter가 환경보다 앞서고 None으로 되돌리면 환경을 다시 읽는다."""
    model = RaonModel(RaonConfig(), None, backend=object())
    monkeypatch.setenv("RAON_PREFILL_MHC_THREADS", "8")
    model.prefill_mhc_threads = 3
    assert model.prefill_mhc_threads == 3
    model.prefill_mhc_threads = None
    assert model.prefill_mhc_threads == 8


def test_bad_thread_environment_fails_prefill_before_flow(tiny_raon_model, monkeypatch):
    """성질 8: 묶음 prefill은 잘못된 환경 변수를 흐름 생성 전에 거부한다."""
    _, config, weights, _ = tiny_raon_model
    model = RaonModel(config, weights, backend=NumpyBackend(weights))
    model.prefill_mhc_batch = True
    monkeypatch.setenv("RAON_PREFILL_MHC_THREADS", "bad")

    def forbidden(*args, **kwargs):
        raise AssertionError("flow work started")

    monkeypatch.setattr(model, "_mhc_front_batch", forbidden)
    with pytest.raises(ValueError, match="RAON_PREFILL_MHC_THREADS"):
        model.prefill([1], logits_positions=[0])
    assert model.last_intermediates == {}

