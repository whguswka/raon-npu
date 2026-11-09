# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Byte-parity tests for chunk-wide prefill MLA attention and head-group threads."""

from __future__ import annotations

import threading
import numpy as np
import pytest

import host.backend
from host.backend import NumpyBackend, _MlaLayerCache, mla_context, mla_threads
from host.raon import RaonConfig, RaonModel
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.weights import bf16_to_f32, f32_to_bf16_rtne


def _oracle_attend(
    cache: _MlaLayerCache, q: np.ndarray, k_u16: np.ndarray, v_u16: np.ndarray, scale: float
) -> np.ndarray:
    """NumpyBackend.mla_attend of main 5bdbe8c (before mla_context) for one token, on a cache the test owns."""
    cache.append(k_u16, v_u16)
    t = cache.length
    q_f32 = bf16_to_f32(q)
    k_f32 = cache.k_f32[:, :t, :]
    v_f32 = cache.v_f32[:, :t, :]
    scores = np.einsum("hd,htd->ht", q_f32, k_f32) * scale
    max_s = np.max(scores, axis=-1, keepdims=True)
    exp_s = np.exp(scores - max_s)
    attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
    return f32_to_bf16_rtne(np.einsum("ht,htd->hd", attn, v_f32).reshape(-1))


def _assert_same_bytes(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes()


def _assert_cache_state(backend: NumpyBackend, oracle: _MlaLayerCache, layer: int = 0) -> None:
    cache = backend._mla_caches[layer]
    assert cache.length == oracle.length
    assert cache.capacity == oracle.capacity
    n = cache.length
    h, _, dk = oracle.k_u16.shape
    _, _, dv = oracle.v_u16.shape

    for attr in ("k_u16", "v_u16", "k_f32", "v_f32"):
        _assert_same_bytes(getattr(cache, attr)[:, :n], getattr(oracle, attr)[:, :n])

    k_view = backend.mla_key_cache[layer]
    v_view = backend.mla_value_cache[layer]
    assert k_view.shape == (h, n, dk)
    assert v_view.shape == (h, n, dv)
    assert np.shares_memory(k_view, cache.k_u16)
    assert np.shares_memory(v_view, cache.v_u16)
    _assert_same_bytes(k_view, oracle.k_u16[:, :n])
    _assert_same_bytes(v_view, oracle.v_u16[:, :n])

    got_k, got_v = backend.get_mla_cache(layer)
    assert got_k is not None and got_v is not None
    _assert_same_bytes(got_k, oracle.k_u16[:, :n][np.newaxis, ...])
    _assert_same_bytes(got_v, oracle.v_u16[:, :n][np.newaxis, ...])


def _bf16_data(
    rng: np.random.Generator, shape: tuple[int, ...], gain: np.ndarray | None = None
) -> np.ndarray:
    x = rng.standard_normal(shape, dtype=np.float32)
    mask = rng.random(shape) < 0.001
    x[mask] *= 30.0
    if gain is not None:
        x *= gain
    return f32_to_bf16_rtne(x)


def _set_threads(monkeypatch: pytest.MonkeyPatch, threads: str | None) -> None:
    if threads is None:
        monkeypatch.delenv("RAON_MLA_THREADS", raising=False)
    else:
        monkeypatch.setenv("RAON_MLA_THREADS", threads)


@pytest.mark.parametrize("threads", ["1", "4", "8", None])
def test_sequential_chunks_match_oracle_and_sharp_head_underflows(monkeypatch, threads):
    _set_threads(monkeypatch, threads)
    rng = np.random.default_rng(20261105)
    h, dk, dv = 40, 192, 128
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]
    chunk_sizes = (64, 64, 64, 64, 64, 7, 61, 1, 64, 64, 64)

    backend = NumpyBackend(None)
    oracle = _MlaLayerCache(h, dk, dv, 257)
    saw_exact_zero_attn = False

    for chunk_len in chunk_sizes:
        qs = _bf16_data(rng, (chunk_len, h, dk), head_gain)
        ks = _bf16_data(rng, (chunk_len, h, dk))
        vs = _bf16_data(rng, (chunk_len, h, dv))

        want = np.stack(
            [_oracle_attend(oracle, qs[i], ks[i], vs[i], scale) for i in range(chunk_len)],
            axis=0,
        )
        if not saw_exact_zero_attn and oracle.length >= 64:
            q_last = bf16_to_f32(qs[-1, -1])
            k_all = oracle.k_f32[-1, : oracle.length]
            scores = np.dot(k_all, q_last) * scale
            exp_s = np.exp(scores - np.max(scores))
            attn = exp_s / np.sum(exp_s)
            if np.any(attn == 0.0):
                saw_exact_zero_attn = True

        got = backend.mla_attend_chunk(0, qs, ks, vs, scale)
        _assert_same_bytes(got, want)
        _assert_cache_state(backend, oracle, layer=0)

    assert oracle.length == 581
    assert oracle.capacity > 513
    assert saw_exact_zero_attn, "sharpest head must produce exact 0.0 softmax weights via exp underflow"


RESTORED_CHUNK_CASES = (
    (3, 61),
    (2, 61),
    (1, 64),
    (66, 61),
    (65, 64),
    (300, 61),
    (600, 64),
    (256, 1),
)


@pytest.mark.parametrize("threads", ["1", "4", "8", None])
@pytest.mark.parametrize("p0,T", RESTORED_CHUNK_CASES)
def test_chunk_after_restored_cache(monkeypatch, p0, T, threads):
    _set_threads(monkeypatch, threads)
    rng = np.random.default_rng(20261110 + p0 * 100 + T)
    h, dk, dv = 40, 192, 128
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    kp = _bf16_data(rng, (h, p0, dk))
    vp = _bf16_data(rng, (h, p0, dv))
    qs = _bf16_data(rng, (T, h, dk), head_gain)
    ks = _bf16_data(rng, (T, h, dk))
    vs = _bf16_data(rng, (T, h, dv))

    oracle = _MlaLayerCache.from_existing(kp, vp)
    want = np.stack([_oracle_attend(oracle, qs[i], ks[i], vs[i], scale) for i in range(T)], axis=0)

    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp, vp)
    got = backend.mla_attend_chunk(0, qs, ks, vs, scale)

    _assert_same_bytes(got, want)
    _assert_cache_state(backend, oracle, layer=0)


@pytest.mark.parametrize("threads", ["1", "4", "8", None])
def test_decode_threads_600_tokens(monkeypatch, threads):
    _set_threads(monkeypatch, threads)
    rng = np.random.default_rng(20261120)
    h, dk, dv = 40, 192, 128
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    backend = NumpyBackend(None)
    oracle = _MlaLayerCache(h, dk, dv, 257)

    for _ in range(600):
        q = _bf16_data(rng, (h, dk), head_gain)
        k = _bf16_data(rng, (h, dk))
        v = _bf16_data(rng, (h, dv))
        want = _oracle_attend(oracle, q, k, v, scale)
        got = backend.mla_attend(0, q, k, v, scale)
        _assert_same_bytes(got, want)

    _assert_cache_state(backend, oracle, layer=0)


def test_decode_auto_threads_across_2048_threshold(monkeypatch):
    _set_threads(monkeypatch, None)
    rng = np.random.default_rng(20261125)
    h, dk, dv = 40, 192, 128
    p0, num_decode = 2040, 16
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    kp = _bf16_data(rng, (h, p0, dk))
    vp = _bf16_data(rng, (h, p0, dv))

    oracle = _MlaLayerCache.from_existing(kp, vp)
    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp, vp)

    for _ in range(num_decode):
        q = _bf16_data(rng, (h, dk), head_gain)
        k = _bf16_data(rng, (h, dk))
        v = _bf16_data(rng, (h, dv))
        want = _oracle_attend(oracle, q, k, v, scale)
        got = backend.mla_attend(0, q, k, v, scale)
        _assert_same_bytes(got, want)

    assert oracle.length == 2056
    _assert_cache_state(backend, oracle, layer=0)


@pytest.mark.parametrize("threads", ["1", "8"])
@pytest.mark.parametrize(
    "target,row,val",
    [
        ("v", 30, np.inf),
        ("v", 30, np.nan),
        ("v", 63, -np.inf),
        ("k", 30, np.nan),
        ("q", 30, np.nan),
        ("vprefix", 150, np.inf),
    ],
)
def test_non_finite_values_match_oracle_uint16(monkeypatch, target, row, val, threads):
    _set_threads(monkeypatch, threads)
    rng = np.random.default_rng(20261130)
    h, dk, dv = 40, 192, 128
    p0, T = 300, 64
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    kp = _bf16_data(rng, (h, p0, dk))
    vp = _bf16_data(rng, (h, p0, dv))
    qs = _bf16_data(rng, (T, h, dk), head_gain)
    ks = _bf16_data(rng, (T, h, dk))
    vs = _bf16_data(rng, (T, h, dv))

    bits = f32_to_bf16_rtne(np.full(3, val, dtype=np.float32))
    if target == "vprefix":
        vp[0, row, :3] = bits
    else:
        {"v": vs, "k": ks, "q": qs}[target][row, 0, :3] = bits

    with np.errstate(all="ignore"):
        oracle = _MlaLayerCache.from_existing(kp, vp)
        want = np.stack(
            [_oracle_attend(oracle, qs[i], ks[i], vs[i], scale) for i in range(T)], axis=0
        )

        backend = NumpyBackend(None)
        backend.set_mla_cache(0, kp, vp)
        got = backend.mla_attend_chunk(0, qs, ks, vs, scale)

    assert np.array_equal(got, want)
    _assert_cache_state(backend, oracle, layer=0)


@pytest.mark.parametrize("threads", ["4", None])
@pytest.mark.parametrize("p0,T", RESTORED_CHUNK_CASES)
def test_non_contiguous_strided_inputs(monkeypatch, p0, T, threads):
    _set_threads(monkeypatch, threads)
    rng = np.random.default_rng(20261110 + p0 * 100 + T)
    h, dk, dv = 40, 192, 128
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    kp = _bf16_data(rng, (h, p0, dk))
    vp = _bf16_data(rng, (h, p0, dv))
    qs_c = _bf16_data(rng, (T, h, dk), head_gain)
    ks_c = _bf16_data(rng, (T, h, dk))
    vs_c = _bf16_data(rng, (T, h, dv))

    oracle = _MlaLayerCache.from_existing(kp, vp)
    want = np.stack(
        [_oracle_attend(oracle, qs_c[i], ks_c[i], vs_c[i], scale) for i in range(T)], axis=0
    )

    wq = np.zeros((T, h, dk + 8), dtype=np.uint16)
    wk = np.zeros((T, h, dk + 8), dtype=np.uint16)
    wv = np.zeros((T, h, dv + 8), dtype=np.uint16)
    wq[:, :, 4:-4] = qs_c
    wk[:, :, 4:-4] = ks_c
    wv[:, :, 4:-4] = vs_c
    qs = wq[:, :, 4:-4]
    ks = wk[:, :, 4:-4]
    vs = wv[:, :, 4:-4]
    assert not qs.flags.c_contiguous
    assert not ks.flags.c_contiguous
    assert not vs.flags.c_contiguous

    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp, vp)
    got = backend.mla_attend_chunk(0, qs, ks, vs, scale)

    _assert_same_bytes(got, want)
    _assert_cache_state(backend, oracle, layer=0)


def test_more_threads_than_heads(monkeypatch):
    _set_threads(monkeypatch, "8")
    rng = np.random.default_rng(20261140)
    h, dk, dv = 4, 24, 16
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    backend_chunk = NumpyBackend(None)
    oracle_chunk = _MlaLayerCache(h, dk, dv, 257)
    for chunk_len in (16, 16, 8, 1):
        qs = _bf16_data(rng, (chunk_len, h, dk), head_gain)
        ks = _bf16_data(rng, (chunk_len, h, dk))
        vs = _bf16_data(rng, (chunk_len, h, dv))
        want = np.stack(
            [_oracle_attend(oracle_chunk, qs[i], ks[i], vs[i], scale) for i in range(chunk_len)],
            axis=0,
        )
        got = backend_chunk.mla_attend_chunk(0, qs, ks, vs, scale)
        _assert_same_bytes(got, want)
        _assert_cache_state(backend_chunk, oracle_chunk, layer=0)

    backend_decode = NumpyBackend(None)
    oracle_decode = _MlaLayerCache(h, dk, dv, 257)
    for _ in range(40):
        q = _bf16_data(rng, (h, dk), head_gain)
        k = _bf16_data(rng, (h, dk))
        v = _bf16_data(rng, (h, dv))
        want = _oracle_attend(oracle_decode, q, k, v, scale)
        got = backend_decode.mla_attend(0, q, k, v, scale)
        _assert_same_bytes(got, want)
    _assert_cache_state(backend_decode, oracle_decode, layer=0)


def test_negative_zero_tripwire(monkeypatch):
    # Query 0 has softmax weights 1.0 and exp(-46.2) ~= 8.7e-21, and the product
    # 8.7e-21 * -1.2e-38 (-2^-126) underflows to -0.0. When numpy rounds the
    # multiplication and addition separately, +0.0 + (-0.0) = +0.0; a numpy build
    # that fused multiply-add in einsum would produce -0.0 on the per-token path,
    # whereas the chunk path adds trailing 0.0 * 1.0 = +0.0 and turns it into +0.0,
    # breaking this test as an early warning on numpy upgrades.
    _set_threads(monkeypatch, "4")
    rng = np.random.default_rng(20261150)
    h, dk, dv = 2, 192, 128
    scale = dk ** -0.5

    kp = np.zeros((h, 1, dk), dtype=np.uint16)
    vp = _bf16_data(rng, (h, 1, dv))
    vp[0, 0, 0] = 0x8000  # -0.0

    qs = np.zeros((2, h, dk), dtype=np.uint16)
    qs[:, :, 0] = 0x3F80  # 1.0
    ks = np.zeros((2, h, dk), dtype=np.uint16)
    ks[0, :, 0] = f32_to_bf16_rtne(np.float32(-640.0))
    vs = _bf16_data(rng, (2, h, dv))
    vs[0, 0, 0] = 0x8080  # -2^-126
    vs[1, 0, 0] = 0x3F80  # 1.0

    oracle = _MlaLayerCache.from_existing(kp, vp)
    want = np.stack([_oracle_attend(oracle, qs[i], ks[i], vs[i], scale) for i in range(2)], axis=0)

    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp, vp)
    got = backend.mla_attend_chunk(0, qs, ks, vs, scale)

    _assert_same_bytes(got, want)
    _assert_cache_state(backend, oracle, layer=0)

    cache = backend._mla_caches[0]
    ctx_f32 = mla_context(bf16_to_f32(qs), cache.k_f32[:, :3, :], cache.v_f32[:, :3, :], scale)
    _assert_same_bytes(f32_to_bf16_rtne(ctx_f32.reshape(2, -1)), want)


def test_mla_threads_schedule_and_env(monkeypatch):
    _set_threads(monkeypatch, None)
    auto_expected = {
        (255, 64): 1,
        (256, 64): 4,
        (511, 64): 4,
        (512, 64): 8,
        (6736, 64): 8,
        (100, 1): 1,
        (256, 1): 1,
        (2047, 1): 1,
        (2048, 1): 4,
        (6736, 1): 4,
    }
    for (n, nq), expected in auto_expected.items():
        assert mla_threads(n, nq) == expected

    explicit_expected = {
        "1": 1,
        "3": 3,
        "0": 1,
        "16": 16,
        " 4 ": 4,
    }
    for env_val, expected in explicit_expected.items():
        _set_threads(monkeypatch, env_val)
        for n, nq in auto_expected:
            assert mla_threads(n, nq) == expected

    for fallback_val in ("", "auto", "AUTO", "x", "-1"):
        _set_threads(monkeypatch, fallback_val)
        for (n, nq), expected in auto_expected.items():
            assert mla_threads(n, nq) == expected


def test_mla_context_dispatch_and_head_groups(monkeypatch):
    rng = np.random.default_rng(20261155)
    h, dk, dv = 40, 192, 128
    T = 64
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    kp300 = _bf16_data(rng, (h, 300, dk))
    vp300 = _bf16_data(rng, (h, 300, dv))
    kp600 = _bf16_data(rng, (h, 600, dk))
    vp600 = _bf16_data(rng, (h, 600, dv))
    qs = _bf16_data(rng, (T, h, dk), head_gain)
    ks = _bf16_data(rng, (T, h, dk))
    vs = _bf16_data(rng, (T, h, dv))

    calls: dict[str, list[str]] = {"chunk_rows": [], "queries": []}
    lock = threading.Lock()
    orig_chunk_rows = host.backend._mla_attend_chunk_rows
    orig_queries = host.backend._mla_attend_queries

    def spy_chunk_rows(*args, **kwargs):
        with lock:
            calls["chunk_rows"].append(threading.current_thread().name)
        return orig_chunk_rows(*args, **kwargs)

    def spy_queries(*args, **kwargs):
        with lock:
            calls["queries"].append(threading.current_thread().name)
        return orig_queries(*args, **kwargs)

    monkeypatch.setattr(host.backend, "_mla_attend_chunk_rows", spy_chunk_rows)
    monkeypatch.setattr(host.backend, "_mla_attend_queries", spy_queries)

    def reset_calls() -> None:
        calls["chunk_rows"].clear()
        calls["queries"].clear()

    caller_thread = threading.current_thread().name
    assert not caller_thread.startswith("raon-mla")

    # 1. RAON_MLA_THREADS="4", restored 300 + chunk 64 (finite V) -> 4 chunk_rows on raon-mla threads
    _set_threads(monkeypatch, "4")
    reset_calls()
    oracle = _MlaLayerCache.from_existing(kp300, vp300)
    want = np.stack([_oracle_attend(oracle, qs[i], ks[i], vs[i], scale) for i in range(T)], axis=0)
    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp300, vp300)
    got = backend.mla_attend_chunk(0, qs, ks, vs, scale)
    _assert_same_bytes(got, want)
    assert len(calls["chunk_rows"]) == 4
    assert len(calls["queries"]) == 0
    assert all(name.startswith("raon-mla") for name in calls["chunk_rows"])

    # 2. Same setup with +Inf in chunk V -> 0 chunk_rows, 4 queries
    vs_inf = vs.copy()
    vs_inf[30, 0, 0] = f32_to_bf16_rtne(np.float32(np.inf))
    reset_calls()
    with np.errstate(all="ignore"):
        backend = NumpyBackend(None)
        backend.set_mla_cache(0, kp300, vp300)
        backend.mla_attend_chunk(0, qs, ks, vs_inf, scale)
    assert len(calls["chunk_rows"]) == 0
    assert len(calls["queries"]) == 4
    assert all(name.startswith("raon-mla") for name in calls["queries"])

    # 3. RAON_MLA_THREADS="1", same finite chunk -> 0 chunk_rows, 1 queries on caller thread
    _set_threads(monkeypatch, "1")
    reset_calls()
    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp300, vp300)
    backend.mla_attend_chunk(0, qs, ks, vs, scale)
    assert len(calls["chunk_rows"]) == 0
    assert calls["queries"] == [caller_thread]

    # 4. Auto threads, restored 600 + chunk 64 (n=664 -> 8 threads) -> 8 chunk_rows
    _set_threads(monkeypatch, None)
    reset_calls()
    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp600, vp600)
    backend.mla_attend_chunk(0, qs, ks, vs, scale)
    assert len(calls["chunk_rows"]) == 8
    assert len(calls["queries"]) == 0
    assert all(name.startswith("raon-mla") for name in calls["chunk_rows"])

    # 5a. Auto threads, restored 300 + decode 1 token (n=301 -> 1 thread) -> 1 queries on caller thread
    reset_calls()
    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp300, vp300)
    backend.mla_attend(0, qs[0], ks[0], vs[0], scale)
    assert calls["queries"] == [caller_thread]
    assert len(calls["chunk_rows"]) == 0

    # 5b. Auto threads, restored 2100 + decode 1 token (n=2101 -> 4 threads) -> 4 queries on raon-mla
    kp2100 = _bf16_data(rng, (h, 2100, dk))
    vp2100 = _bf16_data(rng, (h, 2100, dv))
    oracle2100 = _MlaLayerCache.from_existing(kp2100, vp2100)
    want2100 = _oracle_attend(oracle2100, qs[0], ks[0], vs[0], scale)
    reset_calls()
    backend = NumpyBackend(None)
    backend.set_mla_cache(0, kp2100, vp2100)
    got2100 = backend.mla_attend(0, qs[0], ks[0], vs[0], scale)
    _assert_same_bytes(got2100, want2100)
    assert len(calls["queries"]) == 4
    assert len(calls["chunk_rows"]) == 0
    assert all(name.startswith("raon-mla") for name in calls["queries"])

    # 6. Auto threads, empty cache + chunk 64 (n=64 -> 1 thread) -> 1 queries on caller thread
    reset_calls()
    backend = NumpyBackend(None)
    backend.mla_attend_chunk(0, qs, ks, vs, scale)
    assert calls["queries"] == [caller_thread]
    assert len(calls["chunk_rows"]) == 0


def test_npu_backend_mla_attend_chunk_and_timings(monkeypatch):
    from npu.backend import NpuBackend

    _set_threads(monkeypatch, "4")
    rng = np.random.default_rng(20261160)
    h, dk, dv = 40, 192, 128
    scale = dk ** -0.5
    head_gain = np.geomspace(0.5, 24.0, h).astype(np.float32)[:, np.newaxis]

    npu_be = NpuBackend(mock=True)
    ref_be = NumpyBackend(None)

    for call_idx, chunk_len in enumerate((64, 7), start=1):
        qs = _bf16_data(rng, (chunk_len, h, dk), head_gain)
        ks = _bf16_data(rng, (chunk_len, h, dk))
        vs = _bf16_data(rng, (chunk_len, h, dv))

        got_npu = npu_be.mla_attend_chunk(3, qs, ks, vs, scale)
        got_ref = ref_be.mla_attend_chunk(3, qs, ks, vs, scale)
        _assert_same_bytes(got_npu, got_ref)

        _assert_same_bytes(npu_be.mla_key_cache[3], ref_be.mla_key_cache[3])
        _assert_same_bytes(npu_be.mla_value_cache[3], ref_be.mla_value_cache[3])
        assert np.shares_memory(npu_be.mla_key_cache[3], npu_be._mla_caches[3].k_u16)
        assert np.shares_memory(npu_be.mla_value_cache[3], npu_be._mla_caches[3].v_u16)

        k_npu, v_npu = npu_be.get_mla_cache(3)
        k_ref, v_ref = ref_be.get_mla_cache(3)
        assert k_npu is not None and v_npu is not None
        assert k_ref is not None and v_ref is not None
        _assert_same_bytes(k_npu, k_ref)
        _assert_same_bytes(v_npu, v_ref)

        assert npu_be.call_counts["mla_attend"] == call_idx
        assert npu_be.call_counts["mla_host"] == call_idx
        assert npu_be.timings["mla_attend"] > 0.0


def _install_mla_counters(monkeypatch: pytest.MonkeyPatch, backend: NumpyBackend) -> dict[str, int]:
    counts = {"chunk": 0, "token": 0}
    orig_chunk = backend.mla_attend_chunk
    orig_token = backend.mla_attend

    def counted_chunk(*args, **kwargs):
        counts["chunk"] += 1
        return orig_chunk(*args, **kwargs)

    def counted_token(*args, **kwargs):
        counts["token"] += 1
        return orig_token(*args, **kwargs)

    monkeypatch.setattr(backend, "mla_attend_chunk", counted_chunk)
    monkeypatch.setattr(backend, "mla_attend", counted_token)
    return counts


@pytest.mark.parametrize("threads", [None, "4"])
@pytest.mark.parametrize("num_tokens,chunk", [(40, 16), (33, 16)])
def test_prefill_mla_batch_on_off_byte_identical(
    tiny_raon_model, monkeypatch, num_tokens, chunk, threads
):
    _set_threads(monkeypatch, threads)
    _, host_cfg, weights, _ = tiny_raon_model
    be_on = NumpyBackend(weights)
    be_off = NumpyBackend(weights)
    model_on = RaonModel(host_cfg, weights, backend=be_on)
    model_off = RaonModel(host_cfg, weights, backend=be_off)
    model_on.prefill_mla_batch = True
    model_off.prefill_mla_batch = False

    counts_on = _install_mla_counters(monkeypatch, be_on)
    counts_off = _install_mla_counters(monkeypatch, be_off)

    ids = [(i * 37 + 11) % host_cfg.vocab_size for i in range(num_tokens)]
    positions = range(len(ids))

    logits_on = model_on.prefill(ids, logits_positions=positions, chunk=chunk)
    logits_off = model_off.prefill(ids, logits_positions=positions, chunk=chunk)

    num_chunks = (num_tokens + chunk - 1) // chunk
    num_mla_layers = len(host_cfg.mla_layers)
    assert counts_on["chunk"] == num_chunks * num_mla_layers
    assert counts_on["token"] == 0
    assert counts_off["chunk"] == 0
    assert counts_off["token"] == num_tokens * num_mla_layers

    for pos in positions:
        _assert_same_bytes(logits_on[pos], logits_off[pos])

    for layer_idx in host_cfg.mla_layers:
        k_on, v_on = be_on.get_mla_cache(layer_idx)
        k_off, v_off = be_off.get_mla_cache(layer_idx)
        assert k_on is not None and v_on is not None
        assert k_off is not None and v_off is not None
        _assert_same_bytes(k_on, k_off)
        _assert_same_bytes(v_on, v_off)

    for decode_tok in (19, 73, 142):
        step_on = model_on.step(decode_tok % host_cfg.vocab_size)
        step_off = model_off.step(decode_tok % host_cfg.vocab_size)
        _assert_same_bytes(step_on, step_off)


def test_prefill_mla_batch_switch(monkeypatch):
    model = RaonModel(RaonConfig(), None, backend=object())

    monkeypatch.delenv("RAON_PREFILL_MLA_BATCH", raising=False)
    assert model.prefill_mla_batch is True
    for disabled in ("0", "false", "False", "no"):
        monkeypatch.setenv("RAON_PREFILL_MLA_BATCH", disabled)
        assert model.prefill_mla_batch is False
    monkeypatch.setenv("RAON_PREFILL_MLA_BATCH", "1")
    assert model.prefill_mla_batch is True

    model.prefill_mla_batch = False
    monkeypatch.setenv("RAON_PREFILL_MLA_BATCH", "1")
    assert model.prefill_mla_batch is False
    model.prefill_mla_batch = True
    monkeypatch.setenv("RAON_PREFILL_MLA_BATCH", "0")
    assert model.prefill_mla_batch is True
    model.prefill_mla_batch = None
    assert model.prefill_mla_batch is False
