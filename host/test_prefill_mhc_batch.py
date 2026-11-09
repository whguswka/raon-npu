# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Byte-parity tests for chunk-wide prefill mHC operations."""

from __future__ import annotations

import numpy as np
import pytest

from host.backend import NumpyBackend
from host.raon import RaonConfig, RaonModel, rms_norm
from host.test_host_small import tiny_raon_model  # noqa: F401
from host.weights import bf16_to_f32, f32_to_bf16_rtne


def _assert_same_bytes(actual: np.ndarray, expected: np.ndarray) -> None:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert np.ascontiguousarray(actual).tobytes() == np.ascontiguousarray(expected).tobytes()


def _make_inputs(
    rng: np.random.Generator, tokens: int, streams: int, hidden_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    stream_f32 = rng.standard_normal((tokens, streams, hidden_size)).astype(np.float32)
    flat = stream_f32.reshape(-1)
    outliers = max(1, flat.size // 1000)
    flat[rng.choice(flat.size, size=outliers, replace=False)] *= 30.0
    stream = f32_to_bf16_rtne(stream_f32)
    branch_out = f32_to_bf16_rtne(
        (rng.standard_normal((tokens, hidden_size)) * 0.3).astype(np.float32)
    )
    return stream, bf16_to_f32(stream), branch_out


def _per_token_results(
    model: RaonModel,
    site_prefix: str,
    norm_weight: np.ndarray,
    stream: np.ndarray,
    stream_f32: np.ndarray,
    branch_out: np.ndarray,
) -> tuple[np.ndarray, ...]:
    h_pre = []
    h_post = []
    h_res = []
    branch_in = []
    mixed = []
    for token in range(stream.shape[0]):
        hp, hpost, hres = model._route_mhc(
            stream[token], site_prefix, stream_f32[token]
        )
        h_pre.append(hp)
        h_post.append(hpost)
        h_res.append(hres)
        branch_in.append(
            rms_norm(
                model._mhc_aggregate(stream[token], hp, stream_f32[token]),
                norm_weight,
                eps=model.config.rms_norm_eps,
            )
        )
        mixed.append(
            model._mhc_mix(
                stream[token], branch_out[token], hpost, hres, stream_f32[token]
            )
        )
    return tuple(np.stack(values, axis=0) for values in (h_pre, h_post, h_res, branch_in, mixed))


def _batched_results(
    model: RaonModel,
    site_prefix: str,
    norm_weight: np.ndarray,
    stream_f32: np.ndarray,
    branch_out: np.ndarray,
) -> tuple[np.ndarray, ...]:
    h_pre, h_post, h_res = model._route_mhc_batch(stream_f32, site_prefix)
    branch_in = rms_norm(
        model._mhc_aggregate_batch(stream_f32, h_pre),
        norm_weight,
        eps=model.config.rms_norm_eps,
    )
    mixed = model._mhc_mix_batch(stream_f32, branch_out, h_post, h_res)
    return h_pre, h_post, h_res, branch_in, mixed


def _tiny_sites(model: RaonModel) -> list[tuple[str, np.ndarray]]:
    sites = []
    for layer_idx in range(model.config.num_hidden_layers):
        constants = model._layer_constants(layer_idx)
        layer_prefix = f"model.layers.{layer_idx}"
        sites.append((f"{layer_prefix}.self_attn_raon_mhc", constants["input_norm"]))
        sites.append((f"{layer_prefix}.mlp_raon_mhc", constants["post_norm"]))
    return sites


@pytest.mark.parametrize("tokens", [1, 7, 61, 64])
def test_mhc_batch_functions_match_every_tiny_site(tiny_raon_model, tokens):
    _, host_cfg, weights, _ = tiny_raon_model
    model = RaonModel(host_cfg, weights, backend=NumpyBackend(weights))
    rng = np.random.default_rng(20261105 + tokens)

    for site_prefix, norm_weight in _tiny_sites(model):
        stream, stream_f32, branch_out = _make_inputs(
            rng, tokens, host_cfg.raon_mhc_streams, host_cfg.hidden_size
        )
        expected = _per_token_results(
            model, site_prefix, norm_weight, stream, stream_f32, branch_out
        )
        actual = _batched_results(model, site_prefix, norm_weight, stream_f32, branch_out)
        for got, want in zip(actual, expected):
            _assert_same_bytes(got, want)


def test_mhc_batch_route_special_values_matches_bit_patterns(tiny_raon_model):
    _, host_cfg, weights, _ = tiny_raon_model
    model = RaonModel(host_cfg, weights, backend=NumpyBackend(weights))
    rng = np.random.default_rng(20261106)
    stream, stream_f32, _ = _make_inputs(
        rng, 7, host_cfg.raon_mhc_streams, host_cfg.hidden_size
    )
    stream_f32 = stream_f32.copy()
    stream_f32[0, 0, 0] = np.nan
    stream_f32[1, 1, 1] = np.inf
    stream_f32[2, 2, 2] = -np.inf
    stream = f32_to_bf16_rtne(stream_f32)
    stream_f32 = bf16_to_f32(stream)

    with np.errstate(all="ignore"):
        for site_prefix, _ in _tiny_sites(model):
            expected = tuple(
                np.stack(values, axis=0)
                for values in zip(
                    *(model._route_mhc(stream[t], site_prefix, stream_f32[t]) for t in range(7))
                )
            )
            actual = model._route_mhc_batch(stream_f32, site_prefix)
            for got, want in zip(actual, expected):
                np.testing.assert_array_equal(
                    np.ascontiguousarray(got).view(np.uint32),
                    np.ascontiguousarray(want).view(np.uint32),
                )


def _install_call_counters(monkeypatch, model: RaonModel) -> dict[str, int]:
    counts = {"route_batch": 0, "mix_batch": 0, "route_token": 0}
    for attribute, key in (
        ("_route_mhc_batch", "route_batch"),
        ("_mhc_mix_batch", "mix_batch"),
        ("_route_mhc", "route_token"),
    ):
        original = getattr(model, attribute)

        def counted(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(model, attribute, counted)
    return counts


def test_prefill_mhc_batch_on_off_is_byte_identical_and_uses_selected_path(
    tiny_raon_model, monkeypatch
):
    _, host_cfg, weights, _ = tiny_raon_model
    model_on = RaonModel(host_cfg, weights, backend=NumpyBackend(weights))
    model_off = RaonModel(host_cfg, weights, backend=NumpyBackend(weights))
    model_on.prefill_mhc_batch = True
    model_off.prefill_mhc_batch = False
    counts_on = _install_call_counters(monkeypatch, model_on)
    counts_off = _install_call_counters(monkeypatch, model_off)

    for layer_idx in range(host_cfg.num_hidden_layers):
        constants_on = model_on._layer_constants(layer_idx)
        constants_off = model_off._layer_constants(layer_idx)
        input_norm = (
            1.0
            + 0.5
            * np.random.default_rng(20261110 + 2 * layer_idx).standard_normal(
                constants_on["input_norm"].shape
            )
        ).astype(np.float32)
        post_norm = (
            1.0
            + 0.5
            * np.random.default_rng(20261111 + 2 * layer_idx).standard_normal(
                constants_on["post_norm"].shape
            )
        ).astype(np.float32)
        constants_on["input_norm"] = input_norm
        constants_on["post_norm"] = post_norm
        constants_off["input_norm"] = input_norm.copy()
        constants_off["post_norm"] = post_norm.copy()

    tokens = [(index * 37 + 11) % host_cfg.vocab_size for index in range(40)]

    logits_on = model_on.prefill(tokens, logits_positions=[39], chunk=16)[39]
    logits_off = model_off.prefill(tokens, logits_positions=[39], chunk=16)[39]

    _assert_same_bytes(logits_on, logits_off)
    assert counts_on["route_batch"] > 0
    assert counts_on["mix_batch"] > 0
    assert counts_on["route_token"] == 0
    assert counts_off["route_batch"] == 0
    assert counts_off["mix_batch"] == 0
    assert counts_off["route_token"] > 0


def test_prefill_mhc_batch_switch(monkeypatch):
    model = RaonModel(RaonConfig(), None, backend=object())

    monkeypatch.delenv("RAON_PREFILL_MHC_BATCH", raising=False)
    assert model.prefill_mhc_batch is True
    for disabled in ("0", "false", "False", "no"):
        monkeypatch.setenv("RAON_PREFILL_MHC_BATCH", disabled)
        assert model.prefill_mhc_batch is False
    monkeypatch.setenv("RAON_PREFILL_MHC_BATCH", "1")
    assert model.prefill_mhc_batch is True

    model.prefill_mhc_batch = False
    monkeypatch.setenv("RAON_PREFILL_MHC_BATCH", "1")
    assert model.prefill_mhc_batch is False
    model.prefill_mhc_batch = True
    monkeypatch.setenv("RAON_PREFILL_MHC_BATCH", "0")
    assert model.prefill_mhc_batch is True
    model.prefill_mhc_batch = None
    assert model.prefill_mhc_batch is False


@pytest.mark.parametrize("tokens", [1, 61, 64])
@pytest.mark.parametrize("seed", [20261107, 20261108, 20261109])
def test_operating_size_synthetic_constants_match_per_token(tokens, seed):
    config = RaonConfig(hidden_size=2560, raon_mhc_streams=4)
    model = RaonModel(config, None, backend=object())
    rng = np.random.default_rng(seed)
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
    norm_weight = rng.standard_normal(c).astype(np.float32)
    stream, stream_f32, branch_out = _make_inputs(rng, tokens, n, c)

    expected = _per_token_results(
        model, "synthetic", norm_weight, stream, stream_f32, branch_out
    )
    actual = _batched_results(model, "synthetic", norm_weight, stream_f32, branch_out)
    for got, want in zip(actual, expected):
        _assert_same_bytes(got, want)
