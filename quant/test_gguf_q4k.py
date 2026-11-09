# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors
"""Focused tests for llama.cpp Q4_K decoding and the q4nx bridge."""

from __future__ import annotations

import numpy as np

from quant.gguf_q4k import (
    dequantize_q4_k_blocks,
    hf_to_gguf,
    q4_k_proof,
    q4_k_blocks_to_q4nx,
    unpack_q4_k_codes,
    unpack_q4_k_scale_min,
)
from quant.q4nx import dequantize_simple, pack_decode_proj, parse_decode_proj


def _pack_scale_min(scale: np.ndarray, minimum: np.ndarray) -> np.ndarray:
    """Independent inverse of llama.cpp get_scale_min_k4 for a test vector."""
    out = np.zeros(12, dtype=np.uint8)
    out[:4] = (scale[:4] & 0x3F) | ((scale[4:] >> 4) << 6)
    out[4:8] = (minimum[:4] & 0x3F) | ((minimum[4:] >> 4) << 6)
    out[8:12] = (scale[4:] & 0x0F) | ((minimum[4:] & 0x0F) << 4)
    return out


def test_q4_k_unpack_known_answer() -> None:
    scale = np.array([0, 1, 31, 63, 4, 20, 40, 60], dtype=np.uint8)
    minimum = np.array([63, 32, 17, 0, 3, 19, 35, 51], dtype=np.uint8)
    packed = _pack_scale_min(scale, minimum)
    got_scale, got_minimum = unpack_q4_k_scale_min(packed)
    np.testing.assert_array_equal(got_scale, scale)
    np.testing.assert_array_equal(got_minimum, minimum)

    qs = np.arange(128, dtype=np.uint8)
    codes = unpack_q4_k_codes(qs)
    for pair in range(4):
        np.testing.assert_array_equal(codes[pair * 2], qs[pair * 32:(pair + 1) * 32] & 15)
        np.testing.assert_array_equal(codes[pair * 2 + 1], qs[pair * 32:(pair + 1) * 32] >> 4)


def test_q4_k_to_q4nx_preserves_codes_and_only_rounds_parameters() -> None:
    rng = np.random.default_rng(20261028)
    blocks = np.zeros((7, 144), dtype=np.uint8)
    blocks[:, :4] = rng.integers(0, 256, size=(7, 4), dtype=np.uint8)
    # Replace arbitrary half patterns with finite, normal values.
    blocks[:, 0:2] = np.array([0.001953125, 0.00390625, 0.0078125, 0.015625,
                               0.03125, 0.0625, 0.125], dtype="<f2").view(np.uint8).reshape(7, 2)
    blocks[:, 2:4] = np.array([0.0029296875, 0.005859375, 0.01171875, 0.0234375,
                               0.046875, 0.09375, 0.1875], dtype="<f2").view(np.uint8).reshape(7, 2)
    blocks[:, 4:] = rng.integers(0, 256, size=(7, 140), dtype=np.uint8)

    codes, scales, mins = q4_k_blocks_to_q4nx(blocks)
    reference = dequantize_q4_k_blocks(blocks)
    converted = dequantize_simple(codes, scales, mins)

    np.testing.assert_array_equal(codes.reshape(7, 8, 32), unpack_q4_k_codes(blocks[:, 16:]))
    # The exact observed difference must be covered by the two rounded BF16
    # parameters: |q*delta_scale + delta_min|.
    exact_scale = np.empty_like(scales)
    exact_min = np.empty_like(mins)
    unpacked_scale, unpacked_min = unpack_q4_k_scale_min(blocks[:, 4:16])
    d = blocks[:, :2].copy().reshape(-1).view("<f2").astype(np.float32)
    dmin = blocks[:, 2:4].copy().reshape(-1).view("<f2").astype(np.float32)
    exact_scale[:] = d[:, None] * unpacked_scale
    exact_min[:] = -dmin[:, None] * unpacked_min
    bound = (np.abs(exact_scale - scales)[..., None] * codes.reshape(7, 8, 32)
             + np.abs(exact_min - mins)[..., None] + 2e-6)
    assert np.all(np.abs(reference.reshape(7, 8, 32) - converted.reshape(7, 8, 32)) <= bound)


def test_q4_k_components_feed_decode_projection_packer() -> None:
    rng = np.random.default_rng(44)
    blocks = rng.integers(0, 256, size=(64, 144), dtype=np.uint8)
    blocks[:, :2] = np.array([0.01], dtype="<f2").view(np.uint8)
    blocks[:, 2:4] = np.array([0.02], dtype="<f2").view(np.uint8)
    codes, scales, mins = q4_k_blocks_to_q4nx(blocks)
    packed = pack_decode_proj(codes, scales, mins)
    got_codes, got_scales, got_mins = parse_decode_proj(packed, 0, 64, 256)
    np.testing.assert_array_equal(got_codes, codes)
    np.testing.assert_array_equal(got_scales, scales)
    np.testing.assert_array_equal(got_mins, mins)


def test_g_proj_mapping_uses_actual_tensor_set_for_late_rda_and_mla() -> None:
    names = {"blk.4.ssm_g.weight", "blk.7.attn_gate.weight"}
    assert hf_to_gguf(
        "model.layers.4.self_attn.g_proj.weight", tensor_names=names
    ) == ("blk.4.ssm_g.weight", None)
    assert hf_to_gguf(
        "model.layers.7.self_attn.g_proj.weight", tensor_names=names
    ) == ("blk.7.attn_gate.weight", None)


def test_g_proj_mapping_accepts_converted_config_layers() -> None:
    # Config rda_layers are converted to zero-based indices by the caller.
    rda = {0, 4}
    assert hf_to_gguf(
        "model.layers.4.self_attn.g_proj.weight", rda_layers=rda
    ) == ("blk.4.ssm_g.weight", None)
    assert hf_to_gguf(
        "model.layers.8.self_attn.g_proj.weight", rda_layers=rda
    ) == ("blk.8.attn_gate.weight", None)


def test_q4_k_proof_rejects_non_q4_k_reader() -> None:
    class FakeInfo:
        ggml_type = 1
        type_name = "F16"

    class FakeReader:
        tensors = {"weight": FakeInfo()}

    import pytest

    with pytest.raises(ValueError, match="not Q4_K"):
        q4_k_proof(FakeReader(), "weight")  # type: ignore[arg-type]
