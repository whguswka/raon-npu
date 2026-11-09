# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 raon-npu authors

"""Unit and integration tests for q4nx quantization, padding, and packing.

Execution:
    python3 -m pytest quant/test_q4nx.py -v
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure repo and iron paths are importable
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
iron_path = os.environ.get("RAON_IRON_PATH", "")
sys.path.insert(0, iron_path)

from quant.q4nx import (
    BLOCK_BYTES,
    GROUP,
    K_TILE,
    M_TILE,
    bf16_ceil,
    bf16_floor,
    bf16_nextafter,
    dequantize_simple,
    mask_padded_logits,
    pad,
    pack_decode_proj,
    pack_lm_head,
    parse_decode_proj,
    parse_lm_head,
    quantize,
    round_to_bf16_rne,
    to_bf16_u16,
    unpad,
)

# Reference dequantize from IRON lm_head
from iron.operators.flm.lm_head.reference import dequantize as iron_lm_head_dequantize


# ===========================================================================
# 0. BF16 Directional Rounding & Grid Tests
# ===========================================================================


def test_bf16_direction_rounding_properties():
    """Verify bf16_floor and bf16_ceil adhere to IEEE-754 directed rounding."""
    # Test values across zero, signs, boundaries, subnormals, and large magnitudes
    test_vals = np.array(
        [
            0.0,
            -0.0,
            1.0,
            -1.0,
            1001.0,
            1001.1,
            -1001.0,
            -1001.1,
            1.0 + 2**(-20),
            1.0 - 2**(-20),
            -1.0 + 2**(-20),
            -1.0 - 2**(-20),
            3.3895314e38,  # max finite bf16
            -3.3895314e38,
            1e-40,  # subnormal
            -1e-40,
            9.18355e-41,
            -9.18355e-41,
        ],
        dtype=np.float32,
    )

    floors = bf16_floor(test_vals)
    ceils = bf16_ceil(test_vals)

    # Floor must be <= original, ceil must be >= original
    assert np.all(floors <= test_vals)
    assert np.all(ceils >= test_vals)

    # Verify results are exact bfloat16 representable numbers
    assert np.all(round_to_bf16_rne(floors) == floors)
    assert np.all(round_to_bf16_rne(ceils) == ceils)

    # When strictly between two bf16 grid points, stepping floor with nextafter yields ceil
    inexact = floors != ceils
    assert np.all(bf16_nextafter(floors[inexact]) == ceils[inexact])

    # Test float64 inputs
    vals_f64 = np.array([1001.1, -1001.1, 1e-10, -1e-10, 1.23456789012345], dtype=np.float64)
    fl_f64 = bf16_floor(vals_f64)
    ce_f64 = bf16_ceil(vals_f64)
    assert np.all(fl_f64.astype(np.float64) <= vals_f64)
    assert np.all(ce_f64.astype(np.float64) >= vals_f64)


# ===========================================================================
# 1. Quantization & Numerical Accuracy Tests
# ===========================================================================


def test_quantize_random_bounds():
    """Verify quantization error is <= 0.5 * group scale (+ float rounding tolerance)."""
    rng = np.random.default_rng(2026)
    rows, cols = 64, 512
    w = rng.standard_normal((rows, cols)).astype(np.float32) * 2.5

    codes, scales, mins = quantize(w)

    assert codes.shape == (rows, cols)
    assert scales.shape == (rows, cols // GROUP)
    assert mins.shape == (rows, cols // GROUP)
    assert codes.dtype == np.uint8
    assert np.all((codes >= 0) & (codes <= 15))

    rec = dequantize_simple(codes, scales, mins)
    err = np.abs(w - rec)

    scales_expanded = np.repeat(scales, GROUP, axis=-1)
    # The quantization step is scale. Ideal uniform quant error is <= 0.5 * scale.
    max_allowed_err = 0.5 * scales_expanded + 1e-6
    assert np.all(err <= max_allowed_err), f"Max error exceeded: {np.max(err - max_allowed_err)}"


def test_quantize_non_finite_rejected():
    """Verify quantize rejects inputs containing NaN or Inf with ValueError."""
    for bad_val in [np.nan, np.inf, -np.inf]:
        w = np.ones((32, 64), dtype=np.float32)
        w[0, 10] = bad_val
        with pytest.raises(ValueError, match="non-finite"):
            quantize(w)


def test_quantize_large_magnitude_opposite_signs():
    """Verify handling of wide ranges (e.g. [-3e38, 3e38]) without float32 overflow."""
    rows, cols = 32, 64
    w = np.zeros((rows, cols), dtype=np.float32)
    # Group 0: spans [-3e38, 3e38]
    w[0, 0] = -3e38
    w[0, 1] = 3e38
    w[0, 2:32] = np.linspace(-1e38, 1e38, 30, dtype=np.float32)
    # Group 1: random within reasonable range
    w[:, 32:] = np.linspace(-100.0, 100.0, 32, dtype=np.float32)

    codes, scales, mins = quantize(w)

    assert np.all((codes >= 0) & (codes <= 15))
    assert np.all(np.isfinite(scales))
    assert np.all(np.isfinite(mins))

    rec = dequantize_simple(codes, scales, mins)
    err = np.abs(w.astype(np.float64) - rec.astype(np.float64))
    scales_exp = np.repeat(scales.astype(np.float64), GROUP, axis=-1)

    max_allowed_err = 0.5 * scales_exp * (1.0 + 1e-5)
    assert np.all(err <= max_allowed_err), f"Max error exceeded: {np.max(err - max_allowed_err)}"


def test_quantize_narrow_range_far_from_zero():
    """Verify narrow range offset from zero avoids code saturation (error <= 0.5 * scale)."""
    rows, cols = 32, 64
    w = np.zeros((rows, cols), dtype=np.float32)
    # Group 0: [1001.0, 1001.1] (positive offset)
    w[:, :32] = np.linspace(1001.0, 1001.1, 32, dtype=np.float32)
    # Group 1: [-1001.1, -1001.0] (negative offset)
    w[:, 32:] = np.linspace(-1001.1, -1001.0, 32, dtype=np.float32)

    codes, scales, mins = quantize(w)

    assert np.all((codes >= 0) & (codes <= 15))
    rec = dequantize_simple(codes, scales, mins)
    err = np.abs(w.astype(np.float64) - rec.astype(np.float64))
    scales_exp = np.repeat(scales.astype(np.float64), GROUP, axis=-1)

    max_allowed_err = 0.5 * scales_exp + 1e-5
    assert np.all(err <= max_allowed_err), f"Max error exceeded: {np.max(err - max_allowed_err)}"


def test_quantize_inexact_constant_group():
    """Verify constant group not exactly representable in bf16 (e.g. 0.1)."""
    rows, cols = 32, 32
    w = np.full((rows, cols), 0.1, dtype=np.float32)

    codes, scales, mins = quantize(w)

    # 0.1 is not an exact bf16, so scale_q > 0 and codes in [0, 15]
    assert np.all(scales > 0.0)
    assert np.all((codes >= 0) & (codes <= 15))

    rec = dequantize_simple(codes, scales, mins)
    err = np.abs(w.astype(np.float64) - rec.astype(np.float64))
    scales_exp = np.repeat(scales.astype(np.float64), GROUP, axis=-1)

    max_allowed_err = 0.5 * scales_exp + 1e-6
    assert np.all(err <= max_allowed_err)


def test_quantize_exact_constant_groups():
    """Verify handling of exact bf16 constant groups (max == min)."""
    rows, cols = 32, 128
    # Group 0: all 0.0, Group 1: all 5.0, Group 2: all -12.5, Group 3: mixed
    w = np.zeros((rows, cols), dtype=np.float32)
    w[:, 32:64] = 5.0
    w[:, 64:96] = -12.5
    w[:, 96:128] = np.linspace(1.0, 10.0, 32, dtype=np.float32)

    codes, scales, mins = quantize(w)

    # Exact constant groups must have scale 0.0 and code 0
    assert np.all(scales[:, 0] == 0.0)
    assert np.all(scales[:, 1] == 0.0)
    assert np.all(scales[:, 2] == 0.0)
    assert np.all(scales[:, 3] > 0.0)

    assert np.all(codes[:, :96] == 0)

    # Reconstruction of constant groups must be exact
    rec = dequantize_simple(codes, scales, mins)
    np.testing.assert_allclose(rec[:, :32], 0.0, atol=1e-5)
    np.testing.assert_allclose(rec[:, 32:64], 5.0, atol=1e-5)
    np.testing.assert_allclose(rec[:, 64:96], -12.5, atol=1e-5)


def test_quantize_validation_errors():
    """Verify quantize input shape and dimension validation."""
    with pytest.raises(ValueError, match="must be 2D"):
        quantize(np.ones(64, dtype=np.float32))

    with pytest.raises(ValueError, match="must be 2D"):
        quantize(np.ones((2, 32, 32), dtype=np.float32))

    with pytest.raises(ValueError, match="multiple of GROUP"):
        quantize(np.ones((32, 45), dtype=np.float32))


# ===========================================================================
# 2. Decode Projection Packing & KAT Hash Tests
# ===========================================================================


def kat_inputs(rows: int, cols: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Known-Answer Test inputs matching laptop reference verification."""
    n = rows * cols
    codes = ((np.arange(n, dtype=np.int64) * 7 + 3) % 16).astype(np.uint8).reshape(rows, cols)
    g = rows * (cols // 32)
    i = np.arange(g, dtype=np.int64)
    scales = (((i % 97) - 48) * 1e-3).astype(np.float32).reshape(rows, cols // 32)
    mins = (((i % 89) - 44) * 3e-3).astype(np.float32).reshape(rows, cols // 32)
    return codes, scales, mins


def test_pack_decode_proj_kat_sha256():
    """Verify pack_decode_proj matches byte-level SHA256 hashes from AMD IRON pack_q4nx."""
    kat_expected = [
        ((64, 256), 10240, "2f934f13747e9f5e261c226655ea811bbae8e45ff6784374d6dd31bd91a394fa"),
        ((1088, 2560), 1740800, "155407fbcfd7ba1b3ddd4a242ec7cb16d5725ad201fa001283180f11048af0bc"),
        ((2560, 1280), 2048000, "12846cb24767c9f150311cba31b9186a87df6fdf12d8c517553493be08181fee"),
        ((128, 768), 61440, "7815e5a5b40aafb908cbb0c774914520fed87cbc09a91a72bd76adab59ca5d7f"),
    ]

    for (rows, cols), exp_len, exp_hash in kat_expected:
        codes, scales, mins = kat_inputs(rows, cols)
        blob = pack_decode_proj(codes, scales, mins)
        assert len(blob) == exp_len, f"Length mismatch for ({rows}, {cols}): {len(blob)} vs {exp_len}"
        sha256 = hashlib.sha256(blob).hexdigest()
        assert sha256 == exp_hash, f"SHA256 mismatch for ({rows}, {cols}): {sha256} vs {exp_hash}"


def test_pack_decode_proj_iron_direct_comparison():
    """Direct comparison with IRON pack_q4nx and parse_q4nx when ml_dtypes and aie are available."""
    pytest.importorskip("ml_dtypes")
    pytest.importorskip("aie")
    from iron.operators.flm.layer.reference import pack_q4nx as iron_pack_q4nx
    from iron.operators.flm.layer.reference import parse_q4nx as iron_parse_q4nx

    rows, cols = 64, 256
    codes, scales, mins = kat_inputs(rows, cols)
    blob_raon = pack_decode_proj(codes, scales, mins)
    blob_iron = iron_pack_q4nx(codes, scales, mins)
    blob_raon_u8 = np.frombuffer(blob_raon, dtype=np.uint8)
    np.testing.assert_array_equal(blob_raon_u8, np.asarray(blob_iron).reshape(-1).view(np.uint8))
    parsed_codes, parsed_scales, parsed_mins = iron_parse_q4nx(blob_raon_u8, 0, rows, cols)
    np.testing.assert_array_equal(codes, parsed_codes)
    if parsed_scales.dtype != np.float32:
        parsed_scales = np.asarray(parsed_scales, dtype=np.float32)
    if parsed_mins.dtype != np.float32:
        parsed_mins = np.asarray(parsed_mins, dtype=np.float32)
    np.testing.assert_array_equal(round_to_bf16_rne(scales), parsed_scales)
    np.testing.assert_array_equal(round_to_bf16_rne(mins), parsed_mins)


def test_pack_decode_proj_roundtrip():
    """Verify pack_decode_proj <-> parse_decode_proj round-trip matches bit-for-bit."""
    rng = np.random.default_rng(777)
    rows, cols = 128, 512
    codes = rng.integers(0, 16, (rows, cols), dtype=np.uint8)
    scales = round_to_bf16_rne(rng.uniform(0.01, 1.0, (rows, cols // GROUP)).astype(np.float32))
    mins = round_to_bf16_rne(rng.uniform(-10.0, 10.0, (rows, cols // GROUP)).astype(np.float32))

    packed_bytes = pack_decode_proj(codes, scales, mins)
    expected_len = (rows // M_TILE) * (cols // K_TILE) * BLOCK_BYTES
    assert len(packed_bytes) == expected_len

    parsed_codes, parsed_scales, parsed_mins = parse_decode_proj(packed_bytes, 0, rows, cols)

    np.testing.assert_array_equal(codes, parsed_codes)
    np.testing.assert_array_equal(scales, parsed_scales)
    np.testing.assert_array_equal(mins, parsed_mins)


def test_pack_decode_proj_shape_validation():
    """Verify ValueError on unsupported dimensions."""
    codes = np.zeros((32, 256), dtype=np.uint8)  # rows 32 not multiple of 64
    scales = np.zeros((32, 8), dtype=np.float32)
    mins = np.zeros((32, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of 64"):
        pack_decode_proj(codes, scales, mins)

    codes2 = np.zeros((64, 128), dtype=np.uint8)  # cols 128 not multiple of 256
    scales2 = np.zeros((64, 4), dtype=np.float32)
    mins2 = np.zeros((64, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of 256"):
        pack_decode_proj(codes2, scales2, mins2)


def test_decode_proj_end_to_end():
    """Quantize -> pack_decode_proj -> parse_decode_proj -> dequantize matches within 0.5 * scale."""
    rng = np.random.default_rng(333)
    rows, cols = 64, 256
    w = rng.standard_normal((rows, cols)).astype(np.float32)

    codes, scales, mins = quantize(w)
    blob = pack_decode_proj(codes, scales, mins)

    p_codes, p_scales, p_mins = parse_decode_proj(blob, 0, rows, cols)
    rec = dequantize_simple(p_codes, p_scales, p_mins)
    err = np.abs(w - rec)
    scales_exp = np.repeat(scales, GROUP, axis=-1)
    assert np.all(err <= 0.5 * scales_exp + 1e-6)


# ===========================================================================
# 3. LMHead Vocabulary Packing & IRON Dequantize Tests
# ===========================================================================


def test_pack_lm_head_roundtrip_and_iron_dequantize():
    """Verify pack_lm_head reproduces iron.operators.flm.lm_head.reference.dequantize."""
    cols, rows = 8, 4
    vocab, dim = 1024, 512  # 1 round (8*4*32=1024), 2 k_blocks (512/256=2)
    rng = np.random.default_rng(888)
    w = rng.standard_normal((vocab, dim)).astype(np.float32)

    codes, scales, mins = quantize(w)
    blob = pack_lm_head(codes, scales, mins, cols=cols, rows=rows)

    # Check 1: parse_lm_head round-trip
    p_codes, p_scales, p_mins = parse_lm_head(blob, dim, vocab, cols=cols, rows=rows)
    np.testing.assert_array_equal(codes, p_codes)
    np.testing.assert_array_equal(scales, p_scales)
    np.testing.assert_array_equal(mins, p_mins)

    # Check 2: IRON lm_head dequantize function
    iron_weights = iron_lm_head_dequantize(np.frombuffer(blob, dtype=np.uint8), dim, vocab, cols, rows)
    expected_weights = dequantize_simple(codes, scales, mins)

    max_diff = np.max(np.abs(iron_weights - expected_weights))
    assert max_diff == 0.0, f"Mismatch with IRON reference dequantize: max diff = {max_diff}"


def test_pack_lm_head_multiple_rounds():
    """Verify pack_lm_head across multiple rounds (e.g. 2 rounds = 2048 vocab)."""
    cols, rows = 8, 4
    vocab, dim = 2048, 256  # 2 rounds
    rng = np.random.default_rng(999)
    w = rng.standard_normal((vocab, dim)).astype(np.float32)

    codes, scales, mins = quantize(w)
    blob = pack_lm_head(codes, scales, mins, cols=cols, rows=rows)

    iron_weights = iron_lm_head_dequantize(np.frombuffer(blob, dtype=np.uint8), dim, vocab, cols, rows)
    expected_weights = dequantize_simple(codes, scales, mins)
    np.testing.assert_array_equal(iron_weights, expected_weights)


def test_pack_lm_head_shape_validation():
    """Verify pack_lm_head dimension checks."""
    codes = np.zeros((1000, 256), dtype=np.uint8)  # 1000 not multiple of 1024
    scales = np.zeros((1000, 8), dtype=np.float32)
    mins = np.zeros((1000, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of 1024"):
        pack_lm_head(codes, scales, mins, cols=8, rows=4)

    codes2 = np.zeros((1024, 200), dtype=np.uint8)  # 200 not multiple of 256
    scales2 = np.zeros((1024, 6), dtype=np.float32)
    mins2 = np.zeros((1024, 6), dtype=np.float32)
    with pytest.raises(ValueError, match="multiple of K_TILE"):
        pack_lm_head(codes2, scales2, mins2, cols=8, rows=4)


def test_parse_lm_head_shape_validation():
    """parse_lm_head must reject the same shapes pack_lm_head rejects.

    parse_gemv accepts any M that is a multiple of rows*M_TILE (128); the
    LMHead layout additionally requires a full round (cols*rows*M_TILE =
    1024) and a whole number of K tiles, exactly like pack_lm_head.
    """
    # vocab 1000 is a multiple of 128 (rows*M_TILE) but NOT of 1024
    # (cols*rows*M_TILE) — a valid GEMV shape, invalid LMHead shape.
    blob = np.zeros(1000 * 256 * 5 // 8, dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple of 1024"):
        parse_lm_head(blob, dim=256, vocab=1000, cols=8, rows=4)

    # dim not a multiple of K_TILE must be rejected with the LMHead wording.
    blob2 = np.zeros(1024 * 256 * 5 // 8, dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple of K_TILE"):
        parse_lm_head(blob2, dim=200, vocab=1024, cols=8, rows=4)

    # Sanity: a valid LMHead shape still parses (round-trip of pack_lm_head).
    codes = np.full((1024, 256), 3, dtype=np.uint8)
    scales = np.full((1024, 8), 0.5, dtype=np.float32)
    mins = np.full((1024, 8), 1.0, dtype=np.float32)
    ok_blob = pack_lm_head(codes, scales, mins, cols=8, rows=4)
    p_codes, p_scales, p_mins = parse_lm_head(ok_blob, 256, 1024, cols=8, rows=4)
    np.testing.assert_array_equal(p_codes, codes)
    np.testing.assert_allclose(p_scales, scales, rtol=0, atol=0)
    np.testing.assert_allclose(p_mins, mins, rtol=0, atol=0)


# ===========================================================================
# 4. Raon Model Shapes Padding & Unpadding Tests
# ===========================================================================


def test_pad_unpad_raon_shapes():
    """Verify padding and unpadding for all Raon-1-mini specific weight shapes."""
    rng = np.random.default_rng(1234)

    # 1. MoE gate/up: (1088, 2560)
    w_gate = rng.standard_normal((1088, 2560), dtype=np.float32)
    p_gate, shape_gate = pad(w_gate, m_tile=32, k_tile=256)
    assert p_gate.shape == (1088, 2560)
    np.testing.assert_array_equal(unpad(p_gate, shape_gate), w_gate)

    # 2. MoE down: (2560, 1088)
    w_down = rng.standard_normal((2560, 1088), dtype=np.float32)
    p_down, shape_down = pad(w_down, m_tile=32, k_tile=256)
    assert p_down.shape == (2560, 1280)
    assert np.all(p_down[:, 1088:] == 0.0)
    np.testing.assert_array_equal(unpad(p_down, shape_down), w_down)

    # 3. RDA low-rank: (40, 128)
    w_rda = rng.standard_normal((40, 128), dtype=np.float32)
    p_rda, shape_rda = pad(w_rda, m_tile=32, k_tile=256)
    assert p_rda.shape == (64, 256)
    assert np.all(p_rda[40:, :] == 0.0)
    assert np.all(p_rda[:, 128:] == 0.0)
    np.testing.assert_array_equal(unpad(p_rda, shape_rda), w_rda)

    # 4. MLA q_b: (512, 576)
    w_mla = rng.standard_normal((512, 576), dtype=np.float32)
    p_mla, shape_mla = pad(w_mla, m_tile=32, k_tile=256)
    assert p_mla.shape == (512, 768)
    assert np.all(p_mla[:, 576:] == 0.0)
    np.testing.assert_array_equal(unpad(p_mla, shape_mla), w_mla)

    # 5. Vocabulary: (259840, 2560) - Full Raon dimension
    w_vocab = np.zeros((259840, 2560), dtype=np.float16)  # float16 for memory efficiency
    w_vocab[0, 0] = 1.0
    w_vocab[-1, -1] = 2.0
    p_vocab, shape_vocab = pad(w_vocab, m_tile=1024, k_tile=256)
    assert p_vocab.shape == (260096, 2560)
    assert np.all(p_vocab[259840:, :] == 0.0)
    assert p_vocab[0, 0] == 1.0
    assert p_vocab[259839, 2559] == 2.0
    unpadded = unpad(p_vocab, shape_vocab)
    assert unpadded.shape == (259840, 2560)
    assert unpadded[0, 0] == 1.0
    assert unpadded[-1, -1] == 2.0


# ===========================================================================
# 5. Padded Logits Masking Tests
# ===========================================================================


def test_mask_padded_logits():
    """Verify mask_padded_logits sets padded vocabulary indices to -inf."""
    true_vocab = 259840
    padded_vocab = 260096

    # 1D logits
    logits_1d = np.zeros(padded_vocab, dtype=np.float32)
    masked_1d = mask_padded_logits(logits_1d, true_vocab, inplace=False)
    assert np.all(masked_1d[:true_vocab] == 0.0)
    assert np.all(masked_1d[true_vocab:] == -np.inf)
    assert logits_1d[true_vocab] == 0.0  # verify non-inplace copy

    # 2D logits [batch, vocab]
    logits_2d = np.ones((3, padded_vocab), dtype=np.float32)
    masked_2d = mask_padded_logits(logits_2d, true_vocab, inplace=True)
    assert np.all(masked_2d[:, :true_vocab] == 1.0)
    assert np.all(masked_2d[:, true_vocab:] == -np.inf)
    assert logits_2d[0, true_vocab] == -np.inf  # verify inplace modification


# ===========================================================================
# 6. BF16 Bit-Level Precision Tests
# ===========================================================================


def test_to_bf16_rne_precision():
    """Verify to_bf16_u16 round-to-nearest-even matches IEEE-754 specifications."""
    # Test values with exact ties
    # 1.0 in f32: 0x3F800000. LSB of bf16 is bit 16.
    val_even = 1.0 + 2**(-8)  # bit 16 is 0, bit 15 is 1, sticky 0 -> round to even (1.0)
    u_even = to_bf16_u16(np.array([val_even], dtype=np.float32))[0]
    assert u_even == 0x3F80  # 1.0

    val_odd = 1.0 + 2**(-7) + 2**(-8)  # bit 16 is 1, bit 15 is 1, sticky 0 -> round up to even
    u_odd = to_bf16_u16(np.array([val_odd], dtype=np.float32))[0]
    assert u_odd == 0x3F82
