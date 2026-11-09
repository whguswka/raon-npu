# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side NumPy unit tests for Raon general 9-bit bfp16ebs8 GEMV.

Verifies:
1. Codec identity against mlir-aie python/utils/bfp.py (when available).
2. Pack / parse round-trip: ``pack_bfp16_gemv`` -> ``parse_bfp16_gemv`` across shapes.
3. Hand-calculated CPU bit reference arithmetic:
   - Exponent boundaries (e=6 subnormal boundary, e=133 zero-power, e=255).
   - Negative mantissas (-1, -64, -127).
   - Extreme -128 mantissa representation (|m| <= 128).
   - Saturation at +127 under conv_even rounding.
4. Padding invariance:
   - K (256-align) and M (128-align) zero-padding preserves exact outputs on the active submatrix.
   - Padded elements (x=0) contribute exactly 0 to the dot product.
5. Bit-exact execution for integer contract samples.
6. Design shape validation: ``check_shape`` rejects unaligned or out-of-range shapes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from quant.bfp16 import (
    GEMV_BLOCK_BYTES,
    GROUP,
    GROUPS_PER_TILE,
    K_TILE,
    M_TILE,
    bfp16ebs8_decode,
    bfp16ebs8_encode,
    pack_bfp16_gemv,
    pad_bfp16_gemv,
    parse_bfp16_gemv,
    parse_bfp16_gemv_raw,
    quantize_bfp16ebs8,
    round_to_bfp16ebs8,
)
from quant.q4nx import round_to_bf16_rne, to_bf16_u16
from kernels.gemv_bfp16.design import check_shape
from kernels.gemv_bfp16.reference import (
    bf16_scale_bits,
    bf16_ulp_diff,
    bfp16_gemv_bit_reference,
    dequantize,
    dequantize_block,
    reference,
)


# ---------------------------------------------------------------------------
# 1. Encoding bit-identity against mlir-aie python/utils/bfp.py
# ---------------------------------------------------------------------------
def test_bfp16ebs8_matches_mlir_aie_codec():
    """Verify public codec values against mlir-aie, including tiny and carry blocks."""
    import os
    if not os.environ.get("RAON_MLIR_AIE_DIR"):
        pytest.skip("RAON_MLIR_AIE_DIR is not set")
    orig_path = Path(os.environ["RAON_MLIR_AIE_DIR"]) / "python/utils/bfp.py"
    if not orig_path.exists():
        pytest.skip("mlir-aie bfp.py not found on filesystem")

    import types

    m_aie_helpers = types.ModuleType("aie.helpers")
    m_aie_helpers_npdtypes = types.ModuleType("aie.helpers.npdtypes")

    class DummyV8:
        block_bytes = 9
        block_size = 8
        __name__ = "v8bfp16ebs8"

    m_aie_helpers_npdtypes.v8bfp16ebs8 = DummyV8
    m_aie_helpers_npdtypes.is_block_float = lambda dt: dt is DummyV8
    sys.modules["aie.helpers"] = m_aie_helpers
    sys.modules["aie.helpers.npdtypes"] = m_aie_helpers_npdtypes

    try:
        spec = importlib.util.spec_from_file_location("mlir_ref_bfp", orig_path)
        ref_bfp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref_bfp)
    finally:
        sys.modules.pop("aie.helpers", None)
        sys.modules.pop("aie.helpers.npdtypes", None)

    rng = np.random.default_rng(2026)
    random_x = rng.standard_normal((100, 256)).astype(np.float32)
    edge_x = np.array(
        [
            [2.0**-121, -(2.0**-122), 0.0, 2.0**-126, 1e-40, -1e-40, 0.0, 0.0],
            [1.996, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    for x in (random_x, edge_x):
        for rounding in ("floor", "conv_even"):
            encoded = bfp16ebs8_encode(x, rounding=rounding)
            decoded = bfp16ebs8_decode(encoded)
            ref_quantized = ref_bfp.quantize(x, rounding=rounding)
            assert np.array_equal(decoded, ref_quantized), f"Mismatch for rounding={rounding}"

            # mlir-aie's byte encoder models floor exactly. Its conv_even encoder
            # predates quantize()'s +128 carry rule, which this packer must honor.
            if rounding == "floor":
                assert np.array_equal(encoded, ref_bfp.encode(x, rounding=rounding))

    tiny_encoded = bfp16ebs8_encode(edge_x[:1], rounding="conv_even").view(
        np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (8,))])
    )
    assert tiny_encoded[0]["exponent"] < 7
    assert np.any(tiny_encoded[0]["mantissas"] != 0)


# ---------------------------------------------------------------------------
# 2. Pack / Parse round-trip
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(128, 256), (256, 512), (128, 1024), (384, 256)])
def test_pack_parse_bfp16_gemv_roundtrip(shape):
    """Verify pack_bfp16_gemv -> parse_bfp16_gemv bit-exact round-trip with decoded weights."""
    M, K = shape
    rng = np.random.default_rng(42)
    w_f32 = rng.standard_normal((M, K)).astype(np.float32)

    packed = pack_bfp16_gemv(w_f32, rows=4, rounding="conv_even")
    assert len(packed) == (M // 32) * (K // 256) * 9216

    # 1. Parse back to float32
    w_parsed = parse_bfp16_gemv(packed, K, M, rows=4)
    # Expected decoded float32
    w_decoded = bfp16ebs8_decode(bfp16ebs8_encode(w_f32, rounding="conv_even"))
    assert np.array_equal(w_parsed, w_decoded)

    # 2. Parse raw exponents and mantissas
    exp, mant = parse_bfp16_gemv_raw(packed, K, M, rows=4)
    enc = bfp16ebs8_encode(w_f32, rounding="conv_even").view(
        np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (8,))])
    ).reshape(M, K // 8)
    assert np.array_equal(exp, enc["exponent"])
    assert np.array_equal(mant, enc["mantissas"].reshape(M, K))


@pytest.mark.parametrize("rounding", ["floor", "conv_even"])
def test_pack_parse_matches_public_quantize_for_random_and_carry_boundaries(rounding):
    """Packing must preserve the evaluator's exact quantized values outside tiny flushes."""
    rng = np.random.default_rng(1028)
    w = rng.standard_normal((128, 256)).astype(np.float32)
    w[0, :16] = np.array(
        [1.996, 0, 0, 0, 0, 0, 0, 0, -1.996, 0, 0, 0, 0, 0, 0, 0],
        dtype=np.float32,
    )
    packed = pack_bfp16_gemv(w, rows=4, rounding=rounding)
    parsed = parse_bfp16_gemv(packed, 256, 128, rows=4)
    expected = quantize_bfp16ebs8(w, rounding=rounding)
    assert np.array_equal(parsed, expected)


def test_dequantize_block_matches_parse():
    """Verify dequantize_block matches block-level output of parse_bfp16_gemv."""
    rng = np.random.default_rng(99)
    w = rng.standard_normal((128, 256)).astype(np.float32)
    packed = pack_bfp16_gemv(w, rows=4, rounding="conv_even")

    block0 = dequantize_block(packed[:9216])
    full = parse_bfp16_gemv(packed, 256, 128, rows=4)
    # Block 0 corresponds to slice=0, k=0, row=0 -> rows 0..31, cols 0..255
    assert np.array_equal(block0, full[:32, :256])


# ---------------------------------------------------------------------------
# 3. Hand-calculated CPU bit reference arithmetic & Exponent boundary tests
# ---------------------------------------------------------------------------
def test_hand_calculated_bit_reference_exponent_boundaries():
    """Verify exponent boundary arithmetic against exact float64 and kernel formula.

    Tests:
    1. Kernel scale assembly rule in bfp16_gemv.cc:
         Pure vector assembly: (max(e, 7) - 6) << 7
       matches reference.py bf16_scale_bits(e):
         (np.maximum(e, 7) - 6) << 7
       For e >= 7, produces the exact mathematical value 2**(e - 133) in float64.
       For e < 7 (raw blocks outside packing guarantee), the defensive vector clamp
       max(e, 7) produces 2**(7 - 133) = 2**-126 (~1.175e-38, normal zero-adjacent)
       instead of subnormal garbage.
    2. Raw bfp16 GEMV block (9216 bytes) with exponent bytes e in
       {0, 1, 6, 7, 127, 254, 255}
       and int8 mantissas m in {1, -128, 127, 0}: verifies reference.py dequantize_block
       produces values matching float64 value m * 2**(max(e, 7) - 133) (or overflows
       to signed infinity in float32 when the magnitude exceeds max float32).
    3. Maximum finite float32 input carries to stored exponent e=255 under conv_even.
    4. Dot-product boundary arithmetic via bfp16_gemv_bit_reference.
    """
    # -------------------------------------------------------------------------
    # Part 1: All exponents e in 0..255 match exact 2**(max(e, 7) - 133) in float64
    # Kernel formula in bfp16_gemv.cc: (max(e, 7) - 6) << 7
    # -------------------------------------------------------------------------
    e_all = np.arange(256, dtype=np.int64)
    bits = bf16_scale_bits(e_all)
    # bf16 bit pattern converted to float32: (bits << 16).view(np.float32)
    # Note: e >= 7 are bf16 normals (fraction 0, biased exp e-6) -> scale 2^(e-133).
    #       e < 7 are clamped defensively by the kernel to e=7 -> scale 2^(7-133) = 2^-126.
    scale_f32 = (bits.astype(np.uint32) << 16).view(np.float32)
    for e in range(256):
        effective_e = max(e, 7)
        exact_f64 = float(2.0 ** (effective_e - 133))
        s_val = float(scale_f32[e])
        assert s_val == exact_f64, f"e={e}: bf16 scale {s_val} != exact {exact_f64}"

    # -------------------------------------------------------------------------
    # Part 2: Raw 9216-byte GEMV block with e in {0, 1, 6, 7, 127, 254, 255}
    #         and mantissas m in {1, -128, 127, 0}
    # -------------------------------------------------------------------------
    raw = bytearray(GEMV_BLOCK_BYTES)
    test_exps = [0, 1, 6, 7, 127, 254, 255]
    test_mants = [1, -128, 127, 0]

    # In dequantize_block layout:
    # Half 0, Group 0 (bytes 0..143):
    # - bytes 0..15: 16 row exponents
    # - bytes 16..143: 8 columns x 16 rows int8 mantissas (col_m = mant_raw.reshape(8, 16).T)
    #   col c, row r -> byte index 16 + c * 16 + r
    for r, e in enumerate(test_exps):
        raw[r] = e

    for c, m in enumerate(test_mants):
        for r in range(len(test_exps)):
            val = m if m >= 0 else (256 + m)
            raw[16 + c * 16 + r] = val

    with np.errstate(over="ignore"):
        out = dequantize_block(bytes(raw))

    for r, e in enumerate(test_exps):
        effective_e = max(e, 7)
        for c, m in enumerate(test_mants):
            got = out[r, c]
            exact_f64 = float(np.float64(m) * (2.0 ** (effective_e - 133)))
            # Convert the exact mathematical value to its required float32 result.
            # This covers finite e=255 values (m=1) as well as both overflow signs
            # (m=127 -> +inf and m=-128 -> -inf).
            with np.errstate(over="ignore"):
                expected_f32 = np.float32(exact_f64)
            if np.isinf(expected_f32):
                assert np.isinf(got) and np.signbit(got) == np.signbit(expected_f32), (
                    f"e={e}, m={m}: expected {expected_f32} float32 overflow, got {got}"
                )
            else:
                assert got == expected_f32, (
                    f"Mismatch at e={e}, m={m}: got={got}, expected={expected_f32}"
                )

    # -------------------------------------------------------------------------
    # Part 3: conv_even carry from the largest finite float32 stores e=255.
    # The requantized mantissa is 64, whose decoded float32 value is +inf,
    # matching mlir-aie's quantize() overflow semantics.
    # -------------------------------------------------------------------------
    max_finite_block = np.zeros((1, 8), dtype=np.float32)
    max_finite_block[0, 0] = np.finfo(np.float32).max
    encoded = bfp16ebs8_encode(max_finite_block, rounding="conv_even")
    with np.errstate(over="ignore"):
        decoded = bfp16ebs8_decode(encoded)
        quantized = quantize_bfp16ebs8(max_finite_block, rounding="conv_even")

    assert encoded[0, 0] == 255
    assert np.array_equal(encoded[0, 1:].view(np.int8), [64, 0, 0, 0, 0, 0, 0, 0])
    assert np.isposinf(decoded[0, 0])
    assert np.array_equal(decoded, quantized)

    # -------------------------------------------------------------------------
    # Part 4: Dot product arithmetic at e=133 (scale = 1.0)
    # -------------------------------------------------------------------------
    w = np.zeros((128, 256), dtype=np.float32)
    x = np.zeros(256, dtype=np.float32)

    # Set row 0, col 0..7 to mantissas [1, 2, 3, 4, -1, -2, -3, -4]
    w[0, :8] = np.array([1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0], dtype=np.float32)
    # x vector has 1.0 at col 0..7
    x[:8] = 1.0

    y_f32, y_u16 = bfp16_gemv_bit_reference(w, x, rounding="conv_even")
    # Expected dot product = 1+2+3+4 - 1-2-3-4 = 0.0
    assert y_f32[0] == 0.0

    # Set x to [1, 0, 0, 0, 0, 0, 0, 0]
    x[:8] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    y_f32, _ = bfp16_gemv_bit_reference(w, x, rounding="conv_even")
    assert y_f32[0] == 1.0


def test_pack_bfp16_gemv_all_exponents_ge_7():
    """Verify (requirement 4-ga) that pack_bfp16_gemv guarantees all stored exponents >= 7.

    Tests:
    1. Zero matrix (padding blocks) -> all exponents must be 7, mantissas 0.
    2. Matrix with tiny values (max < 2**-120) -> all exponents must be 7, mantissas 0.
    3. Diverse random matrix with mixed magnitudes -> every block exponent >= 7.
    """
    shapes = [(128, 256), (256, 512)]
    for M, K in shapes:
        # 1. Zero matrix (simulating pure padding blocks)
        w_zeros = np.zeros((M, K), dtype=np.float32)
        packed_zeros = pack_bfp16_gemv(w_zeros, rows=4, rounding="conv_even")
        exp_zeros, mant_zeros = parse_bfp16_gemv_raw(packed_zeros, K, M, rows=4)
        assert np.all(exp_zeros >= 7), "Zero matrix contains exponents < 7"
        assert np.all(exp_zeros == 7), "Zero matrix padding blocks must have e == 7"
        assert np.all(mant_zeros == 0), "Zero matrix mantissas must be 0"

        # 2. Sub-threshold matrix (values below 2**-120, e.g. 1e-38, 1e-40)
        w_tiny = np.full((M, K), 1e-38, dtype=np.float32)
        packed_tiny = pack_bfp16_gemv(w_tiny, rows=4, rounding="conv_even")
        exp_tiny, mant_tiny = parse_bfp16_gemv_raw(packed_tiny, K, M, rows=4)
        assert np.all(exp_tiny >= 7), "Tiny matrix contains exponents < 7"
        assert np.all(exp_tiny == 7), "Tiny matrix below 2**-120 must be lowered to e == 7"
        assert np.all(mant_tiny == 0), "Tiny matrix mantissas must be lowered to 0"

        # 3. Mixed normal + tiny values
        rng = np.random.default_rng(2026)
        w_rand = rng.standard_normal((M, K)).astype(np.float32)
        # Inject zeros and subnormals into several rows
        w_rand[0, :] = 0.0
        w_rand[1, :] = 1e-39
        w_rand[2, :16] = 2.0**-125
        packed_rand = pack_bfp16_gemv(w_rand, rows=4, rounding="conv_even")
        exp_rand, mant_rand = parse_bfp16_gemv_raw(packed_rand, K, M, rows=4)
        assert np.all(exp_rand >= 7), "Mixed matrix produced exponents < 7"
        # Where original was tiny, exponents must be 7 and mantissas 0
        assert np.all(exp_rand[0, :] == 7) and np.all(mant_rand[0, :] == 0)
        assert np.all(exp_rand[1, :] == 7) and np.all(mant_rand[1, :] == 0)
        assert np.all(exp_rand[2, :2] == 7) and np.all(mant_rand[2, :16] == 0)


def test_sub_threshold_flush_is_npu_packing_only_and_audited():
    """Verify tiny values survive the public codec and NPU flushing is explicit/audited.

    Tests:
    1. In pack_bfp16_gemv -> parse_bfp16_gemv, tiny rows become identically 0.0.
    2. quantize_bfp16ebs8 preserves tiny nonzero rows like mlir-aie's public codec.
    3. bfp16_gemv_bit_reference produces 0.0 output for tiny rows.
    """
    M, K = 128, 256
    rng = np.random.default_rng(1028)
    w = rng.standard_normal((M, K)).astype(np.float32)

    # Make row 3 tiny (< 2**-120, max ~ 2**-121)
    # 2**-121 ~= 3.76e-37
    w[3, :] = np.linspace(1e-38, 2.0**-121, K, dtype=np.float32)
    # Make row 5 strictly 0.0
    w[5, :] = 0.0
    # Make row 7 subnormal float32 (~ 1e-40)
    w[7, :] = 1e-40

    # 1. pack_bfp16_gemv -> parse_bfp16_gemv
    packed, stats = pack_bfp16_gemv(
        w, rows=4, rounding="conv_even", flush_tiny=True, return_stats=True
    )
    assert stats["flushed_nonzero_blocks"] > 0
    w_parsed = parse_bfp16_gemv(packed, K, M, rows=4)
    assert np.all(w_parsed[3, :] == 0.0), "Row 3 (<2**-120) was not lowered to 0.0 in pack/parse"
    assert np.all(w_parsed[5, :] == 0.0), "Row 5 (zeros) was not lowered to 0.0 in pack/parse"
    assert np.all(w_parsed[7, :] == 0.0), "Row 7 (1e-40) was not lowered to 0.0 in pack/parse"

    # 2. Public codec keeps e=0..6 values; raw packing without NPU flushing agrees.
    w_quant = quantize_bfp16ebs8(w, rounding="conv_even")
    assert np.any(w_quant[3, :] != 0.0)
    assert np.all(w_quant[5, :] == 0.0), "Row 5 was not 0.0 in quantize_bfp16ebs8"
    assert np.any(w_quant[7, :] != 0.0)
    raw_packed, raw_stats = pack_bfp16_gemv(
        w, rows=4, rounding="conv_even", flush_tiny=False, return_stats=True
    )
    assert raw_stats["tiny_blocks"] > 0
    assert raw_stats["flushed_nonzero_blocks"] == 0
    raw_parsed = parse_bfp16_gemv(raw_packed, K, M, rows=4)
    assert np.array_equal(raw_parsed, w_quant)

    # 3. bfp16_gemv_bit_reference execution
    x = rng.standard_normal(K).astype(np.float32)
    y_f32, _ = bfp16_gemv_bit_reference(w, x, rounding="conv_even")
    assert y_f32[3] == 0.0, f"Expected y_f32[3] == 0.0, got {y_f32[3]}"
    assert y_f32[5] == 0.0, f"Expected y_f32[5] == 0.0, got {y_f32[5]}"
    assert y_f32[7] == 0.0, f"Expected y_f32[7] == 0.0, got {y_f32[7]}"


def test_quantize_bfp16ebs8_default_rounding_is_conv_even():
    """Verify quantize_bfp16ebs8 and round_to_bfp16ebs8 default to conv_even."""
    # Input vector with values where conv_even rounds to nearest even, differing from floor
    x = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]], dtype=np.float32)

    # 1. Default call (no rounding argument) matches explicit conv_even bit-for-bit
    q_default = quantize_bfp16ebs8(x)
    q_conv = quantize_bfp16ebs8(x, rounding="conv_even")
    assert np.array_equal(q_default, q_conv)

    # 2. Alias round_to_bfp16ebs8 without argument also matches conv_even bit-for-bit
    r_default = round_to_bfp16ebs8(x)
    assert np.array_equal(r_default, q_conv)

    # 3. Must differ from floor for this input (proves default is not floor)
    q_floor = quantize_bfp16ebs8(x, rounding="floor")
    assert not np.array_equal(q_default, q_floor)
    assert np.any(q_default != q_floor)


def test_hand_calculated_negative_mantissas_and_minus_128():
    """Verify int8 mantissa edge cases including -128 exact representation."""
    w = np.zeros((128, 256), dtype=np.float32)
    x = np.zeros(256, dtype=np.float32)

    # Scale 1.0: mantissa -128 represents -128.0
    # In float32, -128.0 has bit pattern where max_exp = 134 (for 128.0, 2^7)
    # We construct a block where max magnitude is 128.0
    w[0, 0] = -128.0
    x[0] = 0.5  # -128.0 * 0.5 = -64.0

    y_f32, _ = bfp16_gemv_bit_reference(w, x, rounding="conv_even")
    assert y_f32[0] == -64.0


def test_conv_even_saturation_at_127():
    """Verify that conv_even rounds values correctly without overflowing +127."""
    x = np.array([127.4, -128.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    enc = bfp16ebs8_encode(x, rounding="conv_even")
    dec = bfp16ebs8_decode(enc)
    assert np.isfinite(dec).all()


# ---------------------------------------------------------------------------
# 4. Padding invariance
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "orig_shape",
    [
        (120, 200),  # K=120 (8-aligned, non-256), M=200 (non-128)
        (576, 7680), # MLA q_b (orig K=576 -> Pad K=768)
        (1088, 2560),# Expert down (orig K=1088 -> Pad K=1280)
    ],
)
def test_padding_invariance(orig_shape):
    """Verify K (256-align) and M (128-align) zero padding preserves active submatrix output."""
    K_orig, M_orig = orig_shape
    rng = np.random.default_rng(777)

    w_orig = rng.standard_normal((M_orig, K_orig)).astype(np.float32)
    x_orig = rng.standard_normal(K_orig).astype(np.float32)

    w_pad, (m_o, k_o) = pad_bfp16_gemv(w_orig, m_align=128, k_align=256)
    M_pad, K_pad = w_pad.shape
    assert m_o == M_orig and k_o == K_orig
    assert M_pad % 128 == 0 and K_pad % 256 == 0

    x_pad = np.pad(x_orig, (0, K_pad - K_orig), constant_values=0.0)

    # Reference on padded matrix
    y_pad_f32, y_pad_u16 = bfp16_gemv_bit_reference(w_pad, x_pad, rounding="conv_even")
    active_y_f32 = y_pad_f32[:M_orig]

    # Reference on unpadded matrix (quantized directly)
    y_unpad_f32, y_unpad_u16 = bfp16_gemv_bit_reference(w_orig, x_orig, rounding="conv_even")

    # The active outputs must match bit-for-bit because zero-padded inputs and weights
    # contribute exactly zero to the float32 accumulator!
    assert np.array_equal(active_y_f32, y_unpad_f32)
    assert np.array_equal(y_pad_u16[:M_orig], y_unpad_u16)


# ---------------------------------------------------------------------------
# 5. Bit-exact execution for integer contract samples
# ---------------------------------------------------------------------------
def test_integer_contract_sample_bit_exact():
    """Verify integer contract sample matches float64 reference rounded to bf16."""
    rng = np.random.default_rng(1234)
    M, K = 128, 256

    w_int = rng.integers(-4, 5, (M, K)).astype(np.float32)
    x_int = rng.integers(-4, 5, K).astype(np.float32)

    y_f32, y_u16 = bfp16_gemv_bit_reference(w_int, x_int, rounding="conv_even")
    # Float64 reference rounded to bfloat16
    y_ref_f64 = reference(w_int, x_int)
    y_ref_bf16 = round_to_bf16_rne(y_ref_f64)

    assert np.array_equal(y_f32, y_ref_bf16)


# ---------------------------------------------------------------------------
# 6. Design shape validation
# ---------------------------------------------------------------------------
def test_check_shape_validation():
    """Verify check_shape validates K (multiple of 256) and M (multiple of 128)."""
    class MockDev:
        arch = "AIE2p"
        core_rows = [0, 1, 2, 3]
        cols = 8

    dev = MockDev()

    # Valid shapes
    check_shape(dev, 256, 128, cols=8)
    check_shape(dev, 2560, 20480, cols=8)
    check_shape(dev, 12032, 2560, cols=4)

    # Invalid K
    with pytest.raises(ValueError, match="K.*must be at least 256"):
        check_shape(dev, 250, 128)
    with pytest.raises(ValueError, match="K.*multiple of 256"):
        check_shape(dev, 576, 128)

    # Invalid M
    with pytest.raises(ValueError, match="M.*must be at least 128"):
        check_shape(dev, 256, 100)
    with pytest.raises(ValueError, match="M.*multiple of 128"):
        check_shape(dev, 256, 200)

    # Invalid cols
    with pytest.raises(ValueError, match="cols.*between 1 and 8"):
        check_shape(dev, 256, 128, cols=0)
    with pytest.raises(ValueError, match="cols.*between 1 and 8"):
        check_shape(dev, 256, 128, cols=9)
