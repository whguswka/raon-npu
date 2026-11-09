# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""CPU reference for Raon general 9-bit bfp16ebs8 GEMV, in float64 and bit-exact bf16."""

from __future__ import annotations

import numpy as np

from quant.bfp16 import (
    GROUP,
    GROUPS_PER_TILE,
    HALF_BYTES,
    HALF_ROWS,
    K_TILE,
    M_TILE,
    GEMV_BLOCK_BYTES,
    bfp16ebs8_decode,
    bfp16ebs8_encode,
    parse_bfp16_gemv,
    parse_bfp16_gemv_raw,
    quantize_bfp16ebs8,
)
from quant.q4nx import round_to_bf16_rne, to_bf16_u16


def bf16_scale_bits(e):
    """bf16 bit patterns of the scale factors 2**(max(e, 7) - 133), exactly as the kernel
    (bfp16_gemv.cc) assembles them with a defensive vector ``max(e, 7)`` clamp.

    NPU packing with ``flush_tiny=True`` guarantees e >= 7 for every packed block
    (it lowers e < 7 blocks to zero: mantissa 0, e = 7), so ``(e - 6) << 7`` is always the normal bf16
    bit pattern and no bf16 subnormal is ever assembled. The clamp only matters for
    raw blocks built outside the packer: e < 7 decodes to ~2**-126 (zero times the
    scale) instead of subnormal garbage.
    """
    e = np.asarray(e, dtype=np.int64)
    ec = np.maximum(e, 7)  # defensive vector clamp in the kernel
    bits = (ec - 6) << 7  # normal bf16: biased exponent field e - 6
    return bits.astype(np.uint16)


def bf16_scale(e):
    """Exact float32 values of the bf16 scale factors assembled by ``bf16_scale_bits``."""
    bits = bf16_scale_bits(e)
    return (bits.astype(np.uint32) << 16).view(np.float32)


def dequantize_block(block: bytes | np.ndarray) -> np.ndarray:
    """Dequantize one 9-bit bfp16 GEMV block (9216 bytes) to its [M_TILE, K_TILE] float32 weights."""
    b = bytes(block) if isinstance(block, (np.ndarray, memoryview)) else block
    if len(b) < GEMV_BLOCK_BYTES:
        raise ValueError(
            f"block length ({len(b)}) is less than required GEMV_BLOCK_BYTES ({GEMV_BLOCK_BYTES})"
        )

    out = np.empty((M_TILE, K_TILE), dtype=np.float32)
    raw = np.frombuffer(b[:GEMV_BLOCK_BYTES], dtype=np.uint8)

    for h in range(2):
        h_m_start = h * HALF_ROWS
        h_m_end = h_m_start + HALF_ROWS
        half_offset = h * HALF_BYTES

        for g in range(GROUPS_PER_TILE):
            g_offset = half_offset + g * 144
            exp = raw[g_offset : g_offset + 16]
            # Same rule as the kernel (bfp16_gemv.cc): defensive vector max(e, 7),
            # then the normal bf16 bit pattern (e - 6) << 7. Packed blocks always have
            # e >= 7 (pack_bfp16_gemv lowers e < 7 blocks to mantissa 0, e = 7); for
            # raw blocks with e < 7 this yields ~2**-126, not a subnormal.
            scale = bf16_scale(exp)

            mant_raw = raw[g_offset + 16 : g_offset + 144].view(np.int8)
            col_m = mant_raw.reshape(8, 16).T  # [16 rows, 8 cols]

            col_start = g * 8
            col_end = col_start + 8
            out[h_m_start:h_m_end, col_start:col_end] = (
                col_m.astype(np.float32) * scale[:, None]
            )

    return out


def dequantize(w: bytes | np.ndarray, K: int, M: int, rows: int = 4) -> np.ndarray:
    """Dequantize whole packed bfp16 GEMV buffer to [M, K] float32 weights."""
    return parse_bfp16_gemv(w, K, M, rows)


def reference(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pure linear matrix-vector product in float64 (ideal mathematical output).

    Args:
        weights: [M, K] weight matrix (float32 or float64).
        x: [K] input vector (float32 or float64).

    Returns:
        float64 array of shape [M].
    """
    w = np.asarray(weights, dtype=np.float64)
    x_vec = np.asarray(x, dtype=np.float64).reshape(-1)
    return w @ x_vec


def bf16_ulp_diff(a_u16: np.ndarray, b_u16: np.ndarray) -> np.ndarray:
    """Compute the elementwise ULP difference between two bfloat16 arrays.

    Maps sign-magnitude uint16 bit patterns to ordered signed integers,
    treating both +0.0 (0x0000) and -0.0 (0x8000) as 0.
    """
    ua = np.asarray(a_u16, dtype=np.uint16)
    ub = np.asarray(b_u16, dtype=np.uint16)

    sign_a = (ua >> 15) & 1
    mag_a = (ua & 0x7FFF).astype(np.int32)
    int_a = np.where(sign_a == 1, -mag_a, mag_a)

    sign_b = (ub >> 15) & 1
    mag_b = (ub & 0x7FFF).astype(np.int32)
    int_b = np.where(sign_b == 1, -mag_b, mag_b)

    return np.abs(int_a - int_b)


def bfp16_gemv_bit_reference(
    weights: np.ndarray,
    x: np.ndarray,
    *,
    rounding: str = "conv_even",
) -> tuple[np.ndarray, np.ndarray]:
    """Model of Raon general 9-bit bfp16 GEMV as executed on AMD XDNA 2 AIE2P.

    Hardware pipeline modeled (bfp16_gemv.cc):
    1. Offline quantization / packing:
       Weights are quantized into 9-bit bfp16ebs8 along the K axis in 8-element groups.
       With rounding='conv_even', mantissa clipping/rounding is unbiased.
       NPU packing with ``flush_tiny=True`` guarantees e >= 7: a block with e < 7
       is lowered to zero (mantissa 0, e = 7), exactly as pack_bfp16_gemv stores it.
    2. Zero prep step:
       Unlike q4nx which requires group_sums_32(x) for the minimum offset term,
       bfp16ebs8 weights have no offset (min = 0).
    3. Block accumulation:
       - Mantissa int8 (|m| <= 128) converts to bf16 exactly without rounding loss.
       - Scale factor 2^(e - 133) is an exact power of 2 in bf16: the kernel applies
         the defensive vector max(e, 7) and assembles (e - 6) << 7 (always normal
         bf16 because packing guaranteed e >= 7), then multiplies.
       - Product w = bf16(m) * bf16(scale) is exact in bf16.
       - Each weight multiplies input x (bf16) and accumulates into float32 (accfloat).
       - Zero intermediate narrowing loss occurs across the reduction!
    4. Epilogue:
       The accumulated float32 output is narrowed to bfloat16 using conv_even rounding.

    Args:
        weights: float32 weight matrix [M, K] (or pre-quantized weights).
        x: float32 or bfloat16 input vector [K].
        rounding: Quantization rounding mode ('conv_even' or 'floor').

    Returns:
        y_f32: float32 array [M] of rounded bfloat16 values.
        y_u16: uint16 array [M] of bfloat16 bit patterns.
    """
    w_f32 = np.asarray(weights, dtype=np.float32)
    M, K = w_f32.shape
    x_f32 = round_to_bf16_rne(np.asarray(x, dtype=np.float32).reshape(-1)[:K])

    # 1. Encode into bfp16ebs8 blocks along K and apply the packing guarantee
    # (pack_bfp16_gemv): a block with shared exponent e < 7 is stored as mantissa 0,
    # e = 7, so every packed block satisfies e >= 7.
    enc = bfp16ebs8_encode(w_f32, rounding=rounding)
    blk_view = enc.view(np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (8,))]))
    blk_view = blk_view.reshape(M, K // 8)
    exp = blk_view["exponent"]  # [M, K // 8]
    mant = blk_view["mantissas"]  # [M, K // 8, 8]
    tiny = exp < 7
    exp = np.maximum(exp, 7)
    mant = np.where(tiny[..., None], 0, mant)

    # 2. Dequantize to exact bf16 weights:
    # scale = 2^(e - 133) with the kernel's defensive max(e, 7) clamp and normal-bf16
    # bit assembly ((e - 6) << 7), via bf16_scale. exp is already >= 7 here (step 1),
    # so the clamp is a no-op on packed data but the same rule end to end.
    scale = bf16_scale(exp)  # [M, K // 8]
    scale_rep = np.repeat(scale, 8, axis=1)  # [M, K]
    mant_flat = mant.reshape(M, K).astype(np.float32)  # [M, K]
    w_exact = round_to_bf16_rne(mant_flat * scale_rep)  # exact in bf16

    # 3. Accumulate sequentially in float32 matching hardware loop order
    # (blocks of K_TILE, 8-element groups, 8 columns)
    y_acc = np.zeros(M, dtype=np.float32)
    for k in range(K):
        y_acc = (y_acc + w_exact[:, k] * x_f32[k]).astype(np.float32)

    # 4. Epilogue narrowing to bf16 via conv_even (RNE)
    y_f32 = round_to_bf16_rne(y_acc)
    y_u16 = to_bf16_u16(y_f32)
    return y_f32, y_u16
