# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""CPU reference for Raon general q4nx GEMV, in float64 and bf16."""

from __future__ import annotations

import numpy as np

try:
    from iron.operators.flm.q4nx import (
        BLOCK_BYTES,
        GROUP,
        K_TILE,
        M_TILE,
        bf16_to_f32,
        round_to_bf16_rne,
        to_bf16_u16,
    )
except (ImportError, ModuleNotFoundError):
    from quant.q4nx import (
        BLOCK_BYTES,
        GROUP,
        K_TILE,
        M_TILE,
        bf16_to_f32,
        round_to_bf16_rne,
        to_bf16_u16,
    )


def dequantize_block(block: bytes | np.ndarray) -> np.ndarray:
    """One q4nx block to its (M_TILE, K_TILE) float64 weights."""
    groups = K_TILE // GROUP
    block_bytes = bytes(block) if isinstance(block, (np.ndarray, memoryview)) else block
    params = bf16_to_f32(np.frombuffer(block_bytes[: 4 * groups * M_TILE], "<u2"))
    scales, mins = params.reshape(2, groups, M_TILE).astype(np.float64)
    packed = np.frombuffer(block_bytes[4 * groups * M_TILE :], np.uint8)
    codes = np.empty(M_TILE * K_TILE, np.float64)
    codes[0::2], codes[1::2] = packed & 15, packed >> 4
    codes = codes.reshape(M_TILE // 16, K_TILE // 32, 32, 16)
    codes = codes.transpose(0, 3, 1, 2).reshape(M_TILE, K_TILE)
    k_group = np.arange(K_TILE) // GROUP
    return mins[k_group].T + scales[k_group].T * codes


def dequantize(w: bytes | np.ndarray, K: int, M: int, rows: int = 4) -> np.ndarray:
    """Dequantize whole packed q4nx GEMV buffer to [M, K] float64 weights.

    Matches pack_lm_head and pack_gemv slice layout:
    - Slice = rows * M_TILE (default 4 * 32 = 128) out-features
    - Block ordering: ((slice * k_blocks + k) * rows + row)
    """
    w_bytes = bytes(w) if isinstance(w, (np.ndarray, memoryview)) else w
    k_blocks = K // K_TILE
    total_blocks = (M // M_TILE) * k_blocks
    out = np.empty((M, K), dtype=np.float64)

    for index in range(total_blocks):
        rows_k, row = divmod(index, rows)
        slice_, k = divmod(rows_k, k_blocks)
        n0 = (slice_ * rows + row) * M_TILE
        block = w_bytes[index * BLOCK_BYTES : (index + 1) * BLOCK_BYTES]
        out[n0 : n0 + M_TILE, k * K_TILE : (k + 1) * K_TILE] = dequantize_block(block)
    return out


def reference(weights: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pure linear matrix-vector product for Raon general GEMV (without RMS norm).

    ``weights`` is [M, K] float64 or float32 matrix.
    ``x`` is [K] input vector.
    Returns [M] float64 output.
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


def gemv_bit_reference(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    x: np.ndarray,
    group: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Model of the Raon general q4nx GEMV as the AIE2P kernel accumulates it.

    Hardware pipeline modeled (q4nx_gemv.cc / flm_gemma4 q4nx.h ``q4nx_accumulate``):
    1. q4nx_gemv_prep -> group_sums_32 (utils.h): ``S[g] = reduce_add(x[32g..32g+31])``
       narrowed to bf16 (conv_even). ``aie::reduce_add``'s internal add order is
       NOT defined by the kernel source, so this model uses a single exact sum
       narrowed once.
    2. q4nx_gemv_block -> q4nx_accumulate: for each output row and group, the
       32 columns are accumulated SEQUENTIALLY from column 0 in the fully-unrolled
       MAC chain — first term a multiply, then 31 MACs, each step rounded to
       float32 by the accumulator:
         ``acc = f32(code*x)`` then ``acc = f32(acc + f32(code*x))`` (c = 1..31)
       ``D = bf16_rne(acc)`` (``temp_acc.to_vector<bf16>()`` under conv_even).
       The output accumulator follows the block/group loop order (blocks in k
       order, groups 0..7 inside each block == ascending global group index g):
         ``y = f32(y + bf16(D) * scale)`` then ``y = f32(y + min * bf16(S))``.
    3. q4nx_gemv_epilogue: ``y = bf16_rne(y_acc)`` (narrow_to_bf16, conv_even).

    What is bit-exact vs. modeled:
    - For INTEGER samples (small-integer x/codes/scales/mins, the AMD contract
      sample) every intermediate sum is exactly representable, so all orders
      agree and this model is bit-exact against the NPU output.
    - For NON-INTEGER samples the hardware add order inside ``aie::reduce_add``
      (S) and the hardware accumulator chain are assumptions, not verified
      facts; bit differences there are reported as information only
      (see bench/gemv README section 4).

    Args:
        codes: uint8 array [M, K].
        scales: float32 array [M, K // group].
        mins: float32 array [M, K // group].
        x: float32 or bfloat16 array [K].
        group: Column group size (default 32).

    Returns:
        y_f32: float32 array [M] of the rounded bfloat16 output values.
        y_u16: uint16 array [M] of the bfloat16 bit patterns.
    """
    M, K = codes.shape
    n_groups = K // group
    x_f32 = np.asarray(x, dtype=np.float32).reshape(-1)[:K]
    x_grp = x_f32.reshape(n_groups, group)

    # 1. S: one exact sum per group narrowed once to bf16 via conv_even.
    #    (Order of aie::reduce_add is unspecified; see docstring — exact only
    #    when every partial sum is exactly representable, i.e. integer samples.)
    S_f32 = round_to_bf16_rne(x_grp.astype(np.float64).sum(axis=1))

    # 2. D and y: sequential in kernel order (see docstring).
    codes_f32 = codes.astype(np.float32)
    scales_f32 = np.asarray(scales, dtype=np.float32)
    mins_f32 = np.asarray(mins, dtype=np.float32)

    y = np.zeros(M, dtype=np.float32)
    for g in range(n_groups):
        # D: sequential float32 accumulation of the 32 columns in ascending
        # column order, exactly like the kernel's unrolled MAC chain — the
        # first term is a multiply, the remaining 31 are add-mul MACs.
        cols = slice(g * group, (g + 1) * group)
        prod = codes_f32[:, cols] * x_grp[g]  # exact: bf16 x uint4 in float32
        acc = prod[:, 0].astype(np.float32)
        for c in range(1, group):
            acc = (acc + prod[:, c]).astype(np.float32)
        # D narrowed to bf16 by the conv_even narrowing of temp_acc.
        d_bf16 = round_to_bf16_rne(acc)
        # c_accum = mac(c_accum, bf16(D), scale); then mac(c_accum, mins, S).
        y = (y + d_bf16 * scales_f32[:, g]).astype(np.float32)
        y = (y + mins_f32[:, g] * S_f32[g]).astype(np.float32)

    # 3. q4nx_gemv_epilogue: narrow to bf16 via conv_even (RNE).
    y_f32 = round_to_bf16_rne(y)
    y_u16 = to_bf16_u16(y)
    return y_f32, y_u16
