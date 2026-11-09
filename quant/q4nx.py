# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""q4nx 4-bit weight conversion and packing utilities for AMD XDNA 2 NPU.

This module provides routines to quantize float32 matrices into q4nx format
(4-bit codes with bf16 scales and minima per 32-element group), pad them to
required tile alignments, and pack them into:
1. DecodeLayer projection layouts (matching IRON layer/reference.py pack_q4nx).
2. LMHead vocabulary layouts (matching IRON lm_head/reference.py dequantize).

Hardware execution target: AMD Ryzen AI 7 350 (XDNA 2 AIE2P NPU).
Host conversion runs on x86_64 CPU (CPU host server).
"""

from __future__ import annotations

import numpy as np

# q4nx tiling constants
M_TILE: int = 32
K_TILE: int = 256
GROUP: int = 32

BITS_PER_WEIGHT: int = 4 + 2 * 16 // GROUP  # 5 bits
BLOCK_BYTES: int = M_TILE * K_TILE * BITS_PER_WEIGHT // 8  # 5120 bytes
_SM_BYTES: int = 2 * 2 * M_TILE * K_TILE // GROUP  # 1024 bytes


def to_bf16_u16(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16 bit patterns (uint16) using Round-to-Nearest-Even (RNE).

    Does not depend on ml_dtypes or PyTorch, running purely with numpy bit manipulation.
    """
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    isnan = np.isnan(x)
    bias = 0x7FFF + ((u >> 16) & 1)
    u_rounded = (u + bias) >> 16
    res = u_rounded.astype(np.uint16)
    if np.any(isnan):
        res[isnan] = 0x7FC0
    return res


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    """Convert bfloat16 uint16 bit patterns to float32 values."""
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


def round_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Round float32 array to values representable in bfloat16 using RNE."""
    return bf16_to_f32(to_bf16_u16(x))


def bf16_floor(x: np.ndarray | float) -> np.ndarray | float:
    """Round float values towards -infinity to greatest representable bfloat16."""
    x_arr = np.asarray(x)
    is_scalar = x_arr.ndim == 0
    if x_arr.dtype == np.float64:
        f32 = x_arr.astype(np.float32)
        need_step = f32.astype(np.float64) > x_arr
        f32 = np.where(need_step, np.nextafter(f32, np.float32(-np.inf)), f32)
    else:
        f32 = np.ascontiguousarray(x_arr, dtype=np.float32)

    f32_1d = np.atleast_1d(f32)
    u = f32_1d.view(np.uint32)
    isnan = np.isnan(f32_1d)
    sign = u >> 31
    lower = u & 0xFFFF
    shift = u >> 16
    inc = np.where((sign == 1) & (lower != 0), 1, 0).astype(np.uint32)
    res_u16 = (shift + inc).astype(np.uint16)
    res = (res_u16.astype(np.uint32) << 16).view(np.float32)
    if np.any(isnan):
        res[isnan] = np.nan
    res = res.reshape(f32.shape)
    return res.item() if is_scalar else res


def bf16_ceil(x: np.ndarray | float) -> np.ndarray | float:
    """Round float values towards +infinity to smallest representable bfloat16."""
    x_arr = np.asarray(x)
    is_scalar = x_arr.ndim == 0
    if x_arr.dtype == np.float64:
        f32 = x_arr.astype(np.float32)
        need_step = f32.astype(np.float64) < x_arr
        f32 = np.where(need_step, np.nextafter(f32, np.float32(np.inf)), f32)
    else:
        f32 = np.ascontiguousarray(x_arr, dtype=np.float32)

    f32_1d = np.atleast_1d(f32)
    u = f32_1d.view(np.uint32)
    isnan = np.isnan(f32_1d)
    sign = u >> 31
    lower = u & 0xFFFF
    shift = u >> 16
    inc = np.where((sign == 0) & (lower != 0), 1, 0).astype(np.uint32)
    res_u16 = (shift + inc).astype(np.uint16)
    res = (res_u16.astype(np.uint32) << 16).view(np.float32)
    if np.any(isnan):
        res[isnan] = np.nan
    res = res.reshape(f32.shape)
    return res.item() if is_scalar else res


def bf16_nextafter(x: np.ndarray | float) -> np.ndarray | float:
    """Step an exact bfloat16 value to the next representable bfloat16 towards +infinity."""
    x_arr = np.asarray(x, dtype=np.float32)
    is_scalar = x_arr.ndim == 0
    x_1d = np.atleast_1d(x_arr)
    u = x_1d.view(np.uint32)
    u16 = (u >> 16).astype(np.int32)
    u16 = np.where(u16 == 0x8000, 0, u16)
    is_neg = (u16 & 0x8000) != 0
    next_u16 = np.where(
        u16 == 0,
        0x0001,
        np.where(is_neg, np.where(u16 == 0x8001, 0, u16 - 1), u16 + 1),
    ).astype(np.uint16)
    res = (next_u16.astype(np.uint32) << 16).view(np.float32).reshape(x_arr.shape)
    return res.item() if is_scalar else res


def quantize(
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantize a float32 weight matrix into q4nx format.

    Args:
        w: float32 matrix of shape [N_out, K_in]. K_in must be a multiple of 32 (GROUP).

    Returns:
        codes: uint8 array of shape [N_out, K_in], values in [0, 15].
        scales: float32 array of shape [N_out, K_in / 32] containing bf16-representable scale values.
        mins: float32 array of shape [N_out, K_in / 32] containing bf16-representable minimum values.

    De-quantization formula: ``min + scale * code``.
    Parameters are selected so that the bf16 grid covers the full group range:
    - min_q is the greatest bf16 <= min (bf16_floor).
    - scale_q is the smallest bf16 >= (max - min_q) / 15 (bf16_ceil).
    - If min_q + 15 * scale_q < max in float64, scale_q is stepped to the next bf16.
    - Constant groups where max == min_q have scale_q = 0.0 and codes = 0.
    All calculations (min, max, differences, and division) are performed in float64
    to prevent intermediate float32 overflow.
    """
    w_arr = np.asarray(w)
    if w_arr.ndim != 2:
        raise ValueError(f"Weight matrix must be 2D, got shape {w_arr.shape}")

    if not np.all(np.isfinite(w_arr)):
        raise ValueError("Input weight matrix contains non-finite values (NaN or Inf)")

    w_f32 = w_arr.astype(np.float32)
    n_rows, n_cols = w_f32.shape
    if n_cols % GROUP != 0:
        raise ValueError(
            f"Columns/in-features ({n_cols}) must be a multiple of GROUP ({GROUP})"
        )

    groups = n_cols // GROUP
    wg = w_f32.reshape(n_rows, groups, GROUP)

    w_min = np.min(wg, axis=-1)
    w_max = np.max(wg, axis=-1)

    min_q = bf16_floor(w_min)

    w_max_f64 = w_max.astype(np.float64)
    min_q_f64 = min_q.astype(np.float64)
    diff_f64 = w_max_f64 - min_q_f64

    # scale_q: smallest bf16 >= diff / 15.0
    scale_q = bf16_ceil(diff_f64 / 15.0)

    # Ensure min_q + 15 * scale_q >= max in float64
    covered = (min_q_f64 + 15.0 * scale_q.astype(np.float64)) >= w_max_f64
    while not np.all(covered):
        if not np.all(np.isfinite(scale_q)):
            break
        scale_q = np.where(covered, scale_q, bf16_nextafter(scale_q))
        covered = (min_q_f64 + 15.0 * scale_q.astype(np.float64)) >= w_max_f64

    # If max == min_q (exact bf16 constant group), scale is 0.0
    is_zero_scale = (diff_f64 == 0.0)
    scale_q[is_zero_scale] = 0.0

    if not np.all(np.isfinite(min_q)) or not np.all(np.isfinite(scale_q)):
        raise ValueError(
            "Quantization parameters (scale_q or min_q) overflowed to non-finite values"
        )

    safe_scales = np.where(is_zero_scale, 1.0, scale_q.astype(np.float64))
    norm = (wg.astype(np.float64) - min_q_f64[..., None]) / safe_scales[..., None]
    codes = np.clip(np.rint(norm), 0, 15).astype(np.uint8)
    codes = np.where(is_zero_scale[..., None], 0, codes)

    return codes.reshape(n_rows, n_cols), scale_q.astype(np.float32), min_q.astype(np.float32)


def dequantize_simple(
    codes: np.ndarray, scales: np.ndarray, mins: np.ndarray
) -> np.ndarray:
    """Dequantize q4nx components in natural order using ``min + scale * code``.

    Args:
        codes: uint8 array [N, K].
        scales: float32 array [N, K / 32].
        mins: float32 array [N, K / 32].

    Returns:
        Dequantized float32 matrix of shape [N, K].
    """
    n_rows, n_cols = codes.shape
    groups = n_cols // GROUP
    k_group = np.arange(n_cols) // GROUP
    res = (
        mins[:, k_group].astype(np.float64)
        + scales[:, k_group].astype(np.float64) * codes.astype(np.float64)
    )
    return res.astype(np.float32)


def pad(
    w: np.ndarray, m_tile: int = 32, k_tile: int = 256
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad rows to multiples of m_tile and cols to multiples of k_tile with zeros.

    Args:
        w: 2D numpy array [rows, cols].
        m_tile: Row padding alignment (default 32, LMHead uses 1024).
        k_tile: Column padding alignment (default 256).

    Returns:
        padded_w: Padded 2D array.
        orig_shape: Tuple of (original_rows, original_cols).
    """
    if w.ndim != 2:
        raise ValueError(f"pad requires 2D array, got shape {w.shape}")
    rows, cols = w.shape
    pad_r = (m_tile - (rows % m_tile)) % m_tile
    pad_c = (k_tile - (cols % k_tile)) % k_tile
    if pad_r == 0 and pad_c == 0:
        return w.copy(), (rows, cols)
    padded = np.pad(w, ((0, pad_r), (0, pad_c)), mode="constant", constant_values=0)
    return padded, (rows, cols)


def unpad(padded: np.ndarray, orig_shape: tuple[int, int]) -> np.ndarray:
    """Slice a padded array back to its original dimensions."""
    orig_r, orig_c = orig_shape
    return padded[:orig_r, :orig_c]


def pack_decode_proj(
    codes: np.ndarray, scales: np.ndarray, mins: np.ndarray
) -> bytes:
    """Pack q4nx components into DecodeLayer projection layout.

    Produces byte-identical output to IRON ``layer/reference.py:pack_q4nx``.
    Layout structure:
    - Block holds 32 rows x 256 columns in 5120 bytes (1024 bytes sm + 4096 bytes codes).
    - Blocks are arranged by pairs of 32-row bands (stepping N by 64).
    - Rows must be a multiple of 64 (2 * M_TILE), cols a multiple of 256 (K_TILE).

    Args:
        codes: uint8 array of shape [rows, cols].
        scales: float32 array of shape [rows, cols // 32].
        mins: float32 array of shape [rows, cols // 32].

    Returns:
        bytes: Packed projection buffer.
    """
    rows, cols = codes.shape
    if rows % (2 * M_TILE) != 0:
        raise ValueError(
            f"rows ({rows}) must be a multiple of {2 * M_TILE} (64) for decode projection packing"
        )
    if cols % K_TILE != 0:
        raise ValueError(f"cols ({cols}) must be a multiple of {K_TILE} (256)")

    nb_r, nb_c = rows // M_TILE, cols // K_TILE
    sm = np.stack([scales, mins]).reshape(2, nb_r, M_TILE, nb_c, K_TILE // GROUP)
    sm = sm.transpose(1, 3, 0, 4, 2).reshape(nb_r, nb_c, _SM_BYTES // 2)
    sm = to_bf16_u16(sm).view(np.uint8).reshape(nb_r, nb_c, _SM_BYTES)

    c = (
        codes.reshape(nb_r, 2, 16, nb_c, K_TILE)
        .transpose(0, 3, 1, 4, 2)
        .reshape(nb_r, nb_c, M_TILE * K_TILE)
    )
    qs = (c[..., 0::2] | (c[..., 1::2] << 4)).astype(np.uint8)

    blk = np.concatenate([sm, qs], -1).reshape(nb_r // 2, 2, nb_c, BLOCK_BYTES)
    return blk.transpose(0, 2, 1, 3).reshape(-1).tobytes()


def parse_decode_proj(
    proj: np.ndarray | bytes, offset: int, rows: int, cols: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of ``pack_decode_proj``, matching IRON ``layer/reference.py:parse_q4nx``.

    Args:
        proj: Packed uint8 buffer (or bytes).
        offset: Byte offset into proj.
        rows: Output features (must be multiple of 64).
        cols: In features (must be multiple of 256).

    Returns:
        codes: uint8 array [rows, cols].
        scales: float32 array [rows, cols // 32].
        mins: float32 array [rows, cols // 32].
    """
    proj_u8 = np.frombuffer(proj, dtype=np.uint8) if isinstance(proj, (bytes, bytearray)) else np.asarray(proj, dtype=np.uint8)
    nb_r, nb_c = rows // M_TILE, cols // K_TILE
    blk = proj_u8[offset : offset + nb_r * nb_c * BLOCK_BYTES]
    blk = blk.reshape(nb_r // 2, nb_c, 2, BLOCK_BYTES).transpose(0, 2, 1, 3)
    blk = blk.reshape(nb_r, nb_c, BLOCK_BYTES)

    sm = blk[..., :_SM_BYTES].copy().view("<u2")
    sm = bf16_to_f32(sm.reshape(nb_r, nb_c, 2, K_TILE // GROUP, M_TILE))
    sm = sm.transpose(2, 0, 4, 1, 3).reshape(2, rows, cols // GROUP)

    qs = blk[..., _SM_BYTES:]
    codes = np.empty(qs.shape[:-1] + (M_TILE * K_TILE,), np.uint8)
    codes[..., 0::2] = qs & 0x0F
    codes[..., 1::2] = qs >> 4
    codes = codes.reshape(nb_r, nb_c, 2, K_TILE, 16).transpose(0, 2, 4, 1, 3)

    return codes.reshape(rows, cols), sm[0], sm[1]


def pack_gemv(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    rows: int = 4,
) -> bytes:
    """Pack q4nx matrix into general GEMV layout.

    Layout matches pack_lm_head slice structure:
    - Slice = rows * M_TILE (default 4 * 32 = 128) out-features
    - Block ordering: ((slice * k_blocks + k) * rows + row)
    - M must be a multiple of rows * M_TILE (128)
    - K must be a multiple of K_TILE (256)

    Args:
        codes: uint8 array [M, K].
        scales: float32 array [M, K // 32].
        mins: float32 array [M, K // 32].
        rows: Core rows (default 4).

    Returns:
        bytes: Packed GEMV buffer.
    """
    M, K = codes.shape
    chunk_rows = rows * M_TILE
    if M % chunk_rows != 0:
        raise ValueError(
            f"M ({M}) must be a multiple of {chunk_rows} "
            f"(rows={rows} * M_TILE={M_TILE})"
        )
    if K % K_TILE != 0:
        raise ValueError(f"K ({K}) must be a multiple of K_TILE ({K_TILE})")

    k_blocks = K // K_TILE
    groups = K_TILE // GROUP
    slices = M // (rows * M_TILE)

    s_u16 = to_bf16_u16(scales)
    m_u16 = to_bf16_u16(mins)

    # Transpose scales and mins:
    # Within block: scales.T is [groups, M_TILE] = [8, 32]
    # Block order: (slices, k_blocks, rows)
    s_reshaped = s_u16.reshape(slices, rows, M_TILE, k_blocks, groups).transpose(0, 3, 1, 4, 2)
    m_reshaped = m_u16.reshape(slices, rows, M_TILE, k_blocks, groups).transpose(0, 3, 1, 4, 2)

    sm = np.stack([s_reshaped, m_reshaped], axis=3)  # (slices, k_blocks, rows, 2, groups, M_TILE)
    sm_bytes = sm.reshape(slices * k_blocks * rows, -1).view(np.uint8)

    # Codes internal transposition matching dequantize_block:
    # codes.reshape(M_TILE // 16, K_TILE // 32, 32, 16).transpose(0, 3, 1, 2)
    # Inverse: transpose(0, 2, 3, 1)
    c_blk = codes.reshape(slices, rows, 2, 16, k_blocks, K_TILE // 32, 32)
    c_blk = c_blk.transpose(0, 4, 1, 2, 5, 6, 3)  # (slices, k_blocks, rows, 2, 8, 32, 16)
    c_blk = c_blk.reshape(slices * k_blocks * rows, M_TILE * K_TILE)
    qs = (c_blk[:, 0::2] | (c_blk[:, 1::2] << 4)).astype(np.uint8)

    all_blocks = np.concatenate([sm_bytes, qs], axis=1)  # (total_blocks, BLOCK_BYTES)
    return all_blocks.reshape(-1).tobytes()


def parse_gemv(
    w: np.ndarray | bytes,
    K: int,
    M: int,
    rows: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of ``pack_gemv``, extracting codes, scales, and mins.

    Args:
        w: Packed GEMV buffer (bytes or uint8 array).
        K: In-features / hidden size.
        M: Out-features / output rows.
        rows: Core rows (default 4).

    Returns:
        codes: uint8 array [M, K].
        scales: float32 array [M, K // 32].
        mins: float32 array [M, K // 32].
    """
    buf = np.frombuffer(w, dtype=np.uint8) if isinstance(w, (bytes, bytearray)) else np.asarray(w, dtype=np.uint8)
    k_blocks = K // K_TILE
    groups = K_TILE // GROUP
    slices = M // (rows * M_TILE)
    total_blocks = slices * k_blocks * rows

    blocks = buf[: total_blocks * BLOCK_BYTES].reshape(slices, k_blocks, rows, BLOCK_BYTES)

    # Extract sm
    sm_bytes = blocks[..., :_SM_BYTES].copy().view("<u2")
    params = bf16_to_f32(sm_bytes.reshape(slices, k_blocks, rows, 2, groups, M_TILE))
    s_part = params[:, :, :, 0, :, :].transpose(0, 2, 4, 1, 3).reshape(M, K // GROUP)
    m_part = params[:, :, :, 1, :, :].transpose(0, 2, 4, 1, 3).reshape(M, K // GROUP)

    # Extract codes
    qs = blocks[..., _SM_BYTES:]
    codes_packed = np.empty((slices, k_blocks, rows, M_TILE * K_TILE), dtype=np.uint8)
    codes_packed[..., 0::2] = qs & 0x0F
    codes_packed[..., 1::2] = qs >> 4

    c_blk = codes_packed.reshape(slices, k_blocks, rows, 2, groups, 32, 16)
    codes_res = c_blk.transpose(0, 2, 3, 6, 1, 4, 5).reshape(M, K)

    return codes_res, s_part, m_part


def pad_gemv(
    w: np.ndarray, m_align: int = 128, k_align: int = 256
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad rows to multiples of m_align (128) and cols to multiples of k_align (256) with zeros.

    Args:
        w: 2D numpy array [M, K].
        m_align: Row padding alignment (default 128 = 4 * 32).
        k_align: Column padding alignment (default 256).

    Returns:
        padded_w: Padded 2D array.
        orig_shape: Tuple of (orig_M, orig_K).
    """
    return pad(w, m_tile=m_align, k_tile=k_align)


def pack_lm_head(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    cols: int = 8,
    rows: int = 4,
) -> bytes:
    """Pack q4nx vocabulary matrix into LMHead vocabulary layout.

    Follows the block and indexing order documented in IRON
    ``lm_head/reference.py:dequantize``.
    - Total out-features per round: ``cols * rows * M_TILE`` (default 8*4*32 = 1024).
    - Block index: ``((round * cols + col) * k_blocks + k) * rows + row``.
    - Block internal: scales (256 bf16), mins (256 bf16), then 4096 bytes packed nibbles.

    Args:
        codes: uint8 array [vocab, dim].
        scales: float32 array [vocab, dim // 32].
        mins: float32 array [vocab, dim // 32].
        cols: Core columns (default 8).
        rows: Core rows (default 4).

    Returns:
        bytes: Packed LMHead vocabulary buffer.
    """
    vocab, dim = codes.shape
    vocab_round = cols * rows * M_TILE
    if vocab % vocab_round != 0:
        raise ValueError(
            f"vocab ({vocab}) must be a multiple of {vocab_round} "
            f"(cols={cols} * rows={rows} * M_TILE={M_TILE})"
        )
    return pack_gemv(codes, scales, mins, rows=rows)


def parse_lm_head(
    w: np.ndarray | bytes,
    dim: int,
    vocab: int,
    cols: int = 8,
    rows: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of ``pack_lm_head``, extracting codes, scales, and mins.

    Rejects the same shapes ``pack_lm_head`` rejects: ``vocab`` must be a
    multiple of ``cols * rows * M_TILE`` (default 8*4*32 = 1024) and ``dim``
    a multiple of ``K_TILE`` — the LMHead layout is defined per full round.

    Args:
        w: Packed LMHead buffer (bytes or uint8 array).
        dim: In-features / hidden size.
        vocab: Vocab size / out-features (must be a multiple of cols*rows*M_TILE).
        cols: Core columns (default 8).
        rows: Core rows (default 4).

    Returns:
        codes: uint8 array [vocab, dim].
        scales: float32 array [vocab, dim // 32].
        mins: float32 array [vocab, dim // 32].
    """
    vocab_round = cols * rows * M_TILE
    if vocab % vocab_round != 0:
        raise ValueError(
            f"vocab ({vocab}) must be a multiple of {vocab_round} "
            f"(cols={cols} * rows={rows} * M_TILE={M_TILE})"
        )
    if dim % K_TILE != 0:
        raise ValueError(f"dim ({dim}) must be a multiple of K_TILE ({K_TILE})")
    return parse_gemv(w, K=dim, M=vocab, rows=rows)


def mask_padded_logits(
    logits: np.ndarray, true_vocab: int, inplace: bool = False
) -> np.ndarray:
    """Mask padded vocabulary logits by setting logits[..., true_vocab:] to -inf.

    Args:
        logits: Array of shape [..., vocab] (1D, 2D, or higher).
        true_vocab: Number of valid vocabulary tokens (e.g. 259,840 for Raon).
        inplace: Whether to modify logits in place.

    Returns:
        Masked logits array with padded entries set to -infinity.
    """
    if not inplace:
        logits = logits.copy()
    logits[..., true_vocab:] = -np.inf
    return logits
