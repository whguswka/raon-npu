# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""9-bit bfp16ebs8 quantization codec and GEMV memory layout packing.

Implements:
1. bfp16ebs8 codec matching mlir-aie python/utils/bfp.py and amd/IRON.
   Block = 1 exponent byte (uint8) + 8 mantissa bytes (int8) = 9 bytes.
   Last axis (in-feature K direction) is blocked in groups of 8.
2. GEMV weight packing: pack_bfp16_gemv / parse_bfp16_gemv
   Block = 32 rows x 256 cols (8192 B mantissa + 1024 B exponent = 9216 B, 4-byte aligned).
   Block ordering matches pack_gemv: ((slice * k_blocks + k) * rows + row).
   The public codec preserves every shared exponent, including e=0..6.  The NPU-only
   ``pack_bfp16_gemv`` layout can explicitly flush those blocks to zero so the kernel
   decodes every stored block with a normal bf16 scale.
   Internal block layout optimized for AIE2P vector loads:
   - 2 halves of 16 rows (half 0: rows 0..15, half 1: rows 16..31)
   - 32 groups of 8 columns per half:
     * 16 bytes: group exponent for 16 rows
     * 128 bytes: 8 columns x 16 rows column-major mantissas (16 bytes per column)
     * 144 bytes per group x 32 groups = 4608 bytes per half
     * 4608 x 2 halves = 9216 bytes per block.
3. Shape alignment and padding: pad_bfp16_gemv / pad (K aligned to 256, M aligned to 128).
"""

from __future__ import annotations

import warnings

import numpy as np

# ---------------------------------------------------------------------------
# bfp16ebs8 codec constants and data types
# ---------------------------------------------------------------------------
BLOCK = 8
_BLOCK_DTYPE = np.dtype([("exponent", np.uint8), ("mantissas", np.int8, (BLOCK,))])
BLOCK_BYTES = _BLOCK_DTYPE.itemsize  # 9 bytes
_MANTISSA_SHIFT = 23 - 7 + 1
# The GEMV kernel's bf16 scale assembly requires packed exponents >= 7.  This limit
# belongs to the NPU storage format, not to the public bfp16ebs8 codec.
E_MIN_EXPONENT = 7

# ---------------------------------------------------------------------------
# GEMV layout geometry constants
# ---------------------------------------------------------------------------
M_TILE = 32
K_TILE = 256
GROUP = 8  # bfp16 block size (K direction)
GROUPS_PER_TILE = K_TILE // GROUP  # 32
HALF_ROWS = 16

GEMV_EXP_BYTES_PER_BLOCK = M_TILE * GROUPS_PER_TILE  # 32 * 32 = 1024
GEMV_MANT_BYTES_PER_BLOCK = M_TILE * K_TILE  # 32 * 256 = 8192
GEMV_BLOCK_BYTES = GEMV_EXP_BYTES_PER_BLOCK + GEMV_MANT_BYTES_PER_BLOCK  # 9216

GROUP_BYTES = HALF_ROWS + HALF_ROWS * GROUP  # 16 + 128 = 144
HALF_BYTES = GROUPS_PER_TILE * GROUP_BYTES  # 32 * 144 = 4608
assert HALF_BYTES * 2 == GEMV_BLOCK_BYTES == 9216, "GEMV block geometry mismatch"


def packed_bytes(n_weights: int) -> int:
    """Bytes required to store `n_weights` in 9-bit bfp16ebs8 layout."""
    if n_weights % BLOCK != 0:
        raise ValueError(f"n_weights ({n_weights}) must be a multiple of {BLOCK}")
    return n_weights * 9 // 8


# ---------------------------------------------------------------------------
# Core bfp16ebs8 codec (ported from quant/eval_e2e.py / mlir-aie bfp.py)
# ---------------------------------------------------------------------------
def bfp16ebs8_encode(x: np.ndarray, *, rounding: str = "conv_even") -> np.ndarray:
    """Encode float32 array into bfp16ebs8 format along the last axis (input k direction).

    Args:
        x: float32 array with last dimension divisible by 8.
        rounding: Rounding mode for mantissa truncation ('conv_even' or 'floor').
                  Default is 'conv_even' (nearest-even) to prevent accuracy degradation.

    Returns:
        uint8 array of shape `(*x.shape[:-1], x.shape[-1] // 8 * 9)`.
    """
    if rounding not in ("floor", "conv_even"):
        raise ValueError(
            f"bfp.encode: rounding must be 'floor' or 'conv_even', got {rounding!r}"
        )
    x = np.ascontiguousarray(x, dtype=np.float32)
    n = x.shape[-1]
    if n % BLOCK != 0:
        raise ValueError(f"bfp.encode: last axis {n} is not a multiple of {BLOCK}")
    if not np.isfinite(x).all():
        raise ValueError("bfp.encode: inputs must be finite")
    lead = x.shape[:-1]
    bits = x.view(np.uint32).reshape(*lead, n // BLOCK, BLOCK)
    sign = (bits >> 31).astype(bool)
    exp = (bits >> 23) & 0xFF
    mant = (bits & 0x7FFFFF) | np.where(exp != 0, np.uint32(0x800000), np.uint32(0))
    max_exp = exp.max(axis=-1, keepdims=True)
    if rounding == "conv_even":
        blocks = x.reshape(*lead, n // BLOCK, BLOCK)
        scale = np.ldexp(1.0, max_exp.astype(np.int32) - 127 - 6)
        v = np.rint(blocks.astype(np.float64) / scale)
        carry = (v.max(axis=-1, keepdims=True) > 127)
        if carry.any():
            max_exp = max_exp + carry.astype(max_exp.dtype)
            scale = np.where(carry, scale * 2, scale)
            v = np.rint(blocks.astype(np.float64) / scale)
        v = np.clip(v, -128, 127)
        far = np.zeros_like(v, dtype=bool)
    else:
        shift = (max_exp - exp).astype(np.int64)
        far = shift >= 32
        m32 = np.where(sign, (-mant.astype(np.int64)) & 0xFFFFFFFF, mant).astype(
            np.uint32
        )
        v = (
            ((m32 >> _MANTISSA_SHIFT) & 0xFF)
            .astype(np.uint8)
            .view(np.int8)
            .astype(np.int32)
        )
        v = np.right_shift(v, np.minimum(shift, 31).astype(np.int32))
    out = np.empty(lead + (n // BLOCK,), dtype=_BLOCK_DTYPE)
    out["exponent"] = max_exp[..., 0]
    out["mantissas"] = np.where(far, np.where(sign, -1, 0), v)
    return out.view(np.uint8).reshape(*lead, n // BLOCK * BLOCK_BYTES)


def bfp16ebs8_decode(b: np.ndarray) -> np.ndarray:
    """Decode bfp16ebs8 byte array to float32 array along the last axis."""
    b = np.ascontiguousarray(b, dtype=np.uint8)
    nb = b.shape[-1]
    if nb % BLOCK_BYTES != 0:
        raise ValueError(
            f"bfp.decode: last axis {nb} is not a multiple of {BLOCK_BYTES}"
        )
    lead = b.shape[:-1]
    blk = b.view(_BLOCK_DTYPE)
    scale = np.ldexp(1.0, blk["exponent"].astype(np.int32) - 127 - 6)
    vals = blk["mantissas"].astype(np.float64) * scale[..., None]
    return vals.astype(np.float32).reshape(*lead, nb // BLOCK_BYTES * BLOCK)


def quantize_bfp16ebs8(
    x: np.ndarray, *, rounding: str = "conv_even", rows_per_chunk: int = 4096
) -> np.ndarray:
    """Round float32 array to bfp16ebs8, matching mlir-aie bfp.py quantize().

    bfp16 blocks are 8 consecutive elements along the last axis, so splitting rows
    never changes the result bit-for-bit. Flattening every axis before the last
    into rows and processing ``rows_per_chunk`` rows at a time keeps the encode/
    decode intermediates at chunk size instead of full-matrix size.

    This is the public codec operation and preserves shared exponents e=0..6 exactly
    like mlir-aie's ``bfp.py``.  NPU-specific tiny-block flushing is confined to
    ``pack_bfp16_gemv``.
    """
    if rounding not in ("floor", "conv_even"):
        raise ValueError(
            f"bfp.quantize: rounding must be 'floor' or 'conv_even', got {rounding!r}"
        )
    if rows_per_chunk < 1:
        raise ValueError(
            f"bfp.quantize: rows_per_chunk must be >= 1, got {rows_per_chunk}"
        )
    x = np.asarray(x)
    n = x.shape[-1]
    if n % BLOCK != 0:
        raise ValueError(f"bfp.quantize: last axis {n} is not a multiple of {BLOCK}")
    rows = x.reshape(-1, n)
    out = np.empty(rows.shape, dtype=np.float32)
    for start in range(0, rows.shape[0], rows_per_chunk):
        chunk = np.ascontiguousarray(rows[start : start + rows_per_chunk], np.float32)
        encoded = bfp16ebs8_encode(chunk, rounding=rounding)
        out[start : start + chunk.shape[0]] = bfp16ebs8_decode(encoded)
    return out.reshape(x.shape)


round_to_bfp16ebs8 = quantize_bfp16ebs8


# ---------------------------------------------------------------------------
# GEMV layout packing and parsing
# ---------------------------------------------------------------------------
def pad_bfp16_gemv(
    w: np.ndarray, m_align: int = 128, k_align: int = 256
) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad rows to multiples of m_align (128) and cols to multiples of k_align (256) with zeros.

    Args:
        w: 2D numpy array [M, K].
        m_align: Row padding alignment (default 128 = 4 rows * 32 M_TILE).
        k_align: Column padding alignment (default 256 = K_TILE).

    Returns:
        padded_w: Padded 2D array.
        orig_shape: Tuple of (orig_M, orig_K).
    """
    orig_m, orig_k = w.shape
    pad_m = (m_align - (orig_m % m_align)) % m_align
    pad_k = (k_align - (orig_k % k_align)) % k_align
    if pad_m == 0 and pad_k == 0:
        return w, (orig_m, orig_k)
    padded = np.zeros((orig_m + pad_m, orig_k + pad_k), dtype=w.dtype)
    padded[:orig_m, :orig_k] = w
    return padded, (orig_m, orig_k)


pad = pad_bfp16_gemv


def pack_bfp16_gemv(
    w_f32: np.ndarray,
    rows: int = 4,
    *,
    rounding: str = "conv_even",
    flush_tiny: bool = True,
    return_stats: bool = False,
) -> bytes | tuple[bytes, dict[str, int]]:
    """Pack float32 weight matrix into 9-bit bfp16ebs8 general GEMV layout.

    Layout structure:
    - Slice = rows * M_TILE (default 4 * 32 = 128) out-features
    - Block ordering: ((slice * k_blocks + k) * rows + row)
    - Block = 32 rows x 256 cols (9216 bytes, 4-byte aligned)
    - Inside each block:
      * 2 halves (half 0: rows 0..15, half 1: rows 16..31)
      * 32 groups (K // 8 = 32):
        - 16 bytes: uint8 exponents for 16 rows
        - 128 bytes: int8 mantissas for 8 cols x 16 rows (column-major)
        - 144 bytes per group x 32 groups = 4608 bytes per half
        - 4608 x 2 halves = 9216 bytes per block.

    With ``flush_tiny=True``, every stored exponent is >= E_MIN_EXPONENT (7). A block
    whose shared exponent is below 7 is stored as mantissa 0 and exponent 7. This is
    an NPU-format constraint; the public codec itself does not discard these values.

    Args:
        w_f32: float32 weight matrix [M, K].
        rows: Core rows (default 4).
        rounding: bfp16ebs8 rounding mode ('conv_even' or 'floor'). Default is 'conv_even'.
        flush_tiny: Apply the NPU exponent-floor rule (default True). False is useful
            only for inspecting the raw codec layout and is not kernel-compatible.
        return_stats: Return ``(packed_bytes, stats)`` instead of only bytes. Stats
            report candidate tiny blocks and nonzero blocks actually flushed.

    Returns:
        Packed bytes, or ``(packed bytes, stats)`` when ``return_stats=True``.
    """
    w_f32 = np.ascontiguousarray(w_f32, dtype=np.float32)
    M, K = w_f32.shape
    chunk_rows = rows * M_TILE
    if M % chunk_rows != 0:
        raise ValueError(
            f"M ({M}) must be a multiple of {chunk_rows} (rows={rows} * M_TILE={M_TILE})"
        )
    if K % K_TILE != 0:
        raise ValueError(f"K ({K}) must be a multiple of K_TILE ({K_TILE})")

    k_blocks = K // K_TILE
    slices = M // chunk_rows
    total_blocks = slices * k_blocks * rows

    # Encode entire matrix into bfp16ebs8 blocks along K
    # Shape of enc: (M, K // 8 * 9) -> view as _BLOCK_DTYPE gives (M, K // 8)
    enc = bfp16ebs8_encode(w_f32, rounding=rounding)
    blk_view = enc.view(_BLOCK_DTYPE).reshape(M, K // GROUP)
    exponents = blk_view["exponent"]  # uint8, shape [M, K // 8]
    mantissas = blk_view["mantissas"]  # int8, shape [M, K // 8, 8]

    tiny = exponents < E_MIN_EXPONENT
    tiny_nonzero = tiny & np.any(mantissas != 0, axis=-1)
    stats = {
        "tiny_blocks": int(np.count_nonzero(tiny)),
        "flushed_nonzero_blocks": int(np.count_nonzero(tiny_nonzero)) if flush_tiny else 0,
    }
    if flush_tiny:
        exponents = np.where(tiny, E_MIN_EXPONENT, exponents)
        mantissas = np.where(tiny[..., None], 0, mantissas)
        if stats["flushed_nonzero_blocks"] and not return_stats:
            warnings.warn(
                f"pack_bfp16_gemv flushed {stats['flushed_nonzero_blocks']} nonzero "
                "bfp16ebs8 blocks with exponent below 7",
                RuntimeWarning,
                stacklevel=2,
            )

    # Pre-allocate output buffer for all blocks
    out_buf = np.empty((total_blocks, GEMV_BLOCK_BYTES), dtype=np.uint8)

    # Process each block according to GEMV ordering
    block_idx = 0
    for s in range(slices):
        for kb in range(k_blocks):
            for r in range(rows):
                m_start = (s * rows + r) * M_TILE
                m_end = m_start + M_TILE
                k_start = kb * K_TILE
                k_end = k_start + K_TILE
                g_start = kb * GROUPS_PER_TILE
                g_end = g_start + GROUPS_PER_TILE

                # Extract 32 rows x 32 groups for this block
                b_exp = exponents[m_start:m_end, g_start:g_end]  # [32, 32]
                b_mant = mantissas[m_start:m_end, g_start:g_end, :]  # [32, 32, 8]

                # Destination buffer for this block (9216 bytes)
                dst = out_buf[block_idx]

                # Fill Half 0 (rows 0..15) and Half 1 (rows 16..31)
                for h in range(2):
                    h_m_start = h * HALF_ROWS
                    h_m_end = h_m_start + HALF_ROWS
                    half_exp = b_exp[h_m_start:h_m_end, :]  # [16, 32]
                    half_mant = b_mant[h_m_start:h_m_end, :, :]  # [16, 32, 8]

                    half_dst_offset = h * HALF_BYTES
                    for g in range(GROUPS_PER_TILE):
                        g_dst_offset = half_dst_offset + g * GROUP_BYTES
                        # 1) Exponent: 16 bytes (row 0..15 of group g)
                        dst[g_dst_offset : g_dst_offset + 16] = half_exp[:, g]

                        # 2) Mantissa: 128 bytes (8 columns x 16 rows, column-major)
                        # half_mant[:, g, :] is [16 rows, 8 cols] -> transpose to [8 cols, 16 rows]
                        col_major_mant = np.ascontiguousarray(half_mant[:, g, :].T)
                        dst[
                            g_dst_offset + 16 : g_dst_offset + 144
                        ] = col_major_mant.view(np.uint8).ravel()

                block_idx += 1

    packed = out_buf.tobytes()
    return (packed, stats) if return_stats else packed


def parse_bfp16_gemv_raw(
    w: np.ndarray | bytes,
    K: int,
    M: int,
    rows: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse packed GEMV buffer into (exponents, mantissas).

    Args:
        w: Packed GEMV buffer (bytes or uint8 array).
        K: In-features / hidden size.
        M: Out-features / output rows.
        rows: Core rows (default 4).

    Returns:
        exponents: uint8 array of shape [M, K // 8].
        mantissas: int8 array of shape [M, K].
    """
    buf = (
        np.frombuffer(w, dtype=np.uint8)
        if isinstance(w, (bytes, bytearray))
        else np.asarray(w, dtype=np.uint8)
    )
    chunk_rows = rows * M_TILE
    if M % chunk_rows != 0:
        raise ValueError(
            f"M ({M}) must be a multiple of {chunk_rows} (rows={rows} * M_TILE={M_TILE})"
        )
    if K % K_TILE != 0:
        raise ValueError(f"K ({K}) must be a multiple of K_TILE ({K_TILE})")

    k_blocks = K // K_TILE
    slices = M // chunk_rows
    total_blocks = slices * k_blocks * rows

    blocks = buf[: total_blocks * GEMV_BLOCK_BYTES].reshape(
        total_blocks, GEMV_BLOCK_BYTES
    )

    exponents = np.empty((M, K // GROUP), dtype=np.uint8)
    mantissas = np.empty((M, K), dtype=np.int8)

    block_idx = 0
    for s in range(slices):
        for kb in range(k_blocks):
            for r in range(rows):
                m_start = (s * rows + r) * M_TILE
                m_end = m_start + M_TILE
                k_start = kb * K_TILE
                g_start = kb * GROUPS_PER_TILE

                src = blocks[block_idx]

                for h in range(2):
                    h_m_start = m_start + h * HALF_ROWS
                    h_m_end = h_m_start + HALF_ROWS
                    half_src_offset = h * HALF_BYTES

                    for g in range(GROUPS_PER_TILE):
                        g_src_offset = half_src_offset + g * GROUP_BYTES
                        global_g = g_start + g

                        # Exponent (16 bytes)
                        exponents[h_m_start:h_m_end, global_g] = src[
                            g_src_offset : g_src_offset + 16
                        ]

                        # Mantissa (128 bytes = 8 cols x 16 rows)
                        raw_mant = src[g_src_offset + 16 : g_src_offset + 144].view(
                            np.int8
                        )
                        # reshaped as [8 cols, 16 rows], transpose back to [16 rows, 8 cols]
                        col_m = raw_mant.reshape(8, 16).T
                        col_start = k_start + g * 8
                        col_end = col_start + 8
                        mantissas[h_m_start:h_m_end, col_start:col_end] = col_m

                block_idx += 1

    return exponents, mantissas


def parse_bfp16_gemv(
    w: np.ndarray | bytes,
    K: int,
    M: int,
    rows: int = 4,
) -> np.ndarray:
    """Inverse of ``pack_bfp16_gemv``, restoring dequantized float32 weight matrix.

    Args:
        w: Packed GEMV buffer (bytes or uint8 array).
        K: In-features / hidden size.
        M: Out-features / output rows.
        rows: Core rows (default 4).

    Returns:
        float32 array of shape [M, K].
    """
    exponents, mantissas = parse_bfp16_gemv_raw(w, K, M, rows)
    # Compute scale = 2^(exponent - 127 - 6) = 2^(exponent - 133)
    scale = np.ldexp(1.0, exponents.astype(np.int32) - 127 - 6)  # [M, K // 8]
    # Expand scale across 8 columns
    scale_expanded = np.repeat(scale, 8, axis=1)  # [M, K]
    weights_f32 = (mantissas.astype(np.float64) * scale_expanded).astype(np.float32)
    return weights_f32
