# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Weight packing and buffer organization for Raon-1-mini MoE layers.

This module provides routines to pack a full layer of 129 experts
(128 routed experts e=0..127 + 1 shared expert e=128) into two contiguous
NPU-ready buffers:
1. gate_up buffer: [gate; up] matrices (2176 rows x 2560 cols per expert)
   packed via `pack_gemv` and concatenated in expert order e=0..128.
   Expert unit size: GU_WORDS (870,400 uint32 words = 3,481,600 bytes).
2. down buffer: down matrix (2560 rows x 1088 cols padded to 1280 cols per expert)
   packed via `pack_gemv` and concatenated in expert order e=0..128.
   Expert unit size: DN_WORDS (512,000 uint32 words = 2,048,000 bytes).

Hardware target: AMD Ryzen AI 7 350 (XDNA 2 AIE2P NPU).
Host conversion runs on x86_64 CPU.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from quant.q4nx import (
    BLOCK_BYTES,
    GROUP,
    K_TILE,
    M_TILE,
    pack_gemv,
    pad_gemv,
    parse_gemv,
    quantize,
)
from quant.bfp16 import pack_bfp16_gemv, packed_bytes, parse_bfp16_gemv

# -----------------------------------------------------------------------------
# MoE Architectural Dimensions
# -----------------------------------------------------------------------------
NUM_ROUTED_EXPERTS: int = 128
SHARED_EXPERT_ID: int = 128
NUM_TOTAL_EXPERTS: int = NUM_ROUTED_EXPERTS + 1  # 129 (0..128)
NUM_ACTIVE_ROUTED: int = 6
NUM_ACTIVE_TOTAL: int = NUM_ACTIVE_ROUTED + 1  # 7 (e0..e5 + shared 128)

HIDDEN_SIZE: int = 2560
MOE_INTERMEDIATE_SIZE: int = 1088
CHUNK_ROWS: int = 4 * M_TILE  # 128 rows per chunk

# -----------------------------------------------------------------------------
# B1 (gate+up) Dimensions and Constants
# -----------------------------------------------------------------------------
B1_K: int = HIDDEN_SIZE  # 2560
B1_M_PER_EXPERT: int = 2 * MOE_INTERMEDIATE_SIZE  # 2176 (1088 gate + 1088 up)
B1_CHUNKS_PER_EXPERT: int = B1_M_PER_EXPERT // CHUNK_ROWS  # 17 chunks
B1_W_PER_CHUNK_BYTES: int = CHUNK_ROWS * B1_K * 5 // 8  # 204,800 bytes (0x32000)
B1_W_PER_CHUNK_WORDS: int = B1_W_PER_CHUNK_BYTES // 4  # 51,200 uint32 words
GU_WORDS: int = B1_CHUNKS_PER_EXPERT * B1_W_PER_CHUNK_WORDS  # 870,400 uint32 words
GU_BYTES: int = GU_WORDS * 4  # 3,481,600 bytes (0x352000)
TOTAL_GU_WORDS: int = NUM_TOTAL_EXPERTS * GU_WORDS  # 112,281,600 uint32 words
TOTAL_GU_BYTES: int = NUM_TOTAL_EXPERTS * GU_BYTES  # 449,126,400 bytes
ROUTED_GU_WORDS: int = NUM_ROUTED_EXPERTS * GU_WORDS  # 111,411,200 uint32 words
ROUTED_GU_BYTES: int = NUM_ROUTED_EXPERTS * GU_BYTES  # 445,644,800 bytes

# -----------------------------------------------------------------------------
# B2 (down) Dimensions and Constants
# -----------------------------------------------------------------------------
B2_K_ORIG: int = MOE_INTERMEDIATE_SIZE  # 1088
B2_K_PAD: int = ((B2_K_ORIG + K_TILE - 1) // K_TILE) * K_TILE  # 1280
B2_M_PER_EXPERT: int = HIDDEN_SIZE  # 2560
B2_CHUNKS_PER_EXPERT: int = B2_M_PER_EXPERT // CHUNK_ROWS  # 20 chunks
B2_W_PER_CHUNK_BYTES: int = CHUNK_ROWS * B2_K_PAD * 5 // 8  # 102,400 bytes (0x19000)
B2_W_PER_CHUNK_WORDS: int = B2_W_PER_CHUNK_BYTES // 4  # 25,600 uint32 words
DN_WORDS: int = B2_CHUNKS_PER_EXPERT * B2_W_PER_CHUNK_WORDS  # 512,000 uint32 words
DN_BYTES: int = DN_WORDS * 4  # 2,048,000 bytes (0x1f4000)
TOTAL_DN_WORDS: int = NUM_TOTAL_EXPERTS * DN_WORDS  # 66,048,000 uint32 words
TOTAL_DN_BYTES: int = NUM_TOTAL_EXPERTS * DN_BYTES  # 264,192,000 bytes

# -----------------------------------------------------------------------------
# B2 9-bit bfp16ebs8 Dimensions and Constants (D mixed_full mode)
# -----------------------------------------------------------------------------
B2_BFP16_W_PER_CHUNK_BYTES: int = packed_bytes(CHUNK_ROWS * B2_K_PAD)  # 184,320
B2_BFP16_W_PER_CHUNK_WORDS: int = B2_BFP16_W_PER_CHUNK_BYTES // 4  # 46,080
DN9_BYTES: int = packed_bytes(B2_M_PER_EXPERT * B2_K_PAD)  # 3,686,400
DN9_WORDS: int = DN9_BYTES // 4  # 921,600
TOTAL_DN9_BYTES: int = NUM_TOTAL_EXPERTS * DN9_BYTES  # 475,545,600
TOTAL_DN9_WORDS: int = TOTAL_DN9_BYTES // 4  # 118,886,400


# -----------------------------------------------------------------------------
# Offset Calculation Utilities
# -----------------------------------------------------------------------------
def get_gate_up_chunk_offset_words(e: int, l: int) -> int:
    """Return uint32 element offset for chunk l of expert e in the gate_up buffer.

    Args:
        e: Expert index in [0, 128].
        l: Chunk index in [0, 16].
    """
    return e * GU_WORDS + l * B1_W_PER_CHUNK_WORDS


def get_gate_up_chunk_offset_bytes(e: int, l: int) -> int:
    """Return byte offset for chunk l of expert e in the gate_up buffer."""
    return get_gate_up_chunk_offset_words(e, l) * 4


def get_down_chunk_offset_words(e: int, r: int) -> int:
    """Return uint32 element offset for chunk r (round r) of expert e in the down buffer.

    Args:
        e: Expert index in [0, 128].
        r: Chunk/round index in [0, 19].
    """
    return e * DN_WORDS + r * B2_W_PER_CHUNK_WORDS


def get_down_chunk_offset_bytes(e: int, r: int) -> int:
    """Return byte offset for chunk r of expert e in the down buffer."""
    return get_down_chunk_offset_words(e, r) * 4


def get_down_bfp16_chunk_offset_words(e: int, r: int) -> int:
    """Return uint32 offset for 9-bit down chunk ``r`` of expert ``e``."""
    return e * DN9_WORDS + r * B2_BFP16_W_PER_CHUNK_WORDS


def get_down_bfp16_chunk_offset_bytes(e: int, r: int) -> int:
    """Return byte offset for 9-bit down chunk ``r`` of expert ``e``."""
    return get_down_bfp16_chunk_offset_words(e, r) * 4


# -----------------------------------------------------------------------------
# Packing Functions
# -----------------------------------------------------------------------------
def pack_expert_gate_up(
    gate: np.ndarray,
    up: np.ndarray,
    rows: int = 4,
) -> bytes:
    """Pack a single expert's gate and up matrices into a contiguous q4nx buffer.

    Stacks gate [1088, 2560] and up [1088, 2560] into [2176, 2560], quantizes
    to q4nx, and packs via `pack_gemv`.

    Args:
        gate: Float32 array [1088, 2560].
        up: Float32 array [1088, 2560].
        rows: Core rows (default 4).

    Returns:
        bytes: Packed buffer of length GU_BYTES (3,481,600 bytes).
    """
    g = np.asarray(gate, dtype=np.float32)
    u = np.asarray(up, dtype=np.float32)
    if g.shape != (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE):
        raise ValueError(
            f"Expected gate shape ({MOE_INTERMEDIATE_SIZE}, {HIDDEN_SIZE}), got {g.shape}"
        )
    if u.shape != (MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE):
        raise ValueError(
            f"Expected up shape ({MOE_INTERMEDIATE_SIZE}, {HIDDEN_SIZE}), got {u.shape}"
        )

    w = np.vstack([g, u])  # [2176, 2560]
    codes, scales, mins = quantize(w)
    packed = pack_gemv(codes, scales, mins, rows=rows)
    if len(packed) != GU_BYTES:
        raise ValueError(
            f"Packed gate_up expert byte length mismatch: expected {GU_BYTES}, got {len(packed)}"
        )
    return packed


def pack_expert_gate_up_q4nx(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    rows: int = 4,
) -> bytes:
    """Pack pre-quantized q4nx parameters of a single expert's gate+up matrix [2176, 2560]."""
    packed = pack_gemv(codes, scales, mins, rows=rows)
    if len(packed) != GU_BYTES:
        raise ValueError(
            f"Packed gate_up expert byte length mismatch: expected {GU_BYTES}, got {len(packed)}"
        )
    return packed


def pack_expert_down(
    down: np.ndarray,
    rows: int = 4,
) -> bytes:
    """Pack a single expert's down matrix into a contiguous q4nx buffer.

    Zero-pads down [2560, 1088] to [2560, 1280] via `pad_gemv`, quantizes
    to q4nx, and packs via `pack_gemv`.

    Args:
        down: Float32 array [2560, 1088].
        rows: Core rows (default 4).

    Returns:
        bytes: Packed buffer of length DN_BYTES (2,048,000 bytes).
    """
    d = np.asarray(down, dtype=np.float32)
    if d.shape != (B2_M_PER_EXPERT, B2_K_ORIG):
        raise ValueError(
            f"Expected down shape ({B2_M_PER_EXPERT}, {B2_K_ORIG}), got {d.shape}"
        )

    d_padded, orig_shape = pad_gemv(d, m_align=CHUNK_ROWS, k_align=K_TILE)
    if d_padded.shape != (B2_M_PER_EXPERT, B2_K_PAD):
        raise ValueError(
            f"Expected padded down shape ({B2_M_PER_EXPERT}, {B2_K_PAD}), got {d_padded.shape}"
        )

    codes, scales, mins = quantize(d_padded)
    packed = pack_gemv(codes, scales, mins, rows=rows)
    if len(packed) != DN_BYTES:
        raise ValueError(
            f"Packed down expert byte length mismatch: expected {DN_BYTES}, got {len(packed)}"
        )
    return packed


def pack_expert_down_q4nx(
    codes: np.ndarray,
    scales: np.ndarray,
    mins: np.ndarray,
    rows: int = 4,
) -> bytes:
    """Pack pre-quantized q4nx parameters of a single expert's padded down matrix [2560, 1280]."""
    packed = pack_gemv(codes, scales, mins, rows=rows)
    if len(packed) != DN_BYTES:
        raise ValueError(
            f"Packed down expert byte length mismatch: expected {DN_BYTES}, got {len(packed)}"
        )
    return packed


def pack_expert_down_bfp16(down: np.ndarray, rows: int = 4) -> bytes:
    """Pad and pack one ``[2560, 1088]`` down matrix as 9-bit bfp16ebs8."""
    d = np.asarray(down, dtype=np.float32)
    if d.shape != (B2_M_PER_EXPERT, B2_K_ORIG):
        raise ValueError(
            f"Expected down shape ({B2_M_PER_EXPERT}, {B2_K_ORIG}), got {d.shape}"
        )
    padded = np.zeros((B2_M_PER_EXPERT, B2_K_PAD), dtype=np.float32)
    padded[:, :B2_K_ORIG] = d
    packed = pack_bfp16_gemv(padded, rows=rows, flush_tiny=True)
    if len(packed) != DN9_BYTES:
        raise ValueError(
            f"Packed 9-bit down expert byte length mismatch: expected {DN9_BYTES}, got {len(packed)}"
        )
    return packed


def pack_layer_gate_up(
    gate_list: Sequence[np.ndarray],
    up_list: Sequence[np.ndarray],
    rows: int = 4,
) -> bytes:
    """Pack all 129 experts' gate and up matrices into a full layer buffer.

    Buffer layout: expert 0 .. expert 128 sequentially concatenated.
    Total size: TOTAL_GU_BYTES (449,126,400 bytes).

    Args:
        gate_list: List/sequence of 129 arrays of shape [1088, 2560].
        up_list: List/sequence of 129 arrays of shape [1088, 2560].
        rows: Core rows (default 4).

    Returns:
        bytes: Full layer packed gate_up buffer.
    """
    if len(gate_list) != NUM_TOTAL_EXPERTS:
        raise ValueError(
            f"Expected {NUM_TOTAL_EXPERTS} gate matrices, got {len(gate_list)}"
        )
    if len(up_list) != NUM_TOTAL_EXPERTS:
        raise ValueError(
            f"Expected {NUM_TOTAL_EXPERTS} up matrices, got {len(up_list)}"
        )

    chunks = [
        pack_expert_gate_up(gate_list[e], up_list[e], rows=rows)
        for e in range(NUM_TOTAL_EXPERTS)
    ]
    full_buffer = b"".join(chunks)
    if len(full_buffer) != TOTAL_GU_BYTES:
        raise ValueError(
            f"Full layer gate_up byte length mismatch: expected {TOTAL_GU_BYTES}, got {len(full_buffer)}"
        )
    return full_buffer


def pack_layer_down(
    down_list: Sequence[np.ndarray],
    rows: int = 4,
) -> bytes:
    """Pack all 129 experts' down matrices into a full layer buffer.

    Buffer layout: expert 0 .. expert 128 sequentially concatenated.
    Total size: TOTAL_DN_BYTES (264,192,000 bytes).

    Args:
        down_list: List/sequence of 129 arrays of shape [2560, 1088].
        rows: Core rows (default 4).

    Returns:
        bytes: Full layer packed down buffer.
    """
    if len(down_list) != NUM_TOTAL_EXPERTS:
        raise ValueError(
            f"Expected {NUM_TOTAL_EXPERTS} down matrices, got {len(down_list)}"
        )

    chunks = [
        pack_expert_down(down_list[e], rows=rows) for e in range(NUM_TOTAL_EXPERTS)
    ]
    full_buffer = b"".join(chunks)
    if len(full_buffer) != TOTAL_DN_BYTES:
        raise ValueError(
            f"Full layer down byte length mismatch: expected {TOTAL_DN_BYTES}, got {len(full_buffer)}"
        )
    return full_buffer


def pack_layer_down_bfp16(down_list: Sequence[np.ndarray], rows: int = 4) -> bytes:
    """Pack all 129 down matrices in expert order using 9-bit bfp16ebs8."""
    if len(down_list) != NUM_TOTAL_EXPERTS:
        raise ValueError(
            f"Expected {NUM_TOTAL_EXPERTS} down matrices, got {len(down_list)}"
        )
    full_buffer = b"".join(
        pack_expert_down_bfp16(down_list[e], rows=rows)
        for e in range(NUM_TOTAL_EXPERTS)
    )
    if len(full_buffer) != TOTAL_DN9_BYTES:
        raise ValueError(
            f"Full layer 9-bit down byte length mismatch: expected {TOTAL_DN9_BYTES}, got {len(full_buffer)}"
        )
    return full_buffer


# -----------------------------------------------------------------------------
# Parsing Functions (Inverse Operations)
# -----------------------------------------------------------------------------
def _normalize_to_u8_buffer(buf: bytes | np.ndarray) -> bytes | np.ndarray:
    """Normalize input buffer (bytes, uint8 array, or uint32 array) to a contiguous byte sequence."""
    if isinstance(buf, np.ndarray):
        if not buf.flags.c_contiguous:
            buf = np.ascontiguousarray(buf)
        return buf.view(np.uint8).reshape(-1)
    return buf


def parse_expert_gate_up(
    buf: bytes | np.ndarray,
    e: int,
    rows: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract codes, scales, and mins for expert e from a full or partial gate_up buffer.

    Args:
        buf: Packed gate_up buffer (bytes, uint8, or uint32 array).
        e: Expert index in [0, 128].
        rows: Core rows (default 4).

    Returns:
        codes: uint8 array [2176, 2560].
        scales: float32 array [2176, 80].
        mins: float32 array [2176, 80].
    """
    u8_buf = _normalize_to_u8_buffer(buf)
    offset_bytes = e * GU_BYTES
    chunk = u8_buf[offset_bytes : offset_bytes + GU_BYTES]
    return parse_gemv(chunk, K=B1_K, M=B1_M_PER_EXPERT, rows=rows)


def parse_expert_down(
    buf: bytes | np.ndarray,
    e: int,
    rows: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract codes, scales, and mins for expert e from a full or partial down buffer.

    Args:
        buf: Packed down buffer (bytes, uint8, or uint32 array).
        e: Expert index in [0, 128].
        rows: Core rows (default 4).

    Returns:
        codes: uint8 array [2560, 1280].
        scales: float32 array [2560, 40].
        mins: float32 array [2560, 40].
    """
    u8_buf = _normalize_to_u8_buffer(buf)
    offset_bytes = e * DN_BYTES
    chunk = u8_buf[offset_bytes : offset_bytes + DN_BYTES]
    return parse_gemv(chunk, K=B2_K_PAD, M=B2_M_PER_EXPERT, rows=rows)


def parse_expert_down_bfp16(
    buf: bytes | np.ndarray, e: int, rows: int = 4
) -> np.ndarray:
    """Decode expert ``e`` from a full or partial 9-bit down buffer."""
    u8_buf = _normalize_to_u8_buffer(buf)
    offset_bytes = e * DN9_BYTES
    chunk = u8_buf[offset_bytes : offset_bytes + DN9_BYTES]
    return parse_bfp16_gemv(chunk, K=B2_K_PAD, M=B2_M_PER_EXPERT, rows=rows)
