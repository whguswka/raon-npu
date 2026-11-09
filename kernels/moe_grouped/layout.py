# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Pure-Python device shape and static transfer-plan contracts."""

from __future__ import annotations

from dataclasses import dataclass

from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_K,
    B1_M_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    B2_BFP16_W_PER_CHUNK_BYTES,
    B2_CHUNKS_PER_EXPERT,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    DN9_BYTES,
    GU_BYTES,
)

SLOTS = 8
TMAX = 8
COLS = 8
ROWS = 4
M_TILE = 32
K_TILE = 256
FORMAT_BFP16 = 0
FORMAT_Q4NX = 1
# RTP word 4 of every grouped core: the real rows of the 8 slots, 4 bits each (slot s in bits 4s..4s+3, 0..TMAX).
# The instruction template writes 8 rows everywhere (the full work); B1 calls overwrite it with their real rows
# (patch.py), B2 calls keep it.
ROWS_WORD_TEMPLATE = 0x88888888


@dataclass(frozen=True)
class GroupedShape:
    """One mode of the shared runtime-selected core program."""

    name: str
    mode: int
    k: int
    m: int
    chunks: int
    chunk_bytes: int
    expert_bytes: int
    max_expert: int

    @property
    def k_blocks(self) -> int:
        return self.k // K_TILE

    @property
    def patch_count(self) -> int:
        return SLOTS * self.chunks

    @property
    def balanced(self) -> bool:
        """One chunk past whole rounds (B1: 17 = 2 x 8 + 1): column c takes slot c's last chunk in one extra
        round after all slots, so every column runs the same number of chunk-rounds (DESIGN-prefill.md section 49)."""
        return self.chunks % COLS == 1 and SLOTS == COLS

    @property
    def rounds_by_col(self) -> tuple[int, ...]:
        """Rounds per slot of each column (a balanced shape adds one extra round per column after all slots)."""
        if self.balanced:
            return (self.chunks // COLS,) * COLS
        return tuple(
            self.chunks // COLS + (1 if col < self.chunks % COLS else 0)
            for col in range(COLS)
        )

    @property
    def weight_words(self) -> int:
        return (self.max_expert + 1) * self.expert_bytes // 4


B1 = GroupedShape(
    "b1", FORMAT_Q4NX, B1_K, B1_M_PER_EXPERT, B1_CHUNKS_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES, GU_BYTES, 127,
)
B2 = GroupedShape(
    "b2", FORMAT_BFP16, B2_K_PAD, B2_M_PER_EXPERT, B2_CHUNKS_PER_EXPERT,
    B2_BFP16_W_PER_CHUNK_BYTES, DN9_BYTES, 128,
)
_SHAPES = {shape.name: shape for shape in (B1, B2)}


def get_shape(name: str | GroupedShape) -> GroupedShape:
    if isinstance(name, GroupedShape):
        return name
    try:
        return _SHAPES[name.lower()]
    except KeyError as exc:
        raise KeyError(f"unknown grouped MoE mode {name!r}; choose b1 or b2") from exc


def stream_groups(shape: str | GroupedShape) -> tuple[tuple[dict[str, int], ...], ...]:
    """Return the static runtime sequence: one TaskGroup per (slot, round) in order, then for a balanced shape one
    extra group in which column c serves slot c's last chunk (DESIGN-prefill.md section 49).

    Each record is one column's X/W/Y transfer. W offsets deliberately name template-local chunks. At runtime every
    offset is replaced with ``expert * expert_bytes + chunk * chunk_bytes``. ``round`` is the schedule position (the
    extra group's is chunks // COLS); the kernel runs that extra round as round 0, so it recomputes the q4nx group sums.
    """
    s = get_shape(shape)

    def record(slot: int, rnd: int, col: int, chunk: int) -> dict[str, int]:
        return {
            "slot": slot,
            "round": rnd,
            "col": col,
            "chunk": chunk,
            "x_offset_bf16": (slot * TMAX * s.k),
            "w_offset_bytes": chunk * s.chunk_bytes,
            "y_offset_bf16": (slot * TMAX * s.m + chunk * ROWS * M_TILE),
        }

    groups: list[tuple[dict[str, int], ...]] = []
    for slot in range(SLOTS):
        for rnd in range(max(s.rounds_by_col)):
            groups.append(tuple(
                record(slot, rnd, col, rnd * COLS + col)
                for col, rounds in enumerate(s.rounds_by_col)
                if rnd < rounds
            ))
    if s.balanced:
        groups.append(tuple(record(col, s.chunks // COLS, col, s.chunks - 1) for col in range(COLS)))
    return tuple(groups)


def stream_plan(shape: str | GroupedShape) -> tuple[dict[str, int], ...]:
    """Return every column transfer of ``stream_groups`` in sequence order (the W patch order)."""
    return tuple(record for group in stream_groups(shape) for record in group)


def resource_budget() -> dict[str, object]:
    """Conservative shared-body budget; linked values remain notebook evidence."""
    banks = {
        "bank0": {"stack": (0, 12288), "acc": (12288, 16384)},
        "bank1": {
            "w0_max": (16384, 25600),
            "y0": (25600, 26624),
            "group_sums": (26624, 27904),
        },
        "bank2": {"w1_max": (32768, 41984), "y1": (41984, 43008)},
        "bank3": {"x0": (49152, 57344), "x1": (57344, 65536)},
    }
    return {
        "l1_banks": banks,
        "l1_used_bytes": 52480,
        "l1_limit_bytes": 65536,
        "memtile_bytes": 327680,
        "memtile_limit_bytes": 524288,
        "core_bd_peak": 6,
        "core_bd_limit": 16,
        "core_lock_peak": 7,
        "core_lock_limit": 16,
        "memtile_bd_peak": 24,
        "memtile_bd_limit": 48,
        "memtile_lock_peak": 24,
        "memtile_lock_limit": 64,
        "text_bytes": None,
        "text_limit_bytes": 16384,
        "measured": False,
    }
