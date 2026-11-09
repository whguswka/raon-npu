# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""The stage-1 unified GEMV shape contract and static stream planning.

All offsets are expressed in the element type of the corresponding host BO:
bf16 elements for X/Y and uint32 words for W.  Keeping this module free of
IRON imports makes the stream plan independently testable on the CPU-only host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

K_TILE = 256
M_TILE = 32
ROWS = 4
COLS = 8
M_PER_CHUNK = ROWS * M_TILE
W_BLOCK_BYTES = M_TILE * K_TILE * 9 // 8
W_CHUNK_BYTES = ROWS * W_BLOCK_BYTES
Q4_BLOCK_BYTES = M_TILE * K_TILE * 5 // 8
Q4_CHUNK_BYTES = ROWS * Q4_BLOCK_BYTES
MAX_K = 12032
FORMAT_BFP16 = 0
FORMAT_Q4NX = 1
RTP_FORMAT = 0
RTP_K_BLOCKS = 1
RTP_ROUNDS = 2
RTP_WORDS = 3


@dataclass(frozen=True)
class UnifiedShape:
    """One static instruction stream served by the shared device body.

    ``active_cols`` defaults to ``None``, meaning the active column count is
    resolved against the device column count at build time.  This lets one
    registered shape definition work for both 8-column and 7-column devices
    without maintaining two copies of every shape value.
    """

    name: str
    K: int
    M: int
    format: int = FORMAT_BFP16
    active_cols: int | None = None
    x_slots: int = 1
    chunks_per_col: int | None = None

    def __post_init__(self) -> None:
        if self.K <= 0 or self.K > MAX_K or self.K % K_TILE:
            raise ValueError(f"{self.name}: K must be a multiple of {K_TILE} in 1..{MAX_K}")
        if self.M <= 0 or self.M % M_PER_CHUNK:
            raise ValueError(f"{self.name}: M must be a multiple of {M_PER_CHUNK}")
        if self.format not in (FORMAT_BFP16, FORMAT_Q4NX):
            raise ValueError(f"{self.name}: unsupported format {self.format}")
        if self.active_cols is not None:
            if not 1 <= self.active_cols <= COLS:
                raise ValueError(f"{self.name}: active_cols must be in 1..{COLS}")
            if self.x_slots not in (1, self.active_cols):
                raise ValueError(f"{self.name}: x_slots must be 1 or active_cols")
        else:
            if self.x_slots != 1:
                raise ValueError(
                    f"{self.name}: x_slots must be 1 when active_cols is deferred"
                )
        if self.chunks_per_col is not None:
            if self.active_cols is None:
                raise ValueError(
                    f"{self.name}: active_cols must be explicit when chunks_per_col is set"
                )
            if self.chunks != self.active_cols * self.chunks_per_col:
                raise ValueError(
                    f"{self.name}: chunks must equal active_cols * chunks_per_col"
                )

    @property
    def k_blocks(self) -> int:
        return self.K // K_TILE

    @property
    def chunks(self) -> int:
        return self.M // M_PER_CHUNK

    @property
    def max_rounds(self) -> int:
        return max(self.rounds_by_col())

    @property
    def weight_bytes(self) -> int:
        bits = 9 if self.format == FORMAT_BFP16 else 5
        return self.M * self.K * bits // 8

    @property
    def weight_words(self) -> int:
        assert self.weight_bytes % 4 == 0
        return self.weight_bytes // 4

    @property
    def w_chunk_bytes(self) -> int:
        block_bytes = W_CHUNK_BYTES if self.format == FORMAT_BFP16 else Q4_CHUNK_BYTES
        return self.k_blocks * block_bytes

    @property
    def w_chunk_words(self) -> int:
        assert self.w_chunk_bytes % 4 == 0
        return self.w_chunk_bytes // 4

    def resolve_active_cols(self, cols: int | None = None) -> int:
        """Return the active column count bound by the device column count.

        ``None`` means "use the full device width".  Raises ``ValueError`` if
        an explicitly configured ``active_cols`` exceeds the device width.
        """
        if cols is None:
            cols = COLS
        if self.active_cols is None:
            return cols
        if self.active_cols > cols:
            raise ValueError(
                f"{self.name}: active_cols {self.active_cols} exceeds device cols {cols}"
            )
        return self.active_cols

    def rounds_by_col(self, cols: int | None = None) -> tuple[int, ...]:
        if cols is None:
            cols = COLS
        active = self.resolve_active_cols(cols)
        if self.chunks_per_col is not None:
            return tuple(
                self.chunks_per_col if col < active else 0
                for col in range(cols)
            )
        return tuple(
            self.chunks // active
            + (1 if col < self.chunks % active else 0)
            if col < active
            else 0
            for col in range(cols)
        )


SHAPES: tuple[UnifiedShape, ...] = (
    UnifiedShape("rda_in", 2560, 20736),
    UnifiedShape("f_b", 256, 5120),
    UnifiedShape("o_proj", 5120, 2560),
    UnifiedShape("mla_in", 2560, 1152),
    UnifiedShape("q_b", 768, 7680),
    UnifiedShape("k_b_v_b", 512, 5120),
    UnifiedShape("g", 2560, 5120),
    UnifiedShape("shared_gu", 2560, 2176),
    UnifiedShape("dense_down", 12032, 2560),
    UnifiedShape("lm_head", 2560, 260096),
)
STAGE2_SHAPES: tuple[UnifiedShape, ...] = (
    UnifiedShape("dense_gu", 2560, 24064, format=FORMAT_Q4NX),
    UnifiedShape("moe_b1_6", 2560, 6 * 2176, format=FORMAT_Q4NX),
    UnifiedShape(
        "moe_b2_9",
        1280,
        7 * 2560,
        format=FORMAT_BFP16,
        active_cols=7,
        x_slots=7,
        chunks_per_col=20,
    ),
)
ALL_SHAPES = SHAPES + STAGE2_SHAPES
_BY_NAME = {shape.name: shape for shape in ALL_SHAPES}
_ALIASES = {
    "k_b": "k_b_v_b",
    "v_b": "k_b_v_b",
    "k_b/v_b": "k_b_v_b",
}


def get_shape(name: str) -> UnifiedShape:
    """Return a named supported shape, with a useful error for CLI callers."""
    key = _ALIASES.get(name, name)
    try:
        return _BY_NAME[key]
    except KeyError as exc:
        raise KeyError(f"unknown unified GEMV shape {name!r}; choose {sorted(_BY_NAME)}") from exc


def rtp_table(shape: UnifiedShape, cols: int | None = None) -> tuple[tuple[int, int, int], ...]:
    """Return row-major (col, row) RTP values for every core on the device."""
    if cols is None:
        cols = COLS
    rounds = shape.rounds_by_col(cols)
    return tuple(
        (shape.format, shape.k_blocks, rounds[col])
        for col in range(cols)
        for _row in range(ROWS)
    )


def x_slot_offset(shape: UnifiedShape, col: int, slot_elems: int) -> int:
    """Return the X element offset used by one physical column.

    Multi-slot shapes assign one slot to each active column.  Inactive columns
    still receive X to complete their acquire/release protocol, so they safely
    reuse slot zero instead of addressing past the host tensor.
    """
    return col * slot_elems if col < shape.x_slots else 0


def stream_plan(shape: UnifiedShape, cols: int | None = None) -> tuple[dict[str, int], ...]:
    """Return the shape stream's round-major W/Y transfers.

    ``chunk`` is the canonical output chunk.  W uses uint32 word offsets,
    while Y and X use bf16 element offsets.  Multi-slot shapes give each active
    column its own X slot; inactive columns reuse slot zero so their zero-round
    worker can complete the barrier/acquire/release protocol.
    """
    if cols is None:
        cols = COLS
    active = shape.resolve_active_cols(cols)
    col_rounds = shape.rounds_by_col(cols)
    max_rounds = max(col_rounds) if col_rounds else 0
    records: list[dict[str, int]] = []
    for rnd in range(max_rounds):
        for col in range(cols):
            if rnd >= col_rounds[col]:
                continue
            chunk = (
                col * shape.chunks_per_col + rnd
                if shape.chunks_per_col is not None
                else rnd * active + col
            )
            records.append(
                {
                    "round": rnd,
                    "col": col,
                    "chunk": chunk,
                    "x_offset_bf16": x_slot_offset(shape, col, MAX_K),
                    # The shared xclbin's receiver BD is fixed at MAX_K.  The
                    # host pads X; the core still consumes only shape.K values.
                    "x_length_bf16": MAX_K,
                    "w_offset_words": chunk * shape.w_chunk_words,
                    "w_length_words": shape.w_chunk_words,
                    "y_offset_bf16": chunk * M_PER_CHUNK,
                    "y_length_bf16": M_PER_CHUNK,
                }
            )
    return tuple(records)


def selected_shapes(names: Iterable[str] | None = None) -> tuple[UnifiedShape, ...]:
    """Resolve an optional list of CLI shape names without reordering it."""
    return SHAPES if names is None else tuple(get_shape(name) for name in names)
