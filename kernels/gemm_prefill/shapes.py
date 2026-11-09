# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Stage-2a prefill GEMM shapes, independent of the retired gemv_batch experiment."""

from __future__ import annotations

from dataclasses import dataclass

from kernels.gemv_unified.shapes import COLS, UnifiedShape, get_shape as get_unified_shape

MAX_BATCH_T = 24
SUPPORTED_BATCHES = tuple(range(1, MAX_BATCH_T + 1))
PREFILL_SHAPE_NAMES = (
    "rda_in", "f_b", "o_proj", "mla_in", "q_b", "k_b_v_b", "g", "shared_gu", "dense_down",
)


@dataclass(frozen=True)
class BatchShape:
    """A supported bfp16 projection served by the shared prefill device body."""

    name: str
    unified: UnifiedShape

    @property
    def K(self) -> int:
        return self.unified.K

    @property
    def M(self) -> int:
        return self.unified.M

    @property
    def format(self) -> int:
        return self.unified.format

    @property
    def k_blocks(self) -> int:
        return self.unified.k_blocks

    @property
    def weight_words(self) -> int:
        return self.unified.weight_words

    def rounds_by_col(self, cols: int = COLS) -> tuple[int, ...]:
        return self.unified.rounds_by_col(cols)


SHAPES = tuple(BatchShape(name, get_unified_shape(name)) for name in PREFILL_SHAPE_NAMES)
_BY_NAME = {shape.name: shape for shape in SHAPES}


def get_shape(name: str | BatchShape) -> BatchShape:
    if isinstance(name, BatchShape):
        return name
    try:
        return _BY_NAME[name]
    except KeyError as exc:
        raise KeyError(f"unknown prefill GEMM shape {name!r}; choose {sorted(_BY_NAME)}") from exc


def validate_batch_t(batch_t: int, *, max_batch_t: int = MAX_BATCH_T) -> int:
    if max_batch_t != MAX_BATCH_T:
        raise ValueError(f"stage-1 prefill uses one fixed max_batch_t={MAX_BATCH_T} body")
    if not isinstance(batch_t, int) or isinstance(batch_t, bool) or not 1 <= batch_t <= max_batch_t:
        raise ValueError(f"actual T must be an integer in 1..{max_batch_t}")
    return batch_t


def rtp_table(shape: str | BatchShape, cols: int = COLS) -> tuple[tuple[int, int], ...]:
    """Return shape-dependent ``(K blocks, rounds)`` values for every core."""
    resolved = get_shape(shape)
    rounds = resolved.rounds_by_col(cols)
    return tuple(
        (resolved.k_blocks, rounds[col])
        for col in range(cols)
        for _row in range(4)
    )
