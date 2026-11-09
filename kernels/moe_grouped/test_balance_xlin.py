# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for grouped MoE section 49: column-balanced rounds + one-line contiguous X."""

from __future__ import annotations

import numpy as np
import pytest

from kernels.moe_grouped.layout import (
    B1,
    B2,
    COLS,
    M_TILE,
    ROWS,
    SLOTS,
    TMAX,
    stream_groups,
    stream_plan,
)
from kernels.moe_grouped.op import GroupedMoE
from npu.backend import _grouped_x_tiled


def _old_stream_plan(shape):
    """Reference stream_plan from main 370ec63 before section 49."""
    records = []
    for slot in range(SLOTS):
        for rnd in range(max(shape.rounds_by_col)):
            for col, rounds in enumerate(shape.rounds_by_col):
                if rnd >= rounds:
                    continue
                chunk = rnd * COLS + col
                records.append(
                    {
                        "slot": slot,
                        "round": rnd,
                        "col": col,
                        "chunk": chunk,
                        "x_offset_bf16": (slot * TMAX * shape.k),
                        "w_offset_bytes": chunk * shape.chunk_bytes,
                        "y_offset_bf16": (slot * TMAX * shape.m + chunk * ROWS * M_TILE),
                    }
                )
    return tuple(records)


def test_balanced_shape_flags() -> None:
    assert B1.balanced is True
    assert B2.balanced is False


def test_rounds_by_col_contract() -> None:
    assert B1.rounds_by_col == (2,) * COLS
    assert B2.rounds_by_col == (3, 3, 3, 3, 2, 2, 2, 2)


def test_stream_plan_covers_every_slot_chunk_exactly_once() -> None:
    for shape in (B1, B2):
        plan = stream_plan(shape)
        assert len(plan) == shape.patch_count
        pairs = {(rec["slot"], rec["chunk"]) for rec in plan}
        expected = {(slot, chunk) for slot in range(SLOTS) for chunk in range(shape.chunks)}
        assert pairs == expected


def test_chunk_round_counts_per_column() -> None:
    for shape, expected in (
        (B1, [17] * COLS),
        (B2, [24] * 4 + [16] * 4),
    ):
        counts = [0] * COLS
        for rec in stream_plan(shape):
            counts[rec["col"]] += 1
        assert counts == expected


def test_b1_extra_group_takes_slot_c_last_chunk() -> None:
    groups = stream_groups(B1)
    extra = groups[-1]
    assert len(extra) == COLS
    assert all(rec["round"] == 2 for rec in extra)
    assert [(rec["slot"], rec["col"], rec["chunk"]) for rec in extra] == [
        (col, col, B1.chunks - 1) for col in range(COLS)
    ]


def test_b2_stream_plan_matches_pre_section_49_reference() -> None:
    assert stream_plan(B2) == _old_stream_plan(B2)


def test_b1_stream_plan_is_old_reference_minus_trailing_chunks_plus_extra_group() -> None:
    old = _old_stream_plan(B1)
    # The old plan had 16 records per slot (chunks=17) and is not balanced.
    # The new plan drops the chunk-16 record from each slot's main rounds and appends
    # the balanced extra group where column c takes slot c's chunk 16.
    without_trailing = tuple(rec for rec in old if rec["chunk"] != B1.chunks - 1)
    new = stream_plan(B1)
    assert new[: len(without_trailing)] == without_trailing
    assert len(new) == len(old) + COLS
    extra = new[len(without_trailing):]
    assert [(rec["slot"], rec["col"], rec["chunk"]) for rec in extra] == [
        (col, col, B1.chunks - 1) for col in range(COLS)
    ]


def test_stream_group_records_are_sorted_by_column() -> None:
    for shape in (B1, B2):
        for group in stream_groups(shape):
            cols = [rec["col"] for rec in group]
            assert cols == sorted(cols)


def test_operator_config_name_is_rtp7_and_mode_specific() -> None:
    b1_name = GroupedMoE(B1.name).config_name
    b2_name = GroupedMoE(B2.name).config_name
    assert "rtp7" in b1_name
    assert "rtp7" in b2_name
    assert "rtp5" not in b1_name
    assert "rtp4" not in b1_name
    assert b1_name != b2_name


@pytest.mark.parametrize("shape", [B1, B2])
def test_grouped_x_tiled_matches_old_4d_tab_stride_and_round_trips(shape) -> None:
    """The device X tap is the former 16-byte walk over [SLOTS,TMAX,K], now tiled per slot."""
    rng = np.random.RandomState(4000 + (7 if shape is B1 else 13))
    K = shape.k
    x = rng.randint(0, 65536, size=(SLOTS, TMAX, K), dtype=np.uint16)
    out = _grouped_x_tiled(x, shape)
    assert out.dtype == np.uint16
    assert out.flags.c_contiguous

    # Old tap shape [k_blocks, 32, TMAX, 8] with strides [256, 8, K, 1].
    expected_flat = np.asarray(
        [
            x.reshape(-1)[s * TMAX * K + b * 256 + g * 8 + t * K + e]
            for s in range(SLOTS)
            for b in range(K // 256)
            for g in range(32)
            for t in range(TMAX)
            for e in range(8)
        ],
        dtype=np.uint16,
    )
    np.testing.assert_array_equal(out.reshape(-1), expected_flat)

    round_tripped = (
        out.reshape(SLOTS, K // 256, 32, TMAX, 8)
        .transpose(0, 3, 1, 2, 4)
        .reshape(SLOTS, TMAX, K)
    )
    np.testing.assert_array_equal(round_tripped, x)
