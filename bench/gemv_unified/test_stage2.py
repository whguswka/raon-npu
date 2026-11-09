# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host gates for unified q4nx, MoE B1-6 and B2-9 stream contracts."""

from pathlib import Path

import numpy as np
import pytest

from kernels.gemv_unified import (
    COLS,
    FORMAT_BFP16,
    FORMAT_Q4NX,
    MAX_K,
    Q4_BLOCK_BYTES,
    Q4_CHUNK_BYTES,
    ROWS,
    STAGE2_SHAPES,
    W_BLOCK_BYTES,
    UnifiedMoEDown9,
    UnifiedMoEGateUp,
    UnifiedShape,
    get_shape,
    pad_x,
    rtp_table,
    stream_plan,
)


def test_stage2_shape_table() -> None:
    assert [(s.name, s.K, s.M) for s in STAGE2_SHAPES] == [
        ("dense_gu", 2560, 24064),
        ("moe_b1_6", 2560, 13056),
        ("moe_b2_9", 1280, 17920),
    ]


@pytest.mark.parametrize("shape", STAGE2_SHAPES, ids=lambda s: s.name)
def test_stage2_rtp(shape: UnifiedShape) -> None:
    table = rtp_table(shape)
    assert len(table) == COLS * ROWS
    for col in range(COLS):
        assert {table[col * ROWS + row] for row in range(ROWS)} == {
            (shape.format, shape.k_blocks, shape.rounds_by_col()[col])
        }


@pytest.mark.parametrize("shape", STAGE2_SHAPES, ids=lambda s: s.name)
def test_stage2_plan_coverage(shape: UnifiedShape) -> None:
    plan = stream_plan(shape)
    assert len(plan) == shape.chunks
    assert {r["chunk"] for r in plan} == set(range(shape.chunks))
    for rec in plan:
        assert rec["w_offset_words"] + rec["w_length_words"] <= shape.weight_words
        assert rec["y_offset_bf16"] + rec["y_length_bf16"] <= shape.M


def test_b1_per_call_uses_private_contiguous_shape_only() -> None:
    baseline = get_shape("moe_b1_6")
    op = UnifiedMoEGateUp(sequence_plan="per_call")
    assert baseline.active_cols is None and baseline.chunks_per_col is None
    assert op.shape is baseline
    assert op._op.shape.active_cols == 6
    assert op._op.shape.chunks_per_col == 17
    assert op._op.name.endswith("_pc")


@pytest.mark.parametrize("name", ["dense_gu", "moe_b1_6", "moe_b2_9"])
def test_stage2_x_padding(name: str) -> None:
    s = get_shape(name)
    x = np.arange(s.x_slots * s.K, dtype=np.uint16).reshape(s.x_slots, s.K)
    padded = pad_x(x, s)
    assert padded.size == s.x_slots * MAX_K
    view = padded.reshape(s.x_slots, MAX_K)
    assert np.array_equal(view[:, : s.K], x)
    assert np.count_nonzero(view[:, s.K :]) == 0


@pytest.mark.parametrize("name", ["dense_gu", "moe_b1_6", "moe_b2_9"])
def test_stage2_x_padding_rejects_wrong_size(name: str) -> None:
    s = get_shape(name)
    with pytest.raises(ValueError, match="expected"):
        pad_x(np.zeros(s.x_slots * s.K - 1, np.uint16), s)


def test_format_and_block_constants() -> None:
    assert (FORMAT_BFP16, FORMAT_Q4NX) == (0, 1)
    assert (W_BLOCK_BYTES, Q4_BLOCK_BYTES) == (9216, 5120)
    assert Q4_CHUNK_BYTES == 4 * Q4_BLOCK_BYTES


def test_l1_budget_is_below_64_kib() -> None:
    x = MAX_K * 2
    w = 2 * W_BLOCK_BYTES
    y = 2 * 32 * 2
    acc = 32 * 4
    sums = (MAX_K // 32) * 2
    rtp = 3 * 4
    stack = 10 * 1024
    assert x + w + y + acc + sums + rtp + stack == 53756
    assert 65536 - (x + w + y + acc + sums + rtp + stack) >= 4096


@pytest.mark.parametrize("block_bytes", [W_BLOCK_BYTES, Q4_BLOCK_BYTES])
def test_one_block_bd_word_limit(block_bytes: int) -> None:
    assert block_bytes // 4 <= 16383


@pytest.mark.parametrize(
    ("cls", "name"),
    [(UnifiedMoEGateUp, "moe_b1_6"), (UnifiedMoEDown9, "moe_b2_9")],
)
def test_moe_wrapper_shape(cls, name: str) -> None:
    wrapper = cls(context=None)
    assert wrapper.shape.name == name


@pytest.mark.parametrize(
    "kwargs",
    [
        {"format": 7},
        {"active_cols": 0},
        {"active_cols": 9},
        {"active_cols": 7, "x_slots": 6},
        {"active_cols": 7, "chunks_per_col": 19},
        {"K": 257},
    ],
)
def test_stage2_shape_validation(kwargs: dict) -> None:
    base = dict(name="bad", K=1280, M=7 * 2560)
    base.update(kwargs)
    with pytest.raises(ValueError):
        UnifiedShape(**base)


@pytest.mark.parametrize("shape", STAGE2_SHAPES, ids=lambda s: s.name)
def test_stage2_stream_specific_invariants(shape: UnifiedShape) -> None:
    plan = stream_plan(shape)
    if shape.name == "moe_b2_9":
        assert shape.rounds_by_col() == (20, 20, 20, 20, 20, 20, 20, 0)
        assert {r["x_offset_bf16"] for r in plan} == {
            c * MAX_K for c in range(7)
        }
    else:
        assert all(r["x_offset_bf16"] == 0 for r in plan)
        assert shape.format == FORMAT_Q4NX


@pytest.mark.parametrize(
    "symbol",
    [
        "q4nx_gemv_unified_prep",
        "q4nx_gemv_unified_zero",
        "q4nx_gemv_unified_block",
        "q4nx_gemv_unified_epilogue",
        "bfp16_gemv_unified_block",
    ],
)
def test_combined_cpp_object_exports(symbol: str) -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "kernels"
        / "gemv_unified"
        / "bfp16_gemv.cc"
    ).read_text(encoding="utf-8")
    assert symbol in source
