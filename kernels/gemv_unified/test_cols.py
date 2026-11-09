# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Tests for the device-width ``cols`` parameter in unified GEMV.

The default build keeps the existing 8-column design byte-identical.  A
smaller ``cols`` value lets the same code generate a 7-column (or fewer)
variant without touching the device constants in ``shapes.py``.
"""

import pytest

from kernels.gemv_unified import design as unified_design
from kernels.gemv_unified.op import UnifiedGEMV
from kernels.gemv_unified.routing import unified_stream_topology
from kernels.gemv_unified.shapes import (
    ALL_SHAPES,
    COLS,
    ROWS,
    get_shape,
    rtp_table,
    stream_plan,
)
from kernels.gemv_unified.w_plan import declared_memtile_resources, mem_chunk_plan


# Registered 8-column rounds from the unified GEMV shape table.  They must
# stay stable because the generated xclbin/insts.bin naming depends on them.
EXPECTED_8COL_ROUNDS = {
    "rda_in": (21, 21, 20, 20, 20, 20, 20, 20),
    "f_b": (5, 5, 5, 5, 5, 5, 5, 5),
    "o_proj": (3, 3, 3, 3, 2, 2, 2, 2),
    "mla_in": (2, 1, 1, 1, 1, 1, 1, 1),
    "q_b": (8, 8, 8, 8, 7, 7, 7, 7),
    "k_b_v_b": (5, 5, 5, 5, 5, 5, 5, 5),
    "g": (5, 5, 5, 5, 5, 5, 5, 5),
    "shared_gu": (3, 2, 2, 2, 2, 2, 2, 2),
    "dense_down": (3, 3, 3, 3, 2, 2, 2, 2),
    "lm_head": (254, 254, 254, 254, 254, 254, 254, 254),
}


@pytest.mark.parametrize("name, expected", EXPECTED_8COL_ROUNDS.items())
def test_8col_rounds_match_registered_table(name: str, expected: tuple[int, ...]) -> None:
    # rounds_by_col with cols=8 returns the documented values exactly.
    shape = get_shape(name)
    assert shape.rounds_by_col(cols=COLS) == expected


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_7col_rounds_sum_to_total_chunks(shape) -> None:
    # 7-column scheduling still covers every output chunk.
    rounds = shape.rounds_by_col(cols=7)
    assert len(rounds) == 7
    assert sum(rounds) == shape.chunks


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_7col_stream_plan_covers_all_chunks(shape) -> None:
    # stream_plan chunks for cols=7 partition [0, chunks) without gaps.
    records = stream_plan(shape, cols=7)
    seen = {rec["chunk"] for rec in records}
    assert seen == set(range(shape.chunks))
    cols_seen = {rec["col"] for rec in records}
    assert cols_seen == set(range(7))


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_7col_mem_chunk_plan_sums_to_chunks(shape) -> None:
    # mem_chunk_plan for cols=7 touches exactly the active columns.
    plan = mem_chunk_plan(shape, cols=7)
    assert set(plan.keys()) <= set(range(7))
    distinct_rounds = {
        (col, chunk.first_round)
        for col, chunks in plan.items()
        for chunk in chunks
    }
    assert len(distinct_rounds) == shape.chunks


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_7col_rtp_table_has_one_entry_per_core(shape) -> None:
    # RTP table length equals active cores (cols * rows); each row of a column
    # receives the same (format, k_blocks, rounds[col]) triple.
    table = rtp_table(shape, cols=7)
    assert len(table) == 7 * ROWS
    rounds = shape.rounds_by_col(cols=7)
    for col in range(7):
        col_entries = table[col * ROWS : (col + 1) * ROWS]
        assert all(entry == (shape.format, shape.k_blocks, rounds[col]) for entry in col_entries)


def test_7col_stream_topology_excludes_inactive_columns() -> None:
    # cols=7 topology declares x0..x6 and no x7 paths.
    paths = unified_stream_topology(cols=7)
    names = {path.name for path in paths}
    assert {f"x{j}" for j in range(7)} <= names
    assert not any(name.startswith("x7") for name in names)
    assert {f"w_in_{j}" for j in range(7)} <= names
    assert not any(name.startswith("w_in_7") for name in names)


def test_declared_memtile_resources_scales_with_cols() -> None:
    # Resource declarations span exactly ``cols`` columns.
    res_8 = {r.name for r in declared_memtile_resources(cols=8)}
    res_7 = {r.name for r in declared_memtile_resources(cols=7)}
    assert res_7 < res_8
    assert all("_7_" not in name for name in res_7)
    assert any("_7_" in name for name in res_8)


def test_check_shape_resolves_active_cols_to_cols() -> None:
    # active_cols=None is resolved to cols.
    s = unified_design.check_shape(None, 2560, 2560, cols=7)
    assert s.active_cols == 7


def test_check_shape_rejects_active_cols_larger_than_cols() -> None:
    # active_cols must not exceed the configured device width.
    with pytest.raises(ValueError, match="active_cols"):
        unified_design.check_shape(None, 2560, 2560, cols=7, active_cols=8)


def test_unified_gemv_reference_shape_scales_with_cols() -> None:
    # The minimal reference shape scales with the requested column count.
    op8 = UnifiedGEMV(K=256, M=1024, cols=8, experimental_shape=True)
    assert op8._reference_shape == (256, 8 * 128)
    op7 = UnifiedGEMV(K=256, M=896, cols=7, experimental_shape=True)
    assert op7._reference_shape == (256, 7 * 128)


def test_unified_gemv_for_shape_passes_cols() -> None:
    # UnifiedGEMV.for_shape must pass the requested cols down to the operator.
    shape = get_shape("shared_gu")
    op = UnifiedGEMV.for_shape(shape, cols=8)
    assert op.cols == 8
