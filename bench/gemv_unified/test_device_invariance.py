# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the device-invariance MLIR comparison utility."""

import argparse

import pytest

from bench.gemv_unified.check_device_invariance import (
    DEFAULT_BOUNDARY_SHAPES,
    device_without_runtime_sequence,
    normalize_ssa,
    parse_names,
    parse_shape,
    parse_shapes,
    select_cases,
)
from kernels.gemv_unified.shapes import ALL_SHAPES


def test_shape_parser_accepts_optional_active_columns() -> None:
    assert parse_shape("512:5120") == (512, 5120, 8)
    assert parse_shape("256:128:1") == (256, 128, 1)
    assert parse_shapes("256:128:1,512:5120:8") == (
        (256, 128, 1),
        (512, 5120, 8),
    )


def test_shape_parser_rejects_wrong_field_count() -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=r"K:M\[:cols\]"):
        parse_shape("512")


def test_name_parser_accepts_every_registered_shape() -> None:
    names = tuple(shape.name for shape in ALL_SHAPES)
    assert parse_names(",".join(names)) == names


def test_default_cases_cover_all_exact_shapes_and_boundaries() -> None:
    cases = select_cases(None, None)
    supported = tuple(case.shape for case in cases if not case.experimental)
    boundary = tuple(case.shape for case in cases if case.experimental)
    assert supported == ALL_SHAPES
    assert len(boundary) == len(parse_shapes(DEFAULT_BOUNDARY_SHAPES))
    assert any(shape.format != 0 for shape in supported)
    assert any(
        shape.x_slots == 7 and shape.chunks_per_col == 20 for shape in supported
    )


def test_runtime_sequence_removal_preserves_later_device_operations() -> None:
    mlir = """module {
  aie.device(npu2) {
    %tile = aie.tile(0, 1)
    aie.runtime_sequence(%arg0: memref<1xi32>) {
      scf.if %flag {
        aiex.npu.write32 {value = 1 : i32}
      }
    }
    aie.flow(%tile, DMA : 0, %tile, DMA : 1)
  }
}
"""
    device = device_without_runtime_sequence(mlir)
    assert "runtime_sequence" not in device
    assert "%tile = aie.tile(0, 1)" in device
    assert "aie.flow" in device


def test_ssa_normalization_preserves_reference_structure() -> None:
    left = "%7 = op\nuse %7, %name\n"
    right = "%foo = op\nuse %foo, %other\n"
    assert normalize_ssa(left) == normalize_ssa(right)
