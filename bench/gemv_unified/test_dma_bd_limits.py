# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only mirror of the AIE2 BD field verifier.

Field widths come from the target model, not from the failing notebook log:

* shim: ``AIETargetModel.cpp:850-879``
* memory tile: ``AIETargetModel.cpp:882-914``
* core tile: ``AIETargetModel.cpp:917-940``
* maximum repeat count: ``AIETargetModel.h:848-854``

The size/stride encoding and checks mirror ``BdLowering.h:145-195`` and
``AIEXDialect.cpp:146-258``.  The specifications under test are the same
objects consumed by the real IRON design builders in ``design.py`` and
``exp_short_dma.py``.
"""

from __future__ import annotations

from math import prod

import pytest

from kernels.gemv_unified.dma_layout import (
    DmaTapSpec,
    gemv_dma_tap_specs,
    per_call_shim_tap_specs,
    short_dma_tap_specs,
)
from kernels.gemv_unified.shapes import (
    ALL_SHAPES,
    COLS,
    FORMAT_BFP16,
    FORMAT_Q4NX,
    MAX_K,
    Q4_BLOCK_BYTES,
    ROWS,
    UnifiedShape,
    W_BLOCK_BYTES,
)


# (wrap bits, step bits, iteration bits, linear buffer-length bits)
TARGET_LIMITS = {
    "shim": (10, 20, 6, 32),
    "mem": (10, 17, 6, 17),
    "core": (8, 13, 6, 14),
}
ADDRESS_GRANULARITY_BITS = 32
MAX_REPEAT_COUNT = 255


def _clean_leading_units(
    sizes: tuple[int, ...], strides: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    cleaned = list(strides)
    for index in range(len(sizes) - 1):
        if sizes[index] != 1:
            break
        cleaned[index] = 0
    return sizes, tuple(cleaned)


def _coalesce(
    sizes: tuple[int, ...], strides: tuple[int, ...]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Mirror ``TensorAccessPattern.coalesce`` for constant integer TAPs."""
    kept = [index for index, size in enumerate(sizes) if size != 1]
    if not kept:
        kept = [len(sizes) - 1]
    out_sizes = [sizes[index] for index in kept]
    out_strides = [strides[index] for index in kept]
    index = 0
    while index + 1 < len(out_sizes):
        if out_strides[index] == out_sizes[index + 1] * out_strides[index + 1]:
            out_sizes[index : index + 2] = [
                out_sizes[index] * out_sizes[index + 1]
            ]
            out_strides[index : index + 2] = [out_strides[index + 1]]
        else:
            index += 1
    return tuple(out_sizes), tuple(out_strides)


def _is_contiguous(sizes: tuple[int, ...], strides: tuple[int, ...]) -> bool:
    expected = 1
    for size, stride in zip(reversed(sizes), reversed(strides)):
        if size > 1 and stride != expected:
            return False
        expected *= size
    return True


def _verify_bd(spec: DmaTapSpec) -> None:
    """Translate ``verifyStridesWraps`` and raise AssertionError on violation."""
    wrap_bits, step_bits, iter_bits, length_bits = TARGET_LIMITS[spec.tile]
    sizes, strides = _clean_leading_units(spec.sizes, spec.strides)

    # Tile Bd._emit coalesces ordinary TAPs.  Dimension-preserving q4 receive
    # TAPs intentionally bypass that step.  Shim tasks keep their four slots.
    if spec.tile != "shim" and not spec.preserve_dimensions:
        sizes, strides = _coalesce(sizes, strides)

    total_bits = prod(sizes) * spec.element_bits
    assert total_bits % ADDRESS_GRANULARITY_BITS == 0, spec
    length_words = total_bits // ADDRESS_GRANULARITY_BITS
    assert length_words <= (1 << length_bits) - 1, (
        f"{spec.name}: linear/buffer length {length_words} exceeds "
        f"{spec.tile} {length_bits}-bit limit"
    )

    # A contiguous transfer is lowered in linear mode, where d0/d1 wrap fields
    # are zero and only the tile's wider buffer-length field applies.
    if _is_contiguous(sizes, strides):
        return

    assert len(sizes) <= 4, f"{spec.name}: rank {len(sizes)} exceeds four BD slots"
    pad = 4 - len(sizes)
    outer_sizes = (1,) * pad + sizes
    outer_strides = (0,) * pad + strides
    input_sizes = tuple(reversed(outer_sizes))
    input_strides = tuple(reversed(outer_strides))

    assert all(size > 0 for size in input_sizes), spec
    assert input_sizes[0] * spec.element_bits % ADDRESS_GRANULARITY_BITS == 0, spec
    for index, (size, stride) in enumerate(zip(input_sizes, input_strides)):
        if size > 1:
            assert stride >= (0 if index == 3 else 1), spec
        if index != 0 or stride != 1:
            assert stride * spec.element_bits % ADDRESS_GRANULARITY_BITS == 0, spec
    if spec.element_bits != ADDRESS_GRANULARITY_BITS and input_sizes[0] > 1:
        assert input_strides[0] == 1, spec

    hardware_sizes = (
        input_sizes[0] * spec.element_bits // ADDRESS_GRANULARITY_BITS,
        input_sizes[1],
        input_sizes[2],
        input_sizes[3] - 1 if input_sizes[3] > 1 else 0,
    )

    def encoded_stride(index: int) -> int:
        size, stride = input_sizes[index], input_strides[index]
        if index == 0:
            if stride * spec.element_bits < ADDRESS_GRANULARITY_BITS:
                return 0
        elif size <= 1:
            return 0
        if index == 3 and stride <= 0:
            return 0
        return stride * spec.element_bits // ADDRESS_GRANULARITY_BITS - 1

    hardware_strides = tuple(encoded_stride(index) for index in range(4))
    wrap_max = (1 << wrap_bits) - 1
    step_max = (1 << step_bits) - 1
    assert hardware_sizes[0] <= wrap_max, (
        f"{spec.name}: d0 hardware size {hardware_sizes[0]} > {wrap_max}"
    )
    assert hardware_sizes[1] <= wrap_max, (
        f"{spec.name}: d1 hardware size {hardware_sizes[1]} > {wrap_max}"
    )
    pure_repeat = input_strides[3] == 0
    max_count = MAX_REPEAT_COUNT + 1 if pure_repeat else 1 << iter_bits
    assert input_sizes[3] <= max_count, spec
    for index in range(3):
        assert hardware_strides[index] <= step_max, (
            f"{spec.name}: d{index} hardware stride "
            f"{hardware_strides[index]} > {step_max}"
        )
    if hardware_sizes[3] > 0:
        assert hardware_strides[3] <= step_max, spec


def _flatten(groups: dict[str, tuple[DmaTapSpec, ...]]):
    return (spec for specs in groups.values() for spec in specs)


def _host_tensor_numel(spec: DmaTapSpec) -> int:
    return prod(spec.tensor_dims)


def _host_tensor_last_index(spec: DmaTapSpec) -> int:
    return spec.offset + sum(
        (size - 1) * stride for size, stride in zip(spec.sizes, spec.strides)
    )


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_gemv_dma_specs_reference_only_inside_host_tensor(shape) -> None:
    """Every explicit shim tap points at offsets that exist in the host BO.

    For multi-slot X shapes (currently only MoE B2-9) the inactive trailing
    columns reuse slot 0 so they still receive the barrier/release X fill.
    """
    specs = gemv_dma_tap_specs(shape)
    assert len(specs["x"]) == COLS
    for group in ("x", "w_host", "y"):
        for spec in specs[group]:
            numel = _host_tensor_numel(spec)
            assert 0 <= spec.offset < numel, (
                f"{spec.name}: offset {spec.offset} not in [0, {numel}) "
                f"for tensor_dims {spec.tensor_dims}"
            )
            last = _host_tensor_last_index(spec)
            assert last < numel, (
                f"{spec.name}: last referenced index {last} not in [0, {numel})"
            )
    assert len({spec.offset for spec in specs["x"]}) == shape.x_slots


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_every_unified_shape_bd_fits_target_model(shape) -> None:
    for spec in _flatten(gemv_dma_tap_specs(shape)):
        _verify_bd(spec)


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_every_per_call_shim_tap_fits_target_model(shape) -> None:
    for entries in per_call_shim_tap_specs(shape).values():
        for spec in _flatten(entries):
            _verify_bd(spec)


@pytest.mark.parametrize(
    "words_per_row",
    [W_BLOCK_BYTES // 4, Q4_BLOCK_BYTES // 4],
    ids=["bfp16ebs8", "q4nx"],
)
def test_every_short_dma_experiment_bd_fits_target_model(words_per_row: int) -> None:
    for spec in _flatten(short_dma_tap_specs(words_per_row)):
        _verify_bd(spec)


def test_q4_memory_receive_uses_preserved_hardware_safe_segments() -> None:
    q4_shapes = [shape for shape in ALL_SHAPES if shape.format == FORMAT_Q4NX]
    assert q4_shapes
    for shape in q4_shapes:
        spec = gemv_dma_tap_specs(shape)["mem_in"][0]
        assert spec.sizes == (4, 5, 512)
        assert spec.strides == (4608, 512, 1)
        assert spec.preserve_dimensions

    exp = short_dma_tap_specs(Q4_BLOCK_BYTES // 4)["mem_in"][0]
    assert exp.sizes == (4, 5, 256)
    assert exp.strides == (2304, 256, 1)
    assert exp.preserve_dimensions


def test_full_memory_receive_is_one_contiguous_line() -> None:
    full_shape = next(shape for shape in ALL_SHAPES if shape.format == FORMAT_BFP16)
    spec = gemv_dma_tap_specs(full_shape)["mem_in"][0]
    assert spec.sizes == (4 * 4608,)
    assert spec.strides == (1,)
    assert not spec.preserve_dimensions

    exp = short_dma_tap_specs(W_BLOCK_BYTES // 4)["mem_in"][0]
    assert exp.sizes == (4 * 2304,)
    assert exp.strides == (1,)
    assert not exp.preserve_dimensions


def test_exp_compile_only_exits_before_callable_or_npu(monkeypatch) -> None:
    from bench.gemv_unified import exp_short_dma

    calls = []

    def fake_run(iters: int, *, compile_only: bool = False) -> int:
        calls.append((iters, compile_only))
        return 0

    monkeypatch.setattr(exp_short_dma, "run_npu_experiment", fake_run)
    assert exp_short_dma.main(["--compile-only", "--iters", "17"]) == 0
    assert calls == [(17, True)]


def test_stage2_compile_only_uses_compile_gate_only(monkeypatch) -> None:
    from bench.gemv_unified import stage2_bench

    calls = []
    monkeypatch.setattr(
        stage2_bench, "compile_all_artifacts", lambda: calls.append(True) or 0
    )
    assert stage2_bench.main(["--compile-only"]) == 0
    assert calls == [True]


def test_stage1_compile_only_uses_compile_gate_only(monkeypatch) -> None:
    from bench.gemv_unified import unified_bench

    calls = []
    monkeypatch.setattr(
        unified_bench, "compile_all_artifacts", lambda cols: calls.append(cols) or 0
    )
    assert unified_bench.main(["--compile-only"]) == 0
    assert calls == [8]


# ---------------------------------------------------------------------------
# Runtime-plan simultaneous + static BD budget per tile (host mirror of
# ``aiex.dma_configure_task`` "Too many simultaneously active buffer
# descriptors", AIEAssignRuntimeSequenceBDIDs.cpp).  The runtime numbers are
# produced by replaying the same planner/driver that design.py executes.
# ---------------------------------------------------------------------------

from kernels.gemv_unified import w_plan  # noqa: E402
from kernels.gemv_unified.shapes import get_shape  # noqa: E402

MEM_PARITY_BDS = 24  # AIETargetModel.h:489 / :922 (even ch < 24, odd >= 24)
CORE_BDS = 16
SHIM_BDS = 16
# Static (ObjectFifo) BDs on the memory tile.  y join: four S2MM channels of
# depth 2; the joined MM2S out channel is counted pessimistically as the sum of
# its four sub-FIFOs (4 x depth 2) -- an estimate, not read from compiled MLIR.
MEM_Y_IN_STATIC = 2
MEM_Y_OUT_STATIC = 8
CORE_STATIC = 1 + 2  # x consumer depth 1 + y out depth 2


def _mem_parity_group_bound(usage: "w_plan.BdUsage") -> int:
    """Worst BD demand of one parity group (3 even channels per direction).

    Channel numbers are assigned by the compiler, so bound it by putting the
    three heaviest channels of each direction into the same group.
    """
    s2mm = sorted([usage.mem_in_peak] + [MEM_Y_IN_STATIC] * 4, reverse=True)
    mm2s = sorted([usage.mem_out_peak] * 4 + [MEM_Y_OUT_STATIC], reverse=True)
    return sum(s2mm[:3]) + sum(mm2s[:3])


def _budget_violations(shape, *, continuous: bool) -> list[str]:
    usage = w_plan.simulate_bd_usage(shape, continuous=continuous)
    bad = []
    mem = _mem_parity_group_bound(usage)
    if mem > MEM_PARITY_BDS:
        bad.append(f"mem {mem} > {MEM_PARITY_BDS}")
    if usage.core_w_peak + CORE_STATIC > CORE_BDS:
        bad.append(f"core {usage.core_w_peak + CORE_STATIC} > {CORE_BDS}")
    if usage.shim_peak > SHIM_BDS:
        bad.append(f"shim {usage.shim_peak} > {SHIM_BDS}")
    return bad


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_every_shape_bd_budget_fits_continuous_mode(shape) -> None:
    assert _budget_violations(shape, continuous=True) == []


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_every_shape_bd_budget_fits_default_per_round_mode(shape) -> None:
    # Odd-block shapes use a one-round window; even-block shapes retain depth 2.
    assert _budget_violations(shape, continuous=False) == []


@pytest.mark.parametrize("name", ["q_b", "dense_down", "moe_b2_9"])
def test_odd_per_round_plan_would_overflow_with_depth_two(name: str) -> None:
    shape = get_shape(name)
    usage = w_plan.simulate_bd_usage(shape, continuous=False, depth=2)
    assert _mem_parity_group_bound(usage) > MEM_PARITY_BDS


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_mem_chunks_cover_blocks_exactly_once_in_slot_order(shape) -> None:
    for col, chunks in w_plan.mem_chunk_plan(shape, continuous=True).items():
        rounds = shape.rounds_by_col()[col]
        total = rounds * shape.k_blocks
        nxt = 0
        for chunk in chunks:
            assert chunk.first_block == nxt
            assert chunk.pairs <= MAX_REPEAT_COUNT
            assert chunk.first_slot == nxt % 2
            assert chunk.first_round <= chunk.last_round < rounds
            nxt += chunk.blocks
        assert nxt == total
        # at most one tail, and only at the very end (odd total blocks)
        assert [c.is_tail for c in chunks].count(True) == total % 2
        if chunks and chunks[-1].is_tail:
            assert chunks[-1].first_slot == 0


def test_lm_head_splits_into_repeat_limited_tasks() -> None:
    shape = get_shape("lm_head")
    chunks = w_plan.mem_chunk_plan(shape, continuous=True)[0]
    assert len(chunks) == 5 and all(c.pairs <= MAX_REPEAT_COUNT for c in chunks)
    # A chunk is freed only after the round of its last block; no overlap
    # beyond the two-round window keeps <= 2 chunks live per channel.
    assert w_plan.simulate_bd_usage(shape, continuous=True).mem_out_peak <= 4


@pytest.mark.parametrize("name", ["f_b", "q_b", "dense_down", "moe_b2_9"])
def test_default_per_round_plan_restarts_at_ping_and_uses_safe_window(name: str) -> None:
    shape = get_shape(name)
    assert shape.k_blocks % 2 == 1
    chunks = w_plan.mem_chunk_plan(shape, continuous=False)[0]
    assert all(chunk.first_slot == 0 for chunk in chunks)
    assert w_plan.round_window_depth(shape.k_blocks, continuous=False) == 1


def test_even_per_round_plan_keeps_depth_two() -> None:
    shape = get_shape("k_b_v_b")
    assert shape.k_blocks == 2
    assert w_plan.round_window_depth(shape.k_blocks, continuous=False) == 2


PER_CALL_SHAPES = (
    "f_b",
    "q_b",
    "k_b_v_b",
    "dense_gu",
    "moe_b2_9",
)


def _per_call_intervals(shape, kind: str) -> list[tuple[int, int]]:
    intervals = []
    for entries in per_call_shim_tap_specs(shape).values():
        for spec in entries[kind]:
            run_count = spec.sizes[0]
            run_length = prod(spec.sizes[1:])
            intervals.extend(
                (spec.offset + run * spec.strides[0], run_length)
                for run in range(run_count)
            )
    return sorted(intervals)


@pytest.mark.parametrize("name", PER_CALL_SHAPES)
def test_per_call_shim_ranges_equal_per_round_ranges(name: str) -> None:
    shape = get_shape(name)
    baseline = gemv_dma_tap_specs(shape)
    for kind, baseline_kind in (("w", "w_host"), ("y", "y")):
        before = sorted((spec.offset, prod(spec.sizes)) for spec in baseline[baseline_kind])
        assert _per_call_intervals(shape, kind) == before


@pytest.mark.parametrize("name", PER_CALL_SHAPES)
def test_per_call_bd_budget_fits_selected_shapes(name: str) -> None:
    shape = get_shape(name)
    taps = per_call_shim_tap_specs(shape)
    usage = w_plan.simulate_per_call_bd_usage(
        shape,
        shim_w_bds_by_col={col: len(entries["w"]) for col, entries in taps.items()},
        shim_y_bds_by_col={col: len(entries["y"]) for col, entries in taps.items()},
    )
    assert usage.core_w_peak + CORE_STATIC <= CORE_BDS
    assert usage.shim_peak <= SHIM_BDS
    assert _mem_parity_group_bound(usage) <= MEM_PARITY_BDS


@pytest.mark.parametrize("name", ["f_b", "q_b", "moe_b2_9"])
def test_odd_per_call_core_chain_restarts_at_ping_and_covers_every_block(name: str) -> None:
    shape = get_shape(name)
    for rounds in shape.rounds_by_col():
        if not rounds:
            continue
        chunks = w_plan.per_call_core_chunks(shape.k_blocks, rounds)
        actual = [slot for chunk in chunks for _ in range(chunk.runs) for slot in chunk.slots]
        expected = [block % 2 for _ in range(rounds) for block in range(shape.k_blocks)]
        assert actual == expected
        assert sum(chunk.bds for chunk in chunks) <= CORE_BDS


def test_dense_down_is_not_a_legal_per_call_core_chain() -> None:
    shape = get_shape("dense_down")
    with pytest.raises(ValueError, match="needs 47 BDs"):
        w_plan.per_call_core_chunks(shape.k_blocks, shape.rounds_by_col()[0])


def test_b1_per_call_layout_is_derived_without_changing_baseline() -> None:
    from kernels.gemv_unified.op import UnifiedGEMV

    baseline = get_shape("moe_b1_6")
    assert baseline.active_cols is None
    assert baseline.chunks_per_col is None
    assert max(baseline.rounds_by_col(7)) == 15
    per_call = UnifiedGEMV.for_shape("moe_b1_6", cols=7, sequence_plan="per_call")
    assert per_call.shape.active_cols == 6
    assert per_call.shape.chunks_per_col == 17
    assert max(per_call.shape.rounds_by_col(7)) == 17
    taps = per_call_shim_tap_specs(per_call.shape, 7)
    assert sum(len(entries["w"]) for entries in taps.values()) == 6


RESOURCE_TEST_SHAPES = (
    UnifiedShape("one_block_one_col", 256, 128, active_cols=1),
    UnifiedShape("one_block_eight_cols", 256, 1024),
    UnifiedShape("two_blocks_eight_cols", 512, 5120),
    UnifiedShape("three_blocks_one_col", 768, 128, active_cols=1),
    UnifiedShape("uneven_rounds", 2560, 1152),
    UnifiedShape("max_k", MAX_K, 5120),
)


@pytest.mark.parametrize("continuous", [False, True], ids=["per_round", "continuous"])
def test_memtile_device_resources_are_shape_invariant_and_cover_every_plan(
    continuous: bool,
) -> None:
    reference = w_plan.memtile_resource_plan(
        RESOURCE_TEST_SHAPES[0], continuous=continuous
    )
    declared_names = {resource.name for resource in reference.declared}
    assert len(reference.declared) == COLS * w_plan.DEPTH * 3
    assert len(declared_names) == len(reference.declared)

    for shape in RESOURCE_TEST_SHAPES:
        plan = w_plan.memtile_resource_plan(shape, continuous=continuous)
        assert plan.declared == reference.declared
        assert plan.used_names <= declared_names

    buffers = [resource for resource in reference.declared if resource.kind == "buffer"]
    empty = [resource for resource in reference.declared if "_empty_" in resource.name]
    full = [resource for resource in reference.declared if "_full_" in resource.name]
    assert {resource.elements for resource in buffers} == {4 * 4608}
    assert {resource.init for resource in empty} == {ROWS}
    assert {resource.init for resource in full} == {0}
