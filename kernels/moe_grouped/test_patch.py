# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""NPU-free DDR_PATCH address tests against the decode layout SSOT."""

from __future__ import annotations

import numpy as np
import pytest

from kernels.moe.pack import (
    get_down_bfp16_chunk_offset_bytes,
    get_gate_up_chunk_offset_bytes,
)
from kernels.moe.patch import (
    TXN_OPC_DDR_PATCH,
    TXN_OPC_WRITE,
    HostRuntimeError,
    find_ddr_patches,
)
from .layout import B1, B2, COLS, ROWS, ROWS_WORD_TEMPLATE, SLOTS, stream_plan
from .patch import GroupedPatchedTemplateKernel, find_rows_words, pack_rows


def _write_op(addr: int, value: int, size_bytes: int = 24) -> list[int]:
    return [
        TXN_OPC_WRITE,
        0,
        addr & 0xFFFFFFFF,
        0,
        value & 0xFFFFFFFF,
        size_bytes,
    ]


def _core_ops(shape, col: int, row: int) -> list[int]:
    tile = (col << 25) | (row << 20)
    # RTP layout: words 5-6 right before word 0, words 0-3 right before the rows word (4), barrier after.
    rtp_values = (
        (5, 1 if shape.balanced else 0),
        (6, col),
        (0, shape.k_blocks),
        (1, shape.rounds_by_col[col]),
        (2, shape.mode),
        (3, SLOTS),
        (4, ROWS_WORD_TEMPLATE),
    )
    ops: list[int] = []
    for word, value in rtp_values:
        ops.extend(_write_op(tile | (0x9800 + 4 * word), value))
    ops.extend(_write_op(tile | 0x9820, 1))
    return ops


def _core_ops_without_balanced_rtp(shape, col: int, row: int) -> list[int]:
    """Words 0-4 + barrier with two padding writes before word 0 (no writes at word 5/6 addresses)."""
    tile = (col << 25) | (row << 20)
    ops: list[int] = []
    # Padding writes keep the rows-word far enough into the op list that
    # find_rows_words reaches the RTP word 5/6 check, where it then fails
    # because these addresses are not the expected word-5/word-6 addresses.
    ops.extend(_write_op(tile | 0x9700, 0))
    ops.extend(_write_op(tile | 0x9704, 0))
    for word, value in enumerate((
        shape.k_blocks, shape.rounds_by_col[col], shape.mode, SLOTS,
        ROWS_WORD_TEMPLATE,
    )):
        ops.extend(_write_op(tile | (0x9800 + 4 * word), value))
    ops.extend(_write_op(tile | 0x9820, 1))
    return ops


def _template_without_balanced_rtp(shape) -> np.ndarray:
    words = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            words.extend(_core_ops_without_balanced_rtp(shape, col, row))
    words.extend(_ddr_patch_ops(shape))
    return _finalize_txn(words)


def _ddr_patch_ops(shape) -> list[int]:
    """W patches in stream_plan order (balanced shapes put the trailing chunks last)."""
    ops: list[int] = []
    for rec in stream_plan(shape):
        op = [0] * 12
        op[0] = TXN_OPC_DDR_PATCH
        op[1] = 48
        op[8] = 1
        offset = rec["chunk"] * shape.chunk_bytes
        op[10] = offset & 0xFFFFFFFF
        op[11] = offset >> 32
        ops.extend(op)
    return ops


def _finalize_txn(words: list[int]) -> np.ndarray:
    words[3] = len(words) * 4
    return np.asarray(words, dtype=np.uint32)


def _old_ddr_patch_ops(shape) -> list[int]:
    """Pre-section-49 slot-major order; used only to test rejection."""
    ops: list[int] = []
    for _slot in range(8):
        for chunk in range(shape.chunks):
            op = [0] * 12
            op[0] = TXN_OPC_DDR_PATCH
            op[1] = 48
            op[8] = 1
            offset = chunk * shape.chunk_bytes
            op[10] = offset & 0xFFFFFFFF
            op[11] = offset >> 32
            ops.extend(op)
    return ops


def _template(shape) -> np.ndarray:
    words = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            words.extend(_core_ops(shape, col, row))
    words.extend(_ddr_patch_ops(shape))
    return _finalize_txn(words)


@pytest.mark.parametrize("shape", [B1, B2])
def test_grouped_patch_count_and_decode_offsets(shape) -> None:
    kernel = GroupedPatchedTemplateKernel.from_words(_template(shape), shape)
    experts = (0, 1, 7, 31, 63, 95, 127, 128 if shape is B2 else 5)
    args = {f"e{i}": expert for i, expert in enumerate(experts)}
    patched = kernel.instructions(**args)
    patches = find_ddr_patches(patched, target_arg_idx=1)
    assert len(patches) == shape.patch_count
    for (_pos, _arg, offset), rec in zip(patches, stream_plan(shape)):
        slot, chunk = rec["slot"], rec["chunk"]
        expected = (
            get_gate_up_chunk_offset_bytes(experts[slot], chunk)
            if shape is B1
            else get_down_bfp16_chunk_offset_bytes(experts[slot], chunk)
        )
        assert offset == expected


def test_b1_rejects_shared_expert() -> None:
    kernel = GroupedPatchedTemplateKernel.from_words(_template(B1), B1)
    with pytest.raises(ValueError, match="outside"):
        kernel.instructions(**{f"e{i}": 128 if i == 7 else i for i in range(8)})


def test_b1_rows_patching_and_default_template_preservation() -> None:
    template = _template(B1)
    kernel = GroupedPatchedTemplateKernel.from_words(template.copy(), B1)
    assert kernel.dispatch_params == [f"e{i}" for i in range(8)] + [
        f"r{i}" for i in range(8)
    ]
    assert GroupedPatchedTemplateKernel.from_words(
        _template(B2), B2
    ).dispatch_params == [f"e{i}" for i in range(8)]

    experts = {f"e{i}": i + 1 for i in range(8)}
    rows = {f"r{i}": i for i in range(8)}
    patched_with_r = kernel.instructions(**experts, **rows)
    assert len(kernel._rows_pos) == 32
    assert np.all(patched_with_r[kernel._rows_pos] == np.uint32(0x76543210))

    changed = set(np.flatnonzero(patched_with_r != template).tolist())
    ddr_words = {
        pos + delta
        for pos, _slot, _chunk in kernel._entries
        for delta in (10, 11)
    }
    assert changed <= ddr_words | set(kernel._rows_pos.tolist())
    assert set(kernel._rows_pos.tolist()) <= changed

    patched_without_r = kernel.instructions(**experts)
    assert np.all(
        patched_without_r[kernel._rows_pos] == np.uint32(ROWS_WORD_TEMPLATE)
    )
    assert np.array_equal(kernel._template_words, template)


@pytest.mark.parametrize("shape", [B1, B2])
def test_init_rejects_invalid_templates(shape) -> None:
    base = _template(shape)
    pos = find_rows_words(base, shape)

    # 1. 31 rows-word writes
    bad_31 = base.copy()
    bad_31[pos[0]] = np.uint32(0x88888887)
    with pytest.raises(ValueError, match="31"):
        GroupedPatchedTemplateKernel.from_words(bad_31, shape)

    # 2. One RTP address misaligned
    bad_addr = base.copy()
    bad_addr[pos[3] - 4 - 12 + 2] ^= np.uint32(0x4)
    with pytest.raises(ValueError, match="RTP word"):
        GroupedPatchedTemplateKernel.from_words(bad_addr, shape)

    # 3. Preceding RTP value mismatch (k_blocks)
    bad_k = base.copy()
    bad_k[pos[5] - 24] = np.uint32(shape.k_blocks + 1)
    with pytest.raises(ValueError, match="RTP word 0"):
        GroupedPatchedTemplateKernel.from_words(bad_k, shape)

    # 4. Missing barrier write (value != 1)
    bad_barrier = base.copy()
    bad_barrier[pos[2] + 6] = np.uint32(0)
    with pytest.raises(ValueError, match="barrier"):
        GroupedPatchedTemplateKernel.from_words(bad_barrier, shape)

    # 5. Rows-word write after first DDR_PATCH
    after_words = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            if (col, row) != (7, 5):
                after_words.extend(_core_ops(shape, col, row))
    after_words.extend(_ddr_patch_ops(shape))
    after_words.extend(_core_ops(shape, 7, 5))
    with pytest.raises(ValueError, match="31"):
        GroupedPatchedTemplateKernel.from_words(_finalize_txn(after_words), shape)

    # 6. Duplicate core
    dup_words = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            c, r = (0, 2) if (col, row) == (7, 5) else (col, row)
            dup_words.extend(_core_ops(shape, c, r))
    dup_words.extend(_ddr_patch_ops(shape))
    with pytest.raises(ValueError, match="distinct cores"):
        GroupedPatchedTemplateKernel.from_words(_finalize_txn(dup_words), shape)

    # 7. Column 8 (out of range -> ValueError, not IndexError)
    col8 = base.copy()
    for step in range(-4, 2):
        addr_idx = int(pos[0]) - 4 + step * 6 + 2
        col8[addr_idx] = np.uint32(
            (int(col8[addr_idx]) & ~(0x7F << 25)) | (8 << 25)
        )
    with pytest.raises(ValueError, match="out-of-range core"):
        GroupedPatchedTemplateKernel.from_words(col8, shape)

    # 8. Row 6 tile (out of range -> ValueError)
    row6 = base.copy()
    for step in range(-4, 2):
        addr_idx = int(pos[0]) - 4 + step * 6 + 2
        row6[addr_idx] = np.uint32(
            (int(row6[addr_idx]) & ~(0x1F << 20)) | (6 << 20)
        )
    with pytest.raises(ValueError, match="out-of-range core"):
        GroupedPatchedTemplateKernel.from_words(row6, shape)

    # 9. Corrupted barrier size word (25 bytes -> sz == 6 in parse_txn_ops, but != 24)
    bad_barrier_sz = base.copy()
    bad_barrier_sz[pos[1] + 7] = np.uint32(25)
    with pytest.raises(ValueError, match="barrier"):
        GroupedPatchedTemplateKernel.from_words(bad_barrier_sz, shape)

    # 10. Extra 0x88888888 after DDR_PATCH is ignored when 32 valid precede it
    extra_after = list(base.astype(int))
    extra_after.extend(_write_op(0, ROWS_WORD_TEMPLATE))
    kernel_extra = GroupedPatchedTemplateKernel.from_words(
        _finalize_txn(extra_after), shape
    )
    assert np.array_equal(kernel_extra._rows_pos, pos)

    # 11. No DDR_PATCH operations at all
    no_ddr = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            no_ddr.extend(_core_ops(shape, col, row))
    no_ddr_arr = _finalize_txn(no_ddr)
    with pytest.raises(ValueError, match="DDR_PATCH"):
        find_rows_words(no_ddr_arr, shape)
    with pytest.raises(ValueError):
        GroupedPatchedTemplateKernel.from_words(no_ddr_arr, shape)


@pytest.mark.parametrize("shape", [B1, B2])
def test_find_rows_words_rejects_bad_balanced_rtp_words(shape) -> None:
    """RTP words 5-6 must sit right before word 0 with the balanced-shape values."""
    base = _template(shape)
    pos = find_rows_words(base, shape)
    rows_word_pos = int(pos[0])

    # RTP layout in the template is word 5, word 6, word 0, word 1, word 2, word 3, word 4 (rows).
    word5_value_pos = rows_word_pos - 36
    word6_value_pos = rows_word_pos - 30
    word5_addr_pos = word5_value_pos - 2

    # word 5 (balanced flag) or word 6 (column/slot) has wrong value
    bad_word5 = base.copy()
    bad_word5[word5_value_pos] = np.uint32(0xDEADBEEF)
    with pytest.raises(ValueError, match="RTP word 5"):
        find_rows_words(bad_word5, shape)

    bad_word6 = base.copy()
    bad_word6[word6_value_pos] = np.uint32(0xDEADBEEF)
    with pytest.raises(ValueError, match="RTP word 6"):
        find_rows_words(bad_word6, shape)

    # word 5 at wrong address
    bad_addr = base.copy()
    bad_addr[word5_addr_pos] ^= np.uint32(0x4)
    with pytest.raises(ValueError, match="RTP word 5"):
        find_rows_words(bad_addr, shape)

    # balanced-value mismatch: B1 expects word5==1, B2 expects word5==0
    wrong_balanced = base.copy()
    wrong_balanced[word5_value_pos] = np.uint32(0 if shape.balanced else 1)
    with pytest.raises(ValueError, match="RTP word 5"):
        find_rows_words(wrong_balanced, shape)

    # word 6 must equal the column; corrupt all columns to a different constant
    wrong_col = base.copy()
    for p in pos:
        wrong_col[int(p) - 30] = np.uint32(7)
    with pytest.raises(ValueError, match="RTP word 6"):
        find_rows_words(wrong_col, shape)

    # words 5-6 after the rows word instead of before word 0
    scrambled = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            tile = (col << 25) | (row << 20)
            # write 0-3, rows, barrier, then 5-6 (wrong place)
            for word, value in enumerate((
                shape.k_blocks, shape.rounds_by_col[col], shape.mode, SLOTS,
                ROWS_WORD_TEMPLATE,
            )):
                scrambled.extend(_write_op(tile | (0x9800 + 4 * word), value))
            scrambled.extend(_write_op(tile | 0x9820, 1))
            scrambled.extend(_write_op(tile | (0x9800 + 4 * 5), 1 if shape.balanced else 0))
            scrambled.extend(_write_op(tile | (0x9800 + 4 * 6), col))
    scrambled.extend(_ddr_patch_ops(shape))
    with pytest.raises(ValueError):
        find_rows_words(_finalize_txn(scrambled), shape)


@pytest.mark.parametrize("shape", [B1, B2])
def test_find_rows_words_rejects_missing_balanced_rtp_words(shape) -> None:
    """A template that writes words 0-4 and the barrier but omits 5-6 must be rejected."""
    template = _template_without_balanced_rtp(shape)
    with pytest.raises(ValueError, match="RTP word|rows-word write|unexpected form"):
        find_rows_words(template, shape)


def test_call_rejects_invalid_row_scalars_and_preserves_template() -> None:
    template_b1 = _template(B1)
    kernel_b1 = GroupedPatchedTemplateKernel.from_words(template_b1.copy(), B1)
    kernel_b2 = GroupedPatchedTemplateKernel.from_words(_template(B2), B2)
    experts = {f"e{i}": i for i in range(8)}
    valid_rows = {f"r{i}": 4 for i in range(8)}

    for bad_r in (9, -1, True, 1.5, np.bool_(True)):
        with pytest.raises(ValueError):
            kernel_b1.instructions(**experts, **{**valid_rows, "r3": bad_r})

    # Partial rows (r0..r6)
    partial_rows = {f"r{i}": 4 for i in range(7)}
    with pytest.raises(HostRuntimeError):
        kernel_b1.instructions(**experts, **partial_rows)

    # B2 with rows
    with pytest.raises(HostRuntimeError):
        kernel_b2.instructions(**experts, **valid_rows)

    # Unknown scalar name
    with pytest.raises(HostRuntimeError):
        kernel_b1.instructions(**experts, **valid_rows, extra=1)
    with pytest.raises(HostRuntimeError):
        kernel_b1.instructions(**experts, extra=1)

    assert np.array_equal(kernel_b1._template_words, template_b1)
    kernel_b1.instructions(**experts, **valid_rows)
    assert np.array_equal(kernel_b1._template_words, template_b1)


def test_non_canonical_b1_w_patch_order_is_rejected() -> None:
    """B1's old slot-major order puts chunk 16 in the middle of each slot, not at the end."""
    words = [0, 0, 0, 0]
    for col in range(COLS):
        for row in range(2, ROWS + 2):
            words.extend(_core_ops(B1, col, row))
    words.extend(_old_ddr_patch_ops(B1))
    with pytest.raises(ValueError, match="non-canonical patch order"):
        GroupedPatchedTemplateKernel.from_words(_finalize_txn(words), B1)


def test_pack_rows_boundaries_and_types() -> None:
    assert pack_rows([0] * 8) == 0
    assert pack_rows([8] * 8) == ROWS_WORD_TEMPLATE
    assert pack_rows([np.int32(i) for i in range(8)]) == 0x76543210
    assert pack_rows([0, 0, 0, 0, 0, 0, 0, 8]) == 0x80000000

    kernel = GroupedPatchedTemplateKernel.from_words(_template(B1), B1)
    experts = {f"e{i}": i for i in range(8)}
    slot7_only = {f"r{i}": (8 if i == 7 else 0) for i in range(8)}
    patched = kernel.instructions(**experts, **slot7_only)
    assert patched.dtype == np.uint32
    assert np.all(patched[kernel._rows_pos] == np.uint32(0x80000000))

    for bad_rows in (
        [0] * 7,
        [0] * 9,
        [-1] + [0] * 7,
        [9] + [0] * 7,
        [True] + [0] * 7,
        [np.bool_(False)] + [0] * 7,
        [1.5] + [0] * 7,
    ):
        with pytest.raises(ValueError):
            pack_rows(bad_rows)  # type: ignore[arg-type]
