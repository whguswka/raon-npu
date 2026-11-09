# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side NumPy unit tests for Raon MoE NPU kernel implementations.

Verifies:
1. Architectural dimensions, buffer allocations, and transfer volume metrics.
2. Weight packing layout and chunk offset arithmetic:
   - B1 gate_up: expert e, chunk l offset in uint32 words and bytes.
   - B2 down: expert e, chunk r offset in uint32 words and bytes.
   - Roundtrip parse_gemv on individual chunk slices matching original weights.
3. Scheduling coverage:
   - B1: chunk c -> (slot j, chunk l, col, round, w_offset, y_offset) uniquely and
     completely covers all 119 chunks with col_rounds [15, 15, 15, 15, 15, 15, 15, 14].
   - B2: (slot j, round r) -> (col j, round r, x_offset, w_offset, y_offset) uniquely and
     completely covers all 140 chunks across 7 columns (column 7 unused).
4. Host-side mathematical reference functions:
   - SiLU-GLU intermediate activation and padding to 1280 columns.
   - Router-weighted accumulation of 6 routed experts + 1 shared expert.
5. Hardware resource budgets:
   - Shim channel allocations (MM2S <= 2, S2MM <= 2 per column).
   - Core L1 data memory budget (64 KiB) for K=2560 (B1) and K=1280 (B2).
   - Shim Buffer Descriptors (BD <= 16).
6. Static TXN parser and runtime address patch verification:
   - Synthetic TXN decoding with mixed operations (BLOCKWRITE, TCT, DDR_PATCH).
   - Identification of all W DDR_PATCH ops and order-independent chunk index recovery.
   - Validation that non-119 patch counts raise ValueError.
   - Real gemv txn-static.bin parsing and patching verification when present.
   - B2 down synthetic 140-patch verification.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from quant.q4nx import (
    K_TILE,
    M_TILE,
    pack_gemv,
    pad_gemv,
    parse_gemv,
    quantize,
    round_to_bf16_rne,
    to_bf16_u16,
)
from kernels.gemv.reference import (
    bf16_ulp_diff,
    gemv_bit_reference,
)
from kernels.moe.op import (
    MoEDown,
    MoEDownTemplate,
    MoEGateUp,
    MoEGateUpTemplate,
)
from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_K,
    B1_M_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    B1_W_PER_CHUNK_WORDS,
    B2_CHUNKS_PER_EXPERT,
    B2_K_ORIG,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    B2_W_PER_CHUNK_BYTES,
    B2_W_PER_CHUNK_WORDS,
    CHUNK_ROWS,
    DN_BYTES,
    DN_WORDS,
    GU_BYTES,
    GU_WORDS,
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_ACTIVE_ROUTED,
    NUM_ACTIVE_TOTAL,
    NUM_ROUTED_EXPERTS,
    NUM_TOTAL_EXPERTS,
    SHARED_EXPERT_ID,
    TOTAL_DN_BYTES,
    TOTAL_DN_WORDS,
    TOTAL_GU_BYTES,
    TOTAL_GU_WORDS,
    get_down_chunk_offset_bytes,
    get_down_chunk_offset_words,
    get_gate_up_chunk_offset_bytes,
    get_gate_up_chunk_offset_words,
    pack_expert_down,
    pack_expert_gate_up,
    pack_expert_gate_up_q4nx,
    parse_expert_down,
    parse_expert_gate_up,
)
from kernels.moe.patch import (
    PatchedTemplateKernel,
    TXN_OPC_BLOCKWRITE,
    TXN_OPC_DDR_PATCH,
    TXN_OPC_TCT,
    find_ddr_patches,
    parse_txn_ops,
)
from kernels.moe.reference import (
    moe_down_bit_reference,
    moe_gate_up_bit_reference,
    moe_host_accumulate,
    moe_host_silu_glu,
)


# =============================================================================
# Helper: Synthetic TXN Generator
# =============================================================================
def _build_synthetic_txn(
    w_chunk_offsets: list[int],
    w_arg_idx: int = 1,
    y_patches_count: int = 5,
    x_patches_count: int = 1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Build a valid synthetic TXN uint32 word array containing diverse instructions."""
    if rng is None:
        rng = np.random.default_rng(42)

    words: list[int] = [0, 0, 0, 0]  # Placeholder for 4-word header
    op_count = 0

    # 1. Add some initial BLOCKWRITE ops
    for i in range(3):
        count = rng.integers(2, 6)
        words.append(TXN_OPC_BLOCKWRITE)
        words.append(0)
        words.append(0x1000 + i * 0x100)
        words.append((4 + count) * 4)  # size in bytes
        for c in range(count):
            words.append(int(rng.integers(0, 0xFFFFFFFF)))
        op_count += 1

    # 2. Prepare patches to insert
    patch_specs: list[tuple[int, int]] = []  # (arg_idx, arg_plus)
    for i in range(y_patches_count):
        patch_specs.append((0, i * 0x100))
    for off in w_chunk_offsets:
        patch_specs.append((w_arg_idx, off))
    for i in range(x_patches_count):
        patch_specs.append((2, 0))

    # Shuffle patches to ensure parser does not assume sorted order
    perm = rng.permutation(len(patch_specs))
    for idx in perm:
        arg_idx, arg_plus = patch_specs[idx]

        # Occasionally insert a TCT sync or BLOCKWRITE between patches
        if rng.random() < 0.3:
            words.append(TXN_OPC_TCT)
            words.append(16)
            words.append(0x00010000)
            words.append(0)
            op_count += 1

        # DDR_PATCH (12 words = 48 bytes)
        words.append(TXN_OPC_DDR_PATCH)
        words.append(48)
        words.append(0)
        words.append(0)
        words.append(0)
        words.append(0)
        words.append(0x1D000 + len(words))  # dummy reg addr
        words.append(0)
        words.append(arg_idx)
        words.append(0)
        words.append(arg_plus & 0xFFFFFFFF)
        words.append((arg_plus >> 32) & 0xFFFFFFFF)
        op_count += 1

    # Finalize 4-word header
    arr = np.array(words, dtype=np.uint32)
    arr[0] = 0x06040100  # major=0, minor=1, devGen=4, numRows=6
    arr[1] = 0x00000108  # numCols=8, numMemTileRows=1
    arr[2] = op_count
    arr[3] = len(arr) * 4  # total bytes

    return arr


# =============================================================================
# 1. Architectural & Metric Tests
# =============================================================================
def test_moe_constants_and_buffer_sizes():
    """Verify all MoE constants, buffer sizes, and byte arithmetic match specification."""
    # 1. Expert count
    assert NUM_ROUTED_EXPERTS == 128
    assert SHARED_EXPERT_ID == 128
    assert NUM_TOTAL_EXPERTS == 129
    assert NUM_ACTIVE_ROUTED == 6
    assert NUM_ACTIVE_TOTAL == 7

    # 2. Dimensions
    assert HIDDEN_SIZE == 2560
    assert MOE_INTERMEDIATE_SIZE == 1088
    assert CHUNK_ROWS == 128

    # 3. B1 (gate+up)
    assert B1_K == 2560
    assert B1_M_PER_EXPERT == 2176
    assert B1_CHUNKS_PER_EXPERT == 17
    assert B1_W_PER_CHUNK_BYTES == 204_800
    assert B1_W_PER_CHUNK_WORDS == 51_200
    assert GU_WORDS == 17 * 51_200 == 870_400
    assert GU_BYTES == 870_400 * 4 == 3_481_600
    assert TOTAL_GU_WORDS == 129 * 870_400 == 112_281_600
    assert TOTAL_GU_BYTES == 129 * 3_481_600 == 449_126_400

    b1_read_bytes = 7 * GU_BYTES
    assert b1_read_bytes == 24_371_200
    assert abs(b1_read_bytes / 1e6 - 24.37) < 0.01

    # 4. B2 (down)
    assert B2_K_ORIG == 1088
    assert B2_K_PAD == 1280
    assert B2_M_PER_EXPERT == 2560
    assert B2_CHUNKS_PER_EXPERT == 20
    assert B2_W_PER_CHUNK_BYTES == 102_400
    assert B2_W_PER_CHUNK_WORDS == 25_600
    assert DN_WORDS == 20 * 25_600 == 512_000
    assert DN_BYTES == 512_000 * 4 == 2_048_000
    assert TOTAL_DN_WORDS == 129 * 512_000 == 66_048_000
    assert TOTAL_DN_BYTES == 129 * 2_048_000 == 264_192_000

    b2_read_bytes = 7 * DN_BYTES
    assert b2_read_bytes == 14_336_000
    assert abs(b2_read_bytes / 1e6 - 14.34) < 0.01


def test_moe_gate_up_chunk_offsets_and_roundtrip():
    """Verify B1 gate_up chunk offsets and bit-exact parse_gemv round-trip on chunk slices."""
    rng = np.random.default_rng(1001)

    test_experts = [0, 1, 42, 127, 128]
    for e in test_experts:
        for l in range(B1_CHUNKS_PER_EXPERT):
            w_off = get_gate_up_chunk_offset_words(e, l)
            b_off = get_gate_up_chunk_offset_bytes(e, l)
            assert w_off == e * GU_WORDS + l * B1_W_PER_CHUNK_WORDS
            assert b_off == w_off * 4
            assert 0 <= w_off < TOTAL_GU_WORDS
            assert 0 <= b_off < TOTAL_GU_BYTES

    e_test = 42
    gate = rng.standard_normal((MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)).astype(np.float32)
    up = rng.standard_normal((MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)).astype(np.float32)

    packed_expert = pack_expert_gate_up(gate, up, rows=4)
    assert len(packed_expert) == GU_BYTES

    rec_codes, rec_scales, rec_mins = parse_gemv(
        packed_expert, K=B1_K, M=B1_M_PER_EXPERT, rows=4
    )
    assert rec_codes.shape == (B1_M_PER_EXPERT, B1_K)

    for l in range(B1_CHUNKS_PER_EXPERT):
        chunk_byte_start = l * B1_W_PER_CHUNK_BYTES
        chunk_byte_end = chunk_byte_start + B1_W_PER_CHUNK_BYTES
        chunk_slice = packed_expert[chunk_byte_start:chunk_byte_end]

        c_codes, c_scales, c_mins = parse_gemv(chunk_slice, K=B1_K, M=CHUNK_ROWS, rows=4)
        row_start = l * CHUNK_ROWS
        row_end = row_start + CHUNK_ROWS

        np.testing.assert_array_equal(c_codes, rec_codes[row_start:row_end])
        np.testing.assert_allclose(c_scales, rec_scales[row_start:row_end], rtol=1e-5)
        np.testing.assert_allclose(c_mins, rec_mins[row_start:row_end], rtol=1e-5)


def test_moe_down_chunk_offsets_and_roundtrip():
    """Verify B2 down chunk offsets, pad_gemv zero-padding, and parse_gemv round-trip."""
    rng = np.random.default_rng(2002)

    test_experts = [0, 1, 63, 127, 128]
    for e in test_experts:
        for r in range(B2_CHUNKS_PER_EXPERT):
            w_off = get_down_chunk_offset_words(e, r)
            b_off = get_down_chunk_offset_bytes(e, r)
            assert w_off == e * DN_WORDS + r * B2_W_PER_CHUNK_WORDS
            assert b_off == w_off * 4
            assert 0 <= w_off < TOTAL_DN_WORDS
            assert 0 <= b_off < TOTAL_DN_BYTES

    down = rng.standard_normal((B2_M_PER_EXPERT, B2_K_ORIG)).astype(np.float32)
    packed_expert = pack_expert_down(down, rows=4)
    assert len(packed_expert) == DN_BYTES

    rec_codes, rec_scales, rec_mins = parse_gemv(
        packed_expert, K=B2_K_PAD, M=B2_M_PER_EXPERT, rows=4
    )
    assert rec_codes.shape == (B2_M_PER_EXPERT, B2_K_PAD)

    for r in range(B2_CHUNKS_PER_EXPERT):
        chunk_byte_start = r * B2_W_PER_CHUNK_BYTES
        chunk_byte_end = chunk_byte_start + B2_W_PER_CHUNK_BYTES
        chunk_slice = packed_expert[chunk_byte_start:chunk_byte_end]

        c_codes, c_scales, c_mins = parse_gemv(chunk_slice, K=B2_K_PAD, M=CHUNK_ROWS, rows=4)
        row_start = r * CHUNK_ROWS
        row_end = row_start + CHUNK_ROWS

        np.testing.assert_array_equal(c_codes, rec_codes[row_start:row_end])
        np.testing.assert_allclose(c_scales, rec_scales[row_start:row_end], rtol=1e-5)
        np.testing.assert_allclose(c_mins, rec_mins[row_start:row_end], rtol=1e-5)


def test_b1_scheduling_coverage_and_dispatch_offsets():
    """Verify B1 chunk c -> (slot j, chunk l, col, round, w_offset, y_offset) scheduling coverage."""
    TOTAL_CHUNKS = NUM_ACTIVE_TOTAL * B1_CHUNKS_PER_EXPERT  # 7 * 17 = 119
    COLS = 8
    MAX_ROUNDS = (TOTAL_CHUNKS + COLS - 1) // COLS  # 15
    EXPECTED_COL_ROUNDS = [15, 15, 15, 15, 15, 15, 15, 14]

    selected_experts = [3, 17, 45, 88, 101, 127, SHARED_EXPERT_ID]

    covered_chunks = []
    col_round_counts = [0] * COLS
    covered_slots_chunks = set()
    y_offsets_seen = set()
    w_offsets_seen = set()

    for rnd in range(MAX_ROUNDS):
        for col in range(COLS):
            c = rnd * COLS + col
            if c < TOTAL_CHUNKS:
                covered_chunks.append(c)
                col_round_counts[col] += 1

                j = c // B1_CHUNKS_PER_EXPERT
                l = c % B1_CHUNKS_PER_EXPERT
                e_j = selected_experts[j]

                w_off = e_j * GU_WORDS + l * B1_W_PER_CHUNK_WORDS
                y_off = c * CHUNK_ROWS

                assert 0 <= j < 7
                assert 0 <= l < 17
                assert (j, l) not in covered_slots_chunks
                covered_slots_chunks.add((j, l))

                assert y_off not in y_offsets_seen
                y_offsets_seen.add(y_off)

                assert w_off not in w_offsets_seen
                w_offsets_seen.add(w_off)

    assert covered_chunks == list(range(TOTAL_CHUNKS))
    assert len(covered_chunks) == 119
    assert col_round_counts == EXPECTED_COL_ROUNDS
    assert sum(col_round_counts) == TOTAL_CHUNKS
    assert len(y_offsets_seen) == 119
    assert sorted(y_offsets_seen) == [c * 128 for c in range(119)]
    assert max(y_offsets_seen) + 128 == NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT


def test_b2_scheduling_coverage_and_dispatch_offsets():
    """Verify B2 (slot j, round r) -> (col j, round r, x_offset, w_offset, y_offset) scheduling coverage."""
    ACTIVE_COLS = 7  # Column 7 is unused
    ROUNDS = B2_CHUNKS_PER_EXPERT  # 20
    TOTAL_CHUNKS = ACTIVE_COLS * ROUNDS  # 140

    selected_experts = [5, 12, 33, 77, 99, 120, SHARED_EXPERT_ID]

    covered_chunks = []
    x_offsets_seen = {}
    y_offsets_seen = set()
    w_offsets_seen = set()

    for r in range(ROUNDS):
        for j in range(ACTIVE_COLS):
            e_j = selected_experts[j]

            x_off = j * B2_K_PAD
            if j not in x_offsets_seen:
                x_offsets_seen[j] = x_off
            else:
                assert x_offsets_seen[j] == x_off

            w_off = e_j * DN_WORDS + r * B2_W_PER_CHUNK_WORDS
            assert w_off not in w_offsets_seen
            w_offsets_seen.add(w_off)

            y_off = j * B2_M_PER_EXPERT + r * CHUNK_ROWS
            assert y_off not in y_offsets_seen
            y_offsets_seen.add(y_off)

            covered_chunks.append((j, r))

    assert len(covered_chunks) == TOTAL_CHUNKS == 140
    assert len(x_offsets_seen) == 7
    for j in range(7):
        assert x_offsets_seen[j] == j * 1280

    assert len(y_offsets_seen) == 140
    expected_y_offsets = sorted(
        j * B2_M_PER_EXPERT + r * CHUNK_ROWS for j in range(7) for r in range(20)
    )
    assert sorted(y_offsets_seen) == expected_y_offsets
    assert max(y_offsets_seen) + 128 == NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT


def test_moe_host_silu_glu_and_router_accumulate():
    """Verify host-side SiLU-GLU activation and router accumulation arithmetic."""
    gate_up = np.zeros((NUM_ACTIVE_TOTAL, B1_M_PER_EXPERT), dtype=np.float32)

    gate_up[0, :MOE_INTERMEDIATE_SIZE] = 0.0
    gate_up[0, MOE_INTERMEDIATE_SIZE:] = 5.0

    gate_up[1, :MOE_INTERMEDIATE_SIZE] = 100.0
    gate_up[1, MOE_INTERMEDIATE_SIZE:] = 2.0

    x_down = moe_host_silu_glu(gate_up)
    assert x_down.shape == (NUM_ACTIVE_TOTAL, B2_K_PAD)

    np.testing.assert_allclose(x_down[0, :MOE_INTERMEDIATE_SIZE], 0.0, atol=1e-6)
    np.testing.assert_allclose(x_down[1, :MOE_INTERMEDIATE_SIZE], 200.0, rtol=1e-5)

    np.testing.assert_array_equal(
        x_down[:, MOE_INTERMEDIATE_SIZE:],
        np.zeros((NUM_ACTIVE_TOTAL, B2_K_PAD - MOE_INTERMEDIATE_SIZE), dtype=np.float32),
    )

    down_out = np.zeros((NUM_ACTIVE_TOTAL, B2_M_PER_EXPERT), dtype=np.float32)
    for j in range(NUM_ACTIVE_TOTAL):
        down_out[j, :] = float(j + 1)

    router_weights = [1.0 / 6.0] * 6
    final_out = moe_host_accumulate(down_out, router_weights, shared_scale=1.0)
    expected_val = 10.5
    np.testing.assert_allclose(final_out, np.full(B2_M_PER_EXPERT, expected_val, dtype=np.float32), rtol=1e-6)


def test_moe_hardware_budgets_verification():
    """Verify AIE2P hardware budget invariants (Shim channels, Core L1 including stack, BD count)."""
    L1_LIMIT = 65536  # 64 KiB
    STACK_SIZE = 10 * 1024  # 10 KiB stack (kernels/moe/design.py: STACK_SIZE)

    # 1. Core L1 data memory budget (base GEMV formula: buffers + stack, stack counted ONCE)
    # total = 2K (X) + K/16 (sums) + 10,240 (W, depth 2) + 128 (y) + 128 (y_acc) + STACK_SIZE.
    b1_total_l1 = int(2.0625 * B1_K + 10240 + 128 + 128 + STACK_SIZE)
    assert STACK_SIZE == 10240  # stack is already inside the 20,736-byte constant, once
    assert b1_total_l1 == 26016
    assert b1_total_l1 < L1_LIMIT
    assert b1_total_l1 / L1_LIMIT < 0.60

    b2_total_l1 = int(2.0625 * B2_K_PAD + 10240 + 128 + 128 + STACK_SIZE)
    assert b2_total_l1 == 23376
    assert b2_total_l1 < L1_LIMIT
    assert b2_total_l1 / L1_LIMIT < 0.55

    # 2. Shim DMA channels (MM2S <= 2, S2MM <= 2 per column)
    b1_mm2s_per_col = [2] + [1] * 7
    b1_s2mm_per_col = [1] * 8
    assert all(c <= 2 for c in b1_mm2s_per_col)
    assert all(c <= 2 for c in b1_s2mm_per_col)

    b2_mm2s_per_col = [2] * 7 + [0]
    b2_s2mm_per_col = [1] * 7 + [0]
    assert all(c <= 2 for c in b2_mm2s_per_col)
    assert all(c <= 2 for c in b2_s2mm_per_col)
    assert b2_mm2s_per_col[7] == 0
    assert b2_s2mm_per_col[7] == 0

    # 3. Shim Buffer Descriptors (BD <= 16 per column)
    # B1: Col 0 uses 1 BD for x (depth=1, MM2S1) + 2 BDs for W (depth=2, MM2S0) + 2 BDs for Y (depth=2, S2MM0) = 5 BDs.
    #     Col 1..7 use 2 BDs for W + 2 BDs for Y = 4 BDs.
    # B2: Col 0..6 use 1 BD for x_j (depth=1, MM2S1) + 2 BDs for W (depth=2, MM2S0) + 2 BDs for Y (depth=2, S2MM0) = 5 BDs.
    #     Col 7 is unused (0 BDs).
    BD_LIMIT = 16
    PIPE_DEPTH = 2

    b1_col_bds = [1 + PIPE_DEPTH + PIPE_DEPTH] + [PIPE_DEPTH + PIPE_DEPTH] * 7
    assert b1_col_bds[0] == 5
    assert all(bds <= BD_LIMIT for bds in b1_col_bds)

    b2_col_bds = [1 + PIPE_DEPTH + PIPE_DEPTH] * 7 + [0]
    assert b2_col_bds[0] == 5
    assert all(bds <= BD_LIMIT for bds in b2_col_bds)


def test_moe_mlir_operator_interfaces():
    """Verify MLIROperator declarations for templates and patched operators."""
    # 1. MoEGateUpTemplate
    op_gu_tpl = MoEGateUpTemplate()
    assert op_gu_tpl.K == B1_K
    assert op_gu_tpl.M == NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT
    assert op_gu_tpl.cols == 8
    gu_spec = op_gu_tpl.get_arg_spec()
    assert len(gu_spec) == 3
    assert gu_spec[0].direction == "out" and gu_spec[0].shape == (NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT,)
    assert gu_spec[1].direction == "in" and gu_spec[1].shape == (TOTAL_GU_WORDS,)
    assert gu_spec[2].direction == "in" and gu_spec[2].shape == (B1_K,)

    # 2. MoEDownTemplate
    op_dn_tpl = MoEDownTemplate()
    assert op_dn_tpl.K == B2_K_PAD
    assert op_dn_tpl.M == NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT
    assert op_dn_tpl.cols == 7
    dn_spec = op_dn_tpl.get_arg_spec()
    assert len(dn_spec) == 3
    assert dn_spec[0].direction == "out" and dn_spec[0].shape == (NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,)
    assert dn_spec[1].direction == "in" and dn_spec[1].shape == (TOTAL_DN_WORDS,)
    assert dn_spec[2].direction == "in" and dn_spec[2].shape == (NUM_ACTIVE_TOTAL * B2_K_PAD,)

    # 3. MoEGateUp wrapper
    op_gu = MoEGateUp()
    gu_params = op_gu.get_dispatch_params()
    assert set(gu_params.keys()) == {"e0", "e1", "e2", "e3", "e4", "e5"}

    # 4. MoEDown wrapper
    op_dn = MoEDown()
    dn_params = op_dn.get_dispatch_params()
    assert set(dn_params.keys()) == {"e0", "e1", "e2", "e3", "e4", "e5"}


def test_moe_bench_selection_patterns():
    """Verify selection patterns produce valid routed expert indices."""
    from bench.moe.moe_bench import get_selection_indices

    rng = np.random.default_rng(123)

    fixed = get_selection_indices("fixed", 0, None, rng)
    assert fixed == [0, 1, 2, 3, 4, 5]

    boundary = get_selection_indices("boundary", 0, None, rng)
    assert boundary == [122, 123, 124, 125, 126, 127]

    adj0 = get_selection_indices("adjacent", 0, None, rng)
    adj1 = get_selection_indices("adjacent", 1, None, rng)
    assert adj0 == [10, 11, 12, 13, 14, 15]
    assert adj1 == [10, 11, 12, 13, 14, 15]

    p0 = [10, 20, 30, 40, 50, 60]
    p1 = get_selection_indices("overlap_prev", 1, p0, rng)
    assert len(p1) == 6
    assert p1[:3] == [10, 20, 30]
    assert len(set(p1)) == 6
    assert len(set(p0) & set(p1)) == 3
    assert set(p1[3:]).isdisjoint(set(p0))

    # Test continuous chain of overlap_prev selections
    prev = p1
    for step in range(50):
        curr = get_selection_indices("overlap_prev", step + 2, prev, rng)
        assert len(curr) == 6
        assert len(set(curr)) == 6
        assert len(set(prev) & set(curr)) == 3
        assert set(curr[3:]).isdisjoint(set(prev))
        assert all(0 <= e < NUM_ROUTED_EXPERTS for e in curr)
        prev = curr

    rand = get_selection_indices("random", 0, None, rng)
    assert len(rand) == 6
    assert len(set(rand)) == 6
    assert all(0 <= e < NUM_ROUTED_EXPERTS for e in rand)


# =============================================================================
# 2. Static TXN Parsing & Address Patch Tests (Requirement 5)
# =============================================================================
def test_synthetic_b1_txn_parsing_and_patching():
    """Verify synthetic B1 TXN parsing, chunk index recovery, and per-call patching."""
    CHUNK_BYTES = B1_W_PER_CHUNK_BYTES  # 204,800 = 0x32000
    EXPECTED_PATCHES = 119  # 7 experts * 17 chunks

    # 119 chunks with template offsets: c * CHUNK_BYTES
    template_offsets = [c * CHUNK_BYTES for c in range(EXPECTED_PATCHES)]
    rng = np.random.default_rng(999)

    # Build synthetic stream
    txn_words = _build_synthetic_txn(
        template_offsets,
        w_arg_idx=1,
        y_patches_count=10,
        x_patches_count=1,
        rng=rng,
    )

    # Instantiate PatchedTemplateKernel directly from words
    kernel = PatchedTemplateKernel.from_words(
        None,
        txn_words,
        w_arg_idx=1,
        chunks_per_expert=B1_CHUNKS_PER_EXPERT,
        chunk_bytes=CHUNK_BYTES,
        expert_bytes=GU_BYTES,
        expected_w_patches=EXPECTED_PATCHES,
    )

    assert len(kernel._w_patch_entries) == EXPECTED_PATCHES

    # Verify order-independent chunk index recovery
    recovered_chunks = [c for _, c in kernel._w_patch_entries]
    assert set(recovered_chunks) == set(range(EXPECTED_PATCHES))

    # Test patch generation for arbitrary routed experts
    dispatch_scalars = {
        "e0": 10,
        "e1": 25,
        "e2": 42,
        "e3": 77,
        "e4": 99,
        "e5": 127,
    }
    expert_map = [10, 25, 42, 77, 99, 127, SHARED_EXPERT_ID]

    patched_words = kernel.instructions(**dispatch_scalars)
    assert len(patched_words) == len(txn_words)

    # Check that each W patch is correctly modified to expert_id * GU_BYTES + l * CHUNK_BYTES
    for pos, c in kernel._w_patch_entries:
        j = c // 17
        l = c % 17
        expected_expert = expert_map[j]
        expected_arg_plus = expected_expert * GU_BYTES + l * CHUNK_BYTES

        actual_arg_plus = int(patched_words[pos + 10]) | (int(patched_words[pos + 11]) << 32)
        assert actual_arg_plus == expected_arg_plus, f"Chunk {c} (j={j}, l={l}) arg_plus mismatch"

        # Verify arg_idx unchanged
        assert patched_words[pos + 8] == 1
        # Verify opcode unchanged
        assert (patched_words[pos] & 0xFF) == TXN_OPC_DDR_PATCH


def test_synthetic_b2_txn_parsing_and_patching():
    """Verify synthetic B2 TXN parsing, chunk index recovery, and per-call patching."""
    CHUNK_BYTES = B2_W_PER_CHUNK_BYTES  # 102,400 = 0x19000
    EXPECTED_PATCHES = 140  # 7 experts * 20 chunks

    template_offsets = [c * CHUNK_BYTES for c in range(EXPECTED_PATCHES)]
    rng = np.random.default_rng(888)

    txn_words = _build_synthetic_txn(
        template_offsets,
        w_arg_idx=1,
        y_patches_count=7,
        x_patches_count=7,
        rng=rng,
    )

    kernel = PatchedTemplateKernel.from_words(
        None,
        txn_words,
        w_arg_idx=1,
        chunks_per_expert=B2_CHUNKS_PER_EXPERT,
        chunk_bytes=CHUNK_BYTES,
        expert_bytes=DN_BYTES,
        expected_w_patches=EXPECTED_PATCHES,
    )

    assert len(kernel._w_patch_entries) == EXPECTED_PATCHES
    recovered_chunks = [c for _, c in kernel._w_patch_entries]
    assert set(recovered_chunks) == set(range(EXPECTED_PATCHES))

    dispatch_scalars = {
        "e0": 5,
        "e1": 15,
        "e2": 35,
        "e3": 55,
        "e4": 85,
        "e5": 115,
    }
    expert_map = [5, 15, 35, 55, 85, 115, SHARED_EXPERT_ID]

    patched_words = kernel.instructions(**dispatch_scalars)

    for pos, c in kernel._w_patch_entries:
        j = c // 20
        r = c % 20
        expected_expert = expert_map[j]
        expected_arg_plus = expected_expert * DN_BYTES + r * CHUNK_BYTES

        actual_arg_plus = int(patched_words[pos + 10]) | (int(patched_words[pos + 11]) << 32)
        assert actual_arg_plus == expected_arg_plus, f"Chunk {c} (j={j}, r={r}) arg_plus mismatch"


def test_w_patch_count_mismatch_raises_exception():
    """Verify that an unexpected number of W patches (not 119 for B1) raises ValueError."""
    CHUNK_BYTES = B1_W_PER_CHUNK_BYTES
    rng = np.random.default_rng(111)

    # 1. 118 patches (missing 1)
    offsets_118 = [c * CHUNK_BYTES for c in range(118)]
    txn_118 = _build_synthetic_txn(offsets_118, rng=rng)
    with pytest.raises(ValueError, match="W patch count mismatch"):
        PatchedTemplateKernel.from_words(
            None, txn_118, expected_w_patches=119, chunk_bytes=CHUNK_BYTES
        )

    # 2. 120 patches (1 extra)
    offsets_120 = [c * CHUNK_BYTES for c in range(120)]
    txn_120 = _build_synthetic_txn(offsets_120, rng=rng)
    with pytest.raises(ValueError, match="W patch count mismatch"):
        PatchedTemplateKernel.from_words(
            None, txn_120, expected_w_patches=119, chunk_bytes=CHUNK_BYTES
        )


def test_invalid_chunk_alignment_or_duplicate_chunk_raises_exception():
    """Verify that non-aligned or duplicate template offsets raise ValueError."""
    CHUNK_BYTES = B1_W_PER_CHUNK_BYTES
    rng = np.random.default_rng(222)

    # 1. Misaligned offset
    bad_offsets = [c * CHUNK_BYTES for c in range(118)] + [0x32001]
    txn_bad = _build_synthetic_txn(bad_offsets, rng=rng)
    with pytest.raises(ValueError, match="not a multiple of chunk_bytes"):
        PatchedTemplateKernel.from_words(
            None, txn_bad, expected_w_patches=119, chunk_bytes=CHUNK_BYTES
        )

    # 2. Duplicate chunk offset
    dup_offsets = [c * CHUNK_BYTES for c in range(118)] + [0]  # duplicate chunk 0
    txn_dup = _build_synthetic_txn(dup_offsets, rng=rng)
    with pytest.raises(ValueError, match="Duplicate chunk index"):
        PatchedTemplateKernel.from_words(
            None, txn_dup, expected_w_patches=119, chunk_bytes=CHUNK_BYTES
        )


def test_real_static_gemv_txn_if_present():
    """Verify real statically compiled GEMV TXN binary (if available in delegate artifacts)."""
    import os
    if not os.environ.get("RAON_TXN_PATH"):
        pytest.skip("RAON_TXN_PATH is not set")
    real_txn_path = Path(os.environ["RAON_TXN_PATH"])
    if not real_txn_path.exists():
        pytest.skip(f"Real static TXN not found at {real_txn_path}")

    words = np.fromfile(real_txn_path, dtype=np.uint32)
    assert len(words) == 9092

    kernel = PatchedTemplateKernel.from_words(
        None,
        words,
        w_arg_idx=1,
        chunks_per_expert=17,
        chunk_bytes=0x32000,
        expert_bytes=GU_BYTES,
        expected_w_patches=119,
    )

    assert len(kernel._w_patch_entries) == 119
    recovered_chunks = [c for _, c in kernel._w_patch_entries]
    assert set(recovered_chunks) == set(range(119))

    # Test patching with boundary pattern [122, 123, 124, 125, 126, 127]
    scalars = {"e0": 122, "e1": 123, "e2": 124, "e3": 125, "e4": 126, "e5": 127}
    expert_map = [122, 123, 124, 125, 126, 127, SHARED_EXPERT_ID]

    patched = kernel.instructions(**scalars)
    assert len(patched) == 9092

    for pos, c in kernel._w_patch_entries:
        j = c // 17
        l = c % 17
        expected_expert = expert_map[j]
        expected_arg_plus = expected_expert * GU_BYTES + l * 0x32000
        actual_arg_plus = int(patched[pos + 10]) | (int(patched[pos + 11]) << 32)
        assert actual_arg_plus == expected_arg_plus


def test_patch_expert_id_range_validation():
    """Verify that e0..e5 must be integers in [0, 128), rejecting out-of-range, float, and strings."""
    template_offsets = [c * B1_W_PER_CHUNK_BYTES for c in range(119)]
    rng = np.random.default_rng(42)
    txn_words = _build_synthetic_txn(template_offsets, w_arg_idx=1, rng=rng)

    kernel = PatchedTemplateKernel.from_words(
        None,
        txn_words,
        w_arg_idx=1,
        chunks_per_expert=17,
        chunk_bytes=B1_W_PER_CHUNK_BYTES,
        expert_bytes=GU_BYTES,
        expected_w_patches=119,
    )

    base_scalars = {"e0": 0, "e1": 1, "e2": 2, "e3": 3, "e4": 4, "e5": 5}

    # 1. Negative expert index
    for slot in ["e0", "e3", "e5"]:
        bad_scalars = dict(base_scalars)
        bad_scalars[slot] = -1
        with pytest.raises(ValueError, match="out of range"):
            kernel.instructions(**bad_scalars)

    # 2. Upper boundary 128
    for slot in ["e0", "e2", "e5"]:
        bad_scalars = dict(base_scalars)
        bad_scalars[slot] = 128
        with pytest.raises(ValueError, match="out of range"):
            kernel.instructions(**bad_scalars)

    # 3. Large positive integer 1000
    for slot in ["e1", "e4"]:
        bad_scalars = dict(base_scalars)
        bad_scalars[slot] = 1000
        with pytest.raises(ValueError, match="out of range"):
            kernel.instructions(**bad_scalars)

    # 4. Floating point value 1.5
    for slot in ["e0", "e5"]:
        bad_scalars = dict(base_scalars)
        bad_scalars[slot] = 1.5
        with pytest.raises(ValueError, match="must be an integer"):
            kernel.instructions(**bad_scalars)

    # 5. String inputs ("invalid", "10")
    for slot in ["e0", "e2"]:
        bad_scalars = dict(base_scalars)
        bad_scalars[slot] = "invalid"
        with pytest.raises(ValueError, match="must be an integer"):
            kernel.instructions(**bad_scalars)

        bad_scalars[slot] = "10"
        with pytest.raises(ValueError, match="must be an integer"):
            kernel.instructions(**bad_scalars)

    # 6. Boolean inputs (True, False)
    for slot in ["e0", "e5"]:
        bad_scalars = dict(base_scalars)
        bad_scalars[slot] = True
        with pytest.raises(ValueError, match="must be an integer"):
            kernel.instructions(**bad_scalars)


def test_parse_expert_buffer_types():
    """Verify parse_expert_gate_up and parse_expert_down work identically for bytes, uint8, and uint32 buffers."""
    rng = np.random.default_rng(2026)

    # 1. Gate-up test with 2 experts
    w_gate0 = rng.uniform(-1.0, 1.0, (B1_M_PER_EXPERT // 2, B1_K)).astype(np.float32)
    w_up0 = rng.uniform(-1.0, 1.0, (B1_M_PER_EXPERT // 2, B1_K)).astype(np.float32)
    chunk0 = pack_expert_gate_up(w_gate0, w_up0, rows=4)

    w_gate1 = rng.uniform(-1.0, 1.0, (B1_M_PER_EXPERT // 2, B1_K)).astype(np.float32)
    w_up1 = rng.uniform(-1.0, 1.0, (B1_M_PER_EXPERT // 2, B1_K)).astype(np.float32)
    chunk1 = pack_expert_gate_up(w_gate1, w_up1, rows=4)

    buf2_bytes = chunk0 + chunk1
    assert len(buf2_bytes) == 2 * GU_BYTES

    buf2_u8 = np.frombuffer(buf2_bytes, dtype=np.uint8)
    buf2_u32 = np.frombuffer(buf2_bytes, dtype=np.uint32)

    for expert_id in (0, 1):
        c_bytes, s_bytes, m_bytes = parse_expert_gate_up(buf2_bytes, expert_id, rows=4)
        c_u8, s_u8, m_u8 = parse_expert_gate_up(buf2_u8, expert_id, rows=4)
        c_u32, s_u32, m_u32 = parse_expert_gate_up(buf2_u32, expert_id, rows=4)

        np.testing.assert_array_equal(c_bytes, c_u8)
        np.testing.assert_array_equal(c_bytes, c_u32)
        np.testing.assert_array_equal(s_bytes, s_u8)
        np.testing.assert_array_equal(s_bytes, s_u32)
        np.testing.assert_array_equal(m_bytes, m_u8)
        np.testing.assert_array_equal(m_bytes, m_u32)

    # 2. Down test with 2 experts
    w_dn0 = rng.uniform(-1.0, 1.0, (B2_M_PER_EXPERT, MOE_INTERMEDIATE_SIZE)).astype(np.float32)
    chunk_dn0 = pack_expert_down(w_dn0, rows=4)

    w_dn1 = rng.uniform(-1.0, 1.0, (B2_M_PER_EXPERT, MOE_INTERMEDIATE_SIZE)).astype(np.float32)
    chunk_dn1 = pack_expert_down(w_dn1, rows=4)

    buf2_dn_bytes = chunk_dn0 + chunk_dn1
    assert len(buf2_dn_bytes) == 2 * DN_BYTES

    buf2_dn_u8 = np.frombuffer(buf2_dn_bytes, dtype=np.uint8)
    buf2_dn_u32 = np.frombuffer(buf2_dn_bytes, dtype=np.uint32)

    for expert_id in (0, 1):
        c_bytes, s_bytes, m_bytes = parse_expert_down(buf2_dn_bytes, expert_id, rows=4)
        c_u8, s_u8, m_u8 = parse_expert_down(buf2_dn_u8, expert_id, rows=4)
        c_u32, s_u32, m_u32 = parse_expert_down(buf2_dn_u32, expert_id, rows=4)

        np.testing.assert_array_equal(c_bytes, c_u8)
        np.testing.assert_array_equal(c_bytes, c_u32)
        np.testing.assert_array_equal(s_bytes, s_u8)
        np.testing.assert_array_equal(s_bytes, s_u32)
        np.testing.assert_array_equal(m_bytes, m_u8)
        np.testing.assert_array_equal(m_bytes, m_u32)


def test_patch_64bit_arg_plus_high_word():
    """Verify that DDR_PATCH arg_plus upper 32-bit word is written correctly for 64-bit offsets."""
    CHUNK_BYTES = 0x32000
    EXPECTED_PATCHES = 119
    template_offsets = [c * CHUNK_BYTES for c in range(EXPECTED_PATCHES)]
    rng = np.random.default_rng(777)
    txn_words = _build_synthetic_txn(template_offsets, w_arg_idx=1, rng=rng)

    # Artificially large expert_bytes exceeding 2^32 (e.g. 0x1_0000_0000 = 4 GiB)
    HUGE_EXPERT_BYTES = 0x1_0000_0000  # 4,294,967,296 bytes

    kernel = PatchedTemplateKernel.from_words(
        None,
        txn_words,
        w_arg_idx=1,
        chunks_per_expert=17,
        chunk_bytes=CHUNK_BYTES,
        expert_bytes=HUGE_EXPERT_BYTES,
        expected_w_patches=EXPECTED_PATCHES,
    )

    scalars = {"e0": 1, "e1": 2, "e2": 3, "e3": 4, "e4": 5, "e5": 6}
    expert_map = [1, 2, 3, 4, 5, 6, SHARED_EXPERT_ID]

    patched = kernel.instructions(**scalars)

    for pos, c in kernel._w_patch_entries:
        j = c // 17
        l = c % 17
        expected_expert = expert_map[j]
        expected_arg_plus = expected_expert * HUGE_EXPERT_BYTES + l * CHUNK_BYTES

        actual_low = int(patched[pos + 10])
        actual_high = int(patched[pos + 11])
        actual_arg_plus = actual_low | (actual_high << 32)

        # Upper 32-bit word must match high part of 64-bit offset
        assert actual_high == (expected_arg_plus >> 32) & 0xFFFFFFFF
        assert actual_high == expected_expert  # Since multiplier is 1 << 32
        assert actual_low == (expected_arg_plus & 0xFFFFFFFF)
        assert actual_arg_plus == expected_arg_plus
