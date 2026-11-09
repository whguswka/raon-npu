# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side NumPy unit tests for Raon general q4nx GEMV.

Verifies:
1. Packing byte-identity: ``pack_gemv`` matches pre-refactor golden SHA-256 hashes
   and IRON independent ``iron.operators.flm.lm_head.reference.dequantize``.
2. General M packing: ``pack_gemv`` and ``parse_gemv`` for M % 128 == 0 (non-1024 multiples),
   and rejection of non-128 multiples.
3. CPU reference arithmetic: Hand-calculated exact results for controlled q4nx weights.
4. Padding invariance: K (256-align) and M (128-align) zero-padding preserves exact results
   on the active submatrix (zero x contributes 0 even to the minimum term).
5. Chunk-to-column scheduling coverage: (chunk -> col, round) mapping precisely and uniquely
   covers all chunks across Raon presets and arbitrary shapes.
6. Design shape validation: ``check_shape`` rejects K < 256, M < 128, and unaligned shapes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

# Ensure iron is importable for independent reference dequantize
iron_path = os.environ.get("RAON_IRON_PATH", "")
if iron_path and iron_path not in sys.path:
    sys.path.insert(0, iron_path)

try:
    from iron.operators.flm.lm_head.reference import dequantize as iron_lm_head_dequantize
except (ImportError, ModuleNotFoundError):
    iron_lm_head_dequantize = None

from quant.q4nx import (
    K_TILE,
    M_TILE,
    dequantize_simple,
    pack_gemv,
    pack_lm_head,
    pad_gemv,
    parse_gemv,
    parse_lm_head,
    quantize,
    round_to_bf16_rne,
    to_bf16_u16,
    bf16_floor,
)
from kernels.gemv.reference import (
    bf16_ulp_diff,
    dequantize,
    dequantize_block,
    gemv_bit_reference,
    reference,
)


def test_pack_gemv_matches_pre_refactor_golden_sha256():
    """Verify pack_gemv produces byte-identical output to pre-refactor pack_lm_head golden SHA-256.

    Replaces the tautological pack_gemv == pack_lm_head assertion with fixed golden hashes
    generated from the original pre-refactor pack_lm_head in main branch:
    quant/q4nx.py.

    Hash generation commands:
    -----------------------------------------------------------------------------------------
    1) Shape (M=1024, K=256), seed=1234:
       python3 -c "import hashlib, importlib.util, numpy as np; spec = importlib.util.spec_from_file_location('orig', 'quant/q4nx.py'); orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig); rng = np.random.default_rng(1234); w = rng.standard_normal((1024, 256)).astype(np.float32); c, s, m = orig.quantize(w); b = orig.pack_lm_head(c, s, m, cols=8, rows=4); print(hashlib.sha256(b).hexdigest())"
       Result: 008a66c523441448ae9115edf7de8f1e41f891fe88a5581df47dbacd625071ac

    2) Shape (M=2048, K=512), seed=1234:
       python3 -c "import hashlib, importlib.util, numpy as np; spec = importlib.util.spec_from_file_location('orig', 'quant/q4nx.py'); orig = importlib.util.module_from_spec(spec); spec.loader.exec_module(orig); rng = np.random.default_rng(1234); w = rng.standard_normal((2048, 512)).astype(np.float32); c, s, m = orig.quantize(w); b = orig.pack_lm_head(c, s, m, cols=8, rows=4); print(hashlib.sha256(b).hexdigest())"
       Result: 0ecb33ce812482f45163d5f5743e3673ea4d5054741da78b8f420d605a20ffc6
    -----------------------------------------------------------------------------------------
    """
    golden_cases = [
        # (M, K, seed, expected_sha256, expected_bytes)
        (
            1024,
            256,
            1234,
            "008a66c523441448ae9115edf7de8f1e41f891fe88a5581df47dbacd625071ac",
            163840,
        ),
        (
            2048,
            512,
            1234,
            "0ecb33ce812482f45163d5f5743e3673ea4d5054741da78b8f420d605a20ffc6",
            655360,
        ),
    ]

    for M, K, seed, expected_sha, expected_len in golden_cases:
        rng = np.random.default_rng(seed)
        w = rng.standard_normal((M, K)).astype(np.float32)
        codes, scales, mins = quantize(w)

        b_gemv = pack_gemv(codes, scales, mins, rows=4)

        assert len(b_gemv) == expected_len, f"Byte length mismatch for shape ({M}, {K})"
        computed_sha = hashlib.sha256(b_gemv).hexdigest()
        assert (
            computed_sha == expected_sha
        ), f"SHA-256 hash mismatch for shape ({M}, {K}): got {computed_sha}, expected {expected_sha}"

        # Verify round-trip parsing
        parsed_codes, parsed_scales, parsed_mins = parse_gemv(b_gemv, K=K, M=M, rows=4)
        np.testing.assert_array_equal(parsed_codes, codes)
        np.testing.assert_allclose(parsed_scales, scales, rtol=1e-5)
        np.testing.assert_allclose(parsed_mins, mins, rtol=1e-5)


def test_pack_gemv_iron_independent_dequantize():
    """Verify pack_gemv output accurately dequantizes using IRON reference dequantizer.

    Independent cross-verification using:
    ``iron.operators.flm.lm_head.reference.dequantize(w, dim, vocab, cols, rows)``.

    Tests:
    1. Standard 1024 multiple: M=1024, K=256 (1 round)
    2. Multi-round 1024 multiple: M=2048, K=512 (2 rounds)
    3. Non-1024 multiple (128-aligned): M=1152, K=256 (9 chunks)
       Note: IRON's dequantize indexes slices by ``index // rows // k_blocks``
       and operates strictly in units of ``rows * M_TILE`` (128 rows).
       Therefore, IRON dequantize also successfully unpacks M=1152 without error.
    """
    assert iron_lm_head_dequantize is not None, "iron.operators.flm.lm_head.reference.dequantize not found"

    rng = np.random.default_rng(4321)
    test_shapes = [
        (1024, 256),   # 1024 multiple (1 round)
        (2048, 512),   # 1024 multiple (2 rounds)
        (1152, 256),   # 128 multiple, non-1024 (9 chunks, MLA q_a+kv_a bundle M dimension)
    ]

    for M, K in test_shapes:
        w = rng.standard_normal((M, K)).astype(np.float32)
        codes, scales, mins = quantize(w)

        b_gemv = pack_gemv(codes, scales, mins, rows=4)

        # Dequantize via IRON lm_head reference dequantize
        # Signature: dequantize(w, dim, vocab, cols, rows)
        iron_weights = iron_lm_head_dequantize(
            np.frombuffer(b_gemv, dtype=np.uint8),
            dim=K,
            vocab=M,
            cols=8,
            rows=4,
        )

        expected_weights = dequantize_simple(codes, scales, mins)
        np.testing.assert_array_equal(
            iron_weights,
            expected_weights,
            err_msg=f"IRON reference dequantize output mismatch for shape ({M}, {K})",
        )


def test_pack_gemv_non_1024_multiples():
    """Verify pack_gemv supports M that is a multiple of 128 but NOT 1024."""
    rng = np.random.default_rng(5678)
    shapes = [
        (128, 256),    # Minimal 1-chunk shape
        (256, 256),    # 2 chunks
        (1152, 2560),  # MLA q_a+kv_a bundle (9 chunks)
        (2176, 2560),  # Expert gate+up bundle (17 chunks)
        (7680, 768),   # MLA q_b (60 chunks)
    ]

    for M, K in shapes:
        w = rng.standard_normal((M, K)).astype(np.float32)
        codes, scales, mins = quantize(w)

        # pack_gemv must succeed
        b_gemv = pack_gemv(codes, scales, mins, rows=4)
        assert len(b_gemv) > 0

        # pack_lm_head must raise ValueError because M % 1024 != 0
        with pytest.raises(ValueError, match="multiple of"):
            pack_lm_head(codes, scales, mins, cols=8, rows=4)

        # parse_gemv must accurately reconstruct codes, scales, mins
        rec_codes, rec_scales, rec_mins = parse_gemv(b_gemv, K=K, M=M, rows=4)
        np.testing.assert_array_equal(rec_codes, codes)
        np.testing.assert_allclose(rec_scales, scales, rtol=1e-5)
        np.testing.assert_allclose(rec_mins, mins, rtol=1e-5)

    # Rejection of invalid alignments
    invalid_M = np.zeros((100, 256), dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple of 128"):
        pack_gemv(invalid_M, invalid_M[:, :8], invalid_M[:, :8], rows=4)

    invalid_K = np.zeros((128, 200), dtype=np.uint8)
    with pytest.raises(ValueError, match="multiple of K_TILE"):
        pack_gemv(invalid_K, invalid_K[:, :8], invalid_K[:, :8], rows=4)


def test_gemv_reference_arithmetic_hand_calculated():
    """Verify dequantize and reference arithmetic against exact hand-calculated values."""
    M, K = 128, 256
    # 1 slice (128 rows = 4 rows * 32), 1 k_block (256 cols), 8 groups (256 / 32)
    # Set controlled integer values:
    # code = 3
    # scale = 0.5
    # min = 1.0
    # Expected weight = 1.0 + 0.5 * 3 = 2.5 everywhere
    codes = np.full((M, K), 3, dtype=np.uint8)
    scales = np.full((M, K // 32), 0.5, dtype=np.float32)
    mins = np.full((M, K // 32), 1.0, dtype=np.float32)

    packed = pack_gemv(codes, scales, mins, rows=4)
    weights_deq = dequantize(packed, K=K, M=M, rows=4)

    expected_w = np.full((M, K), 2.5, dtype=np.float64)
    np.testing.assert_allclose(weights_deq, expected_w, rtol=1e-6)

    # Input vector: all 1.0
    x = np.ones(K, dtype=np.float32)
    # y = weights @ x = 256 * 2.5 = 640.0 for every row
    y = reference(weights_deq, x)
    expected_y = np.full(M, 640.0, dtype=np.float64)
    np.testing.assert_allclose(y, expected_y, rtol=1e-6)


def test_gemv_bit_reference_rounding_boundary_hand_calculated():
    """Verify bit reference model on bf16 rounding boundary where floor vs conv_even diverges.

    Boundary arithmetic analysis:
    - In bfloat16, integers between 256 and 512 have a precision of 2 (only even integers
      256, 258, 260, ... can be represented exactly).
    - An odd integer like 259.0 falls exactly on the tie boundary between 258.0 and 260.0.
    - 258.0 in bf16 has an odd mantissa bit pattern (ends in 1).
    - 260.0 in bf16 has an even mantissa bit pattern (ends in 0).
    - Therefore:
        bf16_floor(259.0) narrows to 258.0 (truncated towards -infinity).
        conv_even / round_to_bf16_rne(259.0) rounds to 260.0 (round-to-nearest-even).
      The two rounding modes diverge by 2.0 (exactly 1 bf16 ULP).

    Test construction:
    - M = 128, K = 256 (1 slice, 1 k_block, 8 groups of 32).
    - In group 0: 31 columns have code = 8, 1 column has code = 11.
      x = 1.0 everywhere.
      dot = 31 * 8 * 1.0 + 1 * 11 * 1.0 = 248.0 + 11.0 = 259.0.
    - Groups 1..7: code = 0, so dot = 0.
    - scale = 1.0, min = 0.0 for group 0; scales = 0, mins = 0 for other groups.
    - S = sum(x) = 32.0 for all groups.
    - Under floor: D = 258.0 -> y = 258.0.
    - Under conv_even: D = 260.0 -> y = 260.0.
    """
    M, K = 128, 256
    codes = np.zeros((M, K), dtype=np.uint8)
    # Set group 0 (first 32 columns): 31 elements of 8, 1 element of 11
    codes[:, :31] = 8
    codes[:, 31] = 11

    scales = np.zeros((M, K // 32), dtype=np.float32)
    mins = np.zeros((M, K // 32), dtype=np.float32)
    scales[:, 0] = 1.0  # group 0 scale = 1.0

    x = np.ones(K, dtype=np.float32)

    # 1. Verify scalar narrowing difference
    dot_val = np.float32(259.0)
    assert bf16_floor(dot_val) == 258.0, "Floor must truncate 259.0 to 258.0"
    assert round_to_bf16_rne(dot_val) == 260.0, "conv_even must round 259.0 to 260.0"

    # 2. Run gemv_bit_reference
    y_f32, y_u16 = gemv_bit_reference(codes, scales, mins, x)

    # Must match conv_even prediction (260.0), NOT floor (258.0)
    expected_y = np.full(M, 260.0, dtype=np.float32)
    np.testing.assert_array_equal(y_f32, expected_y)

    floor_y = np.full(M, 258.0, dtype=np.float32)
    ulp_diff = bf16_ulp_diff(y_u16, to_bf16_u16(floor_y))
    # Exactly 1 ULP difference between conv_even and floor results
    assert np.all(ulp_diff == 1)

    # 3. Pack full 128 rows (M=128, K=256) and verify host-side reference dequantize
    # runs unconditionally (including CPU host where AIE/IRON kernel module is absent).
    packed = pack_gemv(codes, scales, mins, rows=4)
    assert len(packed) == 4 * 5120  # 4 tiles of (32, 256)
    sums = x.reshape(8, 32).sum(axis=1).astype(np.float32)
    sums_bf16 = round_to_bf16_rne(sums)

    block_w = dequantize_block(packed[:5120])
    block_y = reference(block_w, x)
    np.testing.assert_array_equal(block_y, np.full(32, 259.0))
    np.testing.assert_array_equal(round_to_bf16_rne(block_y), np.full(32, 260.0, dtype=np.float32))

    # Single block kernel contract reference (if AIE/IRON kernel module is available on laptop)
    try:
        from kernels.gemv.kernel import flm_gemma4_q4nx_gemv_ref
        block_ref_acc = flm_gemma4_q4nx_gemv_ref(packed[:5120], x, sums_bf16, m_tile=32, k_tile=256)
        np.testing.assert_array_equal(block_ref_acc, np.full((1, 32), 260.0, dtype=np.float32))
    except (ImportError, ModuleNotFoundError):
        pass


def test_gemv_padding_invariance():
    """Verify K and M padding preserves identical outputs on active submatrix.

    Key mathematical property:
    For padded input columns where x[k] = 0, the q4nx computation:
      dot = sum(code * x) = 0
      col_sum = sum(x) = 0
      term = scale * dot + min * col_sum = 0
    Therefore, even non-zero min parameters contribute exactly 0 when x == 0.
    """
    rng = np.random.default_rng(9999)
    K_orig, M_orig = 576, 2100
    K_pad, M_pad = 768, 2176  # 576 -> 768 (K_TILE=256), 2100 -> 2176 (128-align)

    w_orig = rng.standard_normal((M_orig, K_orig)).astype(np.float32)
    x_orig = rng.standard_normal(K_orig).astype(np.float32)

    w_padded, orig_shape = pad_gemv(w_orig, m_align=128, k_align=256)
    assert w_padded.shape == (M_pad, K_pad)
    assert orig_shape == (M_orig, K_orig)

    # Quantize padded matrix
    codes_pad, scales_pad, mins_pad = quantize(w_padded)
    w_deq_pad = dequantize_simple(codes_pad, scales_pad, mins_pad)

    # Input x zero-padded to K_pad
    x_padded = np.pad(x_orig, (0, K_pad - K_orig), constant_values=0.0)

    # In float64, the dot product with zero-padded entries is mathematically identical to 0.0
    w_deq_f64 = w_deq_pad.astype(np.float64)
    x_padded_f64 = x_padded.astype(np.float64)
    y_full = w_deq_f64 @ x_padded_f64
    y_sub = w_deq_f64[:M_orig, :K_orig] @ x_orig.astype(np.float64)

    # Output on active rows must match with zero mathematical difference in float64
    np.testing.assert_allclose(y_full[:M_orig], y_sub, rtol=1e-12, atol=1e-12)

    # Verify that the padded columns contribute exactly zero
    padded_col_contribution = w_deq_f64[:, K_orig:] @ x_padded_f64[K_orig:]
    np.testing.assert_array_equal(padded_col_contribution, np.zeros(M_pad, dtype=np.float64))


def test_chunk_scheduling_coverage():
    """Verify chunk -> (col, round) scheduling uniquely and completely covers all chunks."""
    test_cases = [
        # (name, M, cols)
        ("Minimal single chunk", 128, 8),
        ("Small 3 chunks", 384, 8),
        ("RDA q/k/v/g bundle", 20480, 8),
        ("RDA q/k/v/g bundle (cols=4)", 20480, 4),
        ("Output proj", 2560, 8),
        ("Output proj (cols=4)", 2560, 4),
        ("MLA q_a+kv_a bundle", 1152, 8),
        ("MLA q_b", 7680, 8),
        ("Expert 1 gate+up bundle", 2176, 8),
        ("Expert 1 gate+up bundle (cols=4)", 2176, 4),
        ("Dense MLP gate+up", 24064, 8),
        ("LM head reference size", 260096, 8),
    ]

    for name, M, cols in test_cases:
        TOTAL_CHUNKS = M // 128
        ACTIVE_COLS = min(cols, TOTAL_CHUNKS)

        col_rounds = [
            TOTAL_CHUNKS // cols + (1 if j < (TOTAL_CHUNKS % cols) else 0)
            for j in range(ACTIVE_COLS)
        ]
        MAX_ROUNDS = (TOTAL_CHUNKS + cols - 1) // cols

        covered_chunks = []
        col_chunk_counts = [0] * ACTIVE_COLS

        for rnd in range(MAX_ROUNDS):
            for col in range(ACTIVE_COLS):
                c = rnd * cols + col
                if c < TOTAL_CHUNKS:
                    covered_chunks.append(c)
                    col_chunk_counts[col] += 1

        # 1. Every chunk 0..TOTAL_CHUNKS-1 must be scheduled exactly once
        assert covered_chunks == list(range(TOTAL_CHUNKS)), (
            f"Scheduling coverage failed for {name} (M={M}, cols={cols})"
        )

        # 2. Each column executes exactly its designated round count
        assert col_chunk_counts == col_rounds, (
            f"Column round mismatch for {name}: {col_chunk_counts} vs {col_rounds}"
        )

        # 3. Sum of column rounds equals total chunks
        assert sum(col_chunk_counts) == TOTAL_CHUNKS


def test_gemv_check_shape_minimums_and_alignments():
    """Verify check_shape rejects K < 256, M < 128, non-multiples, and invalid dev/cols."""
    from types import SimpleNamespace
    from kernels.gemv.design import check_shape, AIEArch

    dev = SimpleNamespace(arch=AIEArch.AIE2p, core_rows=[0, 1, 2, 3], cols=8)

    # Valid minimum shape
    check_shape(dev, K=256, M=128, cols=8)

    # K < 256 rejection (0 and sub-tile values)
    with pytest.raises(ValueError, match="at least 256"):
        check_shape(dev, K=0, M=128, cols=8)
    with pytest.raises(ValueError, match="at least 256"):
        check_shape(dev, K=128, M=128, cols=8)

    # K not multiple of 256
    with pytest.raises(ValueError, match="positive multiple of 256"):
        check_shape(dev, K=300, M=128, cols=8)

    # M < 128 rejection (0 and sub-chunk values)
    with pytest.raises(ValueError, match="at least 128"):
        check_shape(dev, K=256, M=0, cols=8)
    with pytest.raises(ValueError, match="at least 128"):
        check_shape(dev, K=256, M=64, cols=8)

    # M not multiple of 128
    with pytest.raises(ValueError, match="positive multiple of 128"):
        check_shape(dev, K=256, M=200, cols=8)

    # Architecture rejection (must be AIE2P)
    dev_invalid = SimpleNamespace(arch="AIE2", core_rows=[0, 1, 2, 3], cols=8)
    with pytest.raises(NotImplementedError, match="AIE2P only"):
        check_shape(dev_invalid, K=256, M=128, cols=8)

    # Invalid cols rejection
    with pytest.raises(ValueError, match="cols.*must be between 1 and"):
        check_shape(dev, K=256, M=128, cols=0)
    with pytest.raises(ValueError, match="cols.*must be between 1 and"):
        check_shape(dev, K=256, M=128, cols=9)
