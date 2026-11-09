# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""NumPy-only checks for the D mixed_full MoE layouts and patch tables."""

from __future__ import annotations

import numpy as np

from bench.moe_d import moe_d_bench
from kernels.moe.design import (
    down_bfp16_weight_tap_offsets,
    gate_up_weight_tap_offsets,
)
from kernels.moe.op import (
    MoEDownBfp16,
    MoEDownBfp16Template,
    MoEGateUp,
    MoEGateUpTemplate,
)
from kernels.moe.pack import (
    B1_M_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    B2_BFP16_W_PER_CHUNK_BYTES,
    B2_BFP16_W_PER_CHUNK_WORDS,
    B2_K_ORIG,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    DN9_BYTES,
    DN9_WORDS,
    NUM_ACTIVE_ROUTED,
    NUM_ACTIVE_TOTAL,
    ROUTED_GU_WORDS,
    SHARED_EXPERT_ID,
    TOTAL_DN9_BYTES,
    TOTAL_DN9_WORDS,
    get_down_bfp16_chunk_offset_bytes,
    pack_expert_down_bfp16,
    parse_expert_down_bfp16,
)
from kernels.moe.patch import PatchedTemplateKernel, TXN_OPC_DDR_PATCH, find_ddr_patches
from quant.bfp16 import parse_bfp16_gemv


def _txn(offsets: list[int]) -> np.ndarray:
    words = [0, 0, 0, 0]
    for offset in offsets:
        words.extend(
            [
                TXN_OPC_DDR_PATCH,
                48,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                0,
                offset & 0xFFFFFFFF,
                offset >> 32,
            ]
        )
    out = np.asarray(words, dtype=np.uint32)
    out[2] = len(offsets)
    out[3] = out.nbytes
    return out


def _dispatch() -> dict[str, int]:
    return {f"e{i}": e for i, e in enumerate([2, 17, 31, 64, 95, 127])}


def test_b1_six_slot_production_taps_are_102_ordered_entries():
    offsets = gate_up_weight_tap_offsets(NUM_ACTIVE_ROUTED)
    assert offsets == [c * B1_W_PER_CHUNK_BYTES // 4 for c in range(102)]


def test_b1_six_slot_template_interface_keeps_seven_slot_default():
    six = MoEGateUpTemplate(slots=6)
    seven = MoEGateUpTemplate()
    assert six.M == 6 * B1_M_PER_EXPERT
    assert six.get_arg_spec()[1].shape == (ROUTED_GU_WORDS,)
    assert seven.M == 7 * B1_M_PER_EXPERT
    assert MoEGateUp(slots=6)._template_op.slots == 6
    assert MoEGateUp()._template_op.slots == 7


def test_legacy_positional_gate_up_apis_keep_d1a8d99_meaning():
    operator = MoEGateUp(2560, 2176, 7, 8)
    template = MoEGateUpTemplate(2560, 15232, 8, None)
    assert (operator.K, operator.M_per_expert, operator.active_experts, operator.cols) == (
        2560,
        2176,
        7,
        8,
    )
    assert operator._template_op.M == 15232
    assert operator._template_op.slots == 7
    assert (template.K, template.M, template.cols, template.context) == (
        2560,
        15232,
        8,
        None,
    )
    assert template.slots == 7


def test_b1_six_slot_patch_table_has_no_shared_expert():
    offsets = [c * B1_W_PER_CHUNK_BYTES for c in range(102)]
    kernel = PatchedTemplateKernel.from_words(
        None,
        _txn(offsets),
        chunks_per_expert=17,
        chunk_bytes=B1_W_PER_CHUNK_BYTES,
        expert_bytes=17 * B1_W_PER_CHUNK_BYTES,
        expected_w_patches=102,
        slots=6,
    )
    patched = kernel.instructions(**_dispatch())
    got = [arg_plus for _, _, arg_plus in find_ddr_patches(patched, 1)]
    expected = [
        expert * (17 * B1_W_PER_CHUNK_BYTES) + chunk * B1_W_PER_CHUNK_BYTES
        for expert in _dispatch().values()
        for chunk in range(17)
    ]
    assert got == expected
    assert all(offset < SHARED_EXPERT_ID * 17 * B1_W_PER_CHUNK_BYTES for offset in got)


def test_down9_dimensions_and_production_140_tap_order():
    assert B2_BFP16_W_PER_CHUNK_BYTES == 184_320
    assert B2_BFP16_W_PER_CHUNK_WORDS == 46_080
    assert DN9_BYTES == 3_686_400
    assert DN9_WORDS == 921_600
    assert TOTAL_DN9_BYTES == 129 * DN9_BYTES
    assert TOTAL_DN9_WORDS == 129 * DN9_WORDS
    assert down_bfp16_weight_tap_offsets() == [
        c * B2_BFP16_W_PER_CHUNK_WORDS for c in range(140)
    ]


def test_down9_pack_roundtrip_and_zero_padding():
    rng = np.random.default_rng(20261028)
    down = rng.integers(-4, 5, (B2_M_PER_EXPERT, B2_K_ORIG)).astype(np.float32)
    packed = pack_expert_down_bfp16(down)
    assert len(packed) == DN9_BYTES
    decoded = parse_bfp16_gemv(packed, K=B2_K_PAD, M=B2_M_PER_EXPERT)
    np.testing.assert_array_equal(decoded[:, :B2_K_ORIG], down)
    np.testing.assert_array_equal(decoded[:, B2_K_ORIG:], 0.0)


def test_down9_parse_expert_uses_expert_order():
    first = np.zeros(DN9_BYTES, dtype=np.uint8)
    down = np.ones((B2_M_PER_EXPERT, B2_K_ORIG), dtype=np.float32)
    second = np.frombuffer(pack_expert_down_bfp16(down), dtype=np.uint8)
    decoded = parse_expert_down_bfp16(np.concatenate([first, second]), 1)
    np.testing.assert_array_equal(decoded[:, :B2_K_ORIG], 1.0)


def test_down9_patch_uses_six_routed_then_shared_128():
    offsets = [c * B2_BFP16_W_PER_CHUNK_BYTES for c in range(140)]
    kernel = PatchedTemplateKernel.from_words(
        None,
        _txn(offsets),
        chunks_per_expert=20,
        chunk_bytes=B2_BFP16_W_PER_CHUNK_BYTES,
        expert_bytes=DN9_BYTES,
        expected_w_patches=140,
        slots=7,
    )
    patched = kernel.instructions(**_dispatch())
    got = [arg_plus for _, _, arg_plus in find_ddr_patches(patched, 1)]
    expected = [
        expert * DN9_BYTES + rnd * B2_BFP16_W_PER_CHUNK_BYTES
        for expert in [2, 17, 31, 64, 95, 127, SHARED_EXPERT_ID]
        for rnd in range(20)
    ]
    assert got == expected
    assert got[-1] == get_down_bfp16_chunk_offset_bytes(SHARED_EXPERT_ID, 19)


def test_benchmark_references_use_selected_expert_ids(monkeypatch):
    q4_seen = []
    bfp_seen = []

    def fake_q4_params(expert, seed):
        q4_seen.append((expert, seed))
        return expert, 0, 0

    def fake_q4_ref(codes, scales, mins, x):
        return None, np.asarray([codes], dtype=np.uint16)

    def fake_bfp_weight(expert, seed, rows, cols):
        bfp_seen.append((expert, seed, rows, cols))
        return np.asarray([[expert]], dtype=np.float32)

    def fake_bfp_ref(weight, x):
        return None, np.asarray([int(weight[0, 0])], dtype=np.uint16)

    monkeypatch.setattr(moe_d_bench, "_q4_params", fake_q4_params)
    monkeypatch.setattr(moe_d_bench, "gemv_bit_reference", fake_q4_ref)
    monkeypatch.setattr(moe_d_bench, "_bfp_weight", fake_bfp_weight)
    monkeypatch.setattr(moe_d_bench, "bfp16_gemv_bit_reference", fake_bfp_ref)

    experts = [2, 17, 31, 64, 95, 127]
    np.testing.assert_array_equal(
        moe_d_bench._q4_ref({"seed": 42}, np.zeros(1), experts), experts
    )
    selected = [*experts, SHARED_EXPERT_ID]
    np.testing.assert_array_equal(
        moe_d_bench._bfp_ref({"seed": 42}, np.zeros((7, 1)), selected), selected
    )
    assert q4_seen == [(expert, 42) for expert in experts]
    assert [item[0] for item in bfp_seen] == selected


def test_down9_operator_interfaces():
    template = MoEDownBfp16Template()
    operator = MoEDownBfp16()
    specs = template.get_arg_spec()
    assert specs[0].shape == (NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,)
    assert specs[1].shape == (TOTAL_DN9_WORDS,)
    assert specs[2].shape == (NUM_ACTIVE_TOTAL * B2_K_PAD,)
    assert list(operator.get_dispatch_params()) == [f"e{i}" for i in range(6)]
