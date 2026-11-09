# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit test for NpuBackend without NPU hardware.

Verifies:
1. Bundle name and row-range mapping for RDA, MLA, Dense, Shared MoE, and LM Head.
2. Bundled GEMV reuse rules: single GEMV execution for multiple projections with
   the same input object in the same step, and proper invalidation on step/input change.
3. MoE expert partial sum accumulation order matching NumpyBackend (ascending module ID).
4. RDA recurrent state ping-pong updating and set/get state persistence.
5. Bit-exact parity with NumpyBackend on synthetic small models.
"""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import npu.backend as backend_mod

from host.backend import NumpyBackend, situ_glu
from host.weights import Weights, bf16_to_f32, f32_to_bf16_rtne
from npu.backend import (
    BundleMeta,
    DWeights,
    NpuBackend,
    _load_manifest_specs,
    bfloat16,
    build_default_manifest_specs,
    ensure_mla_cache_expanded,
    resolve_unified_shape_name,
)
from kernels.gemv_unified.shapes import MAX_K, get_shape
from kernels.gemv_unified.stage2 import pad_x
from quant.bfp16 import pack_bfp16_gemv, parse_bfp16_gemv
from quant.q4nx import dequantize_simple, pack_gemv, parse_gemv, quantize


class FakeWeights(Weights):
    """In-memory Weights implementation for unit testing."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = {k: np.array(v, copy=True) for k, v in tensors.items()}
        self.selected_layers = None
        self.load_global = True
        self.skip_moe_experts = False


def test_bundle_mapping():
    """Verify that HF parameter names correctly map to expected bundles and row ranges."""
    backend = NpuBackend(mock=True)

    # 1. Layer 0 RDA input bundle
    expected_rda_slices = [
        ("model.layers.0.self_attn.q_proj.weight", "L00/rda_in", 0, 5120),
        ("model.layers.0.self_attn.k_proj.weight", "L00/rda_in", 5120, 10240),
        ("model.layers.0.self_attn.v_proj.weight", "L00/rda_in", 10240, 15360),
        ("model.layers.0.self_attn.g_proj.weight", "L00/rda_in", 15360, 20480),
        ("model.layers.0.self_attn.f_a_proj.weight", "L00/rda_in", 20480, 20608),
        ("model.layers.0.self_attn.b_proj.weight", "L00/rda_in", 20608, 20648),
        ("model.layers.0.self_attn.rda_w_proj.weight", "L00/rda_in", 20648, 20688),
    ]
    for hf_name, bname, r0, r1 in expected_rda_slices:
        assert hf_name in backend.comp_map, f"Missing {hf_name}"
        comp = backend.comp_map[hf_name]
        assert comp.bundle_name == bname
        assert comp.row_start == r0
        assert comp.row_end == r1

    # RDA f_b (padded K: 128 -> 256)
    fb_comp = backend.comp_map["model.layers.0.self_attn.f_b_proj.weight"]
    assert fb_comp.bundle_name == "L00/f_b"
    assert fb_comp.orig_k == 128
    assert fb_comp.padded_k == 256
    assert fb_comp.row_start == 0 and fb_comp.row_end == 5120

    # 2. Layer 3 MLA bundle
    assert "model.layers.3.self_attn.q_a_proj.weight" in backend.comp_map
    qa_comp = backend.comp_map["model.layers.3.self_attn.q_a_proj.weight"]
    assert qa_comp.bundle_name == "L03/mla_in"
    assert qa_comp.row_start == 0 and qa_comp.row_end == 576

    kva_comp = backend.comp_map["model.layers.3.self_attn.kv_a_proj_with_mqa.weight"]
    assert kva_comp.bundle_name == "L03/mla_in"
    assert kva_comp.row_start == 576 and kva_comp.row_end == 1152

    # MLA q_b (padded K: 576 -> 768)
    qb_comp = backend.comp_map["model.layers.3.self_attn.q_b_proj.weight"]
    assert qb_comp.bundle_name == "L03/q_b"
    assert qb_comp.orig_k == 576
    assert qb_comp.padded_k == 768
    assert qb_comp.row_start == 0 and qb_comp.row_end == 7680

    # 3. Layer 0 Dense MLP
    d_gate = backend.comp_map["model.layers.0.mlp.gate_proj.weight"]
    d_up = backend.comp_map["model.layers.0.mlp.up_proj.weight"]
    assert d_gate.bundle_name == "L00/dense_gu"
    assert d_gate.row_start == 0 and d_gate.row_end == 12032
    assert d_up.bundle_name == "L00/dense_gu"
    assert d_up.row_start == 12032 and d_up.row_end == 24064

    # 4. MoE Shared Expert gate+up
    sh_g = backend.comp_map["model.layers.1.moe.shared_experts.gate_proj.weight"]
    sh_u = backend.comp_map["model.layers.1.moe.shared_experts.up_proj.weight"]
    assert sh_g.bundle_name == "L01/shared_gu"
    assert sh_g.row_start == 0 and sh_g.row_end == 1088
    assert sh_u.bundle_name == "L01/shared_gu"
    assert sh_u.row_start == 1088 and sh_u.row_end == 2176

    # 5. LM Head
    lm_comp = backend.comp_map["lm_head.weight"]
    assert lm_comp.bundle_name == "lm_head"
    assert lm_comp.row_start == 0 and lm_comp.row_end == 259840


def test_bundle_reuse_rules():
    """Verify that multiple linear calls on the same input bundle reuse GEMV output."""
    backend = NpuBackend(mock=True)

    # Provide mock weights for L00/rda_in: shape [20736, 2560]
    rng = np.random.default_rng(123)
    w_rda_in = rng.standard_normal((20736, 2560)).astype(np.float32) * 0.05
    w_rda_in_u16 = f32_to_bf16_rtne(w_rda_in)
    backend.mock_weight_arrays["L00/rda_in"] = w_rda_in_u16

    # Create input activation
    hidden = rng.standard_normal(2560).astype(np.float32)
    hidden_u16 = f32_to_bf16_rtne(hidden)

    # Reset metrics
    backend.cache_hits = 0
    backend.cache_misses = 0

    # 1. Call q_proj -> should trigger GEMV calculation (miss)
    q_out = backend.linear("model.layers.0.self_attn.q_proj.weight", hidden_u16)
    assert backend.cache_misses == 1
    assert backend.cache_hits == 0
    assert q_out.shape == (5120,)

    # 2. Call remaining projections with same hidden object -> must HIT cache
    other_projs = [
        ("model.layers.0.self_attn.k_proj.weight", 5120),
        ("model.layers.0.self_attn.v_proj.weight", 5120),
        ("model.layers.0.self_attn.g_proj.weight", 5120),
        ("model.layers.0.self_attn.f_a_proj.weight", 128),
        ("model.layers.0.self_attn.b_proj.weight", 40),
        ("model.layers.0.self_attn.rda_w_proj.weight", 40),
    ]

    for p_name, expected_len in other_projs:
        out = backend.linear(p_name, hidden_u16)
        assert out.shape == (expected_len,)

    # Cache hits should be 6, misses remained 1!
    assert backend.cache_hits == 6
    assert backend.cache_misses == 1

    # Verify slice correctness against direct matrix multiplication
    w_used_f32 = bf16_to_f32(w_rda_in_u16)
    full_gemv_f32 = np.dot(bf16_to_f32(hidden_u16), w_used_f32.T)
    expected_full_u16 = f32_to_bf16_rtne(full_gemv_f32)

    assert np.array_equal(q_out, expected_full_u16[0:5120])
    k_out = backend.linear("model.layers.0.self_attn.k_proj.weight", hidden_u16)
    assert np.array_equal(k_out, expected_full_u16[5120:10240])

    # 3. Different input array instance -> must NOT reuse previous cache
    different_hidden = rng.standard_normal(2560).astype(np.float32)
    different_hidden_u16 = f32_to_bf16_rtne(different_hidden)

    _ = backend.linear("model.layers.0.self_attn.q_proj.weight", different_hidden_u16)
    assert backend.cache_misses == 2

    # 4. Same hidden object but advanced step_id -> must NOT reuse stale step cache
    backend.set_step(1)
    _ = backend.linear("model.layers.0.self_attn.q_proj.weight", hidden_u16)
    assert backend.cache_misses == 3


def test_moe_order_matches_numpy_backend():
    """Verify that MoE expert partial sum accumulation order strictly matches NumpyBackend."""
    rng = np.random.default_rng(42)
    layer = 1
    dim = 2560
    moe_dim = 1088

    # Create dummy expert weights for 8 routed experts
    top_ids = np.array([45, 12, 88, 3, 102, 55], dtype=np.int32)
    top_weights = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05], dtype=np.float32)

    weights_dict: dict[str, np.ndarray] = {}
    for eid in top_ids:
        g = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        u = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        d = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"] = f32_to_bf16_rtne(g)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"] = f32_to_bf16_rtne(u)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"] = f32_to_bf16_rtne(d)

    fake_weights = FakeWeights(weights_dict)
    numpy_backend = NumpyBackend(fake_weights)
    npu_backend = NpuBackend(host_weights=fake_weights, mock=True)

    x = rng.standard_normal(dim).astype(np.float32)
    x_u16 = f32_to_bf16_rtne(x)

    ref_out = numpy_backend.moe_experts(layer, x_u16, top_ids, top_weights)
    npu_out = npu_backend.moe_experts(layer, x_u16, top_ids, top_weights)

    # In mock mode, NpuBackend._moe_experts_numpy executes the same ascending sort
    diff = int(np.count_nonzero(ref_out != npu_out))
    assert diff == 0, f"MoE output mismatch between NpuBackend and NumpyBackend: {diff} elements"


def test_rda_update_recurrent_state():
    """Verify RDA recurrent state update, double buffering, and state get/set."""
    rng = np.random.default_rng(999)
    backend = NpuBackend(mock=True)

    H, V, K = 40, 128, 128
    layer = 0

    # Initial state
    s_init = rng.standard_normal((H, V, K)).astype(np.float32) * 0.01
    backend.set_rda_state(layer, s_init)

    retrieved = backend.get_rda_state(layer)
    assert retrieved is not None
    assert np.allclose(retrieved.squeeze(0), s_init)

    # Generate 1 step inputs
    q = f32_to_bf16_rtne(rng.standard_normal((H, K)).astype(np.float32))
    k = f32_to_bf16_rtne(rng.standard_normal((H, K)).astype(np.float32))
    v = f32_to_bf16_rtne(rng.standard_normal((H, V)).astype(np.float32))
    log_alpha = rng.uniform(-5.0, 0.0, (H, K)).astype(np.float32)
    beta = f32_to_bf16_rtne(rng.uniform(0.1, 0.9, H).astype(np.float32))

    # Compare against pure NumPy reference
    ref_backend = NumpyBackend(FakeWeights({}))
    ref_backend.set_rda_state(layer, s_init)

    out_ref = ref_backend.rda_update(layer, q, k, v, log_alpha, beta)
    out_npu = backend.rda_update(layer, q, k, v, log_alpha, beta)

    assert np.array_equal(out_ref, out_npu)

    # State in backend should also match ref state
    st_ref = ref_backend.get_rda_state(layer)
    st_npu = backend.get_rda_state(layer)
    assert np.allclose(st_ref, st_npu, atol=1e-7)


def test_mla_attend():
    """Verify MLA attention computation and KV cache update parity with NumpyBackend."""
    rng = np.random.default_rng(777)
    backend = NpuBackend(mock=True)
    ref_backend = NumpyBackend(FakeWeights({}))

    heads = 40
    qk_dim = 192
    v_dim = 128
    scale = qk_dim ** -0.5
    layer = 3

    # Step 1
    q1 = f32_to_bf16_rtne(rng.standard_normal((heads, qk_dim)).astype(np.float32))
    k1 = f32_to_bf16_rtne(rng.standard_normal((heads, qk_dim)).astype(np.float32))
    v1 = f32_to_bf16_rtne(rng.standard_normal((heads, v_dim)).astype(np.float32))

    out_npu_1 = backend.mla_attend(layer, q1, k1, v1, scale)
    out_ref_1 = ref_backend.mla_attend(layer, q1, k1, v1, scale)
    assert np.array_equal(out_npu_1, out_ref_1)

    # Step 2
    q2 = f32_to_bf16_rtne(rng.standard_normal((heads, qk_dim)).astype(np.float32))
    k2 = f32_to_bf16_rtne(rng.standard_normal((heads, qk_dim)).astype(np.float32))
    v2 = f32_to_bf16_rtne(rng.standard_normal((heads, v_dim)).astype(np.float32))

    out_npu_2 = backend.mla_attend(layer, q2, k2, v2, scale)
    out_ref_2 = ref_backend.mla_attend(layer, q2, k2, v2, scale)
    assert np.array_equal(out_npu_2, out_ref_2)

    # Verify cache states match
    k_npu, v_npu = backend.get_mla_cache(layer)
    k_ref, v_ref = ref_backend.get_mla_cache(layer)
    assert np.array_equal(k_npu, k_ref)
    assert np.array_equal(v_npu, v_ref)


def test_mla_kv_b_interleave():
    """Verify that kv_b_proj interleaves 40 heads [k_h, v_h] correctly."""
    backend = NpuBackend(mock=True)
    rng = np.random.default_rng(555)

    # Set mock weights for k_b [5120, 512] and v_b [5120, 512]
    w_k = f32_to_bf16_rtne(rng.standard_normal((5120, 512)).astype(np.float32) * 0.05)
    w_v = f32_to_bf16_rtne(rng.standard_normal((5120, 512)).astype(np.float32) * 0.05)
    backend.mock_weight_arrays["L03/k_b"] = w_k
    backend.mock_weight_arrays["L03/v_b"] = w_v

    x = f32_to_bf16_rtne(rng.standard_normal(512).astype(np.float32))
    kv_out = backend.linear("model.layers.3.self_attn.kv_b_proj.weight", x)

    assert kv_out.shape == (10240,)
    # Reshape to [40, 256]
    kv_heads = kv_out.reshape(40, 256)
    k_expected = backend._execute_gemv("L03/k_b", backend.bundle_map["L03/k_b"], x).reshape(40, 128)
    v_expected = backend._execute_gemv("L03/v_b", backend.bundle_map["L03/v_b"], x).reshape(40, 128)

    assert np.array_equal(kv_heads[:, :128], k_expected)
    assert np.array_equal(kv_heads[:, 128:], v_expected)


def test_lm_head_slice():
    """Verify that lm_head slices original vocab rows [259840] from padded [260096]."""
    backend = NpuBackend(mock=True)
    rng = np.random.default_rng(888)

    w_lm = f32_to_bf16_rtne(rng.standard_normal((260096, 2560)).astype(np.float32) * 0.01)
    backend.mock_weight_arrays["lm_head"] = w_lm

    x = f32_to_bf16_rtne(rng.standard_normal(2560).astype(np.float32))
    logits = backend.linear("lm_head.weight", x)

    assert logits.shape == (259840,)
    full_out = backend._execute_gemv("lm_head", backend.bundle_map["lm_head"], x)
    assert np.array_equal(logits, full_out[:259840])


def test_raon_model_step_parity():
    """Verify RaonModel runs end-to-end with NpuBackend in mock mode and matches NumpyBackend."""
    from host.raon import RaonConfig, RaonModel

    rng = np.random.default_rng(2026)
    c = 64
    vocab = 128
    inter = 128
    moe_inter = 32

    cfg = RaonConfig(
        hidden_size=c,
        num_hidden_layers=2,
        vocab_size=vocab,
        intermediate_size=inter,
        first_k_dense_replace=1,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=moe_inter,
        num_shared_experts=1,
        rda_num_heads=4,
        rda_head_dim=16,
        rda_layers=[0, 1],
        mla_layers=[],
    )

    # Populate all necessary weights
    tensors: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": f32_to_bf16_rtne(rng.standard_normal((vocab, c)).astype(np.float32) * 0.05),
        "model.norm.weight": rng.standard_normal(c).astype(np.float32),
        "lm_head.weight": f32_to_bf16_rtne(rng.standard_normal((vocab, c)).astype(np.float32) * 0.05),
    }

    for ly in range(2):
        lpfx = f"model.layers.{ly}"
        tensors[f"{lpfx}.input_layernorm.weight"] = np.ones(c, dtype=np.float32)
        tensors[f"{lpfx}.post_attention_layernorm.weight"] = np.ones(c, dtype=np.float32)

        # mHC weights
        for site in ["self_attn_raon_mhc", "mlp_raon_mhc"]:
            spfx = f"{lpfx}.{site}"
            tensors[f"{spfx}.phi_pre"] = rng.standard_normal((4, 4 * c)).astype(np.float32) * 0.01
            tensors[f"{spfx}.phi_post"] = rng.standard_normal((4, 4 * c)).astype(np.float32) * 0.01
            tensors[f"{spfx}.res.phi_z"] = rng.standard_normal((9, 4 * c)).astype(np.float32) * 0.01
            tensors[f"{spfx}.alpha_pre"] = np.float32(1.0)
            tensors[f"{spfx}.alpha_post"] = np.float32(1.0)
            tensors[f"{spfx}.res.alpha_z"] = np.float32(1.0)
            tensors[f"{spfx}.b_pre"] = np.zeros(4, dtype=np.float32)
            tensors[f"{spfx}.b_post"] = np.zeros(4, dtype=np.float32)
            tensors[f"{spfx}.res.b_z"] = np.zeros(9, dtype=np.float32)
            tensors[f"{spfx}.res.eta"] = np.float32(0.0)

        # RDA projections
        apfx = f"{lpfx}.self_attn"
        heads = 4
        hdim = 16
        tot_dim = heads * hdim
        for name in ["q_proj", "k_proj", "v_proj", "g_proj"]:
            tensors[f"{apfx}.{name}.weight"] = f32_to_bf16_rtne(rng.standard_normal((tot_dim, c)).astype(np.float32) * 0.05)
        tensors[f"{apfx}.b_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((heads, c)).astype(np.float32) * 0.05)
        tensors[f"{apfx}.rda_w_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((heads, c)).astype(np.float32) * 0.05)
        tensors[f"{apfx}.f_a_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((16, c)).astype(np.float32) * 0.05)
        tensors[f"{apfx}.f_b_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((tot_dim, 16)).astype(np.float32) * 0.05)
        tensors[f"{apfx}.o_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((c, tot_dim)).astype(np.float32) * 0.05)
        tensors[f"{apfx}.o_norm.weight"] = np.ones(hdim, dtype=np.float32)
        tensors[f"{apfx}.dt_bias"] = np.zeros(tot_dim, dtype=np.float32)
        tensors[f"{apfx}.A_log"] = np.zeros(heads, dtype=np.float32)
        tensors[f"{apfx}.q_conv1d.weight"] = np.ones((tot_dim, 4), dtype=np.float32) / 4.0
        tensors[f"{apfx}.k_conv1d.weight"] = np.ones((tot_dim, 4), dtype=np.float32) / 4.0
        tensors[f"{apfx}.v_conv1d.weight"] = np.ones((tot_dim, 4), dtype=np.float32) / 4.0

        if ly == 0:
            # Dense MLP
            mpfx = f"{lpfx}.mlp"
            tensors[f"{mpfx}.gate_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((inter, c)).astype(np.float32) * 0.05)
            tensors[f"{mpfx}.up_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((inter, c)).astype(np.float32) * 0.05)
            tensors[f"{mpfx}.down_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((c, inter)).astype(np.float32) * 0.05)
        else:
            # MoE MLP
            moepfx = f"{lpfx}.moe"
            tensors[f"{moepfx}.gate.weight"] = rng.standard_normal((8, c)).astype(np.float32) * 0.05
            tensors[f"{moepfx}.gate.e_score_correction_bias"] = np.zeros(8, dtype=np.float32)
            for eid in range(8):
                tensors[f"{moepfx}.experts.{eid}.gate_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((moe_inter, c)).astype(np.float32) * 0.05)
                tensors[f"{moepfx}.experts.{eid}.up_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((moe_inter, c)).astype(np.float32) * 0.05)
                tensors[f"{moepfx}.experts.{eid}.down_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((c, moe_inter)).astype(np.float32) * 0.05)
            # Shared experts
            tensors[f"{moepfx}.shared_experts.gate_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((moe_inter, c)).astype(np.float32) * 0.05)
            tensors[f"{moepfx}.shared_experts.up_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((moe_inter, c)).astype(np.float32) * 0.05)
            tensors[f"{moepfx}.shared_experts.down_proj.weight"] = f32_to_bf16_rtne(rng.standard_normal((c, moe_inter)).astype(np.float32) * 0.05)

    weights = FakeWeights(tensors)
    ref_backend = NumpyBackend(weights)
    npu_backend = NpuBackend(host_weights=weights, mock=True)

    ref_model = RaonModel(cfg, weights, backend=ref_backend)
    npu_model = RaonModel(cfg, weights, backend=npu_backend)

    # Step 1
    ref_logits = ref_model.step(7)
    npu_backend.set_step(1)
    npu_logits = npu_model.step(7)

    denom = np.linalg.norm(ref_logits) * np.linalg.norm(npu_logits)
    cos = float(np.dot(ref_logits, npu_logits) / denom)
    assert cos >= 0.9999, f"Step 1 logits cosine {cos:.6f} < 0.9999"
    assert int(np.argmax(ref_logits)) == int(np.argmax(npu_logits))


# -----------------------------------------------------------------------------
# New tests covering fix1 items 1..6
# -----------------------------------------------------------------------------

def test_manifest_validation(tmp_path: Path):
    """Verify manifest loading, host bundle skipping, and validation failures on schema changes."""
    sample_path = Path(__file__).parent / "testdata" / "manifest_sample.json"
    assert sample_path.is_file(), f"Missing manifest sample at {sample_path}"

    # 1. Normal load succeeds and skips host_safetensors bundles
    comp_map, bundle_map = build_default_manifest_specs()
    _load_manifest_specs(sample_path, comp_map, bundle_map, require_complete=False)

    with open(sample_path, "r", encoding="utf-8") as f:
        sample_dict = json.load(f)

    host_entries = [
        b for b in sample_dict.get("bundles", []) if b.get("format") == "host_safetensors"
    ]
    assert len(host_entries) > 0, "manifest_sample.json must contain host_safetensors bundle"

    # 2. Mutated format raises ValueError
    bad_fmt = copy.deepcopy(sample_dict)
    bad_fmt["bundles"][0]["format"] = "invalid_format_name"
    p_bad_fmt = tmp_path / "bad_fmt.json"
    p_bad_fmt.write_text(json.dumps(bad_fmt), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest format mismatch"):
        c_map, b_map = build_default_manifest_specs()
        _load_manifest_specs(p_bad_fmt, c_map, b_map, require_complete=False)

    # 3. Mutated padded_shape raises ValueError
    bad_shape = copy.deepcopy(sample_dict)
    orig_shape = bad_shape["bundles"][0]["padded_shape"]
    bad_shape["bundles"][0]["padded_shape"] = [orig_shape[0] + 128, orig_shape[1]]
    p_bad_shape = tmp_path / "bad_shape.json"
    p_bad_shape.write_text(json.dumps(bad_shape), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest padded_shape mismatch"):
        c_map, b_map = build_default_manifest_specs()
        _load_manifest_specs(p_bad_shape, c_map, b_map, require_complete=False)

    # 4. Mutated component row range raises ValueError
    bad_range = copy.deepcopy(sample_dict)
    mutated = False
    for b in bad_range["bundles"]:
        if b.get("components"):
            for comp in b["components"]:
                if "row_end" in comp:
                    comp["row_end"] += 1
                    mutated = True
                    break
        if mutated:
            break
    assert mutated, "Failed to find component to mutate"
    p_bad_range = tmp_path / "bad_range.json"
    p_bad_range.write_text(json.dumps(bad_range), encoding="utf-8")
    with pytest.raises(ValueError, match="Row range mismatch"):
        c_map, b_map = build_default_manifest_specs()
        _load_manifest_specs(p_bad_range, c_map, b_map, require_complete=False)


def _write_complete_manifest(weights_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Create a complete canonical manifest backed by sparse canonical-size files."""
    import hashlib
    import npu.backend as backend_mod

    comp_map, bundle_map = build_default_manifest_specs()
    entries: list[dict[str, Any]] = []
    for index, meta in enumerate(bundle_map.values()):
        rel_file = f"files/{index:03d}.bin"
        size_bytes, _ = backend_mod._canonical_bundle_size_bytes(meta)
        file_path = weights_dir / rel_file
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as stream:
            stream.truncate(size_bytes)
        sha256 = "0" * 64
        if index == 0:
            hasher = hashlib.sha256()
            with file_path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(block)
            sha256 = hasher.hexdigest()
        components = []
        for component in comp_map.values():
            if component.bundle_name != meta.name:
                continue
            entry_component: dict[str, Any] = {
                "name": component.hf_name,
                "row_start": component.row_start,
                "row_end": component.row_end,
            }
            entry_component.update(component.extra)
            components.append(entry_component)
        entry: dict[str, Any] = {
            "bundle": meta.name,
            "format": meta.fmt,
            "file": rel_file,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "original_shape": meta.orig_shape,
            "padded_shape": meta.padded_shape,
            "components": components,
        }
        if meta.expert_count:
            entry["expert_count"] = meta.expert_count
            entry["expert_bytes"] = meta.expert_bytes
        entries.append(entry)
    entries.append({
        "bundle": "host_global",
        "format": "host_safetensors",
        "file": "host_global.safetensors",
        "size_bytes": 0,
        "sha256": "0" * 64,
    })
    payload = {"bundles": entries}
    manifest = weights_dir / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    return manifest, payload


def _load_integrity_manifest(weights_dir: Path, payload: dict[str, Any]) -> None:
    manifest = weights_dir / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    comp_map, bundle_map = build_default_manifest_specs()
    _load_manifest_specs(manifest, comp_map, bundle_map, weights_dir=weights_dir)


def test_manifest_integrity_rejects_size_sha_and_bundle_set(tmp_path: Path):
    import hashlib
    import npu.backend as backend_mod

    manifest, original = _write_complete_manifest(tmp_path)

    bad_size = copy.deepcopy(original)
    first_file = tmp_path / bad_size["bundles"][0]["file"]
    reduced_size = bad_size["bundles"][0]["size_bytes"] - 4
    with first_file.open("r+b") as stream:
        stream.truncate(reduced_size)
    bad_size["bundles"][0]["size_bytes"] = reduced_size
    bad_size["bundles"][0]["sha256"] = hashlib.sha256(first_file.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonical size_bytes mismatch"):
        _load_integrity_manifest(tmp_path, bad_size)
    with first_file.open("r+b") as stream:
        stream.truncate(original["bundles"][0]["size_bytes"])

    bad_sha = copy.deepcopy(original)
    bad_sha["bundles"][0]["sha256"] = "f" * 64
    _load_integrity_manifest(tmp_path, bad_sha)
    comp_map, bundle_map = build_default_manifest_specs()
    _load_manifest_specs(manifest, comp_map, bundle_map, weights_dir=tmp_path)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        backend_mod._read_verified_bundle(tmp_path, bundle_map[bad_sha["bundles"][0]["bundle"]])

    missing = copy.deepcopy(original)
    missing["bundles"].pop(0)
    with pytest.raises(ValueError, match="missing NPU bundles"):
        _load_integrity_manifest(tmp_path, missing)

    duplicate = copy.deepcopy(original)
    duplicate["bundles"].insert(1, copy.deepcopy(duplicate["bundles"][0]))
    with pytest.raises(ValueError, match="Duplicate NPU bundle"):
        _load_integrity_manifest(tmp_path, duplicate)

    unknown = copy.deepcopy(original)
    unknown["bundles"][0]["bundle"] = "L99/unknown"
    with pytest.raises(ValueError, match="Unknown NPU bundle"):
        _load_integrity_manifest(tmp_path, unknown)

    expert_entry = next(
        entry for entry in original["bundles"] if entry.get("expert_count")
    )
    for field in ("expert_bytes", "expert_count"):
        bad_expert = copy.deepcopy(original)
        target = next(entry for entry in bad_expert["bundles"] if entry["bundle"] == expert_entry["bundle"])
        target[field] += 1
        with pytest.raises(ValueError, match=field):
            _load_integrity_manifest(tmp_path, bad_expert)

    with_component = next(entry for entry in original["bundles"] if entry["components"])
    missing_component = copy.deepcopy(original)
    target = next(entry for entry in missing_component["bundles"] if entry["bundle"] == with_component["bundle"])
    target["components"].pop()
    with pytest.raises(ValueError, match="missing canonical components"):
        _load_integrity_manifest(tmp_path, missing_component)

    unknown_component = copy.deepcopy(original)
    unknown_component["bundles"][0]["components"].append({"name": "not.canonical.weight"})
    with pytest.raises(ValueError, match="Unknown component"):
        _load_integrity_manifest(tmp_path, unknown_component)


def test_canonical_bundle_sizes_are_complete_and_expert_bytes_match():
    import npu.backend as backend_mod

    _, bundle_map = build_default_manifest_specs()
    for meta in bundle_map.values():
        size_bytes, matrix_bytes = backend_mod._canonical_bundle_size_bytes(meta)
        assert size_bytes > 0
        if meta.expert_count:
            assert meta.expert_bytes == matrix_bytes
            assert size_bytes == matrix_bytes * meta.expert_count


def test_real_manifest_matches_canonical_sizes_when_available():
    import os

    weights_value = os.environ.get("RAON_WEIGHTS_D")
    if not weights_value:
        pytest.skip("RAON_WEIGHTS_D is not set; real D manifest unavailable")
    weights_dir = Path(weights_value)
    manifest = weights_dir / "manifest.json"
    if not manifest.is_file():
        pytest.skip(f"RAON_WEIGHTS_D has no manifest.json: {weights_dir}")
    comp_map, bundle_map = build_default_manifest_specs()
    _load_manifest_specs(manifest, comp_map, bundle_map, weights_dir=weights_dir)


def test_manifest_integrity_rejects_path_escape(tmp_path: Path):
    _, payload = _write_complete_manifest(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside")
    escaped = copy.deepcopy(payload)
    escaped["bundles"][0]["file"] = f"../{outside.name}"
    with pytest.raises(ValueError, match="Unsafe file path"):
        _load_integrity_manifest(tmp_path, escaped)

    absolute = copy.deepcopy(payload)
    absolute["bundles"][0]["file"] = str(outside)
    with pytest.raises(ValueError, match="Unsafe file path"):
        _load_integrity_manifest(tmp_path, absolute)

    link = tmp_path / "outside-link.bin"
    link.symlink_to(outside)
    linked = copy.deepcopy(payload)
    linked["bundles"][0]["file"] = link.name
    with pytest.raises(ValueError, match="escapes weights directory"):
        _load_integrity_manifest(tmp_path, linked)


def test_manifest_required_for_real_npu_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import npu.backend as backend_mod

    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    monkeypatch.setattr(NpuBackend, "_init_hardware_operators", lambda self: None)
    with pytest.raises(FileNotFoundError, match="manifest.json is required"):
        NpuBackend(weights_dir=tmp_path, mock=False)


def test_bundle_sha256_is_cached_by_file_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import npu.backend as backend_mod

    _write_complete_manifest(tmp_path)
    backend_mod._FILE_SHA256_CACHE.clear()
    real_sha256 = backend_mod.hashlib.sha256
    calls = 0

    def counting_sha256(data: bytes):
        nonlocal calls
        calls += 1
        return real_sha256(data)

    monkeypatch.setattr(backend_mod.hashlib, "sha256", counting_sha256)
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    monkeypatch.setattr(NpuBackend, "_init_hardware_operators", lambda self: None)
    monkeypatch.setattr(
        backend_mod, "aie_utils", SimpleNamespace(tensor=lambda words: words.copy())
    )
    first = NpuBackend(weights_dir=tmp_path, mock=False)
    bundle_name = next(iter(first.bundle_map))
    first._get_weight_tensor(bundle_name)
    second = NpuBackend(weights_dir=tmp_path, mock=False)
    second._get_weight_tensor(bundle_name)
    assert calls == 1


def test_mock_gemv_without_array_or_d_file_raises():
    backend = NpuBackend(mock=True)
    meta = backend.bundle_map["L00/dense_gu"]
    with pytest.raises(FileNotFoundError, match="No mock array or D bundle"):
        backend._execute_gemv(
            meta.name, meta, np.zeros(meta.k, dtype=np.uint16)
        )


def test_moe_order_with_fake_operators(monkeypatch: pytest.MonkeyPatch):
    """Verify that MoE execution via fake B1/shared/B2 operators matches NumpyBackend bit-for-bit."""
    import npu.backend as backend_mod
    from kernels.moe.pack import (
        B1_K,
        B1_M_PER_EXPERT,
        B2_K_PAD,
        B2_M_PER_EXPERT,
        MOE_INTERMEDIATE_SIZE,
        NUM_ACTIVE_TOTAL,
    )

    class FakeTensor:
        def __init__(self, size: int, dtype: Any = bfloat16):
            self._arr = np.zeros(size, dtype=dtype)

        def __setitem__(self, idx: Any, val: Any) -> None:
            self._arr[idx] = val

        def __getitem__(self, idx: Any) -> Any:
            return self._arr[idx]

        def numpy(self) -> np.ndarray:
            return self._arr  # View

        def to(self, device: str) -> FakeTensor:
            return self

    rng = np.random.default_rng(2026)
    layer = 1
    dim = 2560
    moe_dim = 1088
    top_ids = np.array([45, 12, 88, 3, 102, 55], dtype=np.int32)
    top_weights = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05], dtype=np.float32)

    weights_dict: dict[str, np.ndarray] = {}
    for eid in top_ids:
        g = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        u = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        d = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"] = f32_to_bf16_rtne(g)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"] = f32_to_bf16_rtne(u)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"] = f32_to_bf16_rtne(d)

    # Shared expert
    g_sh = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
    u_sh = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
    d_sh = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
    weights_dict[f"model.layers.{layer}.moe.shared_experts.gate_proj.weight"] = f32_to_bf16_rtne(g_sh)
    weights_dict[f"model.layers.{layer}.moe.shared_experts.up_proj.weight"] = f32_to_bf16_rtne(u_sh)
    weights_dict[f"model.layers.{layer}.moe.shared_experts.down_proj.weight"] = f32_to_bf16_rtne(d_sh)

    fake_weights = FakeWeights(weights_dict)
    numpy_backend = NumpyBackend(fake_weights)

    # Configure NpuBackend with mock=False and fake operators
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend = NpuBackend(host_weights=fake_weights, mock=True, gemv="legacy")
    backend.mock = False

    backend._work_buffers = {
        "moe_y_gu": FakeTensor(6 * B1_M_PER_EXPERT),
        "moe_y_sh": FakeTensor(B1_M_PER_EXPERT),
        "moe_y_dn": FakeTensor(NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT),
        "moe_x_gu": FakeTensor(B1_K),
        "moe_x_dn": FakeTensor(NUM_ACTIVE_TOTAL * B2_K_PAD),
    }

    def run_gu(y_gu: Any, w_gu: Any, x_gu: Any, **scalars: int) -> None:
        x_u16 = np.array(x_gu.numpy().view(np.uint16), copy=True)[:B1_K]
        x_f32 = bf16_to_f32(x_u16)
        gu_out = np.zeros((6, B1_M_PER_EXPERT), dtype=np.uint16)
        for j in range(6):
            eid = scalars[f"e{j}"]
            g_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"])
            u_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"])
            g_out = f32_to_bf16_rtne(np.dot(x_f32, g_w.T))
            u_out = f32_to_bf16_rtne(np.dot(x_f32, u_w.T))
            gu_out[j, :MOE_INTERMEDIATE_SIZE] = g_out
            gu_out[j, MOE_INTERMEDIATE_SIZE:] = u_out
        y_gu[:] = gu_out.reshape(-1).view(bfloat16)

    def run_sh(y_sh: Any, w_sh: Any, x_gu: Any) -> None:
        x_u16 = np.array(x_gu.numpy().view(np.uint16), copy=True)[:B1_K]
        x_f32 = bf16_to_f32(x_u16)
        g_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.gate_proj.weight"])
        u_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.up_proj.weight"])
        g_out = f32_to_bf16_rtne(np.dot(x_f32, g_w.T))
        u_out = f32_to_bf16_rtne(np.dot(x_f32, u_w.T))
        sh_out = np.zeros(B1_M_PER_EXPERT, dtype=np.uint16)
        sh_out[:MOE_INTERMEDIATE_SIZE] = g_out
        sh_out[MOE_INTERMEDIATE_SIZE:] = u_out
        y_sh[:] = sh_out.view(bfloat16)

    def run_dn(y_dn: Any, w_dn: Any, x_dn: Any, **scalars: int) -> None:
        x_arr = np.array(x_dn.numpy().view(np.uint16), copy=True).reshape(NUM_ACTIVE_TOTAL, B2_K_PAD)
        dn_out = np.zeros((NUM_ACTIVE_TOTAL, B2_M_PER_EXPERT), dtype=np.uint16)
        for j in range(6):
            eid = scalars[f"e{j}"]
            d_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"])
            act_u16 = x_arr[j, :MOE_INTERMEDIATE_SIZE]
            dn_out[j] = f32_to_bf16_rtne(np.dot(bf16_to_f32(act_u16), d_w.T))
        sh_dw = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.down_proj.weight"])
        sh_act_u16 = x_arr[6, :MOE_INTERMEDIATE_SIZE]
        dn_out[6] = f32_to_bf16_rtne(np.dot(bf16_to_f32(sh_act_u16), sh_dw.T))
        y_dn[:] = dn_out.reshape(-1).view(bfloat16)

    backend._get_or_compile_moe_ops = lambda: (run_gu, run_sh, run_dn)
    backend._get_weight_tensor = lambda name: "fake_weight_tensor"

    x = rng.standard_normal(dim).astype(np.float32)
    x_u16 = f32_to_bf16_rtne(x)

    ref_out = numpy_backend.moe_experts(layer, x_u16, top_ids, top_weights)
    npu_out = backend.moe_experts(layer, x_u16, top_ids, top_weights)

    diff = int(np.count_nonzero(ref_out != npu_out))
    assert diff == 0, f"MoE fake op output mismatch: {diff} elements differed"
    assert np.array_equal(ref_out, npu_out)


def test_unified_hardware_path_x_layout_and_padding(monkeypatch: pytest.MonkeyPatch):
    """Verify hardware path passes pad_x buffers (MAX_K and 7*MAX_K) with exact layout and zero-padded tails."""
    import npu.backend as backend_mod
    from kernels.gemv_unified.shapes import MAX_K
    from kernels.moe.pack import (
        B1_K,
        B1_M_PER_EXPERT,
        B2_K_PAD,
        B2_M_PER_EXPERT,
        MOE_INTERMEDIATE_SIZE,
        NUM_ACTIVE_TOTAL,
    )

    class FakeTensor:
        def __init__(self, size: int, dtype: Any = bfloat16):
            self._arr = np.zeros(size, dtype=dtype)

        def __setitem__(self, idx: Any, val: Any) -> None:
            self._arr[idx] = val

        def __getitem__(self, idx: Any) -> Any:
            return self._arr[idx]

        def numpy(self) -> np.ndarray:
            return self._arr

        def to(self, device: str) -> FakeTensor:
            return self

    rng = np.random.default_rng(20261030)
    layer = 1
    dim = 2560
    moe_dim = 1088
    top_ids = np.array([45, 12, 88, 3, 102, 55], dtype=np.int32)
    top_weights = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05], dtype=np.float32)

    weights_dict: dict[str, np.ndarray] = {}
    for eid in top_ids:
        g = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        u = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        d = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"] = f32_to_bf16_rtne(g)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"] = f32_to_bf16_rtne(u)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"] = f32_to_bf16_rtne(d)

    # Shared expert
    g_sh = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
    u_sh = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
    d_sh = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
    weights_dict[f"model.layers.{layer}.moe.shared_experts.gate_proj.weight"] = f32_to_bf16_rtne(g_sh)
    weights_dict[f"model.layers.{layer}.moe.shared_experts.up_proj.weight"] = f32_to_bf16_rtne(u_sh)
    weights_dict[f"model.layers.{layer}.moe.shared_experts.down_proj.weight"] = f32_to_bf16_rtne(d_sh)

    fake_weights = FakeWeights(weights_dict)
    numpy_backend = NumpyBackend(fake_weights)

    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend = NpuBackend(host_weights=fake_weights, mock=True, gemv="unified")
    backend.mock = False

    bmeta_fb = backend.bundle_map["L00/f_b"]  # k=256, m=5120
    x_fb = np.full(bmeta_fb.k, 0x3F80, dtype=np.uint16)

    gemv_inspected: dict[str, Any] = {}

    def fake_gemv(y: Any, w: Any, x: Any) -> None:
        x_raw = np.array(x.numpy().view(np.uint16), copy=True)
        gemv_inspected["len"] = len(x_raw)
        gemv_inspected["k_vals"] = x_raw[:bmeta_fb.k].copy()
        gemv_inspected["tail_all_zero"] = bool(np.all(x_raw[bmeta_fb.k:] == 0))
        y._arr[:] = np.full(bmeta_fb.m, 0x4000, dtype=np.uint16).view(bfloat16)

    backend._get_or_compile_gemv = lambda k, m, fmt: fake_gemv
    backend._get_weight_tensor = lambda name: "fake_w"

    # 1. Missing buffer check on unified GEMV: must raise RuntimeError
    with pytest.raises(RuntimeError, match="gemv_x_unified work buffer missing"):
        backend._execute_gemv("L00/f_b", bmeta_fb, x_fb)

    # 2. Setup pre-allocated work buffers matching _init_hardware_operators for unified
    backend._work_buffers = {
        "moe_y_gu": FakeTensor(6 * B1_M_PER_EXPERT),
        "moe_y_sh": FakeTensor(B1_M_PER_EXPERT),
        "moe_y_dn": FakeTensor(NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT),
        "moe_x_gu": FakeTensor(MAX_K),
        "moe_x_dn": FakeTensor(NUM_ACTIVE_TOTAL * MAX_K),
        "gemv_x_unified": FakeTensor(MAX_K),
        f"gemv_y_{bmeta_fb.m}": FakeTensor(bmeta_fb.m),
    }

    # 3. Test _execute_gemv hardware path pad_x size and tail padding
    res_gemv = backend._execute_gemv("L00/f_b", bmeta_fb, x_fb)
    assert gemv_inspected["len"] == MAX_K
    assert np.array_equal(gemv_inspected["k_vals"], x_fb)
    assert gemv_inspected["tail_all_zero"] is True
    assert np.all(res_gemv == 0x4000)

    # 4. Test MoE hardware path pad_x verification: X length, B2 slot layout, zero-padded tails
    b1_inspected: dict[str, Any] = {}
    sh_inspected: dict[str, Any] = {}
    b2_inspected: dict[str, Any] = {}

    def run_gu(y_gu: Any, w_gu: Any, x_gu: Any, **scalars: int) -> None:
        x_raw = np.array(x_gu.numpy().view(np.uint16), copy=True)
        b1_inspected["len"] = len(x_raw)
        b1_inspected["head_vals"] = x_raw[:B1_K].copy()
        b1_inspected["tail_all_zero"] = bool(np.all(x_raw[B1_K:] == 0))

        x_u16_g = x_raw[:B1_K]
        x_f32 = bf16_to_f32(x_u16_g)
        gu_out = np.zeros((6, B1_M_PER_EXPERT), dtype=np.uint16)
        for j in range(6):
            eid = scalars[f"e{j}"]
            g_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"])
            u_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"])
            g_out = f32_to_bf16_rtne(np.dot(x_f32, g_w.T))
            u_out = f32_to_bf16_rtne(np.dot(x_f32, u_w.T))
            gu_out[j, :MOE_INTERMEDIATE_SIZE] = g_out
            gu_out[j, MOE_INTERMEDIATE_SIZE:] = u_out
        y_gu[:] = gu_out.reshape(-1).view(bfloat16)

    def run_sh(y_sh: Any, w_sh: Any, x_gu: Any) -> None:
        x_raw = np.array(x_gu.numpy().view(np.uint16), copy=True)
        sh_inspected["len"] = len(x_raw)
        sh_inspected["head_vals"] = x_raw[:B1_K].copy()
        sh_inspected["tail_all_zero"] = bool(np.all(x_raw[B1_K:] == 0))

        x_u16_s = x_raw[:B1_K]
        x_f32 = bf16_to_f32(x_u16_s)
        g_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.gate_proj.weight"])
        u_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.up_proj.weight"])
        g_out = f32_to_bf16_rtne(np.dot(x_f32, g_w.T))
        u_out = f32_to_bf16_rtne(np.dot(x_f32, u_w.T))
        sh_out = np.zeros(B1_M_PER_EXPERT, dtype=np.uint16)
        sh_out[:MOE_INTERMEDIATE_SIZE] = g_out
        sh_out[MOE_INTERMEDIATE_SIZE:] = u_out
        y_sh[:] = sh_out.view(bfloat16)

    def run_dn(y_dn: Any, w_dn: Any, x_dn: Any, **scalars: int) -> None:
        x_raw = np.array(x_dn.numpy().view(np.uint16), copy=True)
        b2_inspected["len"] = len(x_raw)
        assert len(x_raw) == NUM_ACTIVE_TOTAL * MAX_K

        slots = x_raw.reshape(NUM_ACTIVE_TOTAL, MAX_K)
        b2_inspected["slots_layout_ok"] = True
        b2_inspected["tails_all_zero"] = True

        for j in range(NUM_ACTIVE_TOTAL):
            if not np.all(slots[j, B2_K_PAD:] == 0):
                b2_inspected["tails_all_zero"] = False
            if not np.all(slots[j, MOE_INTERMEDIATE_SIZE:B2_K_PAD] == 0):
                b2_inspected["tails_all_zero"] = False

        dn_out = np.zeros((NUM_ACTIVE_TOTAL, B2_M_PER_EXPERT), dtype=np.uint16)
        for j in range(6):
            eid = scalars[f"e{j}"]
            d_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"])
            act_u16 = slots[j, :MOE_INTERMEDIATE_SIZE]
            dn_out[j] = f32_to_bf16_rtne(np.dot(bf16_to_f32(act_u16), d_w.T))
        sh_dw = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.down_proj.weight"])
        sh_act_u16 = slots[6, :MOE_INTERMEDIATE_SIZE]
        dn_out[6] = f32_to_bf16_rtne(np.dot(bf16_to_f32(sh_act_u16), sh_dw.T))
        y_dn[:] = dn_out.reshape(-1).view(bfloat16)

    backend._get_or_compile_moe_ops = lambda: (run_gu, run_sh, run_dn)

    x = rng.standard_normal(dim).astype(np.float32)
    x_u16 = f32_to_bf16_rtne(x)

    ref_out = numpy_backend.moe_experts(layer, x_u16, top_ids, top_weights)
    npu_out = backend.moe_experts(layer, x_u16, top_ids, top_weights)

    # Check B1 & shared inspections
    assert b1_inspected["len"] == MAX_K
    assert np.array_equal(b1_inspected["head_vals"], x_u16[:B1_K])
    assert b1_inspected["tail_all_zero"] is True

    assert sh_inspected["len"] == MAX_K
    assert np.array_equal(sh_inspected["head_vals"], x_u16[:B1_K])
    assert sh_inspected["tail_all_zero"] is True

    # Check B2 inspections: 7 * MAX_K, slot layout, tail zeros
    assert b2_inspected["len"] == 7 * MAX_K
    assert b2_inspected["slots_layout_ok"] is True
    assert b2_inspected["tails_all_zero"] is True

    # Check numerical output parity
    assert np.array_equal(ref_out, npu_out)


def test_moe_nondefault_situ_betas_match_numpy_backend(monkeypatch: pytest.MonkeyPatch):
    """Non-default SiTU betas must reach both NPU-pipeline and mock paths bit-exactly.

    With gate_beta=3.0/value_beta=20.0 the NpuBackend output must stay bit-exact
    with NumpyBackend.moe_experts and must differ from the default 4.0/25.0 pair
    (otherwise the betas are not being forwarded to the SiTU-GLU activation).
    """
    import npu.backend as backend_mod
    from kernels.moe.pack import (
        B1_K,
        B1_M_PER_EXPERT,
        B2_K_PAD,
        B2_M_PER_EXPERT,
        MOE_INTERMEDIATE_SIZE,
        NUM_ACTIVE_TOTAL,
    )

    class FakeTensor:
        def __init__(self, size: int, dtype: Any = bfloat16):
            self._arr = np.zeros(size, dtype=dtype)

        def __setitem__(self, idx: Any, val: Any) -> None:
            self._arr[idx] = val

        def __getitem__(self, idx: Any) -> Any:
            return self._arr[idx]

        def numpy(self) -> np.ndarray:
            return self._arr  # View

        def to(self, device: str) -> FakeTensor:
            return self

    rng = np.random.default_rng(4242)
    layer = 1
    dim = 2560
    moe_dim = 1088
    top_ids = np.array([45, 12, 88, 3, 102, 55], dtype=np.int32)
    top_weights = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05], dtype=np.float32)

    weights_dict: dict[str, np.ndarray] = {}
    for eid in top_ids:
        g = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        u = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        d = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"] = f32_to_bf16_rtne(g)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"] = f32_to_bf16_rtne(u)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"] = f32_to_bf16_rtne(d)

    # Shared expert
    g_sh = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
    u_sh = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
    d_sh = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
    weights_dict[f"model.layers.{layer}.moe.shared_experts.gate_proj.weight"] = f32_to_bf16_rtne(g_sh)
    weights_dict[f"model.layers.{layer}.moe.shared_experts.up_proj.weight"] = f32_to_bf16_rtne(u_sh)
    weights_dict[f"model.layers.{layer}.moe.shared_experts.down_proj.weight"] = f32_to_bf16_rtne(d_sh)

    fake_weights = FakeWeights(weights_dict)
    numpy_backend = NumpyBackend(fake_weights)

    x = rng.standard_normal(dim).astype(np.float32)
    x_u16 = f32_to_bf16_rtne(x)

    gate_beta, value_beta = 3.0, 20.0

    # 1. Mock path (_moe_experts_numpy): non-default betas must reach SiTU-GLU bit-exactly
    mock_backend = NpuBackend(host_weights=fake_weights, mock=True)
    ref_out = numpy_backend.moe_experts(
        layer, x_u16, top_ids, top_weights, gate_beta=gate_beta, value_beta=value_beta
    )
    mock_out = mock_backend.moe_experts(
        layer, x_u16, top_ids, top_weights, gate_beta=gate_beta, value_beta=value_beta
    )
    assert np.array_equal(ref_out, mock_out)
    ref_default = numpy_backend.moe_experts(layer, x_u16, top_ids, top_weights)
    assert np.count_nonzero(ref_out != ref_default) > 0, (
        "non-default betas match default-beta output — betas are not being forwarded"
    )

    # 2. NPU pipeline path with fake operators: same betas, bit-exact vs NumpyBackend
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend = NpuBackend(host_weights=fake_weights, mock=True, gemv="legacy")
    backend.mock = False

    backend._work_buffers = {
        "moe_y_gu": FakeTensor(6 * B1_M_PER_EXPERT),
        "moe_y_sh": FakeTensor(B1_M_PER_EXPERT),
        "moe_y_dn": FakeTensor(NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT),
        "moe_x_gu": FakeTensor(B1_K),
        "moe_x_dn": FakeTensor(NUM_ACTIVE_TOTAL * B2_K_PAD),
    }

    def run_gu(y_gu: Any, w_gu: Any, x_gu: Any, **scalars: int) -> None:
        x_u16_g = np.array(x_gu.numpy().view(np.uint16), copy=True)[:B1_K]
        x_f32 = bf16_to_f32(x_u16_g)
        gu_out = np.zeros((6, B1_M_PER_EXPERT), dtype=np.uint16)
        for j in range(6):
            eid = scalars[f"e{j}"]
            g_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"])
            u_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"])
            g_out = f32_to_bf16_rtne(np.dot(x_f32, g_w.T))
            u_out = f32_to_bf16_rtne(np.dot(x_f32, u_w.T))
            gu_out[j, :MOE_INTERMEDIATE_SIZE] = g_out
            gu_out[j, MOE_INTERMEDIATE_SIZE:] = u_out
        y_gu[:] = gu_out.reshape(-1).view(bfloat16)

    def run_sh(y_sh: Any, w_sh: Any, x_gu: Any) -> None:
        x_u16_s = np.array(x_gu.numpy().view(np.uint16), copy=True)[:B1_K]
        x_f32 = bf16_to_f32(x_u16_s)
        g_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.gate_proj.weight"])
        u_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.up_proj.weight"])
        g_out = f32_to_bf16_rtne(np.dot(x_f32, g_w.T))
        u_out = f32_to_bf16_rtne(np.dot(x_f32, u_w.T))
        sh_out = np.zeros(B1_M_PER_EXPERT, dtype=np.uint16)
        sh_out[:MOE_INTERMEDIATE_SIZE] = g_out
        sh_out[MOE_INTERMEDIATE_SIZE:] = u_out
        y_sh[:] = sh_out.view(bfloat16)

    def run_dn(y_dn: Any, w_dn: Any, x_dn: Any, **scalars: int) -> None:
        x_arr = np.array(x_dn.numpy().view(np.uint16), copy=True).reshape(NUM_ACTIVE_TOTAL, B2_K_PAD)
        dn_out = np.zeros((NUM_ACTIVE_TOTAL, B2_M_PER_EXPERT), dtype=np.uint16)
        for j in range(6):
            eid = scalars[f"e{j}"]
            d_w = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"])
            act_u16 = x_arr[j, :MOE_INTERMEDIATE_SIZE]
            dn_out[j] = f32_to_bf16_rtne(np.dot(bf16_to_f32(act_u16), d_w.T))
        sh_dw = bf16_to_f32(weights_dict[f"model.layers.{layer}.moe.shared_experts.down_proj.weight"])
        sh_act_u16 = x_arr[6, :MOE_INTERMEDIATE_SIZE]
        dn_out[6] = f32_to_bf16_rtne(np.dot(bf16_to_f32(sh_act_u16), sh_dw.T))
        y_dn[:] = dn_out.reshape(-1).view(bfloat16)

    backend._get_or_compile_moe_ops = lambda: (run_gu, run_sh, run_dn)
    backend._get_weight_tensor = lambda name: "fake_weight_tensor"

    npu_out = backend.moe_experts(
        layer, x_u16, top_ids, top_weights, gate_beta=gate_beta, value_beta=value_beta
    )
    diff = int(np.count_nonzero(ref_out != npu_out))
    assert diff == 0, f"MoE non-default beta output mismatch vs NumpyBackend: {diff} elements differed"
    npu_default = backend.moe_experts(layer, x_u16, top_ids, top_weights)
    assert np.count_nonzero(npu_out != npu_default) > 0, (
        "non-default betas match default-beta output on the NPU path — betas are not being forwarded"
    )


def test_moe_shared_expert_cache_and_validation():
    """Verify shared expert cache reuse after moe_experts and strict input validation on down."""
    backend = NpuBackend(mock=True)
    rng = np.random.default_rng(333)

    # Set mock weight arrays for moe_b1, shared_gu and moe_b2
    backend.mock_weight_arrays["L01/moe_b1"] = f32_to_bf16_rtne(
        rng.standard_normal((128, 2176, 2560)).astype(np.float32) * 0.02
    )
    backend.mock_weight_arrays["L01/shared_gu"] = f32_to_bf16_rtne(
        rng.standard_normal((2176, 2560)).astype(np.float32) * 0.02
    )
    backend.mock_weight_arrays["L01/moe_b2"] = f32_to_bf16_rtne(
        rng.standard_normal((129, 2560, 1280)).astype(np.float32) * 0.02
    )

    x_u16 = f32_to_bf16_rtne(rng.standard_normal(2560).astype(np.float32))
    top_ids = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    top_weights = np.ones(6, dtype=np.float32) / 6.0

    backend.cache_hits = 0
    backend.cache_misses = 0

    # 1. Run MoE -> populates _bundle_cache["L01/shared_gu"] and _cached_shared[1]
    _ = backend.moe_experts(1, x_u16, top_ids, top_weights)
    hits_after_moe = backend.cache_hits
    misses_after_moe = backend.cache_misses

    # 2. Subsequent linear for shared gate and up with the same input object must HIT cache
    _ = backend.linear("model.layers.1.moe.shared_experts.gate_proj.weight", x_u16)
    _ = backend.linear("model.layers.1.moe.shared_experts.up_proj.weight", x_u16)

    assert backend.cache_misses == misses_after_moe, "Shared gate/up linear must not re-trigger GEMV"
    assert backend.cache_hits == hits_after_moe + 2, "Shared gate/up linear should hit bundle cache twice"

    # 3. Test _linear_shared_down validation
    comp_down = backend.comp_map["model.layers.1.moe.shared_experts.down_proj.weight"]
    sh_act = backend._cached_shared[1]["sh_act_u16"]
    assert sh_act is not None

    # (a) Matching input returns cached down result
    valid_down = backend._linear_shared_down(comp_down, sh_act, is_f32=False)
    assert np.array_equal(valid_down, backend._cached_shared[1]["sh_down"])

    # (b) Different input bit raises ValueError
    bad_act = sh_act.copy()
    bad_act[0] ^= 1  # flip 1 bit
    with pytest.raises(ValueError, match="do not match B2 slot-6 input"):
        backend._linear_shared_down(comp_down, bad_act, is_f32=False)

    # (c) Step mismatch raises ValueError
    backend.set_step(backend._step_id + 1)
    with pytest.raises(ValueError, match="step"):
        backend._linear_shared_down(comp_down, sh_act, is_f32=False)

    # (d) Missing cache raises ValueError
    backend._cached_shared.clear()
    with pytest.raises(ValueError, match="without a valid B2 slot-6 cache entry"):
        backend._linear_shared_down(comp_down, sh_act, is_f32=False)


def test_bf16_view_and_work_buffer_safety(monkeypatch: pytest.MonkeyPatch):
    """Verify bitwise preservation into bfloat16 buffers and output copy safety on buffer reuse."""
    import npu.backend as backend_mod

    class FakeAieTensor:
        def __init__(self, shape: tuple[int, ...], dtype: Any = bfloat16):
            self.data = np.zeros(shape, dtype=dtype)

        def __setitem__(self, idx: Any, val: Any) -> None:
            self.data[idx] = val

        def numpy(self) -> np.ndarray:
            return self.data  # VIEW, NOT A COPY

        def to(self, device: str) -> FakeAieTensor:
            return self

    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    # This is an operator-buffer unit test with injected weights, not a real
    # execution startup; bypass the production manifest gate explicitly.
    backend = NpuBackend(mock=True, gemv="legacy")
    backend.mock = False

    bmeta = backend.bundle_map["L00/f_b"]  # k=256, m=5120

    # (a) Verify input bits are preserved directly without numeric float cast
    captured: dict[str, np.ndarray] = {}

    def fake_gemv_inspect(y: Any, w: Any, x: Any) -> None:
        raw_bits = np.array(x.numpy().view(np.uint16), copy=True)
        captured["raw_bits"] = raw_bits
        captured["float_vals"] = bf16_to_f32(raw_bits)
        y.data[:] = np.full(bmeta.m, 0x3F80, dtype=np.uint16).view(bfloat16)

    backend._get_or_compile_gemv = lambda k, m, fmt: fake_gemv_inspect
    backend._get_weight_tensor = lambda name: "fake_w"
    backend._work_buffers[f"gemv_y_{bmeta.m}"] = FakeAieTensor((bmeta.m,))
    backend._work_buffers[f"gemv_x_{bmeta.k}"] = FakeAieTensor((bmeta.k,))

    x_in = np.full(bmeta.k, 0x3F80, dtype=np.uint16)
    _ = backend._execute_gemv("L00/f_b", bmeta, x_in)

    assert captured["raw_bits"][0] == 0x3F80
    assert captured["float_vals"][0] == 1.0

    # (b) Verify that reusing work buffer does NOT mutate previously returned output array
    call_count = 0

    def fake_gemv_mutate(y: Any, w: Any, x: Any) -> None:
        nonlocal call_count
        call_count += 1
        val = 0x1111 if call_count == 1 else 0x9999
        y.data[:] = np.full(bmeta.m, val, dtype=np.uint16).view(bfloat16)

    backend._get_or_compile_gemv = lambda k, m, fmt: fake_gemv_mutate

    res1 = backend._execute_gemv("L00/f_b", bmeta, x_in)
    assert np.all(res1 == 0x1111)

    res2 = backend._execute_gemv("L00/f_b", bmeta, x_in)
    assert np.all(res2 == 0x9999)
    # res1 must remain unchanged because _execute_gemv copies the numpy view
    assert np.all(res1 == 0x1111)


class FakeRuntimeTensor:
    """Fake NPU tensor modeling XRTHostRuntime and tensor_class coherence contracts."""

    def __init__(self, shape, dtype=bfloat16, initial_device="npu"):
        self.shape = shape
        self.dtype = dtype
        self._host_data = np.zeros(shape, dtype=dtype)
        self._device_data = np.zeros(shape, dtype=dtype)
        # Coherence states:
        # "HOST"   : Host has written dirty bytes; to("npu") will sync host->device.
        # "DEVICE" : Kernel wrote to device; numpy() will sync device->host.
        # "CLEAN"  : Host and device buffers match; to("npu") and numpy() are no-ops.
        self._coherence = "CLEAN" if initial_device == "npu" else "HOST"
        self.h2d_syncs = 0
        self.d2h_syncs = 0
        self.overwrites = 0
        self.setitems = 0
        self.to_calls = 0

    class _Overwrite:
        def __init__(self, tensor):
            self.tensor = tensor

        def __enter__(self):
            self.tensor.overwrites += 1
            return self.tensor._host_data

        def __exit__(self, *exc_info):
            self.tensor._coherence = "HOST"
            return False

    def overwrite(self):
        return self._Overwrite(self)

    def __setitem__(self, key, value):
        self.setitems += 1
        self._host_data[key] = value
        self._coherence = "HOST"

    def to(self, device):
        self.to_calls += 1
        if device == "npu":
            if self._coherence == "HOST":
                self._device_data[...] = self._host_data
                self.h2d_syncs += 1
                self._coherence = "CLEAN"
        elif device == "cpu":
            if self._coherence == "DEVICE":
                self._host_data[...] = self._device_data
                self.d2h_syncs += 1
                self._coherence = "CLEAN"
        return self

    def numpy(self):
        if self._coherence == "DEVICE":
            self._host_data[...] = self._device_data
            self.d2h_syncs += 1
            self._coherence = "CLEAN"
        return self._host_data

    @property
    def data(self):
        return self._host_data


def make_fake_runtime(kernel_body, output_tensors_fn, enable_runtime_transfers=True):
    """Wrap kernel execution with XRTHostRuntime-like argument synchronization."""
    def run(*args, **kwargs):
        # 1. XRTHostRuntime: [a.to("npu") for a in args] (hostruntime.py:302)
        if enable_runtime_transfers:
            for a in args:
                if hasattr(a, "to"):
                    a.to("npu")
        # 2. Kernel execution: uses strictly device data
        kernel_body(*args, **kwargs)
        # 3. Kernel marks output BOs as device-owned (tensor_class.py:692-705)
        outs = output_tensors_fn(*args, **kwargs)
        for out in outs:
            if hasattr(out, "_coherence"):
                out._coherence = "DEVICE"
    return run


def test_gemv_reuses_prepared_buffers_without_duplicate_sync(monkeypatch):
    """Repeated GEMV calls allocate no tensors and transfers are owned by runtime."""
    allocations = {"tensor": 0, "full": 0}

    def unexpected_tensor(*args, **kwargs):
        allocations["tensor"] += 1
        raise AssertionError("tensor allocation occurred in decode call")

    def unexpected_full(*args, **kwargs):
        allocations["full"] += 1
        raise AssertionError("output allocation occurred in decode call")

    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        SimpleNamespace(tensor=unexpected_tensor, full=unexpected_full),
    )
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend = NpuBackend(mock=True, gemv="legacy")
    backend.mock = False
    bmeta = backend.bundle_map["L00/f_b"]
    x_tensor = FakeRuntimeTensor((bmeta.k,))
    y_tensor = FakeRuntimeTensor((bmeta.m,))
    backend._work_buffers[f"gemv_x_{bmeta.k}"] = x_tensor
    backend._work_buffers[f"gemv_y_{bmeta.m}"] = y_tensor
    backend._get_weight_tensor = lambda name: "weight"

    def kernel_body(y, weight, x):
        val = x._device_data[0]
        y._device_data[:] = val

    run = make_fake_runtime(kernel_body, lambda y, weight, x: [y], enable_runtime_transfers=True)
    backend._get_or_compile_gemv = lambda k, m, fmt: run

    for i in range(4):
        y_tensor._host_data[:] = 0xDEAD
        x = np.full(bmeta.k, 0x3F80 + i, dtype=np.uint16)
        out = backend._execute_gemv("L00/f_b", bmeta, x)
        assert x_tensor.h2d_syncs == i + 1
        assert x_tensor.d2h_syncs == 0
        assert y_tensor.d2h_syncs == i + 1
        assert y_tensor.h2d_syncs == 0
        assert np.all(out == 0x3F80 + i)

    assert allocations == {"tensor": 0, "full": 0}
    assert x_tensor.overwrites == 4
    assert x_tensor.setitems == 0
    assert x_tensor.h2d_syncs == 4
    assert x_tensor.d2h_syncs == 0
    assert y_tensor.d2h_syncs == 4
    assert y_tensor.h2d_syncs == 0


@pytest.mark.parametrize("gemv_mode", ["legacy", "unified"])
def test_dense_gemv_runtime_owned_transfers_and_zero_alloc(monkeypatch, gemv_mode):
    """Dense GEMV: input transferred once H2D per call, output once D2H on read, zero new allocs."""
    allocations = {"tensor": 0, "full": 0}

    def unexpected_tensor(*args, **kwargs):
        allocations["tensor"] += 1
        raise AssertionError("tensor allocation occurred in decode call")

    def unexpected_full(*args, **kwargs):
        allocations["full"] += 1
        raise AssertionError("output allocation occurred in decode call")

    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        SimpleNamespace(tensor=unexpected_tensor, full=unexpected_full),
    )
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)

    backend = NpuBackend(mock=True, gemv=gemv_mode)
    backend.mock = False
    bmeta = backend.bundle_map["L00/f_b"]
    k, m = bmeta.k, bmeta.m
    x_buf_len = 12032 if gemv_mode == "unified" else k

    x_tensor = FakeRuntimeTensor((x_buf_len,), dtype=bfloat16)
    y_tensor = FakeRuntimeTensor((m,), dtype=bfloat16)
    w_tensor = FakeRuntimeTensor((1,), dtype=bfloat16)

    if gemv_mode == "unified":
        backend._work_buffers["gemv_x_unified"] = x_tensor
    else:
        backend._work_buffers[f"gemv_x_{k}"] = x_tensor
    backend._work_buffers[f"gemv_y_{m}"] = y_tensor
    backend._get_weight_tensor = lambda name: w_tensor

    def kernel_body(y, w, x):
        val = x._device_data[0]
        y._device_data[:] = val

    run_gemv = make_fake_runtime(kernel_body, lambda y, w, x: [y], enable_runtime_transfers=True)
    backend._get_or_compile_gemv = lambda k, m, fmt: run_gemv

    for i in range(4):
        # Poison host output buffer with stale bytes to ensure it is not used as kernel input
        y_tensor._host_data[:] = 0xDEAD
        test_val = 0x3F80 + i
        x_val = np.full(k, test_val, dtype=np.uint16)

        out = backend._execute_gemv("L00/f_b", bmeta, x_val)

        # Input: exactly one host->device transfer per call
        assert x_tensor.h2d_syncs == i + 1
        assert x_tensor.d2h_syncs == 0
        # Output: exactly one device->host transfer on read, zero host->device
        assert y_tensor.d2h_syncs == i + 1
        assert y_tensor.h2d_syncs == 0
        # Output verification: must match computed test_val, not stale 0xDEAD
        assert np.all(out == test_val)

    assert allocations == {"tensor": 0, "full": 0}


@pytest.mark.parametrize("gemv_mode", ["legacy", "unified"])
def test_moe_runtime_owned_transfers_and_zero_alloc(monkeypatch, gemv_mode):
    """MoE (B1/shared/B2): runtime-owned transfers, stale host bytes not read, zero new allocs."""
    allocations = {"tensor": 0, "full": 0}

    def unexpected_tensor(*args, **kwargs):
        allocations["tensor"] += 1
        raise AssertionError("tensor allocation occurred in decode call")

    def unexpected_full(*args, **kwargs):
        allocations["full"] += 1
        raise AssertionError("output allocation occurred in decode call")

    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        SimpleNamespace(tensor=unexpected_tensor, full=unexpected_full),
    )
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)

    backend = NpuBackend(mock=True, gemv=gemv_mode)
    backend.mock = False

    from kernels.moe.pack import (
        B1_K,
        B1_M_PER_EXPERT,
        B2_K_PAD,
        B2_M_PER_EXPERT,
        NUM_ACTIVE_TOTAL,
    )

    x_gu_size = 12032 if gemv_mode == "unified" else B1_K
    x_dn_size = 84224 if gemv_mode == "unified" else NUM_ACTIVE_TOTAL * B2_K_PAD

    x_gu = FakeRuntimeTensor((x_gu_size,), dtype=bfloat16)
    x_dn = FakeRuntimeTensor((x_dn_size,), dtype=bfloat16)
    y_gu = FakeRuntimeTensor((6 * B1_M_PER_EXPERT,), dtype=bfloat16)
    y_sh = FakeRuntimeTensor((B1_M_PER_EXPERT,), dtype=bfloat16)
    y_dn = FakeRuntimeTensor((NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,), dtype=bfloat16)
    w_dummy = FakeRuntimeTensor((1,), dtype=bfloat16)

    backend._work_buffers["moe_x_gu"] = x_gu
    backend._work_buffers["moe_x_dn"] = x_dn
    backend._work_buffers["moe_y_gu"] = y_gu
    backend._work_buffers["moe_y_sh"] = y_sh
    backend._work_buffers["moe_y_dn"] = y_dn
    backend._get_weight_tensor = lambda name: w_dummy

    def run_gu_body(y, w, x, **scalars):
        y._device_data[:] = x._device_data[0]

    def run_sh_body(y, w, x):
        y._device_data[:] = x._device_data[0]

    def run_dn_body(y, w, x, **scalars):
        y._device_data[:] = x._device_data[0]

    run_gu = make_fake_runtime(run_gu_body, lambda y, w, x, **kw: [y], enable_runtime_transfers=True)
    run_sh = make_fake_runtime(run_sh_body, lambda y, w, x: [y], enable_runtime_transfers=True)
    run_dn = make_fake_runtime(run_dn_body, lambda y, w, x, **kw: [y], enable_runtime_transfers=True)

    backend._get_or_compile_moe_ops = lambda: (run_gu, run_sh, run_dn)

    for i in range(4):
        # Poison host output buffers with stale bytes
        y_gu._host_data[:] = 0xDEAD
        y_sh._host_data[:] = 0xDEAD
        y_dn._host_data[:] = 0xDEAD

        x_val = np.full(B1_K, 0x3F80 + i, dtype=np.uint16)
        top_ids = np.array([0, 1, 2, 3, 4, 5], dtype=np.int32)
        top_weights = np.ones(6, dtype=np.float32) / 6.0

        out = backend.moe_experts(0, x_val, top_ids, top_weights)

        assert out.shape == (B2_M_PER_EXPERT,)

        # Transfer counts:
        # x_gu: transferred once per call (during run_gu; run_sh sees it clean, so no extra transfer)
        assert x_gu.h2d_syncs == i + 1
        assert x_gu.d2h_syncs == 0

        # y_gu: device->host once when read by backend, zero host->device
        assert y_gu.d2h_syncs == i + 1
        assert y_gu.h2d_syncs == 0

        # y_sh: device->host once when read by backend, zero host->device
        assert y_sh.d2h_syncs == i + 1
        assert y_sh.h2d_syncs == 0

        # x_dn: host->device once per call
        assert x_dn.h2d_syncs == i + 1
        assert x_dn.d2h_syncs == 0

        # y_dn: device->host once when read by backend, zero host->device
        assert y_dn.d2h_syncs == i + 1
        assert y_dn.h2d_syncs == 0

    assert allocations == {"tensor": 0, "full": 0}


@pytest.mark.parametrize("gemv_mode", ["legacy", "unified"])
def test_rda_runtime_owned_transfers_and_zero_alloc(monkeypatch, gemv_mode):
    """RDA: runtime-owned transfers, recurrent state on NPU, zero new allocs."""
    allocations = {"tensor": 0, "full": 0}

    def unexpected_tensor(*args, **kwargs):
        allocations["tensor"] += 1
        raise AssertionError("tensor allocation occurred in decode call")

    def unexpected_full(*args, **kwargs):
        allocations["full"] += 1
        raise AssertionError("output allocation occurred in decode call")

    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        SimpleNamespace(tensor=unexpected_tensor, full=unexpected_full),
    )
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)

    fake_module = types.ModuleType("bench.rda_state.rda_state")
    def fake_pack_head_bundle(q, k, a, v, beta):
        h_dim = q.shape[0]
        bundle = np.zeros((h_dim, 640), dtype=np.float32)
        bundle[:, 0:128] = q
        bundle[:, 128:256] = k
        bundle[:, 256:384] = a
        bundle[:, 384:512] = v
        bundle[:, 512] = beta
        return bundle

    fake_module.pack_head_bundle = fake_pack_head_bundle
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", fake_module)

    backend = NpuBackend(mock=True, gemv=gemv_mode)
    backend.mock = False

    head_in = FakeRuntimeTensor((40, 640), dtype=np.float32)
    out_npu = FakeRuntimeTensor((40, 128), dtype=np.float32)
    s0 = FakeRuntimeTensor((40, 128, 128), dtype=np.float32)
    s1 = FakeRuntimeTensor((40, 128, 128), dtype=np.float32)

    backend._work_buffers["rda_head_in"] = head_in
    backend._work_buffers["rda_out"] = out_npu
    backend.rda_states[0] = [s0, s1]
    backend.rda_active_slot[0] = 0

    def run_rda_body(s_in, s_out, h_in, h_out):
        h_out._device_data[:] = h_in._device_data[:, :128]

    run_rda = make_fake_runtime(run_rda_body, lambda s_in, s_out, h_in, h_out: [s_out, h_out], enable_runtime_transfers=True)
    backend._compiled_ops["rda_step"] = run_rda

    for i in range(4):
        # Poison host output buffer with stale value
        out_npu._host_data[:] = 999.0

        q = np.full((40, 128), 0x3F80 + i, dtype=np.uint16)
        k = np.full((40, 128), 0x3F80, dtype=np.uint16)
        v = np.full((40, 128), 0x3F80, dtype=np.uint16)
        log_alpha = np.zeros((40, 128), dtype=np.float32)
        beta = np.full(40, 0x3F80, dtype=np.uint16)

        out = backend.rda_update(0, q, k, v, log_alpha, beta)

        assert out.shape == (40, 128)
        assert np.all(out != 999.0)

        # Transfer counts:
        # head_in: host->device once per call
        assert head_in.h2d_syncs == i + 1
        assert head_in.d2h_syncs == 0

        # out_npu: device->host once when read by backend, zero host->device
        assert out_npu.d2h_syncs == i + 1
        assert out_npu.h2d_syncs == 0

        # Persistent recurrent state: stays on NPU, zero host transfers
        assert s0.h2d_syncs == 0
        assert s0.d2h_syncs == 0
        assert s1.h2d_syncs == 0
        assert s1.d2h_syncs == 0

    assert allocations == {"tensor": 0, "full": 0}


def test_regression_omitted_runtime_transfers_fails(monkeypatch):
    """When runtime omits to('npu'), host-to-device transfer does not occur and test fails."""
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    backend = NpuBackend(mock=True, gemv="legacy")
    backend.mock = False
    bmeta = backend.bundle_map["L00/f_b"]
    k, m = bmeta.k, bmeta.m
    x_tensor = FakeRuntimeTensor((k,))
    y_tensor = FakeRuntimeTensor((m,))
    backend._work_buffers[f"gemv_x_{k}"] = x_tensor
    backend._work_buffers[f"gemv_y_{m}"] = y_tensor
    backend._get_weight_tensor = lambda name: "weight"

    def buggy_kernel(y, w, x):
        # Missing: [a.to("npu") for a in (y, w, x)]
        y._device_data[:] = x._device_data[0]
        y._coherence = "DEVICE"

    # Runtime with transfers disabled
    backend._get_or_compile_gemv = lambda k, m, fmt: buggy_kernel

    x_val = np.full(k, 0x3F80, dtype=np.uint16)
    backend._execute_gemv("L00/f_b", bmeta, x_val)

    # Without runtime to('npu'), x_tensor was never transferred to device
    with pytest.raises(AssertionError, match="runtime omitted host-to-device transfer"):
        assert x_tensor.h2d_syncs == 1, "runtime omitted host-to-device transfer"


def test_npu_decode_paths_use_full_overwrite_and_runtime_owned_transfers():
    """Linear, RDA, and MoE paths use overwrite and contain no explicit to()."""
    tree = ast.parse(Path(backend_mod.__file__).read_text(encoding="utf-8"))
    backend_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NpuBackend"
    )
    methods = {
        node.name: node
        for node in backend_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_execute_gemv", "rda_update", "moe_experts"}
    }
    minimum_overwrites = {"_execute_gemv": 2, "rda_update": 1, "moe_experts": 4}
    for name, minimum in minimum_overwrites.items():
        calls = [node for node in ast.walk(methods[name]) if isinstance(node, ast.Call)]
        overwrite_calls = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute)
            and call.func.attr == "_overwrite_buffer"
        ]
        explicit_transfers = [
            call
            for call in calls
            if isinstance(call.func, ast.Attribute) and call.func.attr == "to"
        ]
        assert len(overwrite_calls) >= minimum
        assert not explicit_transfers


def test_dweights_decode_and_parity(tmp_path: Path):
    """Verify DWeights decodes components, routed experts, kv_b, and shared down matching parse functions."""
    rng = np.random.default_rng(404)

    # 1. Component slice: L00/f_b [5120, 256] bfp16_gemv
    w_fb = rng.standard_normal((5120, 256)).astype(np.float32) * 0.05
    bin_fb = pack_bfp16_gemv(w_fb, rows=4)
    (tmp_path / "L00").mkdir(parents=True, exist_ok=True)
    (tmp_path / "L00" / "f_b.bin").write_bytes(bin_fb)
    parse_fb = parse_bfp16_gemv(bin_fb, K=256, M=5120, rows=4)

    # 2. kv_b_proj HF layout: L03/k_b and L03/v_b [5120, 512] bfp16_gemv
    w_kb = rng.standard_normal((5120, 512)).astype(np.float32) * 0.05
    w_vb = rng.standard_normal((5120, 512)).astype(np.float32) * 0.05
    bin_kb = pack_bfp16_gemv(w_kb, rows=4)
    bin_vb = pack_bfp16_gemv(w_vb, rows=4)
    (tmp_path / "L03").mkdir(parents=True, exist_ok=True)
    (tmp_path / "L03" / "k_b.bin").write_bytes(bin_kb)
    (tmp_path / "L03" / "v_b.bin").write_bytes(bin_vb)
    parse_kb = parse_bfp16_gemv(bin_kb, K=512, M=5120, rows=4)
    parse_vb = parse_bfp16_gemv(bin_vb, K=512, M=5120, rows=4)

    # 3. 3D MoE bundle: L01/moe_b1 expert 0 [2176, 2560] q4nx_gemv_bundle
    w_gu = rng.standard_normal((2176, 2560)).astype(np.float32) * 0.02
    codes, scales, mins = quantize(w_gu)
    bin_gu = pack_gemv(codes, scales, mins, rows=4)
    assert len(bin_gu) == 3481600
    (tmp_path / "L01").mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "L01" / "moe_b1.bin", "wb") as f:
        f.write(bin_gu)
    pc, ps, pm = parse_gemv(bin_gu, K=2560, M=2176, rows=4)
    parse_gu = dequantize_simple(pc, ps, pm)

    # 4. 3D MoE bundle: L01/moe_b2 expert 128 (shared down) [2560, 1280] bfp16_gemv_bundle
    w_dn = rng.standard_normal((2560, 1280)).astype(np.float32) * 0.02
    bin_dn = pack_bfp16_gemv(w_dn, rows=4)
    assert len(bin_dn) == 3686400
    with open(tmp_path / "L01" / "moe_b2.bin", "wb") as f:
        f.seek(128 * 3686400)
        f.write(bin_dn)
    parse_dn = parse_bfp16_gemv(bin_dn, K=1280, M=2560, rows=4)

    dw = DWeights(weights_dir=tmp_path)

    # Check 1: Component row range slice
    got_fb = dw.get("model.layers.0.self_attn.f_b_proj.weight")
    assert np.array_equal(got_fb, parse_fb[:5120, :128])

    # Check 2: Single routed expert (3D bundle)
    got_g = dw.get("model.layers.1.moe.experts.0.gate_proj.weight")
    got_u = dw.get("model.layers.1.moe.experts.0.up_proj.weight")
    assert np.array_equal(got_g, parse_gu[:1088, :])
    assert np.array_equal(got_u, parse_gu[1088:, :])

    # Check 3: kv_b_proj interleaved HF layout
    got_kv = dw.get("model.layers.3.self_attn.kv_b_proj.weight")
    k_h = parse_kb.reshape(40, 128, 512)
    v_h = parse_vb.reshape(40, 128, 512)
    exp_kv = np.ascontiguousarray(
        np.concatenate([k_h, v_h], axis=1).reshape(10240, 512), dtype=np.float32
    )
    assert np.array_equal(got_kv, exp_kv)

    # Check 4: Shared down (expert 128, K 1088 only)
    got_sh_dn = dw.get("model.layers.1.moe.shared_experts.down_proj.weight")
    assert np.array_equal(got_sh_dn, parse_dn[:2560, :1088])


def test_2d_linear_and_ensure_mla_cache_expanded():
    """Verify 2D linear row-by-row consistency and ensure_mla_cache_expanded idempotency."""
    from host.raon import RaonConfig

    rng = np.random.default_rng(505)

    # 1. 2D linear [T, K] matches row-by-row 1D executions
    backend = NpuBackend(mock=True)
    w_dense = rng.standard_normal((24064, 2560)).astype(np.float32) * 0.05
    backend.mock_weight_arrays["L00/dense_gu"] = f32_to_bf16_rtne(w_dense)

    X = rng.standard_normal((4, 2560)).astype(np.float32)
    X_u16 = f32_to_bf16_rtne(X)

    out_2d = backend.linear("model.layers.0.mlp.gate_proj.weight", X_u16)
    out_rows = [
        backend.linear("model.layers.0.mlp.gate_proj.weight", X_u16[i]) for i in range(4)
    ]
    expected_stacked = np.stack(out_rows, axis=0)

    assert np.array_equal(out_2d, expected_stacked)
    assert out_2d.shape == (4, 12032)

    # 2. ensure_mla_cache_expanded populates empty cache and skips populated cache
    cfg = RaonConfig(
        hidden_size=2560,
        num_hidden_layers=4,
        rda_layers=[0, 1, 2],
        mla_layers=[3],
        vocab_size=128,
    )
    backend.mock_weight_arrays["L03/k_b"] = f32_to_bf16_rtne(
        rng.standard_normal((5120, 512)).astype(np.float32) * 0.05
    )
    backend.mock_weight_arrays["L03/v_b"] = f32_to_bf16_rtne(
        rng.standard_normal((5120, 512)).astype(np.float32) * 0.05
    )

    class DummyModel:
        def __init__(self, c: RaonConfig, b: NpuBackend):
            self.config = c
            self.backend = b

    model = DummyModel(cfg, backend)
    T = 3
    snap = {
        "layer_003.mla_kv_lat_cache": rng.standard_normal((T, 512)).astype(np.float32),
        "layer_003.mla_k_pe_cache": rng.standard_normal((T, 64)).astype(np.float32),
    }

    # Cache starts empty
    k0, v0 = backend.get_mla_cache(3)
    assert k0 is None and v0 is None

    # First expansion -> expands 1 layer
    expanded1 = ensure_mla_cache_expanded(model, snap)
    assert expanded1 == 1

    k1, v1 = backend.get_mla_cache(3)
    assert k1 is not None and v1 is not None
    assert k1.shape == (1, 40, T, 192)
    assert v1.shape == (1, 40, T, 128)

    # Second expansion -> already populated, expands 0 layers
    expanded2 = ensure_mla_cache_expanded(model, snap)
    assert expanded2 == 0


# =============================================================================
# Unified GEMV Integration Tests (Engine Selection, Shapes, Layout, Parity, Switches)
# =============================================================================

def test_engine_selection(monkeypatch):
    """Test engine selection via constructor argument and RAON_NPU_GEMV environment variable."""
    # 1. Default without args or env -> unified
    monkeypatch.delenv("RAON_NPU_GEMV", raising=False)
    backend_default = NpuBackend(mock=True)
    assert backend_default.gemv == "unified"

    # 2. Constructor argument explicit
    backend_legacy = NpuBackend(mock=True, gemv="legacy")
    assert backend_legacy.gemv == "legacy"

    backend_unified = NpuBackend(mock=True, gemv="unified")
    assert backend_unified.gemv == "unified"

    # 3. Environment variable fallback
    monkeypatch.setenv("RAON_NPU_GEMV", "legacy")
    backend_env_legacy = NpuBackend(mock=True)
    assert backend_env_legacy.gemv == "legacy"

    monkeypatch.setenv("RAON_NPU_GEMV", "unified")
    backend_env_unified = NpuBackend(mock=True)
    assert backend_env_unified.gemv == "unified"

    # 4. Constructor argument overrides environment variable
    monkeypatch.setenv("RAON_NPU_GEMV", "legacy")
    backend_override = NpuBackend(mock=True, gemv="unified")
    assert backend_override.gemv == "unified"

    monkeypatch.setenv("RAON_NPU_GEMV", "unified")
    backend_override_l = NpuBackend(mock=True, gemv="legacy")
    assert backend_override_l.gemv == "legacy"

    # 5. Invalid values raise ValueError
    with pytest.raises(ValueError, match="Invalid gemv engine"):
        NpuBackend(mock=True, gemv="turbo")

    monkeypatch.setenv("RAON_NPU_GEMV", "invalid_val")
    with pytest.raises(ValueError, match="Invalid gemv engine"):
        NpuBackend(mock=True)


def test_manifest_171_bundles_mapping():
    """Verify that all 171 NPU bundles in the real manifest map to valid ALL_SHAPES."""
    weights_d_env = os.environ.get("RAON_WEIGHTS_D")
    if not weights_d_env:
        pytest.skip("RAON_WEIGHTS_D environment variable not set; skipping real manifest check")
    weights_d = Path(weights_d_env)
    manifest_file = weights_d / "manifest.json"
    if not manifest_file.is_file():
        pytest.skip(f"Manifest file not found at {manifest_file}; skipping real manifest check")

    backend = NpuBackend(weights_dir=weights_d, mock=True, gemv="unified")
    assert len(backend.bundle_map) == 171, f"Expected 171 NPU bundles, got {len(backend.bundle_map)}"

    for bname, bmeta in backend.bundle_map.items():
        shape_name = resolve_unified_shape_name(bmeta.k, bmeta.m, bmeta.fmt, bundle_name=bname)
        shape = get_shape(shape_name)
        assert shape is not None, f"Bundle {bname} mapped to unknown shape {shape_name}"


def test_synthetic_manifest_mapping():
    """Verify that all default 171 NPU bundles map to the 13 supported unified shapes."""
    comp_map, bundle_map = build_default_manifest_specs()
    assert len(bundle_map) == 171

    expected_shapes = {
        "rda_in",
        "f_b",
        "o_proj",
        "mla_in",
        "q_b",
        "k_b_v_b",
        "g",
        "shared_gu",
        "dense_down",
        "lm_head",
        "dense_gu",
        "moe_b1_6",
        "moe_b2_9",
    }
    encountered_shapes: set[str] = set()

    for bname, bmeta in bundle_map.items():
        sname = resolve_unified_shape_name(bmeta.k, bmeta.m, bmeta.fmt, bundle_name=bname)
        shape = get_shape(sname)
        assert shape is not None
        # Canonicalize alias
        canon_name = "k_b_v_b" if sname in ("k_b", "v_b") else sname
        encountered_shapes.add(canon_name)

    assert encountered_shapes == expected_shapes, (
        f"Missing shape coverage: {expected_shapes - encountered_shapes}"
    )


def test_unmapped_bundle_raises_value_error():
    """Verify that unmapped bundles cause ValueError at startup or validation."""
    # 1. Invalid (K, M, fmt) tuple
    with pytest.raises(ValueError, match="No unified GEMV shape matching bundle"):
        resolve_unified_shape_name(999, 888, "unsupported_fmt")

    # 2. Unknown bundle in validate_manifest_shapes
    backend = NpuBackend(mock=True, gemv="unified")
    backend.bundle_map["invalid/fake_gemv"] = BundleMeta(
        name="invalid/fake_gemv",
        rel_file="fake.bin",
        fmt="unknown_format",
        orig_shape=[100, 100],
        padded_shape=[100, 100],
        k=100,
        m=100,
        components=[],
    )
    with pytest.raises(ValueError, match="without valid unified GEMV shape mapping"):
        backend.validate_manifest_shapes()


def test_resolve_unified_shape_name_rejects_kind_bypass():
    """Shape must come from (K, M, fmt); bundle kind only validates, never rescues."""
    # 1. Known kind with completely wrong K, M, format
    with pytest.raises(ValueError, match="No unified GEMV shape matching bundle"):
        resolve_unified_shape_name(999, 888, "q4nx_gemv", bundle_name="L00/rda_in")

    # 2. Known kind, correct K/M but wrong format
    with pytest.raises(ValueError, match="No unified GEMV shape matching bundle"):
        resolve_unified_shape_name(2560, 20736, "q4nx_gemv", bundle_name="L00/rda_in")

    # 3. Known kind, correct K/format but wrong M (resolves to a different shape)
    with pytest.raises(ValueError, match="maps to shape 'rda_in'.*resolves to 'mla_in'"):
        resolve_unified_shape_name(2560, 1152, "bfp16_gemv", bundle_name="L00/rda_in")

    # 4. Known kind with a valid (K, M, fmt) that belongs to a different shape
    with pytest.raises(ValueError, match="maps to shape 'rda_in'.*resolves to 'g'"):
        resolve_unified_shape_name(2560, 5120, "bfp16_gemv", bundle_name="L00/rda_in")

    # 5. Unknown kind: kind mismatch against resolved shape
    with pytest.raises(ValueError, match="Unknown bundle kind 'unknown'"):
        resolve_unified_shape_name(2560, 20736, "bfp16_gemv", bundle_name="L99/unknown")


# ------------------------------------------------------------------------------
# Work buffer preparation contract tests
# ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "buffer_name",
    ["moe_x_gu", "moe_x_dn", "moe_y_gu", "moe_y_sh", "moe_y_dn", "gemv_x_unified"],
)
def test_missing_work_buffer_raises_runtime_error(buffer_name: str, monkeypatch: pytest.MonkeyPatch):
    """Using a prepared work buffer before _init_hardware_operators must raise RuntimeError."""
    import npu.backend as backend_mod
    from kernels.moe.pack import (
        B1_K,
        B1_M_PER_EXPERT,
        B2_K_PAD,
        B2_M_PER_EXPERT,
        NUM_ACTIVE_TOTAL,
    )

    class FakeTensor:
        def __init__(self, size: int):
            self._arr = np.zeros(size, dtype=bfloat16)

        def __setitem__(self, idx, val):
            self._arr[idx] = val

        def __getitem__(self, idx):
            return self._arr[idx]

        def numpy(self):
            return self._arr

        def to(self, device: str):
            return self

    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)

    # Build a full set of prepared buffers, then drop the target one.
    prepared_sizes = {
        "moe_y_gu": 6 * B1_M_PER_EXPERT,
        "moe_y_sh": B1_M_PER_EXPERT,
        "moe_y_dn": NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,
        "moe_x_gu": B1_K,
        "moe_x_dn": NUM_ACTIVE_TOTAL * B2_K_PAD,
        "gemv_x_unified": MAX_K,
    }

    # gemv_x_unified is exercised through _execute_gemv
    if buffer_name == "gemv_x_unified":
        backend = NpuBackend(mock=True, gemv="unified")
        backend.mock = False
        backend._work_buffers = {
            name: FakeTensor(size) for name, size in prepared_sizes.items() if name != buffer_name
        }
        backend._get_or_compile_gemv = lambda k, m, fmt: None  # type: ignore[assignment]
        backend._get_weight_tensor = lambda name: "fake_weight_tensor"  # type: ignore[assignment]
        backend._work_buffers[f"gemv_y_{5120}"] = FakeTensor(5120)
        bmeta = backend.bundle_map["L00/f_b"]
        x_in = np.zeros(bmeta.k, dtype=np.uint16)
        with pytest.raises(RuntimeError, match=f"{buffer_name} work buffer missing"):
            backend._execute_gemv("L00/f_b", bmeta, x_in)
        return

    # MoE buffers are exercised through moe_experts
    layer = 1
    dim = 2560
    moe_dim = 1088
    top_ids = np.array([45, 12, 88, 3, 102, 55], dtype=np.int32)
    top_weights = np.array([0.35, 0.25, 0.15, 0.12, 0.08, 0.05], dtype=np.float32)

    weights_dict: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(20261030)
    for eid in top_ids:
        g = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        u = rng.standard_normal((moe_dim, dim)).astype(np.float32) * 0.02
        d = rng.standard_normal((dim, moe_dim)).astype(np.float32) * 0.02
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.gate_proj.weight"] = f32_to_bf16_rtne(g)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.up_proj.weight"] = f32_to_bf16_rtne(u)
        weights_dict[f"model.layers.{layer}.moe.experts.{eid}.down_proj.weight"] = f32_to_bf16_rtne(d)

    fake_weights = FakeWeights(weights_dict)
    backend = NpuBackend(host_weights=fake_weights, mock=True, gemv="legacy")
    backend.mock = False
    backend._work_buffers = {
        name: FakeTensor(size) for name, size in prepared_sizes.items() if name != buffer_name
    }
    backend._get_or_compile_moe_ops = lambda: (None, None, None)  # type: ignore[assignment]
    backend._get_weight_tensor = lambda name: "fake_weight_tensor"  # type: ignore[assignment]

    x = rng.standard_normal(dim).astype(np.float32)
    x_u16 = f32_to_bf16_rtne(x)
    with pytest.raises(RuntimeError, match=f"{buffer_name} work buffer missing"):
        backend.moe_experts(layer, x_u16, top_ids, top_weights)


def test_prepared_buffer_access_only_through_helper_ast():
    """AST rule: every read of a prepared work buffer in backend.py goes through _prepared_buffer."""
    backend_py = Path(__file__).resolve().parent / "backend.py"
    tree = ast.parse(backend_py.read_text(encoding="utf-8"))

    prepared_names = {
        "moe_y_gu",
        "moe_y_sh",
        "moe_y_dn",
        "moe_x_gu",
        "moe_x_dn",
        "gemv_x_unified",
    }

    violations: list[str] = []

    class WorkBufferVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self._inside_helper = False

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            old = self._inside_helper
            self._inside_helper = node.name == "_prepared_buffer"
            self.generic_visit(node)
            self._inside_helper = old

        def visit_Subscript(self, node: ast.Subscript) -> None:
            if (
                not self._inside_helper
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "_work_buffers"
                and isinstance(node.slice, ast.Constant)
                and node.slice.value in prepared_names
                and isinstance(node.ctx, ast.Load)
            ):
                violations.append(
                    f"Line {node.lineno}: direct self._work_buffers[{node.slice.value!r}] read"
                )
            self.generic_visit(node)

    visitor = WorkBufferVisitor()
    visitor.visit(tree)
    assert not violations, "Prepared work buffer read bypasses _prepared_buffer:\n" + "\n".join(violations)


def test_x_padding_and_b2_slots_layout():
    """Verify pad_x dimensions and MoE B2 7-slot zero-padded layout."""
    # 1. General GEMV padding to MAX_K (12,032)
    k_rda = 2560
    x_rda = np.arange(1, k_rda + 1, dtype=np.uint16)
    pad_rda = pad_x(x_rda, "rda_in")
    assert pad_rda.shape == (MAX_K,)
    assert np.array_equal(pad_rda[:k_rda], x_rda)
    assert np.all(pad_rda[k_rda:] == 0)

    # 2. MoE B1 padding to MAX_K
    k_b1 = 2560
    x_b1 = np.arange(1, k_b1 + 1, dtype=np.uint16)
    pad_b1 = pad_x(x_b1, "moe_b1_6")
    assert pad_b1.shape == (MAX_K,)
    assert np.array_equal(pad_b1[:k_b1], x_b1)
    assert np.all(pad_b1[k_b1:] == 0)

    # 3. MoE B2 7-slot padding to 7 * MAX_K (84,224)
    k_b2 = 1280
    slots = 7
    x_b2_2d = np.arange(1, slots * k_b2 + 1, dtype=np.uint16).reshape(slots, k_b2)
    pad_b2 = pad_x(x_b2_2d, "moe_b2_9")
    assert pad_b2.shape == (slots * MAX_K,)
    pad_b2_2d = pad_b2.reshape(slots, MAX_K)

    # Each slot j has exact 1280 input elements, remaining 12032 - 1280 is zero
    for j in range(slots):
        assert np.array_equal(pad_b2_2d[j, :k_b2], x_b2_2d[j]), f"Slot {j} data mismatch"
        assert np.all(pad_b2_2d[j, k_b2:] == 0), f"Slot {j} tail is not zero"

    # 4. Invalid dimensions raise ValueError
    with pytest.raises(ValueError, match="expected 2560 x values"):
        pad_x(np.zeros(100, dtype=np.uint16), "rda_in")

    with pytest.raises(ValueError, match="expected 8960 x values"):
        pad_x(np.zeros(100, dtype=np.uint16), "moe_b2_9")


def test_mock_engines_share_numpy_reference_path():
    """Verify that unified and legacy mock engines follow the same NumPy reference path.

    Both engines use the same mock implementation in this test, so this only
    proves engine selection reaches an identical reference computation. On the
    hardware, the laptop A/B golden runs (unified vs legacy) checked the engines
    instead: identical golden metrics, not a bitwise comparison of outputs.
    """
    backend_u = NpuBackend(mock=True, gemv="unified")
    backend_l = NpuBackend(mock=True, gemv="legacy")

    rng = np.random.default_rng(2026)

    # --- 1. Dense MLP Layer (Layer 0) ---
    w_d_gu = f32_to_bf16_rtne(rng.standard_normal((24064, 2560)).astype(np.float32) * 0.05)
    w_d_dn = f32_to_bf16_rtne(rng.standard_normal((2560, 12032)).astype(np.float32) * 0.05)
    backend_u.mock_weight_arrays["L00/dense_gu"] = w_d_gu
    backend_l.mock_weight_arrays["L00/dense_gu"] = w_d_gu
    backend_u.mock_weight_arrays["L00/dense_down"] = w_d_dn
    backend_l.mock_weight_arrays["L00/dense_down"] = w_d_dn

    hidden_u16 = f32_to_bf16_rtne(rng.standard_normal(2560).astype(np.float32))

    # Dense gate & up projections
    g_u = backend_u.linear("model.layers.0.mlp.gate_proj.weight", hidden_u16)
    g_l = backend_l.linear("model.layers.0.mlp.gate_proj.weight", hidden_u16)
    assert np.array_equal(g_u, g_l)

    u_u = backend_u.linear("model.layers.0.mlp.up_proj.weight", hidden_u16)
    u_l = backend_l.linear("model.layers.0.mlp.up_proj.weight", hidden_u16)
    assert np.array_equal(u_u, u_l)

    # Dense down projection
    act_12032 = f32_to_bf16_rtne(rng.standard_normal(12032).astype(np.float32))
    dn_u = backend_u.linear("model.layers.0.mlp.down_proj.weight", act_12032)
    dn_l = backend_l.linear("model.layers.0.mlp.down_proj.weight", act_12032)
    assert np.array_equal(dn_u, dn_l)

    # --- 2. MoE Layer (Layer 1) ---
    w_moe_b1 = f32_to_bf16_rtne(rng.standard_normal((128, 2176, 2560)).astype(np.float32) * 0.05)
    w_sh_gu = f32_to_bf16_rtne(rng.standard_normal((2176, 2560)).astype(np.float32) * 0.05)
    w_moe_b2 = f32_to_bf16_rtne(rng.standard_normal((129, 2560, 1280)).astype(np.float32) * 0.05)

    backend_u.mock_weight_arrays["L01/moe_b1"] = w_moe_b1
    backend_l.mock_weight_arrays["L01/moe_b1"] = w_moe_b1
    backend_u.mock_weight_arrays["L01/shared_gu"] = w_sh_gu
    backend_l.mock_weight_arrays["L01/shared_gu"] = w_sh_gu
    backend_u.mock_weight_arrays["L01/moe_b2"] = w_moe_b2
    backend_l.mock_weight_arrays["L01/moe_b2"] = w_moe_b2

    top_ids = np.array([3, 17, 45, 78, 102, 115])
    top_weights = np.array([0.35, 0.25, 0.15, 0.10, 0.10, 0.05], dtype=np.float32)

    backend_u.set_step(1)
    backend_l.set_step(1)
    moe_out_u = backend_u.moe_experts(1, hidden_u16, top_ids, top_weights)
    moe_out_l = backend_l.moe_experts(1, hidden_u16, top_ids, top_weights)
    assert np.array_equal(moe_out_u, moe_out_l)

    # Shared down projection via cached slot 6
    sh_act_u = backend_u._cached_shared[1]["sh_act_u16"]
    sh_act_l = backend_l._cached_shared[1]["sh_act_u16"]
    assert np.array_equal(sh_act_u, sh_act_l)
    sh_dn_u = backend_u.linear("model.layers.1.moe.shared_experts.down_proj.weight", sh_act_u)
    sh_dn_l = backend_l.linear("model.layers.1.moe.shared_experts.down_proj.weight", sh_act_l)
    assert np.array_equal(sh_dn_u, sh_dn_l)

    # --- 3. MLA Layer (Layer 3) ---
    w_mla_in = f32_to_bf16_rtne(rng.standard_normal((1152, 2560)).astype(np.float32) * 0.05)
    w_qb = f32_to_bf16_rtne(rng.standard_normal((7680, 768)).astype(np.float32) * 0.05)
    w_kb = f32_to_bf16_rtne(rng.standard_normal((5120, 512)).astype(np.float32) * 0.05)
    w_vb = f32_to_bf16_rtne(rng.standard_normal((5120, 512)).astype(np.float32) * 0.05)
    w_g = f32_to_bf16_rtne(rng.standard_normal((5120, 2560)).astype(np.float32) * 0.05)
    w_o = f32_to_bf16_rtne(rng.standard_normal((2560, 5120)).astype(np.float32) * 0.05)

    for b, w in [
        ("L03/mla_in", w_mla_in),
        ("L03/q_b", w_qb),
        ("L03/k_b", w_kb),
        ("L03/v_b", w_vb),
        ("L03/g", w_g),
        ("L03/o_proj", w_o),
    ]:
        backend_u.mock_weight_arrays[b] = w
        backend_l.mock_weight_arrays[b] = w

    qa_u = backend_u.linear("model.layers.3.self_attn.q_a_proj.weight", hidden_u16)
    qa_l = backend_l.linear("model.layers.3.self_attn.q_a_proj.weight", hidden_u16)
    assert np.array_equal(qa_u, qa_l)

    qb_inp = f32_to_bf16_rtne(rng.standard_normal(576).astype(np.float32))
    qb_u = backend_u.linear("model.layers.3.self_attn.q_b_proj.weight", qb_inp)
    qb_l = backend_l.linear("model.layers.3.self_attn.q_b_proj.weight", qb_inp)
    assert np.array_equal(qb_u, qb_l)

    kvb_inp = f32_to_bf16_rtne(rng.standard_normal(512).astype(np.float32))
    kvb_u = backend_u.linear("model.layers.3.self_attn.kv_b_proj.weight", kvb_inp)
    kvb_l = backend_l.linear("model.layers.3.self_attn.kv_b_proj.weight", kvb_inp)
    assert np.array_equal(kvb_u, kvb_l)

    g_u = backend_u.linear("model.layers.3.self_attn.g_proj.weight", hidden_u16)
    g_l = backend_l.linear("model.layers.3.self_attn.g_proj.weight", hidden_u16)
    assert np.array_equal(g_u, g_l)

    o_inp = f32_to_bf16_rtne(rng.standard_normal(5120).astype(np.float32))
    o_u = backend_u.linear("model.layers.3.self_attn.o_proj.weight", o_inp)
    o_l = backend_l.linear("model.layers.3.self_attn.o_proj.weight", o_inp)
    assert np.array_equal(o_u, o_l)

    # --- 4. RDA Layer (Layer 0) ---
    w_rda_in = f32_to_bf16_rtne(rng.standard_normal((20736, 2560)).astype(np.float32) * 0.05)
    w_fb = f32_to_bf16_rtne(rng.standard_normal((5120, 256)).astype(np.float32) * 0.05)
    backend_u.mock_weight_arrays["L00/rda_in"] = w_rda_in
    backend_l.mock_weight_arrays["L00/rda_in"] = w_rda_in
    backend_u.mock_weight_arrays["L00/f_b"] = w_fb
    backend_l.mock_weight_arrays["L00/f_b"] = w_fb

    rda_q_u = backend_u.linear("model.layers.0.self_attn.q_proj.weight", hidden_u16)
    rda_q_l = backend_l.linear("model.layers.0.self_attn.q_proj.weight", hidden_u16)
    assert np.array_equal(rda_q_u, rda_q_l)

    fb_inp = f32_to_bf16_rtne(rng.standard_normal(128).astype(np.float32))
    fb_u = backend_u.linear("model.layers.0.self_attn.f_b_proj.weight", fb_inp)
    fb_l = backend_l.linear("model.layers.0.self_attn.f_b_proj.weight", fb_inp)
    assert np.array_equal(fb_u, fb_l)

    # RDA recurrence
    q_vec = rng.standard_normal((40, 128)).astype(np.float32)
    k_vec = rng.standard_normal((40, 128)).astype(np.float32)
    v_vec = rng.standard_normal((40, 128)).astype(np.float32)
    log_alpha = -np.abs(rng.standard_normal((40, 128)).astype(np.float32))
    beta = rng.standard_normal(40).astype(np.float32)

    rda_rec_u = backend_u.rda_update(0, q_vec, k_vec, v_vec, log_alpha, beta)
    rda_rec_l = backend_l.rda_update(0, q_vec, k_vec, v_vec, log_alpha, beta)
    assert np.array_equal(rda_rec_u, rda_rec_l)


def test_xclbin_switch_counts_mock():
    """Verify that unified engine eliminates GEMV-to-GEMV xclbin context switches."""
    backend_u = NpuBackend(mock=True, gemv="unified")
    backend_l = NpuBackend(mock=True, gemv="legacy")

    rng = np.random.default_rng(42)
    hidden_u16 = f32_to_bf16_rtne(rng.standard_normal(2560).astype(np.float32))

    # Populate weights for Layer 1 (RDA + MoE)
    backend_u.mock_weight_arrays["L01/rda_in"] = np.zeros((20736, 2560), dtype=np.uint16)
    backend_l.mock_weight_arrays["L01/rda_in"] = np.zeros((20736, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L01/f_b"] = np.zeros((5120, 256), dtype=np.uint16)
    backend_l.mock_weight_arrays["L01/f_b"] = np.zeros((5120, 256), dtype=np.uint16)
    backend_u.mock_weight_arrays["L01/o_proj"] = np.zeros((2560, 5120), dtype=np.uint16)
    backend_l.mock_weight_arrays["L01/o_proj"] = np.zeros((2560, 5120), dtype=np.uint16)
    backend_u.mock_weight_arrays["L01/moe_b1"] = np.zeros((128, 2176, 2560), dtype=np.uint16)
    backend_l.mock_weight_arrays["L01/moe_b1"] = np.zeros((128, 2176, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L01/shared_gu"] = np.zeros((2176, 2560), dtype=np.uint16)
    backend_l.mock_weight_arrays["L01/shared_gu"] = np.zeros((2176, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L01/moe_b2"] = np.zeros((129, 2560, 1280), dtype=np.uint16)
    backend_l.mock_weight_arrays["L01/moe_b2"] = np.zeros((129, 2560, 1280), dtype=np.uint16)

    q = np.zeros((40, 128), dtype=np.float32)
    k = np.zeros((40, 128), dtype=np.float32)
    v = np.zeros((40, 128), dtype=np.float32)
    log_alpha = np.zeros((40, 128), dtype=np.float32)
    beta = np.zeros(40, dtype=np.float32)
    top_ids = np.array([0, 1, 2, 3, 4, 5])
    top_weights = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1], dtype=np.float32)

    # --- Run Layer 1 on Unified ---
    backend_u.set_step(1)
    backend_u.reset_timings()
    backend_u.linear("model.layers.1.self_attn.q_proj.weight", hidden_u16)
    backend_u.linear("model.layers.1.self_attn.f_b_proj.weight", np.zeros(128, dtype=np.uint16))
    backend_u.rda_update(1, q, k, v, log_alpha, beta)
    backend_u.linear("model.layers.1.self_attn.o_proj.weight", np.zeros(5120, dtype=np.uint16))
    backend_u.moe_experts(1, hidden_u16, top_ids, top_weights)

    # In unified: rda_in/f_b (unified) -> rda_step (merged unified, 0 switches) -> o_proj/moe_b1/shared_gu/moe_b2 (unified) = 0 switches
    assert backend_u.xclbin_switches == 0, f"Expected 0 switches in unified RDA layer, got {backend_u.xclbin_switches}"

    # --- Run Layer 1 on Legacy ---
    backend_l.set_step(1)
    backend_l.reset_timings()
    backend_l.linear("model.layers.1.self_attn.q_proj.weight", hidden_u16)
    backend_l.linear("model.layers.1.self_attn.f_b_proj.weight", np.zeros(128, dtype=np.uint16))
    backend_l.rda_update(1, q, k, v, log_alpha, beta)
    backend_l.linear("model.layers.1.self_attn.o_proj.weight", np.zeros(5120, dtype=np.uint16))
    backend_l.moe_experts(1, hidden_u16, top_ids, top_weights)

    # In legacy: rda_in -> f_b (1) -> rda_step (2) -> o_proj (3) -> moe_b1 (4) -> shared_gu (5) -> moe_b2 (6) = 6 switches
    assert backend_l.xclbin_switches == 6, f"Expected 6 switches in legacy RDA layer, got {backend_l.xclbin_switches}"

    # --- Run Layer 3 (MLA + MoE) on Unified ---
    backend_u.mock_weight_arrays["L03/mla_in"] = np.zeros((1152, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/q_b"] = np.zeros((7680, 768), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/k_b"] = np.zeros((5120, 512), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/v_b"] = np.zeros((5120, 512), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/g"] = np.zeros((5120, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/o_proj"] = np.zeros((2560, 5120), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/moe_b1"] = np.zeros((128, 2176, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/shared_gu"] = np.zeros((2176, 2560), dtype=np.uint16)
    backend_u.mock_weight_arrays["L03/moe_b2"] = np.zeros((129, 2560, 1280), dtype=np.uint16)

    backend_u.set_step(2)
    backend_u.reset_timings()
    backend_u.linear("model.layers.3.self_attn.q_a_proj.weight", hidden_u16)
    backend_u.linear("model.layers.3.self_attn.q_b_proj.weight", np.zeros(576, dtype=np.uint16))
    backend_u.linear("model.layers.3.self_attn.kv_b_proj.weight", np.zeros(512, dtype=np.uint16))
    backend_u.linear("model.layers.3.self_attn.g_proj.weight", hidden_u16)
    backend_u.linear("model.layers.3.self_attn.o_proj.weight", np.zeros(5120, dtype=np.uint16))
    backend_u.moe_experts(3, hidden_u16, top_ids, top_weights)

    # In unified MLA layer: NO RDA step! All kernels are gemv_unified!
    # Switches must be 0!
    assert backend_u.xclbin_switches == 0, f"Expected 0 switches in unified MLA layer, got {backend_u.xclbin_switches}"


def test_dweights_f32_cache_identity_readonly_limit_and_instance_scope(
    tmp_path: Path, monkeypatch
):
    raw = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)
    expected = bf16_to_f32(raw)
    dw = DWeights(tmp_path)
    dw.tensors["small"] = raw

    first = dw.get("small", to_float=True)
    second = dw.get("small", to_float=True)
    assert first is second
    assert np.array_equal(first.view(np.uint32), expected.view(np.uint32))
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = 0.0

    raw_path = dw.get("small", to_float=False)
    assert raw_path is raw
    assert raw_path.flags.writeable

    monkeypatch.setattr(backend_mod, "DWEIGHTS_F32_CACHE_MAX_TENSOR_BYTES", 4)
    dw.tensors["too_large"] = raw.copy()
    uncached_1 = dw.get("too_large", to_float=True)
    uncached_2 = dw.get("too_large", to_float=True)
    assert uncached_1 is not uncached_2
    assert "too_large" not in dw._f32_cache

    fresh = DWeights(tmp_path)
    fresh.tensors["small"] = raw.copy()
    assert fresh._f32_cache == {}
    assert fresh.get("small", to_float=True) is not first


class _CountingHostWeights:
    """Fake Weights that records how many times its get() is called."""

    def __init__(self):
        self.tensors = {"host_only_weight": np.array([0x3F80, 0x4000], dtype=np.uint16)}
        self.get_calls = 0

    def get(self, name: str, to_float: bool = True) -> np.ndarray:
        self.get_calls += 1
        arr = self.tensors[name]
        if arr.dtype == np.uint16:
            return bf16_to_f32(arr) if to_float else arr
        return arr.astype(np.float32) if to_float and arr.dtype != np.float32 else arr

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


def test_dweights_f32_cache_checks_before_host_get(tmp_path: Path):
    """When _f32_cache has a host tensor, DWeights.get must not call host_weights.get again."""
    fake = _CountingHostWeights()
    dw = DWeights(tmp_path, host_weights=fake)

    first = dw.get("host_only_weight", to_float=True)
    assert fake.get_calls == 1

    second = dw.get("host_only_weight", to_float=True)
    assert fake.get_calls == 1, "cache hit must not re-invoke host_weights.get"
    assert first is second

    # to_float=False still bypasses the f32 cache and reads from the host source.
    raw = dw.get("host_only_weight", to_float=False)
    assert fake.get_calls == 2
    assert raw is fake.tensors["host_only_weight"]


def test_prepared_dynamic_kernels_load_once_and_run_each_call(monkeypatch):
    class Runtime:
        def __init__(self):
            self.loads = []
            self.runs = []

        def load(self, kernel):
            self.loads.append(kernel)
            return f"handle-{len(self.loads)}"

        def run(self, handle, args, **kwargs):
            self.runs.append((handle, args, kwargs))

    class Kernel:
        dispatch_params = ["e0", "e1"]

        def __init__(self, name):
            self.name = name
            self.generated = []

        def instructions(self, **scalars):
            self.generated.append(scalars)
            return np.array([scalars["e0"], scalars["e1"]], dtype=np.uint32)

    runtime = Runtime()
    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        types.SimpleNamespace(DefaultNPURuntime=runtime),
    )
    b1 = Kernel("b1")
    b2 = Kernel("b2")
    run_b1 = NpuBackend._prepare_dynamic_kernel(b1)
    run_b2 = NpuBackend._prepare_dynamic_kernel(b2)

    y1, w1, x1 = object(), object(), object()
    y2, w2, x2 = object(), object(), object()
    run_b1(y1, w1, x1, e0=3, e1=7)
    run_b1(y1, w1, x1, e0=4, e1=8)
    run_b2(y2, w2, x2, e0=5, e1=9)
    run_b2(y2, w2, x2, e0=6, e1=10)

    assert runtime.loads == [b1, b2]
    assert len(runtime.runs) == 4
    assert runtime.runs[0][1] == [y1, w1, x1]
    assert runtime.runs[2][1] == [y2, w2, x2]
    assert np.array_equal(runtime.runs[1][2]["dispatch_insts"], [4, 8])
    assert np.array_equal(runtime.runs[3][2]["dispatch_insts"], [6, 10])


def test_rda_preparation_builds_and_loads_once(monkeypatch):
    class Runtime:
        def __init__(self):
            self.load_count = 0
            self.runs = []

        def load(self, kernel):
            self.load_count += 1
            return "rda-handle"

        def run(self, handle, args):
            self.runs.append((handle, args))

    class Design:
        def __init__(self):
            self.builds = []
            self.compiles = []

        def _build_compilable(self, kwargs, args):
            self.builds.append((kwargs, args))
            return "fixed-rda-design"

        def _compile_and_build_kernel(self, compilable, cache_key, trace):
            self.compiles.append((compilable, cache_key, trace))
            return "rda-kernel"

    runtime = Runtime()
    design = Design()
    fake_module = types.ModuleType("bench.rda_state.rda_state")
    fake_module.rda_step = design
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", fake_module)
    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        types.SimpleNamespace(DefaultNPURuntime=runtime),
    )

    run_rda = NpuBackend._prepare_rda_kernel()
    args = [object(), object(), object(), object()]
    run_rda(*args)
    run_rda(*args)

    assert len(design.builds) == 1
    assert len(design.compiles) == 1
    assert runtime.load_count == 1
    assert runtime.runs == [("rda-handle", args), ("rda-handle", args)]


def test_rda_prepared_callable_preserves_ping_pong_buffer_arguments(monkeypatch):
    class Tensor:
        def __init__(self, shape):
            self.data = np.zeros(shape, dtype=np.float32)

        def __setitem__(self, key, value):
            self.data[key] = value

        def numpy(self):
            return self.data

    fake_iron = types.SimpleNamespace(
        zeros=lambda shape, **kwargs: Tensor(shape),
    )
    fake_module = types.ModuleType("bench.rda_state.rda_state")
    fake_module.pack_head_bundle = lambda q, k, a, v, beta: np.zeros(
        (40, 640), dtype=np.float32
    )
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", fake_module)
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    monkeypatch.setattr(backend_mod, "iron", fake_iron)

    backend = NpuBackend(mock=True)
    backend.mock = False
    backend._work_buffers["rda_head_in"] = Tensor((40, 640))
    backend._work_buffers["rda_out"] = Tensor((40, 128))
    calls = []
    backend._compiled_ops["rda_step"] = lambda *args: calls.append(args)

    zeros = np.zeros((40, 128), dtype=np.float32)
    beta = np.zeros(40, dtype=np.float32)
    backend.rda_update(3, zeros, zeros, zeros, zeros, beta)
    backend.rda_update(3, zeros, zeros, zeros, zeros, beta)

    state_0, state_1 = backend.rda_states[3]
    assert calls[0] == (
        state_0,
        state_1,
        backend._work_buffers["rda_head_in"],
        backend._work_buffers["rda_out"],
    )
    assert calls[1] == (
        state_1,
        state_0,
        backend._work_buffers["rda_head_in"],
        backend._work_buffers["rda_out"],
    )
    assert backend.rda_active_slot[3] == 0


def test_runtime_load_calls_are_confined_to_preparation_functions():
    tree = ast.parse(Path(backend_mod.__file__).read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def _outer_function_name(node: ast.AST) -> str | None:
        name: str | None = None
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = parent.name
            parent = parents.get(parent)
        return name

    def _is_runtime_load_call(node: ast.Call) -> bool:
        # Direct runtime.load(...) / runtime.load_and_run(...)
        if isinstance(node.func, ast.Attribute):
            if node.func.attr not in {"load", "load_and_run"}:
                return False
            # Exclude obvious non-runtime module methods (np.load, json.load, etc.).
            if isinstance(node.func.value, ast.Name) and node.func.value.id in {"np"}:
                return False
            return True
        # Alias: ld = runtime.load; ld(...)
        if isinstance(node.func, ast.Name):
            return node.func.id in {"load", "load_and_run"}
        # getattr(runtime, "load")(...)
        if (
            isinstance(node.func, ast.Call)
            and isinstance(node.func.func, ast.Attribute)
            and node.func.func.attr == "getattr"
        ):
            args = node.func.args
            if len(args) >= 2 and isinstance(args[1], ast.Constant):
                return args[1].value in {"load", "load_and_run"}
        return False

    found: list[tuple[ast.Call, str | None]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_runtime_load_call(node):
            continue
        # Exclude numpy.load / other non-runtime uses by name.
        if isinstance(node.func, ast.Name):
            # A bare 'load' alias could be anything; we still flag it and let the
            # outer-function check decide.  This keeps the rule strict.
            pass
        outer = _outer_function_name(node)
        found.append((node, outer))

    assert found, "Expected at least one runtime load/load_and_run call"
    violations = [
        (outer, ast.unparse(node))
        for node, outer in found
        if outer is None or not outer.startswith("_prepare_")
    ]
    assert not violations, (
        "runtime load/load_and_run calls must be confined to _prepare_* functions "
        "(nested closures inside preparation functions are allowed):\n"
        + "\n".join(f"{outer}: {code}" for outer, code in violations)
    )


def test_f32_cache_excludes_non_host_branches(tmp_path: Path, monkeypatch):
    """Branch 2-4 to_float=True results must not enter the permanent _f32_cache."""
    dw = DWeights(tmp_path)

    # Make decode paths return small deterministic arrays so the 64 MiB bound
    # would allow caching if the wrong branch were cached.
    monkeypatch.setattr(backend_mod, "DWEIGHTS_F32_CACHE_MAX_TENSOR_BYTES", 64 * 1024 * 1024)

    monkeypatch.setattr(
        dw, "_decode_2d_bundle", lambda bundle_name: np.ones((5120, 512), dtype=np.float32)
    )
    monkeypatch.setattr(dw, "_read_expert_slice", lambda bundle_name, expert_id: b"\x00" * 64)
    monkeypatch.setattr(
        backend_mod,
        "parse_bfp16_gemv",
        lambda data, K, M, rows: np.ones((M, K), dtype=np.float32),
    )

    def fake_parse_gemv(data, K, M, rows):
        return (
            np.ones((M // 4, K // 16), dtype=np.int32),
            np.ones((M // 4, K // 16), dtype=np.float32),
            np.ones((M // 4, K // 16), dtype=np.float32),
        )

    monkeypatch.setattr(backend_mod, "parse_gemv", fake_parse_gemv)
    monkeypatch.setattr(
        backend_mod,
        "dequantize_simple",
        lambda codes, scales, mins: np.ones(
            (codes.shape[0] * 4, codes.shape[1] * 16), dtype=np.float32
        ),
    )

    non_host_names = [
        "model.layers.3.self_attn.kv_b_proj.weight",  # mla_kv_b
        "model.layers.1.moe.shared_experts.down_proj.weight",  # shared_down
        "model.layers.0.self_attn.f_b_proj.weight",  # regular bundle slice
        "model.layers.1.moe.experts.5.gate_proj.weight",  # routed expert
        "model.layers.1.moe.experts.5.down_proj.weight",  # routed expert
    ]
    for name in non_host_names:
        _ = dw.get(name, to_float=True)
        _ = dw.get(name, to_float=True)
        assert name not in dw._f32_cache, f"{name} should not be in _f32_cache"

    # Force expert-cache eviction and confirm the evicted names stay out of _f32_cache.
    dw.max_expert_cache_entries = 1
    for eid in (5, 6):
        _ = dw.get(f"model.layers.1.moe.experts.{eid}.gate_proj.weight", to_float=True)
    for expert_name in dw._expert_cache:
        assert expert_name not in dw._f32_cache


def test_prepared_dynamic_kernel_reloads_invalid_handle(monkeypatch):
    """Invalid prepared handle triggers a single reload before execution."""

    class Handle:
        def __init__(self, name: str):
            self.name = name
            self._is_valid = True

    class Runtime:
        def __init__(self):
            self.loads: list[tuple[Any, Handle]] = []
            self.runs: list[tuple[Handle, list[Any], dict[str, Any]]] = []

        def load(self, kernel: Any) -> Handle:
            h = Handle(f"handle-{len(self.loads)}")
            self.loads.append((kernel, h))
            return h

        def run(self, handle: Handle, args: list[Any], **kwargs: Any) -> None:
            self.runs.append((handle, args, kwargs))

    class Kernel:
        dispatch_params = ["e0", "e1"]

        def __init__(self, name: str):
            self.name = name
            self.generated: list[dict[str, int]] = []

        def instructions(self, **scalars: int) -> np.ndarray:
            self.generated.append(scalars)
            return np.array([scalars["e0"], scalars["e1"]], dtype=np.uint32)

    runtime = Runtime()
    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        types.SimpleNamespace(DefaultNPURuntime=runtime),
    )
    k = Kernel("b1")
    run = NpuBackend._prepare_dynamic_kernel(k)

    y1, w1, x1 = object(), object(), object()
    run(y1, w1, x1, e0=3, e1=7)
    assert len(runtime.loads) == 1
    h1 = runtime.runs[-1][0]

    h1._is_valid = False
    run(y1, w1, x1, e0=4, e1=8)
    assert len(runtime.loads) == 2
    h2 = runtime.runs[-1][0]
    assert h2 is not h1
    assert h2._is_valid is True
    assert np.array_equal(runtime.runs[-1][2]["dispatch_insts"], [4, 8])

    # A valid handle is reused without another load.
    run(y1, w1, x1, e0=5, e1=9)
    assert len(runtime.loads) == 2
    assert runtime.runs[-1][0] is h2


def test_rda_kernel_recovers_xrt_einval_and_reloads_invalid_handle(monkeypatch):
    """RDA run retries once on XRT EINVAL and reloads when the handle is invalid."""
    from types import SimpleNamespace

    call_count = 0
    runtime = SimpleNamespace(load_count=0, runs=[], evicts=[])

    def make_handle() -> Any:
        runtime.load_count += 1
        return SimpleNamespace(
            name=f"handle-{runtime.load_count}",
            _is_valid=True,
            kernel=None,
        )

    def load(kernel: Any) -> Any:
        h = make_handle()
        h.kernel = kernel
        return h

    runtime.load = load
    runtime.evict_context = lambda path: runtime.evicts.append(path)

    class Kernel:
        def __init__(self, name: str, xclbin_path: str = "xclbin.xclbin"):
            self.name = name
            self.xclbin_path = xclbin_path

    class Design:
        def __init__(self):
            self.builds: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
            self.compiles: list[tuple[Any, Any, Any]] = []
            self.kernel_index = 0

        def _build_compilable(
            self, kwargs: dict[str, Any], args: tuple[Any, ...]
        ) -> Any:
            self.builds.append((kwargs, args))
            return "compilable"

        def _compile_and_build_kernel(
            self, compilable: Any, cache_key: Any, trace: Any
        ) -> Kernel:
            self.kernel_index += 1
            self.compiles.append((compilable, cache_key, trace))
            return Kernel(f"kernel-{self.kernel_index}", xclbin_path="xclbin.xclbin")

    design = Design()
    fake_module = types.ModuleType("bench.rda_state.rda_state")
    fake_module.rda_step = design
    fake_module.pack_head_bundle = lambda *args, **kwargs: np.zeros(
        (40, 640), dtype=np.float32
    )
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", fake_module)
    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        types.SimpleNamespace(DefaultNPURuntime=runtime),
    )

    run_rda = NpuBackend._prepare_rda_kernel()
    args: list[Any] = [object(), object(), object(), object()]

    # Phase 1: normal execution prepares once.
    runtime.run = lambda handle, args: runtime.runs.append((handle, args))
    run_rda(*args)
    assert runtime.load_count == 1
    assert len(design.builds) == 1
    assert len(design.compiles) == 1
    assert len(runtime.runs) == 1
    h1 = runtime.runs[0][0]

    # Phase 2: invalid handle reloads the same kernel without rebuilding.
    h1._is_valid = False
    runtime.runs.clear()
    run_rda(*args)
    assert runtime.load_count == 2
    assert len(design.builds) == 1
    assert len(design.compiles) == 1
    assert len(runtime.runs) == 1
    h2 = runtime.runs[0][0]
    assert h2 is not h1
    assert h2._is_valid is True

    # Phase 3: first run raises XRT EINVAL -> evict, rebuild, reload, retry once.
    runtime.runs.clear()
    call_count = 0

    def run_einval_once(handle: Any, args: list[Any]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("XRT error: err=-22 invalid argument")
        runtime.runs.append((handle, args))

    runtime.run = run_einval_once
    run_rda(*args)
    assert call_count == 2
    assert len(runtime.evicts) == 1
    assert runtime.evicts[-1] == "xclbin.xclbin"
    assert len(design.builds) == 2
    assert len(design.compiles) == 2
    assert runtime.load_count == 3
    assert len(runtime.runs) == 1

    # Phase 4: two consecutive EINVALs -> second error propagates, no third try.
    runtime.runs.clear()
    call_count = 0

    def run_double_einval(handle: Any, args: list[Any]) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("XRT error: err=-22 invalid argument")

    runtime.run = run_double_einval
    with pytest.raises(RuntimeError, match="err=-22"):
        run_rda(*args)
    assert call_count == 2
    assert len(runtime.evicts) == 2
    assert len(design.builds) == 3
    assert len(design.compiles) == 3
    assert runtime.load_count == 4

    # Phase 5: non-EINVAL RuntimeError propagates immediately.
    runtime.runs.clear()
    call_count = 0

    def run_other_error(handle: Any, args: list[Any]) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("some other runtime error")

    runtime.run = run_other_error
    with pytest.raises(RuntimeError, match="some other runtime error"):
        run_rda(*args)
    assert call_count == 1
    assert len(runtime.evicts) == 2
    assert len(design.builds) == 3
    assert len(design.compiles) == 3
    assert runtime.load_count == 4

    # Phase 6: bare "Invalid argument" without XRT is not treated as EINVAL.
    runtime.runs.clear()
    call_count = 0

    def run_bare_invalid(handle: Any, args: list[Any]) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Invalid argument")

    runtime.run = run_bare_invalid
    with pytest.raises(RuntimeError, match="Invalid argument"):
        run_rda(*args)
    assert call_count == 1


def test_rda_reload_first_run_einval_recovery(monkeypatch):
    """Re-loaded handle's first run is also covered by the EINVAL retry path."""
    from types import SimpleNamespace

    runtime = SimpleNamespace(load_count=0, runs=[], evicts=[], handles=[])

    def make_handle() -> Any:
        runtime.load_count += 1
        return SimpleNamespace(
            name=f"handle-{runtime.load_count}",
            _is_valid=True,
            kernel=None,
        )

    def load(kernel: Any) -> Any:
        h = make_handle()
        h.kernel = kernel
        runtime.handles.append(h)
        return h

    runtime.load = load
    runtime.evict_context = lambda path: runtime.evicts.append(path)

    class Kernel:
        def __init__(self, name: str, xclbin_path: str = "xclbin.xclbin"):
            self.name = name
            self.xclbin_path = xclbin_path

    class Design:
        def __init__(self):
            self.builds: list[tuple[dict[str, Any], tuple[Any, ...]]] = []
            self.compiles: list[tuple[Any, Any, Any]] = []
            self.kernel_index = 0

        def _build_compilable(
            self, kwargs: dict[str, Any], args: tuple[Any, ...]
        ) -> Any:
            self.builds.append((kwargs, args))
            return "compilable"

        def _compile_and_build_kernel(
            self, compilable: Any, cache_key: Any, trace: Any
        ) -> Kernel:
            self.kernel_index += 1
            self.compiles.append((compilable, cache_key, trace))
            return Kernel(f"kernel-{self.kernel_index}", xclbin_path="xclbin.xclbin")

    design = Design()
    fake_module = types.ModuleType("bench.rda_state.rda_state")
    fake_module.rda_step = design
    fake_module.pack_head_bundle = lambda *args, **kwargs: np.zeros(
        (40, 640), dtype=np.float32
    )
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", fake_module)
    monkeypatch.setattr(
        backend_mod,
        "aie_utils",
        types.SimpleNamespace(DefaultNPURuntime=runtime),
    )

    run_rda = NpuBackend._prepare_rda_kernel()
    args: list[Any] = [object(), object(), object(), object()]

    # Warm-up: the first call prepares and runs once.
    runtime.run = lambda handle, args: runtime.runs.append((handle, args))
    run_rda(*args)
    assert runtime.load_count == 1
    h1 = runtime.runs[0][0]

    # Case 1: invalid handle -> reload -> first run EINVAL -> recover, second run succeeds.
    h1._is_valid = False
    runtime.runs.clear()
    call_count = 0

    def run_reload_then_fail_once(handle: Any, args: list[Any]) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("XRT error: err=-22 invalid argument")
        runtime.runs.append((handle, args))

    runtime.run = run_reload_then_fail_once
    run_rda(*args)
    # initial load + reload + recovery load = 3 loads
    assert runtime.load_count == 3
    assert call_count == 2
    assert len(runtime.evicts) == 1
    assert len(design.builds) == 2
    assert len(design.compiles) == 2
    assert len(runtime.runs) == 1
    h3 = runtime.runs[0][0]
    assert h3.name == "handle-3"

    # Case 2: invalid handle -> reload -> run EINVAL -> recovery run also EINVAL -> propagates, no third try.
    h3._is_valid = False
    runtime.runs.clear()
    call_count = 0

    def run_double_einval_after_reload(handle: Any, args: list[Any]) -> None:
        nonlocal call_count
        call_count += 1
        raise RuntimeError("XRT error: err=-22 invalid argument")

    runtime.run = run_double_einval_after_reload
    with pytest.raises(RuntimeError, match="err=-22"):
        run_rda(*args)
    # previous 3 loads + reload + recovery load = 5 loads, no third attempt
    assert runtime.load_count == 5
    assert call_count == 2
    assert len(runtime.evicts) == 2
    assert len(design.builds) == 3
    assert len(design.compiles) == 3

    # Case 3: invalid handle -> reload itself raises EINVAL -> recover and retry once.
    runtime.handles[-1]._is_valid = False
    runtime.runs.clear()
    fail_next_load = True

    def run_success(handle: Any, args: list[Any]) -> None:
        runtime.runs.append((handle, args))

    def load_fails_once(kernel: Any) -> Any:
        nonlocal fail_next_load
        if fail_next_load:
            fail_next_load = False
            raise RuntimeError("XRT error: err=-22 load failed")
        return load(kernel)

    runtime.run = run_success
    runtime.load = load_fails_once
    run_rda(*args)
    # previous 5 loads + recovery load = 6 loads (failed reload raised before a handle was made)
    assert runtime.load_count == 6
    assert len(runtime.evicts) == 3
    assert len(design.builds) == 4
    assert len(design.compiles) == 4
    assert len(runtime.runs) == 1
    h6 = runtime.runs[0][0]
    assert h6.name == "handle-6"


def test_dweights_no_duplicate_host_copy():
    """When DWeights wraps a Weights host, cached bf16→f32 results share one object.

    Weights.get now caches read-only float32 arrays. DWeights._cache_f32_host must
    reuse that object instead of allocating a second copy.
    """
    rng = np.random.default_rng(20261101)
    bf16_name = "model.layers.0.self_attn.q_proj.weight"
    tensors = {
        bf16_name: f32_to_bf16_rtne(
            rng.standard_normal((64, 64)).astype(np.float32) * 0.05
        ),
    }
    host_weights = FakeWeights(tensors)
    dweights = DWeights("/nonexistent", host_weights=host_weights)

    # Stored as bf16: Weights.get caches a read-only f32 copy; DWeights must reuse it.
    from_host = host_weights.get(bf16_name, to_float=True)
    from_d = dweights.get(bf16_name, to_float=True)
    assert from_d is from_host
    assert from_d.dtype == np.float32
    assert not from_d.flags.writeable
    assert from_d.flags.c_contiguous


# -----------------------------------------------------------------------------
# Unified GEMV continuous_mem_w plumbing and single-xclbin contract
# -----------------------------------------------------------------------------

def test_unified_gemv_for_shape_forwards_continuous_mem_w():
    """``for_shape`` must pass continuous_mem_w to the operator and config_name."""
    from kernels.gemv_unified.op import UnifiedGEMV
    from kernels.gemv_unified.shapes import UnifiedShape, get_shape

    class FakeGEMV(UnifiedGEMV):
        def __post_init__(self) -> None:
            # Bypass device-dependent validation while keeping config_name computable.
            self.shape = get_shape(self.shape_name) if self.shape_name else UnifiedShape(
                "fake", self.K, self.M
            )

    op = FakeGEMV.for_shape("o_proj", continuous_mem_w=True)
    assert op.continuous_mem_w is True
    assert "contw" in op.config_name

    op_default = FakeGEMV.for_shape("o_proj")
    assert op_default.continuous_mem_w is False
    assert "contw" not in op_default.config_name


def test_unified_moe_template_forwards_continuous_mem_w():
    """``UnifiedMoETemplate`` and subclasses must pass continuous_mem_w to ``for_shape``."""
    from kernels.gemv_unified.op import UnifiedGEMV
    from kernels.gemv_unified.shapes import UnifiedShape, get_shape
    from kernels.gemv_unified.stage2 import UnifiedMoEDown9, UnifiedMoEGateUp

    class FakeGEMV(UnifiedGEMV):
        def __post_init__(self) -> None:
            self.shape = get_shape(self.shape_name) if self.shape_name else UnifiedShape(
                "fake", self.K, self.M
            )

    calls: list[tuple[str, bool]] = []

    def recording_for_shape(
        cls,
        shape: Any,
        *,
        cols: int = 8,
        context: Any = None,
        experimental_shape: bool = False,
        continuous_mem_w: bool = False,
        rda_col: int | None = None,
        rda_heads: int = 40,
        **kwargs: Any,
    ) -> Any:
        calls.append((str(shape), continuous_mem_w))
        return FakeGEMV(
            K=2560,
            M=2176,
            cols=cols,
            shape_name="moe_b1_6" if "b1" in str(shape) else "moe_b2_9",
            continuous_mem_w=continuous_mem_w,
            rda_col=rda_col,
            rda_heads=rda_heads,
        )

    import kernels.gemv_unified.op as op_mod
    import kernels.gemv_unified.stage2 as stage2_mod

    orig_for_shape = UnifiedGEMV.for_shape
    op_mod.UnifiedGEMV = FakeGEMV
    stage2_mod.UnifiedGEMV = FakeGEMV
    try:
        UnifiedGEMV.for_shape = classmethod(recording_for_shape)
        FakeGEMV.for_shape = classmethod(recording_for_shape)

        gu = UnifiedMoEGateUp(continuous_mem_w=True)
        assert gu._op.continuous_mem_w is True
        dn = UnifiedMoEDown9(continuous_mem_w=True)
        assert dn._op.continuous_mem_w is True
        gu_default = UnifiedMoEGateUp()
        assert gu_default._op.continuous_mem_w is False

        assert any("moe_b1_6" in call[0] and call[1] is True for call in calls)
        assert any("moe_b2_9" in call[0] and call[1] is True for call in calls)
        assert any("moe_b1_6" in call[0] and call[1] is False for call in calls)
    finally:
        UnifiedGEMV.for_shape = orig_for_shape
        op_mod.UnifiedGEMV = UnifiedGEMV
        stage2_mod.UnifiedGEMV = UnifiedGEMV


def test_require_single_unified_xclbin_rejects_mismatch():
    """Operators resolving to different xclbin paths must raise RuntimeError."""
    from npu.backend import _require_single_unified_xclbin

    class FakeArtifact:
        def __init__(self, filename: str):
            self.filename = filename

    class FakeOp:
        def __init__(self, name: str, filename: str):
            self.name = name
            self.xclbin_artifact = FakeArtifact(filename)

    p1 = FakeOp("dense", "/tmp/fake_contw.xclbin")
    p2 = FakeOp("moe_b2", "/tmp/other.xclbin")
    p3 = FakeOp("shared_gu", "/tmp/fake_contw.xclbin")

    with pytest.raises(RuntimeError, match="unified xclbin path mismatch"):
        _require_single_unified_xclbin([p1, p2])

    # Identical resolved paths are accepted.
    expected = _require_single_unified_xclbin([p1, p3])
    assert expected == Path("/tmp/fake_contw.xclbin").resolve()


def test_backend_unified_operators_pass_continuous_mem_w_via_ast():
    """Source check: unified compilation branches pass the engine-wide constant."""
    backend_py = Path(backend_mod.__file__)
    source = backend_py.read_text(encoding="utf-8")

    # Every unified operator construction (dense GEMV, B1, shared GU, B2, RDA) must use
    # the engine constant, not a hardcoded literal.
    assert source.count("continuous_mem_w=UNIFIED_CONTINUOUS_MEM_W") == 5, (
        "expected 5 unified operator constructions using UNIFIED_CONTINUOUS_MEM_W"
    )


def test_backend_uses_single_unified_xclbin_check():
    """AST check: backend calls the single-xclbin guard exactly three times."""
    backend_py = Path(backend_mod.__file__)
    tree = ast.parse(backend_py.read_text(encoding="utf-8"))
    calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == "_require_single_unified_xclbin":
                calls += 1
    assert calls == 3, f"expected 3 _require_single_unified_xclbin calls, found {calls}"


def test_unified_continuous_mem_w_constant_is_true():
    """The engine-wide switch must be True so decode uses one contw xclbin."""
    assert backend_mod.UNIFIED_CONTINUOUS_MEM_W is True


def test_unified_sequence_plan_uses_shape_allowlist_and_global_rollback(monkeypatch):
    from kernels.gemv_unified.shapes import ALL_SHAPES

    assert backend_mod.UNIFIED_SEQ_PLAN == "per_call"
    assert backend_mod.unified_sequence_plan("f_b") == "per_call"
    assert backend_mod.unified_sequence_plan("q_b") == "per_call"
    assert backend_mod.unified_sequence_plan("moe_b2_9") == "per_call"
    assert backend_mod.unified_sequence_plan("rda_in") == "per_round"
    assert backend_mod.unified_sequence_plan("shared_gu") == "per_round"
    assert backend_mod.unified_sequence_plan("lm_head") == "per_round"
    assert backend_mod.unified_sequence_plan("dense_down") == "per_round"
    monkeypatch.setattr(backend_mod, "UNIFIED_SEQ_PLAN", "per_round")
    assert all(
        backend_mod.unified_sequence_plan(shape.name) == "per_round"
        for shape in ALL_SHAPES
    )


def test_unified_hardware_init_prepares_unified_rda(monkeypatch):
    """unified engine _init_hardware_operators must prepare unified RDA, not legacy rda_step."""
    fake_iron = types.SimpleNamespace(
        zeros=lambda shape, **kwargs: np.zeros(shape, dtype=np.float32),
    )
    fake_aie_utils = types.SimpleNamespace(
        full=lambda shape, val, **kw: np.full(shape, val),
        tensor=lambda arr: arr,
    )
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    monkeypatch.setattr(backend_mod, "iron", fake_iron)
    monkeypatch.setattr(backend_mod, "aie_utils", fake_aie_utils)

    backend = NpuBackend(gemv="unified", mock=True)
    backend.mock = False
    monkeypatch.setattr(backend, "_get_or_compile_moe_ops", lambda: None)

    unified_called = []
    legacy_called = []
    monkeypatch.setattr(backend, "_prepare_unified_rda_kernel", lambda: (unified_called.append(True), "unified_rda")[1])
    monkeypatch.setattr(backend, "_prepare_rda_kernel", lambda: (legacy_called.append(True), "legacy_rda")[1])

    backend._init_hardware_operators()
    assert len(unified_called) == 1, "Expected _prepare_unified_rda_kernel to be called for unified engine"
    assert len(legacy_called) == 0, "_prepare_rda_kernel should not be called for unified engine"
    assert backend._compiled_ops["rda_step"] == "unified_rda"


def test_unified_operators_configuration_parameters(monkeypatch):
    """Unified engine operators (dense GEMV, MoE GU/SH/DN, RDA) all receive cols=7, rda_col=7, continuous_mem_w=True."""
    assert backend_mod.UNIFIED_COLS == 7
    assert backend_mod.UNIFIED_RDA_COL == 7
    assert backend_mod.UNIFIED_CONTINUOUS_MEM_W is True

    backend = NpuBackend(gemv="unified", mock=False)
    backend.mock = False

    created_ops = []

    class MockUnifiedOp:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.name = kwargs.get("shape_name", "mock_op")
            self.cols = kwargs.get("cols")
            self.rda_col = kwargs.get("rda_col")
            self.continuous_mem_w = kwargs.get("continuous_mem_w")
            self.xclbin_artifact = type("Art", (), {"filename": "/tmp/mock.xclbin"})()
            created_ops.append(self)

        @classmethod
        def for_shape(cls, shape_name, **kwargs):
            return cls(shape_name=shape_name, **kwargs)

        def compile(self, *a, **kw):
            pass

        def get_callable(self):
            return lambda *a, **kw: None

    monkeypatch.setattr("kernels.gemv_unified.op.UnifiedGEMV", MockUnifiedOp)
    monkeypatch.setattr("kernels.gemv_unified.stage2.UnifiedMoEGateUp", MockUnifiedOp)
    monkeypatch.setattr("kernels.gemv_unified.stage2.UnifiedMoEDown9", MockUnifiedOp)
    monkeypatch.setattr("kernels.gemv_unified.rda.UnifiedRDA", MockUnifiedOp)
    monkeypatch.setattr(backend, "_prepare_dynamic_kernel", lambda k: k)

    # 1. GEMV (dense_gu: K=2560, M=24064, fmt="q4nx_gemv")
    backend._get_or_compile_gemv(2560, 24064, "q4nx_gemv")
    # 2. MoE ops
    backend._get_or_compile_moe_ops()
    # 3. RDA
    backend._prepare_unified_rda_kernel()

    # Verify all created ops got cols=7, rda_col=7, continuous_mem_w=True
    assert len(created_ops) == 5  # 1 gemv + 3 moe (gu, sh, dn) + 1 rda
    for op in created_ops:
        assert op.cols == 7, f"{op.name} cols={op.cols} != 7"
        assert op.rda_col == 7, f"{op.name} rda_col={op.rda_col} != 7"
        assert op.continuous_mem_w is True, f"{op.name} continuous_mem_w is not True"


def test_unified_rda_update_preserves_ping_pong_order(monkeypatch):
    """rda_update calls the prepared callable with (s_in, s_out, head_in, out) ping-pong order."""
    class MockTensor:
        def __init__(self, shape):
            self.shape = shape
            self._arr = np.zeros(shape, dtype=np.float32)

        def __setitem__(self, idx, val):
            self._arr[idx] = val

        def numpy(self):
            return self._arr

    fake_iron = types.SimpleNamespace(
        zeros=lambda shape, **kwargs: MockTensor(shape),
    )
    fake_module = types.ModuleType("bench.rda_state.rda_state")
    fake_module.pack_head_bundle = lambda q, k, a, v, beta: np.zeros(
        (40, 640), dtype=np.float32
    )
    monkeypatch.setitem(sys.modules, "bench.rda_state.rda_state", fake_module)
    monkeypatch.setattr(backend_mod, "HAVE_NPU", True)
    monkeypatch.setattr(backend_mod, "iron", fake_iron)

    backend = NpuBackend(gemv="unified", mock=True)
    backend.mock = False
    backend._work_buffers["rda_head_in"] = MockTensor((40, 640))
    backend._work_buffers["rda_out"] = MockTensor((40, 128))

    calls = []
    backend._compiled_ops["rda_step"] = lambda *args: calls.append(args)

    q = np.zeros((40, 128), dtype=np.float32)
    beta = np.zeros(40, dtype=np.float32)

    backend.rda_update(0, q, q, q, q, beta)
    backend.rda_update(0, q, q, q, q, beta)

    s0, s1 = backend.rda_states[0]
    head_in = backend._work_buffers["rda_head_in"]
    out_buf = backend._work_buffers["rda_out"]

    assert len(calls) == 2
    # Call 0: s_in is s0, s_out is s1
    assert calls[0] == (s0, s1, head_in, out_buf)
    # Call 1: s_in is s1, s_out is s0
    assert calls[1] == (s1, s0, head_in, out_buf)
    assert backend.rda_active_slot[0] == 0


def test_unified_rda_mismatched_xclbin_raises_runtime_error(monkeypatch):
    """If the RDA operator resolves to a different xclbin path than expected, RuntimeError is raised."""
    backend = NpuBackend(gemv="unified", mock=False)
    backend.mock = False
    backend._unified_xclbin_path = Path("/tmp/moe_expected.xclbin").resolve()

    class MismatchedRDA:
        def __init__(self, *args, **kwargs):
            self.name = "mismatched_rda"
            self.xclbin_artifact = type("Art", (), {"filename": "/tmp/different_rda.xclbin"})()

        def compile(self, *a, **kw):
            pass

        def get_callable(self):
            return lambda *a, **kw: None

    monkeypatch.setattr("kernels.gemv_unified.rda.UnifiedRDA", MismatchedRDA)

    with pytest.raises(RuntimeError, match="unified xclbin path mismatch"):
        backend._prepare_unified_rda_kernel()


def test_unified_vs_legacy_rda_xclbin_recording_and_switches():
    """Unified engine records 'gemv_unified' for RDA (0 switches), legacy records 'rda_step'."""
    # Unified engine
    backend_u = NpuBackend(gemv="unified", mock=True)
    backend_u.reset_timings()
    q = np.zeros((40, 128), dtype=np.float32)
    beta = np.zeros(40, dtype=np.float32)

    # First simulate a GEMV call
    backend_u._record_xclbin_use("gemv_unified")
    assert backend_u.xclbin_switches == 0

    # Then call rda_update on unified
    backend_u.rda_update(0, q, q, q, q, beta)
    assert backend_u.xclbin_calls.get("gemv_unified") == 2
    assert "rda_step" not in backend_u.xclbin_calls
    assert backend_u.xclbin_switches == 0

    # Legacy engine
    backend_l = NpuBackend(gemv="legacy", mock=True)
    backend_l.reset_timings()

    # First simulate a GEMV call
    backend_l._record_xclbin_use("gemv_q4nx_k2560_m20736")
    assert backend_l.xclbin_switches == 0

    # Then call rda_update on legacy
    backend_l.rda_update(0, q, q, q, q, beta)
    assert backend_l.xclbin_calls.get("rda_step") == 1
    assert backend_l.xclbin_switches == 1  # Switched from gemv to rda_step


def test_moe_seven_slot_host_batch_matches_legacy_loop_and_order_is_sensitive():
    """The batched host activation/conversion is bitwise equal to seven calls."""
    rng = np.random.default_rng(604)
    gate_up = rng.integers(0, 65536, size=(7, 34), dtype=np.uint16)
    gate_up[0, :8] = np.array(
        [0x0000, 0x8000, 0x0001, 0x007F, 0x3F80, 0x7F80, 0xFF80, 0x7FC1],
        dtype=np.uint16,
    )
    with np.errstate(all="ignore"):
        legacy = np.stack(
            [
                f32_to_bf16_rtne(situ_glu(row[:17], row[17:]))
                for row in gate_up
            ]
        )
        batched = f32_to_bf16_rtne(situ_glu(gate_up[:, :17], gate_up[:, 17:]))
    assert np.array_equal(legacy, batched)

    dn = rng.integers(0, 65536, size=(7, 127), dtype=np.uint16)
    weights = rng.standard_normal(6).astype(np.float32)
    ids = np.array([91, 2, 77, 8, 45, 16])
    order = sorted(range(6), key=lambda j: int(ids[j]))
    dn_f32 = bf16_to_f32(dn)
    expected = np.zeros(127, dtype=np.float32)
    actual = np.zeros(127, dtype=np.float32)
    for j in order:
        expected += bf16_to_f32(dn[j]) * float(weights[j])
        actual += dn_f32[j] * float(weights[j])
    assert np.array_equal(expected.view(np.uint32), actual.view(np.uint32))

    reversed_sum = np.zeros(127, dtype=np.float32)
    for j in reversed(order):
        reversed_sum += dn_f32[j] * float(weights[j])
    assert np.count_nonzero(expected.view(np.uint32) != reversed_sum.view(np.uint32)) > 0


def test_npu_mla_f32_cache_reset_and_restore():
    backend = NpuBackend(mock=True)
    rng = np.random.default_rng(605)
    key = rng.integers(0, 65536, size=(2, 5, 7), dtype=np.uint16)
    value = rng.integers(0, 65536, size=(2, 5, 3), dtype=np.uint16)
    backend.set_mla_cache(1, key[np.newaxis], value[np.newaxis])
    cache1 = backend._mla_caches[1]
    assert np.array_equal(
        cache1.k_f32[:, :cache1.length, :].view(np.uint32), bf16_to_f32(key).view(np.uint32)
    )
    backend.reset()
    assert backend.mla_key_cache == {}
    assert backend.mla_value_cache == {}
    assert backend._mla_caches == {}


# =============================================================================
# npu-opt6 fix1: NPU MLA preallocated growing cache tests
# =============================================================================

class _LegacyMlaReferenceNpu:
    """af44167 legacy MLA reference for NPU backend tests."""

    def __init__(self):
        self.k: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def reset(self) -> None:
        self.k = None
        self.v = None

    def set_cache(self, k: np.ndarray, v: np.ndarray) -> None:
        k_s = k.squeeze(0) if k.ndim == 4 else k
        v_s = v.squeeze(0) if v.ndim == 4 else v
        self.k = np.ascontiguousarray(k_s, dtype=np.uint16)
        self.v = np.ascontiguousarray(v_s, dtype=np.uint16)

    def attend(
        self,
        q: np.ndarray,
        k_cur: np.ndarray,
        v_cur: np.ndarray,
        scale: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        k_cur_u16 = f32_to_bf16_rtne(k_cur) if k_cur.dtype == np.float32 else k_cur
        v_cur_u16 = f32_to_bf16_rtne(v_cur) if v_cur.dtype == np.float32 else v_cur
        if k_cur_u16.ndim == 3:
            k_cur_u16 = k_cur_u16.squeeze(1)
        if v_cur_u16.ndim == 3:
            v_cur_u16 = v_cur_u16.squeeze(1)

        k_token = k_cur_u16[:, np.newaxis, :]
        v_token = v_cur_u16[:, np.newaxis, :]

        if self.k is None or self.v is None:
            self.k = k_token
            self.v = v_token
        else:
            self.k = np.concatenate([self.k, k_token], axis=1)
            self.v = np.concatenate([self.v, v_token], axis=1)

        q_f32 = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
        k_f32 = bf16_to_f32(self.k)
        v_f32 = bf16_to_f32(self.v)

        scores = np.einsum("hd,htd->ht", q_f32, k_f32) * scale
        max_s = np.max(scores, axis=-1, keepdims=True)
        exp_s = np.exp(scores - max_s)
        attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        context_f32 = np.einsum("ht,htd->hd", attn, v_f32)
        flat_context = context_f32.reshape(-1)
        out = f32_to_bf16_rtne(flat_context)
        return out, self.k, self.v, k_f32, v_f32


def test_npu_mla_prealloc_bit_identical_across_growth_boundaries():
    """Verify NpuBackend preallocated MLA cache matches legacy across growth boundaries."""
    backend = NpuBackend(mock=True)
    legacy = _LegacyMlaReferenceNpu()
    rng = np.random.default_rng(801)

    heads, dk, dv = 4, 16, 12
    scale = 0.25
    layer = 2

    special_vals = np.array([
        0x0000, 0x8000, 0x0001, 0x007F, 0x8001,
        0x7F80, 0xFF80, 0x7FC0, 0x7FC1, 0xFFC0,
    ], dtype=np.uint16)

    saved_snap_250 = None
    saved_snap_250_data = None
    saved_snap_510 = None
    saved_snap_510_data = None

    total_steps = 540

    for step in range(total_steps):
        q = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        k = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        v = rng.integers(0, 65536, size=(heads, dv), dtype=np.uint16)

        inj_idx = step % len(special_vals)
        q[0, 0] = special_vals[inj_idx]
        k[0, 0] = special_vals[(inj_idx + 1) % len(special_vals)]
        v[0, 0] = special_vals[(inj_idx + 2) % len(special_vals)]

        with np.errstate(all="ignore"):
            out_act = backend.mla_attend(layer, q, k, v, scale)
            out_ref, ref_k, ref_v, ref_kf, ref_vf = legacy.attend(q, k, v, scale)

        assert np.array_equal(out_act.view(np.uint16), out_ref.view(np.uint16)), f"Output mismatch at step {step}"
        assert np.array_equal(backend.mla_key_cache[layer], ref_k), f"Key cache mismatch at step {step}"
        assert np.array_equal(backend.mla_value_cache[layer], ref_v), f"Value cache mismatch at step {step}"

        layer_cache = backend._mla_caches[layer]
        act_kf = layer_cache.k_f32[:, :layer_cache.length, :]
        act_vf = layer_cache.v_f32[:, :layer_cache.length, :]
        assert np.array_equal(act_kf.view(np.uint32), ref_kf.view(np.uint32)), f"FP32 key mirror mismatch at step {step}"
        assert np.array_equal(act_vf.view(np.uint32), ref_vf.view(np.uint32)), f"FP32 value mirror mismatch at step {step}"

        if step == 255:
            assert layer_cache.capacity == 257
        elif step == 256:
            assert layer_cache.capacity == 257
        elif step == 257:
            assert layer_cache.capacity == 513
        elif step == 512:
            assert layer_cache.capacity == 513
        elif step == 513:
            assert layer_cache.capacity == 769

        if step == 249:
            k_g, v_g = backend.get_mla_cache(layer)
            assert k_g is not None and v_g is not None
            saved_snap_250 = (k_g, v_g)
            saved_snap_250_data = (k_g.copy(), v_g.copy())
        if step == 259:
            assert saved_snap_250 is not None and saved_snap_250_data is not None
            assert np.array_equal(saved_snap_250[0], saved_snap_250_data[0])
            assert np.array_equal(saved_snap_250[1], saved_snap_250_data[1])

        if step == 509:
            k_g, v_g = backend.get_mla_cache(layer)
            assert k_g is not None and v_g is not None
            saved_snap_510 = (k_g, v_g)
            saved_snap_510_data = (k_g.copy(), v_g.copy())
        if step == 519:
            assert saved_snap_510 is not None and saved_snap_510_data is not None
            assert np.array_equal(saved_snap_510[0], saved_snap_510_data[0])
            assert np.array_equal(saved_snap_510[1], saved_snap_510_data[1])


def test_npu_mla_prealloc_set_cache_reset_and_serve_restore_parity():
    """Verify NpuBackend set_mla_cache, reset, and serve restore parity with legacy."""
    backend = NpuBackend(mock=True)
    legacy = _LegacyMlaReferenceNpu()
    rng = np.random.default_rng(802)

    heads, dk, dv = 3, 14, 10
    scale = 0.2
    layer = 1

    init_t = 20
    k_init = rng.integers(0, 65536, size=(heads, init_t, dk), dtype=np.uint16)
    v_init = rng.integers(0, 65536, size=(heads, init_t, dv), dtype=np.uint16)
    backend.set_mla_cache(layer, k_init, v_init)
    legacy.set_cache(k_init, v_init)

    for step in range(15):
        q = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        k = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        v = rng.integers(0, 65536, size=(heads, dv), dtype=np.uint16)
        with np.errstate(all="ignore"):
            out_act = backend.mla_attend(layer, q, k, v, scale)
            out_ref, ref_k, ref_v, ref_kf, ref_vf = legacy.attend(q, k, v, scale)
        assert np.array_equal(out_act.view(np.uint16), out_ref.view(np.uint16))
        assert np.array_equal(backend.mla_key_cache[layer], ref_k)
        assert np.array_equal(backend.mla_value_cache[layer], ref_v)

    target_len = 25
    k_cache, v_cache = backend.get_mla_cache(layer)
    assert k_cache is not None and v_cache is not None
    if k_cache.ndim == 4:
        k_sliced = np.ascontiguousarray(k_cache[:, :, :target_len, :])
        v_sliced = np.ascontiguousarray(v_cache[:, :, :target_len, :])
    else:
        k_sliced = np.ascontiguousarray(k_cache[:, :target_len, :])
        v_sliced = np.ascontiguousarray(v_cache[:, :target_len, :])

    backend.set_mla_cache(layer, k_sliced, v_sliced)
    legacy.set_cache(k_sliced, v_sliced)
    assert backend._mla_caches[layer].length == target_len

    for step in range(20):
        q = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        k = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        v = rng.integers(0, 65536, size=(heads, dv), dtype=np.uint16)
        with np.errstate(all="ignore"):
            out_act = backend.mla_attend(layer, q, k, v, scale)
            out_ref, ref_k, ref_v, ref_kf, ref_vf = legacy.attend(q, k, v, scale)
        assert np.array_equal(out_act.view(np.uint16), out_ref.view(np.uint16))
        assert np.array_equal(backend.mla_key_cache[layer], ref_k)
        assert np.array_equal(backend.mla_value_cache[layer], ref_v)

    backend.reset()
    legacy.reset()
    assert backend.mla_key_cache == {}
    assert backend.mla_value_cache == {}
    assert backend._mla_caches == {}

    for step in range(10):
        q = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        k = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        v = rng.integers(0, 65536, size=(heads, dv), dtype=np.uint16)
        with np.errstate(all="ignore"):
            out_act = backend.mla_attend(layer, q, k, v, scale)
            out_ref, ref_k, ref_v, ref_kf, ref_vf = legacy.attend(q, k, v, scale)
        assert np.array_equal(out_act.view(np.uint16), out_ref.view(np.uint16))
        assert np.array_equal(backend.mla_key_cache[layer], ref_k)
        assert np.array_equal(backend.mla_value_cache[layer], ref_v)
