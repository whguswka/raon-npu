# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Layer-by-layer comparison tool between PyTorch reference and Host NumPy engine.

Verifies full-layer execution and intermediate steps against golden decode run (ko16, step 2):
1. Layer 0 (RDA + Dense MLP): PyTorch DecoderLayer vs Golden streams_out,
   Host intermediate steps (mixer_in, mixer_out, MLP aggregate, post RMSNorm, SiTU-GLU dense MLP, streams_out),
   and Host vs Golden streams_out.
2. Layer 1 (RDA + MoE MLP): PyTorch DecoderLayer vs Golden streams_out,
   Host intermediate steps (RDA, MoE router logits, top-6 indices, routed weights, experts, streams_out),
   and Host vs Golden streams_out.
3. Layer 3 (MLA + MoE MLP): PyTorch DecoderLayer vs Golden streams_out (with expanded MLA latent cache),
   Host intermediate steps (MLA attention, MoE routing, experts, streams_out),
   and Host vs Golden streams_out.
4. Layer 4 (RDA): PyTorch vs Golden mixer_out, and Host _forward_rda output vs Golden.

Outputs summary table to host/out/check_layer.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_DIR = Path(__file__).resolve().parent / "out"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT_DIR / ".torch_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(CACHE_DIR)
os.environ["TRITON_CACHE_DIR"] = str(CACHE_DIR)
os.environ["TMPDIR"] = str(CACHE_DIR)

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from safetensors import safe_open

import transformers
transformers.initialization = init
sys.modules["transformers.initialization"] = init

from host.backend import NumpyBackend
from host.raon import RaonConfig, RaonModel, rms_norm
from host.weights import Weights, bf16_to_f32, f32_to_bf16_rtne


def cos_sim(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu().numpy()
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()
    norm_a = np.linalg.norm(a_f)
    norm_b = np.linalg.norm(b_f)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if np.array_equal(a_f, b_f) else 0.0
    return float(np.dot(a_f, b_f) / (norm_a * norm_b))


def max_abs_diff(a: np.ndarray | torch.Tensor, b: np.ndarray | torch.Tensor) -> float:
    if isinstance(a, torch.Tensor):
        a = a.detach().float().cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.detach().float().cpu().numpy()
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()
    return float(np.max(np.abs(a_f - b_f)))


def load_hf_modules(hf_dir: Path):
    spec_cfg = importlib.util.spec_from_file_location(
        "raon_pkg.configuration_raon1_mini", hf_dir / "configuration_raon1_mini.py"
    )
    mod_cfg = importlib.util.module_from_spec(spec_cfg)
    sys.modules["raon_pkg"] = type(sys)("raon_pkg")
    sys.modules["raon_pkg.configuration_raon1_mini"] = mod_cfg
    spec_cfg.loader.exec_module(mod_cfg)

    spec_model = importlib.util.spec_from_file_location(
        "raon_pkg.modeling_raon1_mini", hf_dir / "modeling_raon1_mini.py"
    )
    mod_model = importlib.util.module_from_spec(spec_model)
    sys.modules["raon_pkg.modeling_raon1_mini"] = mod_model
    spec_model.loader.exec_module(mod_model)

    return mod_cfg.Raon1MiniConfig, mod_model


def get_layer_state_dict(model_dir: Path, layer_idx: int, prefix: str) -> dict[str, torch.Tensor]:
    index_path = model_dir / "model.safetensors.index.json"
    with open(index_path, encoding="utf-8") as f:
        idx_json = json.load(f)
    weight_map = idx_json["weight_map"]
    needed_shards = {shard for k, shard in weight_map.items() if k.startswith(prefix)}

    state_dict = {}
    for shard in needed_shards:
        with safe_open(str(model_dir / shard), framework="pt") as f:
            for k in f.keys():
                if k.startswith(prefix):
                    state_dict[k[len(prefix):]] = f.get_tensor(k)
    return state_dict


def get_clean_layer_state_dict(model_dir: Path, layer_idx: int, prefix: str, dequantize_fn) -> dict[str, torch.Tensor]:
    """Retrieve layer state dict and dequantize MXFP8 expert weights if present."""
    raw_dict = get_layer_state_dict(model_dir, layer_idx, prefix)
    clean_dict = {}
    for k, v in raw_dict.items():
        if k.endswith(".weight_scale_inv"):
            continue
        scale_key = k[:-6] + "weight_scale_inv" if k.endswith("weight") else None
        if scale_key is not None and scale_key in raw_dict:
            clean_dict[k] = dequantize_fn(v, raw_dict[scale_key], torch.bfloat16)
        else:
            clean_dict[k] = v
    return clean_dict


def summary_exit_code(summary_rows: list[dict]) -> int:
    """Gate exit code: 0 only when every layer row passed, 1 when any row FAILED."""
    return 0 if all(row["pass"] for row in summary_rows) else 1


def run_all_checks(model_dir: Path, golden_dir: Path) -> int:
    Raon1MiniConfig, mod_model = load_hf_modules(model_dir)
    config = Raon1MiniConfig.from_pretrained(str(model_dir))
    plain_payload = config.to_dict()
    plain_payload.pop("quantization_config", None)
    config_plain = Raon1MiniConfig.from_dict(plain_payload)

    host_config = RaonConfig.from_dict(config.to_dict())

    g1 = np.load(golden_dir / "step_000001.npz")
    g2 = np.load(golden_dir / "step_000002.npz")

    summary_rows = []

    # =========================================================================
    # 1. LAYER 0 (RDA + Dense MLP)
    # =========================================================================
    print("=" * 80)
    print("CHECK 1: Layer 0 (RDA + Dense MLP)")
    print("=" * 80)

    weights_0 = Weights(model_dir, selected_layers=[0], load_global=False)
    backend_0 = NumpyBackend(weights_0)
    model_0 = RaonModel(host_config, weights_0, backend=backend_0)
    model_0.load_snapshot(golden_dir / "step_000001.npz")

    # PyTorch layer 0
    layer_0_pt = mod_model.Raon1MiniDecoderLayer(config_plain, 0).bfloat16()
    l0_dict = get_clean_layer_state_dict(model_dir, 0, "model.layers.0.", mod_model.dequantize_mxfp8)
    layer_0_pt.load_state_dict(l0_dict)

    pt_cache_0 = mod_model.Raon1MiniDynamicCache(config_plain)
    pt_cache_0.recurrent_states[0] = torch.from_numpy(g1["layer_000.rda_state"].copy()).float()
    pt_cache_0.conv_states[0] = torch.from_numpy(g1["layer_000.rda_conv_state"].copy()).view(torch.bfloat16)

    s_in_0_u16 = g2["layer_000.streams_in"].squeeze(0).squeeze(0)  # [4, 2560]
    s_in_0_pt = torch.from_numpy(g2["layer_000.streams_in"].copy()).view(torch.bfloat16)
    m_out_golden_0_u16 = g2["layer_000.mixer_out"].copy()
    s_out_golden_0_u16 = g2["layer_000.streams_out"].squeeze(0).squeeze(0)
    s_out_golden_0_pt = torch.from_numpy(g2["layer_000.streams_out"].copy()).view(torch.bfloat16)

    with torch.no_grad():
        pt_sout_0 = layer_0_pt(s_in_0_pt, attention_mask=None, token_mask=None, cache=pt_cache_0)

    cos_pt_golden_sout_0 = cos_sim(pt_sout_0, s_out_golden_0_pt)
    print(f"PyTorch vs Golden streams_out: cos={cos_pt_golden_sout_0:.8f}")

    # Host layer 0 step-by-step
    stream_0_h = s_in_0_u16.copy()
    attn_mhc_pfx_0 = "model.layers.0.self_attn_raon_mhc"
    h_pre_0, h_post_0, h_res_0 = model_0._route_mhc(stream_0_h, attn_mhc_pfx_0)
    agg_attn_0 = model_0._mhc_aggregate(stream_0_h, h_pre_0)
    in_norm_w_0 = weights_0.get("model.layers.0.input_layernorm.weight", to_float=True)
    branch_in_attn_0 = rms_norm(agg_attn_0, in_norm_w_0, eps=host_config.rms_norm_eps)

    # Reset snapshot before forward to ensure clean state
    model_0.load_snapshot(golden_dir / "step_000001.npz")
    host_mout_0_u16 = model_0._forward_rda(0, branch_in_attn_0)
    cos_host_golden_mout_0 = cos_sim(bf16_to_f32(host_mout_0_u16), bf16_to_f32(m_out_golden_0_u16))
    max_diff_mout_0 = max_abs_diff(bf16_to_f32(host_mout_0_u16), bf16_to_f32(m_out_golden_0_u16))
    print(f"Host vs Golden mixer_out:     cos={cos_host_golden_mout_0:.8f}, max_diff={max_diff_mout_0:.6f}")

    stream_mid_0 = model_0._mhc_mix(stream_0_h, host_mout_0_u16, h_post_0, h_res_0)

    # MLP site
    mlp_mhc_pfx_0 = "model.layers.0.mlp_raon_mhc"
    h_pre_mlp_0, h_post_mlp_0, h_res_mlp_0 = model_0._route_mhc(stream_mid_0, mlp_mhc_pfx_0)
    agg_mlp_0 = model_0._mhc_aggregate(stream_mid_0, h_pre_mlp_0)
    post_norm_w_0 = weights_0.get("model.layers.0.post_attention_layernorm.weight", to_float=True)
    branch_in_mlp_0 = rms_norm(agg_mlp_0, post_norm_w_0, eps=host_config.rms_norm_eps)
    branch_out_mlp_0 = model_0._forward_dense_mlp(0, branch_in_mlp_0)
    stream_out_0_h = model_0._mhc_mix(stream_mid_0, branch_out_mlp_0, h_post_mlp_0, h_res_mlp_0)

    cos_host_golden_sout_0 = cos_sim(bf16_to_f32(stream_out_0_h), bf16_to_f32(s_out_golden_0_u16))
    max_diff_sout_0 = max_abs_diff(bf16_to_f32(stream_out_0_h), bf16_to_f32(s_out_golden_0_u16))
    print(f"Host vs Golden streams_out:   cos={cos_host_golden_sout_0:.8f}, max_diff={max_diff_sout_0:.6f}")

    summary_rows.append({
        "layer": "Layer 0 (RDA + Dense MLP)",
        "target": "streams_out",
        "pytorch_vs_golden": f"{cos_pt_golden_sout_0:.6f}",
        "host_vs_golden": f"{cos_host_golden_sout_0:.6f}",
        "max_diff": f"{max_diff_sout_0:.6f}",
        "pass": cos_host_golden_sout_0 >= 0.9999,
    })

    # =========================================================================
    # 2. LAYER 1 (RDA + MoE MLP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CHECK 2: Layer 1 (RDA + MoE MLP)")
    print("=" * 80)

    weights_1 = Weights(model_dir, selected_layers=[1], load_global=False)
    backend_1 = NumpyBackend(weights_1)
    model_1 = RaonModel(host_config, weights_1, backend=backend_1)
    model_1.load_snapshot(golden_dir / "step_000001.npz")

    layer_1_pt = mod_model.Raon1MiniDecoderLayer(config_plain, 1).bfloat16()
    l1_dict = get_clean_layer_state_dict(model_dir, 1, "model.layers.1.", mod_model.dequantize_mxfp8)
    layer_1_pt.load_state_dict(l1_dict)

    pt_cache_1 = mod_model.Raon1MiniDynamicCache(config_plain)
    pt_cache_1.recurrent_states[1] = torch.from_numpy(g1["layer_001.rda_state"].copy()).float()
    pt_cache_1.conv_states[1] = torch.from_numpy(g1["layer_001.rda_conv_state"].copy()).view(torch.bfloat16)

    s_in_1_u16 = g2["layer_001.streams_in"].squeeze(0).squeeze(0)
    s_in_1_pt = torch.from_numpy(g2["layer_001.streams_in"].copy()).view(torch.bfloat16)
    s_out_golden_1_pt = torch.from_numpy(g2["layer_001.streams_out"].copy()).view(torch.bfloat16)
    s_out_golden_1_u16 = g2["layer_001.streams_out"].squeeze(0).squeeze(0)

    with torch.no_grad():
        pt_sout_1 = layer_1_pt(s_in_1_pt, attention_mask=None, token_mask=None, cache=pt_cache_1)

    cos_pt_golden_sout_1 = cos_sim(pt_sout_1, s_out_golden_1_pt)
    print(f"PyTorch vs Golden streams_out: cos={cos_pt_golden_sout_1:.8f}")

    # Host layer 1 step-by-step
    stream_1_h = s_in_1_u16.copy()
    attn_mhc_pfx_1 = "model.layers.1.self_attn_raon_mhc"
    h_pre_1, h_post_1, h_res_1 = model_1._route_mhc(stream_1_h, attn_mhc_pfx_1)
    agg_attn_1 = model_1._mhc_aggregate(stream_1_h, h_pre_1)
    in_norm_w_1 = weights_1.get("model.layers.1.input_layernorm.weight", to_float=True)
    branch_in_attn_1 = rms_norm(agg_attn_1, in_norm_w_1, eps=host_config.rms_norm_eps)

    model_1.load_snapshot(golden_dir / "step_000001.npz")
    host_mout_1_u16 = model_1._forward_rda(1, branch_in_attn_1)
    stream_mid_1 = model_1._mhc_mix(stream_1_h, host_mout_1_u16, h_post_1, h_res_1)

    # MLP site
    mlp_mhc_pfx_1 = "model.layers.1.mlp_raon_mhc"
    h_pre_mlp_1, h_post_mlp_1, h_res_mlp_1 = model_1._route_mhc(stream_mid_1, mlp_mhc_pfx_1)
    agg_mlp_1 = model_1._mhc_aggregate(stream_mid_1, h_pre_mlp_1)
    post_norm_w_1 = weights_1.get("model.layers.1.post_attention_layernorm.weight", to_float=True)
    branch_in_mlp_1 = rms_norm(agg_mlp_1, post_norm_w_1, eps=host_config.rms_norm_eps)
    branch_out_mlp_1 = model_1._forward_moe(1, branch_in_mlp_1, "layer_001")
    stream_out_1_h = model_1._mhc_mix(stream_mid_1, branch_out_mlp_1, h_post_mlp_1, h_res_mlp_1)

    cos_host_golden_sout_1 = cos_sim(bf16_to_f32(stream_out_1_h), bf16_to_f32(s_out_golden_1_u16))
    max_diff_sout_1 = max_abs_diff(bf16_to_f32(stream_out_1_h), bf16_to_f32(s_out_golden_1_u16))
    print(f"Host vs Golden streams_out:   cos={cos_host_golden_sout_1:.8f}, max_diff={max_diff_sout_1:.6f}")

    # Router metrics
    golden_top_1 = g2["layer_001.router_top"].ravel()
    host_top_1 = model_1.last_intermediates["layer_001.router_top"].ravel()
    top_set_match_1 = len(set(golden_top_1) & set(host_top_1)) / float(len(golden_top_1)) * 100.0
    r_wt_diff_1 = max_abs_diff(g2["layer_001.router_weights"].ravel(), model_1.last_intermediates["layer_001.router_weights"].ravel())
    print(f"Host Router top-6 set match:  {top_set_match_1:.1f}%")
    print(f"Router weights max diff:      {r_wt_diff_1:.6f}")

    summary_rows.append({
        "layer": "Layer 1 (RDA + MoE MLP)",
        "target": "streams_out",
        "pytorch_vs_golden": f"{cos_pt_golden_sout_1:.6f}",
        "host_vs_golden": f"{cos_host_golden_sout_1:.6f}",
        "max_diff": f"{max_diff_sout_1:.6f}",
        "pass": cos_host_golden_sout_1 >= 0.9999,
    })

    # =========================================================================
    # 3. LAYER 3 (MLA + MoE MLP)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CHECK 3: Layer 3 (MLA + MoE MLP)")
    print("=" * 80)

    weights_3 = Weights(model_dir, selected_layers=[3], load_global=False)
    backend_3 = NumpyBackend(weights_3)
    model_3 = RaonModel(host_config, weights_3, backend=backend_3)
    model_3.load_snapshot(golden_dir / "step_000001.npz")

    layer_3_pt = mod_model.Raon1MiniDecoderLayer(config_plain, 3).bfloat16()
    l3_dict = get_clean_layer_state_dict(model_dir, 3, "model.layers.3.", mod_model.dequantize_mxfp8)
    layer_3_pt.load_state_dict(l3_dict)

    # PyTorch MLA cache expansion from snapshot latent
    kv_lat_raw = g1["layer_003.mla_kv_lat_cache"]
    k_pe_raw = g1["layer_003.mla_k_pe_cache"]
    kv_lat_pt = torch.from_numpy(kv_lat_raw).view(torch.bfloat16)
    k_pe_pt = torch.from_numpy(k_pe_raw).view(torch.bfloat16)
    with torch.no_grad():
        kv_pt = layer_3_pt.self_attn.kv_b_proj(kv_lat_pt).reshape(1, 20, config.num_attention_heads, config.qk_nope_head_dim + config.v_head_dim)
        k_nope_pt, val_pt = torch.split(kv_pt, [config.qk_nope_head_dim, config.v_head_dim], dim=-1)
        k_pe_exp_pt = k_pe_pt.unsqueeze(2).expand(1, 20, config.num_attention_heads, config.qk_rope_head_dim)
        key_pt = torch.cat([k_nope_pt, k_pe_exp_pt], dim=-1).transpose(1, 2)
        val_pt = val_pt.transpose(1, 2)

    pt_cache_3 = mod_model.Raon1MiniDynamicCache(config_plain)
    pt_cache_3.key_cache[3] = key_pt
    pt_cache_3.value_cache[3] = val_pt

    s_in_3_u16 = g2["layer_003.streams_in"].squeeze(0).squeeze(0)
    s_in_3_pt = torch.from_numpy(g2["layer_003.streams_in"].copy()).view(torch.bfloat16)
    s_out_golden_3_pt = torch.from_numpy(g2["layer_003.streams_out"].copy()).view(torch.bfloat16)
    s_out_golden_3_u16 = g2["layer_003.streams_out"].squeeze(0).squeeze(0)
    m_out_golden_3_u16 = g2["layer_003.mixer_out"].copy()

    with torch.no_grad():
        pt_sout_3 = layer_3_pt(s_in_3_pt, attention_mask=None, token_mask=None, cache=pt_cache_3)

    cos_pt_golden_sout_3 = cos_sim(pt_sout_3, s_out_golden_3_pt)
    print(f"PyTorch vs Golden streams_out: cos={cos_pt_golden_sout_3:.8f}")

    # Host layer 3 step-by-step
    stream_3_h = s_in_3_u16.copy()
    attn_mhc_pfx_3 = "model.layers.3.self_attn_raon_mhc"
    h_pre_3, h_post_3, h_res_3 = model_3._route_mhc(stream_3_h, attn_mhc_pfx_3)
    agg_attn_3 = model_3._mhc_aggregate(stream_3_h, h_pre_3)
    in_norm_w_3 = weights_3.get("model.layers.3.input_layernorm.weight", to_float=True)
    branch_in_attn_3 = rms_norm(agg_attn_3, in_norm_w_3, eps=host_config.rms_norm_eps)

    model_3.load_snapshot(golden_dir / "step_000001.npz")
    host_mout_3_u16 = model_3._forward_mla(3, branch_in_attn_3)
    cos_host_golden_mout_3 = cos_sim(bf16_to_f32(host_mout_3_u16), bf16_to_f32(m_out_golden_3_u16))
    max_diff_mout_3 = max_abs_diff(bf16_to_f32(host_mout_3_u16), bf16_to_f32(m_out_golden_3_u16))
    print(f"Host vs Golden mixer_out:     cos={cos_host_golden_mout_3:.8f}, max_diff={max_diff_mout_3:.6f}")

    stream_mid_3 = model_3._mhc_mix(stream_3_h, host_mout_3_u16, h_post_3, h_res_3)

    # MLP site
    mlp_mhc_pfx_3 = "model.layers.3.mlp_raon_mhc"
    h_pre_mlp_3, h_post_mlp_3, h_res_mlp_3 = model_3._route_mhc(stream_mid_3, mlp_mhc_pfx_3)
    agg_mlp_3 = model_3._mhc_aggregate(stream_mid_3, h_pre_mlp_3)
    post_norm_w_3 = weights_3.get("model.layers.3.post_attention_layernorm.weight", to_float=True)
    branch_in_mlp_3 = rms_norm(agg_mlp_3, post_norm_w_3, eps=host_config.rms_norm_eps)
    branch_out_mlp_3 = model_3._forward_moe(3, branch_in_mlp_3, "layer_003")
    stream_out_3_h = model_3._mhc_mix(stream_mid_3, branch_out_mlp_3, h_post_mlp_3, h_res_mlp_3)

    cos_host_golden_sout_3 = cos_sim(bf16_to_f32(stream_out_3_h), bf16_to_f32(s_out_golden_3_u16))
    max_diff_sout_3 = max_abs_diff(bf16_to_f32(stream_out_3_h), bf16_to_f32(s_out_golden_3_u16))
    print(f"Host vs Golden streams_out:   cos={cos_host_golden_sout_3:.8f}, max_diff={max_diff_sout_3:.6f}")

    golden_top_3 = g2["layer_003.router_top"].ravel()
    host_top_3 = model_3.last_intermediates["layer_003.router_top"].ravel()
    top_set_match_3 = len(set(golden_top_3) & set(host_top_3)) / float(len(golden_top_3)) * 100.0
    r_wt_diff_3 = max_abs_diff(g2["layer_003.router_weights"].ravel(), model_3.last_intermediates["layer_003.router_weights"].ravel())
    print(f"Host Router top-6 set match:  {top_set_match_3:.1f}%")
    print(f"Router weights max diff:      {r_wt_diff_3:.6f}")

    summary_rows.append({
        "layer": "Layer 3 (MLA + MoE MLP)",
        "target": "streams_out",
        "pytorch_vs_golden": f"{cos_pt_golden_sout_3:.6f}",
        "host_vs_golden": f"{cos_host_golden_sout_3:.6f}",
        "max_diff": f"{max_diff_sout_3:.6f}",
        "pass": cos_host_golden_sout_3 >= 0.9999,
    })

    # =========================================================================
    # 4. LAYER 4 (RDA)
    # =========================================================================
    print("\n" + "=" * 80)
    print("CHECK 4: Layer 4 (RDA)")
    print("=" * 80)

    weights_4 = Weights(model_dir, selected_layers=[4], load_global=False, skip_moe_experts=True)
    backend_4 = NumpyBackend(weights_4)
    model_4 = RaonModel(host_config, weights_4, backend=backend_4)
    model_4.load_snapshot(golden_dir / "step_000001.npz")

    # PyTorch layer 4
    rda_4 = mod_model.Raon1MiniRDA(config).bfloat16()
    rda_4_dict = get_layer_state_dict(model_dir, 4, "model.layers.4.self_attn.")
    rda_4.load_state_dict(rda_4_dict)

    pt_cache_4 = mod_model.Raon1MiniDynamicCache(config)
    pt_cache_4.recurrent_states[4] = torch.from_numpy(g1["layer_004.rda_state"].copy()).float()
    pt_cache_4.conv_states[4] = torch.from_numpy(g1["layer_004.rda_conv_state"].copy()).view(torch.bfloat16)

    m4_in_u16 = g2["layer_004.mixer_in"].copy()
    m4_in_pt = torch.from_numpy(m4_in_u16).view(torch.bfloat16)
    m4_out_golden_u16 = g2["layer_004.mixer_out"].copy()
    m4_out_golden_pt = torch.from_numpy(m4_out_golden_u16).view(torch.bfloat16)

    with torch.no_grad():
        pt_out_4 = rda_4(m4_in_pt, pt_cache_4, layer_idx=4)

    cos_pt_golden_4 = cos_sim(pt_out_4, m4_out_golden_pt)
    print(f"PyTorch vs Golden mixer_out: cos={cos_pt_golden_4:.8f}")

    host_out_4_u16 = model_4._forward_rda(4, m4_in_u16.reshape(-1))
    cos_host_golden_4 = cos_sim(bf16_to_f32(host_out_4_u16), bf16_to_f32(m4_out_golden_u16))
    max_diff_4 = max_abs_diff(bf16_to_f32(host_out_4_u16), bf16_to_f32(m4_out_golden_u16))
    print(f"Host vs Golden mixer_out:    cos={cos_host_golden_4:.8f}, max_diff={max_diff_4:.6f}")

    summary_rows.append({
        "layer": "Layer 4 (RDA)",
        "target": "mixer_out",
        "pytorch_vs_golden": f"{cos_pt_golden_4:.6f}",
        "host_vs_golden": f"{cos_host_golden_4:.6f}",
        "max_diff": f"{max_diff_4:.6f}",
        "pass": cos_host_golden_4 >= 0.9999,
    })

    # =========================================================================
    # Write summary table to host/out/check_layer.md
    # =========================================================================
    md_content = [
        "# Layer-by-Layer Verification Report",
        "",
        "| Layer | Target Output | PyTorch vs Golden | Host vs Golden | Max Abs Diff | Pass (cos >= 0.9999) |",
        "|-------|---------------|-------------------|----------------|--------------|----------------------|",
    ]
    for row in summary_rows:
        pass_str = "PASS" if row["pass"] else "FAIL"
        md_content.append(
            f"| {row['layer']} | {row['target']} | {row['pytorch_vs_golden']} | {row['host_vs_golden']} | {row['max_diff']} | {pass_str} |"
        )
    md_content.append("")
    md_content.append("### Measured diagnostics")
    md_content.append(f"- Layer 0 Host mixer_out vs Golden: cos={cos_host_golden_mout_0:.8f}, max_diff={max_diff_mout_0:.6f}")
    md_content.append(f"- Layer 1 Host router: top-set={top_set_match_1:.1f}%, weight max_diff={r_wt_diff_1:.6f}")
    md_content.append(f"- Layer 3 Host mixer_out vs Golden: cos={cos_host_golden_mout_3:.8f}, max_diff={max_diff_mout_3:.6f}")
    md_content.append(f"- Layer 3 Host router: top-set={top_set_match_3:.1f}%, weight max_diff={r_wt_diff_3:.6f}")

    out_file = OUT_DIR / "check_layer.md"
    out_file.write_text("\n".join(md_content), encoding="utf-8")
    print("\n" + "=" * 80)
    print(f"Summary written to {out_file}")
    print("=" * 80)
    # Non-zero exit when any row FAILED, so this tool can gate automation.
    return summary_exit_code(summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=os.environ.get("RAON_MODEL_DIR"),
        help="Path to Raon HF model checkpoint (default: $RAON_MODEL_DIR; otherwise pass --model-dir)",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=os.environ.get("RAON_GOLDEN_DIR"),
        help="Path to golden run directory (default: $RAON_GOLDEN_DIR; otherwise pass --golden-dir)",
    )
    args = parser.parse_args()
    for label, env_var, directory in (
        ("model dir", "RAON_MODEL_DIR", args.model_dir),
        ("golden dir", "RAON_GOLDEN_DIR", args.golden_dir),
    ):
        if directory is None or not directory.is_dir():
            raise FileNotFoundError(
                f"{label} {directory} not found; pass --{'model' if label == 'model dir' else 'golden'}-dir"
                f" or set {env_var} to a local copy"
            )
    raise SystemExit(run_all_checks(args.model_dir, args.golden_dir))
