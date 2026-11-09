# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Unit test for host reference engine against tiny PyTorch Raon model.

No test here reads an untracked results file: rejudge logic is covered by the
synthetic-payload tests below, and the recorded-results rejudge test skips with
a reason when the git-ignored host/out/golden_check.json is absent. PyTorch
cross-checks read the model source from $RAON_SRC_DIR and skip with a reason
when it is not present, like bench/gemv's RAON_IRON_PATH and bench/moe's
if-present skips.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from host import check_golden as check_golden_module
from host import check_layer as check_layer_module
from host.backend import NumpyBackend, situ_glu
from host.check_golden import (
    SUPPORTED_START_STEP,
    evaluate_summary,
    print_evaluation,
    validate_generated_token_ids,
    validate_start_step,
    verify_initial_snapshot,
    verify_requested_step_files,
)
from host.raon import RaonConfig, RaonModel, rms_norm
from host.weights import (
    BF16_IMPL,
    Weights,
    bf16_to_f32,
    f32_to_bf16_rtne,
)

# The local PyTorch Raon model source lives outside this repository; like bench/gemv's
# RAON_IRON_PATH, it is taken from RAON_SRC_DIR and the PyTorch cross-check tests
# skip with this reason when it is absent (e.g. a clean checkout elsewhere).
RAON_SRC_DIR = Path(os.environ.get("RAON_SRC_DIR", "__RAON_SRC_DIR_NOT_SET__"))

# Recorded full-run results file (git-ignored); only the recorded-results rejudge
# test needs it, and it skips with a reason when absent.
GOLDEN_CHECK_JSON = Path(__file__).resolve().parent / "out" / "golden_check.json"


class _DictWeights:
    """Minimal Weights-like mapping over an explicit tensor dict."""

    def __init__(self, tensors: dict):
        self.tensors = dict(tensors)

    def get(self, name: str, to_float: bool = True) -> np.ndarray:
        arr = self.tensors[name]
        if arr.dtype == np.uint16:
            return bf16_to_f32(arr) if to_float else arr
        return arr.astype(np.float32) if to_float and arr.dtype != np.float32 else arr

    def __contains__(self, name: str) -> bool:
        return name in self.tensors


class _FakeWeights(Weights):
    """In-memory Weights subclass that uses Weights.get with its f32 cache."""

    def __init__(self, tensors: dict[str, np.ndarray]):
        self.tensors = {k: np.array(v, copy=True) for k, v in tensors.items()}
        self.selected_layers = None
        self.load_global = True
        self.skip_moe_experts = False
        self._f32_cache: dict[str, np.ndarray] = {}


@pytest.fixture(scope="module")
def tiny_raon_model(tmp_path_factory: pytest.TempPathFactory):
    """Create a 3-layer model (RDA+Dense, MLA+MoE, RDA+MoE) in PyTorch."""
    return _build_tiny_raon_model(tmp_path_factory.mktemp("tiny_raon_host_test"))


def _build_tiny_raon_model(tmp_path: Path, **config_overrides):
    """Build a tiny PyTorch Raon model + host config/weights with optional config overrides."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "golden"))
    required_files = (
        "configuration_raon1_mini.py",
        "modeling_raon1_mini.py",
    )
    if not all((RAON_SRC_DIR / name).is_file() for name in required_files):
        pytest.skip(
            f"PyTorch Raon source not found under {RAON_SRC_DIR}; set RAON_SRC_DIR to a "
            "checkout with configuration_raon1_mini.py/modeling_raon1_mini.py to run the "
            "PyTorch cross-check tests"
        )
    import dump

    configuration, modeling = dump.load_local_raon(RAON_SRC_DIR)

    torch.manual_seed(42)

    config_kwargs: dict = dict(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=16,
        num_experts=8,
        num_experts_per_token=2,
        moe_intermediate_size=32,
        num_shared_experts=1,
        first_k_dense_replace=1,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        rda_decay_rank=16,
        rda_config={
            "num_heads": 4,
            "head_dim": 16,
            "conv_kernel_size": 4,
            "full_rank_gate": True,
            "gate_lower_bound": -5.0,
            "rda_layers": [1, 3],  # 1-based: layers 0 and 2
            "mla_layers": [2],     # 1-based: layer 1
        },
        pad_token_id=0,
        eos_token_id=[2],
        max_position_embeddings=64,
    )
    config_kwargs.update(config_overrides)
    config = configuration.Raon1MiniConfig(**config_kwargs)

    model = modeling.Raon1MiniForCausalLM(config).to(torch.bfloat16).eval()
    model.save_pretrained(tmp_path, safe_serialization=True)

    with open(tmp_path / "config.json", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    host_cfg = RaonConfig.from_dict(cfg_dict)
    weights = Weights(tmp_path)

    return model, host_cfg, weights, tmp_path


@pytest.fixture(scope="module")
def tiny_situ_model(tmp_path_factory: pytest.TempPathFactory):
    """Tiny model with non-default SiTU betas for routed-expert activation checks."""
    return _build_tiny_raon_model(
        tmp_path_factory.mktemp("tiny_situ_host_test"),
        activation_situ_beta=2.0,
        activation_situ_linear_beta=9.0,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float32)
    b_f = b.ravel().astype(np.float32)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0:
        return 1.0 if np.array_equal(a_f, b_f) else 0.0
    return float(np.dot(a_f, b_f) / denom)


def test_tiny_model_step_by_step(tiny_raon_model):
    """Verify decode steps 1, 2, 3 between PyTorch model and NumPy Host engine."""
    pt_model, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    host_model = RaonModel(host_cfg, weights, backend=backend)

    token_sequence = [42, 161, 88]
    pt_past = None

    for step_idx, tok in enumerate(token_sequence):
        with torch.no_grad():
            inp = torch.tensor([[tok]], dtype=torch.long)
            out = pt_model(
                input_ids=inp,
                past_key_values=pt_past,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
            pt_past = out.past_key_values
            pt_logits = out.logits[0, -1, :].float().numpy()

        np_logits = host_model.step(tok)

        cos_sim = cosine_similarity(pt_logits, np_logits)
        assert cos_sim >= 0.9999, f"Step {step_idx+1} logits cosine {cos_sim:.6f} < 0.9999"

        pt_argmax = int(np.argmax(pt_logits))
        np_argmax = int(np.argmax(np_logits))
        assert pt_argmax == np_argmax, f"Step {step_idx+1} argmax mismatch: pt={pt_argmax}, np={np_argmax}"

        # Verify layer intermediate outputs
        for layer_idx in range(host_cfg.num_hidden_layers):
            pfx = f"layer_{layer_idx:03d}"
            # streams_in
            sin = host_model.last_intermediates.get(f"{pfx}.streams_in")
            assert sin is not None and sin.shape == (1, 1, 4, 64)
            # mixer_in, mixer_out
            min_arr = host_model.last_intermediates.get(f"{pfx}.mixer_in")
            mout_arr = host_model.last_intermediates.get(f"{pfx}.mixer_out")
            assert min_arr is not None and mout_arr is not None
            # streams_out
            sout = host_model.last_intermediates.get(f"{pfx}.streams_out")
            assert sout is not None and sout.shape == (1, 1, 4, 64)

            if host_cfg.is_moe_layer(layer_idx):
                rtop = host_model.last_intermediates.get(f"{pfx}.router_top")
                rweights = host_model.last_intermediates.get(f"{pfx}.router_weights")
                assert rtop is not None and rtop.shape == (1, 2)
                assert rweights is not None and rweights.shape == (1, 2)


def test_tiny_model_load_snapshot(tiny_raon_model, tmp_path):
    """Load a golden-format PyTorch cache snapshot and continue in Host."""
    pt_model, host_cfg, weights, _ = tiny_raon_model
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "golden"))
    import dump

    recorder = dump.CaptureRecorder(set(range(host_cfg.num_hidden_layers)))
    recorder.attach(pt_model)
    tok1 = 42
    with torch.no_grad():
        out1 = pt_model(
            input_ids=torch.tensor([[tok1]]),
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    pt_cache = out1.past_key_values

    snapshot_tensors: dict[str, torch.Tensor] = {}
    dump.state_records(
        pt_cache,
        recorder.mla_latent_cache,
        list(range(host_cfg.num_hidden_layers)),
        True,
        snapshot_tensors,
    )
    for layer_idx, conv_state in enumerate(pt_cache.conv_states):
        if conv_state is not None:
            snapshot_tensors[f"layer_{layer_idx:03d}.rda_conv_state"] = conv_state

    recorder.close()

    # Reuse the production golden writer: BF16 is stored as uint16 plus dtype metadata.
    snapshot_file = tmp_path / "step_000001.npz"
    dump.save_tensor_npz(snapshot_file, snapshot_tensors, dump.Budget(1 << 20))
    with np.load(snapshot_file, allow_pickle=False) as saved:
        dtypes = json.loads(str(saved["__dtypes_json__"].item()))
        assert dtypes["layer_000.rda_conv_state"] == "bfloat16"
        assert saved["layer_000.rda_conv_state"].dtype == np.uint16
        assert "layer_001.mla_kv_lat_cache" in saved
        assert "layer_001.mla_k_pe_cache" in saved
        # The pre-run initial-snapshot completeness check accepts this golden-format snapshot.
        verify_initial_snapshot(saved, host_cfg)

    host_model = RaonModel(host_cfg, weights, backend=NumpyBackend(weights))
    host_model.load_snapshot(snapshot_file)

    # Continue both implementations from the PyTorch-produced state.
    tok2 = 161
    with torch.no_grad():
        out2 = pt_model(
            input_ids=torch.tensor([[tok2]]),
            past_key_values=pt_cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
    pt_logits = out2.logits[0, -1].float().numpy()
    host_logits = host_model.step(tok2)

    cos_sim = cosine_similarity(pt_logits, host_logits)
    assert cos_sim >= 0.9999, f"Snapshot continuation logits cosine {cos_sim:.6f} < 0.9999"
    assert int(np.argmax(pt_logits)) == int(np.argmax(host_logits))


def test_config_matches_reference_defaults_and_moe_contract():
    config = RaonConfig.from_dict({})
    assert config.rda_head_dim == 128
    assert config.rda_layers == [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22]
    assert config.mla_layers == [3, 7, 11, 15, 19, 23, 24]

    null_config = RaonConfig.from_dict({
        "raon_mhc_streams": None,
        "raon_mhc_selector_eps": None,
        "raon_mhc_post_factor": None,
        "num_experts": None,
    })
    assert (null_config.raon_mhc_streams, null_config.raon_mhc_selector_eps) == (4, 1e-5)
    assert null_config.raon_mhc_post_factor == 2.0
    assert not null_config.is_moe_layer(1)

    derived_head_dim_config = RaonConfig.from_dict({
        "hidden_size": 2560,
        "num_attention_heads": 40,
        "head_dim": None,
    })
    assert derived_head_dim_config.rda_head_dim == 64

    frequency_config = RaonConfig.from_dict({"first_k_dense_replace": 1, "moe_layer_freq": 2})
    assert not frequency_config.is_moe_layer(1)
    assert frequency_config.is_moe_layer(2)
    with pytest.raises(NotImplementedError, match="sigmoid router"):
        RaonConfig.from_dict({"moe_router_activation_func": "softmax"})


def test_rda_config_without_rda_layers_raises():
    """A partial rda_config that omits rda_layers must raise, not default to all MLA."""
    with pytest.raises(ValueError, match="rda_layers"):
        RaonConfig.from_dict({
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "head_dim": 16,
            "rda_config": {
                "num_heads": 4,
                "head_dim": 16,
                "conv_kernel_size": 4,
                "gate_lower_bound": -5.0,
            },
        })


def test_rda_config_computes_mla_layers_as_complement():
    """When mla_layers is omitted, MLA layers are the complement of rda_layers."""
    config = RaonConfig.from_dict({
        "num_hidden_layers": 5,
        "num_attention_heads": 4,
        "head_dim": 16,
        "rda_config": {
            "num_heads": 4,
            "head_dim": 16,
            "conv_kernel_size": 4,
            "rda_layers": [1, 3, 5],  # 0-based layers 0, 2, 4
        },
    })
    assert config.rda_layers == [0, 2, 4]
    assert config.mla_layers == [1, 3]


def test_rda_config_mla_layers_must_agree_with_complement():
    """An explicit mla_layers that disagrees with the rda_layers complement must raise."""
    with pytest.raises(ValueError, match="complement"):
        RaonConfig.from_dict({
            "num_hidden_layers": 5,
            "num_attention_heads": 4,
            "head_dim": 16,
            "rda_config": {
                "num_heads": 4,
                "head_dim": 16,
                "conv_kernel_size": 4,
                "rda_layers": [1, 3],     # 0-based layers 0, 2
                "mla_layers": [2, 4],     # claims 0-based layers 1, 3, missing 4
            },
        })


def test_layer_streams_out_matches_pytorch(tiny_raon_model):
    """Verify layer-by-layer streams_out (mHC FFN mixing, Dense MLP, and MoE) matches PyTorch."""
    pt_model, host_cfg, weights, _ = tiny_raon_model
    backend = NumpyBackend(weights)
    host_model = RaonModel(host_cfg, weights, backend=backend)

    # Ensure config has post_factor = 2.0
    assert host_cfg.raon_mhc_post_factor == 2.0, "raon_mhc_post_factor must be 2.0"

    tok = 42
    embed_w = weights.get("model.embed_tokens.weight", to_float=False)
    tok_embed = embed_w[tok]
    stream_h = np.tile(tok_embed, (host_cfg.raon_mhc_streams, 1))

    stream_pt = torch.from_numpy(stream_h).unsqueeze(0).unsqueeze(0).view(torch.bfloat16)

    # Test Layer 0 (RDA + Dense MLP)
    layer_0_pt = pt_model.model.layers[0]
    with torch.no_grad():
        pt_out_0 = layer_0_pt(stream_pt, attention_mask=None, token_mask=None, cache=None)

    # Host step for Layer 0
    attn_mhc_pfx = "model.layers.0.self_attn_raon_mhc"
    h_pre, h_post, h_res = host_model._route_mhc(stream_h, attn_mhc_pfx)
    agg_attn = host_model._mhc_aggregate(stream_h, h_pre)
    in_norm_w = weights.get("model.layers.0.input_layernorm.weight", to_float=True)
    branch_in_attn = rms_norm(agg_attn, in_norm_w, eps=host_cfg.rms_norm_eps)
    branch_out_attn = host_model._forward_rda(0, branch_in_attn)
    stream_mid = host_model._mhc_mix(stream_h, branch_out_attn, h_post, h_res)

    mlp_mhc_pfx = "model.layers.0.mlp_raon_mhc"
    h_pre_m, h_post_m, h_res_m = host_model._route_mhc(stream_mid, mlp_mhc_pfx)
    agg_mlp = host_model._mhc_aggregate(stream_mid, h_pre_m)
    post_norm_w = weights.get("model.layers.0.post_attention_layernorm.weight", to_float=True)
    branch_in_mlp = rms_norm(agg_mlp, post_norm_w, eps=host_cfg.rms_norm_eps)
    branch_out_mlp = host_model._forward_dense_mlp(0, branch_in_mlp)
    stream_out_0_h = host_model._mhc_mix(stream_mid, branch_out_mlp, h_post_m, h_res_m)

    cos_0 = cosine_similarity(pt_out_0.squeeze().float().numpy(), bf16_to_f32(stream_out_0_h))
    assert cos_0 >= 0.9999, f"Layer 0 streams_out cos {cos_0:.6f} < 0.9999 (Dense MLP / mHC post mixing drift)"

    # Test Layer 1 (MLA + MoE)
    layer_1_pt = pt_model.model.layers[1]
    with torch.no_grad():
        pt_out_1 = layer_1_pt(pt_out_0, attention_mask=None, token_mask=None, cache=None)

    attn_mhc_1 = "model.layers.1.self_attn_raon_mhc"
    h_pre_1, h_post_1, h_res_1 = host_model._route_mhc(stream_out_0_h, attn_mhc_1)
    agg_attn_1 = host_model._mhc_aggregate(stream_out_0_h, h_pre_1)
    in_norm_w_1 = weights.get("model.layers.1.input_layernorm.weight", to_float=True)
    branch_in_attn_1 = rms_norm(agg_attn_1, in_norm_w_1, eps=host_cfg.rms_norm_eps)
    branch_out_attn_1 = host_model._forward_mla(1, branch_in_attn_1)
    stream_mid_1 = host_model._mhc_mix(stream_out_0_h, branch_out_attn_1, h_post_1, h_res_1)

    mlp_mhc_1 = "model.layers.1.mlp_raon_mhc"
    h_pre_m1, h_post_m1, h_res_m1 = host_model._route_mhc(stream_mid_1, mlp_mhc_1)
    agg_mlp_1 = host_model._mhc_aggregate(stream_mid_1, h_pre_m1)
    post_norm_w_1 = weights.get("model.layers.1.post_attention_layernorm.weight", to_float=True)
    branch_in_mlp_1 = rms_norm(agg_mlp_1, post_norm_w_1, eps=host_cfg.rms_norm_eps)
    branch_out_mlp_1 = host_model._forward_moe(1, branch_in_mlp_1, "layer_001")
    stream_out_1_h = host_model._mhc_mix(stream_mid_1, branch_out_mlp_1, h_post_m1, h_res_m1)

    cos_1 = cosine_similarity(pt_out_1.squeeze().float().numpy(), bf16_to_f32(stream_out_1_h))
    assert cos_1 >= 0.9999, f"Layer 1 (MLA + MoE) streams_out cos {cos_1:.6f} < 0.9999"

    # Test Layer 2 (RDA + MoE)
    layer_2_pt = pt_model.model.layers[2]
    with torch.no_grad():
        pt_out_2 = layer_2_pt(pt_out_1, attention_mask=None, token_mask=None, cache=None)

    attn_mhc_2 = "model.layers.2.self_attn_raon_mhc"
    h_pre_2, h_post_2, h_res_2 = host_model._route_mhc(stream_out_1_h, attn_mhc_2)
    agg_attn_2 = host_model._mhc_aggregate(stream_out_1_h, h_pre_2)
    in_norm_w_2 = weights.get("model.layers.2.input_layernorm.weight", to_float=True)
    branch_in_attn_2 = rms_norm(agg_attn_2, in_norm_w_2, eps=host_cfg.rms_norm_eps)
    branch_out_attn_2 = host_model._forward_rda(2, branch_in_attn_2)
    stream_mid_2 = host_model._mhc_mix(stream_out_1_h, branch_out_attn_2, h_post_2, h_res_2)

    mlp_mhc_2 = "model.layers.2.mlp_raon_mhc"
    h_pre_m2, h_post_m2, h_res_m2 = host_model._route_mhc(stream_mid_2, mlp_mhc_2)
    agg_mlp_2 = host_model._mhc_aggregate(stream_mid_2, h_pre_m2)
    post_norm_w_2 = weights.get("model.layers.2.post_attention_layernorm.weight", to_float=True)
    branch_in_mlp_2 = rms_norm(agg_mlp_2, post_norm_w_2, eps=host_cfg.rms_norm_eps)
    branch_out_mlp_2 = host_model._forward_moe(2, branch_in_mlp_2, "layer_002")
    stream_out_2_h = host_model._mhc_mix(stream_mid_2, branch_out_mlp_2, h_post_m2, h_res_m2)

    cos_2 = cosine_similarity(pt_out_2.squeeze().float().numpy(), bf16_to_f32(stream_out_2_h))
    assert cos_2 >= 0.9999, f"Layer 2 (RDA + MoE) streams_out cos {cos_2:.6f} < 0.9999"


def _mla_latent_snapshot(cfg: RaonConfig, seed: int = 0, t_len: int = 3):
    """Synthetic golden-format MLA latent cache (uint16 BF16 bits, leading batch dim)."""
    rng = np.random.default_rng(seed)
    lat = f32_to_bf16_rtne(rng.standard_normal((1, t_len, cfg.kv_lora_rank)))
    pe = f32_to_bf16_rtne(rng.standard_normal((1, t_len, cfg.qk_rope_head_dim)))
    snapshot = {
        "layer_001.mla_kv_lat_cache": lat,
        "layer_001.mla_k_pe_cache": pe,
    }
    return snapshot, lat[0], pe[0]


def _old_formula_expanded_cache(cfg: RaonConfig, w_kv_b_f32: np.ndarray, lat_u16, pe_u16):
    """Pre-change load_snapshot formula: weights.get -> FP32 dot -> RTNE -> split/pe/transpose."""
    kv_u16 = f32_to_bf16_rtne(np.dot(bf16_to_f32(lat_u16), w_kv_b_f32.T))
    kv_f32 = bf16_to_f32(kv_u16).reshape(
        lat_u16.shape[0], cfg.num_attention_heads, cfg.qk_nope_head_dim + cfg.v_head_dim
    )
    k_nope = kv_f32[:, :, : cfg.qk_nope_head_dim]
    value = kv_f32[:, :, cfg.qk_nope_head_dim:]
    pe_exp = np.tile(bf16_to_f32(pe_u16)[:, np.newaxis, :], (1, cfg.num_attention_heads, 1))
    key = np.concatenate([k_nope, pe_exp], axis=-1)
    return f32_to_bf16_rtne(key.transpose(1, 0, 2)), f32_to_bf16_rtne(value.transpose(1, 0, 2))


def test_mla_latent_expansion_via_backend_is_bit_exact(tiny_raon_model):
    """Backend-routed expansion must be bit-identical to the pre-change weights.get formula."""
    _, host_cfg, weights, _ = tiny_raon_model
    snapshot, lat, pe = _mla_latent_snapshot(host_cfg)
    backend = NumpyBackend(weights)
    model = RaonModel(host_cfg, weights, backend=backend)
    model.load_snapshot(snapshot)

    exp_k, exp_v = _old_formula_expanded_cache(
        host_cfg, weights.get("model.layers.1.self_attn.kv_b_proj.weight", to_float=True), lat, pe
    )
    assert backend.mla_key_cache[1].dtype == np.uint16
    assert backend.mla_key_cache[1].shape == exp_k.shape
    assert np.array_equal(backend.mla_key_cache[1], exp_k), "expanded key bits differ from pre-change formula"
    assert np.array_equal(backend.mla_value_cache[1], exp_v), "expanded value bits differ from pre-change formula"


def test_mla_latent_expansion_when_only_backend_knows_kv_b(tiny_raon_model):
    """A backend that owns kv_b itself (NPU-style bundle) fills the cache with empty host weights."""
    _, host_cfg, weights, _ = tiny_raon_model
    kv_b = "model.layers.1.self_attn.kv_b_proj.weight"
    snapshot, lat, pe = _mla_latent_snapshot(host_cfg, seed=1)

    host_weights = _DictWeights({})
    assert kv_b not in host_weights  # the host weights carry no kv_b; only the backend bundle does
    backend = NumpyBackend(_DictWeights({kv_b: weights.get(kv_b, to_float=False)}))
    model = RaonModel(host_cfg, host_weights, backend=backend)
    model.load_snapshot(snapshot)

    exp_k, exp_v = _old_formula_expanded_cache(
        host_cfg, weights.get(kv_b, to_float=True), lat, pe
    )
    assert backend.mla_key_cache[1] is not None
    assert np.array_equal(backend.mla_key_cache[1], exp_k), "backend-owned kv_b expansion bits differ"
    assert np.array_equal(backend.mla_value_cache[1], exp_v)


def test_mla_latent_expansion_raises_without_kv_b(tiny_raon_model):
    """A latent cache that cannot be expanded must raise, not silently stay empty."""
    _, host_cfg, _, _ = tiny_raon_model
    snapshot, _, _ = _mla_latent_snapshot(host_cfg, seed=2)
    model = RaonModel(host_cfg, _DictWeights({}), backend=NumpyBackend(_DictWeights({})))
    with pytest.raises(KeyError, match="kv_b_proj"):
        model.load_snapshot(snapshot)


def _snapshot_test_config() -> RaonConfig:
    """Tiny standalone config for snapshot-completeness checks (RDA layers 0/2, MLA layer 1)."""
    return RaonConfig.from_dict({
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "head_dim": 16,
        "q_lora_rank": 32,
        "kv_lora_rank": 32,
        "qk_nope_head_dim": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 16,
        "rda_decay_rank": 16,
        "rda_config": {
            "num_heads": 4,
            "head_dim": 16,
            "conv_kernel_size": 4,
            "rda_layers": [1, 3],  # 1-based: layers 0 and 2
            "mla_layers": [2],     # 1-based: layer 1
        },
    })


def _synthetic_state_snapshot(cfg: RaonConfig, *, include_expanded: bool = False, t_len: int = 3):
    """Synthetic golden-format state snapshot (uint16 BF16 bits + __dtypes_json__)."""
    arrays: dict[str, np.ndarray] = {}
    for idx in cfg.rda_layers:
        p = f"layer_{idx:03d}"
        arrays[f"{p}.rda_state"] = np.zeros(
            (1, cfg.rda_num_heads, cfg.rda_head_dim, cfg.rda_head_dim), dtype=np.float32
        )
        arrays[f"{p}.rda_conv_state"] = np.zeros(
            (1, 3 * cfg.rda_num_heads * cfg.rda_head_dim, cfg.rda_conv_kernel_size - 1),
            dtype=np.uint16,
        )
    for idx in cfg.mla_layers:
        p = f"layer_{idx:03d}"
        arrays[f"{p}.mla_kv_lat_cache"] = np.zeros((1, t_len, cfg.kv_lora_rank), dtype=np.uint16)
        arrays[f"{p}.mla_k_pe_cache"] = np.zeros((1, t_len, cfg.qk_rope_head_dim), dtype=np.uint16)
        if include_expanded:
            arrays[f"{p}.mla_key_cache"] = np.zeros(
                (1, cfg.num_attention_heads, t_len, cfg.qk_nope_head_dim + cfg.qk_rope_head_dim),
                dtype=np.uint16,
            )
            arrays[f"{p}.mla_value_cache"] = np.zeros(
                (1, cfg.num_attention_heads, t_len, cfg.v_head_dim), dtype=np.uint16
            )
    dtypes = {k: ("float32" if v.dtype == np.float32 else "bfloat16") for k, v in arrays.items()}
    arrays["__dtypes_json__"] = np.asarray(json.dumps(dtypes, sort_keys=True))
    return arrays


def test_verify_initial_snapshot_accepts_complete_pairs():
    """A snapshot with every RDA pair and a complete MLA latent pair must PASS."""
    cfg = _snapshot_test_config()
    declared = verify_initial_snapshot(_synthetic_state_snapshot(cfg), cfg)
    assert declared["layer_000.rda_state"] == "float32"
    assert declared["layer_000.rda_conv_state"] == "bfloat16"
    assert declared["layer_001.mla_kv_lat_cache"] == "bfloat16"
    # Latent AND expanded pairs both complete -> also accepted.
    verify_initial_snapshot(_synthetic_state_snapshot(cfg, include_expanded=True), cfg)


@pytest.mark.parametrize(
    "missing_key",
    [
        "layer_000.rda_state",
        "layer_000.rda_conv_state",
        "layer_001.mla_kv_lat_cache",
        "layer_001.mla_k_pe_cache",
    ],
)
def test_verify_initial_snapshot_rejects_missing_or_half_state(missing_key):
    """One missing RDA state half or MLA cache-pair half must fail before any run."""
    cfg = _snapshot_test_config()
    snapshot = _synthetic_state_snapshot(cfg)
    snapshot.pop(missing_key)
    with pytest.raises(ValueError, match=re.escape(missing_key)):
        verify_initial_snapshot(snapshot, cfg)


def test_verify_initial_snapshot_rejects_half_expanded_pair():
    cfg = _snapshot_test_config()
    snapshot = _synthetic_state_snapshot(cfg, include_expanded=True)
    snapshot.pop("layer_001.mla_value_cache")
    with pytest.raises(ValueError, match="mla_value_cache"):
        verify_initial_snapshot(snapshot, cfg)


def test_verify_initial_snapshot_rejects_wrong_shape_or_dtype():
    cfg = _snapshot_test_config()

    wrong_shape = _synthetic_state_snapshot(cfg)
    wrong_shape["layer_000.rda_conv_state"] = np.zeros(
        (1, 3 * cfg.rda_num_heads * cfg.rda_head_dim, cfg.rda_conv_kernel_size), dtype=np.uint16
    )
    with pytest.raises(ValueError, match="rda_conv_state.*core shape"):
        verify_initial_snapshot(wrong_shape, cfg)

    wrong_dtype = _synthetic_state_snapshot(cfg)
    wrong_dtype["layer_002.rda_state"] = wrong_dtype["layer_002.rda_state"].astype(np.float64)
    with pytest.raises(ValueError, match="layer_002.rda_state.*does not match declared"):
        verify_initial_snapshot(wrong_dtype, cfg)

    no_dtype_meta = _synthetic_state_snapshot(cfg)
    no_dtype_meta.pop("__dtypes_json__")
    with pytest.raises(KeyError, match="__dtypes_json__"):
        verify_initial_snapshot(no_dtype_meta, cfg)


def test_load_snapshot_rejects_half_mla_cache_pairs(tiny_raon_model):
    """load_snapshot must raise on half caches, never silently run with an empty cache."""
    _, host_cfg, _, _ = tiny_raon_model
    model = RaonModel(host_cfg, _DictWeights({}), backend=NumpyBackend(_DictWeights({})))
    latent_only = _synthetic_state_snapshot(host_cfg, include_expanded=False)
    expanded_pair = _synthetic_state_snapshot(host_cfg, include_expanded=True)

    with pytest.raises(KeyError, match="mla_k_pe_cache"):
        model.load_snapshot(
            {k: v for k, v in latent_only.items() if not k.endswith("layer_001.mla_k_pe_cache")}
        )
    with pytest.raises(KeyError, match="mla_kv_lat_cache"):
        model.load_snapshot(
            {k: v for k, v in latent_only.items() if not k.endswith("layer_001.mla_kv_lat_cache")}
        )
    with pytest.raises(KeyError, match="mla_value_cache"):
        model.load_snapshot(
            {k: v for k, v in expanded_pair.items()
             if not k.endswith("layer_001.mla_value_cache")}
        )
    with pytest.raises(KeyError, match="mla_key_cache"):
        model.load_snapshot(
            {k: v for k, v in expanded_pair.items()
             if not k.endswith("layer_001.mla_key_cache")}
        )


def _make_synthetic_step(
    step: int,
    num_layers: int = 3,
    streams_cos: float = 0.99995,
    argmax_match: bool = True,
    argmax_tie_match: bool = True,
) -> dict:
    layers = []
    for idx in range(num_layers):
        layer = {
            "layer": idx,
            "streams_in_cos": 0.99999,
            "mixer_in_cos": 0.99999,
            "mixer_out_cos": 0.99999,
            "streams_out_cos": streams_cos,
        }
        if idx >= 1:
            layer.update({
                "router_top_set_pct": 100.0,
                "router_top_order_match": True,
                "router_weights_max_err": 1e-4,
            })
        layers.append(layer)
    return {
        "step": step,
        "argmax_match": argmax_match,
        "argmax_tie_match": argmax_tie_match,
        "logits_cos": 0.99999,
        "layers": layers,
    }


def test_verify_requested_step_files_rejects_missing_or_invalid_range(tmp_path):
    """Missing step snapshot files or reversed step ranges must fail immediately."""
    for step in (2, 3):
        np.savez(tmp_path / f"step_{step:06d}.npz", dummy=np.zeros(1))

    files = verify_requested_step_files(tmp_path, 2, 3)
    assert len(files) == 2

    with pytest.raises(FileNotFoundError, match="step_000004.npz"):
        verify_requested_step_files(tmp_path, 2, 4)

    with pytest.raises(ValueError, match="Invalid step range"):
        verify_requested_step_files(tmp_path, 5, 3)


class _StatefulFakeModel:
    """Fake model whose per-step output depends on every token fed since resume."""

    def __init__(self):
        self.fed_tokens: list[int] = []

    def step(self, token_id: int) -> float:
        self.fed_tokens.append(token_id)
        return float(sum(self.fed_tokens))


def _resume_from_snapshot_and_step(model, generated_token_ids, start_step, end_step):
    """Mirror main()'s live-run flow: validate the start step BEFORE stepping."""
    validate_start_step(start_step)
    return [
        model.step(generated_token_ids[step - SUPPORTED_START_STEP])
        for step in range(start_step, end_step + 1)
    ]


def test_start_step_other_than_2_is_rejected():
    """Only start step 2 is supported; anything else must raise, not mis-verify."""
    validate_start_step(SUPPORTED_START_STEP)  # the single supported value must not raise
    for bad in (1, 0, 3, 5, 16):
        with pytest.raises(ValueError, match="not supported"):
            validate_start_step(bad)
    with pytest.raises(ValueError, match="steps 2\\.\\.3 would be skipped"):
        validate_start_step(4)


def test_rejected_start_step_never_steps_the_stateful_model():
    """start_step != 2 raises BEFORE any step runs — no verification on skipped-step state.

    Without the guard, a start_step=3 run resuming from the step-1 snapshot would
    feed token index start_step-2=1 first, silently skipping step 2's token, so
    every step's state-dependent output would come from an under-consumed state.
    """
    generated_token_ids = [10, 11, 12, 13]

    model = _StatefulFakeModel()
    with pytest.raises(ValueError, match="start-step=3 is not supported"):
        _resume_from_snapshot_and_step(model, generated_token_ids, start_step=3, end_step=4)
    assert model.fed_tokens == [], "guard must fire before the model is stepped at all"

    # Had it run, step 3 would have been computed as if token 10 (step 2) never
    # existed: history-dependent output would differ from the full-history one.
    skipped_state = _StatefulFakeModel()
    skipped_out = skipped_state.step(generated_token_ids[1])
    full_state = _StatefulFakeModel()
    for tok in generated_token_ids[:2]:
        full_state.step(tok)
    assert skipped_out != full_state.step(generated_token_ids[1])


def test_supported_start_step_2_feeds_every_step_from_the_snapshot_boundary():
    """start_step=2 (the only supported value) steps the model from token index 0."""
    generated_token_ids = [10, 11, 12, 13]
    model = _StatefulFakeModel()
    outputs = _resume_from_snapshot_and_step(model, generated_token_ids, start_step=2, end_step=4)
    assert model.fed_tokens == [10, 11, 12]
    assert outputs == [10.0, 21.0, 33.0]


def test_routed_moe_uses_config_situ_betas(tiny_situ_model):
    """Routed experts must use the config's SiTU betas, not a hardcoded 4.0/25.0 pair."""
    pt_model, host_cfg, weights, _ = tiny_situ_model
    assert host_cfg.activation_situ_beta == 2.0
    assert host_cfg.activation_situ_linear_beta == 9.0

    backend = NumpyBackend(weights)
    host_model = RaonModel(host_cfg, weights, backend=backend)

    layer = pt_model.model.layers[2]
    assert layer.is_moe
    rng = np.random.default_rng(1234)
    # Large-amplitude activations saturate the SiTU tanh terms, making the two
    # beta pairs (2.0/9.0 vs the 4.0/25.0 defaults) produce visibly different outputs.
    x_u16 = f32_to_bf16_rtne((rng.standard_normal(host_cfg.hidden_size) * 8.0).astype(np.float32))
    x_1d = torch.from_numpy(np.asarray(x_u16)).view(torch.bfloat16)

    with torch.no_grad():
        top_pt, probs_pt = layer.moe.gate(x_1d.float().unsqueeze(0))
        top_np = top_pt[0].tolist()
        probs_np = probs_pt[0].float().numpy()
        routed_pt = torch.zeros(host_cfg.hidden_size, dtype=torch.float32)
        for slot, expert_id in enumerate(top_np):
            expert_out = layer.moe.experts[int(expert_id)](x_1d).float()
            routed_pt += float(probs_np[slot]) * expert_out
        pt_moe_out = layer.moe(x_1d.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0).float()

    host_routed = backend.moe_experts(
        2, x_u16, np.asarray(top_np), np.asarray(probs_np, dtype=np.float32),
        gate_beta=host_cfg.activation_situ_beta,
        value_beta=host_cfg.activation_situ_linear_beta,
    )
    host_routed_default_beta = backend.moe_experts(
        2, x_u16, np.asarray(top_np), np.asarray(probs_np, dtype=np.float32)
    )
    host_full = host_model._forward_moe(2, x_u16, "layer_002")

    assert cosine_similarity(bf16_to_f32(host_routed), routed_pt.numpy()) >= 0.9999
    assert cosine_similarity(
        bf16_to_f32(host_routed_default_beta), routed_pt.numpy()
    ) < 0.9999, "default-beta experts match — the config betas are not being passed through"
    assert cosine_similarity(bf16_to_f32(host_full), pt_moe_out.numpy()) >= 0.9999


def test_evaluate_summary_rejects_empty_missing_and_truncated_results():
    """Empty, truncated, non-contiguous, or metric-missing payloads must FAIL."""
    # 1. Empty step_results
    empty_summary = evaluate_summary({"step_results": []})
    assert empty_summary["criteria"]["results_complete_and_valid"] is False
    assert empty_summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9998"] is False
    assert empty_summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9999"] is False
    assert empty_summary["criteria"]["router_top_match_gte_99"] is False
    assert empty_summary["criteria"]["step_argmax_exact_all_match"] is False
    assert empty_summary["criteria"]["step_argmax_in_golden_tie_all_match"] is False
    assert print_evaluation(empty_summary) is False

    # 2. Early stop / truncated steps (expected 2..4 = 3 steps, only 2 recorded)
    truncated_payload = {
        "start_step": 2,
        "end_step": 4,
        "expected_steps": 3,
        "expected_num_layers": 3,
        "step_results": [
            _make_synthetic_step(2, num_layers=3),
            _make_synthetic_step(3, num_layers=3),
        ],
    }
    truncated_summary = evaluate_summary(truncated_payload)
    assert truncated_summary["criteria"]["results_complete_and_valid"] is False
    assert print_evaluation(truncated_summary) is False

    # 3. Non-contiguous steps (2, 3, 5 instead of 2, 3, 4)
    gap_payload = {
        "start_step": 2,
        "end_step": 4,
        "expected_steps": 3,
        "expected_num_layers": 3,
        "step_results": [
            _make_synthetic_step(2, num_layers=3),
            _make_synthetic_step(3, num_layers=3),
            _make_synthetic_step(5, num_layers=3),
        ],
    }
    gap_summary = evaluate_summary(gap_payload)
    assert gap_summary["criteria"]["results_complete_and_valid"] is False
    assert print_evaluation(gap_summary) is False

    # 4. Missing layer / missing required metric
    missing_metric_step = _make_synthetic_step(2, num_layers=3)
    del missing_metric_step["layers"][1]["router_top_set_pct"]
    missing_metric_payload = {
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "step_results": [missing_metric_step],
    }
    missing_metric_summary = evaluate_summary(missing_metric_payload)
    assert missing_metric_summary["criteria"]["results_complete_and_valid"] is False
    assert print_evaluation(missing_metric_summary) is False


def test_evaluate_summary_preserves_nondefault_moe_layer_placement():
    """Completeness requires router metrics only on the config's actual MoE layers."""
    step = _make_synthetic_step(2, num_layers=6)
    for layer in step["layers"]:
        if layer["layer"] not in (2, 4):
            layer.pop("router_top_set_pct", None)
            layer.pop("router_top_order_match", None)
            layer.pop("router_weights_max_err", None)
    summary = evaluate_summary({
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 6,
        "first_k_dense_replace": 2,
        "moe_layer_freq": 2,
        "num_experts": 8,
        "step_results": [step],
    })
    assert summary["expected_moe_layers"] == [2, 4]
    assert summary["expected_moe_layers_source"] == "legacy payload config values"
    assert summary["criteria"]["results_complete_and_valid"] is True


def test_evaluate_summary_num_experts_none_requires_no_router_metrics():
    """A dense-only config has no expected router-bearing layers."""
    step = _make_synthetic_step(2, num_layers=3)
    for layer in step["layers"]:
        layer.pop("router_top_set_pct", None)
        layer.pop("router_top_order_match", None)
        layer.pop("router_weights_max_err", None)
    summary = evaluate_summary({
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "num_experts": None,
        "step_results": [step],
    })
    assert summary["expected_moe_layers"] == []
    assert summary["criteria"]["results_complete_and_valid"] is True
    assert summary["criteria"]["router_top_match_gte_99"] is True
    assert print_evaluation(summary) is True


def test_evaluate_summary_legacy_payload_uses_released_moe_defaults(capsys):
    """An old payload without config metadata remains rejudgeable and reports fallback."""
    summary = evaluate_summary({
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "step_results": [_make_synthetic_step(2, num_layers=3)],
    })
    assert summary["expected_moe_layers"] == [1, 2]
    assert summary["expected_moe_layers_source"] == "released config defaults"
    assert summary["criteria"]["results_complete_and_valid"] is True
    assert print_evaluation(summary) is True
    assert "source: released config defaults" in capsys.readouterr().out


def test_evaluate_summary_order_only_diff_does_not_truncate_boundary():
    """Order-only router differences (set=100%) must not cut Criterion 1 boundary."""
    step2 = _make_synthetic_step(2, num_layers=3, streams_cos=0.99995)
    # Layer 1 has an order-only difference (same top-k set, different order)
    step2["layers"][1]["router_top_set_pct"] = 100.0
    step2["layers"][1]["router_top_order_match"] = False
    # Layer 2 is after the order-only difference and has cosine 0.99985 (< 0.9999, >= 0.9998)
    step2["layers"][2]["streams_out_cos"] = 0.99985

    payload = {
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "step_results": [step2],
    }
    summary = evaluate_summary(payload)
    assert summary["criteria"]["results_complete_and_valid"] is True
    assert summary["pre_first_router_difference"]["layer_count"] == 3
    assert summary["at_or_after_first_router_difference"]["layer_count"] == 0
    assert len(summary["first_router_differences"]) == 0
    assert len(summary["router_order_differences"]) == 1
    assert summary["router_order_differences"][0]["layer"] == 1
    assert summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9998"] is True
    assert summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9999"] is False
    assert len(summary["layers_below_0_9999_before_first_router_difference"]) == 1
    assert print_evaluation(summary) is True

    # If a layer after an order-only difference drops below 0.9998, Criterion 1 must FAIL.
    step2_low = _make_synthetic_step(2, num_layers=3, streams_cos=0.99995)
    step2_low["layers"][1]["router_top_set_pct"] = 100.0
    step2_low["layers"][1]["router_top_order_match"] = False
    step2_low["layers"][2]["streams_out_cos"] = 0.99975
    summary_low = evaluate_summary({
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "step_results": [step2_low],
    })
    assert summary_low["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9998"] is False
    assert print_evaluation(summary_low) is False


def test_evaluate_summary_set_difference_partitions_boundary():
    """A real router set difference cuts the 0.9998/0.9999 Criterion 1 boundary."""
    step2 = _make_synthetic_step(2, num_layers=3, streams_cos=0.99995)
    # Pre-difference layer 0 is below 0.9999 but above 0.9998.
    step2["layers"][0]["streams_out_cos"] = 0.99985
    # Layer 1 has an actual set difference (not merely order).
    step2["layers"][1]["router_top_set_pct"] = 99.0
    step2["layers"][1]["router_top_order_match"] = False

    summary = evaluate_summary({
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "step_results": [step2],
    })
    assert summary["criteria"]["results_complete_and_valid"] is True
    assert len(summary["first_router_differences"]) == 1
    assert summary["first_router_differences"][0]["layer"] == 1
    assert len(summary["router_set_differences"]) == 1
    assert summary["router_set_differences"][0]["layer"] == 1
    # Pre-difference layers: only layer 0.
    assert summary["pre_first_router_difference"]["layer_count"] == 1
    assert summary["pre_first_router_difference"]["min_streams_out_cos"] == pytest.approx(0.99985)
    # Post-difference layers: layer 1 and 2.
    assert summary["at_or_after_first_router_difference"]["layer_count"] == 2
    assert summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9998"] is True
    assert summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9999"] is False
    assert len(summary["layers_below_0_9999_before_first_router_difference"]) == 1
    assert print_evaluation(summary) is True

    # If a pre-difference layer drops below 0.9998, Criterion 1 must FAIL.
    step2_low = _make_synthetic_step(2, num_layers=3, streams_cos=0.99995)
    step2_low["layers"][0]["streams_out_cos"] = 0.99975
    step2_low["layers"][1]["router_top_set_pct"] = 99.0
    summary_low = evaluate_summary({
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 3,
        "step_results": [step2_low],
    })
    assert summary_low["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9998"] is False
    assert len(summary_low["layers_below_0_9998_before_first_router_difference"]) == 1
    assert print_evaluation(summary_low) is False


def test_check_layer_summary_exit_code():
    """summary_exit_code must be 0 only when every row passed."""
    assert check_layer_module.summary_exit_code([]) == 0
    assert check_layer_module.summary_exit_code([
        {"pass": True},
        {"pass": True},
    ]) == 0
    assert check_layer_module.summary_exit_code([
        {"pass": True},
        {"pass": False},
    ]) == 1
    assert check_layer_module.summary_exit_code([
        {"pass": False},
        {"pass": False},
    ]) == 1


def test_check_golden_missing_dirs_uses_env_and_raises(monkeypatch, tmp_path):
    """main() must use $RAON_MODEL_DIR/$RAON_GOLDEN_DIR and fail clearly when missing."""
    monkeypatch.setenv("RAON_MODEL_DIR", str(tmp_path / "no_model"))
    monkeypatch.setenv("RAON_GOLDEN_DIR", str(tmp_path / "no_golden"))
    monkeypatch.setattr(sys, "argv", ["check_golden.py"])
    with pytest.raises(FileNotFoundError, match="RAON_MODEL_DIR"):
        check_golden_module.main()


def test_check_golden_rejudge_failing_summary_exits_nonzero(monkeypatch, tmp_path):
    """A --rejudge run on a failing summary must exit with code 1, not 0."""
    failing = evaluate_summary({"step_results": []})
    json_path = tmp_path / "failing.json"
    json_path.write_text(json.dumps(failing), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_golden.py", "--rejudge", str(json_path)])
    with pytest.raises(SystemExit) as exc_info:
        check_golden_module.main()
    assert exc_info.value.code == 1


def test_rejudge_recorded_golden_check_json():
    """Rejudge host/out/golden_check.json and verify exact counts and thresholds.

    Skips with a reason when the git-ignored recorded results file is absent,
    so a clean checkout does not fail here.
    """
    if not GOLDEN_CHECK_JSON.is_file():
        pytest.skip(
            f"Recorded golden summary {GOLDEN_CHECK_JSON} not found; "
            "skip recorded-results rejudge test"
        )
    with open(GOLDEN_CHECK_JSON, encoding="utf-8") as f:
        payload = json.load(f)

    summary = evaluate_summary(payload)
    assert summary["criteria"]["results_complete_and_valid"] is True
    assert summary["pre_first_router_difference"]["layer_count"] == 269
    assert summary["at_or_after_first_router_difference"]["layer_count"] == 106
    assert abs(summary["pre_first_router_difference"]["min_streams_out_cos"] - 0.9998517218) < 1e-7
    assert len(summary["layers_below_0_9999_before_first_router_difference"]) == 6
    assert len(summary["router_order_differences"]) == 38
    assert summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9998"] is True
    assert summary["criteria"]["layer_output_cos_before_first_router_difference_gte_0_9999"] is False
    assert summary["criteria"]["router_top_match_gte_99"] is True
    assert summary["criteria"]["step_argmax_in_golden_tie_all_match"] is True
    assert print_evaluation(summary) is True


def test_validate_generated_token_ids():
    """Verify validation of generated_token_ids table according to fix14 specs."""
    # Pass: exactly end_step - 1 tokens (15 tokens for end_step=16)
    valid_tokens = list(range(100, 115))
    res = validate_generated_token_ids(valid_tokens, end_step=16, vocab_size=200)
    assert res == valid_tokens

    # Reject: missing / not a list
    with pytest.raises(ValueError, match="must be a list"):
        validate_generated_token_ids(None, end_step=16)
    with pytest.raises(ValueError, match="must be a list"):
        validate_generated_token_ids("100, 101", end_step=16)

    # Reject: boolean (True/False)
    bad_bool = list(range(100, 114)) + [True]
    with pytest.raises(ValueError, match=r"generated_token_ids\[14\] must be a non-boolean integer"):
        validate_generated_token_ids(bad_bool, end_step=16)

    # Reject: float (1.0)
    bad_float = [1.0] + list(range(101, 115))
    with pytest.raises(ValueError, match=r"generated_token_ids\[0\] must be a non-boolean integer"):
        validate_generated_token_ids(bad_float, end_step=16)

    # Reject: negative (-1)
    bad_neg = list(range(100, 114)) + [-1]
    with pytest.raises(ValueError, match=r"generated_token_ids\[14\] must be >= 0"):
        validate_generated_token_ids(bad_neg, end_step=16)

    # Reject: vocab_size or greater
    bad_vocab = list(range(100, 114)) + [200]
    with pytest.raises(ValueError, match=r"generated_token_ids\[14\] must be < vocab_size"):
        validate_generated_token_ids(bad_vocab, end_step=16, vocab_size=200)

    # Reject: length end_step - 2 (one short: 14 tokens for end_step=16)
    short_tokens = list(range(100, 114))
    with pytest.raises(ValueError, match="length 14 is insufficient for end_step=16; requires at least 15"):
        validate_generated_token_ids(short_tokens, end_step=16)


def test_rejudge_rejects_null_or_false_baseline():
    """A rejudge payload with baseline_passed=null or False must fail completeness."""
    base_payload = {
        "start_step": 2,
        "end_step": 2,
        "expected_steps": 1,
        "expected_num_layers": 25,
        "step_results": [{
            "step": 2,
            "argmax_match": True,
            "argmax_tie_match": True,
            "logits_cos": 0.9999,
            "layers": [
                {
                    "layer": i,
                    "streams_in_cos": 0.9999,
                    "mixer_in_cos": 0.9999,
                    "mixer_out_cos": 0.9999,
                    "streams_out_cos": 0.9999,
                    **({"router_top_set_pct": 100.0, "router_top_order_match": True, "router_weights_max_err": 0.0} if i in range(1, 25) else {})
                }
                for i in range(25)
            ]
        }],
        "baseline_passed": None,
        "baseline_skipped": True,
    }
    summary = evaluate_summary(base_payload)
    assert summary["criteria"]["results_complete_and_valid"] is False
    assert any("baseline" in err for err in summary.get("completeness_errors", []))
    assert print_evaluation(summary) is False

    base_payload["baseline_passed"] = False
    summary_false = evaluate_summary(base_payload)
    assert summary_false["criteria"]["results_complete_and_valid"] is False
    assert print_evaluation(summary_false) is False


def test_check_golden_cli_start_step_rejection(monkeypatch):
    """CLI must reject unsupported start-step values with argparse exit code 2."""
    for bad_step in (1, 3, 10, 16, 4096):
        monkeypatch.setattr(sys, "argv", ["check_golden.py", "--start-step", str(bad_step)])
        with pytest.raises(SystemExit) as exc_info:
            check_golden_module.main()
        assert exc_info.value.code == 2


def _old_f32_to_bf16_rtne(arr: np.ndarray) -> np.ndarray:
    """Reference copy of the pre-optimization f32->bf16 conversion."""
    arr_f32 = np.asarray(arr, dtype=np.float32)
    orig_shape = arr_f32.shape
    u32 = arr_f32.reshape(-1).view(np.uint32)
    lsb = (u32 >> 16) & 1
    bias = 0x7FFF + lsb
    nan_mask = np.isnan(arr_f32.reshape(-1))
    rounded = u32 + bias
    out = (rounded >> 16).astype(np.uint16)
    out[nan_mask] = 0x7FC0
    return out.reshape(orig_shape)


def _old_bf16_to_f32(arr: np.ndarray) -> np.ndarray:
    """Reference copy of the pre-optimization bf16->f32 conversion."""
    arr_u16 = np.asarray(arr, dtype=np.uint16)
    orig_shape = arr_u16.shape
    flat = (arr_u16.reshape(-1).astype(np.uint32) << 16).view(np.float32)
    return flat.reshape(orig_shape)


def _assert_bf16_bit_equal(new: np.ndarray, old: np.ndarray) -> None:
    assert new.dtype == np.uint16
    assert new.shape == old.shape
    assert np.array_equal(new, old)


def _assert_f32_bit_equal(new: np.ndarray, old: np.ndarray) -> None:
    assert new.dtype == np.float32
    assert new.shape == old.shape
    assert np.array_equal(new.view(np.uint32), old.view(np.uint32))


def test_bf16_to_f32_bit_exact_vs_reference():
    """Optimized bf16->f32 must match the reference for all bit patterns and layouts."""
    all_u16 = np.arange(65536, dtype=np.uint16)
    shapes = [
        (65536,),
        (256, 256),
        (),
        (0,),
    ]
    for shape in shapes:
        if shape == ():
            arr = np.array(0x4040, dtype=np.uint16)
        else:
            arr = all_u16[: np.prod(shape, dtype=np.int64)].reshape(shape)
        new = bf16_to_f32(arr)
        old = _old_bf16_to_f32(arr)
        assert new.dtype == np.float32
        assert new.shape == arr.shape
        _assert_f32_bit_equal(new, old)
        assert new.flags.writeable
        if isinstance(arr, np.ndarray):
            assert not np.shares_memory(new, arr)

    # Non-contiguous slice.
    noncontig = all_u16[:16].reshape(4, 4)[:, ::2]
    new_nc = bf16_to_f32(noncontig)
    _assert_f32_bit_equal(new_nc, _old_bf16_to_f32(noncontig))
    assert new_nc.flags.writeable
    assert not np.shares_memory(new_nc, noncontig)

    # Read-only input.
    ro = all_u16[:8].copy()
    ro.flags.writeable = False
    new_ro = bf16_to_f32(ro)
    _assert_f32_bit_equal(new_ro, _old_bf16_to_f32(ro))
    assert new_ro.flags.writeable
    assert not np.shares_memory(new_ro, ro)


@pytest.mark.parametrize("impl_name", ["numpy", "ml_dtypes"])
def test_bf16_to_f32_conversion_contract_matrix(impl_name):
    """Verify bf16_to_f32 contract: different dtypes, non-contiguous, read-only produce new writable arrays."""
    from host.weights import _ML_DTYPES_BFLOAT16, _bf16_to_f32_ml_dtypes, _bf16_to_f32_numpy

    if impl_name == "ml_dtypes" and _ML_DTYPES_BFLOAT16 is None:
        pytest.skip("ml_dtypes is not installed")
    conv_fn = _bf16_to_f32_ml_dtypes if impl_name == "ml_dtypes" else _bf16_to_f32_numpy

    # 1. Other dtypes (int32, uint32, int64, float32, float64)
    for dt in (np.int32, np.uint32, np.int64, np.float32, np.float64):
        raw = np.array([0x3F80, 0x4000, 0x4040, 0xBF80], dtype=dt)
        out = conv_fn(raw)
        assert out.dtype == np.float32
        assert out.shape == raw.shape
        assert out.flags.writeable
        assert not np.shares_memory(out, raw)
        _assert_f32_bit_equal(out, _old_bf16_to_f32(raw))

    # 2. Non-contiguous input
    all_u16 = np.arange(64, dtype=np.uint16).reshape(8, 8)
    noncontig = all_u16[::2, ::2]
    assert not noncontig.flags.c_contiguous
    out_nc = conv_fn(noncontig)
    assert out_nc.dtype == np.float32
    assert out_nc.shape == noncontig.shape
    assert out_nc.flags.writeable
    assert not np.shares_memory(out_nc, noncontig)
    _assert_f32_bit_equal(out_nc, _old_bf16_to_f32(noncontig))

    # 3. Read-only input
    ro = np.array([0x3F80, 0x4000, 0x4040], dtype=np.uint16)
    ro.flags.writeable = False
    out_ro = conv_fn(ro)
    assert out_ro.dtype == np.float32
    assert out_ro.shape == ro.shape
    assert out_ro.flags.writeable
    assert not np.shares_memory(out_ro, ro)
    _assert_f32_bit_equal(out_ro, _old_bf16_to_f32(ro))


def test_f32_to_bf16_rtne_bit_exact_vs_reference():
    """Optimized f32->bf16 must match the reference across rounding boundaries and layouts."""
    old = _old_f32_to_bf16_rtne

    # 1. Every exactly-representable bf16 value round-trips.
    all_u16 = np.arange(65536, dtype=np.uint16)
    all_f32_exact = bf16_to_f32(all_u16)
    new_rt = f32_to_bf16_rtne(all_f32_exact)
    _assert_bf16_bit_equal(new_rt, old(all_f32_exact))

    # 2. Rounding boundary values in the lower 16 bits.
    base_u32 = all_f32_exact.view(np.uint32)
    for pattern in (0x7FFF, 0x8000, 0x8001, 0xFFFF):
        boundary_u32 = (base_u32 & 0xFFFF0000) | pattern
        boundary_f32 = boundary_u32.view(np.float32)
        _assert_bf16_bit_equal(f32_to_bf16_rtne(boundary_f32), old(boundary_f32))

    # 3. Special values: signed zero, infinities, NaNs with varying payloads.
    specials = np.array([
        0x00000000, 0x80000000,          # +0, -0
        0x7F800000, 0xFF800000,          # +inf, -inf
        0x7FC00000, 0x7FC00001,          # qNaN, qNaN+1
        0xFFC00000, 0x7F800001,          # -qNaN, sNaN-like
        0x7F7FFFFF, 0xFF7FFFFF,          # max finite values
        0x00800000, 0x80800000,          # smallest normals
        0x00000001, 0x80000001,          # subnormals
    ], dtype=np.uint32).view(np.float32)
    _assert_bf16_bit_equal(f32_to_bf16_rtne(specials), old(specials))

    # 4. Values that round up to infinity.
    near_inf = np.array([
        0x7F7FFFFF, 0x7F7FFFE0, 0x7F7FFF00, 0x7F7F8000,
        0xFF7FFFFF, 0xFF7FFFE0, 0xFF7FFF00, 0xFF7F8000,
    ], dtype=np.uint32).view(np.float32)
    _assert_bf16_bit_equal(f32_to_bf16_rtne(near_inf), old(near_inf))

    # 5. One million random bit patterns.
    rng = np.random.default_rng(20261031)
    random_bits = rng.integers(0, 2**32, size=1_000_000, dtype=np.uint32)
    random_f32 = random_bits.view(np.float32)
    _assert_bf16_bit_equal(f32_to_bf16_rtne(random_f32), old(random_f32))

    # 6. float64 input (numpy cast to f32 rounds using the same rule as the reference).
    f64 = rng.standard_normal((10000,)).astype(np.float64) * 1e3
    _assert_bf16_bit_equal(f32_to_bf16_rtne(f64), old(f64))

    # 7. Non-contiguous and read-only inputs.
    contig = rng.standard_normal((8, 8)).astype(np.float32)
    noncontig = contig[:, ::2]
    _assert_bf16_bit_equal(f32_to_bf16_rtne(noncontig), old(noncontig))
    ro = contig.copy()
    ro.flags.writeable = False
    new_ro = f32_to_bf16_rtne(ro)
    _assert_bf16_bit_equal(new_ro, old(ro))
    assert new_ro.flags.writeable
    assert not np.shares_memory(new_ro, ro)


def test_bf16_implementation_selection_and_forced_numpy():
    """Selection happens at import; the environment switch always forces NumPy."""
    from host.weights import _ML_DTYPES_BFLOAT16

    forced = os.environ.get("RAON_BF16_IMPL", "auto").strip().lower() == "numpy"
    expected = "numpy" if forced or _ML_DTYPES_BFLOAT16 is None else "ml_dtypes"
    assert BF16_IMPL == expected
    env = os.environ.copy()
    env["RAON_BF16_IMPL"] = "numpy"
    proc = subprocess.run(
        [sys.executable, "-c", "from host.weights import BF16_IMPL; print(BF16_IMPL)"],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert proc.stdout.strip() == "numpy"


def test_copy_elision_functions_match_preoptimization_bits():
    """Removing same-dtype astype copies must not change any result bits."""
    rng = np.random.default_rng(20261101)
    for _ in range(16):
        x = rng.standard_normal((4, 64)).astype(np.float32)
        weight = rng.standard_normal(64).astype(np.float32)
        val_old = x.astype(np.float32)
        w_old = weight.astype(np.float32)
        mean_sq = np.mean(val_old ** 2, axis=-1, keepdims=True)
        expected_norm_f32 = val_old * (1.0 / np.sqrt(mean_sq + 1e-5)) * w_old
        expected_norm = f32_to_bf16_rtne(expected_norm_f32)
        assert np.array_equal(rms_norm(x, weight), expected_norm)
        actual_norm_f32 = rms_norm(x, weight, keep_fp32=True)
        assert np.array_equal(actual_norm_f32.view(np.uint32), expected_norm_f32.view(np.uint32))

        gate = rng.standard_normal(257).astype(np.float32)
        value = rng.standard_normal(257).astype(np.float32)
        g_old = gate.astype(np.float32)
        v_old = value.astype(np.float32)
        sig_old = 1.0 / (1.0 + np.exp(-g_old))
        term1_old = 4.0 * np.tanh(g_old / 4.0) * sig_old
        term2_old = 25.0 * np.tanh(v_old / 25.0)
        expected_situ = term1_old * term2_old
        actual_situ = situ_glu(gate, value)
        assert np.array_equal(actual_situ.view(np.uint32), expected_situ.view(np.uint32))

    cfg = RaonConfig(hidden_size=64, rda_conv_kernel_size=4)
    model = object.__new__(RaonModel)
    model.config = cfg
    model.conv_states = {}
    old_hist = {}
    conv_weight = rng.standard_normal((64, 4)).astype(np.float32)
    for _ in range(16):
        conv_x = rng.standard_normal(64).astype(np.float32)
        x_old = conv_x.astype(np.float32)
        hist = old_hist.get("q")
        if hist is None:
            hist = np.zeros((64, 3), dtype=np.float32)
        window = np.concatenate([hist, x_old[:, np.newaxis]], axis=-1)
        old_hist["q"] = window[:, -3:]
        conv_out = np.sum(window * conv_weight, axis=-1)
        expected_conv = f32_to_bf16_rtne(
            conv_out * (1.0 / (1.0 + np.exp(-conv_out)))
        )
        actual_conv = model._conv_step(0, "q", conv_x, conv_weight)
        assert np.array_equal(actual_conv, expected_conv)

    # Direct bit comparison for RDA host calculation: astype(copy=False) vs legacy astype(copy=True).
    def _pack_head_bundle_ref(q, k, a, v, beta):
        h_dim = q.shape[0]
        bundle = np.zeros((h_dim, 640), dtype=np.float32)
        bundle[:, 0:128] = q
        bundle[:, 128:256] = k
        bundle[:, 256:384] = a
        bundle[:, 384:512] = v
        bundle[:, 512] = beta
        return bundle

    for _ in range(16):
        for use_u16 in (False, True):
            if use_u16:
                q = rng.integers(0, 65536, size=(40, 128), dtype=np.uint16)
                k = rng.integers(0, 65536, size=(40, 128), dtype=np.uint16)
                v = rng.integers(0, 65536, size=(40, 128), dtype=np.uint16)
                beta = rng.integers(0, 65536, size=40, dtype=np.uint16)
            else:
                q = rng.standard_normal((40, 128)).astype(np.float32)
                k = rng.standard_normal((40, 128)).astype(np.float32)
                v = rng.standard_normal((40, 128)).astype(np.float32)
                beta = rng.standard_normal(40).astype(np.float32)
            log_alpha = rng.standard_normal((40, 128)).astype(np.float32)

            # Legacy copy=True (default astype)
            q_old = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32)
            k_old = bf16_to_f32(k) if k.dtype == np.uint16 else k.astype(np.float32)
            v_old = bf16_to_f32(v) if v.dtype == np.uint16 else v.astype(np.float32)
            log_alpha_old = log_alpha.astype(np.float32)
            beta_old = bf16_to_f32(beta) if beta.dtype == np.uint16 else beta.astype(np.float32)
            bundle_old = _pack_head_bundle_ref(q_old, k_old, np.exp(log_alpha_old), v_old, beta_old)

            # Optimized copy=False
            q_new = bf16_to_f32(q) if q.dtype == np.uint16 else q.astype(np.float32, copy=False)
            k_new = bf16_to_f32(k) if k.dtype == np.uint16 else k.astype(np.float32, copy=False)
            v_new = bf16_to_f32(v) if v.dtype == np.uint16 else v.astype(np.float32, copy=False)
            log_alpha_new = log_alpha.astype(np.float32, copy=False)
            beta_new = bf16_to_f32(beta) if beta.dtype == np.uint16 else beta.astype(np.float32, copy=False)
            bundle_new = _pack_head_bundle_ref(q_new, k_new, np.exp(log_alpha_new), v_new, beta_new)

            assert np.array_equal(bundle_new.view(np.uint32), bundle_old.view(np.uint32))


def test_weights_f32_cache_caches_and_reuses(monkeypatch):
    """Same bf16 tensor requested twice as f32 returns the same read-only object."""
    rng = np.random.default_rng(20261101)
    bf16_name = "model.layers.0.self_attn.q_proj.weight"
    f32_name = "model.layers.0.input_layernorm.weight"

    weights = _FakeWeights({
        bf16_name: f32_to_bf16_rtne(rng.standard_normal((32, 32)).astype(np.float32)),
        f32_name: rng.standard_normal(32).astype(np.float32),
    })

    # Stored as float32: to_float=True returns the stored array itself.
    assert weights.get(f32_name, to_float=True) is weights.tensors[f32_name]

    raw = weights.tensors[bf16_name]
    assert raw.dtype == np.uint16

    calls = 0
    orig_bf16_to_f32 = bf16_to_f32

    def counting_bf16_to_f32(arr):
        nonlocal calls
        calls += 1
        return orig_bf16_to_f32(arr)

    monkeypatch.setattr("host.weights.bf16_to_f32", counting_bf16_to_f32)

    first = weights.get(bf16_name, to_float=True)
    assert first.dtype == np.float32
    assert not first.flags.writeable
    assert first.flags.c_contiguous
    assert calls == 1

    second = weights.get(bf16_name, to_float=True)
    assert second is first
    assert calls == 1

    # Bit-identical to the old conversion path.
    expected = orig_bf16_to_f32(raw)
    assert np.array_equal(first, expected)


def test_weights_f32_cache_respects_size_limit(monkeypatch):
    """When the cache size limit is smaller than the tensor, conversion still works
    but the result is not cached (second call re-runs conversion)."""
    rng = np.random.default_rng(20261102)
    name = "model.layers.0.self_attn.q_proj.weight"
    weights = _FakeWeights({
        name: f32_to_bf16_rtne(rng.standard_normal((32, 32)).astype(np.float32)),
    })
    raw = weights.tensors[name]
    assert raw.dtype == np.uint16

    monkeypatch.setattr("host.weights.F32_CACHE_MAX_TENSOR_BYTES", raw.nbytes - 1)

    calls = 0
    orig_bf16_to_f32 = bf16_to_f32

    def counting_bf16_to_f32(arr):
        nonlocal calls
        calls += 1
        return orig_bf16_to_f32(arr)

    monkeypatch.setattr("host.weights.bf16_to_f32", counting_bf16_to_f32)

    first = weights.get(name, to_float=True)
    assert calls == 1
    second = weights.get(name, to_float=True)
    assert calls == 2
    assert first is not second
    assert np.array_equal(first, second)
    assert np.array_equal(first, orig_bf16_to_f32(raw))


def test_weights_f32_cache_to_float_false_returns_stored_array():
    """to_float=False returns the stored array object unchanged."""
    rng = np.random.default_rng(20261103)
    name = "model.layers.0.self_attn.q_proj.weight"
    raw = f32_to_bf16_rtne(rng.standard_normal((16, 16)).astype(np.float32))
    weights = _FakeWeights({name: raw})
    stored = weights.tensors[name]
    assert weights.get(name, to_float=False) is stored


def test_weights_f32_cache_preserves_zero_dim_shape():
    """0-dimensional bf16 tensors must keep shape () after f32 caching, not become (1,)."""
    import warnings

    weights = _FakeWeights({
        "alpha": np.array(0x3F80, dtype=np.uint16),
        "vec": f32_to_bf16_rtne(np.array([1.0, 2.0], dtype=np.float32)),
        "mat": f32_to_bf16_rtne(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)),
    })
    for name in ("alpha", "vec", "mat"):
        stored = weights.tensors[name]
        out = weights.get(name, to_float=True)
        assert out.shape == stored.shape, f"{name}: cached shape {out.shape} != stored shape {stored.shape}"
        assert out.dtype == np.float32
        assert not out.flags.writeable
        assert out.flags.c_contiguous

    # Calling float() on a 0-dimensional cached result must not emit a NumPy
    # deprecation warning about converting an array with shape (1,) to scalar.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        val = float(weights.get("alpha", to_float=True))
    assert val == bf16_to_f32(np.array(0x3F80, dtype=np.uint16)).item()


@pytest.mark.parametrize("fast", [False, True])
def test_qkv_batched_conv_matches_three_legacy_calls_bitwise(fast):
    """Row batching must preserve each depthwise kernel reduction exactly.

    Run for the reference and the fast batched path (DESIGN-prefill.md section 36).
    """
    rng = np.random.default_rng(601)
    cfg = RaonConfig(hidden_size=17, rda_num_heads=1, rda_head_dim=17)
    old = object.__new__(RaonModel)
    old.config = cfg
    old.conv_states = {}
    new = object.__new__(RaonModel)
    new.config = cfg
    new.conv_states = {}
    new._rda_conv_fast = fast
    new._rda_conv_fast_loop = None
    new._conv_fast = {}
    new._conv_wt = {}
    new._in_rda_overlap = False
    weights = [rng.standard_normal((17, 4)).astype(np.float32) for _ in range(3)]
    joined_weight = np.concatenate(weights, axis=0)

    edge_bits = np.array(
        [0x0000, 0x8000, 0x0001, 0x007F, 0x3F80, 0x7F80, 0xFF80, 0x7FC1],
        dtype=np.uint16,
    )
    for step in range(8):
        values = [rng.integers(0, 65536, size=17, dtype=np.uint16) for _ in range(3)]
        if step == 0:
            values[0][: edge_bits.size] = edge_bits
        with np.errstate(all="ignore"):
            expected = tuple(
                old._conv_step(0, kind, value, weight)
                for kind, value, weight in zip(("q", "k", "v"), values, weights)
            )
            actual = new._conv_step_qkv(0, *values, joined_weight)
        for expected_part, actual_part in zip(expected, actual):
            assert np.array_equal(expected_part, actual_part)
        for kind in ("q", "k", "v"):
            assert np.array_equal(
                old.conv_states[0][kind].view(np.uint32),
                new.conv_states[0][kind].view(np.uint32),
            )


def test_mla_f32_mirror_matches_legacy_full_cache_conversion():
    """Incremental FP32 mirrors retain old cache layout and attention bits."""
    rng = np.random.default_rng(602)
    backend = NumpyBackend(_DictWeights({}))
    old_k = None
    old_v = None
    scale = np.float64(7 ** -0.5)

    for step in range(9):
        q = rng.integers(0, 65536, size=(3, 7), dtype=np.uint16)
        k = rng.integers(0, 65536, size=(3, 7), dtype=np.uint16)
        v = rng.integers(0, 65536, size=(3, 5), dtype=np.uint16)
        if step == 0:
            q[0, :4] = np.array([0x0000, 0x8000, 0x7F80, 0x7FC1], dtype=np.uint16)
        k_token = k[:, np.newaxis, :]
        v_token = v[:, np.newaxis, :]
        old_k = k_token if old_k is None else np.concatenate([old_k, k_token], axis=1)
        old_v = v_token if old_v is None else np.concatenate([old_v, v_token], axis=1)
        with np.errstate(all="ignore"):
            q_f32 = bf16_to_f32(q)
            old_k_f32 = bf16_to_f32(old_k)
            old_v_f32 = bf16_to_f32(old_v)
            scores = np.einsum("hd,htd->ht", q_f32, old_k_f32) * scale
            exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
            expected = f32_to_bf16_rtne(
                np.einsum("ht,htd->hd", attn, old_v_f32).reshape(-1)
            )
            actual = backend.mla_attend(2, q, k, v, scale)
        assert np.array_equal(expected, actual)
        assert np.array_equal(backend.mla_key_cache[2], old_k)
        assert np.array_equal(backend.mla_value_cache[2], old_v)
        cache2 = backend._mla_caches[2]
        assert np.array_equal(
            cache2.k_f32[:, :cache2.length, :].view(np.uint32), old_k_f32.view(np.uint32)
        )
        assert np.array_equal(
            cache2.v_f32[:, :cache2.length, :].view(np.uint32), old_v_f32.view(np.uint32)
        )

    restored = NumpyBackend(_DictWeights({}))
    restored.set_mla_cache(2, old_k[np.newaxis], old_v[np.newaxis])
    rest_cache = restored._mla_caches[2]
    assert np.array_equal(
        rest_cache.k_f32[:, :rest_cache.length, :].view(np.uint32), bf16_to_f32(old_k).view(np.uint32)
    )


def test_mhc_reused_f32_and_site_cache_match_legacy_formula():
    rng = np.random.default_rng(603)
    n, c = 4, 13
    prefix = "site"
    tensors = {
        f"{prefix}.phi_pre": rng.standard_normal((n, n * c)).astype(np.float32),
        f"{prefix}.phi_post": rng.standard_normal((n, n * c)).astype(np.float32),
        f"{prefix}.res.phi_z": rng.standard_normal(((n - 1) ** 2, n * c)).astype(np.float32),
        f"{prefix}.alpha_pre": np.array(0.75, dtype=np.float32),
        f"{prefix}.alpha_post": np.array(-0.5, dtype=np.float32),
        f"{prefix}.res.alpha_z": np.array(0.25, dtype=np.float32),
        f"{prefix}.b_pre": rng.standard_normal(n).astype(np.float32),
        f"{prefix}.b_post": rng.standard_normal(n).astype(np.float32),
        f"{prefix}.res.b_z": rng.standard_normal((n - 1) ** 2).astype(np.float32),
        f"{prefix}.res.eta": np.array(-0.2, dtype=np.float32),
    }

    class CountingWeights(_DictWeights):
        def __init__(self, values):
            super().__init__(values)
            self.calls = 0

        def get(self, name, to_float=True):
            self.calls += 1
            return super().get(name, to_float)

    weights = CountingWeights(tensors)
    cfg = RaonConfig(hidden_size=c, raon_mhc_streams=n)
    model = RaonModel(cfg, weights, NumpyBackend(weights))
    stream = f32_to_bf16_rtne(rng.standard_normal((n, c)).astype(np.float32))
    stream_f32 = bf16_to_f32(stream)

    first = model._route_mhc(stream, prefix, stream_f32)
    calls_after_first = weights.calls
    second = model._route_mhc(stream, prefix, stream_f32)
    assert weights.calls == calls_after_first == 10
    for a, b in zip(first, second):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))

    weight = np.concatenate(
        [tensors[f"{prefix}.phi_pre"], tensors[f"{prefix}.phi_post"], tensors[f"{prefix}.res.phi_z"]]
    )
    inv = 1.0 / np.sqrt(np.mean(stream_f32.reshape(-1) ** 2) + cfg.raon_mhc_selector_eps)
    z_f32 = bf16_to_f32(f32_to_bf16_rtne(np.dot(stream_f32.reshape(-1), weight.T))) * inv
    t_pre = float(tensors[f"{prefix}.alpha_pre"]) * z_f32[:n] + tensors[f"{prefix}.b_pre"]
    t_post = float(tensors[f"{prefix}.alpha_post"]) * z_f32[n : 2 * n] + tensors[f"{prefix}.b_post"]
    t_z = float(tensors[f"{prefix}.res.alpha_z"]) * z_f32[2 * n :] + tensors[f"{prefix}.res.b_z"]
    expected_pre = 1.0 / (1.0 + np.exp(-t_pre))
    expected_post = cfg.raon_mhc_post_factor * (1.0 / (1.0 + np.exp(-t_post)))
    helmert, eye, chart_b = __import__("host.raon", fromlist=["get_mhc_constants"]).get_mhc_constants(n)
    e = np.dot(helmert, np.dot(t_z.reshape(n - 1, n - 1), helmert.T))
    rho = max(0.0, float(np.max(-e / chart_b)))
    phi = chart_b + e / (1.0 + rho)
    gamma = 1.0 / (1.0 + np.exp(-float(tensors[f"{prefix}.res.eta"])))
    expected_res = (1.0 - gamma) * eye + gamma * phi
    for actual, expected in zip(first, (expected_pre, expected_post, expected_res)):
        assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))

    h_pre, h_post, h_res = first
    branch = rng.integers(0, 65536, size=c, dtype=np.uint16)
    assert np.array_equal(
        model._mhc_aggregate(stream, h_pre, stream_f32),
        model._mhc_aggregate(stream, h_pre),
    )
    current = model._mhc_mix(stream, branch, h_post, h_res, stream_f32)
    assert np.array_equal(current, model._mhc_mix(stream, branch, h_post, h_res))
    stale = model._mhc_mix(stream, branch, h_post, h_res, np.roll(stream_f32, 1, axis=0))
    assert np.count_nonzero(current != stale) > 0


def test_step_passes_current_not_previous_mhc_f32_copy():
    """Every site must receive the FP32 mirror of that site's uint16 stream."""
    cfg = RaonConfig(
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=8,
        raon_mhc_streams=2,
        num_experts=0,
        rda_num_heads=1,
        rda_head_dim=4,
        rda_layers=[0],
        mla_layers=[],
    )
    weights = _DictWeights(
        {
            "model.embed_tokens.weight": f32_to_bf16_rtne(
                np.arange(32, dtype=np.float32).reshape(8, 4) / 8.0
            ),
            "model.norm.weight": np.ones(4, dtype=np.float32),
        }
    )

    class FinalBackend(NumpyBackend):
        def linear(self, name, x):
            assert name == "lm_head.weight"
            return np.resize(x, cfg.vocab_size).astype(np.uint16, copy=False)

    class TrackingModel(RaonModel):
        def _route_mhc(self, stream, site_prefix, stream_f32=None):
            expected = bf16_to_f32(stream)
            assert stream_f32 is not None
            assert np.array_equal(stream_f32.view(np.uint32), expected.view(np.uint32))
            return (
                np.ones(cfg.raon_mhc_streams, dtype=np.float32),
                np.ones(cfg.raon_mhc_streams, dtype=np.float32),
                np.eye(cfg.raon_mhc_streams, dtype=np.float32),
            )

        def _mhc_aggregate(self, stream, h_pre, stream_f32=None):
            return stream[0].copy()

        def _mhc_mix(self, stream, branch_out, h_post, h_res, stream_f32=None):
            return f32_to_bf16_rtne(bf16_to_f32(stream) + np.float32(1.0))

        def _layer_constants(self, layer_idx):
            return {"input_norm": np.ones(4, np.float32), "post_norm": np.ones(4, np.float32)}

        def _forward_rda(self, layer_idx, hidden):
            return hidden

        def _forward_dense_mlp(self, layer_idx, hidden):
            return hidden

    model = TrackingModel(cfg, weights, FinalBackend(weights))
    logits = model.step(1)
    assert logits.shape == (cfg.vocab_size,)


# =============================================================================
# npu-opt6 fix1: MLA preallocated growing cache tests
# =============================================================================

class _LegacyMlaReference:
    """af44167 legacy MLA reference: concatenate u16 caches and full FP32 conversion each token."""

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


def test_mla_prealloc_bit_identical_across_growth_boundaries():
    """Verify preallocated MLA cache is bit-identical to legacy af44167 across growth boundaries.

    Includes special float values (±0, subnormals, ±inf, NaN) and runs through
    multiple buffer growth points (initial cap 257, 1st growth at 258 -> cap 513,
    2nd growth at 514 -> cap 769).
    Also checks get_mla_cache immutability across subsequent steps and growths.
    """
    backend = NumpyBackend(_DictWeights({}))
    legacy = _LegacyMlaReference()
    rng = np.random.default_rng(701)

    heads, dk, dv = 4, 16, 12
    scale = 0.25
    layer = 1

    special_vals = np.array([
        0x0000,  # +0
        0x8000,  # -0
        0x0001,  # +subnormal
        0x007F,  # subnormal edge
        0x8001,  # -subnormal
        0x7F80,  # +inf
        0xFF80,  # -inf
        0x7FC0,  # qNaN
        0x7FC1,  # NaN payload
        0xFFC0,  # -NaN
    ], dtype=np.uint16)

    saved_snap_250 = None
    saved_snap_250_data = None
    saved_snap_510 = None
    saved_snap_510_data = None

    # Initial capacity with 1 token: 1 + max(256, 0) = 257.
    # Growth 1 occurs at step 258 (cap becomes 257 + 256 = 513).
    # Growth 2 occurs at step 514 (cap becomes 513 + 256 = 769).
    total_steps = 540

    for step in range(total_steps):
        q = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        k = rng.integers(0, 65536, size=(heads, dk), dtype=np.uint16)
        v = rng.integers(0, 65536, size=(heads, dv), dtype=np.uint16)

        # Inject special values cyclically into head 0
        inj_idx = step % len(special_vals)
        q[0, 0] = special_vals[inj_idx]
        k[0, 0] = special_vals[(inj_idx + 1) % len(special_vals)]
        v[0, 0] = special_vals[(inj_idx + 2) % len(special_vals)]

        with np.errstate(all="ignore"):
            out_act = backend.mla_attend(layer, q, k, v, scale)
            out_ref, ref_k, ref_v, ref_kf, ref_vf = legacy.attend(q, k, v, scale)

        # 1. Output uint16 bit-identity
        assert np.array_equal(out_act.view(np.uint16), out_ref.view(np.uint16)), f"Output mismatch at step {step}"

        # 2. Key/Value uint16 cache bit-identity
        k_act, v_act = backend.mla_key_cache[layer], backend.mla_value_cache[layer]
        assert np.array_equal(k_act, ref_k), f"uint16 key cache mismatch at step {step}"
        assert np.array_equal(v_act, ref_v), f"uint16 value cache mismatch at step {step}"

        # 3. FP32 mirror bit-identity
        layer_cache = backend._mla_caches[layer]
        act_kf = layer_cache.k_f32[:, :layer_cache.length, :]
        act_vf = layer_cache.v_f32[:, :layer_cache.length, :]
        assert np.array_equal(act_kf.view(np.uint32), ref_kf.view(np.uint32)), f"FP32 key mirror mismatch at step {step}"
        assert np.array_equal(act_vf.view(np.uint32), ref_vf.view(np.uint32)), f"FP32 value mirror mismatch at step {step}"

        # Check capacity at known boundary points
        if step == 255:  # length 256: first capacity - 1
            assert layer_cache.capacity == 257
        elif step == 256:  # length 257: first capacity (full)
            assert layer_cache.capacity == 257
        elif step == 257:  # length 258: 1st growth triggered
            assert layer_cache.capacity == 513
        elif step == 512:  # length 513: second capacity (full)
            assert layer_cache.capacity == 513
        elif step == 513:  # length 514: 2nd growth triggered
            assert layer_cache.capacity == 769

        # Snapshot for get_mla_cache immutability test
        if step == 249:  # length 250
            k_g, v_g = backend.get_mla_cache(layer)
            assert k_g is not None and v_g is not None
            saved_snap_250 = (k_g, v_g)
            saved_snap_250_data = (k_g.copy(), v_g.copy())
        if step == 259:  # after 1st growth
            assert saved_snap_250 is not None and saved_snap_250_data is not None
            assert np.array_equal(saved_snap_250[0], saved_snap_250_data[0])
            assert np.array_equal(saved_snap_250[1], saved_snap_250_data[1])

        if step == 509:  # length 510
            k_g, v_g = backend.get_mla_cache(layer)
            assert k_g is not None and v_g is not None
            saved_snap_510 = (k_g, v_g)
            saved_snap_510_data = (k_g.copy(), v_g.copy())
        if step == 519:  # after 2nd growth
            assert saved_snap_510 is not None and saved_snap_510_data is not None
            assert np.array_equal(saved_snap_510[0], saved_snap_510_data[0])
            assert np.array_equal(saved_snap_510[1], saved_snap_510_data[1])


def test_mla_prealloc_set_cache_reset_and_serve_restore_parity():
    """Verify set_mla_cache, reset, and serve engine checkpoint restore path."""
    backend = NumpyBackend(_DictWeights({}))
    legacy = _LegacyMlaReference()
    rng = np.random.default_rng(702)

    heads, dk, dv = 3, 14, 10
    scale = 0.2
    layer = 0

    # 1. set_mla_cache -> mla_attend parity
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

    # 2. serve engine restore simulation (slice to target_len and set_mla_cache)
    target_len = 25
    k_cache, v_cache = backend.get_mla_cache(layer)
    assert k_cache is not None and v_cache is not None
    # engine.py line 411-417 logic
    if k_cache.ndim == 4:
        k_sliced = np.ascontiguousarray(k_cache[:, :, :target_len, :])
        v_sliced = np.ascontiguousarray(v_cache[:, :, :target_len, :])
    else:
        k_sliced = np.ascontiguousarray(k_cache[:, :target_len, :])
        v_sliced = np.ascontiguousarray(v_cache[:, :target_len, :])

    backend.set_mla_cache(layer, k_sliced, v_sliced)
    legacy.set_cache(k_sliced, v_sliced)

    assert backend._mla_caches[layer].length == target_len

    # Continue generating from restored state
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

    # 3. reset -> mla_attend parity
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


def test_raon_model_advance_need_logits() -> None:
    """RaonModel.advance(token_id, need_logits)가 need_logits=False일 때
    최종 norm 및 lm_head를 건너뛰고 None을 반환하며, last_intermediates는
    정상 기록되고, need_logits=True일 때는 step()과 동일한 logits를 반환함을 검증한다.
    """
    cfg = RaonConfig(
        hidden_size=4,
        num_hidden_layers=1,
        vocab_size=8,
        raon_mhc_streams=2,
        num_experts=0,
        rda_num_heads=1,
        rda_head_dim=4,
        rda_layers=[0],
        mla_layers=[],
    )
    weights = _DictWeights(
        {
            "model.embed_tokens.weight": f32_to_bf16_rtne(
                np.arange(32, dtype=np.float32).reshape(8, 4) / 8.0
            ),
            "model.norm.weight": np.ones(4, dtype=np.float32),
        }
    )

    class CountingBackend(NumpyBackend):
        def __init__(self, w: Weights) -> None:
            super().__init__(w)
            self.lm_head_calls = 0

        def linear(self, name: str, x: np.ndarray) -> np.ndarray:
            if name == "lm_head.weight":
                self.lm_head_calls += 1
                return np.resize(x, cfg.vocab_size).astype(np.uint16, copy=False)
            return super().linear(name, x)

    class MinimalModel(RaonModel):
        def _route_mhc(self, stream, site_prefix, stream_f32=None):
            return (
                np.ones(cfg.raon_mhc_streams, dtype=np.float32),
                np.ones(cfg.raon_mhc_streams, dtype=np.float32),
                np.eye(cfg.raon_mhc_streams, dtype=np.float32),
            )

        def _mhc_aggregate(self, stream, h_pre, stream_f32=None):
            return stream[0].copy()

        def _mhc_mix(self, stream, branch_out, h_post, h_res, stream_f32=None):
            return f32_to_bf16_rtne(bf16_to_f32(stream) + np.float32(1.0))

        def _layer_constants(self, layer_idx):
            return {"input_norm": np.ones(4, np.float32), "post_norm": np.ones(4, np.float32)}

        def _forward_rda(self, layer_idx, hidden):
            return hidden

        def _forward_dense_mlp(self, layer_idx, hidden):
            return hidden

    be = CountingBackend(weights)
    model = MinimalModel(cfg, weights, be)

    # 1. need_logits=False
    ret_none = model.advance(1, need_logits=False)
    assert ret_none is None
    assert be.lm_head_calls == 0
    # last_intermediates must be preserved
    assert "layer_000.streams_in" in model.last_intermediates
    assert "layer_000.mixer_in" in model.last_intermediates
    assert "layer_000.mixer_out" in model.last_intermediates
    assert "layer_000.streams_out" in model.last_intermediates

    # 2. need_logits=True
    ret_logits = model.advance(1, need_logits=True)
    assert ret_logits is not None
    assert ret_logits.shape == (cfg.vocab_size,)
    assert ret_logits.dtype == np.float32
    assert be.lm_head_calls == 1

    # 3. step() returns identical logits and also calls lm_head
    step_logits = model.step(1)
    assert be.lm_head_calls == 2
    assert np.array_equal(ret_logits, step_logits)
