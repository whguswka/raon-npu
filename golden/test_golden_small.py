# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

import dump
import rda_bf16_drift


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = Path("__RAON_SOURCE_DIR_NOT_SET__")


def run_cli(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *map(str, arguments)],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=160,
    )


def load_npz_tensor(data: np.lib.npyio.NpzFile, dtypes: dict[str, str], name: str) -> torch.Tensor:
    value = torch.from_numpy(np.array(data[name], copy=True))
    if dtypes[name] == "bfloat16":
        return value.view(torch.bfloat16)
    return value.to(getattr(torch, dtypes[name]))


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    source = Path(os.environ.get("RAON_SOURCE_DIR", DEFAULT_SOURCE))
    if not (source / "modeling_raon1_mini.py").is_file():
        pytest.skip(f"Raon source is unavailable: {source}")
    target = tmp_path_factory.mktemp("tiny_raon")
    shutil.copy2(source / "modeling_raon1_mini.py", target)
    shutil.copy2(source / "configuration_raon1_mini.py", target)
    configuration, modeling = dump.load_local_raon(target)
    torch.manual_seed(1234)
    config = configuration.Raon1MiniConfig(
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
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
            "rda_layers": [1, 3],
            "mla_layers": [2, 4],
        },
        pad_token_id=0,
        eos_token_id=[2],
        max_position_embeddings=64,
    )
    model = modeling.Raon1MiniForCausalLM(config).to(torch.bfloat16).eval()
    model.save_pretrained(target, safe_serialization=True)
    (target / "prompt.json").write_text(json.dumps({"input_ids": [11, 12, 13, 14, 15]}), encoding="utf-8")
    return target


def test_dump_cli_artifacts_and_deterministic_hash(tiny_model_dir: Path, tmp_path: Path) -> None:
    outputs = []
    for name in ("first", "second"):
        out = tmp_path / name
        completed = run_cli(
            ROOT / "dump.py",
            "--model-dir", tiny_model_dir,
            "--prompt-file", tiny_model_dir / "prompt.json",
            "--decode-steps", 2,
            "--state-steps", "1",
            "--out", out,
            "--max-gb", "0.25",
            "--threads", "1",
        )
        assert "estimated dump size:" in completed.stdout
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        assert len(manifest["steps"]) == 2
        assert len(manifest["generated_token_ids"]) == 2
        assert manifest["selected_layers"] == [0, 1, 2, 3]
        assert manifest["state_snapshot_steps"] == [1]
        assert manifest["expanded_kv_cache"] is False
        assert [step["step"] for step in manifest["steps"]] == [1, 2]
        assert manifest["model_fingerprint"]["mode"] == "full"
        assert all((out / item["file"]).is_file() for item in manifest["files"])
        with np.load(out / "step_000001.npz", allow_pickle=False) as data:
            assert data["layer_000.streams_in"].shape == (1, 5, 4, 64)
            assert data["layer_000.mixer_in"].shape == (1, 5, 64)
            assert data["layer_001.router_logits"].shape == (5, 8)
            assert data["layer_001.router_top"].shape == (5, 2)
            assert data["logits_top20_ids"].shape == (1, 20)
            assert "layer_000.rda_state" in data
            assert data["layer_001.mla_kv_lat_cache"].shape == (1, 5, 32)
            assert data["layer_001.mla_k_pe_cache"].shape == (1, 5, 8)
            assert "layer_001.mla_key_cache" not in data
        with np.load(out / "step_000002.npz", allow_pickle=False) as data:
            assert "layer_001.mla_kv_lat_cache" not in data
            assert "layer_001.mla_k_pe_cache" not in data
        mla_summary = manifest["steps"][1]["states"]["1"]
        assert mla_summary["kind"] == "mla_latent_cache"
        assert mla_summary["kv_lat"]["shape"] == [1, 6, 32]
        assert mla_summary["k_pe"]["shape"] == [1, 6, 8]
        assert {"norm", "sha256"} <= mla_summary["kv_lat"].keys()
        assert {"norm", "sha256"} <= mla_summary["k_pe"].keys()
        assert "expanded_kv" not in mla_summary
        outputs.append(manifest)
        assert manifest["actual_dump_bytes"] == sum(
            path.stat().st_size for path in out.iterdir() if path.is_file()
        )
    assert [step["logits"]["sha256"] for step in outputs[0]["steps"]] == [
        step["logits"]["sha256"] for step in outputs[1]["steps"]
    ]
    assert outputs[0]["generated_token_ids"] == outputs[1]["generated_token_ids"]


def test_rda_conv_state_snapshot_shape_and_step_boundary(tiny_model_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "rda-conv"
    run_cli(
        ROOT / "dump.py",
        "--model-dir", tiny_model_dir,
        "--prompt-file", tiny_model_dir / "prompt.json",
        "--decode-steps", 2,
        "--state-steps", "1",
        "--out", out,
        "--max-gb", "0.25",
        "--threads", "1",
    )
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    rda_summary = manifest["steps"][0]["states"]["0"]
    assert rda_summary["kind"] == "rda_recurrent"
    # The step-1 forward already folded this step's tokens into conv_states and
    # state_records runs after that forward, so the snapshot is the step-2 start.
    conv = rda_summary["conv_state"]
    # tiny config: rda num_heads 4 * head_dim 16 = proj 64, conv_kernel_size 4
    assert conv["shape"] == [1, 3 * 64, 4 - 1]
    assert conv["dtype"] == "bfloat16"
    assert {"norm", "sha256"} <= conv.keys()
    with np.load(out / "step_000001.npz", allow_pickle=False) as data:
        dtypes = json.loads(str(data["__dtypes_json__"].item()))
        assert dtypes["layer_000.rda_conv_state"] == "bfloat16"
        saved = data["layer_000.rda_conv_state"]
        assert saved.shape == (1, 192, 3)
        assert conv["shape"] == list(saved.shape)
        assert conv["sha256"] == hashlib.sha256(saved.tobytes()).hexdigest()
    with np.load(out / "step_000002.npz", allow_pickle=False) as data:
        assert "layer_000.rda_conv_state" not in data

    # Independently prove the timestamp: restore only the dumped step-1 states,
    # process the next input token, and compare with an uninterrupted step 2.
    model, config = dump.load_model(tiny_model_dir, tiny_model_dir)
    _, modeling = dump.load_local_raon(tiny_model_dir)
    restored_cache = modeling.Raon1MiniDynamicCache(config)
    with np.load(out / "step_000001.npz", allow_pickle=False) as data:
        dtypes = json.loads(str(data["__dtypes_json__"].item()))
        for idx, layer in enumerate(model.model.layers):
            prefix = f"layer_{idx:03d}"
            if layer.is_rda:
                restored_cache.recurrent_states[idx] = load_npz_tensor(
                    data, dtypes, f"{prefix}.rda_state"
                )
                restored_cache.conv_states[idx] = load_npz_tensor(
                    data, dtypes, f"{prefix}.rda_conv_state"
                )
                continue
            kv_lat = load_npz_tensor(data, dtypes, f"{prefix}.mla_kv_lat_cache")
            k_pe = load_npz_tensor(data, dtypes, f"{prefix}.mla_k_pe_cache")
            attention = layer.self_attn
            kv = attention.kv_b_proj(kv_lat).reshape(
                kv_lat.shape[0],
                kv_lat.shape[1],
                attention.num_heads,
                attention.qk_nope + attention.v_dim,
            )
            k_nope, value = torch.split(kv, [attention.qk_nope, attention.v_dim], dim=-1)
            key = torch.cat(
                [
                    k_nope,
                    k_pe.unsqueeze(2).expand(
                        kv_lat.shape[0], kv_lat.shape[1], attention.num_heads, attention.qk_pe
                    ),
                ],
                dim=-1,
            )
            restored_cache.key_cache[idx] = key.transpose(1, 2)
            restored_cache.value_cache[idx] = value.transpose(1, 2)

    prompt_ids = torch.tensor([manifest["prompt_token_ids"]], dtype=torch.long)
    next_ids = torch.tensor([[manifest["generated_token_ids"][0]]], dtype=torch.long)
    next_mask = torch.ones((1, prompt_ids.shape[1] + 1), dtype=torch.long)
    with torch.inference_mode():
        first = model(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        uninterrupted = model(
            input_ids=next_ids,
            attention_mask=next_mask,
            past_key_values=first.past_key_values,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        ).logits
        restored = model(
            input_ids=next_ids,
            attention_mask=next_mask,
            past_key_values=restored_cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        ).logits
    assert uninterrupted.dtype == restored.dtype
    assert torch.equal(uninterrupted.view(torch.uint16), restored.view(torch.uint16))


def test_capture_rda_inputs_and_drift_cli(tiny_model_dir: Path, tmp_path: Path) -> None:
    capture = tmp_path / "capture"
    run_cli(
        ROOT / "dump.py",
        "--model-dir", tiny_model_dir,
        "--prompt-file", tiny_model_dir / "prompt.json",
        "--decode-steps", 0,
        "--capture-rda-inputs",
        "--out", capture,
        "--max-gb", "0.25",
    )
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["capture_rda_inputs_only"] is True
    rda_files = sorted(capture.glob("rda_inputs_layer_*.npz"))
    assert [path.name for path in rda_files] == [
        "rda_inputs_layer_000.npz",
        "rda_inputs_layer_002.npz",
    ]
    with np.load(rda_files[0], allow_pickle=False) as data:
        dtypes = json.loads(str(data["__dtypes_json__"].item()))
        assert set(dtypes) == {"q", "k", "v", "log_alpha", "beta"}
        assert dtypes["q"] == "bfloat16"
        assert dtypes["log_alpha"] == "float32"

    report_path = tmp_path / "drift.json"
    run_cli(
        ROOT / "rda_bf16_drift.py",
        "--input", capture,
        "--positions", "1,3,5",
        "--out", report_path,
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["mode"] == "capture"
    assert len(report["results"]) == 2
    for result in report["results"]:
        assert [row["token_position"] for row in result["metrics"]] == [1, 3, 5]
        assert all(row["output_relative_error"] >= 0 for row in result["metrics"])
        assert all(-1.00001 <= row["output_cosine"] <= 1.00001 for row in result["metrics"])
        assert all(row["state_nrmse"] >= 0 for row in result["metrics"])


def test_synthetic_drift_is_deterministic(tmp_path: Path) -> None:
    reports = []
    for idx in range(2):
        out = tmp_path / f"synthetic-{idx}.json"
        run_cli(
            ROOT / "rda_bf16_drift.py",
            "--synthetic",
            "--tokens", 16,
            "--heads", 2,
            "--dim", 8,
            "--positions", "1,8,16",
            "--seed", 9,
            "--out", out,
        )
        reports.append(json.loads(out.read_text(encoding="utf-8")))
    assert reports[0] == reports[1]
    assert reports[0]["experiment"] == "state rounded to bfloat16 after every token update"


def test_synthetic_mxfp8_loader_is_bit_exact_and_logits_match(
    tiny_model_dir: Path, tmp_path: Path
) -> None:
    mx_dir = tmp_path / "mxfp8"
    mx_dir.mkdir()
    for name in ("modeling_raon1_mini.py", "configuration_raon1_mini.py"):
        shutil.copy2(tiny_model_dir / name, mx_dir / name)
    configuration, modeling = dump.load_local_raon(mx_dir)
    source_model, source_config = dump.load_model(tiny_model_dir, tiny_model_dir)
    expected = {name: value.detach().clone() for name, value in source_model.state_dict().items()}
    stored: dict[str, torch.Tensor] = {}
    paired_weights: list[str] = []
    for name, value in expected.items():
        if ".moe.experts." in name and name.endswith(".weight"):
            quantized, scale = modeling.quantize_mxfp8(value)
            stored[name] = quantized
            stored[f"{name}_scale_inv"] = scale
            expected[name] = modeling.dequantize_mxfp8(
                quantized, scale, torch.bfloat16
            )
            paired_weights.append(name)
        else:
            stored[name] = value
    save_file(stored, mx_dir / "model.safetensors")
    config_payload = source_config.to_dict()
    config_payload["quantization_config"] = {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "dequantize": False,
    }
    (mx_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    loaded, plain_config = dump.load_model(mx_dir, mx_dir)
    assert getattr(plain_config, "quantization_config", None) is None
    loaded_state = loaded.state_dict()
    assert paired_weights
    for name in paired_weights:
        assert loaded_state[name].dtype == torch.bfloat16
        assert torch.equal(loaded_state[name].view(torch.uint16), expected[name].view(torch.uint16))

    bf16_reference = modeling.Raon1MiniForCausalLM(plain_config).to(torch.bfloat16).eval()
    bf16_reference.load_state_dict(expected, strict=True)
    input_ids = torch.tensor([[11, 12, 13]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        actual_logits = loaded(input_ids=input_ids, attention_mask=attention_mask).logits
        expected_logits = bf16_reference(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
    assert torch.equal(actual_logits.view(torch.uint16), expected_logits.view(torch.uint16))


def test_causal_mask_padding_and_decode_k_greater_than_q(tiny_model_dir: Path) -> None:
    _, modeling = dump.load_local_raon(tiny_model_dir)
    key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [1.0, 2.0]]]])
    value = torch.arange(10, dtype=torch.float32).view(1, 1, 5, 2)

    prefill_query = key.clone()
    prefill_padding = torch.tensor([[0, 0, 1, 1, 1]], dtype=torch.long)
    prefill_mask = modeling.create_causal_mask(
        config=None,
        inputs_embeds=torch.zeros(1, 5, 4),
        attention_mask=prefill_padding,
        past_key_values=None,
    )
    prefill_explicit = (
        torch.ones(5, 5, dtype=torch.bool).tril()[None, None]
        & prefill_padding.bool()[:, None, None, :]
    )
    assert torch.equal(prefill_mask, prefill_explicit)
    prefill_actual = torch.nn.functional.scaled_dot_product_attention(
        prefill_query, key, value, attn_mask=prefill_mask
    )
    prefill_oracle = torch.nn.functional.scaled_dot_product_attention(
        prefill_query, key, value, attn_mask=prefill_explicit
    )
    assert torch.equal(prefill_actual, prefill_oracle)

    query = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    padding = torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.long)
    inputs_embeds = torch.zeros(1, 2, 4)
    mask = modeling.create_causal_mask(
        config=None,
        inputs_embeds=inputs_embeds,
        attention_mask=padding,
        past_key_values=object(),
    )
    explicit = torch.tensor(
        [[[[False, True, True, True, False], [False, True, True, True, True]]]]
    )
    assert torch.equal(mask, explicit)
    actual = torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    oracle = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=explicit
    )
    assert torch.equal(actual, oracle)


def _cache_tensors(cache: object) -> list[torch.Tensor]:
    values: list[torch.Tensor] = []
    for attribute in ("recurrent_states", "key_cache", "value_cache"):
        for value in getattr(cache, attribute):
            if value is not None:
                values.append(value)
    return values


def test_capture_hooks_preserve_logits_and_cache_bits(tiny_model_dir: Path) -> None:
    model, _ = dump.load_model(tiny_model_dir, tiny_model_dir)
    ids = torch.tensor([[11, 12, 13]], dtype=torch.long)
    mask = torch.ones_like(ids)
    expected_latent: dict[int, dict[str, torch.Tensor]] = {}
    pending_k_pe: dict[int, torch.Tensor] = {}
    oracle_handles = []
    for idx, layer in enumerate(model.model.layers):
        if layer.is_rda:
            continue
        attention = layer.self_attn

        def projection_post(module, args, output, layer_idx=idx, attn=attention):
            _raw_kv_lat, k_pe = torch.split(
                output, [attn.kv_lora_rank, attn.qk_pe], dim=-1
            )
            pending_k_pe[layer_idx] = k_pe.detach().cpu().clone()

        def norm_post(module, args, output, layer_idx=idx):
            expected_latent[layer_idx] = {
                "kv_lat": output.detach().cpu().clone(),
                "k_pe": pending_k_pe.pop(layer_idx),
            }

        oracle_handles.append(
            attention.kv_a_proj_with_mqa.register_forward_hook(projection_post)
        )
        oracle_handles.append(attention.kv_a_layernorm.register_forward_hook(norm_post))
    try:
        with torch.inference_mode():
            before = model(
                input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True
            )
    finally:
        for handle in oracle_handles:
            handle.remove()
    recorder = dump.CaptureRecorder(set(range(len(model.model.layers))))
    recorder.attach(model)
    try:
        with torch.inference_mode():
            after = model(
                input_ids=ids, attention_mask=mask, use_cache=True, return_dict=True
            )
    finally:
        recorder.close()
    assert torch.equal(before.logits.view(torch.uint16), after.logits.view(torch.uint16))
    before_cache = _cache_tensors(before.past_key_values)
    after_cache = _cache_tensors(after.past_key_values)
    assert len(before_cache) == len(after_cache)
    assert all(torch.equal(left.view(torch.uint8), right.view(torch.uint8)) for left, right in zip(before_cache, after_cache))
    assert recorder.mla_latent_cache.keys() == expected_latent.keys()
    for idx, expected in expected_latent.items():
        actual = recorder.mla_latent_cache[idx]
        assert torch.equal(
            actual["kv_lat"].view(torch.uint16), expected["kv_lat"].view(torch.uint16)
        )
        assert torch.equal(
            actual["k_pe"].view(torch.uint16), expected["k_pe"].view(torch.uint16)
        )


def test_snapshot_steps_are_one_based_and_default_includes_last() -> None:
    assert dump.parse_snapshot_steps(None, 64) == {1, 16, 64}
    assert dump.parse_snapshot_steps(None, 20) == {1, 16, 20}
    assert dump.parse_snapshot_steps("1,16,64", 16) == {1, 16}
    with pytest.raises(ValueError, match="1-based"):
        dump.parse_snapshot_steps("0,1", 4)


def test_expanded_mla_kv_cache_is_opt_in() -> None:
    cache = type(
        "Cache",
        (),
        {
            "recurrent_states": [None],
            "key_cache": [torch.ones(1, 2, 3, 4, dtype=torch.bfloat16)],
            "value_cache": [torch.ones(1, 2, 3, 5, dtype=torch.bfloat16)],
        },
    )()
    latent = {
        0: {
            "kv_lat": torch.ones(1, 3, 6, dtype=torch.bfloat16),
            "k_pe": torch.ones(1, 3, 2, dtype=torch.bfloat16),
        }
    }
    default_tensors: dict[str, torch.Tensor] = {}
    default = dump.state_records(cache, latent, [0], True, default_tensors)
    assert "expanded_kv" not in default["0"]
    assert "layer_000.mla_key_cache" not in default_tensors

    expanded_tensors: dict[str, torch.Tensor] = {}
    expanded = dump.state_records(
        cache, latent, [0], True, expanded_tensors, expanded_kv_cache=True
    )
    assert expanded["0"]["expanded_kv"]["key"]["shape"] == [1, 2, 3, 4]
    assert "layer_000.mla_key_cache" in expanded_tensors
    assert "layer_000.mla_value_cache" in expanded_tensors


def test_npz_budget_rejected_before_write_and_cleans_temporary(tmp_path: Path) -> None:
    target = tmp_path / "too-large.npz"
    tensors = {"value": torch.zeros(1024, dtype=torch.float32)}
    budget = dump.Budget(dump.projected_npz_bytes(tensors) - 1)
    with pytest.raises(RuntimeError, match="before writing"):
        dump.save_tensor_npz(target, tensors, budget)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp*"))
    assert budget.written == 0
    assert budget.reserved == 0


def test_drift_one_token_matches_direct_oracle() -> None:
    inputs = {
        "q": torch.tensor([[[[0.25, -0.5]]]], dtype=torch.bfloat16),
        "k": torch.tensor([[[[0.75, 0.125]]]], dtype=torch.bfloat16),
        "v": torch.tensor([[[[1.25, -0.75]]]], dtype=torch.bfloat16),
        "log_alpha": torch.tensor([[[[-0.2, -0.3]]]], dtype=torch.float32),
        "beta": torch.tensor([[[0.6]]], dtype=torch.bfloat16),
    }
    result = rda_bf16_drift.compare(inputs, [1])[0]
    q, k, v = (inputs[name].float()[:, 0] for name in ("q", "k", "v"))
    beta = inputs["beta"].float()[:, 0][..., None, None]
    reference = beta * torch.einsum("bhv,bhk->bhvk", v, k)
    rounded = reference.to(torch.bfloat16).float()
    output_reference = torch.einsum("bhvk,bhk->bhv", reference, q).to(torch.bfloat16).float()
    output_rounded = torch.einsum("bhvk,bhk->bhv", rounded, q).to(torch.bfloat16).float()
    expected_relative = torch.linalg.vector_norm(output_rounded - output_reference) / torch.linalg.vector_norm(output_reference)
    expected_state = torch.sqrt(torch.mean((rounded - reference).square())) / torch.sqrt(torch.mean(reference.square()))
    expected_cosine = torch.nn.functional.cosine_similarity(
        output_reference.flatten(), output_rounded.flatten(), dim=0
    )
    assert result["output_relative_error"] == pytest.approx(float(expected_relative), abs=1e-12)
    assert result["state_nrmse"] == pytest.approx(float(expected_state), abs=1e-12)
    assert result["output_cosine"] == pytest.approx(float(expected_cosine), abs=1e-12)


def test_drift_rounded_state_is_used_by_later_tokens() -> None:
    inputs = {
        "q": torch.tensor(
            [[[[0.25, -0.5]], [[0.75, 0.125]], [[-0.25, 0.625]]]],
            dtype=torch.bfloat16,
        ),
        "k": torch.tensor(
            [[[[0.75, 0.125]], [[-0.5, 0.375]], [[0.25, 0.875]]]],
            dtype=torch.bfloat16,
        ),
        "v": torch.tensor(
            [[[[1.25, -0.75]], [[0.5, 1.5]], [[-1.0, 0.25]]]],
            dtype=torch.bfloat16,
        ),
        "log_alpha": torch.tensor(
            [[[[ -0.2, -0.3]], [[-0.4, -0.1]], [[-0.15, -0.25]]]],
            dtype=torch.float32,
        ),
        "beta": torch.tensor([[[0.6], [0.35], [0.8]]], dtype=torch.bfloat16),
    }
    records = rda_bf16_drift.compare(inputs, [1, 2, 3])
    q, k, v = (inputs[name].float() for name in ("q", "k", "v"))
    reference = torch.zeros(1, 1, 2, 2)
    rounded = torch.zeros_like(reference)
    expected_state_nrmse = []
    carried_state_changes_prediction = []
    for index in range(3):
        decay = torch.exp(inputs["log_alpha"][:, index]).unsqueeze(-2)
        beta = inputs["beta"].float()[:, index][..., None, None]
        reference = reference * decay
        reference_prediction = torch.einsum("bhvk,bhk->bhv", reference, k[:, index])
        reference = reference + beta * torch.einsum(
            "bhv,bhk->bhvk", v[:, index] - reference_prediction, k[:, index]
        )
        rounded = rounded * decay
        rounded_prediction = torch.einsum("bhvk,bhk->bhv", rounded, k[:, index])
        carried_state_changes_prediction.append(
            not torch.equal(rounded_prediction, reference_prediction)
        )
        rounded = rounded + beta * torch.einsum(
            "bhv,bhk->bhvk", v[:, index] - rounded_prediction, k[:, index]
        )
        rounded = rounded.to(torch.bfloat16).float()
        expected_state_nrmse.append(
            torch.sqrt(torch.mean((rounded - reference).square()))
            / torch.sqrt(torch.mean(reference.square()))
        )
    assert [row["state_nrmse"] for row in records] == pytest.approx(
        [float(value) for value in expected_state_nrmse], abs=1e-12
    )
    # From token 2 onward the prediction consumes the carried BF16-rounded
    # state, and differs from the all-fp32 reference before the new update.
    assert any(carried_state_changes_prediction[1:])
