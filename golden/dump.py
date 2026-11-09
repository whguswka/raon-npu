#!/usr/bin/env python3
# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Deterministic CPU golden dumps for the local Raon-1-mini implementation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    raw = value.view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "norm": float(torch.linalg.vector_norm(value.float()).item()),
        "sha256": tensor_hash(value),
    }


def tensor_array(tensor: torch.Tensor) -> tuple[np.ndarray, str]:
    """Return a pickle-free numpy representation, preserving BF16 bits exactly."""
    value = tensor.detach().contiguous().cpu()
    dtype = str(value.dtype).removeprefix("torch.")
    if value.dtype == torch.bfloat16:
        return value.view(torch.uint16).numpy(), dtype
    return value.numpy(), dtype


def projected_npz_bytes(tensors: dict[str, torch.Tensor]) -> int:
    """Conservative pre-write reservation: payload plus NPZ/JSON container slack."""
    payload = sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())
    dtype_json = json.dumps(
        {name: str(tensor.dtype).removeprefix("torch.") for name, tensor in tensors.items()},
        sort_keys=True,
    ).encode()
    raw = payload + len(dtype_json)
    # ZIP local/central records and NumPy headers dominate when many captured
    # tensors are small, so reserve per-member slack in addition to 8.33%.
    return raw + max(4096, (raw + 11) // 12) + 1024 * (len(tensors) + 1)


def save_tensor_npz(
    path: Path, tensors: dict[str, torch.Tensor], budget: "Budget"
) -> dict[str, str]:
    reservation = budget.reserve(projected_npz_bytes(tensors), path.name)
    arrays: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    temporary = path.with_name(f".{path.name}.tmp.npz")
    try:
        for name, tensor in tensors.items():
            arrays[name], dtypes[name] = tensor_array(tensor)
        arrays["__dtypes_json__"] = np.asarray(json.dumps(dtypes, sort_keys=True))
        np.savez(temporary, **arrays)
        budget.commit(reservation, temporary.stat().st_size, path.name)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        budget.cancel(reservation)
        raise
    return dtypes


def load_local_raon(code_dir: Path):
    """Import the two checkpoint Python files as an isolated local package."""
    code_dir = code_dir.resolve()
    required = [code_dir / "configuration_raon1_mini.py", code_dir / "modeling_raon1_mini.py"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"local Raon source missing: {', '.join(missing)}")
    package_name = "_raon_golden_" + hashlib.sha256(str(code_dir).encode()).hexdigest()[:12]
    if package_name not in sys.modules:
        # The checkpoint source targets Transformers 5.x, where this helper is
        # exported from the package root. 4.57 has the same torch init
        # operations but not that export; supplying the module preserves the
        # source's initialization equations and keeps the source file untouched.
        import transformers

        if not hasattr(transformers, "initialization"):
            transformers.initialization = torch.nn.init
        package = types.ModuleType(package_name)
        package.__path__ = [str(code_dir)]
        package.__package__ = package_name
        sys.modules[package_name] = package
        for stem in ("configuration_raon1_mini", "modeling_raon1_mini"):
            name = f"{package_name}.{stem}"
            spec = importlib.util.spec_from_file_location(name, code_dir / f"{stem}.py")
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot import {code_dir / f'{stem}.py'}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            if stem == "modeling_raon1_mini":
                import inspect

                if "inputs_embeds" not in inspect.signature(module.create_causal_mask).parameters:
                    def create_causal_mask_compat(*, config, inputs_embeds, attention_mask,
                                                  past_key_values, position_ids=None):
                        # Transformers 5.x source API on a 4.57 runtime. The Raon MLA
                        # implementation accepts a boolean [B,1,Q,K] mask and handles
                        # the no-padding causal fast path itself.
                        if attention_mask is None:
                            return None
                        if attention_mask.dim() == 4:
                            return attention_mask
                        if bool(attention_mask.all()):
                            return None
                        batch, query = inputs_embeds.shape[:2]
                        keys = attention_mask.shape[-1]
                        offset = keys - query
                        causal = torch.arange(keys, device=inputs_embeds.device)[None, :] <= (
                            offset + torch.arange(query, device=inputs_embeds.device)[:, None]
                        )
                        valid = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
                        return causal[None, None] & valid[:, None, None, :]

                    module.create_causal_mask = create_causal_mask_compat
    return sys.modules[f"{package_name}.configuration_raon1_mini"], sys.modules[
        f"{package_name}.modeling_raon1_mini"
    ]


def parse_layers(value: str | None, count: int) -> list[int]:
    if value is None or value.strip().lower() in {"", "all"}:
        return list(range(count))
    selected: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            left, right = part.split("-", 1)
            selected.update(range(int(left), int(right) + 1))
        else:
            selected.add(int(part))
    invalid = sorted(item for item in selected if item < 0 or item >= count)
    if invalid:
        raise ValueError(f"layer indices out of range 0..{count - 1}: {invalid}")
    return sorted(selected)


def parse_snapshot_steps(value: str | None, decode_steps: int) -> set[int]:
    if decode_steps <= 0:
        return set()
    if value:
        result = {int(item.strip()) for item in value.split(",") if item.strip()}
    else:
        result = {1, 16, 64, decode_steps}
    invalid = sorted(step for step in result if step < 1)
    if invalid:
        raise ValueError(f"snapshot steps are 1-based and must be >= 1: {invalid}")
    return {step for step in result if step <= decode_steps}


def read_prompt(path: Path, model_dir: Path, apply_chat_template: bool) -> tuple[torch.Tensor, Any]:
    raw = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = raw
    if isinstance(payload, dict) and "input_ids" in payload:
        if apply_chat_template:
            raise ValueError("--chat-template cannot be used with explicit input_ids")
        ids = torch.tensor([payload["input_ids"]], dtype=torch.long)
        return ids, payload
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    if apply_chat_template:
        messages = payload if isinstance(payload, list) else [{"role": "user", "content": str(payload)}]
        ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
    else:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    return ids.long().cpu(), payload


def model_fingerprint(model_dir: Path, mode: str) -> dict[str, Any]:
    candidates = sorted(
        path for path in model_dir.iterdir()
        if path.is_file() and (path.suffix in {".json", ".safetensors", ".bin"} or path.name.endswith(".jinja"))
    )
    entries: list[dict[str, Any]] = []
    aggregate = hashlib.sha256()
    for path in candidates:
        stat = path.stat()
        entry: dict[str, Any] = {"file": path.name, "bytes": stat.st_size}
        if mode == "full":
            entry["sha256"] = sha256_file(path)
        aggregate.update(json.dumps(entry, sort_keys=True).encode())
        entries.append(entry)
    return {"mode": mode, "sha256": aggregate.hexdigest(), "files": entries}


def code_fingerprint(code_dir: Path) -> dict[str, str]:
    return {
        name: sha256_file(code_dir / name)
        for name in ("configuration_raon1_mini.py", "modeling_raon1_mini.py")
    }


def estimate_bytes(
    config: Any,
    prompt_tokens: int,
    decode_steps: int,
    layers: Iterable[int],
    snapshots: set[int],
    capture_rda: bool,
    expanded_kv_cache: bool = False,
) -> int:
    hidden = int(config.hidden_size)
    streams = int(config.raon_mhc_streams)
    total = 0
    if capture_rda:
        for idx in layers:
            if config.is_rda_layer(idx):
                h, d = int(config.rda_num_heads), int(config.rda_head_dim)
                total += prompt_tokens * h * (3 * d * 2 + d * 4 + 2)
        return int(total * 1.08 + 1_000_000)
    calls = [(1, prompt_tokens)] + [(step, 1) for step in range(2, decode_steps + 1)]
    for step, tokens in calls:
        for idx in layers:
            # layer input/output + attention input/output, BF16
            total += tokens * hidden * 2 * (2 * streams + 2)
            if config.is_moe_layer(idx):
                total += tokens * (int(config.num_experts) * 4 + int(config.num_experts_per_token) * 12)
            if step in snapshots:
                if config.is_rda_layer(idx):
                    h, d = int(config.rda_num_heads), int(config.rda_head_dim)
                    total += h * d * d * 4
                else:
                    cache_tokens = prompt_tokens + step - 1
                    total += cache_tokens * (
                        int(config.kv_lora_rank) + int(config.qk_rope_head_dim)
                    ) * 2
                    if expanded_kv_cache:
                        qk = int(config.qk_nope_head_dim + config.qk_rope_head_dim)
                        total += int(config.num_attention_heads) * cache_tokens * (
                            qk + int(config.v_head_dim)
                        ) * 2
    return int(total * 1.08 + 1_000_000)


@dataclass
class CaptureRecorder:
    selected: set[int]
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    mla_latent_cache: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)
    handles: list[Any] = field(default_factory=list)
    _mla_pending: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def _copy(self, name: str, value: torch.Tensor) -> None:
        self.tensors[name] = value.detach().cpu().clone()

    def _append_mla(self, layer_idx: int, kv_lat: torch.Tensor, k_pe: torch.Tensor) -> None:
        current = self.mla_latent_cache.get(layer_idx)
        additions = {
            "kv_lat": kv_lat.detach().cpu().clone(),
            "k_pe": k_pe.detach().cpu().clone(),
        }
        if current is None:
            self.mla_latent_cache[layer_idx] = additions
            return
        for name, value in additions.items():
            current[name] = torch.cat([current[name], value], dim=1)

    def attach(self, model: Any) -> None:
        for idx, layer in enumerate(model.model.layers):
            if idx not in self.selected:
                continue
            prefix = f"layer_{idx:03d}"

            def layer_pre(module, args, kwargs, p=prefix):
                self._copy(f"{p}.streams_in", args[0])

            def layer_post(module, args, kwargs, output, p=prefix):
                self._copy(f"{p}.streams_out", output)

            def mixer_pre(module, args, kwargs, p=prefix):
                self._copy(f"{p}.mixer_in", args[0])

            def mixer_post(module, args, kwargs, output, p=prefix):
                self._copy(f"{p}.mixer_out", output)

            self.handles.append(layer.register_forward_pre_hook(layer_pre, with_kwargs=True))
            self.handles.append(layer.register_forward_hook(layer_post, with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_pre_hook(mixer_pre, with_kwargs=True))
            self.handles.append(layer.self_attn.register_forward_hook(mixer_post, with_kwargs=True))
            if not layer.is_rda:
                attention = layer.self_attn

                def kv_projection_post(module, args, output, layer_idx=idx, attn=attention):
                    # k_pe is consumed directly. kv_lat is cached after the exact
                    # RMSNorm invocation used by kv_b_proj (see the paired hook).
                    self._mla_pending[layer_idx] = torch.split(
                        output, [attn.kv_lora_rank, attn.qk_pe], dim=-1
                    )

                def kv_norm_post(module, args, output, layer_idx=idx):
                    _raw_kv_lat, k_pe = self._mla_pending.pop(layer_idx)
                    self._append_mla(layer_idx, output, k_pe)

                self.handles.append(
                    attention.kv_a_proj_with_mqa.register_forward_hook(kv_projection_post)
                )
                self.handles.append(
                    attention.kv_a_layernorm.register_forward_hook(kv_norm_post)
                )
            if layer.is_moe:
                def router_post(module, args, output, p=prefix):
                    hidden = args[0]
                    logits = torch.nn.functional.linear(hidden.float(), module.weight.float())
                    self._copy(f"{p}.router_logits", logits)
                    self._copy(f"{p}.router_top", output[0])
                    self._copy(f"{p}.router_weights", output[1])

                self.handles.append(layer.moe.gate.register_forward_hook(router_post))

    def clear(self) -> None:
        self.tensors.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self._mla_pending.clear()


class Budget:
    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self.written = 0
        self.reserved = 0

    def reserve(self, size: int, label: str) -> int:
        projected = self.written + self.reserved + size
        if projected > self.max_bytes:
            raise RuntimeError(
                f"dump budget exceeded before writing {label}: "
                f"{projected / 1e9:.3f} GB > {self.max_bytes / 1e9:.3f} GB"
            )
        self.reserved += size
        return size

    def commit(self, reservation: int, actual: int, label: str) -> None:
        if actual > reservation or self.written + actual > self.max_bytes:
            raise RuntimeError(
                f"dump budget exceeded while finalizing {label}: reserved {reservation} bytes, "
                f"actual {actual} bytes"
            )
        self.reserved -= reservation
        self.written += actual

    def cancel(self, reservation: int) -> None:
        if reservation <= self.reserved:
            self.reserved -= reservation


def write_manifest(path: Path, manifest: dict[str, Any], budget: Budget) -> None:
    # Include the manifest itself in actual_dump_bytes. Iterate because the
    # decimal byte count can change the JSON length at a digit boundary.
    rendered = b""
    for _ in range(8):
        manifest["actual_dump_bytes"] = budget.written + len(rendered)
        updated = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
        if len(updated) == len(rendered):
            rendered = updated
            break
        rendered = updated
    manifest["actual_dump_bytes"] = budget.written + len(rendered)
    rendered = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    reservation = budget.reserve(len(rendered), path.name)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(rendered)
        budget.commit(reservation, temporary.stat().st_size, path.name)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        budget.cancel(reservation)
        raise


def install_rda_capture(model: Any, out_dir: Path, selected: set[int], budget: Budget, files: list[dict[str, Any]]):
    restorers = []
    for idx, layer in enumerate(model.model.layers):
        if idx not in selected or not layer.is_rda:
            continue
        original = layer.self_attn._core

        def captured(self, q, k, v, log_alpha, beta, state, *, layer_idx=idx, original_core=original):
            path = out_dir / f"rda_inputs_layer_{layer_idx:03d}.npz"
            dtypes = save_tensor_npz(
                path, {"q": q, "k": k, "v": v, "log_alpha": log_alpha, "beta": beta}, budget
            )
            files.append({"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "dtypes": dtypes})
            return original_core(q, k, v, log_alpha, beta, state)

        layer.self_attn._core = types.MethodType(captured, layer.self_attn)
        restorers.append((layer.self_attn, original))
    return restorers


def state_records(
    cache: Any,
    mla_latent_cache: dict[int, dict[str, torch.Tensor]],
    selected: list[int],
    full: bool,
    tensors: dict[str, torch.Tensor],
    expanded_kv_cache: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    conv_states = getattr(cache, "conv_states", None)
    for idx in selected:
        recurrent = cache.recurrent_states[idx]
        # RDA layers keep a second state: the raw (pre-conv) q|k|v rows of the
        # previous K-1 tokens, [B, 3*proj, K-1] (modeling_raon1_mini.py:325-333).
        conv = conv_states[idx] if conv_states is not None else None
        key = cache.key_cache[idx]
        value = cache.value_cache[idx]
        if recurrent is not None:
            entry: dict[str, Any] = {"kind": "rda_recurrent", **tensor_summary(recurrent)}
            if conv is not None:
                entry["conv_state"] = tensor_summary(conv)
            result[str(idx)] = entry
            if full:
                tensors[f"layer_{idx:03d}.rda_state"] = recurrent.detach().cpu().clone()
                if conv is not None:
                    tensors[f"layer_{idx:03d}.rda_conv_state"] = conv.detach().cpu().clone()
        elif idx in mla_latent_cache:
            latent = mla_latent_cache[idx]
            result[str(idx)] = {
                "kind": "mla_latent_cache",
                "kv_lat": tensor_summary(latent["kv_lat"]),
                "k_pe": tensor_summary(latent["k_pe"]),
            }
            if full:
                tensors[f"layer_{idx:03d}.mla_kv_lat_cache"] = latent["kv_lat"]
                tensors[f"layer_{idx:03d}.mla_k_pe_cache"] = latent["k_pe"]
            if expanded_kv_cache and key is not None and value is not None:
                result[str(idx)]["expanded_kv"] = {
                    "key": tensor_summary(key),
                    "value": tensor_summary(value),
                }
                if full:
                    tensors[f"layer_{idx:03d}.mla_key_cache"] = key.detach().cpu().clone()
                    tensors[f"layer_{idx:03d}.mla_value_cache"] = value.detach().cpu().clone()
    return result


def _checkpoint_weight_map(model_dir: Path) -> dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"invalid or empty weight_map in {index_path}")
        return {str(name): str(shard) for name, shard in weight_map.items()}
    single = model_dir / "model.safetensors"
    if not single.is_file():
        raise FileNotFoundError(f"no safetensors checkpoint under {model_dir}")
    with safe_open(single, framework="pt", device="cpu") as archive:
        return {name: single.name for name in archive.keys()}


def _set_model_tensor(model: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if leaf in parent._parameters:
        old = parent._parameters[leaf]
        requires_grad = bool(old.requires_grad) if old is not None else False
        parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=requires_grad)
    elif leaf in parent._buffers:
        parent._buffers[leaf] = value
    else:
        raise KeyError(f"model has no parameter or buffer named {name}")


def load_mxfp8_checkpoint_into_model(
    model: torch.nn.Module, model_dir: Path, dequantize_mxfp8: Any
) -> None:
    """Stream shards and replace MXFP8 expert pairs with plain BF16 parameters."""
    weight_map = _checkpoint_weight_map(model_dir)
    checkpoint_names = set(weight_map)
    scale_names = {name for name in checkpoint_names if name.endswith(".weight_scale_inv")}
    pairs = {name.removesuffix("_scale_inv"): name for name in scale_names}
    expert_weights = {
        name for name in checkpoint_names
        if ".moe.experts." in name and name.endswith(".weight")
    }
    unpaired_weights = sorted(expert_weights - set(pairs))
    invalid_scales = sorted(
        scale for weight, scale in pairs.items()
        if (
            weight not in checkpoint_names
            or ".moe.experts." not in weight
            or weight_map.get(weight) != weight_map.get(scale)
        )
    )
    expected = set(model.state_dict())
    missing = sorted(expected - (checkpoint_names - scale_names))
    unexpected = sorted(checkpoint_names - expected - scale_names)
    if unpaired_weights or invalid_scales or missing or unexpected:
        raise RuntimeError(
            "MXFP8 checkpoint/model mismatch:\n"
            f"expert weights without scales: {unpaired_weights}\n"
            f"invalid or unpaired scales: {invalid_scales}\n"
            f"missing model tensors: {missing}\n"
            f"unexpected checkpoint tensors: {unexpected}"
        )
    missing_shards = sorted({shard for shard in weight_map.values() if not (model_dir / shard).is_file()})
    if missing_shards:
        raise FileNotFoundError(f"checkpoint shards are missing: {missing_shards}")

    loaded: set[str] = set()
    for shard in sorted(set(weight_map.values())):
        shard_names = {name for name, owner in weight_map.items() if owner == shard}
        with safe_open(model_dir / shard, framework="pt", device="cpu") as archive:
            actual_names = set(archive.keys())
            if actual_names != shard_names:
                raise RuntimeError(
                    f"{shard} header/index mismatch: missing={sorted(shard_names - actual_names)}, "
                    f"extra={sorted(actual_names - shard_names)}"
                )
            for name in sorted(shard_names - scale_names):
                value = archive.get_tensor(name)
                scale_name = pairs.get(name)
                if scale_name is not None:
                    scale = archive.get_tensor(scale_name)
                    if value.dtype != torch.float8_e4m3fn:
                        raise TypeError(f"{name} must be float8_e4m3fn, got {value.dtype}")
                    if scale.dtype != torch.uint8 and str(scale.dtype) != "torch.float8_e8m0fnu":
                        raise TypeError(f"{scale_name} must be uint8/E8M0, got {scale.dtype}")
                    if value.ndim != 2 or value.shape[1] % 32:
                        raise ValueError(f"{name} must be 2D with columns divisible by 32")
                    expected_scale_shape = (value.shape[0], value.shape[1] // 32)
                    if tuple(scale.shape) != expected_scale_shape:
                        raise ValueError(
                            f"{scale_name} shape {tuple(scale.shape)} != {expected_scale_shape} for {name}"
                        )
                    value = dequantize_mxfp8(value, scale, torch.bfloat16)
                    loaded.add(scale_name)
                elif value.dtype == torch.float8_e4m3fn:
                    raise TypeError(f"unpaired E4M3 tensor is not allowed: {name}")
                target = model.state_dict()[name]
                if tuple(value.shape) != tuple(target.shape):
                    raise ValueError(f"{name} shape {tuple(value.shape)} != model {tuple(target.shape)}")
                _set_model_tensor(model, name, value.contiguous())
                loaded.add(name)
    remaining = sorted(checkpoint_names - loaded)
    still_meta = sorted(name for name, tensor in model.state_dict().items() if tensor.is_meta)
    if remaining or still_meta:
        raise RuntimeError(f"incomplete MXFP8 load: remaining={remaining}, meta={still_meta}")


def load_model(model_dir: Path, code_dir: Path):
    config_mod, modeling_mod = load_local_raon(code_dir)
    config = config_mod.Raon1MiniConfig.from_pretrained(str(model_dir), local_files_only=True)
    quant = getattr(config, "quantization_config", None)
    if isinstance(quant, dict) and quant.get("quant_method") == "mxfp8":
        plain_payload = config.to_dict()
        plain_payload.pop("quantization_config", None)
        config = config_mod.Raon1MiniConfig.from_dict(plain_payload)
        with torch.device("meta"):
            model = modeling_mod.Raon1MiniForCausalLM(config)
        load_mxfp8_checkpoint_into_model(model, model_dir, modeling_mod.dequantize_mxfp8)
        model.eval()
        return model, config
    model = modeling_mod.Raon1MiniForCausalLM.from_pretrained(
        str(model_dir), config=config, local_files_only=True, dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, config


def run(args: argparse.Namespace) -> dict[str, Any]:
    model_dir = Path(args.model_dir).resolve()
    code_dir = Path(args.code_dir or args.model_dir).resolve()
    out_dir = Path(args.out).resolve()
    if args.dtype != "bf16":
        raise ValueError("only --dtype bf16 is supported")
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"--out must be new or empty to avoid mixing dump runs: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    input_ids, prompt_payload = read_prompt(Path(args.prompt_file), model_dir, args.chat_template)
    config_mod, _ = load_local_raon(code_dir)
    config = config_mod.Raon1MiniConfig.from_pretrained(str(model_dir), local_files_only=True)
    selected = parse_layers(args.layers, config.num_hidden_layers)
    snapshots = parse_snapshot_steps(args.state_steps, args.decode_steps)
    estimate = estimate_bytes(
        config,
        input_ids.shape[1],
        args.decode_steps,
        selected,
        snapshots,
        args.capture_rda_inputs,
        args.expanded_kv_cache,
    )
    print(f"estimated dump size: {estimate / (1024 ** 3):.3f} GiB", flush=True)
    max_bytes = int(args.max_gb * (1024 ** 3))
    if estimate > max_bytes:
        raise RuntimeError(f"estimated dump exceeds --max-gb: {estimate / (1024 ** 3):.3f} GiB > {args.max_gb:.3f} GiB")

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    model, config = load_model(model_dir, code_dir)
    budget = Budget(max_bytes)
    files: list[dict[str, Any]] = []
    recorder = CaptureRecorder(set(selected))
    restorers = []
    if args.capture_rda_inputs:
        restorers = install_rda_capture(model, out_dir, set(selected), budget, files)
    else:
        recorder.attach(model)

    manifest: dict[str, Any] = {
        "format_version": 1,
        "model_dir": str(model_dir),
        "code_dir": str(code_dir),
        "dtype": "bfloat16",
        "seed": args.seed,
        "prompt": prompt_payload,
        "prompt_token_ids": input_ids[0].tolist(),
        "chat_template_applied": bool(args.chat_template),
        "decode_steps": args.decode_steps,
        "selected_layers": selected,
        "state_snapshot_steps": sorted(snapshots),
        "expanded_kv_cache": bool(args.expanded_kv_cache),
        "capture_rda_inputs_only": bool(args.capture_rda_inputs),
        "estimated_bytes": estimate,
        "model_fingerprint": model_fingerprint(model_dir, args.model_hash),
        "code_sha256": code_fingerprint(code_dir),
        "generated_token_ids": [],
        "steps": [],
        "files": files,
    }
    cache = None
    try:
        with torch.inference_mode():
            if args.capture_rda_inputs:
                # Bypass lm_head: a 32K capture must not allocate
                # [tokens, 259840] logits when only RDA boundary inputs are needed.
                model.model(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    use_cache=True,
                    return_dict=True,
                )
            else:
                current = input_ids
                attention_mask = torch.ones_like(input_ids)
                total_generated = max(0, args.decode_steps)
                calls = max(1, total_generated)
                for zero_based_step in range(calls):
                    generation_step = zero_based_step + 1
                    if zero_based_step == 0:
                        call_ids = current
                    else:
                        call_ids = current[:, -1:]
                        attention_mask = torch.ones(
                            (1, input_ids.shape[1] + zero_based_step), dtype=torch.long
                        )
                    output = model(
                        input_ids=call_ids, attention_mask=attention_mask,
                        past_key_values=cache, use_cache=True, logits_to_keep=1, return_dict=True,
                    )
                    cache = output.past_key_values
                    logits = output.logits[:, -1, :]
                    top_values, top_ids = torch.topk(logits.float(), min(20, logits.shape[-1]), dim=-1)
                    token = torch.argmax(logits, dim=-1, keepdim=True)
                    step_tensors = dict(recorder.tensors)
                    step_tensors["logits_top20_ids"] = top_ids
                    step_tensors["logits_top20_values"] = top_values
                    states = state_records(
                        cache,
                        recorder.mla_latent_cache,
                        selected,
                        generation_step in snapshots,
                        step_tensors,
                        args.expanded_kv_cache,
                    )
                    path = out_dir / f"step_{generation_step:06d}.npz"
                    dtypes = save_tensor_npz(path, step_tensors, budget)
                    file_record = {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path), "dtypes": dtypes}
                    files.append(file_record)
                    manifest["steps"].append({
                        "step": generation_step,
                        "kind": "prefill" if generation_step == 1 else "decode",
                        "input_token_ids": call_ids[0].tolist(),
                        "next_token_id": int(token.item()),
                        "logits": tensor_summary(logits),
                        "top20_ids": top_ids[0].tolist(),
                        "top20_values": top_values[0].tolist(),
                        "states": states,
                        "file": path.name,
                    })
                    if generation_step <= total_generated:
                        manifest["generated_token_ids"].append(int(token.item()))
                    current = torch.cat([current, token.cpu()], dim=1)
                    recorder.clear()
                    if generation_step >= calls:
                        break
    finally:
        recorder.close()
        for module, original in restorers:
            module._core = original

    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest_path, manifest, budget)
    print(f"wrote {len(files)} data files ({budget.written / (1024 ** 2):.2f} MiB) to {out_dir}")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--code-dir", help="local directory containing modeling/configuration Python; defaults to model-dir")
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--decode-steps", type=int, default=1, metavar="N")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=["bf16"], default="bf16")
    parser.add_argument("--layers", help="0-based comma/range selection, e.g. 0,2-4; default all")
    parser.add_argument("--capture-rda-inputs", action="store_true", help="prefill only; save q/k/v/log_alpha/beta per RDA layer")
    parser.add_argument(
        "--state-steps",
        help="comma-separated 1-based generation steps for full-cache snapshots; default 1,16,64,last",
    )
    parser.add_argument(
        "--expanded-kv-cache",
        action="store_true",
        help="also record the model's head-expanded MLA key/value cache",
    )
    parser.add_argument("--max-gb", type=float, default=64.0)
    parser.add_argument("--model-hash", choices=["full", "metadata"], default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.decode_steps < 0:
        raise SystemExit("--decode-steps must be >= 0")
    if args.max_gb <= 0:
        raise SystemExit("--max-gb must be > 0")
    run(args)


if __name__ == "__main__":
    main()
