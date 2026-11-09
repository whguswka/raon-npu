# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Verification script comparing host NumPy engine against golden decode run."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from host.backend import NumpyBackend
from host.raon import RaonConfig, RaonModel
from host.strict_json import loads_strict
from host.weights import Weights, bf16_to_f32


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0:
        return 1.0 if np.array_equal(a_f, b_f) else 0.0
    return float(np.dot(a_f, b_f) / denom)


def relative_error(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.maximum(np.abs(a_f), np.abs(b_f)) + 1e-12
    return float(np.max(np.abs(a_f - b_f) / denom))


def _nearest_changed_router_score_gap(
    golden_top: np.ndarray,
    host_top: np.ndarray,
    golden_logits: np.ndarray,
    host_logits: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Describe the closest golden-score pair among exchanged experts."""
    golden_only = sorted(set(map(int, golden_top)) - set(map(int, host_top)))
    host_only = sorted(set(map(int, host_top)) - set(map(int, golden_top)))
    if not golden_only or not host_only:
        return None

    golden_flat = golden_logits.ravel()
    golden_expert, host_expert = min(
        ((g, h) for g in golden_only for h in host_only),
        key=lambda pair: abs(float(golden_flat[pair[0]]) - float(golden_flat[pair[1]])),
    )
    golden_selected_score = float(golden_flat[golden_expert])
    golden_replacement_score = float(golden_flat[host_expert])
    gap: dict[str, Any] = {
        "source": "golden_router_logits",
        "golden_only_expert": golden_expert,
        "host_only_expert": host_expert,
        "golden_only_score": golden_selected_score,
        "golden_score_for_host_only_expert": golden_replacement_score,
        "abs_score_gap": abs(golden_selected_score - golden_replacement_score),
    }
    if host_logits is not None:
        host_flat = host_logits.ravel()
        host_golden_score = float(host_flat[golden_expert])
        host_selected_score = float(host_flat[host_expert])
        gap.update({
            "host_score_for_golden_only_expert": host_golden_score,
            "host_score_for_host_only_expert": host_selected_score,
            "host_abs_score_gap": abs(host_golden_score - host_selected_score),
        })
    return gap


def _nearest_order_router_score_gap(
    golden_top: np.ndarray,
    golden_logits: np.ndarray,
    host_top: np.ndarray | None = None,
    host_logits: np.ndarray | None = None,
) -> dict[str, Any] | None:
    """Describe the closest golden-score pair among top-k experts with order difference."""
    g_list = [int(x) for x in np.asarray(golden_top).ravel()]
    if len(g_list) < 2:
        return None

    golden_flat = np.asarray(golden_logits).ravel()
    candidate_pairs: list[tuple[int, int]] = []
    if host_top is not None:
        h_list = [int(x) for x in np.asarray(host_top).ravel()]
        if len(h_list) == len(g_list) and set(h_list) == set(g_list):
            pos_h = {exp_id: idx for idx, exp_id in enumerate(h_list)}
            for i in range(len(g_list)):
                for j in range(i + 1, len(g_list)):
                    ea, eb = g_list[i], g_list[j]
                    if pos_h[ea] > pos_h[eb]:
                        candidate_pairs.append((ea, eb))

    if not candidate_pairs:
        for i in range(len(g_list)):
            for j in range(i + 1, len(g_list)):
                candidate_pairs.append((g_list[i], g_list[j]))

    expert_a, expert_b = min(
        candidate_pairs,
        key=lambda pair: abs(float(golden_flat[pair[0]]) - float(golden_flat[pair[1]])),
    )
    score_a = float(golden_flat[expert_a])
    score_b = float(golden_flat[expert_b])
    gap: dict[str, Any] = {
        "source": "golden_router_logits",
        "expert_a": expert_a,
        "expert_b": expert_b,
        "golden_score_a": score_a,
        "golden_score_b": score_b,
        "abs_score_gap": abs(score_a - score_b),
    }
    if host_logits is not None:
        host_flat = np.asarray(host_logits).ravel()
        h_score_a = float(host_flat[expert_a])
        h_score_b = float(host_flat[expert_b])
        gap.update({
            "host_score_a": h_score_a,
            "host_score_b": h_score_b,
            "host_abs_score_gap": abs(h_score_a - h_score_b),
        })
    return gap


def verify_requested_step_files(
    golden_dir: Path,
    start_step: int,
    end_step: int,
) -> list[Path]:
    """Ensure every requested step snapshot exists before or during verification."""
    if start_step > end_step:
        raise ValueError(
            f"Invalid step range: start_step ({start_step}) > end_step ({end_step})"
        )
    step_files: list[Path] = []
    missing: list[str] = []
    for step in range(start_step, end_step + 1):
        step_file = Path(golden_dir) / f"step_{step:06d}.npz"
        if not step_file.is_file():
            missing.append(step_file.name)
        step_files.append(step_file)
    if missing:
        raise FileNotFoundError(
            f"Missing requested golden step snapshot(s) in {golden_dir}: {', '.join(missing)}"
        )
    return step_files


SUPPORTED_START_STEP = 2


def validate_start_step(start_step: int) -> None:
    """Reject any start step other than the one supported start step, 2.

    A run always resumes from snapshot ``step_000001.npz`` and steps from the
    step immediately after that snapshot (step 2, consuming generated token
    index 0). A start step other than 2 would skip steps 2..start_step-1, so
    the model state would never have consumed those tokens and the comparison
    at start_step would run against a wrong, under-consumed state. Reject it
    up front instead of silently verifying the wrong state.
    """
    if start_step != SUPPORTED_START_STEP:
        raise ValueError(
            f"--start-step={start_step} is not supported: a run always resumes from "
            f"snapshot step_000001 and its first decoded step is "
            f"{SUPPORTED_START_STEP}, so steps 2..{start_step - 1} would be skipped and "
            f"verified against an under-consumed model state; only "
            f"--start-step={SUPPORTED_START_STEP} is supported"
        )


def parse_start_step(val_str: Any) -> int:
    try:
        val = int(val_str)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"start_step must be an integer, got {val_str!r}")
    try:
        validate_start_step(val)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))
    return val


parse_start_step.is_ranged = True  # type: ignore[attr-defined]
parse_start_step.type_cls = int  # type: ignore[attr-defined]
parse_start_step.min_val = SUPPORTED_START_STEP  # type: ignore[attr-defined]
parse_start_step.max_val = SUPPORTED_START_STEP  # type: ignore[attr-defined]
parse_start_step.min_inclusive = True  # type: ignore[attr-defined]
parse_start_step.max_inclusive = True  # type: ignore[attr-defined]
parse_start_step.name = "start_step"  # type: ignore[attr-defined]


def validate_generated_token_ids(
    generated_token_ids: Any,
    end_step: int,
    vocab_size: int | None = None,
) -> list[int]:
    """Validate that generated_token_ids is a valid list of tokens for the run.

    Requirements:
    - Must be a list.
    - Each element must be a non-boolean integer >= 0.
    - If vocab_size is known, each element must be < vocab_size.
    - Length must be at least end_step - 1.
    """
    if not isinstance(generated_token_ids, list):
        raise ValueError(
            f"generated_token_ids must be a list, got {type(generated_token_ids).__name__}"
        )
    min_len = end_step - 1
    if len(generated_token_ids) < min_len:
        raise ValueError(
            f"generated_token_ids length {len(generated_token_ids)} is insufficient for "
            f"end_step={end_step}; requires at least {min_len} tokens"
        )
    for idx, tok in enumerate(generated_token_ids):
        if isinstance(tok, bool) or not isinstance(tok, int):
            raise ValueError(
                f"generated_token_ids[{idx}] must be a non-boolean integer, got {tok!r} "
                f"of type {type(tok).__name__}"
            )
        if tok < 0:
            raise ValueError(
                f"generated_token_ids[{idx}] must be >= 0, got {tok}"
            )
        if vocab_size is not None and tok >= vocab_size:
            raise ValueError(
                f"generated_token_ids[{idx}] must be < vocab_size ({vocab_size}), got {tok}"
            )
    return generated_token_ids


_SNAPSHOT_STATE_DTYPES = {
    "rda_state": "float32",
    "rda_conv_state": "bfloat16",
    "mla_kv_lat_cache": "bfloat16",
    "mla_k_pe_cache": "bfloat16",
    "mla_key_cache": "bfloat16",
    "mla_value_cache": "bfloat16",
}
_SNAPSHOT_DTYPE_NUMPY = {"float32": np.float32, "bfloat16": np.uint16}


def _check_snapshot_array(
    snapshot: Any,
    keys: set[str],
    declared: dict[str, str],
    name: str,
    core_shape: tuple,
    errors: list[str],
) -> None:
    """One required snapshot state key: present, declared dtype correct, stored dtype and core shape match."""
    if name not in keys:
        errors.append(f"{name}: required snapshot state key missing")
        return
    expected_dtype = _SNAPSHOT_STATE_DTYPES[name.rsplit(".", 1)[-1]]
    declared_dtype = declared.get(name)
    if declared_dtype is None:
        errors.append(f"{name}: missing from __dtypes_json__")
        return
    if declared_dtype != expected_dtype:
        errors.append(
            f"{name}: __dtypes_json__ declares '{declared_dtype}', expected '{expected_dtype}'"
        )
        return
    arr = np.asarray(snapshot[name])
    if arr.dtype != _SNAPSHOT_DTYPE_NUMPY[expected_dtype]:
        errors.append(
            f"{name}: stored dtype {arr.dtype} does not match declared '{expected_dtype}' "
            f"(expected numpy {np.dtype(_SNAPSHOT_DTYPE_NUMPY[expected_dtype]).name})"
        )
    core = arr
    if arr.ndim == len(core_shape) + 1:
        if arr.shape[0] != 1:
            errors.append(f"{name}: leading batch dim is {arr.shape[0]}, expected 1")
        core = arr[0]
    if core.ndim != len(core_shape) or any(
        expect is not None and int(dim) != int(expect)
        for dim, expect in zip(core.shape, core_shape)
    ):
        errors.append(f"{name}: shape {arr.shape} does not match expected core shape {core_shape}")


def verify_initial_snapshot(snapshot: Any, config: RaonConfig) -> dict[str, str]:
    """Verify the initial snapshot is complete for this config BEFORE any run.

    Requires, for every RDA layer in the config, both ``rda_state`` and
    ``rda_conv_state``, and for every MLA layer one complete cache pair —
    latent + PE or expanded key + value (never one half of a pair) — each with
    config-consistent shape and ``__dtypes_json__`` dtype. Raises ValueError
    listing every violation otherwise.
    """
    keys = set(snapshot.files if hasattr(snapshot, "files") else snapshot.keys())
    if "__dtypes_json__" not in keys:
        raise KeyError("initial snapshot is missing '__dtypes_json__' dtype metadata")
    declared = loads_strict(str(np.asarray(snapshot["__dtypes_json__"]).item()))

    rda_specs = (
        ("rda_state", (config.rda_num_heads, config.rda_head_dim, config.rda_head_dim)),
        ("rda_conv_state",
         (3 * config.rda_num_heads * config.rda_head_dim, config.rda_conv_kernel_size - 1)),
    )
    mla_specs = {
        "mla_kv_lat_cache": (None, config.kv_lora_rank),
        "mla_k_pe_cache": (None, config.qk_rope_head_dim),
        "mla_key_cache": (
            config.num_attention_heads, None,
            config.qk_nope_head_dim + config.qk_rope_head_dim,
        ),
        "mla_value_cache": (config.num_attention_heads, None, config.v_head_dim),
    }

    errors: list[str] = []
    for idx in config.rda_layers:
        pfx = f"layer_{idx:03d}"
        for suffix, core_shape in rda_specs:
            _check_snapshot_array(
                snapshot, keys, declared, f"{pfx}.{suffix}", core_shape, errors
            )

    for idx in config.mla_layers:
        pfx = f"layer_{idx:03d}"
        lat_k, pe_k = f"{pfx}.mla_kv_lat_cache", f"{pfx}.mla_k_pe_cache"
        k_k, v_k = f"{pfx}.mla_key_cache", f"{pfx}.mla_value_cache"
        has_lat, has_pe = lat_k in keys, pe_k in keys
        has_k, has_v = k_k in keys, v_k in keys
        if has_k != has_v:
            present = k_k if has_k else v_k
            missing = v_k if has_k else k_k
            errors.append(
                f"layer {idx}: incomplete MLA expanded cache pair: '{present}' present "
                f"but '{missing}' missing"
            )
        if has_lat != has_pe:
            present = lat_k if has_lat else pe_k
            missing = pe_k if has_lat else lat_k
            errors.append(
                f"layer {idx}: incomplete MLA latent/PE cache pair: '{present}' present "
                f"but '{missing}' missing"
            )
        if not (has_k and has_v) and not (has_lat and has_pe):
            errors.append(
                f"layer {idx}: no complete MLA cache pair (need both "
                f"'{lat_k}' and '{pe_k}', or both '{k_k}' and '{v_k}')"
            )
        for suffix in ("mla_kv_lat_cache", "mla_k_pe_cache", "mla_key_cache", "mla_value_cache"):
            name = f"{pfx}.{suffix}"
            if name in keys:
                _check_snapshot_array(
                    snapshot, keys, declared, name, mla_specs[suffix], errors
                )

    if errors:
        raise ValueError(
            "Initial snapshot is incomplete or mismatched for this config: "
            + "; ".join(errors)
        )
    return declared


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float, np.number)):
        return False
    return bool(math.isfinite(float(val)))


def _resolve_expected_moe_layers(
    payload: dict[str, Any], expected_num_layers: int
) -> tuple[list[int], str, list[str]]:
    """Resolve router-bearing layers, retaining legacy-payload compatibility."""
    if "expected_moe_layers" in payload:
        raw_layers = payload["expected_moe_layers"]
        if not isinstance(raw_layers, list):
            return [], "payload.expected_moe_layers", [
                "expected_moe_layers is not a list"
            ]
        if any(isinstance(layer, bool) or not isinstance(layer, int) for layer in raw_layers):
            return [], "payload.expected_moe_layers", [
                "expected_moe_layers must contain only integer layer numbers"
            ]
        expected_moe_layers = sorted(raw_layers)
        errors: list[str] = []
        if expected_moe_layers != raw_layers:
            errors.append("expected_moe_layers is not sorted")
        if len(set(expected_moe_layers)) != len(expected_moe_layers):
            errors.append("expected_moe_layers contains duplicates")
        invalid = [
            layer for layer in expected_moe_layers
            if layer < 0 or layer >= expected_num_layers
        ]
        if invalid:
            errors.append(
                f"expected_moe_layers contains out-of-range layers {invalid} "
                f"for expected_num_layers={expected_num_layers}"
            )
        return expected_moe_layers, "payload.expected_moe_layers", errors

    config_payload = payload.get("config")
    if not isinstance(config_payload, dict):
        config_payload = {}

    def legacy_value(key: str, default: Any) -> tuple[Any, bool]:
        if key in payload:
            return payload[key], True
        if key in config_payload:
            return config_payload[key], True
        return default, False

    first_dense, has_first_dense = legacy_value("first_k_dense_replace", 1)
    frequency, has_frequency = legacy_value("moe_layer_freq", 1)
    num_experts, has_num_experts = legacy_value("num_experts", 128)
    source = (
        "legacy payload config values"
        if has_first_dense or has_frequency or has_num_experts
        else "released config defaults"
    )
    try:
        first_dense = int(first_dense)
        frequency = int(frequency)
    except (TypeError, ValueError):
        return [], source, [
            "legacy MoE config values first_k_dense_replace and moe_layer_freq must be integers"
        ]
    if frequency <= 0:
        return [], source, [f"invalid moe_layer_freq={frequency}; expected a positive integer"]
    if not num_experts:
        return [], source, []
    return [
        layer for layer in range(expected_num_layers)
        if layer >= first_dense and layer % frequency == 0
    ], source, []


def _validate_payload_completeness(
    payload: dict[str, Any],
) -> tuple[bool, list[str], int, int, int, int, list[int], str]:
    """Validate step range, contiguity, expected count, and required per-layer metrics."""
    errors: list[str] = []

    raw_start = payload.get("start_step", 2)
    start_step = int(raw_start) if isinstance(raw_start, (int, float)) else 2

    if "end_step" in payload and isinstance(payload["end_step"], (int, float)):
        end_step = int(payload["end_step"])
    elif "expected_steps" in payload and isinstance(payload["expected_steps"], (int, float)):
        end_step = start_step + int(payload["expected_steps"]) - 1
    else:
        end_step = 16

    if "expected_steps" in payload and isinstance(payload["expected_steps"], (int, float)):
        expected_steps = int(payload["expected_steps"])
    else:
        expected_steps = end_step - start_step + 1

    raw_layers = payload.get("expected_num_layers", 25)
    expected_num_layers = int(raw_layers) if isinstance(raw_layers, (int, float)) else 25
    expected_moe_layers, expected_moe_layers_source, moe_layer_errors = (
        _resolve_expected_moe_layers(payload, expected_num_layers)
    )
    errors.extend(moe_layer_errors)
    expected_moe_layer_set = set(expected_moe_layers)

    if "baseline_passed" in payload and payload["baseline_passed"] is not True:
        errors.append(
            f"baseline check not passed (baseline_passed={payload['baseline_passed']!r})"
        )

    if start_step > end_step or expected_steps <= 0:
        errors.append(
            f"invalid expected step range {start_step}..{end_step} (expected_steps={expected_steps})"
        )
    elif (end_step - start_step + 1) != expected_steps:
        errors.append(
            f"step range {start_step}..{end_step} count ({end_step - start_step + 1}) "
            f"!= expected_steps ({expected_steps})"
        )

    if expected_num_layers <= 0:
        errors.append(f"invalid expected_num_layers={expected_num_layers}")

    results = payload.get("step_results")
    if not isinstance(results, list) or len(results) == 0:
        errors.append("step_results is empty or missing")
        return (
            False, errors, start_step, end_step, expected_steps, expected_num_layers,
            expected_moe_layers, expected_moe_layers_source,
        )

    if len(results) != expected_steps:
        errors.append(
            f"step_results count ({len(results)}) != expected_steps ({expected_steps})"
        )

    if "total_steps" in payload and payload["total_steps"] != len(results):
        errors.append(
            f"recorded total_steps ({payload['total_steps']}) != len(step_results) ({len(results)})"
        )

    expected_step_sequence = list(range(start_step, end_step + 1))
    actual_steps: list[Any] = [
        s.get("step") if isinstance(s, dict) else None for s in results
    ]
    if actual_steps != expected_step_sequence:
        errors.append(
            f"step sequence {actual_steps} does not match expected contiguous steps {expected_step_sequence}"
        )

    required_step_keys = ("step", "argmax_match", "argmax_tie_match", "logits_cos", "layers")
    required_layer_Save_keys = (
        "streams_in_cos",
        "mixer_in_cos",
        "mixer_out_cos",
        "streams_out_cos",
    )
    required_moe_keys = (
        "router_top_set_pct",
        "router_top_order_match",
        "router_weights_max_err",
    )

    for idx, step_result in enumerate(results):
        if not isinstance(step_result, dict):
            errors.append(f"step_results[{idx}] is not a dict")
            continue
        step_id = step_result.get("step", f"idx_{idx}")
        for req_k in required_step_keys:
            if req_k not in step_result:
                errors.append(f"step {step_id}: missing required key '{req_k}'")

        if "argmax_match" in step_result and not isinstance(step_result["argmax_match"], bool):
            errors.append(f"step {step_id}: 'argmax_match' is not a bool")
        if "argmax_tie_match" in step_result and not isinstance(step_result["argmax_tie_match"], bool):
            errors.append(f"step {step_id}: 'argmax_tie_match' is not a bool")
        if "logits_cos" in step_result and not _is_finite_number(step_result["logits_cos"]):
            errors.append(f"step {step_id}: 'logits_cos' is not a finite number")

        layers = step_result.get("layers")
        if not isinstance(layers, list) or len(layers) == 0:
            errors.append(f"step {step_id}: 'layers' is empty or not a list")
            continue

        if len(layers) != expected_num_layers:
            errors.append(
                f"step {step_id}: layer count ({len(layers)}) != expected_num_layers ({expected_num_layers})"
            )

        actual_layer_ids = [
            l.get("layer") if isinstance(l, dict) else None for l in layers
        ]
        expected_layer_ids = list(range(expected_num_layers))
        if actual_layer_ids != expected_layer_ids:
            errors.append(
                f"step {step_id}: layer sequence {actual_layer_ids} != expected {expected_layer_ids}"
            )

        for l_idx, layer in enumerate(layers):
            if not isinstance(layer, dict):
                errors.append(f"step {step_id}, layer[{l_idx}]: not a dict")
                continue
            lid = layer.get("layer", l_idx)
            for lk in required_layer_Save_keys:
                if lk not in layer or not _is_finite_number(layer[lk]):
                    errors.append(
                        f"step {step_id}, layer {lid}: missing or non-finite '{lk}'"
                    )
            is_moe = isinstance(lid, int) and lid in expected_moe_layer_set
            if is_moe:
                if "router_top_set_pct" not in layer or not _is_finite_number(
                    layer["router_top_set_pct"]
                ):
                    errors.append(
                        f"step {step_id}, layer {lid}: missing or non-finite 'router_top_set_pct'"
                    )
                if "router_top_order_match" not in layer or not isinstance(
                    layer["router_top_order_match"], bool
                ):
                    errors.append(
                        f"step {step_id}, layer {lid}: missing or non-bool 'router_top_order_match'"
                    )
                if "router_weights_max_err" not in layer or not _is_finite_number(
                    layer["router_weights_max_err"]
                ):
                    errors.append(
                        f"step {step_id}, layer {lid}: missing or non-finite 'router_weights_max_err'"
                    )

    return (
        len(errors) == 0, errors, start_step, end_step, expected_steps,
        expected_num_layers, expected_moe_layers, expected_moe_layers_source,
    )


def _load_order_diff_score_gap_from_golden(
    golden_dir: str | Path | None,
    step: int,
    layer_idx: int,
    host_top: list[int] | np.ndarray | None = None,
) -> tuple[list[int] | None, dict[str, Any] | None]:
    """Load golden router top-k and closest score gap from step NPZ if available."""
    if not golden_dir:
        return None, None
    step_path = Path(golden_dir) / f"step_{step:06d}.npz"
    if not step_path.is_file():
        return None, None
    pfx = f"layer_{layer_idx:03d}"
    k_rtop = f"{pfx}.router_top"
    k_rlog = f"{pfx}.router_logits"
    try:
        with np.load(step_path, allow_pickle=False) as npz:
            if k_rtop not in npz or k_rlog not in npz:
                return None, None
            g_top = npz[k_rtop].ravel()
            g_log = npz[k_rlog].ravel()
            gap = _nearest_order_router_score_gap(
                g_top,
                g_log,
                host_top=np.asarray(host_top) if host_top is not None else None,
            )
            return [int(x) for x in g_top], gap
    except Exception:
        return None, None


def evaluate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Recompute criteria from recorded per-step data without loading a model."""
    (
        is_complete,
        completeness_errors,
        start_step,
        end_step,
        expected_steps,
        expected_num_layers,
        expected_moe_layers,
        expected_moe_layers_source,
    ) = _validate_payload_completeness(payload)

    results = payload.get("step_results")
    if not isinstance(results, list):
        results = []
    expected_moe_layer_set = set(expected_moe_layers)

    golden_dir = payload.get("golden_dir")
    pre_difference: list[tuple[int, int, float]] = []
    post_difference: list[tuple[int, int, float]] = []
    first_router_differences: list[dict[str, Any]] = []
    router_set_differences: list[dict[str, Any]] = []
    router_order_differences: list[dict[str, Any]] = []
    all_router_matches: list[float] = []

    for step_result in results:
        if not isinstance(step_result, dict) or "step" not in step_result:
            continue
        step_num = int(step_result["step"])
        layers = step_result.get("layers", [])
        if not isinstance(layers, list):
            continue

        # Criterion 1 boundary is strictly top-6 SET difference (< 100.0%).
        # Order-only differences within the same top-6 set do not cut the boundary.
        first_difference_layer = next(
            (
                int(layer["layer"])
                for layer in layers
                if isinstance(layer, dict)
                and "layer" in layer
                and isinstance(layer["layer"], int)
                and layer["layer"] in expected_moe_layer_set
                and _is_finite_number(layer.get("router_top_set_pct"))
                and float(layer["router_top_set_pct"]) < 100.0
            ),
            None,
        )
        if first_difference_layer is not None:
            first_layer = next(
                layer
                for layer in layers
                if isinstance(layer, dict) and int(layer.get("layer", -1)) == first_difference_layer
            )
            difference: dict[str, Any] = {
                "step": step_num,
                "layer": first_difference_layer,
                "router_top_set_pct": float(first_layer["router_top_set_pct"]),
            }
            for key in (
                "router_top_golden",
                "router_top_host",
                "router_changed_experts",
                "router_nearest_changed_score_gap",
            ):
                if key in first_layer:
                    difference[key] = first_layer[key]
            first_router_differences.append(difference)

        for layer in layers:
            if not isinstance(layer, dict) or "layer" not in layer:
                continue
            layer_num = int(layer["layer"])
            if (
                layer_num in expected_moe_layer_set
                and _is_finite_number(layer.get("router_top_set_pct"))
            ):
                set_pct = float(layer["router_top_set_pct"])
                all_router_matches.append(set_pct)
                if set_pct < 100.0:
                    set_difference: dict[str, Any] = {
                        "step": step_num,
                        "layer": layer_num,
                        "router_top_set_pct": set_pct,
                    }
                    for key in (
                        "router_top_golden",
                        "router_top_host",
                        "router_changed_experts",
                        "router_nearest_changed_score_gap",
                    ):
                        if key in layer:
                            set_difference[key] = layer[key]
                    router_set_differences.append(set_difference)
                elif layer.get("router_top_order_match") is False:
                    order_diff: dict[str, Any] = {
                        "step": step_num,
                        "layer": layer_num,
                        "router_top_set_pct": set_pct,
                    }
                    if "router_weights_max_err" in layer:
                        order_diff["router_weights_max_err"] = float(
                            layer["router_weights_max_err"]
                        )
                    for key in (
                        "router_top_golden",
                        "router_top_host",
                        "router_nearest_order_score_gap",
                    ):
                        if key in layer:
                            order_diff[key] = layer[key]
                    if "router_nearest_order_score_gap" not in order_diff:
                        g_top_loaded, gap_loaded = _load_order_diff_score_gap_from_golden(
                            golden_dir,
                            step_num,
                            layer_num,
                            host_top=order_diff.get("router_top_host"),
                        )
                        if g_top_loaded is not None and "router_top_golden" not in order_diff:
                            order_diff["router_top_golden"] = g_top_loaded
                        if gap_loaded is not None:
                            order_diff["router_nearest_order_score_gap"] = gap_loaded
                    router_order_differences.append(order_diff)

            if not _is_finite_number(layer.get("streams_out_cos")):
                continue
            record = (
                step_num,
                layer_num,
                float(layer["streams_out_cos"]),
            )
            if first_difference_layer is None or record[1] < first_difference_layer:
                pre_difference.append(record)
            else:
                post_difference.append(record)

    total_steps = len(results)
    argmax_matches = sum(
        bool(result.get("argmax_match")) for result in results if isinstance(result, dict)
    )
    argmax_tie_matches = sum(
        bool(result.get("argmax_tie_match")) for result in results if isinstance(result, dict)
    )
    pre_min = min((record[2] for record in pre_difference), default=None)
    post_min = min((record[2] for record in post_difference), default=None)
    overall_min = min(
        (
            float(layer["streams_out_cos"])
            for result in results
            if isinstance(result, dict)
            for layer in result.get("layers", [])
            if isinstance(layer, dict) and _is_finite_number(layer.get("streams_out_cos"))
        ),
        default=None,
    )
    mean_router = (
        float(np.mean(all_router_matches))
        if all_router_matches
        else (100.0 if not expected_moe_layers else None)
    )

    layers_below_0_9999 = [
        {"step": step, "layer": layer, "streams_out_cos": cosine}
        for step, layer, cosine in pre_difference
        if cosine < 0.9999
    ]
    layers_below_0_9998 = [
        {"step": step, "layer": layer, "streams_out_cos": cosine}
        for step, layer, cosine in pre_difference
        if cosine < 0.9998
    ]

    c1_streams_0_9998 = bool(
        is_complete and pre_min is not None and pre_min >= 0.9998
    )
    c1_streams_0_9999 = bool(
        is_complete and pre_min is not None and pre_min >= 0.9999
    )
    c2_router = bool(
        is_complete and mean_router is not None and mean_router >= 99.0
    )
    c3_exact = bool(
        is_complete and total_steps == expected_steps and argmax_matches == total_steps
    )
    c3_argmax = bool(
        is_complete and total_steps == expected_steps and argmax_tie_matches == total_steps
    )

    first_divergence = layers_below_0_9998[0] if layers_below_0_9998 else None
    first_divergence_0_9999 = layers_below_0_9999[0] if layers_below_0_9999 else None

    updated = dict(payload)
    updated.update({
        "start_step": start_step,
        "end_step": end_step,
        "expected_steps": expected_steps,
        "expected_num_layers": expected_num_layers,
        "expected_moe_layers": expected_moe_layers,
        "expected_moe_layers_source": expected_moe_layers_source,
        "total_steps": total_steps,
        "argmax_matches": argmax_matches,
        "argmax_tie_matches": argmax_tie_matches,
        "mean_router_top_pct": mean_router,
        "min_streams_out_cos": overall_min,
        "pre_first_router_difference": {
            "layer_count": len(pre_difference),
            "min_streams_out_cos": pre_min,
            "below_0_9999_count": len(layers_below_0_9999),
            "below_0_9998_count": len(layers_below_0_9998),
        },
        "at_or_after_first_router_difference": {
            "layer_count": len(post_difference),
            "min_streams_out_cos": post_min,
        },
        "first_router_differences": first_router_differences,
        "router_set_differences": router_set_differences,
        "router_order_differences": router_order_differences,
        "layers_below_0_9999_before_first_router_difference": layers_below_0_9999,
        "layers_below_0_9998_before_first_router_difference": layers_below_0_9998,
        "completeness_errors": completeness_errors,
        "criteria": {
            "results_complete_and_valid": is_complete,
            "layer_output_cos_before_first_router_difference_gte_0_9998": c1_streams_0_9998,
            "layer_output_cos_before_first_router_difference_gte_0_9999": c1_streams_0_9999,
            "router_top_match_gte_99": c2_router,
            "step_argmax_exact_all_match": c3_exact,
            "step_argmax_in_golden_tie_all_match": c3_argmax,
        },
        "first_divergence": first_divergence,
        "first_divergence_0_9999": first_divergence_0_9999,
    })
    return updated


def print_evaluation(summary: dict[str, Any]) -> bool:
    """Print the verification gates, 0.9999 diagnostic list, and router diagnostics."""
    criteria = summary["criteria"]
    is_complete = criteria["results_complete_and_valid"]
    c1_streams = criteria[
        "layer_output_cos_before_first_router_difference_gte_0_9998"
    ]
    c1_streams_9999 = criteria[
        "layer_output_cos_before_first_router_difference_gte_0_9999"
    ]
    c2_router = criteria["router_top_match_gte_99"]
    c3_argmax = criteria["step_argmax_in_golden_tie_all_match"]
    pre = summary["pre_first_router_difference"]
    post = summary["at_or_after_first_router_difference"]
    pre_min = pre["min_streams_out_cos"]
    post_min = post["min_streams_out_cos"]
    pre_min_text = "n/a" if pre_min is None else f"{pre_min:.6f}"
    post_min_text = "n/a" if post_min is None else f"{post_min:.6f}"
    mean_router = summary["mean_router_top_pct"]
    mean_router_text = "n/a" if mean_router is None else f"{mean_router:.2f}%"

    print("-" * 80)
    print("CRITERIA EVALUATION:")
    print(
        f"0. Step/layer completeness ({summary['start_step']}..{summary['end_step']}, "
        f"expected={summary['expected_steps']}): {'PASS' if is_complete else 'FAIL'} "
        f"(recorded={summary['total_steps']})"
    )
    print(
        "   Expected MoE layers: "
        f"{summary['expected_moe_layers']} "
        f"(source: {summary['expected_moe_layers_source']})"
    )
    for err in summary.get("completeness_errors", []):
        print(f"   [Completeness error] {err}")
    print(
        "1. Layer output cosine before first router set difference >= 0.9998: "
        f"{'PASS' if c1_streams else 'FAIL'} "
        f"(min={pre_min_text}, layers={pre['layer_count']})"
    )
    below_9999 = summary.get("layers_below_0_9999_before_first_router_difference", [])
    print(
        "   [Reference 0.9999 check] >= 0.9999: "
        f"{'PASS' if c1_streams_9999 else 'FAIL'} "
        f"(below 0.9999: {len(below_9999)} layer(s))"
    )
    for item in below_9999:
        print(
            f"     Step {item['step']}, layer {item['layer']:03d}: "
            f"streams_out_cos={item['streams_out_cos']:.10f}"
        )
    print(
        f"   Excluded at/after first router set difference: layers={post['layer_count']}, "
        f"min={post_min_text}"
    )
    print(
        f"2. Router top-6 set match >= 99.0%:  {'PASS' if c2_router else 'FAIL'} "
        f"(mean={mean_router_text})"
    )
    print(
        f"3. Every host argmax in golden tie:  {'PASS' if c3_argmax else 'FAIL'} "
        f"({summary['argmax_tie_matches']}/{summary['expected_steps']})"
    )
    differences = summary["first_router_differences"]
    if differences:
        print(f"First router set differences: {len(differences)} step(s)")
        for difference in differences:
            print(
                f"  Step {difference['step']}, layer {difference['layer']:03d}: "
                f"top-6 set={difference['router_top_set_pct']:.1f}%"
            )
    set_differences = summary["router_set_differences"]
    if set_differences:
        print(f"Router top-6 set differences: {len(set_differences)} layer(s)")
        for difference in set_differences:
            score_gap = difference.get("router_nearest_changed_score_gap")
            gap_text = (
                f"nearest golden score gap={score_gap['abs_score_gap']:.8g}"
                if score_gap is not None
                else "score gap unavailable in recorded JSON"
            )
            print(
                f"  Step {difference['step']}, layer {difference['layer']:03d}: "
                f"{gap_text}"
            )
    order_differences = summary.get("router_order_differences", [])
    if order_differences:
        print(
            f"Router top-6 order-only differences (set=100%): "
            f"{len(order_differences)} layer(s)"
        )
        for difference in order_differences:
            score_gap = difference.get("router_nearest_order_score_gap")
            gap_text = (
                f"nearest golden top-6 score gap={score_gap['abs_score_gap']:.8g} "
                f"(experts {score_gap['expert_a']} vs {score_gap['expert_b']})"
                if score_gap is not None
                else "score gap unavailable"
            )
            print(
                f"  Step {difference['step']}, layer {difference['layer']:03d}: "
                f"{gap_text}"
            )
    return bool(is_complete and c1_streams and c2_router and c3_argmax)


def main() -> None:
    import resource

    parser = argparse.ArgumentParser(description="Check host engine against golden dump.")
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
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Path to output JSON summary (default: host/out/golden_check.json for a run)",
    )
    parser.add_argument(
        "--rejudge",
        type=Path,
        help="Re-evaluate criteria from an existing JSON without loading the model",
    )
    parser.add_argument(
        "--start-step",
        type=parse_start_step,
        default=SUPPORTED_START_STEP,
        help=(
            f"Start step; only {SUPPORTED_START_STEP} is supported (a run always resumes "
            f"from snapshot step_000001 and its first decoded step is {SUPPORTED_START_STEP})"
        ),
    )
    parser.add_argument(
        "--end-step",
        type=int,
        default=16,
        help="End step (default: 16)",
    )
    args = parser.parse_args()
    if args.start_step > args.end_step:
        parser.error("--start-step must be less than or equal to --end-step")

    if args.rejudge is not None:
        with open(args.rejudge, encoding="utf-8") as f:
            recorded_payload = loads_strict(f.read())
        summary = evaluate_summary(recorded_payload)
        print(f"Rejudging recorded results: {args.rejudge}")
        passed = print_evaluation(summary)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, allow_nan=False)
            print(f"Rejudged results written to {args.out}")
        raise SystemExit(0 if passed else 1)

    # Live runs read the model checkpoint and golden run from local (untracked)
    # directories; fail with a clear reason naming the flag/env var when absent.
    if args.model_dir is None or not args.model_dir.is_dir():
        raise FileNotFoundError(
            f"model dir {args.model_dir} not found; pass --model-dir or set RAON_MODEL_DIR"
        )
    if args.golden_dir is None or not args.golden_dir.is_dir():
        raise FileNotFoundError(
            f"golden dir {args.golden_dir} not found; pass --golden-dir or set RAON_GOLDEN_DIR"
        )

    # Live runs resume from snapshot step_000001 and step from step 2; reject any
    # other start step BEFORE loading the model or stepping it in a wrong state.
    validate_start_step(args.start_step)

    if args.out is None:
        args.out = Path(__file__).resolve().parent / "out" / "golden_check.json"
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Raon-1-mini Host Engine Golden Verification")
    print("=" * 80)
    print(f"Model dir:  {args.model_dir}")
    print(f"Golden dir: {args.golden_dir}")
    print(f"Steps:      {args.start_step} .. {args.end_step}")
    print("-" * 80)

    # 1. Load config
    cfg_path = args.model_dir / "config.json"
    with open(cfg_path, encoding="utf-8") as f:
        cfg_dict = loads_strict(f.read())
    config = RaonConfig.from_dict(cfg_dict)

    # 2. Load manifest
    manifest_path = args.golden_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json in {args.golden_dir}")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = loads_strict(f.read())

    generated_token_ids = validate_generated_token_ids(
        manifest.get("generated_token_ids"),
        args.end_step,
        vocab_size=getattr(config, "vocab_size", None),
    )
    print(f"Manifest decode steps: {manifest.get('decode_steps')}")
    print(f"Generated tokens count: {len(generated_token_ids)}")
    print(f"Prompt tokens count:    {len(manifest.get('prompt_token_ids', []))}")

    # Check snapshot 1 and all requested step snapshots up front
    snap1_path = args.golden_dir / "step_000001.npz"
    if not snap1_path.is_file():
        raise FileNotFoundError(f"Missing initial snapshot: {snap1_path}")
    verify_requested_step_files(args.golden_dir, args.start_step, args.end_step)

    snap1_npz = np.load(snap1_path)
    # Fail immediately unless the initial snapshot carries every RDA/MLA state
    # pair the config requires (correct keys, shapes, and __dtypes_json__).
    verify_initial_snapshot(snap1_npz, config)
    has_conv_state = all(
        f"layer_{idx:03d}.rda_conv_state" in snap1_npz.files
        for idx in config.rda_layers
    )
    print(
        f"[INFO] Initial snapshot completeness verified: {len(config.rda_layers)} RDA "
        f"layer(s) with rda_state+rda_conv_state, {len(config.mla_layers)} MLA layer(s) "
        f"with a complete cache pair; rda_conv_state present={has_conv_state}."
    )

    # 3. Load weights
    t0_load = time.perf_counter()
    print("Loading weights into host memory (BF16 uint16)...")
    weights = Weights(args.model_dir)
    t_load = time.perf_counter() - t0_load
    load_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
    print(f"Weights loaded in {t_load:.2f}s | Current RSS: {load_rss_gb:.2f} GB")

    # 4. Initialize model
    backend = NumpyBackend(weights)
    model = RaonModel(config, weights, backend=backend)

    # 5. Load snapshot 1
    print(f"Loading snapshot from {snap1_path.name}...")
    model.load_snapshot(snap1_path)

    # 6. Run decode steps
    results: list[dict[str, Any]] = []
    total_tokens_evaluated = 0
    argmax_matches = 0
    argmax_tie_matches = 0
    all_router_top_matches: list[float] = []
    all_streams_cos: list[float] = []

    print("-" * 80)
    print(f"{'Step':>4} | {'InputTok':>8} | {'Golden':>8} | {'Host':>8} | {'Match':>5} | {'MinStrmCos':>10} | {'RtrTop%':>7} | {'LogitsCos':>9} | {'Step(s)':>7}")
    print("-" * 80)

    t_decode_start = time.perf_counter()

    for step in range(args.start_step, args.end_step + 1):
        step_file = args.golden_dir / f"step_{step:06d}.npz"
        if not step_file.is_file():
            raise FileNotFoundError(
                f"Missing required step snapshot during decode: {step_file}"
            )

        golden_npz = np.load(step_file)
        input_tok = generated_token_ids[step - 2]
        expected_next_tok = generated_token_ids[step - 1] if (step - 1) < len(generated_token_ids) else None

        t_step0 = time.perf_counter()
        host_logits = model.step(input_tok)
        t_step = time.perf_counter() - t_step0

        # Golden logits top20
        golden_top20_ids = golden_npz["logits_top20_ids"].ravel()[:20]
        golden_top20_vals = golden_npz["logits_top20_values"].ravel()[:20]
        golden_argmax = int(golden_top20_ids[0])
        golden_max = golden_top20_vals[0]
        golden_argmax_ties = {
            int(token_id)
            for token_id, value in zip(golden_top20_ids, golden_top20_vals)
            if value == golden_max
        }

        host_top20_ids = np.argsort(-host_logits)[:20]
        host_top20_vals = host_logits[host_top20_ids]
        host_argmax = int(host_top20_ids[0])

        is_argmax_match = (host_argmax == golden_argmax)
        is_argmax_tie_match = host_argmax in golden_argmax_ties
        if is_argmax_match:
            argmax_matches += 1
        if is_argmax_tie_match:
            argmax_tie_matches += 1
        total_tokens_evaluated += 1

        top20_set_overlap = len(set(host_top20_ids) & set(golden_top20_ids))

        # Partial logits cosine on top-20 tokens
        host_at_golden_top20 = host_logits[golden_top20_ids]
        logits_cos = cosine_similarity(golden_top20_vals, host_at_golden_top20)

        # Layer by layer evaluation
        layer_records: list[dict[str, Any]] = []
        step_min_strm_cos = 1.0
        step_rtr_matches: list[float] = []

        for layer_idx in range(config.num_hidden_layers):
            pfx = f"layer_{layer_idx:03d}"
            rec: dict[str, Any] = {"layer": layer_idx}

            # streams_in
            k_sin = f"{pfx}.streams_in"
            if k_sin in golden_npz and k_sin in model.last_intermediates:
                g_sin = bf16_to_f32(golden_npz[k_sin])
                h_sin = bf16_to_f32(model.last_intermediates[k_sin])
                cos_sin = cosine_similarity(g_sin, h_sin)
                rec["streams_in_cos"] = cos_sin

            # mixer_in
            k_min = f"{pfx}.mixer_in"
            if k_min in golden_npz and k_min in model.last_intermediates:
                g_min = bf16_to_f32(golden_npz[k_min])
                h_min = bf16_to_f32(model.last_intermediates[k_min])
                cos_min = cosine_similarity(g_min, h_min)
                rec["mixer_in_cos"] = cos_min
                rec["mixer_in_relative_error"] = relative_error(g_min, h_min)

            # mixer_out
            k_mout = f"{pfx}.mixer_out"
            if k_mout in golden_npz and k_mout in model.last_intermediates:
                g_mout = bf16_to_f32(golden_npz[k_mout])
                h_mout = bf16_to_f32(model.last_intermediates[k_mout])
                cos_mout = cosine_similarity(g_mout, h_mout)
                rec["mixer_out_cos"] = cos_mout
                rec["mixer_out_relative_error"] = relative_error(g_mout, h_mout)

            # streams_out
            k_sout = f"{pfx}.streams_out"
            if k_sout in golden_npz and k_sout in model.last_intermediates:
                g_sout = bf16_to_f32(golden_npz[k_sout])
                h_sout = bf16_to_f32(model.last_intermediates[k_sout])
                cos_sout = cosine_similarity(g_sout, h_sout)
                rec["streams_out_cos"] = cos_sout
                rec["streams_out_relative_error"] = relative_error(g_sout, h_sout)
                if cos_sout < step_min_strm_cos:
                    step_min_strm_cos = cos_sout
                all_streams_cos.append(cos_sout)

            # MoE router metrics
            if config.is_moe_layer(layer_idx):
                k_rtop = f"{pfx}.router_top"
                k_rwt = f"{pfx}.router_weights"
                k_rlog = f"{pfx}.router_logits"
                if k_rtop in golden_npz and k_rtop in model.last_intermediates:
                    g_top = golden_npz[k_rtop].ravel()
                    h_top = model.last_intermediates[k_rtop].ravel()
                    set_overlap = len(set(g_top) & set(h_top)) / float(len(g_top))
                    order_match = bool(np.array_equal(g_top, h_top))
                    rec["router_top_set_pct"] = set_overlap * 100.0
                    rec["router_top_order_match"] = order_match
                    step_rtr_matches.append(set_overlap * 100.0)
                    all_router_top_matches.append(set_overlap * 100.0)
                    if set_overlap < 1.0:
                        golden_only = sorted(set(map(int, g_top)) - set(map(int, h_top)))
                        host_only = sorted(set(map(int, h_top)) - set(map(int, g_top)))
                        rec["router_top_golden"] = list(map(int, g_top))
                        rec["router_top_host"] = list(map(int, h_top))
                        rec["router_changed_experts"] = {
                            "golden_only": golden_only,
                            "host_only": host_only,
                        }
                        if k_rlog in golden_npz:
                            host_router_logits = model.last_intermediates.get(k_rlog)
                            rec["router_nearest_changed_score_gap"] = (
                                _nearest_changed_router_score_gap(
                                    g_top,
                                    h_top,
                                    golden_npz[k_rlog],
                                    host_router_logits,
                                )
                            )
                    elif not order_match:
                        rec["router_top_golden"] = list(map(int, g_top))
                        rec["router_top_host"] = list(map(int, h_top))
                        if k_rlog in golden_npz:
                            host_router_logits = model.last_intermediates.get(k_rlog)
                            rec["router_nearest_order_score_gap"] = (
                                _nearest_order_router_score_gap(
                                    g_top,
                                    golden_npz[k_rlog],
                                    host_top=h_top,
                                    host_logits=host_router_logits,
                                )
                            )

                if k_rwt in golden_npz and k_rwt in model.last_intermediates:
                    g_wt = golden_npz[k_rwt].ravel()
                    h_wt = model.last_intermediates[k_rwt].ravel()
                    max_wt_err = float(np.max(np.abs(g_wt - h_wt)))
                    rec["router_weights_max_err"] = max_wt_err

            layer_records.append(rec)

        avg_rtr_pct = float(np.mean(step_rtr_matches)) if step_rtr_matches else 100.0

        match_str = "YES" if is_argmax_match else "NO"
        print(
            f"{step:>4} | {input_tok:>8} | {golden_argmax:>8} | {host_argmax:>8} | "
            f"{match_str:>5} | {step_min_strm_cos:>10.6f} | {avg_rtr_pct:>6.1f}% | "
            f"{logits_cos:>9.6f} | {t_step:>6.2f}s"
        )

        results.append({
            "step": step,
            "input_tok": int(input_tok),
            "golden_argmax": golden_argmax,
            "host_argmax": host_argmax,
            "argmax_match": is_argmax_match,
            "argmax_tie_match": is_argmax_tie_match,
            "golden_argmax_ties": sorted(golden_argmax_ties),
            "top20_set_overlap": top20_set_overlap,
            "logits_cos": logits_cos,
            "min_streams_cos": step_min_strm_cos,
            "avg_router_top_pct": avg_rtr_pct,
            "step_time_s": t_step,
            "layers": layer_records,
        })

    t_decode_total = time.perf_counter() - t_decode_start
    final_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

    # 7. Print summary and criteria check
    print("-" * 80)
    print("VERIFICATION SUMMARY")
    print("-" * 80)
    print(f"Total steps evaluated:     {total_tokens_evaluated}")
    print(f"Argmax match count:        {argmax_matches} / {total_tokens_evaluated} ({argmax_matches/max(1,total_tokens_evaluated)*100:.1f}%)")
    print(f"Argmax-in-tie count:       {argmax_tie_matches} / {total_tokens_evaluated} ({argmax_tie_matches/max(1,total_tokens_evaluated)*100:.1f}%)")
    mean_rtr_pct = float(np.mean(all_router_top_matches)) if all_router_top_matches else 100.0
    min_strm_overall = float(np.min(all_streams_cos)) if all_streams_cos else 1.0
    print(f"Mean router top-6 match:   {mean_rtr_pct:.2f}%")
    print(f"Min streams_out cosine:    {min_strm_overall:.6f}")
    print(f"Total decode time:         {t_decode_total:.2f}s ({total_tokens_evaluated/max(0.001, t_decode_total):.2f} tok/s)")
    print(f"Peak memory RSS:           {final_rss_gb:.2f} GB")

    summary_payload = {
        "model_dir": str(args.model_dir),
        "golden_dir": str(args.golden_dir),
        "has_conv_state_in_snapshot": has_conv_state,
        "start_step": int(args.start_step),
        "end_step": int(args.end_step),
        "expected_steps": int(args.end_step - args.start_step + 1),
        "expected_num_layers": int(config.num_hidden_layers),
        "expected_moe_layers": sorted(
            layer_idx
            for layer_idx in range(config.num_hidden_layers)
            if config.is_moe_layer(layer_idx)
        ),
        "total_steps": total_tokens_evaluated,
        "argmax_matches": argmax_matches,
        "argmax_tie_matches": argmax_tie_matches,
        "mean_router_top_pct": mean_rtr_pct,
        "min_streams_out_cos": min_strm_overall,
        "load_time_s": t_load,
        "decode_time_s": t_decode_total,
        "tokens_per_second": total_tokens_evaluated / max(0.001, t_decode_total),
        "peak_rss_gb": final_rss_gb,
        "step_results": results,
    }
    summary_payload = evaluate_summary(summary_payload)
    passed = print_evaluation(summary_payload)
    print("=" * 80)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)
    print(f"Results written to {args.out}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
