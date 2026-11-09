# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Laptop NPU Golden Verification Runner for Raon-1-mini D mode.

Runs sequential decode steps 2..16 from initial golden snapshot 1 on AMD XDNA 2 NPU.
Reports:
1. Per-layer golden intermediate parity (mixer_out, streams_out cosine, router_top overlap).
2. Full 25-layer golden logit parity (Top-1 tie-aware match, Top-20 overlap, Logit Cosine).
3. Parity against host NumPy backend with identical D weights via DWeights (Layer output cos >= 0.999).
4. Pure execution latencies and call counts per category: NPU GEMV by bundle kind, RDA, MoE (B1/shared/B2), MLA attend, and host prep.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import faulthandler
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from host.backend import NumpyBackend
from host.check_golden import (
    SUPPORTED_START_STEP,
    parse_start_step,
    validate_generated_token_ids,
    validate_start_step,
)
from host.raon import RaonConfig, RaonModel
from host.strict_json import loads_strict
from host.weights import Weights, bf16_to_f32
from npu.backend import DWeights, NpuBackend, ensure_mla_cache_expanded


# 2026-10-28 laptop run: 15 steps, tie-aware top-1 14/15, minimum cosine 0.99941.
DEFAULT_MIN_COS = 0.999
DEFAULT_MIN_TOP1_TIE_RATE = 0.9
MAX_GOLDEN_STEPS = 4096
MAX_LAYERS = 25
MAX_TIMEOUT_PER_TOKEN = 3600.0
MAX_TOTAL_TIMEOUT = 86400.0


class _NpuStepDeadlineExceeded(RuntimeError):
    def __init__(self, code: str, elapsed: float, limit: float):
        super().__init__(f"{code}: NPU step took {elapsed:.2f}s, exceeding {limit}s")
        self.code = code
        self.elapsed = elapsed
        self.limit = limit


def _run_npu_step_with_timeout(
    model: RaonModel,
    input_token: int,
    *,
    timeout_per_token: float,
    remaining_total: float | None = None,
) -> tuple[np.ndarray, float]:
    """Run one NPU model step under a hard watchdog and post-return deadlines."""
    if remaining_total is not None and remaining_total <= 0.0:
        raise _NpuStepDeadlineExceeded("total_timeout", 0.0, remaining_total)

    watchdog_timeout = (
        timeout_per_token
        if remaining_total is None
        else min(timeout_per_token, remaining_total)
    )
    watchdog_stream = os.fdopen(os.dup(2), "w", closefd=True)
    started = time.perf_counter()
    try:
        faulthandler.dump_traceback_later(
            watchdog_timeout,
            repeat=False,
            file=watchdog_stream,
            exit=True,
        )
        result = model.step(input_token)
    finally:
        faulthandler.cancel_dump_traceback_later()
        watchdog_stream.close()

    elapsed = time.perf_counter() - started
    if remaining_total is not None and elapsed > remaining_total:
        raise _NpuStepDeadlineExceeded("total_timeout", elapsed, remaining_total)
    if elapsed > timeout_per_token:
        raise _NpuStepDeadlineExceeded("token_timeout", elapsed, timeout_per_token)
    return result, elapsed


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity in FP64 precision."""
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0:
        return 1.0 if np.array_equal(a_f, b_f) else 0.0
    return float(np.dot(a_f, b_f) / denom)


def _to_f32(u: np.ndarray) -> np.ndarray:
    u_arr = np.asarray(u)
    return bf16_to_f32(u_arr) if u_arr.dtype == np.uint16 else u_arr.astype(np.float32)


def _load_and_expand_snapshot(model: RaonModel, snap_path: Path) -> int:
    """Load snapshot into model and expand MLA latent caches if not already populated."""
    model.load_snapshot(snap_path)
    return ensure_mla_cache_expanded(model, snap_path)


GOLDEN_ASSESSMENT_NUMERIC_FIELDS: tuple[str, ...] = ("logit_cosine",)


def _finite_min(
    values: Any,
    *,
    default: float | None = None,
) -> tuple[float | None, bool]:
    """Compute min of values if all are finite numbers.

    Returns:
        (min_value, has_non_finite)
        - If any value is non-finite (NaN, ±Inf) or not a number, returns (None, True).
        - If values is empty, returns (default, False).
        - Otherwise returns (float(min(valid_values)), False).
    """
    import math

    items = list(values)
    if not items:
        return (default, False)

    for v in items:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return (None, True)
        if not math.isfinite(v):
            return (None, True)

    return (float(min(items)), False)


def ranged_int(
    min_val: int | None = None,
    max_val: int | None = None,
    *,
    name: str = "integer",
) -> Any:
    def _parse(val_str: str) -> int:
        try:
            val = int(val_str)
        except (ValueError, TypeError):
            raise argparse.ArgumentTypeError(f"{name} must be an integer, got {val_str!r}")
        if min_val is not None and val < min_val:
            raise argparse.ArgumentTypeError(f"{name} must be >= {min_val}, got {val}")
        if max_val is not None and val > max_val:
            raise argparse.ArgumentTypeError(f"{name} must be <= {max_val}, got {val}")
        return val

    _parse.__name__ = f"ranged_int_{name}"
    _parse.is_ranged = True
    _parse.type_cls = int
    _parse.min_val = min_val
    _parse.max_val = max_val
    _parse.min_inclusive = True
    _parse.max_inclusive = True
    _parse.name = name
    return _parse


def ranged_float(
    min_val: float | None = None,
    max_val: float | None = None,
    *,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
    name: str = "float",
) -> Any:
    import math

    def _parse(val_str: str) -> float:
        try:
            val = float(val_str)
        except (ValueError, TypeError):
            raise argparse.ArgumentTypeError(f"{name} must be a number, got {val_str!r}")
        if not math.isfinite(val):
            raise argparse.ArgumentTypeError(f"{name} must be finite, got {val_str!r}")
        if min_val is not None:
            if min_inclusive and val < min_val:
                raise argparse.ArgumentTypeError(f"{name} must be >= {min_val}, got {val}")
            if not min_inclusive and val <= min_val:
                raise argparse.ArgumentTypeError(f"{name} must be > {min_val}, got {val}")
        if max_val is not None:
            if max_inclusive and val > max_val:
                raise argparse.ArgumentTypeError(f"{name} must be <= {max_val}, got {val}")
            if not max_inclusive and val >= max_val:
                raise argparse.ArgumentTypeError(f"{name} must be < {max_val}, got {val}")
        return val

    _parse.__name__ = f"ranged_float_{name}"
    _parse.is_ranged = True
    _parse.type_cls = float
    _parse.min_val = min_val
    _parse.max_val = max_val
    _parse.min_inclusive = min_inclusive
    _parse.max_inclusive = max_inclusive
    _parse.name = name
    return _parse


def _assess_baseline(
    streams_cosines: list[float],
    mixer_cosines: list[float],
    cos_logits: float,
    *,
    min_threshold: float = 0.999,
) -> dict[str, Any]:
    min_ly_cos, streams_non_finite = _finite_min(streams_cosines)
    min_mixer_cos, mixer_non_finite = _finite_min(mixer_cosines, default=1.0)
    cos_logits_val, logits_non_finite = _finite_min([cos_logits])

    has_non_finite = streams_non_finite or mixer_non_finite or logits_non_finite
    pass_0999 = bool(
        not has_non_finite
        and min_ly_cos is not None
        and min_ly_cos >= min_threshold
    )
    baseline_passed = pass_0999

    res: dict[str, Any] = {
        "status": "PASS" if pass_0999 else "FAIL",
        "logits_cos": cos_logits_val,
        "min_layer_streams_cos": min_ly_cos,
        "min_layer_mixer_cos": min_mixer_cos,
        "pass_0999": pass_0999,
        "baseline_passed": baseline_passed,
    }
    if streams_non_finite:
        res["min_layer_streams_cos_non_finite"] = True
    if mixer_non_finite:
        res["min_layer_mixer_cos_non_finite"] = True
    if logits_non_finite:
        res["logits_cos_non_finite"] = True
    if has_non_finite:
        res["baseline_non_finite"] = True
    return res


def normalize_payload_finite(data: Any, path: str = "") -> Any:
    """Recursively replace non-finite float values with None and record their paths in payload['non_finite_paths']."""
    def _walk(node: Any, current_path: str, paths: list[str]) -> Any:
        if isinstance(node, (float, np.floating)):
            if not math.isfinite(float(node)):
                if current_path:
                    paths.append(current_path)
                return None
            return float(node)
        if isinstance(node, dict):
            new_dict: dict[str, Any] = {}
            for k, v in node.items():
                if k == "non_finite_paths":
                    continue
                sub_path = f"{current_path}.{k}" if current_path else str(k)
                new_dict[k] = _walk(v, sub_path, paths)
            return new_dict
        if isinstance(node, (list, tuple)):
            new_list: list[Any] = []
            for idx, item in enumerate(node):
                sub_path = f"{current_path}[{idx}]" if current_path else f"[{idx}]"
                new_list.append(_walk(item, sub_path, paths))
            return new_list if isinstance(node, list) else tuple(new_list)
        return node

    collected_paths: list[str] = []
    normalized = _walk(data, path, collected_paths)
    if isinstance(normalized, dict):
        if collected_paths:
            normalized["non_finite_paths"] = sorted(collected_paths)
    return normalized


def _latency_seconds_per_token(avg_total_ms: float | None) -> float | None:
    return (avg_total_ms / 1e3) if avg_total_ms is not None else None


def _summarize_metrics(
    step_records: list[dict[str, Any]],
) -> dict[str, Any]:
    all_streams_cos = [
        ly["streams_out"]
        for r in step_records
        for ly in r.get("layers", [])
        if "streams_out" in ly
    ]
    all_mixer_cos = [
        ly["mixer_out"]
        for r in step_records
        for ly in r.get("layers", [])
        if "mixer_out" in ly
    ]
    all_router_overlap = [
        ly["router_overlap"]
        for r in step_records
        for ly in r.get("layers", [])
        if "router_overlap" in ly
    ]
    min_streams_cos, streams_non_finite = _finite_min(all_streams_cos, default=0.0)
    min_mixer_cos, mixer_non_finite = _finite_min(all_mixer_cos, default=0.0)
    if all_router_overlap and any(not math.isfinite(x) for x in all_router_overlap):
        avg_router_overlap = None
    else:
        avg_router_overlap = float(np.mean(all_router_overlap)) if all_router_overlap else 0.0

    res: dict[str, Any] = {
        "min_streams_cos": min_streams_cos,
        "min_mixer_cos": min_mixer_cos,
        "avg_router_overlap": avg_router_overlap,
    }
    if streams_non_finite:
        res["min_streams_cos_non_finite"] = True
    if mixer_non_finite:
        res["min_mixer_cos_non_finite"] = True
    return res


def _records_have_finite_metrics(
    step_records: list[dict[str, Any]],
    fields: tuple[str, ...] = GOLDEN_ASSESSMENT_NUMERIC_FIELDS,
) -> bool:
    """Return True iff all numeric assessment fields across step records are finite."""
    import math

    for record in step_records:
        for field in fields:
            if field in record:
                val = record[field]
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    return False
                if not math.isfinite(val):
                    return False
    return True


def _assess_verification(
    *,
    start_step: int,
    end_step: int,
    step_records: list[dict[str, Any]],
    stop_reason: dict[str, Any],
    full_25_layers: bool,
    baseline_passed: bool | None,
    min_cos: float,
    min_top1_tie_rate: float,
) -> dict[str, Any]:
    """Build the completion and golden-accuracy gates used by the CLI exit status."""
    requested_steps = list(range(start_step, end_step + 1))
    completed_steps = [int(record["step"]) for record in step_records]
    run_complete = (
        completed_steps == requested_steps
        and stop_reason.get("code") == "completed"
    )

    finite_metrics = _records_have_finite_metrics(step_records)

    logit_cosines = [
        float(record["logit_cosine"])
        for record in step_records
        if "logit_cosine" in record
    ]
    tie_matches = sum(bool(record.get("top1_tie_match")) for record in step_records)
    min_logit_cosine, logit_non_finite = _finite_min(logit_cosines, default=None)
    if not finite_metrics:
        min_logit_cosine = None
        logit_non_finite = True
    top1_tie_match_rate = tie_matches / len(step_records) if step_records else 0.0

    if full_25_layers:
        metrics_complete = bool(
            len(logit_cosines) == len(requested_steps)
            and finite_metrics
            and not logit_non_finite
        )
        golden_accuracy_passed = bool(
            metrics_complete
            and min_logit_cosine is not None
            and min_logit_cosine >= min_cos
            and top1_tie_match_rate >= min_top1_tie_rate
        )
    else:
        metrics_complete = False
        golden_accuracy_passed = False

    res: dict[str, Any] = {
        "requested_steps": requested_steps,
        "completed_steps": completed_steps,
        "run_complete": run_complete,
        "golden_metrics_complete": metrics_complete,
        "min_logit_cosine": min_logit_cosine,
        "top1_tie_match_rate": top1_tie_match_rate,
        "min_cos_threshold": min_cos,
        "min_top1_tie_rate_threshold": min_top1_tie_rate,
        "golden_accuracy_passed": golden_accuracy_passed,
        "verification_passed": bool(run_complete and (baseline_passed is True) and golden_accuracy_passed),
    }
    if logit_non_finite:
        res["min_logit_cosine_non_finite"] = True
    return res


def run_golden_verification(
    weights_d_dir: Path,
    golden_dir: Path,
    model_dir: Path | None = None,
    gemv: str = "unified",
    start_step: int = 2,
    end_step: int = 16,
    layer_limit: int | None = None,
    timeout_per_token: float = 120.0,
    total_timeout: float = 1800.0,
    compare_host_d: bool = True,
    preload_weights: bool = True,
    mock: bool = False,
    out_json: Path | None = None,
    min_cos: float = DEFAULT_MIN_COS,
    min_top1_tie_rate: float = DEFAULT_MIN_TOP1_TIE_RATE,
) -> dict[str, Any]:
    """Execute end-to-end golden verification on NPU with early-exit guards."""
    validate_start_step(start_step)
    if start_step > end_step:
        raise ValueError(f"start_step ({start_step}) must be <= end_step ({end_step})")
    if out_json is not None:
        out_json.unlink(missing_ok=True)

    t_run0 = time.perf_counter()

    def log(msg: str) -> None:
        print(f"[{time.perf_counter() - t_run0:8.1f}s] {msg}", flush=True)

    print("=" * 85, flush=True)
    print(f"[{now_str()}] Raon-1-mini NPU Golden Verification (D Mode)", flush=True)
    print("=" * 85, flush=True)
    print(f"GEMV Engine:    {gemv}", flush=True)
    print(f"Weights Dir:    {weights_d_dir}", flush=True)
    print(f"Golden Dir:     {golden_dir}", flush=True)
    print(f"Model Dir:      {model_dir or 'Derived from weights-d'}", flush=True)
    print(f"Step Range:     {start_step} .. {end_step}", flush=True)
    print(f"Layers:         {layer_limit or 25} (of 25)", flush=True)
    print(f"Timeouts:       token={timeout_per_token}s, total={total_timeout}s", flush=True)
    print(f"Golden Gates:   per-step cos>={min_cos}, tie-aware top-1 rate>={min_top1_tie_rate}", flush=True)
    print(f"Execution Mode: {'MOCK / CPU' if mock else 'NPU Hardware (XDNA 2)'}", flush=True)
    print("-" * 85, flush=True)

    # 1. Load Raon configuration
    config_path = (
        (model_dir / "config.json")
        if (model_dir and (model_dir / "config.json").is_file())
        else ((weights_d_dir / "config.json") if (weights_d_dir / "config.json").is_file() else None)
    )
    if config_path and config_path.is_file():
        cfg_dict = loads_strict(config_path.read_bytes())
        config = RaonConfig.from_dict(cfg_dict)
    else:
        config = RaonConfig()

    if layer_limit is not None and layer_limit < config.num_hidden_layers:
        config.num_hidden_layers = layer_limit
        log(f"Layer limit applied: running {config.num_hidden_layers} hidden layers.")

    num_layers = config.num_hidden_layers
    full_25_layers = (num_layers == 25)

    # 2. Load golden manifest
    manifest_path = golden_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing golden manifest.json under {golden_dir}")
    golden_manifest = loads_strict(manifest_path.read_bytes())

    generated_token_ids = validate_generated_token_ids(
        golden_manifest.get("generated_token_ids"),
        end_step,
        vocab_size=getattr(config, "vocab_size", None),
    )
    log(f"Manifest generated tokens: {len(generated_token_ids)} tokens")

    # 3. Verify snapshot 1 exists
    snap1_path = golden_dir / "step_000001.npz"
    if not snap1_path.is_file():
        raise FileNotFoundError(f"Initial golden snapshot missing: {snap1_path}")

    # 4. Load Host weights (skip 128 routed experts to fit laptop RAM)
    host_weights_source = (
        model_dir
        if model_dir is not None
        else ((weights_d_dir / "host") if (weights_d_dir / "host").is_dir() else weights_d_dir)
    )
    log(f"Loading host weights from {host_weights_source} (skip_moe_experts=True)...")
    t0_load = time.perf_counter()
    host_weights: Weights | None = None
    if host_weights_source.is_dir() and any(host_weights_source.glob("*.safetensors")):
        host_weights = Weights(
            host_weights_source,
            selected_layers=list(range(num_layers)),
            skip_moe_experts=True,
        )
        t_load = time.perf_counter() - t0_load
        log(f"Host weights loaded in {t_load:.2f}s ({len(host_weights.tensors)} tensors).")
    else:
        log(f"[NOTICE] No safetensors found in {host_weights_source}; relying on DWeights / NPU binaries.")

    # 5. Initialize NPU Backend and preload weight bundles
    log(f"Initializing NpuBackend (gemv={gemv})...")
    npu_backend = NpuBackend(
        weights_dir=weights_d_dir,
        host_weights=host_weights,
        mock=mock,
        device="cpu" if mock else "npu",
        gemv=gemv,
    )
    preload_summary: dict[str, Any] | None = None
    if preload_weights:
        preload_summary = npu_backend.preload(num_layers=num_layers)

    # Ensure model has a weights container for embeddings/norms
    model_weights: Weights
    if host_weights is not None:
        model_weights = host_weights
    else:
        model_weights = DWeights(
            weights_d_dir,
            host_weights=None,
            selected_layers=list(range(num_layers)),
        )

    model = RaonModel(config, model_weights, backend=npu_backend)

    # 6. Single-step parity comparison against Host NumPy backend with identical D weights
    host_numpy_parity_step: dict[str, Any] | None = None
    baseline_passed: bool | None = None
    baseline_skipped = False
    if compare_host_d:
        baseline_passed = True
        print("-" * 85, flush=True)
        baseline_step = start_step
        log(f"[Baseline Check] Comparing Step {baseline_step} against host NumpyBackend(DWeights)...")
        try:
            d_weights = DWeights(
                weights_d_dir,
                host_weights=host_weights,
                selected_layers=list(range(num_layers)),
            )
            host_backend = NumpyBackend(d_weights)
            host_model = RaonModel(config, d_weights, backend=host_backend)

            _load_and_expand_snapshot(host_model, snap1_path)
            tok_idx = baseline_step - 2
            tok_base = generated_token_ids[tok_idx]

            t0_h = time.perf_counter()
            host_logits_ref = host_model.step(tok_base)
            t_h = time.perf_counter() - t0_h

            _load_and_expand_snapshot(model, snap1_path)
            npu_backend.set_step(baseline_step)
            npu_logits_test, t_n = _run_npu_step_with_timeout(
                model,
                tok_base,
                timeout_per_token=timeout_per_token,
            )

            layer_parity: list[dict[str, Any]] = []
            streams_cosines: list[float] = []
            mixer_cosines: list[float] = []
            for ly in range(num_layers):
                ly_rec: dict[str, Any] = {"layer": ly}
                km = f"layer_{ly:03d}.mixer_out"
                ks = f"layer_{ly:03d}.streams_out"
                if km in model.last_intermediates and km in host_model.last_intermediates:
                    cm = cosine_sim(
                        _to_f32(model.last_intermediates[km]),
                        _to_f32(host_model.last_intermediates[km]),
                    )
                    ly_rec["mixer_out_cos"] = cm
                    mixer_cosines.append(cm)
                if ks in model.last_intermediates and ks in host_model.last_intermediates:
                    cs = cosine_sim(
                        _to_f32(model.last_intermediates[ks]),
                        _to_f32(host_model.last_intermediates[ks]),
                    )
                    ly_rec["streams_out_cos"] = cs
                    streams_cosines.append(cs)
                layer_parity.append(ly_rec)

            if not streams_cosines:
                raise RuntimeError("No layer streams_out intermediates produced during baseline check")

            cos_logits = cosine_sim(npu_logits_test, host_logits_ref)
            base_eval = _assess_baseline(streams_cosines, mixer_cosines, cos_logits)
            min_ly_cos = base_eval["min_layer_streams_cos"]
            min_mixer_cos = base_eval["min_layer_mixer_cos"]
            pass_0999 = base_eval["pass_0999"]
            baseline_passed = base_eval["baseline_passed"]

            log_logits_str = f"{cos_logits:.6f}" if base_eval["logits_cos"] is not None else "non-finite"
            log_ly_str = f"{min_ly_cos:.6f}" if min_ly_cos is not None else "non-finite"
            log_mixer_str = f"{min_mixer_cos:.6f}" if min_mixer_cos is not None else "non-finite"

            log(
                f"[Baseline Check Step {baseline_step}] Logits Cos: {log_logits_str} | "
                f"Min Layer Streams Cos: {log_ly_str} | Min Mixer Cos: {log_mixer_str} | "
                f"Host: {t_h*1e3:.1f}ms, NPU: {t_n*1e3:.1f}ms"
            )

            host_numpy_parity_step = {
                "step": baseline_step,
                **base_eval,
                "host_latency_ms": t_h * 1e3,
                "npu_latency_ms": t_n * 1e3,
                "layers": layer_parity,
            }
            if not pass_0999:
                log(f"[FAIL] Baseline parity below 0.999 ({min_ly_cos} < 0.999)")
            else:
                log("[PASS] Baseline parity verified (cos >= 0.999).")
        except Exception as e:
            baseline_passed = False
            log(f"[FAIL] Host DWeights baseline check failed: {e}")
            host_numpy_parity_step = {
                "step": baseline_step,
                "status": "ERROR",
                "error": str(e),
                "pass_0999": False,
            }
        print("-" * 85, flush=True)
    else:
        baseline_passed = None
        baseline_skipped = True

    # 7. Load initial state from snapshot 1 and expand MLA latent caches if needed
    log(f"Loading initial state from {snap1_path.name}...")
    expanded_mla = _load_and_expand_snapshot(model, snap1_path)
    if expanded_mla > 0:
        log(f"Expanded MLA latent cache via NPU kv_b for {expanded_mla} layer(s).")

    # 8. Sequential Decode Steps Loop (start_step .. end_step)
    log(f"Starting sequential decode steps ({start_step} .. {end_step}), layers={num_layers}...")
    print("-" * 85, flush=True)

    step_records: list[dict[str, Any]] = []
    total_run_start = time.perf_counter()
    top1_matches = 0
    top1_tie_matches = 0
    steps_completed = 0
    stop_reason: dict[str, Any] | None = None

    for step in range(start_step, end_step + 1):
        elapsed_total = time.perf_counter() - total_run_start
        if elapsed_total > total_timeout:
            log(f"[EARLY STOP] Total timeout of {total_timeout}s reached.")
            stop_reason = {
                "code": "total_timeout",
                "step": step,
                "message": f"total timeout of {total_timeout}s reached before step {step}",
            }
            break

        snap_file = golden_dir / f"step_{step:06d}.npz"
        if not snap_file.is_file():
            log(f"Golden snapshot {snap_file.name} not found; terminating run.")
            stop_reason = {
                "code": "missing_snapshot",
                "step": step,
                "message": f"golden snapshot {snap_file.name} not found",
            }
            break

        tok_idx = step - 2
        input_token = generated_token_ids[tok_idx]

        # Notify backend of step and reset per-step timings
        npu_backend.set_step(step)
        npu_backend.reset_timings()

        # Measure pure decode latency (exclude golden loading & metric calculations)
        remaining_total = total_timeout - (time.perf_counter() - total_run_start)
        try:
            logits_f32, t_step_total = _run_npu_step_with_timeout(
                model,
                input_token,
                timeout_per_token=timeout_per_token,
                remaining_total=remaining_total,
            )
        except _NpuStepDeadlineExceeded as e:
            limit_name = "total" if e.code == "total_timeout" else "token"
            log(
                f"[EARLY STOP] Step {step} exceeded {limit_name} timeout "
                f"({e.elapsed:.2f}s > {e.limit}s)."
            )
            stop_reason = {
                "code": e.code,
                "step": step,
                "message": str(e),
            }
            break
        step_total_ms = t_step_total * 1e3

        # Snapshot timing and call counters before any verification work
        timing_s = {
            k: round(float(v), 6)
            for k, v in sorted(npu_backend.timings.items())
            if v > 0.0 or npu_backend.call_counts.get(k, 0) > 0
        }
        calls = {
            k: int(v)
            for k, v in sorted(npu_backend.call_counts.items())
            if v > 0
        }

        npu_gemv_ms = npu_backend.timings.get("linear_npu", 0.0) * 1e3
        npu_rda_ms = npu_backend.timings.get("rda_npu", 0.0) * 1e3
        npu_moe_b1_ms = npu_backend.timings.get("moe_b1_npu", 0.0) * 1e3
        npu_moe_sh_ms = npu_backend.timings.get("moe_sh_npu", 0.0) * 1e3
        npu_moe_dn_ms = npu_backend.timings.get("moe_dn_npu", 0.0) * 1e3
        npu_moe_ms = npu_moe_b1_ms + npu_moe_sh_ms + npu_moe_dn_ms
        mla_attend_ms = npu_backend.timings.get("mla_attend", 0.0) * 1e3
        host_prep_ms = npu_backend.timings.get("host_prep", 0.0) * 1e3
        pure_npu_ms = npu_gemv_ms + npu_rda_ms + npu_moe_ms
        host_ops_ms = max(0.0, step_total_ms - pure_npu_ms)

        # ---------------------------------------------------------------------
        # Evaluation against Golden Snapshot (excluded from decode timing)
        # Follows probe_layers.py format and structure
        # ---------------------------------------------------------------------
        golden_npz = np.load(snap_file)
        rec: dict[str, Any] = {
            "step": step,
            "input_token": int(input_token),
            "seconds": t_step_total,
            "latency_total_ms": step_total_ms,
            "latency_pure_npu_ms": pure_npu_ms,
            "latency_host_ms": host_ops_ms,
            "layers": [],
        }

        for i in range(num_layers):
            r: dict[str, Any] = {"layer": i}
            for k in ("mixer_out", "streams_out"):
                key = f"layer_{i:03d}.{k}"
                if key in golden_npz.files and key in model.last_intermediates:
                    r[k] = cosine_sim(_to_f32(model.last_intermediates[key]), _to_f32(golden_npz[key]))
            rkey = f"layer_{i:03d}.router_top"
            if rkey in golden_npz.files and rkey in model.last_intermediates:
                r["router_overlap"] = len(
                    set(golden_npz[rkey].ravel().tolist())
                    & set(model.last_intermediates[rkey].ravel().tolist())
                )
            rec["layers"].append(r)
            fields_str = " ".join(
                f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in r.items()
                if k != "layer"
            )
            log(f"step {step} layer {i}: {fields_str}")

        if full_25_layers and "logits_top20_ids" in golden_npz.files:
            golden_top20_ids = golden_npz["logits_top20_ids"].ravel()[:20]
            golden_top20_vals = golden_npz["logits_top20_values"].ravel()[:20]

            golden_top1 = int(golden_top20_ids[0])
            golden_max_val = golden_top20_vals[0]
            golden_tie_set = {
                int(tid) for tid, val in zip(golden_top20_ids, golden_top20_vals) if val == golden_max_val
            }

            npu_top20_ids = np.argsort(-logits_f32)[:20]
            npu_top1 = int(npu_top20_ids[0])

            top1_match = (npu_top1 == golden_top1)
            top1_tie_match = npu_top1 in golden_tie_set

            top20_overlap = len(set(npu_top20_ids.tolist()) & set(golden_top20_ids.tolist()))
            logit_cos = cosine_sim(logits_f32[golden_top20_ids], golden_top20_vals)

            rec.update(
                top1=npu_top1,
                npu_top1=npu_top1,
                golden_top1=golden_top1,
                top1_match=top1_match,
                tie=top1_tie_match,
                top1_tie_match=top1_tie_match,
                top20=top20_overlap,
                top20_overlap=top20_overlap,
                logit_cos=logit_cos,
                logit_cosine=logit_cos,
            )
            log(
                f"step {step} logits: top1 {npu_top1} golden {golden_top1} "
                f"tie_ok={top1_tie_match} top20={top20_overlap}/20 cos={logit_cos:.5f}"
            )

        step_xclbin_switches = int(npu_backend.xclbin_switches)
        step_xclbin_calls = dict(npu_backend.xclbin_calls)

        rec["timing_s"] = timing_s
        rec["calls"] = calls
        rec["xclbin_switches"] = step_xclbin_switches
        rec["xclbin_calls"] = step_xclbin_calls
        rec["breakdown"] = {
            "gemv_ms": npu_gemv_ms,
            "rda_ms": npu_rda_ms,
            "moe_b1_ms": npu_moe_b1_ms,
            "moe_sh_ms": npu_moe_sh_ms,
            "moe_dn_ms": npu_moe_dn_ms,
            "moe_ms": npu_moe_ms,
            "mla_attend_ms": mla_attend_ms,
            "host_prep_ms": host_prep_ms,
            "host_ops_ms": host_ops_ms,
        }

        timing_summary = ", ".join(
            f"{k}={v*1e3:.1f}ms/{calls.get(k, 0)}" for k, v in sorted(timing_s.items())
        )
        log(
            f"step {step}: {t_step_total:.2f}s (wall {t_step_total:.4f}s/token, "
            f"xclbin switches={step_xclbin_switches})  {timing_summary}"
        )

        elapsed_total = time.perf_counter() - total_run_start
        if elapsed_total > total_timeout:
            log(
                f"[EARLY STOP] Total timeout of {total_timeout}s reached "
                f"during step {step} verification."
            )
            stop_reason = {
                "code": "total_timeout",
                "step": step,
                "message": (
                    f"total timeout of {total_timeout}s reached during step {step} "
                    f"verification"
                ),
            }
            break

        if rec.get("top1_match"):
            top1_matches += 1
        if rec.get("top1_tie_match"):
            top1_tie_matches += 1
        steps_completed += 1
        step_records.append(rec)

    total_duration = time.perf_counter() - total_run_start
    if stop_reason is None:
        stop_reason = {
            "code": "completed",
            "step": end_step,
            "message": f"completed all requested steps {start_step}..{end_step}",
        }
    assessment = _assess_verification(
        start_step=start_step,
        end_step=end_step,
        step_records=step_records,
        stop_reason=stop_reason,
        full_25_layers=full_25_layers,
        baseline_passed=baseline_passed,
        min_cos=min_cos,
        min_top1_tie_rate=min_top1_tie_rate,
    )
    # Aggregate statistics across all completed steps
    print("-" * 85, flush=True)
    lat_total = [r["latency_total_ms"] for r in step_records if "latency_total_ms" in r]
    avg_total_ms = (
        None if (lat_total and any(not math.isfinite(x) for x in lat_total))
        else (float(np.mean(lat_total)) if lat_total else 0.0)
    )
    lat_npu = [r["latency_pure_npu_ms"] for r in step_records if "latency_pure_npu_ms" in r]
    avg_npu_ms = (
        None if (lat_npu and any(not math.isfinite(x) for x in lat_npu))
        else (float(np.mean(lat_npu)) if lat_npu else 0.0)
    )
    summary_metrics = _summarize_metrics(step_records)
    min_streams_cos = summary_metrics["min_streams_cos"]
    min_mixer_cos = summary_metrics["min_mixer_cos"]
    avg_router_overlap = summary_metrics["avg_router_overlap"]

    # Aggregate per-category timing totals and call counts across completed steps
    category_timing_s: dict[str, float] = {}
    category_calls: dict[str, int] = {}
    for r in step_records:
        for k, v in r.get("timing_s", {}).items():
            category_timing_s[k] = round(category_timing_s.get(k, 0.0) + float(v), 6)
        for k, c in r.get("calls", {}).items():
            category_calls[k] = category_calls.get(k, 0) + int(c)

    total_xclbin_switches = sum(r.get("xclbin_switches", 0) for r in step_records)
    avg_xclbin_switches = (
        float(np.mean([r.get("xclbin_switches", 0) for r in step_records])) if step_records else 0.0
    )
    total_xclbin_calls: dict[str, int] = defaultdict(int)
    for r in step_records:
        for xname, cnt in r.get("xclbin_calls", {}).items():
            total_xclbin_calls[xname] += int(cnt)

    print(f"[{now_str()}] Verification Summary:", flush=True)
    print(f"  GEMV Engine:         {gemv}", flush=True)
    print(f"  Completed Steps:     {steps_completed} / {end_step - start_step + 1}", flush=True)
    print(f"  Avg Xclbin Switches: {avg_xclbin_switches:.1f} / token (Total: {total_xclbin_switches})", flush=True)
    print(f"  Layers Evaluated:    {num_layers}", flush=True)
    str_s = f"{min_streams_cos:.6f}" if min_streams_cos is not None else "non-finite"
    str_m = f"{min_mixer_cos:.6f}" if min_mixer_cos is not None else "non-finite"
    print(f"  Min Streams Cos:     {str_s}", flush=True)
    print(f"  Min Mixer Cos:       {str_m}", flush=True)
    if avg_router_overlap is not None:
        print(f"  Avg Router Overlap:  {avg_router_overlap:.2f} / 6", flush=True)

    avg_cos = 0.0
    pct_top1 = 0.0
    pct_tie = 0.0
    if full_25_layers:
        logit_cos_list = [r["logit_cosine"] for r in step_records if "logit_cosine" in r]
        if logit_cos_list and any(not math.isfinite(x) for x in logit_cos_list):
            avg_cos = None
        else:
            avg_cos = float(np.mean(logit_cos_list)) if logit_cos_list else 0.0
        pct_top1 = (top1_matches / steps_completed * 100.0) if steps_completed else 0.0
        pct_tie = (top1_tie_matches / steps_completed * 100.0) if steps_completed else 0.0
        print(
            f"  Top-1 Match:         {top1_matches}/{steps_completed} ({pct_top1:.1f}%) "
            f"[Tie-aware: {top1_tie_matches}/{steps_completed} ({pct_tie:.1f}%)]",
            flush=True,
        )
        min_golden_cos = assessment["min_logit_cosine"]
        min_golden_cos_text = f"{min_golden_cos:.5f}" if min_golden_cos is not None else "n/a"
        avg_cos_text = f"{avg_cos:.5f}" if avg_cos is not None else "non-finite"
        print(f"  Average Logit Cos:   {avg_cos_text}", flush=True)
        print(f"  Minimum Logit Cos:   {min_golden_cos_text} (Target: >= {min_cos})", flush=True)
        print(
            f"  Tie-aware Top-1:     {assessment['top1_tie_match_rate']:.3f} "
            f"(Target: >= {min_top1_tie_rate})",
            flush=True,
        )

    str_lat_total = f"{avg_total_ms:.1f}" if avg_total_ms is not None else "non-finite"
    str_lat_npu = f"{avg_npu_ms:.1f}" if avg_npu_ms is not None else "non-finite"
    str_lat_host = (
        f"{max(0.0, avg_total_ms - avg_npu_ms):.1f}"
        if (avg_total_ms is not None and avg_npu_ms is not None)
        else "non-finite"
    )
    print(
        f"  Average Latency:     {str_lat_total} ms/token "
        f"(NPU: {str_lat_npu} ms, Host: {str_lat_host} ms)",
        flush=True,
    )
    print(f"  Total Run Duration:  {total_duration:.2f} s", flush=True)
    if compare_host_d:
        print(f"  Host D Baseline:     {'PASS' if baseline_passed else 'FAIL'}", flush=True)
    else:
        print("  Host D Baseline:     SKIPPED", flush=True)
    print(f"  Run Complete:        {'PASS' if assessment['run_complete'] else 'FAIL'}", flush=True)
    if not assessment["run_complete"]:
        print(f"  Stop Reason:         {stop_reason['message']}", flush=True)
    golden_status = "PASS" if assessment["golden_accuracy_passed"] else "FAIL"
    if not full_25_layers:
        golden_status += " (all 25 layers required for logit metrics)"
    print(f"  Golden Accuracy:     {golden_status}", flush=True)
    print("=" * 85, flush=True)

    summary_payload: dict[str, Any] = {
        "gemv": gemv,
        "timestamp": now_str(),
        "weights_d_dir": str(weights_d_dir),
        "golden_dir": str(golden_dir),
        "num_layers": num_layers,
        "steps_completed": steps_completed,
        "min_streams_cos": min_streams_cos,
        "min_mixer_cos": min_mixer_cos,
        "avg_router_overlap": avg_router_overlap,
        "avg_latency_ms": avg_total_ms,
        "avg_latency_s_per_token": _latency_seconds_per_token(avg_total_ms),
        "avg_npu_latency_ms": avg_npu_ms,
        "total_xclbin_switches": total_xclbin_switches,
        "avg_xclbin_switches": avg_xclbin_switches,
        "total_xclbin_calls": dict(total_xclbin_calls),
        "preload": preload_summary,
        "baseline_passed": baseline_passed,
        "baseline_skipped": baseline_skipped,
        "host_numpy_parity": host_numpy_parity_step,
        "category_timing_s": category_timing_s,
        "category_calls": category_calls,
        "step_records": step_records,
        "stop_reason": stop_reason,
        **assessment,
    }
    if summary_metrics.get("min_streams_cos_non_finite"):
        summary_payload["min_streams_cos_non_finite"] = True
    if summary_metrics.get("min_mixer_cos_non_finite"):
        summary_payload["min_mixer_cos_non_finite"] = True
    if full_25_layers:
        summary_payload.update(
            top1_match_pct=pct_top1,
            top1_tie_match_pct=pct_tie,
            avg_logit_cosine=avg_cos,
        )

    summary_payload = normalize_payload_finite(summary_payload)

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, allow_nan=False)
        log(f"wrote {out_json}")

    return summary_payload


class _GoldenArgumentParser(argparse.ArgumentParser):
    def parse_known_args(self, args=None, namespace=None):
        parsed, extras = super().parse_known_args(args, namespace)
        if parsed.gemv is None:
            parsed.gemv = os.environ.get("RAON_NPU_GEMV", "unified")
            if parsed.gemv not in ("unified", "legacy"):
                self.error(
                    "environment variable RAON_NPU_GEMV must be 'unified' or "
                    f"'legacy', got {parsed.gemv!r}"
                )
        return parsed, extras


def _build_parser() -> argparse.ArgumentParser:
    parser = _GoldenArgumentParser(description="Raon NPU Golden Verification Runner")
    parser.add_argument(
        "--gemv",
        type=str,
        default=None,
        choices=["unified", "legacy"],
        help="GEMV execution engine: 'unified' (default) or 'legacy' (default: $RAON_NPU_GEMV or unified)",
    )
    parser.add_argument(
        "--weights-d",
        type=Path,
        default=Path(os.environ["RAON_WEIGHTS_D"]) if os.environ.get("RAON_WEIGHTS_D") else None,
        help="Path to exported D weights directory (default: $RAON_WEIGHTS_D)",
    )
    parser.add_argument(
        "--golden-dir",
        type=Path,
        default=Path(os.environ["RAON_GOLDEN_DIR"]) if os.environ.get("RAON_GOLDEN_DIR") else None,
        help="Path to golden ko16 snapshots directory (default: $RAON_GOLDEN_DIR)",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to host model safetensors directory (optional)",
    )
    parser.add_argument(
        "--start-step",
        type=parse_start_step,
        default=SUPPORTED_START_STEP,
        help="Starting decode step; only 2 is supported (default: 2)",
    )
    parser.add_argument(
        "--end-step",
        type=ranged_int(min_val=1, max_val=MAX_GOLDEN_STEPS, name="end_step"),
        default=16,
        help="Ending decode step (default: 16)",
    )
    parser.add_argument(
        "--layers",
        type=ranged_int(min_val=1, max_val=MAX_LAYERS, name="layers"),
        default=None,
        help="Number of hidden layers to execute (default: all 25)",
    )
    parser.add_argument(
        "--timeout-per-token",
        type=ranged_float(
            min_val=0.0,
            max_val=MAX_TIMEOUT_PER_TOKEN,
            min_inclusive=False,
            max_inclusive=True,
            name="timeout_per_token",
        ),
        default=120.0,
        help=(
            "Wall-clock limit for one NPU model step, including the baseline comparison and "
            "first-step kernel loading. On expiry the process exits 1 and writes all thread "
            "stacks to stderr; a returned over-limit step is recorded as FAIL (default: 120)."
        ),
    )
    parser.add_argument(
        "--total-timeout",
        type=ranged_float(
            min_val=0.0,
            max_val=MAX_TOTAL_TIMEOUT,
            min_inclusive=False,
            max_inclusive=True,
            name="total_timeout",
        ),
        default=1800.0,
        help=(
            "Wall-clock limit for the entire decode loop, including every step's golden loading, "
            "layer comparisons, and logit evaluation; model/weight loading and preload are "
            "excluded (default: 1800)."
        ),
    )
    parser.add_argument(
        "--no-compare-host",
        action="store_true",
        help=(
            "Skip step-2 comparison against host NumpyBackend with D weights "
            "(JSON baseline_passed=null, baseline_skipped=true; verification_passed "
            "will be false and exit code non-zero)"
        ),
    )
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip preloading all weight bundles before decoding",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock NPU kernels with CPU NumPy reference for testing",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for JSON summary report",
    )
    parser.add_argument(
        "--min-cos",
        type=ranged_float(min_val=0.0, max_val=1.0, name="min_cos"),
        default=DEFAULT_MIN_COS,
        help=f"Minimum per-step golden logit cosine (default: {DEFAULT_MIN_COS})",
    )
    parser.add_argument(
        "--min-top1-tie-rate",
        type=ranged_float(min_val=0.0, max_val=1.0, name="min_top1_tie_rate"),
        default=DEFAULT_MIN_TOP1_TIE_RATE,
        help=(
            "Minimum tie-aware golden top-1 match rate "
            f"(default: {DEFAULT_MIN_TOP1_TIE_RATE})"
        ),
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.start_step > args.end_step:
        parser.error("--start-step must be less than or equal to --end-step")
    if (args.end_step - args.start_step + 1) > MAX_GOLDEN_STEPS:
        parser.error(
            f"total steps (end_step - start_step + 1 = {args.end_step - args.start_step + 1}) "
            f"exceeds maximum allowed ({MAX_GOLDEN_STEPS})"
        )
    if not 0.0 <= args.min_cos <= 1.0:
        parser.error("--min-cos must be between 0 and 1")
    if not 0.0 <= args.min_top1_tie_rate <= 1.0:
        parser.error("--min-top1-tie-rate must be between 0 and 1")
    return args


def main() -> None:
    args = _parse_args()
    parser = _build_parser()
    for flag, env_var, value in (
        ("--weights-d", "RAON_WEIGHTS_D", args.weights_d),
        ("--golden-dir", "RAON_GOLDEN_DIR", args.golden_dir),
    ):
        if value is None:
            parser.error(f"{flag} is required: pass {flag} or set ${env_var}")
        if not value.is_dir():
            raise FileNotFoundError(
                f"{flag} {value} not found; pass {flag} or set {env_var} to a local copy"
            )

    summary = run_golden_verification(
        weights_d_dir=args.weights_d,
        golden_dir=args.golden_dir,
        model_dir=args.model_dir,
        gemv=args.gemv,
        start_step=args.start_step,
        end_step=args.end_step,
        layer_limit=args.layers,
        timeout_per_token=args.timeout_per_token,
        total_timeout=args.total_timeout,
        compare_host_d=not args.no_compare_host,
        preload_weights=not args.no_preload,
        mock=args.mock,
        out_json=args.out,
        min_cos=args.min_cos,
        min_top1_tie_rate=args.min_top1_tie_rate,
    )
    if not summary.get("verification_passed", False):
        sys.exit(1)


if __name__ == "__main__":
    main()
