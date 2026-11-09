# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Notebook byte-parity and timing check for context-friendly prefill RDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from host.weights import bf16_to_f32, f32_to_bf16_rtne
from npu.backend import NpuBackend, PREFILL_GEMM_XCLBIN_ID


def _percentiles_ms(values):
    data = np.asarray(values, dtype=np.float64) * 1e3
    return {
        "calls": int(data.size),
        "mean": float(np.mean(data)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
    }


def _rda_inputs(rng, count, num_heads: int = 40, head_dim: int = 128):
    result = []
    scale = head_dim ** -0.5
    for _ in range(count):
        # q: per-head L2-normalized in FP32, cast to BF16, then scaled by head_dim**-0.5 in BF16
        q_h = rng.standard_normal((num_heads, head_dim), dtype=np.float32)
        q_norm = np.linalg.norm(q_h, axis=-1, keepdims=True)
        q_norm_u16 = f32_to_bf16_rtne(q_h / np.maximum(q_norm, 1e-6))
        q_u16 = f32_to_bf16_rtne(bf16_to_f32(q_norm_u16) * scale)

        # k: per-head L2-normalized in FP32, cast to BF16
        k_h = rng.standard_normal((num_heads, head_dim), dtype=np.float32)
        k_norm = np.linalg.norm(k_h, axis=-1, keepdims=True)
        k_u16 = f32_to_bf16_rtne(k_h / np.maximum(k_norm, 1e-6))

        # v: standard normal in BF16
        v_h = rng.standard_normal((num_heads, head_dim), dtype=np.float32)
        v_u16 = f32_to_bf16_rtne(v_h)

        # log_alpha: FP32 (40, 128), negative decay via -softplus(scaled)
        scaled = rng.standard_normal((num_heads, head_dim), dtype=np.float32)
        log_alpha = (
            -np.log1p(np.exp(-np.abs(scaled))) - np.maximum(scaled, 0.0)
        ).astype(np.float32)

        # beta: (40,) sigmoid(beta_raw) in BF16
        b_raw = rng.standard_normal((num_heads,), dtype=np.float32)
        beta = 1.0 / (1.0 + np.exp(-b_raw))
        beta_u16 = f32_to_bf16_rtne(beta)

        result.append((q_u16, k_u16, v_u16, log_alpha, beta_u16))
    return result


def run(args):
    backend = NpuBackend(weights_dir=args.weights_dir, mock=args.mock)
    rng = np.random.default_rng(args.seed)
    updates = _rda_inputs(rng, 64)
    initial_state = np.zeros((40, 128, 128), dtype=np.float32)

    projection = f"model.layers.{args.rda_layer}.self_attn.q_proj.weight"
    projection_input = f32_to_bf16_rtne(
        rng.standard_normal((16, backend.comp_map[projection].orig_k), dtype=np.float32)
    )
    backend.set_rda_state(args.rda_layer, initial_state)
    backend.prefill_rda_enabled = False
    backend.reset_timings()
    decode_outputs = []
    decode_seconds = []
    for values in updates:
        started = time.perf_counter()
        decode_outputs.append(backend.rda_update(args.rda_layer, *values))
        decode_seconds.append(time.perf_counter() - started)
    decode_state = backend.get_rda_state(args.rda_layer)
    decode_timings = dict(backend.timings)
    decode_call_counts = dict(backend.call_counts)
    decode_xclbin_calls = dict(backend.xclbin_calls)

    backend.set_rda_state(args.rda_layer, initial_state)
    # Compile and warm the s2a plate outside the measured mixed window. Each
    # measured group below still issues its own immediately preceding GEMM.
    backend.set_step(1)
    if backend.mock and backend._prefill_xclbin_path is None:
        backend._prefill_xclbin_path = Path(f"{PREFILL_GEMM_XCLBIN_ID}.xclbin")
    backend.linear_batch(projection, projection_input)
    backend.prefill_rda_enabled = True
    backend.reset_timings()
    prefill_outputs = []
    first_seconds = []
    continuous_seconds = []
    step = 1
    for start in range(0, 64, 16):
        step += 1
        backend.set_step(step)
        backend.linear_batch(projection, projection_input)
        for offset, values in enumerate(updates[start : start + 16]):
            started = time.perf_counter()
            prefill_outputs.append(backend.rda_update(args.rda_layer, *values))
            elapsed = time.perf_counter() - started
            (first_seconds if offset == 0 else continuous_seconds).append(elapsed)
    prefill_state = backend.get_rda_state(args.rda_layer)
    prefill_timings = dict(backend.timings)
    prefill_call_counts = dict(backend.call_counts)
    prefill_xclbin_calls = dict(backend.xclbin_calls)

    # Measure first decode RDA calls returning from prefill context
    # Immediately after the prefill loop, the active xclbin is PREFILL_GEMM_XCLBIN_ID.
    return_seconds = []
    return_input = updates[0]
    # First return immediately following the 64-call prefill run:
    backend.prefill_rda_enabled = False
    started = time.perf_counter()
    backend.rda_update(args.rda_layer, *return_input)
    return_seconds.append(time.perf_counter() - started)

    # 3 additional switch pairs so return_seconds has 4 samples, matching first_seconds
    for _ in range(3):
        step += 1
        backend.set_step(step)
        backend.prefill_rda_enabled = True
        backend.linear_batch(projection, projection_input)
        backend.rda_update(args.rda_layer, *return_input)

        backend.prefill_rda_enabled = False
        started = time.perf_counter()
        backend.rda_update(args.rda_layer, *return_input)
        return_seconds.append(time.perf_counter() - started)

    output_equal = [
        bool(np.array_equal(left, right))
        for left, right in zip(decode_outputs, prefill_outputs)
    ]
    state_equal = bool(np.array_equal(decode_state, prefill_state))
    expected_prefill_rda_calls = 64
    expected_prefill_gemm_calls = 4
    expected_prefill_xclbin_calls = expected_prefill_rda_calls + expected_prefill_gemm_calls
    passed = (
        all(output_equal)
        and state_equal
        and decode_call_counts.get("rda_npu_prefill", 0) == 0
        and prefill_call_counts.get("rda_npu_prefill", 0) == expected_prefill_rda_calls
        and prefill_xclbin_calls.get(PREFILL_GEMM_XCLBIN_ID, 0) == expected_prefill_xclbin_calls
    )
    return {
        "schema": "raon-prefill-rda-s2a-v1",
        "weights_dir": str(Path(args.weights_dir).resolve()) if args.weights_dir else None,
        "rda_layer": args.rda_layer,
        "seed": args.seed,
        "calls": 64,
        "byte_parity": {
            "all_outputs_equal": all(output_equal),
            "equal_output_calls": sum(output_equal),
            "final_state_equal": state_equal,
        },
        "ms_per_call": {
            "decode_rda_continuous": _percentiles_ms(decode_seconds),
            "prefill_rda_continuous": _percentiles_ms(continuous_seconds),
            "prefill_first_after_gemm": _percentiles_ms(first_seconds),
            "decode_first_after_prefill": _percentiles_ms(return_seconds),
            "decode_first_return": _percentiles_ms(return_seconds),
        },
        "decode_timings": decode_timings,
        "prefill_timings": prefill_timings,
        "decode_call_counts": decode_call_counts,
        "prefill_call_counts": prefill_call_counts,
        "decode_xclbin_calls": decode_xclbin_calls,
        "prefill_xclbin_calls": prefill_xclbin_calls,
        "xclbin_calls": dict(backend.xclbin_calls),
        "xclbin_switches": backend.xclbin_switches,
        "passed": passed,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir", nargs="?", default=None)
    parser.add_argument("--rda-layer", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--mock", action="store_true", default=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"passed": result["passed"], "output": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
