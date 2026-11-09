# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Notebook benchmark for stage-2a ``NpuBackend.linear_batch`` integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from host.weights import f32_to_bf16_rtne
from npu.backend import NpuBackend, PREFILL_GEMM_MAX_T


def _cases(rda_layer: int, mla_layer: int, moe_layer: int = 1):
    return (
        ("rda_in", f"model.layers.{rda_layer}.self_attn.q_proj.weight"),
        ("f_b", f"model.layers.{rda_layer}.self_attn.f_b_proj.weight"),
        ("o_proj", f"model.layers.{rda_layer}.self_attn.o_proj.weight"),
        ("mla_in", f"model.layers.{mla_layer}.self_attn.q_a_proj.weight"),
        ("q_b", f"model.layers.{mla_layer}.self_attn.q_b_proj.weight"),
        ("k_b_v_b", f"model.layers.{mla_layer}.self_attn.kv_b_proj.weight"),
        ("g", f"model.layers.{mla_layer}.self_attn.g_proj.weight"),
        ("shared_gu", f"model.layers.{moe_layer}.moe.shared_experts.gate_proj.weight"),
        ("dense_down", "model.layers.0.mlp.down_proj.weight"),
    )


def _p(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q))


def evaluate_passed(records: list[dict], shape_gemm_used: dict[str, bool]) -> bool:
    """Evaluate whether benchmark run satisfies byte equality and GEMM usage across all shapes."""
    return (
        all(record["byte_equal"] for record in records)
        and all(shape_gemm_used.values())
    )


def run(args):
    backend = NpuBackend(weights_dir=args.weights_dir, mock=False)
    rng = np.random.default_rng(args.seed)
    records = []
    step = 0
    inputs = {}
    cases = _cases(args.rda_layer, args.mla_layer, args.moe_layer)
    for shape_name, name in cases:
        comp = backend.comp_map[name]
        for t in range(1, PREFILL_GEMM_MAX_T + 1):
            x = f32_to_bf16_rtne(
                rng.standard_normal((t, comp.orig_k), dtype=np.float32)
            )
            inputs[(shape_name, t)] = (name, x)
            step += 1
            backend.set_step(step)
            gemm_before = backend.call_counts.get(f"gemm:{shape_name}", 0)
            got = backend.linear_batch(name, x)
            used_gemm = backend.call_counts.get(f"gemm:{shape_name}", 0) > gemm_before
            step += 1
            backend.set_step(step)
            expected = np.stack([backend.linear(name, row) for row in x])
            equal = bool(np.array_equal(got, expected))
            if not equal:
                raise AssertionError(f"byte mismatch: {shape_name} T={t}")

            gemm_ms = []
            gemv_ms = []
            for _ in range(args.iterations):
                step += 1
                backend.set_step(step)
                start = time.perf_counter()
                np.stack([backend.linear(name, row) for row in x])
                gemv_ms.append((time.perf_counter() - start) * 1e3)
                step += 1
                backend.set_step(step)
                start = time.perf_counter()
                backend.linear_batch(name, x)
                gemm_ms.append((time.perf_counter() - start) * 1e3)
            records.append(
                {
                    "shape": shape_name,
                    "name": name,
                    "t": t,
                    "byte_equal": equal,
                    "used_gemm": used_gemm,
                    "gemm_ms_p50": _p(gemm_ms, 50),
                    "gemm_ms_p90": _p(gemm_ms, 90),
                    "gemm_ms_per_token_p50": _p(gemm_ms, 50) / t,
                    "gemv_ms_p50": _p(gemv_ms, 50),
                    "gemv_ms_per_token_p50": _p(gemv_ms, 50) / t,
                }
            )

    switch_name, switch_x = inputs[("rda_in", 16)]
    before = backend.xclbin_switches
    started = time.perf_counter()
    for _ in range(args.switch_iterations):
        step += 1
        backend.set_step(step)
        backend.linear(switch_name, switch_x[0])
        step += 1
        backend.set_step(step)
        backend.linear_batch(switch_name, switch_x)
    switch_elapsed_ms = (time.perf_counter() - started) * 1e3
    switches = backend.xclbin_switches - before
    shape_gemm_used = {
        sname: any(r["used_gemm"] for r in records if r["shape"] == sname)
        for sname, _ in cases
    }
    result = {
        "schema": "raon-prefill-gemm-s2a-v1",
        "weights_dir": str(Path(args.weights_dir).resolve()),
        "rda_layer": args.rda_layer,
        "mla_layer": args.mla_layer,
        "moe_layer": args.moe_layer,
        "iterations": args.iterations,
        "records": records,
        "shape_gemm_used": shape_gemm_used,
        "context_switch": {
            "iterations": args.switch_iterations,
            "count": switches,
            "window_ms": switch_elapsed_ms,
            "window_ms_per_switch": switch_elapsed_ms / switches if switches else None,
            "note": "wall window includes both kernels; it is not a firmware-only timer",
        },
        "timings": backend.timings,
        "xclbin_calls": dict(backend.xclbin_calls),
        "xclbin_switches": backend.xclbin_switches,
        "passed": evaluate_passed(records, shape_gemm_used),
    }
    return result


def _parse(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("weights_dir")
    parser.add_argument("--rda-layer", type=int, default=1)
    parser.add_argument("--mla-layer", type=int, default=3)
    parser.add_argument("--moe-layer", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--switch-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.iterations < 1 or args.switch_iterations < 1:
        parser.error("iteration counts must be positive")
    return args


def main(argv=None):
    args = _parse(argv)
    result = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "output": str(output)}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
