# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Alternating same-window benchmark for decode GEMV versus grouped GEMM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from host.strict_json import loads_strict
from serve.engine import Engine
from .data import batches, load_records
from .runtime import ReplayResult, ReplayRuntime


def first_mismatch(a: np.ndarray, b: np.ndarray) -> dict[str, object] | None:
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"shape_a": a.shape, "shape_b": b.shape,
                "dtype_a": str(a.dtype), "dtype_b": str(b.dtype)}
    indices = np.argwhere(a != b)
    if not len(indices):
        return None
    index = tuple(int(value) for value in indices[0])
    return {"index": index, "a": int(a[index]), "b": int(b[index])}


def mismatch_count(a: np.ndarray, b: np.ndarray) -> int:
    if a.shape != b.shape or a.dtype != b.dtype:
        return max(1, int(max(a.size, b.size)))
    return int(np.count_nonzero(a != b))


def stress_result_passes(stress: object) -> bool:
    if not isinstance(stress, dict):
        return False
    required = {
        "pass", "other_errors", "timeouts", "mismatches", "calls", "random_checks",
    }
    if not required.issubset(stress):
        return False
    if stress["pass"] is not True:
        return False
    integers = ("other_errors", "timeouts", "mismatches", "calls", "random_checks")
    if any(not isinstance(stress[key], int) or isinstance(stress[key], bool)
           for key in integers):
        return False
    return (
        stress["other_errors"] == 0
        and stress["timeouts"] == 0
        and stress["mismatches"] == 0
        and stress["calls"] >= 10000
        and stress["random_checks"] >= 100
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark grouped MoE against current decode GEMV")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--batches", default="64", help="comma list from 16,32,64")
    parser.add_argument("--layers", help="optional comma-separated layers")
    parser.add_argument("--stress-result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.batches.split(",")]
    if not batch_sizes or any(value not in (16, 32, 64) for value in batch_sizes):
        parser.error("--batches must contain only 16,32,64")
    layers = None if args.layers is None else {int(value) for value in args.layers.split(",")}
    records = load_records(args.records, layers)
    engine = Engine(args.weights_dir, args.model_dir, preload=True)
    if engine.backend is None:
        raise RuntimeError("Engine did not create an NPU backend")
    runtime = ReplayRuntime(engine.backend)
    report: dict[str, object] = {"record_count": len(records), "batches": {}}
    all_valid = True
    try:
        for batch_t in batch_sizes:
            totals = {
                "tokens": 0,
                "baseline_npu_s": 0.0, "baseline_input_write_s": 0.0,
                "baseline_output_read_s": 0.0, "baseline_gather_s": 0.0,
                "baseline_situ_s": 0.0, "baseline_scatter_s": 0.0,
                "grouped_npu_s": 0.0, "grouped_input_write_s": 0.0,
                "grouped_output_read_s": 0.0, "grouped_gather_s": 0.0,
                "grouped_situ_s": 0.0, "grouped_scatter_s": 0.0,
                "jobs": 0, "calls": 0,
                "useful": 0.0, "windows": 0,
                "baseline_mismatch": 0, "grouped_mismatch": 0,
                "shared_baseline_mismatch": 0, "shared_grouped_mismatch": 0,
                "b1_mismatch": 0, "b2_mismatch": 0,
            }
            transition_samples: dict[str, list[float]] = {
                "baseline_to_b1": [], "b1_to_b2": [], "b2_to_baseline": [],
            }
            first: dict[str, object] | None = None
            for index, batch in enumerate(batches(records, batch_t)):
                expected = np.stack([record.output for record in batch])
                expected_shared = np.stack([record.shared_down for record in batch])
                paths = ["baseline", "grouped"] if index % 2 == 0 else ["grouped", "baseline"]
                measured: dict[str, ReplayResult] = {}
                for name in paths:
                    measured[name] = getattr(runtime, name)(batch, warmup=True)
                baseline = measured["baseline"]
                grouped = measured["grouped"]
                samples = runtime.transition_cost_samples(batch, grouped.b1)
                for edge, sample in samples.items():
                    transition_samples[edge].append(sample)
                totals["windows"] += 1
                totals["tokens"] += len(batch)
                totals["baseline_npu_s"] += baseline.timing.npu_s
                totals["baseline_input_write_s"] += baseline.timing.input_write_s
                totals["baseline_output_read_s"] += baseline.timing.output_read_s
                totals["baseline_gather_s"] += baseline.timing.gather_s
                totals["baseline_situ_s"] += baseline.timing.situ_s
                totals["baseline_scatter_s"] += baseline.timing.scatter_s
                totals["grouped_npu_s"] += grouped.timing.npu_s
                totals["grouped_input_write_s"] += grouped.timing.input_write_s
                totals["grouped_output_read_s"] += grouped.timing.output_read_s
                totals["grouped_gather_s"] += grouped.timing.gather_s
                totals["grouped_situ_s"] += grouped.timing.situ_s
                totals["grouped_scatter_s"] += grouped.timing.scatter_s
                totals["jobs"] += grouped.jobs
                totals["calls"] += grouped.calls
                totals["useful"] += grouped.utilization * grouped.calls * 64
                checks = {
                    "baseline_output": (baseline.output, expected),
                    "grouped_output": (grouped.output, expected),
                    "baseline_shared": (baseline.shared_down, expected_shared),
                    "grouped_shared": (grouped.shared_down, expected_shared),
                    "b1": (grouped.b1, baseline.b1),
                    "b2": (grouped.b2, baseline.b2),
                }
                keys = {
                    "baseline_output": "baseline_mismatch",
                    "grouped_output": "grouped_mismatch",
                    "baseline_shared": "shared_baseline_mismatch",
                    "grouped_shared": "shared_grouped_mismatch",
                    "b1": "b1_mismatch", "b2": "b2_mismatch",
                }
                for label, (actual, wanted) in checks.items():
                    count = mismatch_count(actual, wanted)
                    totals[keys[label]] += count
                    if count and first is None:
                        first = {"kind": label, **(first_mismatch(actual, wanted) or {})}
            tokens = int(totals["tokens"])
            baseline_host = sum(float(totals[f"baseline_{name}_s"]) for name in (
                "input_write", "output_read", "gather", "situ", "scatter",
            ))
            baseline_total = float(totals["baseline_npu_s"]) + baseline_host
            grouped_host = sum(float(totals[f"grouped_{name}_s"]) for name in (
                "input_write", "output_read", "gather", "situ", "scatter",
            ))
            grouped_total = float(totals["grouped_npu_s"]) + grouped_host
            ratio = grouped_total / baseline_total if baseline_total else float("inf")
            transition_medians = {
                edge: float(np.median(samples)) if samples else 0.0
                for edge, samples in transition_samples.items()
            }
            transition_estimate = sum(transition_medians.values()) * int(totals["windows"])
            ratio_with_transition = (
                (grouped_total + transition_estimate) / baseline_total
                if baseline_total else float("inf")
            )
            valid = all(int(totals[key]) == 0 for key in (
                "baseline_mismatch", "grouped_mismatch", "shared_baseline_mismatch",
                "shared_grouped_mismatch", "b1_mismatch", "b2_mismatch",
            ))
            all_valid = all_valid and valid
            report["batches"][str(batch_t)] = {
                "tokens": tokens,
                "ms_per_token": {
                    "baseline_npu": 1000 * float(totals["baseline_npu_s"]) / tokens,
                    "baseline_input_write": 1000 * float(totals["baseline_input_write_s"]) / tokens,
                    "baseline_output_read": 1000 * float(totals["baseline_output_read_s"]) / tokens,
                    "baseline_gather": 1000 * float(totals["baseline_gather_s"]) / tokens,
                    "baseline_situ": 1000 * float(totals["baseline_situ_s"]) / tokens,
                    "baseline_scatter": 1000 * float(totals["baseline_scatter_s"]) / tokens,
                    "baseline_host": 1000 * baseline_host / tokens,
                    "baseline_total": 1000 * baseline_total / tokens,
                    "grouped_npu": 1000 * float(totals["grouped_npu_s"]) / tokens,
                    "grouped_input_write": 1000 * float(totals["grouped_input_write_s"]) / tokens,
                    "grouped_output_read": 1000 * float(totals["grouped_output_read_s"]) / tokens,
                    "grouped_gather": 1000 * float(totals["grouped_gather_s"]) / tokens,
                    "grouped_situ": 1000 * float(totals["grouped_situ_s"]) / tokens,
                    "grouped_scatter": 1000 * float(totals["grouped_scatter_s"]) / tokens,
                    "grouped_host": 1000 * grouped_host / tokens,
                    "grouped_total": 1000 * grouped_total / tokens,
                },
                "ratio": ratio,
                "ratios": {
                    "excluding_transition": ratio,
                    "with_measured_transition_estimate": ratio_with_transition,
                },
                "jobs": int(totals["jobs"]), "calls": int(totals["calls"]),
                "slot_utilization": float(totals["useful"]) / (int(totals["calls"]) * 64),
                "mismatches": {key: int(totals[key]) for key in totals if "mismatch" in key},
                "first_mismatch": first,
                "transition_cost_ms": {
                    edge: 1000 * value for edge, value in transition_medians.items()
                },
                "transition_estimate_ms": 1000 * transition_estimate,
                "thresholds": {"ratio_le_0_85": ratio <= 0.85, "bytes_equal": valid},
            }
    finally:
        engine.unload()

    stress_ok = False
    stress_note = "stress result not supplied"
    if args.stress_result is not None:
        stress = loads_strict(args.stress_result.read_text(encoding="utf-8"))
        stress_ok = stress_result_passes(stress)
        stress_note = str(args.stress_result)
    target = report["batches"].get("64")
    ratio_ok = bool(target and target["ratio"] <= 0.85)
    bytes_ok = bool(target and target["thresholds"]["bytes_equal"])
    report["verdict"] = {
        "grouped_ratio_le_0_85": ratio_ok,
        "bytes_equal": bytes_ok and all_valid,
        "stability_10000": stress_ok,
        "stability_evidence": stress_note,
        "pass": ratio_ok and bytes_ok and all_valid and stress_ok,
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["verdict"], sort_keys=True))
    return 0 if report["verdict"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
