# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Alternate grouped B1/B2 calls 10,000 times with periodic decode checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from kernels.moe_grouped.layout import B1, B2, SLOTS, TMAX
from kernels.moe_grouped.planner import build_plan, gather_inputs, make_b2_inputs
from serve.engine import Engine
from .data import batches, load_records
from .runtime import ReplayRuntime


def _strict_mismatch_count(actual: np.ndarray, expected: np.ndarray) -> int:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return max(1, int(max(actual.size, expected.size)))
    return int(np.count_nonzero(actual != expected))


def _checked_random_call(
    runtime: ReplayRuntime,
    shape,
    layer: int,
    experts: tuple[int, ...],
    inputs: np.ndarray,
    valid_rows: tuple[int, ...],
) -> int:
    actual, _timing = runtime._run_grouped_call(shape, layer, experts, inputs)
    expected = runtime._run_decode_reference(
        shape, layer, experts, inputs, valid_rows
    )
    return _strict_mismatch_count(actual, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description="10k grouped MoE alternating-call stress")
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.calls < 1:
        parser.error("--calls must be positive")
    records = load_records(args.records)
    sample = batches(records, 64)[0]
    layer = sample[0].layer
    top_ids = np.stack([record.top_ids for record in sample])
    x = np.stack([record.x for record in sample])
    shared_gu = np.stack([record.shared_gate_up for record in sample])
    b1_plan = build_plan(top_ids)
    b1_real = list(zip(b1_plan.calls, gather_inputs(b1_plan, x)))

    engine = Engine(args.weights_dir, args.model_dir, preload=True)
    if engine.backend is None:
        raise RuntimeError("Engine did not create an NPU backend")
    runtime = ReplayRuntime(engine.backend)
    # Obtain valid B2 activations from one full grouped pass, then use its
    # planner inputs for the real-assignment half of stress.
    grouped = runtime.grouped(sample)
    routed_in, shared_in = make_b2_inputs(
        grouped.b1, shared_gu,
        gate_beta=sample[0].gate_beta, value_beta=sample[0].value_beta,
    )
    b2_plan = build_plan(top_ids, include_shared=True)
    b2_real = list(zip(b2_plan.calls, gather_inputs(b2_plan, routed_in, shared_in)))
    rng = np.random.default_rng(args.seed)
    timeouts = 0
    mismatches = 0
    other_errors = 0
    random_checks = 0
    first_error = None
    try:
        for index in range(args.calls):
            shape = B1 if index % 2 == 0 else B2
            real_pool = b1_real if shape is B1 else b2_real
            check_random = (index + 1) % 100 in (0, 99)
            use_real = bool(index % 4 < 2) and not check_random
            if use_real:
                call, values = real_pool[int(rng.integers(len(real_pool)))]
                experts = call.expert_slots
                inputs = values
                valid_rows = (0,) * SLOTS  # periodic checks always force random inputs
            else:
                high = shape.max_expert + 1
                experts = tuple(int(value) for value in rng.integers(0, high, size=SLOTS))
                source = x if shape is B1 else routed_in.reshape(-1, shape.k)
                inputs = np.zeros((SLOTS, TMAX, shape.k), dtype=np.uint16)
                counts = []
                for slot in range(SLOTS):
                    count = int(rng.integers(0, TMAX + 1))
                    counts.append(count)
                    if count:
                        rows = rng.integers(0, len(source), size=count)
                        inputs[slot, :count] = source[rows]
                valid_rows = tuple(counts)
            try:
                if check_random:
                    mismatches += _checked_random_call(
                        runtime, shape, layer, experts, inputs, valid_rows
                    )
                    random_checks += 1
                else:
                    runtime._run_grouped_call(shape, layer, experts, inputs)
            except Exception as error:  # stress must count and continue
                text = str(error)
                if "ERT_CMD_STATE_TIMEOUT" in text or "TIMEOUT" in text.upper():
                    timeouts += 1
                else:
                    other_errors += 1
                if first_error is None:
                    first_error = {"call": index, "error": text}
    finally:
        engine.unload()
    result = {
        "calls": args.calls, "timeouts": timeouts, "mismatches": mismatches,
        "other_errors": other_errors, "checks": random_checks,
        "random_checks": random_checks,
        "first_error": first_error,
        "pass": timeouts == 0 and mismatches == 0 and other_errors == 0,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
