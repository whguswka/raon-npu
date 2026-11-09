# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""NPU-free tests for grouped MoE planning and deterministic host math."""

from __future__ import annotations

import numpy as np
import pytest

from host.backend import situ_glu
from host.weights import bf16_to_f32, f32_to_bf16_rtne
from .planner import build_plan, gather_inputs, make_b2_inputs, scatter_b1, scatter_b2


@pytest.mark.parametrize("tokens,expected_jobs", [(1, 1), (8, 1), (9, 2), (17, 3)])
def test_split_one_expert_and_zero_fill(tokens: int, expected_jobs: int) -> None:
    top_ids = np.full((tokens, 1), 7, dtype=np.int64)
    plan = build_plan(top_ids)
    assert len(plan.jobs) == expected_jobs
    assert [len(job.assignments) for job in plan.jobs] == [
        min(8, tokens - start) for start in range(0, tokens, 8)
    ]
    x = np.arange(tokens * 5, dtype=np.uint16).reshape(tokens, 5)
    calls = gather_inputs(plan, x)
    restored_outputs = []
    for gathered in calls:
        out = np.zeros((8, 8, 3), dtype=np.uint16)
        out[..., 0] = gathered[..., 0]
        out[..., 1] = gathered[..., 1]
        out[..., 2] = gathered[..., 2]
        restored_outputs.append(out)
    restored = scatter_b1(plan, restored_outputs)
    np.testing.assert_array_equal(restored[:, 0], x[:, :3])
    assert all(call.expert_slots[-1] == 7 for call in plan.calls)
    assert all(np.count_nonzero(arr[slot]) == 0
               for arr, call in zip(calls, plan.calls)
               for slot, job in enumerate(call.jobs) if job is None)
    assert sum(sum(call.rows) for call in plan.calls) == plan.useful_rows == tokens


def test_call_rows_property_full_partial_and_empty_slots() -> None:
    # 11 tokens routed to expert 0 (8 + 3 rows) and 5 tokens to expert 1 (5 rows), rest empty
    top_ids = np.array(
        [[0, 1]] * 5 + [[0, 2]] * 6,
        dtype=np.int64,
    )
    plan = build_plan(top_ids, include_shared=True)
    call = plan.calls[0]
    # Expert 0: [8, 3], Expert 1: [5], Expert 2: [6], Shared(128): [8, 3], then 2 empty slots
    assert call.rows == (8, 3, 5, 6, 8, 3, 0, 0)
    assert sum(call.rows) == plan.useful_rows == top_ids.size + top_ids.shape[0]


def _mock_calls(plan, inputs, weights):
    outputs = []
    for call, gathered in zip(plan.calls, inputs):
        out = np.zeros((8, 8, weights.shape[1]), dtype=np.uint16)
        for slot, job in enumerate(call.jobs):
            if job is None:
                continue
            values = bf16_to_f32(gathered[slot]) @ weights[job.expert].T
            out[slot] = f32_to_bf16_rtne(values)
        outputs.append(out)
    return outputs


def test_gather_scatter_matches_token_moe_numpy() -> None:
    rng = np.random.default_rng(20261102)
    tokens, routes, k, m = 13, 6, 12, 9
    top_ids = np.stack([rng.choice(23, routes, replace=False) for _ in range(tokens)])
    top_weights = rng.random((tokens, routes), dtype=np.float32)
    x = f32_to_bf16_rtne(rng.normal(size=(tokens, k)).astype(np.float32))
    w = rng.normal(scale=0.05, size=(129, m, k)).astype(np.float32)
    shared_x = f32_to_bf16_rtne(rng.normal(size=(tokens, k)).astype(np.float32))

    b1_plan = build_plan(top_ids)
    b1_calls = _mock_calls(b1_plan, gather_inputs(b1_plan, x), w)
    b1 = scatter_b1(b1_plan, b1_calls)
    for token in range(tokens):
        for route in range(routes):
            expected = f32_to_bf16_rtne(
                bf16_to_f32(x[token]) @ w[top_ids[token, route]].T
            )
            np.testing.assert_array_equal(b1[token, route], expected)

    b2_plan = build_plan(top_ids, include_shared=True)
    routed_x = np.repeat(x[:, None, :], routes, axis=1)
    b2_calls = _mock_calls(
        b2_plan, gather_inputs(b2_plan, routed_x, shared_x), w
    )
    actual, actual_shared = scatter_b2(b2_plan, b2_calls, top_ids, top_weights)
    expected = np.zeros_like(actual)
    for token in range(tokens):
        acc = np.zeros(m, dtype=np.float32)
        for route in sorted(range(routes), key=lambda r: int(top_ids[token, r])):
            down = f32_to_bf16_rtne(
                bf16_to_f32(x[token]) @ w[top_ids[token, route]].T
            )
            acc += bf16_to_f32(down) * float(top_weights[token, route])
        expected[token] = f32_to_bf16_rtne(acc)
    expected_shared = np.stack([
        f32_to_bf16_rtne(bf16_to_f32(row) @ w[128].T) for row in shared_x
    ])
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_shared, expected_shared)


def test_b2_input_uses_production_situ_rtne_and_zero_padding() -> None:
    rng = np.random.default_rng(44)
    gate_up = f32_to_bf16_rtne(rng.normal(size=(2, 3, 2176)).astype(np.float32))
    shared_gu = f32_to_bf16_rtne(rng.normal(size=(2, 2176)).astype(np.float32))
    routed, shared = make_b2_inputs(
        gate_up, shared_gu, gate_beta=3.5, value_beta=19.0
    )
    expected = f32_to_bf16_rtne(
        situ_glu(gate_up[..., :1088], gate_up[..., 1088:], 3.5, 19.0)
    )
    expected_shared = f32_to_bf16_rtne(
        situ_glu(shared_gu[..., :1088], shared_gu[..., 1088:], 3.5, 19.0)
    )
    np.testing.assert_array_equal(routed[..., :1088], expected)
    np.testing.assert_array_equal(shared[..., :1088], expected_shared)
    assert not np.any(routed[..., 1088:])
    assert not np.any(shared[..., 1088:])


def test_regression_job_order_accumulation_is_not_equivalent() -> None:
    """A deliberate job-order implementation must disagree with the contract."""
    ids = np.array([[90, 3, 70, 11, 51, 22]], dtype=np.int64)
    plan = build_plan(ids, include_shared=True)
    weights = np.ones(ids.shape, dtype=np.float32)
    # Ascending experts visit routes [1,3,5,4,2,0]: +large and -large
    # cancel before +small. Route order visits +large,+small,-large and loses
    # the small term to FP32 rounding.
    values = np.zeros((1, 6, 5), dtype=np.float32)
    values[0, 1] = 1.0e8
    values[0, 2] = 1.0
    values[0, 3] = -1.0e8
    routed_bits = f32_to_bf16_rtne(values)
    outputs = []
    for call in plan.calls:
        out = np.zeros((8, 8, 5), dtype=np.uint16)
        for slot, job in enumerate(call.jobs):
            if job is None:
                continue
            for row, item in enumerate(job.assignments):
                if not job.shared:
                    out[slot, row] = routed_bits[item.token, item.route]
        outputs.append(out)
    correct, _ = scatter_b2(plan, outputs, ids, weights)
    wrong_acc = np.zeros(5, dtype=np.float32)
    for route in range(6):
        wrong_acc += bf16_to_f32(routed_bits[0, route]) * float(weights[0, route])
    wrong = f32_to_bf16_rtne(wrong_acc)
    assert wrong.tobytes() != correct[0].tobytes()
