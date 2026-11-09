# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Byte-parity tests for call-at-a-time grouped MoE planner helpers."""

from __future__ import annotations

import numpy as np
import pytest

from host.weights import f32_to_bf16_rtne
from kernels.moe.pack import B2_K_PAD, B2_M_PER_EXPERT, MOE_INTERMEDIATE_SIZE
from .layout import SLOTS, TMAX
from .planner import (
    accumulate_call,
    build_plan,
    gather_call,
    gather_inputs,
    make_b2_inputs,
    scatter_b1,
    scatter_b2,
    shared_situ,
    situ_call,
)


TOKEN_COUNTS = (1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 128)


def _ids(tokens: int, dtype=np.int64) -> np.ndarray:
    # Deliberately concentrates more than 16 rows in each routed expert and
    # includes both boundary IDs.
    return np.tile(np.array([0, 1, 2, 3, 4, 127], dtype=dtype), (tokens, 1))


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_gather_call_matches_gather_inputs(tokens: int) -> None:
    rng = np.random.default_rng(1000 + tokens)
    ids = _ids(tokens)
    routed_2d = rng.integers(0, 65536, (tokens, 13), dtype=np.uint16)
    routed_3d = rng.integers(0, 65536, (tokens, 6, 17), dtype=np.uint16)
    shared = rng.integers(0, 65536, (tokens, 17), dtype=np.uint16)

    b1 = build_plan(ids)
    expected_b1 = gather_inputs(b1, routed_2d)
    for index in range(len(b1.calls)):
        np.testing.assert_array_equal(
            gather_call(b1, index, routed_2d), expected_b1[index]
        )

    b2 = build_plan(ids, include_shared=True)
    expected_b2 = gather_inputs(b2, routed_3d, shared)
    for index in range(len(b2.calls)):
        np.testing.assert_array_equal(
            gather_call(b2, index, routed_3d, shared), expected_b2[index]
        )


@pytest.mark.parametrize("tokens", TOKEN_COUNTS)
def test_call_situ_and_accumulate_match_whole_plan(tokens: int) -> None:
    rng = np.random.default_rng(2000 + tokens)
    ids = _ids(tokens)
    gate_beta, value_beta = 3.5, 19.0
    shared_gu = rng.integers(
        0, 65536, (tokens, 2 * MOE_INTERMEDIATE_SIZE), dtype=np.uint16
    )

    b1_plan = build_plan(ids)
    b1_outputs = [
        rng.integers(
            0,
            65536,
            (SLOTS, TMAX, 2 * MOE_INTERMEDIATE_SIZE),
            dtype=np.uint16,
        )
        for _ in b1_plan.calls
    ]
    # Exercise non-finite and signed-zero bit patterns in rows that SiTU reads.
    special = f32_to_bf16_rtne(
        np.array([np.nan, np.inf, -np.inf, 0.0, -0.0], dtype=np.float32)
    )
    b1_outputs[0][0, 0, : special.size] = special
    whole_b1 = scatter_b1(b1_plan, b1_outputs)
    expected_routed, expected_shared = make_b2_inputs(
        whole_b1,
        shared_gu,
        gate_beta=gate_beta,
        value_beta=value_beta,
    )
    actual_routed = np.zeros((tokens, 6, B2_K_PAD), dtype=np.uint16)
    for call, output in zip(b1_plan.calls, b1_outputs):
        situ_call(
            call,
            output,
            actual_routed,
            gate_beta=gate_beta,
            value_beta=value_beta,
        )
    np.testing.assert_array_equal(actual_routed, expected_routed)
    np.testing.assert_array_equal(
        shared_situ(shared_gu, gate_beta=gate_beta, value_beta=value_beta),
        expected_shared,
    )

    b2_plan = build_plan(ids, include_shared=True)
    b2_outputs = [
        rng.integers(
            0,
            65536,
            (SLOTS, TMAX, B2_M_PER_EXPERT),
            dtype=np.uint16,
        )
        for _ in b2_plan.calls
    ]
    b2_outputs[0][0, 0, : special.size] = special
    weights = rng.normal(size=(tokens, 6)).astype(np.float32)
    weights.flat[: min(weights.size, 6)] = np.array(
        [0.0, -0.0, 1.0e20, -1.0e20, 1.0, -1.0], dtype=np.float32
    )[: min(weights.size, 6)]
    expected_routed_down, expected_shared_down = scatter_b2(
        b2_plan, b2_outputs, ids, weights
    )
    routed_acc = np.zeros((tokens, B2_M_PER_EXPERT), dtype=np.float32)
    actual_shared_down = np.zeros((tokens, B2_M_PER_EXPERT), dtype=np.uint16)
    for call, output in zip(b2_plan.calls, b2_outputs):
        accumulate_call(call, output, routed_acc, actual_shared_down, weights)
    np.testing.assert_array_equal(
        f32_to_bf16_rtne(routed_acc), expected_routed_down
    )
    np.testing.assert_array_equal(actual_shared_down, expected_shared_down)


def test_plans_cover_one_two_three_calls_and_shared_tail() -> None:
    assert len(build_plan(_ids(1)).calls) == 1
    assert len(build_plan(_ids(9)).calls) == 2
    assert len(build_plan(_ids(17)).calls) == 3
    shared_plan = build_plan(_ids(9), include_shared=True)
    assert any(job is None for job in shared_plan.calls[-1].jobs)
    assert any(job is not None and job.shared for job in shared_plan.calls[-1].jobs)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
@pytest.mark.parametrize("tokens,routes", [(1, 1), (1, 6), (7, 6)])
def test_distinct_expert_check_accepts_integer_views(dtype, tokens, routes) -> None:
    base = np.tile(np.arange(routes, dtype=dtype), (tokens, 1))
    noncontiguous = base[:, ::-1]
    assert not noncontiguous.flags.c_contiguous or routes == 1
    plan = build_plan(noncontiguous)
    assert plan.token_count == tokens
    assert plan.route_count == routes


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
def test_distinct_expert_check_rejects_duplicate_with_same_message(dtype) -> None:
    ids = np.array([[7, 2, 7]], dtype=dtype)[:, ::-1]
    with pytest.raises(
        ValueError, match="^each token must route to distinct experts$"
    ):
        build_plan(ids)
