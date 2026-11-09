# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for plan_ab_bench fixing zero-input defect and MoE comparison omission."""

from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np

from bench.gemv_unified.plan_ab_bench import (
    CANARY_WORD,
    CANARY_ELEMS,
    build_random_inputs,
    canonical_shape_name,
    execute_ab_loop,
    resolve_weight_bundle,
    _expert_sets,
)
from kernels.gemv_unified.shapes import get_shape


class MockBuffer:
    """Mock AIE/IRON tensor buffer wrapping a NumPy array."""

    def __init__(self, arr: np.ndarray):
        self._arr = np.array(arr)

    def numpy(self) -> np.ndarray:
        return self._arr

    @property
    def shape(self):
        return self._arr.shape


def test_weight_bundle_is_mandatory_and_rejects_none(tmp_path: Path) -> None:
    """Defect 1 regression test: weight bundle must be provided and cannot default to zeros."""
    # 1. Reject None
    with pytest.raises(ValueError, match="Weight bundle is required"):
        resolve_weight_bundle(None, "f_b")

    # 2. Reject nonexistent file
    with pytest.raises(FileNotFoundError, match="Weight bundle does not exist"):
        resolve_weight_bundle(tmp_path / "nonexistent.bin", "f_b")

    # 3. Reject directory missing the shape bundle
    with pytest.raises(FileNotFoundError, match="No weight bundle found"):
        resolve_weight_bundle(tmp_path, "f_b")

    # 4. Resolve successfully when file exists
    valid_file = tmp_path / "f_b.bin"
    valid_file.write_bytes(b"\x00" * 16)
    assert resolve_weight_bundle(valid_file, "f_b") == valid_file
    assert resolve_weight_bundle(tmp_path, "f_b") == valid_file


def test_random_inputs_are_nonzero_and_have_entropy() -> None:
    """Defect 1 regression test: input activations must be nonzero random bf16, not all zeros."""
    for shape_name in ("f_b", "q_b", "moe_b1_6"):
        shape = get_shape(shape_name)
        xs = build_random_inputs(shape, seed=1101, count=2)
        assert len(xs) == 2

        # Verify both inputs are nonzero arrays
        valid_elems = shape.x_slots * shape.K
        for i, x in enumerate(xs):
            x_u16 = x.view(np.uint16)
            nonzero_count = np.count_nonzero(x_u16)
            assert nonzero_count > 0, f"{shape_name} xs[{i}] is all zero"
            # Ensure significant entropy (>90% non-zero in random normal over the valid K slots)
            assert nonzero_count >= int(valid_elems * 0.9), (
                f"{shape_name} xs[{i}] has only {nonzero_count} nonzero elements (expected >= {int(valid_elems * 0.9)})"
            )

        # Verify the two inputs are distinct
        assert not np.array_equal(xs[0], xs[1]), f"{shape_name} inputs xs[0] and xs[1] are identical"


def test_ab_loop_executes_both_plans_for_every_expert_pattern() -> None:
    """Defect 2 regression test: both per_round and per_call must run on every expert set."""
    shape_name = "moe_b1_6"
    shape = get_shape(shape_name)
    sets = _expert_sets(shape_name)
    assert len(sets) == 4

    call_traces: list[tuple[str, int, int]] = []

    def make_mock_fn(plan_name: str):
        def fn(y_buf, w_buf, x_buf, **scalars):
            # Find scalar set index
            si = sets.index(scalars) if scalars in sets else -1
            # Record plan and scalar set index
            call_traces.append((plan_name, si))
            # Write deterministic output into y
            y_arr = y_buf.numpy().view(np.uint16)
            y_arr[: shape.M] = np.uint16(0x1000 + si * 10)

        return fn

    plans = {
        "per_round": {"fn": make_mock_fn("per_round")},
        "per_call": {"fn": make_mock_fn("per_call")},
    }

    canary_arr = np.full((shape.M + CANARY_ELEMS,), CANARY_WORD, dtype=np.uint16)
    outputs = {
        "per_round": MockBuffer(canary_arr.copy()),
        "per_call": MockBuffer(canary_arr.copy()),
    }
    w_mock = MockBuffer(np.zeros(10, dtype=np.uint32))
    xs = [MockBuffer(np.ones(10, dtype=np.uint16)) for _ in range(2)]

    # Run for 16 iterations (covering all combos multiple times)
    done, times, fails, ref, combos = execute_ab_loop(
        shape=shape,
        plans=plans,
        outputs=outputs,
        w_bo=w_mock,
        xs=xs,
        sets=sets,
        iterations=16,
        warmup=0,
    )

    assert not fails
    assert done == 16
    assert len(times["per_round"]) == 16
    assert len(times["per_call"]) == 16

    # Verify that for EVERY expert pattern (si in 0..3), BOTH per_round and per_call were called
    counts = {p: {si: 0 for si in range(len(sets))} for p in ("per_round", "per_call")}
    for plan, si in call_traces:
        counts[plan][si] += 1

    for si in range(len(sets)):
        assert counts["per_round"][si] > 0, f"per_round was never called for expert set {si}"
        assert counts["per_call"][si] > 0, f"per_call was never called for expert set {si}"
        assert counts["per_round"][si] == counts["per_call"][si], (
            f"Unbalanced execution on set {si}: per_round={counts['per_round'][si]} vs per_call={counts['per_call'][si]}"
        )


def test_ab_loop_detects_plan_bit_mismatch_on_moe_pattern() -> None:
    """Defect 2 regression test: defect in one plan on a specific expert set is caught immediately."""
    shape_name = "moe_b1_6"
    shape = get_shape(shape_name)
    sets = _expert_sets(shape_name)

    # Set pattern 2 (slot repetition [7, 7, 7, 7, 7, 7]) to have mismatch in per_call
    target_si = 2

    def make_mock_fn(plan_name: str):
        def fn(y_buf, w_buf, x_buf, **scalars):
            si = sets.index(scalars)
            y_arr = y_buf.numpy().view(np.uint16)
            base = np.uint16(0x2000 + si)
            y_arr[: shape.M] = base
            if plan_name == "per_call" and si == target_si:
                # Corrupt 1 element
                y_arr[0] ^= 0x0001

        return fn

    plans = {
        "per_round": {"fn": make_mock_fn("per_round")},
        "per_call": {"fn": make_mock_fn("per_call")},
    }

    canary_arr = np.full((shape.M + CANARY_ELEMS,), CANARY_WORD, dtype=np.uint16)
    outputs = {
        "per_round": MockBuffer(canary_arr.copy()),
        "per_call": MockBuffer(canary_arr.copy()),
    }
    w_mock = MockBuffer(np.zeros(10, dtype=np.uint32))
    xs = [MockBuffer(np.ones(10, dtype=np.uint16)) for _ in range(2)]

    done, times, fails, ref, combos = execute_ab_loop(
        shape=shape,
        plans=plans,
        outputs=outputs,
        w_bo=w_mock,
        xs=xs,
        sets=sets,
        iterations=16,
        warmup=0,
    )

    # In the flawed scheduler, pattern 2 ran ONLY on per_call (order index % 4 == 2 was per_call),
    # so per_round was never compared against per_call on pattern 2, and the mismatch slipped through.
    # In the fixed implementation, both plans run on every iteration and are compared directly.
    assert len(fails) > 0
    assert any("bit mismatch" in f and f"set={target_si}" in f for f in fails)

    # Defect 2 demonstration: The old flawed scheduler misses this exact defect because
    # order % 4 and pattern % 4 align, so each pattern only executes on one plan.
    def flawed_run() -> list[str]:
        flawed_order = ("per_round", "per_call", "per_call", "per_round")
        flawed_ref: dict[int, np.ndarray] = {}
        flawed_fails: list[str] = []
        for index in range(16):
            p = flawed_order[index % len(flawed_order)]
            scalars = sets[index % len(sets)]
            plans[p]["fn"](outputs[p], w_mock, xs[0], **scalars)
            got = np.asarray(outputs[p].numpy()).view(np.uint16)[: shape.M].copy()
            pid = index % len(sets)
            if pid not in flawed_ref:
                flawed_ref[pid] = got
            elif not np.array_equal(got, flawed_ref[pid]):
                flawed_fails.append(f"mismatch index={index}")
                break
        return flawed_fails

    flawed_fails = flawed_run()
    assert len(flawed_fails) == 0, "Old flawed scheduler should have erroneously passed by missing cross-plan comparison"


def test_ab_loop_detects_canary_corruption() -> None:
    """Canary overwriting beyond shape.M must be flagged as failure."""
    shape_name = "f_b"
    shape = get_shape(shape_name)
    sets = [{}]

    def make_corrupt_canary_fn():
        def fn(y_buf, w_buf, x_buf, **scalars):
            y_arr = y_buf.numpy().view(np.uint16)
            y_arr[: shape.M] = 0x1111
            # Corrupt canary at tail
            y_arr[shape.M + 10] = 0xDEAD

        return fn

    plans = {
        "per_round": {"fn": make_corrupt_canary_fn()},
        "per_call": {"fn": lambda *a, **k: None},
    }
    canary_arr = np.full((shape.M + CANARY_ELEMS,), CANARY_WORD, dtype=np.uint16)
    outputs = {
        "per_round": MockBuffer(canary_arr.copy()),
        "per_call": MockBuffer(canary_arr.copy()),
    }

    done, times, fails, ref, combos = execute_ab_loop(
        shape=shape,
        plans=plans,
        outputs=outputs,
        w_bo=MockBuffer(np.zeros(10, dtype=np.uint32)),
        xs=[MockBuffer(np.ones(10, dtype=np.uint16))],
        sets=sets,
        iterations=5,
        warmup=0,
    )

    assert len(fails) > 0
    assert any("canary overwritten" in f for f in fails)


@pytest.mark.parametrize(
    ("output_word", "expected_failure"),
    [(0, True), (0x1234, False)],
)
def test_ab_loop_rejects_zero_and_accepts_nonzero_matching_outputs(
    output_word: int, expected_failure: bool
) -> None:
    """Matching plans are insufficient when either plan returns only zeros."""
    shape = get_shape("f_b")

    def write_output(y_buf, _w_buf, _x_buf, **_scalars):
        y_buf.numpy().view(np.uint16)[: shape.M] = output_word

    plans = {
        "per_round": {"fn": write_output},
        "per_call": {"fn": write_output},
    }
    canary = np.full((shape.M + CANARY_ELEMS,), CANARY_WORD, dtype=np.uint16)
    outputs = {plan: MockBuffer(canary.copy()) for plan in plans}
    done, _times, fails, _ref, _combos = execute_ab_loop(
        shape=shape,
        plans=plans,
        outputs=outputs,
        w_bo=MockBuffer(np.zeros(1, dtype=np.uint32)),
        xs=[MockBuffer(np.ones(1, dtype=np.uint16))],
        sets=[{}],
        iterations=1,
        warmup=0,
    )

    assert bool(fails) is expected_failure
    if expected_failure:
        assert done == 0
        assert fails == ["all-zero output plan=per_round x=0 set=0"]
    else:
        assert done == 1


def test_canonical_shape_name_aliases() -> None:
    """Verify B1/B2 aliases resolve properly."""
    assert canonical_shape_name("B1") == "moe_b1_6"
    assert canonical_shape_name("b1") == "moe_b1_6"
    assert canonical_shape_name("B2") == "moe_b2_9"
    assert canonical_shape_name("b2") == "moe_b2_9"
    assert canonical_shape_name("f_b") == "f_b"
