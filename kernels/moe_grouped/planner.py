# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Expert-job planning, gather, SiTU preparation, and deterministic scatter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from host.backend import situ_glu
from host.weights import bf16_to_f32, f32_to_bf16_rtne
from kernels.moe.pack import B2_K_PAD, MOE_INTERMEDIATE_SIZE, SHARED_EXPERT_ID
from .layout import SLOTS, TMAX


@dataclass(frozen=True)
class Assignment:
    token: int
    route: int
    expert: int


@dataclass(frozen=True)
class Job:
    expert: int
    assignments: tuple[Assignment, ...]
    shared: bool = False

    def __post_init__(self) -> None:
        if not 1 <= len(self.assignments) <= TMAX:
            raise ValueError(f"job must contain 1..{TMAX} assignments")
        if any(item.expert != self.expert for item in self.assignments):
            raise ValueError("all job assignments must use the job expert")


@dataclass(frozen=True)
class Call:
    jobs: tuple[Job | None, ...]
    expert_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.jobs) != SLOTS or len(self.expert_slots) != SLOTS:
            raise ValueError(f"a call requires exactly {SLOTS} slots")

    @property
    def rows(self) -> tuple[int, ...]:
        """Real rows per slot (0 for an empty slot): the rows RTP word of a B1 call."""
        return tuple(0 if job is None else len(job.assignments) for job in self.jobs)


@dataclass(frozen=True)
class Plan:
    token_count: int
    route_count: int
    jobs: tuple[Job, ...]
    calls: tuple[Call, ...]
    include_shared: bool

    @property
    def useful_rows(self) -> int:
        return sum(len(job.assignments) for job in self.jobs)

    @property
    def slot_utilization(self) -> float:
        denom = len(self.calls) * SLOTS * TMAX
        return self.useful_rows / denom if denom else 0.0


def _chunks(items: list[Assignment], size: int = TMAX) -> Iterable[tuple[Assignment, ...]]:
    for start in range(0, len(items), size):
        yield tuple(items[start : start + size])


def build_plan(top_ids: np.ndarray, *, include_shared: bool = False) -> Plan:
    """Build ascending-expert jobs and groups of eight static slots."""
    ids = np.asarray(top_ids)
    if ids.ndim != 2 or ids.shape[0] < 1 or ids.shape[1] < 1:
        raise ValueError("top_ids must have shape [tokens, routes]")
    if not np.issubdtype(ids.dtype, np.integer):
        raise TypeError("top_ids must be an integer array")
    if np.any(ids < 0) or np.any(ids >= SHARED_EXPERT_ID):
        raise ValueError("routed expert IDs must be in 0..127")
    sorted_ids = np.sort(ids, axis=1)
    if np.any(sorted_ids[:, 1:] == sorted_ids[:, :-1]):
        raise ValueError("each token must route to distinct experts")

    by_expert: dict[int, list[Assignment]] = {}
    for token in range(ids.shape[0]):
        for route in range(ids.shape[1]):
            expert = int(ids[token, route])
            by_expert.setdefault(expert, []).append(Assignment(token, route, expert))
    jobs: list[Job] = []
    for expert in sorted(by_expert):
        jobs.extend(Job(expert, part) for part in _chunks(by_expert[expert]))
    if include_shared:
        shared = [Assignment(token, -1, SHARED_EXPERT_ID) for token in range(ids.shape[0])]
        jobs.extend(Job(SHARED_EXPERT_ID, part, shared=True) for part in _chunks(shared))

    calls: list[Call] = []
    for start in range(0, len(jobs), SLOTS):
        active = jobs[start : start + SLOTS]
        fallback = active[0].expert
        padded = tuple(active) + (None,) * (SLOTS - len(active))
        experts = tuple(job.expert if job is not None else fallback for job in padded)
        calls.append(Call(padded, experts))
    return Plan(ids.shape[0], ids.shape[1], tuple(jobs), tuple(calls), include_shared)


def _validate_gather_inputs(
    plan: Plan, routed: np.ndarray, shared: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray | None]:
    routed_arr = np.asarray(routed, dtype=np.uint16)
    if routed_arr.ndim not in (2, 3) or routed_arr.shape[0] != plan.token_count:
        raise ValueError("routed inputs must be [T,K] or [T,R,K]")
    if routed_arr.ndim == 3 and routed_arr.shape[1] != plan.route_count:
        raise ValueError("routed route dimension does not match the plan")
    shared_arr = None if shared is None else np.asarray(shared, dtype=np.uint16)
    if plan.include_shared and (
        shared_arr is None or shared_arr.shape[0] != plan.token_count
    ):
        raise ValueError("shared inputs [T,K] are required by this plan")
    k = routed_arr.shape[-1]
    if shared_arr is not None and shared_arr.shape != (plan.token_count, k):
        raise ValueError("shared and routed input widths must match")
    return routed_arr, shared_arr


def gather_call(
    plan: Plan, index: int, routed: np.ndarray, shared: np.ndarray | None = None
) -> np.ndarray:
    """Gather one call into a zero-padded ``[8,8,K]`` tensor."""
    routed_arr, shared_arr = _validate_gather_inputs(plan, routed, shared)
    return _gather_call_validated(plan.calls[index], routed_arr, shared_arr)


def _gather_call_validated(
    call: Call, routed_arr: np.ndarray, shared_arr: np.ndarray | None
) -> np.ndarray:
    x = np.zeros((SLOTS, TMAX, routed_arr.shape[-1]), dtype=np.uint16)
    for slot, job in enumerate(call.jobs):
        if job is None:
            continue
        for row, item in enumerate(job.assignments):
            if job.shared:
                x[slot, row] = shared_arr[item.token]
            elif routed_arr.ndim == 2:
                x[slot, row] = routed_arr[item.token]
            else:
                x[slot, row] = routed_arr[item.token, item.route]
    return x


def gather_inputs(
    plan: Plan, routed: np.ndarray, shared: np.ndarray | None = None
) -> list[np.ndarray]:
    """Gather token rows into zero-padded ``[8,8,K]`` call tensors."""
    routed_arr, shared_arr = _validate_gather_inputs(plan, routed, shared)
    return [
        _gather_call_validated(call, routed_arr, shared_arr) for call in plan.calls
    ]


def shared_situ(
    shared_gate_up: np.ndarray, *, gate_beta: float, value_beta: float
) -> np.ndarray:
    """Apply SiTU/RTNE to shared gate+up rows and append B2 padding."""
    sh = np.asarray(shared_gate_up, dtype=np.uint16)
    if sh.ndim != 2 or sh.shape[-1] != 2 * MOE_INTERMEDIATE_SIZE:
        raise ValueError("shared_gate_up must be [T,2176]")
    shared = np.zeros((sh.shape[0], B2_K_PAD), dtype=np.uint16)
    shared[..., :MOE_INTERMEDIATE_SIZE] = f32_to_bf16_rtne(
        situ_glu(
            sh[..., :MOE_INTERMEDIATE_SIZE],
            sh[..., MOE_INTERMEDIATE_SIZE:],
            gate_beta=gate_beta,
            value_beta=value_beta,
        )
    )
    return shared


def situ_call(
    call: Call,
    out: np.ndarray,
    routed_in: np.ndarray,
    *,
    gate_beta: float,
    value_beta: float,
) -> None:
    """Apply SiTU/RTNE to one B1 call and place its routed rows."""
    values = np.asarray(out, dtype=np.uint16)
    if values.ndim != 3 or values.shape[:2] != (SLOTS, TMAX):
        raise ValueError("call output must have shape [8,8,M]")
    if values.shape[-1] != 2 * MOE_INTERMEDIATE_SIZE:
        raise ValueError("B1 call output width must be 2176")
    for slot, job in enumerate(call.jobs):
        if job is None or job.shared:
            continue
        n = len(job.assignments)
        rows = values[slot, :n]
        activated = f32_to_bf16_rtne(
            situ_glu(
                rows[:, :MOE_INTERMEDIATE_SIZE],
                rows[:, MOE_INTERMEDIATE_SIZE:],
                gate_beta=gate_beta,
                value_beta=value_beta,
            )
        )
        for row, item in enumerate(job.assignments):
            routed_in[item.token, item.route, :MOE_INTERMEDIATE_SIZE] = activated[row]


def accumulate_call(
    call: Call,
    out: np.ndarray,
    routed_acc: np.ndarray,
    shared_down: np.ndarray,
    top_weights: np.ndarray,
) -> None:
    """Accumulate one B2 call in slot order, matching ``scatter_b2``."""
    values = np.asarray(out, dtype=np.uint16)
    if values.ndim != 3 or values.shape[:2] != (SLOTS, TMAX):
        raise ValueError("call output must have shape [8,8,M]")
    weights = np.asarray(top_weights, dtype=np.float32)
    for slot, job in enumerate(call.jobs):
        if job is None:
            continue
        n = len(job.assignments)
        rows = values[slot, :n]
        if job.shared:
            for row, item in enumerate(job.assignments):
                shared_down[item.token] = rows[row]
            continue
        for row, item in enumerate(job.assignments):
            routed_acc[item.token] += (
                bf16_to_f32(rows[row]) * weights[item.token, item.route]
            )


def scatter_b1(plan: Plan, outputs: Iterable[np.ndarray]) -> np.ndarray:
    """Restore B1 rows to ``[T,R,M]``; call/job order is irrelevant."""
    arrays = list(outputs)
    if len(arrays) != len(plan.calls):
        raise ValueError("one output tensor is required per call")
    m = np.asarray(arrays[0]).shape[-1] if arrays else 0
    result = np.zeros((plan.token_count, plan.route_count, m), dtype=np.uint16)
    for call, out in zip(plan.calls, arrays):
        values = np.asarray(out, dtype=np.uint16)
        if values.shape != (SLOTS, TMAX, m):
            raise ValueError("call output must have shape [8,8,M]")
        for slot, job in enumerate(call.jobs):
            if job is None or job.shared:
                continue
            for row, item in enumerate(job.assignments):
                result[item.token, item.route] = values[slot, row]
    return result


def make_b2_inputs(
    gate_up: np.ndarray,
    shared_gate_up: np.ndarray,
    *,
    gate_beta: float = 4.0,
    value_beta: float = 25.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the production SiTU/RTNE sequence and append zero padding."""
    gu = np.asarray(gate_up, dtype=np.uint16)
    sh = np.asarray(shared_gate_up, dtype=np.uint16)
    if gu.ndim != 3 or gu.shape[-1] != 2 * MOE_INTERMEDIATE_SIZE:
        raise ValueError("gate_up must be [T,R,2176]")
    if sh.shape != (gu.shape[0], 2 * MOE_INTERMEDIATE_SIZE):
        raise ValueError("shared_gate_up must be [T,2176]")
    routed = np.zeros((*gu.shape[:2], B2_K_PAD), dtype=np.uint16)
    shared = shared_situ(sh, gate_beta=gate_beta, value_beta=value_beta)
    routed[..., :MOE_INTERMEDIATE_SIZE] = f32_to_bf16_rtne(
        situ_glu(
            gu[..., :MOE_INTERMEDIATE_SIZE], gu[..., MOE_INTERMEDIATE_SIZE:],
            gate_beta=gate_beta, value_beta=value_beta,
        )
    )
    return routed, shared


def scatter_b2(
    plan: Plan,
    outputs: Iterable[np.ndarray],
    top_ids: np.ndarray,
    top_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate routed outputs by ascending expert ID and return shared down."""
    arrays = list(outputs)
    ids = np.asarray(top_ids)
    weights = np.asarray(top_weights, dtype=np.float32)
    if ids.shape != (plan.token_count, plan.route_count) or weights.shape != ids.shape:
        raise ValueError("top_ids/top_weights must match the plan")
    if len(arrays) != len(plan.calls):
        raise ValueError("one output tensor is required per call")
    m = np.asarray(arrays[0]).shape[-1] if arrays else 0
    routed_rows: dict[tuple[int, int], np.ndarray] = {}
    shared = np.zeros((plan.token_count, m), dtype=np.uint16)
    for call, out in zip(plan.calls, arrays):
        values = np.asarray(out, dtype=np.uint16)
        if values.shape != (SLOTS, TMAX, m):
            raise ValueError("call output must have shape [8,8,M]")
        for slot, job in enumerate(call.jobs):
            if job is None:
                continue
            for row, item in enumerate(job.assignments):
                if job.shared:
                    shared[item.token] = values[slot, row]
                else:
                    routed_rows[(item.token, item.route)] = values[slot, row].copy()

    final = np.zeros((plan.token_count, m), dtype=np.uint16)
    for token in range(plan.token_count):
        acc = np.zeros(m, dtype=np.float32)
        for route in sorted(range(plan.route_count), key=lambda r: int(ids[token, r])):
            acc += bf16_to_f32(routed_rows[(token, route)]) * float(weights[token, route])
        final[token] = f32_to_bf16_rtne(acc)
    return final, shared
