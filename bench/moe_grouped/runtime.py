# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Laptop-only baseline and grouped NPU replay execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from kernels.gemv_unified.stage2 import pad_x
from kernels.moe.pack import (
    B1_K,
    B1_M_PER_EXPERT,
    B2_M_PER_EXPERT,
    NUM_ACTIVE_ROUTED,
    NUM_ACTIVE_TOTAL,
)
from kernels.moe_grouped.layout import B1, B2, SLOTS, TMAX
from kernels.moe_grouped.op import GroupedMoE, validate_design_pair
from kernels.moe_grouped.planner import (
    build_plan,
    gather_inputs,
    make_b2_inputs,
    scatter_b1,
    scatter_b2,
)
from .data import Record


@dataclass
class Timing:
    npu_s: float = 0.0
    input_write_s: float = 0.0
    output_read_s: float = 0.0
    gather_s: float = 0.0
    situ_s: float = 0.0
    scatter_s: float = 0.0

    @property
    def host_s(self) -> float:
        return (
            self.input_write_s + self.output_read_s + self.gather_s
            + self.situ_s + self.scatter_s
        )

    @property
    def total_s(self) -> float:
        return self.npu_s + self.host_s


@dataclass(frozen=True)
class GroupedCallTiming:
    input_write_s: float
    npu_s: float
    output_read_s: float


@dataclass
class ReplayResult:
    output: np.ndarray
    shared_down: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    timing: Timing
    jobs: int
    calls: int
    utilization: float


class ReplayRuntime:
    """Reuse engine BOs and both static xclbins without changing the engine."""

    def __init__(self, backend: Any) -> None:
        if getattr(backend, "mock", True):
            raise RuntimeError("grouped benchmark requires a real NPU backend")
        if getattr(backend, "gemv", None) != "unified":
            raise RuntimeError("baseline requires the unified decode MoE operators")
        self.backend = backend
        self.run_gu, _run_sh, self.run_dn = backend._get_or_compile_moe_ops()
        self._grouped: dict[str, tuple[Callable[..., None], Any, Any]] = {}
        self._compile_grouped()

    def _compile_grouped(self) -> None:
        from npu import backend as backend_module

        context_type = getattr(backend_module, "AIEContext", None)
        context = context_type() if callable(context_type) else None
        xclbins: dict[str, str] = {}
        for shape in (B1, B2):
            op = GroupedMoE(shape.name, context=context)
            config_mode, stream_mode = op.artifact_designs
            validate_design_pair(config_mode, stream_mode)
            op.compile()
            fn = op.get_callable()
            validate_design_pair(shape, fn.shape)
            path = str(getattr(fn, "xclbin_path", getattr(fn, "_xclbin_path", "")))
            if not path:
                raise RuntimeError(f"grouped {shape.name} callable has no xclbin path")
            if f"_{shape.name}_" not in Path(path).name:
                raise RuntimeError(
                    f"grouped {shape.name} xclbin does not identify its mode: {path}"
                )
            xclbins[shape.name] = path
            run = self.backend._prepare_dynamic_kernel(fn)
            x = backend_module.aie_utils.tensor(
                np.zeros(SLOTS * TMAX * shape.k, dtype=backend_module.bfloat16)
            )
            y = backend_module.aie_utils.full(
                (SLOTS * TMAX * shape.m,), 0.0, dtype=backend_module.bfloat16
            )
            self._grouped[shape.name] = (run, x, y)
        if xclbins[B1.name] == xclbins[B2.name]:
            raise RuntimeError(
                "grouped B1/B2 must use separate mode-specific xclbins: "
                f"{xclbins[B1.name]}"
            )

    def _run_grouped_call(
        self,
        shape,
        layer: int,
        experts: tuple[int, ...],
        values: np.ndarray,
        rows: tuple[int, ...] | None = None,
    ) -> tuple[np.ndarray, GroupedCallTiming]:
        run, x_tensor, y_tensor = self._grouped[shape.name]
        w_tensor = self.backend._get_prefill_weight_tensor(f"L{layer:02d}/moe_{shape.name}")
        from npu import backend as backend_module
        started = time.perf_counter()
        self.backend._overwrite_buffer(
            x_tensor,
            backend_module._grouped_x_tiled(values, shape).reshape(-1).view(
                backend_module.bfloat16
            ),
        )
        input_write_s = time.perf_counter() - started
        scalars = {f"e{i}": int(expert) for i, expert in enumerate(experts)}
        if rows is not None:
            for i, count in enumerate(rows):
                scalars[f"r{i}"] = int(count)
        started = time.perf_counter()
        run(y_tensor, w_tensor, x_tensor, **scalars)
        npu_s = time.perf_counter() - started
        self.backend._record_xclbin_use(f"moe_grouped_{shape.name}")
        started = time.perf_counter()
        output = np.array(y_tensor.numpy().view(np.uint16), copy=True).reshape(
            SLOTS, TMAX, shape.m
        )
        output_read_s = time.perf_counter() - started
        return output, GroupedCallTiming(input_write_s, npu_s, output_read_s)

    @staticmethod
    def _add_grouped_call_timing(timing: Timing, call: GroupedCallTiming) -> None:
        timing.input_write_s += call.input_write_s
        timing.npu_s += call.npu_s
        timing.output_read_s += call.output_read_s

    def _run_decode_reference(
        self,
        shape,
        layer: int,
        experts: tuple[int, ...],
        values: np.ndarray,
        valid_rows: tuple[int, ...],
    ) -> np.ndarray:
        """Independently replay grouped rows through the decode GEMV operators."""
        from npu import backend as backend_module

        if len(experts) != SLOTS or len(valid_rows) != SLOTS:
            raise ValueError("reference requires one expert and row count per slot")
        # Decode templates are sized by the architecture, not by SLOTS: B1
        # carries the routed experts, B2 adds the shared expert as its last row.
        output = np.zeros((SLOTS, TMAX, shape.m), dtype=np.uint16)
        if shape is B1:
            run = self.run_gu
            weight = self.backend._get_weight_tensor(f"L{layer:02d}/moe_b1")
            y_tensor = self.backend._prepared_buffer("moe_y_gu")
            x_tensor = self.backend._prepared_buffer("moe_x_gu")
            mode = "moe_b1_6"
            decode_rows = NUM_ACTIVE_ROUTED
        elif shape is B2:
            run = self.run_dn
            weight = self.backend._get_weight_tensor(f"L{layer:02d}/moe_b2")
            y_tensor = self.backend._prepared_buffer("moe_y_dn")
            x_tensor = self.backend._prepared_buffer("moe_x_dn")
            mode = "moe_b2_9"
            decode_rows = NUM_ACTIVE_TOTAL
        else:
            raise ValueError(f"unsupported grouped shape {shape}")

        for slot, (expert, count) in enumerate(zip(experts, valid_rows)):
            if count < 0 or count > TMAX:
                raise ValueError(f"slot {slot}: invalid row count {count}")
            for row in range(count):
                if shape is B1:
                    packed = pad_x(values[slot, row, :B1_K], mode)
                    selected = 0
                    scalar_expert = expert
                else:
                    decode_inputs = np.zeros((NUM_ACTIVE_TOTAL, shape.k), dtype=np.uint16)
                    if expert == shape.max_expert:
                        decode_inputs[-1] = values[slot, row]
                        selected = NUM_ACTIVE_TOTAL - 1
                        scalar_expert = 0
                    else:
                        decode_inputs[0] = values[slot, row]
                        selected = 0
                        scalar_expert = expert
                    packed = pad_x(decode_inputs, mode)
                self.backend._overwrite_buffer(
                    x_tensor,
                    np.ascontiguousarray(packed, dtype=np.uint16).view(
                        backend_module.bfloat16
                    ),
                )
                scalars = {f"e{i}": int(scalar_expert) for i in range(NUM_ACTIVE_ROUTED)}
                run(y_tensor, weight, x_tensor, **scalars)
                self.backend._record_xclbin_use("gemv_unified")
                decoded = np.array(
                    y_tensor.numpy().view(np.uint16), copy=True
                ).reshape(decode_rows, shape.m)
                output[slot, row] = decoded[selected]
        return output

    def _run_baseline_probe(self, record: Record) -> float:
        """Run one real-input decode B1 call and return only its NPU call time."""
        from npu import backend as backend_module

        layer = record.layer
        experts = tuple(int(value) for value in record.top_ids)
        scalars = {f"e{i}": value for i, value in enumerate(experts)}
        w_gu = self.backend._get_weight_tensor(f"L{layer:02d}/moe_b1")
        y_gu = self.backend._prepared_buffer("moe_y_gu")
        x_gu = self.backend._prepared_buffer("moe_x_gu")
        x_pad = pad_x(record.x[:B1_K], "moe_b1_6")
        self.backend._overwrite_buffer(
            x_gu,
            np.ascontiguousarray(x_pad, dtype=np.uint16).view(backend_module.bfloat16),
        )
        started = time.perf_counter()
        self.run_gu(y_gu, w_gu, x_gu, **scalars)
        elapsed = time.perf_counter() - started
        self.backend._record_xclbin_use("gemv_unified")
        return elapsed

    @staticmethod
    def _switch_minus_steady(
        switched: Callable[[], float], steady: Callable[[], float]
    ) -> float:
        return switched() - steady()

    def transition_cost_samples(
        self, records: list[Record], grouped_b1: np.ndarray
    ) -> dict[str, float]:
        """Measure one real-input sample for each context transition edge."""
        if not records:
            raise ValueError("transition probe requires at least one record")
        layer = records[0].layer
        top_ids = np.stack([record.top_ids for record in records])
        b1_plan = build_plan(top_ids)
        b1_inputs = gather_inputs(b1_plan, np.stack([record.x for record in records]))
        b1_call, b1_values = b1_plan.calls[0], b1_inputs[0]
        shared_gu = np.stack([record.shared_gate_up for record in records])
        routed_in, shared_in = make_b2_inputs(
            grouped_b1,
            shared_gu,
            gate_beta=records[0].gate_beta,
            value_beta=records[0].value_beta,
        )
        b2_plan = build_plan(top_ids, include_shared=True)
        b2_inputs = gather_inputs(b2_plan, routed_in, shared_in)
        b2_call, b2_values = b2_plan.calls[0], b2_inputs[0]
        baseline = lambda: self._run_baseline_probe(records[0])
        b1 = lambda: self._run_grouped_call(
            B1, layer, b1_call.expert_slots, b1_values, rows=b1_call.rows
        )[1].npu_s
        b2 = lambda: self._run_grouped_call(
            B2, layer, b2_call.expert_slots, b2_values
        )[1].npu_s

        baseline()  # establish the starting decode context; not a sample
        baseline_to_b1 = self._switch_minus_steady(b1, b1)
        b1_to_b2 = self._switch_minus_steady(b2, b2)
        b2_to_baseline = self._switch_minus_steady(baseline, baseline)
        return {
            "baseline_to_b1": baseline_to_b1,
            "b1_to_b2": b1_to_b2,
            "b2_to_baseline": b2_to_baseline,
        }

    def baseline(self, records: list[Record], *, warmup: bool = False) -> ReplayResult:
        if not records:
            raise ValueError("baseline requires at least one record")
        layer = records[0].layer
        if any(record.layer != layer for record in records):
            raise ValueError("one replay batch must contain one layer")
        timing = Timing()
        from npu import backend as backend_module
        all_b1, all_b2, all_out, all_shared = [], [], [], []
        w_gu = self.backend._get_weight_tensor(f"L{layer:02d}/moe_b1")
        w_dn = self.backend._get_weight_tensor(f"L{layer:02d}/moe_b2")
        y_gu = self.backend._prepared_buffer("moe_y_gu")
        y_dn = self.backend._prepared_buffer("moe_y_dn")
        x_gu = self.backend._prepared_buffer("moe_x_gu")
        x_dn = self.backend._prepared_buffer("moe_x_dn")
        if warmup:
            self._run_baseline_probe(records[0])
        for source in records:
            record = source.fresh()
            experts = tuple(int(value) for value in record.top_ids)
            scalars = {f"e{i}": value for i, value in enumerate(experts)}
            started = time.perf_counter()
            x_pad = pad_x(record.x[:B1_K], "moe_b1_6")
            timing.gather_s += time.perf_counter() - started
            started = time.perf_counter()
            self.backend._overwrite_buffer(
                x_gu, np.ascontiguousarray(x_pad, dtype=np.uint16).view(backend_module.bfloat16)
            )
            timing.input_write_s += time.perf_counter() - started
            started = time.perf_counter()
            self.run_gu(y_gu, w_gu, x_gu, **scalars)
            timing.npu_s += time.perf_counter() - started
            self.backend._record_xclbin_use("gemv_unified")
            started = time.perf_counter()
            b1 = np.array(y_gu.numpy().view(np.uint16), copy=True).reshape(
                len(experts), B1_M_PER_EXPERT
            )
            timing.output_read_s += time.perf_counter() - started
            started = time.perf_counter()
            routed_in, shared_in = make_b2_inputs(
                b1[None, ...], record.shared_gate_up[None, ...],
                gate_beta=record.gate_beta, value_beta=record.value_beta,
            )
            slots = np.concatenate([routed_in[0], shared_in], axis=0)
            x_pad_dn = pad_x(slots, "moe_b2_9")
            timing.situ_s += time.perf_counter() - started
            started = time.perf_counter()
            self.backend._overwrite_buffer(
                x_dn, np.ascontiguousarray(x_pad_dn, dtype=np.uint16).view(backend_module.bfloat16)
            )
            timing.input_write_s += time.perf_counter() - started
            started = time.perf_counter()
            self.run_dn(y_dn, w_dn, x_dn, **scalars)
            timing.npu_s += time.perf_counter() - started
            self.backend._record_xclbin_use("gemv_unified")
            started = time.perf_counter()
            b2 = np.array(y_dn.numpy().view(np.uint16), copy=True).reshape(
                len(experts) + 1, B2_M_PER_EXPERT
            )
            timing.output_read_s += time.perf_counter() - started
            started = time.perf_counter()
            plan = build_plan(record.top_ids[None, :], include_shared=True)
            # Convert the one decode call into the common scatter representation.
            packed = np.zeros((SLOTS, TMAX, B2_M_PER_EXPERT), dtype=np.uint16)
            for slot, job in enumerate(plan.calls[0].jobs):
                if job is not None:
                    packed[slot, 0] = b2[-1] if job.shared else b2[job.assignments[0].route]
            output, shared = scatter_b2(
                plan, [packed], record.top_ids[None, :], record.top_weights[None, :]
            )
            timing.scatter_s += time.perf_counter() - started
            all_b1.append(b1); all_b2.append(b2[:-1])
            all_out.append(output[0]); all_shared.append(shared[0])
        return ReplayResult(
            np.stack(all_out), np.stack(all_shared), np.stack(all_b1),
            np.stack(all_b2), timing, len(records) * 2, len(records) * 2, 1.0,
        )

    def grouped(self, records: list[Record], *, warmup: bool = False) -> ReplayResult:
        if not records:
            raise ValueError("grouped replay requires at least one record")
        layer = records[0].layer
        if any(record.layer != layer for record in records):
            raise ValueError("one replay batch must contain one layer")
        fresh = [record.fresh() for record in records]
        top_ids = np.stack([record.top_ids for record in fresh])
        top_weights = np.stack([record.top_weights for record in fresh])
        x = np.stack([record.x for record in fresh])
        shared_gu = np.stack([record.shared_gate_up for record in fresh])
        betas = {(record.gate_beta, record.value_beta) for record in fresh}
        if len(betas) != 1:
            raise ValueError("one replay batch must use one SiTU beta pair")
        gate_beta, value_beta = next(iter(betas))
        timing = Timing()

        started = time.perf_counter()
        b1_plan = build_plan(top_ids)
        b1_inputs = gather_inputs(b1_plan, x)
        timing.gather_s += time.perf_counter() - started
        b1_outputs = []
        if warmup and b1_plan.calls:
            call, values = b1_plan.calls[0], b1_inputs[0]
            self._run_grouped_call(
                B1, layer, call.expert_slots, values, rows=call.rows
            )
        for call, values in zip(b1_plan.calls, b1_inputs):
            output, call_timing = self._run_grouped_call(
                B1, layer, call.expert_slots, values, rows=call.rows
            )
            self._add_grouped_call_timing(timing, call_timing)
            b1_outputs.append(output)
        started = time.perf_counter()
        b1 = scatter_b1(b1_plan, b1_outputs)
        timing.scatter_s += time.perf_counter() - started

        started = time.perf_counter()
        routed_in, shared_in = make_b2_inputs(
            b1, shared_gu, gate_beta=gate_beta, value_beta=value_beta
        )
        timing.situ_s += time.perf_counter() - started
        started = time.perf_counter()
        b2_plan = build_plan(top_ids, include_shared=True)
        b2_inputs = gather_inputs(b2_plan, routed_in, shared_in)
        timing.gather_s += time.perf_counter() - started
        b2_outputs = []
        if warmup and b2_plan.calls:
            call, values = b2_plan.calls[0], b2_inputs[0]
            self._run_grouped_call(B2, layer, call.expert_slots, values)
        for call, values in zip(b2_plan.calls, b2_inputs):
            output, call_timing = self._run_grouped_call(
                B2, layer, call.expert_slots, values
            )
            self._add_grouped_call_timing(timing, call_timing)
            b2_outputs.append(output)
        started = time.perf_counter()
        final, shared = scatter_b2(b2_plan, b2_outputs, top_ids, top_weights)
        timing.scatter_s += time.perf_counter() - started

        jobs = len(b1_plan.jobs) + len(b2_plan.jobs)
        calls = len(b1_plan.calls) + len(b2_plan.calls)
        useful = b1_plan.useful_rows + b2_plan.useful_rows
        utilization = useful / (calls * SLOTS * TMAX) if calls else 0.0
        # Extract routed B2 rows for intermediate comparison.
        routed_b2 = np.zeros((len(fresh), top_ids.shape[1], B2_M_PER_EXPERT), dtype=np.uint16)
        for call, output in zip(b2_plan.calls, b2_outputs):
            for slot, job in enumerate(call.jobs):
                if job is None or job.shared:
                    continue
                for row, item in enumerate(job.assignments):
                    routed_b2[item.token, item.route] = output[slot, row]
        return ReplayResult(
            final, shared, b1, routed_b2, timing, jobs, calls, utilization
        )
