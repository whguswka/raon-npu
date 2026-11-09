# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""CPU reference and bit-level simulation for Raon MoE layers.

Provides:
- Exact bit-level simulation for B1 (gate+up) and B2 (down) using `gemv_bit_reference`.
- Exact float64 reference models for linear projections.
- Host SiLU-GLU activation and router accumulation models.
- Full end-to-end pipeline verification helpers.
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from kernels.gemv.reference import (
    bf16_ulp_diff,
    dequantize,
    gemv_bit_reference,
    reference as gemv_reference,
)
from quant.q4nx import (
    K_TILE,
    M_TILE,
    round_to_bf16_rne,
    to_bf16_u16,
)
from kernels.moe.pack import (
    B1_K,
    B1_M_PER_EXPERT,
    B2_K_ORIG,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    MOE_INTERMEDIATE_SIZE,
    NUM_ACTIVE_TOTAL,
    SHARED_EXPERT_ID,
)


def moe_gate_up_bit_reference(
    expert_codes: Sequence[np.ndarray],
    expert_scales: Sequence[np.ndarray],
    expert_mins: Sequence[np.ndarray],
    x: np.ndarray,
    expert_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact bit-level simulation of B1 gate+up for 7 active experts.

    Args:
        expert_codes: Sequence of uint8 arrays [2176, 2560] for each expert e.
        expert_scales: Sequence of float32 arrays [2176, 80].
        expert_mins: Sequence of float32 arrays [2176, 80].
        x: float32 or bfloat16 input vector [2560].
        expert_indices: Sequence of 7 expert IDs [e0..e5, e6=128].

    Returns:
        y_f32: float32 array [7, 2176] of rounded bfloat16 output values.
        y_u16: uint16 array [7, 2176] of bfloat16 bit patterns.
    """
    if len(expert_indices) != NUM_ACTIVE_TOTAL:
        raise ValueError(
            f"Expected {NUM_ACTIVE_TOTAL} active experts, got {len(expert_indices)}"
        )

    y_f32_list = []
    y_u16_list = []

    for slot, e in enumerate(expert_indices):
        c = expert_codes[e]
        s = expert_scales[e]
        m = expert_mins[e]
        out_f32, out_u16 = gemv_bit_reference(c, s, m, x)
        y_f32_list.append(out_f32)
        y_u16_list.append(out_u16)

    return np.stack(y_f32_list, axis=0), np.stack(y_u16_list, axis=0)


def moe_down_bit_reference(
    expert_codes: Sequence[np.ndarray],
    expert_scales: Sequence[np.ndarray],
    expert_mins: Sequence[np.ndarray],
    x_slots: np.ndarray,
    expert_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Exact bit-level simulation of B2 down for 7 active experts.

    Each expert receives its dedicated input vector x_j (shape [1280]).

    Args:
        expert_codes: Sequence of uint8 arrays [2560, 1280] for each expert e.
        expert_scales: Sequence of float32 arrays [2560, 40].
        expert_mins: Sequence of float32 arrays [2560, 40].
        x_slots: float32 or bfloat16 input array [7, 1280].
        expert_indices: Sequence of 7 expert IDs [e0..e5, e6=128].

    Returns:
        y_f32: float32 array [7, 2560] of rounded bfloat16 output values.
        y_u16: uint16 array [7, 2560] of bfloat16 bit patterns.
    """
    if len(expert_indices) != NUM_ACTIVE_TOTAL:
        raise ValueError(
            f"Expected {NUM_ACTIVE_TOTAL} active experts, got {len(expert_indices)}"
        )
    if x_slots.shape != (NUM_ACTIVE_TOTAL, B2_K_PAD):
        raise ValueError(
            f"Expected x_slots shape ({NUM_ACTIVE_TOTAL}, {B2_K_PAD}), got {x_slots.shape}"
        )

    y_f32_list = []
    y_u16_list = []

    for slot, e in enumerate(expert_indices):
        c = expert_codes[e]
        s = expert_scales[e]
        m = expert_mins[e]
        x_j = x_slots[slot]
        out_f32, out_u16 = gemv_bit_reference(c, s, m, x_j)
        y_f32_list.append(out_f32)
        y_u16_list.append(out_u16)

    return np.stack(y_f32_list, axis=0), np.stack(y_u16_list, axis=0)


def moe_host_silu_glu(
    gate_up_out: np.ndarray,
    *,
    gate_beta: float | None = None,
    value_beta: float | None = None,
) -> np.ndarray:
    """Host-side SiLU/SiTU-GLU and padding to B2 input dimension [7, 1280].

    For each active expert slot j in [0..6]:
        gate = gate_up_out[j, :1088]
        up = gate_up_out[j, 1088:]
        inter = gate*sigmoid(gate) * up                 (default SiLU-GLU)
        inter = gb*tanh(gate/gb)*sigmoid(gate)
                * vb*tanh(up/vb)                       (SiTU-GLU)
        inter_pad = pad(inter, 1280)

    Args:
        gate_up_out: Array of shape [7, 2176] (float32, float64, or bfloat16).
        gate_beta: SiTU gate beta. Must be provided together with ``value_beta``.
        value_beta: SiTU linear beta. Must be provided together with ``gate_beta``.

    Returns:
        float32 array [7, 1280] ready for B2 down projection.
    """
    if (gate_beta is None) != (value_beta is None):
        raise ValueError("gate_beta and value_beta must be provided together")

    out_arr = np.asarray(gate_up_out, dtype=np.float32)
    x_down = np.zeros((NUM_ACTIVE_TOTAL, B2_K_PAD), dtype=np.float32)

    for j in range(NUM_ACTIVE_TOTAL):
        gate = out_arr[j, :MOE_INTERMEDIATE_SIZE]
        up = out_arr[j, MOE_INTERMEDIATE_SIZE:]
        # exp(-abs(gate)) keeps sigmoid finite for both signs, including very
        # large-magnitude inputs.
        exp_neg_abs = np.exp(-np.abs(gate))
        sigmoid_gate = np.where(
            gate >= 0,
            1.0 / (1.0 + exp_neg_abs),
            exp_neg_abs / (1.0 + exp_neg_abs),
        )
        if gate_beta is None:
            gate_term = gate * sigmoid_gate
            value_term = up
        else:
            # With beta=4 and linear beta=25 this exactly follows
            # modeling_raon1_mini.py::situ_glu:
            # gb*tanh(g/gb)*sigmoid(g) * vb*tanh(v/vb).
            gate_term = gate_beta * np.tanh(gate / gate_beta) * sigmoid_gate
            value_term = value_beta * np.tanh(up / value_beta)
        inter = gate_term * value_term
        x_down[j, :MOE_INTERMEDIATE_SIZE] = inter

    return x_down


def moe_host_accumulate(
    down_out: np.ndarray,
    router_weights: Sequence[float] | np.ndarray,
    shared_scale: float = 1.0,
) -> np.ndarray:
    """Host-side weighted accumulation of 7 expert down-projections.

    final_y = sum_{j=0..5} (router_weights[j] * down_out[j]) + shared_scale * down_out[6]

    Args:
        down_out: Array of shape [7, 2560].
        router_weights: Weights for the 6 routed experts (length 6).
        shared_scale: Scalar multiplier for shared expert (default 1.0).

    Returns:
        float32 or float64 array of shape [2560].
    """
    down = np.asarray(down_out)
    rw = np.asarray(router_weights, dtype=down.dtype)
    if len(rw) != 6:
        raise ValueError(f"Expected 6 router weights, got {len(rw)}")

    # Routed experts accumulation
    routed_acc = np.zeros(B2_M_PER_EXPERT, dtype=down.dtype)
    for j in range(6):
        routed_acc += rw[j] * down[j]

    # Shared expert addition
    shared_part = shared_scale * down[6]
    return routed_acc + shared_part
