#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Laptop NPU benchmark runner for Raon MoE kernels (Static Templates & Patched Dispatch).

Executes a staged sequence on AMD XDNA 2 AIE2P NPU:
1. Stage 1: B1 Static Template (moe_gate_up_template, no patching, experts 0..6) correctness verification.
2. Stage 2: B1 Patched Dispatch (PatchedTemplateKernel, 5 patterns: fixed, random, adjacent, boundary, overlap)
            correctness verification, NPU/host latency, and bandwidth measurement.
3. Stage 3: B2 Static Template (moe_down_template, dedicated per-col x_j, no patching, experts 0..6)
            correctness verification.
4. Stage 4: B2 Patched Dispatch (PatchedTemplateKernel, 5 patterns) correctness verification, NPU/host latency.
5. Stage 5: Random 1,000 consecutive iterations stress test (correctness check every 100 iterations).

Failure Isolation:
- Each stage runs in an isolated child process with a 180-second timeout.
- If ANY stage fails or times out, execution HALTS IMMEDIATELY to prevent driver TDR cascades.
- All log messages include ISO timestamps and explicit flush=True for real-time visibility.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure repository and IRON paths are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

IRON_DEFAULT_PATH = Path(r"C:\dev\IRON")
if IRON_DEFAULT_PATH.exists() and str(IRON_DEFAULT_PATH) not in sys.path:
    sys.path.insert(0, str(IRON_DEFAULT_PATH))

try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    bfloat16 = np.uint16

from quant.q4nx import (
    K_TILE,
    M_TILE,
    pack_gemv,
    pad_gemv,
    parse_gemv,
    quantize,
    round_to_bf16_rne,
    to_bf16_u16,
)
from kernels.gemv.reference import (
    bf16_ulp_diff,
    gemv_bit_reference,
    reference as gemv_reference,
)
from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_K,
    B1_M_PER_EXPERT,
    B1_W_PER_CHUNK_BYTES,
    B1_W_PER_CHUNK_WORDS,
    B2_CHUNKS_PER_EXPERT,
    B2_K_ORIG,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    B2_W_PER_CHUNK_BYTES,
    B2_W_PER_CHUNK_WORDS,
    CHUNK_ROWS,
    DN_BYTES,
    DN_WORDS,
    GU_BYTES,
    GU_WORDS,
    HIDDEN_SIZE,
    MOE_INTERMEDIATE_SIZE,
    NUM_ACTIVE_ROUTED,
    NUM_ACTIVE_TOTAL,
    NUM_ROUTED_EXPERTS,
    NUM_TOTAL_EXPERTS,
    SHARED_EXPERT_ID,
    TOTAL_DN_BYTES,
    TOTAL_DN_WORDS,
    TOTAL_GU_BYTES,
    TOTAL_GU_WORDS,
    pack_expert_down,
    pack_expert_gate_up,
    parse_expert_down,
    parse_expert_gate_up,
)
from kernels.moe.reference import (
    moe_down_bit_reference,
    moe_gate_up_bit_reference,
    moe_host_accumulate,
    moe_host_silu_glu,
)

SENTINEL = 99.0
B1_TRANSFER_BYTES = 7 * GU_BYTES  # 24,371,200 bytes (~24.4 MB)
B2_TRANSFER_BYTES = 7 * DN_BYTES  # 14,336,000 bytes (~14.3 MB)

SELECTION_PATTERNS = [
    "fixed",
    "random",
    "adjacent",
    "boundary",
    "overlap_prev",
]


def _now_str() -> str:
    """Return formatted current timestamp string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _extract_npu_time_ns(ret) -> int | None:
    """Extract hardware NPU execution time in nanoseconds from callable return value."""
    candidate = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
    return getattr(candidate, "npu_time", None)


def _compute_cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    u_f64 = u.astype(np.float64).ravel()
    v_f64 = v.astype(np.float64).ravel()
    norm_u = np.linalg.norm(u_f64)
    norm_v = np.linalg.norm(v_f64)
    if norm_u == 0.0 or norm_v == 0.0:
        return 1.0 if norm_u == norm_v else 0.0
    return float(np.dot(u_f64, v_f64) / (norm_u * norm_v))


def generate_synthetic_layer_weights(seed: int = 42) -> dict[str, Any]:
    """Generate synthetic weights for all 129 experts rapidly using vector tiling."""
    rng = np.random.default_rng(seed)
    print(
        f"[{_now_str()}] Generating synthetic weights for {NUM_TOTAL_EXPERTS} experts (seed={seed})...",
        flush=True,
    )
    t0 = time.perf_counter()

    # 1. Base expert for gate_up
    gate_base = rng.standard_normal((MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)).astype(np.float32) * 0.05
    up_base = rng.standard_normal((MOE_INTERMEDIATE_SIZE, HIDDEN_SIZE)).astype(np.float32) * 0.05
    w_gu_base = np.vstack([gate_base, up_base])  # [2176, 2560]
    c_gu_base, s_gu_base, m_gu_base = quantize(w_gu_base)
    packed_gu_base = pack_gemv(c_gu_base, s_gu_base, m_gu_base, rows=4)
    gu_base_words = np.frombuffer(packed_gu_base, dtype=np.uint8).view(np.uint32)

    # 2. Base expert for down
    down_base = rng.standard_normal((B2_M_PER_EXPERT, B2_K_ORIG)).astype(np.float32) * 0.05
    down_pad_base, _ = pad_gemv(down_base, m_align=CHUNK_ROWS, k_align=K_TILE)
    c_dn_base, s_dn_base, m_dn_base = quantize(down_pad_base)
    packed_dn_base = pack_gemv(c_dn_base, s_dn_base, m_dn_base, rows=4)
    dn_base_words = np.frombuffer(packed_dn_base, dtype=np.uint8).view(np.uint32)

    w_gu_all = np.empty(TOTAL_GU_WORDS, dtype=np.uint32)
    w_dn_all = np.empty(TOTAL_DN_WORDS, dtype=np.uint32)

    codes_gu_all, scales_gu_all, mins_gu_all = [], [], []
    codes_dn_all, scales_dn_all, mins_dn_all = [], [], []

    for e in range(NUM_TOTAL_EXPERTS):
        e_mask = np.uint32(e * 0x01010101)
        w_gu_slice = gu_base_words ^ (e_mask & 0x000F000F)
        w_dn_slice = dn_base_words ^ (e_mask & 0x000F000F)

        w_gu_all[e * GU_WORDS : (e + 1) * GU_WORDS] = w_gu_slice
        w_dn_all[e * DN_WORDS : (e + 1) * DN_WORDS] = w_dn_slice

        # Extract quantization params for bit-exact reference
        c_g, s_g, m_g = parse_gemv(w_gu_slice.view(np.uint8), K=B1_K, M=B1_M_PER_EXPERT, rows=4)
        c_d, s_d, m_d = parse_gemv(w_dn_slice.view(np.uint8), K=B2_K_PAD, M=B2_M_PER_EXPERT, rows=4)
        codes_gu_all.append(c_g)
        scales_gu_all.append(s_g)
        mins_gu_all.append(m_g)
        codes_dn_all.append(c_d)
        scales_dn_all.append(s_d)
        mins_dn_all.append(m_d)

    t_gen = time.perf_counter() - t0
    print(
        f"[{_now_str()}] Synthetic weights ready in {t_gen:.2f}s: "
        f"gate_up={w_gu_all.nbytes / 1e6:.2f} MB, "
        f"down={w_dn_all.nbytes / 1e6:.2f} MB.",
        flush=True,
    )

    return {
        "w_gu_np": w_gu_all,
        "w_dn_np": w_dn_all,
        "codes_gu": codes_gu_all,
        "scales_gu": scales_gu_all,
        "mins_gu": mins_gu_all,
        "codes_dn": codes_dn_all,
        "scales_dn": scales_dn_all,
        "mins_dn": mins_dn_all,
    }


def get_selection_indices(
    pattern: str, iter_idx: int, prev_indices: list[int] | None, rng: np.random.Generator
) -> list[int]:
    """Generate 6 routed expert indices [e0..e5] for the specified selection pattern."""
    if pattern == "fixed":
        return [0, 1, 2, 3, 4, 5]
    elif pattern == "adjacent":
        return [10, 11, 12, 13, 14, 15]
    elif pattern == "boundary":
        return [122, 123, 124, 125, 126, 127]
    elif pattern == "overlap_prev":
        if prev_indices is None:
            return [0, 1, 2, 3, 4, 5]
        kept = list(prev_indices[:3])
        available = [x for x in range(NUM_ROUTED_EXPERTS) if x not in prev_indices]
        fresh = list(rng.choice(available, size=3, replace=False))
        return kept + fresh
    elif pattern == "random":
        return list(rng.choice(NUM_ROUTED_EXPERTS, size=6, replace=False))
    else:
        raise ValueError(f"Unknown selection pattern: {pattern}")


# =============================================================================
# Stage 1: B1 Static Template (moe_gate_up_template, no patching, experts 0..6)
# =============================================================================
def _run_stage_1_b1_static(weights_dict: dict, is_mock: bool):
    """Compile and verify B1 Static Template without patching (experts 0..6)."""
    print(f"\n[{_now_str()}] =======================================================", flush=True)
    print(f"[{_now_str()}] STAGE 1: B1 Static Template (moe_gate_up_template, experts 0..6)", flush=True)
    print(f"[{_now_str()}] =======================================================", flush=True)

    rng = np.random.default_rng(101)
    x_np = rng.integers(-4, 5, B1_K).astype(np.float32)
    x_bf16 = x_np.astype(bfloat16)
    w_gu_np = weights_dict["w_gu_np"]
    codes_gu = weights_dict["codes_gu"]
    scales_gu = weights_dict["scales_gu"]
    mins_gu = weights_dict["mins_gu"]

    # CPU reference for contiguous experts 0..6
    ref_indices = [0, 1, 2, 3, 4, 5, 6]
    ref_f32, ref_u16 = moe_gate_up_bit_reference(
        codes_gu, scales_gu, mins_gu, x_bf16, ref_indices
    )
    ref_u16_flat = ref_u16.ravel()

    if is_mock:
        print(f"[{_now_str()}] [Mock] Verified CPU reference shape: {ref_f32.shape}.", flush=True)
        print(f"[{_now_str()}] STAGE 1 PASSED (Mock).", flush=True)
        return

    import aie.utils as aie_utils
    from iron.common import AIEContext
    from kernels.moe.op import MoEGateUpTemplate

    print(f"[{_now_str()}] Compiling MoEGateUpTemplate static operator...", flush=True)
    t0 = time.perf_counter()
    op_tpl = MoEGateUpTemplate(context=AIEContext())
    op_tpl.compile()
    print(f"[{_now_str()}] MoEGateUpTemplate compiled in {time.perf_counter() - t0:.2f}s.", flush=True)

    run_fn = op_tpl.get_callable()

    y_buf = aie_utils.full((NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    w_buf = aie_utils.tensor(w_gu_np.view(np.uint32))
    x_buf = aie_utils.tensor(x_bf16)

    print(f"[{_now_str()}] Executing B1 static template on NPU (no patching)...", flush=True)
    y_buf[:] = SENTINEL
    y_buf.to("npu")
    t_start = time.perf_counter()
    ret = run_fn(y_buf, w_buf, x_buf)
    t_call = time.perf_counter() - t_start

    got_bit_y_u16 = y_buf.numpy().view(np.uint16)
    mismatch_mask = (got_bit_y_u16 != ref_u16_flat)
    bit_mismatches = int(np.sum(mismatch_mask))
    ulp_diffs = bf16_ulp_diff(got_bit_y_u16, ref_u16_flat)
    bit_max_ulp = int(np.max(ulp_diffs)) if len(ulp_diffs) > 0 else 0

    print(
        f"[{_now_str()}] B1 Static Template Result: Mismatches={bit_mismatches}/{got_bit_y_u16.size}, "
        f"MaxULP={bit_max_ulp}, CallTime={t_call*1000:.2f}ms",
        flush=True,
    )

    if bit_mismatches != 0:
        raise RuntimeError(
            f"B1 Static Template verification failed: Mismatches={bit_mismatches}/{got_bit_y_u16.size}, MaxULP={bit_max_ulp}"
        )
    print(f"[{_now_str()}] STAGE 1 PASSED successfully.", flush=True)


# =============================================================================
# Stage 2: B1 Patched Dispatch (PatchedTemplateKernel, 5 patterns)
# =============================================================================
def _run_stage_2_b1_patched(
    weights_dict: dict,
    warmup: int,
    iters: int,
    seed: int,
    is_mock: bool,
):
    """Benchmark B1 PatchedTemplateKernel across 5 selection patterns."""
    print(f"\n[{_now_str()}] =======================================================", flush=True)
    print(f"[{_now_str()}] STAGE 2: B1 Patched Dispatch (PatchedTemplateKernel, 5 patterns)", flush=True)
    print(f"[{_now_str()}] =======================================================", flush=True)

    rng = np.random.default_rng(seed)
    w_gu_np = weights_dict["w_gu_np"]
    codes_gu = weights_dict["codes_gu"]
    scales_gu = weights_dict["scales_gu"]
    mins_gu = weights_dict["mins_gu"]

    if not is_mock:
        import aie.utils as aie_utils
        from iron.common import AIEContext
        from kernels.moe.op import MoEGateUp

        print(f"[{_now_str()}] Preparing MoEGateUp patched operator...", flush=True)
        op_gu = MoEGateUp(context=AIEContext())
        op_gu.compile()
        run_fn = op_gu.get_callable()

        y_buf = aie_utils.full((NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
        w_buf = aie_utils.tensor(w_gu_np.view(np.uint32))
        dummy_x = np.zeros(B1_K, dtype=bfloat16)
        x_buf = aie_utils.tensor(dummy_x)
    else:
        run_fn = None

    for pattern in SELECTION_PATTERNS:
        print(f"\n[{_now_str()}] --- Testing B1 Pattern: {pattern} ---", flush=True)
        prev_indices = None
        latencies_call = []
        latencies_npu = []

        total_runs = warmup + iters
        for idx in range(total_runs):
            indices_6 = get_selection_indices(pattern, idx, prev_indices, rng)
            prev_indices = indices_6
            active_7 = indices_6 + [SHARED_EXPERT_ID]

            x_np = rng.integers(-4, 5, B1_K).astype(np.float32)
            x_bf16 = x_np.astype(bfloat16)

            if is_mock:
                if idx >= warmup:
                    latencies_call.append(0.5e-3)
                    latencies_npu.append(0.4e-3)
                continue

            x_buf[:] = x_bf16
            x_buf.to("npu")
            y_buf[:] = SENTINEL
            y_buf.to("npu")

            scalars = {f"e{i}": indices_6[i] for i in range(6)}

            t_start = time.perf_counter()
            ret = run_fn(y_buf, w_buf, x_buf, **scalars)
            t_call = time.perf_counter() - t_start

            check_idx = 1 if pattern == "overlap_prev" else 0
            if idx == check_idx:
                ref_f32, ref_u16 = moe_gate_up_bit_reference(
                    codes_gu, scales_gu, mins_gu, x_bf16, active_7
                )
                ref_u16_flat = ref_u16.ravel()
                got_bit_y_u16 = y_buf.numpy().view(np.uint16)
                mismatch_mask = (got_bit_y_u16 != ref_u16_flat)
                bit_mismatches = int(np.sum(mismatch_mask))
                ulp_diffs = bf16_ulp_diff(got_bit_y_u16, ref_u16_flat)
                bit_max_ulp = int(np.max(ulp_diffs)) if len(ulp_diffs) > 0 else 0
                print(
                    f"[{_now_str()}] Pattern {pattern} check (experts={indices_6}): Mismatches={bit_mismatches}/{got_bit_y_u16.size}, MaxULP={bit_max_ulp}",
                    flush=True,
                )
                if bit_mismatches != 0:
                    raise RuntimeError(
                        f"B1 pattern {pattern} failed verification: Mismatches={bit_mismatches}/{got_bit_y_u16.size}, MaxULP={bit_max_ulp}"
                    )

            if idx >= warmup:
                latencies_call.append(t_call)
                npu_ns = _extract_npu_time_ns(ret)
                if npu_ns:
                    latencies_npu.append(npu_ns / 1e9)

        avg_call_ms = np.mean(latencies_call) * 1000.0
        p50_call_ms = float(np.percentile(latencies_call, 50)) * 1000.0
        p99_call_ms = float(np.percentile(latencies_call, 99)) * 1000.0
        bw_call_gbps = (B1_TRANSFER_BYTES / np.mean(latencies_call)) / 1e9

        if latencies_npu:
            avg_npu_ms = np.mean(latencies_npu) * 1000.0
            p50_npu_ms = float(np.percentile(latencies_npu, 50)) * 1000.0
            p99_npu_ms = float(np.percentile(latencies_npu, 99)) * 1000.0
            bw_npu_gbps = (B1_TRANSFER_BYTES / np.mean(latencies_npu)) / 1e9
            print(
                f"[{_now_str()}] Pattern {pattern:12s} Result: HostCall avg={avg_call_ms:.2f}ms p50={p50_call_ms:.2f}ms p99={p99_call_ms:.2f}ms ({bw_call_gbps:.2f} GB/s), "
                f"NPU avg={avg_npu_ms:.2f}ms p50={p50_npu_ms:.2f}ms p99={p99_npu_ms:.2f}ms ({bw_npu_gbps:.2f} GB/s)",
                flush=True,
            )
        else:
            print(
                f"[{_now_str()}] Pattern {pattern:12s} Result: HostCall avg={avg_call_ms:.2f}ms p50={p50_call_ms:.2f}ms p99={p99_call_ms:.2f}ms ({bw_call_gbps:.2f} GB/s)",
                flush=True,
            )

    print(f"\n[{_now_str()}] STAGE 2 PASSED successfully.", flush=True)


# =============================================================================
# Stage 3: B2 Static Template (moe_down_template, dedicated per-col x_j, experts 0..6)
# =============================================================================
def _run_stage_3_b2_static(weights_dict: dict, is_mock: bool):
    """Compile and verify B2 Static Template without patching (experts 0..6)."""
    print(f"\n[{_now_str()}] =======================================================", flush=True)
    print(f"[{_now_str()}] STAGE 3: B2 Static Template (moe_down_template, experts 0..6)", flush=True)
    print(f"[{_now_str()}] =======================================================", flush=True)

    rng = np.random.default_rng(202)
    x_slots_np = rng.integers(-4, 5, (NUM_ACTIVE_TOTAL, B2_K_PAD)).astype(np.float32)
    # Zero out padded columns [1088..1280]
    x_slots_np[:, MOE_INTERMEDIATE_SIZE:] = 0.0
    x_slots_bf16 = x_slots_np.astype(bfloat16)

    w_dn_np = weights_dict["w_dn_np"]
    codes_dn = weights_dict["codes_dn"]
    scales_dn = weights_dict["scales_dn"]
    mins_dn = weights_dict["mins_dn"]

    ref_indices = [0, 1, 2, 3, 4, 5, 6]
    ref_f32, ref_u16 = moe_down_bit_reference(
        codes_dn, scales_dn, mins_dn, x_slots_bf16, ref_indices
    )
    ref_u16_flat = ref_u16.ravel()

    if is_mock:
        print(f"[{_now_str()}] [Mock] Verified CPU reference shape: {ref_f32.shape}.", flush=True)
        print(f"[{_now_str()}] STAGE 3 PASSED (Mock).", flush=True)
        return

    import aie.utils as aie_utils
    from iron.common import AIEContext
    from kernels.moe.op import MoEDownTemplate

    print(f"[{_now_str()}] Compiling MoEDownTemplate static operator...", flush=True)
    t0 = time.perf_counter()
    op_tpl = MoEDownTemplate(context=AIEContext())
    op_tpl.compile()
    print(f"[{_now_str()}] MoEDownTemplate compiled in {time.perf_counter() - t0:.2f}s.", flush=True)

    run_fn = op_tpl.get_callable()

    y_buf = aie_utils.full((NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    w_buf = aie_utils.tensor(w_dn_np.view(np.uint32))
    x_buf = aie_utils.tensor(x_slots_bf16.ravel())

    print(f"[{_now_str()}] Executing B2 static template on NPU (no patching)...", flush=True)
    y_buf[:] = SENTINEL
    y_buf.to("npu")
    t_start = time.perf_counter()
    ret = run_fn(y_buf, w_buf, x_buf)
    t_call = time.perf_counter() - t_start

    got_bit_y_u16 = y_buf.numpy().view(np.uint16)
    mismatch_mask = (got_bit_y_u16 != ref_u16_flat)
    bit_mismatches = int(np.sum(mismatch_mask))
    ulp_diffs = bf16_ulp_diff(got_bit_y_u16, ref_u16_flat)
    bit_max_ulp = int(np.max(ulp_diffs)) if len(ulp_diffs) > 0 else 0

    print(
        f"[{_now_str()}] B2 Static Template Result: Mismatches={bit_mismatches}/{got_bit_y_u16.size}, "
        f"MaxULP={bit_max_ulp}, CallTime={t_call*1000:.2f}ms",
        flush=True,
    )

    if bit_mismatches != 0:
        raise RuntimeError(
            f"B2 Static Template verification failed: Mismatches={bit_mismatches}/{got_bit_y_u16.size}, MaxULP={bit_max_ulp}"
        )
    print(f"[{_now_str()}] STAGE 3 PASSED successfully.", flush=True)


# =============================================================================
# Stage 4: B2 Patched Dispatch (PatchedTemplateKernel, 5 patterns)
# =============================================================================
def _run_stage_4_b2_patched(
    weights_dict: dict,
    warmup: int,
    iters: int,
    seed: int,
    is_mock: bool,
):
    """Benchmark B2 PatchedTemplateKernel across 5 selection patterns."""
    print(f"\n[{_now_str()}] =======================================================", flush=True)
    print(f"[{_now_str()}] STAGE 4: B2 Patched Dispatch (PatchedTemplateKernel, 5 patterns)", flush=True)
    print(f"[{_now_str()}] =======================================================", flush=True)

    rng = np.random.default_rng(seed)
    w_dn_np = weights_dict["w_dn_np"]
    codes_dn = weights_dict["codes_dn"]
    scales_dn = weights_dict["scales_dn"]
    mins_dn = weights_dict["mins_dn"]

    if not is_mock:
        import aie.utils as aie_utils
        from iron.common import AIEContext
        from kernels.moe.op import MoEDown

        print(f"[{_now_str()}] Preparing MoEDown patched operator...", flush=True)
        op_dn = MoEDown(context=AIEContext())
        op_dn.compile()
        run_fn = op_dn.get_callable()

        y_buf = aie_utils.full((NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
        w_buf = aie_utils.tensor(w_dn_np.view(np.uint32))
        dummy_x = np.zeros(NUM_ACTIVE_TOTAL * B2_K_PAD, dtype=bfloat16)
        x_buf = aie_utils.tensor(dummy_x)
    else:
        run_fn = None

    for pattern in SELECTION_PATTERNS:
        print(f"\n[{_now_str()}] --- Testing B2 Pattern: {pattern} ---", flush=True)
        prev_indices = None
        latencies_call = []
        latencies_npu = []

        total_runs = warmup + iters
        for idx in range(total_runs):
            indices_6 = get_selection_indices(pattern, idx, prev_indices, rng)
            prev_indices = indices_6
            active_7 = indices_6 + [SHARED_EXPERT_ID]

            x_slots_np = rng.integers(-4, 5, (NUM_ACTIVE_TOTAL, B2_K_PAD)).astype(np.float32)
            x_slots_np[:, MOE_INTERMEDIATE_SIZE:] = 0.0
            x_slots_bf16 = x_slots_np.astype(bfloat16)

            if is_mock:
                if idx >= warmup:
                    latencies_call.append(0.35e-3)
                    latencies_npu.append(0.25e-3)
                continue

            x_buf[:] = x_slots_bf16.ravel()
            x_buf.to("npu")
            y_buf[:] = SENTINEL
            y_buf.to("npu")

            scalars = {f"e{i}": indices_6[i] for i in range(6)}

            t_start = time.perf_counter()
            ret = run_fn(y_buf, w_buf, x_buf, **scalars)
            t_call = time.perf_counter() - t_start

            check_idx = 1 if pattern == "overlap_prev" else 0
            if idx == check_idx:
                ref_f32, ref_u16 = moe_down_bit_reference(
                    codes_dn, scales_dn, mins_dn, x_slots_bf16, active_7
                )
                ref_u16_flat = ref_u16.ravel()
                got_bit_y_u16 = y_buf.numpy().view(np.uint16)
                mismatch_mask = (got_bit_y_u16 != ref_u16_flat)
                bit_mismatches = int(np.sum(mismatch_mask))
                ulp_diffs = bf16_ulp_diff(got_bit_y_u16, ref_u16_flat)
                bit_max_ulp = int(np.max(ulp_diffs)) if len(ulp_diffs) > 0 else 0
                print(
                    f"[{_now_str()}] Pattern {pattern} check (experts={indices_6}): Mismatches={bit_mismatches}/{got_bit_y_u16.size}, MaxULP={bit_max_ulp}",
                    flush=True,
                )
                if bit_mismatches != 0:
                    raise RuntimeError(
                        f"B2 pattern {pattern} failed verification: Mismatches={bit_mismatches}/{got_bit_y_u16.size}, MaxULP={bit_max_ulp}"
                    )

            if idx >= warmup:
                latencies_call.append(t_call)
                npu_ns = _extract_npu_time_ns(ret)
                if npu_ns:
                    latencies_npu.append(npu_ns / 1e9)

        avg_call_ms = np.mean(latencies_call) * 1000.0
        p50_call_ms = float(np.percentile(latencies_call, 50)) * 1000.0
        p99_call_ms = float(np.percentile(latencies_call, 99)) * 1000.0
        bw_call_gbps = (B2_TRANSFER_BYTES / np.mean(latencies_call)) / 1e9

        if latencies_npu:
            avg_npu_ms = np.mean(latencies_npu) * 1000.0
            p50_npu_ms = float(np.percentile(latencies_npu, 50)) * 1000.0
            p99_npu_ms = float(np.percentile(latencies_npu, 99)) * 1000.0
            bw_npu_gbps = (B2_TRANSFER_BYTES / np.mean(latencies_npu)) / 1e9
            print(
                f"[{_now_str()}] Pattern {pattern:12s} Result: HostCall avg={avg_call_ms:.2f}ms p50={p50_call_ms:.2f}ms p99={p99_call_ms:.2f}ms ({bw_call_gbps:.2f} GB/s), "
                f"NPU avg={avg_npu_ms:.2f}ms p50={p50_npu_ms:.2f}ms p99={p99_npu_ms:.2f}ms ({bw_npu_gbps:.2f} GB/s)",
                flush=True,
            )
        else:
            print(
                f"[{_now_str()}] Pattern {pattern:12s} Result: HostCall avg={avg_call_ms:.2f}ms p50={p50_call_ms:.2f}ms p99={p99_call_ms:.2f}ms ({bw_call_gbps:.2f} GB/s)",
                flush=True,
            )

    print(f"\n[{_now_str()}] STAGE 4 PASSED successfully.", flush=True)


# =============================================================================
# Stage 5: Random 1,000 Consecutive Stress Test (B1 + B2, verification every 100)
# =============================================================================
def _run_stage_5_stress(
    weights_dict: dict,
    stress_iters: int,
    seed: int,
    is_mock: bool,
):
    """Execute consecutive random MoE calls running both B1 and B2 with periodic verification."""
    print(f"\n[{_now_str()}] =======================================================", flush=True)
    print(f"[{_now_str()}] STAGE 5: Random {stress_iters} Stress Test (B1+B2, Check every 100)", flush=True)
    print(f"[{_now_str()}] =======================================================", flush=True)

    rng = np.random.default_rng(seed)

    w_gu_np = weights_dict["w_gu_np"]
    codes_gu = weights_dict["codes_gu"]
    scales_gu = weights_dict["scales_gu"]
    mins_gu = weights_dict["mins_gu"]

    w_dn_np = weights_dict["w_dn_np"]
    codes_dn = weights_dict["codes_dn"]
    scales_dn = weights_dict["scales_dn"]
    mins_dn = weights_dict["mins_dn"]

    if not is_mock:
        import aie.utils as aie_utils
        from iron.common import AIEContext
        from kernels.moe.op import MoEDown, MoEGateUp

        ctx = AIEContext()
        print(f"[{_now_str()}] Preparing MoEGateUp and MoEDown for stress test...", flush=True)
        op_gu = MoEGateUp(context=ctx)
        op_gu.compile()
        run_fn_gu = op_gu.get_callable()

        op_dn = MoEDown(context=ctx)
        op_dn.compile()
        run_fn_dn = op_dn.get_callable()

        # B1 buffers
        y_buf_gu = aie_utils.full((NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
        w_buf_gu = aie_utils.tensor(w_gu_np.view(np.uint32))
        dummy_x_gu = np.zeros(B1_K, dtype=bfloat16)
        x_buf_gu = aie_utils.tensor(dummy_x_gu)

        # B2 buffers
        y_buf_dn = aie_utils.full((NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
        w_buf_dn = aie_utils.tensor(w_dn_np.view(np.uint32))
        dummy_x_dn = np.zeros(NUM_ACTIVE_TOTAL * B2_K_PAD, dtype=bfloat16)
        x_buf_dn = aie_utils.tensor(dummy_x_dn)
    else:
        run_fn_gu = None
        run_fn_dn = None

    t_start_all = time.perf_counter()
    block_start = time.perf_counter()
    block_t_b1 = 0.0
    block_t_b2 = 0.0
    total_t_b1 = 0.0
    total_t_b2 = 0.0

    for it in range(1, stress_iters + 1):
        # Shared random expert selection for both B1 and B2
        indices_6 = list(rng.choice(NUM_ROUTED_EXPERTS, size=6, replace=False))
        scalars = {f"e{i}": indices_6[i] for i in range(6)}

        x_np_gu = rng.integers(-4, 5, B1_K).astype(np.float32)
        x_bf16_gu = x_np_gu.astype(bfloat16)

        x_slots_np = rng.integers(-4, 5, (NUM_ACTIVE_TOTAL, B2_K_PAD)).astype(np.float32)
        x_slots_np[:, MOE_INTERMEDIATE_SIZE:] = 0.0
        x_slots_bf16_dn = x_slots_np.astype(bfloat16)

        do_check = (it % 100 == 0) or (it == 1)

        if not is_mock:
            # 1. Run B1 (MoEGateUp)
            x_buf_gu[:] = x_bf16_gu
            x_buf_gu.to("npu")
            y_buf_gu[:] = SENTINEL
            y_buf_gu.to("npu")

            t0 = time.perf_counter()
            ret_gu = run_fn_gu(y_buf_gu, w_buf_gu, x_buf_gu, **scalars)
            dt_b1 = time.perf_counter() - t0

            # 2. Run B2 (MoEDown)
            x_buf_dn[:] = x_slots_bf16_dn.ravel()
            x_buf_dn.to("npu")
            y_buf_dn[:] = SENTINEL
            y_buf_dn.to("npu")

            t1 = time.perf_counter()
            ret_dn = run_fn_dn(y_buf_dn, w_buf_dn, x_buf_dn, **scalars)
            dt_b2 = time.perf_counter() - t1

            block_t_b1 += dt_b1
            block_t_b2 += dt_b2
            total_t_b1 += dt_b1
            total_t_b2 += dt_b2

            if do_check:
                active_7 = indices_6 + [SHARED_EXPERT_ID]
                # Verify B1
                ref_gu_f32, ref_gu_u16 = moe_gate_up_bit_reference(
                    codes_gu, scales_gu, mins_gu, x_bf16_gu, active_7
                )
                ref_gu_u16_flat = ref_gu_u16.ravel()
                got_gu_u16 = y_buf_gu.numpy().view(np.uint16)
                mismatches_gu = int(np.sum(got_gu_u16 != ref_gu_u16_flat))
                if mismatches_gu != 0:
                    raise RuntimeError(
                        f"Stress B1 check failed at iter {it}: Mismatches={mismatches_gu}/{got_gu_u16.size}"
                    )

                # Verify B2
                ref_dn_f32, ref_dn_u16 = moe_down_bit_reference(
                    codes_dn, scales_dn, mins_dn, x_slots_bf16_dn, active_7
                )
                ref_dn_u16_flat = ref_dn_u16.ravel()
                got_dn_u16 = y_buf_dn.numpy().view(np.uint16)
                mismatches_dn = int(np.sum(got_dn_u16 != ref_dn_u16_flat))
                if mismatches_dn != 0:
                    raise RuntimeError(
                        f"Stress B2 check failed at iter {it}: Mismatches={mismatches_dn}/{got_dn_u16.size}"
                    )
        else:
            dt_b1 = 0.5e-3
            dt_b2 = 0.35e-3
            block_t_b1 += dt_b1
            block_t_b2 += dt_b2
            total_t_b1 += dt_b1
            total_t_b2 += dt_b2

        if it % 100 == 0:
            elapsed_block = time.perf_counter() - block_start
            avg_b1_ms = (block_t_b1 / 100.0) * 1000.0
            avg_b2_ms = (block_t_b2 / 100.0) * 1000.0
            avg_tot_ms = ((block_t_b1 + block_t_b2) / 100.0) * 1000.0
            print(
                f"[{_now_str()}] Completed {it:4d}/{stress_iters} iterations "
                f"(B1: {avg_b1_ms:.2f} ms, B2: {avg_b2_ms:.2f} ms, Total: {avg_tot_ms:.2f} ms/iter, block time: {elapsed_block:.2f}s) - PASS",
                flush=True,
            )
            block_start = time.perf_counter()
            block_t_b1 = 0.0
            block_t_b2 = 0.0

    t_total = time.perf_counter() - t_start_all
    avg_b1_all = (total_t_b1 / stress_iters) * 1000.0
    avg_b2_all = (total_t_b2 / stress_iters) * 1000.0
    avg_tot_all = ((total_t_b1 + total_t_b2) / stress_iters) * 1000.0
    print(
        f"[{_now_str()}] STAGE 5 PASSED: {stress_iters} iterations completed in {t_total:.2f}s "
        f"(B1: {avg_b1_all:.2f} ms, B2: {avg_b2_all:.2f} ms, Total: {avg_tot_all:.2f} ms/iter).",
        flush=True,
    )


# =============================================================================
# Process-isolated Stage Runner
# =============================================================================
def run_stage_in_child_process(
    target_func,
    args: tuple,
    timeout_sec: int,
    stage_name: str,
):
    """Run target stage function in an isolated child process with timeout enforcement."""
    print(f"[{_now_str()}] Starting child process for {stage_name} (timeout={timeout_sec}s)...", flush=True)
    p = multiprocessing.Process(target=target_func, args=args)
    p.start()
    p.join(timeout=timeout_sec)

    if p.is_alive():
        print(
            f"[{_now_str()}] ERROR: {stage_name} TIMED OUT after {timeout_sec} seconds! Terminating process...",
            flush=True,
        )
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join()
        sys.exit(1)

    if p.exitcode != 0:
        print(
            f"[{_now_str()}] ERROR: {stage_name} FAILED with exit code {p.exitcode}! Halting pipeline...",
            flush=True,
        )
        sys.exit(p.exitcode)

    print(f"[{_now_str()}] Finished {stage_name} successfully.", flush=True)


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Raon MoE Staged Benchmark Runner")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without NPU hardware")
    parser.add_argument("--timeout", type=int, default=180, help="Per-stage timeout in seconds (default 180)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per pattern")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations per pattern (default 20)")
    parser.add_argument("--stress-iters", type=int, default=1000, help="Stage 5 stress iterations (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="Stages to execute: 1, 2, 3, 4, 5",
    )
    args = parser.parse_args()

    print(f"[{_now_str()}] =======================================================", flush=True)
    print(f"[{_now_str()}] Raon MoE Staged Benchmark Suite", flush=True)
    print(f"[{_now_str()}] Mock={args.mock}, Timeout={args.timeout}s, Stages={args.stages}", flush=True)
    print(f"[{_now_str()}] =======================================================", flush=True)

    weights_dict = generate_synthetic_layer_weights(seed=args.seed)

    # Stage 1: B1 Static Template
    if "1" in args.stages:
        t0 = time.perf_counter()
        run_stage_in_child_process(
            _run_stage_1_b1_static,
            (weights_dict, args.mock),
            timeout_sec=args.timeout,
            stage_name="Stage 1 (B1 Static Template)",
        )
        print(f"[{_now_str()}] Stage 1 completed in {time.perf_counter() - t0:.2f}s.\n", flush=True)

    # Stage 2: B1 Patched Dispatch
    if "2" in args.stages:
        t0 = time.perf_counter()
        run_stage_in_child_process(
            _run_stage_2_b1_patched,
            (weights_dict, args.warmup, args.iters, args.seed, args.mock),
            timeout_sec=args.timeout,
            stage_name="Stage 2 (B1 Patched Dispatch)",
        )
        print(f"[{_now_str()}] Stage 2 completed in {time.perf_counter() - t0:.2f}s.\n", flush=True)

    # Stage 3: B2 Static Template
    if "3" in args.stages:
        t0 = time.perf_counter()
        run_stage_in_child_process(
            _run_stage_3_b2_static,
            (weights_dict, args.mock),
            timeout_sec=args.timeout,
            stage_name="Stage 3 (B2 Static Template)",
        )
        print(f"[{_now_str()}] Stage 3 completed in {time.perf_counter() - t0:.2f}s.\n", flush=True)

    # Stage 4: B2 Patched Dispatch
    if "4" in args.stages:
        t0 = time.perf_counter()
        run_stage_in_child_process(
            _run_stage_4_b2_patched,
            (weights_dict, args.warmup, args.iters, args.seed, args.mock),
            timeout_sec=args.timeout,
            stage_name="Stage 4 (B2 Patched Dispatch)",
        )
        print(f"[{_now_str()}] Stage 4 completed in {time.perf_counter() - t0:.2f}s.\n", flush=True)

    # Stage 5: Stress Test
    if "5" in args.stages:
        t0 = time.perf_counter()
        run_stage_in_child_process(
            _run_stage_5_stress,
            (weights_dict, args.stress_iters, args.seed, args.mock),
            timeout_sec=args.timeout,
            stage_name="Stage 5 (Stress Test)",
        )
        print(f"[{_now_str()}] Stage 5 completed in {time.perf_counter() - t0:.2f}s.\n", flush=True)

    print(f"[{_now_str()}] All requested stages successfully completed!", flush=True)


if __name__ == "__main__":
    main()
