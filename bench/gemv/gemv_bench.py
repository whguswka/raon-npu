#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Laptop NPU benchmark runner for Raon general q4nx GEMV.

Benchmarks AMD XDNA 2 AIE2P NPU for general projection GEMV layers
(RDA, MLA, Expert, Dense MLP, Output layer, and minimal shapes).
Includes:
- Per-shape child process timeout protection against NPU deadlocks (--timeout, default: 300s)
- Dual accuracy evaluation:
  (a) Bit-exact verification against conv_even simulated bit model (AMD sample contract)
  (b) Realistic float sample verification against float64 ideal (cosine >= 0.9999)
- Benchmark iterations (--warmup, --iters) with NPU hardware execution timing
- Effective bandwidth (packed weight bytes / NPU execution time)
- Flexible CLI: --preset raon, --shape K,M (repeatable), --cols, --json-out
- Dedicated one-time NPU tensor allocation (never pass numpy on each call)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np

# Ensure repository and IRON paths are in sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

IRON_DEFAULT_PATH = Path(r"C:\dev\IRON")
if IRON_DEFAULT_PATH.exists() and str(IRON_DEFAULT_PATH) not in sys.path:
    sys.path.insert(0, str(IRON_DEFAULT_PATH))

from quant.q4nx import (
    K_TILE,
    M_TILE,
    dequantize_simple,
    pack_gemv,
    pad_gemv,
    quantize,
    round_to_bf16_rne,
    to_bf16_u16,
)
from kernels.gemv.reference import (
    bf16_ulp_diff,
    gemv_bit_reference,
    reference,
)

SENTINEL = 99.0

# Raon preset shapes: (name, K_orig, M_orig, K_pad, M_pad, cols)
RAON_PRESETS = [
    ("RDA q/k/v/g bundle", 2560, 20480, 2560, 20480, 8),
    ("RDA q/k/v/g bundle (cols=4)", 2560, 20480, 2560, 20480, 4),
    ("Output proj (RDA/MLA o_proj)", 5120, 2560, 5120, 2560, 8),
    ("Output proj (RDA/MLA o_proj, cols=4)", 5120, 2560, 5120, 2560, 4),
    ("MLA q_a+kv_a bundle", 2560, 1152, 2560, 1152, 8),
    ("MLA q_b", 576, 7680, 768, 7680, 8),
    ("Expert 1 gate+up bundle", 2560, 2176, 2560, 2176, 8),
    ("Expert 1 gate+up bundle (cols=4)", 2560, 2176, 2560, 2176, 4),
    ("Expert 1 down", 1088, 2560, 1280, 2560, 8),
    ("Dense MLP gate+up", 2560, 24064, 2560, 24064, 8),
    ("Dense MLP down", 12032, 2560, 12032, 2560, 8),
    ("LM head reference size", 2560, 260096, 2560, 260096, 8),
    ("Fixed overhead test", 256, 128, 256, 128, 8),
]


def _extract_npu_time_ns(ret) -> int | None:
    """Extract hardware NPU execution time in nanoseconds from callable return value."""
    candidate = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
    return getattr(candidate, "npu_time", None)


def _compute_cosine_sim(u: np.ndarray, v: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    u_f64 = u.astype(np.float64)
    v_f64 = v.astype(np.float64)
    norm_u = np.linalg.norm(u_f64)
    norm_v = np.linalg.norm(v_f64)
    if norm_u == 0.0 or norm_v == 0.0:
        return 1.0 if norm_u == norm_v else 0.0
    return float(np.dot(u_f64, v_f64) / (norm_u * norm_v))


def _run_single_case(
    name: str,
    K_orig: int,
    M_orig: int,
    K_pad: int,
    M_pad: int,
    cols: int,
    warmup: int,
    iters: int,
    seed: int,
) -> dict:
    """Run compile, dual verification, and benchmark for a single GEMV shape on NPU."""
    from ml_dtypes import bfloat16
    import aie.utils as aie_utils
    from iron.common import AIEContext
    from kernels.gemv.op import GEMV

    rng = np.random.default_rng(seed)

    # 1. Compile NPU operator
    op = GEMV(K=K_pad, M=M_pad, cols=cols, context=AIEContext())
    t0 = time.perf_counter()
    op.compile()
    t_compile = time.perf_counter() - t0
    run = op.get_callable()

    # Allocate NPU output buffer (M_pad bf16 elements)
    y_buf = aie_utils.full((M_pad,), SENTINEL, dtype=bfloat16)

    # -------------------------------------------------------------------------
    # 2. (가) 비트 대조 (Bit-exact Verification via AMD sample contract)
    # -------------------------------------------------------------------------
    # Generate exact contract sample: small integers for x (-4..4), codes (0..15),
    # and small powers of two for scales and minima.
    n_groups = K_pad // 32
    x_exact_orig = rng.integers(-4, 5, K_orig)
    x_exact_pad = np.pad(x_exact_orig, (0, K_pad - K_orig), constant_values=0.0).astype(bfloat16)

    codes_exact = rng.integers(0, 16, (M_pad, K_pad), dtype=np.uint8)
    params_exact = rng.choice([-1.0, 1.0], (M_pad, 2 * n_groups)) * (
        2.0 ** (-rng.integers(0, 5, (M_pad, 2 * n_groups)))
    )
    scales_exact = params_exact[:, :n_groups].astype(np.float32)
    mins_exact = params_exact[:, n_groups:].astype(np.float32)

    w_exact_bytes = pack_gemv(codes_exact, scales_exact, mins_exact, rows=4)
    w_exact_np = np.frombuffer(w_exact_bytes, dtype=np.uint8)

    # Simulated bit reference model (conv_even rounding for D and S, float32 accumulation, conv_even epilogue)
    _, ref_bit_y_u16 = gemv_bit_reference(codes_exact, scales_exact, mins_exact, x_exact_pad)
    ref_bit_active_u16 = ref_bit_y_u16[:M_orig]

    # NPU execution for bit-exact sample
    w_exact_buf = aie_utils.tensor(w_exact_np.view(np.uint32))
    x_exact_buf = aie_utils.tensor(x_exact_pad)
    y_buf[:] = SENTINEL
    y_buf.to("npu")
    run(y_buf, w_exact_buf, x_exact_buf)

    got_bit_y_u16 = y_buf.numpy()[:M_orig].view(np.uint16)
    mismatch_mask = (got_bit_y_u16 != ref_bit_active_u16)
    bit_mismatches = int(np.sum(mismatch_mask))
    ulp_diffs = bf16_ulp_diff(got_bit_y_u16, ref_bit_active_u16)
    bit_max_ulp = int(np.max(ulp_diffs)) if len(ulp_diffs) > 0 else 0
    bit_pass = bool(bit_mismatches == 0)

    # -------------------------------------------------------------------------
    # 3. (나) 현실 표본 (Realistic Sample Verification against float64 ideal)
    # -------------------------------------------------------------------------
    w_real_orig = rng.standard_normal((M_orig, K_orig)).astype(np.float32) * 0.05
    w_real_pad, _ = pad_gemv(w_real_orig, m_align=128, k_align=256)

    codes_real, scales_real, mins_real = quantize(w_real_pad)
    w_real_bytes = pack_gemv(codes_real, scales_real, mins_real, rows=4)
    w_real_np = np.frombuffer(w_real_bytes, dtype=np.uint8)

    x_real_orig = rng.standard_normal(K_orig).astype(np.float32)
    x_real_pad = np.pad(x_real_orig, (0, K_pad - K_orig), constant_values=0.0).astype(bfloat16)

    # CPU Reference (float64 dequantized ideal on active submatrix)
    w_deq = dequantize_simple(codes_real, scales_real, mins_real).astype(np.float64)
    ref_real_y = (w_deq[:M_orig, :K_orig] @ x_real_orig.astype(np.float64)).astype(np.float64)

    # Allocate NPU buffers for realistic weights & input
    w_real_buf = aie_utils.tensor(w_real_np.view(np.uint32))
    x_real_buf = aie_utils.tensor(x_real_pad)

    y_buf[:] = SENTINEL
    y_buf.to("npu")
    run(y_buf, w_real_buf, x_real_buf)
    got_real_y = y_buf.numpy()[:M_orig].astype(np.float64)

    err = np.abs(got_real_y - ref_real_y)
    max_abs_err = float(np.max(err))
    norm_ref = np.linalg.norm(ref_real_y)
    rel_l2_err = float(np.linalg.norm(err) / norm_ref) if norm_ref > 0 else 0.0
    cos_sim = _compute_cosine_sim(got_real_y, ref_real_y)
    real_pass = bool(cos_sim >= 0.9999)

    # -------------------------------------------------------------------------
    # 3-1. (다) 전용 비정수 표본 비트 대조 (Dedicated Non-integer Bit Check - Info Only)
    # -------------------------------------------------------------------------
    # Dedicated non-integer bf16 sample:
    # - x, scales, mins are non-integer bf16 values.
    # - codes are random uint8 (0..15).
    # - At least one group (group 0) has a confirmed tie boundary where
    #   D = sum(code * x) falls exactly on the midpoint between two bf16 values:
    #   x[:32] = 0.5 (non-integer bf16), codes[:31] = 8, codes[31] = 11 -> sum(code) = 259.
    #   With x=0.5, D = 129.5 (exact tie between 129.0 and 130.0; conv_even -> 130.0).
    # - Informational only: not used for pass/fail decision because hardware
    #   accumulator rounding order and aie::reduce_add add-order inside utils.h
    #   are unspecified in kernel code.
    x_nonint_raw = rng.uniform(0.125, 1.875, K_orig)
    x_nonint_orig = x_nonint_raw.astype(bfloat16)
    x_nonint_pad = np.pad(x_nonint_orig, (0, K_pad - K_orig), constant_values=0.0).astype(bfloat16)
    x_nonint_pad[:32] = bfloat16(0.5)

    codes_nonint = rng.integers(0, 16, (M_pad, K_pad), dtype=np.uint8)
    codes_nonint[:, :31] = 8
    codes_nonint[:, 31] = 11

    scales_nonint = (rng.uniform(0.125, 1.875, (M_pad, n_groups))).astype(bfloat16).astype(np.float32)
    mins_nonint = (rng.uniform(-1.875, -0.125, (M_pad, n_groups))).astype(bfloat16).astype(np.float32)
    scales_nonint[:, 0] = np.float32(0.75)
    mins_nonint[:, 0] = np.float32(-0.25)

    w_nonint_bytes = pack_gemv(codes_nonint, scales_nonint, mins_nonint, rows=4)
    w_nonint_np = np.frombuffer(w_nonint_bytes, dtype=np.uint8)
    w_nonint_buf = aie_utils.tensor(w_nonint_np.view(np.uint32))
    x_nonint_buf = aie_utils.tensor(x_nonint_pad)

    y_buf[:] = SENTINEL
    y_buf.to("npu")
    run(y_buf, w_nonint_buf, x_nonint_buf)

    got_nonint_y_u16 = y_buf.numpy()[:M_orig].view(np.uint16)
    _, ref_nonint_y_u16 = gemv_bit_reference(codes_nonint, scales_nonint, mins_nonint, x_nonint_pad)
    ref_nonint_active_u16 = ref_nonint_y_u16[:M_orig]
    nonint_mismatch_mask = (got_nonint_y_u16 != ref_nonint_active_u16)
    nonint_bit_mismatches = int(np.sum(nonint_mismatch_mask))
    nonint_ulp_diffs = bf16_ulp_diff(got_nonint_y_u16, ref_nonint_active_u16)
    nonint_bit_max_ulp = int(np.max(nonint_ulp_diffs)) if len(nonint_ulp_diffs) > 0 else 0

    # -------------------------------------------------------------------------
    # 4. Benchmark iterations (Timing using allocated NPU buffers)
    # -------------------------------------------------------------------------
    npu_times_ms: list[float] = []
    e2e_times_ms: list[float] = []
    total_calls = warmup + iters

    for i in range(total_calls):
        t_start = time.perf_counter()
        ret = run(y_buf, w_real_buf, x_real_buf)
        t_end = time.perf_counter()

        e2e_ms = (t_end - t_start) * 1e3
        npu_ns = _extract_npu_time_ns(ret)
        npu_ms = (npu_ns / 1e6) if npu_ns is not None else e2e_ms

        if i >= warmup:
            npu_times_ms.append(npu_ms)
            e2e_times_ms.append(e2e_ms)

    npu_arr = np.array(npu_times_ms)
    e2e_arr = np.array(e2e_times_ms)

    npu_avg_ms = float(np.mean(npu_arr))
    npu_p50_ms = float(np.percentile(npu_arr, 50))
    npu_p99_ms = float(np.percentile(npu_arr, 99))
    npu_min_ms = float(np.min(npu_arr))
    npu_max_ms = float(np.max(npu_arr))

    e2e_avg_ms = float(np.mean(e2e_arr))

    packed_size_bytes = len(w_real_bytes)
    read_bw_gbs = (packed_size_bytes / (npu_avg_ms * 1e-3)) / 1e9 if npu_avg_ms > 0 else 0.0

    return {
        "name": name,
        "K_orig": K_orig,
        "M_orig": M_orig,
        "K_pad": K_pad,
        "M_pad": M_pad,
        "cols": cols,
        "compile_s": t_compile,
        "packed_bytes": packed_size_bytes,
        "bit_exact": {
            "mismatch_count": bit_mismatches,
            "total_elements": M_orig,
            "max_ulp_err": bit_max_ulp,
            "pass": bit_pass,
        },
        "nonint_bit": {
            "mismatch_count": nonint_bit_mismatches,
            "total_elements": M_orig,
            "max_ulp_err": nonint_bit_max_ulp,
            "info_only": True,
        },
        "realistic": {
            "cosine_sim": cos_sim,
            "rel_l2_err": rel_l2_err,
            "max_abs_err": max_abs_err,
            "pass": real_pass,
        },
        "bit_mismatches": bit_mismatches,
        "bit_max_ulp": bit_max_ulp,
        "bit_pass": bit_pass,
        "nonint_bit_mismatches": nonint_bit_mismatches,
        "nonint_bit_max_ulp": nonint_bit_max_ulp,
        "cosine_sim": cos_sim,
        "rel_l2_err": rel_l2_err,
        "max_abs_err": max_abs_err,
        "accuracy_pass": (bit_pass and real_pass),
        "npu_avg_ms": npu_avg_ms,
        "npu_p50_ms": npu_p50_ms,
        "npu_p99_ms": npu_p99_ms,
        "npu_min_ms": npu_min_ms,
        "npu_max_ms": npu_max_ms,
        "e2e_avg_ms": e2e_avg_ms,
        "read_bw_gbs": read_bw_gbs,
        "status": "PASS" if (bit_pass and real_pass) else "FAIL",
    }


def _single_shape_worker(
    case: tuple[str, int, int, int, int, int],
    warmup: int,
    iters: int,
    seed: int,
    queue: multiprocessing.Queue,
) -> None:
    """Worker function executed inside dedicated child process per shape."""
    try:
        name, K_orig, M_orig, K_pad, M_pad, cols = case
        res = _run_single_case(
            name=name,
            K_orig=K_orig,
            M_orig=M_orig,
            K_pad=K_pad,
            M_pad=M_pad,
            cols=cols,
            warmup=warmup,
            iters=iters,
            seed=seed,
        )
        queue.put({"status": "SUCCESS", "result": res})
    except Exception as e:
        import traceback
        traceback.print_exc()
        queue.put({"status": "ERROR", "error": str(e)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Laptop NPU benchmark runner for Raon general q4nx GEMV")
    parser.add_argument(
        "--preset",
        choices=["raon", "none"],
        default="none",
        help="Run predefined Raon projection GEMV shapes",
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=str,
        help="Specify custom shape as K,M (e.g. --shape 2560,20480). Can be repeated.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=8,
        help="Active core columns for custom shapes (default: 8)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations per shape (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark iterations per shape (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds per shape to protect against NPU deadlocks (default: 300)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatable weight generation (default: 42)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Path to output JSONL file",
    )
    args = parser.parse_args()

    # Build cases list: (name, K_orig, M_orig, K_pad, M_pad, cols)
    cases = []
    if args.preset == "raon":
        cases.extend(RAON_PRESETS)

    if args.shape:
        for s in args.shape:
            k_str, m_str = s.split(",")
            k_val, m_val = int(k_str), int(m_str)
            k_pad = -(-k_val // 256) * 256
            m_pad = -(-m_val // 128) * 128
            case_name = f"Custom K={k_val}->{k_pad}, M={m_val}->{m_pad}"
            cases.append((case_name, k_val, m_val, k_pad, m_pad, args.cols))

    if not cases:
        print("No shapes specified. Please use --preset raon or --shape K,M.")
        sys.exit(1)

    print("=" * 80)
    print("Raon General q4nx GEMV NPU Benchmark (Per-Shape Deadlock-Guarded)")
    print("=" * 80)
    print(f"Total shapes to run  : {len(cases)}")
    print(f"Warmup / Iters       : {args.warmup} / {args.iters}")
    print(f"Timeout per shape    : {args.timeout} s")
    print(f"Random seed          : {args.seed}")
    if args.json_out:
        print(f"JSONL output path    : {args.json_out}")
    print("-" * 80)

    t_total_start = time.perf_counter()
    results = []

    for idx, case in enumerate(cases, 1):
        name, K_orig, M_orig, K_pad, M_pad, cols = case
        k_str = f"{K_orig}->{K_pad}" if K_orig != K_pad else f"{K_orig}"
        m_str = f"{M_orig}->{M_pad}" if M_orig != M_pad else f"{M_orig}"
        print(f"\n[{idx}/{len(cases)}] Running: {name} [K={k_str}, M={m_str}, cols={cols}] (timeout: {args.timeout}s) ...")

        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(
            target=_single_shape_worker,
            args=(case, args.warmup, args.iters, args.seed + idx, queue),
        )

        proc.start()
        proc.join(timeout=args.timeout)

        if proc.is_alive():
            print(f"    [TIMEOUT] Execution exceeded {args.timeout}s (deadlock guard triggered).")
            proc.terminate()
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join(timeout=2)
            res = {
                "name": name,
                "K_orig": K_orig,
                "M_orig": M_orig,
                "K_pad": K_pad,
                "M_pad": M_pad,
                "cols": cols,
                "compile_s": None,
                "packed_bytes": None,
                "bit_exact": None,
                "nonint_bit": None,
                "realistic": None,
                "bit_mismatches": None,
                "bit_max_ulp": None,
                "bit_pass": False,
                "nonint_bit_mismatches": None,
                "nonint_bit_max_ulp": None,
                "cosine_sim": None,
                "rel_l2_err": None,
                "max_abs_err": None,
                "accuracy_pass": False,
                "npu_avg_ms": None,
                "npu_p50_ms": None,
                "npu_p99_ms": None,
                "npu_min_ms": None,
                "npu_max_ms": None,
                "e2e_avg_ms": None,
                "read_bw_gbs": None,
                "status": "TIMEOUT",
                "error": f"Execution timed out after {args.timeout}s",
            }
            results.append(res)
            print("    Result [TIMEOUT]: Process terminated and NPU freed. Proceeding to next shape...")
            continue

        if queue.empty():
            print(f"    [ERROR] Process exited abnormally (exitcode {proc.exitcode}).")
            res = {
                "name": name,
                "K_orig": K_orig,
                "M_orig": M_orig,
                "K_pad": K_pad,
                "M_pad": M_pad,
                "cols": cols,
                "compile_s": None,
                "packed_bytes": None,
                "bit_exact": None,
                "nonint_bit": None,
                "realistic": None,
                "bit_mismatches": None,
                "bit_max_ulp": None,
                "bit_pass": False,
                "nonint_bit_mismatches": None,
                "nonint_bit_max_ulp": None,
                "cosine_sim": None,
                "rel_l2_err": None,
                "max_abs_err": None,
                "accuracy_pass": False,
                "npu_avg_ms": None,
                "npu_p50_ms": None,
                "npu_p99_ms": None,
                "npu_min_ms": None,
                "npu_max_ms": None,
                "e2e_avg_ms": None,
                "read_bw_gbs": None,
                "status": "CRASH",
                "error": f"Child process crashed with exit code {proc.exitcode}",
            }
            results.append(res)
            print("    Result [CRASH]: Proceeding to next shape...")
            continue

        item = queue.get()
        if item.get("status") == "SUCCESS":
            r = item["result"]
            results.append(r)
            bit_info = f"Bit: {r['bit_mismatches']} mm ({r['bit_max_ulp']} ulp) [{'PASS' if r['bit_pass'] else 'FAIL'}]"
            nonint_info = (
                f"Non-int bit: {r['nonint_bit_mismatches']} mm ({r['nonint_bit_max_ulp']} ulp) [INFO]"
                if r.get('nonint_bit_mismatches') is not None
                else "Non-int bit: - [INFO]"
            )
            real_info = f"Real: cos={r['cosine_sim']:.6f}, rel_l2={r['rel_l2_err']:.4e} [{'PASS' if r['realistic']['pass'] else 'FAIL'}]"
            perf_info = f"NPU avg={r['npu_avg_ms']:.3f} ms (p50={r['npu_p50_ms']:.3f}) | BW={r['read_bw_gbs']:.1f} GB/s"
            print(f"    Result [{r['status']}]: {bit_info} | {nonint_info} | {real_info} | {perf_info}")
        else:
            err_msg = item.get("error", "Unknown error")
            print(f"    [ERROR] Shape failed with error: {err_msg}")
            res = {
                "name": name,
                "K_orig": K_orig,
                "M_orig": M_orig,
                "K_pad": K_pad,
                "M_pad": M_pad,
                "cols": cols,
                "compile_s": None,
                "packed_bytes": None,
                "bit_exact": None,
                "nonint_bit": None,
                "realistic": None,
                "bit_mismatches": None,
                "bit_max_ulp": None,
                "bit_pass": False,
                "nonint_bit_mismatches": None,
                "nonint_bit_max_ulp": None,
                "cosine_sim": None,
                "rel_l2_err": None,
                "max_abs_err": None,
                "accuracy_pass": False,
                "npu_avg_ms": None,
                "npu_p50_ms": None,
                "npu_p99_ms": None,
                "npu_min_ms": None,
                "npu_max_ms": None,
                "e2e_avg_ms": None,
                "read_bw_gbs": None,
                "status": "ERROR",
                "error": err_msg,
            }
            results.append(res)

    total_elapsed = time.perf_counter() - t_total_start

    # Print summary table
    print("\n" + "=" * 136)
    print(
        f"{'Shape / Layer':<30} | {'K -> Pad':>10} | {'M -> Pad':>13} | {'Cols':>4} | {'Status':>7} | "
        f"{'Bit (ULP)':>11} | {'NonInt(ULP)':>11} | {'Cosine':>8} | {'NPU Avg':>9} | {'p50':>8} | {'GB/s':>6}"
    )
    print("-" * 136)
    pass_count = 0
    for r in results:
        k_str = f"{r['K_orig']}->{r['K_pad']}" if r['K_orig'] != r['K_pad'] else f"{r['K_orig']}"
        m_str = f"{r['M_orig']}->{r['M_pad']}" if r['M_orig'] != r['M_pad'] else f"{r['M_orig']}"
        status = r.get("status", "FAIL")
        if status == "PASS":
            pass_count += 1
            bit_str = f"{r['bit_mismatches']} ({r['bit_max_ulp']}u)"
            nonint_str = f"{r['nonint_bit_mismatches']} ({r['nonint_bit_max_ulp']}u)" if r.get('nonint_bit_mismatches') is not None else "-"
            cos_str = f"{r['cosine_sim']:.4f}" if r['cosine_sim'] is not None else "-"
            avg_str = f"{r['npu_avg_ms']:.3f} ms" if r['npu_avg_ms'] is not None else "-"
            p50_str = f"{r['npu_p50_ms']:.3f} ms" if r['npu_p50_ms'] is not None else "-"
            bw_str = f"{r['read_bw_gbs']:.1f}" if r['read_bw_gbs'] is not None else "-"
        else:
            bit_str = f"{r['bit_mismatches']} ({r['bit_max_ulp']}u)" if r.get('bit_mismatches') is not None else "-"
            nonint_str = f"{r['nonint_bit_mismatches']} ({r['nonint_bit_max_ulp']}u)" if r.get('nonint_bit_mismatches') is not None else "-"
            cos_str = f"{r['cosine_sim']:.4f}" if r.get('cosine_sim') is not None else "-"
            avg_str = status
            p50_str = "-"
            bw_str = "-"
        print(
            f"{r['name']:<30} | "
            f"{k_str:>10} | "
            f"{m_str:>13} | "
            f"{r['cols']:4d} | "
            f"{status:>7} | "
            f"{bit_str:>11} | "
            f"{nonint_str:>11} | "
            f"{cos_str:>8} | "
            f"{avg_str:>9} | "
            f"{p50_str:>8} | "
            f"{bw_str:>6}"
        )
    print("=" * 136)
    print(f"Total benchmark finished in {total_elapsed:.2f} seconds ({pass_count}/{len(results)} shapes passed).")

    # Write JSONL if requested
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Results successfully saved to {args.json_out}")


if __name__ == "__main__":
    main()
