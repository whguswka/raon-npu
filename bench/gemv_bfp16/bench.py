#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Laptop NPU benchmark runner for Raon general 9-bit bfp16ebs8 GEMV.

Benchmarks AMD XDNA 2 AIE2P NPU for 9-bit bfp16ebs8 projection layers
(RDA q/k/v/g bundle, o_proj, MLA q_a+kv_a, MLA q_b, Expert down, Dense MLP down, Output layer, Fixed overhead).
Includes:
- Per-shape child process timeout protection against NPU deadlocks (--timeout, default: 300s)
- Dual accuracy evaluation:
  (a) Bit-exact verification against conv_even simulated bit model (integer contract sample)
  (b) Realistic float sample verification against float64 ideal (cosine >= 0.9999)
  (c) Dedicated non-integer sample bit check (informational)
- Benchmark iterations (--warmup, --iters) with NPU hardware execution timing
- Effective bandwidth (9-bit packed weight bytes / NPU execution time)
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

from quant.bfp16 import (
    K_TILE,
    M_TILE,
    pack_bfp16_gemv,
    pad_bfp16_gemv,
    parse_bfp16_gemv,
    quantize_bfp16ebs8,
)
from quant.q4nx import round_to_bf16_rne, to_bf16_u16
from kernels.gemv_bfp16.reference import (
    bf16_ulp_diff,
    bfp16_gemv_bit_reference,
    reference,
)

SENTINEL = 99.0

# Raon preset shapes for 9-bit bfp16ebs8: (name, K_orig, M_orig, K_pad, M_pad, cols)
RAON_BFP16_PRESETS = [
    ("RDA q/k/v/g bundle", 2560, 20480, 2560, 20480, 8),
    ("RDA q/k/v/g bundle (cols=4)", 2560, 20480, 2560, 20480, 4),
    ("Output proj (RDA/MLA o_proj)", 5120, 2560, 5120, 2560, 8),
    ("Output proj (RDA/MLA o_proj, cols=4)", 5120, 2560, 5120, 2560, 4),
    ("MLA q_a+kv_a bundle", 2560, 1152, 2560, 1152, 8),
    ("MLA q_b", 576, 7680, 768, 7680, 8),
    ("Expert down", 1088, 2560, 1280, 2560, 8),
    ("Dense MLP down", 12032, 2560, 12032, 2560, 8),
    ("Output layer (LM head)", 2560, 260096, 2560, 260096, 8),
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
    """Run compile, dual verification, and benchmark for a single 9-bit GEMV shape on NPU."""
    from ml_dtypes import bfloat16
    import aie.utils as aie_utils
    from iron.common import AIEContext
    from kernels.gemv_bfp16.op import GEMV_BFP16

    rng = np.random.default_rng(seed)

    # 1. Compile NPU operator
    op = GEMV_BFP16(K=K_pad, M=M_pad, cols=cols, context=AIEContext())
    t0 = time.perf_counter()
    op.compile()
    t_compile = time.perf_counter() - t0
    run = op.get_callable()

    # Allocate NPU output buffer (M_pad bf16 elements)
    y_buf = aie_utils.full((M_pad,), SENTINEL, dtype=bfloat16)

    # -------------------------------------------------------------------------
    # 2. (가) 비트 대조 (Bit-exact Verification via Integer Sample)
    # -------------------------------------------------------------------------
    # Small integers for weights and x (-4..4) ensure all dot products are exact in float32.
    w_exact_raw = rng.integers(-4, 5, (M_pad, K_pad)).astype(np.float32)
    # Zero out padded submatrix regions to preserve active outputs
    w_exact_raw[M_orig:, :] = 0.0
    w_exact_raw[:, K_orig:] = 0.0

    w_exact_bytes = pack_bfp16_gemv(w_exact_raw, rows=4, rounding="conv_even")
    w_exact_np = np.frombuffer(w_exact_bytes, dtype=np.uint8)

    x_exact_orig = rng.integers(-4, 5, K_orig)
    x_exact_pad = np.pad(x_exact_orig, (0, K_pad - K_orig), constant_values=0.0).astype(bfloat16)

    # Bit reference model (conv_even rounding, float32 accumulation, conv_even epilogue)
    _, ref_bit_y_u16 = bfp16_gemv_bit_reference(w_exact_raw, x_exact_pad, rounding="conv_even")
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
    w_real_pad, _ = pad_bfp16_gemv(w_real_orig, m_align=128, k_align=256)

    w_real_bytes = pack_bfp16_gemv(w_real_pad, rows=4, rounding="conv_even")
    w_real_np = np.frombuffer(w_real_bytes, dtype=np.uint8)

    x_real_orig = rng.standard_normal(K_orig).astype(np.float32)
    x_real_pad = np.pad(x_real_orig, (0, K_pad - K_orig), constant_values=0.0).astype(bfloat16)

    # CPU Reference (float64 dequantized ideal on active submatrix)
    w_deq = parse_bfp16_gemv(w_real_bytes, K_pad, M_pad, rows=4).astype(np.float64)
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
    x_nonint_raw = rng.uniform(0.125, 1.875, K_orig)
    x_nonint_orig = x_nonint_raw.astype(bfloat16)
    x_nonint_pad = np.pad(x_nonint_orig, (0, K_pad - K_orig), constant_values=0.0).astype(bfloat16)
    x_nonint_pad[:16] = bfloat16(0.5)

    w_nonint_orig = rng.uniform(-1.0, 1.0, (M_orig, K_orig)).astype(np.float32)
    w_nonint_pad, _ = pad_bfp16_gemv(w_nonint_orig, m_align=128, k_align=256)
    w_nonint_bytes = pack_bfp16_gemv(w_nonint_pad, rows=4, rounding="conv_even")
    w_nonint_np = np.frombuffer(w_nonint_bytes, dtype=np.uint8)

    w_nonint_buf = aie_utils.tensor(w_nonint_np.view(np.uint32))
    x_nonint_buf = aie_utils.tensor(x_nonint_pad)

    y_buf[:] = SENTINEL
    y_buf.to("npu")
    run(y_buf, w_nonint_buf, x_nonint_buf)

    got_nonint_y_u16 = y_buf.numpy()[:M_orig].view(np.uint16)
    _, ref_nonint_y_u16 = bfp16_gemv_bit_reference(w_nonint_pad, x_nonint_pad, rounding="conv_even")
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
    timing_sources: list[str] = []
    total_calls = warmup + iters

    for i in range(total_calls):
        t_start = time.perf_counter()
        ret = run(y_buf, w_real_buf, x_real_buf)
        t_end = time.perf_counter()

        e2e_ms = (t_end - t_start) * 1e3
        npu_ns = _extract_npu_time_ns(ret)
        if npu_ns is not None:
            npu_ms = npu_ns / 1e6
            src = "npu"
        else:
            npu_ms = e2e_ms
            src = "e2e"

        if i >= warmup:
            npu_times_ms.append(npu_ms)
            e2e_times_ms.append(e2e_ms)
            timing_sources.append(src)

    npu_arr = np.array(npu_times_ms)
    e2e_arr = np.array(e2e_times_ms)

    npu_avg_ms = float(np.mean(npu_arr))
    npu_p50_ms = float(np.percentile(npu_arr, 50))
    npu_p99_ms = float(np.percentile(npu_arr, 99))

    # Actual transferred weight buffer bytes (padded bfp16 GEMV block layout)
    packed_bytes = len(w_real_bytes)
    read_bw_gbs = (packed_bytes / 1e6) / npu_avg_ms if npu_avg_ms > 0 else 0.0

    # Timing source: "npu" if hardware timer extracted, else "e2e"
    timing_source = "npu" if all(s == "npu" for s in timing_sources) else (
        "e2e" if all(s == "e2e" for s in timing_sources) else "mixed"
    )

    return {
        "name": name,
        "K_orig": K_orig,
        "M_orig": M_orig,
        "K_pad": K_pad,
        "M_pad": M_pad,
        "cols": cols,
        "t_compile_s": t_compile,
        # Accuracy: Bit-exact (Integer contract sample)
        "bit_pass": bit_pass,
        "bit_mismatches": bit_mismatches,
        "bit_max_ulp": bit_max_ulp,
        # Accuracy: Realistic sample vs float64 ideal
        "real_pass": real_pass,
        "cos_sim": cos_sim,
        "rel_l2_err": rel_l2_err,
        "max_abs_err": max_abs_err,
        # Accuracy: Non-integer bit comparison (info only)
        "nonint_bit_mismatches": nonint_bit_mismatches,
        "nonint_bit_max_ulp": nonint_bit_max_ulp,
        # Timing & Performance
        "npu_avg_ms": npu_avg_ms,
        "npu_p50_ms": npu_p50_ms,
        "npu_p99_ms": npu_p99_ms,
        "e2e_avg_ms": float(np.mean(e2e_arr)),
        "packed_bytes": packed_bytes,
        "read_bw_gbs": read_bw_gbs,
        "gb_s": read_bw_gbs,
        "timing_source": timing_source,
    }


def _worker_proc(queue: multiprocessing.Queue, case_args: dict):
    """Worker process target to isolate NPU crash or hang."""
    try:
        res = _run_single_case(**case_args)
        queue.put(("ok", res))
    except Exception as exc:
        import traceback

        queue.put(("err", str(exc), traceback.format_exc()))


def main():
    parser = argparse.ArgumentParser(description="Raon general 9-bit bfp16ebs8 GEMV benchmark.")
    parser.add_argument(
        "--preset",
        choices=["raon"],
        default="raon",
        help="Preset shapes (default: raon)",
    )
    parser.add_argument(
        "--shape",
        action="append",
        metavar="K,M",
        help="Custom shape K,M (repeatable, e.g. --shape 2560,20480)",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Override active core columns (1..8)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="Benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-shape timeout in seconds (default: 300s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="RNG seed (default: 1234)",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to output results JSON",
    )
    args = parser.parse_args()

    # Determine cases to run
    cases = []
    if args.shape:
        for idx, s in enumerate(args.shape):
            parts = [int(p.strip()) for p in s.split(",")]
            k, m = parts[0], parts[1]
            k_pad = ((k + 255) // 256) * 256
            m_pad = ((m + 127) // 128) * 128
            cols = args.cols if args.cols is not None else 8
            cases.append((f"Custom shape #{idx+1}", k, m, k_pad, m_pad, cols))
    else:
        for name, k, m, kp, mp, c in RAON_BFP16_PRESETS:
            cols = args.cols if args.cols is not None else c
            cases.append((name, k, m, kp, mp, cols))

    print("=" * 110)
    print("Raon-1-mini 9-bit bfp16ebs8 GEMV Benchmark (Dual Accuracy & Hardware Timing)")
    print(f"Configurations: {len(cases)} shapes | Warmup: {args.warmup} | Iters: {args.iters} | Timeout: {args.timeout}s")
    print("=" * 110)

    results = []
    total_start = time.perf_counter()

    for idx, (name, k, m, kp, mp, cols) in enumerate(cases, 1):
        case_args = {
            "name": name,
            "K_orig": k,
            "M_orig": m,
            "K_pad": kp,
            "M_pad": mp,
            "cols": cols,
            "warmup": args.warmup,
            "iters": args.iters,
            "seed": args.seed + idx,
        }

        print(f"\n[{idx}/{len(cases)}] Shape: {name} (K={k}, M={m} -> Pad K={kp}, M={mp}, cols={cols})")
        sys.stdout.flush()

        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=_worker_proc, args=(queue, case_args))
        proc.start()
        proc.join(timeout=args.timeout)

        if proc.is_alive():
            proc.kill()
            proc.join()
            print(f"  Result [TIMEOUT]: Process killed after exceeding {args.timeout}s limit.")
            res_entry = {
                "name": name,
                "K_orig": k,
                "M_orig": m,
                "K_pad": kp,
                "M_pad": mp,
                "cols": cols,
                "status": "TIMEOUT",
            }
        else:
            if not queue.empty():
                status, *payload = queue.get()
                if status == "ok":
                    data = payload[0]
                    data["status"] = "PASS" if (data["bit_pass"] and data["real_pass"]) else "FAIL"
                    res_entry = data
                    t_src_str = f" ({data['timing_source']})" if "timing_source" in data else ""
                    print(
                        f"  Result [{res_entry['status']}]: "
                        f"Bit (ULP) mismatches={data['bit_mismatches']}, max_ulp={data['bit_max_ulp']} | "
                        f"Real cosine={data['cos_sim']:.6f}, rel_l2={data['rel_l2_err']:.6f} | "
                        f"NPU avg={data['npu_avg_ms']:.4f} ms (p50={data['npu_p50_ms']:.4f}, p99={data['npu_p99_ms']:.4f}){t_src_str} | "
                        f"BW={data['read_bw_gbs']:.2f} GB/s"
                    )
                else:
                    err_msg, tb = payload[0], payload[1]
                    print(f"  Result [FAIL]: Exception in worker:\n{err_msg}\n{tb}")
                    res_entry = {
                        "name": name,
                        "K_orig": k,
                        "M_orig": m,
                        "K_pad": kp,
                        "M_pad": mp,
                        "cols": cols,
                        "status": "CRASH",
                        "error": err_msg,
                    }
            else:
                exit_code = proc.exitcode
                print(f"  Result [CRASH]: Worker process crashed with exit code {exit_code}")
                res_entry = {
                    "name": name,
                    "K_orig": k,
                    "M_orig": m,
                    "K_pad": kp,
                    "M_pad": mp,
                    "cols": cols,
                    "status": "CRASH",
                    "exit_code": exit_code,
                }

        results.append(res_entry)
        sys.stdout.flush()

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 110)
    print("Summary of Raon 9-bit bfp16ebs8 GEMV Benchmark Results")
    print("=" * 110)
    print(
        f"{'Shape / Layer':<36} | {'K x M':<15} | {'Bit (ULP)':<10} | {'Cosine':<10} | "
        f"{'NPU avg':<10} | {'p50 / p99 (ms)':<16} | {'GB/s':<8} | {'Status'}"
    )
    print("-" * 110)
    for r in results:
        shape_str = f"{r['K_orig']}x{r['M_orig']}"
        if r.get("status") in ("PASS", "FAIL"):
            bit_str = f"0 ({r['bit_max_ulp']})" if r['bit_pass'] else f"{r['bit_mismatches']} ({r['bit_max_ulp']})"
            cos_str = f"{r['cos_sim']:.6f}"
            npu_str = f"{r['npu_avg_ms']:.3f} ms"
            p_str = f"{r['npu_p50_ms']:.3f} / {r['npu_p99_ms']:.3f}"
            bw_str = f"{r['gb_s']:.2f}"
            print(
                f"{r['name']:<36} | {shape_str:<15} | {bit_str:<10} | {cos_str:<10} | "
                f"{npu_str:<10} | {p_str:<16} | {bw_str:<8} | {r['status']}"
            )
        else:
            status = r.get("status", "UNKNOWN")
            print(f"{r['name']:<36} | {shape_str:<15} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':<16} | {'N/A':<8} | {status}")
    print("=" * 110)
    print(f"Total benchmark finished in {total_elapsed:.2f} s")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "total_elapsed_s": total_elapsed,
                    "results": results,
                },
                f,
                indent=2,
            )
        print(f"Saved JSON results to: {args.json_out}")


if __name__ == "__main__":
    main()
