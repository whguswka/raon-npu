#!/usr/bin/env python3
# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402

"""Laptop NPU benchmark for the D mixed_full MoE execution order.

Stages stop at the first failure and each run in a child process with a 240 s
default limit.  The runtime path is static-template plus DDR_PATCH only; it
never uses DispatchTime lowering.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    bfloat16 = np.uint16

from bench.moe.moe_bench import get_selection_indices, SELECTION_PATTERNS
from kernels.gemv.reference import bf16_ulp_diff, gemv_bit_reference
from kernels.gemv_bfp16.reference import bfp16_gemv_bit_reference
from kernels.moe.pack import (
    B1_K,
    B1_M_PER_EXPERT,
    B2_K_ORIG,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    DN9_BYTES,
    GU_BYTES,
    NUM_ACTIVE_ROUTED,
    NUM_ACTIVE_TOTAL,
    NUM_ROUTED_EXPERTS,
    NUM_TOTAL_EXPERTS,
)
from kernels.moe.reference import moe_host_silu_glu
from quant.bfp16 import pack_bfp16_gemv
from quant.q4nx import pack_gemv

SENTINEL = 99.0
B1_SLOTS = NUM_ACTIVE_ROUTED
B1_BYTES = B1_SLOTS * GU_BYTES
B2_BYTES = NUM_ACTIVE_TOTAL * DN9_BYTES


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _expert_rows(expert: int, rows: int, seed: int) -> np.ndarray:
    """Return a unique, integer-valued row signature for expert 0..128."""
    encoded = expert ^ (seed & 0xFF)
    bits = (encoded >> (np.arange(rows) % 9)) & 1
    return (bits * 2 - 1).astype(np.float32)


def _q4_params(expert: int, seed: int):
    """Build exact q4nx parameters for an expert-specific constant-row matrix."""
    rows = _expert_rows(expert, B1_M_PER_EXPERT, seed)
    codes = np.zeros((B1_M_PER_EXPERT, B1_K), dtype=np.uint8)
    scales = np.zeros((B1_M_PER_EXPERT, B1_K // 32), dtype=np.float32)
    mins = np.broadcast_to(rows[:, None], scales.shape)
    return codes, scales, mins


def _bfp_weight(expert: int, seed: int, rows: int, cols: int) -> np.ndarray:
    signature = _expert_rows(expert, rows, seed)
    return np.broadcast_to(signature[:, None], (rows, cols))


def _base_weights(seed: int, *, need_gu: bool, need_shared: bool, need_down: bool):
    """Create packed buffers whose expert IDs have distinct integer weights."""
    out = {}
    out["seed"] = seed
    if need_gu:
        w_gu = np.empty(NUM_ROUTED_EXPERTS * GU_BYTES // 4, dtype=np.uint32)
        for expert in range(NUM_ROUTED_EXPERTS):
            packed = pack_gemv(*_q4_params(expert, seed), rows=4)
            start = expert * GU_BYTES // 4
            w_gu[start : start + GU_BYTES // 4] = np.frombuffer(packed, np.uint32)
        out["w_gu"] = w_gu
    if need_shared:
        shared = _bfp_weight(128, seed, B1_M_PER_EXPERT, B1_K)
        packed = pack_bfp16_gemv(shared, flush_tiny=True)
        out.update(shared=shared, w_shared=np.frombuffer(packed, np.uint32).copy())
    if need_down:
        w_down = np.empty(NUM_TOTAL_EXPERTS * DN9_BYTES // 4, dtype=np.uint32)
        for expert in range(NUM_TOTAL_EXPERTS):
            down = _bfp_weight(expert, seed, B2_M_PER_EXPERT, B2_K_PAD)
            packed = pack_bfp16_gemv(down, flush_tiny=True)
            start = expert * DN9_BYTES // 4
            w_down[start : start + DN9_BYTES // 4] = np.frombuffer(packed, np.uint32)
        out["w_down"] = w_down
    return out


def _q4_ref(weights, x, experts):
    y = [
        gemv_bit_reference(*_q4_params(expert, weights["seed"]), x)[1]
        for expert in experts
    ]
    return np.stack(y).reshape(-1)


def _bfp_ref(weights, x_slots: np.ndarray, experts) -> np.ndarray:
    y = [
        bfp16_gemv_bit_reference(
            _bfp_weight(expert, weights["seed"], B2_M_PER_EXPERT, B2_K_PAD), x
        )[1]
        for expert, x in zip(experts, x_slots, strict=True)
    ]
    return np.stack(y).reshape(-1)


def _check(label: str, got, expected):
    got_u16 = np.asarray(got).view(np.uint16).reshape(-1)
    exp_u16 = np.asarray(expected, np.uint16).reshape(-1)
    mismatches = int(np.count_nonzero(got_u16 != exp_u16))
    max_ulp = int(bf16_ulp_diff(got_u16, exp_u16).max(initial=0))
    print(
        f"[{now()}] {label}: mismatches={mismatches}/{got_u16.size}, max_ulp={max_ulp}",
        flush=True,
    )
    if mismatches:
        raise RuntimeError(f"{label} bit comparison failed")


def _check_tolerance(
    label: str,
    got,
    expected,
    *,
    max_mismatch_ratio: float = 1e-3,
    max_ulp_limit: int = 4,
):
    got_u16 = np.asarray(got).view(np.uint16).reshape(-1)
    exp_u16 = np.asarray(expected, np.uint16).reshape(-1)
    mismatches = int(np.count_nonzero(got_u16 != exp_u16))
    mismatch_ratio = mismatches / got_u16.size
    max_ulp = int(bf16_ulp_diff(got_u16, exp_u16).max(initial=0))
    print(
        f"[{now()}] {label}: mismatches={mismatches}/{got_u16.size}, "
        f"mismatch_ratio={mismatch_ratio:.6g} (limit={max_mismatch_ratio:.6g}), "
        f"max_ulp={max_ulp} (limit={max_ulp_limit})",
        flush=True,
    )
    if mismatch_ratio > max_mismatch_ratio or max_ulp > max_ulp_limit:
        raise RuntimeError(f"{label} tolerance comparison failed")


def _stats(label: str, times: list[float], transfer_bytes: int):
    ms = np.asarray(times) * 1000.0
    bw = transfer_bytes / np.mean(times) / 1e9
    print(
        f"[{now()}] {label}: avg={ms.mean():.3f} ms p50={np.percentile(ms, 50):.3f} ms "
        f"p99={np.percentile(ms, 99):.3f} ms ({bw:.2f} GB/s)",
        flush=True,
    )


def _npu_api():
    import aie.utils as aie_utils
    from iron.common import AIEContext

    return aie_utils, AIEContext


def stage1(seed: int, mock: bool):
    print(f"[{now()}] Stage 1: B1 six-slot static template", flush=True)
    if mock:
        return
    from kernels.moe.op import MoEGateUpTemplate

    aie, Context = _npu_api()
    weights = _base_weights(seed, need_gu=True, need_shared=False, need_down=False)
    x = np.random.default_rng(seed + 1).integers(-4, 5, B1_K).astype(bfloat16)
    op = MoEGateUpTemplate(slots=6, context=Context())
    op.compile()
    run = op.get_callable()
    y = aie.full((6 * B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    run(y, aie.tensor(weights["w_gu"]), aie.tensor(x))
    _check("B1-6 static", y.numpy(), _q4_ref(weights, x, range(6)))


def stage2(seed: int, warmup: int, iters: int, mock: bool):
    print(f"[{now()}] Stage 2: B1 six-slot patching, five patterns", flush=True)
    if mock:
        for p in SELECTION_PATTERNS:
            _stats(f"B1 {p}", [0.0005] * iters, B1_BYTES)
        return
    from kernels.moe.op import MoEGateUp

    aie, Context = _npu_api()
    rng = np.random.default_rng(seed)
    weights = _base_weights(seed, need_gu=True, need_shared=False, need_down=False)
    op = MoEGateUp(slots=6, context=Context())
    op.compile()
    run = op.get_callable()
    y = aie.full((6 * B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    wb = aie.tensor(weights["w_gu"])
    xb = aie.tensor(np.zeros(B1_K, dtype=bfloat16))
    for pattern in SELECTION_PATTERNS:
        prev = None
        timings = []
        for i in range(warmup + iters):
            experts = get_selection_indices(pattern, i, prev, rng)
            prev = experts
            x = rng.integers(-4, 5, B1_K).astype(bfloat16)
            xb[:] = x
            xb.to("npu")
            y[:] = SENTINEL
            y.to("npu")
            t0 = time.perf_counter()
            run(y, wb, xb, **{f"e{j}": experts[j] for j in range(6)})
            if i >= warmup:
                timings.append(time.perf_counter() - t0)
            if i == 0:
                _check(f"B1 {pattern}", y.numpy(), _q4_ref(weights, x, experts))
        _stats(f"B1 {pattern}", timings, B1_BYTES)


def stage3(seed: int, mock: bool):
    print(f"[{now()}] Stage 3: B2 9-bit seven-slot static template", flush=True)
    if mock:
        return
    from kernels.moe.op import MoEDownBfp16Template

    aie, Context = _npu_api()
    rng = np.random.default_rng(seed + 3)
    weights = _base_weights(seed, need_gu=False, need_shared=False, need_down=True)
    x = rng.integers(-4, 5, (7, B2_K_PAD)).astype(np.float32)
    x[:, B2_K_ORIG:] = 0
    x = x.astype(bfloat16)
    op = MoEDownBfp16Template(context=Context())
    op.compile()
    run = op.get_callable()
    y = aie.full((7 * B2_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    run(y, aie.tensor(weights["w_down"]), aie.tensor(x.reshape(-1)))
    _check("B2-9 static", y.numpy(), _bfp_ref(weights, x, range(7)))


def stage4(seed: int, warmup: int, iters: int, mock: bool):
    print(f"[{now()}] Stage 4: B2 9-bit patching, five patterns", flush=True)
    if mock:
        for p in SELECTION_PATTERNS:
            _stats(f"B2-9 {p}", [0.0007] * iters, B2_BYTES)
        return
    from kernels.moe.op import MoEDownBfp16

    aie, Context = _npu_api()
    rng = np.random.default_rng(seed + 4)
    weights = _base_weights(seed, need_gu=False, need_shared=False, need_down=True)
    op = MoEDownBfp16(context=Context())
    op.compile()
    run = op.get_callable()
    y = aie.full((7 * B2_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    wb = aie.tensor(weights["w_down"])
    xb = aie.tensor(np.zeros(7 * B2_K_PAD, dtype=bfloat16))
    for pattern in SELECTION_PATTERNS:
        prev = None
        timings = []
        for i in range(warmup + iters):
            experts = get_selection_indices(pattern, i, prev, rng)
            prev = experts
            x = rng.integers(-4, 5, (7, B2_K_PAD)).astype(np.float32)
            x[:, B2_K_ORIG:] = 0
            x = x.astype(bfloat16)
            xb[:] = x.reshape(-1)
            xb.to("npu")
            y[:] = SENTINEL
            y.to("npu")
            t0 = time.perf_counter()
            run(y, wb, xb, **{f"e{j}": experts[j] for j in range(6)})
            if i >= warmup:
                timings.append(time.perf_counter() - t0)
            if i == 0:
                selected = [*experts, 128]
                _check(f"B2-9 {pattern}", y.numpy(), _bfp_ref(weights, x, selected))
        _stats(f"B2-9 {pattern}", timings, B2_BYTES)


def stage5(seed: int, stress_iters: int, mock: bool):
    print(
        f"[{now()}] Stage 5: MoE projection chain "
        f"(B1-6 -> shared GU-9 -> B2-9), {stress_iters} iterations",
        flush=True,
    )
    if mock:
        for i in range(100, stress_iters + 1, 100):
            print(
                f"[{now()}] stress {i} integer bit checks: "
                "B1 mismatches=0, shared mismatches=0, B2 mismatches=0",
                flush=True,
            )
            print(
                f"[{now()}] stress {i} real chain B2: mismatches=0/17920, "
                "mismatch_ratio=0 (limit=0.001), max_ulp=0 (limit=4)",
                flush=True,
            )
            print(f"[{now()}] {i}/{stress_iters} PASS", flush=True)
        return
    import aie.utils as aie
    from iron.common import AIEContext
    from kernels.gemv_bfp16.op import GEMV_BFP16
    from kernels.moe.op import MoEDownBfp16, MoEGateUp

    rng = np.random.default_rng(seed + 5)
    weights = _base_weights(seed, need_gu=True, need_shared=True, need_down=True)
    ctx = AIEContext()
    gu = MoEGateUp(slots=6, context=ctx)
    shared = GEMV_BFP16(K=B1_K, M=B1_M_PER_EXPERT, context=ctx)
    dn = MoEDownBfp16(context=ctx)
    for op in (gu, shared, dn):
        op.compile()
    run_gu, run_shared, run_dn = (
        gu.get_callable(),
        shared.get_callable(),
        dn.get_callable(),
    )
    y_gu = aie.full((6 * B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    y_sh = aie.full((B1_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    y_dn = aie.full((7 * B2_M_PER_EXPERT,), SENTINEL, dtype=bfloat16)
    w_gu, w_sh, w_dn = (
        aie.tensor(weights["w_gu"]),
        aie.tensor(weights["w_shared"]),
        aie.tensor(weights["w_down"]),
    )
    x_gu = aie.tensor(np.zeros(B1_K, dtype=bfloat16))
    x_dn = aie.tensor(np.zeros(7 * B2_K_PAD, dtype=bfloat16))
    timings = []
    for i in range(1, stress_iters + 1):
        experts = list(rng.choice(NUM_ROUTED_EXPERTS, 6, replace=False))
        scalars = {f"e{j}": experts[j] for j in range(6)}
        x = rng.integers(-4, 5, B1_K).astype(bfloat16)
        x_gu[:] = x
        x_gu.to("npu")
        t0 = time.perf_counter()
        run_gu(y_gu, w_gu, x_gu, **scalars)
        run_shared(y_sh, w_sh, x_gu)
        gate_up = np.concatenate(
            [y_gu.numpy().reshape(6, -1), y_sh.numpy().reshape(1, -1)]
        )
        down_x = moe_host_silu_glu(
            gate_up, gate_beta=4.0, value_beta=25.0
        ).astype(bfloat16)
        x_dn[:] = down_x.reshape(-1)
        x_dn.to("npu")
        run_dn(y_dn, w_dn, x_dn, **scalars)
        timings.append(time.perf_counter() - t0)
        if i % 100 == 0:
            _check(f"stress {i} B1", y_gu.numpy(), _q4_ref(weights, x, experts))
            _check(
                f"stress {i} shared",
                y_sh.numpy(),
                bfp16_gemv_bit_reference(weights["shared"], x)[1],
            )
            _check_tolerance(
                f"stress {i} real chain B2",
                y_dn.numpy(),
                _bfp_ref(weights, down_x, [*experts, 128]),
            )

            integer_down_x = rng.integers(
                -4, 5, (NUM_ACTIVE_TOTAL, B2_K_PAD)
            ).astype(np.float32)
            integer_down_x[:, B2_K_ORIG:] = 0
            integer_down_x = integer_down_x.astype(bfloat16)
            x_dn[:] = integer_down_x.reshape(-1)
            x_dn.to("npu")
            run_dn(y_dn, w_dn, x_dn, **scalars)
            _check(
                f"stress {i} integer B2",
                y_dn.numpy(),
                _bfp_ref(weights, integer_down_x, [*experts, 128]),
            )
            print(f"[{now()}] {i}/{stress_iters} PASS", flush=True)
    _stats(
        "MoE projection chain",
        timings,
        B1_BYTES + len(weights["w_shared"]) * 4 + B2_BYTES,
    )


def _run(target, args, timeout: int, name: str):
    proc = multiprocessing.Process(target=target, args=args)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        raise SystemExit(f"{name} timed out after {timeout}s")
    if proc.exitcode:
        raise SystemExit(f"{name} failed with exit code {proc.exitcode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--stress-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stages", nargs="+", default=list("12345"))
    a = parser.parse_args()
    stages = {
        "1": (stage1, (a.seed, a.mock)),
        "2": (stage2, (a.seed, a.warmup, a.iters, a.mock)),
        "3": (stage3, (a.seed, a.mock)),
        "4": (stage4, (a.seed, a.warmup, a.iters, a.mock)),
        "5": (stage5, (a.seed, a.stress_iters, a.mock)),
    }
    for key in a.stages:
        fn, args = stages[key]
        _run(fn, args, a.timeout, f"stage {key}")
    print(f"[{now()}] all requested stages passed", flush=True)


if __name__ == "__main__":
    main()
