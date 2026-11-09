# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Laptop runner for unified q4nx, B1-6, B2-9 and mixed-cycle gates.

The short-DMA gate always runs first with its own phased watchdogs. All operators are constructed
from ``kernels.gemv_unified`` and the runner refuses to start if their resolved
xclbin path or mtime differs.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.moe.moe_bench import SELECTION_PATTERNS, get_selection_indices
from bench.moe_d.moe_d_bench import (
    SENTINEL,
    _base_weights,
    _bfp_ref,
    _check,
    _q4_ref,
)
from kernels.gemv.reference import gemv_bit_reference
from kernels.gemv_unified import (
    ALL_SHAPES,
    MAX_K,
    UnifiedGEMV,
    UnifiedMoEDown9,
    UnifiedMoEGateUp,
    assert_one_xclbin,
    get_shape,
    pad_x,
)
from kernels.moe.pack import (
    B1_K,
    B1_M_PER_EXPERT,
    B2_K_ORIG,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    NUM_ACTIVE_TOTAL,
)
from quant.q4nx import pack_gemv

try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    bfloat16 = np.uint16


def log(message: str) -> None:
    print(message, flush=True)


def _dense_case(seed: int):
    s = get_shape("dense_gu")
    rng = np.random.default_rng(seed)
    x = rng.integers(-4, 5, s.K).astype(np.float32)
    rows = np.where(np.arange(s.M) & 1, 1.0, -1.0).astype(np.float32)
    codes = np.zeros((s.M, s.K), np.uint8)
    scales = np.zeros((s.M, s.K // 32), np.float32)
    mins = np.broadcast_to(rows[:, None], scales.shape)
    packed = np.frombuffer(pack_gemv(codes, scales, mins, rows=4), np.uint32).copy()
    ref = gemv_bit_reference(codes, scales, mins, x)[1]
    return s, packed, x.astype(bfloat16), ref


def _runtime():
    import aie.utils as aie
    from iron.common import AIEContext

    return aie, AIEContext


def _build(context, names):
    ops = {}
    for name in names:
        if name == "moe_b1_6":
            op = UnifiedMoEGateUp(context=context)
        elif name == "moe_b2_9":
            op = UnifiedMoEDown9(context=context)
        else:
            op = UnifiedGEMV.for_shape(name, context=context)
        op.compile()
        ops[name] = op.get_callable()
    path, mtime = assert_one_xclbin(ops.values())
    log(f"[identity] {len(ops)} callables xclbin={path} mtime_ns={mtime}")
    return ops


def compile_all_artifacts() -> int:
    """Compile the experiment plus every unified shape without loading/running NPU."""
    exp_cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "exp_short_dma.py"),
        "--compile-only",
    ]
    subprocess.run(exp_cmd, check=True)

    _aie, Context = _runtime()
    context = Context()
    for shape in ALL_SHAPES:
        op = UnifiedGEMV.for_shape(shape, context=context)
        op.compile()
        log(
            f"[compile-only] {shape.name}: "
            f"xclbin={op.xclbin_artifact.filename} "
            f"insts={op.insts_artifact.filename}"
        )
    log(
        f"[compile-only] PASS: short-DMA artifacts + 1 shared GEMV xclbin + "
        f"{len(ALL_SHAPES)} instruction streams"
    )
    return 0


def q4_smoke(seed: int, mock: bool) -> None:
    log("[q4-smoke] dense_gu 2560->24064")
    if mock:
        return
    aie, Context = _runtime()
    s, w, x, ref = _dense_case(seed)
    run = _build(Context(), ["dense_gu"])["dense_gu"]
    y = aie.full((s.M,), SENTINEL, dtype=bfloat16)
    run(y, aie.tensor(w), aie.tensor(pad_x(x, s)))
    _check("unified dense_gu", y.numpy(), ref)


def moe_patterns(seed: int, which: str, mock: bool) -> None:
    log(f"[{which}] five expert-selection patterns")
    if mock:
        for name in SELECTION_PATTERNS:
            log(f"[{which}] {name}: PASS")
        return
    aie, Context = _runtime()
    rng = np.random.default_rng(seed)
    b1 = which == "b1"
    shape_name = "moe_b1_6" if b1 else "moe_b2_9"
    run = _build(Context(), [shape_name])[shape_name]
    weights = _base_weights(
        seed, need_gu=b1, need_shared=False, need_down=not b1
    )
    wb = aie.tensor(weights["w_gu" if b1 else "w_down"])
    y_size = 6 * B1_M_PER_EXPERT if b1 else 7 * B2_M_PER_EXPERT
    y = aie.full((y_size,), SENTINEL, dtype=bfloat16)
    for pattern in SELECTION_PATTERNS:
        experts = get_selection_indices(pattern, 0, None, rng)
        scalars = {f"e{i}": experts[i] for i in range(6)}
        if b1:
            x = rng.integers(-4, 5, B1_K).astype(bfloat16)
            ref = _q4_ref(weights, x, experts)
        else:
            x = rng.integers(-4, 5, (7, B2_K_PAD)).astype(np.float32)
            x[:, B2_K_ORIG:] = 0
            x = x.astype(bfloat16)
            ref = _bfp_ref(weights, x, [*experts, 128])
        y[:] = SENTINEL
        y.to("npu")
        run(y, wb, aie.tensor(pad_x(x, shape_name)), **scalars)
        _check(f"unified {which} {pattern}", y.numpy(), ref)


def mixed_cycle(seed: int, iterations: int, mock: bool) -> None:
    names = ["rda_in", "dense_gu", "moe_b1_6", "shared_gu", "moe_b2_9", "o_proj"]
    log(f"[mixed] {' -> '.join(names)}, iterations={iterations}")
    if mock:
        log(f"[mixed] {iterations}/{iterations} PASS")
        return
    aie, Context = _runtime()
    rng = np.random.default_rng(seed)
    runs = _build(Context(), names)
    weights = _base_weights(seed, need_gu=True, need_shared=True, need_down=True)

    # Zero-weight BOs are sufficient for the three ordinary projections here;
    # their bit gates ran in stage 1.  This stage targets lock/command-BO lifetime.
    ordinary = {}
    for name in ("rda_in", "shared_gu", "o_proj"):
        s = get_shape(name)
        ordinary[name] = (
            aie.zeros((s.M,), dtype=bfloat16),
            aie.zeros((s.weight_words,), dtype=np.uint32),
            aie.zeros((MAX_K,), dtype=bfloat16),
        )
    _, dense_w, dense_x, _ = _dense_case(seed + 9)
    dense_y = aie.zeros((get_shape("dense_gu").M,), dtype=bfloat16)
    dense_x_bo = aie.tensor(pad_x(dense_x, "dense_gu"))
    b1_y = aie.zeros((6 * B1_M_PER_EXPERT,), dtype=bfloat16)
    b2_y = aie.zeros((7 * B2_M_PER_EXPERT,), dtype=bfloat16)
    b1_w = aie.tensor(weights["w_gu"])
    b2_w = aie.tensor(weights["w_down"])
    dense_w_bo = aie.tensor(dense_w)
    b1_x_bo = aie.zeros((MAX_K,), dtype=bfloat16)
    b2_x_bo = aie.zeros((NUM_ACTIVE_TOTAL * MAX_K,), dtype=bfloat16)

    t0 = time.perf_counter()
    for step in range(1, iterations + 1):
        experts = list(rng.choice(128, 6, replace=False))
        scalars = {f"e{i}": experts[i] for i in range(6)}
        for name in ("rda_in",):
            runs[name](*ordinary[name])
        runs["dense_gu"](dense_y, dense_w_bo, dense_x_bo)
        b1_x = rng.integers(-4, 5, B1_K).astype(bfloat16)
        b1_x_bo[:] = pad_x(b1_x, "moe_b1_6")
        b1_x_bo.to("npu")
        runs["moe_b1_6"](b1_y, b1_w, b1_x_bo, **scalars)
        runs["shared_gu"](*ordinary["shared_gu"])
        b2_x = np.zeros((NUM_ACTIVE_TOTAL, B2_K_PAD), dtype=bfloat16)
        b2_x_bo[:] = pad_x(b2_x, "moe_b2_9")
        b2_x_bo.to("npu")
        runs["moe_b2_9"](b2_y, b2_w, b2_x_bo, **scalars)
        runs["o_proj"](*ordinary["o_proj"])
        if step % 1000 == 0 or step == iterations:
            log(f"[mixed] {step}/{iterations} elapsed={time.perf_counter() - t0:.1f}s PASS")


def _run_child(stage: str, args) -> int:
    if stage == "q4":
        q4_smoke(args.seed, args.mock)
    elif stage == "b1":
        moe_patterns(args.seed, "b1", args.mock)
    elif stage == "b2":
        moe_patterns(args.seed, "b2", args.mock)
    else:
        mixed_cycle(args.seed, args.iterations, args.mock)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("q4", "b1", "b2", "mixed", "all"), default="q4")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--timeout", type=float, default=240.0)
    p.add_argument("--exp-timeout", type=float, default=2400.0)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--mock", action="store_true")
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile short-DMA and every unified shape artifact without NPU execution.",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.compile_only:
        if args.mock or args.child:
            p.error("--compile-only cannot be combined with --mock or --child")
        return compile_all_artifacts()
    if args.child:
        return _run_child(args.stage, args)
    exp_cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve().parent / "exp_short_dma.py"),
        "--iters",
        str(args.iterations),
    ]
    if args.mock:
        exp_cmd.append("--simulate-only")
    try:
        subprocess.run(exp_cmd, check=True, timeout=args.exp_timeout)
    except subprocess.TimeoutExpired:
        log(f"[exp-short-dma] TIMEOUT after {args.exp_timeout:.1f}s")
        return 124
    except subprocess.CalledProcessError as exc:
        log(f"[exp-short-dma] FAIL rc={exc.returncode}; stage-2 tests not started")
        return int(exc.returncode)
    stages = ("q4", "b1", "b2", "mixed") if args.stage == "all" else (args.stage,)
    for stage in stages:
        cmd = [
            sys.executable, "-u", str(Path(__file__).resolve()), "--child",
            "--stage", stage, "--iterations", str(args.iterations),
            "--seed", str(args.seed),
        ]
        if args.mock:
            cmd.append("--mock")
        try:
            subprocess.run(cmd, check=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            log(f"[{stage}] TIMEOUT after {args.timeout:.1f}s")
            return 124
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
