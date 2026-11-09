# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Stage-1 unified 9-bit bfp16ebs8 GEMV benchmark and verification harness.

Stages (run incrementally with early-abort, timeout, and flush=True):
  1. `smoke`  : Single shape (`o_proj` 5120->2560) bit-exact check + NPU latency.
  2. `alt2`   : Two-shape alternation (`o_proj` <-> `f_b`) without xclbin reload.
  3. `all10`  : All 10 shapes bit-exact check + 1,000-iter p50 latency (+ optional
                comparison against per-shape static `GEMV_BFP16`).
  4. `stress` : 10,000 random shape-alternating dispatches in one process, checking
                bit-exactness against `bfp16_gemv_bit_reference` every 1,000 calls.
  5. `full`   : Runs `smoke` -> `alt2` -> `all10` -> `stress` in sequence.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IRON_DIR = os.environ.get("RAON_IRON_DIR")
if IRON_DIR:
    iron_path = Path(IRON_DIR).expanduser()
    if str(iron_path) not in sys.path:
        sys.path.insert(0, str(iron_path))

from kernels.gemv_bfp16.reference import (  # noqa: E402
    bf16_ulp_diff,
    bfp16_gemv_bit_reference,
)
from kernels.gemv_unified.op import (  # noqa: E402
    SharedXclbinCallable,
    UnifiedGEMV,
    assert_shared_xclbin,
)
from kernels.gemv_unified.shapes import (  # noqa: E402
    COLS,
    MAX_K,
    SHAPES,
    UnifiedShape,
    get_shape,
    selected_shapes,
)
from quant.bfp16 import pack_bfp16_gemv  # noqa: E402
from quant.q4nx import round_to_bf16_rne, to_bf16_u16  # noqa: E402

SENTINEL = 99.0


def log(msg: str) -> None:
    print(msg, flush=True)


def bf16_u16_to_float32(u16: np.ndarray) -> np.ndarray:
    """Convert uint16 bfloat16 bit patterns to float32."""
    return (np.asarray(u16, dtype=np.uint32) << 16).view(np.float32)


def _extract_npu_time_ns(ret: Any) -> int | None:
    """Extract hardware NPU execution time in nanoseconds from callable return value."""
    candidate = ret[1] if isinstance(ret, tuple) and len(ret) >= 2 else ret
    return getattr(candidate, "npu_time", None)


def make_shape_test_case(
    K: int,
    M: int,
    *,
    seed: int,
    ref_chunk_m: int = 2048,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build deterministic packed 9-bit weights, bf16 input, and bit-exact reference.

    Uses the integer contract sample (-4..4) matching `bench/gemv_bfp16/bench.py`
    so every dot product is bit-exact against `bfp16_gemv_bit_reference`.
    For very large M (e.g. `lm_head` M=260,096), processes rows in multiples of 128
    to bound peak host memory.
    """
    rng = np.random.default_rng(seed)
    x_f32 = round_to_bf16_rne(rng.integers(-4, 5, K).astype(np.float32))
    x_u16 = to_bf16_u16(x_f32)

    w_words = np.empty((M * K * 9) // 32, dtype=np.uint32)
    y_ref_u16 = np.empty(M, dtype=np.uint16)

    words_per_row = (K * 9) // 32
    step = max(128, (ref_chunk_m // 128) * 128)
    for m0 in range(0, M, step):
        m1 = min(M, m0 + step)
        rows = m1 - m0
        w_chunk_f32 = rng.integers(-4, 5, (rows, K)).astype(np.float32)
        packed_bytes = pack_bfp16_gemv(w_chunk_f32, rows=4, rounding="conv_even")
        w_words[m0 * words_per_row : m1 * words_per_row] = np.frombuffer(
            packed_bytes, dtype=np.uint32
        )
        _, ref_u16 = bfp16_gemv_bit_reference(
            w_chunk_f32, x_f32, rounding="conv_even"
        )
        y_ref_u16[m0:m1] = ref_u16

    return w_words, x_u16, y_ref_u16


def compare_bf16_bits(got_u16: np.ndarray, ref_u16: np.ndarray) -> dict[str, Any]:
    got_u16 = np.asarray(got_u16, dtype=np.uint16)
    ref_u16 = np.asarray(ref_u16, dtype=np.uint16)
    diff_mask = got_u16 != ref_u16
    mismatches = int(np.count_nonzero(diff_mask))
    ulp = bf16_ulp_diff(got_u16, ref_u16)
    got_f32 = bf16_u16_to_float32(got_u16)
    ref_f32 = bf16_u16_to_float32(ref_u16)
    abs_err = np.abs(got_f32 - ref_f32)
    first_bad = None
    if mismatches:
        idx = int(np.flatnonzero(diff_mask)[0])
        first_bad = {
            "index": idx,
            "got_hex": f"0x{int(got_u16[idx]):04x}",
            "ref_hex": f"0x{int(ref_u16[idx]):04x}",
            "got_f32": float(got_f32[idx]),
            "ref_f32": float(ref_f32[idx]),
        }
    return {
        "mismatches": mismatches,
        "total": int(ref_u16.size),
        "max_ulp": int(ulp.max(initial=0)),
        "max_abs_err": float(abs_err.max(initial=0.0)),
        "first_mismatch": first_bad,
    }


class ShapeRunner:
    """Holds preallocated NPU BOs, reference output, and callable for one shape."""

    def __init__(
        self,
        shape: UnifiedShape,
        *,
        cols: int,
        context: Any,
        aie_utils: Any,
        bfloat16_dtype: Any,
        seed: int,
    ) -> None:
        self.shape = shape
        self.op = UnifiedGEMV.for_shape(shape, cols=cols, context=context)
        self.op.set_up_artifacts()
        self.op.compile()
        self.fn: SharedXclbinCallable = self.op.get_callable()

        w_words, x_u16, y_ref_u16 = make_shape_test_case(shape.K, shape.M, seed=seed)
        self.y_ref_u16 = y_ref_u16

        # Allocate NPU tensors ONCE outside timing loops
        # The shared x ObjectFifo has a fixed MAX_K-sized receiver BD.  A short
        # transfer leaves that BD waiting forever, so zero-pad every host X BO.
        x_padded = np.zeros(MAX_K, dtype=np.uint16)
        x_padded[: shape.K] = x_u16
        self.X_bo = aie_utils.tensor(x_padded.view(bfloat16_dtype))
        self.W_bo = aie_utils.tensor(w_words)
        self.Y_bo = aie_utils.full((shape.M,), SENTINEL, dtype=bfloat16_dtype)

    def invoke(self) -> Any:
        return self.fn(self.Y_bo, self.W_bo, self.X_bo)

    def read_output_u16(self) -> np.ndarray:
        return np.asarray(self.Y_bo.numpy()).view(np.uint16).copy()

    def verify(self) -> dict[str, Any]:
        got_u16 = self.read_output_u16()
        return compare_bf16_bits(got_u16, self.y_ref_u16)

    def time_dispatches(self, *, warmup: int, iters: int) -> dict[str, Any]:
        for _ in range(warmup):
            self.invoke()
        npu_us = np.empty(iters, dtype=np.float64)
        e2e_us = np.empty(iters, dtype=np.float64)
        src = "e2e"
        for i in range(iters):
            t0 = time.perf_counter_ns()
            ret = self.invoke()
            t1 = time.perf_counter_ns()
            wall_us = (t1 - t0) / 1e3
            e2e_us[i] = wall_us
            hw_ns = _extract_npu_time_ns(ret)
            if hw_ns is not None:
                npu_us[i] = hw_ns / 1e3
                src = "npu"
            else:
                npu_us[i] = wall_us
        return {
            "source": src,
            "mean_us": float(npu_us.mean()),
            "min_us": float(npu_us.min()),
            "p50_us": float(np.percentile(npu_us, 50)),
            "p90_us": float(np.percentile(npu_us, 90)),
            "p99_us": float(np.percentile(npu_us, 99)),
            "max_us": float(npu_us.max()),
            "e2e_p50_us": float(np.percentile(e2e_us, 50)),
        }


def build_runners(
    shapes: tuple[UnifiedShape, ...],
    *,
    cols: int,
    seed_base: int,
) -> tuple[dict[str, ShapeRunner], dict[str, Any]]:
    from ml_dtypes import bfloat16
    import aie.utils as aie_utils
    from iron.common import AIEContext

    ctx = AIEContext()
    runners: dict[str, ShapeRunner] = {}
    for idx, s in enumerate(shapes):
        t0 = time.perf_counter()
        log(f"[init] preparing shape {s.name} (K={s.K}, M={s.M}, chunks={s.chunks})...")
        runner = ShapeRunner(
            s,
            cols=cols,
            context=ctx,
            aie_utils=aie_utils,
            bfloat16_dtype=bfloat16,
            seed=seed_base + idx * 97,
        )
        runners[s.name] = runner
        dt = time.perf_counter() - t0
        log(
            f"[init] ready {s.name} in {dt:.2f}s | "
            f"xclbin={runner.fn.xclbin_path.name} (mtime_ns={runner.fn.xclbin_mtime_ns}) | "
            f"insts={runner.fn.insts_path.name}"
        )

    shared_sig = assert_shared_xclbin([r.fn for r in runners.values()])
    log(
        f"[assert] shared xclbin verified across {len(runners)} shape(s): "
        f"path={shared_sig['xclbin_path']} mtime_ns={shared_sig['xclbin_mtime_ns']} "
        f"kernel={shared_sig['kernel_name']}"
    )
    return runners, {
        "xclbin_path": str(shared_sig["xclbin_path"]),
        "xclbin_mtime_ns": int(shared_sig["xclbin_mtime_ns"]),
        "kernel_name": str(shared_sig["kernel_name"]),
        "abi_signature": [list(x) for x in shared_sig["abi_signature"]],
    }


def measure_static_baseline(
    shape: UnifiedShape,
    *,
    cols: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    from ml_dtypes import bfloat16
    import aie.utils as aie_utils
    from iron.common import AIEContext
    from kernels.gemv_bfp16.op import GEMV_BFP16

    ctx = AIEContext()
    op = GEMV_BFP16(K=shape.K, M=shape.M, cols=cols, context=ctx)
    op.set_up_artifacts()
    op.compile()
    fn = op.get_callable()

    X_bo = aie_utils.zeros((shape.K,), dtype=bfloat16)
    W_bo = aie_utils.zeros((shape.weight_words,), dtype=np.uint32)
    Y_bo = aie_utils.zeros((shape.M,), dtype=bfloat16)

    for _ in range(warmup):
        fn(Y_bo, W_bo, X_bo)
    times_us = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        t0 = time.perf_counter_ns()
        ret = fn(Y_bo, W_bo, X_bo)
        t1 = time.perf_counter_ns()
        hw_ns = _extract_npu_time_ns(ret)
        times_us[i] = (hw_ns / 1e3) if hw_ns is not None else ((t1 - t0) / 1e3)
    return {
        "p50_us": float(np.percentile(times_us, 50)),
        "mean_us": float(times_us.mean()),
        "p90_us": float(np.percentile(times_us, 90)),
    }


def run_smoke_stage(
    runners: dict[str, ShapeRunner],
    *,
    warmup: int,
    iters: int,
    fail_fast: bool,
) -> dict[str, Any]:
    runner = runners["o_proj"]
    log("=== [Stage 1: smoke] single shape o_proj (5120 -> 2560) ===")
    runner.invoke()
    acc = runner.verify()
    timing = runner.time_dispatches(warmup=warmup, iters=iters)
    acc_after = runner.verify()
    passed = acc["mismatches"] == 0 and acc_after["mismatches"] == 0
    log(
        f"[smoke] o_proj mismatches={acc['mismatches']}/{acc['total']} "
        f"after_timing={acc_after['mismatches']}/{acc_after['total']} "
        f"max_ulp={max(acc['max_ulp'], acc_after['max_ulp'])} "
        f"p50={timing['p50_us']:.1f} us (mean={timing['mean_us']:.1f} us, p90={timing['p90_us']:.1f} us)"
    )
    if fail_fast and not passed:
        raise RuntimeError(f"[smoke] o_proj bit-exact verification failed: {acc}")
    return {
        "shape": "o_proj",
        "passed": passed,
        "accuracy_initial": acc,
        "accuracy_after_timing": acc_after,
        "timing": timing,
    }


def run_alt2_stage(
    runners: dict[str, ShapeRunner],
    *,
    pairs: int,
    fail_fast: bool,
) -> dict[str, Any]:
    r_o = runners["o_proj"]
    r_f = runners["f_b"]
    assert_shared_xclbin([r_o.fn, r_f.fn])
    log(
        f"=== [Stage 2: alt2] alternating o_proj (5120->2560, rounds=3/2) "
        f"<-> f_b (256->5120, rounds=5) for {pairs} pairs ==="
    )

    t_o_us = np.empty(pairs, dtype=np.float64)
    t_f_us = np.empty(pairs, dtype=np.float64)
    for i in range(pairs):
        t0 = time.perf_counter_ns()
        ret_o = r_o.invoke()
        t1 = time.perf_counter_ns()
        ret_f = r_f.invoke()
        t2 = time.perf_counter_ns()
        hw_o = _extract_npu_time_ns(ret_o)
        hw_f = _extract_npu_time_ns(ret_f)
        t_o_us[i] = (hw_o / 1e3) if hw_o is not None else ((t1 - t0) / 1e3)
        t_f_us[i] = (hw_f / 1e3) if hw_f is not None else ((t2 - t1) / 1e3)

    acc_o = r_o.verify()
    acc_f = r_f.verify()
    passed = acc_o["mismatches"] == 0 and acc_f["mismatches"] == 0
    res = {
        "pairs": pairs,
        "passed": passed,
        "o_proj": {
            "accuracy": acc_o,
            "p50_us": float(np.percentile(t_o_us, 50)),
            "mean_us": float(t_o_us.mean()),
            "p90_us": float(np.percentile(t_o_us, 90)),
        },
        "f_b": {
            "accuracy": acc_f,
            "p50_us": float(np.percentile(t_f_us, 50)),
            "mean_us": float(t_f_us.mean()),
            "p90_us": float(np.percentile(t_f_us, 90)),
        },
    }
    log(
        f"[alt2] o_proj: mismatches={acc_o['mismatches']} p50={res['o_proj']['p50_us']:.1f} us | "
        f"f_b: mismatches={acc_f['mismatches']} p50={res['f_b']['p50_us']:.1f} us"
    )
    if fail_fast and not passed:
        raise RuntimeError(f"[alt2] alternating verification failed: {res}")
    return res


def run_all10_stage(
    runners: dict[str, ShapeRunner],
    *,
    shapes: tuple[UnifiedShape, ...],
    cols: int,
    warmup: int,
    iters: int,
    compare_static: bool,
    fail_fast: bool,
) -> dict[str, Any]:
    log(f"=== [Stage 3: all10] {len(shapes)} shapes bit-exact & {iters}-iter p50 ===")
    rows: list[dict[str, Any]] = []
    all_passed = True

    for s in shapes:
        runner = runners[s.name]
        shape_iters = min(iters, 50) if s.M >= 100000 else iters
        shape_warmup = min(warmup, 5) if s.M >= 100000 else warmup

        runner.invoke()
        acc = runner.verify()
        timing = runner.time_dispatches(warmup=shape_warmup, iters=shape_iters)
        acc_after = runner.verify()

        static_timing = None
        delta_pct = None
        gate_5pct = True
        if compare_static:
            static_timing = measure_static_baseline(
                s, cols=cols, warmup=shape_warmup, iters=shape_iters
            )
            delta_pct = (
                (timing["p50_us"] - static_timing["p50_us"])
                / max(static_timing["p50_us"], 1e-9)
                * 100.0
            )
            gate_5pct = delta_pct <= 5.0

        passed = (
            acc["mismatches"] == 0
            and acc_after["mismatches"] == 0
            and gate_5pct
        )
        all_passed = all_passed and passed
        row = {
            "name": s.name,
            "K": s.K,
            "M": s.M,
            "k_blocks": s.k_blocks,
            "chunks": s.chunks,
            "rounds_by_col": list(s.rounds_by_col()),
            "iters": shape_iters,
            "accuracy": acc_after,
            "timing": timing,
            "static_timing": static_timing,
            "delta_p50_pct": delta_pct,
            "passed": passed,
        }
        rows.append(row)
        delta_str = f" delta={delta_pct:+.2f}%" if delta_pct is not None else ""
        log(
            f"[all10] {s.name:10s} (K={s.K:5d}, M={s.M:6d}): "
            f"mismatches={acc_after['mismatches']}/{acc_after['total']} "
            f"max_ulp={acc_after['max_ulp']} "
            f"p50={timing['p50_us']:.1f} us (p90={timing['p90_us']:.1f} us){delta_str}"
        )
        if fail_fast and not passed:
            raise RuntimeError(f"[all10] shape {s.name} failed gate: {row}")

    return {"passed": all_passed, "shapes": rows}


def run_stress_stage(
    runners: dict[str, ShapeRunner],
    *,
    stress_iters: int,
    check_every: int,
    include_lm_head: bool,
    seed: int,
    fail_fast: bool,
) -> dict[str, Any]:
    names = [
        name
        for name, r in runners.items()
        if include_lm_head or r.shape.M < 100000
    ]
    if not names:
        names = list(runners.keys())

    assert_shared_xclbin([runners[n].fn for n in names])
    log(
        f"=== [Stage 4: stress] {stress_iters} random dispatches across {len(names)} shapes "
        f"(bit-exact check every {check_every}) ==="
    )

    rng = np.random.default_rng(seed)
    checkpoints: list[dict[str, Any]] = []
    t_start = time.perf_counter()

    for step in range(1, stress_iters + 1):
        chosen = names[int(rng.integers(0, len(names)))]
        runner = runners[chosen]
        runner.invoke()

        if step % check_every == 0 or step == stress_iters:
            bad_shapes: dict[str, Any] = {}
            for n in names:
                runners[n].invoke()
                acc = runners[n].verify()
                if acc["mismatches"] != 0:
                    bad_shapes[n] = acc
            elapsed = time.perf_counter() - t_start
            ckpt = {
                "step": step,
                "elapsed_s": elapsed,
                "last_shape": chosen,
                "failed_shapes": bad_shapes,
                "passed": len(bad_shapes) == 0,
            }
            checkpoints.append(ckpt)
            log(
                f"[stress] step {step}/{stress_iters} ({elapsed:.1f}s) "
                f"last={chosen} failed_shapes={len(bad_shapes)}"
            )
            if fail_fast and bad_shapes:
                raise RuntimeError(f"[stress] bit-exact failure at step {step}: {bad_shapes}")

    return {
        "stress_iters": stress_iters,
        "check_every": check_every,
        "participating_shapes": names,
        "passed": all(c["passed"] for c in checkpoints),
        "checkpoints": checkpoints,
    }


def compile_all_artifacts(cols: int = COLS) -> int:
    """Compile the shared xclbin and all stage-1 streams without NPU execution."""
    from iron.common import AIEContext

    ctx = AIEContext()
    for shape in SHAPES:
        op = UnifiedGEMV.for_shape(shape, cols=cols, context=ctx)
        op.compile()
        log(
            f"[compile-only] {shape.name}: "
            f"xclbin={op.xclbin_artifact.filename} "
            f"insts={op.insts_artifact.filename}"
        )
    log(f"[compile-only] PASS: 1 shared xclbin, {len(SHAPES)} instruction streams")
    return 0


def run_in_current_process(args: argparse.Namespace) -> int:
    if args.compile_only:
        return compile_all_artifacts(args.cols)

    stage = args.stage
    if stage == "smoke":
        req_shapes = (get_shape("o_proj"),)
    elif stage == "alt2":
        req_shapes = (get_shape("o_proj"), get_shape("f_b"))
    else:
        req_shapes = selected_shapes(args.shapes)

    runners, shared_info = build_runners(
        req_shapes,
        cols=args.cols,
        seed_base=args.seed,
    )

    report: dict[str, Any] = {
        "stage": stage,
        "shared_xclbin": shared_info,
        "results": {},
    }

    if stage in ("smoke", "full"):
        report["results"]["smoke"] = run_smoke_stage(
            runners,
            warmup=args.warmup,
            iters=args.smoke_iters,
            fail_fast=args.fail_fast,
        )
    if stage in ("alt2", "full"):
        report["results"]["alt2"] = run_alt2_stage(
            runners,
            pairs=args.alt_pairs,
            fail_fast=args.fail_fast,
        )
    if stage in ("all10", "full"):
        report["results"]["all10"] = run_all10_stage(
            runners,
            shapes=req_shapes,
            cols=args.cols,
            warmup=args.warmup,
            iters=args.iters,
            compare_static=args.compare_static,
            fail_fast=args.fail_fast,
        )
    if stage in ("stress", "full"):
        report["results"]["stress"] = run_stress_stage(
            runners,
            stress_iters=args.stress_iters,
            check_every=args.check_every,
            include_lm_head=args.stress_include_lm_head,
            seed=args.seed + 999,
            fail_fast=args.fail_fast,
        )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"[done] wrote JSON report to {out_path}")

    all_ok = all(section.get("passed", False) for section in report["results"].values())
    log(f"[summary] stage={stage} passed={all_ok}")
    return 0 if all_ok else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        choices=("smoke", "alt2", "all10", "stress", "full"),
        default="smoke",
        help="Execution stage (run incrementally: smoke -> alt2 -> all10 -> stress).",
    )
    p.add_argument(
        "--shapes",
        nargs="*",
        default=None,
        help="Optional subset of shape names for all10/stress.",
    )
    p.add_argument("--cols", type=int, default=COLS)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--smoke-iters", type=int, default=100)
    p.add_argument("--alt-pairs", type=int, default=200)
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--stress-iters", type=int, default=10000)
    p.add_argument("--check-every", type=int, default=1000)
    p.add_argument(
        "--stress-include-lm-head",
        action="store_true",
        help="Include lm_head (M=260,096) in the 10,000-iter random alternation pool.",
    )
    p.add_argument(
        "--compare-static",
        action="store_true",
        help="Also measure per-shape static GEMV_BFP16 p50 and check <= +5%% gate.",
    )
    p.add_argument("--seed", type=int, default=20261028)
    p.add_argument(
        "--no-fail-fast",
        dest="fail_fast",
        action="store_false",
        default=True,
        help="Continue running remaining shapes even if one fails.",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=0.0,
        help="If > 0, run the benchmark in a child process with this timeout (seconds).",
    )
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile the shared xclbin and all 10 shape instruction streams without NPU execution.",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.timeout > 0 and not args.child:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child"] + (
            argv if argv is not None else sys.argv[1:]
        )
        log(f"[watchdog] spawning child with timeout={args.timeout:.1f}s: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.run(cmd, timeout=args.timeout, env=env, check=False)
            return int(proc.returncode)
        except subprocess.TimeoutExpired:
            log(f"[watchdog] ERROR: stage={args.stage} timed out after {args.timeout:.1f}s")
            return 124
    return run_in_current_process(args)


if __name__ == "__main__":
    raise SystemExit(main())
