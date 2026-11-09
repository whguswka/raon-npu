# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded per-round/per-call bit and latency A/B for unified shapes on NPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.gemv_unified.op import UnifiedGEMV
from kernels.gemv_unified.shapes import get_shape
from kernels.gemv_unified.stage2 import UnifiedMoEDown9, UnifiedMoEGateUp, pad_x
from kernels.moe.patch import TXN_OPC_DDR_PATCH, parse_txn_ops

try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    bfloat16 = np.float16

CANARY_WORD = np.uint16(0x5A3C)
CANARY_ELEMS = 256

# Recommended execution order starting from small odd-K shapes
RECOMMENDED_ORDER = [
    "f_b",        # K_blocks=1 (odd)
    "q_b",        # K_blocks=3 (odd)
    "k_b_v_b",    # K_blocks=2 (even)
    "moe_b2_9",   # B2: K_blocks=5 (odd, 7-slot down)
    "moe_b1_6",   # B1: K_blocks=10 (even, 6-expert gate/up)
    "dense_gu",   # K_blocks=10 (even)
]

SHAPE_ALIASES = {
    "b1": "moe_b1_6",
    "B1": "moe_b1_6",
    "b2": "moe_b2_9",
    "B2": "moe_b2_9",
}


def canonical_shape_name(name: str) -> str:
    return SHAPE_ALIASES.get(name, name)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _stats(path: Path) -> dict[str, int]:
    words = np.fromfile(path, dtype=np.uint32)
    ops = parse_txn_ops(words)
    return {
        "words": int(words.size),
        "txn_ops": len(ops),
        "ddr_patch": sum(opcode == TXN_OPC_DDR_PATCH for opcode, _pos, _size in ops),
    }


def _operator(name: str, plan: str, context):
    kwargs = {"context": context, "cols": 7, "rda_col": 7, "continuous_mem_w": True, "sequence_plan": plan}
    if name == "moe_b1_6":
        return UnifiedMoEGateUp(**kwargs)
    if name == "moe_b2_9":
        return UnifiedMoEDown9(**kwargs)
    return UnifiedGEMV.for_shape(name, **kwargs)


def _expert_sets(name: str) -> list[dict[str, int]]:
    """Return at least 3 expert ID sets: min, max, repetition across slots, and mixed."""
    if not name.startswith("moe_"):
        return [{}]
    # 4 patterns covering:
    # 1) Min expert IDs: [0, 1, 2, 3, 4, 5]
    # 2) Max expert IDs: [127, 126, 125, 124, 123, 122]
    # 3) Repeated across slots (slot repetition): [7, 7, 7, 7, 7, 7]
    # 4) Mixed min/max/repeated: [0, 127, 0, 127, 42, 42]
    patterns = [
        [0, 1, 2, 3, 4, 5],
        [127, 126, 125, 124, 123, 122],
        [7, 7, 7, 7, 7, 7],
        [0, 127, 0, 127, 42, 42],
    ]
    return [{f"e{i}": values[i] for i in range(6)} for values in patterns]


def resolve_weight_bundle(weights_arg: Path | str | None, shape_name: str) -> Path:
    """Resolve weight bundle file path. No zero-array fallback allowed."""
    if weights_arg is None:
        raise ValueError(f"Weight bundle is required for shape {shape_name} (specify --weights)")
    weights_path = Path(weights_arg)
    if weights_path.is_dir():
        candidates = [
            weights_path / f"{shape_name}.bin",
            weights_path / f"{shape_name}.npy",
        ]
        for cand in candidates:
            if cand.exists():
                return cand
        raise FileNotFoundError(
            f"No weight bundle found for {shape_name} in {weights_path} (checked {candidates})"
        )
    if not weights_path.exists():
        raise FileNotFoundError(f"Weight bundle does not exist: {weights_path}")
    return weights_path


def load_weight_words(bundle_path: Path, shape_name: str, shape) -> np.ndarray:
    """Load packed uint32 weight words from .bin or .npy file."""
    if str(bundle_path).endswith(".npy"):
        words = np.load(bundle_path)
        if words.dtype != np.uint32:
            raise ValueError(f"{bundle_path} must have dtype uint32, got {words.dtype}")
    else:
        words = np.fromfile(bundle_path, dtype=np.uint32)
    moe = shape_name.startswith("moe_")
    if not moe and words.size != shape.weight_words:
        raise SystemExit(f"bundle has {words.size} words, shape expects {shape.weight_words}")
    return words


def build_random_inputs(shape, seed: int = 1101, count: int = 2) -> list[np.ndarray]:
    """Generate nonzero random bfloat16 activation inputs padded to shape."""
    rng = np.random.default_rng(seed)
    xs = []
    for _ in range(count):
        x = rng.standard_normal(shape.x_slots * shape.K, dtype=np.float32).astype(bfloat16)
        xs.append(np.asarray(pad_x(x, shape), dtype=bfloat16))
    return xs


def execute_ab_loop(
    shape,
    plans: dict[str, dict],
    outputs: dict[str, object],
    w_bo: object,
    xs: list[object],
    sets: list[dict[str, int]],
    iterations: int,
    warmup: int = 5,
    canary_word: int = int(CANARY_WORD),
) -> tuple[int, dict[str, list[float]], list[str], dict[tuple[int, int], np.ndarray], list[tuple[int, int]]]:
    """Execute A/B loop testing BOTH plans on every input/expert combo with bit comparison."""
    combos = [(xi, si) for xi in range(len(xs)) for si in range(len(sets))]

    def call(p: str, xi: int, si: int) -> tuple[float, np.ndarray, bool]:
        t0 = time.perf_counter_ns()
        plans[p]["fn"](outputs[p], w_bo, xs[xi], **sets[si])
        dt = (time.perf_counter_ns() - t0) / 1e3
        got = np.asarray(outputs[p].numpy()).view(np.uint16)
        canary_ok = bool(np.all(got[shape.M:] == canary_word))
        return dt, got[: shape.M].copy(), canary_ok

    for p in plans:
        for i in range(warmup):
            call(p, 0, i % len(sets))

    times = {p: [] for p in plans}
    fails: list[str] = []
    ref: dict[tuple[int, int], np.ndarray] = {}
    done = 0

    for it in range(iterations):
        xi, si = combos[it % len(combos)]
        res: dict[str, np.ndarray] = {}
        order = ("per_round", "per_call") if it % 2 == 0 else ("per_call", "per_round")
        for p in order:
            dt, y, canary_ok = call(p, xi, si)
            times[p].append(dt)
            res[p] = y
            if not canary_ok:
                fails.append(f"canary overwritten plan={p} it={it}")
                break
            if not np.any(y):
                fails.append(f"all-zero output plan={p} x={xi} set={si}")
                break
        if fails:
            break
        if not np.array_equal(res["per_round"], res["per_call"]):
            diff = int(np.sum(res["per_round"] != res["per_call"]))
            fails.append(f"bit mismatch it={it} x={xi} set={si} elements={diff}")
            break
        if (xi, si) in ref and not np.array_equal(ref[(xi, si)], res["per_round"]):
            fails.append(f"per_round not repeatable it={it} x={xi} set={si}")
            break
        ref.setdefault((xi, si), res["per_round"])
        done = it + 1

    return done, times, fails, ref, combos


def _child(args: argparse.Namespace) -> int:
    import aie.utils as aie
    from iron.common import AIEContext

    shape_name = canonical_shape_name(args.shape)
    shape = get_shape(shape_name)
    moe = shape_name.startswith("moe_")
    context = AIEContext()

    t_compile = time.perf_counter()
    built = {}
    for plan in ("per_round", "per_call"):
        op = _operator(shape_name, plan, context)
        op.compile()
        xclbin = Path(op.xclbin_artifact.filename).resolve()
        insts_path = Path(op.insts_artifact.filename)
        stats = _stats(insts_path)
        if moe:
            # Same as NpuBackend._prepare_dynamic_kernel: load once, patch the instructions per call
            kernel = op.get_callable()
            runtime = aie.DefaultNPURuntime
            handle = runtime.load(kernel)

            def make_dynamic_fn(k, h, rt):
                def fn(*args_call, **scalars):
                    rt.run(h, list(args_call), dispatch_insts=k.instructions(**scalars))
                return fn

            callable_fn = make_dynamic_fn(kernel, handle, runtime)
        else:
            callable_fn = op.get_callable()

        built[plan] = {
            "op": op,
            "fn": callable_fn,
            "xclbin": xclbin,
            "xclbin_sha256": _sha256(xclbin),
            "insts": stats,
            "insts_name": insts_path.name,
        }
    compile_s = time.perf_counter() - t_compile

    if built["per_round"]["xclbin"] != built["per_call"]["xclbin"]:
        raise SystemExit("per_round and per_call resolve to different xclbins")
    if built["per_round"]["xclbin_sha256"] != built["per_call"]["xclbin_sha256"]:
        raise SystemExit("per_round and per_call resolve to different xclbin hashes")

    bundle_path = resolve_weight_bundle(args.weights, shape_name)
    weight_words = load_weight_words(bundle_path, shape_name, shape)
    w_bo = aie.tensor(weight_words)

    if args.input_npy:
        x_loaded = np.load(args.input_npy).astype(bfloat16)
        xs = [aie.tensor(np.asarray(pad_x(x_loaded, shape), dtype=bfloat16))]
    else:
        raw_xs = build_random_inputs(shape, seed=1101, count=2)
        xs = [aie.tensor(x) for x in raw_xs]

    canary_val = np.asarray([CANARY_WORD], dtype=np.uint16).view(bfloat16)[0]
    outputs = {
        plan: aie.full((shape.M + CANARY_ELEMS,), canary_val, dtype=bfloat16)
        for plan in built
    }

    sets = _expert_sets(shape_name)
    done, times, fails, ref, combos = execute_ab_loop(
        shape=shape,
        plans=built,
        outputs=outputs,
        w_bo=w_bo,
        xs=xs,
        sets=sets,
        iterations=args.iterations,
        warmup=args.warmup,
        canary_word=int(CANARY_WORD),
    )

    nonzero = {f"x{k[0]}s{k[1]}": int(np.count_nonzero(v)) for k, v in ref.items()}
    result = {
        "shape": shape_name,
        "bundle": str(bundle_path),
        "compile_s": round(compile_s, 2),
        "iterations": done,
        "combos": len(combos),
        "warmup": args.warmup,
        "xclbin": str(built["per_round"]["xclbin"]),
        "xclbin_sha256": built["per_round"]["xclbin_sha256"],
        "nonzero_outputs": nonzero,
        "fails": fails,
        "plans": {},
    }
    for plan, values in times.items():
        arr = np.asarray(values, dtype=np.float64)
        result["plans"][plan] = {
            "insts": built[plan]["insts_name"],
            "words": built[plan]["insts"]["words"],
            "txn_ops": built[plan]["insts"]["txn_ops"],
            "ddr_patch": built[plan]["insts"]["ddr_patch"],
            "calls": int(arr.size),
            "p50_us": round(float(np.percentile(arr, 50)), 1) if arr.size else None,
            "p95_us": round(float(np.percentile(arr, 95)), 1) if arr.size else None,
            "mean_us": round(float(arr.mean()), 1) if arr.size else None,
        }

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    for p, v in result["plans"].items():
        print(
            f"{p:9s} words {v['words']:6d} txn {v['txn_ops']:5d} patch {v['ddr_patch']:4d} "
            f"p50 {v['p50_us'] if v['p50_us'] is not None else 0.0:8.1f} "
            f"p95 {v['p95_us'] if v['p95_us'] is not None else 0.0:8.1f} us",
            flush=True,
        )
    if fails:
        for f in fails:
            print(f"FAIL: {f}", flush=True)
    print(f"RESULT: {'FAIL' if fails else 'PASS'}", flush=True)
    return 1 if fails else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--shape",
        default=None,
        help="Shape name or alias ('f_b', 'q_b', 'k_b_v_b', 'B2'/'moe_b2_9', 'B1'/'moe_b1_6', 'dense_gu', or 'all')",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Run all recommended shapes sequentially in separate child processes",
    )
    p.add_argument(
        "--weights",
        "--bundle",
        "--weights-bundle",
        "--weights-npy",
        dest="weights",
        type=Path,
        required=False,
        help="Weight bundle file or directory containing <shape>.bin or <shape>.npy files (required)",
    )
    p.add_argument(
        "--iterations",
        "--calls",
        dest="iterations",
        type=int,
        default=200,
        help="Total benchmark iterations (default: 200, min 100 for deadlock checking)",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations discarded from statistics (default: 5, min 5)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-shape timeout in seconds (default: 300.0)",
    )
    p.add_argument("--out-json", type=Path, help="Path to write JSON benchmark summary")
    p.add_argument("--input-npy", type=Path, help="Input activations file (default: random bf16)")
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p


def _run_single_shape(shape_name: str, args: argparse.Namespace, bundle_path: Path) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        "--shape",
        shape_name,
        "--weights",
        str(bundle_path),
        "--iterations",
        str(args.iterations),
        "--warmup",
        str(args.warmup),
    ]
    if args.input_npy:
        cmd.extend(["--input-npy", str(args.input_npy)])
    if args.out_json:
        cmd.extend(["--out-json", str(args.out_json)])

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    try:
        completed = subprocess.run(cmd, timeout=args.timeout, env=env, check=False)
        return completed.returncode
    except subprocess.TimeoutExpired:
        print(f"[{shape_name}] RESULT: FAIL (timeout {args.timeout:.1f}s)", flush=True)
        return 124


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.child:
        return _child(args)

    if not args.weights:
        print("Error: Weight bundle is required (specify --weights <path-or-dir>)", file=sys.stderr)
        return 1

    if args.iterations < 100:
        raise SystemExit("--iterations must be at least 100 for the lock/BD accumulation gate")
    if args.warmup < 5:
        raise SystemExit("--warmup must be at least 5")

    if args.all or args.shape in ("all", "ALL"):
        shapes_to_run = RECOMMENDED_ORDER
    elif args.shape:
        shapes_to_run = [canonical_shape_name(args.shape)]
    else:
        print("Error: Specify --shape <name> (e.g. f_b, q_b, k_b_v_b, B2, B1, dense_gu) or --all", file=sys.stderr)
        return 1

    overall_failed = False
    for shape_name in shapes_to_run:
        print(f"=== Running A/B benchmark for shape: {shape_name} ===", flush=True)
        try:
            bundle_path = resolve_weight_bundle(args.weights, shape_name)
        except (ValueError, FileNotFoundError) as exc:
            print(f"Error resolving weight bundle for {shape_name}: {exc}", file=sys.stderr)
            overall_failed = True
            break
        rc = _run_single_shape(shape_name, args, bundle_path)
        if rc != 0:
            overall_failed = True
            print(f"=== Shape {shape_name} FAILED (rc={rc}) ===", flush=True)
            break

    print(f"RESULT: {'FAIL' if overall_failed else 'PASS'}", flush=True)
    return 1 if overall_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
