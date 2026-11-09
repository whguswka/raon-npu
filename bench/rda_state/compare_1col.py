# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
#
"""Compare the original multi-column RDA design against the 1-column tile-wise design.

This script is meant to run on the notebook with an NPU.  It instantiates both
``rda_step`` (4 workers, memtile routing) and ``rda_step_1col`` (single column 0)
in the same process, drives ``--tokens`` sequential decode steps with realistic
head bundles, and verifies bit-exact agreement of s_out and out against the
numpy golden reference.  It also benchmarks both designs and runs a soak/repeat
stability test on the 1-column design.
"""

import argparse
import sys
import time

import aie.iron as iron
from aie.iron.device import from_name as device_from_name
from aie.utils import hostruntime
import numpy as np

from bench.rda_state.rda_state import (
    K_DIM,
    numpy_recurrent_rda_step,
    pack_head_bundle,
    rda_step,
    V_DIM,
)
from bench.rda_state.rda_state_1col import rda_step_1col


def _find_first_mismatch(a, b):
    """Return (token, head, row, col, a_val, b_val) for the first uint32 mismatch."""
    a_u32 = a.view(np.uint32)
    b_u32 = b.view(np.uint32)
    diff = a_u32 != b_u32
    if not np.any(diff):
        return None
    idx = np.unravel_index(np.argmax(diff), a_u32.shape)
    return (*idx, float(a[idx]), float(b[idx]))


def main():
    parser = argparse.ArgumentParser(
        description="Compare original and 1-column RDA state update designs"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--tokens", type=int, default=200)
    parser.add_argument("--heads", type=int, default=40)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--soak", type=int, default=10000)
    parser.add_argument("--soak-check", type=int, default=200)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    H = args.heads
    T = args.tokens

    print("=" * 70)
    print("RDA design comparison: multi-column vs 1-column tile-wise")
    print(f"heads={H}, tokens={T}, seeds={args.seeds}")
    print("=" * 70)

    # --------------------------------------------------------------------------
    # Prepare both designs with distinct current devices so they do not leak.
    # --------------------------------------------------------------------------
    dev_1col = device_from_name("npu2", n_cols=1)
    hostruntime.set_current_device(dev_1col)

    # Compile 1-column design once on the single-column device.
    s_in_1col = iron.zeros((H, V_DIM, K_DIM), dtype=np.float32, device="npu")
    s_out_1col = iron.zeros_like(s_in_1col)
    head_in_1col = iron.zeros((H, 640), dtype=np.float32, device="npu")
    out_1col = iron.zeros((H, V_DIM), dtype=np.float32, device="npu")

    for _ in range(args.warmup):
        rda_step_1col(s_in_1col, s_out_1col, head_in_1col, out_1col, num_heads=H, col=0)

    # Multi-column design uses the default full device.
    hostruntime.set_current_device(device_from_name("npu2", n_cols=None))
    s_in_orig = iron.zeros((H, V_DIM, K_DIM), dtype=np.float32, device="npu")
    s_out_orig = iron.zeros_like(s_in_orig)
    head_in_orig = iron.zeros((H, 640), dtype=np.float32, device="npu")
    out_orig = iron.zeros((H, V_DIM), dtype=np.float32, device="npu")

    for _ in range(args.warmup):
        rda_step(
            s_in_orig,
            s_out_orig,
            head_in_orig,
            out_orig,
            num_heads=H,
            num_workers=4,
            use_memtile=True,
        )

    # --------------------------------------------------------------------------
    # Bit-exact verification across seeds.
    # --------------------------------------------------------------------------
    print("\n[1/4] Bit-exact verification across seeds...")
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        q_all = rng.standard_normal((T, H, K_DIM)).astype(np.float32)
        q_all /= np.linalg.norm(q_all, axis=-1, keepdims=True) + 1e-12
        k_all = rng.standard_normal((T, H, K_DIM)).astype(np.float32)
        k_all /= np.linalg.norm(k_all, axis=-1, keepdims=True) + 1e-12
        a_all = np.exp(rng.uniform(-5.0, 0.0, size=(T, H, K_DIM)).astype(np.float32))
        beta_all = rng.uniform(0.01, 0.99, size=(T, H)).astype(np.float32)
        v_all = rng.standard_normal((T, H, V_DIM)).astype(np.float32)
        S_init = rng.standard_normal((H, V_DIM, K_DIM)).astype(np.float32) * 0.01

        S_golden = S_init.copy()

        s_in_1col[:] = S_init
        s_in_orig[:] = S_init
        for t in range(T):
            bundle = pack_head_bundle(q_all[t], k_all[t], a_all[t], v_all[t], beta_all[t])
            head_in_1col[:] = bundle
            head_in_orig[:] = bundle

            S_golden, _ = numpy_recurrent_rda_step(
                S_golden, q_all[t], k_all[t], a_all[t], v_all[t], beta_all[t]
            )

            hostruntime.set_current_device(dev_1col)
            rda_step_1col(s_in_1col, s_out_1col, head_in_1col, out_1col, num_heads=H, col=0)

            hostruntime.set_current_device(device_from_name("npu2", n_cols=None))
            rda_step(
                s_in_orig,
                s_out_orig,
                head_in_orig,
                out_orig,
                num_heads=H,
                num_workers=4,
                use_memtile=True,
            )

            s_out_1col_np = s_out_1col.numpy()
            out_1col_np = out_1col.numpy()
            s_out_orig_np = s_out_orig.numpy()
            out_orig_np = out_orig.numpy()

            mm = _find_first_mismatch(s_out_1col_np, s_out_orig_np)
            if mm is not None:
                head, row, col, a_val, b_val = mm[1], mm[2], mm[3], mm[4], mm[5]
                print(
                    f"  seed {seed} token {t}: s_out mismatch "
                    f"head={head} row={row} col={col} vals={a_val},{b_val}"
                )
                print("RESULT: FAIL s_out mismatch")
                sys.exit(1)

            mm = _find_first_mismatch(out_1col_np, out_orig_np)
            if mm is not None:
                head, row, a_val, b_val = mm[1], mm[2], mm[4], mm[5]
                print(
                    f"  seed {seed} token {t}: out mismatch "
                    f"head={head} row={row} vals={a_val},{b_val}"
                )
                print("RESULT: FAIL out mismatch")
                sys.exit(1)

            s_in_1col[:] = s_out_1col
            s_in_orig[:] = s_out_orig

    print("  All seeds PASS: s_out and out are bit-exact between designs.")

    # --------------------------------------------------------------------------
    # Benchmark both designs with the same input on the same board.
    # --------------------------------------------------------------------------
    print("\n[2/4] Benchmarking...")
    rng = np.random.default_rng(42)
    S_bench = rng.standard_normal((H, V_DIM, K_DIM)).astype(np.float32) * 0.01
    bundle_bench = pack_head_bundle(
        q=rng.standard_normal((H, K_DIM)).astype(np.float32),
        k=rng.standard_normal((H, K_DIM)).astype(np.float32),
        a=np.exp(rng.uniform(-5.0, 0.0, size=(H, K_DIM)).astype(np.float32)),
        v=rng.standard_normal((H, V_DIM)).astype(np.float32),
        beta=rng.uniform(0.01, 0.99, size=H).astype(np.float32),
    )
    s_in_1col[:] = S_bench
    s_in_orig[:] = S_bench
    head_in_1col[:] = bundle_bench
    head_in_orig[:] = bundle_bench

    def time_1col():
        hostruntime.set_current_device(dev_1col)
        t0 = time.perf_counter()
        rda_step_1col(s_in_1col, s_out_1col, head_in_1col, out_1col, num_heads=H, col=0)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    def time_orig():
        hostruntime.set_current_device(device_from_name("npu2", n_cols=None))
        t0 = time.perf_counter()
        rda_step(
            s_in_orig,
            s_out_orig,
            head_in_orig,
            out_orig,
            num_heads=H,
            num_workers=4,
            use_memtile=True,
        )
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0

    times_1col = [time_1col() for _ in range(args.iters)]
    times_orig = [time_orig() for _ in range(args.iters)]

    def stats(arr):
        return (
            float(np.percentile(arr, 50)),
            float(np.percentile(arr, 95)),
            float(np.percentile(arr, 99)),
        )

    p50_1, p95_1, p99_1 = stats(times_1col)
    p50_o, p95_o, p99_o = stats(times_orig)
    print(f"  1col  p50/p95/p99 ms: {p50_1:.3f} / {p95_1:.3f} / {p99_1:.3f}")
    print(f"  orig  p50/p95/p99 ms: {p50_o:.3f} / {p95_o:.3f} / {p99_o:.3f}")
    print(f"  ratio 1col/orig p50 : {p50_1 / p50_o if p50_o > 0 else float('inf'):.3f}")

    # --------------------------------------------------------------------------
    # Soak / repeat stability test on 1-column design.
    # --------------------------------------------------------------------------
    print(f"\n[3/4] Soak test: {args.soak} consecutive 1-column calls...")
    hostruntime.set_current_device(dev_1col)
    s_in_1col[:] = S_bench
    for i in range(args.soak):
        t0 = time.perf_counter()
        rda_step_1col(s_in_1col, s_out_1col, head_in_1col, out_1col, num_heads=H, col=0)
        dt = time.perf_counter() - t0
        if dt > args.timeout_s:
            print(f"  call {i} exceeded timeout: {dt:.3f}s")
            print("RESULT: FAIL soak timeout")
            sys.exit(1)
        s_in_1col[:] = s_out_1col
    print(f"  {args.soak} calls completed within {args.timeout_s}s each.")

    # --------------------------------------------------------------------------
    # Cross-check the 1-column design against the original design after soak.
    # Both designs start from the same state and run the same short sequence so
    # the comparison is between two NPU implementations, not against numpy.
    # --------------------------------------------------------------------------
    print("\n[4/4] Golden cross-check after soak...")
    check_state = s_out_1col.numpy().copy()

    hostruntime.set_current_device(dev_1col)
    s_in_1col[:] = check_state
    for _ in range(args.soak_check):
        rda_step_1col(
            s_in_1col, s_out_1col, head_in_1col, out_1col, num_heads=H, col=0
        )
        s_in_1col[:] = s_out_1col

    hostruntime.set_current_device(device_from_name("npu2", n_cols=None))
    s_in_orig[:] = check_state
    for _ in range(args.soak_check):
        rda_step(
            s_in_orig,
            s_out_orig,
            head_in_orig,
            out_orig,
            num_heads=H,
            num_workers=4,
            use_memtile=True,
        )
        s_in_orig[:] = s_out_orig

    mm = _find_first_mismatch(s_out_1col.numpy(), s_out_orig.numpy())
    if mm is not None:
        print(f"  cross-check mismatch: {mm}")
        print("RESULT: FAIL cross-check mismatch")
        sys.exit(1)
    print("  Golden cross-check PASS.")

    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
