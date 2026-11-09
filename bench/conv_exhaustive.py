# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Exhaustive and timing comparison of NumPy and ml_dtypes BF16 conversions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from host import weights as weights_mod


def _first_differences(bits: np.ndarray, got: np.ndarray, expected: np.ndarray) -> str:
    indices = np.flatnonzero(got != expected)[:8]
    rows = []
    for idx in indices:
        rows.append(
            f"input=0x{int(bits[idx]):08X} ml=0x{int(got[idx]):04X} "
            f"numpy=0x{int(expected[idx]):04X}"
        )
    return "\n".join(rows)


def _time_call(fn, arr: np.ndarray, repeats: int) -> float:
    fn(arr)
    start = time.perf_counter()
    for _ in range(repeats):
        fn(arr)
    return (time.perf_counter() - start) * 1e6 / repeats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-size", type=int, default=1 << 24)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.repeats <= 0:
        parser.error("--chunk-size and --repeats must be positive")

    if weights_mod._ML_DTYPES_BFLOAT16 is None:
        print("ml_dtypes is unavailable; exhaustive comparison was not run.")
        print("RESULT: FAIL")
        return 2

    total = 1 << 32
    for start in range(0, total, args.chunk_size):
        count = min(args.chunk_size, total - start)
        bits = np.arange(count, dtype=np.uint32) + np.uint32(start)
        values = bits.view(np.float32)
        expected = weights_mod._f32_to_bf16_numpy(values)
        got = weights_mod._f32_to_bf16_ml_dtypes(values)
        if not np.array_equal(got, expected):
            print(f"float32 mismatch in [0x{start:08X}, 0x{start + count:08X})")
            print(_first_differences(bits, got, expected))
            print("RESULT: FAIL")
            return 1

    bf16_bits = np.arange(1 << 16, dtype=np.uint16)
    expected_f32 = weights_mod._bf16_to_f32_numpy(bf16_bits).view(np.uint32)
    got_f32 = weights_mod._bf16_to_f32_ml_dtypes(bf16_bits).view(np.uint32)
    if not np.array_equal(got_f32, expected_f32):
        indices = np.flatnonzero(got_f32 != expected_f32)[:8]
        for idx in indices:
            print(
                f"bf16=0x{idx:04X} ml=0x{int(got_f32[idx]):08X} "
                f"numpy=0x{int(expected_f32[idx]):08X}"
            )
        print("RESULT: FAIL")
        return 1

    rng = np.random.default_rng(20261101)
    print("size       f32->bf16 numpy/ml (us)   bf16->f32 numpy/ml (us)")
    for size in (2560, 5120, 20736, 655360):
        f32 = rng.standard_normal(size).astype(np.float32)
        u16 = weights_mod._f32_to_bf16_numpy(f32)
        f_np = _time_call(weights_mod._f32_to_bf16_numpy, f32, args.repeats)
        f_ml = _time_call(weights_mod._f32_to_bf16_ml_dtypes, f32, args.repeats)
        b_np = _time_call(weights_mod._bf16_to_f32_numpy, u16, args.repeats)
        b_ml = _time_call(weights_mod._bf16_to_f32_ml_dtypes, u16, args.repeats)
        print(f"{size:6d}     {f_np:10.2f}/{f_ml:10.2f}       {b_np:10.2f}/{b_ml:10.2f}")

    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
