# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Compile and invoke one bounded unified-GEMV stall diagnostic at a time."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.gemv_bfp16.reference import (  # noqa: E402
    bf16_ulp_diff,
    bfp16_gemv_bit_reference,
)
from kernels.gemv_unified.kernel import unified_bfp16_kernel  # noqa: E402
from kernels.gemv_unified.shapes import MAX_K, W_BLOCK_BYTES  # noqa: E402
from quant.bfp16 import pack_bfp16_gemv  # noqa: E402
from quant.q4nx import round_to_bf16_rne, to_bf16_u16  # noqa: E402

STAGES = {"core1": (1, 1), "col4": (1, 4), "cols8": (8, 4)}


def log(message: str) -> None:
    print(message, flush=True)


def make_diagnostic_case(rows: int, total_m: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create one-block data in the row-interleave used by this topology."""
    rng = np.random.default_rng(20261029)
    x_f32 = round_to_bf16_rne(rng.integers(-4, 5, 256).astype(np.float32))
    w_f32 = rng.integers(-4, 5, (total_m, 256)).astype(np.float32)
    packed = pack_bfp16_gemv(w_f32, rows=rows, rounding="conv_even")
    _, ref = bfp16_gemv_bit_reference(w_f32, x_f32, rounding="conv_even")
    return np.frombuffer(packed, dtype=np.uint32), to_bf16_u16(x_f32), ref


def compare_bits(got: np.ndarray, ref: np.ndarray) -> tuple[int, int]:
    got = np.asarray(got, dtype=np.uint16)
    ref = np.asarray(ref, dtype=np.uint16)
    return int(np.count_nonzero(got != ref)), int(bf16_ulp_diff(got, ref).max(initial=0))


def run(stage: str, counts: str, prepare_only: bool) -> int:
    import aie.utils as aie_utils
    from ml_dtypes import bfloat16
    from iron.common import (
        AIEContext,
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
    from iron.common.compilation import InstsBinArtifact, XclbinArtifact

    cols, rows = STAGES[stage]
    dynamic = counts == "dynamic"
    total_m = cols * rows * 32
    w_words = cols * rows * W_BLOCK_BYTES // 4
    ctx = AIEContext()

    class DiagnosticOp(MLIROperator):
        def __init__(self, context: Any):
            super().__init__(context=context)

        def set_up_artifacts(self) -> None:
            kernel = unified_bfp16_kernel()
            kernel_art = KernelObjectArtifact.from_extern(kernel)
            design = Path(__file__).resolve().parents[2] / "kernels" / "gemv_unified" / "diagnostic.py"
            dev = aie_utils.get_current_device()
            prefix = f"Raon_GEMV_Diag_{stage}_{counts}_{dev.resolve().name}"
            mlir = PythonGeneratedMLIRArtifact(
                f"{prefix}.mlir",
                DesignGenerator(
                    design,
                    "gemv_diagnostic",
                    (dev, cols, rows, dynamic),
                    {"gemv_kernel": kernel},
                ),
            )
            flags = ["--dynamic-objFifos"] if dynamic else []
            self.xclbin_artifact = XclbinArtifact(
                f"{prefix}.xclbin", mlir_input=mlir,
                dependencies=[mlir, kernel_art], kernel_name="MLIR_AIE",
                extra_flags=flags,
            )
            self.insts_artifact = InstsBinArtifact(
                f"{prefix}.bin", mlir_input=mlir,
                dependencies=[mlir, kernel_art], extra_flags=flags,
            )
            self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

        def get_arg_spec(self):
            return [
                AIERuntimeArgSpec("out", (total_m,), dtype=bfloat16),
                AIERuntimeArgSpec("in", (w_words,), dtype=np.uint32),
                AIERuntimeArgSpec("in", (MAX_K,), dtype=bfloat16),
            ]

    log(f"[diag] compile stage={stage} topology={cols}x{rows} counts={counts}")
    op = DiagnosticOp(context=ctx)
    op.set_up_artifacts()
    op.compile()
    if prepare_only:
        log("[diag] compile complete; no NPU invocation requested")
        return 0

    w, x, ref = make_diagnostic_case(rows, total_m)
    x_padded = np.zeros(MAX_K, dtype=np.uint16)
    x_padded[:256] = x
    y_bo = aie_utils.zeros((total_m,), dtype=bfloat16)
    w_bo = aie_utils.tensor(w)
    x_bo = aie_utils.tensor(x_padded.view(bfloat16))
    log("[diag] invoking exactly once")
    op.get_callable()(y_bo, w_bo, x_bo)
    got = np.asarray(y_bo.numpy()).view(np.uint16)
    mismatches, max_ulp = compare_bits(got, ref)
    log(f"[diag] completed mismatches={mismatches} max_ulp={max_ulp}")
    return 0 if mismatches == 0 else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=tuple(STAGES), required=True)
    p.add_argument("--counts", choices=("static", "dynamic"), required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--timeout", type=float, default=0.0)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.timeout > 0 and not args.child:
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--child"] + (
            argv if argv is not None else sys.argv[1:]
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        log(f"[watchdog] timeout={args.timeout:.1f}s")
        try:
            return subprocess.run(cmd, timeout=args.timeout, env=env, check=False).returncode
        except subprocess.TimeoutExpired:
            log(f"[watchdog] TIMEOUT stage={args.stage} counts={args.counts}")
            return 124
    return run(args.stage, args.counts, args.prepare_only)


if __name__ == "__main__":
    raise SystemExit(main())
