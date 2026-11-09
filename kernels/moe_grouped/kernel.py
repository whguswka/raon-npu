# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""External-function contract for the shared grouped MoE AIE program."""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from ml_dtypes import bfloat16
    _BF16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    _BF16 = np.dtype[np.uint16]

_IMPORT_ERROR: Exception | None = None
try:
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
    from aie.utils import config
    from aie.utils.compile.jit.markers import In, InOut, Out
    from aie.utils.verify import Tolerance
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as error:
    _IMPORT_ERROR = error
    NPU_AVAILABLE = False


def grouped_kernel(mode: int = 0):
    if not NPU_AVAILABLE:
        raise RuntimeError(f"AIE/IRON unavailable: {_IMPORT_ERROR}") from _IMPORT_ERROR
    if mode not in (0, 1):
        raise ValueError("grouped kernel mode must be 0 (9-bit) or 1 (q4nx)")
    # Both signatures resolve to the same source/object and therefore one core
    # program. The dynamic ObjectFifo type follows the selected instruction
    # stream so q4nx consumes 5,120 B per K256 block without repacking W.
    w = np.ndarray[((9216 if mode == 0 else 5120) // 2,), _BF16]
    x = np.ndarray[(8 * 256,), _BF16]
    acc = np.ndarray[(8 * 32,), np.dtype[np.float32]]
    # B1 k_blocks (K 2560 // 256 = 10) * 8 groups * 8 tokens = 640 elements (1,280 B).
    # Must match B1 k_blocks.
    sums = np.ndarray[(10 * 8 * 8,), _BF16]
    y = np.ndarray[(8 * 32,), _BF16]
    kdir = Path(config.aie_kernels_dir())
    flags = ["-fstack-size-section", "-UAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"]
    base = _make_extern(
        "grouped_block", Path(__file__).with_name("grouped_mmul.cc"),
        [w, x, acc, sums, np.int32, np.int32, np.int32, np.int32, np.int32], compile_flags=flags,
        include_dirs=[config.cxx_header_path(), str(kdir), str(kdir / "flm_gemma4")],
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, In, InOut, InOut, Param, Param, Param, Param, Param),
            tolerance=Tolerance.exact(note="match decode GEMV BF16 bytes"),
            ops_per_call=2 * 8 * 32 * 256, acc_dtype=np.float32, reduction=256,
        ),
    )

    def sibling(symbol, arg_types, roles, note):
        return ExternalFunction(
            symbol, object_file_name=base.object_file_name, source_file=base.source_file,
            arg_types=arg_types, include_dirs=base.include_dirs,
            compile_flags=base.compile_flags, symbol_prefix=base.object_file.symbol_prefix,
            contract=KernelContract(
                trace=Trace.whole_call(), roles=roles,
                tolerance=Tolerance.exact(note=note), ops_per_call=0,
            ),
        )

    base.zero = sibling("grouped_zero", [acc, np.int32], (Out, Param), "zero")
    base.epilogue = sibling(
        "grouped_epilogue", [y, acc, np.int32], (Out, In, Param), "BF16 RTNE"
    )
    return base
