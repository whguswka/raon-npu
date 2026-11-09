# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Exact BF16 prefill kernel factory with no gemv_batch dependency."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .shapes import MAX_BATCH_T

try:
    from ml_dtypes import bfloat16
    _BF16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    _BF16 = np.dtype[np.uint16]

_F32 = np.dtype[np.float32]
_IMPORT_ERROR: Exception | None = None
try:
    from aie.iron.kernel import ExternalFunction
    from aie.iron.kernels._common import KernelContract, Param, Trace, _make_extern
    from aie.utils import config
    from aie.utils.compile.jit.markers import In, InOut, Out
    from aie.utils.verify import Tolerance
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False
    ExternalFunction = object
    KernelContract = Param = Trace = _make_extern = object
    config = In = InOut = Out = Tolerance = object


def _on_object(base, symbol, arg_types, roles, *, note):
    return ExternalFunction(
        symbol, object_file_name=base.object_file_name, source_file=base.source_file,
        arg_types=arg_types, include_dirs=base.include_dirs,
        compile_flags=base.compile_flags, symbol_prefix=base.object_file.symbol_prefix,
        contract=KernelContract(trace=Trace.whole_call(), roles=roles,
                                tolerance=Tolerance.exact(note=note), ops_per_call=0),
    )


def compile_flags(max_batch_t=MAX_BATCH_T, variant="exact", fmt=0):
    if max_batch_t != MAX_BATCH_T or variant != "exact" or fmt != 0:
        raise ValueError(f"the prefill device supports exact bfp16 GEMM with max T={MAX_BATCH_T} only")
    return ["-fstack-size-section", f"-DPREFILL_MAX_T={max_batch_t}", "-DPREFILL_Q4=0",
            "-UAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16"]


def prefill_kernel(max_batch_t=MAX_BATCH_T, variant="exact", fmt=0):
    flags = compile_flags(max_batch_t, variant, fmt)
    if not NPU_AVAILABLE:
        raise RuntimeError(f"AIE/IRON unavailable: {_IMPORT_ERROR}") from _IMPORT_ERROR
    w = np.ndarray[(9216 // 2,), _BF16]
    x = np.ndarray[(max_batch_t * 256,), _BF16]
    acc = np.ndarray[(max_batch_t * 32,), _F32]
    y = np.ndarray[(max_batch_t * 32,), _BF16]
    kdir = Path(config.aie_kernels_dir())
    base = _make_extern(
        "prefill_block", Path(__file__).with_name("mmul.cc"),
        [w, x, acc, np.int32, np.int32], compile_flags=flags,
        include_dirs=[config.cxx_header_path(), str(kdir), str(kdir / "flm_gemma4")],
        contract=KernelContract(
            trace=Trace.whole_call(), roles=(In, In, InOut, Param, Param),
            tolerance=Tolerance.exact(note="BF16 output must equal decode GEMV"),
            ops_per_call=2 * max_batch_t * 32 * 256,
            acc_dtype=np.float32, reduction=256,
        ),
    )
    base.zero = _on_object(base, "prefill_zero", [acc, np.int32], (Out, Param), note="zero")
    base.epilogue = _on_object(base, "prefill_epilogue", [y, acc, np.int32], (Out, In, Param), note="BF16 RTNE")
    return base
