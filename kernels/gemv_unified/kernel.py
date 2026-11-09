# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Fixed-geometry external kernel object for unified bfp16 GEMV."""

from __future__ import annotations

from pathlib import Path
from typing import get_args

import numpy as np

try:
    from ml_dtypes import bfloat16
    _BF16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    _BF16 = np.dtype[np.uint16]

_F32 = np.dtype[np.float32]

_IMPORT_ERROR: Exception | None = None
try:
    from aie.iron.kernel import ExternalFunction, Kernel
    from aie.iron.kernels._common import (
        KernelContract,
        Param,
        Trace,
        _arch_traits,
        _make_extern,
    )
    from aie.iron.kernels.linalg import _ZeroInitializedKernel
    from aie.utils import config
    from aie.utils.compile.jit.markers import In, InOut, Out
    from aie.utils.verify import Tolerance
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False
    ExternalFunction = Kernel = object
    KernelContract = Param = Trace = _arch_traits = _make_extern = object
    _ZeroInitializedKernel = object
    config = object
    In = InOut = Out = object
    Tolerance = object

from .shapes import K_TILE, MAX_K, M_TILE, W_BLOCK_BYTES

_SOURCE = Path(__file__).resolve().parent / "bfp16_gemv.cc"


class _UnifiedBfp16Kernel(_ZeroInitializedKernel):
    bfp16_gemv_unified_zero: Kernel
    bfp16_gemv_unified_block: ExternalFunction
    bfp16_gemv_unified_epilogue: Kernel
    bfp16_gemv_zero: Kernel
    bfp16_gemv_block: ExternalFunction
    bfp16_gemv_epilogue: Kernel
    q4nx_gemv_unified_prep: ExternalFunction
    q4nx_gemv_unified_zero: ExternalFunction
    q4nx_gemv_unified_block: ExternalFunction
    q4nx_gemv_unified_epilogue: ExternalFunction


def _on_object(base, symbol, arg_types, contract) -> ExternalFunction:
    """``symbol`` of ``base``'s object, judged by ``contract``."""
    return ExternalFunction(
        symbol,
        object_file_name=base.object_file_name,
        source_file=base.source_file,
        arg_types=arg_types,
        include_dirs=base.include_dirs,
        compile_flags=base.compile_flags,
        symbol_prefix=base.object_file.symbol_prefix,
        contract=contract,
    )


def _gemv_zero(fn) -> ExternalFunction:
    """``fn``'s ``bfp16_gemv_unified_zero``, the initializer of its accumulator."""
    y_acc = fn.arg_types()[2]
    return _on_object(
        fn,
        "bfp16_gemv_unified_zero",
        [y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            reference=lambda: np.zeros((1, *get_args(y_acc)[0]), np.float32),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )


def _gemv_epilogue(fn) -> ExternalFunction:
    """``fn``'s ``bfp16_gemv_unified_epilogue``, the epilogue narrowing y_acc to bf16."""
    y_acc = fn.arg_types()[2]
    y = np.ndarray[(M_TILE,), _BF16]
    return _on_object(
        fn,
        "bfp16_gemv_unified_epilogue",
        [y, y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out, In),
            tolerance=Tolerance.exact(note="epilogue conv_even narrow"),
            ops_per_call=0,
        ),
    )


def unified_bfp16_kernel(K: int | None = None) -> ExternalFunction:
    """Return one object containing both bfp16ebs8 and q4nx entry points.

    ``K`` overrides the X buffer size used in the kernel signatures.  When
    ``None`` the shared configuration size ``MAX_K`` is used.  A smaller X is
    needed for ``diagnostic_small_x`` designs, where the ObjectFifo and host
    X BO are sized to the actual K.
    """
    if not NPU_AVAILABLE:
        raise RuntimeError(
            f"unified_bfp16_kernel: AIE/IRON packages are not available: {_IMPORT_ERROR}"
        ) from _IMPORT_ERROR
    if not _arch_traits().bfp16:
        raise NotImplementedError("unified bfp16 GEMV requires AIE2P bfp16 support")

    try:
        import aie.utils as aie_utils
        import aie.utils.hostruntime as hostruntime
        hostruntime.set_current_device(aie_utils.get_current_device())
    except Exception:
        pass

    max_k = K if K is not None else MAX_K
    w_ty = np.ndarray[(W_BLOCK_BYTES // 2,), _BF16]
    x_ty = np.ndarray[(max_k,), _BF16]
    acc_ty = np.ndarray[(M_TILE,), _F32]
    aie_kdir = Path(config.aie_kernels_dir())
    include_dirs = [
        str(_SOURCE.parent),
        config.cxx_header_path(),
        str(aie_kdir),
        str(aie_kdir / "flm_gemma4"),
    ]
    fn = _make_extern(
        "bfp16_gemv_unified_block",
        _SOURCE,
        [w_ty, x_ty, acc_ty, np.int32],
        compile_flags=[
            f"-DBFP16_M_TILE={M_TILE}",
            f"-DBFP16_K_TILE={K_TILE}",
            "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
        ],
        include_dirs=include_dirs,
        cls=_UnifiedBfp16Kernel,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, InOut, Param),
            parameter_bindings=((3, 0),),
            initializers=((2, _gemv_zero),),
            tolerance=Tolerance.exact(
                note="exact float32 accumulation without intermediate narrowing"
            ),
            ops_per_call=2 * M_TILE * K_TILE,
            acc_dtype=np.float32,
            reduction=K_TILE,
        ),
    )
    fn.bfp16_gemv_unified_block = fn
    fn.bfp16_gemv_unified_zero = _gemv_zero(fn)
    fn.bfp16_gemv_unified_epilogue = _gemv_epilogue(fn)
    fn.bfp16_gemv_block = fn.bfp16_gemv_unified_block
    fn.bfp16_gemv_zero = fn.bfp16_gemv_unified_zero
    fn.bfp16_gemv_epilogue = fn.bfp16_gemv_unified_epilogue

    sums_ty = np.ndarray[(max_k // 32,), _BF16]
    y_ty = np.ndarray[(M_TILE,), _BF16]
    fn.q4nx_gemv_unified_prep = _on_object(
        fn,
        "q4nx_gemv_unified_prep",
        [x_ty, sums_ty, np.int32],
        KernelContract(
            trace=Trace.whole_call(), roles=(In, Out, Param), ops_per_call=0
        ),
    )
    fn.q4nx_gemv_unified_zero = _on_object(
        fn,
        "q4nx_gemv_unified_zero",
        [acc_ty],
        KernelContract(
            trace=Trace.whole_call(), roles=(Out,), ops_per_call=0,
            tolerance=Tolerance.exact(note="zero fill"),
        ),
    )
    fn.q4nx_gemv_unified_block = _on_object(
        fn,
        "q4nx_gemv_unified_block",
        [w_ty, x_ty, acc_ty, sums_ty, np.int32],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, InOut, Param, Param),
            parameter_bindings=((4, 0),),
            tolerance=Tolerance.exact(note="q4nx conv_even accumulation"),
            ops_per_call=2 * M_TILE * K_TILE,
            acc_dtype=np.float32,
            reduction=K_TILE,
        ),
    )
    fn.q4nx_gemv_unified_epilogue = _on_object(
        fn,
        "q4nx_gemv_unified_epilogue",
        [y_ty, acc_ty],
        KernelContract(
            trace=Trace.whole_call(), roles=(Out, In), ops_per_call=0,
            tolerance=Tolerance.exact(note="epilogue conv_even narrow"),
        ),
    )
    return fn
