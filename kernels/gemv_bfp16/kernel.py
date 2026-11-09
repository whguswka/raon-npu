# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/mlir-aie (Apache-2.0 WITH LLVM-exception); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modified by the raon-npu authors, 2026.

"""Custom AIE2P kernel factory for Raon-1-mini general GEMV (9-bit bfp16ebs8 projection).

Compiles kernels/gemv_bfp16/bfp16_gemv.cc with -DRAON_GEMV_K={K} to produce linear GEMV outputs.
Eliminates RMSNorm, softcap, and prep group-sums (no min offsets in bfp16ebs8).
Accumulates directly in float32 without intermediate bf16 narrowing.
"""

from functools import partial
from pathlib import Path
from typing import get_args

import numpy as np
try:
    from ml_dtypes import bfloat16
    _BF16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    _BF16 = np.uint16

_F32 = np.dtype[np.float32]

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
except (ImportError, ModuleNotFoundError):
    ExternalFunction = Kernel = object
    KernelContract = Param = Trace = _arch_traits = _make_extern = object
    _ZeroInitializedKernel = object
    config = object
    In = InOut = Out = object
    Tolerance = object


def _bf16_round(x):
    """Narrow float32 to bfloat16 using Round-to-Nearest-Even (conv_even)."""
    bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    isnan = np.isnan(x)
    bias = 0x7FFF + ((bits >> 16) & 1)
    u_rounded = (bits + bias) >> 16
    res = (u_rounded.astype(np.uint32) << 16).view(np.float32)
    if np.any(isnan):
        res[isnan] = np.nan
    return res


_KERNEL_PATH = Path(__file__).resolve().parent / "bfp16_gemv.cc"


class _GemvBfp16Kernel(_ZeroInitializedKernel):
    bfp16_gemv_zero: Kernel
    bfp16_gemv_block: ExternalFunction
    bfp16_gemv_epilogue: Kernel


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


def _gemv_geometry(K, m_tile, k_tile, group):
    for name, value in dict(K=K, m_tile=m_tile, k_tile=k_tile, group=group).items():
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"raon_bfp16_gemv: {name} must be a positive integer")
    if group != 8:
        raise ValueError("raon_bfp16_gemv: group must be 8 (bfp16ebs8 shared exponent block)")
    if k_tile % group:
        raise ValueError("raon_bfp16_gemv: k_tile must be a multiple of group")
    if m_tile % 16:
        raise ValueError("raon_bfp16_gemv: m_tile must be a multiple of 16")
    if K % k_tile:
        raise ValueError("raon_bfp16_gemv: K must be a multiple of k_tile")


def _f32(x):
    """Round float64 to float32 once, as an AIE float accumulator does."""
    return np.asarray(x, np.float64).astype(np.float32)


def raon_bfp16_gemv_ref(w, x, *, m_tile=32, k_tile=256, group=8):
    """Numpy reference for one ``bfp16_gemv_block`` call.

    Block is 9216 bytes (m_tile * k_tile * 9 // 8).
    Unpacks exponents and mantissas, exact bf16 weights, and accumulates in float32.
    """
    from kernels.gemv_bfp16.reference import dequantize_block

    w_bytes = bytes(w) if isinstance(w, (np.ndarray, memoryview)) else w
    calls = len(w_bytes) // 9216
    w_block_bytes = 9216
    acc = np.zeros((calls, m_tile), dtype=np.float32)
    x_vec = np.asarray(x, dtype=np.float32).reshape(-1)[:k_tile]

    for c in range(calls):
        b = w_bytes[c * w_block_bytes : (c + 1) * w_block_bytes]
        weights = dequantize_block(b)  # [m_tile, k_tile]
        # Sequential float32 accumulation
        y_c = np.zeros(m_tile, dtype=np.float32)
        for k in range(k_tile):
            y_c = (y_c + weights[:, k] * x_vec[k]).astype(np.float32)
        acc[c] = y_c

    return acc


def _bfp16_gemv_sample(rng, calls, *, K, m_tile, k_tile, group):
    """Sample generator for general bfp16 GEMV contract verification.

    Generates integer-representable samples where all partial sums are exact in float32.
    """
    from quant.bfp16 import pack_bfp16_gemv

    # Small integer weights in [-4, 4]
    w_raw = rng.integers(-4, 5, (calls * m_tile, k_tile)).astype(np.float32)
    w_bytes = pack_bfp16_gemv(w_raw, rows=calls * (m_tile // 32), rounding="conv_even")
    w_buf = np.frombuffer(w_bytes, dtype=_BF16)

    # Small integer x in [-4, 4]
    x = rng.integers(-4, 5, K).astype(_BF16)
    return [w_buf, x]


def _gemv_zero(fn) -> ExternalFunction:
    """``fn``'s ``bfp16_gemv_zero``, the initializer of its accumulator."""
    y_acc = fn.arg_types()[2]
    return _on_object(
        fn,
        "bfp16_gemv_zero",
        [y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            reference=lambda: np.zeros((1, *get_args(y_acc)[0]), np.float32),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )


def raon_bfp16_gemv(
    *, K: int = 2560, m_tile: int = 32, k_tile: int = 256, group: int = 8
) -> ExternalFunction:
    """AIE2P linear projection from a 9-bit bfp16ebs8 matrix.

    Source: ``kernels/gemv_bfp16/bfp16_gemv.cc``.
    Geometry: in-features K (multiple of 256), m_tile=32, k_tile=256, group=8.
    """
    _gemv_geometry(K, m_tile, k_tile, group)
    if not _arch_traits().bfp16:
        raise NotImplementedError("raon_bfp16_gemv() is only available on aie2p.")

    geometry = dict(m_tile=m_tile, k_tile=k_tile, group=group)
    x = np.ndarray[(K,), _BF16]
    y_acc = np.ndarray[(m_tile,), _F32]
    y = np.ndarray[(m_tile,), _BF16]

    # Include dirs for compiling custom kernel
    cxx_inc = config.cxx_header_path()
    aie_kdir = Path(config.aie_kernels_dir())
    include_dirs = [
        str(_KERNEL_PATH.parent),
        cxx_inc,
        str(aie_kdir),
        str(aie_kdir / "flm_gemma4"),
    ]

    compile_flags = [
        f"-DBFP16_M_TILE={m_tile}",
        f"-DBFP16_K_TILE={k_tile}",
        f"-DRAON_GEMV_K={K}",
    ]

    # w_blk is 9216 bytes = (m_tile * k_tile * 9 // 8) bytes = 4608 bf16 elements
    w_blk_len = m_tile * k_tile * 9 // 8 // 2
    fn = _make_extern(
        "bfp16_gemv_block",
        _KERNEL_PATH,
        [np.ndarray[(w_blk_len,), _BF16], x, y_acc, np.int32],
        compile_flags=compile_flags,
        include_dirs=include_dirs,
        cls=_GemvBfp16Kernel,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, InOut, Param),
            parameter_bindings=((3, 0),),
            initializers=((2, _gemv_zero),),
            reference=partial(raon_bfp16_gemv_ref, **geometry),
            sample=partial(_bfp16_gemv_sample, K=K, **geometry),
            tolerance=Tolerance.exact(
                note="exact float32 accumulation without intermediate narrowing"
            ),
            ops_per_call=2 * m_tile * k_tile,
            acc_dtype=np.float32,
            reduction=k_tile,
        ),
    )

    bind = fn.object_file.bind
    fn.bfp16_gemv_block = fn
    fn.bfp16_gemv_zero = bind("bfp16_gemv_zero", [y_acc])
    fn.bfp16_gemv_epilogue = bind("bfp16_gemv_epilogue", [y, y_acc])
    return fn
