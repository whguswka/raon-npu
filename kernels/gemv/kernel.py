# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/mlir-aie (Apache-2.0 WITH LLVM-exception); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Modified by the raon-npu authors, 2026.

"""Custom AIE2P kernel factory for Raon-1-mini general GEMV (q4nx linear projection).

Compiles kernels/gemv/q4nx_gemv.cc with -DRAON_GEMV_K={K} to produce linear GEMV outputs
without RMSNorm or tanh softcap epilogue.
Ported from amd/mlir-aie python/iron/kernels/flm_gemma4.py (flm_gemma4_q4nx_lm_head).
Preserves original reference, sample, and exact tolerance contracts from amd/mlir-aie.
"""

from functools import partial
from pathlib import Path
from typing import get_args

import numpy as np
from ml_dtypes import bfloat16

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


try:
    from aie.iron.kernels.quant import _bf16_floor
except ImportError:
    def _bf16_floor(x):
        """Narrow float32 toward minus infinity, not numpy's bf16 nearest-even."""
        bits = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
        upper = bits >> 16
        upper += ((bits >> 31 != 0) & (bits & 0xFFFF != 0)).astype(np.uint32)
        return (upper << 16).view(np.float32)

_BF16, _F32 = np.dtype[bfloat16], np.dtype[np.float32]

_KERNEL_PATH = Path(__file__).resolve().parent / "q4nx_gemv.cc"


class _GemvKernel(_ZeroInitializedKernel):
    q4nx_gemv_prep: Kernel
    q4nx_gemv_zero: Kernel
    q4nx_gemv_block: ExternalFunction
    q4nx_gemv_epilogue: Kernel


def _on_object(base, symbol, arg_types, contract) -> ExternalFunction:
    """``symbol`` of ``base``'s object, judged by ``contract``.

    It names ``base``'s object and compile recipe, so a design that uses both
    compiles and links one copy of the source.
    """
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
            raise ValueError(f"raon_q4nx_gemv: {name} must be a positive integer")
    if group != 32:
        raise ValueError("raon_q4nx_gemv: group must be 32, the span of one column sum")
    if k_tile % group:
        raise ValueError("raon_q4nx_gemv: k_tile must be a multiple of group")
    if m_tile % 16:
        raise ValueError("raon_q4nx_gemv: m_tile must be a multiple of 16")
    if K % k_tile:
        raise ValueError("raon_q4nx_gemv: K must be a multiple of k_tile")


def _f32(x):
    """Round float64 to float32 once, as an AIE float accumulator does."""
    return np.asarray(x, np.float64).astype(np.float32)


def flm_gemma4_q4nx_gemv_ref(w, x, sums, *, m_tile=32, k_tile=256, group=32):
    """Numpy reference for one ``q4nx_gemv_block`` call.

    Preserved from amd/mlir-aie flm_gemma4.py for arithmetic contract verification.
    """
    w = np.ascontiguousarray(w).view(np.uint8).reshape(-1, m_tile * k_tile * 5 // 8)
    calls, n_groups = len(w), k_tile // group
    params = np.ascontiguousarray(w[:, : 4 * m_tile * n_groups]).view("<u2")
    params = (params.astype(np.uint32) << 16).view(np.float32)
    scales, mins = params.reshape(calls, 2, n_groups, m_tile).transpose(1, 0, 2, 3)
    packed = w[:, 4 * m_tile * n_groups :]
    codes = np.empty((calls, m_tile * k_tile), np.float64)
    codes[:, 0::2], codes[:, 1::2] = packed & 15, packed >> 4
    codes = codes.reshape(calls, m_tile // 16, n_groups, 32, 16)
    x = np.asarray(x, np.float32).reshape(-1)[:k_tile]
    x = x.astype(np.float64).reshape(1, n_groups, 32)
    sums = np.asarray(sums, np.float32).reshape(-1)[:n_groups]
    sums = sums.astype(np.float64).reshape(1, n_groups)
    acc = np.zeros((calls, m_tile // 16, 16), np.float32)
    for g in range(n_groups):
        dot = np.zeros((calls, m_tile // 16, 16), np.float32)
        for c in range(32):
            dot = _f32(dot + codes[:, :, g, c, :] * x[:, g, c, None, None])
        dot = _bf16_round(dot).astype(np.float64)
        scale = scales[:, g].reshape(calls, m_tile // 16, 16).astype(np.float64)
        low = mins[:, g].reshape(calls, m_tile // 16, 16).astype(np.float64)
        acc = _f32(acc + dot * scale)
        acc = _f32(acc + low * sums[:, g, None, None])
    return acc.reshape(calls, m_tile)


def _q4nx_gemv_sample(rng, calls, *, K, m_tile, k_tile, group):
    """Sample generator for general GEMV contract verification."""
    n_groups = k_tile // group
    params = rng.choice([-1.0, 1.0], (calls, 2 * n_groups * m_tile)) * 2.0 ** (
        -rng.integers(0, 5, (calls, 2 * n_groups * m_tile))
    )
    params = params.astype(bfloat16).view(np.uint16).astype("<u2").view(np.uint8)
    packed = rng.integers(0, 256, (calls, m_tile * k_tile // 2), dtype=np.uint8)
    w = np.concatenate([params, packed], axis=1).view(bfloat16)
    x = rng.integers(-4, 5, K).astype(bfloat16)
    sums = x.astype(np.float32).reshape(K // group, group).sum(axis=1)
    return [w, x, sums.astype(bfloat16)]


def _gemv_zero(fn) -> ExternalFunction:
    """``fn``'s ``q4nx_gemv_zero``, the initializer of its accumulator.

    It shares ``fn``'s object, so the design links one copy of the source.
    """
    y_acc = fn.arg_types()[2]
    return _on_object(
        fn,
        "q4nx_gemv_zero",
        [y_acc],
        KernelContract(
            trace=Trace.whole_call(),
            roles=(Out,),
            reference=lambda: np.zeros((1, *get_args(y_acc)[0]), np.float32),
            tolerance=Tolerance.exact(note="zero fill"),
            ops_per_call=0,
        ),
    )


def raon_q4nx_gemv(
    *, K: int = 2560, m_tile: int = 32, k_tile: int = 256, group: int = 32
) -> ExternalFunction:
    """AIE2P linear projection from a q4nx matrix without RMSNorm or softcap.

    Source: ``kernels/gemv/q4nx_gemv.cc``.
    Geometry: in-features K (multiple of 256), m_tile=32, k_tile=256, group=32.
    Preserves original reference, sample, and exact tolerance contracts from amd/mlir-aie.
    """
    _gemv_geometry(K, m_tile, k_tile, group)
    if not _arch_traits().bfp16:
        raise NotImplementedError("raon_q4nx_gemv() is only available on aie2p.")

    geometry = dict(m_tile=m_tile, k_tile=k_tile, group=group)
    x = np.ndarray[(K,), _BF16]
    sums = np.ndarray[(K // group,), _BF16]
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
        f"-DQ4NX_M_TILE={m_tile}",
        f"-DQ4NX_K_TILE={k_tile}",
        f"-DQ4NX_GROUP={group}",
        f"-DRAON_GEMV_K={K}",
        "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
    ]

    fn = _make_extern(
        "q4nx_gemv_block",
        _KERNEL_PATH,
        [np.ndarray[(m_tile * k_tile * 5 // 8 // 2,), _BF16], x, y_acc, sums, np.int32],
        compile_flags=compile_flags,
        include_dirs=include_dirs,
        cls=_GemvKernel,
        contract=KernelContract(
            trace=Trace.whole_call(),
            roles=(In, Param, InOut, Param, Param),
            parameter_bindings=((4, 0),),
            initializers=((2, _gemv_zero),),
            reference=partial(flm_gemma4_q4nx_gemv_ref, **geometry),
            sample=partial(_q4nx_gemv_sample, K=K, **geometry),
            tolerance=Tolerance.exact(
                note="the sample keeps every sum exact in float32; the conv_even "
                "(nearest-even) bf16 narrowing of each dot product is modeled"
            ),
            ops_per_call=2 * m_tile * k_tile,
            acc_dtype=np.float32,
            reduction=k_tile,
        ),
    )

    bind = fn.object_file.bind
    fn.q4nx_gemv_block = fn
    fn.q4nx_gemv_prep = bind("q4nx_gemv_prep", [x, sums])
    fn.q4nx_gemv_zero = bind("q4nx_gemv_zero", [y_acc])
    fn.q4nx_gemv_epilogue = bind("q4nx_gemv_epilogue", [y, y_acc])
    return fn
