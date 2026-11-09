# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""AIE2P design for Raon-1-mini general 9-bit bfp16ebs8 GEMV.

Supports general K and M projection shapes across an AMD XDNA 2 NPU (up to 8 cols x 4 rows).
Each round, an active core computes M_TILE (32) out-features across K_TILE (256) in-feature blocks.
Chunks of 128 output rows (ROWS * M_TILE = 4 * 32) are distributed across cores:
chunk c is assigned to column (c % cols) and round (c // cols).
Each core executes only the number of rounds corresponding to its assigned chunks.
Input vector x is stored in an ObjectFifo of depth 1, preserving core L1 memory for large K.

Key differences from kernels/gemv/design.py (q4nx GEMV):
-----------------------------------------------------------------------------------------
1. Weight size:
   Uses 9-bit bfp16ebs8 packed bytes (n_weights * 9 // 8), vs q4nx 5-bit (n_weights * 5 // 8).
   One 32x256 block is 9216 bytes (4608 bf16 elements) vs 5120 bytes.
2. Removed prep step (k_prep / b_group_sums):
   bfp16ebs8 does not have minimum offsets, so sum(x) group reduction is not required.
   Eliminates the prep call from core_fn and removes the sums L1 Buffer completely.
3. Kernel symbols & arguments:
   Uses bfp16_gemv_zero, bfp16_gemv_block (without sums argument), and bfp16_gemv_epilogue.
-----------------------------------------------------------------------------------------
"""

import numpy as np
try:
    from ml_dtypes import bfloat16
    bf16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    bf16 = np.uint16

try:
    from aie.dialects._aie_enum_gen import AIEArch
    from aie.helpers.npdtypes import np_ndarray_type_get_shape
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (
        Buffer,
        ObjectFifo,
        Program,
        Runtime,
        TaskGroup,
        Worker,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
except (ImportError, ModuleNotFoundError):
    class AIEArch:
        AIE2p = "AIE2p"
        AIE2 = "AIE2"
    Buffer = ObjectFifo = Program = Runtime = TaskGroup = Worker = object
    range_ = range
    Tile = object
    np_ndarray_type_get_shape = TensorAccessPattern = object

from quant.bfp16 import K_TILE, M_TILE, packed_bytes

# The sequence finishes each round's TaskGroup DEPTH rounds later.
DEPTH = 2

# Core stack bytes.
STACK_SIZE = 8 * 1024


def check_shape(dev, K: int, M: int, cols: int = 8):
    """Reject a device or shape the GEMV design does not support."""
    if dev.arch != AIEArch.AIE2p:
        raise NotImplementedError("the bfp16_gemv kernel is AIE2P only")
    if K < K_TILE:
        raise ValueError(f"K ({K}) must be at least {K_TILE} (one K_TILE block)")
    if K % K_TILE != 0:
        raise ValueError(f"K ({K}) must be a positive multiple of {K_TILE}")
    rows = len(dev.core_rows)
    m_per_chunk = rows * M_TILE
    if M < m_per_chunk:
        raise ValueError(
            f"M ({M}) must be at least {m_per_chunk} (rows={rows} * M_TILE={M_TILE})"
        )
    if M % m_per_chunk != 0:
        raise ValueError(
            f"M ({M}) must be a positive multiple of {m_per_chunk} (rows={rows} * M_TILE={M_TILE})"
        )
    if cols <= 0 or cols > dev.cols:
        raise ValueError(f"cols ({cols}) must be between 1 and {dev.cols}")


def gemv(dev, K: int, M: int, cols: int = 8, trace_size: int = 0, *, gemv_kernel):
    """Program for Raon general 9-bit bfp16ebs8 GEMV.

    ``gemv_kernel`` is the ``raon_bfp16_gemv`` kernel compiled for ``K``.
    X holds the input vector of length ``K`` (bf16).
    W holds the 9-bit bfp16 packed weights of shape ``[M, K]``.
    Y receives the projected output vector of length ``M`` (bf16).
    """
    check_shape(dev, K, M, cols)

    ROWS = len(dev.core_rows)
    M_PER_CHUNK = ROWS * M_TILE
    TOTAL_CHUNKS = M // M_PER_CHUNK

    # If chunks are fewer than cols, instantiate only the needed active columns.
    ACTIVE_COLS = min(cols, TOTAL_CHUNKS)

    # Number of rounds each active column will execute.
    # Chunk c is mapped to col = c % cols, rnd = c // cols.
    col_rounds = [
        TOTAL_CHUNKS // cols + (1 if j < (TOTAL_CHUNKS % cols) else 0)
        for j in range(ACTIVE_COLS)
    ]

    # Kernel argument types and core-local scratch buffers
    w_blk_ty, x_ty, y_acc_ty, _ = gemv_kernel.bfp16_gemv_block.arg_types()
    y_blk_ty, _ = gemv_kernel.bfp16_gemv_epilogue.arg_types()

    (w_blk,) = np_ndarray_type_get_shape(w_blk_ty)
    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, M_TILE), bf16]

    IT = [Tile(j, 0) for j in range(ACTIVE_COLS)]

    def core_fn(x_in, w_in, y_out, k_zero, k_block, k_epi, y_acc, rounds):
        x = x_in.acquire(1)
        for _ in range_(rounds):
            k_zero(y_acc)
            for k in range_(K // K_TILE):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    y_l3_ty = np.ndarray[(M,), bf16]
    w_l3_ty = np.ndarray[
        (packed_bytes(M * K) // np.dtype(np.uint32).itemsize,),
        np.dtype[np.uint32],
    ]
    x_l3_ty = np.ndarray[(K,), bf16]

    # ObjectFifo depth 1 for x: acquired once and held until all rounds complete,
    # avoiding double-buffering overhead and keeping L1 usage safe for K up to 12032+.
    of_x = ObjectFifo(x_ty, depth=1, name="x")
    x_prod = of_x.prod(tile=IT[0])

    workers, w_prods, y_conses = [], [], []
    for j in range(ACTIVE_COLS):
        of_w = ObjectFifo(w_col_ty, name=f"w{j}")
        w_prods.append(of_w.prod(tile=IT[j]))
        w_cores = of_w.cons().split(
            [w_blk * i for i in range(ROWS)],
            obj_types=[w_blk_ty] * ROWS,
            names=[f"w{j}_{i}" for i in range(ROWS)],
        )

        of_y = ObjectFifo(y_col_ty, name=f"y{j}")
        y_conses.append(of_y.cons(tile=IT[j]))
        y_cores = of_y.prod().join(
            [M_TILE * i for i in range(ROWS)],
            obj_types=[y_blk_ty] * ROWS,
            names=[f"y{j}_{i}" for i in range(ROWS)],
        )

        rounds_for_this_col = col_rounds[j]
        for i in range(ROWS):
            y_acc = Buffer(type=y_acc_ty, name=f"yacc_{i}_{j}")
            workers.append(
                Worker(
                    core_fn,
                    [
                        of_x.cons(),
                        w_cores[i].cons(),
                        y_cores[i].prod(),
                        gemv_kernel.bfp16_gemv_zero,
                        gemv_kernel.bfp16_gemv_block,
                        gemv_kernel.bfp16_gemv_epilogue,
                        y_acc,
                        rounds_for_this_col,
                    ],
                    stack_size=STACK_SIZE,
                )
            )

    W_PER_CHUNK = packed_bytes(M_PER_CHUNK * K) // np.dtype(np.uint32).itemsize

    def row_taps(rows, cols_len):
        return [
            TensorAccessPattern((rows, cols_len), i * cols_len, [1, 1, 1, cols_len], [0, 0, 0, 1])
            for i in range(rows)
        ]

    y_taps = row_taps(TOTAL_CHUNKS, M_PER_CHUNK)
    w_taps = row_taps(TOTAL_CHUNKS, W_PER_CHUNK)

    def sequence(Y, W, X, x_prod, w_prods, y_conses):
        x_task = x_prod.fill(X, wait=False, managed=False)

        max_rounds = (TOTAL_CHUNKS + cols - 1) // cols
        window = []
        for rnd in range(max_rounds):
            tg = TaskGroup()
            for col in range(ACTIVE_COLS):
                c = rnd * cols + col
                if c < TOTAL_CHUNKS:
                    y_conses[col].drain(Y, y_taps[c], wait=True, group=tg)
                    w_prods[col].fill(W, w_taps[c], wait=False, group=tg)

            window.append(tg)
            if len(window) > DEPTH:
                window.pop(0).finish()

        for tg in window:
            tg.finish()
        x_task.free()

    rt = Runtime(sequence, [y_l3_ty, w_l3_ty, x_l3_ty, x_prod, w_prods, y_conses])
    prog = Program(dev, rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)
    return prog.resolve_program()
