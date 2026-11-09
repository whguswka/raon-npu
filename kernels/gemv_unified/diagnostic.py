# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Small, single-dispatch GEMV programs used to localize NPU stalls.

Each program executes exactly one K block and one output round.  ``cols`` and
``rows`` select the topology; ``dynamic_counts`` selects RTP/barrier counts or
compile-time constants without changing the dataflow graph.
"""

from __future__ import annotations

import numpy as np

from kernels.gemv_unified.shapes import MAX_K, M_TILE, W_BLOCK_BYTES


def gemv_diagnostic(dev, cols: int, rows: int, dynamic_counts: bool, *, gemv_kernel):
    """Build one of the 1x1, 1x4, or 8x4 diagnostic programs."""
    from ml_dtypes import bfloat16

    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, ObjectFifo, Program, Runtime, TaskGroup, Worker
    from aie.iron import WorkerRuntimeBarrier
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

    if dev is None:
        import aie.utils as aie_utils

        dev = aie_utils.get_current_device()

    if dev is not None:
        try:
            import aie.utils.hostruntime as hostruntime

            hostruntime.set_current_device(dev)
        except Exception:
            pass

    if (cols, rows) not in ((1, 1), (1, 4), (8, 4)):
        raise ValueError(f"unsupported diagnostic topology {cols}x{rows}")

    k_zero = gemv_kernel.bfp16_gemv_unified_zero
    k_block = gemv_kernel.bfp16_gemv_unified_block
    k_epi = gemv_kernel.bfp16_gemv_unified_epilogue
    w_elems = W_BLOCK_BYTES // 2
    m_per_col = rows * M_TILE
    total_m = cols * m_per_col

    x_ty = np.ndarray[(MAX_K,), np.dtype[bfloat16]]
    w_row_ty = np.ndarray[(w_elems,), np.dtype[bfloat16]]
    w_col_ty = np.ndarray[(rows * w_elems,), np.dtype[bfloat16]]
    y_row_ty = np.ndarray[(M_TILE,), np.dtype[bfloat16]]
    y_col_ty = np.ndarray[(m_per_col,), np.dtype[bfloat16]]
    acc_ty = np.ndarray[(M_TILE,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    x_host_ty = np.ndarray[(MAX_K,), np.dtype[bfloat16]]
    w_host_ty = np.ndarray[(cols * rows * W_BLOCK_BYTES // 4,), np.dtype[np.uint32]]
    y_host_ty = np.ndarray[(total_m,), np.dtype[bfloat16]]

    x_prods = []
    w_prods = []
    y_conss = []
    workers = []
    rtps = []
    barriers = []

    def static_core(x_in, w_in, y_out, y_acc):
        x = x_in.acquire(1)
        for _ in range_(1):
            k_zero(y_acc)
            for k in range_(1):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    def dynamic_core(x_in, w_in, y_out, y_acc, rtp, barrier):
        barrier.wait_for_value(1)
        k_blocks = rtp[0]
        rounds = rtp[1]
        barrier.release_with_value(1)
        x = x_in.acquire(1)
        for _ in range_(rounds):
            k_zero(y_acc)
            for k in range_(k_blocks):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    for col in range(cols):
        shim = Tile(col, 0)
        of_x = ObjectFifo(x_ty, depth=1, name=f"diag_x{col}")
        x_prods.append(of_x.prod(tile=shim))

        of_w = ObjectFifo(w_col_ty, depth=2, name=f"diag_w{col}")
        w_prods.append(of_w.prod(tile=shim))
        w_rows = of_w.cons().split(
            [row * w_elems for row in range(rows)],
            obj_types=[w_row_ty] * rows,
            names=[f"diag_w{col}_{row}" for row in range(rows)],
        )

        of_y = ObjectFifo(y_col_ty, depth=2, name=f"diag_y{col}")
        y_conss.append(of_y.cons(tile=shim))
        y_rows = of_y.prod().join(
            [row * M_TILE for row in range(rows)],
            obj_types=[y_row_ty] * rows,
            names=[f"diag_y{col}_{row}" for row in range(rows)],
        )

        for row in range(rows):
            y_acc = Buffer(type=acc_ty, name=f"diag_acc_{row}_{col}")
            if dynamic_counts:
                rtp = Buffer(
                    type=rtp_ty,
                    name=f"diag_rtp_{row}_{col}",
                    initial_value=np.zeros(2, dtype=np.int32),
                    use_write_rtp=True,
                )
                barrier = WorkerRuntimeBarrier()
                rtps.append(rtp)
                barriers.append(barrier)
                args = [
                    of_x.cons(), w_rows[row].cons(), y_rows[row].prod(),
                    y_acc, rtp, barrier,
                ]
                workers.append(
                    Worker(
                        dynamic_core, args, stack_size=8192,
                        dynamic_objfifo_lowering=True,
                    )
                )
            else:
                args = [of_x.cons(), w_rows[row].cons(), y_rows[row].prod(), y_acc]
                workers.append(Worker(static_core, args, stack_size=8192))

    x_tap = TensorAccessPattern(
        (1, MAX_K), 0, [1, 1, 1, MAX_K], [1, 1, 1, 1]
    )
    w_words_per_col = rows * W_BLOCK_BYTES // 4
    w_taps = [
        TensorAccessPattern(
            (1, cols * w_words_per_col),
            col * w_words_per_col,
            [1, 1, 1, w_words_per_col],
            [1, 1, 1, 1],
        )
        for col in range(cols)
    ]
    y_taps = [
        TensorAccessPattern(
            (1, total_m), col * m_per_col, [1, 1, 1, m_per_col], [1, 1, 1, 1]
        )
        for col in range(cols)
    ]

    def sequence(Y, W, X, x_handles, w_handles, y_handles):
        if dynamic_counts:
            for rtp in rtps:
                rtp[0] = 1
                rtp[1] = 1
            for barrier in barriers:
                barrier.set(1)
        x_tasks = [h.fill(X, x_tap, wait=False, managed=False) for h in x_handles]
        tg = TaskGroup()
        for col in range(cols):
            y_handles[col].drain(Y, y_taps[col], wait=True, group=tg)
            w_handles[col].fill(W, w_taps[col], wait=False, group=tg)
        tg.finish()
        for task in x_tasks:
            task.free()

    rt = Runtime(sequence, [y_host_ty, w_host_ty, x_host_ty, x_prods, w_prods, y_conss])
    return Program(dev, rt, workers=workers).resolve_program()
