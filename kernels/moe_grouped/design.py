# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Static IRON device design for eight sequential expert slots."""

from __future__ import annotations

import numpy as np

from kernels.moe_grouped.layout import (
    COLS,
    ROWS,
    ROWS_WORD_TEMPLATE,
    SLOTS,
    TMAX,
    GroupedShape,
    get_shape,
    resource_budget,
    stream_groups,
)

try:
    from ml_dtypes import bfloat16
    bf16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    bf16 = np.uint16

_IMPORT_ERROR: Exception | None = None
try:
    from aie.helpers.npdtypes import np_ndarray_type_get_shape
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, ObjectFifo, Program, Runtime, TaskGroup, Worker, WorkerRuntimeBarrier
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as error:
    _IMPORT_ERROR = error
    NPU_AVAILABLE = False
    range_ = range


def check_design(shape: str | GroupedShape) -> GroupedShape:
    resolved = get_shape(shape)
    budget = resource_budget()
    if int(budget["l1_used_bytes"]) > int(budget["l1_limit_bytes"]):
        raise ValueError("grouped core exceeds L1 budget")
    return resolved


def moe_grouped(dev=None, shape_name="b1", *, grouped_kernel=None):
    """Build a static stream; mode RTP is read only after the runtime barrier."""
    if not NPU_AVAILABLE:
        raise RuntimeError(f"AIE/IRON unavailable: {_IMPORT_ERROR}")
    if dev is None:
        import aie.utils as aie_utils
        dev = aie_utils.get_current_device()
    shape = check_design(shape_name)
    if grouped_kernel is None:
        from kernels.moe_grouped.kernel import grouped_kernel as make_kernel
        grouped_kernel = make_kernel(shape.mode)

    w_blk_ty, x_ty, acc_ty, sums_ty, *_ = grouped_kernel.arg_types()
    y_blk_ty, _, _ = grouped_kernel.epilogue.arg_types()
    (w_elems,) = np_ndarray_type_get_shape(w_blk_ty)
    w_col_ty = np.ndarray[(ROWS, w_elems), bf16]
    y_col_ty = np.ndarray[(ROWS, TMAX, 32), bf16]
    rtp_ty = np.ndarray[(7,), np.dtype[np.int32]]
    tiles = [Tile(col, 0) for col in range(COLS)]

    def core(x_in, w_in, y_out, acc, sums, rtp, barrier, zero, block, epilogue):
        barrier.wait_for_value(1)
        k_blocks = rtp[0]
        rounds = rtp[1]
        mode = rtp[2]
        slots = rtp[3]
        rows = rtp[4]  # real rows of the 8 slots, 4 bits each (layout.ROWS_WORD_TEMPLATE)
        extra = rtp[5]  # balanced shape: 1 extra round after all slots (layout.stream_groups), else 0
        extra_slot = rtp[6]  # the slot whose last chunk this column takes in the extra round
        barrier.release_with_value(1)
        for _slot in range_(slots):
            for _round in range_(rounds):
                zero(acc, TMAX)
                for _k in range_(k_blocks):
                    x = x_in.acquire(1)
                    w = w_in.acquire(1)
                    block(w, x, acc, sums, rows, mode, _round, _k, _slot)
                    w_in.release(1)
                    x_in.release(1)
                y = y_out.acquire(1)
                epilogue(y, acc, TMAX)
                y_out.release(1)
        # The extra round passes round 0 (_e) so q4nx recomputes that slot's group sums from the same X.
        for _e in range_(extra):
            zero(acc, TMAX)
            for _k in range_(k_blocks):
                x = x_in.acquire(1)
                w = w_in.acquire(1)
                block(w, x, acc, sums, rows, mode, _e, _k, extra_slot)
                w_in.release(1)
                x_in.release(1)
            y = y_out.acquire(1)
            epilogue(y, acc, TMAX)
            y_out.release(1)

    x_fifos = [ObjectFifo(x_ty, depth=2, name=f"group_x_{c}") for c in range(COLS)]
    x_prods = [fifo.prod(tile=tiles[c]) for c, fifo in enumerate(x_fifos)]
    w_prods, y_conses, workers, rtps, barriers = [], [], [], [], []
    for col in range(COLS):
        of_w = ObjectFifo(w_col_ty, depth=2, name=f"group_w_{col}")
        w_prods.append(of_w.prod(tile=tiles[col]))
        w_core = of_w.cons().split(
            [row * w_elems for row in range(ROWS)], obj_types=[w_blk_ty] * ROWS,
            names=[f"group_w_{col}_{row}" for row in range(ROWS)],
        )
        of_y = ObjectFifo(y_col_ty, depth=2, name=f"group_y_{col}")
        y_conses.append(of_y.cons(tile=tiles[col]))
        y_core = of_y.prod().join(
            [row * TMAX * 32 for row in range(ROWS)], obj_types=[y_blk_ty] * ROWS,
            names=[f"group_y_{col}_{row}" for row in range(ROWS)],
        )
        col_rtps, col_barriers = [], []
        for row in range(ROWS):
            acc = Buffer(type=acc_ty, name=f"group_acc_{col}_{row}")
            sums = Buffer(type=sums_ty, name=f"group_sums_{col}_{row}")
            rtp = Buffer(type=rtp_ty, name=f"group_rtp_{col}_{row}",
                         initial_value=np.zeros(7, dtype=np.int32), use_write_rtp=True)
            barrier = WorkerRuntimeBarrier()
            col_rtps.append(rtp); col_barriers.append(barrier)
            workers.append(Worker(
                core,
                [x_fifos[col].cons(), w_core[row].cons(), y_core[row].prod(),
                 acc, sums, rtp, barrier, grouped_kernel.zero, grouped_kernel,
                 grouped_kernel.epilogue],
                stack_size=12288,
            ))
        rtps.append(col_rtps); barriers.append(col_barriers)

    y_host_ty = np.ndarray[(SLOTS, TMAX, shape.m), bf16]
    w_host_ty = np.ndarray[(shape.weight_words,), np.dtype[np.uint32]]
    x_host_ty = np.ndarray[(SLOTS, TMAX, shape.k), bf16]
    rounds = shape.rounds_by_col

    def sequence(Y, W, X, x_ps, w_ps, y_cs):
        for col in range(COLS):
            for row in range(ROWS):
                # words 5-6 first: patch.find_rows_words wants words 0-3 right before the rows word, the barrier after
                rtps[col][row][5] = 1 if shape.balanced else 0
                rtps[col][row][6] = col
                rtps[col][row][0] = shape.k_blocks
                rtps[col][row][1] = rounds[col]
                rtps[col][row][2] = shape.mode
                rtps[col][row][3] = SLOTS
                rtps[col][row][4] = ROWS_WORD_TEMPLATE - (1 << 32)  # int32 bits: 8 rows in every slot
                barriers[col][row].set(1)
        live = []
        for records in stream_groups(shape):
            group = TaskGroup()
            for rec in records:
                col = rec["col"]
                # The host stores each slot's X pre-tiled as [k_block][32][TMAX][8], so one contiguous run per
                # transfer gives the cores the bytes of the former 16-byte walk over [SLOTS, TMAX, K].
                x_tap = TensorAccessPattern(
                    (SLOTS, TMAX, shape.k), rec["x_offset_bf16"],
                    [1, 1, 1, TMAX * shape.k], [0, 0, 0, 1],
                )
                w_tap = TensorAccessPattern(
                    (shape.weight_words,), rec["w_offset_bytes"] // 4,
                    [1, 1, 1, shape.chunk_bytes // 4], [0, 0, 0, 1],
                )
                y_tap = TensorAccessPattern(
                    (SLOTS, TMAX, shape.m), rec["y_offset_bf16"],
                    [1, ROWS, TMAX, 32], [0, 32, shape.m, 1],
                )
                y_cs[col].drain(Y, y_tap, wait=True, group=group)
                w_ps[col].fill(W, w_tap, wait=False, group=group)
                x_ps[col].fill(X, x_tap, wait=False, group=group)
            live.append(group)
            if len(live) > 2:
                live.pop(0).finish()
        for group in live:
            group.finish()

    runtime = Runtime(
        sequence,
        [y_host_ty, w_host_ty, x_host_ty] + [x_prods, w_prods, y_conses],
    )
    return Program(dev, runtime, workers=workers).resolve_program()
