# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Static IRON design for the isolated prefill mmul xclbin.

X is staged as ``[32, MAX_BATCH_T, 8]`` objects.  Every execution computes all ``MAX_BATCH_T`` (24) rows;
the caller zero-pads a short final chunk.  The runtime instruction stream writes
shape-dependent K-block and round counts to RTPs.  W is
split across four rows exactly like the existing static GEMV designs; Y is
scattered to a padded ``[max_batch_t, M]`` host BO, whose first ``batch_t``
rows are the logical contiguous output.  Column 7 is the prefill RDA channel
that updates ``rda_tokens`` tokens per call (bench/rda_state/rda_state_multi.py,
DESIGN-prefill.md section 36): one core program, one instruction stream per T.
"""

from __future__ import annotations

import numpy as np

from kernels.gemv_unified.shapes import (
    COLS,
    K_TILE,
    M_TILE,
    ROWS,
)

from kernels.gemm_prefill.shapes import MAX_BATCH_T, BatchShape, get_shape, validate_batch_t
from kernels.gemm_prefill.budget import resource_budget
from bench.rda_state.rda_state_1col import MODEL_RDA_HEADS, TILES_PER_HEAD
from bench.rda_state.rda_state_multi import (
    build_rda_multi_channel,
    check_rda_tokens,
    rda_multi_sequence_body,
)

try:
    from ml_dtypes import bfloat16

    bf16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    bf16 = np.uint16

_IMPORT_ERROR: Exception | None = None
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
        WorkerRuntimeBarrier,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    from kernels.gemv_unified.w_dma import declare_runtime_buffers

    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False

    class AIEArch:
        AIE2p = "AIE2p"

    Buffer = ObjectFifo = Program = Runtime = TaskGroup = Worker = object
    WorkerRuntimeBarrier = object
    TensorAccessPattern = Tile = object
    np_ndarray_type_get_shape = object
    range_ = range
    declare_runtime_buffers = None

DEPTH = 2


def check_design(
    dev,
    shape: str | BatchShape,
    batch_t: int,
    max_batch_t: int,
    cols: int,
) -> BatchShape:
    resolved = get_shape(shape)
    validate_batch_t(batch_t, max_batch_t=max_batch_t)
    if not 1 <= cols <= COLS:
        raise ValueError(f"cols must be in 1..{COLS}, got {cols}")
    if dev is not None and getattr(dev, "arch", AIEArch.AIE2p) != AIEArch.AIE2p:
        raise NotImplementedError("batch GEMV is AIE2P only")
    return resolved


def gemm_prefill(
    dev=None,
    shape_name: str = "rda_in",
    batch_t: int = MAX_BATCH_T,
    max_batch_t: int = MAX_BATCH_T,
    cols: int = COLS,
    trace_size: int = 0,
    *,
    gemv_kernel=None,
    variant="exact",
    mode=0,
    rda_col=None,
    execution="gemm",
    rda_tokens=1,
):
    """Build one static prefill instruction stream and experimental xclbin."""
    if not NPU_AVAILABLE:
        raise RuntimeError(f"AIE/IRON packages are unavailable: {_IMPORT_ERROR}")
    if dev is None:
        import aie.utils as aie_utils

        dev = aie_utils.get_current_device()
    shape = check_design(dev, shape_name, batch_t, max_batch_t, cols)
    if execution not in ("gemm", "rda"):
        raise ValueError("execution must be 'gemm' or 'rda'")
    if rda_col is not None and (cols != 7 or rda_col != 7):
        raise ValueError("the prototype partition is exactly GEMM columns 0..6 plus RDA column 7")
    if execution == "rda" and rda_col is None:
        raise ValueError("rda execution requires rda_col=7")
    if execution == "rda":
        check_rda_tokens(rda_tokens)
    if mode not in (0, 1, 2, 3):
        raise ValueError("mode must be 0..3")
    if not resource_budget(shape, max_batch_t, cols=cols).limits_ok:
        raise ValueError("resource budget exceeded")
    if gemv_kernel is None:
        from kernels.gemm_prefill.kernel import prefill_kernel

        gemv_kernel = prefill_kernel(max_batch_t, variant, shape.format)

    total_chunks = shape.M // (ROWS * M_TILE)
    active_cols = min(cols, total_chunks)
    col_rounds = tuple(
        total_chunks // active_cols + (1 if col < total_chunks % active_cols else 0)
        if col < active_cols else 0
        for col in range(cols)
    )

    w_blk_ty, x_ty, y_acc_ty, _, _ = gemv_kernel.arg_types()
    y_blk_ty, _, _ = gemv_kernel.epilogue.arg_types()
    (w_blk_elems,) = np_ndarray_type_get_shape(w_blk_ty)
    w_col_ty = np.ndarray[(ROWS, w_blk_elems), bf16]
    y_col_ty = np.ndarray[(ROWS, max_batch_t, M_TILE), bf16]
    rtp_ty = np.ndarray[(3,), np.dtype[np.int32]]
    interface_tiles = [Tile(col, 0) for col in range(cols)]

    def core(x_in, w_in, y_out, y_acc, rtp,
             k_zero, k_block, k_epilogue, barrier):
        barrier.wait_for_value(1)
        k_blocks = rtp[0]
        rounds = rtp[1]
        diagnostic = rtp[2]
        barrier.release_with_value(1)
        actual_t = max_batch_t
        for _ in range_(rounds):
            k_zero(y_acc, actual_t)
            for _k in range_(k_blocks):
                x = x_in.acquire(1)
                w = w_in.acquire(1)
                k_block(w, x, y_acc, actual_t, diagnostic)
                w_in.release(1)
                x_in.release(1)
            y = y_out.acquire(1)
            k_epilogue(y, y_acc, actual_t)
            y_out.release(1)

    x_fifos = [
        ObjectFifo(x_ty, depth=2, name=f"batch_x_{col}")
        for col in range(cols)
    ]
    x_prods = [
        fifo.prod(tile=interface_tiles[col]) for col, fifo in enumerate(x_fifos)
    ]
    workers = []
    w_prods = []
    y_conses = []
    rtps = []
    barriers = []
    for col in range(cols):
        of_w = ObjectFifo(w_col_ty, depth=2, name=f"batch_w_{col}")
        w_prods.append(of_w.prod(tile=interface_tiles[col]))
        w_cores = of_w.cons().split(
            [row * w_blk_elems for row in range(ROWS)],
            obj_types=[w_blk_ty] * ROWS,
            names=[f"batch_w_{col}_{row}" for row in range(ROWS)],
        )
        of_y = ObjectFifo(y_col_ty, depth=2, name=f"batch_y_{col}")
        y_conses.append(of_y.cons(tile=interface_tiles[col]))
        y_cores = of_y.prod().join(
            [row * max_batch_t * M_TILE for row in range(ROWS)],
            obj_types=[y_blk_ty] * ROWS,
            names=[f"batch_y_{col}_{row}" for row in range(ROWS)],
        )
        col_rtps = []
        col_barriers = []
        for row in range(ROWS):
            y_acc = Buffer(type=y_acc_ty, name=f"batch_acc_{col}_{row}")
            rtp = Buffer(
                type=rtp_ty,
                name=f"batch_t_{col}_{row}",
                initial_value=np.zeros(3, dtype=np.int32),
                use_write_rtp=True,
            )
            col_rtps.append(rtp)
            barrier = WorkerRuntimeBarrier()
            col_barriers.append(barrier)
            args = [
                x_fifos[col].cons(),
                w_cores[row].cons(),
                y_cores[row].prod(),
                y_acc,
            ]
            args.extend([rtp, gemv_kernel.zero,
                         gemv_kernel, gemv_kernel.epilogue, barrier])
            workers.append(Worker(core, args, stack_size=resource_budget(shape, max_batch_t, cols=cols).stack_bytes))
        rtps.append(col_rtps)
        barriers.append(col_barriers)

    rda_channel = None
    if rda_col is not None:
        rda_channel = build_rda_multi_channel(rda_col, MODEL_RDA_HEADS)
        workers.extend(rda_channel.workers)

    y_host_ty = np.ndarray[(max_batch_t, shape.M), bf16]
    w_host_ty = np.ndarray[(shape.weight_words,), np.dtype[np.uint32]]
    x_host_ty = np.ndarray[(max_batch_t, shape.K), bf16]
    w_words_per_chunk = shape.unified.w_chunk_words
    w_taps = [
        TensorAccessPattern(
            (shape.weight_words,),
            chunk * w_words_per_chunk,
            [1, 1, 1, w_words_per_chunk],
            [0, 0, 0, 1],
        )
        for chunk in range(total_chunks)
    ]
    # Rank-4 traversal: [K tile][K8 group][token][8 contiguous bf16].
    # Innermost transfers are 16 B and every stride is 4-byte aligned.
    x_tap = TensorAccessPattern(
        (max_batch_t, shape.K),
        0,
        [shape.k_blocks, K_TILE // 8, max_batch_t, 8],
        [K_TILE, 8, shape.K, 1],
    )
    y_taps = [
        TensorAccessPattern(
            (max_batch_t, shape.M),
            chunk * ROWS * M_TILE,
            [1, ROWS, max_batch_t, M_TILE],
            [0, M_TILE, shape.M, 1],
        )
        for chunk in range(total_chunks)
    ]

    def sequence(Y, W, X, x_ps, w_ps, y_cs, s_in_prod=None, head_in_prod=None):
        if rda_channel is not None:
            declare_runtime_buffers(rda_channel.agg_mem_buffers)
        for col in range(cols):
            for row in range(ROWS):
                rtps[col][row][0] = shape.k_blocks
                rtps[col][row][1] = col_rounds[col]
                rtps[col][row][2] = mode
                barriers[col][row].set(1)
        window = []
        max_rounds = max(col_rounds)
        for rnd in range(max_rounds):
            tg = TaskGroup()
            for col in range(active_cols):
                if rnd >= col_rounds[col]:
                    continue
                chunk = rnd * active_cols + col
                y_cs[col].drain(Y, y_taps[chunk], wait=True, group=tg)
                w_ps[col].fill(W, w_taps[chunk], wait=False, group=tg)
                x_ps[col].fill(X, x_tap, wait=False, group=tg)
            window.append(tg)
            if len(window) > DEPTH:
                window.pop(0).finish()
        for tg in window:
            tg.finish()

    shim_handles = [x_prods, w_prods, y_conses]
    if rda_channel is not None:
        shim_handles.extend([rda_channel.s_in_prod, rda_channel.head_in_prod])
    if execution == "gemm":
        runtime = Runtime(sequence, [y_host_ty, w_host_ty, x_host_ty] + shim_handles)
    else:
        def rda_sequence(s_in, s_out, head_in, out, x_ps, w_ps, y_cs,
                         s_in_prod, head_in_prod):
            declare_runtime_buffers(rda_channel.agg_mem_buffers)
            rda_multi_sequence_body(
                rda_channel, s_in, s_out, head_in, out, s_in_prod, head_in_prod,
                rda_tokens,
            )

        runtime = Runtime(
            rda_sequence,
            [rda_channel.s_l3_ty, rda_channel.s_l3_ty,
             rda_channel.head_in_l3_ty(rda_tokens),
             rda_channel.out_l3_ty(rda_tokens)] + shim_handles,
        )
    if rda_channel is not None:
        for slot in range(2):
            for tile_index in range(TILES_PER_HEAD):
                runtime.add_lock(rda_channel.agg_empty[slot][tile_index])
                runtime.add_lock(rda_channel.agg_full[slot][tile_index])
        for tile_index in range(TILES_PER_HEAD):
            runtime.add_lock(rda_channel.core_empty[tile_index])
            runtime.add_lock(rda_channel.core_full[tile_index])
            runtime.add_flow(rda_channel.flows_c2m[tile_index])
        runtime.add_flow(rda_channel.flow_state_m2s)
        runtime.add_flow(rda_channel.flow_out_m2s)
    program = Program(dev, runtime, workers=workers)
    if trace_size > 0:
        program.enable_trace(trace_size)
    return program.resolve_program()


gemv = gemm_prefill
