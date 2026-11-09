# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Unified stage-1 9-bit bfp16ebs8 GEMV IRON design.

Fixed 32-core (8 columns x 4 rows) device configuration shared across all 10
stage-1 shapes:
  - Per-column x ObjectFifo: consumer depth 1, max_K = 12,032 bf16 (24,064 B).
    Runtime X buffers are padded to that full extent: a short shim transfer cannot
    complete the fixed-size ObjectFifo receiver BD and leaves workers blocked.
  - Per-column W path: explicit shim->mem->core Flows and fixed-size ping/pong
    Buffers (9,216 B x 2/core).  Runtime DMA tasks configure the receiving BDs
    to 9,216 B for bfp16ebs8 or 5,120 B for q4nx in each instruction stream.
    These mem/core tasks issue no completion token; they are freed after the
    Y drain await of the last round they serve (see the runtime sequence
    comment).  Memory-tile W tasks restart each round by default; the fix4
    cross-round chain is retained behind ``continuous_mem_w``.
  - Per-column y ObjectFifo: depth 2, joined across 4 rows (64 B x 2 = 128 B/core).
  - Per-core y_acc buffer: 32 x float32 (128 B).
  - Per-core RTP buffer: 3 x int32 (format, k_blocks, rounds) + WorkerRuntimeBarrier.
  - Dynamic ObjectFifo lowering (`dynamic_objfifo_lowering=True` + `--dynamic-objFifos`)
    so the core loops stay rolled and buffer ping/pong state persists across
    odd-round dispatches.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from kernels.gemv_unified.kernel import unified_bfp16_kernel
from kernels.gemv_unified.dma_layout import gemv_dma_tap_specs, per_call_shim_tap_specs
from kernels.gemv_unified.shapes import (
    COLS,
    FORMAT_BFP16,
    FORMAT_Q4NX,
    M_TILE,
    MAX_K,
    ROWS,
    RTP_FORMAT,
    RTP_K_BLOCKS,
    RTP_ROUNDS,
    RTP_WORDS,
    W_BLOCK_BYTES,
    UnifiedShape,
    x_slot_offset,
)
from kernels.gemv_unified.routing import W_TASKS_ISSUE_TOKEN, unified_stream_topology
from kernels.gemv_unified.w_plan import (
    W_MEM_BUFFER_ELEMS,
    declared_memtile_resources,
    drive_rounds,
    mem_chunk_plan,
    per_call_core_chunks,
    round_window_depth,
)
from bench.rda_state.rda_state_1col import (
    RDAChannel,
    TILES_PER_HEAD,
    build_rda_channel,
    rda_sequence_body,
)

_IMPORT_ERROR: Exception | None = None
try:
    from ml_dtypes import bfloat16

    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import Buffer, Flow, Lock, ObjectFifo, Program, Runtime, TaskGroup, Worker, WorkerRuntimeBarrier
    from aie.iron.controlflow import else_, if_, range_
    from aie.iron.device import NPU2, Tile
    from aie.utils import TraceConfig
    from kernels.gemv_unified.w_dma import (
        configure_chunk_tasks,
        configure_core_call_tasks,
        configure_ping_pong_tasks,
        declare_runtime_buffers,
        finish_all,
        start_all,
        tap_from_spec,
    )
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:  # pragma: no cover - CPU-only host lacks IRON runtime
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False
    bfloat16 = None  # type: ignore[assignment]
    TensorAccessPattern = Buffer = Flow = Lock = ObjectFifo = Program = Runtime = TaskGroup = Worker = WorkerRuntimeBarrier = object
    else_ = if_ = range_ = NPU2 = Tile = TraceConfig = object
    configure_chunk_tasks = configure_core_call_tasks = configure_ping_pong_tasks = declare_runtime_buffers = finish_all = start_all = tap_from_spec = None

STACK_SIZE = 10 * 1024
DEPTH = 2
W_ELEMS_PER_BLOCK = W_BLOCK_BYTES // 2  # 4608 bf16-sized elements (9216 B)


def _tap_from_spec(spec):
    """Build a TAP, retaining the lightweight fake used by host-only tests."""
    if tap_from_spec is not None:
        return tap_from_spec(spec)
    return TensorAccessPattern(spec.tensor_dims, spec.offset, spec.sizes, spec.strides)


def check_shape(
    dev,
    K: int,
    M: int,
    cols: int = COLS,
    *,
    format: int = FORMAT_BFP16,
    active_cols: int | None = None,
    x_slots: int = 1,
    chunks_per_col: int | None = None,
) -> UnifiedShape:
    """Validate device and shape constraints and return a UnifiedShape descriptor."""
    if dev is not None and NPU2 is not None and not isinstance(dev, NPU2):
        raise ValueError(f"Only AIE2p (NPU2) is supported, got {type(dev)}")
    if not 1 <= cols <= COLS:
        raise ValueError(f"Unified GEMV cols must be in 1..{COLS}, got {cols}")
    if active_cols is None:
        active_cols = cols
    if not 1 <= active_cols <= cols:
        raise ValueError(
            f"Unified GEMV active_cols must be in 1..cols, got active_cols={active_cols}, cols={cols}"
        )
    return UnifiedShape(
        name=f"k{K}_m{M}", K=K, M=M, format=format,
        active_cols=active_cols, x_slots=x_slots, chunks_per_col=chunks_per_col,
    )


def gemv_unified(
    dev=None,
    K: int = 2560,
    M: int = 2560,
    cols: int = COLS,
    trace_size: int = 0,
    *,
    gemv_kernel=None,
    format: int = FORMAT_BFP16,
    active_cols: int | None = None,
    x_slots: int = 1,
    chunks_per_col: int | None = None,
    continuous_mem_w: bool = False,
    sequence_plan: str = "per_round",
    diagnostic_receive_only: bool = False,
    diagnostic_static_core: bool = False,
    diagnostic_small_x: bool = False,
    rda_col: int | None = None,
    rda_heads: int = 40,
    mode: str = "gemv",
):
    """Build the unified stage-1 GEMV IRON Program for shape (K, M).

    The device body (Tiles, ObjectFifos, Buffers, Workers, core_fn) is invariant
    with respect to (K, M); only the Runtime sequence (RTP writes, barrier sets,
    and DMA TAPs) varies with (K, M).
    """
    if dev is None:
        try:
            import aie.utils as aie_utils
            dev = aie_utils.get_current_device()
        except Exception:
            pass

    if dev is not None:
        try:
            import aie.utils.hostruntime as hostruntime
            hostruntime.set_current_device(dev)
        except Exception:
            pass

    shape = check_shape(
        dev, K, M, cols, format=format, active_cols=active_cols,
        x_slots=x_slots, chunks_per_col=chunks_per_col,
    )

    if sequence_plan not in ("per_round", "per_call"):
        raise ValueError(
            f"sequence_plan must be 'per_round' or 'per_call', got {sequence_plan!r}"
        )
    if mode not in ("gemv", "rda"):
        raise ValueError(f"mode must be 'gemv' or 'rda', got {mode!r}")
    if rda_col is not None:
        if not (0 <= rda_col < COLS):
            raise ValueError(
                f"rda_col must be within device columns 0..{COLS - 1}, got {rda_col}"
            )
        if rda_col < cols:
            raise ValueError(
                f"rda_col {rda_col} overlaps GEMV active columns 0..{cols - 1}"
            )
        if rda_heads <= 0:
            raise ValueError(f"rda_heads must be positive, got {rda_heads}")
    elif mode == "rda":
        raise ValueError("rda mode requires rda_col to be set")

    if gemv_kernel is None:
        gemv_kernel = unified_bfp16_kernel(
            K=shape.K if diagnostic_small_x else None
        )
    if isinstance(gemv_kernel, (list, tuple)):
        k_zero, k_block, k_epi = gemv_kernel
        q_prep = q_zero = q_block = q_epi = None
    else:
        k_zero = getattr(gemv_kernel, "bfp16_gemv_unified_zero", getattr(gemv_kernel, "bfp16_gemv_zero", None))
        k_block = getattr(gemv_kernel, "bfp16_gemv_unified_block", getattr(gemv_kernel, "bfp16_gemv_block", gemv_kernel))
        k_epi = getattr(gemv_kernel, "bfp16_gemv_unified_epilogue", getattr(gemv_kernel, "bfp16_gemv_epilogue", None))
        q_prep = gemv_kernel.q4nx_gemv_unified_prep
        q_zero = gemv_kernel.q4nx_gemv_unified_zero
        q_block = gemv_kernel.q4nx_gemv_unified_block
        q_epi = gemv_kernel.q4nx_gemv_unified_epilogue

    max_rounds = shape.max_rounds
    col_rounds = shape.rounds_by_col()
    K_BLOCKS = shape.k_blocks
    W_TOTAL_WORDS = shape.weight_words
    dma_specs = gemv_dma_tap_specs(shape, cols)

    # Fixed device buffer types (shape-independent outside diagnostic mode).
    x_elems = shape.K if diagnostic_small_x else dma_specs["x_core"][0].sizes[0]
    x_ty = np.ndarray[(x_elems,), np.dtype[bfloat16]]
    w_blk_ty = np.ndarray[(W_ELEMS_PER_BLOCK,), np.dtype[bfloat16]]
    w_col_ty = np.ndarray[(W_MEM_BUFFER_ELEMS,), np.dtype[bfloat16]]
    y_blk_ty = np.ndarray[(dma_specs["y_core"][0].sizes[0],), np.dtype[bfloat16]]
    y_col_ty = np.ndarray[(dma_specs["y_shim"][0].sizes[0],), np.dtype[bfloat16]]
    y_acc_ty = np.ndarray[(M_TILE,), np.dtype[np.float32]]
    sums_ty = np.ndarray[
        (shape.K // 32 if diagnostic_small_x else MAX_K // 32,), np.dtype[bfloat16]
    ]
    rtp_ty = np.ndarray[(RTP_WORDS,), np.dtype[np.int32]]

    # Host tensor types (shape-dependent)
    # The ObjectFifo receiver BD is fixed at MAX_K elements by the shared xclbin.
    # Sending only K elements completes the shim BD but not that receiver BD, so
    # its lock is never released and every worker hangs in x_in.acquire(1).
    X_host_ty = np.ndarray[(shape.x_slots * x_elems,), np.dtype[bfloat16]]
    W_host_ty = np.ndarray[(W_TOTAL_WORDS,), np.dtype[np.uint32]]
    Y_host_ty = np.ndarray[(M,), np.dtype[bfloat16]]

    IT = [Tile(j, 0) for j in range(cols)]
    MT = [Tile(j, 1) for j in range(cols)]
    CT = [[Tile(j, i + 2) for i in range(ROWS)] for j in range(cols)]
    stream_paths = {path.name: path for path in unified_stream_topology(cols)}

    # Per-column x FIFO: consumer depth 1, max_K = 12,032 bf16
    x_paths = [stream_paths[f"x{j}"] for j in range(cols)]
    of_x_list = [ObjectFifo(x_ty, depth=1, name=path.name) for path in x_paths]
    x_prods = [
        of_x_list[j].prod(tile=IT[path.source.col])
        for j, path in enumerate(x_paths)
    ]

    of_y_conss = []
    w_in_flows = []
    w_out_flows = []
    w_mem_buffers = []
    w_mem_empty_locks = []
    w_mem_full_locks = []
    w_core_buffers = [[None for _ in range(ROWS)] for _ in range(cols)]
    w_core_empty_locks = [[None for _ in range(ROWS)] for _ in range(cols)]
    w_core_full_locks = [[None for _ in range(ROWS)] for _ in range(cols)]
    rtps = [[None for _ in range(ROWS)] for _ in range(cols)]
    barriers = [[None for _ in range(ROWS)] for _ in range(cols)]
    workers = []
    memtile_resources = {
        resource.name: resource for resource in declared_memtile_resources(cols)
    }

    def dynamic_core_fn(
        x_in, w_ping, w_pong, w_ping_empty, w_pong_empty,
        w_ping_full, w_pong_full, y_out, y_acc, sums, my_rtp,
        k_zero, k_block, k_epi, q_prep, q_zero, q_block, q_epi, barrier,
    ):
        barrier.wait_for_value(1)
        _fmt = my_rtp[RTP_FORMAT]
        k_blocks = my_rtp[RTP_K_BLOCKS]
        rounds = my_rtp[RTP_ROUNDS]
        barrier.release_with_value(1)

        x = x_in.acquire(1)
        if diagnostic_receive_only:
            for _ in range_(rounds):
                for k in range_(0, k_blocks, 2):
                    w_ping_full.acquire()
                    w_ping_empty.release()
                    with if_(k + 1 < k_blocks):
                        w_pong_full.acquire()
                        w_pong_empty.release()
                y = y_out.acquire(1)
                for index in range(M_TILE):
                    y[index] = 0
                y_out.release(1)
            x_in.release(1)
            return

        with if_(_fmt == FORMAT_Q4NX) as q4_branch:
            q_prep(x, sums, k_blocks)
            for _ in range_(rounds):
                q_zero(y_acc)
                for k in range_(0, k_blocks, 2):
                    w_ping_full.acquire()
                    q_block(w_ping, x, y_acc, sums, k)
                    w_ping_empty.release()
                    with if_(k + 1 < k_blocks):
                        w_pong_full.acquire()
                        q_block(w_pong, x, y_acc, sums, k + 1)
                        w_pong_empty.release()
                y = y_out.acquire(1)
                q_epi(y, y_acc)
                y_out.release(1)
        with else_(q4_branch):
            for _ in range_(rounds):
                k_zero(y_acc)
                for k in range_(0, k_blocks, 2):
                    w_ping_full.acquire()
                    k_block(w_ping, x, y_acc, k)
                    w_ping_empty.release()
                    with if_(k + 1 < k_blocks):
                        w_pong_full.acquire()
                        k_block(w_pong, x, y_acc, k + 1)
                        w_pong_empty.release()
                y = y_out.acquire(1)
                k_epi(y, y_acc)
                y_out.release(1)
        x_in.release(1)

    def make_static_core(rounds_for_col: int):
        """Create a Python-unrolled ping/pong core for stall isolation."""

        def static_core_fn(
            x_in, w_ping, w_pong, w_ping_empty, w_pong_empty,
            w_ping_full, w_pong_full, y_out, y_acc, sums, my_rtp,
            k_zero, k_block, k_epi, q_prep, q_zero, q_block, q_epi, barrier,
        ):
            barrier.wait_for_value(1)
            barrier.release_with_value(1)
            x = x_in.acquire(1)
            if shape.format == FORMAT_Q4NX and not diagnostic_receive_only:
                q_prep(x, sums, K_BLOCKS)
            for _ in range(rounds_for_col):
                if diagnostic_receive_only:
                    pass
                elif shape.format == FORMAT_Q4NX:
                    q_zero(y_acc)
                else:
                    k_zero(y_acc)
                for k in range(K_BLOCKS):
                    if k % 2:
                        w_pong_full.acquire()
                        if not diagnostic_receive_only:
                            if shape.format == FORMAT_Q4NX:
                                q_block(w_pong, x, y_acc, sums, k)
                            else:
                                k_block(w_pong, x, y_acc, k)
                        w_pong_empty.release()
                    else:
                        w_ping_full.acquire()
                        if not diagnostic_receive_only:
                            if shape.format == FORMAT_Q4NX:
                                q_block(w_ping, x, y_acc, sums, k)
                            else:
                                k_block(w_ping, x, y_acc, k)
                        w_ping_empty.release()
                y = y_out.acquire(1)
                if diagnostic_receive_only:
                    for index in range(M_TILE):
                        y[index] = 0
                elif shape.format == FORMAT_Q4NX:
                    q_epi(y, y_acc)
                else:
                    k_epi(y, y_acc)
                y_out.release(1)
            x_in.release(1)

        return static_core_fn

    for j in range(cols):
        w_in_path = stream_paths[f"w_in_{j}"]
        w_out_paths = [stream_paths[f"w_out_{j}_{row}"] for row in range(ROWS)]
        w_in_flows.append(
            Flow(
                IT[w_in_path.source.col],
                MT[w_in_path.destinations[0].col],
                name=w_in_path.name,
            )
        )
        w_out_flows.append(
            [
                Flow(
                    MT[path.source.col],
                    CT[path.destinations[0].col][path.destinations[0].row - 2],
                    name=path.name,
                )
                for path in w_out_paths
            ]
        )
        mem_buffers = [
            Buffer(type=w_col_ty, name=f"w_mem_{j}_{slot}", tile=MT[j])
            for slot in range(DEPTH)
        ]
        mem_empty = [
            Lock(
                MT[j],
                init=memtile_resources[f"w_mem_empty_{j}_{slot}"].init,
                name=f"w_mem_empty_{j}_{slot}",
            )
            for slot in range(DEPTH)
        ]
        mem_full = [
            Lock(
                MT[j],
                init=memtile_resources[f"w_mem_full_{j}_{slot}"].init,
                name=f"w_mem_full_{j}_{slot}",
            )
            for slot in range(DEPTH)
        ]
        w_mem_buffers.append(mem_buffers)
        w_mem_empty_locks.append(mem_empty)
        w_mem_full_locks.append(mem_full)

        y_path = stream_paths[f"y{j}"]
        y_join_paths = [
            stream_paths[f"y{j}_{row}"] for row in range(ROWS)
        ]
        of_y = ObjectFifo(y_col_ty, name=y_path.name)
        of_y_conss.append(of_y.cons(tile=IT[j]))
        y_cores = of_y.prod().join(
            [r * M_TILE for r in range(ROWS)],
            tile=MT[y_path.source.col],
            obj_types=[y_blk_ty] * ROWS,
            names=[path.name for path in y_join_paths],
        )

        for i in range(ROWS):
            core_buffers = [
                Buffer(type=w_blk_ty, name=f"w_core_{j}_{i}_{slot}", tile=CT[j][i])
                for slot in range(DEPTH)
            ]
            core_empty = [
                Lock(CT[j][i], init=1, name=f"w_core_empty_{j}_{i}_{slot}")
                for slot in range(DEPTH)
            ]
            core_full = [
                Lock(CT[j][i], init=0, name=f"w_core_full_{j}_{i}_{slot}")
                for slot in range(DEPTH)
            ]
            w_core_buffers[j][i] = core_buffers
            w_core_empty_locks[j][i] = core_empty
            w_core_full_locks[j][i] = core_full
            y_acc = Buffer(
                type=y_acc_ty,
                name=f"yacc_{i}_{j}",
            )
            sums = Buffer(type=sums_ty, name=f"sums_{i}_{j}")
            rtp = Buffer(
                type=rtp_ty,
                name=f"rtp_{i}_{j}",
                initial_value=np.zeros(RTP_WORDS, dtype=np.int32),
                use_write_rtp=True,
            )
            barrier = WorkerRuntimeBarrier()
            rtps[j][i] = rtp
            barriers[j][i] = barrier

            worker_core = (
                make_static_core(col_rounds[j])
                if diagnostic_static_core
                else dynamic_core_fn
            )
            workers.append(
                Worker(
                    worker_core,
                    [
                        of_x_list[j].cons(),
                        core_buffers[0],
                        core_buffers[1],
                        core_empty[0],
                        core_empty[1],
                        core_full[0],
                        core_full[1],
                        y_cores[i].prod(),
                        y_acc,
                        sums,
                        rtp,
                        k_zero,
                        k_block,
                        k_epi,
                        q_prep,
                        q_zero,
                        q_block,
                        q_epi,
                        barrier,
                    ],
                    tile=CT[j][i],
                    stack_size=STACK_SIZE,
                    dynamic_objfifo_lowering=True,
                )
            )

    rda_channel: RDAChannel | None = None
    if rda_col is not None:
        rda_channel = build_rda_channel(rda_col, rda_heads)
        workers.extend(rda_channel.workers)

    if diagnostic_small_x:
        x_taps = [
            TensorAccessPattern(
                (1, shape.x_slots * x_elems),
                x_slot_offset(shape, col, x_elems),
                [1, 1, 1, x_elems],
                [1, 1, 1, 1],
            )
            for col in range(cols)
        ]
    else:
        x_taps = [_tap_from_spec(spec) for spec in dma_specs["x"]]
    w_taps = [_tap_from_spec(spec) for spec in dma_specs["w_host"]]
    y_taps = [_tap_from_spec(spec) for spec in dma_specs["y"]]
    mem_in_tap = _tap_from_spec(dma_specs["mem_in"][0])
    mem_out_taps = [_tap_from_spec(spec) for spec in dma_specs["mem_out"]]
    core_in_tap = _tap_from_spec(dma_specs["core_in"][0])
    per_call_taps = {}
    if sequence_plan == "per_call":
        per_call_specs = per_call_shim_tap_specs(shape, cols)
        per_call_taps = {
            col: {
                kind: tuple(_tap_from_spec(spec) for spec in specs)
                for kind, specs in entries.items()
            }
            for col, entries in per_call_specs.items()
        }

    def sequence(
        Y,
        W,
        X,
        x_prods,
        of_y_conss,
        s_in_prod=None,
        head_in_prod=None,
        *extra_shim_handles,
    ):
        # Runtime DMA buffers are normally declared only when a task first uses
        # them.  Anchor the complete fixed set so every shape's device body is
        # identical even when its command stream only references slot zero.
        declare_runtime_buffers(
            buffer for column in w_mem_buffers for buffer in column
        )
        if rda_channel is not None:  # Keep the runtime-only buffer prologue the same in both modes
            declare_runtime_buffers(rda_channel.agg_mem_buffers)

        # 1. Write per-core RTP values before releasing worker barriers
        for col in range(cols):
            rounds_for_col = col_rounds[col]
            for row in range(ROWS):
                rtps[col][row][RTP_FORMAT] = shape.format
                rtps[col][row][RTP_K_BLOCKS] = K_BLOCKS
                rtps[col][row][RTP_ROUNDS] = rounds_for_col

        # 2. Release all worker runtime barriers
        for col in range(cols):
            for row in range(ROWS):
                barriers[col][row].set(1)

        # 3. Broadcast X to each column's consumer buffer
        x_tasks = [
            x_prods[col].fill(
                X,
                x_taps[col],
                wait=(col_rounds[col] == 0),
                managed=False,
            )
            for col in range(cols)
        ]

        # 4. Stream W chunks in and drain Y chunks out with depth-2 task groups.
        # Completion tokens come ONLY from shim tasks (x fill, W-in fill, Y
        # drain).  Memory-tile / core W tasks are created without a token:
        # ``aie-generate-column-control-overlay`` (route-shim-to-tct=shim-only)
        # would otherwise add a TileControl -> shim South packet flow per
        # token-issuing tile, i.e. 4 cores + 1 memory tile = 5 downward paths
        # per column, which with the 4 y-join streams exceeds the memory tile's
        # North source limit of 4 (5 > 4 -> "Unable to find a legal routing").
        # Their BDs are freed (no await) only after the Y drain await of the
        # last round they serve: a core emits y only after its last W block
        # arrives, the memory tile sends a block before the core receives it,
        # and receives it before sending it, so Y drain completion implies
        # every W task of that round has finished (``dma_free_task`` is only
        # safe under such synchronization).
        #
        # Optional fix4 mode: memory-tile W tasks run ACROSS rounds
        # (``w_plan.continuous_mem_chunks``): slot = cumulative block % 2, one
        # two-BD repeat task per <=255 pairs plus an odd tail, started at the
        # round of their first block and freed after the Y await of the round of
        # their last block.  Core W tasks stay per round (ping-first each round,
        # own buffers/locks; core BDs are far below the 16 limit), so the core
        # program is unchanged.  Shim W/Y tasks stay per round.
        mem_plan = mem_chunk_plan(shape, cols, continuous=continuous_mem_w)

        # A repeated core chain starts every round at ping.  Even K uses a
        # two-BD ping/pong chain; odd K uses one position-specific BD per block
        # and therefore must fit the core's 16-BD pool.
        if sequence_plan == "per_call":
            if K_BLOCKS % 2 and K_BLOCKS > 16:
                raise ValueError(
                    f"per_call core chain needs {K_BLOCKS} BDs; core limit is 16"
                )
            mem_plan = mem_chunk_plan(shape, cols, continuous=True)
            core_call_tasks = {}
            mem_call_tasks = {}
            shim_groups = {}

            for col in range(shape.active_cols):
                rounds_for_col = col_rounds[col]
                if not rounds_for_col:
                    continue
                core_chunks = per_call_core_chunks(K_BLOCKS, rounds_for_col)
                core_sets = []
                for row in range(ROWS):
                    for chunk in core_chunks:
                        core_sets.append(
                            configure_core_call_tasks(
                                w_out_flows[col][row].endpoint(CT[col][row]),
                                w_core_buffers[col][row],
                                [core_in_tap, core_in_tap],
                                w_core_empty_locks[col][row],
                                w_core_full_locks[col][row],
                                chunk=chunk,
                                issue_token=W_TASKS_ISSUE_TOKEN,
                            )
                        )
                core_call_tasks[col] = core_sets

                mem_sets = []
                for chunk in mem_plan[col]:
                    mem_sets.extend(
                        configure_chunk_tasks(
                            w_out_flows[col][row].endpoint(MT[col]),
                            w_mem_buffers[col],
                            [mem_out_taps[row], mem_out_taps[row]],
                            w_mem_full_locks[col],
                            w_mem_empty_locks[col],
                            chunk=chunk,
                            issue_token=W_TASKS_ISSUE_TOKEN,
                        )
                        for row in range(ROWS)
                    )
                    mem_sets.append(
                        configure_chunk_tasks(
                            w_in_flows[col].endpoint(MT[col]),
                            w_mem_buffers[col],
                            [mem_in_tap, mem_in_tap],
                            w_mem_empty_locks[col],
                            w_mem_full_locks[col],
                            chunk=chunk,
                            acquire_value=ROWS,
                            release_value=ROWS,
                            issue_token=W_TASKS_ISSUE_TOKEN,
                        )
                    )
                mem_call_tasks[col] = mem_sets
                start_all(core_sets + mem_sets)

                tg = TaskGroup()
                shim_groups[col] = tg
                for tap in per_call_taps[col]["w"]:
                    w_in_flows[col].fill(W, tap=tap, wait=False, group=tg)
                for tap in per_call_taps[col]["y"]:
                    of_y_conss[col].drain(Y, tap, wait=True, group=tg)

            # Completing every token-producing Y task proves all core/memory W
            # tasks in that column ended.
            for col, tg in shim_groups.items():
                tg.finish()
                finish_all(core_call_tasks[col])
                finish_all(mem_call_tasks[col])

            for col, t in enumerate(x_tasks):
                if col_rounds[col] == 0:
                    t.await_()
                t.free()
            return

        window_depth = round_window_depth(
            K_BLOCKS, continuous=continuous_mem_w
        )
        groups = {}
        core_round_tasks = {}
        mem_live = {}

        def issue_round(rnd):
            tg = TaskGroup()
            groups[rnd] = tg
            round_tasks = core_round_tasks.setdefault(rnd, [])
            for col in range(cols):
                if col >= shape.active_cols or rnd >= col_rounds[col]:
                    continue
                c = (
                    col * shape.chunks_per_col + rnd
                    if shape.chunks_per_col is not None
                    else rnd * shape.active_cols + col
                )
                core_tasks = [
                    configure_ping_pong_tasks(
                        w_out_flows[col][row].endpoint(CT[col][row]),
                        w_core_buffers[col][row],
                        [core_in_tap, core_in_tap],
                        w_core_empty_locks[col][row],
                        w_core_full_locks[col][row],
                        blocks=K_BLOCKS,
                        issue_token=W_TASKS_ISSUE_TOKEN,
                    )
                    for row in range(ROWS)
                ]
                task_sets = list(core_tasks)
                for idx, chunk in enumerate(mem_plan[col]):
                    if chunk.first_round != rnd:
                        continue
                    chunk_sets = [
                        configure_chunk_tasks(
                            w_out_flows[col][row].endpoint(MT[col]),
                            w_mem_buffers[col],
                            [mem_out_taps[row], mem_out_taps[row]],
                            w_mem_full_locks[col],
                            w_mem_empty_locks[col],
                            chunk=chunk,
                            issue_token=W_TASKS_ISSUE_TOKEN,
                        )
                        for row in range(ROWS)
                    ]
                    chunk_sets.append(
                        configure_chunk_tasks(
                            w_in_flows[col].endpoint(MT[col]),
                            w_mem_buffers[col],
                            [mem_in_tap, mem_in_tap],
                            w_mem_empty_locks[col],
                            w_mem_full_locks[col],
                            chunk=chunk,
                            acquire_value=ROWS,
                            release_value=ROWS,
                            issue_token=W_TASKS_ISSUE_TOKEN,
                        )
                    )
                    mem_live[(col, idx)] = chunk_sets
                    task_sets.extend(chunk_sets)
                start_all(task_sets)
                round_tasks.extend(core_tasks)
                of_y_conss[col].drain(Y, y_taps[c], wait=True, group=tg)
                w_in_flows[col].fill(W, tap=w_taps[c], wait=False, group=tg)

        def retire_round(rnd):
            groups.pop(rnd).finish()  # awaits the round's Y drain (shim token)
            finish_all(core_round_tasks.pop(rnd))  # free-only; safe after Y await
            for col in range(cols):
                for idx, chunk in enumerate(mem_plan.get(col, ())):
                    if chunk.last_round == rnd:
                        finish_all(mem_live.pop((col, idx)))  # free-only

        drive_rounds(max_rounds, issue_round, retire_round, window_depth)

        # 5. Clean up X fill tasks
        for col, t in enumerate(x_tasks):
            if col_rounds[col] == 0:
                t.await_()
            t.free()

    # Both modes must pass identical ObjectFifo shim handles in identical order
    # to Runtime so that IRON determines identical device bodies for both.
    if rda_channel is not None:
        shim_handles = [
            x_prods,
            of_y_conss,
            rda_channel.s_in_prod,
            rda_channel.head_in_prod,
        ]
    else:
        shim_handles = [x_prods, of_y_conss]

    if mode == "gemv":
        rt = Runtime(sequence, [Y_host_ty, W_host_ty, X_host_ty] + shim_handles)
    else:
        if rda_channel is None:
            raise RuntimeError("rda mode requires rda_col to be set")

        def rda_mode_sequence(
            s_in_arg,
            s_out_arg,
            head_in_arg,
            out_arg,
            x_prods_seq,
            of_y_conss_seq,
            s_in_prod_seq,
            head_in_prod_seq,
        ):
            declare_runtime_buffers(  # Same buffer prologue as the gemv-mode sequence
                buffer for column in w_mem_buffers for buffer in column
            )
            declare_runtime_buffers(rda_channel.agg_mem_buffers)
            rda_sequence_body(
                rda_channel,
                s_in_arg,
                s_out_arg,
                head_in_arg,
                out_arg,
                s_in_prod_seq,
                head_in_prod_seq,
            )

        rt = Runtime(
            rda_mode_sequence,
            [
                rda_channel.s_l3_ty,
                rda_channel.s_l3_ty,
                rda_channel.head_in_l3_ty,
                rda_channel.out_l3_ty,
            ]
            + shim_handles,
        )

    for column in w_mem_empty_locks + w_mem_full_locks:
        for lock in column:
            rt.add_lock(lock)
    for col in range(cols):
        rt.add_flow(w_in_flows[col])
        for row in range(ROWS):
            rt.add_flow(w_out_flows[col][row])

    if rda_channel is not None:
        for s in range(2):
            for t in range(TILES_PER_HEAD):
                rt.add_lock(rda_channel.agg_empty[s][t])
                rt.add_lock(rda_channel.agg_full[s][t])
        for t in range(TILES_PER_HEAD):
            rt.add_lock(rda_channel.core_empty[t])
            rt.add_lock(rda_channel.core_full[t])
            rt.add_flow(rda_channel.flows_c2m[t])
        rt.add_flow(rda_channel.flow_state_m2s)
        rt.add_flow(rda_channel.flow_out_m2s)

    prog = Program(dev, rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)
    return prog.resolve_program()


gemv = gemv_unified
