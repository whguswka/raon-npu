# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""AIE2P static template designs for Raon-1-mini MoE layers.

Provides two robust, statically-compiled NPU operator templates:
1. B1 (moe_gate_up_template):
   - Computes gate+up projection for 7 contiguous experts (0..6) across 8 columns x 4 rows.
   - Exact structural clone of `kernels/gemv/design.py` (K=2560, M=15232 = 7 x 2176, cols=8).
   - Only structural difference: W L3 buffer shape declared as [TOTAL_GU_WORDS] (129 experts)
     instead of 7 experts, so the static insts.bin DDR_PATCH arg_plus offsets index
     into the full layer weight buffer.
   - Chunk c (0..118) W tap offset: c * B1_W_PER_CHUNK_WORDS (c * 0x32000 bytes).
   - Preserves x BD0 throughout the entire sequence, eliminating driver TDR hangs.

2. B2 (moe_down_template):
   - Computes down projection for 7 contiguous experts (0..6) across 7 columns x 4 rows (col 7 unused).
   - Dedicated ObjectFifo (depth 1) for each column j's slice x_j [1280] of X [7 x 1280].
   - 20 rounds of execution per column.
   - Chunk c (0..139) W tap offset: c * B2_W_PER_CHUNK_WORDS (c * 0x19000 bytes) where c = j * 20 + r.
   - Preserves all 7 x_j BDs throughout all 20 rounds before deallocating at sequence completion.
"""

from __future__ import annotations

import numpy as np

try:
    from ml_dtypes import bfloat16

    bf16 = np.dtype[bfloat16]
except (ImportError, ModuleNotFoundError):
    bf16 = np.uint16

from kernels.moe.pack import (
    B1_CHUNKS_PER_EXPERT,
    B1_K,
    B1_M_PER_EXPERT,
    B1_W_PER_CHUNK_WORDS,
    B2_CHUNKS_PER_EXPERT,
    B2_K_PAD,
    B2_M_PER_EXPERT,
    B2_W_PER_CHUNK_WORDS,
    B2_BFP16_W_PER_CHUNK_WORDS,
    CHUNK_ROWS,
    DN_WORDS,
    GU_WORDS,
    NUM_ACTIVE_TOTAL,
    NUM_ROUTED_EXPERTS,
    TOTAL_DN_WORDS,
    TOTAL_DN9_WORDS,
    TOTAL_GU_WORDS,
)

# Pipeline depth matching kernels/gemv/design.py
DEPTH: int = 2

# Core stack bytes matching kernels/gemv/design.py
STACK_SIZE: int = 10 * 1024

try:
    import aie.iron as iron
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

    HAVE_AIE = True
except (ImportError, ModuleNotFoundError):
    HAVE_AIE = False

    class AIEArch:
        AIE2p = "AIE2p"
        AIE2 = "AIE2"

    Buffer = ObjectFifo = Program = Runtime = TaskGroup = Worker = object
    range_ = range
    Tile = object
    np_ndarray_type_get_shape = TensorAccessPattern = object


def check_moe_device(dev):
    """Reject device architectures other than AIE2P."""
    if not HAVE_AIE:
        return
    if dev.arch != AIEArch.AIE2p:
        raise NotImplementedError("Raon MoE kernels are AIE2P only")


def gate_up_weight_tap_offsets(slots: int) -> list[int]:
    """Return the production B1 weight-tap offsets, in emitted command order."""
    if slots not in (6, 7):
        raise ValueError(f"slots must be 6 or 7, got {slots}")
    return [
        chunk * B1_W_PER_CHUNK_WORDS
        for chunk in range(slots * B1_CHUNKS_PER_EXPERT)
    ]


def down_bfp16_weight_tap_offsets() -> list[int]:
    """Return the production 9-bit B2 weight-tap offsets in command order."""
    return [
        chunk * B2_BFP16_W_PER_CHUNK_WORDS
        for chunk in range(NUM_ACTIVE_TOTAL * B2_CHUNKS_PER_EXPERT)
    ]


# -----------------------------------------------------------------------------
# B1 Static Template: moe_gate_up_template
# -----------------------------------------------------------------------------
def moe_gate_up_template(
    dev=None,
    K: int = B1_K,
    M: int = NUM_ACTIVE_TOTAL * B1_M_PER_EXPERT,  # 15,232 (7 x 2,176)
    cols: int = 8,
    trace_size: int = 0,
    *,
    gemv_kernel=None,
    slots: int = NUM_ACTIVE_TOTAL,
):
    """Program for Raon MoE B1 gate+up static projection template.

    Exact clone of `kernels/gemv/design.py:gemv` with W L3 buffer sized to
    TOTAL_GU_WORDS (112,281,600 uint32 words) so that DDR_PATCH arg_plus
    targets the 129-expert layer buffer.

    Arguments in runtime sequence: Y, W, X.
    X holds input vector [2560] (bf16) broadcast to all 32 cores (ObjectFifo depth 1).
    W holds full layer packed weights [TOTAL_GU_WORDS] (uint32).
    Y receives ``slots * 2176`` bf16 outputs.  Six slots use a routed-only
    128-expert W buffer; the seven-slot default keeps the legacy 129-expert buffer.
    """
    if slots not in (6, 7):
        raise ValueError(f"slots must be 6 or 7, got {slots}")
    expected_m = slots * B1_M_PER_EXPERT
    if M != expected_m:
        raise ValueError(
            f"M must equal slots * {B1_M_PER_EXPERT}: expected {expected_m}, got {M}"
        )
    gu_buffer_words = (NUM_ROUTED_EXPERTS * GU_WORDS) if slots == 6 else TOTAL_GU_WORDS
    if not HAVE_AIE:
        raise RuntimeError("aie.iron is not available in this environment")

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

    check_moe_device(dev)

    if gemv_kernel is None:
        from kernels.gemv.kernel import raon_q4nx_gemv
        from quant.q4nx import GROUP, K_TILE, M_TILE

        gemv_kernel = raon_q4nx_gemv(K=K, m_tile=M_TILE, k_tile=K_TILE, group=GROUP)

    from quant.q4nx import K_TILE, M_TILE

    ROWS = len(dev.core_rows)  # 4
    M_PER_CHUNK = ROWS * M_TILE  # 128
    TOTAL_CHUNKS = M // M_PER_CHUNK  # 119
    ACTIVE_COLS = min(cols, TOTAL_CHUNKS)  # 8

    col_rounds = [
        TOTAL_CHUNKS // cols + (1 if j < (TOTAL_CHUNKS % cols) else 0)
        for j in range(ACTIVE_COLS)
    ]

    w_blk_ty, x_ty, y_acc_ty, sums_ty, _ = gemv_kernel.q4nx_gemv_block.arg_types()
    y_blk_ty, _ = gemv_kernel.q4nx_gemv_epilogue.arg_types()

    (w_blk,) = np_ndarray_type_get_shape(w_blk_ty)
    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, M_TILE), bf16]

    IT = [Tile(j, 0) for j in range(ACTIVE_COLS)]

    def core_fn(x_in, w_in, y_out, k_prep, k_zero, k_block, k_epi, y_acc, sums, rounds):
        x = x_in.acquire(1)
        k_prep(x, sums)
        for _ in range_(rounds):
            k_zero(y_acc)
            for k in range_(K // K_TILE):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    y_l3_ty = np.ndarray[(M,), bf16]
    # Sized to full 129-expert layer buffer
    w_l3_ty = np.ndarray[(gu_buffer_words,), np.dtype[np.uint32]]
    x_l3_ty = np.ndarray[(K,), bf16]

    # ObjectFifo depth 1 for x: held until all rounds complete (preserves BD0)
    of_x = ObjectFifo(x_ty, depth=1, name="x_b1")
    x_prod = of_x.prod(tile=IT[0])

    workers, w_prods, y_conses = [], [], []
    for j in range(ACTIVE_COLS):
        of_w = ObjectFifo(w_col_ty, name=f"w_b1_{j}")
        w_prods.append(of_w.prod(tile=IT[j]))
        w_cores = of_w.cons().split(
            [w_blk * i for i in range(ROWS)],
            obj_types=[w_blk_ty] * ROWS,
            names=[f"w_b1_{j}_{i}" for i in range(ROWS)],
        )

        of_y = ObjectFifo(y_col_ty, name=f"y_b1_{j}")
        y_conses.append(of_y.cons(tile=IT[j]))
        y_cores = of_y.prod().join(
            [M_TILE * i for i in range(ROWS)],
            obj_types=[y_blk_ty] * ROWS,
            names=[f"y_b1_{j}_{i}" for i in range(ROWS)],
        )

        rounds_for_this_col = col_rounds[j]
        for i in range(ROWS):
            y_acc = Buffer(type=y_acc_ty, name=f"yacc_b1_{i}_{j}")
            sums = Buffer(type=sums_ty, name=f"sums_b1_{i}_{j}")
            workers.append(
                Worker(
                    core_fn,
                    [
                        of_x.cons(),
                        w_cores[i].cons(),
                        y_cores[i].prod(),
                        gemv_kernel.q4nx_gemv_prep,
                        gemv_kernel.q4nx_gemv_zero,
                        gemv_kernel.q4nx_gemv_block,
                        gemv_kernel.q4nx_gemv_epilogue,
                        y_acc,
                        sums,
                        rounds_for_this_col,
                    ],
                    stack_size=STACK_SIZE,
                )
            )

    W_PER_CHUNK = B1_W_PER_CHUNK_WORDS  # 51,200 uint32 words

    y_taps = [
        TensorAccessPattern(
            (M,),
            c * M_PER_CHUNK,
            [1, 1, 1, M_PER_CHUNK],
            [0, 0, 0, 1],
        )
        for c in range(TOTAL_CHUNKS)
    ]
    w_taps = [
        TensorAccessPattern(
            (gu_buffer_words,),
            offset,
            [1, 1, 1, W_PER_CHUNK],
            [0, 0, 0, 1],
        )
        for offset in gate_up_weight_tap_offsets(slots)
    ]

    def sequence(Y, W, X, x_p, w_ps, y_cs):
        x_task = x_p.fill(X, wait=False, managed=False)

        max_rounds = (TOTAL_CHUNKS + cols - 1) // cols
        window = []
        for rnd in range(max_rounds):
            tg = TaskGroup()
            for col in range(ACTIVE_COLS):
                c = rnd * cols + col
                if c < TOTAL_CHUNKS:
                    y_cs[col].drain(Y, y_taps[c], wait=True, group=tg)
                    w_ps[col].fill(W, w_taps[c], wait=False, group=tg)

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


# -----------------------------------------------------------------------------
# B2 9-bit Static Template: moe_down_bfp16_template
# -----------------------------------------------------------------------------
def moe_down_bfp16_template(dev=None, trace_size: int = 0, *, gemv_kernel=None):
    """Seven-slot B2 template using the 9-bit bfp16ebs8 GEMV kernel.

    Slot columns 0..5 are routed experts and column 6 is shared expert 128.
    The static W taps cover experts 0..6; ``PatchedTemplateKernel`` rewrites all
    140 offsets to ``e_j * DN9_BYTES + r * chunk9_bytes`` per invocation.
    """
    if not HAVE_AIE:
        raise RuntimeError("aie.iron is not available in this environment")
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
    check_moe_device(dev)

    if gemv_kernel is None:
        from kernels.gemv_bfp16.kernel import raon_bfp16_gemv
        from quant.bfp16 import GROUP, K_TILE, M_TILE

        gemv_kernel = raon_bfp16_gemv(
            K=B2_K_PAD, m_tile=M_TILE, k_tile=K_TILE, group=GROUP
        )
    from quant.bfp16 import K_TILE, M_TILE

    rows = len(dev.core_rows)
    active_cols = NUM_ACTIVE_TOTAL
    rounds_per_col = B2_CHUNKS_PER_EXPERT
    m_total = active_cols * B2_M_PER_EXPERT
    x_total = active_cols * B2_K_PAD
    total_chunks = active_cols * rounds_per_col

    w_blk_ty, x_ty, y_acc_ty, _ = gemv_kernel.bfp16_gemv_block.arg_types()
    y_blk_ty, _ = gemv_kernel.bfp16_gemv_epilogue.arg_types()
    (w_blk,) = np_ndarray_type_get_shape(w_blk_ty)
    w_col_ty = np.ndarray[(rows, w_blk), bf16]
    y_col_ty = np.ndarray[(rows, M_TILE), bf16]
    interface_tiles = [Tile(j, 0) for j in range(active_cols)]

    def core_fn(x_in, w_in, y_out, k_zero, k_block, k_epi, y_acc, rounds):
        x = x_in.acquire(1)
        for _ in range_(rounds):
            k_zero(y_acc)
            for k in range_(B2_K_PAD // K_TILE):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    y_l3_ty = np.ndarray[(m_total,), bf16]
    w_l3_ty = np.ndarray[(TOTAL_DN9_WORDS,), np.dtype[np.uint32]]
    x_l3_ty = np.ndarray[(x_total,), bf16]
    x_fifos = [
        ObjectFifo(x_ty, depth=1, name=f"x_b2_9_{j}") for j in range(active_cols)
    ]
    x_prods = [x_fifos[j].prod(tile=interface_tiles[j]) for j in range(active_cols)]

    workers, w_prods, y_conses = [], [], []
    for j in range(active_cols):
        of_w = ObjectFifo(w_col_ty, name=f"w_b2_9_{j}")
        w_prods.append(of_w.prod(tile=interface_tiles[j]))
        w_cores = of_w.cons().split(
            [w_blk * i for i in range(rows)],
            obj_types=[w_blk_ty] * rows,
            names=[f"w_b2_9_{j}_{i}" for i in range(rows)],
        )
        of_y = ObjectFifo(y_col_ty, name=f"y_b2_9_{j}")
        y_conses.append(of_y.cons(tile=interface_tiles[j]))
        y_cores = of_y.prod().join(
            [M_TILE * i for i in range(rows)],
            obj_types=[y_blk_ty] * rows,
            names=[f"y_b2_9_{j}_{i}" for i in range(rows)],
        )
        for i in range(rows):
            y_acc = Buffer(type=y_acc_ty, name=f"yacc_b2_9_{i}_{j}")
            workers.append(
                Worker(
                    core_fn,
                    [
                        x_fifos[j].cons(),
                        w_cores[i].cons(),
                        y_cores[i].prod(),
                        gemv_kernel.bfp16_gemv_zero,
                        gemv_kernel.bfp16_gemv_block,
                        gemv_kernel.bfp16_gemv_epilogue,
                        y_acc,
                        rounds_per_col,
                    ],
                    stack_size=STACK_SIZE,
                )
            )

    y_taps = [
        TensorAccessPattern(
            (m_total,),
            (c // rounds_per_col) * B2_M_PER_EXPERT + (c % rounds_per_col) * CHUNK_ROWS,
            [1, 1, 1, CHUNK_ROWS],
            [0, 0, 0, 1],
        )
        for c in range(total_chunks)
    ]
    w_taps = [
        TensorAccessPattern(
            (TOTAL_DN9_WORDS,),
            offset,
            [1, 1, 1, B2_BFP16_W_PER_CHUNK_WORDS],
            [0, 0, 0, 1],
        )
        for offset in down_bfp16_weight_tap_offsets()
    ]
    x_taps = [
        TensorAccessPattern((x_total,), j * B2_K_PAD, [1, 1, 1, B2_K_PAD], [0, 0, 0, 1])
        for j in range(active_cols)
    ]

    def sequence(Y, W, X, x_ps, w_ps, y_cs):
        x_tasks = [
            x_ps[j].fill(X, x_taps[j], wait=False, managed=False)
            for j in range(active_cols)
        ]
        window = []
        for r in range(rounds_per_col):
            tg = TaskGroup()
            for j in range(active_cols):
                c = j * rounds_per_col + r
                y_cs[j].drain(Y, y_taps[c], wait=True, group=tg)
                w_ps[j].fill(W, w_taps[c], wait=False, group=tg)
            window.append(tg)
            if len(window) > DEPTH:
                window.pop(0).finish()
        for tg in window:
            tg.finish()
        for task in x_tasks:
            task.free()

    rt = Runtime(sequence, [y_l3_ty, w_l3_ty, x_l3_ty, x_prods, w_prods, y_conses])
    prog = Program(dev, rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)
    return prog.resolve_program()


# -----------------------------------------------------------------------------
# B2 Static Template: moe_down_template
# -----------------------------------------------------------------------------
def moe_down_template(
    dev=None,
    trace_size: int = 0,
    *,
    gemv_kernel=None,
):
    """Program for Raon MoE B2 down static projection template.

    Computes 7 contiguous experts (0..6) across columns 0..6 (column 7 unused).
    Each column j executes 20 rounds of down projection for expert j.
    Dedicated per-column ObjectFifos (depth 1) for x_j [1280] are filled once
    and held until all 20 rounds complete.

    Arguments in runtime sequence: Y, W, X.
    X holds concatenated input vector [7 x 1280] = [8960] (bf16).
    W holds full layer packed weights [TOTAL_DN_WORDS] (uint32).
    Y receives projected partial sums [7 x 2560] = [17920] (bf16).
    """
    if not HAVE_AIE:
        raise RuntimeError("aie.iron is not available in this environment")

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

    check_moe_device(dev)

    if gemv_kernel is None:
        from kernels.gemv.kernel import raon_q4nx_gemv
        from quant.q4nx import GROUP, K_TILE, M_TILE

        gemv_kernel = raon_q4nx_gemv(
            K=B2_K_PAD, m_tile=M_TILE, k_tile=K_TILE, group=GROUP
        )

    from quant.q4nx import K_TILE, M_TILE

    ROWS = len(dev.core_rows)  # 4
    ACTIVE_COLS = NUM_ACTIVE_TOTAL  # 7 columns (cols 0..6; col 7 unused)
    ROUNDS_PER_COL = B2_CHUNKS_PER_EXPERT  # 20 rounds
    M_TOTAL = NUM_ACTIVE_TOTAL * B2_M_PER_EXPERT  # 17,920
    X_TOTAL = NUM_ACTIVE_TOTAL * B2_K_PAD  # 8,960
    TOTAL_CHUNKS = ACTIVE_COLS * ROUNDS_PER_COL  # 140
    W_PER_CHUNK = B2_W_PER_CHUNK_WORDS  # 25,600 uint32 words

    w_blk_ty, x_ty, y_acc_ty, sums_ty, _ = gemv_kernel.q4nx_gemv_block.arg_types()
    y_blk_ty, _ = gemv_kernel.q4nx_gemv_epilogue.arg_types()

    (w_blk,) = np_ndarray_type_get_shape(w_blk_ty)
    w_col_ty = np.ndarray[(ROWS, w_blk), bf16]
    y_col_ty = np.ndarray[(ROWS, M_TILE), bf16]

    IT = [Tile(j, 0) for j in range(ACTIVE_COLS)]

    def core_fn(x_in, w_in, y_out, k_prep, k_zero, k_block, k_epi, y_acc, sums, rounds):
        x = x_in.acquire(1)
        k_prep(x, sums)
        for _ in range_(rounds):
            k_zero(y_acc)
            for k in range_(B2_K_PAD // K_TILE):
                w = w_in.acquire(1)
                k_block(w, x, y_acc, sums, k)
                w_in.release(1)
            y = y_out.acquire(1)
            k_epi(y, y_acc)
            y_out.release(1)
        x_in.release(1)

    y_l3_ty = np.ndarray[(M_TOTAL,), bf16]
    w_l3_ty = np.ndarray[(TOTAL_DN_WORDS,), np.dtype[np.uint32]]
    x_l3_ty = np.ndarray[(X_TOTAL,), bf16]

    # Dedicated ObjectFifo for each column's slice x_j (depth 1, held for all 20 rounds)
    of_x_list = [
        ObjectFifo(x_ty, depth=1, name=f"x_b2_{j}") for j in range(ACTIVE_COLS)
    ]
    x_prods = [of_x_list[j].prod(tile=IT[j]) for j in range(ACTIVE_COLS)]

    workers, w_prods, y_conses = [], [], []
    for j in range(ACTIVE_COLS):
        of_w = ObjectFifo(w_col_ty, name=f"w_b2_{j}")
        w_prods.append(of_w.prod(tile=IT[j]))
        w_cores = of_w.cons().split(
            [w_blk * i for i in range(ROWS)],
            obj_types=[w_blk_ty] * ROWS,
            names=[f"w_b2_{j}_{i}" for i in range(ROWS)],
        )

        of_y = ObjectFifo(y_col_ty, name=f"y_b2_{j}")
        y_conses.append(of_y.cons(tile=IT[j]))
        y_cores = of_y.prod().join(
            [M_TILE * i for i in range(ROWS)],
            obj_types=[y_blk_ty] * ROWS,
            names=[f"y_b2_{j}_{i}" for i in range(ROWS)],
        )

        for i in range(ROWS):
            y_acc = Buffer(type=y_acc_ty, name=f"yacc_b2_{i}_{j}")
            sums = Buffer(type=sums_ty, name=f"sums_b2_{i}_{j}")
            workers.append(
                Worker(
                    core_fn,
                    [
                        of_x_list[j].cons(),
                        w_cores[i].cons(),
                        y_cores[i].prod(),
                        gemv_kernel.q4nx_gemv_prep,
                        gemv_kernel.q4nx_gemv_zero,
                        gemv_kernel.q4nx_gemv_block,
                        gemv_kernel.q4nx_gemv_epilogue,
                        y_acc,
                        sums,
                        ROUNDS_PER_COL,
                    ],
                    stack_size=STACK_SIZE,
                )
            )

    # Static TAP offsets for Y, W, X
    # For chunk c (0..139):
    #   col/slot j = c // 20
    #   round r = c % 20
    #   W offset: c * W_PER_CHUNK (chunk c in continuous 0..6 placement)
    #   Y offset: j * B2_M_PER_EXPERT + r * CHUNK_ROWS
    y_taps = [
        TensorAccessPattern(
            (M_TOTAL,),
            (c // ROUNDS_PER_COL) * B2_M_PER_EXPERT + (c % ROUNDS_PER_COL) * CHUNK_ROWS,
            [1, 1, 1, CHUNK_ROWS],
            [0, 0, 0, 1],
        )
        for c in range(TOTAL_CHUNKS)
    ]
    w_taps = [
        TensorAccessPattern(
            (TOTAL_DN_WORDS,),
            c * W_PER_CHUNK,
            [1, 1, 1, W_PER_CHUNK],
            [0, 0, 0, 1],
        )
        for c in range(TOTAL_CHUNKS)
    ]
    x_taps = [
        TensorAccessPattern(
            (X_TOTAL,),
            j * B2_K_PAD,
            [1, 1, 1, B2_K_PAD],
            [0, 0, 0, 1],
        )
        for j in range(ACTIVE_COLS)
    ]

    def sequence(Y, W, X, x_ps, w_ps, y_cs):
        # Fill each column's x_j slice (held for all 20 rounds)
        x_tasks = []
        for j in range(ACTIVE_COLS):
            x_tasks.append(x_ps[j].fill(X, x_taps[j], wait=False, managed=False))

        window = []
        for r in range(ROUNDS_PER_COL):
            tg = TaskGroup()
            for j in range(ACTIVE_COLS):
                c = j * ROUNDS_PER_COL + r
                y_cs[j].drain(Y, y_taps[c], wait=True, group=tg)
                w_ps[j].fill(W, w_taps[c], wait=False, group=tg)

            window.append(tg)
            if len(window) > DEPTH:
                window.pop(0).finish()

        for tg in window:
            tg.finish()

        for t in x_tasks:
            t.free()

    rt = Runtime(
        sequence,
        [
            y_l3_ty,
            w_l3_ty,
            x_l3_ty,
            x_prods,
            w_prods,
            y_conses,
        ],
    )
    prog = Program(dev, rt, workers=workers)
    if trace_size > 0:
        prog.enable_trace(trace_size)
    return prog.resolve_program()


# DesignGenerator entry point aliases
gemv = moe_gate_up_template
