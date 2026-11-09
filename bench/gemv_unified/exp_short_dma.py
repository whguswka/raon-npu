# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Gate experiment for runtime-configured 9,216 B / 5,120 B W receive BDs.

One shared xclbin contains fixed 9,216-byte ping/pong buffers on a memory tile
and four compute tiles. Each instruction stream configures finite runtime DMA
tasks whose mem/core receive BDs use the format's true length. The path is
shim -> memory tile -> four cores, matching GEMV.

Run this gate before q4 smoke, B1, B2, or mixed tests. The parent process runs
three separately watched phases: first full+short calls, a short repetition,
then the requested (default 10,000) alternating calls.

fix5: Adds --variant to sweep multi-block DMA/lock shapes and shim task
granularity.  V0 is the original single-block ping transfer; V1..V5 explore
pong, 2-BD chains, and per-block shim tasks.  See VARIANTS below.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.gemv_unified.dma_layout import DmaTapSpec, short_dma_tap_specs
from kernels.gemv_unified.shapes import MAX_K

W_FULL_BYTES = 9216
W_SHORT_BYTES = 5120
W_SENTINEL_BYTES = W_FULL_BYTES - W_SHORT_BYTES
W_FULL_U32 = W_FULL_BYTES // 4
W_SHORT_U32 = W_SHORT_BYTES // 4
W_SENTINEL_U32 = W_SENTINEL_BYTES // 4
ROWS = 4
DEPTH = 2
_ECHO_SOURCE = Path(__file__).resolve().parent / "short_dma_echo.cc"


class VariantSpec:
    """One short-DMA experiment variant.

    Fields
    ------
    blocks:
        Number of W blocks transferred in a single NPU call.
    mem_chain:
        If True, the memory-tile W tasks use one 2-BD repeat task per
        endpoint (``configure_ping_pong_tasks``).  If False, each block gets a
        separate 1-BD task.
    core_chain:
        Same choice for the core receive tasks.
    shim_chain:
        Same choice for the shim W fill tasks.
    alt_sizes:
        True only for V0: keep the original alternating full/short behaviour.
    transfer_words:
        Active W row length for this variant (full by default).
    """

    __slots__ = (
        "name",
        "blocks",
        "mem_chain",
        "core_chain",
        "shim_chain",
        "alt_sizes",
        "transfer_words",
        "x_like_fifo",
        "rounds",
    )

    def __init__(
        self,
        name: str,
        blocks: int,
        mem_chain: bool,
        core_chain: bool,
        shim_chain: bool,
        alt_sizes: bool = False,
        transfer_words: int = W_FULL_U32,
        x_like_fifo: bool = False,
        rounds: int = 1,
    ) -> None:
        self.name = name
        self.blocks = blocks
        self.mem_chain = mem_chain
        self.core_chain = core_chain
        self.shim_chain = shim_chain
        self.alt_sizes = alt_sizes
        self.transfer_words = transfer_words
        self.x_like_fifo = x_like_fifo
        self.rounds = rounds

    def __repr__(self) -> str:
        return (
            f"VariantSpec({self.name!r}, blocks={self.blocks}, "
            f"mem_chain={self.mem_chain}, core_chain={self.core_chain}, "
            f"shim_chain={self.shim_chain}, alt_sizes={self.alt_sizes}, "
            f"transfer_words={self.transfer_words}, "
            f"x_like_fifo={self.x_like_fifo}, rounds={self.rounds})"
        )


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("V0", 1, False, False, False, alt_sizes=True),
    VariantSpec("V1S1", 2, False, False, True),
    VariantSpec("V1S2", 2, False, False, False),
    VariantSpec("V2S1", 2, True, False, True),
    VariantSpec("V2S2", 2, True, False, False),
    VariantSpec("V3S1", 2, False, True, True),
    VariantSpec("V3S2", 2, False, True, False),
    VariantSpec("V4S1", 2, True, True, True),
    VariantSpec("V4S2", 2, True, True, False),
    VariantSpec("V5S1", 10, True, True, True),
    VariantSpec("V5S2", 10, True, True, False),
    # fix6: channel-1 W path (x-like ObjectFifo on S2MM channel 0).
    VariantSpec("V6", 2, True, True, True, x_like_fifo=True),
    VariantSpec("V7", 4, True, True, True, x_like_fifo=True, rounds=2),
    VariantSpec("V8", 1, False, False, False, x_like_fifo=True),
)
_VARIANTS_BY_NAME = {v.name: v for v in VARIANTS}


def log(msg: str) -> None:
    print(msg, flush=True)


def _shim_tap_spec(
    name: str, length: int, offset: int = 0, tensor_length: int | None = None
) -> DmaTapSpec:
    """Return a flat shim-form DmaTapSpec of ``length`` uint32 words.

    ``tensor_length`` allows the underlying host tensor to be larger than the
    transferred slice (needed for multi-block S2 variants whose per-block taps
    reference a single host W tensor).
    """
    total = tensor_length if tensor_length is not None else length
    return DmaTapSpec(
        name=name,
        tensor_dims=(1, total),
        offset=offset,
        sizes=(1, 1, 1, length),
        strides=(1, 1, 1, 1),
        tile="shim",
        element_bits=32,
    )


def simulate_short_dma_ping_pong(
    iters: int = 100,
    *,
    depth: int = DEPTH,
    sentinel_word: int = 0xDEADBEEF,
) -> dict[str, Any]:
    """Reference exact-length writes and the per-dispatch ping re-arm contract."""
    if depth != DEPTH:
        raise ValueError(f"runtime task design fixes depth={DEPTH}, got {depth}")
    slots = np.full((depth, ROWS, W_FULL_U32), sentinel_word, dtype=np.uint32)
    sentinel_preserved = True
    compared_words = 0
    for step in range(iters):
        active = W_FULL_U32 if step % 2 == 0 else W_SHORT_U32
        payload = np.arange(ROWS * active, dtype=np.uint32).reshape(ROWS, active)
        payload ^= np.uint32(0xA5A50000 | (step & 0xFFFF))
        # A finite runtime task starts at ping after its predecessor was awaited
        # and freed. Only the active prefix is written.
        before_tail = slots[0, :, active:].copy()
        slots[0, :, :active] = payload
        sentinel_preserved &= np.array_equal(slots[0, :, active:], before_tail)
        compared_words += payload.size
    return {
        "iters": iters,
        "depth": depth,
        "final_slot_idx": 0,
        "sentinel_preserved": bool(sentinel_preserved),
        "compared_words": compared_words,
    }


def simulate_short_dma_variant(
    variant: VariantSpec,
    calls: int = 5,
    *,
    sentinel_word: int = 0xDEADBEEF,
) -> dict[str, Any]:
    """Lock-order simulation for a multi-block variant.

    Models the per-call lock dance (mem empty -> mem full -> core full -> core
    empty) for every block and verifies that counts return to their initial
    values after each call.  Data is copied through exactly once per block.
    """
    if variant.alt_sizes:
        raise ValueError("use simulate_short_dma_ping_pong for the V0 alternating case")
    active = variant.transfer_words
    slots = np.full((DEPTH, ROWS, W_FULL_U32), sentinel_word, dtype=np.uint32)
    sentinel_preserved = True
    compared_words = 0

    for c in range(calls):
        mem_empty = [ROWS, ROWS]
        mem_full = [0, 0]
        core_empty = [1, 1]
        core_full = [0, 0]

        payload = np.arange(
            variant.blocks * ROWS * active, dtype=np.uint32
        ).reshape(variant.blocks, ROWS, active)
        payload ^= np.uint32(0xA5A50000 | (c & 0xFFFF))

        for b in range(variant.blocks):
            slot = b % 2
            # 1) shim -> memory tile receive
            if mem_empty[slot] < ROWS:
                raise AssertionError(
                    f"{variant.name} call={c} block={b}: mem_empty[{slot}] underflow"
                )
            mem_empty[slot] -= ROWS
            mem_full[slot] += ROWS

            for row in range(ROWS):
                # 2a) memory tile MM2S: consume one row from the filled slot
                if mem_full[slot] < 1:
                    raise AssertionError(
                        f"{variant.name} call={c} block={b} row={row}: "
                        f"mem_full[{slot}] underflow"
                    )
                mem_full[slot] -= 1
                mem_empty[slot] += 1

                # 2b) core S2MM receive for the same row (same DMA transfer)
                if core_empty[slot] < 1:
                    raise AssertionError(
                        f"{variant.name} call={c} block={b} row={row}: "
                        f"core_empty[{slot}] underflow"
                    )
                core_empty[slot] -= 1
                core_full[slot] += 1

                # 3) core worker consumes the arrived row and echoes to y
                if core_full[slot] < 1:
                    raise AssertionError(
                        f"{variant.name} call={c} block={b} row={row}: "
                        f"core_full[{slot}] underflow"
                    )
                core_full[slot] -= 1
                core_empty[slot] += 1

                before_tail = slots[slot, row, active:].copy()
                slots[slot, row, :active] = payload[b, row]
                sentinel_preserved &= np.array_equal(
                    slots[slot, row, active:], before_tail
                )
                compared_words += active

        if mem_empty != [ROWS, ROWS] or mem_full != [0, 0]:
            raise AssertionError(
                f"{variant.name} call={c}: memory lock counts did not reset"
            )
        if core_empty != [1, 1] or core_full != [0, 0]:
            raise AssertionError(
                f"{variant.name} call={c}: core lock counts did not reset"
            )

    def task_count(chain: bool, blocks: int) -> int:
        # configure_ping_pong_tasks returns one task for even blocks; for odd
        # blocks it adds a 1-BD tail task.  Our variants use 1 or even blocks.
        if chain:
            return 1 if blocks % 2 == 0 else 2
        return blocks

    return {
        "variant": variant.name,
        "blocks": variant.blocks,
        "rounds": variant.rounds,
        "x_like_fifo": variant.x_like_fifo,
        "calls": calls,
        "sentinel_preserved": bool(sentinel_preserved),
        "compared_words": compared_words,
        "mem_tasks": task_count(variant.mem_chain, variant.blocks),
        "core_tasks_per_row": task_count(variant.core_chain, variant.blocks),
        "shim_tasks": task_count(variant.shim_chain, variant.blocks),
    }


def _echo_kernel():
    from aie.iron.kernel import ExternalFunction

    row_ty = np.ndarray[(W_FULL_U32,), np.dtype[np.uint32]]
    return ExternalFunction(
        "short_dma_echo",
        source_file=str(_ECHO_SOURCE),
        arg_types=[row_ty, row_ty, np.int32],
    )


def short_dma_program(
    dev=None,
    transfer_words_per_row: int = W_FULL_U32,
    *,
    echo_kernel=None,
    variant: VariantSpec | None = None,
):
    """Build the invariant device and one exact-length runtime instruction stream.

    When ``variant`` is None the program is the original single-block V0 shape.
    Otherwise it follows the variant's block count and chain flags.
    """
    from aie.helpers.taplib import TensorAccessPattern
    from aie.iron import (
        Acquire,
        Bd,
        Buffer,
        Flow,
        Lock,
        ObjectFifo,
        Program,
        Release,
        Runtime,
        TaskGroup,
        Worker,
        WorkerRuntimeBarrier,
    )
    from aie.iron.device import Tile
    from kernels.gemv_unified.routing import W_TASKS_ISSUE_TOKEN
    from kernels.gemv_unified.w_dma import (
        RuntimeDmaTasks,
        configure_ping_pong_tasks,
        finish_all,
        start_all,
        tap_from_spec,
    )

    if dev is None:
        import aie.utils as aie_utils

        dev = aie_utils.get_current_device()
    if dev is not None:
        try:
            import aie.utils.hostruntime as hostruntime

            hostruntime.set_current_device(dev)
        except Exception:
            pass
    if transfer_words_per_row not in (W_FULL_U32, W_SHORT_U32):
        raise ValueError(
            f"transfer_words_per_row must be {W_FULL_U32} or {W_SHORT_U32}"
        )
    if echo_kernel is None:
        echo_kernel = _echo_kernel()

    if variant is None:
        blocks = 1
        mem_chain = False
        core_chain = False
        shim_chain = True  # blocks==1, so the result is identical to separate
    else:
        blocks = variant.blocks
        mem_chain = variant.mem_chain
        core_chain = variant.core_chain
        shim_chain = variant.shim_chain

    def local_tap_from_spec(spec: DmaTapSpec) -> TensorAccessPattern:
        # ``tap_from_spec`` needs ``aie``; keep the import local.
        return tap_from_spec(spec)

    dma_specs = short_dma_tap_specs(transfer_words_per_row)
    row_ty = np.ndarray[(dma_specs["y_core"][0].sizes[0],), np.dtype[np.uint32]]
    col_ty = np.ndarray[(dma_specs["y_shim"][0].sizes[0],), np.dtype[np.uint32]]
    rtp_ty = np.ndarray[(1,), np.dtype[np.int32]]

    if variant is None or variant.alt_sizes:
        host_w_ty = np.ndarray[(ROWS * transfer_words_per_row,), np.dtype[np.uint32]]
        host_y_ty = np.ndarray[(ROWS * W_FULL_U32,), np.dtype[np.uint32]]
        shim_w_taps = [local_tap_from_spec(dma_specs["host_w"][0])]
        host_y_tap = local_tap_from_spec(dma_specs["host_y"][0])
    else:
        per_block_w = ROWS * transfer_words_per_row
        total_w = blocks * per_block_w
        total_y = blocks * ROWS * W_FULL_U32
        host_w_ty = np.ndarray[(total_w,), np.dtype[np.uint32]]
        host_y_ty = np.ndarray[(total_y,), np.dtype[np.uint32]]
        if shim_chain:
            shim_w_taps = [
                local_tap_from_spec(_shim_tap_spec("exp_host_w_total", total_w))
            ]
        else:
            shim_w_taps = [
                local_tap_from_spec(
                    _shim_tap_spec(
                        f"exp_host_w_block{b}",
                        per_block_w,
                        offset=b * per_block_w,
                        tensor_length=total_w,
                    )
                )
                for b in range(blocks)
            ]
        host_y_tap = local_tap_from_spec(
            _shim_tap_spec("exp_host_y_total", total_y)
        )

    shim = Tile(0, 0)
    mem = Tile(0, 1)
    cores = [Tile(0, row + 2) for row in range(ROWS)]
    in_flow = Flow(shim, mem, name="w_exp_in")
    out_flows = [
        Flow(mem, core, name=f"w_exp_out_{row}")
        for row, core in enumerate(cores)
    ]

    mem_buffers = [
        Buffer(type=col_ty, name=f"w_exp_mem_{slot}", tile=mem)
        for slot in range(DEPTH)
    ]
    mem_empty = [
        Lock(mem, init=ROWS, name=f"w_exp_mem_empty_{slot}")
        for slot in range(DEPTH)
    ]
    mem_full = [
        Lock(mem, init=0, name=f"w_exp_mem_full_{slot}")
        for slot in range(DEPTH)
    ]

    of_y = ObjectFifo(col_ty, name="y_exp")
    y_cons = of_y.cons(tile=shim)
    # Pin the y join to the same column's memory tile as W (same constraint as
    # the unified design): memory tiles have no East/West ports, so an unpinned
    # join (AnyMemTile) may land in another column and cannot be routed back.
    y_rows = of_y.prod().join(
        [row * W_FULL_U32 for row in range(ROWS)],
        tile=mem,
        obj_types=[row_ty] * ROWS,
        names=[f"y_exp_{row}" for row in range(ROWS)],
    )

    # Optional dummy x ObjectFifo to move the explicit W flow to S2MM channel 1
    # (matching the unified design).  It is filled once per call and held across
    # all rounds; the core only acquires/releases it to reserve channel 0.
    if variant is not None and variant.x_like_fifo:
        x_ty = np.ndarray[(MAX_K,), np.dtype[np.uint32]]
        of_x = ObjectFifo(x_ty, depth=1, name="x_exp")
        x_prod = of_x.prod(tile=shim)
        x_cons = [of_x.cons() for _ in cores]
        x_host_ty = np.ndarray[(MAX_K,), np.dtype[np.uint32]]
        x_tap = local_tap_from_spec(_shim_tap_spec("exp_host_x", MAX_K))
    else:
        x_prod = None
        x_cons = [None] * ROWS

    core_buffers = []
    core_empty = []
    core_full = []
    rtps = []
    barriers = []
    workers = []

    def core_fn(
        x_in,
        w_ping,
        w_pong,
        ping_empty,
        pong_empty,
        ping_full,
        pong_full,
        y_out,
        rtp,
        barrier,
        kernel,
    ):
        barrier.wait_for_value(1)
        words = rtp[0]
        barrier.release_with_value(1)
        if x_in is not None:
            x_in.acquire(1)
        bufs = [w_ping, w_pong]
        full = (ping_full, pong_full)
        empty = (ping_empty, pong_empty)
        for _ in range(blocks):
            slot = _ % 2
            full[slot].acquire()
            y = y_out.acquire(1)
            kernel(bufs[slot], y, words)
            y_out.release(1)
            empty[slot].release()
        if x_in is not None:
            x_in.release(1)

    for row, core in enumerate(cores):
        buffers = [
            Buffer(type=row_ty, name=f"w_exp_core_{row}_{slot}", tile=core)
            for slot in range(DEPTH)
        ]
        empty = [
            Lock(core, init=1, name=f"w_exp_core_empty_{row}_{slot}")
            for slot in range(DEPTH)
        ]
        full = [
            Lock(core, init=0, name=f"w_exp_core_full_{row}_{slot}")
            for slot in range(DEPTH)
        ]
        rtp = Buffer(
            type=rtp_ty,
            name=f"rtp_exp_{row}",
            initial_value=np.zeros(1, dtype=np.int32),
            use_write_rtp=True,
        )
        barrier = WorkerRuntimeBarrier()
        core_buffers.append(buffers)
        core_empty.append(empty)
        core_full.append(full)
        rtps.append(rtp)
        barriers.append(barrier)
        workers.append(
            Worker(
                core_fn,
                [
                    x_cons[row],
                    buffers[0],
                    buffers[1],
                    empty[0],
                    empty[1],
                    full[0],
                    full[1],
                    y_rows[row].prod(),
                    rtp,
                    barrier,
                    echo_kernel,
                ],
                tile=core,
                stack_size=4096,
            )
        )

    mem_in_tap = local_tap_from_spec(dma_specs["mem_in"][0])
    mem_out_taps = [local_tap_from_spec(spec) for spec in dma_specs["mem_out"]]
    core_tap = local_tap_from_spec(dma_specs["core_in"][0])

    def configure_block_tasks(
        endpoint: Any,
        buffers: Sequence[Any],
        taps: Sequence[Any],
        empty_locks: Sequence[Any],
        full_locks: Sequence[Any],
        *,
        chain: bool,
        acquire_value: int = 1,
        release_value: int = 1,
    ) -> RuntimeDmaTasks:
        """Configure one endpoint for ``blocks`` blocks, chained or separate."""
        if chain:
            return configure_ping_pong_tasks(
                endpoint,
                buffers,
                taps,
                empty_locks,
                full_locks,
                blocks=blocks,
                acquire_value=acquire_value,
                release_value=release_value,
                issue_token=W_TASKS_ISSUE_TOKEN,
            )

        def bd(slot: int) -> Bd:
            return Bd(
                buffers[slot],
                tap=taps[slot],
                acquires=[Acquire(empty_locks[slot], value=acquire_value)],
                releases=[Release(full_locks[slot], value=release_value)],
            )

        return RuntimeDmaTasks(
            [
                endpoint.task(bd(b % 2), runs=1, wait=W_TASKS_ISSUE_TOKEN)
                for b in range(blocks)
            ],
            issue_token=W_TASKS_ISSUE_TOKEN,
        )

    rounds_per_call = 1 if variant is None else variant.rounds
    if blocks % rounds_per_call:
        raise ValueError(
            f"blocks={blocks} must be divisible by rounds={rounds_per_call}"
        )
    blocks_per_round = blocks // rounds_per_call

    def _round_shim_tap(name: str, length: int, offset: int, total: int):
        return local_tap_from_spec(
            _shim_tap_spec(name, length, offset=offset, tensor_length=total)
        )

    def _build_continuous_tasks() -> list[Any]:
        core = [
            configure_block_tasks(
                out_flows[row].endpoint(cores[row]),
                core_buffers[row],
                [core_tap, core_tap],
                core_empty[row],
                core_full[row],
                chain=core_chain,
            )
            for row in range(ROWS)
        ]
        mem_out = [
            configure_block_tasks(
                out_flows[row].endpoint(mem),
                mem_buffers,
                [mem_out_taps[row], mem_out_taps[row]],
                mem_full,
                mem_empty,
                chain=mem_chain,
            )
            for row in range(ROWS)
        ]
        mem_in = configure_block_tasks(
            in_flow.endpoint(mem),
            mem_buffers,
            [mem_in_tap, mem_in_tap],
            mem_empty,
            mem_full,
            chain=mem_chain,
            acquire_value=ROWS,
            release_value=ROWS,
        )
        return [*core, *mem_out, mem_in]

    def _run_rounds(Y, W, y_handle, group: TaskGroup, rnd: int):
        y_words = blocks_per_round * ROWS * W_FULL_U32
        w_words = blocks_per_round * ROWS * transfer_words_per_row
        y_offset = rnd * y_words
        y_tap = _round_shim_tap("exp_host_y_round", y_words, y_offset, total_y)
        y_handle.drain(Y, y_tap, wait=True, group=group)
        if shim_chain:
            w_offset = rnd * w_words
            w_tap = _round_shim_tap("exp_host_w_round", w_words, w_offset, total_w)
            in_flow.fill(W, tap=w_tap, wait=False, group=group)
        else:
            base_block = rnd * blocks_per_round
            for b in range(blocks_per_round):
                block = base_block + b
                w_offset = block * ROWS * transfer_words_per_row
                w_tap = _round_shim_tap(
                    f"exp_host_w_block{block}",
                    ROWS * transfer_words_per_row,
                    w_offset,
                    total_w,
                )
                in_flow.fill(W, tap=w_tap, wait=False, group=group)

    if x_prod is not None:

        def sequence(Y, W, X, x_handle, y_handle):
            for row in range(ROWS):
                rtps[row][0] = transfer_words_per_row
                barriers[row].set(1)

            task_sets = _build_continuous_tasks()
            start_all(task_sets)

            x_task = x_handle.fill(X, tap=x_tap, wait=False, managed=False)
            if rounds_per_call == 1:
                tg = TaskGroup()
                y_handle.drain(Y, host_y_tap, wait=True, group=tg)
                for shim_tap in shim_w_taps:
                    in_flow.fill(W, tap=shim_tap, wait=False, group=tg)
                tg.finish()
            else:
                for rnd in range(rounds_per_call):
                    tg = TaskGroup()
                    _run_rounds(Y, W, y_handle, tg, rnd)
                    tg.finish()
            x_task.free()
            finish_all(task_sets)  # free-only

        rt = Runtime(sequence, [host_y_ty, host_w_ty, x_host_ty, x_prod, y_cons])
    else:

        def sequence(Y, W, y_handle):
            for row in range(ROWS):
                rtps[row][0] = transfer_words_per_row
                barriers[row].set(1)

            task_sets = _build_continuous_tasks()
            start_all(task_sets)

            if rounds_per_call == 1:
                tg = TaskGroup()
                y_handle.drain(Y, host_y_tap, wait=True, group=tg)
                for shim_tap in shim_w_taps:
                    in_flow.fill(W, tap=shim_tap, wait=False, group=tg)
                tg.finish()
            else:
                for rnd in range(rounds_per_call):
                    tg = TaskGroup()
                    _run_rounds(Y, W, y_handle, tg, rnd)
                    tg.finish()
            finish_all(task_sets)  # free-only

        rt = Runtime(sequence, [host_y_ty, host_w_ty, y_cons])
    rt.add_flow(in_flow)
    for flow in out_flows:
        rt.add_flow(flow)
    return Program(dev, rt, workers=workers).resolve_program()


def run_npu_experiment(iters: int, *, compile_only: bool = False) -> int:
    """Compile one xclbin, alternate exact streams, and compare every active word."""
    import aie.iron as iron
    import aie.utils as aie_utils
    from iron.common import (
        AIEContext,
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
    from iron.common.compilation import InstsBinArtifact, XclbinArtifact

    kernel = _echo_kernel()

    class ShortDMAOp(MLIROperator):
        def __init__(self, transfer_words: int, context: Any):
            self.transfer_words = transfer_words
            super().__init__(context=context)

        @property
        def config_name(self) -> str:
            return f"Raon_ExpShortDMA_{aie_utils.get_current_device().resolve().name}"

        @property
        def name(self) -> str:
            return f"{self.config_name}_w{self.transfer_words}"

        def _mlir_artifact(self, transfer_words: int):
            return PythonGeneratedMLIRArtifact(
                f"{self.config_name}_w{transfer_words}.mlir",
                DesignGenerator(
                    Path(__file__).resolve(),
                    "short_dma_program",
                    (aie_utils.get_current_device(), transfer_words),
                    {"echo_kernel": kernel},
                ),
            )

        def get_mlir_artifact(self):
            return self._mlir_artifact(self.transfer_words)

        def get_kernel_artifacts(self):
            return [KernelObjectArtifact.from_extern(kernel)]

        def set_up_artifacts(self) -> None:
            kernels = self.get_kernel_artifacts()
            cfg_mlir = self._mlir_artifact(W_FULL_U32)
            shape_mlir = self.get_mlir_artifact()
            self.xclbin_artifact = XclbinArtifact(
                f"{self.config_name}.xclbin",
                mlir_input=cfg_mlir,
                dependencies=[cfg_mlir, *kernels],
                kernel_name="MLIR_AIE",
                extra_flags=["--dynamic-objFifos"],
            )
            self.insts_artifact = InstsBinArtifact(
                f"{self.name}.bin",
                mlir_input=shape_mlir,
                dependencies=[shape_mlir, *kernels],
                extra_flags=["--dynamic-objFifos"],
            )
            if hasattr(self, "add_artifacts") and callable(getattr(self, "add_artifacts")):
                self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

        def register_artifacts(self, artifacts: list[Any]) -> None:
            if hasattr(self, "add_artifacts") and callable(getattr(self, "add_artifacts")):
                self.add_artifacts(artifacts)

        def get_arg_spec(self):
            return [
                AIERuntimeArgSpec("out", (ROWS * W_FULL_U32,), dtype=np.uint32),
                AIERuntimeArgSpec(
                    "in", (ROWS * self.transfer_words,), dtype=np.uint32
                ),
            ]

    ctx = AIEContext()
    op_full = ShortDMAOp(W_FULL_U32, context=ctx)
    op_short = ShortDMAOp(W_SHORT_U32, context=ctx)
    for op in (op_full, op_short):
        op.set_up_artifacts()
        op.compile()
        log(
            f"[exp_short_dma] compiled xclbin={op.xclbin_artifact.filename} "
            f"insts={op.insts_artifact.filename}"
        )
    if compile_only:
        log("[exp_short_dma] compile-only PASS (1 shared xclbin, 2 instruction streams)")
        return 0
    fn_full, fn_short = op_full.get_callable(), op_short.get_callable()
    assert Path(op_full.xclbin_artifact.filename).resolve() == Path(
        op_short.xclbin_artifact.filename
    ).resolve(), "xclbin path mismatch"

    full = (
        np.arange(ROWS * W_FULL_U32, dtype=np.uint32)
        ^ np.uint32(0xA5A50000)
    ).reshape(ROWS, W_FULL_U32)
    short = (
        np.arange(ROWS * W_SHORT_U32, dtype=np.uint32)
        ^ np.uint32(0x5A5A0000)
    ).reshape(ROWS, W_SHORT_U32)
    full_bo = iron.tensor(full.reshape(-1), dtype=np.uint32, device="npu")
    short_bo = iron.tensor(short.reshape(-1), dtype=np.uint32, device="npu")
    y_bo = iron.zeros((ROWS * W_FULL_U32,), dtype=np.uint32, device="npu")

    t0 = time.perf_counter()
    for step in range(iters):
        is_full = step % 2 == 0
        expected = full if is_full else short
        run = fn_full if is_full else fn_short
        run(y_bo, full_bo if is_full else short_bo)
        y_bo._sync_from_device()
        actual = np.asarray(y_bo).reshape(ROWS, W_FULL_U32)[:, : expected.shape[1]]
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)[0]
            row, word = int(mismatch[0]), int(mismatch[1])
            raise AssertionError(
                f"step={step + 1} row={row} word={word}: "
                f"got=0x{int(actual[row, word]):08x} "
                f"expected=0x{int(expected[row, word]):08x}"
            )
        if step < 2 or (step + 1) % 1000 == 0 or step + 1 == iters:
            log(f"[exp_short_dma] {step + 1}/{iters} exact bit compare PASS")
    log(
        f"[exp_short_dma] PASS: {iters} alternating calls in "
        f"{time.perf_counter() - t0:.2f}s"
    )
    return 0


def run_variant_experiment(
    variant: VariantSpec,
    iters: int,
    *,
    compile_only: bool = False,
) -> int:
    """Compile and run a single non-V0 variant."""
    if variant.alt_sizes:
        return run_npu_experiment(iters, compile_only=compile_only)

    import aie.iron as iron
    import aie.utils as aie_utils
    from iron.common import (
        AIEContext,
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
    from iron.common.compilation import InstsBinArtifact, XclbinArtifact

    kernel = _echo_kernel()
    transfer_words = variant.transfer_words

    class ShortDMAVariantOp(MLIROperator):
        def __init__(self, context: Any):
            super().__init__(context=context)

        @property
        def config_name(self) -> str:
            return (
                f"Raon_ExpShortDMA_"
                f"{aie_utils.get_current_device().resolve().name}_{variant.name}"
            )

        @property
        def name(self) -> str:
            return f"{self.config_name}_w{transfer_words}"

        def get_mlir_artifact(self):
            return PythonGeneratedMLIRArtifact(
                f"{self.config_name}_w{transfer_words}.mlir",
                DesignGenerator(
                    Path(__file__).resolve(),
                    "short_dma_program",
                    (aie_utils.get_current_device(), transfer_words),
                    {"echo_kernel": kernel, "variant": variant},
                ),
            )

        def get_kernel_artifacts(self):
            return [KernelObjectArtifact.from_extern(kernel)]

        def set_up_artifacts(self) -> None:
            mlir = self.get_mlir_artifact()
            kernels = self.get_kernel_artifacts()
            self.xclbin_artifact = XclbinArtifact(
                f"{self.config_name}.xclbin",
                mlir_input=mlir,
                dependencies=[mlir, *kernels],
                kernel_name="MLIR_AIE",
                extra_flags=["--dynamic-objFifos"],
            )
            self.insts_artifact = InstsBinArtifact(
                f"{self.name}.bin",
                mlir_input=mlir,
                dependencies=[mlir, *kernels],
                extra_flags=["--dynamic-objFifos"],
            )
            if hasattr(self, "add_artifacts") and callable(
                getattr(self, "add_artifacts")
            ):
                self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

        def register_artifacts(self, artifacts: list[Any]) -> None:
            if hasattr(self, "add_artifacts") and callable(
                getattr(self, "add_artifacts")
            ):
                self.add_artifacts(artifacts)

        def get_arg_spec(self):
            per_block_w = ROWS * transfer_words
            total_w = variant.blocks * per_block_w
            total_y = variant.blocks * ROWS * W_FULL_U32
            specs = [
                AIERuntimeArgSpec("out", (total_y,), dtype=np.uint32),
                AIERuntimeArgSpec("in", (total_w,), dtype=np.uint32),
            ]
            if variant.x_like_fifo:
                specs.append(AIERuntimeArgSpec("x", (MAX_K,), dtype=np.uint32))
            return specs

    ctx = AIEContext()
    op = ShortDMAVariantOp(context=ctx)
    op.set_up_artifacts()
    op.compile()
    log(
        f"[exp_short_dma:{variant.name}] compiled "
        f"xclbin={op.xclbin_artifact.filename} "
        f"insts={op.insts_artifact.filename}"
    )
    if compile_only:
        log(f"[exp_short_dma:{variant.name}] compile-only PASS")
        return 0

    fn = op.get_callable()
    total_w = variant.blocks * ROWS * transfer_words
    total_y = variant.blocks * ROWS * W_FULL_U32

    w_host = (
        np.arange(total_w, dtype=np.uint32)
        ^ np.uint32(0xA5A50000)
    ).reshape(variant.blocks, ROWS, transfer_words)
    w_bo = iron.tensor(w_host.reshape(-1), dtype=np.uint32, device="npu")
    y_bo = iron.zeros((total_y,), dtype=np.uint32, device="npu")
    expected = w_host.reshape(variant.blocks, ROWS, transfer_words)
    if variant.x_like_fifo:
        x_bo = iron.zeros((MAX_K,), dtype=np.uint32, device="npu")

    t0 = time.perf_counter()
    for step in range(iters):
        if variant.x_like_fifo:
            fn(y_bo, w_bo, x_bo)
        else:
            fn(y_bo, w_bo)
        y_bo._sync_from_device()
        actual = np.asarray(y_bo).reshape(variant.blocks, ROWS, W_FULL_U32)[
            :, :, :transfer_words
        ]
        if not np.array_equal(actual, expected):
            mismatch = np.argwhere(actual != expected)[0]
            block, row, word = int(mismatch[0]), int(mismatch[1]), int(mismatch[2])
            raise AssertionError(
                f"[exp_short_dma:{variant.name}] step={step + 1} "
                f"block={block} row={row} word={word}: "
                f"got=0x{int(actual[block, row, word]):08x} "
                f"expected=0x{int(expected[block, row, word]):08x}"
            )
        if step < 2 or (step + 1) % 100 == 0 or step + 1 == iters:
            log(
                f"[exp_short_dma:{variant.name}] {step + 1}/{iters} "
                f"exact bit compare PASS"
            )
    log(
        f"[exp_short_dma:{variant.name}] PASS: {iters} calls "
        f"({variant.blocks} blocks/call) in {time.perf_counter() - t0:.2f}s"
    )
    return 0


def _run_phase(phase: str, iters: int, simulate_only: bool) -> int:
    count = 2 if phase == "first" else 10 if phase == "short" else iters
    sim = simulate_short_dma_ping_pong(count)
    assert sim["sentinel_preserved"], sim
    log(f"[exp_short_dma:{phase}] host model PASS: {sim}")
    return 0 if simulate_only else run_npu_experiment(count)


def _variant_phases(variant: VariantSpec, base_iters: int) -> list[tuple[str, int, float]]:
    """Return (phase_name, iters, timeout) for a variant."""
    if variant.alt_sizes:
        return [
            ("first", 2, 180.0),
            ("short", 10, 240.0),
            ("soak", base_iters, 1800.0),
        ]
    return [
        ("first", 2, 60.0),
        ("short", 10, 120.0),
        ("soak", 100, 300.0),
    ]


def _run_variant_child(variant_name: str, phase: str, iters: int) -> int:
    variant = _VARIANTS_BY_NAME[variant_name]
    status = run_variant_experiment(variant, iters)
    log(
        f"RESULT variant={variant_name} phase={phase} status=PASS calls={iters}"
    )
    return status


def _compile_all_variants(base_iters: int) -> int:
    """Compile every variant once."""
    log("[exp_short_dma] compile-only: all variants")
    try:
        import aie.utils as aie_utils

        aie_available = aie_utils is not None
    except Exception:  # pragma: no cover - CPU-only host lacks IRON runtime
        aie_available = False

    # Keep the original V0 compile path (the one exercised by host tests).
    v0_status = run_npu_experiment(base_iters, compile_only=True)
    if v0_status != 0:
        return v0_status

    if not aie_available:
        log(
            "[exp_short_dma] aie not available on this host; "
            "skipping multi-variant compiles"
        )
        return 0

    failures = 0
    for variant in VARIANTS:
        if variant.alt_sizes:
            continue
        try:
            run_variant_experiment(variant, base_iters, compile_only=True)
        except Exception as exc:
            log(f"[exp_short_dma:{variant.name}] compile FAILED: {exc}")
            failures += 1
    log(
        f"[exp_short_dma] compile-only finished: "
        f"{len(VARIANTS) - 1 - failures}/{len(VARIANTS) - 1} extra variants compiled"
    )
    return 0 if failures == 0 else 1


def _simulate_all_variants() -> int:
    log("[exp_short_dma] simulate-only: all variants")
    failures = 0
    for variant in VARIANTS:
        if variant.alt_sizes:
            sim = simulate_short_dma_ping_pong(iters=200)
        else:
            sim = simulate_short_dma_variant(variant, calls=5)
        ok = sim["sentinel_preserved"]
        log(
            f"[exp_short_dma:{variant.name}] simulate "
            f"{'PASS' if ok else 'FAIL'}: {sim}"
        )
        if not ok:
            failures += 1
    return 0 if failures == 0 else 1


def _run_all_variants(base_iters: int) -> int:
    """Run every variant with per-phase watchdogs and print a result table."""
    log("[exp_short_dma] running all variants")
    results: list[tuple[str, str, int]] = []
    for variant in VARIANTS:
        phases = _variant_phases(variant, base_iters)
        calls_completed = 0
        status = "PASS"
        for phase, phase_iters, timeout in phases:
            cmd = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--variant",
                variant.name,
                "--phase",
                phase,
                "--iters",
                str(phase_iters),
            ]
            try:
                subprocess.run(cmd, check=True, timeout=timeout)
                calls_completed += phase_iters
            except subprocess.TimeoutExpired:
                log(
                    f"[exp_short_dma:{variant.name}:{phase}] TIMEOUT after "
                    f"{timeout:.1f}s"
                )
                status = "TIMEOUT"
                break
            except subprocess.CalledProcessError as exc:
                log(
                    f"[exp_short_dma:{variant.name}:{phase}] FAILED rc={exc.returncode}"
                )
                status = f"FAIL(rc={exc.returncode})"
                break
        results.append((variant.name, status, calls_completed))
        log(
            f"[exp_short_dma:{variant.name}] summary: {status} calls={calls_completed}"
        )

    log("")
    log("| 변형 | blocks | rounds | x-like | mem | core | shim | 결과 | 호출 수 |")
    log("|---|---|---|---|---|---|---|---|---|")
    for variant, status, calls in zip(VARIANTS, [s for _, s, _ in results], [c for _, _, c in results]):
        chain_label = lambda chain: "chain" if chain else "sep"
        log(
            f"| {variant.name} | {variant.blocks} | {variant.rounds} | "
            f"{'yes' if variant.x_like_fifo else 'no'} | "
            f"{chain_label(variant.mem_chain)} | "
            f"{chain_label(variant.core_chain)} | "
            f"{chain_label(variant.shim_chain)} | {status} | {calls} |"
        )
    log("")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--simulate-only", action="store_true")
    p.add_argument(
        "--compile-only",
        action="store_true",
        help="Compile all variants, then exit.",
    )
    p.add_argument("--first-timeout", type=float, default=180.0)
    p.add_argument("--short-timeout", type=float, default=240.0)
    p.add_argument("--soak-timeout", type=float, default=1800.0)
    p.add_argument(
        "--phase", choices=("first", "short", "soak"), help=argparse.SUPPRESS
    )
    p.add_argument(
        "--variant",
        choices=["all", *(v.name for v in VARIANTS)],
        default="all",
        help="Variant to run, or 'all'.",
    )
    args = p.parse_args(argv)
    if args.iters <= 0:
        p.error("--iters must be positive")

    if args.phase:
        if args.variant == "all":
            p.error("--phase requires a specific --variant")
        return _run_variant_child(args.variant, args.phase, args.iters)

    if args.compile_only:
        if args.simulate_only:
            p.error("--compile-only cannot be combined with --simulate-only")
        return _compile_all_variants(args.iters)

    if args.simulate_only:
        return _simulate_all_variants()

    return _run_all_variants(args.iters)


if __name__ == "__main__":
    raise SystemExit(main())
