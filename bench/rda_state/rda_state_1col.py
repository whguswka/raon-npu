# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
#
"""Single-column, tile-wise RDA state update IRON design.

This design follows the adopted architecture in
the internal RDA-merge design note (section 2.4, not published).
One AIE column (parameter ``col``) handles all 40 heads:
worker ``t`` runs on core ``(col, 2 + t)`` and is responsible for V-tile ``t``
(rows ``t*32:(t+1)*32``) of every head.

Resource summary (per column):

| Resource                          | Count / Size                     |
|-----------------------------------|----------------------------------|
| Shim MM2S channels                | 2 (s_in, head_in)               |
| Shim S2MM channels                | 2 (s_out, out)                  |
| MemTile S2MM channels             | 5 (s_in 1, packed core 4)         |
| MemTile MM2S channels             | 6 (s_in split 4, s_out 1, out 1)|
| Core S2MM channels (each)         | 2 (s_in tile, head bundle)      |
| Core MM2S channels (each)         | 1 (packed [state tile + out])    |
| MemTile memory                    | ~257 KiB / 512 KiB              |
| Core L1 data footprint (each)     | ~49.3 KiB / 64 KiB              |
| MemTile locks (aggregate + cores) | 16 + 8 = 24 / 64                |
| MemTile explicit BDs              | 8 write + 16 read = 24 (+ s_in split ObjectFifo) |
| Shim active BDs (runtime sequence)| | 4 explicit fills/drains (+ s_in ObjectFifo producer) |

The packed output element is 4128 floats: 4096 for the updated 32x128 state
tile followed by 32 output floats.  The memory tile aggregates four such
elements (one per core) into a ``[4, 4128]`` ping/pong buffer, then two
memory-tile MM2S tasks read the same buffer with different TAPs:

* state read: offset 0, stride 4128, 4 x 4096 floats -> s_out BO
* out read:   offset 4096, stride 4128, 4 x 32 floats -> out BO

Lock protocol (per slot ``s`` and core ``t``):

* Producer (core ``t`` -> memtile slot ``s``): acquire ``agg_empty[s][t]``
  value 2, write packed element ``[t*4128:(t+1)*4128]``, release
  ``agg_full[s][t]`` value 2.
* State consumer (memtile slot ``s``, core ``t`` segment): acquire
  ``agg_full[s][t]`` value 1, read 4096 floats, release ``agg_empty[s][t]``
  value 1.
* Out consumer (memtile slot ``s``, core ``t`` segment): acquire
  ``agg_full[s][t]`` value 1, read 32 floats, release ``agg_empty[s][t]``
  value 1.

The producer cannot reuse a slot until both consumers have returned their
empty units (2 total), so no read is overwritten by the next head.

Shim transfers move the whole tensor once per invocation instead of once per
head, staying under the 16 shim BD limit.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# Shared RDA geometry constants (mirrored from bench/rda_state/rda_state.py)
# --------------------------------------------------------------------------
V_DIM = 128
K_DIM = 128
TILE_V = 32
TILES_PER_HEAD = V_DIM // TILE_V  # 4
HEAD_BUNDLE_SIZE = 640

# Head count fixed by the Raon-1-mini model contract (``rda_num_heads``).  The
# merged unified config path (``UnifiedGEMV`` with ``rda_col`` / ``UnifiedRDA``)
# accepts exactly this value: its shared xclbin/insts artifact names do not carry
# a head count.  The standalone ``rda_step_1col`` design keeps the head count as
# a compile-time JIT argument (so the JIT cache keys artifacts on it) but, like
# this module's BD chains, only accepts positive even counts because both output
# drain chains use ``runs=H // 2`` and would leave an odd last head unconsumed.
MODEL_RDA_HEADS = 40


def pack_head_bundle(q, k, a, v, beta):
    """Pack per-head vectors into contiguous head bundle [H, 640] fp32."""
    H = q.shape[0]
    bundle = np.zeros((H, HEAD_BUNDLE_SIZE), dtype=np.float32)
    bundle[:, 0:128] = q
    bundle[:, 128:256] = k
    bundle[:, 256:384] = a
    bundle[:, 384:512] = v
    bundle[:, 512] = beta
    return bundle


def numpy_recurrent_rda_step(S, q, k, a, v, beta):
    """Numpy reference calculation for one RDA decode step."""
    S_decay = S * a[:, None, :]
    pred = np.einsum("hvk,hk->hv", S_decay, k)
    err = v - pred
    scale = (beta[:, None] * err)[:, :, None]
    S_new = S_decay + scale * k[:, None, :]
    out = np.einsum("hvk,hk->hv", S_new, q)
    return S_new, out

# --------------------------------------------------------------------------
# IRON availability
# --------------------------------------------------------------------------
# On the CPU host server there is no compiled IRON package, so the JIT design
# cannot be compiled here.  The pure-Python TAP helpers below are still
# exercised by the host-side tests.  The notebook has a working IRON install
# and runs the comparison harness.
try:
    import aie.iron as iron
    from aie.helpers.taplib import TensorAccessPattern as _TensorAccessPattern
    from aie.iron import (
        Acquire,
        Bd,
        Buffer,
        CompileTime,
        ExternalFunction,
        Flow,
        In,
        Lock,
        ObjectFifo,
        Out,
        Program,
        Release,
        Runtime,
        TaskGroup,
        Worker,
        WorkerRuntimeBarrier,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile
    from aie.utils.config import cxx_header_path

    _IRON_AVAILABLE = True
except Exception:

    class _TensorAccessPattern:
        """Minimal stand-in for host-side TAP address verification."""

        def __init__(self, tensor_dims, offset, sizes, strides):
            self.tensor_dims = tuple(tensor_dims)
            self.offset = offset
            self.sizes = tuple(sizes)
            self.strides = tuple(strides)

        def gather(self, tensor):
            flat = np.asarray(tensor).reshape(-1)
            indices = []

            def rec(pos, dim):
                if dim == len(self.sizes):
                    indices.append(pos)
                    return
                for i in range(int(self.sizes[dim])):
                    rec(pos + i * int(self.strides[dim]), dim + 1)

            rec(int(self.offset), 0)
            return flat[np.array(indices, dtype=np.int64)]

    _IRON_AVAILABLE = False

    def cxx_header_path():
        return ""

TensorAccessPattern = _TensorAccessPattern

_KERNEL_PATH = Path(__file__).resolve().parent.parent.parent / "kernels" / "rda_state.cc"

# One packed result element per core: updated state tile + output slice.
PACKED_ELEMS = TILE_V * K_DIM + TILE_V  # 4096 + 32 = 4128


def _aggregate_buffer_shape():
    """Shape of the memory-tile output aggregation buffer."""
    return (TILES_PER_HEAD, PACKED_ELEMS)


def _s_in_head_tap(head_idx, num_heads=40):
    """TAP for one head's four state tiles from the s_in BO."""
    return TensorAccessPattern(
        (num_heads, V_DIM, K_DIM),
        offset=head_idx * V_DIM * K_DIM,
        sizes=(TILES_PER_HEAD, TILE_V, K_DIM),
        strides=(TILE_V * K_DIM, K_DIM, 1),
    )


def _s_in_tile_tap(head_idx, tile_idx, num_heads=40):
    """TAP for tile ``tile_idx`` of head ``head_idx`` from the s_in BO."""
    return TensorAccessPattern(
        (num_heads, V_DIM, K_DIM),
        offset=head_idx * V_DIM * K_DIM + tile_idx * TILE_V * K_DIM,
        sizes=(1, TILE_V, K_DIM),
        strides=(0, K_DIM, 1),
    )


def _s_out_tile_tap(head_idx, tile_idx, num_heads=40):
    """TAP for writing tile ``tile_idx`` of head ``head_idx`` into s_out BO."""
    return TensorAccessPattern(
        (num_heads, V_DIM, K_DIM),
        offset=head_idx * V_DIM * K_DIM + tile_idx * TILE_V * K_DIM,
        sizes=(1, TILE_V, K_DIM),
        strides=(0, K_DIM, 1),
    )


def _out_tile_tap(head_idx, tile_idx, num_heads=40):
    """TAP for writing tile ``tile_idx`` of head ``head_idx`` into out BO."""
    return TensorAccessPattern(
        (num_heads, V_DIM),
        offset=head_idx * V_DIM + tile_idx * TILE_V,
        sizes=(1, TILE_V),
        strides=(0, 1),
    )


def _aggregate_core_write_tap(tile_idx):
    """TAP for core ``tile_idx`` writing its packed element into the aggregate buffer."""
    return TensorAccessPattern(
        _aggregate_buffer_shape(),
        offset=tile_idx * PACKED_ELEMS,
        sizes=(1, PACKED_ELEMS),
        strides=(0, 1),
    )


def _aggregate_state_read_tap(tile_idx):
    """TAP reading the state portion of core ``tile_idx`` from the aggregate buffer."""
    return TensorAccessPattern(
        _aggregate_buffer_shape(),
        offset=tile_idx * PACKED_ELEMS,
        sizes=(1, TILE_V * K_DIM),
        strides=(0, 1),
    )


def _aggregate_out_read_tap(tile_idx):
    """TAP reading the output portion of core ``tile_idx`` from the aggregate buffer."""
    return TensorAccessPattern(
        _aggregate_buffer_shape(),
        offset=tile_idx * PACKED_ELEMS + TILE_V * K_DIM,
        sizes=(1, TILE_V),
        strides=(0, 1),
    )


def _packed_state_read_tap(out_offset=4096):
    """TAP reading the state portion from the packed aggregate buffer.

    Reads 4 x 4096 floats with stride 4128 starting at offset 0.
    The returned flat stream is exactly one head's [V, K] row-major data.
    """
    return TensorAccessPattern(
        _aggregate_buffer_shape(),
        offset=0,
        sizes=(TILES_PER_HEAD, TILE_V * K_DIM),
        strides=(PACKED_ELEMS, 1),
    )


def _packed_out_read_tap(out_offset=4096):
    """TAP reading the output portion from the packed aggregate buffer.

    Reads 4 x 32 floats with stride 4128 starting at ``out_offset``.
    The returned flat stream is exactly one head's [V] row-major data.
    """
    return TensorAccessPattern(
        _aggregate_buffer_shape(),
        offset=out_offset,
        sizes=(TILES_PER_HEAD, TILE_V),
        strides=(PACKED_ELEMS, 1),
    )


@dataclass
class RdaResource:
    """One fixed resource declared by a single-column RDA channel."""

    kind: str
    name: str
    init: int | None = None
    elements: int | None = None


def rda_column_plan(col: int, num_heads: int = 40) -> tuple[RdaResource, ...]:
    """Return the canonical resource table for one RDA column.

    This table is shared by the standalone 1-column design and the merged
    unified GEMV design so both declare the same ObjectFifo/Buffer/Lock/Flow
    names with identical initial values.
    """
    c = col
    resources: list[RdaResource] = []
    resources.append(RdaResource("objectfifo", f"rda_s_in_l3l2_{c}"))
    for t in range(TILES_PER_HEAD):
        resources.append(RdaResource("objectfifo", f"rda_s_in_l2l1_{c}_{t}"))
    resources.append(RdaResource("objectfifo", f"rda_head_in_{c}"))
    for s in range(2):
        resources.append(
            RdaResource(
                "buffer",
                f"rda_agg_mem_{c}_{s}",
                elements=TILES_PER_HEAD * PACKED_ELEMS,
            )
        )
    for t in range(TILES_PER_HEAD):
        resources.append(
            RdaResource("buffer", f"rda_packed_core_{c}_{t}", elements=PACKED_ELEMS)
        )
    for s in range(2):
        for t in range(TILES_PER_HEAD):
            resources.append(
                RdaResource("lock", f"rda_agg_empty_{c}_{s}_{t}", init=2)
            )
            resources.append(
                RdaResource("lock", f"rda_agg_full_{c}_{s}_{t}", init=0)
            )
    for t in range(TILES_PER_HEAD):
        resources.append(
            RdaResource("lock", f"rda_packed_empty_{c}_{t}", init=1)
        )
        resources.append(
            RdaResource("lock", f"rda_packed_full_{c}_{t}", init=0)
        )
    for t in range(TILES_PER_HEAD):
        resources.append(RdaResource("flow", f"rda_packed_c2m_{c}_{t}"))
    resources.append(RdaResource("flow", f"rda_state_m2s_{c}"))
    resources.append(RdaResource("flow", f"rda_out_m2s_{c}"))
    return tuple(resources)


@dataclass
class RDAChannel:
    """All IRON objects belonging to a single-column RDA channel."""

    col: int
    num_heads: int
    shim_tile: Any
    memtile: Any
    core_tiles: list[Any]
    s_l3_ty: Any
    head_in_l3_ty: Any
    out_l3_ty: Any
    s_in_prod: Any
    head_in_prod: Any
    flows_c2m: list[Any]
    flow_state_m2s: Any
    flow_out_m2s: Any
    agg_mem_buffers: list[Any]
    packed_core_buffers: list[Any]
    agg_empty: list[list[Any]]
    agg_full: list[list[Any]]
    core_empty: list[Any]
    core_full: list[Any]
    barriers: list[Any]
    workers: list[Any]


def make_rda_kernel_object():
    """Return the IRON ExternalFunction object for ``rda_update_tile_packed``."""
    if not _IRON_AVAILABLE:
        return None
    tile_s_ty = np.ndarray[(TILE_V, K_DIM), np.dtype[np.float32]]
    tile_head_bundle_ty = np.ndarray[(HEAD_BUNDLE_SIZE,), np.dtype[np.float32]]
    packed_ty = np.ndarray[(PACKED_ELEMS,), np.dtype[np.float32]]
    return ExternalFunction(
        "rda_update_tile_packed",
        source_file=str(_KERNEL_PATH),
        arg_types=[tile_s_ty, packed_ty, tile_head_bundle_ty, np.int32],
        include_dirs=[cxx_header_path()],
    )


def build_rda_channel(col: int, num_heads: int = MODEL_RDA_HEADS) -> RDAChannel:
    """Build the IRON objects for a single-column RDA channel.

    The standalone design takes the head count as a compile-time JIT argument, so
    each accepted count gets its own cached artifacts.  Only positive even counts
    are accepted: the core-to-mem and output drain BD chains all run ``H // 2``
    times, so an odd ``H`` would never consume the last head and the invocation
    could stall.
    """
    if num_heads <= 0 or num_heads % 2:
        raise ValueError(
            f"num_heads must be a positive even head count: every BD chain in the "
            f"design runs runs=H//2 times, so an odd count (or a non-positive one) "
            f"would leave heads unconsumed and could stall, got {num_heads}"
        )
    if not _IRON_AVAILABLE:
        raise RuntimeError("IRON is not available; cannot build RDA channel")

    H = num_heads
    c = col

    shim_tile = Tile(c, 0)
    memtile = Tile(c, 1)
    core_tiles = [Tile(c, 2 + t) for t in range(TILES_PER_HEAD)]

    s_l3_ty = np.ndarray[(H, V_DIM, K_DIM), np.dtype[np.float32]]
    head_in_l3_ty = np.ndarray[(H, HEAD_BUNDLE_SIZE), np.dtype[np.float32]]
    out_l3_ty = np.ndarray[(H, V_DIM), np.dtype[np.float32]]

    tile_s_ty = np.ndarray[(TILE_V, K_DIM), np.dtype[np.float32]]
    tile_head_bundle_ty = np.ndarray[(HEAD_BUNDLE_SIZE,), np.dtype[np.float32]]
    packed_ty = np.ndarray[(PACKED_ELEMS,), np.dtype[np.float32]]
    agg_mem_ty = np.ndarray[_aggregate_buffer_shape(), np.dtype[np.float32]]

    rda_kernel = make_rda_kernel_object()

    head_element_ty = np.ndarray[
        (TILES_PER_HEAD, TILE_V, K_DIM), np.dtype[np.float32]
    ]
    of_s_in_l3l2 = ObjectFifo(head_element_ty, depth=2, name=f"rda_s_in_l3l2_{c}")
    s_in_prod = of_s_in_l3l2.prod(tile=shim_tile)

    of_s_in_subs = of_s_in_l3l2.cons().split(
        offsets=[t * TILE_V * K_DIM for t in range(TILES_PER_HEAD)],
        tile=memtile,
        obj_types=[tile_s_ty] * TILES_PER_HEAD,
        names=[f"rda_s_in_l2l1_{c}_{t}" for t in range(TILES_PER_HEAD)],
        depths=[1] * TILES_PER_HEAD,
    )
    core_s_in_handles = [sub.cons() for sub in of_s_in_subs]

    of_head_in = ObjectFifo(
        tile_head_bundle_ty, depth=2, name=f"rda_head_in_{c}"
    )
    head_in_prod = of_head_in.prod(tile=shim_tile)
    core_head_in_handles = [
        of_head_in.cons(tile=core_tiles[t]) for t in range(TILES_PER_HEAD)
    ]

    agg_mem_buffers = [
        Buffer(type=agg_mem_ty, tile=memtile, name=f"rda_agg_mem_{c}_{s}")
        for s in range(2)
    ]
    packed_core_buffers = [
        Buffer(
            type=packed_ty,
            tile=core_tiles[t],
            name=f"rda_packed_core_{c}_{t}",
        )
        for t in range(TILES_PER_HEAD)
    ]

    agg_empty = [
        [
            Lock(memtile, init=2, name=f"rda_agg_empty_{c}_{s}_{t}")
            for t in range(TILES_PER_HEAD)
        ]
        for s in range(2)
    ]
    agg_full = [
        [
            Lock(memtile, init=0, name=f"rda_agg_full_{c}_{s}_{t}")
            for t in range(TILES_PER_HEAD)
        ]
        for s in range(2)
    ]

    core_empty = [
        Lock(core_tiles[t], init=1, name=f"rda_packed_empty_{c}_{t}")
        for t in range(TILES_PER_HEAD)
    ]
    core_full = [
        Lock(core_tiles[t], init=0, name=f"rda_packed_full_{c}_{t}")
        for t in range(TILES_PER_HEAD)
    ]

    flows_c2m = [
        Flow(core_tiles[t], memtile, name=f"rda_packed_c2m_{c}_{t}")
        for t in range(TILES_PER_HEAD)
    ]
    flow_state_m2s = Flow(memtile, shim_tile, name=f"rda_state_m2s_{c}")
    flow_out_m2s = Flow(memtile, shim_tile, name=f"rda_out_m2s_{c}")

    barriers = [WorkerRuntimeBarrier() for _ in range(TILES_PER_HEAD)]

    def core_fn(
        s_in_handle,
        head_in_handle,
        packed_buf,
        kernel,
        tile_idx,
        core_empty_lock,
        core_full_lock,
        barrier,
    ):
        barrier.wait_for_value(1)
        barrier.release_with_value(1)
        for _ in range_(H):
            core_empty_lock.acquire(1)
            elem_head = head_in_handle.acquire(1)
            elem_s_in = s_in_handle.acquire(1)

            kernel(elem_s_in, packed_buf, elem_head, tile_idx)
            core_full_lock.release(1)

            s_in_handle.release(1)
            head_in_handle.release(1)

    workers = []
    for t in range(TILES_PER_HEAD):
        worker = Worker(
            core_fn,
            [
                core_s_in_handles[t],
                core_head_in_handles[t],
                packed_core_buffers[t],
                rda_kernel,
                t,
                core_empty[t],
                core_full[t],
                barriers[t],
            ],
            tile=core_tiles[t],
            stack_size=12288,
        )
        workers.append(worker)

    return RDAChannel(
        col=c,
        num_heads=H,
        shim_tile=shim_tile,
        memtile=memtile,
        core_tiles=core_tiles,
        s_l3_ty=s_l3_ty,
        head_in_l3_ty=head_in_l3_ty,
        out_l3_ty=out_l3_ty,
        s_in_prod=s_in_prod,
        head_in_prod=head_in_prod,
        flows_c2m=flows_c2m,
        flow_state_m2s=flow_state_m2s,
        flow_out_m2s=flow_out_m2s,
        agg_mem_buffers=agg_mem_buffers,
        packed_core_buffers=packed_core_buffers,
        agg_empty=agg_empty,
        agg_full=agg_full,
        core_empty=core_empty,
        core_full=core_full,
        barriers=barriers,
        workers=workers,
    )


def rda_sequence_body(
    channel: RDAChannel,
    s_in_arg,
    s_out_arg,
    head_in_arg,
    out_arg,
    s_in_prod_seq,
    head_in_prod_seq,
):
    """Runtime sequence for one RDA token using a pre-built ``RDAChannel``."""
    H = channel.num_heads
    memtile = channel.memtile
    core_tiles = channel.core_tiles
    agg_mem_buffers = channel.agg_mem_buffers
    packed_core_buffers = channel.packed_core_buffers
    agg_empty = channel.agg_empty
    agg_full = channel.agg_full
    core_empty = channel.core_empty
    core_full = channel.core_full
    flows_c2m = channel.flows_c2m
    flow_state_m2s = channel.flow_state_m2s
    flow_out_m2s = channel.flow_out_m2s

    tg = TaskGroup()

    # Release RDA worker barriers so they can process the incoming head.
    for barrier in channel.barriers:
        barrier.set(1)

    s_in_prod_seq.fill(s_in_arg, wait=False, group=tg)
    head_in_prod_seq.fill(head_in_arg, wait=False, group=tg)

    core_to_mem_tasks = []
    for t in range(TILES_PER_HEAD):
        core_tap = TensorAccessPattern.full((PACKED_ELEMS,))
        mem_tap = _aggregate_core_write_tap(t)
        core_task = flows_c2m[t].endpoint(core_tiles[t]).task(
            Bd(
                packed_core_buffers[t],
                tap=core_tap,
                acquires=[Acquire(core_full[t], value=1)],
                releases=[Release(core_empty[t], value=1)],
            ),
            runs=H,
            wait=False,
        )
        mem_task = flows_c2m[t].endpoint(memtile).task(
            Bd(
                agg_mem_buffers[0],
                tap=mem_tap,
                acquires=[Acquire(agg_empty[0][t], value=2)],
                releases=[Release(agg_full[0][t], value=2)],
            ),
            Bd(
                agg_mem_buffers[1],
                tap=mem_tap,
                acquires=[Acquire(agg_empty[1][t], value=2)],
                releases=[Release(agg_full[1][t], value=2)],
            ),
            runs=H // 2,
            wait=False,
        )
        core_task.start()
        mem_task.start()
        core_to_mem_tasks.append((core_task, mem_task))

    state_bds = []
    out_bds = []
    for s in range(2):
        for t in range(TILES_PER_HEAD):
            state_bds.append(
                Bd(
                    agg_mem_buffers[s],
                    tap=_aggregate_state_read_tap(t),
                    acquires=[Acquire(agg_full[s][t], value=1)],
                    releases=[Release(agg_empty[s][t], value=1)],
                )
            )
            out_bds.append(
                Bd(
                    agg_mem_buffers[s],
                    tap=_aggregate_out_read_tap(t),
                    acquires=[Acquire(agg_full[s][t], value=1)],
                    releases=[Release(agg_empty[s][t], value=1)],
                )
            )
    state_drain_task = flow_state_m2s.endpoint(memtile).task(
        *state_bds,
        runs=H // 2,
        wait=False,
    )
    out_drain_task = flow_out_m2s.endpoint(memtile).task(
        *out_bds,
        runs=H // 2,
        wait=False,
    )
    state_drain_task.start()
    out_drain_task.start()

    flow_state_m2s.drain(s_out_arg, wait=True, group=tg)
    flow_out_m2s.drain(out_arg, wait=True, group=tg)
    tg.finish()

    for core_task, mem_task in core_to_mem_tasks:
        core_task.free()
        mem_task.free()
    state_drain_task.free()
    out_drain_task.free()


if _IRON_AVAILABLE:

    @iron.jit
    def rda_step_1col(
        s_in: In,
        s_out: Out,
        head_in: In,
        out: Out,
        *,
        num_heads: CompileTime[int] = MODEL_RDA_HEADS,
        col: CompileTime[int] = 0,
    ):
        """Run one decode token step of recurrent RDA on a single column."""
        channel = build_rda_channel(col, num_heads)

        def sequence(
            s_in_arg,
            s_out_arg,
            head_in_arg,
            out_arg,
            s_in_prod_seq,
            head_in_prod_seq,
        ):
            rda_sequence_body(
                channel,
                s_in_arg,
                s_out_arg,
                head_in_arg,
                out_arg,
                s_in_prod_seq,
                head_in_prod_seq,
            )

        rt = Runtime(
            sequence,
            [
                channel.s_l3_ty,
                channel.s_l3_ty,
                channel.head_in_l3_ty,
                channel.out_l3_ty,
                channel.s_in_prod,
                channel.head_in_prod,
            ],
        )
        for s in range(2):
            for t in range(TILES_PER_HEAD):
                rt.add_lock(channel.agg_empty[s][t])
                rt.add_lock(channel.agg_full[s][t])
        for t in range(TILES_PER_HEAD):
            rt.add_lock(channel.core_empty[t])
            rt.add_lock(channel.core_full[t])
            rt.add_flow(channel.flows_c2m[t])
        rt.add_flow(channel.flow_state_m2s)
        rt.add_flow(channel.flow_out_m2s)

        prog = Program(iron.get_current_device(), rt, workers=channel.workers)
        return prog.resolve_program()

    def main():
        parser = argparse.ArgumentParser(
            description="RDA 1-column tile-wise state update"
        )
        parser.add_argument("--tokens", type=int, default=64)
        parser.add_argument("--heads", type=int, default=40)
        parser.add_argument("--col", type=int, default=0)
        parser.add_argument("--warmup", type=int, default=5)
        parser.add_argument("--seed", type=int, default=42)
        args = parser.parse_args()

        H = args.heads
        T = args.tokens
        c = args.col

        print("=" * 70)
        print("RDA 1-column tile-wise state update microbenchmark")
        print(f"Configuration: col={c}, Heads={H}, Tokens={T}")
        print("=" * 70)

        rng = np.random.default_rng(args.seed)
        q_all = rng.standard_normal((T, H, K_DIM)).astype(np.float32)
        q_all /= np.linalg.norm(q_all, axis=-1, keepdims=True) + 1e-12
        k_all = rng.standard_normal((T, H, K_DIM)).astype(np.float32)
        k_all /= np.linalg.norm(k_all, axis=-1, keepdims=True) + 1e-12
        log_alpha_all = rng.uniform(-5.0, 0.0, size=(T, H, K_DIM)).astype(np.float32)
        a_all = np.exp(log_alpha_all)
        beta_all = rng.uniform(0.01, 0.99, size=(T, H)).astype(np.float32)
        v_all = rng.standard_normal((T, H, V_DIM)).astype(np.float32)
        S_init = rng.standard_normal((H, V_DIM, K_DIM)).astype(np.float32) * 0.01

        print("\n[1/3] Computing CPU golden reference...")
        ref_outs = np.zeros((T, H, V_DIM), dtype=np.float32)
        S_ref = S_init.copy()
        for t in range(T):
            S_ref, ref_outs[t] = numpy_recurrent_rda_step(
                S_ref, q_all[t], k_all[t], a_all[t], v_all[t], beta_all[t]
            )
        print("Golden reference computed.")

        print(
            f"\n[2/3] Allocating NPU buffers and warming up JIT ({args.warmup} runs)..."
        )
        s_in_npu = iron.tensor(S_init, device="npu")
        s_out_npu = iron.zeros_like(s_in_npu)
        head_in_npu = iron.zeros(
            (H, HEAD_BUNDLE_SIZE), dtype=np.float32, device="npu"
        )
        out_npu = iron.zeros((H, V_DIM), dtype=np.float32, device="npu")

        head_in_npu[:] = pack_head_bundle(
            q_all[0], k_all[0], a_all[0], v_all[0], beta_all[0]
        )
        for _ in range(args.warmup):
            rda_step_1col(
                s_in_npu, s_out_npu, head_in_npu, out_npu, num_heads=H, col=c
            )

        print(f"\n[3/3] Executing {T} sequential tokens...")
        s_in_npu[:] = S_init
        npu_outs = np.zeros((T, H, V_DIM), dtype=np.float32)
        token_latencies_us = []
        for t in range(T):
            head_in_npu[:] = pack_head_bundle(
                q_all[t], k_all[t], a_all[t], v_all[t], beta_all[t]
            )
            t_start = time.perf_counter()
            rda_step_1col(
                s_in_npu, s_out_npu, head_in_npu, out_npu, num_heads=H, col=c
            )
            t_end = time.perf_counter()
            token_latencies_us.append((t_end - t_start) * 1e6)
            npu_outs[t] = out_npu.numpy()
            s_in_npu[:] = s_out_npu

        s_final_npu = s_out_npu.numpy().copy()

        out_diff = npu_outs - ref_outs
        out_max_err = float(np.max(np.abs(out_diff)))
        out_rel_l2 = float(
            np.linalg.norm(out_diff) / (np.linalg.norm(ref_outs) + 1e-12)
        )

        s_diff = s_final_npu - S_ref
        s_max_err = float(np.max(np.abs(s_diff)))
        s_rmse = float(np.sqrt(np.mean(s_diff**2)))
        s_range = float(np.max(S_ref) - np.min(S_ref) + 1e-12)
        s_nrmse = s_rmse / s_range

        print("\n" + "=" * 70)
        print("RDA 1-column tile-wise evaluation report")
        print("=" * 70)
        print(f"Tokens Processed : {T}")
        print(f"Heads / Column   : {H} / {c}")
        print("-" * 70)
        print("ACCURACY:")
        print(f"  out max err    : {out_max_err:.6e}")
        print(f"  out rel l2     : {out_rel_l2:.6e}")
        print(f"  s max err      : {s_max_err:.6e}")
        print(f"  s nrmse        : {s_nrmse:.6e}")
        p50 = float(np.percentile(token_latencies_us, 50))
        print("-" * 70)
        print(f"TIMING: p50={p50:.1f} us")
        print("=" * 70)

        if s_nrmse <= 1e-4 and out_rel_l2 <= 1e-5:
            print("VERDICT: PASS")
        else:
            print("VERDICT: FAIL")
            sys.exit(1)

    if __name__ == "__main__":
        main()
