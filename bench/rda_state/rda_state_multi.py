# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
#
"""Prefill RDA channel that updates T tokens per call (DESIGN-prefill.md section 36, stage R3-2).

The decode channel (``rda_state_1col.build_rda_channel``) moves every head's state in and out once per token.  This
prefill-only channel keeps its column layout - core ``(col, 2 + t)`` owns V-tile ``t`` of every head, the memory tile
gathers the four cores' packed elements and splits them into the state and output BOs - but a core acquires a head's
state tile once, runs ``rda_update_tile_multi`` (kernels/rda_state_multi.cc) for each of the call's T tokens and
sends its packed element once:

* T comes from a one-word runtime parameter per core that the instruction stream writes before the barrier; the core
  reads it once right after the barrier, so one core program serves the instruction stream of every T in
  ``RDA_TOKEN_CHOICES``;
* head bundles stream head-major: head_in is [H, T, 640] (the bundle of token t for head h at [h, t]);
* the packed element holds the state tile and ``RDA_MAX_TOKENS`` output slices (4096 + 32 * 8 floats); the core sends
  it whole, the memory tile drains the state and the first 32 * T outputs, so out is [H, 4 tiles, T, 32].

The decode builder, its kernel object and the decode xclbin are untouched: this module only adds objects (named
``rdam_*``), and its kernel is a separate translation unit that includes rda_state.cc.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bench.rda_state.rda_state_1col import (
    _IRON_AVAILABLE,
    _KERNEL_PATH as RDA_STATE_KERNEL_PATH,
    HEAD_BUNDLE_SIZE,
    K_DIM,
    MODEL_RDA_HEADS,
    TILE_V,
    TILES_PER_HEAD,
    V_DIM,
    TensorAccessPattern,
    cxx_header_path,
)

if _IRON_AVAILABLE:
    from aie.iron import (
        Acquire,
        Bd,
        Buffer,
        ExternalFunction,
        Flow,
        Lock,
        ObjectFifo,
        Release,
        TaskGroup,
        Worker,
        WorkerRuntimeBarrier,
    )
    from aie.iron.controlflow import range_
    from aie.iron.device import Tile

RDA_MAX_TOKENS = 8
RDA_TOKEN_CHOICES = (1, 2, 4, 8)
MULTI_PACKED_ELEMS = TILE_V * K_DIM + TILE_V * RDA_MAX_TOKENS  # 4096 + 256 = 4352
MULTI_KERNEL_PATH = Path(__file__).resolve().parent.parent.parent / "kernels" / "rda_state_multi.cc"


def check_rda_tokens(tokens: int) -> int:
    """Return ``tokens`` if it is one of ``RDA_TOKEN_CHOICES``, else raise ValueError."""
    if type(tokens) is not int or tokens not in RDA_TOKEN_CHOICES:
        raise ValueError(f"RDA tokens per call must be one of {RDA_TOKEN_CHOICES}, got {tokens!r}")
    return tokens


@dataclass
class RDAMultiChannel:
    """All IRON objects of the prefill T-token RDA channel."""

    col: int
    num_heads: int
    memtile: Any
    core_tiles: list[Any]
    s_l3_ty: Any
    s_in_prod: Any
    head_in_prod: Any
    flows_c2m: list[Any]
    flow_state_m2s: Any
    flow_out_m2s: Any
    agg_mem_buffers: list[Any]
    packed_core_buffers: list[Any]
    token_rtps: list[Any]
    agg_empty: list[list[Any]]
    agg_full: list[list[Any]]
    core_empty: list[Any]
    core_full: list[Any]
    barriers: list[Any]
    workers: list[Any]

    def head_in_l3_ty(self, tokens: int):
        """Host head BO type of a ``tokens``-token call: [H, T, 640] fp32, head-major."""
        return np.ndarray[(self.num_heads, check_rda_tokens(tokens), HEAD_BUNDLE_SIZE), np.dtype[np.float32]]

    def out_l3_ty(self, tokens: int):
        """Host output BO type of a ``tokens``-token call: [H, 4, T, 32] fp32."""
        return np.ndarray[
            (self.num_heads, TILES_PER_HEAD, check_rda_tokens(tokens), TILE_V), np.dtype[np.float32]
        ]


def make_rda_multi_kernel_object():
    """Return the IRON ExternalFunction for ``rda_update_tile_multi`` (None without IRON).

    The kernels folder is an include directory because the compiler sees a staged copy of rda_state_multi.cc and its
    ``#include "rda_state.cc"`` must still find the decode kernel.
    """
    if not _IRON_AVAILABLE:
        return None
    f32 = np.dtype[np.float32]
    return ExternalFunction(
        "rda_update_tile_multi",
        source_file=str(MULTI_KERNEL_PATH),
        arg_types=[
            np.ndarray[(TILE_V, K_DIM), f32],
            np.ndarray[(MULTI_PACKED_ELEMS,), f32],
            np.ndarray[(HEAD_BUNDLE_SIZE,), f32],
            np.int32,
            np.int32,
            np.int32,
        ],
        include_dirs=[cxx_header_path(), str(MULTI_KERNEL_PATH.parent)],
    )


def build_rda_multi_channel(col: int, num_heads: int = MODEL_RDA_HEADS) -> RDAMultiChannel:
    """Build the IRON objects of the T-token RDA channel on column ``col``.

    Like the decode channel only positive even head counts are accepted: the memory-tile chains run ``H // 2`` times.
    """
    if num_heads <= 0 or num_heads % 2:
        raise ValueError(f"num_heads must be a positive even head count, got {num_heads}")
    if not _IRON_AVAILABLE:
        raise RuntimeError("IRON is not available; cannot build the RDA multi-token channel")

    H = num_heads
    c = col
    P = MULTI_PACKED_ELEMS
    f32 = np.dtype[np.float32]
    shim_tile = Tile(c, 0)
    memtile = Tile(c, 1)
    core_tiles = [Tile(c, 2 + t) for t in range(TILES_PER_HEAD)]
    tile_s_ty = np.ndarray[(TILE_V, K_DIM), f32]
    bundle_ty = np.ndarray[(HEAD_BUNDLE_SIZE,), f32]
    packed_ty = np.ndarray[(P,), f32]
    rda_kernel = make_rda_multi_kernel_object()

    of_s_in_l3l2 = ObjectFifo(np.ndarray[(TILES_PER_HEAD, TILE_V, K_DIM), f32], depth=2, name=f"rdam_s_in_l3l2_{c}")
    s_in_prod = of_s_in_l3l2.prod(tile=shim_tile)
    of_s_in_subs = of_s_in_l3l2.cons().split(
        offsets=[t * TILE_V * K_DIM for t in range(TILES_PER_HEAD)],
        tile=memtile,
        obj_types=[tile_s_ty] * TILES_PER_HEAD,
        names=[f"rdam_s_in_l2l1_{c}_{t}" for t in range(TILES_PER_HEAD)],
        depths=[1] * TILES_PER_HEAD,
    )
    of_head_in = ObjectFifo(bundle_ty, depth=2, name=f"rdam_head_in_{c}")
    head_in_prod = of_head_in.prod(tile=shim_tile)

    agg_mem_buffers = [
        Buffer(type=np.ndarray[(TILES_PER_HEAD, P), f32], tile=memtile, name=f"rdam_agg_mem_{c}_{s}")
        for s in range(2)
    ]
    packed_core_buffers = [
        Buffer(type=packed_ty, tile=core_tiles[t], name=f"rdam_packed_core_{c}_{t}") for t in range(TILES_PER_HEAD)
    ]
    token_rtps = [
        Buffer(
            type=np.ndarray[(1,), np.dtype[np.int32]],
            tile=core_tiles[t],
            name=f"rdam_tokens_{c}_{t}",
            initial_value=np.ones(1, dtype=np.int32),
            use_write_rtp=True,
        )
        for t in range(TILES_PER_HEAD)
    ]
    agg_empty = [[Lock(memtile, init=2, name=f"rdam_agg_empty_{c}_{s}_{t}") for t in range(TILES_PER_HEAD)]
                 for s in range(2)]
    agg_full = [[Lock(memtile, init=0, name=f"rdam_agg_full_{c}_{s}_{t}") for t in range(TILES_PER_HEAD)]
                for s in range(2)]
    core_empty = [Lock(core_tiles[t], init=1, name=f"rdam_packed_empty_{c}_{t}") for t in range(TILES_PER_HEAD)]
    core_full = [Lock(core_tiles[t], init=0, name=f"rdam_packed_full_{c}_{t}") for t in range(TILES_PER_HEAD)]
    flows_c2m = [Flow(core_tiles[t], memtile, name=f"rdam_packed_c2m_{c}_{t}") for t in range(TILES_PER_HEAD)]
    flow_state_m2s = Flow(memtile, shim_tile, name=f"rdam_state_m2s_{c}")
    flow_out_m2s = Flow(memtile, shim_tile, name=f"rdam_out_m2s_{c}")
    barriers = [WorkerRuntimeBarrier() for _ in range(TILES_PER_HEAD)]

    def core_fn(s_in_handle, head_in_handle, packed_buf, kernel, tile_idx, core_empty_lock, core_full_lock, rtp,
                barrier):
        barrier.wait_for_value(1)
        tokens = rtp[0]  # read once per call, right after the barrier
        barrier.release_with_value(1)
        for _ in range_(H):
            core_empty_lock.acquire(1)
            elem_s_in = s_in_handle.acquire(1)
            for t in range_(tokens):
                elem_head = head_in_handle.acquire(1)
                kernel(elem_s_in, packed_buf, elem_head, tile_idx, t, tokens)
                head_in_handle.release(1)
            s_in_handle.release(1)
            core_full_lock.release(1)

    workers = [
        Worker(
            core_fn,
            [of_s_in_subs[t].cons(), of_head_in.cons(tile=core_tiles[t]), packed_core_buffers[t], rda_kernel, t,
             core_empty[t], core_full[t], token_rtps[t], barriers[t]],
            tile=core_tiles[t],
            stack_size=12288,
        )
        for t in range(TILES_PER_HEAD)
    ]

    return RDAMultiChannel(
        col=c,
        num_heads=H,
        memtile=memtile,
        core_tiles=core_tiles,
        s_l3_ty=np.ndarray[(H, V_DIM, K_DIM), f32],
        s_in_prod=s_in_prod,
        head_in_prod=head_in_prod,
        flows_c2m=flows_c2m,
        flow_state_m2s=flow_state_m2s,
        flow_out_m2s=flow_out_m2s,
        agg_mem_buffers=agg_mem_buffers,
        packed_core_buffers=packed_core_buffers,
        token_rtps=token_rtps,
        agg_empty=agg_empty,
        agg_full=agg_full,
        core_empty=core_empty,
        core_full=core_full,
        barriers=barriers,
        workers=workers,
    )


def rda_multi_sequence_body(
    channel: RDAMultiChannel,
    s_in_arg,
    s_out_arg,
    head_in_arg,
    out_arg,
    s_in_prod_seq,
    head_in_prod_seq,
    tokens: int,
):
    """Runtime sequence of one ``tokens``-token call: T to every core, barrier, fills, then the state and out drains."""
    check_rda_tokens(tokens)
    H = channel.num_heads
    P = MULTI_PACKED_ELEMS
    memtile = channel.memtile
    tg = TaskGroup()

    for rtp in channel.token_rtps:
        rtp[0] = tokens
    for barrier in channel.barriers:
        barrier.set(1)

    s_in_prod_seq.fill(s_in_arg, wait=False, group=tg)
    head_in_prod_seq.fill(head_in_arg, wait=False, group=tg)

    core_to_mem_tasks = []
    for t in range(TILES_PER_HEAD):
        core_task = channel.flows_c2m[t].endpoint(channel.core_tiles[t]).task(
            Bd(
                channel.packed_core_buffers[t],
                tap=TensorAccessPattern.full((P,)),
                acquires=[Acquire(channel.core_full[t], value=1)],
                releases=[Release(channel.core_empty[t], value=1)],
            ),
            runs=H,
            wait=False,
        )
        mem_tap = TensorAccessPattern((TILES_PER_HEAD, P), offset=t * P, sizes=(1, P), strides=(0, 1))
        mem_task = channel.flows_c2m[t].endpoint(memtile).task(
            *[
                Bd(
                    channel.agg_mem_buffers[s],
                    tap=mem_tap,
                    acquires=[Acquire(channel.agg_empty[s][t], value=2)],
                    releases=[Release(channel.agg_full[s][t], value=2)],
                )
                for s in range(2)
            ],
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
                    channel.agg_mem_buffers[s],
                    tap=TensorAccessPattern((TILES_PER_HEAD, P), offset=t * P, sizes=(1, TILE_V * K_DIM),
                                            strides=(0, 1)),
                    acquires=[Acquire(channel.agg_full[s][t], value=1)],
                    releases=[Release(channel.agg_empty[s][t], value=1)],
                )
            )
            out_bds.append(
                Bd(
                    channel.agg_mem_buffers[s],
                    tap=TensorAccessPattern((TILES_PER_HEAD, P), offset=t * P + TILE_V * K_DIM,
                                            sizes=(1, TILE_V * tokens), strides=(0, 1)),
                    acquires=[Acquire(channel.agg_full[s][t], value=1)],
                    releases=[Release(channel.agg_empty[s][t], value=1)],
                )
            )
    state_drain_task = channel.flow_state_m2s.endpoint(memtile).task(*state_bds, runs=H // 2, wait=False)
    out_drain_task = channel.flow_out_m2s.endpoint(memtile).task(*out_bds, runs=H // 2, wait=False)
    state_drain_task.start()
    out_drain_task.start()

    channel.flow_state_m2s.drain(s_out_arg, wait=True, group=tg)
    channel.flow_out_m2s.drain(out_arg, wait=True, group=tg)
    tg.finish()

    for core_task, mem_task in core_to_mem_tasks:
        core_task.free()
        mem_task.free()
    state_drain_task.free()
    out_drain_task.free()
