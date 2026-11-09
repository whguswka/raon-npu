# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-testable runtime-sequence plan for the unified GEMV W path.

fix3 configured, per round and per channel, one pair task plus (odd K) one tail
task: 3 runtime BDs, and with the depth-2 round window 6 BDs per channel.  The
memory tile has five W channels (S2MM w_in + four MM2S w_out) and its BD pool is
split by channel parity (``AIETargetModel.h:489`` ``getNumBDsForChannel`` /
``:922`` ``isBdChannelAccessible``: even channel -> BD id < 24, odd -> >= 24), so
odd block counts (q_b k=3, B2-9 k=5, dense_down k=47) overflowed 24.

The optional fix4 mode makes **memory-tile** W tasks run across rounds.  The
ping/pong slot is the cumulative block number modulo two, so one dispatch needs
one two-BD repeat task per ``MAX_REPEAT`` pairs plus (odd total) a one-BD tail.
The core W receive is *not* continued: the core tile keeps its per-round
ping-first protocol (its own buffers and locks), so no core code or
program-memory change is needed.  Data order is all that couples the two sides.

``drive_rounds`` is the single round/window driver used both by ``design.py`` and
by the BD-budget simulation, so the host test counts what the design emits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kernels.gemv_unified.shapes import COLS, ROWS, W_BLOCK_BYTES, UnifiedShape

# ``AIETargetModel.h:853`` getMaxRepeatCount (host mirror: test_dma_bd_limits).
MAX_REPEAT = 255
DEPTH = 2  # rounds in flight in the runtime sequence
W_MEM_BUFFER_ELEMS = ROWS * W_BLOCK_BYTES // 2


@dataclass(frozen=True)
class MemtileResource:
    """One fixed device-side memory-tile resource used by runtime W tasks."""

    name: str
    kind: str
    elements: int | None = None
    init: int | None = None


@dataclass(frozen=True)
class MemtileResourcePlan:
    """Fixed declarations and the subset used by one shape's command plan."""

    declared: tuple[MemtileResource, ...]
    used_names: frozenset[str]


def declared_memtile_resources(cols: int = COLS) -> tuple[MemtileResource, ...]:
    """Return the shape-independent memtile buffer/lock declaration set."""
    resources = []
    for col in range(cols):
        for slot in range(DEPTH):
            resources.extend(
                (
                    MemtileResource(
                        f"w_mem_{col}_{slot}", "buffer", elements=W_MEM_BUFFER_ELEMS
                    ),
                    MemtileResource(
                        f"w_mem_empty_{col}_{slot}", "lock", init=ROWS
                    ),
                    MemtileResource(f"w_mem_full_{col}_{slot}", "lock", init=0),
                )
            )
    return tuple(resources)


def used_memtile_resource_names(
    shape: UnifiedShape,
    cols: int = COLS,
    *,
    continuous: bool = False,
) -> frozenset[str]:
    """Return every fixed memtile resource referenced by a shape's DMA plan."""
    names = set()
    for col, chunks in mem_chunk_plan(shape, cols, continuous=continuous).items():
        for chunk in chunks:
            slots = (chunk.first_slot,) if chunk.is_tail else (0, 1)
            for slot in slots:
                names.update(
                    (
                        f"w_mem_{col}_{slot}",
                        f"w_mem_empty_{col}_{slot}",
                        f"w_mem_full_{col}_{slot}",
                    )
                )
    return frozenset(names)


def memtile_resource_plan(
    shape: UnifiedShape,
    cols: int = COLS,
    *,
    continuous: bool = False,
) -> MemtileResourcePlan:
    """Pair the invariant declarations with one shape's referenced subset."""
    return MemtileResourcePlan(
        declared=declared_memtile_resources(cols),
        used_names=used_memtile_resource_names(shape, cols, continuous=continuous),
    )


@dataclass(frozen=True)
class MemChunk:
    """One memory-tile W task: ``pairs`` x (ping, pong) or a single tail BD."""

    first_block: int  # cumulative block number of the first block
    pairs: int  # 0 for a tail task
    k_blocks: int  # blocks per round (to map blocks to rounds)
    restart_each_round: bool = False

    @property
    def is_tail(self) -> bool:
        return self.pairs == 0

    @property
    def blocks(self) -> int:
        return 1 if self.is_tail else 2 * self.pairs

    @property
    def bds(self) -> int:
        return 1 if self.is_tail else 2

    @property
    def first_slot(self) -> int:
        block = (
            self.first_block % self.k_blocks
            if self.restart_each_round
            else self.first_block
        )
        return block % 2

    @property
    def first_round(self) -> int:
        return self.first_block // self.k_blocks

    @property
    def last_round(self) -> int:
        return (self.first_block + self.blocks - 1) // self.k_blocks


def continuous_mem_chunks(
    k_blocks: int, rounds: int, max_repeat: int = MAX_REPEAT
) -> tuple[MemChunk, ...]:
    """Memory-tile tasks covering ``rounds * k_blocks`` blocks continuously."""
    total = k_blocks * rounds
    chunks: list[MemChunk] = []
    pairs_left = total // 2
    block = 0
    while pairs_left:
        pairs = min(pairs_left, max_repeat)
        chunks.append(MemChunk(block, pairs, k_blocks))
        block += 2 * pairs
        pairs_left -= pairs
    if total & 1:
        chunks.append(MemChunk(block, 0, k_blocks))
    return tuple(chunks)


def per_round_mem_chunks(k_blocks: int, rounds: int) -> tuple[MemChunk, ...]:
    """fix3 layout (kept for the before/after budget test): restart at ping."""
    chunks: list[MemChunk] = []
    for rnd in range(rounds):
        base = rnd * k_blocks
        if k_blocks // 2:
            chunks.append(
                MemChunk(base, k_blocks // 2, k_blocks, restart_each_round=True)
            )
        if k_blocks & 1:
            chunks.append(
                MemChunk(base + k_blocks - 1, 0, k_blocks, restart_each_round=True)
            )
    return tuple(chunks)


def mem_chunk_plan(
    shape: UnifiedShape,
    cols: int = COLS,
    *,
    continuous: bool = False,
) -> dict[int, tuple[MemChunk, ...]]:
    """Per-column memory-tile W tasks (columns have different round counts)."""
    make = (
        continuous_mem_chunks
        if continuous
        else lambda k, r: per_round_mem_chunks(k, r)
    )
    active = shape.resolve_active_cols(cols)
    return {
        col: make(shape.k_blocks, rounds)
        for col, rounds in enumerate(shape.rounds_by_col(cols))
        if rounds > 0 and col < active
    }


def round_window_depth(k_blocks: int, *, continuous: bool) -> int:
    """Return a BD-safe number of in-flight rounds for the selected W plan.

    A per-round odd-block plan needs three BDs per W channel (ping/pong pair
    plus ping tail).  Keeping only one such round live avoids the memory-tile
    parity-pool overflow that motivated fix4, without changing the transfer
    protocol for even-block shapes.  Continuous mode retains the depth-2
    window used by fix4.
    """
    return 1 if not continuous and k_blocks % 2 else DEPTH


def core_round_bds(k_blocks: int) -> int:
    """Runtime BDs of one core W task set: pair task (2) + odd tail (1)."""
    return (2 if k_blocks // 2 else 0) + (1 if k_blocks & 1 else 0)


def drive_rounds(
    max_rounds: int,
    on_round: Callable[[int], None],
    on_retire: Callable[[int], None],
    depth: int = DEPTH,
) -> None:
    """Runtime-sequence order: retire the oldest round, then issue the next.

    ``on_retire(r)`` awaits round r's Y drain and then frees everything whose
    last use is round r.  Used verbatim by ``design.py``.
    """
    window: list[int] = []
    for rnd in range(max_rounds):
        if len(window) == depth:
            on_retire(window.pop(0))
        on_round(rnd)
        window.append(rnd)
    for rnd in window:
        on_retire(rnd)


# ---------------------------------------------------------------------------
# BD budget (host mirror of AIEAssignRuntimeSequenceBDIDs live-range counting)
# ---------------------------------------------------------------------------

MEM_TILE_BDS_PER_PARITY = 24  # AIETargetModel.h:489/922 (even ch <24, odd >=24)
CORE_TILE_BDS = 16
SHIM_TILE_BDS = 16
ROWS_PER_COL = 4


@dataclass(frozen=True)
class BdUsage:
    """Peak simultaneously-live runtime BDs per channel/tile for one shape."""

    mem_in_peak: int  # memory-tile S2MM w_in channel
    mem_out_peak: int  # each memory-tile MM2S w_out channel (four of them)
    core_w_peak: int  # core-tile W S2MM channel
    shim_peak: int  # shim tile, all runtime tasks (x + W fill + Y drain)


@dataclass(frozen=True)
class CoreCallChunk:
    """One core task: a BD chain repeated ``runs`` times."""

    slots: tuple[int, ...]
    runs: int

    @property
    def bds(self) -> int:
        return len(self.slots)


def per_call_core_chunks(k_blocks: int, rounds: int) -> tuple[CoreCallChunk, ...]:
    """Return a ping-first core chain covering every round of one call.

    Even block counts repeat a two-BD ping/pong chain.  Odd block counts use
    one position-specific BD per block, so repeating the whole chain restarts
    every round at ping.  A chain must fit the core tile's 16-BD pool.
    """
    if k_blocks <= 0 or rounds <= 0:
        raise ValueError("k_blocks and rounds must be positive")
    if k_blocks % 2 == 0:
        slots = (0, 1)
        repetitions = rounds * (k_blocks // 2)
    else:
        if k_blocks > CORE_TILE_BDS:
            raise ValueError(
                f"odd per-call core chain needs {k_blocks} BDs, limit is {CORE_TILE_BDS}"
            )
        slots = tuple(block % 2 for block in range(k_blocks))
        repetitions = rounds
    chunks = []
    while repetitions:
        runs = min(repetitions, MAX_REPEAT)
        chunks.append(CoreCallChunk(slots, runs))
        repetitions -= runs
    return tuple(chunks)


def simulate_per_call_bd_usage(
    shape: UnifiedShape,
    cols: int = COLS,
    *,
    shim_w_bds_by_col: dict[int, int] | None = None,
    shim_y_bds_by_col: dict[int, int] | None = None,
) -> BdUsage:
    """Count simultaneously-live BDs emitted by the per-call plan.

    Odd chains restart at ping by repeating the complete position-specific BD
    chain.  Odd chains longer than the core BD pool are not a legal per-call
    plan and must remain per-round at the engine selection layer.
    """
    mem = mem_chunk_plan(shape, cols, continuous=True)
    rounds = shape.rounds_by_col(cols)
    mem_peak = max((sum(chunk.bds for chunk in chunks) for chunks in mem.values()), default=0)
    core_peak = max(
        (sum(chunk.bds for chunk in per_call_core_chunks(shape.k_blocks, count))
         for count in rounds if count),
        default=0,
    )
    shim_peak = max(
        (1
         + (shim_w_bds_by_col or {}).get(col, 1)
         + (shim_y_bds_by_col or {}).get(col, 1)
         for col, count in enumerate(rounds) if count),
        default=1,
    )
    return BdUsage(mem_peak, mem_peak, core_peak, shim_peak)


def simulate_bd_usage(
    shape: UnifiedShape,
    cols: int = COLS,
    *,
    continuous: bool = False,
    depth: int | None = None,
) -> BdUsage:
    """Replay ``design.py``'s runtime-sequence order and track live runtime BDs."""
    plan = mem_chunk_plan(shape, cols, continuous=continuous)
    if depth is None:
        depth = round_window_depth(shape.k_blocks, continuous=continuous)
    col_rounds = shape.rounds_by_col(cols)
    live_mem = {col: 0 for col in plan}
    live_core = {col: 0 for col in plan}
    # x fill is a shim task (waited/freed at the end); count it for every column.
    live_shim = {col: 1 for col in range(cols)}
    peaks = {"mem": 0, "core": 0, "shim": 0}
    core_bds = core_round_bds(shape.k_blocks)

    def bump() -> None:
        peaks["mem"] = max(peaks["mem"], *live_mem.values())
        peaks["core"] = max(peaks["core"], *live_core.values())
        peaks["shim"] = max(peaks["shim"], *live_shim.values())

    def on_round(rnd: int) -> None:
        for col, chunks in plan.items():
            if rnd >= col_rounds[col]:
                continue
            for chunk in chunks:
                if chunk.first_round == rnd:
                    live_mem[col] += chunk.bds
            live_core[col] += core_bds
            live_shim[col] += 2  # Y drain + W fill
        bump()

    def on_retire(rnd: int) -> None:
        for col, chunks in plan.items():
            if rnd >= col_rounds[col]:
                continue
            for chunk in chunks:
                if chunk.last_round == rnd:
                    live_mem[col] -= chunk.bds
            live_core[col] -= core_bds
            live_shim[col] -= 2

    drive_rounds(shape.max_rounds, on_round, on_retire, depth)
    assert all(v == 0 for v in live_mem.values()), live_mem
    return BdUsage(peaks["mem"], peaks["mem"], peaks["core"], peaks["shim"])
