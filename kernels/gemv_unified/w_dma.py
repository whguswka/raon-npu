# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Runtime-configured ping/pong DMA helpers for unified GEMV weights.

The shared xclbin owns fixed-size L1 buffers, while each instruction stream
configures the mem/core receive BDs with the active format's true transfer
length.  Core tiles use ``configure_ping_pong_tasks`` per round (a task starts at
ping; even pairs use one two-BD task and an odd final block one one-BD task).
Memory tiles use ``configure_chunk_tasks`` with ``w_plan.MemChunk``.  The
default plan restarts at ping each round; optional fix4 mode uses the cumulative
block number modulo two so one two-BD repeat task can cover many rounds.  A
dispatch ends with no resident task waiting for a block that will never arrive.

Completion tokens (fix3): these memory-tile / core-tile tasks do NOT issue a
token (``issue_token=False``).  ``aie-generate-column-control-overlay`` (option
``route-shim-to-tct`` = ``shim-only``, ``AIEPasses.td:653``) adds a TileControl ->
shim South packet flow, going DOWN the column, for every non-shim tile that issues
a token from a runtime-sequence task.  Per column that was 4 core rows + the
memory tile = 5 downward paths, and with the 4 y-join streams the memory tile's
North source port (limit 4, ``AIETargetModel.cpp:1019-1110``) was oversubscribed
(4 + 1 = 5 > 4).  Tokens now come only from shim tasks (x, W input, Y drain).

Release rule: ``aiex.dma_free_task`` does not wait for the task
(``AIEAssignRuntimeSequenceBDIDs.cpp:236-315``: "only safe when some other
synchronization already guarantees that task has finished").  The caller frees
these tasks only after that round's Y drain (shim) has been awaited.  That await
proves every W task of the round is done:
  * a core emits its y only after receiving its last W block, so Y drain
    completion implies every core-side W receive finished;
  * the memory tile sends a block to a core before the core can receive it, so
    every memory-tile send finished before the core receives (send <= recv);
  * the memory tile receives a block from the shim before it can send it, so
    every memory-tile receive finished before the send (recv <= send).
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Acquire, Bd, Release
from aie.iron.runtime._context import active_sequence

from .dma_layout import DmaTapSpec
from .w_plan import CoreCallChunk


class _DimensionPreservingTap(TensorAccessPattern):
    """A TAP whose hardware-safe dimension split must survive ``Bd._emit``."""

    def coalesce(self):
        return self


def tap_from_spec(spec: DmaTapSpec) -> TensorAccessPattern:
    """Build the exact TAP described by the host-testable shared specification."""
    tap_type = _DimensionPreservingTap if spec.preserve_dimensions else TensorAccessPattern
    return tap_type(spec.tensor_dims, spec.offset, spec.sizes, spec.strides)


class RuntimeDmaTasks:
    """Configured tasks on one endpoint, owned by one runtime-sequence round."""

    def __init__(self, tasks: Iterable[Any], *, issue_token: bool = False) -> None:
        self.tasks = tuple(tasks)
        self.issue_token = issue_token

    def start(self) -> None:
        """Queue the pair task before the optional odd-tail task."""
        for task in self.tasks:
            task.start()

    def finish(self) -> None:
        """Return every runtime BD (awaiting first only if tokens are issued).

        Without a token the task cannot be awaited; the caller must already have
        awaited the round's Y drain, which guarantees these tasks finished.
        """
        for task in self.tasks:
            if self.issue_token:
                task.await_()
            task.free()


def declare_runtime_buffers(buffers: Iterable[Any]) -> None:
    """Declare runtime-only buffers even when a shape never configures their slot.

    IRON normally resolves such buffers lazily when a runtime DMA task first
    references them.  The shared-xclbin design needs the complete fixed set in
    every generated device body, including instruction streams that only use
    slot zero.
    """
    active = active_sequence()
    for buffer in buffers:
        active.resolve_in_device(buffer)


def configure_ping_pong_tasks(
    endpoint: Any,
    buffers: Sequence[Any],
    taps: Sequence[Any],
    empty_locks: Sequence[Any],
    full_locks: Sequence[Any],
    *,
    blocks: int,
    acquire_value: int = 1,
    release_value: int = 1,
    issue_token: bool = False,
) -> RuntimeDmaTasks:
    """Configure an exact-length, finite ping/pong transfer on ``endpoint``.

    Separate locks per slot prevent one channel from spending another slot's
    readiness count.  The returned tasks issue NO completion token by default
    (see module docstring); callers must call ``finish`` only after the round's Y
    drain await, which proves the task and its consumers have finished.
    """
    if blocks <= 0:
        raise ValueError(f"blocks must be positive, got {blocks}")
    if not all(len(items) == 2 for items in (buffers, taps, empty_locks, full_locks)):
        raise ValueError("ping/pong buffers, taps, and lock lists must each have length 2")

    def bd(slot: int) -> Bd:
        return Bd(
            buffers[slot],
            tap=taps[slot],
            acquires=[Acquire(empty_locks[slot], value=acquire_value)],
            releases=[Release(full_locks[slot], value=release_value)],
        )

    tasks = []
    pairs = blocks // 2
    if pairs:
        tasks.append(endpoint.task(bd(0), bd(1), runs=pairs, wait=issue_token))
    if blocks & 1:
        tasks.append(endpoint.task(bd(0), runs=1, wait=issue_token))
    return RuntimeDmaTasks(tasks, issue_token=issue_token)


def configure_chunk_tasks(
    endpoint: Any,
    buffers: Sequence[Any],
    taps: Sequence[Any],
    empty_locks: Sequence[Any],
    full_locks: Sequence[Any],
    *,
    chunk: Any,
    acquire_value: int = 1,
    release_value: int = 1,
    issue_token: bool = False,
) -> RuntimeDmaTasks:
    """Configure one memory-tile task described by ``w_plan.MemChunk``.

    A pair chunk is one two-BD repeat task starting at ``chunk.first_slot``; a
    tail chunk is one BD.  The planner selects either per-round ping restart or
    cross-round continuous ping/pong semantics.
    """
    if not all(len(items) == 2 for items in (buffers, taps, empty_locks, full_locks)):
        raise ValueError("ping/pong buffers, taps, and lock lists must each have length 2")

    def bd(slot: int) -> Bd:
        return Bd(
            buffers[slot],
            tap=taps[slot],
            acquires=[Acquire(empty_locks[slot], value=acquire_value)],
            releases=[Release(full_locks[slot], value=release_value)],
        )

    first = chunk.first_slot
    if chunk.is_tail:
        task = endpoint.task(bd(first), runs=1, wait=issue_token)
    else:
        task = endpoint.task(bd(first), bd(1 - first), runs=chunk.pairs, wait=issue_token)
    return RuntimeDmaTasks([task], issue_token=issue_token)


def configure_core_call_tasks(
    endpoint: Any,
    buffers: Sequence[Any],
    taps: Sequence[Any],
    empty_locks: Sequence[Any],
    full_locks: Sequence[Any],
    *,
    chunk: CoreCallChunk,
    acquire_value: int = 1,
    release_value: int = 1,
    issue_token: bool = False,
) -> RuntimeDmaTasks:
    """Configure one repeated, position-specific core BD chain."""
    if not all(len(items) == 2 for items in (buffers, taps, empty_locks, full_locks)):
        raise ValueError("ping/pong buffers, taps, and lock lists must each have length 2")

    def bd(slot: int) -> Bd:
        return Bd(
            buffers[slot],
            tap=taps[slot],
            acquires=[Acquire(empty_locks[slot], value=acquire_value)],
            releases=[Release(full_locks[slot], value=release_value)],
        )

    task = endpoint.task(
        *(bd(slot) for slot in chunk.slots),
        runs=chunk.runs,
        wait=issue_token,
    )
    return RuntimeDmaTasks([task], issue_token=issue_token)


def start_all(task_sets: Iterable[RuntimeDmaTasks]) -> None:
    """Start several endpoints after every endpoint has been configured."""
    for task_set in task_sets:
        task_set.start()


def finish_all(task_sets: Iterable[RuntimeDmaTasks]) -> None:
    """Await and free several endpoints in deterministic order."""
    for task_set in task_sets:
        task_set.finish()
