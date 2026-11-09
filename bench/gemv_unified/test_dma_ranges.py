# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only bounds checks for every TAP emitted by the unified design."""

from __future__ import annotations

from math import prod
from typing import Any

import numpy as np

from bench.gemv_unified.check_device_invariance import select_cases
import kernels.gemv_unified.design as unified_design
from kernels.gemv_unified.shapes import ALL_SHAPES, COLS, MAX_K, x_slot_offset


MODES = (
    ("default", {}),
    ("continuous_mem_w", {"continuous_mem_w": True}),
    ("diagnostic_small_x", {"diagnostic_small_x": True}),
    ("diagnostic_receive_only", {"diagnostic_receive_only": True}),
    ("diagnostic_static_core", {"diagnostic_static_core": True}),
)


def _install_fake_iron(monkeypatch, taps: list[Any]) -> None:
    """Replace IRON with small recorders while retaining the runtime sequence."""

    class FakeTile:
        def __init__(self, col: int, row: int) -> None:
            self.col = col
            self.row = row

    class FakeTask:
        def await_(self) -> None:
            pass

        def free(self) -> None:
            pass

    class FakeEndpoint:
        def __init__(self, name: str, tile: Any = None, depth: int | None = None) -> None:
            self.name = name
            self.tile = tile
            self.depth = depth

        def join(self, offsets, tile=None, depths=None, obj_types=None, names=None, **kwargs):
            depths = depths or [self.depth or 2] * len(offsets)
            obj_types = obj_types or [None] * len(offsets)
            names = names or [f"{self.name}_join{i}" for i in range(len(offsets))]
            return [
                FakeObjectFifo(obj_type, depth, name)
                for obj_type, depth, name in zip(obj_types, depths, names)
            ]

        def fill(self, buf, tap, wait=False, managed=True, group=None):
            return FakeTask()

        def drain(self, buf, tap, wait=False, group=None) -> None:
            pass

    class FakeObjectFifo:
        def __init__(self, obj_type, depth=2, name="") -> None:
            self.obj_type = obj_type
            self.depth = depth
            self.name = name

        def prod(self, tile=None, depth=None):
            return FakeEndpoint(self.name, tile, depth)

        def cons(self, tile=None, depth=None):
            return FakeEndpoint(self.name, tile, depth)

    class FakeBuffer:
        def __init__(self, type, name, **kwargs) -> None:
            self.type = type
            self.name = name

        def __setitem__(self, index, value) -> None:
            pass

    class FakeFlow:
        def __init__(self, src, dst, name=None, **kwargs) -> None:
            self.name = name or "flow"

        def endpoint(self, tile):
            return FakeEndpoint(self.name, tile)

        def fill(self, buf, tap, wait=False, group=None) -> None:
            pass

    class FakeLock:
        def __init__(self, tile, **kwargs) -> None:
            self.tile = tile

    class FakeBarrier:
        def set(self, value) -> None:
            pass

    class FakeWorker:
        def __init__(self, core_fn, fn_args, **kwargs) -> None:
            pass

    class FakeTaskGroup:
        def finish(self) -> None:
            pass

    class FakeTAP:
        def __init__(self, tensor_dims, offset, sizes, strides) -> None:
            self.tensor_dims = tuple(tensor_dims)
            self.offset = int(offset)
            self.sizes = tuple(sizes)
            self.strides = tuple(strides)
            taps.append(self)

    class FakeDmaTaskSet:
        def start(self) -> None:
            pass

        def finish(self) -> None:
            pass

    class FakeRuntime:
        def __init__(self, seq_fn=None, fn_args=None, **kwargs) -> None:
            self.seq_fn = seq_fn
            self.fn_args = fn_args or []

        def add_flow(self, flow) -> None:
            pass

        def add_lock(self, lock) -> None:
            pass

        def run(self) -> None:
            self.seq_fn(*self.fn_args)

    class FakeProgram:
        def __init__(self, dev, runtime, workers=None) -> None:
            self.runtime = runtime

        def resolve_program(self, **kwargs):
            self.runtime.run()
            return self

    def fake_configure(*args, **kwargs):
        return FakeDmaTaskSet()

    def fake_start_all(task_sets) -> None:
        for task_set in task_sets:
            task_set.start()

    def fake_finish_all(task_sets) -> None:
        for task_set in task_sets:
            task_set.finish()

    replacements = {
        "NPU2": None,
        "bfloat16": np.uint16,
        "Tile": FakeTile,
        "ObjectFifo": FakeObjectFifo,
        "Buffer": FakeBuffer,
        "Flow": FakeFlow,
        "Lock": FakeLock,
        "WorkerRuntimeBarrier": FakeBarrier,
        "Worker": FakeWorker,
        "TaskGroup": FakeTaskGroup,
        "TensorAccessPattern": FakeTAP,
        "Runtime": FakeRuntime,
        "Program": FakeProgram,
        # Force _tap_from_spec through FakeTAP even when the optional real
        # IRON package is available in the second host-side test invocation.
        "tap_from_spec": None,
        "configure_ping_pong_tasks": fake_configure,
        "configure_chunk_tasks": fake_configure,
        "start_all": fake_start_all,
        "finish_all": fake_finish_all,
        "declare_runtime_buffers": lambda buffers: None,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(unified_design, name, value, raising=False)


def _assert_tap_inside_tensor(tap, *, context: str) -> None:
    assert tap.offset >= 0, f"{context}: negative offset in {tap.__dict__}"
    assert len(tap.sizes) == len(tap.strides), context
    assert tap.sizes and all(size > 0 for size in tap.sizes), context
    tensor_elems = prod(tap.tensor_dims)
    largest_address = tap.offset + sum(
        (size - 1) * stride for size, stride in zip(tap.sizes, tap.strides)
    )
    assert largest_address < tensor_elems, (
        f"{context}: largest address {largest_address} is outside "
        f"{tap.tensor_dims} ({tensor_elems} elements); tap={tap.__dict__}"
    )


def test_every_shape_and_mode_emits_only_in_bounds_dma_taps(monkeypatch) -> None:
    """Cover 13 product + 6 boundary shapes across all five design modes."""
    taps: list[Any] = []
    _install_fake_iron(monkeypatch, taps)
    cases = select_cases(None, None)
    assert len(ALL_SHAPES) == 13
    assert len(cases) == 19

    for case in cases:
        shape = case.shape
        for mode_name, mode_kwargs in MODES:
            start = len(taps)
            unified_design.gemv_unified(
                dev=None,
                K=shape.K,
                M=shape.M,
                cols=COLS,
                format=shape.format,
                active_cols=shape.active_cols,
                x_slots=shape.x_slots,
                chunks_per_col=shape.chunks_per_col,
                gemv_kernel=["zero", "block", "epilogue"],
                **mode_kwargs,
            )
            emitted = taps[start:]
            # X(8), shim W/Y(2 per chunk), mem in/out(1+4), core W in(1).
            assert len(emitted) == COLS + 2 * shape.chunks + 6
            context = f"{case.label} mode={mode_name}"
            for tap in emitted:
                _assert_tap_inside_tensor(tap, context=context)

            x_taps = emitted[:COLS]
            slot_elems = shape.K if mode_name == "diagnostic_small_x" else MAX_K
            assert [tap.offset for tap in x_taps] == [
                x_slot_offset(shape, col, slot_elems) for col in range(COLS)
            ]

    # Product-path snapshot: fix13 must not change the established B2 layout.
    b2 = next(shape for shape in ALL_SHAPES if shape.name == "moe_b2_9")
    assert [x_slot_offset(b2, col, MAX_K) for col in range(COLS)] == [
        0, 12032, 24064, 36096, 48128, 60160, 72192, 0
    ]
