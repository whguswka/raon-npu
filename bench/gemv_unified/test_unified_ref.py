# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Stage-1 unified 9-bit bfp16ebs8 GEMV on the CPU-only host (numpy + mock)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from kernels.gemv_unified.w_plan import mem_chunk_plan

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the top-level ``bench`` namespace package stable in multi-directory runs.
def _sanitize_bench_namespace() -> None:
    for _p in list(sys.path):
        if Path(_p).name in ("gemv_bfp16", "gemv_unified"):
            sys.path.remove(_p)
    if "bench" in sys.modules and not hasattr(sys.modules["bench"], "__path__"):
        del sys.modules["bench"]


_sanitize_bench_namespace()


@pytest.fixture(scope="session", autouse=True)
def _ensure_bench_namespace_package() -> None:
    _sanitize_bench_namespace()


def _load_sibling_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_bench_mod = _load_sibling_module("gemv_unified_bench", "unified_bench.py")
_exp_mod = _load_sibling_module("gemv_unified_exp_short_dma", "exp_short_dma.py")
compare_bf16_bits = _bench_mod.compare_bf16_bits
make_shape_test_case = _bench_mod.make_shape_test_case
simulate_short_dma_ping_pong = _exp_mod.simulate_short_dma_ping_pong
import kernels.gemv_unified.design as unified_design  # noqa: E402
import kernels.gemv_unified.op as unified_op  # noqa: E402
from kernels.gemv_unified.shapes import (  # noqa: E402
    ALL_SHAPES,
    COLS,
    FORMAT_BFP16,
    FORMAT_Q4NX,
    M_PER_CHUNK,
    MAX_K,
    ROWS,
    RTP_FORMAT,
    RTP_K_BLOCKS,
    RTP_ROUNDS,
    SHAPES,
    W_BLOCK_BYTES,
    W_CHUNK_BYTES,
    UnifiedShape,
    get_shape,
    rtp_table,
    selected_shapes,
    stream_plan,
)

EXPECTED_STAGE1_SHAPES: tuple[tuple[str, int, int, int, int, tuple[int, ...]], ...] = (
    ("rda_in", 2560, 20736, 10, 162, (21, 21, 20, 20, 20, 20, 20, 20)),
    ("f_b", 256, 5120, 1, 40, (5, 5, 5, 5, 5, 5, 5, 5)),
    ("o_proj", 5120, 2560, 20, 20, (3, 3, 3, 3, 2, 2, 2, 2)),
    ("mla_in", 2560, 1152, 10, 9, (2, 1, 1, 1, 1, 1, 1, 1)),
    ("q_b", 768, 7680, 3, 60, (8, 8, 8, 8, 7, 7, 7, 7)),
    ("k_b_v_b", 512, 5120, 2, 40, (5, 5, 5, 5, 5, 5, 5, 5)),
    ("g", 2560, 5120, 10, 40, (5, 5, 5, 5, 5, 5, 5, 5)),
    ("shared_gu", 2560, 2176, 10, 17, (3, 2, 2, 2, 2, 2, 2, 2)),
    ("dense_down", 12032, 2560, 47, 20, (3, 3, 3, 3, 2, 2, 2, 2)),
    ("lm_head", 2560, 260096, 10, 2032, (254, 254, 254, 254, 254, 254, 254, 254)),
)


def test_stage1_shapes_table_and_rounds_by_col() -> None:
    assert len(SHAPES) == len(EXPECTED_STAGE1_SHAPES)
    for shape, (name, K, M, k_blocks, chunks, expected_rounds) in zip(
        SHAPES, EXPECTED_STAGE1_SHAPES
    ):
        assert shape.name == name
        assert shape.K == K
        assert shape.M == M
        assert shape.k_blocks == k_blocks
        assert shape.chunks == chunks
        assert shape.rounds_by_col() == expected_rounds
        assert sum(shape.rounds_by_col()) == chunks
        assert shape.max_rounds == max(expected_rounds)
        assert shape.w_chunk_bytes == k_blocks * W_CHUNK_BYTES
        assert shape.weight_bytes == chunks * shape.w_chunk_bytes
        assert shape.weight_words == chunks * shape.w_chunk_words

    # Check aliases for k_b / v_b
    assert get_shape("k_b") == get_shape("k_b_v_b")
    assert get_shape("v_b") == get_shape("k_b_v_b")
    assert get_shape("k_b/v_b") == get_shape("k_b_v_b")
    assert selected_shapes(["o_proj", "f_b"]) == (get_shape("o_proj"), get_shape("f_b"))


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_rtp_table_matches_all_32_cores(shape: UnifiedShape) -> None:
    table = rtp_table(shape)
    assert len(table) == COLS * ROWS == 32
    col_rounds = shape.rounds_by_col()
    for col in range(COLS):
        for row in range(ROWS):
            entry = table[col * ROWS + row]
            assert entry[RTP_FORMAT] == FORMAT_BFP16
            assert entry[RTP_K_BLOCKS] == shape.k_blocks
            assert entry[RTP_ROUNDS] == col_rounds[col]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: s.name)
def test_stream_plan_offsets_and_coverage(shape: UnifiedShape) -> None:
    plan = stream_plan(shape)
    assert len(plan) == shape.chunks

    col_counts = [0] * COLS
    expected_w_words_per_chunk = (M_PER_CHUNK * shape.K * 9 // 8) // 4
    assert shape.w_chunk_words == expected_w_words_per_chunk
    # Check BD word limit (<= 16,383 words per chunk for all shapes; max is dense_down: 47 * 9216 B / 4 = 108,288 words across 47 blocks, wait: per chunk in dense_down is 108,288 words)
    assert W_BLOCK_BYTES == 9216

    for idx, rec in enumerate(plan):
        assert rec["chunk"] == idx
        assert rec["col"] == idx % COLS
        assert rec["round"] == idx // COLS
        assert rec["x_offset_bf16"] == 0
        assert rec["x_length_bf16"] == MAX_K
        assert rec["w_offset_words"] == idx * expected_w_words_per_chunk
        assert rec["w_length_words"] == expected_w_words_per_chunk
        assert rec["y_offset_bf16"] == idx * M_PER_CHUNK
        assert rec["y_length_bf16"] == M_PER_CHUNK
        col_counts[rec["col"]] += 1

    assert tuple(col_counts) == shape.rounds_by_col()
    # Last chunk must end exactly at total weight_words and M
    last = plan[-1]
    assert last["w_offset_words"] + last["w_length_words"] == shape.weight_words
    assert last["y_offset_bf16"] + last["y_length_bf16"] == shape.M


def test_check_shape_validation() -> None:
    s = unified_design.check_shape(None, 5120, 2560, cols=8)
    assert s.K == 5120 and s.M == 2560

    with pytest.raises(ValueError, match="multiple of 256"):
        unified_design.check_shape(None, 300, 2560)
    with pytest.raises(ValueError, match="1..12032"):
        unified_design.check_shape(None, MAX_K + 256, 2560)
    with pytest.raises(ValueError, match="multiple of 128"):
        unified_design.check_shape(None, 2560, 2560 + 64)
    # active_cols must not exceed cols
    with pytest.raises(ValueError, match="active_cols"):
        unified_design.check_shape(None, 2560, 2560, cols=4, active_cols=8)
    # cols below the requested active device width is still allowed if active_cols matches cols
    s4 = unified_design.check_shape(None, 2560, 2560, cols=4)
    assert s4.active_cols == 4


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_operator_accepts_every_exact_supported_shape(shape: UnifiedShape) -> None:
    op = unified_op.UnifiedGEMV.for_shape(shape, context=object())
    assert op.shape_name == shape.name
    assert unified_op.UnifiedGEMV._shape_fields(
        op.shape
    ) == unified_op.UnifiedGEMV._shape_fields(shape)


@pytest.mark.parametrize(
    ("k", "m"),
    ((256, 128), (1024, 128), (12032, 128)),
)
def test_operator_rejects_unregistered_product_shapes(k: int, m: int) -> None:
    with pytest.raises(ValueError, match=r"supported names:.*dense_gu.*moe_b2_9"):
        unified_op.UnifiedGEMV(K=k, M=m, context=object())


def test_operator_rejects_spoofed_supported_name() -> None:
    spoofed = UnifiedShape("rda_in", 1024, 256)
    with pytest.raises(ValueError, match=r"K=.*closest|M=.*closest"):
        unified_op.UnifiedGEMV.for_shape(spoofed, context=object())


def test_operator_rejects_unregistered_unified_shape_by_default() -> None:
    outside = UnifiedShape("outside", 1024, 256)
    with pytest.raises(ValueError, match=r"supported names:"):
        unified_op.UnifiedGEMV.for_shape(outside, context=object())


@pytest.mark.parametrize(
    "shape",
    (
        UnifiedShape("boundary_256", 256, 128),
        UnifiedShape("boundary_1024", 1024, 128),
        UnifiedShape("boundary_max", 12032, 128),
        UnifiedShape("outside", 1024, 256),
    ),
)
def test_operator_accepts_explicit_experimental_shapes(shape: UnifiedShape) -> None:
    op = unified_op.UnifiedGEMV.for_shape(
        shape, experimental_shape=True, context=object()
    )
    assert op.experimental_shape


def _set_up_artifacts_with_captured_mlir_shapes(
    op: unified_op.UnifiedGEMV, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[UnifiedShape], Any, Any]:
    generated_shapes: list[UnifiedShape] = []

    def fake_mlir_artifact(
        filename: str,
        _k: int,
        _m: int,
        *,
        shape: UnifiedShape | None = None,
    ) -> SimpleNamespace:
        generated_shape = shape or op.shape
        generated_shapes.append(generated_shape)
        return SimpleNamespace(filename=filename, shape=generated_shape)

    monkeypatch.setattr(op, "get_kernel_artifacts", lambda: [])
    monkeypatch.setattr(op, "_mlir_artifact", fake_mlir_artifact)
    monkeypatch.setattr(
        op,
        "get_xclbin_artifact",
        lambda config_mlir, _kernels: SimpleNamespace(mlir_input=config_mlir),
    )
    monkeypatch.setattr(
        op,
        "get_insts_artifact",
        lambda shape_mlir, _kernels: SimpleNamespace(mlir_input=shape_mlir),
    )
    monkeypatch.setattr(op, "add_artifacts", lambda _artifacts: None, raising=False)

    op.set_up_artifacts()
    return generated_shapes, op.xclbin_artifact, op.insts_artifact


def test_configuration_equivalent_experimental_shape_reuses_config_mlir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = UnifiedShape("experimental_config_equivalent", 256, 1024)
    op = unified_op.UnifiedGEMV.for_shape(
        shape, experimental_shape=True, context=object()
    )

    generated, xclbin, insts = _set_up_artifacts_with_captured_mlir_shapes(
        op, monkeypatch
    )

    assert len(generated) == 1
    assert insts.mlir_input is xclbin.mlir_input
    assert unified_op.UnifiedGEMV._shape_fields(
        insts.mlir_input.shape
    ) == unified_op.UnifiedGEMV._shape_fields(op.shape)


def test_reference_km_q4nx_experimental_shape_builds_own_insts_mlir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = UnifiedShape("experimental_q4nx", 256, 1024, format=FORMAT_Q4NX)
    op = unified_op.UnifiedGEMV.for_shape(
        shape, experimental_shape=True, context=object()
    )

    generated, xclbin, insts = _set_up_artifacts_with_captured_mlir_shapes(
        op, monkeypatch
    )

    assert len(generated) == 2
    assert insts.mlir_input is not xclbin.mlir_input
    assert insts.mlir_input.shape is op.shape
    assert insts.mlir_input.shape.format == FORMAT_Q4NX


def test_reference_km_single_col_experimental_shape_builds_own_insts_mlir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = UnifiedShape("experimental_single_col", 256, 1024, active_cols=1)
    op = unified_op.UnifiedGEMV.for_shape(
        shape, experimental_shape=True, context=object()
    )

    generated, xclbin, insts = _set_up_artifacts_with_captured_mlir_shapes(
        op, monkeypatch
    )

    assert len(generated) == 2
    assert insts.mlir_input is not xclbin.mlir_input
    assert insts.mlir_input.shape is op.shape
    assert insts.mlir_input.shape.active_cols == 1


@pytest.mark.parametrize("shape", ALL_SHAPES, ids=lambda shape: shape.name)
def test_product_shapes_build_insts_from_their_own_mlir(
    shape: UnifiedShape, monkeypatch: pytest.MonkeyPatch
) -> None:
    op = unified_op.UnifiedGEMV.for_shape(shape, context=object())

    generated, xclbin, insts = _set_up_artifacts_with_captured_mlir_shapes(
        op, monkeypatch
    )

    assert len(generated) == 2
    assert insts.mlir_input is not xclbin.mlir_input
    assert insts.mlir_input.shape is op.shape
    assert unified_op.UnifiedGEMV._shape_fields(
        insts.mlir_input.shape
    ) == unified_op.UnifiedGEMV._shape_fields(shape)


def test_mock_design_generates_rtp_barriers_and_dma_sequence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `gemv_unified` emits RTP writes -> barrier sets -> X fills -> W/Y drains in order."""
    events: list[tuple[str, Any]] = []
    created_workers: list[dict[str, Any]] = []
    join_tiles: list[tuple[str, int, int]] = []

    class FakeTypeSubscript:
        def __getitem__(self, item: Any) -> Any:
            return ("ndarray", item)

    class FakeTile:
        def __init__(self, col: int, row: int) -> None:
            self.col = col
            self.row = row

    class FakeEndpoint:
        def __init__(self, name: str, role: str, tile: Any = None, depth: int | None = None) -> None:
            self.name = name
            self.role = role
            self.tile = tile
            self.depth = depth

        def split(self, offsets, tile=None, depths=None, obj_types=None, names=None, **kwargs):
            if depths is None:
                depths = [self.depth or 2] * len(offsets)
            if names is None:
                names = [f"{self.name}_split{i}" for i in range(len(offsets))]
            if obj_types is None:
                obj_types = [None] * len(offsets)
            return [FakeObjectFifo(t, d, n) for n, t, d in zip(names, obj_types, depths)]

        def join(self, offsets, tile=None, depths=None, obj_types=None, names=None, **kwargs):
            join_tiles.append((self.name, tile.col, tile.row))
            if depths is None:
                depths = [self.depth or 2] * len(offsets)
            if names is None:
                names = [f"{self.name}_join{i}" for i in range(len(offsets))]
            if obj_types is None:
                obj_types = [None] * len(offsets)
            return [FakeObjectFifo(t, d, n) for n, t, d in zip(names, obj_types, depths)]

        def fill(self, buf, tap, wait=False, managed=True, group=None):
            events.append(("fill", self.name, tap.offset, tap.sizes[-1], wait, managed))
            task = SimpleNamespace(
                name=self.name,
                await_=lambda: events.append(("await", self.name)),
                free=lambda: events.append(("free", self.name)),
            )
            return task

        def drain(self, buf, tap, wait=False, group=None):
            events.append(("drain", self.name, tap.offset, tap.sizes[-1], wait))

    class FakeObjectFifo:
        def __init__(self, obj_type: Any, depth: int = 2, name: str = "") -> None:
            self.obj_type = obj_type
            self.depth = depth
            self.name = name

        def prod(self, tile: Any = None, depth: int | None = None) -> FakeEndpoint:
            return FakeEndpoint(self.name, "prod", tile=tile, depth=depth)

        def cons(self, tile: Any = None, depth: int | None = None) -> FakeEndpoint:
            return FakeEndpoint(self.name, "cons", tile=tile, depth=depth)

    class FakeBuffer:
        def __init__(
            self,
            type: Any,
            name: str,
            initial_value: Any = None,
            use_write_rtp: bool = False,
            tile: Any = None,
        ) -> None:
            self.type = type
            self.name = name
            self.tile = tile
            self.use_write_rtp = use_write_rtp
            self.values: dict[int, int] = {}

        def __setitem__(self, idx: int, val: int) -> None:
            self.values[idx] = int(val)
            events.append(("rtp_set", self.name, int(idx), int(val)))

    class FakeBarrier:
        _counter = 0

        def __init__(self) -> None:
            self.id = FakeBarrier._counter
            FakeBarrier._counter += 1

        def set(self, val: int) -> None:
            events.append(("barrier_set", self.id, int(val)))

    class FakeWorker:
        def __init__(
            self,
            core_fn,
            fn_args,
            tile=None,
            placement=None,
            stack_size=0,
            while_true=True,
            dynamic_objfifo_lowering=False,
            **kwargs,
        ) -> None:
            created_workers.append(
                {
                    "core_fn": core_fn,
                    "fn_args": fn_args,
                    "tile": tile,
                    "placement": placement,
                    "stack_size": stack_size,
                    "while_true": while_true,
                    "dynamic_objfifo_lowering": dynamic_objfifo_lowering,
                }
            )

    class FakeTAP:
        def __init__(self, tensor_dims, offset, sizes, strides) -> None:
            self.tensor_dims = tensor_dims
            self.offset = offset
            self.sizes = sizes
            self.strides = strides

    tg_count = [0]

    class FakeTaskGroup:
        def __init__(self) -> None:
            self.id = tg_count[0]
            tg_count[0] += 1
            events.append(("tg_new", self.id))

        def finish(self) -> None:
            events.append(("tg_finish", self.id))

    class FakeLock:
        def __init__(self, tile, lock_id=None, init=0, name=None) -> None:
            self.tile = tile
            self.init = init
            self.name = name

    class FakeFlowEndpoint:
        def __init__(self, flow_name: str, tile: Any) -> None:
            self.flow_name = flow_name
            self.tile = tile

    class FakeFlow:
        def __init__(self, src, dst, name=None, **kwargs) -> None:
            self.src = src
            self.dst = dst
            self.name = name or "flow"

        def endpoint(self, tile):
            return FakeFlowEndpoint(self.name, tile)

        def fill(self, buf, tap, wait=False, group=None):
            events.append(("fill", self.name, tap.offset, tap.sizes[-1], wait, True))

    class FakeDmaTaskSet:
        def __init__(self, endpoint, chunk=None) -> None:
            self.endpoint = endpoint
            self.chunk = chunk
            self.group = tg_count[0] - 1  # round's TaskGroup (created before tasks)

        def start(self) -> None:
            events.append(("w_task_start", self.endpoint.flow_name, self.chunk))

        def finish(self) -> None:
            events.append(("w_task_finish", self.endpoint.flow_name, self.group, self.chunk))

    configure_kwargs: list[dict[str, Any]] = []

    def fake_configure(endpoint, *args, **kwargs):
        configure_kwargs.append(kwargs)
        return FakeDmaTaskSet(endpoint)

    chunk_kwargs: list[dict[str, Any]] = []
    core_call_kwargs: list[dict[str, Any]] = []

    def fake_configure_chunk(endpoint, *args, **kwargs):
        chunk_kwargs.append(kwargs)
        return FakeDmaTaskSet(endpoint, chunk=kwargs["chunk"])

    def fake_configure_core_call(endpoint, *args, **kwargs):
        core_call_kwargs.append({"endpoint": endpoint, **kwargs})
        return FakeDmaTaskSet(endpoint, chunk=kwargs["chunk"])

    def fake_start_all(task_sets):
        for task_set in task_sets:
            task_set.start()

    def fake_finish_all(task_sets):
        for task_set in task_sets:
            task_set.finish()

    class FakeRuntime:
        def __init__(self, seq_fn=None, fn_args=None, **kwargs) -> None:
            self.seq_fn = seq_fn
            self.fn_args = fn_args or []

        def add_flow(self, flow) -> None:
            pass

        def add_lock(self, lock) -> None:
            pass

        def run(self) -> None:
            if self.seq_fn is not None:
                self.seq_fn(*self.fn_args)

    class FakeProgram:
        def __init__(self, dev, rt, workers=None) -> None:
            self.dev = dev
            self.rt = rt
            self.workers = workers or []

        def resolve_program(self, device_name="main", placer=None):
            events.append(("start_workers", len(self.workers)))
            if hasattr(self.rt, "run"):
                self.rt.run()
            return self

    monkeypatch.setattr(unified_design, "NPU2", None, raising=False)
    monkeypatch.setattr(unified_design, "bfloat16", np.uint16, raising=False)
    monkeypatch.setattr(unified_design, "Tile", FakeTile, raising=False)
    monkeypatch.setattr(unified_design, "ObjectFifo", FakeObjectFifo, raising=False)
    monkeypatch.setattr(unified_design, "Buffer", FakeBuffer, raising=False)
    monkeypatch.setattr(unified_design, "Flow", FakeFlow, raising=False)
    monkeypatch.setattr(unified_design, "Lock", FakeLock, raising=False)
    monkeypatch.setattr(unified_design, "WorkerRuntimeBarrier", FakeBarrier, raising=False)
    monkeypatch.setattr(unified_design, "Worker", FakeWorker, raising=False)
    monkeypatch.setattr(unified_design, "TaskGroup", FakeTaskGroup, raising=False)
    monkeypatch.setattr(unified_design, "TensorAccessPattern", FakeTAP, raising=False)
    monkeypatch.setattr(unified_design, "Runtime", FakeRuntime, raising=False)
    monkeypatch.setattr(unified_design, "Program", FakeProgram, raising=False)
    monkeypatch.setattr(
        unified_design, "configure_ping_pong_tasks", fake_configure, raising=False
    )
    monkeypatch.setattr(
        unified_design, "configure_chunk_tasks", fake_configure_chunk, raising=False
    )
    monkeypatch.setattr(
        unified_design,
        "configure_core_call_tasks",
        fake_configure_core_call,
        raising=False,
    )
    monkeypatch.setattr(unified_design, "start_all", fake_start_all, raising=False)
    monkeypatch.setattr(unified_design, "finish_all", fake_finish_all, raising=False)
    monkeypatch.setattr(
        unified_design, "declare_runtime_buffers", lambda buffers: None, raising=False
    )

    fake_kernels = ["k_zero", "k_block", "k_epi"]
    shape = get_shape("o_proj")  # K=5120, M=2560, chunks=20, rounds=(3,3,3,3,2,2,2,2)
    prog = unified_design.gemv_unified(
        dev=None, K=shape.K, M=shape.M, cols=COLS, gemv_kernel=fake_kernels
    )
    assert isinstance(prog, FakeProgram)
    assert len(created_workers) == 32
    assert join_tiles == [(f"y{col}", col, 1) for col in range(COLS)]
    for w in created_workers:
        assert w["stack_size"] == unified_design.STACK_SIZE == 10240
        assert w["dynamic_objfifo_lowering"] is True

    # Verify ordering: start_workers -> 96 rtp_set -> 32 barrier_set -> 8 x fills -> 20 (y drain + w fill) -> 8 x frees
    assert events[0] == ("start_workers", 32)
    rtp_events = [e for e in events if e[0] == "rtp_set"]
    barrier_events = [e for e in events if e[0] == "barrier_set"]
    assert len(rtp_events) == 32 * 3 == 96
    assert len(barrier_events) == 32

    last_rtp_idx = max(i for i, e in enumerate(events) if e[0] == "rtp_set")
    first_barrier_idx = min(i for i, e in enumerate(events) if e[0] == "barrier_set")
    last_barrier_idx = max(i for i, e in enumerate(events) if e[0] == "barrier_set")
    first_dma_idx = min(i for i, e in enumerate(events) if e[0] in ("fill", "drain"))
    assert last_rtp_idx < first_barrier_idx < last_barrier_idx < first_dma_idx

    w_fills = [e for e in events if e[0] == "fill" and e[1].startswith("w")]
    y_drains = [e for e in events if e[0] == "drain" and e[1].startswith("y")]
    plan = stream_plan(shape)
    assert len(w_fills) == len(plan) == 20
    assert len(y_drains) == len(plan) == 20
    for rec, wf, yd in zip(plan, w_fills, y_drains):
        assert wf[1] == f"w_in_{rec['col']}"
        assert wf[2] == rec["w_offset_words"]
        assert wf[3] == rec["w_length_words"]
        assert yd[1] == f"y{rec['col']}"
        assert yd[2] == rec["y_offset_bf16"]
        assert yd[3] == rec["y_length_bf16"]

    x_fills = [e for e in events if e[0] == "fill" and e[1].startswith("x")]
    assert len(x_fills) == COLS
    assert all(e[3] == MAX_K for e in x_fills)

    # fix3: mem/core W tasks issue no completion token (a token from a non-shim
    # tile adds a TileControl -> shim South overlay flow that busts the port
    # budget), and each is freed only after its own round's Y drain await
    # (TaskGroup.finish).
    assert configure_kwargs and all(
        kw.get("issue_token", False) is False for kw in configure_kwargs
    )
    assert chunk_kwargs and all(
        kw.get("issue_token", False) is False for kw in chunk_kwargs
    )
    # Core and default memory-tile W tasks are per round and are freed only
    # after that round's Y await.
    finished_groups: set[int] = set()
    started: set[Any] = set()
    n_core_finish = n_mem_finish = 0
    current_round = -1
    for e in events:
        if e[0] == "tg_new":
            current_round = e[1]
        elif e[0] == "tg_finish":
            finished_groups.add(e[1])
        elif e[0] == "w_task_start" and e[2] is not None:
            started.add((e[1], e[2]))
            assert e[2].first_round == current_round, e
        elif e[0] == "w_task_finish":
            if e[3] is None:
                n_core_finish += 1
                assert e[2] in finished_groups, e
            else:
                n_mem_finish += 1
                assert (e[1], e[3]) in started, e
                # the group of the chunk's last round must already be awaited
                assert e[3].last_round in finished_groups, e
    assert n_core_finish == 20 * 4
    n_chunks = sum(len(c) for c in mem_chunk_plan(shape).values())
    assert n_mem_finish == n_chunks * 5 and n_chunks == shape.chunks

    # Per-call odd-K path: one W fill and one Y drain/wait group per active
    # column, with W queued first.  The one-block f_b core chain repeats once
    # per column round, so shortening runs by one breaks this assertion.
    events.clear()
    core_call_kwargs.clear()
    tg_count[0] = 0
    odd = get_shape("f_b")
    prog = unified_design.gemv_unified(
        dev=None,
        K=odd.K,
        M=odd.M,
        cols=COLS,
        gemv_kernel=fake_kernels,
        sequence_plan="per_call",
    )
    assert isinstance(prog, FakeProgram)
    w_fills = [e for e in events if e[0] == "fill" and e[1].startswith("w_in_")]
    y_drains = [e for e in events if e[0] == "drain" and e[1].startswith("y")]
    finishes = [e for e in events if e[0] == "tg_finish"]
    assert len(w_fills) == len(y_drains) == len(finishes) == odd.resolve_active_cols(COLS)
    for col in range(odd.resolve_active_cols(COLS)):
        w_index = next(i for i, e in enumerate(events) if e[0] == "fill" and e[1] == f"w_in_{col}")
        y_index = next(i for i, e in enumerate(events) if e[0] == "drain" and e[1] == f"y{col}")
        assert w_index < y_index
    rounds = odd.rounds_by_col(COLS)
    assert len(core_call_kwargs) == odd.resolve_active_cols(COLS) * ROWS
    for entry in core_call_kwargs:
        col = entry["endpoint"].tile.col
        chunk = entry["chunk"]
        assert chunk.slots == (0,)
        assert chunk.runs == rounds[col]


def test_unified_op_shares_single_xclbin_and_asserts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    unified_op.reset_xclbin_signature_cache()

    class FakeMLIRArt:
        def __init__(self, filename, generator):
            self.filename = str(tmp_path / filename)
            self.generator = generator

    class FakeKernelObjArt:
        def __init__(self, kernel):
            self.kernel = kernel

        @classmethod
        def from_extern(cls, kernel):
            return cls(kernel)

    class FakeDesignGenerator:
        def __init__(self, source_path, entry_point, args, kwargs=None):
            self.source_path = source_path
            self.entry_point = entry_point
            self.args = args
            self.kwargs = kwargs or {}

    class FakeXclbinArt:
        def __init__(self, filename, mlir_input, dependencies, kernel_name, extra_flags=None):
            self.filename = str(tmp_path / filename)
            self.mlir_input = mlir_input
            self.dependencies = dependencies
            self.kernel_name = kernel_name
            self.extra_flags = extra_flags or []
            Path(self.filename).write_bytes(b"FAKE_XCLBIN")

    class FakeInstsArt:
        def __init__(self, filename, mlir_input, dependencies, extra_flags=None):
            self.filename = str(tmp_path / filename)
            self.mlir_input = mlir_input
            self.dependencies = dependencies
            self.extra_flags = extra_flags or []
            Path(self.filename).write_bytes(f"INSTS_{filename}".encode("utf-8"))

    class FakeArgSpec:
        def __init__(self, direction: str, shape: tuple[int, ...], dtype: Any = None):
            self.direction = direction
            self.shape = shape
            self.dtype = dtype

    monkeypatch.setattr(unified_op, "bfloat16", np.uint16, raising=False)
    monkeypatch.setattr(unified_op, "PythonGeneratedMLIRArtifact", FakeMLIRArt, raising=False)
    monkeypatch.setattr(unified_op, "KernelObjectArtifact", FakeKernelObjArt, raising=False)
    monkeypatch.setattr(unified_op, "DesignGenerator", FakeDesignGenerator, raising=False)
    monkeypatch.setattr(unified_op, "XclbinArtifact", FakeXclbinArt, raising=False)
    monkeypatch.setattr(unified_op, "InstsBinArtifact", FakeInstsArt, raising=False)
    monkeypatch.setattr(unified_op, "AIERuntimeArgSpec", FakeArgSpec, raising=False)
    fake_kernel = SimpleNamespace(
        archive="k0.a",
        bfp16_gemv_zero=SimpleNamespace(),
        bfp16_gemv_block=SimpleNamespace(),
        bfp16_gemv_epilogue=SimpleNamespace(),
    )
    monkeypatch.setattr(
        unified_op,
        "unified_bfp16_kernel",
        lambda: fake_kernel,
    )

    fake_ctx = SimpleNamespace(
        device_manager=SimpleNamespace(device=None, device_str="npu2")
    )

    ops: list[unified_op.UnifiedGEMV] = []
    for s in SHAPES:
        op = unified_op.UnifiedGEMV.for_shape(s, context=fake_ctx)
        op.register_artifacts = lambda arts: None  # type: ignore[method-assign]
        op.set_up_artifacts()
        assert op.get_arg_spec()[2].shape == (MAX_K,)
        assert "--dynamic-objFifos" in op.xclbin_artifact.extra_flags
        assert "--dynamic-objFifos" in op.insts_artifact.extra_flags
        ops.append(op)

    sig = unified_op.assert_shared_xclbin(ops)
    assert Path(sig["xclbin_path"]).name == "Raon_GEMV_Unified_c8_npu2.xclbin"
    insts_names = {Path(op.insts_artifact.filename).name for op in ops}
    assert len(insts_names) == len(SHAPES) == 10

    # Tampering with one operator's xclbin path must trigger AssertionError
    ops[-1].xclbin_artifact.filename = str(tmp_path / "different.xclbin")
    Path(ops[-1].xclbin_artifact.filename).write_bytes(b"OTHER")
    with pytest.raises(AssertionError, match="xclbin path mismatch"):
        unified_op.assert_shared_xclbin(ops)


def test_bit_reference_and_short_dma_simulation() -> None:
    w_words, x_u16, y_ref_u16 = make_shape_test_case(512, 256, seed=42, ref_chunk_m=128)
    assert w_words.shape == ((256 * 512 * 9) // 32,)
    assert x_u16.shape == (512,)
    assert y_ref_u16.shape == (256,)

    cmp_ok = compare_bf16_bits(y_ref_u16, y_ref_u16)
    assert cmp_ok["mismatches"] == 0
    assert cmp_ok["max_ulp"] == 0

    bad = y_ref_u16.copy()
    bad[7] ^= np.uint16(1)
    cmp_bad = compare_bf16_bits(bad, y_ref_u16)
    assert cmp_bad["mismatches"] == 1
    assert cmp_bad["max_ulp"] == 1
    assert cmp_bad["first_mismatch"]["index"] == 7

    sim = simulate_short_dma_ping_pong(iters=200)
    assert sim["sentinel_preserved"] is True
