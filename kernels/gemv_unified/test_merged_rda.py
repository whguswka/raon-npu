# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side unit tests for merged GEMV (cols=7) + RDA (col 7) in a single PDI.

Tests run in pure Python without requiring an NPU, verifying:
  1. rda_col parameter validation (overlap rejection, range bounds).
  2. Configuration naming (Raon_GEMV_Unified_c7_rda7_... parity).
  3. UnifiedRDA argument contract (4-BO ABI: s_in, s_out, head_in, out).
  4. RDA column plan parity between standalone 1-col design and merged design.
  5. Exact device body identity (Tiles, Workers, ObjectFifos, Buffers, Locks, Flows)
     between mode='gemv' and mode='rda'.
  6. Shared xclbin artifact identity between UnifiedGEMV and UnifiedRDA.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
import numpy as np
import pytest

from bench.rda_state.rda_state_1col import (
    HEAD_BUNDLE_SIZE,
    K_DIM,
    V_DIM,
    RdaResource,
    rda_column_plan,
)
from kernels.gemv_unified import design as design_mod
from kernels.gemv_unified.op import UnifiedGEMV
from kernels.gemv_unified.rda import SharedRdaCallable, UnifiedRDA
from kernels.gemv_unified.shapes import M_PER_CHUNK
from kernels.gemv_unified.stage2 import UnifiedMoEDown9, UnifiedMoEGateUp


# --------------------------------------------------------------------------
# 1. rda_col validation
# --------------------------------------------------------------------------
class TestRdaColValidation:
    """Check constraints on cols and rda_col in design generator and operators."""

    def test_rda_col_overlap_with_active_cols_raises(self) -> None:
        # cols=7 uses cols 0..6, so rda_col=6 overlaps.
        with pytest.raises(ValueError, match="overlaps GEMV active columns"):
            UnifiedGEMV(K=256, M=896, cols=7, rda_col=6, experimental_shape=True)

        with pytest.raises(ValueError, match="overlaps GEMV active columns"):
            UnifiedRDA(cols=7, rda_col=6)

    def test_rda_col_overlap_with_8cols_raises(self) -> None:
        # cols=8 uses all columns 0..7, so rda_col=7 overlaps.
        with pytest.raises(ValueError, match="overlaps GEMV active columns"):
            UnifiedGEMV(K=256, M=1024, cols=8, rda_col=7, experimental_shape=True)

        with pytest.raises(ValueError, match="overlaps GEMV active columns"):
            UnifiedRDA(cols=8, rda_col=7)

    def test_rda_col_valid_at_col7_with_cols7(self) -> None:
        # cols=7 with rda_col=7 is valid (disjoint partition: 0..6 for GEMV, 7 for RDA).
        op = UnifiedGEMV(K=256, M=896, cols=7, rda_col=7, experimental_shape=True)
        assert op.cols == 7
        assert op.rda_col == 7

        rda_op = UnifiedRDA(cols=7, rda_col=7)
        assert rda_op.cols == 7
        assert rda_op.rda_col == 7

    def test_rda_col_out_of_bounds_raises(self) -> None:
        for invalid_col in (-1, 8, 99):
            with pytest.raises(ValueError, match="must be within device columns"):
                UnifiedGEMV(K=256, M=896, cols=7, rda_col=invalid_col, experimental_shape=True)

            with pytest.raises(ValueError, match="must be within device columns"):
                UnifiedRDA(cols=7, rda_col=invalid_col)

    def test_merged_path_accepts_only_contract_heads(self) -> None:
        # The merged config path supports exactly the model contract count (40):
        # the shared xclbin/insts artifact names carry no head count, so any other
        # value would silently share artifacts compiled for 40 heads.
        op = UnifiedGEMV(
            K=256, M=896, cols=7, rda_col=7, rda_heads=40, experimental_shape=True
        )
        assert op.rda_heads == 40

        rda_op = UnifiedRDA(cols=7, rda_col=7, rda_heads=40)
        assert rda_op.rda_heads == 40

    def test_merged_path_rejects_non_contract_heads(self) -> None:
        # Odd counts would additionally stall: every RDA BD chain runs runs=H//2.
        for bad_heads in (39, 41, 0, -2):
            with pytest.raises(
                ValueError,
                match=r"the merged config path supports only rda_heads=40",
            ):
                UnifiedGEMV(
                    K=256,
                    M=896,
                    cols=7,
                    rda_col=7,
                    rda_heads=bad_heads,
                    experimental_shape=True,
                )

            with pytest.raises(
                ValueError,
                match=r"UnifiedRDA supports only rda_heads=40",
            ):
                UnifiedRDA(cols=7, rda_col=7, rda_heads=bad_heads)

    def test_rda_mode_without_rda_col_raises(self) -> None:
        with pytest.raises(ValueError, match="rda mode requires rda_col"):
            design_mod.gemv_unified(None, 256, 896, cols=7, rda_col=None, mode="rda")


# --------------------------------------------------------------------------
# 2. Configuration naming
# --------------------------------------------------------------------------
class TestConfigurationNaming:
    """Verify that rda_col is reflected in the shared configuration xclbin name."""

    def test_config_name_includes_c7_rda7(self) -> None:
        op = UnifiedGEMV(K=256, M=896, cols=7, rda_col=7, experimental_shape=True)
        assert "c7_rda7" in op.config_name
        assert op.config_name.startswith("Raon_GEMV_Unified_c7_rda7_")

    def test_config_name_without_rda_keeps_c8(self) -> None:
        op8 = UnifiedGEMV(K=256, M=1024, cols=8, rda_col=None, experimental_shape=True)
        assert "c8" in op8.config_name
        assert "rda" not in op8.config_name

    def test_unified_rda_shares_exact_gemv_config_name(self) -> None:
        gemv_op = UnifiedGEMV(K=256, M=896, cols=7, rda_col=7, experimental_shape=True)
        rda_op = UnifiedRDA(cols=7, rda_col=7)
        assert rda_op._gemv_op.config_name == gemv_op.config_name
        assert rda_op.name == f"{gemv_op.config_name}_rda"

    def test_moe_templates_share_c7_rda7_config(self) -> None:
        gu = UnifiedMoEGateUp(cols=7, rda_col=7)
        dn = UnifiedMoEDown9(cols=7, rda_col=7)
        assert "c7_rda7" in gu._op.config_name
        assert gu._op.config_name == dn._op.config_name


# --------------------------------------------------------------------------
# 3. UnifiedRDA argument contract (4-BO ABI)
# --------------------------------------------------------------------------
class TestUnifiedRDAContract:
    """Verify the 4-BO ABI specifications for UnifiedRDA."""

    def test_arg_spec_contains_exactly_four_bos(self) -> None:
        rda_op = UnifiedRDA(cols=7, rda_col=7, rda_heads=40)
        specs = rda_op.get_arg_spec()
        assert len(specs) == 4

        # s_in: in fp32 [40, 128, 128]
        assert specs[0].direction == "in"
        assert specs[0].shape == (40, V_DIM, K_DIM)
        assert np.dtype(specs[0].dtype) == np.float32

        # s_out: out fp32 [40, 128, 128]
        assert specs[1].direction == "out"
        assert specs[1].shape == (40, V_DIM, K_DIM)
        assert np.dtype(specs[1].dtype) == np.float32

        # head_in: in fp32 [40, 640]
        assert specs[2].direction == "in"
        assert specs[2].shape == (40, HEAD_BUNDLE_SIZE)
        assert np.dtype(specs[2].dtype) == np.float32

        # out: out fp32 [40, 128]
        assert specs[3].direction == "out"
        assert specs[3].shape == (40, V_DIM)
        assert np.dtype(specs[3].dtype) == np.float32

    def test_shared_rda_callable_requires_positional_four_args(self) -> None:
        inner_calls = []

        def fake_inner(*args: Any) -> Any:
            inner_calls.append(args)
            return "done"

        op_path = "/fake/path.xclbin"
        callable_wrapper = SharedRdaCallable(
            fake_inner,
            xclbin_path=op_path,  # type: ignore[arg-type]
            xclbin_mtime_ns=1234567,
            kernel_name="MLIR_AIE",
            arg_spec=[],
        )

        # 4 args succeeds
        ret = callable_wrapper("s_in", "s_out", "head_in", "out")
        assert ret == "done"
        assert len(inner_calls) == 1

        # Wrong argument count fails
        with pytest.raises(ValueError, match="requires positional 4-BO ABI"):
            callable_wrapper("s_in", "s_out", "head_in")  # 3 args

        with pytest.raises(ValueError, match="requires positional 4-BO ABI"):
            callable_wrapper("s_in", "s_out", "head_in", "out", "extra")  # 5 args

        with pytest.raises(ValueError, match="requires positional 4-BO ABI"):
            callable_wrapper("s_in", "s_out", "head_in", out="out")  # keyword arg


# --------------------------------------------------------------------------
# 4. Resource table parity between standalone 1-col design and merged design
# --------------------------------------------------------------------------
class TestResourceTableParity:
    """Verify that rda_column_plan matches the RDA resources in col 7."""

    def test_rda_column_plan_counts_and_types(self) -> None:
        plan = rda_column_plan(col=7, num_heads=40)
        assert isinstance(plan, tuple)
        assert len(plan) == 42

        by_kind: dict[str, list[RdaResource]] = {}
        for r in plan:
            by_kind.setdefault(r.kind, []).append(r)

        # ObjectFifos: 1 l3l2 + 4 l2l1 + 1 head_in = 6
        assert len(by_kind["objectfifo"]) == 6
        assert by_kind["objectfifo"][0].name == "rda_s_in_l3l2_7"
        assert by_kind["objectfifo"][-1].name == "rda_head_in_7"

        # Buffers: 2 agg_mem (16,512 floats) + 4 packed_core (4,128 floats) = 6
        assert len(by_kind["buffer"]) == 6
        assert [b.elements for b in by_kind["buffer"][:2]] == [16512, 16512]
        assert [b.elements for b in by_kind["buffer"][2:]] == [4128, 4128, 4128, 4128]

        # Locks: 8 agg_empty(init=2) + 8 agg_full(init=0) + 4 core_empty(init=1) + 4 core_full(init=0) = 24
        assert len(by_kind["lock"]) == 24
        agg_empty = [lock for lock in by_kind["lock"] if "agg_empty" in lock.name]
        assert len(agg_empty) == 8
        assert all(lock.init == 2 for lock in agg_empty)

        agg_full = [lock for lock in by_kind["lock"] if "agg_full" in lock.name]
        assert len(agg_full) == 8
        assert all(lock.init == 0 for lock in agg_full)

        core_empty = [lock for lock in by_kind["lock"] if "packed_empty" in lock.name]
        assert len(core_empty) == 4
        assert all(lock.init == 1 for lock in core_empty)

        core_full = [lock for lock in by_kind["lock"] if "packed_full" in lock.name]
        assert len(core_full) == 4
        assert all(lock.init == 0 for lock in core_full)

        # Flows: 4 packed_c2m + 1 state_m2s + 1 out_m2s = 6
        assert len(by_kind["flow"]) == 6
        flow_names = [f.name for f in by_kind["flow"]]
        assert flow_names == [
            "rda_packed_c2m_7_0",
            "rda_packed_c2m_7_1",
            "rda_packed_c2m_7_2",
            "rda_packed_c2m_7_3",
            "rda_state_m2s_7",
            "rda_out_m2s_7",
        ]


# --------------------------------------------------------------------------
# 5. Exact device body identity between mode='gemv' and mode='rda'
# --------------------------------------------------------------------------
class RecordedResource:
    def __init__(self, kind: str, name: str, **kwargs: Any) -> None:
        self.kind = kind
        self.name = name
        self.kwargs = kwargs

    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"{self.kind}({self.name}{', ' + kw if kw else ''})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, RecordedResource):
            return False
        return (self.kind, self.name, self.kwargs) == (other.kind, other.name, other.kwargs)


@contextmanager
def record_device_declarations():
    """Intercept all IRON declarations in design.py and record order of declaration."""
    import bench.rda_state.rda_state_1col as rda_mod

    class EventList(list):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.runtimes: list[Any] = []

    events = EventList()

    class MockTile:
        def __init__(self, col: int, row: int) -> None:
            self.col = col
            self.row = row
            events.append(RecordedResource("Tile", f"Tile({col},{row})", col=col, row=row))

        def __repr__(self) -> str:
            return f"Tile({self.col},{self.row})"

    class MockObjectFifoHandle:
        def __init__(self, parent: Any, role: str, tile: Any = None) -> None:
            self.parent = parent
            self.role = role
            self.tile = tile

        def cons(self, tile: Any = None) -> Any:
            return MockObjectFifoHandle(self.parent, "cons", tile)

        def prod(self, tile: Any = None) -> Any:
            return MockObjectFifoHandle(self.parent, "prod", tile)

        def join(self, offsets: Any, tile: Any, obj_types: Any, names: Any) -> list[Any]:
            events.append(RecordedResource("FifoJoin", self.parent.name, names=tuple(names)))
            return [MockObjectFifoHandle(self.parent, "joined", tile) for _ in names]

        def split(self, offsets: Any, tile: Any, obj_types: Any, names: Any, depths: Any) -> list[Any]:
            events.append(RecordedResource("FifoSplit", self.parent.name, names=tuple(names)))
            return [MockObjectFifoHandle(self.parent, "split", tile) for _ in names]

    class MockObjectFifo:
        def __init__(self, ty: Any, depth: int = 1, name: str = "") -> None:
            self.name = name
            self.depth = depth
            events.append(RecordedResource("ObjectFifo", name, depth=depth))

        def prod(self, tile: Any = None) -> MockObjectFifoHandle:
            return MockObjectFifoHandle(self, "prod", tile)

        def cons(self, tile: Any = None) -> MockObjectFifoHandle:
            return MockObjectFifoHandle(self, "cons", tile)

    class MockBuffer:
        def __init__(self, type: Any = None, name: str = "", tile: Any = None, **kw: Any) -> None:
            self.name = name
            self.tile = str(tile) if tile is not None else None
            events.append(RecordedResource("Buffer", name, tile=self.tile))

    class MockLock:
        def __init__(self, tile: Any, init: int = 0, name: str = "") -> None:
            self.name = name
            self.tile = str(tile)
            self.init = init
            events.append(RecordedResource("Lock", name, tile=self.tile, init=init))

    class MockFlow:
        def __init__(self, src: Any, dst: Any, name: str = "") -> None:
            self.name = name
            self.src = str(src)
            self.dst = str(dst)
            events.append(RecordedResource("Flow", name, src=self.src, dst=self.dst))

    class MockBarrier:
        def __init__(self) -> None:
            events.append(RecordedResource("Barrier", "barrier"))

    class MockWorker:
        def __init__(self, fn: Any, args: Any, tile: Any, **kw: Any) -> None:
            self.tile = str(tile)
            fn_name = getattr(fn, "__name__", "fn")
            events.append(RecordedResource("Worker", fn_name, tile=self.tile))

    class MockRuntime:
        def __init__(self, fn: Any, arg_types: Any) -> None:
            self.fn = fn
            self.arg_types = arg_types
            events.runtimes.append(self)
            for arg in arg_types:
                if isinstance(arg, list):
                    for item in arg:
                        if hasattr(item, "parent") and hasattr(item, "role"):
                            events.append(
                                RecordedResource("rt.handle", item.parent.name, role=item.role)
                            )
                elif hasattr(arg, "parent") and hasattr(arg, "role"):
                    events.append(
                        RecordedResource("rt.handle", arg.parent.name, role=arg.role)
                    )

        def add_lock(self, lock: Any) -> None:
            events.append(RecordedResource("rt.add_lock", lock.name, init=lock.init))

        def add_flow(self, flow: Any) -> None:
            events.append(RecordedResource("rt.add_flow", flow.name))

    class MockProgram:
        def __init__(self, dev: Any, rt: Any, workers: Any = None) -> None:
            self.dev = dev
            self.rt = rt
            self.workers = workers

        def enable_trace(self, trace_size: int) -> None:
            pass

        def resolve_program(self) -> list[RecordedResource]:
            return list(events)

    class MockTAP:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    from types import SimpleNamespace
    fake_gemv_kernel = SimpleNamespace(
        archive="fake_k0.a",
        bfp16_gemv_zero=lambda *a: None,
        bfp16_gemv_block=lambda *a: None,
        bfp16_gemv_epilogue=lambda *a: None,
        bfp16_gemv_unified_zero=lambda *a: None,
        bfp16_gemv_unified_block=lambda *a: None,
        bfp16_gemv_unified_epilogue=lambda *a: None,
        q4nx_gemv_unified_prep=lambda *a: None,
        q4nx_gemv_unified_zero=lambda *a: None,
        q4nx_gemv_unified_block=lambda *a: None,
        q4nx_gemv_unified_epilogue=lambda *a: None,
    )

    # Save originals
    orig_design_symbols = {
        name: getattr(design_mod, name)
        for name in (
            "Tile", "ObjectFifo", "Buffer", "Lock", "Flow",
            "Worker", "WorkerRuntimeBarrier", "Runtime", "Program",
            "TensorAccessPattern", "unified_bfp16_kernel",
        )
        if hasattr(design_mod, name)
    }
    orig_rda_iron = getattr(rda_mod, "_IRON_AVAILABLE", False)
    orig_rda_symbols = {
        name: getattr(rda_mod, name)
        for name in (
            "Tile", "ObjectFifo", "Buffer", "Lock", "Flow",
            "Worker", "WorkerRuntimeBarrier", "ExternalFunction",
        )
        if hasattr(rda_mod, name)
    }

    try:
        # Patch design_mod
        design_mod.Tile = MockTile
        design_mod.ObjectFifo = MockObjectFifo
        design_mod.Buffer = MockBuffer
        design_mod.Lock = MockLock
        design_mod.Flow = MockFlow
        design_mod.Worker = MockWorker
        design_mod.WorkerRuntimeBarrier = MockBarrier
        design_mod.Runtime = MockRuntime
        design_mod.Program = MockProgram
        design_mod.TensorAccessPattern = MockTAP
        design_mod.unified_bfp16_kernel = lambda *a, **kw: fake_gemv_kernel

        # Patch rda_mod
        rda_mod._IRON_AVAILABLE = True
        rda_mod.Tile = MockTile
        rda_mod.ObjectFifo = MockObjectFifo
        rda_mod.Buffer = MockBuffer
        rda_mod.Lock = MockLock
        rda_mod.Flow = MockFlow
        rda_mod.Worker = MockWorker
        rda_mod.WorkerRuntimeBarrier = MockBarrier
        rda_mod.ExternalFunction = lambda *a, **kw: "external_function"

        yield events
    finally:
        for k, v in orig_design_symbols.items():
            setattr(design_mod, k, v)
        rda_mod._IRON_AVAILABLE = orig_rda_iron
        for k, v in orig_rda_symbols.items():
            setattr(rda_mod, k, v)


class TestDeviceBodyIdentity:
    """Verify that mode='gemv' and mode='rda' produce bit-for-bit identical device bodies."""

    def test_gemv_and_rda_modes_declare_identical_device_resources(self) -> None:
        # 1. Run in mode='gemv'
        with record_device_declarations():
            prog_gemv = design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="gemv",
            )

        # 2. Run in mode='rda'
        with record_device_declarations():
            prog_rda = design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="rda",
            )

        # Both modes must yield the exact same resource list and order
        assert prog_gemv == prog_rda
        assert len(prog_gemv) > 0

        # Assert RDA components are present in both
        res_names = [r.name for r in prog_gemv]
        assert "rda_s_in_l3l2_7" in res_names
        assert "rda_agg_mem_7_0" in res_names
        assert "rda_state_m2s_7" in res_names
        assert "rda_out_m2s_7" in res_names

        # GEMV workers (7 cols x 4 rows = 28) + RDA workers (4) = 32 workers
        workers = [r for r in prog_gemv if r.kind == "Worker"]
        assert len(workers) == 7 * 4 + 4

    def test_revert_failure_simulation(self) -> None:
        """Requirement 2: If RDA components were conditionally declared ONLY in rda mode,

        the identity assertion must fail.
        """
        with record_device_declarations():
            # Mode gemv with NO RDA components (simulating a broken conditional declaration)
            prog_gemv_no_rda = design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=None,  # No RDA
                mode="gemv",
            )

        with record_device_declarations():
            # Mode rda with RDA components declared
            prog_rda_with_rda = design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                mode="rda",
            )

        # Confirm they differ
        assert prog_gemv_no_rda != prog_rda_with_rda

    def test_gemv_and_rda_runtime_handles_identity(self) -> None:
        """CPU host test: verify mode='gemv' and mode='rda' pass identical ObjectFifo handles in identical order."""
        with record_device_declarations() as ev_gemv:
            design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="gemv",
            )
        with record_device_declarations() as ev_rda:
            design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="rda",
            )

        rt_gemv = ev_gemv.runtimes[0]
        rt_rda = ev_rda.runtimes[0]

        handles_gemv = extract_runtime_handles(rt_gemv)
        handles_rda = extract_runtime_handles(rt_rda)

        expected = [
            ("x0", "prod"), ("x1", "prod"), ("x2", "prod"), ("x3", "prod"),
            ("x4", "prod"), ("x5", "prod"), ("x6", "prod"),
            ("y0", "cons"), ("y1", "cons"), ("y2", "cons"), ("y3", "cons"),
            ("y4", "cons"), ("y5", "cons"), ("y6", "cons"),
            ("rda_s_in_l3l2_7", "prod"),
            ("rda_head_in_7", "prod"),
        ]
        assert handles_gemv == expected
        assert handles_rda == expected
        assert handles_gemv == handles_rda

    def test_runtime_handles_mismatch_fails_when_rda_omitted(self) -> None:
        """Requirement: Revert test.

        When GEMV mode omits RDA handles (as before fix1), the comparison must fail.
        """
        with record_device_declarations() as ev_gemv_buggy:
            design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="gemv",
            )
            # Simulate old bug: GEMV mode omitted RDA handles (s_in_prod, head_in_prod)
            rt_buggy = ev_gemv_buggy.runtimes[0]
            rt_buggy.arg_types = rt_buggy.arg_types[:-2]

        with record_device_declarations() as ev_rda:
            design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="rda",
            )

        rt_rda = ev_rda.runtimes[0]
        handles_buggy = extract_runtime_handles(rt_buggy)
        handles_rda = extract_runtime_handles(rt_rda)

        assert len(handles_buggy) == 14  # only x_prods (7) + of_y_conss (7)
        assert len(handles_rda) == 16    # + s_in_prod (1) + head_in_prod (1)
        with pytest.raises(AssertionError, match=r"Right contains 2 more items"):
            assert handles_buggy == handles_rda


def extract_runtime_handles(rt: Any) -> list[tuple[str, str]]:
    """Extract list of (fifo_name, role) for all ObjectFifo handles in rt.arg_types."""
    handles: list[tuple[str, str]] = []
    for arg in getattr(rt, "arg_types", ()):
        if isinstance(arg, list):
            for item in arg:
                if hasattr(item, "parent") and hasattr(item, "role"):
                    handles.append((item.parent.name, item.role))
        elif hasattr(arg, "parent") and hasattr(arg, "role"):
            handles.append((arg.parent.name, arg.role))
    return handles


class TestStripRuntimeSequence:
    """Verify runtime sequence stripping logic used by merged_rda_check.py."""

    def test_strip_aie_runtime_sequence(self) -> None:
        from bench.gemv_unified.merged_rda_check import strip_runtime_sequence

        fake_mlir = (
            "module @test {\n"
            "  %tile0 = aie.tile(0, 0)\n"
            "  %tile1 = aie.tile(0, 1)\n"
            '  aie.runtime_sequence(%arg0: memref<1024xi32>) {\n'
            '    aie.dma_start("fake {brace} inside string")\n'
            "    aie.use_lock(%l0, acquire, 1)\n"
            "  }\n"
            "  %tile2 = aie.tile(0, 2)\n"
            "}\n"
        )
        stripped = strip_runtime_sequence(fake_mlir)
        assert "%tile0 = aie.tile(0, 0)" in stripped
        assert "%tile1 = aie.tile(0, 1)" in stripped
        assert "%tile2 = aie.tile(0, 2)" in stripped
        assert "aie.runtime_sequence" not in stripped
        assert "aie.dma_start" not in stripped

    def test_strip_aiex_runtime_sequence(self) -> None:
        from bench.gemv_unified.merged_rda_check import strip_runtime_sequence

        fake_mlir = (
            "module @test {\n"
            "  %buf = aie.buffer(%t) : memref<256xi32>\n"
            '  aiex.runtime_sequence(%arg0: memref<1024xi32>) {\n'
            "    aiex.npu.dma_memcpy_nd(0, 0, %arg0)\n"
            "  }\n"
            "  %lock = aie.lock(%t, 0)\n"
            "}\n"
        )
        stripped = strip_runtime_sequence(fake_mlir)
        assert "%buf = aie.buffer(%t)" in stripped
        assert "%lock = aie.lock(%t, 0)" in stripped
        assert "aiex.runtime_sequence" not in stripped
        assert "aiex.npu.dma_memcpy_nd" not in stripped

    def test_device_body_comparison_succeeds_when_sequences_differ(self) -> None:
        from bench.gemv_unified.merged_rda_check import strip_runtime_sequence

        # Two MLIR modules with the identical device body but different sequences
        body_prefix = (
            "module @main {\n"
            "  %t0 = aie.tile(0, 0)\n"
            "  %t1 = aie.tile(7, 1)\n"
            "  %fifo = aie.objectfifo @rda_head_in_7(%t0, {%t1}, 2)\n"
        )
        body_suffix = "  %worker = aie.core(%t1) {}\n}\n"

        mlir_gemv = (
            body_prefix
            + "  aie.runtime_sequence(%arg0: memref<256xbf16>, %arg1: memref<1024xbf16>) {\n"
            + "    // GEMV specific DMA sequence\n"
            + "  }\n"
            + body_suffix
        )
        mlir_rda = (
            body_prefix
            + "  aiex.runtime_sequence(%s_in: memref<40x128x128xf32>, %out: memref<40x128xf32>) {\n"
            + "    // RDA specific DMA sequence\n"
            + "  }\n"
            + body_suffix
        )

        stripped_gemv = strip_runtime_sequence(mlir_gemv)
        stripped_rda = strip_runtime_sequence(mlir_rda)
        assert stripped_gemv == stripped_rda


# --------------------------------------------------------------------------
# 7. Verification of main-copy fixes (CPU-only host tests)
# --------------------------------------------------------------------------
class TestMainFixesVerification:
    """CPU host unit tests verifying the three main-copy fixes in rda.py and design.py."""

    def test_rda_design_args_gemv_kernel_matches_config(self, monkeypatch: Any) -> None:
        """Fix 1: UnifiedRDA must pass self._gemv_op._kernel() (not None) as gemv_kernel."""
        captured_kwargs: dict[str, Any] = {}
        fake_kernel = object()

        def mock_design_generator(path: Any, fn_name: str, args: Any, kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return "mock_generator"

        import kernels.gemv_unified.rda as rda_mod
        monkeypatch.setattr(rda_mod, "DesignGenerator", mock_design_generator)
        monkeypatch.setattr(rda_mod, "PythonGeneratedMLIRArtifact", lambda name, gen: gen)

        rda_op = UnifiedRDA(cols=7, rda_col=7)
        monkeypatch.setattr(rda_op._gemv_op, "_kernel", lambda: fake_kernel)
        rda_op._mlir_artifact("test.mlir")

        assert captured_kwargs.get("gemv_kernel") is not None
        assert captured_kwargs["gemv_kernel"] is fake_kernel

    def test_runtime_sequence_buffer_prologue_order_matches(self) -> None:
        """Fix 2: gemv and rda runtime sequences must declare identical runtime buffers in identical order."""
        prologue_calls_gemv: list[list[str]] = []
        prologue_calls_rda: list[list[str]] = []

        class StopAfterPrologue(Exception):
            pass

        def make_mock_declare(target: list[list[str]]):
            def _declare(buffers: Any) -> None:
                names = [b.name for b in buffers]
                target.append(names)
                if len(target) == 2:
                    raise StopAfterPrologue()
            return _declare

        with record_device_declarations() as ev_gemv:
            design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="gemv",
            )
        rt_gemv = ev_gemv.runtimes[0]

        with record_device_declarations() as ev_rda:
            design_mod.gemv_unified(
                None,
                K=256,
                M=7 * M_PER_CHUNK,
                cols=7,
                rda_col=7,
                rda_heads=40,
                mode="rda",
            )
        rt_rda = ev_rda.runtimes[0]

        orig_declare = design_mod.declare_runtime_buffers
        try:
            design_mod.declare_runtime_buffers = make_mock_declare(prologue_calls_gemv)
            try:
                rt_gemv.fn(*(None for _ in range(16)))
            except StopAfterPrologue:
                pass

            design_mod.declare_runtime_buffers = make_mock_declare(prologue_calls_rda)
            try:
                rt_rda.fn(*(None for _ in range(8)))
            except StopAfterPrologue:
                pass
        finally:
            design_mod.declare_runtime_buffers = orig_declare

        assert len(prologue_calls_gemv) == 2
        assert len(prologue_calls_rda) == 2
        # First call: W mem buffers (cols 0..6 ping/pong = 14 buffers)
        assert prologue_calls_gemv[0] == prologue_calls_rda[0]
        assert len(prologue_calls_gemv[0]) == 14
        # Second call: RDA agg mem buffers (2 buffers: rda_agg_mem_7_0, rda_agg_mem_7_1)
        assert prologue_calls_gemv[1] == prologue_calls_rda[1]
        assert prologue_calls_gemv[1] == ["rda_agg_mem_7_0", "rda_agg_mem_7_1"]

    def test_compile_preserves_gemv_xclbin_artifact_identity(self) -> None:
        """Fix 3: compile() must keep self.xclbin_artifact identical to self._gemv_op.xclbin_artifact."""
        fake_gemv_xclbin = object()
        fake_other_xclbin = object()

        class MockMLIROperator:
            def compile(self, *args: Any, **kwargs: Any) -> str:
                self.xclbin_artifact = fake_other_xclbin
                return "super_compiled"

        class TestableUnifiedRDA(UnifiedRDA, MockMLIROperator):
            pass

        rda_op = TestableUnifiedRDA(cols=7, rda_col=7)

        gemv_compiled = []

        def mock_gemv_compile(*args: Any, **kwargs: Any) -> str:
            gemv_compiled.append(True)
            rda_op._gemv_op.xclbin_artifact = fake_gemv_xclbin
            return "gemv_compiled"

        setattr(rda_op._gemv_op, "compile", mock_gemv_compile)

        res = rda_op.compile()
        assert res == "super_compiled"
        assert len(gemv_compiled) == 1
        assert rda_op.xclbin_artifact is fake_gemv_xclbin
        assert rda_op.xclbin_artifact is not fake_other_xclbin

