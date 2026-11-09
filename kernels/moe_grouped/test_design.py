# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Static shape, transfer-count, and resource-budget checks."""

import importlib.util
from pathlib import Path

import pytest

import kernels.moe_grouped.kernel as kernel_module
import kernels.moe_grouped.op as op_module
from kernels.moe_grouped.dump_mlir import device_without_runtime_sequence
from kernels.moe_grouped.layout import B1, B2, resource_budget, stream_plan
from kernels.moe_grouped.op import GroupedMoE, validate_design_pair


def test_design_imports_without_package_context() -> None:
    source = Path(__file__).with_name("design.py")
    spec = importlib.util.spec_from_file_location(source.name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.NPU_AVAILABLE is False


def test_static_command_shape_and_patch_counts() -> None:
    for shape, count in ((B1, 136), (B2, 160)):
        plan = stream_plan(shape)
        assert len(plan) == count == shape.patch_count
        assert {record["slot"] for record in plan} == set(range(8))
        assert all(0 <= record["col"] < 8 for record in plan)
        assert all(record["w_offset_bytes"] < shape.expert_bytes for record in plan)


def test_budget_and_runtime_barrier_contract() -> None:
    budget = resource_budget()
    assert budget["l1_used_bytes"] <= budget["l1_limit_bytes"]
    assert budget["memtile_bytes"] <= budget["memtile_limit_bytes"]
    assert budget["core_bd_peak"] <= budget["core_bd_limit"]
    assert budget["memtile_bd_peak"] <= budget["memtile_bd_limit"]
    source = Path(__file__).with_name("design.py").read_text(encoding="utf-8")
    assert source.index("barrier.wait_for_value(1)") < source.index("mode = rtp[2]")
    assert "ObjectFifo" in source
    assert "from .layout" not in source
    assert "from .kernel" not in source


def test_runtime_uses_current_iron_constructor_contract() -> None:
    source = Path(__file__).with_name("design.py").read_text(encoding="utf-8")
    assert "def sequence(Y, W, X, x_ps, w_ps, y_cs):" in source
    assert "Runtime(\n        sequence," in source
    assert "[x_prods, w_prods, y_conses]" in source
    assert "Program(dev, runtime, workers=workers)" in source
    assert "runtime.sequence" not in source
    assert "runtime.start" not in source


def test_each_operator_pairs_its_own_device_and_commands() -> None:
    for shape in (B1, B2):
        op = GroupedMoE(shape.name)
        assert "rtp7" in op.config_name
        assert "rtp5" not in op.config_name
        assert "rtp4" not in op.config_name
        assert shape.name in op.config_name
        assert op.artifact_designs == (shape.name, shape.name)
        assert validate_design_pair(*op.artifact_designs).name == shape.name
    with pytest.raises(ValueError, match="xclbin=b1, instructions=b2"):
        validate_design_pair("b1", "b2")


def test_rtp7_design_and_kernel_source_contract() -> None:
    design_src = Path(__file__).with_name("design.py").read_text(encoding="utf-8")
    assert "rtp_ty = np.ndarray[(7," in design_src
    assert "initial_value=np.zeros(7, dtype=np.int32)" in design_src
    wait_idx = design_src.index("barrier.wait_for_value(1)")
    rows_idx = design_src.index("rows = rtp[4]")
    extra_idx = design_src.index("extra = rtp[5]")
    extra_slot_idx = design_src.index("extra_slot = rtp[6]")
    release_idx = design_src.index("barrier.release_with_value(1)")
    assert wait_idx < rows_idx < extra_idx < extra_slot_idx < release_idx
    assert "block(w, x, acc, sums, rows, mode, _round, _k, _slot)" in design_src
    assert "block(w, x, acc, sums, rows, mode, _e, _k, extra_slot)" in design_src
    assert "rtps[col][row][5] = 1 if shape.balanced else 0" in design_src
    assert "rtps[col][row][6] = col" in design_src

    kernel_src = Path(__file__).with_name("kernel.py").read_text(encoding="utf-8")
    assert (
        "[w, x, acc, sums, np.int32, np.int32, np.int32, np.int32, np.int32]"
        in kernel_src
    )
    assert "roles=(In, In, InOut, InOut, Param, Param, Param, Param, Param)" in kernel_src


def test_artifact_setup_uses_one_mode_for_xclbin_and_commands(monkeypatch) -> None:
    class FakeMLIR:
        def __init__(self, filename: str, mode: str) -> None:
            self.filename = filename
            self.mode = mode

    class FakeKernelArtifact:
        @staticmethod
        def from_extern(kernel):
            return ("artifact", kernel)

    class FakeArtifact:
        def __init__(self, filename, mlir_input, dependencies, **_kwargs) -> None:
            self.filename = filename
            self.mlir_input = mlir_input
            self.dependencies = dependencies

    monkeypatch.setattr(op_module, "NPU_AVAILABLE", True)
    monkeypatch.setattr(op_module, "KernelObjectArtifact", FakeKernelArtifact, raising=False)
    monkeypatch.setattr(op_module, "XclbinArtifact", FakeArtifact, raising=False)
    monkeypatch.setattr(op_module, "InstsBinArtifact", FakeArtifact, raising=False)
    monkeypatch.setattr(kernel_module, "grouped_kernel", lambda mode: ("kernel", mode))
    monkeypatch.setattr(
        GroupedMoE,
        "_mlir",
        lambda _self, filename, mode: FakeMLIR(filename, mode),
    )
    for shape in (B1, B2):
        op = GroupedMoE(shape.name)
        assert hasattr(op, "get_kernel_artifacts")
        assert hasattr(op, "get_mlir_artifact")
        op.set_up_artifacts()
        assert op.xclbin_artifact.mlir_input.mode == shape.name
        assert op.insts_artifact.mlir_input.mode == shape.name
        assert shape.name in op.xclbin_artifact.filename
        assert shape.name in op.insts_artifact.filename
        expected = ("artifact", ("kernel", shape.mode))
        assert expected in op.xclbin_artifact.dependencies
        assert expected in op.insts_artifact.dependencies


def test_device_text_excludes_only_runtime_sequence() -> None:
    mlir = """module {
  aie.device @npu2 {
    %0 = aie.objectfifo @w
    aiex.runtime_sequence(%arg0: memref<4xi32>) {
      scf.for %i = %c0 to %c4 step %c1 {
        aiex.npu.dma_memcpy_nd(%arg0)
      }
    }
    aie.core(%0) { aie.end }
  }
}
"""
    device = device_without_runtime_sequence(mlir)
    assert "runtime_sequence" not in device
    assert "objectfifo @w" in device
    assert "aie.core" in device
