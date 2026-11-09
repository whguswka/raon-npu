# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Verification that artifact filenames uniquely and deterministically map to generator inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from bench.gemv_unified.check_device_invariance import select_cases
import kernels.gemv_unified.op as unified_op
from kernels.gemv_unified.op import UnifiedGEMV, _shape_tag
from kernels.gemv_unified.shapes import ALL_SHAPES, UnifiedShape


@dataclass
class CapturedGeneratorInvocation:
    design_path: str
    func_name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def normalized_signature(self) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]:
        norm_args = tuple(
            str(a) if not isinstance(a, (int, bool, type(None), str)) else a
            for a in self.args
        )
        norm_kwargs = tuple(
            sorted(
                (
                    k,
                    str(v) if not isinstance(v, (int, bool, type(None), str)) else v,
                )
                for k, v in self.kwargs.items()
            )
        )
        return (norm_args, norm_kwargs)


class CapturedMLIRArtifact:
    def __init__(self, filename: str, generator: Any):
        self.filename = filename
        self.generator = generator


class CapturedXclbinArtifact:
    def __init__(
        self,
        filename: str,
        mlir_input: Any = None,
        dependencies: list[Any] | None = None,
        kernel_name: str | None = None,
        extra_flags: list[str] | None = None,
    ):
        self.filename = filename
        self.mlir_input = mlir_input
        self.dependencies = dependencies or []
        self.kernel_name = kernel_name
        self.extra_flags = extra_flags or []


class CapturedInstsBinArtifact:
    def __init__(
        self,
        filename: str,
        mlir_input: Any = None,
        dependencies: list[Any] | None = None,
        extra_flags: list[str] | None = None,
    ):
        self.filename = filename
        self.mlir_input = mlir_input
        self.dependencies = dependencies or []
        self.extra_flags = extra_flags or []


def build_test_operators() -> list[UnifiedGEMV]:
    """Build the comprehensive test matrix (155 operator instances).

    1. 19 shapes from select_cases(None, None): 13 ALL_SHAPES + 6 boundary shapes
       x 5 execution modes = 95 combinations.
    2. 12 experimental shape variations for (256, 1024) spanning:
       active_cols (1, 8), x_slots (1, 8), format (0, 1), chunks_per_col (None, 8//active_cols)
       x 5 execution modes = 60 combinations.
    Total = 155 operator combinations.
    """
    cases = select_cases(None, None)

    modes = [
        {},  # 기본 (Default)
        {"continuous_mem_w": True},
        {"diagnostic_small_x": True},
        {"diagnostic_receive_only": True},
        {"diagnostic_static_core": True},
    ]

    operators: list[UnifiedGEMV] = []

    # 1. 19 cases x 5 modes = 95 operators
    for case in cases:
        for mode in modes:
            op = UnifiedGEMV(
                K=case.shape.K,
                M=case.shape.M,
                format=case.shape.format,
                active_cols=case.shape.active_cols,
                x_slots=case.shape.x_slots,
                chunks_per_col=case.shape.chunks_per_col,
                experimental_shape=case.experimental,
                context=object(),
                **mode,
            )
            operators.append(op)

    # 2. 12 experimental variations for (256, 1024) x 5 modes = 60 operators
    for fmt in (0, 1):
        for ac, xs in [(8, 1), (8, 8), (1, 1)]:
            for cpc in [None, 8 // ac]:
                for mode in modes:
                    op = UnifiedGEMV(
                        K=256,
                        M=1024,
                        format=fmt,
                        active_cols=ac,
                        x_slots=xs,
                        chunks_per_col=cpc,
                        experimental_shape=True,
                        context=object(),
                        **mode,
                    )
                    operators.append(op)

    return operators


def test_artifact_names_are_deterministic_functions_of_generator_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enforce that artifact filenames uniquely identify their generator inputs."""
    captured_mlir_artifacts: list[CapturedMLIRArtifact] = []

    def fake_design_generator(design_path, func_name, args, kwargs):
        return CapturedGeneratorInvocation(
            design_path=str(design_path),
            func_name=str(func_name),
            args=args,
            kwargs=dict(kwargs),
        )

    def fake_mlir_artifact(filename, generator):
        art = CapturedMLIRArtifact(filename, generator)
        captured_mlir_artifacts.append(art)
        return art

    monkeypatch.setattr(unified_op, "DesignGenerator", fake_design_generator)
    monkeypatch.setattr(unified_op, "PythonGeneratedMLIRArtifact", fake_mlir_artifact)
    monkeypatch.setattr(unified_op, "XclbinArtifact", CapturedXclbinArtifact)
    monkeypatch.setattr(unified_op, "InstsBinArtifact", CapturedInstsBinArtifact)
    monkeypatch.setattr(unified_op.UnifiedGEMV, "get_kernel_artifacts", lambda self: [])
    monkeypatch.setattr(
        unified_op.UnifiedGEMV,
        "_kernel",
        lambda self: f"fake_kernel_K_{self.K if self.diagnostic_small_x else None}",
    )
    monkeypatch.setattr(
        unified_op.UnifiedGEMV, "add_artifacts", lambda self, arts: None, raising=False
    )

    ops = build_test_operators()
    assert len(ops) == 155

    captured_xclbins: list[CapturedXclbinArtifact] = []
    captured_insts: list[CapturedInstsBinArtifact] = []

    for op in ops:
        op.set_up_artifacts()
        captured_xclbins.append(op.xclbin_artifact)
        captured_insts.append(op.insts_artifact)

    # 1. Check: If MLIR filename is identical, generator inputs must be bit-identical.
    mlir_by_filename: dict[str, tuple[Any, ...]] = {}
    for art in captured_mlir_artifacts:
        sig = art.generator.normalized_signature()
        if art.filename in mlir_by_filename:
            existing = mlir_by_filename[art.filename]
            assert existing == sig, (
                f"MLIR filename '{art.filename}' collision with conflicting generator inputs:\n"
                f"  Existing: {existing}\n"
                f"  Incoming: {sig}"
            )
        else:
            mlir_by_filename[art.filename] = sig

    # 2. Check: If xclbin filename is identical, input config MLIR must be identical.
    xclbin_by_filename: dict[str, str] = {}
    for art in captured_xclbins:
        input_mlir = art.mlir_input.filename
        if art.filename in xclbin_by_filename:
            existing = xclbin_by_filename[art.filename]
            assert existing == input_mlir, (
                f"xclbin filename '{art.filename}' collision with conflicting input MLIR:\n"
                f"  Existing: {existing}\n"
                f"  Incoming: {input_mlir}"
            )
        else:
            xclbin_by_filename[art.filename] = input_mlir

    # 3. Check: If insts.bin filename is identical, input shape MLIR must be identical.
    insts_by_filename: dict[str, str] = {}
    for art in captured_insts:
        input_mlir = art.mlir_input.filename
        if art.filename in insts_by_filename:
            existing = insts_by_filename[art.filename]
            assert existing == input_mlir, (
                f"insts filename '{art.filename}' collision with conflicting input MLIR:\n"
                f"  Existing: {existing}\n"
                f"  Incoming: {input_mlir}"
            )
        else:
            insts_by_filename[art.filename] = input_mlir

    # 4. Check product shapes preserve shared config xclbin name
    for op in ops:
        if not op.experimental_shape and not (
            op.diagnostic_receive_only
            or op.diagnostic_static_core
            or op.diagnostic_small_x
        ):
            expected_xclbin = (
                "Raon_GEMV_Unified_c8_npu2_contw.xclbin"
                if op.continuous_mem_w
                else "Raon_GEMV_Unified_c8_npu2.xclbin"
            )
            assert op.xclbin_artifact.filename == expected_xclbin


def test_shape_tag_distinguishes_all_six_fields() -> None:
    """Verify that different shape fields produce mutually distinct tags and insts names."""
    base = UnifiedShape("base", 256, 1024, format=0, active_cols=8, x_slots=1, chunks_per_col=None)
    diff_cols = UnifiedShape("diff_cols", 256, 1024, format=0, active_cols=1, x_slots=1, chunks_per_col=None)
    diff_slots = UnifiedShape("diff_slots", 256, 1024, format=0, active_cols=8, x_slots=8, chunks_per_col=None)
    diff_chunks = UnifiedShape("diff_chunks", 256, 1024, format=0, active_cols=8, x_slots=1, chunks_per_col=1)
    diff_fmt = UnifiedShape("diff_fmt", 256, 1024, format=1, active_cols=8, x_slots=1, chunks_per_col=None)
    diff_k = UnifiedShape("diff_k", 512, 1024, format=0, active_cols=8, x_slots=1, chunks_per_col=None)
    diff_m = UnifiedShape("diff_m", 256, 2048, format=0, active_cols=8, x_slots=1, chunks_per_col=None)

    shapes = [base, diff_cols, diff_slots, diff_chunks, diff_fmt, diff_k, diff_m]
    tags = [_shape_tag(s) for s in shapes]
    assert len(set(tags)) == len(shapes) == 7

    assert _shape_tag(base) == "f0_k256_m1024_a8_x1_cauto"
    assert _shape_tag(diff_cols) == "f0_k256_m1024_a1_x1_cauto"
    assert _shape_tag(diff_slots) == "f0_k256_m1024_a8_x8_cauto"
    assert _shape_tag(diff_chunks) == "f0_k256_m1024_a8_x1_c1"
    assert _shape_tag(diff_fmt) == "f1_k256_m1024_a8_x1_cauto"


def test_diagnostic_static_core_artifacts_differ_by_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify that static-core diagnostic builds for f0 and f1 have different xclbin and config MLIR names."""
    monkeypatch.setattr(unified_op, "DesignGenerator", lambda *a, **k: None)
    monkeypatch.setattr(
        unified_op,
        "PythonGeneratedMLIRArtifact",
        lambda filename, gen: CapturedMLIRArtifact(filename, gen),
    )
    monkeypatch.setattr(unified_op, "XclbinArtifact", CapturedXclbinArtifact)
    monkeypatch.setattr(unified_op, "InstsBinArtifact", CapturedInstsBinArtifact)
    monkeypatch.setattr(unified_op.UnifiedGEMV, "get_kernel_artifacts", lambda self: [])
    monkeypatch.setattr(unified_op.UnifiedGEMV, "_kernel", lambda self: "fake_kernel")
    monkeypatch.setattr(
        unified_op.UnifiedGEMV, "add_artifacts", lambda self, arts: None, raising=False
    )

    op_f0 = UnifiedGEMV(
        K=256, M=1024, format=0, diagnostic_static_core=True, context=object()
    )
    op_f1 = UnifiedGEMV(
        K=256, M=1024, format=1, diagnostic_static_core=True, context=object()
    )

    op_f0.set_up_artifacts()
    op_f1.set_up_artifacts()

    assert op_f0.config_name != op_f1.config_name
    assert op_f0.xclbin_artifact.filename != op_f1.xclbin_artifact.filename
    assert op_f0.xclbin_artifact.mlir_input.filename != op_f1.xclbin_artifact.mlir_input.filename
    assert "f0" in op_f0.xclbin_artifact.filename
    assert "f1" in op_f1.xclbin_artifact.filename


def test_product_13_shapes_generator_inputs_and_insts_names() -> None:
    """Verify that the 13 product shapes have uniquely resolved insts names."""
    ops = [UnifiedGEMV.for_shape(s, context=object()) for s in ALL_SHAPES]
    insts_names = [f"{op.name}.bin" for op in ops]
    assert len(set(insts_names)) == len(ALL_SHAPES) == 13

    # Every product shape shares the canonical xclbin
    for op in ops:
        assert op.config_name == "Raon_GEMV_Unified_c8_npu2"

