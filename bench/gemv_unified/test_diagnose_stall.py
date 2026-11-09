# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-only contract checks for the bounded stall diagnosis matrix."""

from bench.gemv_unified import diagnose_stall
from bench.gemv_unified.diagnose_stall import PROBES, make_probes, parser
from kernels.gemv_unified.op import UnifiedGEMV


def test_diagnosis_matrix_isolates_each_unified_design_dimension() -> None:
    by_name = {probe.name: probe for probe in PROBES}
    assert not by_name["baseline"].receive_only
    assert by_name["receive_only"].receive_only
    assert by_name["static_core"].static_core
    assert by_name["small_x"].small_x
    assert by_name["one_col"].active_cols == 1
    assert by_name["continuous_w"].continuous_mem_w

    minimal = by_name["minimal_dynamic"]
    assert minimal.receive_only and not minimal.static_core and minimal.small_x
    assert minimal.active_cols == 1
    assert "minimal_static" not in by_name
    assert "minimal_static_continuous" not in by_name


def test_probe_dimensions_use_cli_defaults() -> None:
    baseline = make_probes()[0]
    assert baseline.k == 512
    assert baseline.m == 5120
    one_col = make_probes(1024, 256, 4096)[4]
    assert one_col.name == "one_col"
    assert one_col.k == 1024
    assert one_col.m == 256


def test_parser_accepts_k_m_and_case_filter() -> None:
    args = parser().parse_args(["--k", "1024", "--m-one-col", "256", "--cases", "baseline,small_x"])
    assert args.k == 1024
    assert args.m_one_col == 256
    assert args.m_all_cols == 5120
    assert args.cases == ["baseline", "small_x"]


def test_child_path_selects_probe_by_name(monkeypatch) -> None:
    selected = []

    def fake_run_child(probe, *, prepare_only, timeout):
        selected.append((probe, prepare_only, timeout))
        return 17

    monkeypatch.setattr(diagnose_stall, "run_child", fake_run_child)
    result = diagnose_stall.main(
        ["--child", "one_col", "--k", "1024", "--m-one-col", "256"]
    )
    assert result == 17
    probe, prepare_only, timeout = selected[0]
    assert probe.name == "one_col"
    assert probe.k == 1024
    assert probe.m == 256
    assert prepare_only is False
    assert timeout == 120.0


def test_diagnostic_config_name_includes_shape() -> None:
    op = UnifiedGEMV(
        K=512,
        M=128,
        active_cols=1,
        diagnostic_receive_only=True,
        diagnostic_static_core=True,
        diagnostic_small_x=True,
        context=object(),
    )
    name = op.config_name
    assert "f0" in name
    assert "k512" in name
    assert "m128" in name
    assert "a1" in name
    assert "x1" in name
    assert "cauto" in name
    assert "recv" in name
    assert "static" in name
    assert "smallx" in name


def test_diagnostic_static_core_config_name_differs_by_format() -> None:
    op_f0 = UnifiedGEMV(
        K=256,
        M=1024,
        format=0,
        diagnostic_static_core=True,
        context=object(),
    )
    op_f1 = UnifiedGEMV(
        K=256,
        M=1024,
        format=1,
        diagnostic_static_core=True,
        context=object(),
    )
    assert op_f0.config_name != op_f1.config_name
    assert "f0" in op_f0.config_name
    assert "f1" in op_f1.config_name


def test_non_diagnostic_config_name_stays_shape_independent() -> None:
    op = UnifiedGEMV(K=2560, M=5120, active_cols=8, context=object())
    assert op.config_name == "Raon_GEMV_Unified_c8_npu2"


def test_log_paths_follow_case_and_phase() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        import os

        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            # run_matrix creates diag-logs and writes <case>-{prepare,run}.log.
            path = diagnose_stall.Path("diag-logs") / "baseline-run.log"
            assert str(path) == "diag-logs/baseline-run.log"
        finally:
            os.chdir(old_cwd)


def test_small_x_kernel_binding_uses_k_in_mlir_artifact(monkeypatch) -> None:
    """_kernel() builds the gemv_kernel fed to _mlir_artifact with K only for small_x."""
    captured: dict[str, object] = {}

    def fake_unified_bfp16_kernel(*, K: int | None = None) -> object:
        captured["K"] = K
        return object()

    monkeypatch.setattr(
        "kernels.gemv_unified.op.unified_bfp16_kernel", fake_unified_bfp16_kernel
    )

    small_x_op = UnifiedGEMV(
        K=512,
        M=128,
        active_cols=1,
        diagnostic_small_x=True,
        context=object(),
    )
    small_x_op._kernel()
    assert captured.get("K") == 512

    captured.clear()
    normal_op = UnifiedGEMV(
        K=512,
        M=128,
        active_cols=1,
        diagnostic_small_x=False,
        experimental_shape=True,
        context=object(),
    )
    normal_op._kernel()
    assert captured.get("K") is None
