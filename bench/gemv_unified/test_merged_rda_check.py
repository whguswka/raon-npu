# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Host-side unit tests for merged_rda_check result judging logic.

Tests execute in pure Python without requiring an NPU or IRON toolchain, verifying:
  1. Exception strings and mismatches record as FAIL and yield verdict FAIL.
  2. B2 COMPILE_ONLY is counted separately in counts['compile_only'] and never
     counted in counts['pass'].
  3. A single FAIL among required checks forces overall verdict to FAIL.
  4. All required checks PASS plus B2 COMPILE_ONLY produces verdict PASS.
  5. Exceptions in required separate comparison propagate instead of being skipped.
  6. Failure in separate comparison forces main() output to 'RESULT: FAIL' and exit 1.
  7. reset_xclbin_signature_cache is invoked before separate 8-col callable creation.
  8. cp949-encoded stdout completes verdict and summary logging without UnicodeEncodeError.
  9. B2 produces exactly one result key ('moe_b2_9') and counts as compile-only 1.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from bench.gemv_unified.merged_rda_check import (
    COMPILE_ONLY_STATUS,
    compare_separate_xclbins,
    configure_streams,
    judge_and_log_summary,
    judge_results,
    log,
    main,
    run_gemv_checks,
)


def test_exception_and_mismatch_record_fail_and_verdict_fail() -> None:
    results = {
        "dense_gu": "FAIL: RuntimeError('device stall')",
        "shape_1": "FAIL: mismatches=16",
    }
    verdict = judge_results(results)
    assert verdict["verdict"] == "FAIL"
    assert len(verdict["failures"]) == 2
    assert "dense_gu=FAIL: RuntimeError('device stall')" in verdict["failures"]
    assert "shape_1=FAIL: mismatches=16" in verdict["failures"]
    assert verdict["counts"]["pass"] == 0
    assert verdict["counts"]["compile_only"] == 0


def test_b2_compile_only_not_counted_in_pass_and_counted_separately() -> None:
    results = {"moe_b2_9": COMPILE_ONLY_STATUS}
    verdict = judge_results(results)
    assert verdict["counts"]["compile_only"] == 1
    assert verdict["counts"]["pass"] == 0
    assert len(verdict["failures"]) == 0
    assert verdict["verdict"] == "PASS"


def test_single_fail_among_required_checks_forces_verdict_fail() -> None:
    results = {
        "shape_1": "PASS",
        "shape_2": "PASS",
        "dense_gu": "FAIL: BrokenPipeError",
        "moe_b2_9": COMPILE_ONLY_STATUS,
    }
    verdict = judge_results(results)
    assert verdict["verdict"] == "FAIL"
    assert verdict["failures"] == ["dense_gu=FAIL: BrokenPipeError"]
    assert verdict["counts"]["pass"] == 2
    assert verdict["counts"]["compile_only"] == 1


def test_all_pass_plus_b2_compile_only_yields_verdict_pass() -> None:
    results = {
        "shape_1": "PASS",
        "shape_2": "PASS",
        "dense_gu": "PASS",
        "moe_b2_9": COMPILE_ONLY_STATUS,
    }
    verdict = judge_results(results)
    assert verdict["verdict"] == "PASS"
    assert verdict["counts"]["pass"] == 3
    assert verdict["counts"]["compile_only"] == 1
    assert len(verdict["failures"]) == 0


class DummyBO:
    def numpy(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)


class DummyRunner:
    def __init__(self) -> None:
        self.Y_bo = DummyBO()
        self.W_bo = DummyBO()
        self.X_bo = DummyBO()

    def invoke(self) -> Any:
        return None


def test_separate_comparison_exception_fails_and_propagates() -> None:
    """A failure in the required separate comparison must raise and not be skipped."""
    runner = DummyRunner()

    def failing_callable(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated NPU kernel execution stall")

    def dummy_rda(*args: Any, **kwargs: Any) -> Any:
        return None

    with pytest.raises(RuntimeError) as exc_info:
        compare_separate_xclbins(
            gemv_runner=runner,
            s_in=DummyBO(),
            s_out=DummyBO(),
            head_in=DummyBO(),
            out=DummyBO(),
            heads=40,
            iterations=5,
            context=None,
            sep_gemv_fn=failing_callable,
            rda_step_fn=dummy_rda,
        )

    msg = str(exc_info.value)
    assert "Separate XCLBINs comparison failed" in msg
    assert "simulated NPU kernel execution stall" in msg


def test_separate_comparison_exception_forces_main_verdict_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """When separate comparison fails, main() must output 'RESULT: FAIL <reason>' and exit 1."""
    import types

    ml_dtypes = types.ModuleType("ml_dtypes")
    ml_dtypes.bfloat16 = np.uint16
    monkeypatch.setitem(sys.modules, "ml_dtypes", ml_dtypes)

    aie = types.ModuleType("aie")
    aie_utils = types.ModuleType("aie.utils")
    aie_utils.zeros = lambda *args, **kwargs: DummyBO()
    aie_utils.zeros_like = lambda *args, **kwargs: DummyBO()
    aie.utils = aie_utils
    monkeypatch.setitem(sys.modules, "aie", aie)
    monkeypatch.setitem(sys.modules, "aie.utils", aie_utils)

    iron = types.ModuleType("iron")
    iron_common = types.ModuleType("iron.common")

    class MockAIEContext:
        pass

    iron_common.AIEContext = MockAIEContext
    iron.common = iron_common
    monkeypatch.setitem(sys.modules, "iron", iron)
    monkeypatch.setitem(sys.modules, "iron.common", iron_common)

    monkeypatch.setattr("sys.argv", ["merged_rda_check.py", "--iters", "1"])
    # Mock preceding steps so we reach alternation benchmark
    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.check_device_body_identity",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.run_gemv_checks",
        lambda **kwargs: {"shape_1": "PASS"},
    )
    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.run_rda_checks",
        lambda **kwargs: None,
    )

    class MockRdaOp:
        def compile(self) -> None:
            pass

        def get_callable(self) -> Any:
            return lambda *args, **kwargs: None

    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.MergedShapeRunner",
        lambda *args, **kwargs: DummyRunner(),
    )
    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.UnifiedRDA",
        lambda *args, **kwargs: MockRdaOp(),
    )

    class MockFailingSepOp:
        def compile(self) -> None:
            pass

        def get_callable(self) -> Any:
            def _failing_call(*args: Any, **kwargs: Any) -> Any:
                raise RuntimeError("timeout on 8-col kernel")

            return _failing_call

    class MockFailingSepGEMV:
        @classmethod
        def for_shape(cls, *args: Any, **kwargs: Any) -> MockFailingSepOp:
            return MockFailingSepOp()

    import kernels.gemv_unified.op as op_mod

    monkeypatch.setattr(op_mod, "UnifiedGEMV", MockFailingSepGEMV)

    # Allow standalone rda_step fallback if bench.rda_state not present
    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.rda_step",
        lambda *args, **kwargs: None,
    )

    rc = main()
    assert rc == 1
    captured = capsys.readouterr()
    assert "RESULT: FAIL Separate XCLBINs comparison failed: timeout on 8-col kernel" in captured.out


def test_reset_signature_cache_called_before_separate_callable_creation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset_xclbin_signature_cache must be called before creating the 8-col callable."""
    call_log: list[str] = []

    def mock_reset() -> None:
        call_log.append("reset_xclbin_signature_cache")

    class MockSepOp:
        def compile(self) -> None:
            call_log.append("SepGEMV.compile")

        def get_callable(self) -> Any:
            call_log.append("SepGEMV.get_callable")
            return lambda *args, **kwargs: None

    class MockSepGEMV:
        @classmethod
        def for_shape(cls, *args: Any, **kwargs: Any) -> MockSepOp:
            call_log.append("SepGEMV.for_shape")
            return MockSepOp()

    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.reset_xclbin_signature_cache",
        mock_reset,
    )
    import kernels.gemv_unified.op as op_mod

    monkeypatch.setattr(op_mod, "UnifiedGEMV", MockSepGEMV)

    runner = DummyRunner()

    compare_separate_xclbins(
        gemv_runner=runner,
        s_in=DummyBO(),
        s_out=DummyBO(),
        head_in=DummyBO(),
        out=DummyBO(),
        heads=40,
        iterations=2,
        context=None,
        sep_gemv_fn=None,  # Forces creation of callable
        rda_step_fn=lambda *args, **kwargs: None,
    )

    assert "reset_xclbin_signature_cache" in call_log
    assert "SepGEMV.get_callable" in call_log

    reset_idx = call_log.index("reset_xclbin_signature_cache")
    callable_idx = call_log.index("SepGEMV.get_callable")
    assert reset_idx < callable_idx, (
        f"reset_xclbin_signature_cache (idx {reset_idx}) must precede "
        f"SepGEMV.get_callable (idx {callable_idx})"
    )


def test_cp949_stdout_verdict_and_summary_output_without_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge and summary output must complete without UnicodeEncodeError on cp949 stdout."""
    import io

    # 1. Test with strict cp949 stream after configure_streams
    raw_strict = io.BytesIO()
    fake_strict = io.TextIOWrapper(raw_strict, encoding="cp949", errors="strict")
    monkeypatch.setattr(sys, "stdout", fake_strict)

    configure_streams()

    results = {
        "shape_1": "PASS",
        "dense_gu": "PASS",
        "moe_b2_9": COMPILE_ONLY_STATUS,
    }
    verdict = judge_and_log_summary(results)
    assert verdict["verdict"] == "PASS"

    # Also verify that the B2 skip line produces valid cp949 output
    log("  [COMPILE_ONLY -- 기능 판정은 엔진 골든에서] moe_b2_9 (K= 256, M= 2048, x_slots=7)")

    fake_strict.flush()
    content = raw_strict.getvalue().decode("cp949")
    assert "GEMV checks summary: 2 passed, 1 compile-only (engine golden required)" in content
    assert "기능 판정은 엔진 골든에서" in content

    # 2. Test with stream without reconfigure attribute (e.g. custom stream)
    class CustomCp949Writer:
        def __init__(self) -> None:
            self.buf = io.BytesIO()

        def write(self, s: str) -> int:
            b = s.encode("cp949")
            return self.buf.write(b)

        def flush(self) -> None:
            pass

    custom_out = CustomCp949Writer()
    monkeypatch.setattr(sys, "stdout", custom_out)
    # configure_streams must not fail when reconfigure is missing
    configure_streams()
    judge_and_log_summary(results)
    log("  [COMPILE_ONLY -- 기능 판정은 엔진 골든에서] moe_b2_9 (K= 256, M= 2048, x_slots=7)")
    custom_content = custom_out.buf.getvalue().decode("cp949")
    assert "GEMV checks summary: 2 passed, 1 compile-only (engine golden required)" in custom_content


def test_b2_single_compile_only_in_gemv_checks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """B2 must produce exactly one result key ('moe_b2_9') and count as compile-only 1."""
    import types

    class MockRunner:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def invoke(self) -> None:
            pass

        def verify(self) -> dict[str, Any]:
            return {"mismatches": 0}

    monkeypatch.setattr(
        "bench.gemv_unified.merged_rda_check.MergedShapeRunner", MockRunner
    )

    dummy_shape = types.SimpleNamespace(K=256, M=256, x_slots=1)
    monkeypatch.setattr(
        "bench.gemv_unified.stage2_bench._dense_case",
        lambda seed: (dummy_shape, np.zeros(1), np.zeros(1), np.zeros(1)),
    )
    import kernels.gemv_unified.op as op_mod

    class MockDenseOp:
        def compile(self) -> None:
            pass

        def get_callable(self) -> Any:
            return lambda *args, **kwargs: None

    monkeypatch.setattr(
        op_mod.UnifiedGEMV,
        "for_shape",
        classmethod(lambda cls, *args, **kwargs: MockDenseOp()),
    )
    monkeypatch.setattr("bench.moe_d.moe_d_bench._check", lambda *args, **kwargs: None)

    class MockTmpl:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def compile(self) -> None:
            pass

    monkeypatch.setattr("bench.gemv_unified.merged_rda_check.UnifiedMoEGateUp", MockTmpl)
    monkeypatch.setattr("bench.gemv_unified.merged_rda_check.UnifiedMoEDown9", MockTmpl)

    mock_aie = types.SimpleNamespace(
        full=lambda *args, **kwargs: DummyBO(),
        tensor=lambda *args, **kwargs: DummyBO(),
    )

    results = run_gemv_checks(
        cols=7,
        rda_col=7,
        continuous_mem_w=True,
        context=None,
        aie=mock_aie,
        bfloat16_dtype=np.uint16,
        seed=42,
    )

    b2_keys = [k for k in results if "moe_b2_9" in k]
    assert b2_keys == ["moe_b2_9"], f"Expected exactly one B2 result key, got {b2_keys}"

    verdict = judge_results(results)
    assert verdict["counts"]["compile_only"] == 1, (
        f"Expected compile-only count 1, got {verdict['counts']['compile_only']}"
    )

    captured = capsys.readouterr()
    assert "1 compile-only (engine golden required)" in captured.out
    assert "2 compile-only" not in captured.out

