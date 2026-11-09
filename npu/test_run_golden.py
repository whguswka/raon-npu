# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Focused exit-gate tests for the NPU golden runner."""

import ast
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import numpy as np
import pytest

from host.check_golden import (
    evaluate_summary,
    print_evaluation,
    validate_generated_token_ids,
)
from npu.run_golden import (
    GOLDEN_ASSESSMENT_NUMERIC_FIELDS,
    MAX_GOLDEN_STEPS,
    _assess_baseline,
    _assess_verification,
    _build_parser,
    _latency_seconds_per_token,
    _parse_args,
    _summarize_metrics,
    normalize_payload_finite,
    run_golden_verification,
)


def test_none_average_latency_serializes_as_json_null():
    payload = {"avg_latency_s_per_token": _latency_seconds_per_token(None)}
    assert json.loads(json.dumps(payload)) == {"avg_latency_s_per_token": None}


def test_invalid_gemv_environment_fails_during_argument_parsing(monkeypatch):
    monkeypatch.setenv("RAON_NPU_GEMV", "invalid")
    with pytest.raises(SystemExit) as exc_info:
        _build_parser().parse_args([])
    assert exc_info.value.code == 2


def _record(step: int, cosine: float, tie: bool = True) -> dict:
    return {
        "step": step,
        "logit_cosine": cosine,
        "top1_tie_match": tie,
    }


def test_golden_assessment_rejects_incomplete_requested_range():
    result = _assess_verification(
        start_step=2,
        end_step=4,
        step_records=[_record(2, 1.0), _record(3, 1.0)],
        stop_reason={"code": "missing_snapshot"},
        full_25_layers=True,
        baseline_passed=True,
        min_cos=0.999,
        min_top1_tie_rate=0.9,
    )

    assert result["completed_steps"] == [2, 3]
    assert result["run_complete"] is False
    assert result["verification_passed"] is False


def test_golden_assessment_requires_every_step_cosine_threshold():
    result = _assess_verification(
        start_step=2,
        end_step=3,
        step_records=[_record(2, 0.9995), _record(3, 0.9989)],
        stop_reason={"code": "completed"},
        full_25_layers=True,
        baseline_passed=True,
        min_cos=0.999,
        min_top1_tie_rate=0.5,
    )

    assert result["run_complete"] is True
    assert result["min_logit_cosine"] == 0.9989
    assert result["golden_accuracy_passed"] is False
    assert result["verification_passed"] is False


def test_golden_assessment_applies_configurable_tie_rate():
    records = [_record(step, 0.9995, tie=step != 4) for step in range(2, 12)]

    passing = _assess_verification(
        start_step=2,
        end_step=11,
        step_records=records,
        stop_reason={"code": "completed"},
        full_25_layers=True,
        baseline_passed=True,
        min_cos=0.999,
        min_top1_tie_rate=0.9,
    )
    failing = _assess_verification(
        start_step=2,
        end_step=11,
        step_records=records,
        stop_reason={"code": "completed"},
        full_25_layers=True,
        baseline_passed=True,
        min_cos=0.999,
        min_top1_tie_rate=0.91,
    )

    assert passing["top1_tie_match_rate"] == 0.9
    assert passing["verification_passed"] is True
    assert failing["verification_passed"] is False


def test_golden_assessment_rejects_partial_layer_run_without_logit_metrics():
    result = _assess_verification(
        start_step=2,
        end_step=2,
        step_records=[{"step": 2}],
        stop_reason={"code": "completed"},
        full_25_layers=False,
        baseline_passed=True,
        min_cos=0.999,
        min_top1_tie_rate=0.9,
    )

    assert result["run_complete"] is True
    assert result["golden_metrics_complete"] is False
    assert result["golden_accuracy_passed"] is False
    assert result["verification_passed"] is False


def test_golden_assessment_rejects_non_finite_metrics_at_any_step_position():
    """15단계 완주 기록에서 처음·중간·끝 위치마다, 판정이 쓰는 숫자 칸마다 nan·inf·-inf 주입 시 실패 확인."""
    steps = list(range(2, 17))
    assert len(steps) == 15

    positions = [0, len(steps) // 2, len(steps) - 1]
    bad_values = [float("nan"), float("inf"), float("-inf")]

    assert len(GOLDEN_ASSESSMENT_NUMERIC_FIELDS) > 0

    for pos in positions:
        for field in GOLDEN_ASSESSMENT_NUMERIC_FIELDS:
            for bad_val in bad_values:
                records = [_record(s, 0.9995, tie=True) for s in steps]
                records[pos][field] = bad_val

                result = _assess_verification(
                    start_step=2,
                    end_step=16,
                    step_records=records,
                    stop_reason={"code": "completed"},
                    full_25_layers=True,
                    baseline_passed=True,
                    min_cos=0.999,
                    min_top1_tie_rate=0.9,
                )

                assert result["golden_metrics_complete"] is False, (
                    f"Expected golden_metrics_complete=False for {field}={bad_val} at step {records[pos]['step']}"
                )
                assert result["golden_accuracy_passed"] is False, (
                    f"Expected golden_accuracy_passed=False for {field}={bad_val} at step {records[pos]['step']}"
                )
                assert result["verification_passed"] is False, (
                    f"Expected verification_passed=False for {field}={bad_val} at step {records[pos]['step']}"
                )


def test_baseline_assessment_rejects_non_finite_at_any_layer_position():
    """처음·중간·끝 층 위치 × 측정 칸(streams/mixer/logits) × nan·inf·-inf 조합 주입 시 baseline 실패 및 JSON 확인."""
    num_layers = 25
    positions = [0, num_layers // 2, num_layers - 1]
    bad_values = [float("nan"), float("inf"), float("-inf")]

    # 1. streams_cosines에 비유한 값 주입
    for pos in positions:
        for bad_val in bad_values:
            streams = [0.9995] * num_layers
            streams[pos] = bad_val
            mixer = [0.9995] * num_layers
            logits = 0.9995

            res = _assess_baseline(streams, mixer, logits)
            assert res["pass_0999"] is False
            assert res["baseline_passed"] is False
            assert res["status"] == "FAIL"
            assert res["min_layer_streams_cos"] is None
            assert res.get("min_layer_streams_cos_non_finite") is True

            verif = _assess_verification(
                start_step=2,
                end_step=16,
                step_records=[_record(s, 0.9995) for s in range(2, 17)],
                stop_reason={"code": "completed"},
                full_25_layers=True,
                baseline_passed=res["baseline_passed"],
                min_cos=0.999,
                min_top1_tie_rate=0.9,
            )
            assert verif["verification_passed"] is False

            # JSON 직렬화 시 NaN 문자열이 없어야 함
            dumped = json.dumps(res)
            assert "NaN" not in dumped
            assert "null" in dumped

    # 2. mixer_cosines에 비유한 값 주입
    for pos in positions:
        for bad_val in bad_values:
            streams = [0.9995] * num_layers
            mixer = [0.9995] * num_layers
            mixer[pos] = bad_val
            logits = 0.9995

            res = _assess_baseline(streams, mixer, logits)
            assert res["pass_0999"] is False
            assert res["baseline_passed"] is False
            assert res["status"] == "FAIL"
            assert res["min_layer_mixer_cos"] is None
            assert res.get("min_layer_mixer_cos_non_finite") is True

            verif = _assess_verification(
                start_step=2,
                end_step=16,
                step_records=[_record(s, 0.9995) for s in range(2, 17)],
                stop_reason={"code": "completed"},
                full_25_layers=True,
                baseline_passed=res["baseline_passed"],
                min_cos=0.999,
                min_top1_tie_rate=0.9,
            )
            assert verif["verification_passed"] is False

            dumped = json.dumps(res)
            assert "NaN" not in dumped
            assert "null" in dumped

    # 3. cos_logits에 비유한 값 주입
    for bad_val in bad_values:
        streams = [0.9995] * num_layers
        mixer = [0.9995] * num_layers

        res = _assess_baseline(streams, mixer, bad_val)
        assert res["pass_0999"] is False
        assert res["baseline_passed"] is False
        assert res["status"] == "FAIL"
        assert res["logits_cos"] is None
        assert res.get("logits_cos_non_finite") is True

        verif = _assess_verification(
            start_step=2,
            end_step=16,
            step_records=[_record(s, 0.9995) for s in range(2, 17)],
            stop_reason={"code": "completed"},
            full_25_layers=True,
            baseline_passed=res["baseline_passed"],
            min_cos=0.999,
            min_top1_tie_rate=0.9,
        )
        assert verif["verification_passed"] is False

        dumped = json.dumps(res)
        assert "NaN" not in dumped
        assert "null" in dumped


def test_summarize_metrics_rejects_non_finite_at_any_layer_position():
    """요약 함수도 처음·중간·끝 층 위치에 비유한 값이 들어오면 None 및 non_finite 플래그 처리 검증."""
    num_layers = 25
    positions = [0, num_layers // 2, num_layers - 1]
    bad_values = [float("nan"), float("inf"), float("-inf")]

    for pos in positions:
        for bad_val in bad_values:
            layers_streams = [{"streams_out": 0.9995, "mixer_out": 0.9995, "router_overlap": 6} for _ in range(num_layers)]
            layers_streams[pos]["streams_out"] = bad_val
            records = [{"step": 2, "layers": layers_streams}]

            summary = _summarize_metrics(records)
            assert summary["min_streams_cos"] is None
            assert summary.get("min_streams_cos_non_finite") is True

            dumped = json.dumps(summary)
            assert "NaN" not in dumped
            assert "null" in dumped

            layers_mixer = [{"streams_out": 0.9995, "mixer_out": 0.9995, "router_overlap": 6} for _ in range(num_layers)]
            layers_mixer[pos]["mixer_out"] = bad_val
            records = [{"step": 2, "layers": layers_mixer}]

            summary = _summarize_metrics(records)
            assert summary["min_mixer_cos"] is None
            assert summary.get("min_mixer_cos_non_finite") is True

            dumped = json.dumps(summary)
            assert "NaN" not in dumped
            assert "null" in dumped


def test_run_golden_ast_min_max_usage_rule():
    """ast로 npu/run_golden.py를 읽어 내장 min/max(단일 반복가능 인자) 호출이 도우미 안에만 있음을 단언."""
    golden_py = Path(__file__).resolve().parent / "run_golden.py"
    tree = ast.parse(golden_py.read_text(encoding="utf-8"))

    violations: list[str] = []

    class MinMaxVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None

        def visit_FunctionDef(self, node):
            old = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = old

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
                fn_name = node.func.id
                # 도우미 함수 내부 호출은 허용
                if self.current_function in {"_finite_min", "_run_npu_step_with_timeout"}:
                    self.generic_visit(node)
                    return
                # 인자가 2개 이상이고 그 중 하나가 상수 0 또는 0.0인 clipping 형태(예: max(0.0, x))는 허용
                if len(node.args) >= 2:
                    has_zero_const = any(
                        isinstance(arg, ast.Constant) and arg.value in (0, 0.0)
                        for arg in node.args
                    )
                    if has_zero_const:
                        self.generic_visit(node)
                        return
                violations.append(
                    f"Line {node.lineno}: call to {fn_name}() in function {self.current_function!r} "
                    f"with {len(node.args)} arguments without using helper"
                )
            self.generic_visit(node)

    visitor = MinMaxVisitor()
    visitor.visit(tree)

    assert not violations, "Rule violation in run_golden.py:\n" + "\n".join(violations)


def test_run_golden_cli_numeric_arguments_use_ranged_parser():
    """모든 CLI 동작(_actions) 중 숫자 기본값 또는 int/float 타입인 동작은 모두 범위를 가진 파서여야 한다 (1b)."""
    parser = _build_parser()
    for action in parser._actions:
        if not action.option_strings:
            continue
        opt_name = action.option_strings[0]
        is_num_default = isinstance(action.default, (int, float)) and not isinstance(action.default, bool)
        is_num_type = action.type in (int, float)

        if is_num_default or is_num_type:
            assert getattr(action.type, "is_ranged", False) is True, (
                f"Action {opt_name} must use a ranged parser, got type={action.type!r}, default={action.default!r}"
            )
            assert hasattr(action.type, "min_val") and action.type.min_val is not None, (
                f"Action {opt_name} ranged parser min_val must not be None"
            )
            assert hasattr(action.type, "max_val") and action.type.max_val is not None, (
                f"Action {opt_name} ranged parser max_val must not be None"
            )


def test_run_golden_cli_ranged_parser_boundary_and_non_finite():
    """모든 ranged parser에 nan, inf, -inf 및 범위 밖 값을 주면 SystemExit(2) 검증 (1c)."""
    parser = _build_parser()
    for action in parser._actions:
        if not getattr(action.type, "is_ranged", False):
            continue
        opt = action.option_strings[0]
        ptype = action.type

        # 1. nan, inf, -inf
        for bad_str in ("nan", "inf", "-inf", "NaN", "Infinity", "-Infinity"):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, bad_str])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={bad_str}"

        # 2. 범위 밖 값 (위 한계 바로 위, 아래 한계 바로 아래)
        min_v = ptype.min_val
        max_v = ptype.max_val
        type_cls = getattr(ptype, "type_cls", float)

        if min_v is not None:
            if getattr(ptype, "min_inclusive", True):
                below = min_v - 1 if type_cls is int else min_v - 0.001
            else:
                below = min_v
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, str(below)])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={below} (below min {min_v})"

        if max_v is not None:
            if getattr(ptype, "max_inclusive", True):
                above = max_v + 1 if type_cls is int else max_v + 0.001
            else:
                above = max_v
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, str(above)])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={above} (above max {max_v})"

        # 2-b. 거대 수 10**400, -10**400, 1e308, -1e308
        huge_int_str = "1" + "0" * 400
        neg_huge_int_str = "-1" + "0" * 400
        for huge_str in (huge_int_str, neg_huge_int_str, "1e308", "-1e308"):
            with pytest.raises(SystemExit) as exc_info:
                parser.parse_args([opt, huge_str])
            assert exc_info.value.code == 2, f"Expected code 2 for {opt}={huge_str}"

        # 3. 유효한 기본값은 통과
        if action.default is not None:
            args = parser.parse_args([opt, str(action.default)])
            dest = action.dest
            assert getattr(args, dest) == action.default


def test_run_golden_step_range_limit():
    """단언 1e: 골든 end_step 검증 및 start_step > end_step 거부."""
    # 1. 상한 경계 내는 통과
    args = _parse_args(["--start-step", "2", "--end-step", str(MAX_GOLDEN_STEPS)])
    assert args.start_step == 2
    assert args.end_step == MAX_GOLDEN_STEPS

    # 2. end_step 상한 초과(4097)는 ranged_int에서 SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["--start-step", "2", "--end-step", str(MAX_GOLDEN_STEPS + 1)])
    assert exc_info.value.code == 2

    # 3. start_step > end_step 검사도 SystemExit(2)
    with pytest.raises(SystemExit) as exc_info:
        _parse_args(["--start-step", "2", "--end-step", "1"])
    assert exc_info.value.code == 2


def test_golden_summary_payload_finite_serialization_and_non_finite_paths():
    """단언 2a: 단계(처음·중간·끝) × 층(처음·중간·끝) × 측정 칸에 nan·inf·-inf를 넣은 완주 기록으로
    요약 payload를 만들어 실제로 직렬화한다.
    그 글자를 json.loads(text, parse_constant=…)(상수를 만나면 예외)로 읽어 비유한 글자가 없음을 확인한다.
    non_finite_paths에 넣은 자리가 모두 있어야 한다.
    """
    def _make_full_step_records(num_steps: int = 15, num_layers: int = 25) -> list[dict]:
        records = []
        for s_idx in range(num_steps):
            step_num = s_idx + 2
            layer_list = []
            for l_idx in range(num_layers):
                layer_list.append({
                    "layer": l_idx,
                    "streams_out": 0.9995,
                    "mixer_out": 0.9995,
                    "router_overlap": 6.0,
                })
            records.append({
                "step": step_num,
                "layers": layer_list,
                "logit_cosine": 0.9995,
                "top1_match": True,
                "top1_tie_match": True,
                "latency_total_ms": 100.0,
                "latency_pure_npu_ms": 50.0,
            })
        return records

    def _build_test_summary_payload(records: list[dict], num_layers: int = 25) -> dict[str, Any]:
        assessment = _assess_verification(
            start_step=records[0]["step"],
            end_step=records[-1]["step"],
            step_records=records,
            stop_reason={"code": "completed"},
            full_25_layers=(num_layers == 25),
            baseline_passed=True,
            min_cos=0.999,
            min_top1_tie_rate=0.9,
        )
        summary_metrics = _summarize_metrics(records)
        lat_total = [r["latency_total_ms"] for r in records if "latency_total_ms" in r]
        avg_total_ms = (
            None if (lat_total and any(not math.isfinite(x) for x in lat_total))
            else (float(np.mean(lat_total)) if lat_total else 0.0)
        )
        lat_npu = [r["latency_pure_npu_ms"] for r in records if "latency_pure_npu_ms" in r]
        avg_npu_ms = (
            None if (lat_npu and any(not math.isfinite(x) for x in lat_npu))
            else (float(np.mean(lat_npu)) if lat_npu else 0.0)
        )
        logit_cos_list = [r["logit_cosine"] for r in records if "logit_cosine" in r]
        avg_cos = (
            None if (logit_cos_list and any(not math.isfinite(x) for x in logit_cos_list))
            else (float(np.mean(logit_cos_list)) if logit_cos_list else 0.0)
        )

        payload = {
            "timestamp": "2026-10-30 12:00:00",
            "weights_d_dir": "/tmp/weights",
            "golden_dir": "/tmp/golden",
            "num_layers": num_layers,
            "steps_completed": len(records),
            "min_streams_cos": summary_metrics["min_streams_cos"],
            "min_mixer_cos": summary_metrics["min_mixer_cos"],
            "avg_router_overlap": summary_metrics["avg_router_overlap"],
            "avg_latency_ms": avg_total_ms,
            "avg_npu_latency_ms": avg_npu_ms,
            "preload": {},
            "baseline_passed": True,
            "host_numpy_parity": None,
            "category_timing_s": {},
            "category_calls": {},
            "step_records": records,
            "stop_reason": {"code": "completed", "step": records[-1]["step"], "message": "done"},
            "avg_logit_cosine": avg_cos,
            **assessment,
        }
        return normalize_payload_finite(payload)

    def _no_constants(val: str) -> None:
        raise ValueError(f"Encountered non-standard JSON constant: {val!r}")

    num_steps = 15
    num_layers = 25
    step_positions = [0, num_steps // 2, num_steps - 1]  # 0, 7, 14 (처음, 중간, 끝)
    layer_positions = [0, num_layers // 2, num_layers - 1]  # 0, 12, 24 (처음, 중간, 끝)
    bad_values = [float("nan"), float("inf"), float("-inf")]

    # 1. 층별 측정 칸 (streams_out, mixer_out) 주입 검증
    for step_pos in step_positions:
        for layer_pos in layer_positions:
            for field in ("streams_out", "mixer_out"):
                for bad_val in bad_values:
                    records = _make_full_step_records(num_steps, num_layers)
                    records[step_pos]["layers"][layer_pos][field] = bad_val

                    payload = _build_test_summary_payload(records, num_layers)
                    expected_path = f"step_records[{step_pos}].layers[{layer_pos}].{field}"

                    assert "non_finite_paths" in payload, f"Expected non_finite_paths for {expected_path}"
                    assert expected_path in payload["non_finite_paths"], (
                        f"Expected {expected_path} in {payload['non_finite_paths']}"
                    )

                    # 직렬화 검증: allow_nan=False로 안전하게 직렬화되어야 함
                    serialized = json.dumps(payload, indent=2, allow_nan=False)

                    # parse_constant로 비유한 문자열(NaN, Infinity, -Infinity)이 없음을 검증
                    parsed = json.loads(serialized, parse_constant=_no_constants)
                    assert parsed is not None
                    assert "NaN" not in serialized
                    assert "Infinity" not in serialized

    # 2. 단계별 측정 칸 (logit_cosine) 주입 검증
    for step_pos in step_positions:
        for bad_val in bad_values:
            records = _make_full_step_records(num_steps, num_layers)
            records[step_pos]["logit_cosine"] = bad_val

            payload = _build_test_summary_payload(records, num_layers)
            expected_path = f"step_records[{step_pos}].logit_cosine"

            assert "non_finite_paths" in payload
            assert expected_path in payload["non_finite_paths"]

            serialized = json.dumps(payload, indent=2, allow_nan=False)
            parsed = json.loads(serialized, parse_constant=_no_constants)
            assert parsed is not None
            assert "NaN" not in serialized
            assert "Infinity" not in serialized


def test_run_golden_ast_json_dump_allow_nan_false():
    """단언 2b: ast로 npu/run_golden.py를 읽어, 모든 json.dump·json.dumps 호출에 allow_nan=False가 있음을 단언."""
    golden_py = Path(__file__).resolve().parent / "run_golden.py"
    tree = ast.parse(golden_py.read_text(encoding="utf-8"))

    violations: list[str] = []

    class JsonDumpVisitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr in ("dump", "dumps"):
                if isinstance(func.value, ast.Name) and func.value.id == "json":
                    has_allow_nan_false = any(
                        kw.arg == "allow_nan" and isinstance(kw.value, ast.Constant) and kw.value.value is False
                        for kw in node.keywords
                    )
                    if not has_allow_nan_false:
                        violations.append(
                            f"Line {node.lineno}: call to json.{func.attr}() without allow_nan=False"
                        )
            self.generic_visit(node)

    visitor = JsonDumpVisitor()
    visitor.visit(tree)
    assert not violations, "Found json.dump/dumps calls without allow_nan=False:\n" + "\n".join(violations)


def test_start_step_table_rejected_function_and_cli():
    """시작 단계 표: 1·3·10·16·4096 은 함수와 CLI 모두 거부해야 한다. 파일을 읽기 전에 거부하는지 없는 경로로 확인한다. 2 는 통과."""
    bad_steps = [1, 3, 10, 16, 4096]
    nonexistent = Path("/nonexistent/path/for/test")

    for step in bad_steps:
        # 함수: 없는 경로더라도 파일을 읽기 전에 ValueError 발생
        with pytest.raises(ValueError, match="is not supported"):
            run_golden_verification(
                weights_d_dir=nonexistent,
                golden_dir=nonexistent,
                start_step=step,
                end_step=step,
            )

        # CLI: argparse ArgumentTypeError -> SystemExit 2
        with pytest.raises(SystemExit) as exc_info:
            _parse_args([
                "--weights-d", str(nonexistent),
                "--golden-dir", str(nonexistent),
                "--start-step", str(step),
            ])
        assert exc_info.value.code == 2

    # 2는 시작 단계 검사를 통과 (없는 경로이므로 파일 로드 단계에서 FileNotFoundError 등으로 진행)
    with pytest.raises(FileNotFoundError):
        run_golden_verification(
            weights_d_dir=nonexistent,
            golden_dir=nonexistent,
            start_step=2,
            end_step=16,
        )

    parsed = _parse_args([
        "--weights-d", str(nonexistent),
        "--golden-dir", str(nonexistent),
        "--start-step", "2",
    ])
    assert parsed.start_step == 2


def test_start_step_greater_than_end_step_rejected():
    """start_step > end_step 은 함수에서 거부."""
    nonexistent = Path("/nonexistent/path/for/test")
    with pytest.raises(ValueError, match="must be <= end_step"):
        run_golden_verification(
            weights_d_dir=nonexistent,
            golden_dir=nonexistent,
            start_step=2,
            end_step=1,
        )


def test_token_list_table_rejected_and_accepted():
    """토큰 목록 표: 거부(키 없음, 목록 아님, True, 1.0, -1, vocab_size 이상, 길이 end_step - 2), 통과(정확한 길이)."""
    end_step = 16
    vocab_size = 259216

    # 1. 통과: 정확한 길이 (end_step - 1 = 15개)
    valid_tokens = [1000 + i for i in range(15)]
    res = validate_generated_token_ids(valid_tokens, end_step=end_step, vocab_size=vocab_size)
    assert res == valid_tokens

    # 2. 거부: 키 없음 (None)
    with pytest.raises(ValueError, match="must be a list"):
        validate_generated_token_ids(None, end_step=end_step, vocab_size=vocab_size)

    # 3. 거부: 목록 아님
    with pytest.raises(ValueError, match="must be a list"):
        validate_generated_token_ids("1000, 1001", end_step=end_step, vocab_size=vocab_size)

    # 4. 거부: True
    bad_true = valid_tokens[:14] + [True]
    with pytest.raises(ValueError, match=r"generated_token_ids\[14\] must be a non-boolean integer"):
        validate_generated_token_ids(bad_true, end_step=end_step, vocab_size=vocab_size)

    # 5. 거부: 1.0
    bad_float = [1.0] + valid_tokens[1:]
    with pytest.raises(ValueError, match=r"generated_token_ids\[0\] must be a non-boolean integer"):
        validate_generated_token_ids(bad_float, end_step=end_step, vocab_size=vocab_size)

    # 6. 거부: -1
    bad_neg = valid_tokens[:14] + [-1]
    with pytest.raises(ValueError, match=r"generated_token_ids\[14\] must be >= 0"):
        validate_generated_token_ids(bad_neg, end_step=end_step, vocab_size=vocab_size)

    # 7. 거부: vocab_size 이상
    bad_vocab = valid_tokens[:14] + [vocab_size]
    with pytest.raises(ValueError, match=r"generated_token_ids\[14\] must be < vocab_size"):
        validate_generated_token_ids(bad_vocab, end_step=end_step, vocab_size=vocab_size)

    # 8. 거부: 길이 end_step - 2 (하나 모자람: 14개)
    short_tokens = valid_tokens[:14]
    with pytest.raises(ValueError, match=r"insufficient for end_step=16; requires at least 15"):
        validate_generated_token_ids(short_tokens, end_step=end_step, vocab_size=vocab_size)


def test_zero_fallback_eliminated_fails_before_npu_step(tmp_path, monkeypatch):
    """0 대체가 다시 생기지 않는지: 토큰 하나 모자란 manifest로 골든 실행(가짜 백엔드)이 NPU 단계에 들어가기 전에 실패한다."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    class FakeConfig:
        vocab_size = 1000
        num_hidden_layers = 25

        @classmethod
        def from_dict(cls, data: dict) -> "FakeConfig":
            return cls()

    import npu.run_golden as golden
    monkeypatch.setattr(golden, "RaonConfig", FakeConfig)

    # manifest.json with 14 tokens (needs 15 for end_step=16)
    manifest = {
        "decode_steps": 16,
        "generated_token_ids": [10 + i for i in range(14)],
    }
    (golden_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # NPU 단계 진입 전 manifest 검증 단계에서 실패
    with pytest.raises(ValueError, match="insufficient for end_step=16"):
        run_golden_verification(
            weights_d_dir=weights_dir,
            golden_dir=golden_dir,
            start_step=2,
            end_step=16,
            mock=True,
        )


def test_no_compare_host_json_output_and_rejudge(tmp_path, capsys, monkeypatch):
    """--no-compare-host: JSON(baseline_passed: null, baseline_skipped: true), 출력(SKIPPED), verification_passed=False, rejudge도 실패, CLI exit 1."""
    golden_dir = tmp_path / "golden"
    golden_dir.mkdir()
    weights_dir = tmp_path / "weights"
    weights_dir.mkdir()

    class FakeConfig:
        vocab_size = 1000
        num_hidden_layers = 25

        @classmethod
        def from_dict(cls, data: dict) -> "FakeConfig":
            return cls()

    class FakeBackend:
        def __init__(self, *args, **kwargs):
            self.timings = {}
            self.call_counts = {}
            self.xclbin_switches = 0
            self.xclbin_calls = {}

        def preload(self, *args, **kwargs):
            return None

        def set_step(self, step: int):
            pass

        def reset_timings(self):
            pass

    class FakeWeights:
        def __init__(self, *args, **kwargs):
            self.tensors = {}

    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.last_intermediates = {}

        def load_snapshot(self, path: Path):
            pass

        def step(self, token: int):
            logits = np.zeros(1000, dtype=np.float32)
            logits[:20] = 10.0
            return logits

    import npu.run_golden as golden
    monkeypatch.setattr(golden, "RaonConfig", FakeConfig)
    monkeypatch.setattr(golden, "NpuBackend", FakeBackend)
    monkeypatch.setattr(golden, "DWeights", FakeWeights)
    monkeypatch.setattr(golden, "RaonModel", FakeModel)
    monkeypatch.setattr(golden, "ensure_mla_cache_expanded", lambda *args: 0)

    manifest = {
        "decode_steps": 2,
        "generated_token_ids": [100, 101],
    }
    (golden_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    # snapshot 1
    snap1_file = golden_dir / "step_000001.npz"
    np.savez(snap1_file, dummy=np.zeros(1))
    # step 2 snapshot (25층 및 나머지 조건 통과를 위해 logits_top20 포함)
    step2_file = golden_dir / "step_000002.npz"
    golden_top20_ids = np.arange(20, dtype=np.int32)
    golden_top20_vals = np.full(20, 10.0, dtype=np.float32)
    np.savez(step2_file, logits_top20_ids=golden_top20_ids, logits_top20_values=golden_top20_vals)

    out_json = tmp_path / "out.json"

    # compare_host_d=False, 나머지 판정(run_complete, golden_accuracy_passed)은 모두 통과
    summary = run_golden_verification(
        weights_d_dir=weights_dir,
        golden_dir=golden_dir,
        start_step=2,
        end_step=2,
        layer_limit=None,
        compare_host_d=False,
        preload_weights=False,
        mock=True,
        out_json=out_json,
    )

    captured = capsys.readouterr()
    # 출력 검증
    assert "Host D Baseline:     SKIPPED" in captured.out

    # JSON 검증: 나머지 판정은 통과했으나 baseline_passed가 None이어서 verification_passed는 False
    assert summary["run_complete"] is True
    assert summary["golden_accuracy_passed"] is True
    assert summary["baseline_passed"] is None
    assert summary["baseline_skipped"] is True
    assert summary["verification_passed"] is False

    # out.json 검증
    written_json = json.loads(out_json.read_text(encoding="utf-8"))
    assert written_json["run_complete"] is True
    assert written_json["golden_accuracy_passed"] is True
    assert written_json["baseline_passed"] is None
    assert written_json["baseline_skipped"] is True
    assert written_json["verification_passed"] is False

    # host rejudge 검증: null을 통과로 읽지 않음
    rejudge_summary = evaluate_summary(written_json)
    assert rejudge_summary["criteria"]["results_complete_and_valid"] is False
    assert print_evaluation(rejudge_summary) is False

    # CLI 종료 코드가 0이 아님(sys.exit(1)) 확인
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_golden.py",
            "--weights-d",
            str(weights_dir),
            "--golden-dir",
            str(golden_dir),
            "--start-step",
            "2",
            "--end-step",
            "2",
            "--mock",
            "--no-preload",
            "--no-compare-host",
            "--out",
            str(out_json),
        ],
    )
    with pytest.raises(SystemExit) as exc_info:
        golden.main()
    assert exc_info.value.code == 1


@pytest.mark.parametrize(
    ("baseline_passed", "expected_verification"),
    [
        (True, True),
        (False, False),
        (None, False),
    ],
)
def test_golden_assessment_baseline_passed_table(baseline_passed, expected_verification):
    """baseline_passed가 True·False·None일 때(나머지 조건은 모두 통과) verification_passed는 각각 참·거짓·거짓."""
    records = [_record(s, 1.0, tie=True) for s in range(2, 5)]
    res = _assess_verification(
        start_step=2,
        end_step=4,
        step_records=records,
        stop_reason={"code": "completed"},
        full_25_layers=True,
        baseline_passed=baseline_passed,
        min_cos=0.999,
        min_top1_tie_rate=0.9,
    )
    assert res["run_complete"] is True
    assert res["golden_accuracy_passed"] is True
    assert res["verification_passed"] is expected_verification


def test_windows_import_without_resource(monkeypatch):
    """Windows에서 불러오기: sys.modules['resource'] = None 상태에서 host.check_golden과 npu.run_golden 로드 성공."""
    monkeypatch.setitem(sys.modules, "resource", None)

    # 기존 모듈 언로드 후 재임포트
    for mod_name in ["host.check_golden", "npu.run_golden"]:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    import host.check_golden as cg
    import npu.run_golden as rg

    assert hasattr(cg, "validate_start_step")
    assert hasattr(cg, "validate_generated_token_ids")
    assert hasattr(rg, "run_golden_verification")


def _collect_npu_backend_read_attrs(run_golden_path: Path) -> set[str]:
    tree = ast.parse(run_golden_path.read_text(encoding="utf-8"), filename=str(run_golden_path))
    attrs: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "npu_backend"
            and isinstance(node.ctx, ast.Load)
        ):
            attrs.add(node.attr)
    return attrs


def _find_fake_backend_classes(npu_dir: Path) -> list[tuple[Path, ast.ClassDef]]:
    results: list[tuple[Path, ast.ClassDef]] = []
    for test_file in sorted(npu_dir.glob("test_*.py")):
        tree = ast.parse(test_file.read_text(encoding="utf-8"), filename=str(test_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name.endswith("Backend"):
                results.append((test_file, node))
    return results


def _get_self_target_attrs(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        names.add(node.attr)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(_get_self_target_attrs(elt))
    return names


def _get_name_targets(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, (ast.Tuple, ast.List)):
        for elt in node.elts:
            names.update(_get_name_targets(elt))
    return names


def _extract_class_defined_attrs(class_node: ast.ClassDef) -> set[str]:
    attrs: set[str] = set()
    for item in class_node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            attrs.add(item.name)
            if item.name == "__init__":
                for sub in ast.walk(item):
                    if isinstance(sub, ast.Assign):
                        for target in sub.targets:
                            attrs.update(_get_self_target_attrs(target))
                    elif isinstance(sub, ast.AnnAssign):
                        attrs.update(_get_self_target_attrs(sub.target))
        elif isinstance(item, ast.Assign):
            for target in item.targets:
                attrs.update(_get_name_targets(target))
        elif isinstance(item, ast.AnnAssign):
            attrs.update(_get_name_targets(item.target))
    return attrs


def test_fake_backends_satisfy_run_golden_attributes():
    """모든 가짜 백엔드 클래스가 npu/run_golden.py에서 npu_backend.<attr>로 읽는 모든 속성을 정의하는지 검증."""
    npu_dir = Path(__file__).resolve().parent
    run_golden_path = Path(os.environ.get("RUN_GOLDEN_PATH", str(npu_dir / "run_golden.py")))
    assert run_golden_path.is_file(), f"run_golden file not found: {run_golden_path}"

    required_attrs = _collect_npu_backend_read_attrs(run_golden_path)
    assert required_attrs, "No npu_backend attributes found in run_golden.py"

    fake_backends = _find_fake_backend_classes(npu_dir)
    assert len(fake_backends) >= 2, f"Expected at least 2 fake backends in {npu_dir}, found {len(fake_backends)}"

    failures: list[str] = []
    for test_file, class_node in fake_backends:
        cls_attrs = _extract_class_defined_attrs(class_node)
        missing = sorted(required_attrs - cls_attrs)
        if missing:
            failures.append(
                f"Class '{class_node.name}' in {test_file.name} missing required attributes: {missing}"
            )

    assert not failures, "Fake backend attribute contract violation:\n" + "\n".join(failures)
