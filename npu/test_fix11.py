# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for fix11 wrapper, timeout, and strict-JSON contracts."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pytest

from host.weights import SafeTensorsReader, Weights
from npu.backend import _load_manifest_specs
import npu.run_golden as golden


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_powershell_external_process_exit_code_rules():
    scripts = sorted((REPO_ROOT / "npu").rglob("*.ps1")) + sorted((REPO_ROOT / "serve").rglob("*.ps1"))
    assert scripts

    violations: list[str] = []
    for script in scripts:
        lines = script.read_text(encoding="utf-8").splitlines()
        captured_vars: set[str] = set()
        for index, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("& "):
                following = next(
                    (candidate.strip() for candidate in lines[index + 1 :] if candidate.strip() and not candidate.lstrip().startswith("#")),
                    "",
                )
                if "= $LASTEXITCODE" not in following:
                    violations.append(f"{script.relative_to(REPO_ROOT)}:{index + 1}: next statement is {following!r}")
                elif following.startswith("$"):
                    captured_vars.add(following.split("=", 1)[0].strip())
        for variable in captured_vars:
            exit_indexes = [index for index, line in enumerate(lines) if f"exit {variable}" in line]
            if not exit_indexes:
                violations.append(f"{script.relative_to(REPO_ROOT)}: no exit {variable}")
                continue
            null_check_indexes = [
                index for index, line in enumerate(lines) if f"$null -eq {variable}" in line
            ]
            for exit_index in exit_indexes:
                if not any(null_check_index < exit_index for null_check_index in null_check_indexes):
                    violations.append(
                        f"{script.relative_to(REPO_ROOT)}:{exit_index + 1}: "
                        f"exit {variable} lacks a preceding $null check"
                    )
        for index, line in enumerate(lines):
            if "Start-Process" in line:
                violations.append(f"{script.relative_to(REPO_ROOT)}:{index + 1}: Start-Process is forbidden")

    assert not violations, "PowerShell external-process rules failed:\n" + "\n".join(violations)


def test_powershell_scripts_do_not_write_to_files():
    scripts = sorted((REPO_ROOT / "npu").rglob("*.ps1")) + sorted((REPO_ROOT / "serve").rglob("*.ps1"))
    assert scripts

    violations: list[str] = []
    for script in scripts:
        text = script.read_text(encoding="utf-8")
        stripped = re.sub(r"<#.*?#>", lambda m: "\n" * m.group(0).count("\n") + " ", text, flags=re.DOTALL)
        stripped = re.sub(r"#[^\r\n]*", " ", stripped)
        stripped = re.sub(r'"(""|`[\s\S]|[^"])*"', " ", stripped)
        stripped = re.sub(r"'(''|[^'])*'", " ", stripped)

        for lineno, line in enumerate(stripped.splitlines(), 1):
            for m in re.finditer(r"\b(Tee-Object|Out-File|Set-Content|Add-Content)\b", line, re.IGNORECASE):
                violations.append(f"{script.relative_to(REPO_ROOT)}:{lineno}: forbidden cmdlet {m.group(0)!r}")
            for m in re.finditer(r"(?<!-)(?:\*|\d)?>>?&?\d?", line):
                if ">" in m.group(0):
                    violations.append(f"{script.relative_to(REPO_ROOT)}:{lineno}: forbidden redirection operator {m.group(0)!r}")

    assert not violations, (
        "PowerShell file-write rules failed (PS 5.1 UTF-16 encoding hazard):\n"
        + "\n".join(violations)
    )


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell.exe") is None, reason="Windows PowerShell required")
@pytest.mark.parametrize("relative_script", ["npu/run-golden.ps1", "serve/run-serve.ps1"])
@pytest.mark.parametrize("fake_exit", [0, 3])
def test_powershell_wrappers_propagate_fake_launcher_exit(tmp_path: Path, relative_script: str, fake_exit: int):
    fake = tmp_path / "fake-python.cmd"
    fake.write_text("@echo off\r\necho fake stderr 1>&2\r\nexit /b %FAKE_EXIT_CODE%\r\n", encoding="utf-8")
    env = os.environ.copy()
    env["FAKE_EXIT_CODE"] = str(fake_exit)
    command = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(REPO_ROOT / relative_script), "-PythonCmd", str(fake),
    ]
    if relative_script.startswith("npu/"):
        command.extend(["-OutFile", str(tmp_path / "report.json")])
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command.extend(["-Port", str(port), "-Log", str(tmp_path / "serve.log")])

    result = subprocess.run(command, text=True, capture_output=True, env=env, timeout=20, check=False)
    assert result.returncode == fake_exit, result.stdout + result.stderr
    assert "fake stderr" in result.stdout + result.stderr


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("powershell.exe") is None, reason="Windows PowerShell required")
@pytest.mark.parametrize("relative_script", ["npu/run-golden.ps1", "serve/run-serve.ps1"])
def test_powershell_wrappers_reject_missing_launcher(tmp_path: Path, relative_script: str):
    missing = tmp_path / "missing-python.cmd"
    command = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(REPO_ROOT / relative_script), "-PythonCmd", str(missing),
    ]
    if relative_script.startswith("npu/"):
        command.extend(["-OutFile", str(tmp_path / "report.json")])
    else:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        command.extend(["-Port", str(port), "-Log", str(tmp_path / "serve.log")])

    result = subprocess.run(command, text=True, capture_output=True, timeout=20, check=False)
    assert result.returncode != 0, result.stdout + result.stderr


def test_server_rejects_already_bound_port():
    from serve.server import Server, Handler

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", 1), 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        with pytest.raises(OSError):
            Server(("0.0.0.0", port), Handler)


def test_main_rejects_blocked_port_before_creating_engine(monkeypatch: pytest.MonkeyPatch):
    import serve.server as server_mod

    engine_created = []

    class FakeEngine:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            engine_created.append(True)
            raise AssertionError("Engine must not be created when port is blocked")

    fake_mod = type(sys)("serve.engine")
    fake_mod.Engine = FakeEngine
    fake_mod.GenOptions = object
    fake_mod.Chunk = object
    monkeypatch.setitem(sys.modules, "serve.engine", fake_mod)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", 1), 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        with pytest.raises(SystemExit) as excinfo:
            server_mod.main(["--port", str(port)])
        assert excinfo.value.code != 0
        assert not engine_created, "Engine was created before port binding check"


@pytest.mark.skipif(
    sys.platform != "win32" or shutil.which("powershell.exe") is None,
    reason="Windows PowerShell required",
)
def test_run_serve_rejects_an_already_listening_port(tmp_path: Path):
    log_path = tmp_path / "serve.log"

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        if os.name == "nt":
            listener.setsockopt(socket.SOL_SOCKET, getattr(socket, "SO_EXCLUSIVEADDRUSE", 1), 1)
        listener.bind(("0.0.0.0", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        proc = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(REPO_ROOT / "serve" / "run-serve.ps1"),
                "-Port",
                str(port),
                "-PythonCmd",
                sys.executable,
                "-Log",
                str(log_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = proc.communicate(timeout=20)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/T", "/F", "/PID", str(proc.pid)], check=False
                )
            else:
                proc.kill()
            stdout_bytes, stderr_bytes = proc.communicate()
            raise
        finally:
            if proc.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(
                        ["taskkill", "/T", "/F", "/PID", str(proc.pid)], check=False
                    )
                else:
                    proc.kill()
                proc.communicate()

    stdout = (stdout_bytes or b"").decode("utf-8", errors="replace")
    stderr = (stderr_bytes or b"").decode("utf-8", errors="replace")
    assert returncode != 0, f"stdout={stdout}\nstderr={stderr}"

    log_bytes = log_path.read_bytes() if log_path.exists() else b""
    log_content = log_bytes.decode("utf-8", errors="replace")
    if log_path.exists():
        log_bytes.decode("utf-8")

    combined = stdout + stderr + "\n" + log_content
    assert any(
        phrase in combined.lower()
        for phrase in ["failed to bind", "10048", "address already in use"]
    ), f"Expected bind failure message in output/log, got:\n{combined}"
    assert "engine loaded" not in combined.lower()


def test_ast_all_npu_model_steps_use_timeout_helper():
    tree = ast.parse((REPO_ROOT / "npu" / "run_golden.py").read_text(encoding="utf-8"))
    violations: list[str] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr == "step":
                receiver = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                allowed_host_reference = receiver == "host_model"
                allowed_timeout_helper = self.functions and self.functions[-1] == "_run_npu_step_with_timeout"
                if not allowed_host_reference and not allowed_timeout_helper:
                    violations.append(f"line {node.lineno}: direct {receiver or '<expression>'}.step call")
            self.generic_visit(node)

    Visitor().visit(tree)
    assert not violations, "NPU step calls outside timeout helper:\n" + "\n".join(violations)


@pytest.mark.parametrize(
    ("timeout_per_token", "remaining_total"),
    [(1.0, None), (60.0, 1.0)],
)
def test_npu_step_watchdog_force_exits_subprocess(timeout_per_token: float, remaining_total: float | None):
    code = f"""
import time
from npu.run_golden import _run_npu_step_with_timeout
class Sleeper:
    def step(self, token):
        time.sleep(60)
_run_npu_step_with_timeout(Sleeper(), 0, timeout_per_token={timeout_per_token!r}, remaining_total={remaining_total!r})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 1
    assert "Timeout" in result.stderr


@pytest.mark.parametrize(
    ("timeout_per_token", "remaining_total", "expected"),
    [(1.0, None, "token_timeout"), (60.0, 1.0, "total_timeout")],
)
def test_npu_step_post_return_timeout_classification(
    monkeypatch: pytest.MonkeyPatch,
    timeout_per_token: float,
    remaining_total: float | None,
    expected: str,
):
    class Immediate:
        def step(self, token: int) -> np.ndarray:
            return np.zeros(1, dtype=np.float32)

    ticks = iter([10.0, 12.0])
    monkeypatch.setattr(golden.faulthandler, "dump_traceback_later", lambda *args, **kwargs: None)
    monkeypatch.setattr(golden.faulthandler, "cancel_dump_traceback_later", lambda: None)
    monkeypatch.setattr(golden.time, "perf_counter", lambda: next(ticks))
    with pytest.raises(golden._NpuStepDeadlineExceeded) as exc_info:
        golden._run_npu_step_with_timeout(
            Immediate(),
            0,
            timeout_per_token=timeout_per_token,
            remaining_total=remaining_total,
        )
    assert exc_info.value.code == expected


def test_npu_step_does_not_call_model_when_total_time_is_exhausted():
    class NeverCalled:
        called = False

        def step(self, token: int) -> np.ndarray:
            self.called = True
            return np.zeros(1, dtype=np.float32)

    model = NeverCalled()
    with pytest.raises(golden._NpuStepDeadlineExceeded) as exc_info:
        golden._run_npu_step_with_timeout(model, 0, timeout_per_token=1.0, remaining_total=0.0)
    assert exc_info.value.code == "total_timeout"
    assert model.called is False


def test_run_golden_last_step_postprocessing_total_timeout_is_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    golden_dir = tmp_path / "golden"
    weights_dir = tmp_path / "weights"
    golden_dir.mkdir()
    weights_dir.mkdir()
    (golden_dir / "manifest.json").write_text('{"generated_token_ids":[1,2]}', encoding="utf-8")
    for step in (1, 2, 3):
        np.savez(golden_dir / f"step_{step:06d}.npz")

    class FakeConfig:
        num_hidden_layers = 1

    class FakeBackend:
        def __init__(self, **kwargs: Any):
            self.timings: dict[str, float] = {}
            self.call_counts: dict[str, int] = {}
            self.xclbin_switches: int = 0
            self.xclbin_calls: dict[str, int] = {}

        def preload(self, *args: Any, **kwargs: Any) -> None:
            pass

        def set_step(self, step: int) -> None:
            pass

        def reset_timings(self) -> None:
            pass

    class FakeWeights:
        def __init__(self, *args: Any, **kwargs: Any):
            pass

    class FakeModel:
        calls = 0

        def __init__(self, *args: Any, **kwargs: Any):
            self.last_intermediates: dict[str, np.ndarray] = {}

        def load_snapshot(self, path: Path) -> None:
            pass

        def step(self, token: int) -> np.ndarray:
            FakeModel.calls += 1
            return np.zeros(32, dtype=np.float32)

    monkeypatch.setattr(golden, "RaonConfig", FakeConfig)
    monkeypatch.setattr(golden, "NpuBackend", FakeBackend)
    monkeypatch.setattr(golden, "DWeights", FakeWeights)
    monkeypatch.setattr(golden, "RaonModel", FakeModel)
    monkeypatch.setattr(golden, "ensure_mla_cache_expanded", lambda *args: 0)
    monkeypatch.setattr(golden.faulthandler, "dump_traceback_later", lambda *args, **kwargs: None)
    monkeypatch.setattr(golden.faulthandler, "cancel_dump_traceback_later", lambda: None)
    real_np_load = golden.np.load

    def slow_last_golden_load(path: Path, *args: Any, **kwargs: Any):
        if Path(path).name == "step_000003.npz":
            time.sleep(0.12)
        return real_np_load(path, *args, **kwargs)

    monkeypatch.setattr(golden.np, "load", slow_last_golden_load)

    summary = golden.run_golden_verification(
        weights_dir,
        golden_dir,
        start_step=2,
        end_step=3,
        timeout_per_token=1.0,
        total_timeout=0.1,
        compare_host_d=False,
        preload_weights=False,
        out_json=tmp_path / "summary.json",
    )
    assert FakeModel.calls == 2
    assert summary["verification_passed"] is False
    assert summary["stop_reason"]["code"] == "total_timeout"
    assert summary["steps_completed"] == 1


def test_golden_stop_reason_table_and_assessment_fail_closed():
    tree = ast.parse((REPO_ROOT / "npu" / "run_golden.py").read_text(encoding="utf-8"))
    emitted_codes: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "code"
                    and isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                ):
                    emitted_codes.add(value.value)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_NpuStepDeadlineExceeded"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            emitted_codes.add(node.args[0].value)

    expected_codes = {"completed", "missing_snapshot", "token_timeout", "total_timeout"}
    assert emitted_codes == expected_codes
    records = [
        {"step": 2, "logit_cosine": 1.0, "top1_tie_match": True},
        {"step": 3, "logit_cosine": 1.0, "top1_tie_match": True},
    ]
    for code in expected_codes - {"completed"}:
        assessment = golden._assess_verification(
            start_step=2,
            end_step=3,
            step_records=records,
            stop_reason={"code": code},
            full_25_layers=True,
            baseline_passed=True,
            min_cos=0.999,
            min_top1_tie_rate=0.9,
        )
        assert assessment["verification_passed"] is False, code


def test_run_golden_removes_stale_summary_before_reading_inputs(tmp_path: Path):
    out_json = tmp_path / "stale.json"
    out_json.write_text('{"verification_passed":true}', encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        golden.run_golden_verification(tmp_path / "weights", tmp_path / "missing-golden", out_json=out_json)
    assert not out_json.exists()


def test_golden_timeout_option_table_covers_all_parser_timeouts():
    timeout_contracts = [
        {"dest": "timeout_per_token", "scope": "one NPU step", "hard_exit": True},
        {"dest": "total_timeout", "scope": "decode loop", "hard_exit": True},
    ]
    parser_dests = {
        action.dest
        for action in golden._build_parser()._actions
        if "timeout" in action.dest
    }
    assert {row["dest"] for row in timeout_contracts} == parser_dests


def _leaf_paths(value: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    if isinstance(value, dict):
        return [path for key, child in value.items() for path in _leaf_paths(child, prefix + (key,))]
    if isinstance(value, list):
        return [path for index, child in enumerate(value) for path in _leaf_paths(child, prefix + (index,))]
    return [prefix]


def _json_with_raw_value(value: Any, path: tuple[Any, ...], raw_token: str) -> str:
    if not path:
        return raw_token
    head, *tail = path
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(key) + ":" + (_json_with_raw_value(child, tuple(tail), raw_token) if key == head else json.dumps(child))
            for key, child in value.items()
        ) + "}"
    return "[" + ",".join(
        _json_with_raw_value(child, tuple(tail), raw_token) if index == head else json.dumps(child)
        for index, child in enumerate(value)
    ) + "]"


@pytest.mark.parametrize("raw_token", ["NaN", "Infinity", "-Infinity", "1e400"])
def test_manifest_rejects_non_finite_at_every_value_position(tmp_path: Path, raw_token: str):
    manifest = {"bundles": [], "metadata": {"count": 0, "nested": [True, {"name": "ok"}]}}
    for index, path in enumerate(_leaf_paths(manifest)):
        manifest_path = tmp_path / f"manifest-{index}.json"
        manifest_path.write_text(_json_with_raw_value(manifest, path, raw_token), encoding="utf-8")
        with pytest.raises(ValueError):
            _load_manifest_specs(manifest_path, {}, {}, require_complete=False)


def test_safetensors_header_rejects_non_finite_json(tmp_path: Path):
    header = b'{"tensor":{"dtype":"F32","shape":[1],"data_offsets":[0,NaN]}}'
    path = tmp_path / "bad.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header)
    with pytest.raises(ValueError):
        SafeTensorsReader(path)


def test_weights_index_rejects_non_finite_json(tmp_path: Path):
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":NaN},"weight_map":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        Weights(tmp_path)


def _module_path(module: str) -> Path | None:
    direct = REPO_ROOT.joinpath(*module.split(".")).with_suffix(".py")
    package = REPO_ROOT.joinpath(*module.split("."), "__init__.py")
    if direct.is_file():
        return direct
    if package.is_file():
        return package
    return None


def _reachable_repo_modules(entry_modules: set[str]) -> set[Path]:
    pending = list(entry_modules)
    seen_modules: set[str] = set()
    paths: set[Path] = set()
    while pending:
        module = pending.pop()
        if module in seen_modules:
            continue
        seen_modules.add(module)
        path = _module_path(module)
        if path is None:
            continue
        paths.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        package_parts = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    keep = len(package_parts) - (node.level - 1)
                    base_parts = package_parts[:keep]
                else:
                    base_parts = []
                if node.module:
                    base_parts += node.module.split(".")
                base = ".".join(base_parts)
                if base:
                    pending.append(base)
                for alias in node.names:
                    candidate = ".".join(base_parts + [alias.name])
                    if candidate:
                        pending.append(candidate)
    return paths


def test_ast_all_json_loads_in_strict_helpers():
    target_files = {
        path
        for directory in (REPO_ROOT / "npu", REPO_ROOT / "serve")
        for path in directory.rglob("*.py")
        if not path.name.startswith("test_") and not path.name.endswith("_test.py")
    }
    reachable = _reachable_repo_modules({"serve.server", "npu.run_golden"})
    target_files |= reachable
    assert REPO_ROOT / "host" / "weights.py" in reachable
    assert REPO_ROOT / "npu" / "backend.py" in reachable

    violations: list[str] = []
    strict_path = REPO_ROOT / "host" / "strict_json.py"

    class Visitor(ast.NodeVisitor):
        def __init__(self, path: Path):
            self.path = path
            self.functions: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.functions.append(node.name)
            self.generic_visit(node)
            self.functions.pop()

        def _allowed(self) -> bool:
            return self.path == strict_path and self.functions and self.functions[-1] == "loads_strict"

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute) and node.func.attr in {"from_file", "from_pretrained"}:
                violations.append(
                    f"{self.path.relative_to(REPO_ROOT)}:{node.lineno}: {node.func.attr}; "
                    "글자를 읽어 loads_strict 로 검증한 뒤 넘겨라"
                )
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                if node.func.value.id == "json" and node.func.attr in {"load", "loads", "JSONDecoder"} and not self._allowed():
                    violations.append(f"{self.path.relative_to(REPO_ROOT)}:{node.lineno}: json.{node.func.attr}")
            self.generic_visit(node)

        def visit_Attribute(self, node: ast.Attribute) -> None:
            if isinstance(node.value, ast.Name) and node.value.id == "json" and node.attr == "decoder" and not self._allowed():
                violations.append(f"{self.path.relative_to(REPO_ROOT)}:{node.lineno}: json.decoder")
            self.generic_visit(node)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module == "json" and not self._allowed():
                violations.append(f"{self.path.relative_to(REPO_ROOT)}:{node.lineno}: from json import")

        def visit_Import(self, node: ast.Import) -> None:
            for alias in node.names:
                if alias.name == "json" and alias.asname not in (None, "json") and not self._allowed():
                    violations.append(f"{self.path.relative_to(REPO_ROOT)}:{node.lineno}: import json as {alias.asname}")

    for path in sorted(target_files):
        Visitor(path).visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
    assert not violations, "Strict JSON AST violations:\n" + "\n".join(violations)


def test_json_input_filename_table_covers_non_test_code():
    actual: set[str] = set()
    for directory in (REPO_ROOT / "npu", REPO_ROOT / "serve"):
        for path in directory.rglob("*.py"):
            if path.name.startswith("test_") or path.name.endswith("_test.py"):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            actual.update(
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value.endswith(".json")
            )
    expected = {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "manifest.json",
        "model.safetensors.index.json",
    }
    assert actual == expected


_STRICT_JSON_IDS = ["nan", "inf", "-inf", "1e400", "deep-nesting"]


def _strict_json_corruptions() -> list[str]:
    return [
        '{"value":NaN}',
        '{"value":Infinity}',
        '{"value":-Infinity}',
        '{"value":1e400}',
        "[" * 100_000 + "0" + "]" * 100_000,
    ]


def _load_real_engine_class():
    import importlib.util

    if "_real_serve_engine" in sys.modules:
        return sys.modules["_real_serve_engine"].Engine
    engine_path = REPO_ROOT / "serve" / "engine.py"
    spec = importlib.util.spec_from_file_location("_real_serve_engine", engine_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_real_serve_engine"] = mod
    spec.loader.exec_module(mod)
    return mod.Engine


@pytest.mark.parametrize("raw_json", _strict_json_corruptions(), ids=_STRICT_JSON_IDS)
def test_engine_and_golden_config_reject_corrupt_json(tmp_path: Path, raw_json: str):
    engine_cls = _load_real_engine_class()

    engine_dir = tmp_path / "engine"
    engine_dir.mkdir()
    (engine_dir / "config.json").write_text(raw_json, encoding="utf-8")
    engine = object.__new__(engine_cls)
    engine.model_dir = engine_dir
    engine.weights_dir = engine_dir
    with pytest.raises(ValueError):
        engine._build_model(preload=False, backend=object())

    golden_dir = tmp_path / "golden"
    weights_dir = tmp_path / "weights"
    golden_dir.mkdir()
    weights_dir.mkdir()
    (weights_dir / "config.json").write_text(raw_json, encoding="utf-8")
    (golden_dir / "manifest.json").write_text('{"generated_token_ids":[]}', encoding="utf-8")
    with pytest.raises(ValueError):
        golden.run_golden_verification(weights_dir, golden_dir)


@pytest.mark.parametrize("raw_json", _strict_json_corruptions(), ids=_STRICT_JSON_IDS)
def test_npu_and_golden_manifests_reject_corrupt_json(tmp_path: Path, raw_json: str):
    npu_manifest = tmp_path / "manifest.json"
    npu_manifest.write_text(raw_json, encoding="utf-8")
    with pytest.raises(ValueError):
        _load_manifest_specs(npu_manifest, {}, {}, require_complete=False)

    golden_dir = tmp_path / "golden"
    weights_dir = tmp_path / "weights"
    golden_dir.mkdir()
    weights_dir.mkdir()
    (golden_dir / "manifest.json").write_text(raw_json, encoding="utf-8")
    with pytest.raises(ValueError):
        golden.run_golden_verification(weights_dir, golden_dir)


@pytest.mark.parametrize("filename", ["tokenizer.json", "generation_config.json", "tokenizer_config.json"])
@pytest.mark.parametrize("raw_json", _strict_json_corruptions(), ids=_STRICT_JSON_IDS)
def test_tokenizer_json_inputs_reject_corruption(tmp_path: Path, filename: str, raw_json: str):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from serve.tokenize import RaonTokenizer

    model_dir = tmp_path / filename.replace(".json", "")
    model_dir.mkdir()
    tokenizer = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    (model_dir / "tokenizer.json").write_text(tokenizer.to_str(), encoding="utf-8")
    (model_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    if filename == "tokenizer_config.json":
        (model_dir / "chat_template.jinja").unlink()
    (model_dir / filename).write_text(raw_json, encoding="utf-8")
    with pytest.raises(ValueError):
        RaonTokenizer(model_dir)


@pytest.mark.parametrize("raw_json", _strict_json_corruptions(), ids=_STRICT_JSON_IDS)
def test_weights_index_rejects_all_strict_json_corruption(tmp_path: Path, raw_json: str):
    (tmp_path / "model.safetensors.index.json").write_text(raw_json, encoding="utf-8")
    with pytest.raises(ValueError):
        Weights(tmp_path)


@pytest.mark.parametrize("raw_json", _strict_json_corruptions(), ids=_STRICT_JSON_IDS)
def test_api_tags_skips_corrupt_manifest(
    tmp_path: Path, raw_json: str, monkeypatch: pytest.MonkeyPatch
):
    from serve.server import Handler

    monkeypatch.delenv("RAON_WEIGHTS_D", raising=False)
    (tmp_path / "manifest.json").write_text(raw_json, encoding="utf-8")
    old_weights_dir = Handler.weights_dir
    old_model_dir = Handler.model_dir
    try:
        Handler.weights_dir = str(tmp_path)
        Handler.model_dir = ""
        Handler._cached_manifest_info = None
        Handler._cached_manifest_key = None
        info = Handler._get_manifest_info()
    finally:
        Handler.weights_dir = old_weights_dir
        Handler.model_dir = old_model_dir
        Handler._cached_manifest_info = None
        Handler._cached_manifest_key = None
    assert info["total_size_bytes"] == 0


def test_npu_and_serve_files_have_no_machine_specific_repo_paths():
    forbidden = (
        "C:\\dev\\raon\\integ",
        "C:\\dev\\raon\\unified",
        "C:\\dev\\raon\\srv-",
        "/" + "data/" + "ana" + "conda3",
        "/" + "home/",
        "C:\\Users\\",
        "/tmp/" + "claude-",
    )
    violations: list[str] = []
    for directory in (REPO_ROOT / "npu", REPO_ROOT / "serve"):
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".py", ".ps1", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for banned in forbidden:
                if banned in text:
                    violations.append(f"{path.relative_to(REPO_ROOT)}: {banned}")
    assert not violations, "Machine-specific paths found:\n" + "\n".join(violations)


def test_all_collected_nodeids_under_1000_chars(request: pytest.FixtureRequest):
    items = request.session.items
    oversized = [
        item.nodeid
        for item in items
        if len(item.nodeid) > 1000
    ]
    assert not oversized, (
        f"Found {len(oversized)} tests with nodeid > 1000 characters:\n"
        + "\n".join(f"{n[:80]}... ({len(n)} chars)" for n in oversized[:10])
    )
