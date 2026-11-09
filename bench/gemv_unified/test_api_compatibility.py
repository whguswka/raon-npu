# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""AST-based API compatibility test against CPU host source trees (mlir-aie & iron).

Verifies that all `aie.*` and `iron.*` module and symbol imports in
`kernels/gemv_unified/*.py` and `bench/gemv_unified/*.py` actually exist
in the CPU host source repositories (located via $RAON_MLIR_AIE_PY and $RAON_IRON_PY):
  - `aie.iron.X`    -> `$RAON_MLIR_AIE_PY/iron/X`
  - `aie.utils.X`   -> `$RAON_MLIR_AIE_PY/utils/X`
  - `aie.helpers.X` -> `$RAON_MLIR_AIE_PY/helpers/X`
  - `aie.dialects.X`-> `$RAON_MLIR_AIE_PY/dialects/X`
  - `aie.X`         -> `$RAON_MLIR_AIE_PY/X`
  - `iron.X`        -> `$RAON_IRON_PY/X`
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import pytest

# Placeholder path that is never a file or directory, so tests skip when unset.
_UNSET = Path(os.devnull)
MLIR_AIE_PY = Path(os.environ.get("RAON_MLIR_AIE_PY", _UNSET))
IRON_PY = Path(os.environ.get("RAON_IRON_PY", _UNSET))

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_module_file(mod_name: str) -> tuple[Path | None, bool]:
    """Map a Python import path like 'aie.iron.kernel' to a file in CPU host sources."""
    parts = mod_name.split(".")
    if parts[0] == "aie":
        base = MLIR_AIE_PY
        subparts = parts[1:]
    elif parts[0] == "iron":
        base = IRON_PY
        subparts = parts[1:]
    else:
        return None, False

    if not subparts:
        init_file = base / "__init__.py"
        if init_file.is_file():
            return init_file, True
        return None, False

    target = base.joinpath(*subparts)
    py_file = target.with_suffix(".py")
    if py_file.is_file():
        return py_file, False
    init_file = target / "__init__.py"
    if init_file.is_file():
        return init_file, True
    return None, False


def get_module_symbols(file_path: Path) -> tuple[set[str], bool]:
    """Extract all functions, classes, assignments, and imports defined in file_path."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    symbols: set[str] = set()
    has_getattr = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
            if node.name == "__getattr__":
                has_getattr = True
        elif isinstance(node, ast.ClassDef):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    symbols.add(t.id)
                elif isinstance(t, (ast.Tuple, ast.List)):
                    for elt in t.elts:
                        if isinstance(elt, ast.Name):
                            symbols.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                symbols.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                symbols.add(a.asname or a.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                symbols.add(a.asname or a.name)
    return symbols, has_getattr


def verify_file_imports(py_file: Path) -> list[str]:
    """Check that all `aie.*` and `iron.*` imports in py_file exist in CPU host sources."""
    errors: list[str] = []
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and (node.module.startswith("aie") or node.module.startswith("iron")):
                mfile, _ = resolve_module_file(node.module)
                if mfile is None:
                    errors.append(
                        f"{py_file.name}:{node.lineno}: Module '{node.module}' not found in CPU host sources"
                    )
                    continue
                syms, has_getattr = get_module_symbols(mfile)
                for alias in node.names:
                    if alias.name != "*" and alias.name not in syms:
                        if has_getattr:
                            continue
                        submod = mfile.parent / f"{alias.name}.py"
                        subpkg = mfile.parent / alias.name / "__init__.py"
                        if not (submod.is_file() or subpkg.is_file()):
                            errors.append(
                                f"{py_file.name}:{node.lineno}: Symbol '{alias.name}' not defined in '{node.module}' ({mfile})"
                            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("aie") or alias.name.startswith("iron"):
                    mfile, _ = resolve_module_file(alias.name)
                    if mfile is None:
                        errors.append(
                            f"{py_file.name}:{node.lineno}: Module '{alias.name}' not found in CPU host sources"
                        )
    return errors


def test_gemv_unified_aie_and_iron_api_consistency() -> None:
    """Verify all production code in kernels/gemv_unified and bench/gemv_unified against CPU host sources."""
    if not MLIR_AIE_PY.is_dir() or not IRON_PY.is_dir():
        pytest.skip(
            "$RAON_MLIR_AIE_PY and $RAON_IRON_PY not set or not directories on this host"
        )

    target_dirs = [
        REPO_ROOT / "kernels" / "gemv_unified",
        REPO_ROOT / "bench" / "gemv_unified",
    ]
    all_errors: list[str] = []
    checked_files = 0

    for d in target_dirs:
        for py_path in sorted(d.glob("*.py")):
            checked_files += 1
            errs = verify_file_imports(py_path)
            all_errors.extend(errs)

    assert checked_files >= 5, f"Expected at least 5 target python files, checked {checked_files}"
    assert not all_errors, "API inconsistencies found against CPU host sources:\n" + "\n".join(all_errors)


def test_api_checker_catches_known_bad_patterns(tmp_path: Path) -> None:
    """Regression test: verify that the AST checker catches known invalid imports."""
    if not MLIR_AIE_PY.is_dir() or not IRON_PY.is_dir():
        pytest.skip(
            "$RAON_MLIR_AIE_PY and $RAON_IRON_PY not set or not directories on this host"
        )
    bad_code = """
from aie.utils.compile.jit.markers import Param  # Param is actually in aie.iron.kernels._common
from aie.iron.placers import SequentialPlacer     # aie.iron.placers does not exist
from iron.common import OperatorArgSpec           # OperatorArgSpec is actually AIERuntimeArgSpec
from iron.common.device_Context import AIEContext # device_Context does not exist
"""
    bad_file = tmp_path / "bad.py"
    bad_file.write_text(bad_code, encoding="utf-8")
    errs = verify_file_imports(bad_file)
    assert len(errs) == 4, f"Expected 4 errors, got {len(errs)}: {errs}"
    assert any("Param" in e for e in errs)
    assert any("SequentialPlacer" in e or "aie.iron.placers" in e for e in errs)
    assert any("OperatorArgSpec" in e for e in errs)
    assert any("device_Context" in e for e in errs)


def _get_function_or_method_signature(
    file_path: Path, class_name: str | None, func_name: str
) -> tuple[list[str], set[str], set[str], bool]:
    """Return positional, allowed, required, and **kwargs data for one callable."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))

    def signature(node: ast.FunctionDef | ast.AsyncFunctionDef):
        positional = list(node.args.posonlyargs) + list(node.args.args)
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        names = [arg.arg for arg in positional]
        allowed = set(names) | {arg.arg for arg in node.args.kwonlyargs}
        num_required_pos = len(positional) - len(node.args.defaults)
        required = set(names[:num_required_pos])
        required.update(
            arg.arg
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if default is None
        )
        return names, allowed, required, node.args.kwarg is not None

    for node in ast.walk(tree):
        if class_name:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == func_name:
                        return signature(sub)
        else:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
                return signature(node)
    return [], set(), set(), False


def _get_class_attributes(file_path: Path, class_name: str) -> set[str]:
    """Extract dataclass fields, methods, and properties declared on a class."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            attrs: set[str] = set()
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    attrs.add(sub.name)
                elif isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    attrs.add(sub.target.id)
                elif isinstance(sub, ast.Assign):
                    attrs.update(
                        target.id for target in sub.targets if isinstance(target, ast.Name)
                    )
            return attrs
    return set()


def _get_dataclass_signature(
    file_path: Path, class_name: str
) -> tuple[list[str], set[str], set[str], bool]:
    """Extract generated ``__init__`` parameters from a source dataclass."""
    tree = ast.parse(file_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            fields: list[str] = []
            required: set[str] = set()
            for sub in node.body:
                if isinstance(sub, ast.AnnAssign) and isinstance(sub.target, ast.Name):
                    fields.append(sub.target.id)
                    if sub.value is None:
                        required.add(sub.target.id)
            return fields, set(fields), required, False
    return [], set(), set(), False


def verify_api_calls(
    py_file: Path,
    signatures: dict[str, tuple[list[str], set[str], set[str], bool]],
    aie_context_attrs: set[str],
) -> list[str]:
    """Check constructor/method calls and removed AIEContext attributes."""
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    errors: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "device_manager":
            if node.attr not in aie_context_attrs:
                errors.append(
                    f"{py_file.name}:{node.lineno}: AIEContext has no attribute "
                    "'device_manager'; use aie_utils.get_current_device()"
                )

        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            target_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            target_name = node.func.attr
        else:
            continue
        if target_name not in signatures:
            continue

        positional, allowed, required, has_var_kw = signatures[target_name]
        keywords = {kw.arg for kw in node.keywords if kw.arg is not None}
        if not has_var_kw:
            unknown = keywords - allowed
            if unknown:
                errors.append(
                    f"{py_file.name}:{node.lineno}: unknown keyword(s) {sorted(unknown)} "
                    f"for {target_name}; allowed={sorted(allowed)}"
                )

        provided = set(positional[: len(node.args)]) | keywords
        missing = required - provided
        if missing:
            errors.append(
                f"{py_file.name}:{node.lineno}: missing required argument(s) "
                f"{sorted(missing)} for {target_name}"
            )
    return errors


def test_iron_keyword_arguments_consistency() -> None:
    """Verify all GEMV bench/device API call sites against CPU host iron definitions."""
    if not MLIR_AIE_PY.is_dir():
        pytest.skip("$RAON_MLIR_AIE_PY not set or not a directory on this host")

    worker_py = MLIR_AIE_PY / "iron" / "worker.py"
    buffer_py = MLIR_AIE_PY / "iron" / "buffer.py"
    objfifo_py = MLIR_AIE_PY / "iron" / "dataflow" / "objectfifo.py"
    runtime_py = MLIR_AIE_PY / "iron" / "runtime" / "runtime.py"
    program_py = MLIR_AIE_PY / "iron" / "program.py"
    flow_py = MLIR_AIE_PY / "iron" / "dataflow" / "flow.py"
    tile_dma_py = MLIR_AIE_PY / "iron" / "dataflow" / "tile_dma.py"
    lock_py = MLIR_AIE_PY / "iron" / "lock.py"
    task_py = MLIR_AIE_PY / "iron" / "runtime" / "dmataskhandle.py"
    tap_py = MLIR_AIE_PY / "helpers" / "taplib" / "tap.py"
    kernel_py = MLIR_AIE_PY / "iron" / "kernel.py"
    context_py = IRON_PY / "common" / "context.py"

    signatures = {
        "Worker": _get_function_or_method_signature(worker_py, "Worker", "__init__"),
        "Buffer": _get_function_or_method_signature(buffer_py, "Buffer", "__init__"),
        "WorkerRuntimeBarrier": _get_function_or_method_signature(worker_py, "WorkerRuntimeBarrier", "__init__"),
        "ObjectFifo": _get_function_or_method_signature(objfifo_py, "ObjectFifo", "__init__"),
        "split": _get_function_or_method_signature(objfifo_py, "ObjectFifoHandle", "split"),
        "join": _get_function_or_method_signature(objfifo_py, "ObjectFifoHandle", "join"),
        "prod": _get_function_or_method_signature(objfifo_py, "ObjectFifo", "prod"),
        "cons": _get_function_or_method_signature(objfifo_py, "ObjectFifo", "cons"),
        "Runtime": _get_function_or_method_signature(runtime_py, "Runtime", "__init__"),
        "Program": _get_function_or_method_signature(program_py, "Program", "__init__"),
        "Flow": _get_function_or_method_signature(flow_py, "Flow", "__init__"),
        "Lock": _get_function_or_method_signature(lock_py, "Lock", "__init__"),
        "TileDma": _get_function_or_method_signature(tile_dma_py, "TileDma", "__init__"),
        "TensorAccessPattern": _get_function_or_method_signature(
            tap_py, "TensorAccessPattern", "__init__"
        ),
        "ExternalFunction": _get_function_or_method_signature(
            kernel_py, "ExternalFunction", "__init__"
        ),
        "Acquire": _get_dataclass_signature(tile_dma_py, "Acquire"),
        "Release": _get_dataclass_signature(tile_dma_py, "Release"),
        "Bd": _get_dataclass_signature(tile_dma_py, "Bd"),
        "DmaChannel": _get_dataclass_signature(tile_dma_py, "DmaChannel"),
        "task": _get_function_or_method_signature(tile_dma_py, "DmaEndpoint", "task"),
        "start": _get_function_or_method_signature(task_py, "Task", "start"),
        "await_": _get_function_or_method_signature(task_py, "Task", "await_"),
        "free": _get_function_or_method_signature(task_py, "Task", "free"),
    }

    # Ensure signatures were successfully extracted
    for name, (positional, allowed, _, _) in signatures.items():
        if name not in {"await_", "free"}:
            assert positional or allowed, f"Failed to extract parameters for {name}"

    aie_context_attrs = _get_class_attributes(context_py, "AIEContext")
    assert aie_context_attrs, "Failed to extract AIEContext attributes"

    target_files = sorted((REPO_ROOT / "bench" / "gemv_unified").glob("*.py"))
    target_files += sorted((REPO_ROOT / "kernels" / "gemv_unified").glob("*.py"))
    errors: list[str] = []
    for py_file in target_files:
        errors.extend(verify_api_calls(py_file, signatures, aie_context_attrs))
    assert not errors, "API call inconsistencies found against CPU host sources:\n" + "\n".join(errors)


def test_low_level_weight_dma_exports_and_runtime_task_contract() -> None:
    """Pin every low-level W-path API to the v1.4.4 source contract."""
    iron_init = MLIR_AIE_PY / "iron" / "__init__.py"
    tile_dma_py = MLIR_AIE_PY / "iron" / "dataflow" / "tile_dma.py"
    if not iron_init.is_file() or not tile_dma_py.is_file():
        pytest.skip("CPU host mlir-aie source path not present")

    exports, _ = get_module_symbols(iron_init)
    required_exports = {
        "Acquire",
        "Bd",
        "Buffer",
        "DmaChannel",
        "Flow",
        "Lock",
        "Release",
        "TileDma",
    }
    assert required_exports <= exports, sorted(required_exports - exports)

    task_sig = _get_function_or_method_signature(tile_dma_py, "DmaEndpoint", "task")
    positional, allowed, required, has_var_kw = task_sig
    assert positional == ["bds"] or positional == []  # ``*bds`` is not positional AST data.
    assert {"runs", "wait", "out_of_order"} <= allowed
    assert not required
    assert has_var_kw is False

    for name in ("Acquire", "Release", "Bd", "DmaChannel"):
        positional, allowed, _, has_var_kw = _get_dataclass_signature(tile_dma_py, name)
        assert positional and allowed
        assert has_var_kw is False


def test_api_call_checker_catches_required_args_and_removed_context_attr(tmp_path: Path) -> None:
    """Regression: catch Runtime() and the removed AIEContext.device_manager API."""
    runtime_py = MLIR_AIE_PY / "iron" / "runtime" / "runtime.py"
    context_py = IRON_PY / "common" / "context.py"
    if not runtime_py.is_file() or not context_py.is_file():
        pytest.skip("CPU host source paths not present on this host")

    signatures = {
        "Runtime": _get_function_or_method_signature(runtime_py, "Runtime", "__init__")
    }
    bad_file = tmp_path / "bad_calls.py"
    bad_file.write_text(
        "from aie.iron import Runtime\n"
        "from iron.common import AIEContext\n"
        "ctx = AIEContext()\n"
        "rt = Runtime()\n"
        "dev = ctx.device_manager.device\n",
        encoding="utf-8",
    )
    errors = verify_api_calls(
        bad_file, signatures, _get_class_attributes(context_py, "AIEContext")
    )
    assert any("missing required" in error and "seq_fn" in error for error in errors)
    assert any("device_manager" in error for error in errors)
