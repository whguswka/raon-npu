# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MLIR kernel hash normalization in artifact checking."""

from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import types

import pytest

import bench.gemv_unified.plan_artifact_check as artifact_check
from bench.gemv_unified.plan_artifact_check import (
    PER_CALL_SHAPES,
    capture_kernel_inputs,
    compare_artifact_sets,
    kernel_input_fingerprint,
    normalize_kernel_hashes,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_normalize_kernel_hashes_equates_different_build_hashes() -> None:
    # Digit-starting hash (requires MLIR quoting)
    snippet_quoted = (
        'func.func private @"628d8ca8_bfp16_gemv_unified_block"(memref<128xi32>, memref<256xbf16>) {\n'
        '  aie.call @"628d8ca8_bfp16_gemv_unified_block"() { link_with = "bfp16_gemv_unified_block_628d8ca8.o" }\n'
        '  return\n'
        '}\n'
    )
    # Letter-starting hash (bare identifier in MLIR)
    snippet_bare = (
        'func.func private @fdc930c8_bfp16_gemv_unified_block(memref<128xi32>, memref<256xbf16>) {\n'
        '  aie.call @fdc930c8_bfp16_gemv_unified_block() { link_with = "bfp16_gemv_unified_block_fdc930c8.o" }\n'
        '  return\n'
        '}\n'
    )

    # Before normalization: raw strings and SHAs are different
    assert snippet_quoted != snippet_bare
    assert _sha(snippet_quoted) != _sha(snippet_bare)

    # After normalization: identical placeholder and identical SHA
    norm_quoted = normalize_kernel_hashes(snippet_quoted)
    norm_bare = normalize_kernel_hashes(snippet_bare)

    assert norm_quoted == norm_bare
    assert _sha(norm_quoted) == _sha(norm_bare)
    assert "628d8ca8" not in norm_quoted
    assert "fdc930c8" not in norm_bare
    assert "bfp16_gemv_unified_block_HASH.o" in norm_quoted
    assert "@HASH_bfp16_gemv_unified_block" in norm_quoted


def test_normalize_kernel_hashes_detects_one_char_real_difference() -> None:
    snippet1 = (
        'func.func private @"628d8ca8_bfp16_gemv_unified_block"(memref<128xi32>) {\n'
        '  aie.call @"628d8ca8_bfp16_gemv_unified_block"() { link_with = "bfp16_gemv_unified_block_628d8ca8.o" }\n'
        '  return\n'
        '}\n'
    )
    # Difference: memref<128xi32> vs memref<129xi32> (one character)
    snippet2 = (
        'func.func private @fdc930c8_bfp16_gemv_unified_block(memref<129xi32>) {\n'
        '  aie.call @fdc930c8_bfp16_gemv_unified_block() { link_with = "bfp16_gemv_unified_block_fdc930c8.o" }\n'
        '  return\n'
        '}\n'
    )

    norm1 = normalize_kernel_hashes(snippet1)
    norm2 = normalize_kernel_hashes(snippet2)

    assert norm1 != norm2
    assert _sha(norm1) != _sha(norm2)


class FakeKernel:
    pass


def _make_repo(root: Path, source: bytes = b"kernel source") -> tuple[Path, Path]:
    repo = root
    source_path = repo / "kernels" / "fake.cc"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(source)
    include_dir = repo / "kernels" / "include"
    include_dir.mkdir()
    (include_dir / "fake.h").write_bytes(b"header")
    return source_path, include_dir


def _fingerprint(repo: Path, flags: list[str] | None = None) -> dict[str, object]:
    source, include_dir = _make_repo(repo)
    return kernel_input_fingerprint(
        repo,
        ("fake_kernel", source, ["memref<32xbf16>"]),
        {
            "compile_flags": flags or ["-DROWS=32"],
            "include_dirs": [include_dir],
            "cls": FakeKernel,
        },
    )


def _record(
    insts_sha: str = "insts", *, name: str = "fake", device: str = "device"
) -> dict[str, object]:
    return {
        "name": name,
        "mlir_raw_sha256": "raw",
        "mlir_norm_sha256": "norm",
        "device_norm_sha256": device,
        "insts_sha256": insts_sha,
    }


def _payload(fingerprints: list[dict[str, object]], insts_sha: str = "insts") -> dict[str, object]:
    records = {
        "fake:per_round": _record(insts_sha),
        "rda:per_round": _record(insts_sha, name="rda"),
    }
    for shape in PER_CALL_SHAPES:
        records[f"{shape}:per_round"] = _record(insts_sha, name=shape)
        records[f"{shape}:per_call"] = _record(insts_sha, name=f"{shape}_pc")
    return {
        "records": records,
        "kernel_fingerprints": fingerprints,
    }


def test_capture_kernel_inputs_wraps_fake_module_and_returns_original_result(
    tmp_path: Path, monkeypatch
) -> None:
    repo = tmp_path / "repo"
    source, include_dir = _make_repo(repo)
    module = types.ModuleType("kernels.fake_fingerprint_test")
    sentinel = object()

    def fake_make_extern(*_args, **_kwargs):
        return sentinel

    module._make_extern = fake_make_extern
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with capture_kernel_inputs(repo) as captured:
        result = module._make_extern(
            "fake_kernel",
            source,
            ["arg"],
            compile_flags=["-DVALUE=1"],
            include_dirs=[include_dir],
            cls=FakeKernel,
        )

    assert result is sentinel
    assert len(captured) == 1
    assert captured[0]["source"]["path"] == "kernels/fake.cc"
    assert module._make_extern is fake_make_extern


def test_kernel_source_byte_difference_fails(tmp_path: Path) -> None:
    old_repo = tmp_path / "old"
    new_repo = tmp_path / "new"
    old_fp = _fingerprint(old_repo)
    _fingerprint(new_repo)
    (new_repo / "kernels" / "fake.cc").write_bytes(b"changed source")
    new_fp = kernel_input_fingerprint(
        new_repo,
        ("fake_kernel", new_repo / "kernels" / "fake.cc", ["memref<32xbf16>"]),
        {
            "compile_flags": ["-DROWS=32"],
            "include_dirs": [new_repo / "kernels" / "include"],
            "cls": FakeKernel,
        },
    )

    failed, _rows, diagnostics = compare_artifact_sets(
        _payload([old_fp]), _payload([new_fp]), compile_insts=False
    )
    assert failed
    assert diagnostics[0].startswith("커널 입력 지문: 다름 ")


def test_kernel_compile_flags_difference_fails(tmp_path: Path) -> None:
    old_fp = _fingerprint(tmp_path / "old", ["-DVALUE=1"])
    new_fp = _fingerprint(tmp_path / "new", ["-DVALUE=2"])

    failed, _rows, _diagnostics = compare_artifact_sets(
        _payload([old_fp]), _payload([new_fp]), compile_insts=False
    )
    assert failed


def test_kernel_fingerprint_is_checkout_path_independent(tmp_path: Path) -> None:
    old_fp = _fingerprint(tmp_path / "checkout-a")
    new_fp = _fingerprint(tmp_path / "checkout-b")

    failed, _rows, diagnostics = compare_artifact_sets(
        _payload([old_fp]), _payload([new_fp]), compile_insts=False
    )
    assert not failed
    assert diagnostics == ["커널 입력 지문: 동일 (1개: fake_kernel)"]


def test_include_fingerprint_ignores_non_c_cpp_and_cache_files(tmp_path: Path) -> None:
    old_repo = tmp_path / "old"
    new_repo = tmp_path / "new"
    _fingerprint(old_repo)
    _fingerprint(new_repo)
    old_include = old_repo / "kernels" / "include"
    new_include = new_repo / "kernels" / "include"
    (old_include / "helper.py").write_text("old", encoding="utf-8")
    (new_include / "helper.py").write_text("changed", encoding="utf-8")
    (old_include / "README.md").write_text("old", encoding="utf-8")
    (new_include / "README.md").write_text("changed", encoding="utf-8")
    (new_include / "__pycache__").mkdir()
    (new_include / "__pycache__" / "x.pyc").write_bytes(b"cache")
    (new_include / ".generated").mkdir()
    (new_include / ".generated" / "hidden.h").write_bytes(b"hidden header")

    old_fp = kernel_input_fingerprint(
        old_repo,
        ("fake_kernel", old_repo / "kernels" / "fake.cc", ["memref<32xbf16>"]),
        {"include_dirs": [old_include]},
    )
    new_fp = kernel_input_fingerprint(
        new_repo,
        ("fake_kernel", new_repo / "kernels" / "fake.cc", ["memref<32xbf16>"]),
        {"include_dirs": [new_include]},
    )

    assert old_fp == new_fp


def test_include_header_difference_fails_with_sorted_function_names(tmp_path: Path) -> None:
    old_repo = tmp_path / "old"
    new_repo = tmp_path / "new"
    old_fp = _fingerprint(old_repo)
    _fingerprint(new_repo)
    (new_repo / "kernels" / "include" / "fake.h").write_bytes(b"changed header")
    new_fp = kernel_input_fingerprint(
        new_repo,
        ("fake_kernel", new_repo / "kernels" / "fake.cc", ["memref<32xbf16>"]),
        {
            "compile_flags": ["-DROWS=32"],
            "include_dirs": [new_repo / "kernels" / "include"],
            "cls": FakeKernel,
        },
    )
    old_other = {**old_fp, "function": "aaa_kernel"}
    new_other = {**new_fp, "function": "aaa_kernel"}

    failed, _rows, diagnostics = compare_artifact_sets(
        _payload([old_fp, old_other]), _payload([new_fp, new_other]), compile_insts=False
    )

    assert failed
    assert diagnostics[0] == (
        "커널 입력 지문: 다름 "
        "(baseline 2개: aaa_kernel, fake_kernel; current 2개: aaa_kernel, fake_kernel)"
    )


def test_per_round_insts_sha_difference_fails(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    failed, rows, _diagnostics = compare_artifact_sets(
        _payload([fingerprint], "before"),
        _payload([fingerprint], "after"),
        compile_insts=True,
    )
    assert failed
    assert rows[0]["insts_sha_equal"] is False


def test_empty_kernel_fingerprint_fails() -> None:
    failed, _rows, diagnostics = compare_artifact_sets(
        _payload([]), _payload([]), compile_insts=False
    )
    assert failed
    assert diagnostics == ["커널 입력 지문: 비어 있음"]


def test_per_call_shapes_match_backend_allowlist() -> None:
    from npu.backend import UNIFIED_PER_CALL_SHAPES

    assert PER_CALL_SHAPES == UNIFIED_PER_CALL_SHAPES


def test_complete_required_per_call_records_pass(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")

    failed, rows, _diagnostics = compare_artifact_sets(
        _payload([fingerprint]), _payload([fingerprint]), compile_insts=True
    )

    assert not failed
    required_rows = {row["name"]: row for row in rows if row["name"] in PER_CALL_SHAPES}
    assert required_rows.keys() == PER_CALL_SHAPES
    assert all(row["call_device_equal"] is True for row in required_rows.values())
    assert all(row["call_name_ok"] is True for row in required_rows.values())


def test_missing_required_per_call_fails(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    del new["records"]["f_b:per_call"]

    failed, rows, diagnostics = compare_artifact_sets(old, new, compile_insts=False)

    assert failed
    assert next(row for row in rows if row["name"] == "f_b")["call_device_equal"] is False
    assert "f_b:per_call: 필수 레코드 누락" in diagnostics


def test_per_call_name_without_pc_suffix_fails(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    new["records"]["f_b:per_call"]["name"] = "different_but_no_suffix"

    failed, rows, _diagnostics = compare_artifact_sets(old, new, compile_insts=False)

    assert failed
    assert next(row for row in rows if row["name"] == "f_b")["call_name_ok"] is False


def test_per_call_error_fails(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    new["records"]["f_b:per_call"] = {"error": "generation failed"}

    failed, rows, diagnostics = compare_artifact_sets(old, new, compile_insts=False)

    assert failed
    assert next(row for row in rows if row["name"] == "f_b")["call_name_ok"] is False
    assert "f_b:per_call: error=generation failed" in diagnostics


def test_per_call_device_body_difference_fails(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    new["records"]["f_b:per_call"]["device_norm_sha256"] = "different"

    failed, rows, _diagnostics = compare_artifact_sets(old, new, compile_insts=False)

    assert failed
    assert next(row for row in rows if row["name"] == "f_b")["call_device_equal"] is False


def test_empty_per_round_key_sets_fail(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = {"records": {}, "kernel_fingerprints": [fingerprint]}
    new = {"records": {}, "kernel_fingerprints": [fingerprint]}

    failed, _rows, diagnostics = compare_artifact_sets(old, new, compile_insts=False)

    assert failed
    assert "per_round 레코드 집합: 비어 있음" in diagnostics


def test_different_per_round_key_sets_fail(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    del new["records"]["fake:per_round"]

    failed, _rows, diagnostics = compare_artifact_sets(old, new, compile_insts=False)

    assert failed
    assert any(item.startswith("per_round 레코드 집합: 다름 ") for item in diagnostics)


@pytest.mark.parametrize("side", ["baseline", "current"])
@pytest.mark.parametrize(
    ("key", "compile_insts"),
    [
        ("name", False),
        ("mlir_norm_sha256", False),
        ("device_norm_sha256", False),
        ("insts_sha256", True),
    ],
)
def test_missing_comparison_key_fails(
    tmp_path: Path, side: str, key: str, compile_insts: bool
) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    payload = old if side == "baseline" else new
    del payload["records"]["fake:per_round"][key]

    failed, rows, _diagnostics = compare_artifact_sets(
        old, new, compile_insts=compile_insts
    )

    assert failed
    error_row = next(row for row in rows if row["name"] == "fake")
    assert error_row["error"] == f"missing comparison keys: {side}.{key}"


def test_missing_comparison_key_on_both_sides_fails(tmp_path: Path) -> None:
    fingerprint = _fingerprint(tmp_path / "repo")
    old = _payload([fingerprint])
    new = _payload([fingerprint])
    del old["records"]["fake:per_round"]["name"]
    del new["records"]["fake:per_round"]["name"]

    failed, rows, _diagnostics = compare_artifact_sets(
        old, new, compile_insts=False
    )

    assert failed
    error_row = next(row for row in rows if row["name"] == "fake")
    assert error_row["error"] == (
        "missing comparison keys: baseline.name, current.name"
    )


def test_main_prints_truthful_header_and_dashes_for_per_round_only_shapes(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    payload = _payload([{"function": "fake_kernel"}])
    responses = iter((payload, payload))
    monkeypatch.setattr(
        artifact_check, "_run_worker", lambda *_args, **_kwargs: next(responses)
    )
    args = types.SimpleNamespace(
        current=tmp_path / "current",
        baseline=tmp_path / "baseline",
        compile_insts=False,
    )

    assert artifact_check._main(args) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "shape round_equal round_name_equal call_device_equal call_name_ok "
        "words_before words_after txn_before txn_after patch_before patch_after "
        "raw_sha_diff norm_sha_equal insts_sha_equal"
    )
    assert next(line for line in lines if line.startswith("fake ")).startswith(
        "fake True True - - "
    )
    assert lines[-1] == "RESULT: PASS"
