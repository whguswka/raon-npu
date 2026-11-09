# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Compare baseline/current per-round artifacts and per-call device bodies.

Run this on the NPU notebook, where the ``aie`` Python bindings are installed.
The baseline tree must be an af44167 checkout/archive and is never modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Iterator


_C_CPP_INCLUDE_SUFFIXES = {
    ".h", ".hh", ".hpp", ".hxx", ".inc", ".ipp",
    ".c", ".cc", ".cpp", ".cxx",
}

PER_CALL_SHAPES = frozenset(
    {"f_b", "q_b", "k_b_v_b", "dense_gu", "moe_b1_6", "moe_b2_9"}
)
_PER_ROUND_COMPARE_KEYS = ("name", "mlir_norm_sha256", "device_norm_sha256")


def _sha(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def normalize_kernel_hashes(mlir: str, placeholder: str = "HASH") -> str:
    """Normalize directory-derived kernel object hashes in MLIR text.

    Kernel compilation derives an 8-hex object hash from the build path.
    If the hash starts with a digit, MLIR quotes the symbol name (e.g.
    @"628d8ca8_bfp16_gemv_unified_block"); if it starts with a letter, the
    symbol is bare (e.g. @fdc930c8_bfp16_gemv_unified_block). The linked
    object also embeds the hash (e.g. link_with = "bfp16_gemv_unified_block_628d8ca8.o").

    This function normalizes both forms into a deterministic placeholder
    so that genuine MLIR differences can be compared without path-hash jitter.
    """
    # 1. Object filename link_with: bfp16_gemv_unified_block_<hash>.o
    res = re.sub(
        r"bfp16_gemv_unified_block_[0-9a-fA-F]{8}\.o",
        f"bfp16_gemv_unified_block_{placeholder}.o",
        mlir,
    )
    # Generic .o link_with containing an 8-hex hash
    res = re.sub(
        r'(link_with\s*=\s*"[^"]*?)[0-9a-fA-F]{8}(\.o")',
        rf"\g<1>{placeholder}\2",
        res,
    )
    # 2. Quoted 8-hex prefix symbol: "628d8ca8_name" -> HASH_name (stripping quotes so it matches bare symbol)
    res = re.sub(
        r'"[0-9a-fA-F]{8}_([a-zA-Z0-9_]+)"',
        rf"{placeholder}_\1",
        res,
    )
    # 3. Bare 8-hex prefix symbol: fdc930c8_name -> HASH_name
    res = re.sub(
        r"\b[0-9a-fA-F]{8}_([a-zA-Z0-9_]+)\b",
        rf"{placeholder}_\1",
        res,
    )
    return res


def _repo_path(path: Path, repo: Path) -> str | None:
    """Return a repository-relative POSIX path, or ``None`` when external."""
    try:
        return path.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return None


def _include_dir_fingerprint(value: object, repo: Path) -> object:
    """Describe repository include trees without embedding checkout paths."""
    raw = os.fspath(value)
    path = Path(raw)
    relative = _repo_path(path, repo)
    if relative is None:
        return raw

    files = []
    if path.is_dir():
        # Only C/C++ compiler inputs outside hidden and __pycache__ directories belong here.
        candidates = (
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file()
            and candidate.suffix.lower() in _C_CPP_INCLUDE_SUFFIXES
            and all(
                part != "__pycache__" and not part.startswith(".")
                for part in candidate.relative_to(path).parts[:-1]
            )
        )
        for candidate in sorted(candidates):
            files.append(
                {
                    "path": candidate.relative_to(path).as_posix(),
                    "sha256": _sha(candidate.read_bytes()),
                }
            )
    return {"path": relative, "files": files}


def kernel_input_fingerprint(
    repo: Path, args: tuple[object, ...], kwargs: dict[str, object]
) -> dict[str, object]:
    """Build a stable fingerprint from one public ``_make_extern`` call."""
    func_name = args[0] if len(args) > 0 else kwargs.get("func_name")
    source_value = args[1] if len(args) > 1 else kwargs.get("source_path")
    arg_types = args[2] if len(args) > 2 else kwargs.get("arg_types")
    if source_value is None:
        raise ValueError("_make_extern call has no source_path")

    source = Path(os.fspath(source_value))
    source_relative = _repo_path(source, repo)
    source_path = source_relative if source_relative is not None else os.fspath(source_value)
    cls = kwargs.get("cls")
    return {
        "function": str(func_name),
        "source": {"path": source_path, "sha256": _sha(source.read_bytes())},
        "arg_types": repr(arg_types),
        "compile_flags": list(kwargs.get("compile_flags") or ()),
        "include_dirs": [
            _include_dir_fingerprint(value, repo)
            for value in (kwargs.get("include_dirs") or ())
        ],
        "cls": getattr(cls, "__name__", None),
    }


@contextmanager
def capture_kernel_inputs(repo: Path) -> Iterator[list[dict[str, object]]]:
    """Wrap imported repository kernel factories and record their public inputs."""
    captured: list[dict[str, object]] = []
    wrapped: list[tuple[object, object]] = []
    for module_name, module in tuple(sys.modules.items()):
        original = getattr(module, "_make_extern", None)
        if not module_name.startswith("kernels.") or not callable(original):
            continue

        @functools.wraps(original)
        def recording_make_extern(*args, __original=original, **kwargs):
            captured.append(kernel_input_fingerprint(repo, args, kwargs))
            return __original(*args, **kwargs)

        setattr(module, "_make_extern", recording_make_extern)
        wrapped.append((module, original))
    try:
        yield captured
    finally:
        for module, original in wrapped:
            setattr(module, "_make_extern", original)


def _fingerprint_set(values: list[dict[str, object]]) -> set[str]:
    return {json.dumps(value, sort_keys=True, separators=(",", ":")) for value in values}


def _fingerprint_names(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({
        str(value.get("function"))
        for value in values
        if isinstance(value, dict) and value.get("function") is not None
    })


def compare_artifact_sets(
    old: dict[str, object], new: dict[str, object], compile_insts: bool
) -> tuple[bool, list[dict[str, object]], list[str]]:
    """Compare worker payloads and return ``(failed, rows, diagnostics)``."""
    failed = False
    rows: list[dict[str, object]] = []
    diagnostics: list[str] = []
    old_records = old.get("records", {})
    new_records = new.get("records", {})
    assert isinstance(old_records, dict) and isinstance(new_records, dict)

    old_round_keys = {key for key in old_records if key.endswith(":per_round")}
    new_round_keys = {key for key in new_records if key.endswith(":per_round")}
    if not old_round_keys or not new_round_keys:
        diagnostics.append("per_round 레코드 집합: 비어 있음")
        failed = True
    if old_round_keys != new_round_keys:
        diagnostics.append(
            "per_round 레코드 집합: 다름 "
            f"(baseline only={sorted(old_round_keys - new_round_keys)}, "
            f"current only={sorted(new_round_keys - old_round_keys)})"
        )
        failed = True
    if "rda:per_round" not in old_round_keys or "rda:per_round" not in new_round_keys:
        diagnostics.append("per_round 레코드 집합: rda 누락")
        failed = True
    for shape in sorted(PER_CALL_SHAPES):
        round_key = f"{shape}:per_round"
        call_key = f"{shape}:per_call"
        if round_key not in new_round_keys:
            diagnostics.append(f"{round_key}: 필수 현재 레코드 누락")
            failed = True
        call = new_records.get(call_key)
        if not isinstance(call, dict):
            diagnostics.append(f"{call_key}: 필수 레코드 누락")
            failed = True
        elif "error" in call:
            diagnostics.append(f"{call_key}: error={call['error']}")
            failed = True

    names = sorted(
        key.removesuffix(":per_round")
        for key in old_round_keys & new_round_keys
    )
    for name in names:
        before = old_records[f"{name}:per_round"]
        now = new_records[f"{name}:per_round"]
        assert isinstance(before, dict) and isinstance(now, dict)
        if "error" in before or "error" in now:
            rows.append(
                {
                    "name": name,
                    "error": f"before={before.get('error')} now={now.get('error')}",
                }
            )
            failed = True
            continue

        compare_keys = list(_PER_ROUND_COMPARE_KEYS)
        if compile_insts:
            compare_keys.append("insts_sha256")
        missing_keys = [
            f"{side}.{key}"
            for side, record in (("baseline", before), ("current", now))
            for key in compare_keys
            if key not in record
        ]
        if missing_keys:
            rows.append(
                {
                    "name": name,
                    "error": "missing comparison keys: " + ", ".join(missing_keys),
                }
            )
            failed = True
            continue

        raw_equal = before.get("mlir_raw_sha256") == now.get("mlir_raw_sha256")
        norm_equal = before["mlir_norm_sha256"] == now["mlir_norm_sha256"]
        name_equal = before["name"] == now["name"]
        insts_equal: bool | None = None
        if compile_insts:
            insts_equal = before["insts_sha256"] == now["insts_sha256"]

        call_device_equal: bool | None = None
        call_name_ok: bool | None = None
        after_record = now
        if name in PER_CALL_SHAPES:
            call_key = f"{name}:per_call"
            call = new_records.get(call_key)
            if not isinstance(call, dict):
                call_device_equal = False
                call_name_ok = False
                failed = True
            elif "error" in call:
                call_device_equal = False
                call_name_ok = False
                after_record = call
                failed = True
            else:
                call_missing = [
                    key for key in ("name", "device_norm_sha256") if key not in call
                ]
                if call_missing:
                    diagnostics.append(
                        f"{call_key}: missing comparison keys: {', '.join(call_missing)}"
                    )
                    failed = True
                call_device_equal = (
                    "device_norm_sha256" in call
                    and now["device_norm_sha256"] == call["device_norm_sha256"]
                )
                call_name = call.get("name")
                call_name_ok = (
                    isinstance(call_name, str)
                    and call_name.endswith("_pc")
                    and call_name != now["name"]
                )
                after_record = call
                failed |= not (call_device_equal and call_name_ok)

        failed |= not (norm_equal and name_equal)
        if compile_insts:
            failed |= not bool(insts_equal)
        rows.append(
            {
                "name": name,
                "round_equal": norm_equal and name_equal,
                "round_name_equal": name_equal,
                "call_device_equal": call_device_equal,
                "call_name_ok": call_name_ok,
                "words_before": before.get("words"),
                "words_after": after_record.get("words"),
                "txn_before": before.get("txn_ops"),
                "txn_after": after_record.get("txn_ops"),
                "patch_before": before.get("ddr_patch"),
                "patch_after": after_record.get("ddr_patch"),
                "raw_sha_diff": not raw_equal,
                "norm_sha_equal": norm_equal,
                "insts_sha_equal": insts_equal,
            }
        )

    old_values = old.get("kernel_fingerprints", [])
    new_values = new.get("kernel_fingerprints", [])
    assert isinstance(old_values, list) and isinstance(new_values, list)
    old_fingerprints = _fingerprint_set(old_values)
    new_fingerprints = _fingerprint_set(new_values)
    old_names = _fingerprint_names(old_values)
    new_names = _fingerprint_names(new_values)
    if not old_fingerprints or not new_fingerprints:
        diagnostics.append("커널 입력 지문: 비어 있음")
        failed = True
    elif old_fingerprints != new_fingerprints:
        diagnostics.append(
            "커널 입력 지문: 다름 "
            f"(baseline {len(old_fingerprints)}개: {', '.join(old_names)}; "
            f"current {len(new_fingerprints)}개: {', '.join(new_names)})"
        )
        for item in sorted(old_fingerprints - new_fingerprints):
            diagnostics.append(f"  baseline only: {item}")
        for item in sorted(new_fingerprints - old_fingerprints):
            diagnostics.append(f"  current only: {item}")
        failed = True
    else:
        diagnostics.append(
            f"커널 입력 지문: 동일 ({len(old_fingerprints)}개: {', '.join(old_names)})"
        )
    return failed, rows, diagnostics


def _worker(repo: Path, out: Path, compile_insts: bool) -> int:
    repo = repo.resolve()
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parents[2]]
    sys.path.insert(0, str(repo))
    os.chdir(repo)

    import numpy as np
    from iron.common import AIEContext
    from kernels.gemv_unified.op import UnifiedGEMV
    from kernels.gemv_unified.shapes import ALL_SHAPES
    from kernels.moe.patch import TXN_OPC_DDR_PATCH, parse_txn_ops
    from bench.gemv_unified.check_device_invariance import device_without_runtime_sequence

    supports_plan = "sequence_plan" in UnifiedGEMV.for_shape.__func__.__code__.co_varnames
    records: dict[str, dict[str, object]] = {}
    out.mkdir(parents=True, exist_ok=True)
    capture_context = capture_kernel_inputs(repo)
    kernel_fingerprints = capture_context.__enter__()
    for shape in ALL_SHAPES:
        plans = (
            ("per_round", "per_call")
            if supports_plan and shape.name in PER_CALL_SHAPES
            else ("per_round",)
        )
        for plan in plans:
            kwargs = {
                "cols": 7,
                "rda_col": 7,
                "continuous_mem_w": True,
                "context": AIEContext(build_dir=out / "build"),
            }
            if supports_plan:
                kwargs["sequence_plan"] = plan
            op = UnifiedGEMV.for_shape(shape, **kwargs)
            mlir = op.get_mlir_artifact().generator()
            norm_mlir = normalize_kernel_hashes(mlir)
            dev = device_without_runtime_sequence(mlir)
            # Architecture/device lines remain part of this text. Only the
            # generated 8-hex kernel object/symbol suffixes are normalized.
            norm_dev = normalize_kernel_hashes(dev)
            key = f"{shape.name}:{plan}"
            record: dict[str, object] = {
                "name": op.name,
                "mlir_raw_sha256": _sha(mlir),
                "mlir_norm_sha256": _sha(norm_mlir),
                "device_raw_sha256": _sha(dev),
                "device_norm_sha256": _sha(norm_dev),
                # Compatibility aliases
                "mlir_sha256": _sha(norm_mlir),
                "device_sha256": _sha(norm_dev),
            }
            (out / f"{shape.name}-{plan}.mlir").write_text(mlir, encoding="utf-8")
            (out / f"{shape.name}-{plan}.norm.mlir").write_text(norm_mlir, encoding="utf-8")
            if compile_insts:
                op.compile()
                insts_path = Path(op.insts_artifact.filename)
                insts_bytes = insts_path.read_bytes()
                words = np.frombuffer(insts_bytes, dtype=np.uint32)
                ops = parse_txn_ops(words)
                counts = Counter(opcode for opcode, _pos, _size in ops)
                record.update(
                    {
                        "insts_path": str(insts_path.resolve()),
                        "insts_sha256": _sha(insts_bytes),
                        "words": int(words.size),
                        "txn_ops": len(ops),
                        "ddr_patch": counts[TXN_OPC_DDR_PATCH],
                    }
                )
            records[key] = record

    # RDA mode comparison (UnifiedRDA shares GEMV configuration xclbin)
    try:
        from kernels.gemv_unified.rda import UnifiedRDA
        rda_kwargs = {
            "cols": 7,
            "rda_col": 7,
            "continuous_mem_w": True,
            "context": AIEContext(build_dir=out / "build"),
        }
        if hasattr(UnifiedRDA, "__dataclass_fields__") and "sequence_plan" in UnifiedRDA.__dataclass_fields__:
            rda_kwargs["sequence_plan"] = "per_round"
        rda_op = UnifiedRDA(**rda_kwargs)
        rda_mlir = rda_op.get_mlir_artifact().generator()
        norm_rda_mlir = normalize_kernel_hashes(rda_mlir)
        rda_dev = device_without_runtime_sequence(rda_mlir)
        norm_rda_dev = normalize_kernel_hashes(rda_dev)
        rda_record: dict[str, object] = {
            "name": rda_op.name,
            "mlir_raw_sha256": _sha(rda_mlir),
            "mlir_norm_sha256": _sha(norm_rda_mlir),
            "device_raw_sha256": _sha(rda_dev),
            "device_norm_sha256": _sha(norm_rda_dev),
            "mlir_sha256": _sha(norm_rda_mlir),
            "device_sha256": _sha(norm_rda_dev),
        }
        (out / "rda-per_round.mlir").write_text(rda_mlir, encoding="utf-8")
        (out / "rda-per_round.norm.mlir").write_text(norm_rda_mlir, encoding="utf-8")
        if compile_insts:
            rda_op.compile()
            insts_path = Path(rda_op.insts_artifact.filename)
            insts_bytes = insts_path.read_bytes()
            words = np.frombuffer(insts_bytes, dtype=np.uint32)
            ops = parse_txn_ops(words)
            counts = Counter(opcode for opcode, _pos, _size in ops)
            rda_record.update(
                {
                    "insts_path": str(insts_path.resolve()),
                    "insts_sha256": _sha(insts_bytes),
                    "words": int(words.size),
                    "txn_ops": len(ops),
                    "ddr_patch": counts[TXN_OPC_DDR_PATCH],
                }
            )
        records["rda:per_round"] = rda_record
    except Exception as e:
        records["rda:per_round"] = {"error": str(e)}

    capture_context.__exit__(None, None, None)
    payload = {"records": records, "kernel_fingerprints": kernel_fingerprints}
    (out / "records.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


def _run_worker(script: Path, repo: Path, out: Path, compile_insts: bool) -> dict[str, object]:
    cmd = [sys.executable, str(script), "--worker", "--repo", str(repo), "--out", str(out)]
    if compile_insts:
        cmd.append("--compile-insts")
    subprocess.run(cmd, check=True, cwd=repo)
    return json.loads((out / "records.json").read_text(encoding="utf-8"))


def _main(args: argparse.Namespace) -> int:
    current = Path(args.current).resolve()
    baseline = Path(args.baseline).resolve()
    with tempfile.TemporaryDirectory(prefix="npu-opt7-artifacts-") as tmp:
        root = Path(tmp)
        old = _run_worker(Path(__file__).resolve(), baseline, root / "baseline", args.compile_insts)
        new = _run_worker(Path(__file__).resolve(), current, root / "current", args.compile_insts)
        failed, rows, diagnostics = compare_artifact_sets(old, new, args.compile_insts)
        print(
            "shape round_equal round_name_equal call_device_equal call_name_ok "
            "words_before words_after txn_before txn_after patch_before patch_after "
            "raw_sha_diff norm_sha_equal insts_sha_equal"
        )
        for row_data in rows:
            if "error" in row_data:
                print(f"{row_data['name']} ERROR {row_data['error']}")
                continue
            row = [
                str(row_data[key])
                for key in ("name", "round_equal", "round_name_equal")
            ]
            row.extend(
                "-" if row_data[key] is None else str(row_data[key])
                for key in ("call_device_equal", "call_name_ok")
            )
            if (
                args.compile_insts
                and row_data["words_before"] is not None
                and row_data["words_after"] is not None
            ):
                row.extend(
                    str(row_data[key])
                    for key in (
                        "words_before", "words_after", "txn_before", "txn_after",
                        "patch_before", "patch_after",
                    )
                )
            else:
                row.extend(["미확인"] * 6)
            row.extend(
                [
                    str(row_data["raw_sha_diff"]),
                    str(row_data["norm_sha_equal"]),
                    str(row_data["insts_sha_equal"])
                    if args.compile_insts
                    else "미확인",
                ]
            )
            print(" ".join(row))
        for diagnostic in diagnostics:
            print(diagnostic)
    if not args.compile_insts:
        print("insts.bin 비교: 안 함(--compile-insts 필요)")
    print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    return 1 if failed else 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, help="af44167 source tree")
    p.add_argument("--current", type=Path, default=Path.cwd())
    p.add_argument("--compile-insts", action="store_true")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--repo", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    return p


if __name__ == "__main__":
    ns = parser().parse_args()
    if ns.worker:
        if ns.repo is None or ns.out is None:
            raise SystemExit("--worker requires --repo and --out")
        raise SystemExit(_worker(ns.repo, ns.out, ns.compile_insts))
    if ns.baseline is None:
        raise SystemExit("--baseline is required")
    raise SystemExit(_main(ns))
