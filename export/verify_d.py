# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Verification tool for exported Raon-1-mini D mode (mixed_full) weights.

Verifies:
1. SHA256 integrity against `manifest.json`.
2. Bit-exact parity between decoded NPU layouts (parse_gemv / parse_bfp16_gemv)
   and `quant/eval_e2e.py`'s mixed_full expected weights.
3. Zero padding in padded rows and padded columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.bfp16 import (
    GEMV_BLOCK_BYTES,
    K_TILE,
    M_TILE,
    parse_bfp16_gemv,
    quantize_bfp16ebs8,
)
from quant.gguf_q4k import GGUFReader, GGUFWeightReader
from quant.q4nx import dequantize_simple, parse_gemv



RDA_LAYERS = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22}
GU_BYTES = 3481600
DN9_BYTES = 3686400

# Map both numpy dtype names and safetensors header dtype names to a common
# safetensors header representation so manifest ``stored_dtype``/``dtype`` can be
# compared with the on-disk safetensors header.
_DTYPE_TO_HEADER: dict[str, str] = {
    "float32": "F32",
    "bfloat16": "BF16",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "uint8": "U8",
    "uint16": "U16",
    "F32": "F32",
    "BF16": "BF16",
    "F16": "F16",
    "I64": "I64",
    "I32": "I32",
    "U8": "U8",
    "U16": "U16",
}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read just the JSON header of a safetensors file."""
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


def parse_expert_gate_up(buf: bytes | np.ndarray, e: int, rows: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse one expert from a full moe_b1 packed buffer."""
    u8 = np.frombuffer(buf, dtype=np.uint8) if isinstance(buf, (bytes, bytearray)) else buf.view(np.uint8).reshape(-1)
    chunk = u8[e * GU_BYTES : (e + 1) * GU_BYTES]
    return parse_gemv(chunk, K=2560, M=2176, rows=rows)


def parse_expert_down_bfp16(buf: bytes | np.ndarray, e: int, rows: int = 4) -> np.ndarray:
    """Parse one expert from a full moe_b2 packed buffer."""
    u8 = np.frombuffer(buf, dtype=np.uint8) if isinstance(buf, (bytes, bytearray)) else buf.view(np.uint8).reshape(-1)
    chunk = u8[e * DN9_BYTES : (e + 1) * DN9_BYTES]
    return parse_bfp16_gemv(chunk, K=1280, M=2560, rows=rows)


class DModeVerifier:
    """Verifies exported D mode weights against manifest and GGUF source."""

    def __init__(
        self,
        manifest_path: str | Path,
        weights_dir: str | Path | None = None,
        gguf_path: str | Path | None = None,
    ):
        self.manifest_path = Path(manifest_path)
        self.weights_dir = Path(weights_dir) if weights_dir else self.manifest_path.parent
        if gguf_path is None:
            raise ValueError("provide gguf_path / --gguf")
        self.gguf_path = Path(gguf_path)

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        print(f"[Verifier] Loading GGUF source from {self.gguf_path}...")
        self.reader = GGUFReader(self.gguf_path)
        self.weight_reader = GGUFWeightReader(self.reader, rda_layers=RDA_LAYERS)

    def verify_bundle(self, entry: dict[str, Any], expert_sample_count: int = 4) -> dict[str, Any]:
        """Verify a single bundle entry."""
        bundle_name = entry["bundle"]
        rel_file = entry["file"]
        file_path = self.weights_dir / rel_file
        fmt = entry["format"]

        # Decide classification up front so every early failure is bucketed correctly.
        check_type = "host_integrity" if fmt == "host_safetensors" else "npu_bit_exact"

        result: dict[str, Any] = {
            "bundle": bundle_name,
            "check_type": check_type,
            "status": "PASS",
            "sha_match": False,
            "bit_exact": None if check_type == "host_integrity" else True,
            "zero_padding_exact": None if check_type == "host_integrity" else True,
            "errors": [],
            "checked_tensors": [],
        }

        if not file_path.is_file():
            result["status"] = "FAIL"
            result["errors"].append(f"File not found: {file_path}")
            return result

        # 1. Check SHA256
        actual_sha = compute_sha256(file_path)
        expected_sha = entry["sha256"]
        if actual_sha != expected_sha:
            result["status"] = "FAIL"
            result["errors"].append(f"SHA256 mismatch: {actual_sha} != {expected_sha}")
            return result
        result["sha_match"] = True

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        if len(raw_bytes) != entry["size_bytes"]:
            result["status"] = "FAIL"
            result["errors"].append(f"Size mismatch: {len(raw_bytes)} != {entry['size_bytes']}")
            return result

        # Host safetensors are integrity-checked only (SHA256 + size + header).
        # Bit-exact verification against HF originals requires a separate --hf pass.
        if fmt == "host_safetensors":
            try:
                header = _read_safetensors_header(file_path)
            except Exception as exc:
                result["status"] = "FAIL"
                result["errors"].append(f"Failed to read safetensors header: {exc}")
                return result

            if not entry.get("components"):
                result["status"] = "FAIL"
                result["errors"].append("Host bundle has no 'components' in the manifest; nothing to verify against")
                return result

            expected_names = {comp["name"] for comp in entry["components"]}
            header_names = set(header) - {"__metadata__"}
            missing = expected_names - header_names
            extra = header_names - expected_names
            if missing:
                result["status"] = "FAIL"
                result["errors"].append(f"Header missing tensors: {sorted(missing)}")
                return result
            if extra:
                result["status"] = "FAIL"
                result["errors"].append(f"Header has extra tensors: {sorted(extra)}")
                return result

            # Manifest stored dtype/shape must match the real on-disk header.
            header_dtype_mismatches = []
            header_shape_mismatches = []
            for comp in entry["components"]:
                c_name = comp["name"]
                if c_name == "padding":
                    continue
                actual_header = header.get(c_name, {})
                actual_dtype = actual_header.get("dtype")

                # A manifest component without dtype/shape cannot be verified: fail
                # rather than skipping the comparison.
                missing_keys = [k for k in ("dtype", "shape") if comp.get(k) is None]
                if missing_keys:
                    header_dtype_mismatches.append(
                        f"{c_name}: manifest component is missing {missing_keys}; cannot verify"
                    )
                    continue

                # Original dtype bfloat16 always requires a BF16 header, regardless of
                # manifest stored_dtype. This catches pre-fix manifests that still record
                # stored_dtype="uint16" for tensors whose bits are actually bfloat16.
                if comp.get("dtype") == "bfloat16":
                    if actual_dtype != "BF16":
                        header_dtype_mismatches.append(
                            f"{c_name}: dtype mismatch (original dtype is bfloat16, "
                            f"expects header 'BF16', got header {actual_dtype!r})"
                        )
                else:
                    stored = comp.get("stored_dtype") or comp.get("dtype")
                    expected_header_dtype = _DTYPE_TO_HEADER.get(stored)
                    if expected_header_dtype is None:
                        header_dtype_mismatches.append(
                            f"{c_name}: manifest stored_dtype/dtype={stored!r} is not a recognized dtype"
                        )
                    elif actual_dtype != expected_header_dtype:
                        header_dtype_mismatches.append(
                            f"{c_name}: dtype mismatch (manifest stored_dtype={stored!r} "
                            f"expects header {expected_header_dtype!r}, got header {actual_dtype!r})"
                        )

                actual_shape = actual_header.get("shape")
                expected_shape = comp.get("shape")
                if actual_shape is None or list(actual_shape) != list(expected_shape):
                    header_shape_mismatches.append(
                        f"{c_name}: shape mismatch (manifest {list(expected_shape)!r}, "
                        f"header {None if actual_shape is None else list(actual_shape)!r})"
                    )

            if header_dtype_mismatches:
                result["status"] = "FAIL"
                result["errors"].append(f"Header dtype mismatch(es): {header_dtype_mismatches}")
            if header_shape_mismatches:
                result["status"] = "FAIL"
                result["errors"].append(f"Header shape mismatch(es): {header_shape_mismatches}")
            if result["status"] == "FAIL":
                return result

            result["integrity_only"] = True
            result["checked_tensors"] = []
            result["status"] = "PASS"
            return result

        # 2. Decode and verify bit equality according to format
        if fmt == "q4nx_gemv":
            m_pad, k_pad = entry["padded_shape"]
            codes, scales, mins = parse_gemv(raw_bytes, K=k_pad, M=m_pad, rows=4)
            restored = dequantize_simple(codes, scales, mins)

            for comp in entry.get("components", []):
                r_start = comp["row_start"]
                r_end = comp["row_end"]
                c_name = comp["name"]

                if comp.get("gguf_type") == "ZERO_PAD" or c_name == "padding":
                    pad_slice = restored[r_start:r_end, :]
                    if not np.all(pad_slice == 0.0):
                        result["zero_padding_exact"] = False
                        result["errors"].append(f"Padding rows [{r_start}:{r_end}] contain non-zero values")
                    continue

                expected = self.weight_reader.q4_restore(c_name)
                mat_slice = restored[r_start:r_end, :expected.shape[1]]
                if not np.array_equal(mat_slice.view(np.uint32), expected.view(np.uint32)):
                    result["bit_exact"] = False
                    result["errors"].append(f"Bit mismatch in component {c_name}")
                result["checked_tensors"].append(c_name)

        elif fmt == "bfp16_gemv":
            m_pad, k_pad = entry["padded_shape"]
            restored = parse_bfp16_gemv(raw_bytes, K=k_pad, M=m_pad, rows=4)

            # Check column padding if K was padded
            orig_k = entry["original_shape"][1]
            if k_pad > orig_k:
                col_pad = restored[:, orig_k:]
                if not np.all(col_pad == 0.0):
                    result["zero_padding_exact"] = False
                    result["errors"].append(f"Column padding [:, {orig_k}:{k_pad}] contains non-zero values")

            for comp in entry.get("components", []):
                c_name = comp["name"]

                if comp.get("gguf_type") == "ZERO_PAD" or c_name == "padding":
                    r_start = comp["row_start"]
                    r_end = comp["row_end"]
                    pad_slice = restored[r_start:r_end, :]
                    if not np.all(pad_slice == 0.0):
                        result["zero_padding_exact"] = False
                        result["errors"].append(f"Row padding [{r_start}:{r_end}] contains non-zero values")
                    continue

                gguf_name = comp.get("gguf_name")

                # Compute expected mixed_full weight via quantize_bfp16ebs8(rounding='conv_even')
                if "k_b" in bundle_name:
                    k_raw = self.reader.dequantize(gguf_name)
                    k_mat = np.ascontiguousarray(k_raw.transpose(0, 2, 1).reshape(-1, 512))
                    expected = quantize_bfp16ebs8(k_mat, rounding="conv_even")
                elif "v_b" in bundle_name:
                    v_raw = self.reader.dequantize(gguf_name)
                    v_mat = np.ascontiguousarray(v_raw.reshape(-1, 512))
                    expected = quantize_bfp16ebs8(v_mat, rounding="conv_even")
                else:
                    raw_f32 = self.reader.dequantize(gguf_name)
                    expected = quantize_bfp16ebs8(raw_f32, rounding="conv_even")

                if "row_start" in comp and "row_end" in comp:
                    r_start = comp["row_start"]
                    r_end = comp["row_end"]
                elif len(entry["components"]) == 1:
                    # Single-component bundles (f_b, o_proj, q_b, ...) legitimately carry no
                    # row range: the sole tensor starts at row 0.  The decoded rows are
                    # still compared bit-for-bit below, so a wrong default cannot pass.
                    r_start = 0
                    r_end = expected.shape[0]
                else:
                    result["bit_exact"] = False
                    result["errors"].append(
                        f"Component {c_name} of multi-component bundle lacks row_start/row_end in the manifest"
                    )
                    continue
                mat_slice = restored[r_start:r_end, :expected.shape[1]]

                if not np.array_equal(mat_slice.view(np.uint32), expected.view(np.uint32)):
                    result["bit_exact"] = False
                    max_diff = float(np.max(np.abs(mat_slice - expected)))
                    result["errors"].append(f"Bit mismatch in component {c_name} (max_diff={max_diff})")
                result["checked_tensors"].append(c_name)

        elif fmt == "q4nx_gemv_bundle":
            # moe_b1: verify a subset of experts
            layer = entry["layer"]
            expert_count = entry["expert_count"]
            check_indices = sorted(random.sample(range(expert_count), min(expert_sample_count, expert_count)))

            for e in check_indices:
                codes, scales, mins = parse_expert_gate_up(raw_bytes, e, rows=4)
                restored = dequantize_simple(codes, scales, mins)

                exp_g = self.weight_reader.q4_restore(f"model.layers.{layer}.moe.experts.{e}.gate_proj.weight")
                exp_u = self.weight_reader.q4_restore(f"model.layers.{layer}.moe.experts.{e}.up_proj.weight")
                expected = np.vstack([exp_g, exp_u])

                if not np.array_equal(restored.view(np.uint32), expected.view(np.uint32)):
                    result["bit_exact"] = False
                    result["errors"].append(f"Expert {e} gate+up bit mismatch")
                result["checked_tensors"].append(f"L{layer:02d}/moe_b1/exp_{e:03d}")

        elif fmt == "bfp16_gemv_bundle":
            # moe_b2: verify routed experts and shared expert
            layer = entry["layer"]
            expert_count = entry["expert_count"]
            routed_samples = random.sample(range(expert_count - 1), min(expert_sample_count, expert_count - 1))
            check_indices = sorted(routed_samples + [128])  # Always check shared expert (e=128)

            for e in check_indices:
                restored = parse_expert_down_bfp16(raw_bytes, e, rows=4)

                # Check column padding (1088..1280)
                col_pad = restored[:, 1088:]
                if not np.all(col_pad == 0.0):
                    result["zero_padding_exact"] = False
                    result["errors"].append(f"Expert {e} down column padding contains non-zero values")

                if e < 128:
                    raw_f32 = self.reader.dequantize(f"blk.{layer}.ffn_down_exps.weight", outer_index=e)
                else:
                    raw_f32 = self.reader.dequantize(f"blk.{layer}.ffn_down_shexp.weight")

                expected = quantize_bfp16ebs8(raw_f32, rounding="conv_even")
                mat_slice = restored[:, :1088]

                if not np.array_equal(mat_slice.view(np.uint32), expected.view(np.uint32)):
                    result["bit_exact"] = False
                    max_diff = float(np.max(np.abs(mat_slice - expected)))
                    result["errors"].append(f"Expert {e} down bit mismatch (max_diff={max_diff})")
                result["checked_tensors"].append(f"L{layer:02d}/moe_b2/exp_{e:03d}")

        else:
            # Unknown / misspelled / corrupted NPU format: nothing was compared bit for
            # bit, so it must never fall through to the PASS default.
            result["status"] = "FAIL"
            result["bit_exact"] = False
            result["zero_padding_exact"] = False
            result["errors"].append(f"Unsupported NPU format {fmt!r}: cannot verify bit-exactness")
            return result

        if not result["checked_tensors"] and result["bit_exact"] and result["zero_padding_exact"]:
            # Nothing was actually compared (e.g. empty components): not a PASS.
            result["bit_exact"] = False
            result["errors"].append(f"No tensors were bit-compared for format {fmt!r}")

        if not result["bit_exact"] or not result["zero_padding_exact"]:
            result["status"] = "FAIL"

        return result

    def verify(
        self,
        sample_count: int | None = 10,
        bundle_names: Sequence[str] | None = None,
        verify_all: bool = False,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Run verification over selected or sampled bundles."""
        random.seed(seed)
        all_bundles = self.manifest.get("bundles", [])
        host_bundles = [b for b in all_bundles if b.get("format") == "host_safetensors"]
        non_host_bundles = [b for b in all_bundles if b.get("format") != "host_safetensors"]

        if bundle_names:
            target_names = set(bundle_names)
            selected = [b for b in all_bundles if b["bundle"] in target_names]
            # A requested bundle name absent from the manifest is an error: silently
            # verifying 0 bundles and exiting 0 would hide typos and stale names.
            missing = sorted(target_names - {b["bundle"] for b in selected})
            if missing:
                raise ValueError(
                    f"Requested bundle(s) not present in manifest {self.manifest_path}: {missing}"
                )
        elif verify_all or sample_count is None or sample_count >= len(non_host_bundles):
            selected = all_bundles
        else:
            # Random evaluation targets only NPU bundles. Host bundles are verified
            # separately as integrity-only checks and reported as such.
            sampled = random.sample(non_host_bundles, min(sample_count, len(non_host_bundles)))
            selected = sampled + host_bundles

        if not selected:
            # Zero bundles verified must not read as "no failures".
            raise ValueError(f"No bundles selected for verification in manifest {self.manifest_path}")

        results = []
        integrity_checked = integrity_passed = integrity_failed = 0
        bit_exact_checked = bit_exact_passed = bit_exact_failed = 0

        for b in selected:
            b_name = b["bundle"]
            res = self.verify_bundle(b)
            results.append(res)
            ctype = res.get("check_type")
            status = res["status"]
            if ctype == "host_integrity":
                integrity_checked += 1
                if status == "PASS":
                    integrity_passed += 1
                    print(f"  [PASS] {b_name:30s} (integrity only: SHA256 + header verified)")
                else:
                    integrity_failed += 1
                    print(f"  [FAIL] {b_name:30s} ERRORS: {res['errors']}")
            else:
                bit_exact_checked += 1
                if status == "PASS":
                    bit_exact_passed += 1
                    print(f"  [PASS] {b_name:30s} ({len(res['checked_tensors'])} tensors checked, SHA256 verified)")
                else:
                    bit_exact_failed += 1
                    print(f"  [FAIL] {b_name:30s} ERRORS: {res['errors']}")

        failed = integrity_failed + bit_exact_failed

        # Integrity-only host checks cannot establish the advertised NPU
        # bit-exactness.  Treat a run that compared no NPU bundle as an overall
        # verification failure, regardless of how the bundles were selected.
        if bit_exact_checked == 0:
            failed += 1
            print("[Verifier] FAIL: no NPU bundles were bit-exact checked")

        # A manifest that records a failed host export is never a clean whole-manifest
        # verification, even when --bundles selects a surviving subset.
        host_status = self.manifest.get("host_tensors_status")
        if host_status is not None and host_status.get("status") != "ok":
            failed += 1
            print(f"[Verifier] FAIL: manifest records host_tensors_status={host_status}")

        summary = {
            "total_checked": len(selected),
            "failed": failed,
            "host_tensors_status": host_status,
            "integrity_checked": integrity_checked,
            "integrity_passed": integrity_passed,
            "integrity_failed": integrity_failed,
            "bit_exact_checked": bit_exact_checked,
            "bit_exact_passed": bit_exact_passed,
            "bit_exact_failed": bit_exact_failed,
            "results": results,
        }
        print(f"[Verifier] Integrity:  {integrity_checked} checked, {integrity_passed} passed, {integrity_failed} failed")
        print(f"[Verifier] Bit-exact: {bit_exact_checked} checked, {bit_exact_passed} passed, {bit_exact_failed} failed")
        return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify exported Raon-1-mini D mode weights")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to manifest.json")
    parser.add_argument("--weights-dir", type=Path, default=None, help="Path to weights directory")
    parser.add_argument("--gguf", type=Path, default=None, help="Path to GGUF file")
    parser.add_argument("--all", action="store_true", help="Verify all bundles in manifest")
    parser.add_argument("--sample-count", type=int, default=10, help="Number of random bundles to sample")
    parser.add_argument("--bundles", type=str, default=None, help="Comma-separated list of bundles to verify")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")

    parser.add_argument("--expected", type=Path, help="Portable sha256sum list; compare all files without loading GGUF")
    args = parser.parse_args()
    from export.artifacts import require_tokenizer_files, verify_expected
    if args.manifest is None and args.weights_dir is None:
        parser.error("provide --manifest or --weights-dir")
    weights = args.weights_dir or args.manifest.parent
    try:
        require_tokenizer_files(weights)
        if args.expected:
            errors = verify_expected(weights, args.expected)
            for error in errors:
                print(error)
            print("RESULT: FAIL" if errors else "RESULT: PASS")
            raise SystemExit(1 if errors else 0)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.gguf is None:
        parser.error("--gguf is required for bit-exact verification")
    if args.manifest is None:
        args.manifest = weights / "manifest.json"
    bundles = [b.strip() for b in args.bundles.split(",")] if args.bundles else None

    verifier = DModeVerifier(
        manifest_path=args.manifest,
        weights_dir=args.weights_dir,
        gguf_path=args.gguf,
    )
    res = verifier.verify(
        sample_count=args.sample_count,
        bundle_names=bundles,
        verify_all=args.all,
        seed=args.seed,
    )
    if res["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
