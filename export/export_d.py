# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""NPU weight export tool for Raon-1-mini D mode (mixed_full).

Packs eligible model weights into XDNA 2 NPU GEMV layouts with bit-exact parity
to `quant/eval_e2e.py`'s mixed_full mode:
- Q4_K tensors in GGUF -> code-preserving q4nx (pack_gemv)
- Non-Q4_K tensors -> GGUF dequantization + 9-bit bfp16ebs8 (pack_bfp16_gemv with flush_tiny=True)
- Non-target tensors (norm, router, mHC, conv1d, embeddings) -> preserved in HF precision
  (read via torch framework; the on-disk safetensors header stores BF16, and only during internal
  conversion do we temporarily view the bits as uint16 since numpy lacks bfloat16 here).

Outputs per-layer files under `L00/`..`L24/`, `lm_head.bin`, host safetensors, and `manifest.json`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable
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
    pad_bfp16_gemv,
    pack_bfp16_gemv,
    packed_bytes,
)
from quant.gguf_q4k import GGUFReader, GGUFWeightReader
from quant.q4nx import pack_gemv, pad_gemv



# Every bundle name accepted by --only / only_bundles (see ``DModeExporter._should_export``).
KNOWN_BUNDLE_NAMES = frozenset({
    "rda_in", "f_b", "o_proj", "mla_in", "q_b", "k_b", "v_b", "g",
    "dense_gu", "dense_down", "moe_b1", "moe_b2", "shared_gu", "lm_head", "host",
})

NUM_LAYERS = 25
RDA_LAYERS = {0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22}
MLA_LAYERS = {3, 7, 11, 15, 19, 23, 24}
NUM_ROUTED_EXPERTS = 128
NUM_TOTAL_EXPERTS = 129
VOCAB_SIZE = 259840
VOCAB_PADDED = 260096
HIDDEN_SIZE = 2560


def _write_json_atomic(path: Path, data: dict) -> None:
    """Write JSON to a unique temp file in the same directory, then os.replace() into place.

    An interrupted write can never leave a half-written manifest/index behind: readers
    see either the old complete file or the new complete one.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        encoding="utf-8",
    ) as tf:
        tmp_path = Path(tf.name)
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _tensor_bits(t) -> np.ndarray:
    """Raw bit pattern of a torch CPU tensor as a numpy array (bf16 -> uint16 bits)."""
    import torch

    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16)
    return t.numpy()


def compute_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def _build_incomplete_host_status(
    old_status: dict[str, Any] | None,
    missing_bundles: list[str],
    invalid_bundles: list[str],
    missing_tensors: list[str],
) -> dict[str, Any]:
    """Build a host_tensors_status entry, preserving an earlier failure reason if present."""
    if old_status is not None:
        error = old_status.get("error", "Host tensor export is incomplete")
        note = old_status.get("note", "")
    else:
        error = "Host tensor export is incomplete"
        note = ""

    details: list[str] = []
    if missing_bundles:
        details.append(f"missing bundles: {missing_bundles}")
    if missing_tensors:
        details.append(f"missing tensors: {missing_tensors}")
    if invalid_bundles:
        details.append(f"invalid bundles: {invalid_bundles}")
    detail_text = "; ".join(details)

    if note:
        note = f"{note}; {detail_text}" if detail_text else note
    else:
        note = detail_text

    return {
        "status": "failed",
        "error": error,
        "note": note,
    }


def _save_tensors_verified(torch_dict: dict, out_file: Path, expected_bits: dict[str, np.ndarray]) -> None:
    """Atomically write torch tensors to `out_file`, verifying the temp file before replace.

    Writes to a unique temp file in the same directory, reads the temp file back, and
    verifies tensor names, header dtypes, shapes, and exact bit patterns against
    `expected_bits` BEFORE `os.replace()` swaps it into place. On any failure the temp
    file is removed and the exception re-raised, leaving the previous `out_file` bytes
    untouched.
    """
    import safetensors.torch as st
    from safetensors import safe_open

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_file.parent, prefix=f".{out_file.name}.", suffix=".safetensors.tmp", delete=False,
    ) as tf:
        tmp_path = Path(tf.name)
    try:
        st.save_file(torch_dict, str(tmp_path))
        header = _read_safetensors_header(tmp_path)
        expected_names = set(torch_dict)
        with safe_open(str(tmp_path), framework="pt") as f:
            written_names = set(f.keys())
            if written_names != expected_names:
                raise RuntimeError(
                    f"Temp safetensors {tmp_path} has tensors {sorted(written_names ^ expected_names)} "
                    f"mismatching the expected set; refusing to replace {out_file}"
                )
            for name in sorted(expected_names):
                t = f.get_tensor(name)
                want = torch_dict[name]
                hdr_dtype = header.get(name, {}).get("dtype")
                if hdr_dtype != _torch_header_dtype(want):
                    raise RuntimeError(
                        f"Temp safetensors header dtype {hdr_dtype} != expected "
                        f"{_torch_header_dtype(want)} for {name}; refusing to replace {out_file}"
                    )
                if list(t.shape) != list(want.shape):
                    raise RuntimeError(
                        f"Temp safetensors shape {list(t.shape)} != expected {list(want.shape)} "
                        f"for {name}; refusing to replace {out_file}"
                    )
                if not np.array_equal(_tensor_bits(t).view(np.uint16), expected_bits[name].view(np.uint16)):
                    raise RuntimeError(
                        f"Bit pattern changed while writing {out_file} (tensor {name}); keeping original file"
                    )
        os.replace(tmp_path, out_file)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def write_and_record(
    out_path: Path,
    data: bytes | Iterable[bytes],
    bundle_name: str,
    layer_idx: int | None,
    fmt: str,
    orig_shape: list[int] | tuple[int, ...],
    padded_shape: list[int] | tuple[int, ...],
    components: list[dict[str, Any]],
    gguf_types: list[str],
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically write binary data to file and return a manifest entry.

    Data is written to a unique temp file in the same directory; the file size and
    SHA256 are computed from the complete temp file before ``os.replace()`` swaps it
    into place. On any failure the temp file is removed, leaving any previous file
    untouched.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent, prefix=f".{out_path.name}.", suffix=".tmp", delete=False,
    ) as tf:
        tmp_path = Path(tf.name)
    try:
        with open(tmp_path, "wb") as f:
            if isinstance(data, (bytes, bytearray)):
                f.write(data)
            else:
                for chunk in data:
                    f.write(chunk)
        size = tmp_path.stat().st_size
        sha = compute_sha256(tmp_path)
        os.replace(tmp_path, out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    record: dict[str, Any] = {
        "bundle": bundle_name,
        "layer": layer_idx,
        "format": fmt,
        "file": str(out_path.name) if layer_idx is None else f"{out_path.parent.name}/{out_path.name}",
        "size_bytes": size,
        "original_shape": list(orig_shape),
        "padded_shape": list(padded_shape),
        "components": components,
        "gguf_types": gguf_types,
        "sha256": sha,
    }
    if extra_meta:
        record.update(extra_meta)
    return record


def host_tensor_to_numpy(t) -> tuple[np.ndarray, str]:
    """Convert a torch CPU tensor (read with framework="pt") to numpy for host storage.

    This environment's numpy lacks bfloat16 (no ml_dtypes), so BF16 tensors are read via
    torch and returned as raw uint16 bit patterns: t.view(torch.int16).numpy().view(np.uint16).
    The caller is expected to re-encode them as torch.bfloat16 before writing so the
    safetensors header dtype becomes "BF16", which host/weights.py expects. All other
    dtypes (fp32/fp16/int) are returned as-is numpy arrays. Returns (array, original HF
    dtype string).
    """
    import torch

    if t.dtype == torch.bfloat16:
        return t.view(torch.int16).numpy().view(np.uint16), "bfloat16"
    arr = t.numpy()
    return arr, str(arr.dtype)


def _host_numpy_to_torch(arr: np.ndarray, src_dtype: str) -> Any:
    """Convert host numpy storage back to a torch tensor with the on-disk dtype.

    BF16 source tensors are stored as uint16 bit patterns in numpy; write them out as
    torch.bfloat16 so the safetensors header dtype is "BF16".  fp32/fp16/int tensors
    are written with their native dtype.
    """
    import torch

    if src_dtype == "bfloat16":
        return torch.from_numpy(arr.view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_header_dtype(t) -> str:
    """Return the safetensors header dtype string for a torch tensor."""
    import torch

    mapping = {
        torch.bfloat16: "BF16",
        torch.float32: "F32",
        torch.float16: "F16",
        torch.int64: "I64",
        torch.int32: "I32",
        torch.uint8: "U8",
    }
    return mapping.get(t.dtype, str(t.dtype))


def _save_host_safetensors(
    tensors: dict[str, np.ndarray],
    source_dtypes: dict[str, str],
    out_file: Path,
) -> list[dict[str, Any]]:
    """Atomically write a host safetensors file and return component metadata.

    Tensors are written to a unique temp file in the same directory via
    `_save_tensors_verified`, which verifies the temp file's header (tensor names,
    header dtypes, shapes) and bit patterns before `os.replace()`-ing it into place,
    so a failed write never damages the previous output file.
    """
    torch_dict = {
        name: _host_numpy_to_torch(tensors[name], source_dtypes[name])
        for name in tensors
    }
    _save_tensors_verified(
        torch_dict,
        out_file,
        {name: _tensor_bits(t) for name, t in torch_dict.items()},
    )

    return [
        {
            "name": name,
            "shape": list(tensors[name].shape),
            "dtype": source_dtypes[name],
            "stored_dtype": _torch_header_dtype(torch_dict[name]),
        }
        for name in tensors
    ]


def _read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read just the JSON header of a safetensors file."""
    import struct

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(header_len).decode("utf-8"))


def write_host_index(
    out_dir: Path,
    entries: list[dict[str, Any]],
    merge: bool = True,
) -> None:
    """Write `model.safetensors.index.json` mapping each host tensor to its file.

    Partial runs (`--layers` / `--only`) merge into an existing index so the aggregate
    file still points every host tensor to the correct shard.
    """
    index_path = out_dir / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    metadata: dict[str, Any] = {}

    if merge and index_path.is_file():
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                old_index = json.load(f)
            # Preserve metadata from the old index, but rebuild weight_map from the
            # merged manifest so stale tensor->file mappings from removed bundles
            # or shards do not survive.
            metadata.update(old_index.get("metadata", {}))
        except Exception as exc:
            print(f"[Exporter] Notice: failed to merge existing host index: {exc}")

    for entry in entries:
        if entry.get("format") != "host_safetensors":
            continue
        rel_file = entry.get("file")
        if not rel_file:
            raise RuntimeError(
                f"Host bundle {entry.get('bundle')} has no 'file' in the manifest; "
                f"refusing to write an index entry pointing to nothing"
            )
        for comp in entry["components"]:
            weight_map[comp["name"]] = rel_file

    index_data = {"metadata": metadata, "weight_map": weight_map}
    _write_json_atomic(index_path, index_data)


def _load_existing_manifest(manifest_path: Path) -> dict[str, Any] | None:
    """Load and validate an existing manifest before an export can touch outputs."""
    if not manifest_path.is_file():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not isinstance(manifest, dict):
            raise TypeError("top level is not an object")
        bundles = manifest.get("bundles")
        if not isinstance(bundles, list):
            raise TypeError("'bundles' is not a list")
        seen: set[str] = set()
        for pos, bundle in enumerate(bundles):
            if not isinstance(bundle, dict):
                raise TypeError(f"bundles[{pos}] is not an object")
            bundle_id = bundle.get("bundle")
            if not isinstance(bundle_id, str) or not bundle_id.strip():
                raise ValueError(f"bundles[{pos}] has no usable bundle identifier")
            if bundle_id in seen:
                raise ValueError(f"duplicate bundle identifier {bundle_id!r}")
            seen.add(bundle_id)
            size = bundle.get("size_bytes")
            if not isinstance(size, int) or isinstance(size, bool) or size < 0:
                raise ValueError(f"bundle {bundle_id!r} has invalid size_bytes {size!r}")
            layer = bundle.get("layer")
            if layer is not None and (not isinstance(layer, int) or isinstance(layer, bool)):
                raise ValueError(f"bundle {bundle_id!r} has invalid layer {layer!r}")
            if bundle.get("format") == "host_safetensors":
                rel_file = bundle.get("file")
                components = bundle.get("components")
                if not isinstance(rel_file, str) or not rel_file:
                    raise ValueError(f"host bundle {bundle_id!r} has no usable file")
                if not isinstance(components, list) or not components:
                    raise ValueError(f"host bundle {bundle_id!r} has no usable components")
                for c_pos, component in enumerate(components):
                    if not isinstance(component, dict) or not isinstance(component.get("name"), str) or not component["name"]:
                        raise ValueError(
                            f"host bundle {bundle_id!r} component {c_pos} has no usable name"
                        )
        return manifest
    except Exception as exc:
        raise RuntimeError(
            f"Existing manifest {manifest_path} could not be read/merged ({exc}); "
            f"refusing to touch output files. Fix or remove it, then re-run."
        ) from exc


def _preflight_existing_host_index(out_dir: Path) -> None:
    """Reject a malformed existing host index before any bundle is replaced."""
    index_path = out_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        if not isinstance(index, dict):
            raise TypeError("top level is not an object")
        if not isinstance(index.get("metadata", {}), dict):
            raise TypeError("'metadata' is not an object")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise TypeError("'weight_map' is not an object")
        for name, rel_file in weight_map.items():
            if not isinstance(name, str) or not name or not isinstance(rel_file, str) or not rel_file:
                raise ValueError("weight_map contains an invalid tensor name or file")
    except Exception as exc:
        raise RuntimeError(
            f"Existing host index {index_path} is invalid ({exc}); refusing to touch output files."
        ) from exc


class DModeExporter:
    """Exports Raon-1-mini weights into D mode (mixed_full) layout."""

    def __init__(
        self,
        gguf_path: str | Path,
        hf_dir: str | Path | None,
        out_dir: str | Path,
        layers: Sequence[int] | None = None,
        only_bundles: Sequence[str] | None = None,
        export_host: bool = True,
    ):
        self.gguf_path = Path(gguf_path)
        self.hf_dir = Path(hf_dir) if hf_dir is not None else None
        self.out_dir = Path(out_dir)
        self.layers = sorted(layers) if layers is not None else list(range(NUM_LAYERS))
        invalid_layers = [layer for layer in self.layers if not isinstance(layer, int) or isinstance(layer, bool) or not 0 <= layer < NUM_LAYERS]
        if invalid_layers:
            raise ValueError(f"Layer indices out of range 0..{NUM_LAYERS - 1}: {invalid_layers}")
        if layers is not None and not self.layers:
            raise ValueError("Explicit layer selection is empty; refusing a zero-layer export")
        self.only_bundles = set(only_bundles) if only_bundles is not None else None
        if self.only_bundles is not None:
            if not self.only_bundles:
                raise ValueError("only_bundles is empty; refusing an export that selects nothing")
            # A misspelled bundle name would otherwise select nothing and "succeed".
            unknown = sorted(self.only_bundles - KNOWN_BUNDLE_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown bundle name(s) in only_bundles: {unknown}; "
                    f"valid names: {sorted(KNOWN_BUNDLE_NAMES)}"
                )
        self.export_host = export_host

        if not self._planned_bundle_ids():
            raise ValueError(
                "The layer/only/skip-host selection produces no bundles; refusing a zero-bundle export"
            )

        print(f"[Exporter] Loading GGUF header from {self.gguf_path}...")
        self.reader = GGUFReader(self.gguf_path)
        self.weight_reader = GGUFWeightReader(self.reader, rda_layers=RDA_LAYERS)
        self.manifest_entries: list[dict[str, Any]] = []

    def _should_export(self, bundle_name: str) -> bool:
        if self.only_bundles is None:
            return True
        return bundle_name in self.only_bundles

    def _planned_bundle_ids(self) -> list[str]:
        """Return the concrete bundle ids selected by the current arguments."""
        planned: list[str] = []
        rda_names = ("rda_in", "f_b", "o_proj", "dense_gu", "dense_down")
        mla_names = ("mla_in", "q_b", "k_b", "v_b", "g", "o_proj", "moe_b1", "moe_b2", "shared_gu")
        for layer in self.layers:
            names = rda_names if layer in RDA_LAYERS else mla_names
            if layer != 0:
                names = tuple(name for name in names if name not in {"dense_gu", "dense_down"})
            for name in names:
                if self._should_export(name):
                    planned.append(f"L{layer:02d}/{name}")
        if self._should_export("lm_head"):
            planned.append("lm_head")
        if self.export_host and self._should_export("host"):
            planned.append("host")
        return planned

    def _preflight_host_inputs(self) -> None:
        """Validate every selected HF shard/tensor before NPU or host output writes."""
        if not (self.export_host and self._should_export("host")):
            return
        if self.hf_dir is None or not self.hf_dir.is_dir():
            raise FileNotFoundError(
                f"HF directory {self.hf_dir} is missing or not a directory; "
                f"host tensor export cannot proceed."
            )
        weight_map = self._load_hf_weight_map()
        layer_weights, global_weights = self._collect_host_tensor_names(self.layers, weight_map)
        required = set(global_weights)
        for names in layer_weights.values():
            required.update(names)
        if not required:
            raise ValueError("Host export selected no tensors from the HF weight map")

        from safetensors import safe_open
        by_shard: dict[str, set[str]] = {}
        for name in required:
            shard = weight_map.get(name)
            if not isinstance(shard, str) or not shard:
                raise ValueError(f"HF weight_map has no usable shard for required tensor {name}")
            by_shard.setdefault(shard, set()).add(name)
        for shard, names in sorted(by_shard.items()):
            shard_path = self.hf_dir / shard
            if not shard_path.is_file():
                raise FileNotFoundError(f"HF shard {shard_path} is missing")
            with safe_open(str(shard_path), framework="pt") as f:
                missing = names - set(f.keys())
            if missing:
                raise KeyError(f"Host tensors {sorted(missing)} missing from shard {shard}")

    def _preflight_gguf_inputs(self) -> None:
        """Validate selected GGUF tensor metadata and payload bounds before writing."""
        selected = set(self._planned_bundle_ids())
        selected.discard("host")
        if not selected:
            return

        expected: dict[str, tuple[int, ...]] = {}
        for layer in self.layers:
            prefix = f"L{layer:02d}/"
            if layer in RDA_LAYERS:
                if prefix + "rda_in" in selected:
                    expected.update({
                        f"blk.{layer}.attn_q.weight": (5120, HIDDEN_SIZE),
                        f"blk.{layer}.attn_k.weight": (5120, HIDDEN_SIZE),
                        f"blk.{layer}.attn_v.weight": (5120, HIDDEN_SIZE),
                        f"blk.{layer}.ssm_g.weight": (5120, HIDDEN_SIZE),
                        f"blk.{layer}.ssm_f_a.weight": (128, HIDDEN_SIZE),
                        f"blk.{layer}.ssm_beta.weight": (40, HIDDEN_SIZE),
                        f"blk.{layer}.ssm_rda.weight": (40, HIDDEN_SIZE),
                    })
                if prefix + "f_b" in selected:
                    expected[f"blk.{layer}.ssm_f_b.weight"] = (5120, 128)
            else:
                if prefix + "mla_in" in selected:
                    expected[f"blk.{layer}.attn_q_a.weight"] = (576, HIDDEN_SIZE)
                    expected[f"blk.{layer}.attn_kv_a_mqa.weight"] = (576, HIDDEN_SIZE)
                if prefix + "q_b" in selected:
                    expected[f"blk.{layer}.attn_q_b.weight"] = (7680, 576)
                if prefix + "k_b" in selected:
                    expected[f"blk.{layer}.attn_k_b.weight"] = (40, 512, 128)
                if prefix + "v_b" in selected:
                    expected[f"blk.{layer}.attn_v_b.weight"] = (40, 128, 512)
                if prefix + "g" in selected:
                    expected[f"blk.{layer}.attn_gate.weight"] = (5120, HIDDEN_SIZE)
                if prefix + "moe_b1" in selected:
                    expected[f"blk.{layer}.ffn_gate_exps.weight"] = (128, 1088, HIDDEN_SIZE)
                    expected[f"blk.{layer}.ffn_up_exps.weight"] = (128, 1088, HIDDEN_SIZE)
                if prefix + "moe_b2" in selected:
                    expected[f"blk.{layer}.ffn_down_exps.weight"] = (128, HIDDEN_SIZE, 1088)
                    expected[f"blk.{layer}.ffn_down_shexp.weight"] = (HIDDEN_SIZE, 1088)
                if prefix + "shared_gu" in selected:
                    expected[f"blk.{layer}.ffn_gate_shexp.weight"] = (1088, HIDDEN_SIZE)
                    expected[f"blk.{layer}.ffn_up_shexp.weight"] = (1088, HIDDEN_SIZE)
            if prefix + "o_proj" in selected:
                expected[f"blk.{layer}.attn_output.weight"] = (HIDDEN_SIZE, 5120)

        if "L00/dense_gu" in selected:
            expected["blk.0.ffn_gate.weight"] = (12032, HIDDEN_SIZE)
            expected["blk.0.ffn_up.weight"] = (12032, HIDDEN_SIZE)
        if "L00/dense_down" in selected:
            expected["blk.0.ffn_down.weight"] = (HIDDEN_SIZE, 12032)
        if "lm_head" in selected:
            expected["output.weight"] = (VOCAB_SIZE, HIDDEN_SIZE)

        file_size = self.gguf_path.stat().st_size
        for name, shape in expected.items():
            info = self.reader.tensors.get(name)
            if info is None:
                raise KeyError(f"Selected GGUF tensor {name} is missing")
            if info.shape != shape:
                raise ValueError(f"Shape mismatch for {name}: {info.shape} != {shape}")
            try:
                payload_end = self.reader.data_offset + info.offset + info.nbytes
            except Exception as exc:
                raise ValueError(f"Selected GGUF tensor {name} has an unsupported layout: {exc}") from exc
            if payload_end > file_size:
                raise EOFError(
                    f"Selected GGUF tensor {name} payload ends at {payload_end}, beyond file size {file_size}"
                )

    def export_rda_in(self, layer: int) -> dict[str, Any]:
        """Export RDA rda_in bundle: [q; k; v; g; f_a; b; rda_w] (M=20688 -> 20736, K=2560)."""
        tensors_spec = [
            ("q", f"blk.{layer}.attn_q.weight", f"model.layers.{layer}.self_attn.q_proj.weight", 5120),
            ("k", f"blk.{layer}.attn_k.weight", f"model.layers.{layer}.self_attn.k_proj.weight", 5120),
            ("v", f"blk.{layer}.attn_v.weight", f"model.layers.{layer}.self_attn.v_proj.weight", 5120),
            ("g", f"blk.{layer}.ssm_g.weight", f"model.layers.{layer}.self_attn.g_proj.weight", 5120),
            ("f_a", f"blk.{layer}.ssm_f_a.weight", f"model.layers.{layer}.self_attn.f_a_proj.weight", 128),
            ("b", f"blk.{layer}.ssm_beta.weight", f"model.layers.{layer}.self_attn.b_proj.weight", 40),
            ("rda_w", f"blk.{layer}.ssm_rda.weight", f"model.layers.{layer}.self_attn.rda_w_proj.weight", 40),
        ]

        slices = []
        components = []
        gguf_types = []
        row_offset = 0

        for short_name, gguf_name, hf_name, expected_rows in tensors_spec:
            mat = self.reader.dequantize(gguf_name)
            info = self.reader.tensors[gguf_name]
            gguf_types.append(info.type_name)
            if mat.shape != (expected_rows, HIDDEN_SIZE):
                raise ValueError(f"Shape mismatch for {gguf_name}: {mat.shape} != ({expected_rows}, {HIDDEN_SIZE})")
            slices.append(mat)
            components.append({
                "name": hf_name,
                "gguf_name": gguf_name,
                "short_name": short_name,
                "row_start": row_offset,
                "row_end": row_offset + expected_rows,
                "shape": [expected_rows, HIDDEN_SIZE],
                "gguf_type": info.type_name,
            })
            row_offset += expected_rows

        assert row_offset == 20688, f"Expected 20688 total rows for rda_in, got {row_offset}"
        stacked = np.vstack(slices)
        del slices
        gc.collect()

        padded, orig_shape = pad_bfp16_gemv(stacked, m_align=128, k_align=256)
        del stacked
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        components.append({
            "name": "padding",
            "short_name": "pad",
            "row_start": 20688,
            "row_end": 20736,
            "shape": [48, HIDDEN_SIZE],
            "gguf_type": "ZERO_PAD",
        })

        out_file = self.out_dir / f"L{layer:02d}" / "rda_in.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/rda_in",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=(20736, HIDDEN_SIZE),
            components=components,
            gguf_types=list(dict.fromkeys(gguf_types)),
            extra_meta={"description": "RDA input projection bundle [q; k; v; g; f_a; b; rda_w]"},
        )

    def export_f_b(self, layer: int) -> dict[str, Any]:
        """Export RDA f_b projection: (5120, 128) -> K padded to 256."""
        gguf_name = f"blk.{layer}.ssm_f_b.weight"
        hf_name = f"model.layers.{layer}.self_attn.f_b_proj.weight"
        mat = self.reader.dequantize(gguf_name)
        info = self.reader.tensors[gguf_name]
        if mat.shape != (5120, 128):
            raise ValueError(f"Shape mismatch for {gguf_name}: {mat.shape} != (5120, 128)")

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "f_b.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/f_b",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=(5120, 256),
            components=[{
                "name": hf_name,
                "gguf_name": gguf_name,
                "shape": [5120, 128],
                "padded_shape": [5120, 256],
                "gguf_type": info.type_name,
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "RDA f_b projection (K padded 128->256)"},
        )

    def export_o_proj(self, layer: int) -> dict[str, Any]:
        """Export mixer o_proj: (2560, 5120)."""
        gguf_name = f"blk.{layer}.attn_output.weight"
        hf_name = f"model.layers.{layer}.self_attn.o_proj.weight"
        mat = self.reader.dequantize(gguf_name)
        info = self.reader.tensors[gguf_name]
        if mat.shape != (2560, 5120):
            raise ValueError(f"Shape mismatch for {gguf_name}: {mat.shape} != (2560, 5120)")

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "o_proj.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/o_proj",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[{
                "name": hf_name,
                "gguf_name": gguf_name,
                "shape": [2560, 5120],
                "gguf_type": info.type_name,
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "Mixer output projection (o_proj)"},
        )

    def export_mla_in(self, layer: int) -> dict[str, Any]:
        """Export MLA mla_in bundle: [q_a; kv_a] (M=1152, K=2560)."""
        q_a_name = f"blk.{layer}.attn_q_a.weight"
        kv_a_name = f"blk.{layer}.attn_kv_a_mqa.weight"

        q_a = self.reader.dequantize(q_a_name)
        kv_a = self.reader.dequantize(kv_a_name)
        q_a_info = self.reader.tensors[q_a_name]
        kv_a_info = self.reader.tensors[kv_a_name]

        stacked = np.vstack([q_a, kv_a])
        del q_a, kv_a
        gc.collect()

        padded, orig_shape = pad_bfp16_gemv(stacked, m_align=128, k_align=256)
        del stacked
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "mla_in.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/mla_in",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[
                {
                    "name": f"model.layers.{layer}.self_attn.q_a_proj.weight",
                    "gguf_name": q_a_name,
                    "row_start": 0,
                    "row_end": 576,
                    "shape": [576, HIDDEN_SIZE],
                    "gguf_type": q_a_info.type_name,
                },
                {
                    "name": f"model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight",
                    "gguf_name": kv_a_name,
                    "row_start": 576,
                    "row_end": 1152,
                    "shape": [576, HIDDEN_SIZE],
                    "gguf_type": kv_a_info.type_name,
                },
            ],
            gguf_types=list(dict.fromkeys([q_a_info.type_name, kv_a_info.type_name])),
            extra_meta={"description": "MLA input projection bundle [q_a; kv_a]"},
        )

    def export_mla_q_b(self, layer: int) -> dict[str, Any]:
        """Export MLA q_b projection: (7680, 576) -> K padded to 768."""
        gguf_name = f"blk.{layer}.attn_q_b.weight"
        hf_name = f"model.layers.{layer}.self_attn.q_b_proj.weight"
        mat = self.reader.dequantize(gguf_name)
        info = self.reader.tensors[gguf_name]

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "q_b.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/q_b",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=(7680, 768),
            components=[{
                "name": hf_name,
                "gguf_name": gguf_name,
                "shape": [7680, 576],
                "padded_shape": [7680, 768],
                "gguf_type": info.type_name,
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "MLA q_b projection (K padded 576->768)"},
        )

    def export_mla_k_b(self, layer: int) -> dict[str, Any]:
        """Export MLA k_b projection: (40, 512, 128) -> transposed to (5120, 512)."""
        gguf_name = f"blk.{layer}.attn_k_b.weight"
        mat_raw = self.reader.dequantize(gguf_name)  # [40, 512, 128]
        info = self.reader.tensors[gguf_name]
        # Transpose to [40, 128, 512] and flatten to [5120, 512]
        mat = np.ascontiguousarray(mat_raw.transpose(0, 2, 1).reshape(-1, 512))
        del mat_raw
        gc.collect()

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "k_b.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/k_b",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[{
                "name": f"model.layers.{layer}.self_attn.kv_b_proj.weight (key_part)",
                "gguf_name": gguf_name,
                "shape": [5120, 512],
                "raw_gguf_shape": [40, 512, 128],
                "gguf_type": info.type_name,
                "hf_interleave_rule": "Head h=0..39 keys occupy rows [h*256 : h*256+128] in HF kv_b_proj.weight",
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "MLA k_b projection (transposed [40, 128, 512] -> [5120, 512])"},
        )

    def export_mla_v_b(self, layer: int) -> dict[str, Any]:
        """Export MLA v_b projection: (40, 128, 512) -> flattened to (5120, 512)."""
        gguf_name = f"blk.{layer}.attn_v_b.weight"
        mat_raw = self.reader.dequantize(gguf_name)  # [40, 128, 512]
        info = self.reader.tensors[gguf_name]
        mat = np.ascontiguousarray(mat_raw.reshape(-1, 512))
        del mat_raw
        gc.collect()

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "v_b.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/v_b",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[{
                "name": f"model.layers.{layer}.self_attn.kv_b_proj.weight (val_part)",
                "gguf_name": gguf_name,
                "shape": [5120, 512],
                "raw_gguf_shape": [40, 128, 512],
                "gguf_type": info.type_name,
                "hf_interleave_rule": "Head h=0..39 values occupy rows [h*256+128 : (h+1)*256] in HF kv_b_proj.weight",
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "MLA v_b projection ([40, 128, 512] -> [5120, 512])"},
        )

    def export_mla_g(self, layer: int) -> dict[str, Any]:
        """Export MLA attention gate: (5120, 2560)."""
        gguf_name = f"blk.{layer}.attn_gate.weight"
        hf_name = f"model.layers.{layer}.self_attn.g_proj.weight"
        mat = self.reader.dequantize(gguf_name)
        info = self.reader.tensors[gguf_name]

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "g.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/g",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[{
                "name": hf_name,
                "gguf_name": gguf_name,
                "shape": [5120, HIDDEN_SIZE],
                "gguf_type": info.type_name,
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "MLA attention gate (attn_gate / g_proj)"},
        )

    def export_dense_gu(self) -> dict[str, Any]:
        """Export layer 0 dense MLP gate+up: [gate; up] (24064, 2560) q4nx."""
        gate_name = "blk.0.ffn_gate.weight"
        up_name = "blk.0.ffn_up.weight"

        g_codes, g_scales, g_mins = self.reader.q4nx_components(gate_name)
        u_codes, u_scales, u_mins = self.reader.q4nx_components(up_name)
        g_info = self.reader.tensors[gate_name]
        u_info = self.reader.tensors[up_name]

        codes = np.vstack([g_codes, u_codes])
        scales = np.vstack([g_scales, u_scales])
        mins = np.vstack([g_mins, u_mins])
        del g_codes, g_scales, g_mins, u_codes, u_scales, u_mins
        gc.collect()

        packed = pack_gemv(codes, scales, mins, rows=4)
        del codes, scales, mins
        gc.collect()

        out_file = self.out_dir / "L00" / "dense_gu.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name="L00/dense_gu",
            layer_idx=0,
            fmt="q4nx_gemv",
            orig_shape=(24064, HIDDEN_SIZE),
            padded_shape=(24064, HIDDEN_SIZE),
            components=[
                {
                    "name": "model.layers.0.mlp.gate_proj.weight",
                    "gguf_name": gate_name,
                    "row_start": 0,
                    "row_end": 12032,
                    "shape": [12032, HIDDEN_SIZE],
                    "gguf_type": g_info.type_name,
                },
                {
                    "name": "model.layers.0.mlp.up_proj.weight",
                    "gguf_name": up_name,
                    "row_start": 12032,
                    "row_end": 24064,
                    "shape": [12032, HIDDEN_SIZE],
                    "gguf_type": u_info.type_name,
                },
            ],
            gguf_types=[g_info.type_name, u_info.type_name],
            extra_meta={"description": "Layer 0 dense MLP gate+up projection bundle (q4nx)"},
        )

    def export_dense_down(self) -> dict[str, Any]:
        """Export layer 0 dense MLP down: (2560, 12032) 9-bit bfp16."""
        gguf_name = "blk.0.ffn_down.weight"
        hf_name = "model.layers.0.mlp.down_proj.weight"
        mat = self.reader.dequantize(gguf_name)
        info = self.reader.tensors[gguf_name]

        padded, orig_shape = pad_bfp16_gemv(mat, m_align=128, k_align=256)
        del mat
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / "L00" / "dense_down.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name="L00/dense_down",
            layer_idx=0,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[{
                "name": hf_name,
                "gguf_name": gguf_name,
                "shape": [2560, 12032],
                "gguf_type": info.type_name,
            }],
            gguf_types=[info.type_name],
            extra_meta={"description": "Layer 0 dense MLP down projection (9-bit bfp16)"},
        )

    def export_moe_b1(self, layer: int) -> dict[str, Any]:
        """Export MoE moe_b1: 128 routed experts gate+up q4nx (e=0..127, shared expert excluded)."""
        gate_exps_name = f"blk.{layer}.ffn_gate_exps.weight"
        up_exps_name = f"blk.{layer}.ffn_up_exps.weight"
        g_info = self.reader.tensors[gate_exps_name]
        u_info = self.reader.tensors[up_exps_name]

        out_file = self.out_dir / f"L{layer:02d}" / "moe_b1.bin"

        expert_bytes = 3481600

        def _pack_all_experts():
            for e in range(NUM_ROUTED_EXPERTS):
                g_c, g_s, g_m = self.reader.q4nx_components(gate_exps_name, outer_index=e)
                u_c, u_s, u_m = self.reader.q4nx_components(up_exps_name, outer_index=e)

                c = np.vstack([g_c, u_c])
                s = np.vstack([g_s, u_s])
                m = np.vstack([g_m, u_m])
                del g_c, g_s, g_m, u_c, u_s, u_m

                packed_e = pack_gemv(c, s, m, rows=4)
                del c, s, m
                if len(packed_e) != expert_bytes:
                    raise ValueError(f"Expert {e} packed size mismatch: {len(packed_e)} != {expert_bytes}")
                yield packed_e
                del packed_e

        record = write_and_record(
            out_file,
            _pack_all_experts(),
            bundle_name=f"L{layer:02d}/moe_b1",
            layer_idx=layer,
            fmt="q4nx_gemv_bundle",
            orig_shape=(NUM_ROUTED_EXPERTS, 2176, HIDDEN_SIZE),
            padded_shape=(NUM_ROUTED_EXPERTS, 2176, HIDDEN_SIZE),
            components=[
                {
                    "name": f"model.layers.{layer}.moe.experts.0..127.(gate|up)_proj.weight",
                    "gguf_name": f"{gate_exps_name} + {up_exps_name}",
                    "shape_per_expert": [2176, HIDDEN_SIZE],
                    "expert_range": [0, 128],
                    "gguf_type": g_info.type_name,
                }
            ],
            gguf_types=[g_info.type_name, u_info.type_name],
            extra_meta={
                "expert_count": NUM_ROUTED_EXPERTS,
                "expert_bytes": expert_bytes,
                "description": "128 routed experts gate+up q4nx (shared expert excluded)",
            },
        )
        gc.collect()
        return record

    def export_moe_b2(self, layer: int) -> dict[str, Any]:
        """Export MoE moe_b2: 128 routed experts + 1 shared expert (e=128) down 9-bit bfp16."""
        down_exps_name = f"blk.{layer}.ffn_down_exps.weight"
        down_shexp_name = f"blk.{layer}.ffn_down_shexp.weight"
        d_info = self.reader.tensors[down_exps_name]
        sh_info = self.reader.tensors[down_shexp_name]

        out_file = self.out_dir / f"L{layer:02d}" / "moe_b2.bin"

        expert_bytes = 3686400

        def _pack_all_experts():
            # 1. 128 routed experts (e=0..127)
            for e in range(NUM_ROUTED_EXPERTS):
                w = self.reader.dequantize(down_exps_name, outer_index=e)  # [2560, 1088]
                padded, _ = pad_bfp16_gemv(w, m_align=128, k_align=256)   # [2560, 1280]
                del w
                packed_e = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
                del padded
                if len(packed_e) != expert_bytes:
                    raise ValueError(f"Expert {e} packed size mismatch: {len(packed_e)} != {expert_bytes}")
                yield packed_e
                del packed_e

            # 2. 1 shared expert (e=128)
            w_sh = self.reader.dequantize(down_shexp_name)  # [2560, 1088]
            padded_sh, _ = pad_bfp16_gemv(w_sh, m_align=128, k_align=256)
            del w_sh
            packed_sh = pack_bfp16_gemv(padded_sh, rows=4, flush_tiny=True)
            del padded_sh
            if len(packed_sh) != expert_bytes:
                raise ValueError(f"Shared expert packed size mismatch: {len(packed_sh)} != {expert_bytes}")
            yield packed_sh
            del packed_sh

        record = write_and_record(
            out_file,
            _pack_all_experts(),
            bundle_name=f"L{layer:02d}/moe_b2",
            layer_idx=layer,
            fmt="bfp16_gemv_bundle",
            orig_shape=(NUM_TOTAL_EXPERTS, HIDDEN_SIZE, 1088),
            padded_shape=(NUM_TOTAL_EXPERTS, HIDDEN_SIZE, 1280),
            components=[
                {
                    "name": f"model.layers.{layer}.moe.experts.0..127.down_proj.weight",
                    "gguf_name": down_exps_name,
                    "shape_per_expert": [HIDDEN_SIZE, 1088],
                    "padded_shape_per_expert": [HIDDEN_SIZE, 1280],
                    "expert_range": [0, 128],
                    "gguf_type": d_info.type_name,
                },
                {
                    "name": f"model.layers.{layer}.moe.shared_experts.down_proj.weight",
                    "gguf_name": down_shexp_name,
                    "shape_per_expert": [HIDDEN_SIZE, 1088],
                    "padded_shape_per_expert": [HIDDEN_SIZE, 1280],
                    "expert_id": 128,
                    "gguf_type": sh_info.type_name,
                },
            ],
            gguf_types=list(dict.fromkeys([d_info.type_name, sh_info.type_name])),
            extra_meta={
                "expert_count": NUM_TOTAL_EXPERTS,
                "expert_bytes": expert_bytes,
                "description": "128 routed experts + 1 shared expert (e=128) down 9-bit bfp16 (K padded 1088->1280)",
            },
        )
        gc.collect()
        return record

    def export_shared_gu(self, layer: int) -> dict[str, Any]:
        """Export MoE shared_gu: [gate; up] (2176, 2560) 9-bit bfp16."""
        g_name = f"blk.{layer}.ffn_gate_shexp.weight"
        u_name = f"blk.{layer}.ffn_up_shexp.weight"

        g = self.reader.dequantize(g_name)
        u = self.reader.dequantize(u_name)
        g_info = self.reader.tensors[g_name]
        u_info = self.reader.tensors[u_name]

        stacked = np.vstack([g, u])
        del g, u
        gc.collect()

        padded, orig_shape = pad_bfp16_gemv(stacked, m_align=128, k_align=256)
        del stacked
        gc.collect()

        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / f"L{layer:02d}" / "shared_gu.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name=f"L{layer:02d}/shared_gu",
            layer_idx=layer,
            fmt="bfp16_gemv",
            orig_shape=orig_shape,
            padded_shape=orig_shape,
            components=[
                {
                    "name": f"model.layers.{layer}.moe.shared_experts.gate_proj.weight",
                    "gguf_name": g_name,
                    "row_start": 0,
                    "row_end": 1088,
                    "shape": [1088, HIDDEN_SIZE],
                    "gguf_type": g_info.type_name,
                },
                {
                    "name": f"model.layers.{layer}.moe.shared_experts.up_proj.weight",
                    "gguf_name": u_name,
                    "row_start": 1088,
                    "row_end": 2176,
                    "shape": [1088, HIDDEN_SIZE],
                    "gguf_type": u_info.type_name,
                },
            ],
            gguf_types=list(dict.fromkeys([g_info.type_name, u_info.type_name])),
            extra_meta={"description": "MoE shared expert gate+up projection bundle (9-bit bfp16)"},
        )

    def export_lm_head(self) -> dict[str, Any]:
        """Export lm_head: (259840 -> 260096, 2560) 9-bit bfp16."""
        gguf_name = "output.weight"
        hf_name = "lm_head.weight"
        print("[Exporter] Dequantizing output.weight for lm_head...")
        mat = self.reader.dequantize(gguf_name)
        info = self.reader.tensors[gguf_name]

        padded = np.zeros((VOCAB_PADDED, HIDDEN_SIZE), dtype=np.float32)
        padded[:VOCAB_SIZE, :] = mat
        del mat
        gc.collect()

        print("[Exporter] Packing lm_head as 9-bit bfp16 GEMV...")
        packed = pack_bfp16_gemv(padded, rows=4, flush_tiny=True)
        del padded
        gc.collect()

        out_file = self.out_dir / "lm_head.bin"
        return write_and_record(
            out_file,
            packed,
            bundle_name="lm_head",
            layer_idx=None,
            fmt="bfp16_gemv",
            orig_shape=(VOCAB_SIZE, HIDDEN_SIZE),
            padded_shape=(VOCAB_PADDED, HIDDEN_SIZE),
            components=[
                {
                    "name": hf_name,
                    "gguf_name": gguf_name,
                    "row_start": 0,
                    "row_end": VOCAB_SIZE,
                    "shape": [VOCAB_SIZE, HIDDEN_SIZE],
                    "gguf_type": info.type_name,
                },
                {
                    "name": "padding",
                    "row_start": VOCAB_SIZE,
                    "row_end": VOCAB_PADDED,
                    "shape": [VOCAB_PADDED - VOCAB_SIZE, HIDDEN_SIZE],
                    "gguf_type": "ZERO_PAD",
                },
            ],
            gguf_types=[info.type_name],
            extra_meta={"description": "Output vocabulary projection (lm_head, 259840 padded to 260096)"},
        )

    def _load_hf_weight_map(self) -> dict[str, str]:
        """Load the HF weight map from model.safetensors.index.json or a single model.safetensors file."""
        index_file = self.hf_dir / "model.safetensors.index.json"
        single_file = self.hf_dir / "model.safetensors"
        if index_file.is_file():
            with open(index_file, "r", encoding="utf-8") as f:
                index_data = json.load(f)
            weight_map = index_data.get("weight_map") if isinstance(index_data, dict) else None
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    f"HF index {index_file} has no usable (non-empty) 'weight_map'; "
                    f"host tensor set cannot be determined."
                )
            return weight_map
        if single_file.is_file():
            from safetensors import safe_open
            with safe_open(str(single_file), framework="pt") as f:
                return {name: "model.safetensors" for name in f.keys()}
        raise FileNotFoundError(
            f"No safetensors index ({index_file}) or single model.safetensors "
            f"({single_file}) found in {self.hf_dir}; host tensor export cannot proceed."
        )

    def _collect_host_tensor_names(
        self,
        selected_layers: Sequence[int],
        weight_map: dict[str, str],
    ) -> tuple[dict[int, list[str]], list[str]]:
        """Return non-target HF tensor names grouped by layer/global for the given layers."""
        layer_set = set(selected_layers)
        layer_weights: dict[int, list[str]] = {l: [] for l in layer_set}
        global_weights: list[str] = []

        # Target weight name patterns (quantized for NPU; excluded from host files)
        target_suffixes = (
            ".self_attn.q_proj.weight",
            ".self_attn.k_proj.weight",
            ".self_attn.v_proj.weight",
            ".self_attn.g_proj.weight",
            ".self_attn.o_proj.weight",
            ".self_attn.f_a_proj.weight",
            ".self_attn.f_b_proj.weight",
            ".self_attn.b_proj.weight",
            ".self_attn.rda_w_proj.weight",
            ".self_attn.q_a_proj.weight",
            ".self_attn.q_b_proj.weight",
            ".self_attn.kv_a_proj_with_mqa.weight",
            ".self_attn.kv_b_proj.weight",
            ".mlp.gate_proj.weight",
            ".mlp.up_proj.weight",
            ".mlp.down_proj.weight",
            ".moe.shared_experts.gate_proj.weight",
            ".moe.shared_experts.up_proj.weight",
            ".moe.shared_experts.down_proj.weight",
        )

        for name in weight_map:
            # Skip MXFP8 scale inverse tensors
            if name.endswith(".weight_scale_inv"):
                continue
            # Skip routed expert projections
            if ".moe.experts." in name and (
                name.endswith(".gate_proj.weight")
                or name.endswith(".up_proj.weight")
                or name.endswith(".down_proj.weight")
            ):
                continue
            # Skip lm_head
            if name == "lm_head.weight":
                continue
            # Check if it's an NPU target
            if any(name.endswith(sfx) for sfx in target_suffixes):
                continue

            # Classify as layer or global
            if name.startswith("model.layers."):
                parts = name.split(".")
                if len(parts) > 2 and parts[2].isdigit():
                    l_idx = int(parts[2])
                    if l_idx in layer_set:
                        layer_weights[l_idx].append(name)
            else:
                global_weights.append(name)

        return layer_weights, global_weights

    def _verify_host_tensors_complete(
        self,
        exported_layers: Sequence[int],
        merged_bundles: list[dict[str, Any]],
    ) -> tuple[list[str], list[str], list[str]]:
        """Check that merged manifest contains every required host bundle and tensor.

        Returns (missing_bundle_names, invalid_bundle_messages, missing_tensor_names).
        An existing host bundle is invalid if its file is missing, the size differs,
        or its sha256 no longer matches the manifest.
        """
        missing_bundles: list[str] = []
        invalid_bundles: list[str] = []
        missing_tensors: list[str] = []

        if self.hf_dir is None or not self.hf_dir.is_dir():
            # The required host set cannot be determined, so completeness cannot be
            # confirmed: report that as incomplete (never "nothing is missing").
            return (
                [f"(HF directory {self.hf_dir} is unavailable; required host set unknown)"],
                invalid_bundles,
                missing_tensors,
            )

        try:
            weight_map = self._load_hf_weight_map()
            layer_weights, global_weights = self._collect_host_tensor_names(exported_layers, weight_map)
        except Exception:
            # Cannot determine the required set from the HF index; stay conservative
            # and report that the verification could not complete.
            return ["(required host set could not be loaded from HF index)"], invalid_bundles, missing_tensors

        required_bundles: set[str] = set()
        required_tensors: set[str] = set(global_weights)
        if global_weights:
            # Global host tensors live in their own bundle; a global tensor that is
            # (wrongly) present inside another bundle must not satisfy completeness.
            required_bundles.add("host_global")
        for l_idx, names in layer_weights.items():
            if names:
                required_bundles.add(f"L{l_idx:02d}/host")
                required_tensors.update(names)

        host_bundles = {
            b["bundle"]: b
            for b in merged_bundles
            if b.get("format") == "host_safetensors"
        }
        missing_bundles = sorted(required_bundles - set(host_bundles.keys()))

        actual_tensor_names: set[str] = set()
        for bundle_name, bundle in host_bundles.items():
            file_path = self.out_dir / bundle["file"]
            if not file_path.is_file():
                invalid_bundles.append(f"{bundle_name} (file missing)")
                continue
            if file_path.stat().st_size != bundle["size_bytes"]:
                invalid_bundles.append(f"{bundle_name} (size mismatch)")
                continue
            if compute_sha256(file_path) != bundle["sha256"]:
                invalid_bundles.append(f"{bundle_name} (sha256 mismatch)")
                continue
            for comp in bundle.get("components", []):
                name = comp.get("name")
                if name:
                    actual_tensor_names.add(name)

        missing_tensors = sorted(required_tensors - actual_tensor_names)
        return missing_bundles, invalid_bundles, missing_tensors

    def export_host_tensors(self, selected_layers: Sequence[int]) -> list[dict[str, Any]]:
        """Export non-target tensors from HF safetensors preserving original precision.

        Tensors are read with framework="pt" because this environment's numpy cannot read
        bfloat16 (no ml_dtypes). BF16 tensors are written out as torch.bfloat16 so the
        safetensors header dtype is "BF16", matching what host/weights.py expects, while
        keeping the bit pattern identical.  fp32/fp16/int tensors are written with their
        native dtype.  The manifest component keys are:
          - "dtype": original HF dtype (e.g. "bfloat16", "float32")
          - "stored_dtype": on-disk safetensors header dtype (e.g. "BF16", "F32")
        """
        if not self.export_host:
            return []
        if self.hf_dir is None or not self.hf_dir.is_dir():
            raise FileNotFoundError(
                f"HF directory {self.hf_dir} is missing or not a directory; "
                f"host tensor export cannot proceed."
            )

        weight_map = self._load_hf_weight_map()
        layer_weights, global_weights = self._collect_host_tensor_names(selected_layers, weight_map)

        from safetensors import safe_open

        # Process one output file at a time (one layer, then global): read ONLY the
        # tensors that output file needs (framework="pt": numpy here has no bfloat16
        # dtype), write it atomically, and release the tensors before the next group.
        # A tensor required by a group but absent from its shard is a hard failure.
        entries: list[dict[str, Any]] = []

        groups: list[tuple[Path, str, int | None, list[str]]] = [
            (self.out_dir / f"L{l_idx:02d}" / "host.safetensors", f"L{l_idx:02d}/host", l_idx, names)
            for l_idx, names in sorted(layer_weights.items())
            if names
        ]
        if global_weights:
            groups.append((self.out_dir / "host_global.safetensors", "host_global", None, global_weights))

        try:
            for out_file, bundle_name, l_idx, t_names in groups:
                needed = set(t_names)
                by_shard: dict[str, set[str]] = {}
                for name in sorted(needed):
                    if name not in weight_map:
                        raise KeyError(
                            f"Host tensor {name} required by {bundle_name} missing from HF index weight_map"
                        )
                    by_shard.setdefault(weight_map[name], set()).add(name)

                loaded_tensors: dict[str, np.ndarray] = {}
                source_dtypes: dict[str, str] = {}
                for shard in sorted(by_shard):
                    shard_path = self.hf_dir / shard
                    with safe_open(str(shard_path), framework="pt") as f:
                        shard_names = set(f.keys())
                        for k in sorted(by_shard[shard]):
                            if k not in shard_names:
                                raise KeyError(
                                    f"Host tensor {k} missing from shard {shard} "
                                    f"(required by {bundle_name})"
                                )
                            arr, src_dtype = host_tensor_to_numpy(f.get_tensor(k))
                            loaded_tensors[k] = arr
                            source_dtypes[k] = src_dtype

                missing = needed - loaded_tensors.keys()
                if missing:
                    raise KeyError(
                        f"Host tensors {sorted(missing)} required by {bundle_name} were not loaded "
                        f"from any shard"
                    )

                sub_dict = {name: loaded_tensors[name] for name in t_names}
                comp_info = _save_host_safetensors(sub_dict, source_dtypes, out_file)

                if l_idx is not None:
                    description = f"Host tensors for layer {l_idx} (norms, routers, conv1d, mHC)"
                else:
                    description = "Global host tensors (embed_tokens, output_norm)"
                entries.append({
                    "bundle": bundle_name,
                    "layer": l_idx,
                    "format": "host_safetensors",
                    "file": out_file.relative_to(self.out_dir).as_posix(),
                    "size_bytes": out_file.stat().st_size,
                    "components": comp_info,
                    "sha256": compute_sha256(out_file),
                    "description": description,
                })

                # Release this group's tensors before loading the next group.
                loaded_tensors.clear()
                source_dtypes.clear()
                del sub_dict
                gc.collect()
        except Exception as exc:
            # Keep entries for host bundles already written so the manifest can record them
            exc.host_partial_entries = entries
            raise

        return entries

    def export_layer(self, layer: int) -> list[dict[str, Any]]:
        """Export all NPU bundles for a given layer."""
        entries = []
        is_rda = layer in RDA_LAYERS
        is_dense = (layer == 0)

        print(f"[Exporter] Processing layer {layer:02d} ({'RDA' if is_rda else 'MLA'}, {'Dense' if is_dense else 'MoE'})...")

        # 1. Mixer projections
        if is_rda:
            if self._should_export("rda_in"):
                entries.append(self.export_rda_in(layer))
            if self._should_export("f_b"):
                entries.append(self.export_f_b(layer))
            if self._should_export("o_proj"):
                entries.append(self.export_o_proj(layer))
        else:
            if self._should_export("mla_in"):
                entries.append(self.export_mla_in(layer))
            if self._should_export("q_b"):
                entries.append(self.export_mla_q_b(layer))
            if self._should_export("k_b"):
                entries.append(self.export_mla_k_b(layer))
            if self._should_export("v_b"):
                entries.append(self.export_mla_v_b(layer))
            if self._should_export("g"):
                entries.append(self.export_mla_g(layer))
            if self._should_export("o_proj"):
                entries.append(self.export_o_proj(layer))

        # 2. MLP projections
        if is_dense:
            if self._should_export("dense_gu"):
                entries.append(self.export_dense_gu())
            if self._should_export("dense_down"):
                entries.append(self.export_dense_down())
        else:
            if self._should_export("moe_b1"):
                entries.append(self.export_moe_b1(layer))
            if self._should_export("moe_b2"):
                entries.append(self.export_moe_b2(layer))
            if self._should_export("shared_gu"):
                entries.append(self.export_shared_gu(layer))

        return entries

    def run(self) -> dict[str, Any]:
        """Execute full or selected export pipeline."""
        start_time = time.time()
        manifest_path = self.out_dir / "manifest.json"
        # Every fallible existing-output/HF input check must happen before mkdir or
        # an atomic replacement of an NPU/host bundle.
        old_manifest = _load_existing_manifest(manifest_path)
        _preflight_existing_host_index(self.out_dir)
        self._preflight_gguf_inputs()
        self._preflight_host_inputs()
        self.out_dir.mkdir(parents=True, exist_ok=True)

        all_entries: list[dict[str, Any]] = []

        # Export layers
        for l in self.layers:
            layer_entries = self.export_layer(l)
            all_entries.extend(layer_entries)

        # Export output layer (lm_head)
        if self._should_export("lm_head"):
            print("[Exporter] Exporting lm_head...")
            all_entries.append(self.export_lm_head())

        # Export host non-target tensors
        host_attempted = self.export_host and self._should_export("host")
        host_failure: Exception | None = None
        if host_attempted:
            print("[Exporter] Exporting host non-target tensors...")
            if self.hf_dir is None or not self.hf_dir.is_dir():
                host_failure = FileNotFoundError(
                    f"HF directory {self.hf_dir} is missing or not a directory; "
                    f"host tensor export cannot proceed."
                )
            else:
                try:
                    all_entries.extend(self.export_host_tensors(self.layers))
                except Exception as exc:
                    host_failure = exc
                    partial = getattr(exc, "host_partial_entries", None)
                    if partial:
                        all_entries.extend(partial)
            if host_failure is not None:
                print(
                    f"[Exporter] ERROR: host tensor export FAILED: {type(host_failure).__name__}: {host_failure}"
                )
                print(
                    "[Exporter] Writing manifest for bundles written so far "
                    f"({len(all_entries)} entries); re-raising to signal failure."
                )

        # Write manifest.json
        total_size = sum(e["size_bytes"] for e in all_entries)
        manifest_data = {
            "version": "1.0",
            "model": "Raon-1-mini-preview",
            "mode": "mixed_full_d",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "num_layers_total": NUM_LAYERS,
            "exported_layers": sorted(set(self.layers)) if self.layers else [],
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_size_gb": round(total_size / (1024 * 1024 * 1024), 3),
            "bundles": all_entries,
        }
        if host_failure is not None:
            manifest_data["host_tensors_status"] = {
                "status": "failed",
                "error": f"{type(host_failure).__name__}: {host_failure}",
                "note": (
                    "Host tensor export failed; bundles listed in this manifest were written "
                    "before the failure and host bundles are missing or incomplete."
                ),
            }

        # If a manifest already exists and we ran partially, merge entries
        if old_manifest is not None:
            old_map = {b["bundle"]: b for b in old_manifest["bundles"]}
            for new_b in all_entries:
                old_map[new_b["bundle"]] = new_b
            merged_bundles = sorted(old_map.values(), key=lambda x: x["bundle"])
            total_merged = sum(b["size_bytes"] for b in merged_bundles)
            manifest_data["bundles"] = merged_bundles
            manifest_data["total_size_bytes"] = total_merged
            manifest_data["total_size_mb"] = round(total_merged / (1024 * 1024), 2)
            manifest_data["total_size_gb"] = round(total_merged / (1024 * 1024 * 1024), 3)

        # Recalculate exported_layers from the merged bundles so partial runs
        # (e.g. --layers 1 after layer 0) report all layers present in the output.
        merged_bundles = manifest_data.get("bundles", [])
        layers_in_bundles: set[int] = set()
        for b in merged_bundles:
            layer = b.get("layer")
            if isinstance(layer, int):
                layers_in_bundles.add(layer)
        manifest_data["exported_layers"] = sorted(layers_in_bundles)

        if not host_attempted:
            # Host export was not part of this run; keep any previously recorded status.
            if old_manifest is not None and "host_tensors_status" in old_manifest:
                manifest_data["host_tensors_status"] = old_manifest["host_tensors_status"]
        elif host_failure is None:
            # Host export succeeded this run, but the merged output can still be incomplete
            # because earlier runs failed or omitted some layers.  Only clear the failure
            # status when the merged manifest actually contains every required host bundle
            # and tensor, and each host file exists with the recorded size and sha256.
            missing_bundles, invalid_bundles, missing_tensors = self._verify_host_tensors_complete(
                manifest_data["exported_layers"], merged_bundles
            )
            if missing_bundles or invalid_bundles or missing_tensors:
                old_status = old_manifest.get("host_tensors_status") if old_manifest else None
                manifest_data["host_tensors_status"] = _build_incomplete_host_status(
                    old_status, missing_bundles, invalid_bundles, missing_tensors
                )
            else:
                manifest_data.pop("host_tensors_status", None)

        _write_json_atomic(manifest_path, manifest_data)

        # Write/update host tensor index so host/weights.py can locate every tensor.
        write_host_index(self.out_dir, manifest_data.get("bundles", []), merge=True)

        if host_failure is not None:
            print(f"[Exporter] Manifest saved to {manifest_path} with {len(all_entries)} bundles before host failure.")
            raise host_failure

        elapsed = time.time() - start_time
        print(f"[Exporter] Completed export of {len(all_entries)} bundles in {elapsed:.2f}s.")
        print(f"[Exporter] Manifest saved to {manifest_path} ({manifest_data['total_size_mb']} MB total).")
        return manifest_data


def parse_layers_arg(arg: str | None) -> list[int] | None:
    """Parse layers argument like '0,1,3' or '0-3' or 'all'."""
    if not arg or arg.lower() == "all":
        return None
    res = set()
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            s, e = part.split("-", 1)
            res.update(range(int(s), int(e) + 1))
        else:
            res.add(int(part))
    return sorted(res)


# Original (manifest ``dtype``) numpy dtype name -> safetensors header dtype, for the
# non-bfloat16 host tensors; an unknown name is a failure, never a pass.
_NPY_DTYPE_TO_HEADER: dict[str, str] = {
    "float64": "F64",
    "float32": "F32",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "uint16": "U16",
    "bool": "BOOL",
}


def fix_host_only(out_dir: str | Path) -> dict[str, Any]:
    """Rewrite host safetensors so bf16 tensors use the 'BF16' header dtype.

    Existing NPU .bin bundles are left untouched.  The function is meant to repair
    outputs produced before the BF16-header fix without re-running the full export.
    Which tensors need the fix is decided from the **actual on-disk safetensors header**
    (not the manifest's ``stored_dtype``), and every host file, manifest component tensor,
    and header entry must exist — a missing host file or a missing tensor/header entry is
    a hard failure.  Before ANY file is touched, all host bundles are preflighted: the
    manifest shape must equal the header shape, and a bfloat16 tensor must have a ``BF16``
    header (kept) or a same-shape legacy ``U16`` header (reinterpreted); ``F32`` or any
    other dtype fails the whole run with every original file left byte-identical.  Rewrite is atomic: tensors go to a unique temp file in the same
    directory whose header, tensor names, and bit patterns are verified BEFORE
    ``os.replace()`` swaps it in, so a failed rewrite leaves the original file untouched.
    The manifest component ``stored_dtype`` values are updated, repaired bundles'
    ``size_bytes`` and the top-level ``total_size_bytes``/``total_size_mb``/``total_size_gb``
    are recomputed from the bundle ``size_bytes`` sum (the rewritten safetensors header
    has a different length), and ``model.safetensors.index.json`` is rewritten from the
    repaired manifest.
    """
    import torch
    from safetensors import safe_open

    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest found at {manifest_path}")

    manifest = _load_existing_manifest(manifest_path)
    assert manifest is not None  # guarded by is_file() above
    _preflight_existing_host_index(out_dir)

    # ---- Phase 1: preflight every host bundle WITHOUT touching any file. ----------
    # The manifest shape/dtype of each component is cross-checked against the actual
    # on-disk header.  For an original dtype of bfloat16 only a ``BF16`` header (already
    # correct) or a legacy ``U16`` header of the SAME shape (reinterpreted as BF16) is
    # accepted; ``F32`` or any other header dtype, and any shape mismatch, is a hard
    # failure raised before any host file is rewritten.
    plans: list[tuple[dict[str, Any], Path, dict[str, Any], set[str]]] = []
    for entry in manifest.get("bundles", []):
        if entry.get("format") != "host_safetensors":
            continue
        rel_file = entry.get("file")
        if not rel_file:
            raise RuntimeError(
                f"Host bundle {entry.get('bundle')} has no 'file' in the manifest"
            )
        file_path = out_dir / rel_file
        if not file_path.is_file():
            raise FileNotFoundError(
                f"Host bundle file {file_path} missing for bundle {entry.get('bundle')}"
            )
        components = entry.get("components")
        if not components:
            raise RuntimeError(
                f"Host bundle {entry.get('bundle')} has no 'components' in the manifest"
            )

        # Judge from the ACTUAL on-disk header, not the manifest's stored_dtype field.
        header = _read_safetensors_header(file_path)
        header_names = set(header) - {"__metadata__"}
        comp_names = {comp["name"] for comp in components}
        missing_in_header = comp_names - header_names
        extra_in_file = header_names - comp_names
        if missing_in_header:
            raise RuntimeError(
                f"Manifest tensors {sorted(missing_in_header)} missing from {file_path} header"
            )
        if extra_in_file:
            raise RuntimeError(
                f"File {file_path} contains tensors {sorted(extra_in_file)} not listed in the manifest"
            )

        rewrite_names: set[str] = set()
        for comp in components:
            c_name = comp["name"]
            for key in ("shape", "dtype"):
                if comp.get(key) is None:
                    raise RuntimeError(
                        f"Manifest component {c_name} of {file_path} has no '{key}'; "
                        f"cannot validate it against the on-disk header"
                    )
            h = header[c_name]
            h_dtype = h.get("dtype")
            h_shape = h.get("shape")
            if h_shape is None or list(h_shape) != list(comp["shape"]):
                raise RuntimeError(
                    f"Shape mismatch for {c_name} in {file_path}: manifest {list(comp['shape'])}, "
                    f"header {None if h_shape is None else list(h_shape)}; refusing to rewrite"
                )
            if comp["dtype"] == "bfloat16":
                if h_dtype == "U16":
                    rewrite_names.add(c_name)
                elif h_dtype != "BF16":
                    raise RuntimeError(
                        f"Tensor {c_name} in {file_path} is declared bfloat16 but its header dtype "
                        f"is {h_dtype!r} (only 'BF16' or legacy 'U16' are accepted); refusing to rewrite"
                    )
            else:
                expected_h = _NPY_DTYPE_TO_HEADER.get(comp["dtype"])
                if expected_h is None or h_dtype != expected_h:
                    raise RuntimeError(
                        f"Tensor {c_name} in {file_path}: manifest dtype {comp['dtype']!r} "
                        f"(expected header {expected_h!r}) does not match header dtype {h_dtype!r}"
                    )
        plans.append((entry, file_path, header, rewrite_names))

    if not plans:
        raise RuntimeError(
            f"Manifest {manifest_path} contains no host_safetensors bundles; nothing can be repaired"
        )

    # ---- Phase 2: refresh metadata / rewrite files (every check already passed). ----
    updated = False
    for entry, file_path, header, bf16_names in plans:
        # Always refresh manifest metadata from the actual on-disk file, even when
        # the header is already BF16. This guarantees the manifest matches the file
        # after an interrupted run where some files were rewritten but the manifest
        # was not yet updated.
        for comp in entry["components"]:
            comp["stored_dtype"] = header[comp["name"]]["dtype"]
        entry["size_bytes"] = file_path.stat().st_size
        entry["sha256"] = compute_sha256(file_path)
        updated = True

        if not bf16_names:
            continue

        old_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(file_path), framework="pt") as f:
            for name in f.keys():
                old_tensors[name] = f.get_tensor(name)

        new_tensors: dict[str, torch.Tensor] = {}
        expected_bits: dict[str, np.ndarray] = {}
        for name, t in old_tensors.items():
            if name in bf16_names:
                # Same uint16 bit pattern, but declared as bfloat16 in the header.
                new_tensors[name] = t.view(torch.int16).view(torch.bfloat16)
            else:
                new_tensors[name] = t
            expected_bits[name] = _tensor_bits(old_tensors[name])

        # Atomic: writes to a unique temp file in the same directory; header, tensor
        # names, and bit patterns are verified on the temp file BEFORE os.replace()
        # swaps it in, so a failed rewrite leaves the original file untouched.
        _save_tensors_verified(new_tensors, file_path, expected_bits)

        for comp in entry["components"]:
            if comp["name"] in bf16_names:
                comp["stored_dtype"] = "BF16"
        entry["size_bytes"] = file_path.stat().st_size
        entry["sha256"] = compute_sha256(file_path)

    if updated:
        # Rewritten safetensors headers change host bundle sizes, so recompute the
        # top-level totals from the bundle size_bytes sum (stale totals no longer match).
        total_size = sum(e["size_bytes"] for e in manifest.get("bundles", []))
        manifest["total_size_bytes"] = total_size
        manifest["total_size_mb"] = round(total_size / (1024 * 1024), 2)
        manifest["total_size_gb"] = round(total_size / (1024 * 1024 * 1024), 3)
        _write_json_atomic(manifest_path, manifest)

    # Rebuild/merge the host index from the (possibly repaired) manifest.
    write_host_index(out_dir, manifest.get("bundles", []), merge=True)

    return {
        "manifest_path": str(manifest_path),
        "updated": updated,
        "host_bundles_checked": sum(
            1 for e in manifest.get("bundles", []) if e.get("format") == "host_safetensors"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Raon-1-mini weights to D mode (mixed_full)")
    parser.add_argument("--gguf", type=Path, default=None, help="Path to input GGUF file")
    parser.add_argument("--hf", type=Path, default=None, help="Path to HF model directory")
    parser.add_argument("--out", type=Path, default=None, help="Output directory")
    parser.add_argument("--layers", type=str, default=None, help="Layers to export, e.g. '0,1,3' or 'all'")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated bundle names, e.g. 'lm_head,rda_in'")
    parser.add_argument("--skip-host", action="store_true", help="Skip host non-target tensor export")
    parser.add_argument("--fix-host-only", type=Path, default=None, metavar="OUT_DIR", help="Rewrite existing host safetensors U16 headers to BF16 and refresh index/manifest")

    args = parser.parse_args()
    if args.fix_host_only is not None:
        result = fix_host_only(args.fix_host_only)
        print(f"[fix-host-only] Updated={result['updated']}, checked {result['host_bundles_checked']} host bundles.")
        print(f"[fix-host-only] Manifest: {result['manifest_path']}")
        return

    if any(value is None for value in (args.gguf, args.hf, args.out)):
        parser.error("--gguf, --hf and --out are required (except with --fix-host-only)")

    layers = parse_layers_arg(args.layers)
    only_bundles = [b.strip() for b in args.only.split(",")] if args.only else None

    from export.artifacts import require_tokenizer_files, copy_tokenizer_files
    require_tokenizer_files(args.hf)
    exporter = DModeExporter(
        gguf_path=args.gguf,
        hf_dir=args.hf,
        out_dir=args.out,
        layers=layers,
        only_bundles=only_bundles,
        export_host=not args.skip_host,
    )
    exporter.run()
    copy_tokenizer_files(args.hf, args.out)
    # The engine reads model configuration from the exported root.
    import shutil
    if (args.hf / "config.json").resolve() != (args.out / "config.json").resolve():
        shutil.copyfile(args.hf / "config.json", args.out / "config.json")


if __name__ == "__main__":
    main()
