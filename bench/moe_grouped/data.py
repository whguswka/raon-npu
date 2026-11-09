# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Versioned per-layer/per-token replay record format."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from host.strict_json import loads_strict

FORMAT_VERSION = 1


@dataclass(frozen=True)
class Record:
    layer: int
    token: int
    x: np.ndarray
    top_ids: np.ndarray
    top_weights: np.ndarray
    shared_gate_up: np.ndarray
    output: np.ndarray
    shared_down: np.ndarray
    gate_beta: float = 4.0
    value_beta: float = 25.0

    def fresh(self) -> "Record":
        return Record(
            self.layer, self.token, self.x.copy(), self.top_ids.copy(),
            self.top_weights.copy(), self.shared_gate_up.copy(),
            self.output.copy(), self.shared_down.copy(),
            self.gate_beta, self.value_beta,
        )


def record_path(root: Path, layer: int, token: int) -> Path:
    return root / f"layer{layer:02d}_token{token:06d}.npz"


def save_record(root: Path, record: Record) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = record_path(root, record.layer, record.token)
    np.savez(
        path,
        format_version=np.asarray(FORMAT_VERSION, dtype=np.int64),
        layer=np.asarray(record.layer, dtype=np.int64),
        token=np.asarray(record.token, dtype=np.int64),
        x=np.asarray(record.x, dtype=np.uint16),
        top_ids=np.asarray(record.top_ids, dtype=np.int64),
        top_weights=np.asarray(record.top_weights, dtype=np.float32),
        shared_gate_up=np.asarray(record.shared_gate_up, dtype=np.uint16),
        output=np.asarray(record.output, dtype=np.uint16),
        shared_down=np.asarray(record.shared_down, dtype=np.uint16),
        gate_beta=np.asarray(record.gate_beta, dtype=np.float32),
        value_beta=np.asarray(record.value_beta, dtype=np.float32),
    )
    return path


def load_record(path: Path) -> Record:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "format_version", "layer", "token", "x", "top_ids", "top_weights",
            "shared_gate_up", "output", "shared_down", "gate_beta", "value_beta",
        }
        if set(data.files) != required:
            missing = sorted(required - set(data.files))
            extra = sorted(set(data.files) - required)
            raise ValueError(f"record fields differ in {path}: missing={missing}, extra={extra}")
        _validate_raw_arrays(data, path)
        version = int(data["format_version"])
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported record version {version} in {path}")
        record = Record(
            int(data["layer"]), int(data["token"]),
            np.array(data["x"], copy=True),
            np.array(data["top_ids"], copy=True),
            np.array(data["top_weights"], copy=True),
            np.array(data["shared_gate_up"], copy=True),
            np.array(data["output"], copy=True),
            np.array(data["shared_down"], copy=True),
            float(data["gate_beta"]), float(data["value_beta"]),
        )
    _validate_record(record, path)
    return record


def _validate_raw_arrays(data: object, path: Path) -> None:
    exact = {
        "format_version": ((), np.dtype(np.int64)),
        "layer": ((), np.dtype(np.int64)),
        "token": ((), np.dtype(np.int64)),
        "x": ((2560,), np.dtype(np.uint16)),
        "shared_gate_up": ((2176,), np.dtype(np.uint16)),
        "output": ((2560,), np.dtype(np.uint16)),
        "shared_down": ((2560,), np.dtype(np.uint16)),
        "gate_beta": ((), np.dtype(np.float32)),
        "value_beta": ((), np.dtype(np.float32)),
    }
    for name, (shape, dtype) in exact.items():
        value = data[name]  # type: ignore[index]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{name} must be {dtype}{list(shape)} in {path}; "
                f"got {value.dtype}{list(value.shape)}"
            )
    top_ids = data["top_ids"]  # type: ignore[index]
    top_weights = data["top_weights"]  # type: ignore[index]
    if top_ids.ndim != 1 or top_ids.dtype != np.dtype(np.int64):
        raise ValueError(f"top_ids must be one-dimensional int64 in {path}")
    if top_weights.shape != top_ids.shape or top_weights.dtype != np.dtype(np.float32):
        raise ValueError(f"top_weights must be float32 with the top_ids shape in {path}")


def _validate_record(record: Record, path: Path | None = None) -> None:
    where = f" in {path}" if path is not None else ""
    if record.x.shape != (2560,) or record.x.dtype != np.uint16:
        raise ValueError(f"x must be uint16[2560]{where}")
    if record.top_ids.ndim != 1 or record.top_weights.shape != record.top_ids.shape:
        raise ValueError(f"top_ids/top_weights shape mismatch{where}")
    if record.top_ids.dtype != np.int64 or record.top_weights.dtype != np.float32:
        raise ValueError(f"top_ids/top_weights dtype mismatch{where}")
    if record.shared_gate_up.shape != (2176,) or record.shared_gate_up.dtype != np.uint16:
        raise ValueError(f"shared_gate_up must be uint16[2176]{where}")
    if (
        record.output.shape != (2560,) or record.output.dtype != np.uint16
        or record.shared_down.shape != (2560,) or record.shared_down.dtype != np.uint16
    ):
        raise ValueError(f"outputs must be uint16[2560]{where}")


def load_records(root: Path, layers: set[int] | None = None) -> list[Record]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"replay manifest not found: {manifest_path}")
    manifest = loads_strict(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"invalid replay manifest version in {manifest_path}")
    manifest_layers = manifest.get("layers")
    manifest_counts = manifest.get("records_by_layer")
    if (
        not isinstance(manifest_layers, list)
        or not all(isinstance(layer, int) and not isinstance(layer, bool)
                   for layer in manifest_layers)
        or len(set(manifest_layers)) != len(manifest_layers)
        or not isinstance(manifest_counts, dict)
    ):
        raise ValueError(f"invalid layer metadata in {manifest_path}")
    expected: dict[int, int] = {}
    for layer in manifest_layers:
        count = manifest_counts.get(str(layer))
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(f"invalid record count for layer {layer} in {manifest_path}")
        expected[layer] = count
    if set(manifest_counts) != {str(layer) for layer in manifest_layers}:
        raise ValueError(f"records_by_layer does not match layers in {manifest_path}")

    paths = sorted(root.glob("layer*_token*.npz"))
    records = [load_record(path) for path in paths]
    actual: dict[int, list[Record]] = {}
    for path, record in zip(paths, records):
        if path != record_path(root, record.layer, record.token):
            raise ValueError(f"record filename does not match its layer/token: {path}")
        actual.setdefault(record.layer, []).append(record)
    if set(actual) != {layer for layer, count in expected.items() if count}:
        raise ValueError(
            f"record layers differ from manifest: expected={sorted(expected)}, "
            f"actual={sorted(actual)}"
        )
    for layer, count in expected.items():
        tokens = sorted(record.token for record in actual.get(layer, []))
        if tokens != list(range(count)):
            raise ValueError(
                f"layer {layer}: tokens must be contiguous 0..{count - 1}; got {tokens}"
            )
    if layers is not None:
        unknown = layers - set(expected)
        if unknown:
            raise ValueError(f"requested layers are not in manifest: {sorted(unknown)}")
        records = [record for record in records if record.layer in layers]
    if not records:
        raise ValueError(f"no replay records found in {root}")
    return records


def batches(records: Iterable[Record], size: int) -> list[list[Record]]:
    if size not in (16, 32, 64):
        raise ValueError("batch size must be 16, 32, or 64")
    by_layer: dict[int, list[Record]] = {}
    for record in records:
        by_layer.setdefault(record.layer, []).append(record)
    result: list[list[Record]] = []
    for layer in sorted(by_layer):
        ordered = sorted(by_layer[layer], key=lambda item: item.token)
        for start in range(0, len(ordered), size):
            result.append(ordered[start : start + size])
    return result


def write_manifest(root: Path, payload: dict[str, object]) -> None:
    data = {"format_version": FORMAT_VERSION, **payload}
    (root / "manifest.json").write_text(
        json.dumps(data, indent=2, sort_keys=True), encoding="utf-8"
    )
