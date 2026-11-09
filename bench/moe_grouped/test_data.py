# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Record format and layer-local batching tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from . import record as record_module
import pytest

from .data import Record, batches, load_record, load_records, save_record, write_manifest


def _record(layer: int, token: int) -> Record:
    return Record(
        layer, token, np.zeros(2560, dtype=np.uint16),
        np.arange(6, dtype=np.int64), np.ones(6, dtype=np.float32),
        np.zeros(2176, dtype=np.uint16), np.zeros(2560, dtype=np.uint16),
        np.zeros(2560, dtype=np.uint16),
    )


def test_npz_round_trip_without_pickle(tmp_path) -> None:
    expected = _record(3, 17)
    path = save_record(tmp_path, expected)
    actual = load_record(path)
    assert (actual.layer, actual.token) == (3, 17)
    for name in ("x", "top_ids", "top_weights", "shared_gate_up", "output", "shared_down"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert (actual.gate_beta, actual.value_beta) == (4.0, 25.0)


def _manifest(root: Path, count: int = 2) -> None:
    write_manifest(
        root,
        {"tokens": count, "layers": [3], "records_by_layer": {"3": count}},
    )


def test_load_records_requires_manifest(tmp_path) -> None:
    save_record(tmp_path, _record(3, 0))
    with pytest.raises(ValueError, match="manifest not found"):
        load_records(tmp_path)


def test_load_records_rejects_missing_file(tmp_path) -> None:
    save_record(tmp_path, _record(3, 0))
    missing = save_record(tmp_path, _record(3, 1))
    _manifest(tmp_path, 2)
    missing.unlink()
    with pytest.raises(ValueError, match="contiguous"):
        load_records(tmp_path)


def test_load_records_rejects_token_gap(tmp_path) -> None:
    save_record(tmp_path, _record(3, 0))
    save_record(tmp_path, _record(3, 2))
    _manifest(tmp_path, 2)
    with pytest.raises(ValueError, match="contiguous"):
        load_records(tmp_path)


def test_load_record_rejects_dtype_before_conversion(tmp_path) -> None:
    path = save_record(tmp_path, _record(3, 0))
    with np.load(path, allow_pickle=False) as original:
        arrays = {name: np.array(original[name], copy=True) for name in original.files}
    arrays["x"] = arrays["x"].astype(np.int32)
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="x must be uint16"):
        load_record(path)


def test_batches_never_mix_layers() -> None:
    records = [_record(2, token) for token in range(19)] + [
        _record(4, token) for token in range(18)
    ]
    result = batches(reversed(records), 16)
    assert [len(batch) for batch in result] == [16, 3, 16, 2]
    assert all(len({record.layer for record in batch}) == 1 for batch in result)
    assert all([record.token for record in batch] == sorted(record.token for record in batch)
               for batch in result)


def test_tools_unload_engine_instead_of_calling_close() -> None:
    root = Path(__file__).parent
    for name in ("record.py", "benchmark.py", "stress.py"):
        source = (root / name).read_text(encoding="utf-8")
        assert "engine.unload()" in source
        assert "engine.close()" not in source


def test_record_unloads_before_writing_manifest(monkeypatch, tmp_path) -> None:
    events: list[str] = []

    class FakeBackend:
        def __init__(self) -> None:
            self.moe_experts = lambda *_args, **_kwargs: None

        def reset(self) -> None:
            pass

    class FakeModel:
        config = type("Config", (), {"num_hidden_layers": 1})()

        def reset_state(self) -> None:
            pass

        def prefill(self, *_args, **_kwargs) -> None:
            pass

    class FakeEngine:
        def __init__(self, *_args, **_kwargs) -> None:
            self.model = FakeModel()
            self.backend = FakeBackend()

        def unload(self) -> None:
            events.append("unload")

    out = tmp_path / "records"
    monkeypatch.setattr(record_module, "Engine", FakeEngine)
    monkeypatch.setattr(record_module, "load_tokens", lambda _args: [1])
    monkeypatch.setattr(
        sys,
        "argv",
        ["record", "--weights-dir", str(tmp_path), "--prompt", "x", "--out", str(out)],
    )
    assert record_module.main() == 0
    assert events == ["unload"]
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["tokens"] == 1
