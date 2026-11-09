# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import pytest
from export.artifacts import TOKENIZER_FILES, copy_tokenizer_files, require_tokenizer_files, verify_expected


def test_tokenizer_copy_is_byte_exact(tmp_path):
    source, out = tmp_path / 'hf', tmp_path / 'out'
    source.mkdir()
    for name in TOKENIZER_FILES:
        (source / name).write_bytes(b'\xef\xbb\xbf\x00\r\n' + name.encode())
    copy_tokenizer_files(source, out)
    for name in TOKENIZER_FILES:
        assert (source / name).read_bytes() == (out / name).read_bytes()
    copy_tokenizer_files(source, source)


@pytest.mark.parametrize('missing', TOKENIZER_FILES)
def test_missing_tokenizer_prevents_partial_copy(tmp_path, missing):
    source, out = tmp_path / 'hf', tmp_path / 'out'
    source.mkdir()
    for name in TOKENIZER_FILES:
        if name != missing:
            (source / name).write_bytes(b'{}')
    with pytest.raises(FileNotFoundError, match=missing):
        copy_tokenizer_files(source, out)
    assert not out.exists()
    with pytest.raises(FileNotFoundError):
        require_tokenizer_files(source)


def test_checksums_missing_extra_changed_and_timestamp(tmp_path):
    out = tmp_path / 'weights'
    out.mkdir()
    digest = hashlib.sha256(b'abc').hexdigest()
    expected = tmp_path / 'expected.txt'
    expected.write_text(f'{digest}  space name.bin\n')
    assert verify_expected(out, expected) == ['MISSING: space name.bin']
    (out / 'space name.bin').write_bytes(b'abc')
    (out / 'manifest.json').write_text('{"created_at":"today"}')
    assert verify_expected(out, expected) == []
    (out / 'space name.bin').write_bytes(b'abd')
    (out / 'extra.bin').write_bytes(b'')
    assert verify_expected(out, expected) == ['UNEXPECTED: extra.bin', 'MISMATCH: space name.bin']


@pytest.mark.parametrize('line', ['', 'bad line', '0'*64+'  ../escape', '0'*64+'  /absolute', '0'*64+'  C:\\escape', ('0'*64+'  x\n')*2])
def test_invalid_checksum_lists(tmp_path, line):
    expected = tmp_path / 'expected.txt'
    expected.write_text(line)
    with pytest.raises(ValueError):
        verify_expected(tmp_path / 'weights', expected)


def test_index_json_keeps_lf_with_windows_text_default(tmp_path, monkeypatch):
    import builtins
    import export.export_d as exporter
    original_open = builtins.open
    def windows_open(file, mode='r', *args, **kwargs):
        if 'w' in mode and 'b' not in mode:
            kwargs.setdefault('newline', '\r\n')
        return original_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(exporter, 'open', windows_open, raising=False)
    destination = tmp_path / 'model.safetensors.index.json'
    exporter._write_json_atomic(destination, {'weight_map': {'a': 'a.safetensors'}})
    assert destination.read_bytes() == b'{\n  "weight_map": {\n    "a": "a.safetensors"\n  }\n}'
