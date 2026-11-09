# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import sys
from types import SimpleNamespace

import pytest
from scripts import kernel_object_audit as audit_module, selfcheck


def test_audit_reads_then_fixes_only_stale_copies(tmp_path):
    arch = tmp_path / 'build/aie2p'
    arch.mkdir(parents=True)
    (arch / 'x.o').write_bytes(b'new')
    stale = tmp_path / 'build/stale.mlir.d'
    fresh = tmp_path / 'build/fresh.mlir.d'
    stale.mkdir()
    fresh.mkdir()
    (stale / 'x.o').write_bytes(b'old')
    (stale / 'x.o.d').write_bytes(b'deps')
    (fresh / 'x.o').write_bytes(b'new')
    (stale / 'keep.txt').write_bytes(b'keep')
    assert len(audit_module.audit(tmp_path)['stale']) == 1
    assert (stale / 'x.o').read_bytes() == b'old'
    result = audit_module.audit(tmp_path, fix=True)
    assert result['checked'] == 2 and result['fixed_files'] == 2
    assert not (stale / 'x.o').exists()
    assert (fresh / 'x.o').read_bytes() == b'new'
    assert (stale / 'keep.txt').read_bytes() == b'keep'
    assert audit_module.audit(tmp_path)['stale'] == []


@pytest.mark.parametrize('value', [None, '', '64'])
@pytest.mark.parametrize('passes', [True, False])
def test_selfcheck_chunk_contract(tmp_path, monkeypatch, capsys, value, passes):
    # Track the original value even when selfcheck creates the environment entry.
    monkeypatch.setenv('RAON_PREFILL_CHUNK', '')
    if value is None:
        monkeypatch.delenv('RAON_PREFILL_CHUNK', raising=False)
    else:
        monkeypatch.setenv('RAON_PREFILL_CHUNK', value)
    monkeypatch.setenv('RAON_WEIGHTS_D', str(tmp_path))
    monkeypatch.setattr(sys, 'argv', ['selfcheck'])
    monkeypatch.setattr(selfcheck, 'audit', lambda repo: {'stale': []})
    digest = hashlib.sha256('thought|answer'.encode()).hexdigest()[:16]
    monkeypatch.setattr(selfcheck, 'EXPECTED_HASH', digest if passes else 'wrong')
    class Engine:
        def __init__(self, weights, model, preload):
            assert weights == model == tmp_path and preload
        def chat(self, messages, options):
            assert messages == selfcheck.messages
            assert options.temperature == 0 and options.max_tokens == 32
            yield SimpleNamespace(text='', thinking='thought', completion_tokens=1, prefill_s=2, decode_s=3)
            yield SimpleNamespace(text='answer', thinking='', completion_tokens=32, prefill_s=4, decode_s=5)
    monkeypatch.setitem(sys.modules, 'serve.engine', SimpleNamespace(Engine=Engine, GenOptions=SimpleNamespace))
    assert selfcheck.main() == (0 if passes else 1)
    output = capsys.readouterr().out
    assert f'hash={digest}' in output
    assert 'prefill_s=4, decode_s=5, completion_tokens=32' in output
    assert ('RESULT: PASS' if passes else 'RESULT: FAIL') in output
    import os
    assert os.environ['RAON_PREFILL_CHUNK'] == (value or '256')


def test_selfcheck_stale_stops_before_engine_import(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, 'argv', ['selfcheck', '--weights-d', str(tmp_path)])
    monkeypatch.setattr(selfcheck, 'audit', lambda repo: {'stale': ['old.o']})
    monkeypatch.setitem(sys.modules, 'serve.engine', None)
    assert selfcheck.main() == 1
    assert '--fix' in capsys.readouterr().out


def test_selfcheck_requires_weights(monkeypatch):
    monkeypatch.delenv('RAON_WEIGHTS_D', raising=False)
    monkeypatch.setattr(sys, 'argv', ['selfcheck'])
    with pytest.raises(SystemExit) as exc:
        selfcheck.main()
    assert exc.value.code == 2
