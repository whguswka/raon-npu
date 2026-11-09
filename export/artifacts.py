# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Byte-preserving tokenizer assets and portable sha256sum verification."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil

TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja', 'generation_config.json')


def require_tokenizer_files(directory):
    directory = Path(directory)
    missing = [name for name in TOKENIZER_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Missing tokenizer files in {directory}: {", ".join(missing)}')


def copy_tokenizer_files(source, destination):
    source, destination = Path(source), Path(destination)
    require_tokenizer_files(source)
    destination.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        if (source / name).resolve() != (destination / name).resolve():
            shutil.copyfile(source / name, destination / name)


def verify_expected(directory, expected):
    """Compare every listed file and report missing, unexpected and changed files.

    manifest.json is intentionally excluded because it contains a creation timestamp.
    """
    directory = Path(directory).resolve()
    entries = {}
    for number, line in enumerate(Path(expected).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r'([0-9a-fA-F]{64}) [ *](.+)', line)
        if not match:
            raise ValueError(f'Invalid sha256sum line {number}')
        digest, name = match.groups()
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError(f'Unsafe relative filename on line {number}: {name}')
        if name in entries:
            raise ValueError(f'Duplicate filename on line {number}: {name}')
        entries[name] = digest.lower()
    if not entries:
        raise ValueError('Expected checksum file is empty')
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    errors = []
    for name in sorted(entries.keys() - actual):
        errors.append(f'MISSING: {name}')
    for name in sorted(actual - entries.keys() - {'manifest.json'}):
        errors.append(f'UNEXPECTED: {name}')
    for name in sorted(entries.keys() & actual):
        path = directory / name
        if not path.resolve().is_relative_to(directory):
            errors.append(f'OUTSIDE ROOT: {name}')
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(block)
        if digest.hexdigest() != entries[name]:
            errors.append(f'MISMATCH: {name}')
    return errors
