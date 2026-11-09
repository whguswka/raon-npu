# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Check copied kernel objects: IRON on Windows does not refresh existing copies."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def audit(repo, fix=False):
    build = Path(repo) / 'build'
    arch = build / 'aie2p'
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    fresh = {p.name: digest(p) for p in sorted(arch.glob('*.o'))}
    stale = []
    checked = 0
    for name, want in fresh.items():
        for copy in sorted(build.glob(f'*.mlir.d/{name}')):
            checked += 1
            if digest(copy) != want:
                stale.append({'object': name, 'copy': str(copy)})
    fixed = 0
    if fix:
        for item in stale:
            for p in (Path(item['copy']), Path(item['copy'] + '.d')):
                if p.exists():
                    p.unlink()
                    fixed += 1
        for name in {item['object'] for item in stale}:
            os.utime(arch / name, None)
    print(f'Objects: {len(fresh)}, copies: {checked}, stale: {len(stale)}, removed: {fixed}')
    for item in stale:
        print(f"STALE: {item['copy']}")
    return {'fresh': fresh, 'checked': checked, 'stale': stale, 'fixed_files': fixed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--fix', action='store_true')
    args = parser.parse_args()
    if not args.repo.is_dir():
        parser.error('--repo must be an existing directory')
    result = audit(args.repo, args.fix)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return int(bool(result['stale']) and not args.fix)


if __name__ == '__main__':
    raise SystemExit(main())
