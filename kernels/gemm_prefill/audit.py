# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""링크된 코어 ELF .text 크기를 읽는 의존성 없는 노트북 검증 CLI."""
import argparse
import json
from pathlib import Path
import struct

from .budget import CORE_PROGRAM_LIMIT_BYTES


def section_sizes(path):
    data = Path(path).read_bytes()
    if data[:4] != b'\x7fELF' or data[4] not in (1,2) or data[5] not in (1,2):
        raise ValueError(f'{path}: not an ELF32/ELF64 file')
    endian = '<' if data[5] == 1 else '>'
    if data[4] == 1:
        header = struct.unpack_from(endian+'HHIIIIIHHHHHH',data,16)
        fmt = endian+'IIIIIIIIII'
    else:
        header = struct.unpack_from(endian+'HHIQQQIHHHHHH',data,16)
        fmt = endian+'IIQQQQIIQQ'
    shoff, entsize, count, stridx = header[5],header[10],header[11],header[12]
    if not count or stridx >= count or entsize < struct.calcsize(fmt):
        raise ValueError('missing/unsupported section table')
    sections = [struct.unpack_from(fmt,data,shoff+i*entsize) for i in range(count)]
    names_section = sections[stridx]
    names = data[names_section[4]:names_section[4]+names_section[5]]
    result = {}
    for section in sections:
        end = names.find(b'\0',section[0])
        name = names[section[0]:end].decode('utf-8', errors='replace')
        result[name] = result.get(name, 0) + section[5]
    return result


def text_size(path):
    sections = section_sizes(path)
    total = sum(size for name, size in sections.items()
                if name == '.text' or name.startswith('.text.'))
    if not total:
        raise ValueError('ELF has no .text section')
    return total


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--elf',type=Path,action='append',required=True)
    args=parser.parse_args(argv)
    try:
        rows=[dict(elf=str(p),text_bytes=text_size(p)) for p in args.elf]
        passed=all(r['text_bytes'] <= CORE_PROGRAM_LIMIT_BYTES for r in rows)
        print(json.dumps(rows,indent=2))
    except (ValueError,OSError,struct.error) as exc:
        print(str(exc))
        passed=False
    print('RESULT: '+('PASS' if passed else 'FAIL'))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
