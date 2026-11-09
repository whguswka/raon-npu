#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Count nonzero bfp16ebs8 blocks that NPU packing would flush.

The GGUF is memory-mapped and processed one tensor and a bounded row chunk at a
time. Native Q4_K tensors are excluded because they stay on the q4nx path.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant.bfp16 import _BLOCK_DTYPE, bfp16ebs8_encode
from quant.gguf_q4k import GGML_TYPES, GGUFReader




def count_tensor(
    reader: GGUFReader,
    name: str,
    *,
    rows_per_chunk: int,
    rounding: str,
) -> dict[str, int | str | list[int]]:
    """Dequantize and encode one matrix tensor in bounded row chunks."""
    info = reader.tensors[name]
    shape = info.shape
    width = shape[-1]
    rows = int(np.prod(shape[:-1]))
    tiny_blocks = 0
    tiny_nonzero_blocks = 0

    for start in range(0, rows, rows_per_chunk):
        stop = min(start + rows_per_chunk, rows)
        values = reader.dequantize(name, row_slice=slice(start, stop))
        matrix = np.ascontiguousarray(values.reshape(stop - start, width), dtype=np.float32)
        encoded = bfp16ebs8_encode(matrix, rounding=rounding)
        blocks = encoded.view(_BLOCK_DTYPE).reshape(stop - start, width // 8)
        tiny = blocks["exponent"] < 7
        tiny_blocks += int(np.count_nonzero(tiny))
        tiny_nonzero_blocks += int(
            np.count_nonzero(tiny & np.any(blocks["mantissas"] != 0, axis=-1))
        )

    return {
        "name": name,
        "type": info.type_name,
        "shape": list(shape),
        "blocks": rows * width // 8,
        "tiny_blocks": tiny_blocks,
        "tiny_nonzero_blocks": tiny_nonzero_blocks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Count nonzero e<7 blocks in non-Q4_K GGUF weight tensors."
    )
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--include", help="regular expression applied to GGUF tensor names")
    parser.add_argument("--limit", type=int, help="stop after this many matching tensors")
    parser.add_argument("--rows-per-chunk", type=int, default=4096)
    parser.add_argument("--rounding", choices=("floor", "conv_even"), default="conv_even")
    args = parser.parse_args()
    if args.rows_per_chunk < 1:
        parser.error("--rows-per-chunk must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")

    include = re.compile(args.include) if args.include else None
    reader = GGUFReader(args.gguf)
    names = []
    for name, info in reader.tensors.items():
        if not name.endswith(".weight") or len(info.shape) < 2:
            continue
        if info.ggml_type == 12 or info.ggml_type not in GGML_TYPES:  # native Q4_K / unsupported
            continue
        if info.shape[-1] % 8 or (include and not include.search(name)):
            continue
        names.append(name)
    names.sort()
    if args.limit is not None:
        names = names[: args.limit]

    results = []
    for name in names:
        result = count_tensor(
            reader,
            name,
            rows_per_chunk=args.rows_per_chunk,
            rounding=args.rounding,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)

    summary = {
        "gguf": str(args.gguf),
        "rounding": args.rounding,
        "tensors": len(results),
        "blocks": sum(int(item["blocks"]) for item in results),
        "tiny_blocks": sum(int(item["tiny_blocks"]) for item in results),
        "tiny_nonzero_blocks": sum(
            int(item["tiny_nonzero_blocks"]) for item in results
        ),
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
