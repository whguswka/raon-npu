# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Generate grouped B1/B2 MLIR and compare device-only configuration text."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path

from kernels.moe_grouped.design import NPU_AVAILABLE, moe_grouped
from kernels.moe_grouped.kernel import grouped_kernel
from kernels.moe_grouped.layout import B1, B2, GroupedShape
from kernels.moe_grouped.op import validate_design_pair


def _brace_delta(line: str) -> int:
    delta = 0
    quoted = False
    escaped = False
    for char in line:
        if escaped:
            escaped = False
        elif char == "\\" and quoted:
            escaped = True
        elif char == '"':
            quoted = not quoted
        elif not quoted and char == "{":
            delta += 1
        elif not quoted and char == "}":
            delta -= 1
    return delta


def device_without_runtime_sequence(mlir: str) -> str:
    """Return MLIR with its complete runtime-sequence operation removed."""
    kept: list[str] = []
    removing = False
    depth = 0
    found = False
    for line in mlir.splitlines():
        if not removing and "runtime_sequence" in line:
            removing = True
            found = True
            depth = _brace_delta(line)
            if depth <= 0:
                removing = False
            continue
        if removing:
            depth += _brace_delta(line)
            if depth <= 0:
                removing = False
            continue
        kept.append(line.rstrip())
    if not found or removing:
        raise ValueError("could not isolate a complete runtime_sequence operation")
    return "\n".join(kept) + "\n"


def _generate(dev: object, shape: GroupedShape) -> str:
    return str(
        moe_grouped(
            dev,
            shape.name,
            grouped_kernel=grouped_kernel(shape.mode),
        )
    )


def _diff_counts(left: str, right: str) -> tuple[int, int]:
    diff = difflib.unified_diff(left.splitlines(), right.splitlines(), lineterm="")
    added = removed = 0
    for line in diff:
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dump grouped B1/B2 MLIR without compiling or opening the NPU"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not NPU_AVAILABLE:
        raise RuntimeError("AIE/IRON packages are required to generate MLIR")

    from aie import utils as aie_utils
    from aie.iron.device import NPU2

    dev = NPU2()
    aie_utils.set_current_device(dev)
    args.out.mkdir(parents=True, exist_ok=True)
    generated: dict[str, str] = {}
    devices: dict[str, str] = {}
    for shape in (B1, B2):
        validate_design_pair(shape, shape)
        config_mlir = _generate(dev, shape)
        command_mlir = _generate(dev, shape)
        config_device = device_without_runtime_sequence(config_mlir)
        command_device = device_without_runtime_sequence(command_mlir)
        same = config_device == command_device
        print(
            f"{shape.name}: xclbin={shape.name}, instructions={shape.name}, "
            f"device={'MATCH' if same else 'MISMATCH'}"
        )
        if not same:
            raise RuntimeError(f"{shape.name} xclbin/instruction device configuration mismatch")
        path = args.out / f"moe_grouped_{shape.name}.mlir"
        path.write_text(command_mlir, encoding="utf-8")
        generated[shape.name] = command_mlir
        devices[shape.name] = command_device

    added, removed = _diff_counts(devices[B1.name], devices[B2.name])
    relation = "identical" if devices[B1.name] == devices[B2.name] else "different"
    print(f"b1 vs b2 device: {relation}; diff +{added}/-{removed} lines")
    print(f"wrote {len(generated)} MLIR files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
