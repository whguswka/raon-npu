# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Compare unified GEMV device MLIR across shapes without compiling it."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.gemv_unified.op import NPU_AVAILABLE, UnifiedGEMV
from kernels.gemv_unified.shapes import ALL_SHAPES, COLS, MAX_K, UnifiedShape, get_shape


DEFAULT_REFERENCE = "256:1024:8"
DEFAULT_BOUNDARY_SHAPES = ",".join(
    (
        "256:128:1",
        "256:1024:8",
        "512:5120:8",
        "768:128:1",
        "2560:1152:8",
        f"{MAX_K}:5120:8",
    )
)
SSA = re.compile(r"%[-$._A-Za-z0-9]+")


@dataclass(frozen=True)
class InvarianceCase:
    """One exact supported shape or an explicitly experimental boundary shape."""

    shape: UnifiedShape
    experimental: bool

    @property
    def label(self) -> str:
        prefix = self.shape.name if not self.experimental else "boundary"
        return (
            f"{prefix} K={self.shape.K} M={self.shape.M} "
            f"format={self.shape.format} cols={self.shape.active_cols} "
            f"x_slots={self.shape.x_slots} chunks_per_col={self.shape.chunks_per_col}"
        )


def parse_shape(text: str) -> tuple[int, int, int]:
    """Parse K:M[:active_cols]."""
    fields = text.split(":")
    if len(fields) not in (2, 3):
        raise argparse.ArgumentTypeError(f"expected K:M[:cols], got {text!r}")
    try:
        k, m = (int(value) for value in fields[:2])
        cols = int(fields[2]) if len(fields) == 3 else COLS
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape values must be integers: {text!r}") from exc
    return k, m, cols


def parse_shapes(text: str) -> tuple[tuple[int, int, int], ...]:
    """Parse a comma-separated list of K:M[:active_cols] shapes."""
    shapes = tuple(parse_shape(item.strip()) for item in text.split(",") if item.strip())
    if not shapes:
        raise argparse.ArgumentTypeError("--shapes must contain at least one shape")
    return shapes


def parse_names(text: str) -> tuple[str, ...]:
    """Parse and validate a comma-separated list of registered shape names."""
    names = tuple(item.strip() for item in text.split(",") if item.strip())
    if not names:
        raise argparse.ArgumentTypeError("--names must contain at least one shape name")
    for name in names:
        try:
            get_shape(name)
        except KeyError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc
    return names


def select_cases(
    names: tuple[str, ...] | None,
    shapes: tuple[tuple[int, int, int], ...] | None,
) -> tuple[InvarianceCase, ...]:
    """Resolve CLI selection, using all supported plus boundary cases by default."""
    if names is None and shapes is None:
        names = tuple(shape.name for shape in ALL_SHAPES)
        shapes = parse_shapes(DEFAULT_BOUNDARY_SHAPES)

    cases = [
        InvarianceCase(get_shape(name), experimental=False)
        for name in (names or ())
    ]
    cases.extend(
        InvarianceCase(
            UnifiedShape(
                f"boundary_{k}_{m}_{active_cols}",
                k,
                m,
                active_cols=active_cols,
            ),
            experimental=True,
        )
        for k, m, active_cols in (shapes or ())
    )
    return tuple(cases)


def _brace_delta(line: str) -> int:
    """Count braces outside MLIR string literals."""
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
    """Remove the complete aie.runtime_sequence operation from textual MLIR."""
    kept = []
    removing = False
    depth = 0
    found = False
    for line in mlir.splitlines():
        if not removing and "aie.runtime_sequence" in line:
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
        raise ValueError("could not isolate a complete aie.runtime_sequence operation")
    return "\n".join(kept) + "\n"


def normalize_ssa(text: str) -> str:
    """Canonicalize SSA identifiers by first occurrence, preserving use structure."""
    names: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        name = match.group(0)
        if name not in names:
            names[name] = f"%ssa{len(names)}"
        return names[name]

    return SSA.sub(replace, text)


def generate_mlir(case: InvarianceCase, *, continuous: bool) -> str:
    """Generate one MLIR module through the operator's lazy generator."""
    from iron.common import AIEContext

    context = AIEContext(build_dir=Path.cwd() / "build")
    if case.experimental:
        shape = case.shape
        op = UnifiedGEMV(
            K=shape.K,
            M=shape.M,
            format=shape.format,
            active_cols=shape.active_cols,
            x_slots=shape.x_slots,
            chunks_per_col=shape.chunks_per_col,
            continuous_mem_w=continuous,
            experimental_shape=True,
            context=context,
        )
    else:
        op = UnifiedGEMV.for_shape(case.shape, context=context)
        op.continuous_mem_w = continuous
    return op.get_mlir_artifact().generator()


def print_first_differences(reference: str, candidate: str, limit: int = 20) -> None:
    """Print the first differing line pairs with stable line numbers."""
    shown = 0
    for line_no, (left, right) in enumerate(
        zip_longest(reference.splitlines(), candidate.splitlines(), fillvalue="<EOF>"),
        start=1,
    ):
        if left == right:
            continue
        print(f"  line {line_no}:")
        print(f"    ref: {left}")
        print(f"    got: {right}")
        shown += 1
        if shown == limit:
            break


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=parse_shape, default=parse_shape(DEFAULT_REFERENCE))
    p.add_argument("--names", type=parse_names, help="comma-separated registered shape names")
    p.add_argument("--shapes", type=parse_shapes, help="experimental K:M[:cols] shapes")
    p.add_argument("--continuous", action="store_true", help="generate continuous-W plans")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not NPU_AVAILABLE:
        raise SystemExit("IRON/aie Python packages are required to generate MLIR")

    ref_k, ref_m, ref_cols = args.reference
    reference_case = InvarianceCase(
        UnifiedShape("reference", ref_k, ref_m, active_cols=ref_cols),
        experimental=True,
    )
    reference = device_without_runtime_sequence(
        generate_mlir(reference_case, continuous=args.continuous)
    )
    failed = False
    for case in select_cases(args.names, args.shapes):
        candidate = device_without_runtime_sequence(
            generate_mlir(case, continuous=args.continuous)
        )
        label = case.label
        if candidate == reference:
            print(f"PASS {label}")
            continue
        normalized_reference = normalize_ssa(reference)
        normalized_candidate = normalize_ssa(candidate)
        if normalized_candidate == normalized_reference:
            print(f"PASS {label} (SSA names normalized)")
            continue
        failed = True
        print(f"FAIL {label} (showing up to 20 normalized differing lines)")
        print_first_differences(normalized_reference, normalized_candidate)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
