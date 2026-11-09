# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Run bounded unified-GEMV stall probes and print one comparison table.

Every probe is a fresh child process.  A hung compile or NPU invocation is
terminated at the per-case timeout, and the remaining probes still run.

PASS means an all-zero W/X invocation completed within the timeout and left
every output bit zero.  This is a deadlock/stall diagnostic, not a general
numerical-accuracy gate; use ``diagnose.py`` for a bit-exact nonzero check.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.gemv_unified.op import UnifiedGEMV  # noqa: E402
from kernels.gemv_unified.shapes import MAX_K  # noqa: E402


DEFAULT_K = 512
DEFAULT_M_ONE_COL = 128
DEFAULT_M_ALL_COLS = 5120


@dataclass(frozen=True)
class Probe:
    name: str
    receive_only: bool = False
    static_core: bool = False
    small_x: bool = False
    active_cols: int = 8
    continuous_mem_w: bool = False
    k: int = DEFAULT_K
    m: int = DEFAULT_M_ALL_COLS


def make_probes(
    k: int = DEFAULT_K,
    m_one_col: int = DEFAULT_M_ONE_COL,
    m_all_cols: int = DEFAULT_M_ALL_COLS,
) -> tuple[Probe, ...]:
    """Build the 7 supported diagnostic cases for a given (K, M) pair."""
    return (
        Probe("baseline", k=k, m=m_all_cols),
        Probe("receive_only", receive_only=True, k=k, m=m_all_cols),
        Probe("static_core", static_core=True, k=k, m=m_all_cols),
        Probe("small_x", small_x=True, k=k, m=m_all_cols),
        Probe("one_col", active_cols=1, k=k, m=m_one_col),
        Probe("continuous_w", continuous_mem_w=True, k=k, m=m_all_cols),
        Probe(
            "minimal_dynamic",
            receive_only=True,
            small_x=True,
            active_cols=1,
            k=k,
            m=m_one_col,
        ),
    )


PROBES = make_probes(DEFAULT_K, DEFAULT_M_ONE_COL, DEFAULT_M_ALL_COLS)
PROBE_BY_NAME = {probe.name: probe for probe in PROBES}


def probe_by_name(probes: tuple[Probe, ...], name: str) -> Probe:
    """Select a probe by its stable CLI name."""
    return {probe.name: probe for probe in probes}[name]


def log(message: str) -> None:
    print(message, flush=True)


def run_child(probe: Probe, *, prepare_only: bool, timeout: float) -> int:
    """Compile and optionally invoke one all-zero, one-block diagnostic."""
    os.environ.setdefault("IRON_HSA_TIMEOUT", str(timeout))

    import aie.utils as aie_utils
    from iron.common import AIEContext
    from ml_dtypes import bfloat16

    op = UnifiedGEMV(
        K=probe.k,
        M=probe.m,
        active_cols=probe.active_cols,
        context=AIEContext(),
        continuous_mem_w=probe.continuous_mem_w,
        diagnostic_receive_only=probe.receive_only,
        diagnostic_static_core=probe.static_core,
        diagnostic_small_x=probe.small_x,
        experimental_shape=True,
    )
    log(f"[child] compile {probe.name} K={probe.k} M={probe.m}")
    op.set_up_artifacts()
    op.compile()
    if prepare_only:
        log("[child] PASS compile-only")
        return 0

    x_elems = probe.k if probe.small_x else MAX_K
    y_bo = aie_utils.full((probe.m,), 99.0, dtype=bfloat16)
    w_bo = aie_utils.zeros((op.shape.weight_words,), dtype=np.uint32)
    x_bo = aie_utils.zeros((x_elems,), dtype=bfloat16)
    log(f"[child] invoke {probe.name}")
    op.get_callable()(y_bo, w_bo, x_bo)
    got = np.asarray(y_bo.numpy()).view(np.uint16)
    nonzero = int(np.count_nonzero(got))
    log(f"[child] {'PASS' if nonzero == 0 else 'FAIL'} nonzero={nonzero}")
    return 0 if nonzero == 0 else 2


def run_matrix(
    probes: tuple[Probe, ...],
    k: int,
    m_one_col: int,
    m_all_cols: int,
    timeout: float,
    *,
    prepare_only: bool,
) -> list[dict[str, Any]]:
    """Run all probes independently, preserving results after timeouts."""
    results: list[dict[str, Any]] = []
    script = str(Path(__file__).resolve())
    log_dir = Path("diag-logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    phase = "prepare" if prepare_only else "run"

    for probe in probes:
        cmd = [
            sys.executable,
            "-u",
            script,
            "--child",
            probe.name,
            "--k",
            str(k),
            "--m-one-col",
            str(m_one_col),
            "--m-all-cols",
            str(m_all_cols),
            "--timeout",
            str(timeout),
        ]
        if prepare_only:
            cmd.append("--prepare-only")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["IRON_HSA_TIMEOUT"] = str(timeout)
        log_path = log_dir / f"{probe.name}-{phase}.log"
        started = time.monotonic()
        try:
            with log_path.open("w", encoding="utf-8") as log_file:
                completed = subprocess.run(
                    cmd,
                    timeout=timeout,
                    env=env,
                    check=False,
                    text=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
            output = log_path.read_text(encoding="utf-8").strip().splitlines()
            status = "PASS" if completed.returncode == 0 else "FAIL"
            note = output[-1] if output else f"exit={completed.returncode}"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
            output = (
                log_path.read_text(encoding="utf-8").strip().splitlines()
                if log_path.exists()
                else []
            )
            note = output[-1] if output else f"exceeded {timeout:.0f}s"
        elapsed = time.monotonic() - started
        results.append(
            {
                "case": probe.name,
                "kernel": "receive" if probe.receive_only else "gemv",
                "core": "static" if probe.static_core else "dynamic",
                "x": f"K={probe.k}" if probe.small_x else f"MAX_K={MAX_K}",
                "K": probe.k,
                "M": probe.m,
                "cols": probe.active_cols,
                "mem_w": "continuous" if probe.continuous_mem_w else "per-round",
                "result": status,
                "seconds": round(elapsed, 1),
                "note": note,
            }
        )
        log(f"[matrix] {probe.name}: {status} ({elapsed:.1f}s) -> {log_path}")
        if status != "PASS" and output:
            log(f"[matrix] last {min(20, len(output))} lines of {probe.name}:")
            for line in output[-20:]:
                log("  " + line)
    return results


def print_table(results: list[dict[str, Any]]) -> None:
    columns = (
        "case",
        "kernel",
        "core",
        "x",
        "K",
        "M",
        "cols",
        "mem_w",
        "result",
        "seconds",
    )
    widths = {
        column: max(len(column), *(len(str(row[column])) for row in results))
        for column in columns
    }
    print("| " + " | ".join(column.ljust(widths[column]) for column in columns) + " |")
    print("|-" + "-|-".join("-" * widths[column] for column in columns) + "-|")
    for row in results:
        print(
            "| "
            + " | ".join(str(row[column]).ljust(widths[column]) for column in columns)
            + " |"
        )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=DEFAULT_K, help="K dimension (default 512)")
    p.add_argument(
        "--m-one-col", type=int, default=DEFAULT_M_ONE_COL, help="M for active_cols=1"
    )
    p.add_argument(
        "--m-all-cols", type=int, default=DEFAULT_M_ALL_COLS, help="M for active_cols=8"
    )
    p.add_argument(
        "--cases",
        type=lambda s: [c.strip() for c in s.split(",") if c.strip()],
        default=None,
        help="comma-separated case names (default: all)",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="seconds per child")
    p.add_argument("--prepare-only", action="store_true", help="compile without NPU invoke")
    p.add_argument("--json", type=Path, help="also write the result rows as JSON")
    p.add_argument("--child", choices=tuple(PROBE_BY_NAME), help=argparse.SUPPRESS)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.child:
        probe = probe_by_name(
            make_probes(args.k, args.m_one_col, args.m_all_cols), args.child
        )
        return run_child(probe, prepare_only=args.prepare_only, timeout=args.timeout)
    if args.timeout <= 0:
        raise SystemExit("--timeout must be positive")

    probes = make_probes(args.k, args.m_one_col, args.m_all_cols)
    if args.cases:
        unknown = set(args.cases) - {probe.name for probe in probes}
        if unknown:
            raise SystemExit(f"Unknown cases: {', '.join(sorted(unknown))}")
        probes = tuple(probe for probe in probes if probe.name in args.cases)

    results = run_matrix(
        probes,
        args.k,
        args.m_one_col,
        args.m_all_cols,
        args.timeout,
        prepare_only=args.prepare_only,
    )
    print_table(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0 if all(row["result"] == "PASS" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
