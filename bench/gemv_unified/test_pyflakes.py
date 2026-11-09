# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Undefined-name and dead-import gate for unified GEMV Python sources."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def test_gemv_unified_pyflakes_is_clean() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyflakes",
            str(root / "kernels" / "gemv_unified"),
            str(root / "bench" / "gemv_unified"),
        ],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert completed.returncode == 0, completed.stdout
