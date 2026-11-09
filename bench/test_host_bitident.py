# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_host_bitident_missing_weights_exits_two(tmp_path: Path):
    env = os.environ.copy()
    env.pop("RAON_MODEL_DIR", None)
    env.pop("RAON_WEIGHTS_D", None)
    proc = subprocess.run(
        [sys.executable, "bench/host_bitident.py", "--model-dir", str(tmp_path / "missing")],
        cwd=Path(__file__).resolve().parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "no host checkpoint found" in proc.stderr
