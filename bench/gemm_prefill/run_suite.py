# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entry point for the consolidated notebook benchmark."""

from bench.gemm_prefill.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
