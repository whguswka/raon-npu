# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""Raon NPU computation backend for AMD XDNA 2.

GEMV Engine Selection:
- Selected via `RAON_NPU_GEMV` ('unified' [default] or 'legacy') or `run_golden.py --gemv`.
- Benchmark (25 layers, 15 steps): unified 406.5 ms/token (36 switches per token) vs legacy 874.1 ms/token (182 switches per token).
"""

from .backend import DWeights, NpuBackend, ensure_mla_cache_expanded, legacy_gemv_xclbin_name

__all__ = ["DWeights", "NpuBackend", "ensure_mla_cache_expanded", "legacy_gemv_xclbin_name"]
