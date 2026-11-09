# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""Raon general q4nx GEMV kernel and operator package."""

try:
    from kernels.gemv.kernel import raon_q4nx_gemv
    from kernels.gemv.op import GEMV
    __all__ = ["GEMV", "raon_q4nx_gemv"]
except (ImportError, ModuleNotFoundError):
    # Guard against environments without AIE/IRON/ml_dtypes (e.g. host testing on CPU host)
    __all__ = []
