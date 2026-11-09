# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""GEMV MLIROperator for Raon-1-mini general projection without softcap or RMSNorm."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.utils as aie_utils
from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
)
from iron.operators.flm.q4nx import GROUP, K_TILE, M_TILE, packed_bytes

from kernels.gemv.design import check_shape
from kernels.gemv.kernel import raon_q4nx_gemv


@dataclass
class GEMV(MLIROperator):
    """Linear projection from a q4nx matrix (softcap-free, RMS-free).

    ``K`` in-features (multiple of 256), ``M`` out-features (multiple of 128).
    ``cols`` active core columns (default 8).
    X holds the input vector of length K (bf16).
    W holds the q4nx packed weights of shape [M, K].
    Y receives the output vector of length M (bf16).
    """

    K: int = 2560
    M: int = 2560
    cols: int = 8
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        check_shape(aie_utils.get_current_device(), self.K, self.M, self.cols)
        MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev = aie_utils.get_current_device().resolve().name
        return f"Raon_GEMV_k{self.K}_m{self.M}_c{self.cols}_{dev}"

    def quantized_size(self) -> int:
        """Bytes of q4nx weights the operator reads."""
        return packed_bytes(self.M * self.K)

    def _kernel(self):
        """The custom raon_q4nx_gemv kernel for K."""
        return raon_q4nx_gemv(
            K=self.K, m_tile=M_TILE, k_tile=K_TILE, group=GROUP
        )

    def get_mlir_artifact(self):
        op_dir = Path(__file__).resolve().parent
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                op_dir / "design.py",
                "gemv",
                (
                    aie_utils.get_current_device(),
                    self.K,
                    self.M,
                    self.cols,
                ),
                {"gemv_kernel": self._kernel()},
            ),
        )

    def get_kernel_artifacts(self):
        return [KernelObjectArtifact.from_extern(self._kernel())]

    def get_arg_spec(self):
        # The runtime sequence's argument order: y, w, x.
        return [
            AIERuntimeArgSpec("out", (self.M,), dtype=bfloat16),
            AIERuntimeArgSpec(
                "in",
                (self.quantized_size() // np.dtype(np.uint32).itemsize,),
                dtype=np.uint32,
            ),
            AIERuntimeArgSpec("in", (self.K,), dtype=bfloat16),
        ]

    def reference(self, w, x):
        """CPU reference in float64."""
        from kernels.gemv.reference import dequantize, reference

        dev = aie_utils.get_current_device()
        weights = dequantize(w, self.K, self.M, len(dev.core_rows))
        return reference(weights, np.asarray(x, np.float64))
