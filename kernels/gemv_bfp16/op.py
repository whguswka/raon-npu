# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# Portions ported from amd/IRON (Apache-2.0); modified for raon-npu
# SPDX-License-Identifier: Apache-2.0
# Modified by the raon-npu authors, 2026.

"""GEMV_BFP16 MLIROperator for Raon-1-mini general 9-bit bfp16ebs8 projection."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
try:
    from ml_dtypes import bfloat16
except (ImportError, ModuleNotFoundError):
    bfloat16 = np.uint16

try:
    import aie.utils as aie_utils
    from iron.common import (
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
except (ImportError, ModuleNotFoundError):
    aie_utils = None
    AIERuntimeArgSpec = DesignGenerator = KernelObjectArtifact = object
    MLIROperator = PythonGeneratedMLIRArtifact = object

from quant.bfp16 import GROUP, K_TILE, M_TILE, packed_bytes
from kernels.gemv_bfp16.design import check_shape
from kernels.gemv_bfp16.kernel import raon_bfp16_gemv


@dataclass
class GEMV_BFP16(MLIROperator):
    """Linear projection from a 9-bit bfp16ebs8 matrix (softcap-free, RMS-free).

    ``K`` in-features (multiple of 256), ``M`` out-features (multiple of 128).
    ``cols`` active core columns (default 8).
    X holds the input vector of length K (bf16).
    W holds the 9-bit bfp16 packed weights of shape [M, K].
    Y receives the output vector of length M (bf16).
    """

    K: int = 2560
    M: int = 2560
    cols: int = 8
    context: object = field(default=None, repr=False)

    def __post_init__(self):
        if aie_utils is not None:
            check_shape(aie_utils.get_current_device(), self.K, self.M, self.cols)
            MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        dev = aie_utils.get_current_device().resolve().name if aie_utils else "npu"
        return f"Raon_GEMV_BFP16_k{self.K}_m{self.M}_c{self.cols}_{dev}"

    def quantized_size(self) -> int:
        """Bytes of 9-bit bfp16ebs8 weights the operator reads."""
        return packed_bytes(self.M * self.K)

    def _kernel(self):
        """The custom raon_bfp16_gemv kernel for K."""
        return raon_bfp16_gemv(
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
        from kernels.gemv_bfp16.reference import dequantize, reference

        dev = aie_utils.get_current_device() if aie_utils else None
        rows = len(dev.core_rows) if dev else 4
        weights = dequantize(w, self.K, self.M, rows)
        return reference(weights, np.asarray(x, np.float64))
