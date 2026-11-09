# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
"""MLIR operator and patched callable for grouped B1/B2 streams."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from kernels.moe_grouped.layout import SLOTS, TMAX, GroupedShape, get_shape
from kernels.moe_grouped.patch import GroupedPatchedTemplateKernel

_IMPORT_ERROR: Exception | None = None
try:
    import aie.utils as aie_utils
    from ml_dtypes import bfloat16
    from iron.common import AIERuntimeArgSpec, DesignGenerator, KernelObjectArtifact, MLIROperator, PythonGeneratedMLIRArtifact
    from iron.common.compilation import InstsBinArtifact, XclbinArtifact
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as error:
    _IMPORT_ERROR = error
    NPU_AVAILABLE = False
    aie_utils = None
    bfloat16 = np.uint16
    MLIROperator = object


def validate_design_pair(
    xclbin_shape: str | GroupedShape,
    insts_shape: str | GroupedShape,
) -> GroupedShape:
    """Reject an instruction stream paired with another mode's device image."""
    config = get_shape(xclbin_shape)
    stream = get_shape(insts_shape)
    if config != stream:
        raise ValueError(
            "grouped artifact mode mismatch: "
            f"xclbin={config.name}, instructions={stream.name}"
        )
    return config


@dataclass
class GroupedMoE(MLIROperator):
    shape_name: str
    context: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.shape: GroupedShape = get_shape(self.shape_name)
        if MLIROperator is not object:
            MLIROperator.__init__(self, context=self.context)

    @property
    def config_name(self) -> str:
        dev = aie_utils.get_current_device().resolve().name if aie_utils is not None else "npu2"
        # "rtp7": the device reads 7 RTP words (rows in word 4, the balanced extra round in words 5-6; DESIGN-prefill.md
        # section 49). The 4- and 5-word designs' files keep their old names, so a stale xclbin is never paired with
        # this instruction stream (or the reverse).
        return f"Raon_MoE_Grouped_e8_t8_rtp7_{self.shape.name}_{dev}"

    @property
    def name(self) -> str:
        return f"{self.config_name}_commands"

    @property
    def artifact_designs(self) -> tuple[str, str]:
        """Return the xclbin and instruction-stream design modes, in order."""
        return (self.shape.name, self.shape.name)

    def _mlir(self, filename: str, shape_name: str):
        dev = aie_utils.get_current_device() if aie_utils is not None else None
        from kernels.moe_grouped.kernel import grouped_kernel
        shape = get_shape(shape_name)
        return PythonGeneratedMLIRArtifact(
            filename,
            DesignGenerator(
                Path(__file__).with_name("design.py"), "moe_grouped",
                (dev, shape_name), {"grouped_kernel": grouped_kernel(shape.mode)},
            ),
        )

    def get_kernel_artifacts(self):
        from kernels.moe_grouped.kernel import grouped_kernel
        return [KernelObjectArtifact.from_extern(grouped_kernel(self.shape.mode))]

    def get_mlir_artifact(self):
        """Return the instruction-stream MLIR required by ``MLIROperator``."""
        return self._mlir(f"{self.name}.mlir", self.shape.name)

    def set_up_artifacts(self) -> None:
        if not NPU_AVAILABLE:
            raise RuntimeError(f"AIE/IRON packages are unavailable: {_IMPORT_ERROR}")
        kernels = self.get_kernel_artifacts()
        config_mode, stream_mode = self.artifact_designs
        validate_design_pair(config_mode, stream_mode)
        config = self._mlir(f"{self.config_name}.mlir", config_mode)
        stream = self.get_mlir_artifact()
        self.mlir_artifact = stream
        self.xclbin_artifact = XclbinArtifact(
            f"{self.config_name}.xclbin", mlir_input=config,
            dependencies=[config] + kernels, kernel_name="MLIR_AIE",
            extra_flags=["--dynamic-objFifos"],
        )
        self.insts_artifact = InstsBinArtifact(
            f"{self.name}.bin", mlir_input=stream,
            dependencies=[stream] + kernels, extra_flags=["--dynamic-objFifos"],
        )
        if hasattr(self, "add_artifacts"):
            self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("out", (SLOTS * TMAX * self.shape.m,), dtype=bfloat16),
            AIERuntimeArgSpec("in", (self.shape.weight_words,), dtype=np.uint32),
            AIERuntimeArgSpec("in", (SLOTS * TMAX * self.shape.k,), dtype=bfloat16),
        ]

    def get_callable(self) -> GroupedPatchedTemplateKernel:
        if getattr(self, "xclbin_artifact", None) is None:
            raise RuntimeError("compile/set_up_artifacts must run before get_callable")
        return GroupedPatchedTemplateKernel(
            Path(self.xclbin_artifact.filename).resolve(),
            Path(self.insts_artifact.filename).resolve(), self.shape,
        )
