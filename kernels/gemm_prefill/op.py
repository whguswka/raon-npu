# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Operator for the isolated, statically generated batch-GEMV experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .budget import resource_budget
from .kernel import prefill_kernel
from .shapes import MAX_BATCH_T, BatchShape, get_shape, validate_batch_t
from bench.rda_state.rda_state_1col import (
    HEAD_BUNDLE_SIZE,
    K_DIM,
    MODEL_RDA_HEADS,
    TILE_V,
    TILES_PER_HEAD,
    V_DIM,
)
from bench.rda_state.rda_state_multi import (
    RDA_MAX_TOKENS,
    RDA_STATE_KERNEL_PATH,
    check_rda_tokens,
    make_rda_multi_kernel_object,
)

_IMPORT_ERROR: Exception | None = None
try:
    import aie.utils as aie_utils
    from ml_dtypes import bfloat16
    from iron.common import (
        AIERuntimeArgSpec,
        DesignGenerator,
        KernelObjectArtifact,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
    from iron.common.compilation import InstsBinArtifact, SourceArtifact, XclbinArtifact

    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False
    aie_utils = None
    bfloat16 = np.uint16
    class AIERuntimeArgSpec:  # type: ignore[no-redef]
        def __init__(self, direction, shape, dtype=None):
            self.direction = direction
            self.shape = shape
            self.dtype = dtype

    DesignGenerator = KernelObjectArtifact = object
    MLIROperator = PythonGeneratedMLIRArtifact = object
    InstsBinArtifact = SourceArtifact = XclbinArtifact = object

KERNEL_NAME = "MLIR_AIE"


class PrefillCallable:
    """Validate the fixed padded BO ABI and expose the logical output view."""

    def __init__(self, inner: Any, op: "PrefillGEMM") -> None:
        self._inner = inner
        self.op = op

    def __call__(self, Y: Any, W: Any, X: Any) -> Any:
        return self._inner(Y, W, X)

    def logical_output(self, Y: Any) -> np.ndarray:
        return np.asarray(Y.numpy()).reshape(self.op.max_batch_t, self.op.M)[
            : self.op.batch_t
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class PrefillGEMM(MLIROperator):
    """One shape stream sharing the revision-s2a, fixed-T (MAX_BATCH_T rows) device body.

    ``rda_tokens`` is the T of an ``execution="rda"`` stream (1, 2, 4 or 8 tokens per call).
    """

    shape_name: str
    batch_t: int
    max_batch_t: int = MAX_BATCH_T
    cols: int = 7
    variant: str = "exact"
    mode: int = 0
    rda_col: int | None = 7
    execution: str = "gemm"
    rda_tokens: int = 1
    context: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.variant != "exact" or self.mode not in (0, 1, 2, 3):
            raise ValueError("invalid variant or diagnostic mode")
        if self.execution not in ("gemm", "rda"):
            raise ValueError("execution must be 'gemm' or 'rda'")
        if self.rda_col is not None and (self.cols != 7 or self.rda_col != 7):
            raise ValueError("RDA prototype requires cols=7 and rda_col=7")
        if self.execution == "rda" and self.rda_col is None:
            raise ValueError("rda execution requires rda_col=7")
        if self.execution == "rda":
            check_rda_tokens(self.rda_tokens)
        elif self.rda_tokens != 1:
            raise ValueError("rda_tokens applies to the rda execution only")
        self.shape: BatchShape = get_shape(self.shape_name)
        validate_batch_t(self.batch_t, max_batch_t=self.max_batch_t)
        budget = resource_budget(self.shape, self.max_batch_t, cols=self.cols)
        if not budget.core_fits:
            raise ValueError(
                f"{self.shape.name} T={self.max_batch_t} exceeds core L1: "
                f"{budget.core_total_bytes} > 65536"
            )
        if not budget.memtile_fits:
            raise ValueError(
                f"{self.shape.name} T={self.max_batch_t} exceeds conservative "
                f"memtile budget: {budget.memtile_total_bytes} > 524288"
            )
        if MLIROperator is not object:
            MLIROperator.__init__(self, context=self.context)

    @classmethod
    def for_shape(
        cls,
        shape: str | BatchShape,
        *,
        batch_t: int,
        max_batch_t: int = MAX_BATCH_T,
        cols: int = 7,
        context: object = None,
        variant: str = "exact",
        mode: int = 0,
        rda_col: int | None = 7,
        execution: str = "gemm",
        rda_tokens: int = 1,
    ) -> "PrefillGEMM":
        resolved = get_shape(shape)
        return cls(
            resolved.name,
            batch_t,
            max_batch_t=max_batch_t,
            cols=cols,
            context=context,
            variant=variant,
            mode=mode,
            rda_col=rda_col,
            execution=execution,
            rda_tokens=rda_tokens,
        )

    @property
    def K(self) -> int:
        return self.shape.K

    @property
    def M(self) -> int:
        return self.shape.M

    @property
    def config_name(self) -> str:
        dev = (
            aie_utils.get_current_device().resolve().name
            if aie_utils is not None
            else "npu2"
        )
        rda = f"_rda7t{RDA_MAX_TOKENS}" if self.rda_col is not None else ""
        return f"Raon_GEMM_Prefill_s2a_tmax{self.max_batch_t}_c{self.cols}{rda}_{self.variant}_{dev}"

    @property
    def name(self) -> str:
        suffix = self.shape.name if self.execution == "gemm" else f"rda_t{self.rda_tokens}"
        return f"{self.config_name}_{suffix}_mode{self.mode}"

    def _kernel(self):
        return prefill_kernel(self.max_batch_t, self.variant, self.shape.format)

    def get_kernel_artifacts(self):
        artifacts = [KernelObjectArtifact.from_extern(self._kernel())]
        if self.rda_col is not None:
            rda_kernel = make_rda_multi_kernel_object()
            if rda_kernel is not None:
                rda_object = KernelObjectArtifact.from_extern(rda_kernel)
                # rda_state_multi.cc includes rda_state.cc: the object is out of date when either file is newer.
                rda_object.dependencies.add(SourceArtifact(str(RDA_STATE_KERNEL_PATH)))
                artifacts.append(rda_object)
        return artifacts

    def _mlir_artifact(
        self, filename: str, batch_t: int, mode=None, shape_name=None, execution=None
    ):
        dev = aie_utils.get_current_device() if aie_utils is not None else None
        return PythonGeneratedMLIRArtifact(
            filename,
            DesignGenerator(
                Path(__file__).parent / "design.py",
                "gemm_prefill",
                (
                    dev,
                    shape_name or self.shape.name,
                    batch_t,
                    self.max_batch_t,
                    self.cols,
                    0,
                ),
                {"gemv_kernel": self._kernel(), "variant": self.variant,
                 "mode": self.mode if mode is None else mode,
                 "rda_col": self.rda_col,
                 "execution": self.execution if execution is None else execution,
                 "rda_tokens": self.rda_tokens if (execution or self.execution) == "rda" else 1},
            ),
        )

    def get_mlir_artifact(self):
        """Return the instruction-stream MLIR required by ``MLIROperator``."""
        return self._mlir_artifact(f"{self.name}.mlir", self.batch_t)

    def set_up_artifacts(self) -> None:
        if not NPU_AVAILABLE:
            raise RuntimeError(f"AIE/IRON packages are unavailable: {_IMPORT_ERROR}")
        kernels = self.get_kernel_artifacts()
        config_mlir = self._mlir_artifact(
            f"{self.config_name}.mlir", self.max_batch_t, mode=0, shape_name="rda_in",
            execution="gemm",
        )
        stream_mlir = self.get_mlir_artifact()
        self.mlir_artifact = stream_mlir
        self.xclbin_artifact = XclbinArtifact(
            f"{self.config_name}.xclbin",
            mlir_input=config_mlir,
            dependencies=[config_mlir] + kernels,
            kernel_name=KERNEL_NAME,
            extra_flags=["--dynamic-objFifos"],
        )
        self.insts_artifact = InstsBinArtifact(
            f"{self.name}.bin",
            mlir_input=stream_mlir,
            dependencies=[stream_mlir] + kernels,
            extra_flags=["--dynamic-objFifos"],
        )
        if hasattr(self, "add_artifacts"):
            self.add_artifacts([self.xclbin_artifact, self.insts_artifact])

    def get_arg_spec(self):
        if self.execution == "rda":
            return [
                AIERuntimeArgSpec("in", (MODEL_RDA_HEADS, V_DIM, K_DIM), dtype=np.float32),
                AIERuntimeArgSpec("out", (MODEL_RDA_HEADS, V_DIM, K_DIM), dtype=np.float32),
                AIERuntimeArgSpec(
                    "in", (MODEL_RDA_HEADS, self.rda_tokens, HEAD_BUNDLE_SIZE), dtype=np.float32
                ),
                AIERuntimeArgSpec(
                    "out", (MODEL_RDA_HEADS, TILES_PER_HEAD, self.rda_tokens, TILE_V), dtype=np.float32
                ),
            ]
        return [
            AIERuntimeArgSpec(
                "out", (self.max_batch_t * self.M,), dtype=bfloat16
            ),
            AIERuntimeArgSpec(
                "in", (self.shape.weight_words,), dtype=np.uint32
            ),
            AIERuntimeArgSpec(
                "in", (self.max_batch_t * self.K,), dtype=bfloat16
            ),
        ]

    def get_callable(self) -> PrefillCallable:
        inner = super().get_callable()
        return inner if self.execution == "rda" else PrefillCallable(inner, self)


GEMM_Prefill = PrefillGEMM
