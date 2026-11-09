# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0

"""Unified RDA operator sharing the merged GEMV configuration-only xclbin."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from kernels.gemv_unified.op import KERNEL_NAME, UnifiedGEMV
from kernels.gemv_unified.shapes import COLS, M_PER_CHUNK, UnifiedShape

_IMPORT_ERROR: Exception | None = None
try:
    import aie.utils as aie_utils
    from iron.common import (
        AIERuntimeArgSpec,
        DesignGenerator,
        MLIROperator,
        PythonGeneratedMLIRArtifact,
    )
    from iron.common.compilation import InstsBinArtifact
    NPU_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as err:
    _IMPORT_ERROR = err
    NPU_AVAILABLE = False
    aie_utils = None
    class AIERuntimeArgSpec:  # type: ignore[no-redef]
        def __init__(self, direction: str, shape: tuple[int, ...], dtype: Any = None) -> None:
            self.direction = direction
            self.shape = shape
            self.dtype = dtype

    DesignGenerator = object  # type: ignore[assignment,misc]
    MLIROperator = object  # type: ignore[assignment,misc]
    PythonGeneratedMLIRArtifact = object  # type: ignore[assignment,misc]
    InstsBinArtifact = object  # type: ignore[assignment,misc]

from bench.rda_state.rda_state_1col import (
    HEAD_BUNDLE_SIZE,
    K_DIM,
    MODEL_RDA_HEADS,
    V_DIM,
)


def _abi_signature_from_specs(specs: Iterable[Any]) -> tuple[tuple[str, str, int], ...]:
    """Extract direction, dtype, and rank for each BO in the 4-BO RDA ABI."""
    sig = []
    for s in specs:
        direction = getattr(s, "direction", None)
        dtype = str(np.dtype(getattr(s, "dtype", None)))
        shape = getattr(s, "shape", ())
        sig.append((str(direction), dtype, len(shape)))
    return tuple(sig)


class SharedRdaCallable:
    """Thin callable wrapper that preserves metadata for the shared RDA xclbin."""

    def __init__(
        self,
        inner: Any,
        *,
        xclbin_path: Path,
        xclbin_mtime_ns: int,
        kernel_name: str,
        arg_spec: list[Any],
    ) -> None:
        self._inner = inner
        self.xclbin_path = xclbin_path
        self.xclbin_mtime_ns = xclbin_mtime_ns
        self.kernel_name = kernel_name
        self.arg_spec = list(arg_spec)
        self.abi_signature = _abi_signature_from_specs(self.arg_spec)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if len(args) != 4 or kwargs:
            raise ValueError(
                f"UnifiedRDA callable requires positional 4-BO ABI "
                f"(s_in, s_out, head_in, out), got {len(args)} args and kwargs={list(kwargs)}"
            )
        return self._inner(*args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


@dataclass
class UnifiedRDA(MLIROperator):
    """RDA state update operator sharing the merged unified GEMV xclbin."""

    cols: int = COLS
    rda_col: int | None = None
    rda_heads: int = MODEL_RDA_HEADS
    continuous_mem_w: bool = False
    sequence_plan: str = "per_round"
    context: object = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.rda_col is None:
            raise ValueError("UnifiedRDA requires rda_col")
        if not (0 <= self.rda_col < COLS):
            raise ValueError(
                f"rda_col must be within device columns 0..{COLS - 1}, got {self.rda_col}"
            )
        if self.rda_col < self.cols:
            raise ValueError(
                f"rda_col {self.rda_col} overlaps GEMV active columns 0..{self.cols - 1}"
            )
        if self.rda_heads != MODEL_RDA_HEADS:
            raise ValueError(
                f"UnifiedRDA supports only rda_heads={MODEL_RDA_HEADS} (the model "
                f"contract head count): the shared xclbin/insts artifact names carry no "
                f"head count, and the RDA BD chains run runs=H//2, so any other count "
                f"would share artifacts wrongly or leave heads unconsumed, "
                f"got {self.rda_heads}"
            )

        # Reference shape used only to generate the identical device body as the
        # shared GEMV xclbin; the RDA runtime sequence does not depend on (K, M).
        self._gemv_op = UnifiedGEMV(
            K=256,
            M=self.cols * M_PER_CHUNK,
            cols=self.cols,
            rda_col=self.rda_col,
            rda_heads=self.rda_heads,
            experimental_shape=True,
            continuous_mem_w=self.continuous_mem_w,
            sequence_plan=self.sequence_plan,
            context=self.context,
        )

        if MLIROperator is not object:
            MLIROperator.__init__(self, context=self.context)

    @property
    def name(self) -> str:
        return f"{self._gemv_op.config_name}_rda"

    def _reference_shape(self) -> UnifiedShape:
        return UnifiedShape("rda", 256, self.cols * M_PER_CHUNK, active_cols=self.cols)

    def _mlir_artifact(self, filename: str) -> PythonGeneratedMLIRArtifact:
        design_path = Path(__file__).parent / "design.py"
        dev = aie_utils.get_current_device() if aie_utils is not None else None
        s = self._reference_shape()
        return PythonGeneratedMLIRArtifact(
            filename,
            DesignGenerator(
                design_path,
                "gemv_unified",
                (
                    dev,
                    s.K,
                    s.M,
                    self.cols,
                    0,
                ),
                {
                    "mode": "rda",
                    "rda_col": self.rda_col,
                    "rda_heads": self.rda_heads,
                    "format": s.format,
                    "active_cols": s.active_cols,
                    "x_slots": s.x_slots,
                    "chunks_per_col": s.chunks_per_col,
                    "continuous_mem_w": self.continuous_mem_w,
                    "sequence_plan": self.sequence_plan,
                    "gemv_kernel": self._gemv_op._kernel(),  # Reuse the same GEMV workers as the config body
                },
            ),
        )

    def get_kernel_artifacts(self):  # Implements an MLIROperator abstract method
        return self._gemv_op.get_kernel_artifacts()

    def get_mlir_artifact(self):  # Implements an MLIROperator abstract method
        return self._mlir_artifact(f"{self.name}.mlir")

    def set_up_artifacts(self) -> None:
        """Register the shared xclbin (via the GEMV op) and the RDA instruction stream."""
        if not getattr(self._gemv_op, "artifacts", None):  # Do not recreate already-compiled artifacts
            self._gemv_op.set_up_artifacts()
        self.xclbin_artifact = self._gemv_op.xclbin_artifact
        self.mlir_artifact = self._mlir_artifact(f"{self.name}.mlir")
        self.insts_artifact = InstsBinArtifact(
            f"{self.name}.bin",
            mlir_input=self.mlir_artifact,
            dependencies=[self.mlir_artifact] + self.get_kernel_artifacts(),
            extra_flags=["--dynamic-objFifos"],
        )
        if hasattr(self, "add_artifacts") and callable(getattr(self, "add_artifacts")):
            self.add_artifacts([self.insts_artifact])

    def compile(self, *args: Any, **kwargs: Any) -> Any:
        """Compile the shared xclbin first, then the RDA instruction stream."""
        self._gemv_op.compile(*args, **kwargs)
        result = super().compile(*args, **kwargs)
        # IRON moves artifact paths into the build dir in place; keep the
        # GEMV op's compiled xclbin object (set_up_artifacts must not recreate it).
        self.xclbin_artifact = self._gemv_op.xclbin_artifact
        return result

    def get_arg_spec(self) -> list[Any]:
        """Fixed 4-BO ABI for RDA state update."""
        H = self.rda_heads
        return [
            AIERuntimeArgSpec("in", (H, V_DIM, K_DIM), dtype=np.float32),
            AIERuntimeArgSpec("out", (H, V_DIM, K_DIM), dtype=np.float32),
            AIERuntimeArgSpec("in", (H, HEAD_BUNDLE_SIZE), dtype=np.float32),
            AIERuntimeArgSpec("out", (H, V_DIM), dtype=np.float32),
        ]

    def get_callable(self) -> SharedRdaCallable:
        """Return an XRT callable and verify shared-xclbin metadata at startup."""
        assert self.xclbin_artifact is not None, "xclbin_artifact not initialized"
        assert self.insts_artifact is not None, "insts_artifact not initialized"

        xclbin_path = Path(self.xclbin_artifact.filename).resolve()
        insts_path = Path(self.insts_artifact.filename).resolve()
        assert xclbin_path.is_file(), f"Shared xclbin file does not exist: {xclbin_path}"
        assert insts_path.is_file(), f"RDA insts.bin file does not exist: {insts_path}"

        mtime_ns = xclbin_path.stat().st_mtime_ns
        kernel_name = getattr(self.xclbin_artifact, "kernel_name", KERNEL_NAME)
        arg_spec = self.get_arg_spec()

        expected_abi = (
            ("in", str(np.dtype(np.float32)), 3),
            ("out", str(np.dtype(np.float32)), 3),
            ("in", str(np.dtype(np.float32)), 2),
            ("out", str(np.dtype(np.float32)), 2),
        )
        actual_abi = _abi_signature_from_specs(arg_spec)
        assert actual_abi == expected_abi, (
            f"UnifiedRDA ABI {actual_abi} does not match expected 4-BO ABI {expected_abi}"
        )
        assert kernel_name == KERNEL_NAME, (
            f"UnifiedRDA kernel_name={kernel_name!r}, expected {KERNEL_NAME!r}"
        )

        inner = super().get_callable()
        return SharedRdaCallable(
            inner,
            xclbin_path=xclbin_path,
            xclbin_mtime_ns=mtime_ns,
            kernel_name=kernel_name,
            arg_spec=arg_spec,
        )
